# chatbot.py
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import re
from langdetect import detect
import requests
import random
import os

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "tinyllama"   # keep it small; works on low RAM

# -----------------------------
# Globals (lazy-initialized)
# -----------------------------
_sentence_model = None
_manual_sentences = None
_manual_embeddings = None

# -----------------------------
# Helpers: Translation
# -----------------------------
def _translate_to_en(txt: str) -> str:
    if not txt or not txt.strip():
        return ""
    try:
        return GoogleTranslator(source="auto", target="en").translate(txt) or txt
    except Exception:
        return txt

def _translate_from_en(txt: str, lang: str) -> str:
    if not txt or not txt.strip():
        return ""
    if not lang or lang == "en":
        return txt
    try:
        return GoogleTranslator(source="en", target=lang).translate(txt) or txt
    except Exception:
        return txt

# -----------------------------
# Loaders (manual + model)
# -----------------------------
def _load_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _sentence_model

def _load_manual_sentences():
    """
    Load + lightly clean the manual text and split into “meaningful”
    sentences. Uses Clean_Nexon_manual.txt if present, else Nexon_manual.txt.
    """
    global _manual_sentences
    if _manual_sentences is not None:
        return _manual_sentences

    path = "Clean_Nexon_manual.txt"
    if not os.path.exists(path):
        path = "Clean_Nexon_manual.txt"

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # normalize spaces and junk
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,!?\'\"()/%\-& ]+", " ", text)

    # split by sentence enders
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # keep only reasonably long sentences
    sentences = [s.strip() for s in sentences if len(s.split()) > 6]

    _manual_sentences = sentences
    return _manual_sentences

def _ensure_manual_embeddings():
    """
    Build sentence embeddings once and cache.
    """
    global _manual_embeddings
    if _manual_embeddings is not None:
        return _manual_embeddings

    model = _load_sentence_model()
    sents = _load_manual_sentences()
    _manual_embeddings = model.encode(sents, convert_to_tensor=True)
    return _manual_embeddings

# -----------------------------
# Heuristics: Greeting & Car Detection
# -----------------------------
_GREETING_REGEX = re.compile(
    r"\b(h+i+|hey+|hello+|yo+|sup+|wass?up+|whats\s*up+|hi+ bro+|hey bro+|namaste|mama|bro|dude)\b",
    re.IGNORECASE
)

# Add a few Indic casual phrases too
_GREETING_HINTS = [
    "ela unnav", "bagunnava", "bagunnara",  # Telugu casual
    "kaise ho", "kya haal", "aap kaise hain"  # Hindi casual
]

_CAR_KEYWORDS = [
    "car","engine","battery","tyre","tire","oil","brake","gear","headlight",
    "nexon","vehicle","clutch","service","maintenance","mode","fuel","drive",
    "coolant","milage","mileage","odometer","speedometer","airbag","abs","tpms",
    "infotainment","ac","air conditioner","warranty","manual","boot","hood"
]

_FILTER_MAP = {
    "engine": ["engine","oil","coolant","rpm","temperature","check engine","misfire","overheat"],
    "mileage": ["mileage","milage","fuel economy","average","kmpl","mpg"],
    "fuel": ["fuel","petrol","diesel","octane","fuel pump","fuel filter","range"],
    "battery": ["battery","charging","jump start","voltage"],
    "brake": ["brake","abs","pad","disc","fluid"],
    "tyre": ["tyre","tire","pressure","tpms","puncture","spare"],
    "mode": ["mode","eco","city","sport","driving mode"]
}

def _looks_like_greeting(text_en: str) -> bool:
    if _GREETING_REGEX.search(text_en):
        return True
    lowered = text_en.lower()
    return any(hint in lowered for hint in _GREETING_HINTS)

def _is_car_related(text_en: str) -> bool:
    t = text_en.lower()
    return any(kw in t for kw in _CAR_KEYWORDS)

def _keyword_filter_sentences(query_en: str, sentences: list[str]) -> list[str]:
    """
    Before similarity, filter the manual to sections relevant to the query.
    This improves accuracy for queries like "engine mileage".
    """
    q = query_en.lower()
    bucket = set()
    for key, words in _FILTER_MAP.items():
        if key in q or any(w in q for w in words):
            bucket.update(words)
    if not bucket:
        # generic: if user typed any car kw, keep sentences where that kw appears
        kws = [kw for kw in _CAR_KEYWORDS if kw in q]
        bucket.update(kws)

    if not bucket:
        # no filter – return original
        return sentences

    filtered = []
    for s in sentences:
        ls = s.lower()
        if any(w in ls for w in bucket):
            filtered.append(s)

    # fallback if too small
    return filtered if len(filtered) >= 20 else sentences

# -----------------------------
# Ollama Helpers
# -----------------------------
def _ollama_generate(prompt: str, temperature: float = 0.6, max_retries: int = 1) -> str | None:
    """
    Call Ollama (TinyLlama) to generate a SHORT casual reply.
    Returns None if Ollama not reachable.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            "You are a friendly AI assistant. "
            "Always reply in ONE short line (max ~18 words). "
            "Be natural, casual, and avoid long explanations.\n\n"
            f"User: {prompt}\nAssistant:"
        ),
        "options": {"temperature": temperature}
    }
    for _ in range(max_retries+1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                text = (data.get("response") or "").strip()
                if text:
                    # squash newlines
                    return re.sub(r"\s+", " ", text)
            return None
        except Exception:
            pass
    return None

def _classify_intent_ollama(text_en: str) -> str | None:
    """
    Ask TinyLlama to classify intent as greeting/car/other.
    Returns one of: 'greeting' | 'car' | 'other' | None
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            "Classify the message as exactly one word: greeting, car, or other.\n"
            f"Message: {text_en}\nLabel:"
        ),
        "options": {"temperature": 0.1}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=20)
        if r.status_code == 200:
            label = (r.json().get("response") or "").strip().lower()
            if "greeting" in label:
                return "greeting"
            if "car" in label:
                return "car"
            if "other" in label:
                return "other"
    except Exception:
        return None
    return None

# -----------------------------
# History
# -----------------------------
try:
    _history = pd.read_csv("history.csv")
except Exception:
    _history = pd.DataFrame(columns=["question"])

def _save_history(q: str):
    global _history
    new = pd.DataFrame({"question": [q]})
    _history = pd.concat([_history, new], ignore_index=True)
    _history.to_csv("history.csv", index=False)

# -----------------------------
# Public: get_best_answer
# -----------------------------
def get_best_answer(user_input: str) -> str:
    """
    Handles:
    - greetings/small-talk (AI via TinyLlama if available)
    - car queries via manual + embeddings + AI fallback
    - non-car queries: polite refusal (AI phrased if available)
    """
    if not user_input or not user_input.strip():
        return "⚙️ I’m here, but didn’t get that properly. Try again?"

    _save_history(user_input)

    # Detect language
    try:
        lang = detect(user_input)
    except Exception:
        lang = "en"

    text_en = _translate_to_en(user_input).strip() or user_input.strip()

    # Greeting check
    if _looks_like_greeting(text_en):
        ai = _ollama_generate(f"The user said '{text_en}'. Respond casually like a friend in one short line.")
        return _translate_from_en(ai or "Hey there! How are you?", lang)

    # Intent classification (AI)
    intent = _classify_intent_ollama(text_en)
    if intent == "greeting":
        ai = _ollama_generate(f"The user said '{text_en}'. Greet back warmly.")
        return _translate_from_en(ai or "Hey! All good here, how about you?", lang)

    # Detect if car-related
    is_car = _is_car_related(text_en) or (intent == "car")
    if not is_car:
        ai = _ollama_generate(
            f"The user said '{text_en}'. Politely reply that you can only answer automobile-related questions in one short line."
        )
        return _translate_from_en(ai or "Sorry! I can answer only automobile-related questions.", lang)

    # Car-related: manual + fallback
    model = _load_sentence_model()
    sentences = _load_manual_sentences()
    filtered = _keyword_filter_sentences(text_en, sentences)

    if filtered is sentences:
        emb = _ensure_manual_embeddings()
    else:
        emb = model.encode(filtered, convert_to_tensor=True)

    user_emb = model.encode(text_en, convert_to_tensor=True)
    sims = util.cos_sim(user_emb, emb)[0]
    best_idx = torch.argmax(sims).item()
    best_score = sims[best_idx].item()

    if best_score < 0.55:
        # 🔥 AI fallback for vague questions like "engine mileage"
        ai_prompt = (
            f"The user asked '{text_en}'. Give a short, helpful automobile-related answer (not from manual). "
            f"Limit to one or two sentences, human tone."
        )
        ai_reply = _ollama_generate(ai_prompt)
        if not ai_reply:
            ai_reply = (
                "Engine mileage depends on driving mode, maintenance, and fuel quality. "
                "Regular servicing improves efficiency."
            )
        return _translate_from_en(ai_reply, lang)

    # Else return the best manual answer
    best_text = (filtered[best_idx] if filtered is not sentences else sentences[best_idx]).strip()
    best_text = re.sub(r"\s+", " ", best_text)
    if len(best_text) > 400:
        best_text = best_text[:best_text.rfind(".")] + "."

    return _translate_from_en(best_text, lang)

# -----------------------------
# Public: predict_issue
# -----------------------------
def predict_issue() -> str:
    issues = ["battery", "engine", "oil", "brake", "service"]
    freq = {k: 0 for k in issues}
    for q in _history["question"]:
        ql = str(q).lower()
        for k in issues:
            if k in ql:
                freq[k] += 1
    top = max(freq, key=freq.get)
    if freq[top] >= 3:
        return f"⚠️ It seems your {top} might need attention soon."
    return ""
