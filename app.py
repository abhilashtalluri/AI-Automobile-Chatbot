# app.py
import streamlit as st
from chatbot import get_best_answer, predict_issue
from voice_helper import listen, speak
import time
import requests

st.set_page_config(page_title="TEAM C automobile Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<h1 style='text-align:center; color:white;'>TEAM C automobile Chatbot</h1>
<p style='text-align:center; color:#b0d6ff;'>mutli talented</p>
<hr>
""", unsafe_allow_html=True)

# ---- Ollama status ----
def ollama_up() -> bool:
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

st.markdown(
    "<p style='color:{}'>{}</p>".format(
        "lightgreen" if ollama_up() else "orange",
        "✅ AI mode active (TinyLlama connected)" if ollama_up() else "⚠️ AI mode disabled — Ollama not running"
    ),
    unsafe_allow_html=True
)

# ---- Input section ----
col1, col2 = st.columns([2, 1])
with col1:
    mode = st.radio("Mode:", ["Text 💬", "Voice 🎙️"])
with col2:
    st.markdown("<p style='color:#b0d6ff;'>🌍 Auto language detection enabled</p>", unsafe_allow_html=True)

user_input = ""

if mode.startswith("Text"):
    user_input = st.text_input("💭 Type here:")
else:
    if st.button("🎧 Speak now"):
        st.info("Listening...")
        user_input = listen()
        if user_input:
            st.success(f"Captured: “{user_input}”")
        else:
            st.warning("No voice detected. Try again.")

# ---- Response handling ----
if user_input:
    st.markdown(
        f"<div style='background:#48b1bf;color:white;padding:10px;border-radius:10px;'><b>You:</b> {user_input}</div>",
        unsafe_allow_html=True
    )

    stop = st.button("🛑 Stop Generating")
    if not stop:
        with st.spinner("🤔 Thinking..."):
            reply = get_best_answer(user_input)

            pred = predict_issue()
            if pred:
                reply += f"\n\n🔧 {pred}"

            # ensure non-empty
            if not reply or not reply.strip():
                reply = "⚙️ I’m here, but didn’t get that properly. Try again?"

            # typing animation
            placeholder = st.empty()
            acc = ""
            for ch in reply:
                acc += ch
                placeholder.markdown(
                    f"<div style='background:#f2f2f2;color:black;padding:10px;border-radius:10px;'><b>Bot:</b> {acc}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.01)

            # speak (best-effort)
            try:
                speak(reply)
            except Exception:
                pass
    else:
        st.warning("🛑 Chat stopped by user.")
