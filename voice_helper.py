# voice_helper.py
import speech_recognition as sr
from gtts import gTTS
from playsound3 import playsound
import tempfile
import os

def listen() -> str:
    """Capture microphone speech and return recognized text (Google SR)."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        r.energy_threshold = 300
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except Exception:
        return ""

def speak(text: str):
    """Say text aloud using gTTS (temp file, safe)."""
    if not text or not text.strip():
        return
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts = gTTS(text=text, lang="en")
            tts.save(f.name)
            path = f.name
        playsound(path)
    except Exception:
        pass
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
