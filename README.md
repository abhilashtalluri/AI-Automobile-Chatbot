# 🚗 AI-Automobile-Chatbot

An **AI-powered automobile chatbot** built using **Streamlit**, **TinyLlama**, and **Natural Language Processing (NLP)**.  
It supports **voice, text, and multilingual queries**, and intelligently answers automobile-related questions from a car manual.

---

## 🧩 Features

✅ Voice-based query recognition  
✅ AI natural conversation using **TinyLlama (via Ollama)**  
✅ Text-based multilingual support  
✅ Integration with **Nexon car manual** for knowledge-based answers  
✅ Smart issue prediction (engine, oil, brake, battery)  
✅ FAQ semantic matching using **Sentence Transformers**  
✅ Streamlit-based modern UI  

---

## 🛠️ Technologies Used

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| NLP Model | SentenceTransformer (`paraphrase-MiniLM-L3-v2`) |
| AI Model | TinyLlama (via Ollama) |
| Voice Recognition | SpeechRecognition + Google TTS |
| Data | Nexon Car Manual + FAQs |
| Deployment | Localhost / GitHub |

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/abhilashtalluri/AI-Automobile-Chatbot.git
cd AI-Automobile-Chatbot
