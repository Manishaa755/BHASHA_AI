# BHASHA_AI
Got it ✅ — here’s a **professional `README.md`** for your **Bhasha Multilingual Voice AI** project.
It explains what the project does, how to run it, and how to use the endpoints.

---

# 📢 Bhasha — Multilingual Voice AI

Bhasha is an **open-source multilingual AI assistant** that supports **speech-to-text (STT)**, **text-to-speech (TTS)**, **speech-to-speech (S2S)**, and **multilingual reasoning**.
It allows natural conversations in multiple languages and can be embedded into websites or apps as a chatbot widget.

---

## 🚀 Features

* 🎙 **Speech-to-Text (STT)** — Powered by OpenAI Whisper (`whisper-1`).
* 💬 **Chat Reasoning** — Uses OpenAI GPT (`gpt-4o-mini`, `gpt-4o`) for smart replies.
* 🔊 **Text-to-Speech (TTS)** — Natural speech with **ElevenLabs** (preferred) or **OpenAI TTS fallback**.
* 🔄 **Speech-to-Speech (S2S)** — Record voice → AI understands → AI replies with both **text + speech**.
* 🌍 **Multilingual** — Detects language automatically and replies in the same language.
* 📄 **Document Reasoning** (optional) — Summarization & Q&A over PDF, DOCX, PPTX, XLSX.
* 🌐 **Web Search Integration** (optional) — Adds real-time answers for current events and live queries.
* 🖼 **Frontend-ready** — Includes demo chatbot widget + static frontend mount.

---

## 📂 Project Structure

```
backend/
 ├── app.py             # Main FastAPI backend
 ├── demo_proxy.py      # Optional proxy for demos
 ├── frontend/          # Static frontend (HTML/JS/CSS)
 ├── requirements.txt   # Python dependencies
 └── README.md          # Project docs
```

---

## ⚙️ Requirements

* Python **3.9+**
* [FFmpeg](https://ffmpeg.org/download.html) (for audio conversions with `pydub`)
* OpenAI API key (required)
* ElevenLabs API key (optional, for higher-quality voices)

---

## 🔑 Environment Variables

Create a `.env` file in the backend folder:

```ini
# OpenAI
OPENAI_API_KEY=sk-xxxx
OPENAI_API_BASE=https://api.openai.com/v1

# Models
WHISPER_MODEL=whisper-1
CHAT_MODEL=gpt-4o-mini
TTS_MODEL=tts-1
TTS_VOICE=alloy

# ElevenLabs (optional)
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=alloy

# Frontend
FRONTEND_DIR=./frontend

# Search (optional)
SEARCH_PROVIDER=bing
BING_API_KEY=xxxx
SERPAPI_KEY=xxxx
```

---

## ▶️ Run Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📡 API Endpoints

### Health

```http
GET /health
```

---

### Speech-to-Text Chat

```http
POST /chat/
```

* Input: Audio file (`file`)
* Output: JSON `{ user_text, bot_reply }`

---

### Text-to-Speech

```http
POST /v1/tts-speak
```

* Form fields: `text`, `voice`, `fmt`
* Output: Audio (MP3/WAV) + headers

---

### Speech-to-Speech

```http
POST /v1/s2s
```

* Input: Audio file (`file`)
* Output: AI reply audio + headers (`X-Reply-Text-B64`, `X-Source-Text`)

---

### Demo Endpoints

* `/demo/chat_text` → `{ text }` → `{ bot_reply }`
* `/demo/s2s` → file upload → audio reply + headers
  *(used by chatbot widget, safe for frontend since no API key is exposed)*

---

### Document Reasoning (optional)

```http
POST /v1/reason-doc
```

* Supports `task=summarize` or `qa`
* Upload a document or raw text
* Returns summary/answer in target language

---

## 🖼 Frontend Chatbot Widget

The project ships with a **floating chat widget** (`frontend/index.html`) that connects to the `/demo/*` endpoints.
It lets users try **text + speech chat** directly in the browser without exposing API keys.

---

## 🌍 Deployment

You can deploy the backend easily on:

* [Vercel](https://vercel.com/) (with serverless FastAPI adapter)
* [Railway](https://railway.app/)
* [Render](https://render.com/)
* [Docker](https://www.docker.com/)

---

## 🔮 Roadmap

* [ ] Improve live web search integration
* [ ] Add conversation memory
* [ ] Support streaming TTS for lower latency
* [ ] Add more voices & custom styles

---

## 📜 License

MIT License © 2025 — Bhasha AI

---

👉 Would you like me to also create a **`requirements.txt`** for you (with FastAPI, httpx, pydub, openpyxl, etc.) so you can set up your environment in one command?
