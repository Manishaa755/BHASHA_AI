import os
import re
import base64
import asyncio
import sqlite3
import hmac
import hashlib
import time
import json
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from pydub import AudioSegment
from langdetect import detect, DetectorFactory
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Form
from fastapi.responses import (
    JSONResponse, FileResponse, HTMLResponse, RedirectResponse, Response, PlainTextResponse
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# --------------------------------------------------------------------
# Optional fastText language id (≈176 langs) and CLD3 (100+ langs)
# --------------------------------------------------------------------
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "")
_ft = None
if FASTTEXT_MODEL_PATH:
    try:
        import fasttext
        if os.path.isfile(FASTTEXT_MODEL_PATH):
            _ft = fasttext.load_model(FASTTEXT_MODEL_PATH)
        else:
            print(f"⚠️ FASTTEXT_MODEL_PATH not found: {FASTTEXT_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ fastText unavailable ({e}). Falling back when missing.")

_cld3 = None
try:
    import pycld3  # optional
    _cld3 = pycld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=2048)
except Exception as e:
    print(f"⚠️ pycld3 unavailable ({e}).")

# deterministic langdetect
DetectorFactory.seed = 42

APP_TITLE = "Bhasha — Multilingual Voice AI (STT + TTS + S2S + 200+ Lang Detect)"


def _coerce_model(value, valid, default, name):
    if not value or value.lower() not in valid:
        print(f"⚠️ {name}='{value}' invalid/unset; using '{default}'")
        return default
    return value.lower()


VALID_STT = {"whisper-1"}
VALID_CHAT = {"gpt-4o-mini", "gpt-4o"}
VALID_TTS = {"tts-1", "gpt-4o-mini-tts"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
WHISPER_MODEL = _coerce_model(
    os.getenv("WHISPER_MODEL"), VALID_STT,  "whisper-1", "WHISPER_MODEL")
CHAT_MODEL = _coerce_model(os.getenv("CHAT_MODEL"),
                           VALID_CHAT, "gpt-4o-mini", "CHAT_MODEL")
TTS_MODEL = _coerce_model(os.getenv("TTS_MODEL"),
                          VALID_TTS,  "tts-1", "TTS_MODEL")
TTS_DEFAULT_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_DEFAULT_FORMAT = os.getenv("TTS_FORMAT", "mp3")

# --------- Auth config (SQLite + HMAC token) ----------
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "auth.db")
AUTH_TOKEN_SECRET = (os.getenv("AUTH_TOKEN_SECRET")
                     or "dev-secret-change-me").encode("utf-8")
AUTH_TOKEN_TTL = int(os.getenv("AUTH_TOKEN_TTL", "86400"))  # 24h

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = os.getenv("FRONTEND_DIR", str(PROJECT_ROOT.parent / "frontend"))

app = FastAPI(title=APP_TITLE)

# --------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------
EXPOSE = "X-Reply-Text-B64, X-Source-Lang, X-Target-Lang"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # 🔒 tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=EXPOSE.split(", "),
)

# --------------------------------------------------------------------
# Header-safety helpers
# --------------------------------------------------------------------
_CTL_RE = re.compile(r"[\x00-\x1F\x7F]")


def safe_header_value(value: str, max_len: int = 1024) -> str:
    if value is None:
        return ""
    value = _CTL_RE.sub("", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    return value[:max_len]


def b64_header(value: str, max_len: int = 8192) -> str:
    if not value:
        return ""
    b = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return safe_header_value(b, max_len=max_len)

# --------------------------------------------------------------------
# Last-mile CORS + safety
# --------------------------------------------------------------------


@app.middleware("http")
async def ensure_cors_header(request: Request, call_next):
    try:
        resp = await call_next(request)
    except Exception as e:
        print("❌ Unhandled exception:", repr(e))
        resp = JSONResponse(status_code=500, content={
                            "error": "internal_server_error", "detail": str(e)[:800]})
    if "access-control-allow-origin" not in (k.lower() for k in resp.headers.keys()):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "*"
        resp.headers["Access-Control-Expose-Headers"] = EXPOSE
    return resp

# --------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------
if not os.path.isdir(STATIC_DIR):
    print(f"⚠️ FRONTEND_DIR not found: {STATIC_DIR}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/frontend/static", StaticFiles(directory=STATIC_DIR),
          name="frontend_static")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# --------------------------------------------------------------------
# Utilities (language & audio)
# --------------------------------------------------------------------


def _normalize_label(code: str) -> str:
    """
    Normalize a label to a coarse ISO 639-1/3 base:
    - lower, strip region (pt-BR -> pt)
    - trim to <= 3 chars
    """
    if not code:
        return ""
    c = code.strip().lower()
    if "-" in c or "_" in c:
        c = re.split(r"[-_]", c)[0]
    if len(c) > 3:
        c = c[:3]
    return c


def detect_language_fasttext_topk(text: str, k: int = 5) -> List[Dict]:
    out: List[Dict] = []
    if not _ft:
        return out
    t = (text or "").strip()
    if not t:
        return out
    try:
        labels, probs = _ft.predict(t.replace("\n", " "), k=k)
        for lab, p in zip(labels, probs):
            raw = lab.replace("__label__", "")
            out.append(
                {"code_raw": raw, "code_base": _normalize_label(raw), "prob": float(p)})
    except Exception:
        pass
    return out


def detect_language_cld3_topk(text: str, k: int = 5) -> List[Dict]:
    out: List[Dict] = []
    if not _cld3:
        return out
    t = (text or "").strip()
    if not t:
        return out
    try:
        langs = _cld3.FindTopNMostFreqLangs(t, k)
        for item in (langs or []):
            code = (item.language or "").lower()
            prob = float(item.probability or 0.0)
            out.append(
                {"code_raw": code, "code_base": _normalize_label(code), "prob": prob})
    except Exception:
        pass
    return out


def detect_language_langdetect(text: str) -> str:
    try:
        return detect(text) or ""
    except Exception:
        return ""

# ---------- LLM helpers: 200+ languages ----------


async def llm_detect_language(text: str) -> Dict:
    """
    Ask the LLM to identify language (ISO 639-1 if available else 639-3) + name.
    Returns: {"code_raw","code_base","name","prob"}
    """
    if not OPENAI_API_KEY:
        return {}
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    system = (
        "Identify the language of the given text across 200+ languages. "
        "Return ONLY a compact JSON: {\"code\":\"<iso>\",\"name\":\"<language name>\",\"confidence\":<0..1>} "
        "Use ISO 639-1 if available else ISO 639-3 for the code. No prose."
    )
    payload = {
        "model": CHAT_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            r = await client.post(url, headers=headers, json=payload)
            if not r.is_success:
                return {}
            data = r.json()
            content = data.get("choices", [{}])[0].get(
                "message", {}).get("content", "").strip()
            # parse JSON (robust)
            try:
                obj = json.loads(content)
            except Exception:
                obj = {}
                m = re.search(r'"code"\s*:\s*"([^"]+)"', content)
                n = re.search(r'"name"\s*:\s*"([^"]+)"', content)
                c = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
                if m:
                    obj["code"] = m.group(1)
                if n:
                    obj["name"] = n.group(1)
                if c:
                    obj["confidence"] = float(c.group(1))
            code = (obj.get("code") or "").strip().lower()
            name = (obj.get("name") or "").strip()
            conf = float(obj.get("confidence") or 0.85)
            if not code:
                return {}
            return {"code_raw": code, "code_base": _normalize_label(code), "name": name, "prob": conf}
    except Exception:
        return {}


async def llm_infer_target_language(user_text: str, default_code: str) -> str:
    """
    If the user explicitly asks to answer in some language, return its ISO code; else default_code.
    """
    if not OPENAI_API_KEY:
        return default_code
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    system = (
        "Decide the requested reply language for the user text. "
        "If user requested a language, return only its ISO code (639-1 or 639-3). "
        f"Otherwise return only '{default_code}'. No prose."
    )
    payload = {
        "model": CHAT_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            r = await client.post(url, headers=headers, json=payload)
            if not r.is_success:
                return default_code
            code = (r.json().get("choices", [{}])[0].get(
                "message", {}).get("content", "")).strip().lower()
            code = _normalize_label(code)
            return code or default_code
    except Exception:
        return default_code


async def llm_detect_dictionary_intent(text: str) -> Dict:
    """
    Detect 'dictionary' intent (e.g., “X ko English me kya kehte hain?”) in ANY language.
    Return {"intent":"dict","term":"..."} or {"intent":"none"}.
    """
    if not OPENAI_API_KEY:
        return {"intent": "none"}
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    system = (
        "Your job: decide if the user is asking 'what is <word/phrase> in English' (dictionary intent) "
        "in ANY language. Return ONLY JSON. "
        "If yes, return {\"intent\":\"dict\",\"term\":\"<the queried term>\"}. "
        "If not, return {\"intent\":\"none\"}."
    )
    payload = {
        "model": CHAT_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            if not r.is_success:
                return {"intent": "none"}
            content = r.json().get("choices", [{}])[0].get(
                "message", {}).get("content", "").strip()
            try:
                obj = json.loads(content)
            except Exception:
                obj = {"intent": "none"}
            if obj.get("intent") == "dict" and obj.get("term"):
                return {"intent": "dict", "term": str(obj["term"]).strip()}
            return {"intent": "none"}
    except Exception:
        return {"intent": "none"}


async def ensemble_best_code(text: str) -> Dict:
    """
    Combine fastText + CLD3 + langdetect + LLM fallback.
    """
    ft = detect_language_fasttext_topk(text, k=5)
    cd = detect_language_cld3_topk(text, k=5)
    ld_raw = detect_language_langdetect(text)
    ld_base = _normalize_label(ld_raw) if ld_raw else ""
    llm = await llm_detect_language(text)

    scores: Dict[str, float] = {}
    for e in ft:
        scores[e["code_base"]] = scores.get(e["code_base"], 0.0) + e["prob"]
    for e in cd:
        scores[e["code_base"]] = scores.get(e["code_base"], 0.0) + e["prob"]
    if ld_base:
        scores[ld_base] = scores.get(ld_base, 0.0) + 0.60
    if llm:
        scores[llm["code_base"]] = scores.get(
            llm["code_base"], 0.0) + float(llm.get("prob", 0.85))

    best_code, best_score = "", -1.0
    for k, v in scores.items():
        if v > best_score:
            best_code, best_score = k, v

    return {
        "best": {"code_base": best_code, "score": best_score},
        "scores": scores,
        "detectors": {
            "fasttext":  {"available": bool(_ft),  "top": ft},
            "cld3":      {"available": bool(_cld3), "top": cd},
            "langdetect": {"available": True,        "code_raw": ld_raw, "code_base": ld_base},
            "llm":       {"available": bool(OPENAI_API_KEY), "top": llm},
        }
    }


def _audio_content_type(fmt: str) -> str:
    fmt = (fmt or "").lower()
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
    }.get(fmt, "application/octet-stream")


def to_wav_16k_mono(raw_bytes: bytes) -> bytes:
    from io import BytesIO
    buf_in = BytesIO(raw_bytes)
    audio = AudioSegment.from_file(buf_in)
    audio = audio.set_frame_rate(16000).set_channels(1)
    buf_out = BytesIO()
    audio.export(buf_out, format="wav")
    return buf_out.getvalue()

# --------------------------------------------------------------------
# OpenAI calls (STT, chat, TTS)
# --------------------------------------------------------------------


async def openai_transcribe(file_bytes: bytes, filename: str, language: Optional[str] = None) -> dict:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY.")
    url = f"{OPENAI_API_BASE.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    def make_files():
        files = {
            "file": (filename or "audio.webm", file_bytes, "application/octet-stream"),
            "model": (None, WHISPER_MODEL),
        }
        if language:
            files["language"] = (None, language)
        return files

    max_tries, backoff = 4, 1.0
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(1, max_tries + 1):
            resp = await client.post(url, headers=headers, files=make_files())
            if resp.status_code == 400 and attempt == 1:
                try:
                    wav = to_wav_16k_mono(file_bytes)
                    files = {
                        "file": ("converted.wav", wav, "audio/wav"), "model": (None, WHISPER_MODEL)}
                    if language:
                        files["language"] = (None, language)
                    resp = await client.post(url, headers=headers, files=files)
                except Exception:
                    pass
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt == max_tries:
                    break
                delay = float(resp.headers.get("retry-after") or backoff)
                await asyncio.sleep(delay)
                backoff *= 2
                continue
            if not resp.is_success:
                raise HTTPException(
                    status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            return {"text": (data.get("text") or "").strip(), "language": data.get("language")}
    raise HTTPException(
        status_code=502, detail="Transcription failed after retries.")


async def openai_chat_reply(user_text: str, lang_hint: Optional[str], enforce_lang: bool = True) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY.")
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}

    base = ("You are a multilingual assistant. Be concise (2–4 sentences). "
            "If unsure, ask a clarifying question; do not guess.")
    if enforce_lang and lang_hint and lang_hint != "auto":
        system_msg = f"{base} Always reply exclusively in language code '{lang_hint}'."
    else:
        system_msg = f"{base} Choose the most appropriate language for the user."

    payload = {
        "model": CHAT_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ],
    }

    max_tries, backoff = 4, 1.0
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(1, max_tries + 1):
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt == max_tries:
                    break
                delay = float(resp.headers.get("retry-after") or backoff)
                await asyncio.sleep(delay)
                backoff *= 2
                continue
            if not resp.is_success:
                raise HTTPException(
                    status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "").strip())
    raise HTTPException(
        status_code=502, detail="Chat completion failed after retries.")


async def openai_dictionary_translate_to_english(term: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY.")
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": CHAT_MODEL,
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": "You are a bilingual dictionary. Return ONLY the most common English equivalent. No punctuation."},
            {"role": "user", "content": term},
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        translation = (data.get("choices", [{}])[0].get(
            "message", {}).get("content", "")).strip()
        return translation.split("\n")[0].strip(" '\".,;:()")


async def openai_tts(text: str, voice: Optional[str], fmt: Optional[str], model: Optional[str]) -> bytes:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY.")
    voice = (voice or TTS_DEFAULT_VOICE).strip()
    fmt = (fmt or TTS_DEFAULT_FORMAT).lower().strip()
    model = (model or TTS_MODEL).strip()

    url = f"{OPENAI_API_BASE.rstrip('/')}/audio/speech"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": model, "input": text, "voice": voice, "format": fmt}

    max_tries, backoff = 4, 1.0
    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(1, max_tries + 1):
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt == max_tries:
                    break
                delay = float(resp.headers.get("retry-after") or backoff)
                await asyncio.sleep(delay)
                backoff *= 2
                continue
            if not resp.is_success:
                raise HTTPException(
                    status_code=resp.status_code, detail=resp.text)
            return resp.content
    raise HTTPException(status_code=502, detail="TTS failed after retries.")

# --------------------------------------------------------------------
# Auth helpers (SQLite + PBKDF2)
# --------------------------------------------------------------------


def _db():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        pw_salt TEXT NOT NULL,
        pw_hash TEXT NOT NULL,
        created_at INTEGER NOT NULL
      )
    """)
    conn.commit()
    return conn


def _pbkdf2(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000, dklen=32)


def hash_password(password: str) -> Tuple[str, str]:
    salt = os.urandom(16)
    digest = _pbkdf2(password, salt)
    return base64.b64encode(salt).decode("ascii"), base64.b64encode(digest).decode("ascii")


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        test = _pbkdf2(password, salt)
        return hmac.compare_digest(test, expected)
    except Exception:
        return False


def sign_token(email: str, ttl: int = AUTH_TOKEN_TTL) -> str:
    payload = {"sub": email, "iat": int(
        time.time()), "exp": int(time.time()) + ttl}
    body = base64.urlsafe_b64encode(json.dumps(
        payload).encode()).decode().rstrip("=")
    sig = hmac.new(AUTH_TOKEN_SECRET, body.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{body}.{sig_b64}"

# --------------------------------------------------------------------
# Routes — health
# --------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_model": WHISPER_MODEL,
        "chat_model": CHAT_MODEL,
        "tts_model": TTS_MODEL,
        "detectors": {
            "fasttext": bool(_ft),
            "cld3": bool(_cld3),
            "langdetect": True,
            "llm": bool(OPENAI_API_KEY),
        },
        "frontend_dir": STATIC_DIR,
        "has_api_key": bool(OPENAI_API_KEY),
        "auth_db": os.path.abspath(AUTH_DB_PATH),
    }

# --------------------------------------------------------------------
# Routes — Auth
# --------------------------------------------------------------------


@app.post("/auth/signup")
async def auth_signup(request: Request):
    try:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            data = await request.json()
        else:
            form = await request.form()
            data = {k: form.get(k) for k in ("name", "email", "password")}
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "")
        if not name or not email or not password:
            raise HTTPException(
                status_code=400, detail="Missing name/email/password.")
        conn = _db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        if cur.fetchone():
            raise HTTPException(
                status_code=409, detail="Email already registered.")
        salt, digest = hash_password(password)
        cur.execute("INSERT INTO users (name,email,pw_salt,pw_hash,created_at) VALUES (?,?,?,?,?)",
                    (name, email, salt, digest, int(time.time())))
        conn.commit()
        token = sign_token(email)
        return JSONResponse({"ok": True, "token": token, "email": email, "name": name})
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /auth/signup error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})


@app.post("/auth/login")
async def auth_login(request: Request):
    try:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            data = await request.json()
        else:
            form = await request.form()
            data = {k: form.get(k) for k in ("email", "password")}
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "")
        if not email or not password:
            raise HTTPException(
                status_code=400, detail="Missing email/password.")
        conn = _db()
        cur = conn.cursor()
        cur.execute(
            "SELECT name,pw_salt,pw_hash FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        name, salt_b64, hash_b64 = row
        if not verify_password(password, salt_b64, hash_b64):
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        token = sign_token(email)
        return JSONResponse({"ok": True, "token": token, "email": email, "name": name})
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /auth/login error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})

# --------------------------------------------------------------------
# NEW: Translate route (text -> text)
# --------------------------------------------------------------------


@app.post("/v1/translate")
async def translate(payload: dict = Body(...)):
    try:
        src_text = (payload.get("text") or "").strip()
        target_lang = _normalize_label(
            (payload.get("target_lang") or "en").strip().lower())
        if not src_text:
            raise HTTPException(status_code=400, detail="Missing 'text'.")
        translated = await openai_chat_reply(
            f"Translate this to {target_lang}:\n\n{src_text}",
            target_lang,
            enforce_lang=True
        )
        return {"input": src_text, "translated": translated, "target_lang": target_lang}
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /v1/translate error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})

# --------------------------------------------------------------------
# Routes — STT (speech -> text chat reply)
# --------------------------------------------------------------------


@app.post("/chat/")
async def chat_with_bot(
    file: UploadFile = File(...),
    lang: str = Query(
        "auto", description="Language hint (e.g., hi, bn, ta-IN) or 'auto'")
):
    try:
        if not file or not file.filename:
            raise HTTPException(
                status_code=400, detail="No audio file provided.")
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=400, detail="Uploaded file is empty.")

        lang_hint = None if lang.lower() == "auto" else lang
        stt = await openai_transcribe(audio_bytes, file.filename, lang_hint)
        user_text = stt.get("text", "")

        ensemble = await ensemble_best_code(user_text)
        src_lang = ensemble["best"]["code_base"] or (
            stt.get("language") or "auto")
        tgt_lang = await llm_infer_target_language(user_text, src_lang)

        # Dictionary intent via LLM (no hardcoded patterns)
        d = await llm_detect_dictionary_intent(user_text)
        if d.get("intent") == "dict":
            english = await openai_dictionary_translate_to_english(d["term"])
            reply = await openai_chat_reply(
                f"Say in {tgt_lang} that the '{d['term']}' is called '{english}' in English, succinctly.",
                tgt_lang, enforce_lang=True
            )
        else:
            reply = await openai_chat_reply(user_text, tgt_lang, enforce_lang=True)

        return JSONResponse({
            "user_text": user_text,
            "bot_reply": reply,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "lang_detect": ensemble
        })
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /chat error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})

# --------------------------------------------------------------------
# Routes — TTS (text -> speech)
# --------------------------------------------------------------------


@app.options("/v1/tts-speak")
async def tts_speak_options():
    return PlainTextResponse("", status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Expose-Headers": EXPOSE,
    })


@app.post("/v1/tts-speak")
async def tts_speak(payload: dict = Body(...)):
    try:
        txt = payload.get("text")
        user_text = (txt.strip() if isinstance(txt, str) else "")
        if not user_text:
            raise HTTPException(status_code=400, detail="Missing 'text'.")

        ensemble = await ensemble_best_code(user_text)
        src = ensemble["best"]["code_base"] or "auto"
        tgt_lang = await llm_infer_target_language(user_text, src)

        voice = (payload.get("voice") or TTS_DEFAULT_VOICE).strip()
        fmt = (payload.get("format") or TTS_DEFAULT_FORMAT).lower().strip()
        tts_model = (payload.get("tts_model") or TTS_MODEL).strip()

        d = await llm_detect_dictionary_intent(user_text)
        if d.get("intent") == "dict":
            english = await openai_dictionary_translate_to_english(d["term"])
            reply_text = await openai_chat_reply(
                f"Say in {tgt_lang} that the '{d['term']}' is called '{english}' in English, succinctly.",
                tgt_lang, enforce_lang=True
            )
        else:
            reply_text = await openai_chat_reply(user_text, tgt_lang, enforce_lang=True)

        audio_bytes = await openai_tts(reply_text, voice, fmt, tts_model)
        media_type = _audio_content_type(fmt)

        headers = {
            "X-Reply-Text-B64": b64_header(reply_text),
            "X-Source-Lang": safe_header_value(src, 32),
            "X-Target-Lang": safe_header_value(tgt_lang, 32),
            "Cache-Control": "no-store",
            "Content-Disposition": safe_header_value(f'inline; filename="reply.{fmt}"'),
        }
        return Response(content=audio_bytes, media_type=media_type, headers=headers)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /v1/tts-speak error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})

# --------------------------------------------------------------------
# Routes — Language Detection API (text or audio) — 200+ coverage
# --------------------------------------------------------------------


@app.post("/v1/lang-detect")
async def lang_detect(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        detected_text = None
        if file and file.filename:
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(
                    status_code=400, detail="Uploaded audio is empty.")
            stt = await openai_transcribe(audio_bytes, file.filename, None)
            detected_text = (stt.get("text") or "").strip()
        elif text:
            detected_text = text.strip()

        if not detected_text:
            raise HTTPException(status_code=400, detail="No input provided.")

        ensemble = await ensemble_best_code(detected_text)

        # Friendly combined top view
        combined: Dict[str, float] = {}
        for e in ensemble["detectors"]["fasttext"]["top"]:
            combined[e["code_base"]] = combined.get(
                e["code_base"], 0.0) + e["prob"]
        for e in ensemble["detectors"]["cld3"]["top"]:
            combined[e["code_base"]] = combined.get(
                e["code_base"], 0.0) + e["prob"]
        ld_base = ensemble["detectors"]["langdetect"]["code_base"]
        if ld_base:
            combined[ld_base] = combined.get(ld_base, 0.0) + 0.60
        llm = ensemble["detectors"]["llm"]
        if llm and llm.get("top", {}).get("code_base"):
            combined[llm["top"]["code_base"]] = combined.get(
                llm["top"]["code_base"], 0.0) + float(llm["top"].get("prob", 0.85))

        top_combined = sorted(
            [{"code_base": k, "score": v} for k, v in combined.items()],
            key=lambda x: x["score"], reverse=True
        )[:5]

        return JSONResponse({
            "input": detected_text,
            "best": ensemble["best"],
            "top_combined": top_combined,
            "detectors": ensemble["detectors"]
        })
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /v1/lang-detect error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})

# --------------------------------------------------------------------
# Routes — S2S (speech -> speech)
# --------------------------------------------------------------------


@app.post("/v1/s2s")
async def speech_to_speech(
    file: UploadFile = File(...,
                            description="Audio file from MediaRecorder or upload"),
    target_lang: Optional[str] = Query(
        None, description="(Optional) 'same'/'auto'/language code override"),
    voice: Optional[str] = Query(None, description="TTS voice"),
    format: Optional[str] = Query(
        None, description="Audio format: mp3|wav|opus|aac|flac"),
    tts_model: Optional[str] = Query(None, description="TTS model")
):
    try:
        if not file or not file.filename:
            raise HTTPException(
                status_code=400, detail="No audio file provided.")
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=400, detail="Uploaded file is empty.")

        # 1) STT
        stt = await openai_transcribe(audio_bytes, file.filename, None)
        user_text = (stt.get("text") or "").strip()
        if not user_text:
            raise HTTPException(
                status_code=400, detail="Could not understand audio.")

        # 2) Detect src via ensemble (200+)
        ens = await ensemble_best_code(user_text)
        src_lang = ens["best"]["code_base"] or (stt.get("language") or "auto")

        # 3) target language
        if target_lang:
            tl = target_lang.lower().strip()
            if tl == "same":
                tgt_lang = src_lang
                enforce = True
            elif tl == "auto":
                tgt_lang = await llm_infer_target_language(user_text, src_lang)
                enforce = (tgt_lang != "auto")
            else:
                tgt_lang = _normalize_label(tl)
                enforce = True
        else:
            tgt_lang = await llm_infer_target_language(user_text, src_lang)
            enforce = (tgt_lang != "auto")

        # 4) Dictionary intent via LLM
        d = await llm_detect_dictionary_intent(user_text)
        if d.get("intent") == "dict":
            english = await openai_dictionary_translate_to_english(d["term"])
            reply_text = await openai_chat_reply(
                f"Say in {tgt_lang if tgt_lang != 'auto' else 'the best fitting language'} that the '{d['term']}' is called '{english}' in English, succinctly.",
                tgt_lang, enforce_lang=enforce
            )
        else:
            reply_text = await openai_chat_reply(user_text, tgt_lang, enforce_lang=enforce)

        # 5) TTS
        v = (voice or TTS_DEFAULT_VOICE).strip()
        fmt = (format or TTS_DEFAULT_FORMAT).lower().strip()
        ttsm = (tts_model or TTS_MODEL).strip()
        audio_bytes_out = await openai_tts(reply_text, v, fmt, ttsm)
        media_type = _audio_content_type(fmt)

        headers = {
            "X-Reply-Text-B64": b64_header(reply_text),
            "X-Source-Lang": safe_header_value(src_lang, 32),
            "X-Target-Lang": safe_header_value(tgt_lang, 32),
            "Cache-Control": "no-store",
            "Content-Disposition": safe_header_value(f'inline; filename="reply.{fmt}"'),
        }
        return Response(content=audio_bytes_out, media_type=media_type, headers=headers)
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
    except Exception as e:
        print("❌ /v1/s2s error:", repr(e))
        return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(e)[:800]})

# --------------------------------------------------------------------
# Page serving
# --------------------------------------------------------------------


@app.get("/", response_class=FileResponse)
async def home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse(f"<h1>Missing index.html</h1><p>Expected at: <code>{index_path}</code></p>", status_code=200)
    return FileResponse(index_path)


@app.get("/stt")
async def stt_redirect():
    return RedirectResponse(url="/static/stt.html", status_code=307)


@app.get("/tts")
async def tts_redirect():
    return RedirectResponse(url="/static/tts.html", status_code=307)


@app.get("/s2s")
async def s2s_redirect():
    return RedirectResponse(url="/static/sts.html", status_code=307)


@app.get("/langdetect")
async def langdetect_redirect():
    return RedirectResponse(url="/static/langdetect.html", status_code=307)
