# api_key_demo.py — minimal API-key (token) auth demo

import os, time, hmac, json, base64, hashlib, sqlite3
from typing import Tuple
from fastapi import FastAPI, HTTPException, Request, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "auth_demo.db")
AUTH_TOKEN_SECRET = (os.getenv("AUTH_TOKEN_SECRET") or "dev-secret-change-me").encode("utf-8")
AUTH_TOKEN_TTL = int(os.getenv("AUTH_TOKEN_TTL", "86400"))  # 24h

app = FastAPI(title="Bhasha API Key Demo")

# ---------------- DB & PW helpers ----------------
def _db():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        pw_salt BLOB NOT NULL,
        pw_hash BLOB NOT NULL,
        created_at INTEGER NOT NULL
      )
    """)
    conn.commit()
    return conn

def _pbkdf2(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000, dklen=32)

def hash_password(password: str) -> Tuple[bytes, bytes]:
    salt = os.urandom(16)
    digest = _pbkdf2(password, salt)
    return salt, digest

def verify_password(password: str, salt: bytes, digest: bytes) -> bool:
    return hmac.compare_digest(_pbkdf2(password, salt), digest)

# ---------------- Token helpers (same shape as your app) ----------------
def sign_token(email: str, ttl: int = AUTH_TOKEN_TTL) -> str:
    payload = {"sub": email, "iat": int(time.time()), "exp": int(time.time()) + ttl}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig  = hmac.new(AUTH_TOKEN_SECRET, body.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{body}.{sig_b64}"

def _urlsafe_b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def verify_token(token: str) -> dict:
    try:
        body_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        raise HTTPException(401, "Malformed token")
    expected_sig = hmac.new(AUTH_TOKEN_SECRET, body_b64.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, _urlsafe_b64decode(sig_b64)):
        raise HTTPException(401, "Invalid signature")
    payload = json.loads(_urlsafe_b64decode(body_b64).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(401, "Token expired")
    return payload

security = HTTPBearer(auto_error=False)
async def require_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not creds or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise HTTPException(401, "Missing bearer token")
    return verify_token(creds.credentials)

# ---------------- Public routes ----------------
@app.post("/auth/signup")
async def signup(data: dict = Body(...)):
    name  = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    pw    = (data.get("password") or "")
    if not name or not email or not pw:
        raise HTTPException(400, "Missing name/email/password")

    conn = _db(); cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE email=?", (email,))
    if cur.fetchone():
        raise HTTPException(409, "Email already registered")
    salt, digest = hash_password(pw)
    cur.execute("INSERT INTO users (name,email,pw_salt,pw_hash,created_at) VALUES (?,?,?,?,?)",
                (name, email, salt, digest, int(time.time())))
    conn.commit()
    token = sign_token(email)
    return {"ok": True, "token": token, "email": email, "name": name}

@app.post("/auth/login")
async def login(data: dict = Body(...)):
    email = (data.get("email") or "").strip().lower()
    pw    = (data.get("password") or "")
    if not email or not pw:
        raise HTTPException(400, "Missing email/password")

    conn = _db(); cur = conn.cursor()
    cur.execute("SELECT name,pw_salt,pw_hash FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    if not row: raise HTTPException(401, "Invalid credentials")
    name, salt, digest = row
    if not verify_password(pw, salt, digest):
        raise HTTPException(401, "Invalid credentials")
    token = sign_token(email)
    return {"ok": True, "token": token, "email": email, "name": name}

# ---------------- Protected route ----------------
@app.post("/v1/echo")
async def echo(payload: dict = Body(...), user=Depends(require_token)):
    """
    Example protected endpoint. Replace with your own (e.g., /v1/translate, /v1/s2s).
    """
    return {"you_sent": payload, "auth_subject": user["sub"]}
