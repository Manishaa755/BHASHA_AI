# backend/auth.py
# Simple authentication router that stores opaque tokens in the DB.
# Endpoints:
#   POST /auth/signup  -> { token, user: { id, email } }
#   POST /auth/login   -> { token, user: { id, email } }
#   GET  /auth/me      -> { id, email } (requires Authorization: Bearer <token>)

import os
import secrets
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, DateTime, select, Boolean
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
import bcrypt

# Config - keep same DB as api_keys module (override via API_KEYS_DB env if needed)
DB_PATH = os.getenv("API_KEYS_DB", "backend/api_keys.db")
ENGINE_URL = f"sqlite:///{DB_PATH}"

engine: Engine = create_engine(ENGINE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

# Users table
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("email", String, nullable=False, unique=True),
    Column("password_hash", String, nullable=False),
    Column("api_token", String, nullable=True, unique=True),  # opaque bearer token
    Column("is_active", Boolean, default=True),
    Column("created_at", DateTime, default=datetime.utcnow),
)

# create table if missing
metadata.create_all(engine)

router = APIRouter(prefix="/auth", tags=["auth"])


# Pydantic models
class SignupIn(BaseModel):
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _generate_token() -> str:
    return secrets.token_urlsafe(32)


# Helper to fetch user row as mapping (dict-like)
def _get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    email = (email or "").lower().strip()
    with engine.begin() as conn:
        sel = select(
            users.c.id,
            users.c.email,
            users.c.password_hash,
            users.c.api_token,
            users.c.is_active,
            users.c.created_at,
        ).where(users.c.email == email)
        res = conn.execute(sel)
        row = res.mappings().fetchone()
    return dict(row) if row else None


def _get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        sel = select(
            users.c.id,
            users.c.email,
            users.c.password_hash,
            users.c.api_token,
            users.c.is_active,
            users.c.created_at,
        ).where(users.c.api_token == token)
        res = conn.execute(sel)
        row = res.mappings().fetchone()
    return dict(row) if row else None


@router.post("/signup")
async def signup(payload: SignupIn):
    email = payload.email.lower().strip()
    password = payload.password or ""
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    # Check existing
    existing = _get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    pw_hash = _hash_password(password)
    token = _generate_token()
    now = datetime.utcnow()
    try:
        with engine.begin() as conn:
            ins = users.insert().values(
                email=email,
                password_hash=pw_hash,
                api_token=token,
                is_active=True,
                created_at=now,
            )
            res = conn.execute(ins)
            # inserted_primary_key may be None or a tuple
            pk = None
            try:
                pk = res.inserted_primary_key[0]
            except Exception:
                # fallback: attempt to select the row we just inserted
                pk = None
            if not pk:
                # try to read user back
                row = _get_user_by_email(email)
                user_id = row["id"] if row else None
            else:
                user_id = pk
    except IntegrityError as e:
        # race or duplicate insert
        raise HTTPException(status_code=400, detail="Email already registered.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {"token": token, "user": {"id": user_id, "email": email}}


@router.post("/login")
async def login(payload: LoginIn):
    email = payload.email.lower().strip()
    password = payload.password or ""
    row = _get_user_by_email(email)
    if not row:
        # don't leak whether user exists
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not _check_password(password, row.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # rotate token on login
    token = _generate_token()
    try:
        with engine.begin() as conn:
            upd = users.update().where(users.c.id == row["id"]).values(api_token=token)
            conn.execute(upd)
    except Exception:
        # non-fatal; continue with existing token if rotation failed (but we generated a new one)
        pass

    return {"token": token, "user": {"id": row["id"], "email": row["email"]}}


# Dependency to get current user from Authorization header
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Accepts: Authorization: Bearer <token>
    Returns user dict or raises 401.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    user = _get_user_by_token(token)
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # return minimal user info
    return {"id": user["id"], "email": user["email"]}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "email": current_user["email"]}


# For other modules that expect validate_api_key_header in auth.py, export alias
# (Note: this is different — it's user token validation, not X-API-Key used for third-party API keys.)
validate_api_key_header = get_current_user  # if some modules import auth.validate_api_key_header
