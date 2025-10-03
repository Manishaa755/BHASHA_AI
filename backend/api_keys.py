# backend/api_keys.py
"""
API keys manager (SQLAlchemy core).
- create_api_key_for_user(user_id, name=None, scopes=None) -> returns dict with kid, secret, preview, created_at
- list_api_keys_for_user(user_id) -> list of rows (previews only)
- get_api_key_by_kid(kid) / get_api_key_by_secret(secret)
- revoke_api_key(kid, user_id=None, admin=False) -> soft-disable
- delete_api_key_hard(kid) -> hard delete (and delete usage)
- record_usage(api_key_id, tool, units, cost_in_inr, client_ip=None, user_agent=None)
- validate_api_key_header(x_api_key) -> FastAPI dependency (returns api_key_info)
- reporting helpers: summarize_usage_by_key_since, list_all_keys
"""
import os
import secrets
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import bcrypt
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Boolean, DateTime, Float, select, func
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from fastapi import Header, HTTPException

# DB path (change via env var if needed)
DB_PATH = os.getenv("API_KEYS_DB", os.path.join(os.path.dirname(__file__), "api_keys.db"))
ENGINE_URL = f"sqlite:///{DB_PATH}"

# engine & metadata
engine: Engine = create_engine(ENGINE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

# Tables
api_keys_table = Table(
    "api_keys",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kid", String(64), unique=True, nullable=False),      # public identifier
    Column("secret_hash", String(200), nullable=False),          # bcrypt hash of secret
    Column("user_id", String(200), nullable=False),              # owner identifier
    Column("name", String(200), nullable=True),
    Column("preview", String(100), nullable=True),               # masked preview shown to client
    Column("scopes", String(1000), nullable=True),              # JSON string
    Column("disabled", Boolean, default=False),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)

api_key_usage_table = Table(
    "api_key_usage",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("api_key_id", Integer, nullable=False),
    Column("tool", String(200), nullable=False),
    Column("units", Float, default=0.0),
    Column("cost_in_inr", Float, default=0.0),
    Column("client_ip", String(100), nullable=True),
    Column("user_agent", String(500), nullable=True),
    Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
)

# Create tables if they don't exist
metadata.create_all(engine)

# Constants
KID_LEN = 12


# ---------- Utility helpers ----------
def _generate_secret() -> str:
    return secrets.token_urlsafe(40)


def _generate_kid() -> str:
    # generate a short kid; trim to KID_LEN
    return secrets.token_urlsafe(KID_LEN)[:KID_LEN]


def _hash_secret(secret: str) -> str:
    return bcrypt.hashpw(secret.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_secret(secret: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(secret.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    s = secret.strip()
    if len(s) <= 6:
        # keep first 2 and last 1 if very short
        return s[:2] + "..." + s[-1:]
    return s[:2] + "..." + s[-2:]


# ---------- Public API ----------
def create_api_key_for_user(user_id: str, name: Optional[str] = None, scopes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create new API key for user_id. Returns:
      { kid, secret, preview, user_id, name, created_at }
    IMPORTANT: 'secret' is only returned here (one-time). Only store the hash in DB.
    """
    if not user_id:
        raise ValueError("user_id required")

    secret = _generate_secret()
    kid = _generate_kid()
    preview = _mask_secret(secret)
    scopes_s = json.dumps(scopes) if scopes else None
    now = datetime.utcnow()
    hashed = _hash_secret(secret)

    with engine.begin() as conn:
        ins = api_keys_table.insert().values(
            kid=kid,
            secret_hash=hashed,
            user_id=str(user_id),
            name=name,
            preview=preview,
            scopes=scopes_s,
            disabled=False,
            created_at=now
        )
        try:
            conn.execute(ins)
        except IntegrityError:
            # rare kid collision: try again once
            kid = _generate_kid()
            ins = api_keys_table.insert().values(
                kid=kid,
                secret_hash=hashed,
                user_id=str(user_id),
                name=name,
                preview=preview,
                scopes=scopes_s,
                disabled=False,
                created_at=now
            )
            conn.execute(ins)

    return {"kid": kid, "secret": secret, "preview": preview, "user_id": str(user_id), "name": name, "created_at": now.isoformat()}


def list_api_keys_for_user(user_id: str) -> List[Dict[str, Any]]:
    """
    Returns list of api keys for a user (previews only).
    Each item: { kid, preview, name, disabled, created_at }
    """
    with engine.begin() as conn:
        sel = select(
            api_keys_table.c.kid,
            api_keys_table.c.preview,
            api_keys_table.c.name,
            api_keys_table.c.disabled,
            api_keys_table.c.created_at
        ).where(api_keys_table.c.user_id == str(user_id)).order_by(api_keys_table.c.created_at.desc())
        rows = conn.execute(sel).fetchall()

    out = []
    for r in rows:
        out.append({
            "kid": r[0],
            "preview": r[1],
            "name": r[2],
            "disabled": bool(r[3]),
            "created_at": (r[4].isoformat() if r[4] else None)
        })
    return out


def get_api_key_by_kid(kid: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        sel = select(api_keys_table).where(api_keys_table.c.kid == kid)
        r = conn.execute(sel).mappings().fetchone()
    return dict(r) if r else None


def get_api_key_by_secret(secret: str) -> Optional[Dict[str, Any]]:
    """
    Finds key row by testing secret against stored bcrypt hashes.
    Iterates all rows with a hash and checks bcrypt. Suitable for small-to-moderate number of keys.
    """
    if not secret:
        return None

    with engine.begin() as conn:
        sel = select(
            api_keys_table.c.id,
            api_keys_table.c.kid,
            api_keys_table.c.secret_hash,
            api_keys_table.c.user_id,
            api_keys_table.c.disabled,
            api_keys_table.c.preview,
            api_keys_table.c.scopes
        )
        rows = conn.execute(sel).fetchall()

    for r in rows:
        _id, kid, stored_hash, user_id, disabled, preview, scopes = r
        if not stored_hash:
            continue
        if _verify_secret(secret, stored_hash):
            return {
                "id": int(_id),
                "kid": kid,
                "user_id": user_id,
                "disabled": bool(disabled),
                "preview": preview,
                "scopes": json.loads(scopes) if scopes else None
            }
    return None


def revoke_api_key(kid: str, user_id: Optional[str] = None, admin: bool = False) -> bool:
    """
    Soft-disable (mark disabled=True) the given kid.
    If user_id provided and admin=False, only revoke if owner matches.
    Returns True if updated, False if not found or not permitted.
    """
    with engine.begin() as conn:
        sel = select(api_keys_table.c.id, api_keys_table.c.user_id).where(api_keys_table.c.kid == kid)
        r = conn.execute(sel).fetchone()
        if not r:
            return False
        owner = r[1]
        if (user_id is not None) and (not admin) and (str(owner) != str(user_id)):
            return False
        upd = api_keys_table.update().where(api_keys_table.c.kid == kid).values(disabled=True)
        conn.execute(upd)
    return True


def delete_api_key_hard(kid: str) -> bool:
    """
    Hard delete the key and its usage rows. Use with caution.
    """
    with engine.begin() as conn:
        sel = select(api_keys_table.c.id).where(api_keys_table.c.kid == kid)
        r = conn.execute(sel).fetchone()
        if not r:
            return False
        api_key_id = int(r[0])
        conn.execute(api_key_usage_table.delete().where(api_key_usage_table.c.api_key_id == api_key_id))
        conn.execute(api_keys_table.delete().where(api_keys_table.c.kid == kid))
    return True


def record_usage(api_key_id: int, tool: str, units: float, cost_in_inr: float, client_ip: Optional[str] = None, user_agent: Optional[str] = None):
    """
    Append usage row. cost_in_inr should be provided by caller (INR).
    """
    with engine.begin() as conn:
        conn.execute(api_key_usage_table.insert().values(
            api_key_id=int(api_key_id),
            tool=str(tool),
            units=float(units or 0.0),
            cost_in_inr=float(cost_in_inr or 0.0),
            client_ip=client_ip,
            user_agent=user_agent,
            created_at=datetime.utcnow()
        ))


# ---------- FastAPI dependency ----------
async def validate_api_key_header(x_api_key: Optional[str] = Header(None)):
    """
    FastAPI dependency for protecting /tool/* endpoints.
    Expects header: X-API-Key: <full_secret>
    Returns api_key_info dict: { id, kid, user_id, preview, scopes } on success; raises HTTPException on failure.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    api = get_api_key_by_secret(x_api_key.strip())
    if not api:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if api.get("disabled"):
        raise HTTPException(status_code=403, detail="API key disabled")

    # return minimal metadata for downstream handlers
    return {
        "id": api.get("id"),
        "kid": api.get("kid"),
        "user_id": api.get("user_id"),
        "preview": api.get("preview"),
        "scopes": api.get("scopes")
    }


# ---------- Reporting / admin helpers ----------
def summarize_usage_by_key_since(days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Returns top keys by total cost_in_inr over last `days`.
    Output items: { kid, user_id, total_inr }
    """
    since = datetime.utcnow() - timedelta(days=days)
    with engine.begin() as conn:
        # join and aggregate
        stmt = select(
            api_keys_table.c.kid,
            api_keys_table.c.user_id,
            func.coalesce(func.sum(api_key_usage_table.c.cost_in_inr), 0).label("total_inr")
        ).select_from(
            api_keys_table.join(api_key_usage_table, api_keys_table.c.id == api_key_usage_table.c.api_key_id)
        ).where(
            api_key_usage_table.c.created_at >= since
        ).group_by(api_keys_table.c.id).order_by(func.sum(api_key_usage_table.c.cost_in_inr).desc()).limit(limit)
        rows = conn.execute(stmt).fetchall()

    out = []
    for r in rows:
        out.append({"kid": r[0], "user_id": r[1], "total_inr": float(r[2] or 0.0)})
    return out


def list_all_keys(limit: int = 1000) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        sel = select(
            api_keys_table.c.kid,
            api_keys_table.c.preview,
            api_keys_table.c.user_id,
            api_keys_table.c.name,
            api_keys_table.c.disabled,
            api_keys_table.c.created_at
        ).order_by(api_keys_table.c.created_at.desc()).limit(limit)
        rows = conn.execute(sel).fetchall()

    out = []
    for r in rows:
        out.append({
            "kid": r[0],
            "preview": r[1],
            "user_id": r[2],
            "name": r[3],
            "disabled": bool(r[4]),
            "created_at": (r[5].isoformat() if r[5] else None)
        })
    return out
