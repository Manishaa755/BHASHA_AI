# backend/billing.py
# Simple billing/accounting module for Bhasha.
# - Uses same SQLite DB as auth/api_keys (API_KEYS_DB or backend/api_keys.db)
# - Provides routes: /billing/overview, /billing/topup (requires Authorization: Bearer <user token>)
# - Exposes helper charge_for_usage(user_id, amount_cents, description) to deduct credits.

import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, DateTime, Text, select, Boolean
)
from sqlalchemy.engine import Engine

# Use same DB path as other modules (auth/api_keys)
DB_PATH = os.getenv("API_KEYS_DB", "backend/api_keys.db")
ENGINE_URL = f"sqlite:///{DB_PATH}"
engine: Engine = create_engine(ENGINE_URL, connect_args={"check_same_thread": False})
metadata = MetaData()

# Accounts table: one row per user (by user_id/email)
accounts = Table(
    "billing_accounts",
    metadata,
    Column("user_id", String, primary_key=True),  # choose user_id/email as key
    Column("balance_cents", Integer, nullable=False, default=0),
    Column("created_at", DateTime, default=datetime.utcnow),
)

# Events table: ledger of topups/charges
events = Table(
    "billing_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False),
    Column("amount_cents", Integer, nullable=False),  # positive for credit, negative for charge
    Column("description", String, nullable=True),
    Column("ts", DateTime, default=datetime.utcnow),
)

# Ensure tables exist (safe to call on startup)
metadata.create_all(engine)

router = APIRouter(prefix="/billing", tags=["billing"])


# -------------------------
# Helper DB functions
# -------------------------
def _ensure_account_exists(user_id: str):
    """Create account row with zero balance if missing."""
    with engine.begin() as conn:
        # try to select
        sel = select([accounts.c.user_id]).where(accounts.c.user_id == user_id)
        r = conn.execute(sel).fetchone()
        if not r:
            ins = accounts.insert().values(user_id=user_id, balance_cents=0, created_at=datetime.utcnow())
            conn.execute(ins)


def get_balance(user_id: str) -> int:
    """Return balance in cents (int)."""
    _ensure_account_exists(user_id)
    with engine.begin() as conn:
        sel = select([accounts.c.balance_cents]).where(accounts.c.user_id == user_id)
        r = conn.execute(sel).fetchone()
        return int(r[0] or 0)


def record_event(user_id: str, amount_cents: int, description: Optional[str] = "") -> Dict[str, Any]:
    """Insert an event row and return it (id, user_id, amount_cents, description, ts)."""
    with engine.begin() as conn:
        ins = events.insert().values(user_id=user_id, amount_cents=amount_cents, description=description, ts=datetime.utcnow())
        res = conn.execute(ins)
        eid = res.inserted_primary_key[0] if res.inserted_primary_key else None
        sel = select([events]).where(events.c.id == eid)
        row = conn.execute(sel).fetchone()
    return dict(row) if row else {"id": eid, "user_id": user_id, "amount_cents": amount_cents, "description": description}


def top_up(user_id: str, amount_cents: int, description: Optional[str] = "topup") -> Dict[str, Any]:
    """Add funds to user's account and record event. Returns updated balance."""
    if amount_cents <= 0:
        raise ValueError("Top-up must be positive amount_cents")
    _ensure_account_exists(user_id)
    with engine.begin() as conn:
        # update balance
        upd_sql = accounts.update().where(accounts.c.user_id == user_id).values(balance_cents=accounts.c.balance_cents + amount_cents)
        conn.execute(upd_sql)
        # record event
        ins = events.insert().values(user_id=user_id, amount_cents=amount_cents, description=description, ts=datetime.utcnow())
        conn.execute(ins)
        # read back balance
        sel = select([accounts.c.balance_cents]).where(accounts.c.user_id == user_id)
        r = conn.execute(sel).fetchone()
        balance = int(r[0] or 0)
    return {"user_id": user_id, "balance_cents": balance}


def charge_for_usage(user_id: str, amount_cents: int, description: str = "usage") -> Dict[str, Any]:
    """
    Atomically charge the user's account.
    If insufficient funds -> raise HTTPException(status_code=402).
    Returns dict { user_id, amount_cents, balance_cents, event }
    """
    if amount_cents <= 0:
        raise ValueError("Charge amount must be positive integer (cents).")
    _ensure_account_exists(user_id)
    with engine.begin() as conn:
        # fetch current balance (FOR UPDATE semantics are not available with SQLite, but the transaction helps)
        sel = select([accounts.c.balance_cents]).where(accounts.c.user_id == user_id)
        r = conn.execute(sel).fetchone()
        balance = int(r[0] or 0)
        if balance < amount_cents:
            raise HTTPException(status_code=402, detail="Insufficient balance")
        # deduct
        upd = accounts.update().where(accounts.c.user_id == user_id).values(balance_cents=accounts.c.balance_cents - amount_cents)
        conn.execute(upd)
        # record negative event
        ins = events.insert().values(user_id=user_id, amount_cents=-int(amount_cents), description=description, ts=datetime.utcnow())
        res = conn.execute(ins)
        eid = res.inserted_primary_key[0] if res.inserted_primary_key else None
        # read back balance
        sel2 = select([accounts.c.balance_cents]).where(accounts.c.user_id == user_id)
        r2 = conn.execute(sel2).fetchone()
        new_balance = int(r2[0] or 0)
        # fetch event row
        ev = None
        if eid:
            row = conn.execute(select([events]).where(events.c.id == eid)).fetchone()
            ev = dict(row) if row else None
    return {"user_id": user_id, "amount_cents": -int(amount_cents), "balance_cents": new_balance, "event": ev}


def recent_events_for_user(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        sel = select([events]).where(events.c.user_id == user_id).order_by(events.c.ts.desc()).limit(limit)
        rows = conn.execute(sel).fetchall()
    return [dict(r) for r in rows]


# -------------------------
# FastAPI routes (require user auth)
# -------------------------
# Note: depends on auth.get_current_user to provide {"id":..., "email":...} or similar
try:
    import auth  # expects auth.get_current_user dependency
    get_current_user = auth.get_current_user
except Exception:
    # fallback: require a dependency injection in your app to pass get_current_user
    def get_current_user():
        raise HTTPException(status_code=500, detail="Auth dependency not available on server. Ensure auth.get_current_user is importable.")

@router.get("/overview")
async def overview(current_user: dict = Depends(get_current_user)):
    """
    Returns:
    {
      "user_id": "...",
      "balance_cents": 12345,
      "recent_events": [...]
    }
    """
    user_id = current_user.get("user_id") or current_user.get("id") or current_user.get("email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user identity")
    bal = get_balance(user_id)
    events = recent_events_for_user(user_id, limit=50)
    return {"user_id": user_id, "balance_cents": bal, "recent_events": events}


@router.post("/topup")
async def api_topup(body: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """
    Body: { amount_cents: int, description?: str }
    Returns updated balance.
    (For development/testing only)
    """
    user_id = current_user.get("user_id") or current_user.get("id") or current_user.get("email")
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user identity")
    amount = int(body.get("amount_cents") or 0)
    if amount <= 0:
        raise HTTPException(status_code=400, detail="amount_cents must be positive integer")
    desc = body.get("description") or "topup"
    res = top_up(user_id, amount, description=desc)
    return res


# Optional: endpoint to charge by admin (for testing)
@router.post("/charge")
async def api_charge(body: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """
    Body: { user_id: str (optional, default current user), amount_cents: int, description?: str }
    Admin or user can call to charge (if calling for other user, ensure proper check).
    """
    caller = current_user.get("user_id") or current_user.get("id") or current_user.get("email")
    target = body.get("user_id") or caller
    amount = int(body.get("amount_cents") or 0)
    if amount <= 0:
        raise HTTPException(status_code=400, detail="amount_cents must be positive")
    desc = body.get("description") or "charge"
    try:
        res = charge_for_usage(target, amount, description=desc)
        return res
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export helper functions for other modules to import
# e.g., from billing import charge_for_usage
__all__ = ["router", "charge_for_usage", "get_balance", "top_up", "record_event"]
