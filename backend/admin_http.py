# backend/admin_http.py
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import os

import api_keys
import auth  # reuse your auth.get_current_user dependency

router = APIRouter(prefix="/admin", tags=["admin"])


def _is_admin_user(user: dict) -> bool:
    """
    Determine if given user dict is an admin.
    Accept either explicit flags on the user dict (is_admin / is_superuser)
    or match the user's email against the ADMINS env var (comma-separated).
    """
    if not user:
        return False
    if user.get("is_admin") or user.get("is_superuser"):
        return True
    admins = [e.strip().lower() for e in os.getenv("ADMINS", "").split(",") if e.strip()]
    if admins and (user.get("email", "").lower() in admins):
        return True
    return False


def require_admin(user: dict = Depends(auth.get_current_user)):
    """
    Dependency to require admin privileges; raises 403 if not admin.
    """
    if not user or not _is_admin_user(user):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


@router.get("/overview")
def admin_overview(days: int = Query(30, ge=1), user: dict = Depends(require_admin)):
    """
    Returns summary overview:
      - total_keys
      - active_keys
      - revoked_keys
      - total_usage_in_inr (last `days`)
      - top_keys (by cost in INR)
    """
    since = datetime.utcnow() - timedelta(days=days)

    # keys summary
    try:
        keys = api_keys.list_all_keys(limit=10000)
    except Exception:
        keys = []

    total_keys = len(keys)
    active_keys = sum(1 for k in keys if not k.get("disabled"))
    revoked_keys = total_keys - active_keys

    # top keys by cost (uses helper which already aggregates)
    try:
        top = api_keys.summarize_usage_by_key_since(days=days, limit=20)
    except Exception:
        top = []

    # compute total usage in INR over last N days (query usage table directly)
    total_usage = 0.0
    try:
        from sqlalchemy import select, func
        with api_keys.engine.begin() as conn:
            s = select(func.coalesce(func.sum(api_keys.api_key_usage_table.c.cost_in_inr), 0)).where(
                api_keys.api_key_usage_table.c.created_at >= since
            )
            total_usage = float(conn.execute(s).scalar() or 0.0)
    except Exception:
        total_usage = 0.0

    return {
        "total_keys": total_keys,
        "active_keys": active_keys,
        "revoked_keys": revoked_keys,
        "total_usage_in_inr": total_usage,
        "top_keys": top,
    }


@router.get("/keys")
def admin_list_keys(user: dict = Depends(require_admin), limit: int = Query(1000, ge=1, le=5000)):
    """
    List all API keys (previews only).
    """
    try:
        rows = api_keys.list_all_keys(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list keys: {e}")
    return {"api_keys": rows}


@router.get("/usage")
def admin_usage(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    tool: Optional[str] = Query(None),
    user: dict = Depends(require_admin)
):
    """
    Returns aggregated usage grouped by day (and optionally tool).
    Query params:
      - start, end in YYYY-MM-DD (defaults: last 30 days)
      - tool: filter by tool name
    Response: { "usage": [ { day: "YYYY-MM-DD", tool: "tool_name", total_inr: 12.34, requests: 10 }, ... ] }
    """
    # parse dates
    try:
        if start:
            start_dt = datetime.fromisoformat(start)
        else:
            start_dt = datetime.utcnow() - timedelta(days=30)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start date format (use YYYY-MM-DD)")

    try:
        if end:
            end_dt = datetime.fromisoformat(end)
        else:
            end_dt = datetime.utcnow()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid end date format (use YYYY-MM-DD)")

    # Ensure start <= end
    if start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start must be <= end")

    try:
        from sqlalchemy import select, func
        # For portability, aggregate by date string (SQLite / Postgres compatible using func.date)
        with api_keys.engine.begin() as conn:
            stmt = select(
                func.date(api_keys.api_key_usage_table.c.created_at).label("day"),
                api_keys.api_key_usage_table.c.tool,
                func.coalesce(func.sum(api_keys.api_key_usage_table.c.cost_in_inr), 0).label("total_inr"),
                func.count().label("requests")
            ).where(
                api_keys.api_key_usage_table.c.created_at >= start_dt,
            ).where(
                api_keys.api_key_usage_table.c.created_at <= end_dt
            )

            if tool:
                stmt = stmt.where(api_keys.api_key_usage_table.c.tool == tool)

            stmt = stmt.group_by(func.date(api_keys.api_key_usage_table.c.created_at), api_keys.api_key_usage_table.c.tool).order_by(func.date(api_keys.api_key_usage_table.c.created_at).asc())

            rows = conn.execute(stmt).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Usage query failed: {e}")

    out = []
    for r in rows:
        # r[0] may be a date string or datetime depending on DB; normalize to str
        day_val = r[0]
        out.append({"day": str(day_val), "tool": r[1], "total_inr": float(r[2]), "requests": int(r[3])})

    return {"usage": out}


@router.post("/revoke")
def admin_revoke(payload: dict = Body(...), user: dict = Depends(require_admin)):
    """
    Admin revoke of a key.
    Body: { "kid": "<kid>" , "hard": false }
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")
    kid = (payload.get("kid") or "").strip()
    if not kid:
        raise HTTPException(status_code=400, detail="Missing kid")
    hard = bool(payload.get("hard", False))

    try:
        if hard:
            ok = api_keys.delete_api_key_hard(kid)
        else:
            ok = api_keys.revoke_api_key(kid, user_id=None, admin=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Revoke operation failed: {e}")

    if not ok:
        raise HTTPException(status_code=404, detail="Key not found or not permitted")
    return {"ok": True, "kid": kid, "hard": hard}
