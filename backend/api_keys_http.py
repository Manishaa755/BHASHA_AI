# backend/api_keys_http.py
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Optional, Dict, Any
import api_keys  # backend/api_keys.py
import auth

router = APIRouter(tags=["api_keys"])


@router.post("/user/apikeys/create")
async def create_key(body: Dict[str, Any] = Body(None), user: dict = Depends(auth.get_current_user)):
    """
    Create a new API key for the authenticated user.
    Request body (optional): { "name": "friendly name", "scopes": ["tool_stt", ...] }
    Response includes the ONE-TIME secret: { kid, secret, preview, user_id, name, created_at }
    """
    body = body or {}
    name = (body.get("name") or "web-key")
    user_id = user.get("id") or user.get("email") or "unknown_user"
    try:
        rec = api_keys.create_api_key_for_user(
            user_id=user_id, name=name, scopes=body.get("scopes"))
        return rec
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/apikeys")
async def list_keys(user: dict = Depends(auth.get_current_user)):
    """
    List API keys (previews only) for the authenticated user.
    Response shape: { api_keys: [ { kid, preview, name, disabled, created_at }, ... ] }
    """
    user_id = user.get("id") or user.get("email") or "unknown_user"
    try:
        rows = api_keys.list_api_keys_for_user(user_id)
        return {"api_keys": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/apikeys/revoke")
async def revoke_key(payload: Dict[str, Any] = Body(None), user: dict = Depends(auth.get_current_user)):
    """
    Revoke an API key owned by the authenticated user.

    Body: { "kid": "<kid>", "hard": true|false (optional, default false) }

    - soft revoke (default): calls api_keys.revoke_api_key(kid, user_id=...)
      which marks disabled=True and the key will be inactive (still present in DB).
    - hard delete (hard=true): calls api_keys.delete_api_key_hard(kid)
      which fully removes the key and its usage records (use with caution).
    """
    payload = payload or {}
    kid = payload.get("kid")
    if not kid:
        raise HTTPException(
            status_code=400, detail="Missing 'kid' in request body")

    hard = bool(payload.get("hard", False))
    user_id = user.get("id") or user.get("email")

    try:
        if hard:
            # Attempt hard delete — only allowed for owner (revoke if owner matches)
            # delete_api_key_hard returns True if deleted
            deleted = api_keys.delete_api_key_hard(kid)
            if not deleted:
                raise HTTPException(
                    status_code=404, detail="API key not found or you don't have permission to delete it")
            return {"ok": True, "deleted": True, "hard": True}

        # Soft revoke by owner
        revoked = api_keys.revoke_api_key(
            kid=kid, user_id=user_id, admin=False)
        if not revoked:
            # Could be not found or not owner
            raise HTTPException(
                status_code=404, detail="API key not found or you don't have permission to revoke it")
        return {"ok": True, "revoked": True, "hard": False}

    except HTTPException:
        # re-raise FastAPI HTTPExceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
