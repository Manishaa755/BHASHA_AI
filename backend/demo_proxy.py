# demo_proxy.py
import os
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import asyncio

router = APIRouter()

# Put full Bhasha secret into DEMO_BHASHA_KEY on server (do NOT commit).
BHASHA_KEY = os.getenv("DEMO_BHASHA_KEY")
# Where your Bhasha app (local) is accessible; usually the same host:port
BHASHA_API_BASE = os.getenv("BHASHA_API_BASE", "http://127.0.0.1:8000").rstrip('/')

if not BHASHA_KEY:
    # Keep this print for local debugging; on production you may want to log differently.
    print("⚠️ DEMO proxy: DEMO_BHASHA_KEY not set — proxy will return 500")


async def _safe_response_text(resp: httpx.Response) -> str:
    """
    Safely obtain a textual representation of an httpx.Response body without
    assuming `.text` is awaitable or callable. httpx.Response.text is a property,
    so calling or awaiting it will cause TypeError in some cases.
    """
    text_attr = getattr(resp, "text", None)
    try:
        # If it's awaitable (coroutine) -> await it
        if asyncio.iscoroutine(text_attr):
            return await text_attr
        # If it's callable (function) -> call it, and await if needed
        if callable(text_attr):
            maybe = text_attr()
            if asyncio.iscoroutine(maybe):
                return await maybe
            return str(maybe)
        # Otherwise assume it's a string property
        if text_attr is not None:
            return str(text_attr)
    except Exception:
        # Fall through to reading content
        pass

    # Fallback: return decoded bytes content (best-effort)
    try:
        content = resp.content
        if isinstance(content, (bytes, bytearray)):
            return content.decode(errors="ignore")
        return str(content)
    except Exception:
        return "<unreadable response body>"


@router.post("/demo/s2s")
async def demo_s2s(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and proxy it to the internal BHASHA /tool/s2s endpoint.
    Returns streamed audio from the upstream, and forwards X-Reply-Text-B64 header if present.
    """
    if not BHASHA_KEY:
        raise HTTPException(status_code=500, detail="Server not configured (DEMO_BHASHA_KEY missing)")

    # Read incoming file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    files = {
        "file": (file.filename or "recording.webm", file_bytes, file.content_type or "audio/webm")
    }

    url = f"{BHASHA_API_BASE}/tool/s2s"
    headers = {"X-API-Key": BHASHA_KEY}

    # Use a reasonable timeout; adjust if needed
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            r = await client.post(url, headers=headers, files=files)
        except httpx.RequestError as e:
            # Network /connectivity errors
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Unexpected upstream error: {e}")

    # If upstream returned an error status, include upstream body safely in the detail
    if r.status_code >= 400:
        upstream_body = await _safe_response_text(r)
        # Return structured detail for easier debugging
        raise HTTPException(
            status_code=r.status_code,
            detail={
                "upstream_status": r.status_code,
                "upstream_body": upstream_body,
                "upstream_headers": dict(r.headers),
            }
        )

    # Success: stream back audio content, and include X-Reply-Text-B64 header if present
    content_type = r.headers.get("content-type", "audio/mpeg")
    reply_b64 = r.headers.get("X-Reply-Text-B64")
    headers_out: Dict[str, str] = {}
    if reply_b64:
        headers_out["X-Reply-Text-B64"] = reply_b64

    # StreamingResponse expects an iterator/generator of bytes. Use a simple iterator for small responses,
    # or use .aiter_bytes() for larger upstream responses if available.
    async def upstream_iter():
        # httpx.Response supports aiter_bytes for async streaming
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        except AttributeError:
            # Fallback: yield complete content once
            yield r.content

    return StreamingResponse(upstream_iter(), media_type=content_type, headers=headers_out)
