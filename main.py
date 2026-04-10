from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import uuid
import random
import asyncio
import logging
import sqlite3
import threading
import secrets
from datetime import datetime, date

import httpx
import keep_alive

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_VERSION        = os.getenv("APP_VERSION", "1.0.0")
ENVIRONMENT        = os.getenv("ENVIRONMENT", "production")

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "").strip() or None
OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")

ADMIN_KEY          = os.getenv("ADMIN_KEY", "").strip() or None
REQUIRE_API_KEY    = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"

HTTP_TIMEOUT       = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
MAX_RETRIES        = int(os.getenv("UPSTREAM_MAX_RETRIES", "3"))
BACKOFF_BASE       = float(os.getenv("UPSTREAM_BACKOFF_BASE_SECONDS", "0.5"))
BACKOFF_MAX        = float(os.getenv("UPSTREAM_BACKOFF_MAX_SECONDS", "4.0"))

# Default quota: max images per key per day (0 = unlimited)
DEFAULT_DAILY_QUOTA = int(os.getenv("DEFAULT_DAILY_QUOTA", "100"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-generation-api")

# ---------------------------------------------------------------------------
# SQLite key store + usage tracking
# ---------------------------------------------------------------------------
DB_PATH  = os.getenv("DB_PATH", "image_api_keys.db")
_db_lock = threading.Lock()

def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _db_lock:
        c = _conn()
        c.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key         TEXT PRIMARY KEY,
                label       TEXT NOT NULL,
                active      INTEGER NOT NULL DEFAULT 1,
                daily_quota INTEGER NOT NULL DEFAULT 100,
                created_at  TEXT NOT NULL
            )
        """)
        # Migrate: add daily_quota if upgrading from old schema
        try:
            c.execute("ALTER TABLE api_keys ADD COLUMN daily_quota INTEGER NOT NULL DEFAULT 100")
        except Exception:
            pass

        # Usage log: one row per request
        c.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key_prefix  TEXT NOT NULL,
                full_key    TEXT NOT NULL,
                ip          TEXT,
                model       TEXT,
                size        TEXT,
                quality     TEXT,
                n           INTEGER,
                success     INTEGER NOT NULL DEFAULT 1,
                error_msg   TEXT,
                request_id  TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        c.commit()
        c.close()

def db_add_key(key: str, label: str, daily_quota: int = None) -> None:
    quota = daily_quota if daily_quota is not None else DEFAULT_DAILY_QUOTA
    with _db_lock:
        c = _conn()
        c.execute(
            "INSERT INTO api_keys (key, label, active, daily_quota, created_at) VALUES (?,?,1,?,?)",
            (key, label, quota, datetime.utcnow().isoformat())
        )
        c.commit(); c.close()

def db_valid(key: str) -> bool:
    with _db_lock:
        c = _conn()
        row = c.execute("SELECT 1 FROM api_keys WHERE key=? AND active=1", (key,)).fetchone()
        c.close()
        return row is not None

def db_revoke(key: str) -> bool:
    with _db_lock:
        c = _conn()
        cur = c.execute("UPDATE api_keys SET active=0 WHERE key=?", (key,))
        c.commit(); c.close()
        return cur.rowcount > 0

def db_list() -> list:
    with _db_lock:
        c = _conn()
        rows = c.execute(
            "SELECT key, label, active, daily_quota, created_at FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
        c.close()
        return [dict(r) for r in rows]

def db_get_daily_usage(key: str, day: str) -> int:
    """Total images successfully requested by this key on `day` (YYYY-MM-DD)."""
    with _db_lock:
        c = _conn()
        row = c.execute(
            "SELECT COALESCE(SUM(n), 0) FROM usage_log WHERE full_key=? AND success=1 AND created_at LIKE ?",
            (key, f"{day}%")
        ).fetchone()
        c.close()
        return int(row[0]) if row else 0

def db_get_quota(key: str) -> int:
    with _db_lock:
        c = _conn()
        row = c.execute("SELECT daily_quota FROM api_keys WHERE key=?", (key,)).fetchone()
        c.close()
        return int(row["daily_quota"]) if row else DEFAULT_DAILY_QUOTA

def db_log_usage(key: str, ip: str, model: str, size: str, quality: str,
                 n: int, success: bool, request_id: str, error_msg: str = None) -> None:
    with _db_lock:
        c = _conn()
        c.execute(
            """INSERT INTO usage_log
               (key_prefix, full_key, ip, model, size, quality, n, success, error_msg, request_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (key[:8] + "••••••••", key, ip, model, size, quality, n,
             1 if success else 0, error_msg, request_id, datetime.utcnow().isoformat())
        )
        c.commit(); c.close()

def db_usage_summary(key: str = None, days: int = 7) -> list:
    with _db_lock:
        c = _conn()
        if key:
            rows = c.execute(
                """SELECT DATE(created_at) as day,
                          COUNT(*) as requests,
                          COALESCE(SUM(n), 0) as images,
                          SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as errors
                   FROM usage_log
                   WHERE full_key=? AND created_at >= DATE('now', ? || ' days')
                   GROUP BY day ORDER BY day DESC""",
                (key, f"-{days}")
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT DATE(created_at) as day,
                          COUNT(*) as requests,
                          COALESCE(SUM(n), 0) as images,
                          SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as errors
                   FROM usage_log
                   WHERE created_at >= DATE('now', ? || ' days')
                   GROUP BY day ORDER BY day DESC""",
                (f"-{days}",)
            ).fetchall()
        c.close()
        return [dict(r) for r in rows]

init_db()

def bootstrap_keys():
    raw = os.getenv("BOOTSTRAP_KEYS", "").strip()
    if not raw:
        return
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 1)
        key   = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else "bootstrapped"
        with _db_lock:
            c = _conn()
            exists = c.execute("SELECT 1 FROM api_keys WHERE key=?", (key,)).fetchone()
            if not exists:
                c.execute(
                    "INSERT INTO api_keys (key, label, active, daily_quota, created_at) VALUES (?,?,1,?,?)",
                    (key, label, DEFAULT_DAILY_QUOTA, datetime.utcnow().isoformat())
                )
                c.commit()
            c.close()

bootstrap_keys()

# ---------------------------------------------------------------------------
# Rate limiter — IP-level backstop only
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="Image Generation API",
    version=APP_VERSION,
    description="Image generation powered by DALL-E 3 via OpenAI.",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Rate limit exceeded."})

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
api_key_header   = APIKeyHeader(name="X-API-Key",   auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

async def require_api_key(key: str = Depends(api_key_header)) -> str:
    if not REQUIRE_API_KEY:
        return ""
    if not key or not db_valid(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key

async def require_admin_key(request: Request):
    key = request.headers.get("x-admin-key") or request.headers.get("X-Admin-Key")
    if not ADMIN_KEY:
        raise HTTPException(status_code=503, detail="Admin key not configured on this server.")
    if not key or key.strip() != ADMIN_KEY.strip():
        raise HTTPException(status_code=401, detail="Invalid or missing admin key.")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ImageRequest(BaseModel):
    prompt:  str = Field(..., min_length=1, max_length=4000)
    model:   Optional[str] = None
    size:    str = Field(default="1024x1024", pattern="^(1024x1024|1792x1024|1024x1792)$")
    n:       int = Field(default=1, ge=1, le=4)
    quality: str = Field(default="standard", pattern="^(standard|hd)$")

class ImageItem(BaseModel):
    url:      Optional[str] = None
    b64_json: Optional[str] = None

class ImageResponse(BaseModel):
    provider:   str
    model:      str
    images:     List[ImageItem]
    request_id: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _backoff(attempt: int) -> float:
    exp = BACKOFF_BASE * (2 ** (attempt - 1))
    return min(BACKOFF_MAX, exp + random.uniform(0, min(0.25, exp / 4)))

def _retryable(code: int) -> bool:
    return code in {408, 425, 429, 500, 502, 503, 504}

async def _post(client: httpx.AsyncClient, url: str, headers: dict, body: dict) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client.request("POST", url, headers=headers, json=body)
            if r.status_code >= 400:
                if _retryable(r.status_code) and attempt < MAX_RETRIES:
                    await asyncio.sleep(_backoff(attempt)); continue
                raise HTTPException(status_code=r.status_code,
                                    detail=f"Upstream error from OpenAI: {r.text[:300]}")
            return r.json()
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
            if attempt >= MAX_RETRIES:
                raise HTTPException(status_code=504, detail="Timeout from OpenAI image service")
            await asyncio.sleep(_backoff(attempt))
    raise HTTPException(status_code=502, detail="Failed to reach OpenAI") from last_err

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
async def _generate_image(client: httpx.AsyncClient, payload: ImageRequest) -> ImageResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    model = payload.model or OPENAI_IMAGE_MODEL
    body  = {"model": model, "prompt": payload.prompt, "size": payload.size,
             "n": payload.n, "quality": payload.quality}
    data  = await _post(client, f"{OPENAI_BASE_URL}/images/generations",
                        {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, body)
    images = [ImageItem(url=i.get("url"), b64_json=i.get("b64_json")) for i in data.get("data", [])]
    return ImageResponse(provider="openai", model=model, images=images, request_id=str(uuid.uuid4()))

# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------
@app.post("/v1/keys/generate")
async def generate_key(
    request: Request,
    label: str,
    daily_quota: int = Query(default=None, description="Max images/day (0 = unlimited). Defaults to DEFAULT_DAILY_QUOTA env var.")
):
    await require_admin_key(request)
    key   = "img-" + secrets.token_urlsafe(32)
    quota = daily_quota if daily_quota is not None else DEFAULT_DAILY_QUOTA
    db_add_key(key, label, quota)
    return {"label": label, "api_key": key, "daily_quota": quota,
            "note": "Store securely — cannot be retrieved again."}

@app.get("/v1/keys")
async def list_keys(request: Request):
    await require_admin_key(request)
    keys = db_list()
    for k in keys:
        k["key"] = k["key"][:8] + "••••••••"
    return {"count": len(keys), "keys": keys}

@app.delete("/v1/keys/revoke")
async def revoke_key(request: Request, key: str):
    await require_admin_key(request)
    if not db_revoke(key):
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"revoked": True, "key": key[:8] + "••••••••"}

@app.patch("/v1/keys/quota")
async def update_quota(request: Request, key: str, daily_quota: int):
    """Update the daily image quota for a specific API key."""
    await require_admin_key(request)
    with _db_lock:
        c = _conn()
        cur = c.execute("UPDATE api_keys SET daily_quota=? WHERE key=?", (daily_quota, key))
        c.commit(); c.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"updated": True, "key": key[:8] + "••••••••", "daily_quota": daily_quota}

# ---------------------------------------------------------------------------
# Usage / reporting endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/usage")
async def get_usage(
    request: Request,
    days: int = Query(default=7, ge=1, le=90),
    key: Optional[str] = Query(default=None, description="Filter by a specific API key"),
):
    """Daily usage summary for the past N days."""
    await require_admin_key(request)
    return {"days": days, "summary": db_usage_summary(key=key, days=days)}

@app.get("/v1/usage/keys")
async def get_usage_by_key(
    request: Request,
    days: int = Query(default=7, ge=1, le=90),
):
    """Usage totals broken down per API key for the past N days."""
    await require_admin_key(request)
    with _db_lock:
        c = _conn()
        rows = c.execute(
            """SELECT key_prefix,
                      COUNT(*) as requests,
                      COALESCE(SUM(n), 0) as images,
                      SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as errors,
                      MAX(created_at) as last_used
               FROM usage_log
               WHERE created_at >= DATE('now', ? || ' days')
               GROUP BY key_prefix
               ORDER BY images DESC""",
            (f"-{days}",)
        ).fetchall()
        c.close()
    return {"days": days, "by_key": [dict(r) for r in rows]}

# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "environment": ENVIRONMENT, "version": APP_VERSION,
            "providers": {"openai": bool(OPENAI_API_KEY)}}

@app.get("/ready")
async def ready():
    checks = {"openai_configured": bool(OPENAI_API_KEY)}
    return {"ready": all(checks.values()), "checks": checks}

@app.post("/v1/image/generate", response_model=ImageResponse)
@limiter.limit("60/minute")   # IP-level backstop
async def image_generate(
    request: Request,
    payload: ImageRequest,
    api_key: str = Depends(require_api_key),
):
    model      = payload.model or OPENAI_IMAGE_MODEL
    request_id = str(uuid.uuid4())
    client_ip  = request.client.host if request.client else "unknown"

    # ── Per-key daily quota enforcement ──────────────────────────────────────
    if REQUIRE_API_KEY and api_key:
        quota = db_get_quota(api_key)
        if quota > 0:                          # 0 = unlimited
            today = date.today().isoformat()
            used  = db_get_daily_usage(api_key, today)
            if used + payload.n > quota:
                remaining = max(0, quota - used)
                logger.warning("Quota exceeded key=%s used=%d quota=%d requested=%d",
                               api_key[:8], used, quota, payload.n)
                raise HTTPException(
                    status_code=429,
                    detail=(f"Daily quota exceeded. Quota: {quota} images/day. "
                            f"Used today: {used}. Remaining: {remaining}.")
                )

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            result = await _generate_image(client, payload)
        result.request_id = request_id

        if REQUIRE_API_KEY and api_key:
            db_log_usage(key=api_key, ip=client_ip, model=model,
                         size=payload.size, quality=payload.quality,
                         n=payload.n, success=True, request_id=request_id)

        logger.info("Generated %d image(s) key=%s model=%s size=%s ip=%s id=%s",
                    payload.n, api_key[:8] if api_key else "anon",
                    model, payload.size, client_ip, request_id)
        return result

    except HTTPException as exc:
        if REQUIRE_API_KEY and api_key:
            db_log_usage(key=api_key, ip=client_ip, model=model,
                         size=payload.size, quality=payload.quality,
                         n=payload.n, success=False, request_id=request_id,
                         error_msg=str(exc.detail))
        raise
        
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive.keep_alive())