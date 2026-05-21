# === FILE: app/redis_service.py ===
# Redis layer: image cache, rate limiting, session history, API usage tracking.
# This is the cost-control brain of the system.

import redis
import json
import hashlib
import time
from datetime import datetime


# ── Key templates ─────────────────────────────────────────────────────────────
KEY_IMAGE_CACHE    = "freight:cache:img:{img_hash}"      # Gemini result cache
KEY_RATE_WINDOW    = "freight:rate:{minute_bucket}"      # Per-minute API counter
KEY_DAILY_COUNTER  = "freight:stats:calls:{date}"        # Daily Gemini call count
KEY_SESSION_HIST   = "freight:history:session:{sid}"     # Per-session truck log
KEY_GLOBAL_HIST    = "freight:history:global"            # Global inspection list
KEY_STATUS         = "freight:system:status"             # System health blob


def get_redis_client(host: str = "localhost", port: int = 6379, db: int = 0):
    """Return a connected Redis client, or None if unavailable."""
    try:
        r = redis.Redis(host=host, port=port, db=db,
                        decode_responses=True,
                        socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


# ── Image hash ────────────────────────────────────────────────────────────────

def compute_image_hash(image_bytes: bytes) -> str:
    """SHA-256 of raw image bytes — used as cache key."""
    return hashlib.sha256(image_bytes).hexdigest()


# ── Result cache ──────────────────────────────────────────────────────────────

def get_cached_result(r, img_hash: str) -> dict | None:
    """Return cached Gemini+decision result, or None on miss."""
    if r is None:
        return None
    key = KEY_IMAGE_CACHE.format(img_hash=img_hash)
    raw = r.get(key)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def cache_result(r, img_hash: str, result: dict, ttl_seconds: int = 86400) -> bool:
    """Store result in Redis with TTL (default 24 hours)."""
    if r is None:
        return False
    key = KEY_IMAGE_CACHE.format(img_hash=img_hash)
    try:
        r.setex(key, ttl_seconds, json.dumps(result))
        return True
    except Exception:
        return False


# ── Rate limiting ─────────────────────────────────────────────────────────────

def check_rate_limit(r, max_calls_per_minute: int = 10) -> tuple[bool, int, int]:
    """
    Returns (allowed: bool, current_count: int, limit: int).
    Uses a sliding 60-second window keyed by current minute bucket.
    """
    if r is None:
        return True, 0, max_calls_per_minute   # Redis down → allow, no tracking

    bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
    key = KEY_RATE_WINDOW.format(minute_bucket=bucket)

    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 120)           # 2-min TTL so key cleans itself up
    results = pipe.execute()

    current = results[0]
    allowed = current <= max_calls_per_minute
    return allowed, current, max_calls_per_minute


def get_current_rate_usage(r, max_calls_per_minute: int = 10) -> tuple[int, int]:
    """Return (current_calls_this_minute, limit) without incrementing."""
    if r is None:
        return 0, max_calls_per_minute
    bucket = datetime.utcnow().strftime("%Y%m%d%H%M")
    key = KEY_RATE_WINDOW.format(minute_bucket=bucket)
    val = r.get(key)
    return (int(val) if val else 0), max_calls_per_minute


# ── Daily API call counter ────────────────────────────────────────────────────

def increment_daily_counter(r) -> int:
    """Increment and return today's Gemini API call count."""
    if r is None:
        return 0
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key = KEY_DAILY_COUNTER.format(date=today)
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 86400 * 3)    # Keep 3 days
    results = pipe.execute()
    return results[0]


def get_daily_counter(r) -> int:
    """Return today's Gemini API call count without incrementing."""
    if r is None:
        return 0
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key = KEY_DAILY_COUNTER.format(date=today)
    val = r.get(key)
    return int(val) if val else 0


# ── Session + global history ──────────────────────────────────────────────────

def push_to_history(r, session_id: str, record: dict) -> None:
    """
    Append an inspection record to both session and global history lists.
    Global list is capped at 200 entries; session list at 50.
    """
    if r is None:
        return
    payload = json.dumps(record)
    s_key = KEY_SESSION_HIST.format(sid=session_id)
    g_key = KEY_GLOBAL_HIST

    pipe = r.pipeline()
    pipe.lpush(s_key, payload)
    pipe.ltrim(s_key, 0, 49)
    pipe.expire(s_key, 3600 * 4)   # Session expires in 4 hours
    pipe.lpush(g_key, payload)
    pipe.ltrim(g_key, 0, 199)
    pipe.execute()


def get_session_history(r, session_id: str) -> list[dict]:
    """Return this session's inspection records, newest first."""
    if r is None:
        return []
    key = KEY_SESSION_HIST.format(sid=session_id)
    items = r.lrange(key, 0, -1)
    records = []
    for item in items:
        try:
            records.append(json.loads(item))
        except Exception:
            pass
    return records


def get_global_history(r, count: int = 20) -> list[dict]:
    """Return last N global inspection records."""
    if r is None:
        return []
    items = r.lrange(KEY_GLOBAL_HIST, 0, count - 1)
    records = []
    for item in items:
        try:
            records.append(json.loads(item))
        except Exception:
            pass
    return records


# ── System status ─────────────────────────────────────────────────────────────

def get_redis_stats(r) -> dict:
    """Return Redis memory + key count for dashboard display."""
    if r is None:
        return {"connected": False}
    try:
        info = r.info("memory")
        return {
            "connected": True,
            "used_memory_human": info.get("used_memory_human", "?"),
            "total_keys": r.dbsize(),
        }
    except Exception:
        return {"connected": False}