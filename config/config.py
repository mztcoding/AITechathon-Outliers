# === FILE: config/config.py ===

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Max Gemini Vision calls per 60-second window.
# Keep low during demo to show the caching/rate-limit story clearly.
GEMINI_RATE_LIMIT_PER_MINUTE = int(os.getenv("GEMINI_RATE_LIMIT_PER_MINUTE", "8"))

# ── Cache ─────────────────────────────────────────────────────────────────────
# Identical images served from Redis cache for this many seconds (24 hrs default).
IMAGE_CACHE_TTL_SECONDS = int(os.getenv("IMAGE_CACHE_TTL_SECONDS", "86400"))

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "LOW":      (0,  30),
    "MEDIUM":   (31, 60),
    "HIGH":     (61, 80),
    "CRITICAL": (81, 100),
}

RISK_ACTIONS = {
    "LOW":      "ALLOW PASSAGE",
    "MEDIUM":   "INSPECT",
    "HIGH":     "STOP FOR WEIGHING",
    "CRITICAL": "STOP FOR WEIGHING — IMMEDIATE ACTION REQUIRED",
}