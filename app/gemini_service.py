# === FILE: app/gemini_service.py ===
# Gemini Vision API wrapper — only called when Redis cache misses and rate limit allows.

import google.generativeai as genai
from PIL import Image
import io
import json
import re

from app.prompts import TRUCK_ANALYSIS_PROMPT
from app.utils import parse_json_response


def init_gemini(api_key: str):
    """Initialize and return Gemini generative model."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def analyze_truck_image(model, image_bytes: bytes) -> dict:
    """
    Send truck crop to Gemini Vision for overload analysis.
    Returns a structured dict. Raises on unrecoverable error.
    This function should ONLY be called after checking Redis cache
    and confirming the rate limit allows another call.
    """
    image = Image.open(io.BytesIO(image_bytes))

    response = model.generate_content(
        [TRUCK_ANALYSIS_PROMPT, image],
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text.strip()

    # ── Parse attempt 1: clean JSON ──────────────────────────────────────────
    try:
        return parse_json_response(raw_text)
    except Exception:
        pass

    # ── Parse attempt 2: fix truncated JSON by balancing braces ──────────────
    try:
        fixed = raw_text
        open_b  = fixed.count("{")
        close_b = fixed.count("}")
        if open_b > close_b:
            fixed = fixed + "}" * (open_b - close_b)
        return json.loads(fixed)
    except Exception:
        pass

    # ── Parse attempt 3: regex field extraction ───────────────────────────────
    def _extract(pattern, text, default):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    axle       = _extract(r'"axle_count_estimate"\s*:\s*(\d+)', raw_text, "4")
    truck_cls  = _extract(r'"truck_class"\s*:\s*"([^"]+)"', raw_text, "heavy")
    cargo_ext  = "true" in raw_text.lower() and "cargo_extension" in raw_text
    score      = _extract(r'"risk_score_raw"\s*:\s*(\d+)', raw_text, "50")

    signals_m  = re.search(r'"visible_overload_signals"\s*:\s*\[([^\]]*)', raw_text)
    signals    = re.findall(r'"([^"]+)"', signals_m.group(1)) if signals_m else []

    reason_m   = re.search(r'"reasoning"\s*:\s*"([^"]{10,})', raw_text)
    reasoning  = reason_m.group(1) if reason_m else "Analysis partially completed."

    return {
        "axle_count_estimate":     int(axle),
        "truck_class":             truck_cls,
        "cargo_extension_detected": cargo_ext,
        "visible_overload_signals": signals or ["Analysis truncated — manual review needed"],
        "reasoning":               reasoning,
        "risk_score_raw":          int(score),
    }