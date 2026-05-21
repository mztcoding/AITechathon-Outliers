"""
core/gemini_client.py
Gemini Vision API client — the "premium reasoning layer" of FreightGuard AI.

STRICT RULES (enforced in code):
1. Only called for CONFIRMED, UNIQUE, HIGH-QUALITY truck crops
2. Rate-limited via Redis token bucket
3. Results cached in Redis — never re-calls for known truck IDs
4. Only CROPPED images sent (not full frames)
5. Images resized to ≤512px before sending
6. Structured JSON output enforced via prompt engineering

SDK: Uses google-genai (new SDK). Install with:
  pip install google-genai

Cost model:
  Gemini 2.0 Flash: ~$0.000315/image
  With our filters: ~5-10 calls/min max vs 1800/min naive = 99.7% savings
"""

import json
import time
import re
from typing import Optional
from dataclasses import dataclass
from loguru import logger

# New google-genai SDK (replaces deprecated google.generativeai)
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    google_genai = None   # type: ignore
    genai_types = None    # type: ignore
    _GENAI_AVAILABLE = False

from PIL import Image
import io
import numpy as np
import cv2

from config.settings import config
from core.redis_client import redis_client


@dataclass
class GeminiResult:
    """Structured output from Gemini analysis."""
    risk_level: str          # "low" | "medium" | "high"
    confidence: float        # 0.0 - 1.0
    signals: list            # Observed visual signals
    explanation: str         # Human-readable explanation
    raw_response: str = ""
    error: Optional[str] = None
    was_cached: bool = False
    truck_id: str = ""


# System prompt — carefully engineered to get reliable JSON output
GEMINI_SYSTEM_PROMPT = """You are an expert freight inspection AI assistant for highway authorities.
Your task is to analyze images of trucks and assess overloading risk.

CRITICAL: You MUST respond with ONLY a valid JSON object. No markdown, no explanation outside JSON.

Analyze the truck image for these overloading signals:
- Suspension compression (rear axle sagging)
- Tire bulging or excessive flattening
- Cargo visible above cab height
- Visible overloading of flatbed/trailer
- Uneven load distribution (truck tilting)
- Smoke/exhaust indicating engine strain
- Visible trailer frame stress
- License plate visibility (obscured plates = suspicious)
- Overall vehicle condition

Respond with EXACTLY this JSON structure:
{
  "risk_level": "low|medium|high",
  "confidence": <float 0.0-1.0>,
  "signals": ["<signal1>", "<signal2>"],
  "explanation": "<1-2 sentence explanation for inspector>"
}

Risk level definitions:
- "low": No significant overloading indicators visible
- "medium": Some suspicious indicators — warrants attention
- "high": Clear overloading signals — priority inspection needed
"""


class GeminiClient:
    """
    Cost-optimized Gemini Vision client for truck analysis.
    Uses the new google-genai SDK.
    All calls go through Redis cache check first.
    """

    def __init__(self):
        self._configured = False
        self._client = None
        self._setup()

    def _setup(self):
        """Initialize Gemini API client using new google-genai SDK.
        Safe to call multiple times — re-initializes if a key becomes available.
        """
        if not _GENAI_AVAILABLE:
            logger.warning(
                "⚠️  google-genai not installed. Run: pip install google-genai\n"
                "    (The old google.generativeai package is deprecated.)"
            )
            return

        # Always re-read config so sidebar key injection works
        api_key = config.gemini.api_key
        if not api_key:
            logger.warning("⚠️  No GEMINI_API_KEY set. Gemini calls will be simulated.")
            self._configured = False
            self._client = None
            return

        try:
            self._client = google_genai.Client(api_key=api_key)
            self._configured = True
            logger.info(f"✅ Gemini (google-genai SDK) configured: {config.gemini.model}")
        except Exception as e:
            logger.error(f"Gemini setup failed: {e}")
            self._configured = False

    def analyze_truck(
        self,
        truck_id: str,
        crop_image: np.ndarray,
        force: bool = False,
    ) -> GeminiResult:
        """
        Analyze a truck crop image for overloading risk.

        Flow:
        1. Check Redis cache → return cached result if exists
        2. Check rate limit → skip if exceeded
        3. Call Gemini API
        4. Parse and validate response
        5. Cache result in Redis
        6. Return result
        """
        # Step 1: Cache check
        if not force:
            cached = redis_client.get_truck_result(truck_id)
            if cached:
                logger.debug(f"✅ Cache hit for truck {truck_id} — no Gemini call")
                redis_client.increment_stat("gemini_skipped")
                return GeminiResult(
                    truck_id=truck_id,
                    was_cached=True,
                    **{k: cached[k] for k in
                       ["risk_level", "confidence", "signals", "explanation"]},
                )

        # Step 2: Rate limit check
        if not redis_client.check_and_consume_rate_limit():
            logger.warning(f"🚦 Rate limited — skipping Gemini for truck {truck_id}")
            redis_client.increment_stat("gemini_skipped")
            return self._fallback_result(truck_id, "Rate limited")

        # Step 3: API call (or simulation)
        if not self._configured:
            return self._simulate_result(truck_id, crop_image)

        result = self._call_gemini(truck_id, crop_image)
        logger.warning(f"🔍 DEBUG analyze_truck result: error={result.error} risk={result.risk_level} conf={result.confidence}")
        
        # Step 4: Cache successful results
        if result.error is None:
            cache_data = {
                "risk_level": result.risk_level,
                "confidence": result.confidence,
                "signals": result.signals,
                "explanation": result.explanation,
            }
            redis_client.cache_truck_result(truck_id, cache_data)
            redis_client.increment_stat("gemini_calls")
            redis_client.increment_stat(f"{result.risk_level}_risk_count")

        return result

    def _call_gemini(self, truck_id: str, crop_image: np.ndarray) -> GeminiResult:
        """Execute Gemini API call using new google-genai SDK."""
        for attempt in range(config.gemini.max_retries):
            try:
                # Convert numpy BGR → PIL RGB → resize → JPEG bytes
                pil_img = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
                max_dim = 512
                if max(pil_img.size) > max_dim:
                    pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

                # Encode as JPEG bytes for the new SDK
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                image_bytes = buf.getvalue()

                # Build content parts for new SDK
                image_part = genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )

                response = self._client.models.generate_content(
                    model=config.gemini.model,
                    contents=[GEMINI_SYSTEM_PROMPT, image_part],
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                        response_mime_type="application/json",
                    ),
                )

                raw_text = response.text.strip()
                logger.debug(f"Gemini raw for truck {truck_id}: {raw_text[:200]}")

                parsed = self._parse_response(raw_text)
                if parsed:
                    return GeminiResult(
                        truck_id=truck_id,
                        raw_response=raw_text,
                        **parsed,
                    )
                else:
                    logger.warning(f"Failed to parse Gemini response for truck {truck_id}")
                    return self._fallback_result(truck_id, "Parse error")

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {wait}s")
                if attempt < config.gemini.max_retries - 1:
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini failed after {config.gemini.max_retries} attempts")
                    return self._fallback_result(truck_id, str(e))

        return self._fallback_result(truck_id, "Max retries exceeded")

    def _parse_response(self, raw_text: str) -> Optional[dict]:
        """Parse and validate Gemini JSON response."""
        cleaned = re.sub(r"```json|```", "", raw_text).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return None
            else:
                return None

        required = {"risk_level", "confidence", "signals", "explanation"}
        if not required.issubset(data.keys()):
            logger.warning(f"Missing fields in response: {required - set(data.keys())}")
            return None

        risk_level = str(data.get("risk_level", "medium")).lower()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "medium"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        signals = data.get("signals", [])
        if not isinstance(signals, list):
            signals = [str(signals)]

        explanation = str(data.get("explanation", "No explanation provided."))[:500]

        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "signals": signals,
            "explanation": explanation,
        }

    def _fallback_result(self, truck_id: str, reason: str) -> GeminiResult:
        """
        Conservative fallback when Gemini is unavailable.
        Returns medium risk with error flag so decision engine
        uses local heuristics only (which still produce a valid score).
        """
        return GeminiResult(
            truck_id=truck_id,
            risk_level="medium",
            confidence=0.3,
            signals=["Gemini API unavailable — local heuristics applied"],
            explanation=f"Gemini unavailable ({reason}). Score based on YOLO + tracking signals. Manual check recommended.",
            error=reason,
        )

    def _simulate_result(self, truck_id: str, crop_image: np.ndarray) -> GeminiResult:
        """
        Simulate Gemini response for demo mode (no API key configured).
        Uses image features to produce a semi-realistic distribution.
        """
        import random
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std = float(np.std(gray))

        seed = int(mean_brightness * 100 + std * 10) % 1000
        rng = random.Random(seed)

        risk_options = [
            ("low",    0.85, ["No visible overloading", "Normal suspension height", "Clear license plate"],
             "Truck appears within legal load limits."),
            ("medium", 0.70, ["Slight suspension compression", "Load visible at trailer edges"],
             "Moderate indicators detected. Recommend standard check."),
            ("high",   0.92, ["Severe axle sagging", "Cargo exceeds cab height", "Rear tire bulging"],
             "Clear overloading signals. Priority inspection required."),
        ]

        weights = [0.45, 0.35, 0.20]
        chosen = rng.choices(risk_options, weights=weights, k=1)[0]
        risk_level, confidence, signals, explanation = chosen

        result = GeminiResult(
            truck_id=truck_id,
            risk_level=risk_level,
            confidence=min(1.0, max(0.0, confidence + rng.uniform(-0.08, 0.08))),
            signals=signals,
            explanation=explanation,
        )

        redis_client.cache_truck_result(truck_id, {
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "signals": result.signals,
            "explanation": result.explanation,
        })
        redis_client.increment_stat("gemini_calls")
        redis_client.increment_stat(f"{result.risk_level}_risk_count")

        logger.info(f"🎭 Simulated Gemini for truck {truck_id}: {risk_level} ({confidence:.2f})")
        return result


# Singleton
gemini_client = GeminiClient()
