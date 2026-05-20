import json
import re
import hashlib
import math


def parse_json_response(text: str) -> dict:
    """Safely parse JSON from Gemini response, stripping markdown fences if present."""
    text = text.strip()
    # Remove ```json ... ``` fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract JSON object with regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        raise ValueError(f"Could not parse JSON from model response: {e}\nRaw: {text[:300]}")


def get_risk_category(score: int) -> str:
    if score <= 30:
        return "LOW"
    elif score <= 60:
        return "MEDIUM"
    elif score <= 80:
        return "HIGH"
    else:
        return "CRITICAL"


def get_risk_action(category: str) -> str:
    actions = {
        "LOW": "ALLOW PASSAGE",
        "MEDIUM": "INSPECT",
        "HIGH": "STOP FOR WEIGHING",
        "CRITICAL": "STOP FOR WEIGHING — IMMEDIATE ACTION REQUIRED",
    }
    return actions.get(category, "INSPECT")


def risk_color(category: str) -> str:
    colors = {
        "LOW": "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH": "#f97316",
        "CRITICAL": "#ef4444",
    }
    return colors.get(category, "#6b7280")


def text_to_embedding(text: str, dim: int = 768) -> list[float]:
    """
    Deterministic pseudo-embedding from text hash.
    Used as fallback when no embedding API is available.
    This is fine for a hackathon demo — it allows Pinecone to store/retrieve vectors.
    """
    # Use SHA-256 to get consistent bytes from text
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    # Seed a simple LCG with the hash to generate `dim` floats in [-1, 1]
    seed = int.from_bytes(hash_bytes[:8], "big")
    values = []
    a, c, m = 1664525, 1013904223, 2**32
    for _ in range(dim):
        seed = (a * seed + c) % m
        values.append((seed / m) * 2 - 1)
    # L2-normalize
    magnitude = math.sqrt(sum(v * v for v in values))
    return [v / magnitude for v in values]


def format_gemini_analysis_for_embedding(analysis: dict) -> str:
    """Convert Gemini JSON output to a text string suitable for embedding."""
    signals = ", ".join(analysis.get("visible_overload_signals", []))
    return (
        f"Truck class: {analysis.get('truck_class', 'unknown')}. "
        f"Axles: {analysis.get('axle_count_estimate', '?')}. "
        f"Cargo extension: {analysis.get('cargo_extension_detected', False)}. "
        f"Overload signals: {signals}. "
        f"Risk score: {analysis.get('risk_score_raw', 0)}. "
        f"Reasoning: {analysis.get('reasoning', '')}"
    )
