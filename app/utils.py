# === FILE: app/utils.py ===

import json
import re


def parse_json_response(text: str) -> dict:
    """Safely parse JSON from Gemini response, stripping markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*",     "", text)
    text = re.sub(r"\s*```$",     "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        raise ValueError(
            f"Could not parse JSON from model response: {e}\nRaw: {text[:300]}"
        )


def get_risk_category(score: int) -> str:
    if score <= 30:   return "LOW"
    elif score <= 60: return "MEDIUM"
    elif score <= 80: return "HIGH"
    else:             return "CRITICAL"


def get_risk_action(category: str) -> str:
    return {
        "LOW":      "ALLOW PASSAGE",
        "MEDIUM":   "INSPECT",
        "HIGH":     "STOP FOR WEIGHING",
        "CRITICAL": "STOP FOR WEIGHING — IMMEDIATE ACTION REQUIRED",
    }.get(category, "INSPECT")


def risk_color(category: str) -> str:
    return {
        "LOW":      "#22c55e",
        "MEDIUM":   "#f59e0b",
        "HIGH":     "#f97316",
        "CRITICAL": "#ef4444",
    }.get(category, "#6b7280")