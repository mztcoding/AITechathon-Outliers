# === FILE: app/decision_engine.py ===
# Deterministic rule-based decision engine.
# Replaces the Pinecone RAG layer — produces final PASS/INSPECT/STOP decisions
# by applying rule boosts on top of Gemini's raw risk score.
# No external API calls. Runs in milliseconds.

from app.utils import risk_color


# ── Historical reference cases (in-memory, no DB needed) ─────────────────────
# These are displayed as "similar cases" to the adjudicator.
# Seeded from real Pakistani highway incident patterns.

REFERENCE_CASES = [
    {
        "id": "REF-001", "truck_class": "heavy", "axle_count": 6,
        "cargo_extension": True,  "risk_score": 92,
        "overload_label": "CRITICAL",
        "location": "M-2 Motorway",      "cargo_type": "Sugarcane",
        "outcome": "Stopped — fined Rs. 50,000, cargo partially offloaded",
    },
    {
        "id": "REF-002", "truck_class": "extra-heavy", "axle_count": 8,
        "cargo_extension": True,  "risk_score": 97,
        "overload_label": "CRITICAL",
        "location": "Lahore Ring Road",   "cargo_type": "Cement mixer",
        "outcome": "Vehicle impounded, driver arrested",
    },
    {
        "id": "REF-003", "truck_class": "medium", "axle_count": 4,
        "cargo_extension": False, "risk_score": 35,
        "overload_label": "MEDIUM",
        "location": "N-55 National Hwy", "cargo_type": "Cotton bales",
        "outcome": "Allowed after inspection",
    },
    {
        "id": "REF-004", "truck_class": "light", "axle_count": 2,
        "cargo_extension": False, "risk_score": 12,
        "overload_label": "LOW",
        "location": "Karachi Port Road",  "cargo_type": "Packaged goods",
        "outcome": "Cleared immediately",
    },
    {
        "id": "REF-005", "truck_class": "heavy", "axle_count": 6,
        "cargo_extension": True,  "risk_score": 78,
        "overload_label": "HIGH",
        "location": "GT Road Rawalpindi", "cargo_type": "Bricks",
        "outcome": "Stopped, weighed, fined",
    },
    {
        "id": "REF-006", "truck_class": "extra-heavy", "axle_count": 10,
        "cargo_extension": True,  "risk_score": 99,
        "overload_label": "CRITICAL",
        "location": "Faisalabad Industrial Zone", "cargo_type": "Steel coils",
        "outcome": "Emergency stop — road closed temporarily",
    },
    {
        "id": "REF-007", "truck_class": "light", "axle_count": 2,
        "cargo_extension": False, "risk_score": 8,
        "overload_label": "LOW",
        "location": "Multan Bypass",     "cargo_type": "Vegetables",
        "outcome": "Cleared immediately",
    },
    {
        "id": "REF-008", "truck_class": "medium", "axle_count": 4,
        "cargo_extension": False, "risk_score": 55,
        "overload_label": "MEDIUM",
        "location": "M-3 Motorway",     "cargo_type": "Fertilizer bags",
        "outcome": "Directed to weigh station",
    },
]


def _match_similar_cases(gemini_analysis: dict, top_k: int = 3) -> list[dict]:
    """
    Simple rule-based similarity matching against reference cases.
    Scores each case by how closely it matches the current truck's class,
    axle count, cargo extension, and risk range. Returns top-K matches
    with a synthetic similarity percentage.
    """
    truck_class = gemini_analysis.get("truck_class", "medium")
    axle_count  = gemini_analysis.get("axle_count_estimate", 4)
    cargo_ext   = gemini_analysis.get("cargo_extension_detected", False)
    raw_score   = gemini_analysis.get("risk_score_raw", 50)

    scored = []
    for case in REFERENCE_CASES:
        sim = 0.0
        # Class match
        if case["truck_class"] == truck_class:
            sim += 0.35
        # Axle proximity (within ±2)
        axle_diff = abs(case["axle_count"] - axle_count)
        sim += max(0.0, 0.25 - axle_diff * 0.08)
        # Cargo extension match
        if case["cargo_extension"] == cargo_ext:
            sim += 0.25
        # Risk score proximity
        score_diff = abs(case["risk_score"] - raw_score)
        sim += max(0.0, 0.15 - score_diff * 0.002)

        scored.append({**case, "similarity": round(sim * 100, 1)})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def build_final_decision(gemini_analysis: dict) -> dict:
    """
    Deterministic risk decision engine.
    Applies rule-based score boosts on top of Gemini's raw risk score,
    then maps to a category and recommended action.
    """
    risk_score = gemini_analysis.get("risk_score_raw", 50)
    signals    = gemini_analysis.get("visible_overload_signals", [])
    truck_cls  = gemini_analysis.get("truck_class", "unknown")
    axle_count = gemini_analysis.get("axle_count_estimate", 0)
    cargo_ext  = gemini_analysis.get("cargo_extension_detected", False)

    # ── Rule boosts ──────────────────────────────────────────────────────────
    boost_reasons = []

    if len(signals) >= 3:
        risk_score += 10
        boost_reasons.append(f"+10 for ≥3 overload signals detected")

    if cargo_ext:
        risk_score += 10
        boost_reasons.append("+10 for cargo extending beyond chassis")

    if axle_count <= 2 and truck_cls in ("heavy", "extra-heavy"):
        risk_score += 10
        boost_reasons.append("+10 for underspecified axle count vs truck class")

    if axle_count >= 8:
        risk_score += 5
        boost_reasons.append("+5 for extra-heavy axle configuration")

    # Clamp 0–100
    risk_score = max(0, min(100, risk_score))

    # ── Categorise ───────────────────────────────────────────────────────────
    if risk_score >= 81:
        category = "CRITICAL"
        action   = "STOP FOR WEIGHING — IMMEDIATE ACTION REQUIRED"
    elif risk_score >= 61:
        category = "HIGH"
        action   = "STOP FOR WEIGHING"
    elif risk_score >= 31:
        category = "MEDIUM"
        action   = "INSPECT"
    else:
        category = "LOW"
        action   = "ALLOW PASSAGE"

    reasoning = (
        f"Truck classified as {truck_cls} with {axle_count} axle(s). "
        f"Detected overload indicators: "
        f"{', '.join(signals) if signals else 'none'}. "
        + (" Rule boosts applied: " + "; ".join(boost_reasons) if boost_reasons else "No rule boosts applied.")
    )

    return {
        "final_risk_score": risk_score,
        "risk_category":    category,
        "inspection_action": action,
        "reasoning":        reasoning,
        "boost_reasons":    boost_reasons,
    }


def run_decision_pipeline(
    gemini_analysis: dict,
    top_k: int = 3,
) -> tuple[dict, list[dict]]:
    """
    Full decision pipeline (no external calls):
    1. Match similar reference cases
    2. Apply rule engine to produce final decision
    Returns (final_decision, similar_cases)
    """
    similar_cases  = _match_similar_cases(gemini_analysis, top_k=top_k)
    final_decision = build_final_decision(gemini_analysis)
    return final_decision, similar_cases