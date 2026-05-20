import json

from app.utils import format_gemini_analysis_for_embedding
from app.pinecone_service import query_similar_cases


def format_retrieved_cases(matches: list[dict]) -> str:
    """
    Format Pinecone matches into compact readable text.
    """
    if not matches:
        return "No similar historical cases found."

    lines = []

    for i, match in enumerate(matches, 1):
        meta = match.get("metadata", {})
        similarity = round(match.get("score", 0) * 100, 1)

        lines.append(
            f"Case {i} ({similarity}% similar): "
            f"Truck={meta.get('truck_class', '?')} | "
            f"Axles={meta.get('axle_count', '?')} | "
            f"Risk={meta.get('risk_score', '?')} | "
            f"Outcome={meta.get('outcome', 'Unknown')}"
        )

    return "\n".join(lines)


def build_final_decision(gemini_analysis: dict) -> dict:
    """
    Deterministic risk engine.
    Much more stable than second Gemini call.
    """

    risk_score = gemini_analysis.get("risk_score_raw", 50)

    signals = gemini_analysis.get(
        "visible_overload_signals",
        []
    )

    truck_class = gemini_analysis.get(
        "truck_class",
        "unknown"
    )

    axle_count = gemini_analysis.get(
        "axle_count_estimate",
        0
    )

    # ------------------------------
    # Additional rule-based boosts
    # ------------------------------

    if len(signals) >= 3:
        risk_score += 10

    if axle_count <= 2 and truck_class == "heavy":
        risk_score += 10

    if gemini_analysis.get(
        "cargo_extension_detected",
        False
    ):
        risk_score += 10

    # Clamp score
    risk_score = max(0, min(100, risk_score))

    # ------------------------------
    # Risk categorization
    # ------------------------------

    if risk_score >= 85:
        category = "CRITICAL"
        action = "STOP FOR WEIGHING"

    elif risk_score >= 65:
        category = "HIGH"
        action = "INSPECT"

    elif risk_score >= 40:
        category = "MEDIUM"
        action = "MONITOR"

    else:
        category = "LOW"
        action = "ALLOW"

    # ------------------------------
    # Explanation
    # ------------------------------

    reasoning = (
        f"Truck classified as {truck_class} "
        f"with {axle_count} axles. "
        f"Detected overload indicators: "
        f"{', '.join(signals) if signals else 'none'}."
    )

    return {
        "final_risk_score": risk_score,
        "risk_category": category,
        "inspection_action": action,
        "reasoning": reasoning,
    }


def run_rag_pipeline(
    gemini_model,
    pinecone_index,
    gemini_analysis: dict,
    top_k: int = 2,
) -> tuple[dict, list[dict]]:
    """
    Stable RAG pipeline:
    1. Query Pinecone for similar cases
    2. Use deterministic rules for final decision
    """

    # ---------------------------------
    # Step 1: Build embedding query
    # ---------------------------------

    query_text = format_gemini_analysis_for_embedding(
        gemini_analysis
    )

    # ---------------------------------
    # Step 2: Retrieve similar cases
    # ---------------------------------

    matches = query_similar_cases(
        pinecone_index,
        query_text,
        top_k=top_k,
    )

    # ---------------------------------
    # Step 3: Build stable final decision
    # ---------------------------------

    final_decision = build_final_decision(
        gemini_analysis
    )

    return final_decision, matches