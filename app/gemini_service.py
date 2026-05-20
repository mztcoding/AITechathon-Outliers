import google.generativeai as genai
from PIL import Image
import io
import json
import re
from app.prompts import TRUCK_ANALYSIS_PROMPT
from app.utils import parse_json_response


def init_gemini(api_key: str):
    """Initialize and return Gemini client."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def analyze_truck_image(model, image_bytes: bytes) -> dict:
    """
    Send truck image to Gemini for visual analysis.
    Returns structured dict with truck details and raw risk score.
    """
    image = Image.open(io.BytesIO(image_bytes))

    response = model.generate_content(
        [TRUCK_ANALYSIS_PROMPT, image],
        generation_config=genai.types.GenerationConfig(
        temperature=0.1,
        max_output_tokens=2048,
        response_mime_type="application/json",
),
    )

    raw_text = response.text.strip()

    # Try normal parse first
    try:
        return parse_json_response(raw_text)
    except Exception:
        pass

    # If truncated, try to salvage by closing open JSON
    try:
        fixed = raw_text
        # Count open/close braces to detect truncation
        open_b = fixed.count("{")
        close_b = fixed.count("}")
        if open_b > close_b:
            fixed = fixed + "}" * (open_b - close_b)
        return json.loads(fixed)
    except Exception:
        pass

    # Last resort: extract whatever fields we can with regex
    def extract_field(pattern, text, default):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    axle = extract_field(r'"axle_count_estimate"\s*:\s*(\d+)', raw_text, 4)
    truck_class = extract_field(r'"truck_class"\s*:\s*"([^"]+)"', raw_text, "heavy")
    cargo_ext = "true" in raw_text.lower() and "cargo_extension" in raw_text
    score = extract_field(r'"risk_score_raw"\s*:\s*(\d+)', raw_text, 50)

    # Extract signals array if present
    signals_match = re.search(r'"visible_overload_signals"\s*:\s*\[([^\]]*)', raw_text)
    signals = []
    if signals_match:
        raw_signals = signals_match.group(1)
        signals = re.findall(r'"([^"]+)"', raw_signals)

    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]{10,})', raw_text)
    reasoning = reasoning_match.group(1) if reasoning_match else "Analysis partially completed due to response length."

    return {
        "axle_count_estimate": int(axle),
        "truck_class": truck_class,
        "cargo_extension_detected": cargo_ext,
        "visible_overload_signals": signals if signals else ["Analysis truncated — manual review needed"],
        "reasoning": reasoning,
        "risk_score_raw": int(score),
    }