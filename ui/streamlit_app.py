import streamlit as st
import sys
import os
import time
# ─── Path setup so imports work when run from project root ─────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME,
    PINECONE_DIMENSION, TOP_K_RESULTS,
)
from app.gemini_service import init_gemini, analyze_truck_image
from app.pinecone_service import init_pinecone, upsert_historical_cases, check_index_populated
from app.rag_service import run_rag_pipeline
from app.utils import get_risk_category, get_risk_action, risk_color
# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreightGuard AI — Pakistan Truck Inspection",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
/* Dark industrial theme */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.header-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0;
    line-height: 1.2;
}
.header-sub {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 4px;
}
.pk-badge {
    background: #1e3a5f;
    color: #58a6ff;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #388bfd40;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-top: 8px;
}
/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.metric-label {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 8px;
}
/* Action banner */
.action-banner {
    border-radius: 10px;
    padding: 18px 24px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-align: center;
    text-transform: uppercase;
    margin: 16px 0;
}
/* Section headers */
.section-header {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}
/* Signal tags */
.signal-tag {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    color: #d1d5db;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px 4px 3px 0;
}
/* Case card */
.case-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #388bfd;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.case-card-header {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 6px;
}
.case-card-meta {
    font-size: 0.80rem;
    color: #8b949e;
    line-height: 1.7;
}
/* Progress bar track */
.risk-bar-track {
    background: #21262d;
    border-radius: 8px;
    height: 14px;
    overflow: hidden;
    margin: 10px 0;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}
/* Reasoning box */
.reasoning-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.7;
    font-family: 'JetBrains Mono', monospace;
}
/* Streamlit overrides */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px);
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
    background: #161b22 !important;
}
.stSpinner > div {
    border-top-color: #58a6ff !important;
}
</style>
""", unsafe_allow_html=True)
# ─── Session state ───────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "gemini_model": None,
        "pinecone_index": None,
        "initialized": False,
        "analysis_done": False,
        "gemini_result": None,
        "final_decision": None,
        "matches": None,
        "image_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session()

# ─── Auto-initialize on startup ──────────────────────────────────────────────
if not st.session_state["initialized"]:
    gemini_key = GEMINI_API_KEY
    pinecone_key = PINECONE_API_KEY

    if not gemini_key or not pinecone_key:
        st.error("❌ API keys missing. Please set GEMINI_API_KEY and PINECONE_API_KEY in your .env file.")
        st.stop()

    with st.spinner("🔗 Connecting to Gemini..."):
        try:
            model = init_gemini(gemini_key)
            st.session_state["gemini_model"] = model
        except Exception as e:
            st.error(f"Gemini error: {e}")
            st.stop()

    with st.spinner("🔗 Connecting to Pinecone..."):
        try:
            index = init_pinecone(pinecone_key, PINECONE_INDEX_NAME, PINECONE_DIMENSION)
            st.session_state["pinecone_index"] = index
        except Exception as e:
            st.error(f"Pinecone error: {e}")
            st.stop()

    with st.spinner("🌱 Seeding historical cases..."):
        try:
            idx = st.session_state["pinecone_index"]
            if not check_index_populated(idx):
                upsert_historical_cases(idx)
        except Exception as e:
            st.warning(f"Seed warning: {e}")

    st.session_state["initialized"] = True

# ─── Sidebar — Status only ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
    if st.session_state["initialized"]:
        st.markdown("🟢 Gemini Vision: Active")
        st.markdown("🟢 Pinecone RAG: Active")
        st.markdown("🟢 Historical DB: Loaded")
    else:
        st.markdown("🔴 Not initialized")
    st.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#8b949e;line-height:1.7">
    <b>Risk Categories</b><br>
    🟢 LOW (0–30): Allow Passage<br>
    🟡 MEDIUM (31–60): Inspect<br>
    🟠 HIGH (61–80): Stop for Weighing<br>
    🔴 CRITICAL (81–100): Immediate Action
    </div>
    """, unsafe_allow_html=True)

# ─── Main Content ─────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div class="header-banner">
    <div style="font-size:3rem">🚛</div>
    <div>
        <div class="header-title">FreightGuard AI</div>
        <div class="header-sub">AI-Assisted Freight Inspection Prioritization System</div>
        <div class="pk-badge">🇵🇰 PAKISTAN MOTORWAY POLICE · NATIONAL HIGHWAY AUTHORITY</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state["initialized"]:
    st.info("⏳ System is initializing, please wait...")
    st.stop()

# ─── Upload section ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📸 Upload Truck Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a truck photo for inspection",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
if uploaded_file:
    image_bytes = uploaded_file.read()
    st.session_state["image_bytes"] = image_bytes
    col_img, col_analyze = st.columns([3, 1])
    with col_img:
        st.image(image_bytes, caption="Uploaded truck image", use_container_width=True)
    with col_analyze:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Truck", use_container_width=True)
    if analyze_btn:
        # ── Step 1: Gemini Analysis ──────────────────────────────────────────
        with st.spinner("🤖 Gemini is analyzing the truck image..."):
            try:
                gemini_result = analyze_truck_image(
                    st.session_state["gemini_model"], image_bytes
                )
                st.session_state["gemini_result"] = gemini_result
            except Exception as e:
                st.error(f"Gemini analysis failed: {e}")
                st.stop()
        # ── Step 2: RAG Pipeline ────────────────────────────────────────────
        with st.spinner("🗄️ Retrieving similar historical cases from Pinecone..."):
            try:
                final_decision, matches = run_rag_pipeline(
                    st.session_state["gemini_model"],
                    st.session_state["pinecone_index"],
                    gemini_result,
                    top_k=TOP_K_RESULTS,
                )
                st.session_state["final_decision"] = final_decision
                st.session_state["matches"] = matches
                st.session_state["analysis_done"] = True
            except Exception as e:
                st.error(f"RAG pipeline failed: {e}")
                st.stop()
        st.rerun()
# ─── Results Dashboard ────────────────────────────────────────────────────────
if st.session_state["analysis_done"] and st.session_state["final_decision"]:
    gres = st.session_state["gemini_result"]
    fdec = st.session_state["final_decision"]
    matches = st.session_state["matches"]
    risk_score = fdec.get("final_risk_score", gres.get("risk_score_raw", 0))
    risk_cat = fdec.get("risk_category", get_risk_category(risk_score))
    action = fdec.get("inspection_action", get_risk_action(risk_cat))
    explanation = fdec.get("explanation", "No explanation generated.")
    key_signals = fdec.get("key_signals", gres.get("visible_overload_signals", []))
    hist_summary = fdec.get("historical_match_summary", "")
    color = risk_color(risk_cat)
    st.markdown("---")
    # ── Top KPI Row ──────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value" style="color:{color}">{risk_score}</div>
            <div style="color:#8b949e;font-size:0.78rem;margin-top:4px">out of 100</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Category</div>
            <div class="risk-badge" style="background:{color}20;color:{color};border:1px solid {color}40;margin-top:4px">
                {risk_cat}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Truck Class</div>
            <div class="metric-value" style="font-size:1.4rem;color:#f0f6fc;margin-top:4px">
                {gres.get("truck_class", "?").upper()}
            </div>
            <div style="color:#8b949e;font-size:0.78rem;margin-top:4px">{gres.get("axle_count_estimate", "?")} axles detected</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        cargo_ext = gres.get("cargo_extension_detected", False)
        ext_color = "#ef4444" if cargo_ext else "#22c55e"
        ext_label = "YES — EXCEEDS CHASSIS" if cargo_ext else "NO — WITHIN BOUNDS"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cargo Extension</div>
            <div style="color:{ext_color};font-size:0.85rem;font-weight:600;margin-top:8px">{ext_label}</div>
        </div>
        """, unsafe_allow_html=True)
    # ── Risk Score Bar ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin:8px 0 4px 0">
        <div style="display:flex;justify-content:space-between;color:#8b949e;font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em">
            <span>LOW</span><span>MEDIUM</span><span>HIGH</span><span>CRITICAL</span>
        </div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{risk_score}%;background:linear-gradient(90deg, #22c55e, #f59e0b, #f97316, #ef4444)"></div>
        </div>
        <div style="text-align:right;color:#8b949e;font-size:0.75rem">Score: {risk_score}/100</div>
    </div>
    """, unsafe_allow_html=True)
    # ── Action Banner ─────────────────────────────────────────────────────────
    action_icon = {"ALLOW PASSAGE": "✅", "INSPECT": "⚠️", "STOP FOR WEIGHING": "🛑"}.get(action, "🛑")
    st.markdown(f"""
    <div class="action-banner" style="background:{color}15;border:2px solid {color};color:{color}">
        {action_icon} INSPECTOR ACTION: {action}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # ── Two column layout: Signals + Cases ───────────────────────────────────
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown('<div class="section-header">🔬 Visual Analysis (Gemini Vision)</div>', unsafe_allow_html=True)
        # Key signals
        st.markdown("**Detected Overload Signals:**")
        if key_signals:
            tags = "".join([f'<span class="signal-tag">⚠ {s}</span>' for s in key_signals])
            st.markdown(f'<div style="margin:8px 0 16px 0">{tags}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="signal-tag">✓ No significant signals detected</span>', unsafe_allow_html=True)
        st.markdown("**Gemini Reasoning:**")
        reasoning = gres.get("reasoning", "No reasoning provided.")
        st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)
        st.markdown("**Final Explanation:**")
        st.markdown(f'<div class="reasoning-box" style="border-color:{color}40">{explanation}</div>', unsafe_allow_html=True)
        if hist_summary:
            st.markdown("**Historical Case Pattern:**")
            st.markdown(f'<div class="reasoning-box" style="border-color:#388bfd40;color:#58a6ff">{hist_summary}</div>', unsafe_allow_html=True)
    with right_col:
        st.markdown(f'<div class="section-header">🗄️ Similar Historical Cases (Top {TOP_K_RESULTS})</div>', unsafe_allow_html=True)
        if matches:
            for i, match in enumerate(matches, 1):
                meta = match.get("metadata", {})
                sim = round(match.get("score", 0) * 100, 1)
                label = meta.get("overload_label", "?")
                lcolor = risk_color(label)
                st.markdown(f"""
                <div class="case-card">
                    <div class="case-card-header">
                        Case #{i} — <span style="color:{lcolor}">{label}</span>
                        <span style="float:right;color:#8b949e;font-size:0.78rem">{sim}% similar</span>
                    </div>
                    <div class="case-card-meta">
                        📍 {meta.get("location", "Unknown")} &nbsp;|&nbsp;
                        🚛 {meta.get("truck_class", "?").title()} ({meta.get("axle_count", "?")} axles)<br>
                        📦 Cargo: {meta.get("cargo_type", "Unknown")}<br>
                        📊 Historical risk score: <b style="color:{lcolor}">{meta.get("risk_score", "?")}</b><br>
                        ✅ Outcome: {meta.get("outcome", "Unknown")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("No similar cases retrieved from database.")
    # ── Detailed JSON (collapsible) ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Full Analysis JSON (for developers)"):
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.markdown("**Gemini Raw Analysis**")
            st.json(gres)
        with col_j2:
            st.markdown("**RAG Final Decision**")
            st.json(fdec)
    # ── Reset Button ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Analyze Another Truck"):
        st.session_state["analysis_done"] = False
        st.session_state["gemini_result"] = None
        st.session_state["final_decision"] = None
        st.session_state["matches"] = None
        st.session_state["image_bytes"] = None
        st.rerun()
# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;border-top:1px solid #21262d;padding-top:16px;text-align:center;color:#484f58;font-size:0.78rem">
    FreightGuard AI · Pakistan NHA Hackathon MVP · Powered by Gemini Vision + Pinecone RAG
</div>
""", unsafe_allow_html=True)