# === FILE: streamlit_app.py ===
# FreightGuard AI — Pakistan NHA Freight Inspection System
# Improvements over v1:
#   • Pinecone removed — replaced by in-memory reference cases (no fake embeddings)
#   • Redis added for: image result cache, per-minute rate limiting, session/global history
#   • Every Gemini call is gated by Redis cache check + rate limit check
#   • Dashboard shows Redis cost metrics: cache hits, API calls saved, rate usage
#   • History tab shows all inspections this session and globally

import sys
import os
import uuid
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from config.config import (
    GEMINI_API_KEY,
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    GEMINI_RATE_LIMIT_PER_MINUTE,
    IMAGE_CACHE_TTL_SECONDS,
)
from app.gemini_service   import init_gemini, analyze_truck_image
from app.decision_engine  import run_decision_pipeline
from app.redis_service    import (
    get_redis_client,
    compute_image_hash,
    get_cached_result,
    cache_result,
    check_rate_limit,
    get_current_rate_usage,
    increment_daily_counter,
    get_daily_counter,
    push_to_history,
    get_session_history,
    get_global_history,
    get_redis_stats,
)
from app.utils import get_risk_category, get_risk_action, risk_color

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreightGuard AI — Pakistan NHA",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* ── Hide or style Streamlit default header ── */
[data-testid="stHeader"] {
    background: #080c10 !important;
}

[data-testid="stAppViewContainer"] > div:first-child {
    background: #080c10 !important;
}

/* Optional: hide toolbar/menu */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }

.stApp { background: #080c10; color: #dde3ea; }

section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a38;
}

/* ── Header ── */
.hdr {
    background: linear-gradient(120deg, #0a1628 0%, #091220 60%, #060d18 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hdr::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg, transparent, transparent 40px,
        rgba(56,139,253,0.03) 40px, rgba(56,139,253,0.03) 41px
    );
    pointer-events: none;
}
.hdr-title {
    font-size: 2.4rem; font-weight: 800; color: #f0f6fc;
    letter-spacing: -0.03em; margin: 0; line-height: 1.1;
}
.hdr-sub { color: #6e7f91; font-size: 0.95rem; margin-top: 6px; }
.hdr-badge {
    display: inline-block;
    background: rgba(56,139,253,0.1);
    color: #58a6ff;
    border: 1px solid rgba(56,139,253,0.3);
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    padding: 4px 14px; border-radius: 20px; margin-top: 10px;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.mcard {
    background: #0d1117;
    border: 1px solid #1e2a38;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 14px;
    position: relative;
}
.mcard-accent { border-left: 3px solid; }
.mlabel {
    color: #6e7f91; font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;
}
.mvalue {
    font-size: 2rem; font-weight: 800;
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}

/* ── Redis cost panel ── */
.redis-panel {
    background: #0a1220;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.redis-title {
    color: #58a6ff; font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;
}
.redis-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #1e2a38; font-size: 0.83rem;
}
.redis-row:last-child { border-bottom: none; }
.redis-key { color: #8b949e; }
.redis-val { color: #f0f6fc; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.redis-val.green  { color: #22c55e; }
.redis-val.yellow { color: #f59e0b; }
.redis-val.red    { color: #ef4444; }

/* ── Rate bar ── */
.rate-track {
    background: #1e2a38; border-radius: 6px; height: 10px; overflow: hidden; margin: 6px 0;
}
.rate-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }

/* ── Risk bar ── */
.risk-track {
    background: #1e2a38; border-radius: 8px; height: 16px; overflow: hidden; margin: 10px 0;
}
.risk-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #22c55e 0%, #f59e0b 40%, #f97316 70%, #ef4444 100%);
}

/* ── Action banner ── */
.action-banner {
    border-radius: 10px; padding: 20px 28px;
    font-size: 1.1rem; font-weight: 800; letter-spacing: 0.06em;
    text-align: center; text-transform: uppercase; margin: 18px 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Signal tags ── */
.stag {
    display: inline-block;
    background: #12191f; border: 1px solid #2a3a4a;
    color: #c9d1d9; padding: 4px 12px; border-radius: 20px;
    font-size: 0.80rem; margin: 3px 3px 3px 0;
}

/* ── Case card ── */
.ccard {
    background: #0d1117; border: 1px solid #1e2a38;
    border-left: 3px solid; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 10px;
}
.ccard-head { font-size: 0.88rem; font-weight: 700; color: #f0f6fc; margin-bottom: 6px; }
.ccard-meta { font-size: 0.78rem; color: #8b949e; line-height: 1.8; }

/* ── Reasoning box ── */
.rbox {
    background: #060a0f; border: 1px solid #1e2a38;
    border-radius: 8px; padding: 14px 18px;
    font-size: 0.84rem; color: #b0b9c4; line-height: 1.7;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Cache hit banner ── */
.cache-banner {
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.3);
    border-radius: 10px; padding: 14px 20px; margin: 12px 0;
    color: #22c55e; font-weight: 700; font-size: 0.9rem; text-align: center;
}

/* ── History table rows ── */
.hist-row {
    background: #0d1117; border: 1px solid #1e2a38;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    display: grid; grid-template-columns: 1fr 1fr 1fr 2fr;
    gap: 12px; align-items: center; font-size: 0.82rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #21262d !important; color: #f0f6fc !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
    font-weight: 700 !important; font-family: 'Syne', sans-serif !important;
    padding: 10px 24px !important; transition: all 0.2s !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    background: #30363d !important; border-color: #58a6ff !important;
    color: #58a6ff !important;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important; background: #0a1220 !important;
}
.stSpinner > div { border-top-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


# ── Session init ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "session_id":      str(uuid.uuid4())[:8],
        "gemini_model":    None,
        "redis_client":    None,
        "initialized":     False,
        "analysis_done":   False,
        "gemini_result":   None,
        "final_decision":  None,
        "similar_cases":   None,
        "image_bytes":     None,
        "served_from_cache": False,
        "cache_hit_count": 0,
        "api_call_count":  0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Auto-init on first load ───────────────────────────────────────────────────
if not st.session_state["initialized"]:
    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY missing. Add it to your .env file.")
        st.stop()

    with st.spinner("🔗 Connecting to Gemini..."):
        try:
            st.session_state["gemini_model"] = init_gemini(GEMINI_API_KEY)
        except Exception as e:
            st.error(f"Gemini init failed: {e}")
            st.stop()

    with st.spinner("🔗 Connecting to Redis..."):
        r = get_redis_client(REDIS_HOST, REDIS_PORT, REDIS_DB)
        st.session_state["redis_client"] = r
        if r is None:
            st.warning(
                "⚠️ Redis unavailable — running without cache/rate-limiting. "
                "Start Redis with: `redis-server`"
            )

    st.session_state["initialized"] = True


# ── Sidebar ───────────────────────────────────────────────────────────────────
r = st.session_state["redis_client"]

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.markdown("<hr style='border-color:#1e2a38'>", unsafe_allow_html=True)

    gemini_ok = st.session_state["gemini_model"] is not None
    redis_ok  = r is not None

    st.markdown(f"{'🟢' if gemini_ok else '🔴'} Gemini Vision: {'Active' if gemini_ok else 'Offline'}")
    st.markdown(f"{'🟢' if redis_ok  else '🔴'} Redis Cache: {'Active' if redis_ok  else 'Offline'}")

    st.markdown("<hr style='border-color:#1e2a38'>", unsafe_allow_html=True)

    # ── Redis cost stats ──────────────────────────────────────────────────────
    st.markdown("### 💰 Cost Controls")

    current_rate, rate_limit = get_current_rate_usage(r, GEMINI_RATE_LIMIT_PER_MINUTE)
    rate_pct = min(100, int((current_rate / rate_limit) * 100)) if rate_limit else 0
    rate_color = "#22c55e" if rate_pct < 60 else "#f59e0b" if rate_pct < 85 else "#ef4444"

    daily_calls  = get_daily_counter(r)
    redis_stats  = get_redis_stats(r)

    cache_hits  = st.session_state["cache_hit_count"]
    api_calls   = st.session_state["api_call_count"]
    total_reqs  = cache_hits + api_calls
    savings_pct = round((cache_hits / total_reqs) * 100) if total_reqs > 0 else 0

    st.markdown(f"""
    <div class="redis-panel">
        <div class="redis-title">⚡ Redis Cost Shield</div>
        <div class="redis-row">
            <span class="redis-key">Rate (this min)</span>
            <span class="redis-val {'green' if rate_pct < 60 else 'yellow' if rate_pct < 85 else 'red'}">{current_rate}/{rate_limit}</span>
        </div>
        <div class="rate-track">
            <div class="rate-fill" style="width:{rate_pct}%;background:{rate_color}"></div>
        </div>
        <div class="redis-row">
            <span class="redis-key">Gemini calls today</span>
            <span class="redis-val">{daily_calls}</span>
        </div>
        <div class="redis-row">
            <span class="redis-key">Cache hits (session)</span>
            <span class="redis-val green">{cache_hits}</span>
        </div>
        <div class="redis-row">
            <span class="redis-key">API calls saved</span>
            <span class="redis-val green">{savings_pct}%</span>
        </div>
        <div class="redis-row">
            <span class="redis-key">Redis memory</span>
            <span class="redis-val">{redis_stats.get('used_memory_human', 'N/A')}</span>
        </div>
        <div class="redis-row">
            <span class="redis-key">Cached results</span>
            <span class="redis-val">{redis_stats.get('total_keys', 'N/A')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2a38'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.76rem;color:#6e7f91;line-height:1.8">
    <b style="color:#8b949e">Risk Categories</b><br>
    🟢 LOW (0–30): Allow Passage<br>
    🟡 MEDIUM (31–60): Inspect<br>
    🟠 HIGH (61–80): Stop for Weighing<br>
    🔴 CRITICAL (81–100): Immediate Action
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.72rem;color:#484f58;margin-top:16px">
    Session ID: {st.session_state['session_id']}
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <div style="display:flex;align-items:center;gap:20px">
        <div style="font-size:3.5rem;line-height:1">🚛</div>
        <div>
            <div class="hdr-title">FreightGuard AI</div>
            <div class="hdr-sub">AI-Assisted Freight Inspection Prioritization System</div>
            <div class="hdr-badge">🇵🇰 Pakistan Motorway Police · National Highway Authority</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state["initialized"]:
    st.info("⏳ Initializing system...")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_inspect, tab_history, tab_about = st.tabs(["🔍 Inspect Truck", "📋 Inspection History", "ℹ️ How It Works"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — INSPECT
# ════════════════════════════════════════════════════════════════════════════════
with tab_inspect:

    st.markdown('<div style="color:#6e7f91;font-size:0.9rem;margin-bottom:20px">Upload a truck photograph from a toll plaza or highway camera for AI-assisted overload risk assessment.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload truck image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()
        st.session_state["image_bytes"] = image_bytes

        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.image(image_bytes, caption="Uploaded truck image", use_container_width=True)
        with col_btn:
            st.markdown("<br><br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analyze Truck", use_container_width=True)

        if analyze_btn:
            img_hash = compute_image_hash(image_bytes)

            # ── Step 1: Redis cache check ─────────────────────────────────────
            cached = get_cached_result(r, img_hash)

            if cached:
                # Cache HIT — zero Gemini cost
                st.session_state["gemini_result"]     = cached["gemini_result"]
                st.session_state["final_decision"]    = cached["final_decision"]
                st.session_state["similar_cases"]     = cached["similar_cases"]
                st.session_state["served_from_cache"] = True
                st.session_state["analysis_done"]     = True
                st.session_state["cache_hit_count"]  += 1
                st.rerun()

            else:
                # Cache MISS — check rate limit before calling Gemini
                allowed, current_count, limit = check_rate_limit(r, GEMINI_RATE_LIMIT_PER_MINUTE)

                if not allowed:
                    st.error(
                        f"🚦 **Rate limit reached** ({current_count}/{limit} calls this minute). "
                        f"Please wait ~60 seconds before analyzing a new image. "
                        f"This protects against unexpected Gemini API cost spikes."
                    )
                    st.stop()

                # ── Step 2: Gemini Vision call ────────────────────────────────
                with st.spinner("🤖 Gemini Vision is analyzing the truck..."):
                    try:
                        gemini_result = analyze_truck_image(
                            st.session_state["gemini_model"], image_bytes
                        )
                        st.session_state["gemini_result"] = gemini_result
                        increment_daily_counter(r)
                        st.session_state["api_call_count"] += 1
                    except Exception as e:
                        st.error(f"Gemini analysis failed: {e}")
                        st.stop()

                # ── Step 3: Decision engine (local, no API) ───────────────────
                with st.spinner("⚙️ Running decision engine..."):
                    final_decision, similar_cases = run_decision_pipeline(gemini_result)
                    st.session_state["final_decision"] = final_decision
                    st.session_state["similar_cases"]  = similar_cases

                # ── Step 4: Write to Redis cache ──────────────────────────────
                cache_payload = {
                    "gemini_result":  gemini_result,
                    "final_decision": final_decision,
                    "similar_cases":  similar_cases,
                }
                cache_result(r, img_hash, cache_payload, IMAGE_CACHE_TTL_SECONDS)

                # ── Step 5: Push to history ───────────────────────────────────
                history_record = {
                    "timestamp":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "session_id":     st.session_state["session_id"],
                    "truck_class":    gemini_result.get("truck_class", "?"),
                    "axle_count":     gemini_result.get("axle_count_estimate", "?"),
                    "risk_score":     final_decision.get("final_risk_score", 0),
                    "risk_category":  final_decision.get("risk_category", "?"),
                    "action":         final_decision.get("inspection_action", "?"),
                    "cargo_ext":      gemini_result.get("cargo_extension_detected", False),
                    "signals":        gemini_result.get("visible_overload_signals", []),
                    "from_cache":     False,
                    "img_hash":       img_hash[:12] + "...",
                }
                push_to_history(r, st.session_state["session_id"], history_record)

                st.session_state["served_from_cache"] = False
                st.session_state["analysis_done"]     = True
                st.rerun()


    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state["analysis_done"] and st.session_state["final_decision"]:
        gres  = st.session_state["gemini_result"]
        fdec  = st.session_state["final_decision"]
        cases = st.session_state["similar_cases"] or []

        risk_score = fdec.get("final_risk_score", gres.get("risk_score_raw", 0))
        risk_cat   = fdec.get("risk_category", get_risk_category(risk_score))
        action     = fdec.get("inspection_action", get_risk_action(risk_cat))
        color      = risk_color(risk_cat)
        signals    = gres.get("visible_overload_signals", [])
        cargo_ext  = gres.get("cargo_extension_detected", False)

        # ── Cache hit banner ──────────────────────────────────────────────────
        if st.session_state["served_from_cache"]:
            st.markdown("""
            <div class="cache-banner">
                ⚡ RESULT SERVED FROM REDIS CACHE — No Gemini API call made · Zero additional cost
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:#6e7f91;font-size:0.78rem;text-align:right;margin-bottom:8px">'
                f'✅ Result cached in Redis · Next identical image = free</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── KPI Row ───────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="mcard mcard-accent" style="border-left-color:{color}">
                <div class="mlabel">Risk Score</div>
                <div class="mvalue" style="color:{color}">{risk_score}</div>
                <div style="color:#6e7f91;font-size:0.75rem;margin-top:4px">out of 100</div>
            </div>""", unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="mcard">
                <div class="mlabel">Risk Category</div>
                <div style="margin-top:8px">
                    <span style="background:{color}18;color:{color};border:1px solid {color}40;
                    padding:8px 18px;border-radius:8px;font-weight:800;font-size:1rem;
                    letter-spacing:0.06em;font-family:'IBM Plex Mono',monospace">{risk_cat}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="mcard">
                <div class="mlabel">Truck Class</div>
                <div class="mvalue" style="font-size:1.3rem;color:#f0f6fc;margin-top:4px">
                    {gres.get("truck_class", "?").upper()}
                </div>
                <div style="color:#6e7f91;font-size:0.75rem;margin-top:4px">
                    {gres.get("axle_count_estimate","?")} axles detected
                </div>
            </div>""", unsafe_allow_html=True)

        with k4:
            ext_color = "#ef4444" if cargo_ext else "#22c55e"
            ext_label = "YES — EXCEEDS CHASSIS" if cargo_ext else "NO — WITHIN BOUNDS"
            st.markdown(f"""
            <div class="mcard">
                <div class="mlabel">Cargo Extension</div>
                <div style="color:{ext_color};font-size:0.88rem;font-weight:700;margin-top:10px">
                    {ext_label}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Risk bar ──────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="margin:4px 0">
            <div style="display:flex;justify-content:space-between;
            color:#6e7f91;font-size:0.70rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px">
                <span>LOW</span><span>MEDIUM</span><span>HIGH</span><span>CRITICAL</span>
            </div>
            <div class="risk-track">
                <div class="risk-fill" style="width:{risk_score}%"></div>
            </div>
            <div style="text-align:right;color:#6e7f91;font-size:0.72rem">{risk_score}/100</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Action banner ─────────────────────────────────────────────────────
        icons = {"ALLOW PASSAGE": "✅", "INSPECT": "⚠️",
                 "STOP FOR WEIGHING": "🛑",
                 "STOP FOR WEIGHING — IMMEDIATE ACTION REQUIRED": "🚨"}
        icon = next((v for k, v in icons.items() if k in action), "🛑")
        st.markdown(f"""
        <div class="action-banner" style="background:{color}12;border:2px solid {color};color:{color}">
            {icon} &nbsp; INSPECTOR ACTION: {action}
        </div>
        """, unsafe_allow_html=True)

        # ── Detail columns ────────────────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div style="color:#6e7f91;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;border-bottom:1px solid #1e2a38;padding-bottom:8px;margin-bottom:14px">🔬 Visual Analysis — Gemini Vision</div>', unsafe_allow_html=True)

            st.markdown("**Detected Overload Signals:**")
            if signals:
                tags = "".join([f'<span class="stag">⚠ {s}</span>' for s in signals])
                st.markdown(f'<div style="margin:8px 0 16px">{tags}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="stag">✓ No significant signals detected</span>', unsafe_allow_html=True)

            st.markdown("**Gemini Reasoning:**")
            st.markdown(f'<div class="rbox">{gres.get("reasoning","No reasoning provided.")}</div>', unsafe_allow_html=True)

            st.markdown("<br>**Decision Engine Output:**")
            st.markdown(f'<div class="rbox" style="border-color:{color}30">{fdec.get("reasoning","")}</div>', unsafe_allow_html=True)

            if fdec.get("boost_reasons"):
                st.markdown("**Rule Boosts Applied:**")
                boosts = "".join([f'<span class="stag" style="border-color:#388bfd40;color:#58a6ff">{b}</span>' for b in fdec["boost_reasons"]])
                st.markdown(f'<div style="margin:8px 0">{boosts}</div>', unsafe_allow_html=True)

        with right:
            st.markdown(f'<div style="color:#6e7f91;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;border-bottom:1px solid #1e2a38;padding-bottom:8px;margin-bottom:14px">📁 Similar Historical Cases (Top {len(cases)})</div>', unsafe_allow_html=True)

            if cases:
                for i, case in enumerate(cases, 1):
                    lbl   = case.get("overload_label", "?")
                    lclr  = risk_color(lbl)
                    sim   = case.get("similarity", 0)
                    st.markdown(f"""
                    <div class="ccard" style="border-left-color:{lclr}">
                        <div class="ccard-head">
                            Case {case.get('id','#'+str(i))} —
                            <span style="color:{lclr}">{lbl}</span>
                            <span style="float:right;color:#6e7f91;font-size:0.75rem">{sim}% match</span>
                        </div>
                        <div class="ccard-meta">
                            📍 {case.get('location','?')} &nbsp;|&nbsp;
                            🚛 {str(case.get('truck_class','?')).title()} ({case.get('axle_count','?')} axles)<br>
                            📦 Cargo: {case.get('cargo_type','Unknown')}<br>
                            📊 Historical risk: <b style="color:{lclr}">{case.get('risk_score','?')}</b><br>
                            ✅ {case.get('outcome','Unknown')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("No similar cases found.")

        # ── Raw JSON expander ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Full Analysis JSON"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Gemini Raw Output**")
                st.json(gres)
            with c2:
                st.markdown("**Decision Engine Output**")
                st.json(fdec)

        if st.button("🔄 Analyze Another Truck"):
            for k in ["analysis_done","gemini_result","final_decision",
                      "similar_cases","image_bytes","served_from_cache"]:
                st.session_state[k] = None if k != "analysis_done" else False
                if k == "served_from_cache":
                    st.session_state[k] = False
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ════════════════════════════════════════════════════════════════════════════════
with tab_history:

    st.markdown("### 📋 Inspection History")

    if r is None:
        st.warning("Redis unavailable — history requires Redis to be running.")
    else:
        h_col1, h_col2 = st.columns(2)

        with h_col1:
            st.markdown("#### This Session")
            session_records = get_session_history(r, st.session_state["session_id"])
            if not session_records:
                st.markdown('<div style="color:#6e7f91;font-size:0.88rem">No inspections this session yet.</div>', unsafe_allow_html=True)
            else:
                for rec in session_records:
                    score = rec.get("risk_score", 0)
                    cat   = rec.get("risk_category", "?")
                    clr   = risk_color(cat)
                    st.markdown(f"""
                    <div class="ccard" style="border-left-color:{clr}">
                        <div class="ccard-head">
                            {rec.get('truck_class','?').upper()} — {rec.get('axle_count','?')} axles
                            <span style="float:right;color:{clr};font-family:'IBM Plex Mono',monospace;font-size:0.88rem">{score}/100</span>
                        </div>
                        <div class="ccard-meta">
                            🕐 {rec.get('timestamp','?')}<br>
                            📋 Action: <b style="color:{clr}">{rec.get('action','?')}</b><br>
                            {'⚡ From cache' if rec.get('from_cache') else '🤖 Gemini API call'} &nbsp;|&nbsp;
                            Hash: {rec.get('img_hash','?')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with h_col2:
            st.markdown("#### Global (All Sessions)")
            global_records = get_global_history(r, count=20)
            if not global_records:
                st.markdown('<div style="color:#6e7f91;font-size:0.88rem">No global records yet.</div>', unsafe_allow_html=True)
            else:
                for rec in global_records:
                    score = rec.get("risk_score", 0)
                    cat   = rec.get("risk_category", "?")
                    clr   = risk_color(cat)
                    st.markdown(f"""
                    <div class="ccard" style="border-left-color:{clr}">
                        <div class="ccard-head">
                            {rec.get('truck_class','?').upper()} — {cat}
                            <span style="float:right;color:{clr};font-family:'IBM Plex Mono',monospace;font-size:0.88rem">{score}/100</span>
                        </div>
                        <div class="ccard-meta">
                            🕐 {rec.get('timestamp','?')}<br>
                            Session: {rec.get('session_id','?')} &nbsp;|&nbsp;
                            {'⚡ Cache' if rec.get('from_cache') else '🤖 API'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ System Architecture")
    st.markdown("""
    <div style="color:#8b949e;font-size:0.9rem;line-height:1.9;max-width:760px">

    <b style="color:#f0f6fc">FreightGuard AI</b> is a cost-optimized truck overload detection system
    built for Pakistan's National Highway Authority. Here's how each component earns its place:

    <br><br>

    <b style="color:#58a6ff">1. Image Upload & Hashing</b><br>
    Every uploaded image is SHA-256 hashed before any AI call is made.
    This hash becomes the Redis cache key.

    <br><br>

    <b style="color:#58a6ff">2. Redis Cache Check (first gate)</b><br>
    If the same truck image was analyzed before (today or in the last 24 hours),
    the result is served from Redis instantly — <b style="color:#22c55e">zero Gemini API cost</b>.
    This handles duplicate uploads, page refreshes, and re-inspections of the same vehicle.

    <br><br>

    <b style="color:#58a6ff">3. Rate Limiter (second gate)</b><br>
    A Redis counter tracks Gemini API calls per 60-second window.
    If the limit is reached, the request is queued/deferred rather than dropped,
    protecting against unexpected cost spikes during busy inspection periods.

    <br><br>

    <b style="color:#58a6ff">4. Gemini Vision Analysis</b><br>
    Only images that miss the cache AND pass the rate limit reach Gemini.
    The model returns a structured JSON: truck class, axle count, overload signals, raw risk score.

    <br><br>

    <b style="color:#58a6ff">5. Deterministic Decision Engine (local, free)</b><br>
    A rule-based engine applies score boosts based on signal count, cargo extension,
    and axle-class mismatches. No second API call needed. Result is reproducible and auditable.

    <br><br>

    <b style="color:#58a6ff">6. Result Cached & Logged</b><br>
    The final result is written back to Redis (24-hour TTL) and appended to both
    session and global inspection history lists.

    <br><br>

    <b style="color:#f59e0b">Cost Impact</b><br>
    Without Redis: every inspection = 1 Gemini call.<br>
    With Redis: repeated images = 0 calls. Rate spikes = throttled automatically.<br>
    Estimated savings on a busy checkpoint: <b style="color:#22c55e">60–80% reduction</b> in API spend.

    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:56px;border-top:1px solid #1e2a38;padding-top:16px;
text-align:center;color:#3a4450;font-size:0.75rem">
    FreightGuard AI · Pakistan NHA Hackathon · Gemini Vision + Redis Cost Shield · No Pinecone
</div>
""", unsafe_allow_html=True)