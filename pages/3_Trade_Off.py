"""
pages/3_Trade_Off.py
Trade-Off Optimization page — multi-objective analysis & what-if scenarios for HM_SI and HM_TEMP
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.predictor import load_all_pipelines, predict_regression

st.set_page_config(
    page_title="Trade-Off Optimizer — BF Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 40%, #111827 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%) !important;
    border-right: 1px solid rgba(249,115,22,0.15);
}
.page-header {
    background: linear-gradient(135deg, rgba(251,191,36,0.12) 0%, rgba(249,115,22,0.1) 100%);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
}
.page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #fbbf24, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.page-subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 6px; }

.result-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}
.result-value {
    font-size: 2.2rem; font-weight: 800;
    line-height: 1; margin: 6px 0;
}
.result-label {
    font-size: 0.8rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1px;
}
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 2px solid rgba(251,191,36,0.3);
    padding-bottom: 10px; margin: 24px 0 16px;
}
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(251,191,36,0.3), transparent);
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0;'>
        <div style='font-size:2.5rem;'>🔥</div>
        <div style='font-size:1rem; font-weight:700; color:#f97316;'>BF Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📌 Navigation**")
    st.markdown("- [🏠 Home](/)")
    st.markdown("- [🎯 Prediction](/Prediction)")
    st.markdown("- [📊 Analysis](/Analysis)")
    st.markdown("- ⚖️ **Trade-Off** ← You are here")
    st.markdown("---")

# ── Load Models ───────────────────────────────────────────────
artifacts = load_all_pipelines()

# ── Page Header ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">⚖️ Trade-Off Optimization</div>
    <div class="page-subtitle">
        Simulate how process parameters affect both Silicon Content (HM_SI) and Hot Metal Temperature (HM_TEMP) simultaneously.
    </div>
</div>
""", unsafe_allow_html=True)

if not artifacts or 'meta' not in artifacts:
    st.error("⚠️ Model artifacts not found. Please wait for model training to complete.")
    st.stop()

meta = artifacts['meta']['regression']
shared_feats = list(dict.fromkeys(meta['HM_Si']['top_features'] + meta['HM_Temp']['top_features']))

FRIENDLY_NAMES = {
    'Cold Blast Volume': 'Cold Blast Volume (Nm³/hr)',
    'HBT': 'Hot Blast Temp (°C)',
    'HBP': 'Hot Blast Pressure (kPa)',
    'Oxygen Flow': 'Oxygen Flow Rate (Nm³/hr)',
    'Steam': 'Steam Injection (kg/hr)',
    'Coal Actual': 'Coal Injection Rate (kg/tHM)',
    'PROD_RATE': 'Production Rate (t/hr)',
    'SLAG_RATE': 'Slag Rate (kg/tHM)',
    'Permeabilty': 'Permeability Index',
    'OreCokeRatio': 'Ore / Coke Ratio',
    'SinterFrac': 'Sinter Fraction',
    'FluxIronRatio': 'Flux / Iron Ratio',
}

# ════════════════════════════════════════════════════════════
#  SECTION 1: WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔮 Operational What-If Simulator</div>', unsafe_allow_html=True)

slider_col, result_col = st.columns([1.5, 1.0])

input_vals = {}
with slider_col:
    st.markdown("**🎛️ Adjust Operating Parameters:**")
    n_feats = len(shared_feats)
    for i in range(0, n_feats, 2):
        c1, c2 = st.columns(2)
        batch = shared_feats[i:i+2]
        for idx, feat in enumerate(batch):
            target_col = c1 if idx == 0 else c2
            r_info = meta['HM_Si']['feature_ranges'].get(feat, meta['HM_Temp']['feature_ranges'].get(feat, {'min': 0.0, 'max': 100.0, 'mean': 50.0}))
            fname = FRIENDLY_NAMES.get(feat, feat)
            with target_col:
                val = st.slider(
                    fname,
                    min_value=float(r_info['min']),
                    max_value=float(r_info['max']),
                    value=float(r_info['mean']),
                    key=f"tradeoff_{feat}"
                )
                input_vals[feat] = val

    st.markdown("<br>", unsafe_allow_html=True)
    btn_to = st.button("⚖️ Run Trade-Off Simulation", type="primary", key="btn_run_tradeoff", use_container_width=True)

if btn_to:
    st.session_state['tradeoff_preds'] = predict_regression(artifacts, input_vals)

with result_col:
    st.markdown("**🎯 Simultaneous Predictions**")
    
    if 'tradeoff_preds' in st.session_state:
        preds = st.session_state['tradeoff_preds']

        si_val = preds['HM_Si']['prediction']
        temp_val = preds['HM_Temp']['prediction']

        si_color = "#059669" if 0.25 <= si_val <= 0.80 else ("#2563eb" if si_val < 0.25 else "#dc2626")
        temp_color = "#059669" if 1480 <= temp_val <= 1535 else ("#2563eb" if temp_val < 1480 else "#dc2626")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted HM_SI</div>
            <div class="result-value" style="color:{si_color};">{si_val:.3f} %Si</div>
            <div style="color:{si_color}; font-weight:600;">Spec Window: 0.25 – 0.80 %Si</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted HM_TEMP</div>
            <div class="result-value" style="color:{temp_color};">{temp_val:.1f} °C</div>
            <div style="color:{temp_color}; font-weight:600;">Spec Window: 1480 – 1535 °C</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ Operating Window Status")
        if 0.25 <= si_val <= 0.80 and 1480 <= temp_val <= 1535:
            st.success("✅ **Optimal Operating Window**: Both HM_SI and HM_TEMP are strictly within spec targets.")
        else:
            st.warning("⚠️ **Out of Ideal Operating Window**: Adjust blast thermal inputs or burden ratio.")
    else:
        st.info("💡 Adjust operating parameters on the left and click **'Run Trade-Off Simulation'** to calculate simultaneous outcomes.")
