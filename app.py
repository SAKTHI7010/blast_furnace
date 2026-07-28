"""
app.py — Blast Furnace Intelligence Platform
Home Page — Streamlit Community Cloud Compatible
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Blast Furnace Intelligence Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models ──────────────────────────────────────────────
from utils.predictor import load_all_models, TARGETS, TARGET_LABELS, TARGET_UNITS

feature_meta, models, scalers, imputers = load_all_models()

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 40%, #111827 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%) !important;
    border-right: 1px solid rgba(255,140,0,0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarNav"] a { color: #f97316 !important; }

/* Hero section */
.hero-container {
    background: linear-gradient(135deg, rgba(249,115,22,0.12) 0%, rgba(234,179,8,0.08) 50%, rgba(239,68,68,0.1) 100%);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 20px;
    padding: 50px 40px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(249,115,22,0.06) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.05); opacity: 1; }
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316 0%, #fbbf24 50%, #ef4444 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    margin-top: 12px;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(249,115,22,0.2);
    border: 1px solid rgba(249,115,22,0.4);
    color: #f97316;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(17,24,39,0.95) 100%);
    border: 1px solid rgba(249,115,22,0.2);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(249,115,22,0.6);
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(249,115,22,0.1);
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f97316, #fbbf24, #ef4444);
}
.metric-title {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 10px;
}
.metric-value {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f97316;
    line-height: 1;
}
.metric-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 6px;
}

/* Feature card */
.feature-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 28px;
    height: 100%;
    transition: all 0.3s ease;
}
.feature-card:hover {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 8px 32px rgba(99,102,241,0.1);
}
.feature-icon { font-size: 2.5rem; margin-bottom: 12px; }
.feature-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.feature-desc  { font-size: 0.9rem; color: #64748b; line-height: 1.6; }

/* Status indicator */
.status-ok    { color: #22c55e; }
.status-warn  { color: #f97316; }
.status-error { color: #ef4444; }

/* Section header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(249,115,22,0.2);
    margin-bottom: 24px;
}

/* Nav highlight */
[data-testid="stSidebarNav"] li:first-child { display: none; }

/* Divider */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(249,115,22,0.4), transparent);
    margin: 32px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.8rem;
    padding: 24px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:3rem;'>🔥</div>
        <div style='font-size:1rem; font-weight:700; color:#f97316;'>BF Intelligence</div>
        <div style='font-size:0.75rem; color:#475569; margin-top:4px;'>ML Prediction Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📌 Navigation**")
    st.markdown("- 🏠 **Home** ← You are here")
    st.markdown("- 🎯 **Prediction** — Predict targets")
    st.markdown("- 📊 **Analysis** — EDA & Insights")
    st.markdown("- ⚖️ **Trade-Off** — Optimization")

    st.markdown("---")
    st.markdown("**📋 Dataset Info**")
    if feature_meta:
        st.info(f"""
        📅 {feature_meta['date_range']['start']} → {feature_meta['date_range']['end']}
        
        🗂 {feature_meta['n_records']:,} records merged
        """)
    else:
        st.warning("Models not trained yet. Run `train_models.py` first.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; text-align:center;'>
        JSW Blast Furnace No.3<br>
        Powered by XGBoost + Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── HERO SECTION ─────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔥 AI-Powered Steel Production</div>
    <h1 class="hero-title">Blast Furnace<br>Intelligence Platform</h1>
    <p class="hero-subtitle">
        Real-time ML predictions for Silicon Content, Hot Metal Temperature & Production Rate<br>
        Powered by XGBoost • Random Forest • Gradient Boosting
    </p>
</div>
""", unsafe_allow_html=True)

# ── SYSTEM STATUS ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
model_count = len(models) if models else 0
n_records = feature_meta['n_records'] if feature_meta else 0
date_str  = f"{feature_meta['date_range']['start']}" if feature_meta else "N/A"

metrics_display = [
    ("🤖 Models Loaded",    f"{model_count}/3",      "Active & Ready"),
    ("📊 Training Records", f"{n_records:,}",         "Merged Taps"),
    ("🎯 Prediction Targets", "3",                    "Si · Temp · Prod"),
    ("🧬 Features/Model",   "9",                      "Top Importance"),
]

for col, (title, val, sub) in zip([col1, col2, col3, col4], metrics_display):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── MODEL PERFORMANCE OVERVIEW ────────────────────────────────
if feature_meta and feature_meta.get('model_metrics'):
    st.markdown('<div class="section-header">📈 Model Performance Overview</div>', unsafe_allow_html=True)

    metrics_list = feature_meta['model_metrics']
    perf_cols = st.columns(3)
    target_configs = {
        'HM_Si':     {'icon': '⚗️', 'color': '#6366f1', 'label': 'Silicon (HM_Si)'},
        'HM_Temp':   {'icon': '🌡️', 'color': '#f97316', 'label': 'HM Temperature'},
        'Prod_Rate': {'icon': '⚙️', 'color': '#22c55e', 'label': 'Production Rate'},
    }

    for col, (tgt, cfg) in zip(perf_cols, target_configs.items()):
        if tgt in metrics_list:
            m = metrics_list[tgt]
            r2_pct = max(0, min(100, m['r2'] * 100))
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=r2_pct,
                    number={'suffix': '%', 'font': {'size': 28, 'color': cfg['color']}},
                    title={'text': f"{cfg['icon']} {cfg['label']}<br><span style='font-size:0.75em;color:#64748b'>{m['best_model']}</span>",
                           'font': {'size': 13}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickfont': {'size': 10}},
                        'bar': {'color': cfg['color'], 'thickness': 0.3},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0,  50], 'color': 'rgba(239,68,68,0.15)'},
                            {'range': [50, 75], 'color': 'rgba(249,115,22,0.15)'},
                            {'range': [75,100], 'color': 'rgba(34,197,94,0.15)'},
                        ],
                        'threshold': {
                            'line': {'color': "#fbbf24", 'width': 3},
                            'thickness': 0.8, 'value': r2_pct
                        }
                    }
                ))
                fig.update_layout(
                    height=250, margin=dict(t=40, b=20, l=20, r=20),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'},
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div style='text-align:center; margin-top:-10px;'>
                    <span style='color:#64748b; font-size:0.8rem;'>MAE: <b style='color:{cfg['color']}'>{m['mae']:.4f}</b> &nbsp;|&nbsp;
                    RMSE: <b style='color:{cfg['color']}'>{m['rmse']:.4f}</b></span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── FEATURE SECTIONS ─────────────────────────────────────────
st.markdown('<div class="section-header">🧭 Platform Features</div>', unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)
features_info = [
    ("🎯", "Prediction Engine", "Enter 8–9 process parameters and get instant ML predictions for Silicon Content, HM Temperature, and Production Rate with confidence indicators.", "#f97316"),
    ("📊", "EDA & Analysis",    "Explore data distributions, time-series trends, correlation heatmaps, and feature importance rankings across all three targets.", "#6366f1"),
    ("⚖️", "Trade-Off Optimizer","Interactively tune process parameters and visualize how changes impact all three targets simultaneously. Find the optimal operating window.", "#22c55e"),
    ("🔬", "Feature Insights",  "Understand which process variables (blast temperature, oxygen enrichment, coke rate, etc.) drive each prediction target most strongly.", "#f59e0b"),
]
for col, (icon, title, desc, color) in zip([f1, f2, f3, f4], features_info):
    with col:
        st.markdown(f"""
        <div class="feature-card" style="border-color: rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.25);">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── QUICK STATS (Top Features) ────────────────────────────────
if feature_meta and feature_meta.get('top_features'):
    st.markdown('<div class="section-header">🏆 Top Predictive Features per Target</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚗️ HM Silicon", "🌡️ HM Temperature", "⚙️ Production Rate"])
    tabs_map = [('HM_Si', tab1), ('HM_Temp', tab2), ('Prod_Rate', tab3)]

    for tgt, tab in tabs_map:
        with tab:
            top_feats = feature_meta['top_features'].get(tgt, [])
            fi_dict   = feature_meta['feature_importance'].get(tgt, {})
            if fi_dict:
                fi_items = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:9]
                names = [x[0] for x in fi_items]
                vals  = [x[1] for x in fi_items]
                colors_list = ['#f97316' if i == 0 else '#6366f1' if i < 3 else '#334155'
                               for i in range(len(names))]

                fig = go.Figure(go.Bar(
                    x=vals, y=names, orientation='h',
                    marker_color=colors_list,
                    text=[f'{v:.4f}' for v in vals],
                    textposition='outside',
                    textfont={'size': 11, 'color': '#94a3b8'},
                ))
                fig.update_layout(
                    height=340, margin=dict(t=20, b=20, l=20, r=60),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'},
                    xaxis={'title': 'Importance Score', 'color': '#64748b',
                           'gridcolor': 'rgba(255,255,255,0.05)'},
                    yaxis={'autorange': 'reversed', 'color': '#94a3b8',
                           'gridcolor': 'rgba(255,255,255,0.05)'},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run `train_models.py` to populate feature importances.")

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── HOW TO USE ───────────────────────────────────────────────
st.markdown('<div class="section-header">📖 How to Use This Platform</div>', unsafe_allow_html=True)

step_cols = st.columns(4)
steps = [
    ("1", "🔥", "Select a Page", "Use the sidebar to navigate to Prediction, Analysis, or Trade-Off pages."),
    ("2", "🎛️", "Input Parameters", "Adjust the process parameter sliders to match current furnace conditions."),
    ("3", "🤖", "Get Prediction", "Click 'Predict' to instantly get ML-powered predictions with confidence scores."),
    ("4", "📈", "Analyze & Optimize", "Use the Analysis and Trade-Off pages to understand trends and optimize operations."),
]
for col, (num, icon, title, desc) in zip(step_cols, steps):
    with col:
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(249,115,22,0.15);
                    border-radius:12px; padding:20px; text-align:center;'>
            <div style='font-size:2rem; margin-bottom:8px;'>{icon}</div>
            <div style='background:rgba(249,115,22,0.2); color:#f97316; border-radius:50%;
                        width:28px; height:28px; display:inline-flex; align-items:center;
                        justify-content:center; font-weight:800; font-size:0.9rem;
                        margin-bottom:10px;'>{num}</div>
            <div style='font-weight:700; color:#e2e8f0; margin-bottom:6px;'>{title}</div>
            <div style='font-size:0.8rem; color:#64748b;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔥 Blast Furnace Intelligence Platform &nbsp;|&nbsp; 
    Built with Streamlit &amp; Scikit-Learn &nbsp;|&nbsp; 
    JSW Steel — Blast Furnace No.3
</div>
""", unsafe_allow_html=True)
