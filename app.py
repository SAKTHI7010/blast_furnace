"""
app.py — Blast Furnace Intelligence Platform
Home Page — Multi-Pipeline AI Platform
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

from utils.predictor import load_all_pipelines

artifacts = load_all_pipelines()
feature_meta = artifacts['meta'] if (artifacts and 'meta' in artifacts) else None

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

/* Hero section */
.hero-container {
    background: linear-gradient(135deg, rgba(249,115,22,0.12) 0%, rgba(234,179,8,0.08) 50%, rgba(239,68,68,0.1) 100%);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 20px;
    padding: 44px 36px;
    text-align: center;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316 0%, #fbbf24 50%, #ef4444 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.1rem;
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
    margin-bottom: 16px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(17,24,39,0.95) 100%);
    border: 1px solid rgba(249,115,22,0.2);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(249,115,22,0.6);
    transform: translateY(-4px);
}
.metric-title {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.4rem;
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
    padding: 24px;
    height: 100%;
}
.feature-icon { font-size: 2.2rem; margin-bottom: 10px; }
.feature-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.feature-desc  { font-size: 0.85rem; color: #64748b; line-height: 1.5; }

.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(249,115,22,0.2);
    margin-bottom: 20px;
}

.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(249,115,22,0.4), transparent);
    margin: 28px 0;
}

.footer {
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 36px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 10px;'>
        <div style='font-size:3rem;'>🔥</div>
        <div style='font-size:1rem; font-weight:700; color:#f97316;'>BF Intelligence</div>
        <div style='font-size:0.75rem; color:#475569; margin-top:4px;'>Multi-Pipeline Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📌 Navigation**")
    st.markdown("- 🏠 **Home** ← You are here")
    st.markdown("- 🎯 **Prediction Engine**")
    st.markdown("- 📊 **Analysis**")
    st.markdown("- ⚖️ **Trade-Off**")

    st.markdown("---")
    st.markdown("**📋 Dataset Summary**")
    if feature_meta:
        summary = feature_meta.get('data_summary', {})
        st.info(f"""
        📅 {summary.get('date_start', 'N/A')} → {summary.get('date_end', 'N/A')}  
        🗂 {summary.get('n_records', 0):,} Hourly Records  
        ⚗️ HM_SI Mean: {summary.get('HM_SI_mean', 0):.3f} %Si  
        🌡️ HM_TEMP Mean: {summary.get('HM_TEMP_mean', 0):.1f} °C
        """)
    else:
        st.warning("Models not trained. Please run `train_models.py`.")

# ── HERO SECTION ─────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔥 Multi-Pipeline AI Platform</div>
    <h1 class="hero-title">Blast Furnace Quality<br>Intelligence Platform</h1>
    <p class="hero-subtitle">
        Comprehensive Machine Learning Engine with Regression, Classification, and Time-Series Forecasting<br>
        Optimizing Hot Metal Silicon (HM_SI) & Temperature (HM_TEMP) for Blast Furnace Operations
    </p>
</div>
""", unsafe_allow_html=True)

# ── SYSTEM METRICS ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
n_rec = feature_meta['data_summary']['n_records'] if feature_meta else 0

metrics = [
    ("⚙️ ML Pipelines", "3", "Reg / Class / Time-Series"),
    ("📊 Merged Dataset", f"{n_rec:,}", "PARA + BURDEN + HM"),
    ("🎯 Targets Monitored", "HM_SI & HM_TEMP", "Silicon & Hot Metal Temp"),
    ("🤖 Candidate Models", "Multiple", "RF, GBM, XGBoost Evaluated"),
]

for col, (title, val, sub) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── PIPELINE ARCHITECTURE OVERVIEW ───────────────────────────
st.markdown('<div class="section-header">🧠 Machine Learning Pipeline Architecture</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)

pipes = [
    ("📈", "1. Regression ML Pipeline",
     "<b>Input:</b> PARA + BURDEN<br><b>Target:</b> HM_SI and HM_TEMP (Actual Float Values)<br>Predicts exact silicon content (%) and hot metal temperature (°C) continuously.", "#6366f1"),
    ("🏷️", "2. Classification Pipeline",
     "<b>Input:</b> PARA + BURDEN<br><b>Target:</b> HM_SI Class & HM_TEMP Class (Low / Normal / High)<br>Classifies operational quality grade with multi-class probability scores.", "#059669"),
    ("⏱️", "3. Time-Series Pipeline",
     "<b>Input:</b> Previous HM + PARA + BURDEN<br><b>Target:</b> Future HM_SI and Future HM_TEMP<br>Autoregressive horizon forecasting for next-tap predictions.", "#f97316"),
]

for col, (icon, title, desc, color) in zip([p1, p2, p3], pipes):
    with col:
        st.markdown(f"""
        <div class="feature-card" style="border-color:{color}44;">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title" style="color:{color};">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔥 Blast Furnace Intelligence Platform &nbsp;|&nbsp; 
    JSW Blast Furnace Operations &nbsp;|&nbsp; 
    Powered by XGBoost, Random Forest & Streamlit
</div>
""", unsafe_allow_html=True)
