"""
pages/1_Prediction.py
Prediction page for HM_Si, HM_Temp, and Production Rate
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.predictor import load_all_models, TARGETS, TARGET_LABELS, TARGET_UNITS, predict_single, get_feature_contributions

st.set_page_config(
    page_title="Prediction — BF Intelligence",
    page_icon="🎯",
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
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(249,115,22,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
.page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #f97316 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.page-subtitle { color: #64748b; font-size: 0.95rem; margin-top: 6px; }

.target-card {
    background: rgba(15,23,42,0.9);
    border-radius: 16px;
    padding: 28px;
    border: 1px solid rgba(99,102,241,0.2);
    margin-bottom: 16px;
    transition: border-color 0.3s;
}
.target-card:hover { border-color: rgba(99,102,241,0.5); }

.target-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px;
}
.target-icon { font-size: 2rem; }
.target-title { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; }
.target-desc  { font-size: 0.85rem; color: #64748b; }

.prediction-result {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(249,115,22,0.1));
    border: 2px solid rgba(99,102,241,0.4);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.pred-label { font-size: 0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.pred-value { font-size: 3rem; font-weight: 800; color: #6366f1; line-height: 1; }
.pred-unit  { font-size: 1.1rem; color: #94a3b8; margin-top: 4px; }

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(249,115,22,0.3), transparent);
    margin: 32px 0;
}

.slider-label {
    font-size: 0.85rem; color: #94a3b8; font-weight: 500;
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
    st.markdown("- [🏠 Home](/) ")
    st.markdown("- 🎯 **Prediction** ← You are here")
    st.markdown("- [📊 Analysis](/Analysis)")
    st.markdown("- [⚖️ Trade-Off](/Trade_Off)")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#475569;'>
        <b style='color:#f97316;'>ℹ️ How to use:</b><br>
        Adjust the sliders for each process parameter, then click <b>Predict</b> to get instant ML predictions.
    </div>
    """, unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
feature_meta, models, scalers, imputers = load_all_models()

# ── Page Header ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">🎯 Prediction Engine</div>
    <div class="page-subtitle">
        Adjust process parameters for each target and click Predict to get real-time ML predictions.
        Each model uses the top 9 most predictive features selected from 100+ process variables.
    </div>
</div>
""", unsafe_allow_html=True)

if not feature_meta or not models:
    st.error("⚠️ Models not found. Please run `python train_models.py` first.")
    st.info("This trains the ML models on your datasets and saves them to the `models/` directory.")
    st.stop()

# ── Target Configurations ─────────────────────────────────────
TARGET_CONFIG = {
    'HM_Si': {
        'icon': '⚗️', 'label': 'Silicon Content (HM_Si)', 'unit': '%',
        'color': '#6366f1', 'desc': 'Predicts the silicon content in hot metal (%). Lower Si means better quality.',
        'good_range': (0.3, 0.6), 'alert_low': 0.2, 'alert_high': 0.9,
    },
    'HM_Temp': {
        'icon': '🌡️', 'label': 'Hot Metal Temperature', 'unit': '°C',
        'color': '#f97316', 'desc': 'Predicts the temperature of hot metal at the taphole (°C).',
        'good_range': (1480, 1560), 'alert_low': 1450, 'alert_high': 1580,
    },
    'Prod_Rate': {
        'icon': '⚙️', 'label': 'Production Rate', 'unit': 't/hr',
        'color': '#22c55e', 'desc': 'Predicts the production rate of hot metal (tonnes per hour).',
        'good_range': (140, 200), 'alert_low': 100, 'alert_high': 250,
    },
}

# ── Render each target in a tab ────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚗️ Silicon (HM_Si)",
    "🌡️ Hot Metal Temperature",
    "⚙️ Production Rate"
])

def render_prediction_tab(tgt_name, cfg, tab):
    """Render a full prediction tab for one target."""
    with tab:
        top_feats = feature_meta['top_features'].get(tgt_name, [])
        ranges    = feature_meta['feature_ranges'].get(tgt_name, {})

        if not top_feats:
            st.warning(f"No features found for {tgt_name}. Please retrain models.")
            return

        col_inputs, col_results = st.columns([1.5, 1])

        with col_inputs:
            st.markdown(f"""
            <div class="target-card">
                <div class="target-header">
                    <span class="target-icon">{cfg['icon']}</span>
                    <div>
                        <div class="target-title">{cfg['label']}</div>
                        <div class="target-desc">{cfg['desc']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**🎛️ Adjust the {len(top_feats)} key parameters:**")

            # Pretty feature name mapping
            FRIENDLY_NAMES = {
                'Cold Blast Volume': 'Cold Blast Volume (Nm³/hr)',
                'HBT': 'Hot Blast Temperature (°C)',
                'HBP': 'Hot Blast Pressure (kPa)',
                'Raft': 'RAFT — Flame Temperature (°C)',
                'Oxygen Flow': 'Oxygen Flow Rate (Nm³/hr)',
                'Steam': 'Steam Injection (kg/hr)',
                'Coal Actual': 'Coal Injection Rate (kg/tHM)',
                'PROD_RATE': 'Production Rate (t/hr)',
                'SLAG_RATE': 'Slag Rate (kg/tHM)',
                'Permeabilty': 'Permeability Index',
                'Heat_Load Q1': 'Heat Load Q1 (MW)',
                'Heat_Load Q2': 'Heat Load Q2 (MW)',
                'Heat_Load Q3': 'Heat Load Q3 (MW)',
                'Heat_Load Q4': 'Heat Load Q4 (MW)',
                'Radar Level': 'Stock Level (m)',
                'B MOIST': 'Blast Moisture (g/Nm³)',
                'CS12_Moisture': 'Coke Moisture CS1/2 (%)',
                'CS34_Moisture': 'Coke Moisture CS3/4 (%)',
                'CS56_Moisture': 'Coke Moisture CS5/6 (%)',
                'TAP1_HM_TEMP': 'Taphole 1 HM Temp (°C)',
                'TAP2_HM_TEMP': 'Taphole 2 HM Temp (°C)',
                'TAP3_HM_TEMP': 'Taphole 3 HM Temp (°C)',
                'TAP4_HM_TEMP': 'Taphole 4 HM Temp (°C)',
                'Uptake1': 'Uptake Temperature 1 (°C)',
                'Uptake2': 'Uptake Temperature 2 (°C)',
                'Uptake3': 'Uptake Temperature 3 (°C)',
                'Heat Flow Flux': 'Heat Flow Flux (MW/m²)',
                'Coal Inj. SP': 'Coal Injection Setpoint',
                'FTP': 'Flame Temperature Probe (°C)',
            }

            input_vals = {}
            n_feats = len(top_feats)
            # Show in 3 columns of sliders
            rows = (n_feats + 2) // 3
            for row_i in range(rows):
                scol1, scol2, scol3 = st.columns(3)
                for si, scol in enumerate([scol1, scol2, scol3]):
                    feat_idx = row_i * 3 + si
                    if feat_idx >= n_feats:
                        break
                    feat = top_feats[feat_idx]
                    r = ranges.get(feat, {'min': 0, 'max': 1, 'mean': 0.5})
                    fname = FRIENDLY_NAMES.get(feat, feat)
                    # Trim feature label
                    short_name = feat if len(feat) <= 20 else feat[:20] + '…'
                    with scol:
                        val = st.slider(
                            FRIENDLY_NAMES.get(feat, feat),
                            min_value=float(r['min']),
                            max_value=float(r['max']),
                            value=float(r.get('mean', (r['min'] + r['max']) / 2)),
                            step=float((r['max'] - r['min']) / 200),
                            key=f"slider_{tgt_name}_{feat}",
                        )
                        input_vals[feat] = val

            st.markdown("---")
            predict_btn = st.button(
                f"🚀 Predict {cfg['label']}",
                key=f"btn_{tgt_name}",
                type="primary",
                use_container_width=True,
            )

        with col_results:
            st.markdown("### 📊 Prediction Result")

            if predict_btn or st.session_state.get(f"pred_{tgt_name}"):
                result = predict_single(
                    tgt_name, input_vals,
                    feature_meta, models, scalers, imputers
                )
                pred_val = result['prediction']
                st.session_state[f"pred_{tgt_name}"] = pred_val

                # Status color
                good_lo, good_hi = cfg['good_range']
                if pred_val < cfg['alert_low'] or pred_val > cfg['alert_high']:
                    status_color = '#ef4444'; status_label = '⚠️ Out of Range'
                elif good_lo <= pred_val <= good_hi:
                    status_color = '#22c55e'; status_label = '✅ Normal Range'
                else:
                    status_color = '#f97316'; status_label = '🟡 Monitor Closely'

                # Gauge chart
                ds = feature_meta['data_stats'].get(tgt_name, {})
                gauge_min = ds.get('min', pred_val * 0.5)
                gauge_max = ds.get('max', pred_val * 1.5)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_val,
                    delta={'reference': ds.get('mean', pred_val), 'valueformat': '.3f'},
                    number={'suffix': f' {cfg["unit"]}',
                            'font': {'size': 32, 'color': cfg['color']}},
                    title={'text': f"{cfg['icon']} Predicted {cfg['label']}<br>"
                                   f"<span style='font-size:0.8em;color:{status_color}'>{status_label}</span>",
                           'font': {'size': 13}},
                    gauge={
                        'axis': {'range': [gauge_min, gauge_max]},
                        'bar':  {'color': cfg['color'], 'thickness': 0.25},
                        'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                        'steps': [
                            {'range': [gauge_min, good_lo], 'color': 'rgba(249,115,22,0.2)'},
                            {'range': [good_lo, good_hi],   'color': 'rgba(34,197,94,0.2)'},
                            {'range': [good_hi, gauge_max], 'color': 'rgba(239,68,68,0.2)'},
                        ],
                        'threshold': {
                            'line': {'color': '#fbbf24', 'width': 3},
                            'thickness': 0.8, 'value': pred_val
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=280, margin=dict(t=60, b=20, l=20, r=20),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'},
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Stats row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Predicted", f"{pred_val:.4f}", f"{cfg['unit']}")
                with m2:
                    st.metric("Dataset Mean", f"{ds.get('mean', 0):.4f}", f"{cfg['unit']}")
                with m3:
                    delta_from_mean = pred_val - ds.get('mean', pred_val)
                    st.metric("Δ from Mean", f"{delta_from_mean:+.4f}", f"{cfg['unit']}")

                # Feature contributions
                st.markdown("#### 🔍 Feature Contributions")
                contrib_df = get_feature_contributions(
                    tgt_name, input_vals,
                    feature_meta, models, scalers, imputers
                )
                colors_bar = ['#22c55e' if v >= 0 else '#ef4444'
                              for v in contrib_df['Contribution']]
                fig_contrib = go.Figure(go.Bar(
                    x=contrib_df['Contribution'],
                    y=contrib_df['Feature'],
                    orientation='h',
                    marker_color=colors_bar,
                    text=[f'{v:+.4f}' for v in contrib_df['Contribution']],
                    textposition='outside',
                    textfont={'size': 10, 'color': '#94a3b8'},
                ))
                fig_contrib.update_layout(
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=60),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'},
                    xaxis={'title': 'Impact on Prediction', 'color': '#64748b',
                           'gridcolor': 'rgba(255,255,255,0.05)', 'zeroline': True,
                           'zerolinecolor': 'rgba(255,255,255,0.2)'},
                    yaxis={'autorange': 'reversed', 'color': '#94a3b8',
                           'gridcolor': 'rgba(255,255,255,0.05)'},
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

            else:
                st.markdown("""
                <div style='text-align:center; padding:60px 20px; color:#475569;'>
                    <div style='font-size:3rem; margin-bottom:16px;'>🎯</div>
                    <div style='font-size:1rem; font-weight:600;'>Adjust parameters and click Predict</div>
                    <div style='font-size:0.85rem; margin-top:8px;'>
                        The model will instantly generate a prediction based on your input values.
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Render each tab
render_prediction_tab('HM_Si',     TARGET_CONFIG['HM_Si'],     tab1)
render_prediction_tab('HM_Temp',   TARGET_CONFIG['HM_Temp'],   tab2)
render_prediction_tab('Prod_Rate', TARGET_CONFIG['Prod_Rate'], tab3)
