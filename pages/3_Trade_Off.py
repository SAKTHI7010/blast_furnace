"""
pages/3_Trade_Off.py
Trade-Off Optimization page — multi-objective analysis, Pareto front, what-if scenarios
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.predictor import load_all_models, predict_single

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
    padding: 28px 32px;
    margin-bottom: 28px;
}
.page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #fbbf24, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.page-subtitle { color: #64748b; font-size: 0.95rem; margin-top: 6px; }

.tradeoff-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}
.result-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.result-value {
    font-size: 2rem; font-weight: 800;
    line-height: 1; margin-bottom: 4px;
}
.result-label {
    font-size: 0.75rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1px;
}
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 2px solid rgba(251,191,36,0.3);
    padding-bottom: 10px; margin: 28px 0 20px;
}
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(251,191,36,0.3), transparent);
    margin: 32px 0;
}
.rec-card {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 10px;
    padding: 16px;
}
.warn-card {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 16px;
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
    st.markdown("""
    <div style='font-size:0.8rem; color:#475569;'>
        <b style='color:#fbbf24;'>⚖️ About Trade-Off:</b><br>
        This page helps you find the optimal operating window that balances all three targets simultaneously.
    </div>
    """, unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
feature_meta, models, scalers, imputers = load_all_models()

PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': '#e2e8f0'},
)
GRID = dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)')

# ── Page Header ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">⚖️ Trade-Off Optimization</div>
    <div class="page-subtitle">
        Explore how process parameters affect all three targets simultaneously.
        Find the optimal operating window for quality and productivity.
    </div>
</div>
""", unsafe_allow_html=True)

if not feature_meta or not models:
    st.error("⚠️ Models not found. Please run `python train_models.py` first.")
    st.stop()

# ── Helper: build a shared "common" parameter set ─────────────
def get_shared_features():
    """Get features common to at least 2 targets for the shared slider panel."""
    all_target_feats = {}
    for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
        all_target_feats[tgt] = set(feature_meta['top_features'].get(tgt, []))
    # Union of all features
    all_feats = set()
    for s in all_target_feats.values():
        all_feats.update(s)
    return list(all_feats)

SHARED_FEATS = get_shared_features()

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
    'Radar Level': 'Stock Level (m)',
    'B MOIST': 'Blast Moisture (g/Nm³)',
    'TAP1_HM_TEMP': 'Taphole 1 HM Temp (°C)',
    'TAP2_HM_TEMP': 'Taphole 2 HM Temp (°C)',
}

TARGET_CONFIG = {
    'HM_Si':     {'label': 'HM Silicon (%)',       'unit': '%',    'color': '#6366f1', 'icon': '⚗️'},
    'HM_Temp':   {'label': 'HM Temperature (°C)',  'unit': '°C',   'color': '#f97316', 'icon': '🌡️'},
    'Prod_Rate': {'label': 'Production Rate (t/hr)','unit': 't/hr', 'color': '#22c55e', 'icon': '⚙️'},
}

# ════════════════════════════════════════════════════════════
#  SECTION 1: WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔮 What-If Simulator</div>', unsafe_allow_html=True)
st.markdown("Adjust process parameters below and see predictions for **all three targets simultaneously**.")

slider_col, result_col = st.columns([1.6, 1])

input_vals_shared = {}
with slider_col:
    st.markdown("**🎛️ Adjust Process Parameters:**")
    n_feats = len(SHARED_FEATS)
    feat_rows = (n_feats + 1) // 2
    for row_i in range(feat_rows):
        rcol1, rcol2 = st.columns(2)
        for si, rcol in enumerate([rcol1, rcol2]):
            fi = row_i * 2 + si
            if fi >= n_feats:
                break
            feat = SHARED_FEATS[fi]
            # Get range from any target that has this feature
            r = None
            for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
                if feat in (feature_meta['feature_ranges'].get(tgt) or {}):
                    r = feature_meta['feature_ranges'][tgt][feat]
                    break
            if r is None:
                continue
            with rcol:
                val = st.slider(
                    FRIENDLY_NAMES.get(feat, feat),
                    min_value=float(r['min']),
                    max_value=float(r['max']),
                    value=float(r.get('mean', (r['min'] + r['max']) / 2)),
                    step=float((r['max'] - r['min']) / 200),
                    key=f"tradeoff_{feat}",
                )
                input_vals_shared[feat] = val

    predict_all_btn = st.button(
        "🚀 Predict All Targets",
        type="primary",
        use_container_width=True,
        key="predict_all",
    )

with result_col:
    st.markdown("**📊 Simultaneous Predictions:**")

    if predict_all_btn or st.session_state.get('tradeoff_results'):
        all_preds = {}
        for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
            res = predict_single(tgt, input_vals_shared, feature_meta, models, scalers, imputers)
            all_preds[tgt] = res.get('prediction', 0)

        if predict_all_btn:
            st.session_state['tradeoff_results'] = all_preds
        else:
            all_preds = st.session_state['tradeoff_results']

        # Show each prediction
        THRESHOLDS = {
            'HM_Si':     {'lo': 0.3, 'hi': 0.6},
            'HM_Temp':   {'lo': 1480, 'hi': 1560},
            'Prod_Rate': {'lo': 140, 'hi': 200},
        }
        for tgt, cfg in TARGET_CONFIG.items():
            pval = all_preds.get(tgt, 0)
            lo, hi = THRESHOLDS[tgt]['lo'], THRESHOLDS[tgt]['hi']
            if lo <= pval <= hi:
                color, badge = cfg['color'], '✅ Normal'
            elif pval < lo:
                color, badge = '#ef4444', '⬇️ Low'
            else:
                color, badge = '#f59e0b', '⬆️ High'

            st.markdown(f"""
            <div class="result-card" style="border-color: {color}44; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="text-align:left;">
                        <div style="font-size:0.8rem; color:#64748b;">{cfg['icon']} {cfg['label']}</div>
                        <div class="result-value" style="color:{color};">{pval:.3f}</div>
                        <div style="font-size:0.8rem; color:#475569;">{cfg['unit']}</div>
                    </div>
                    <div style="background:{color}22; color:{color}; padding:6px 12px;
                                border-radius:20px; font-size:0.8rem; font-weight:600;">
                        {badge}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Radar / spider chart for normalized targets
        ds_stats = feature_meta.get('data_stats', {})
        radar_tgts = ['HM_Si', 'HM_Temp', 'Prod_Rate']
        radar_vals = []
        for tgt in radar_tgts:
            pv = all_preds.get(tgt, 0)
            mn = ds_stats.get(tgt, {}).get('min', pv * 0.8)
            mx = ds_stats.get(tgt, {}).get('max', pv * 1.2)
            radar_vals.append((pv - mn) / max(mx - mn, 1e-6) * 100)

        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=['Si (%)', 'Temp (°C)', 'Prod (t/hr)', 'Si (%)'],
            fill='toself',
            fillcolor='rgba(249,115,22,0.15)',
            line={'color': '#f97316', 'width': 2},
            marker={'color': '#f97316', 'size': 8},
        ))
        fig_radar.update_layout(
            **PLOT_THEME, height=250,
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='rgba(255,255,255,0.1)',
                                tickfont={'color': '#64748b', 'size': 9}),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                                 tickfont={'color': '#94a3b8', 'size': 11}),
            ),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#475569;'>
            <div style='font-size:3rem; margin-bottom:16px;'>⚖️</div>
            <div style='font-size:1rem; font-weight:600;'>Adjust sliders and click Predict All Targets</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SECTION 2: SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)
st.markdown("See how **one variable** affects all three targets while keeping others at default.")

sel_feat = st.selectbox(
    "Select parameter to vary:",
    options=SHARED_FEATS,
    format_func=lambda f: FRIENDLY_NAMES.get(f, f),
    key="sens_feat",
)

if sel_feat:
    # Get range
    r = None
    for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
        if sel_feat in (feature_meta['feature_ranges'].get(tgt) or {}):
            r = feature_meta['feature_ranges'][tgt][sel_feat]
            break

    if r:
        n_points = 30
        sweep_vals = np.linspace(r['min'], r['max'], n_points)

        # Default values for all features
        default_vals = {}
        for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
            for feat, frange in (feature_meta['feature_ranges'].get(tgt) or {}).items():
                if feat not in default_vals:
                    default_vals[feat] = frange.get('mean', (frange['min'] + frange['max']) / 2)

        sens_results = {tgt: [] for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']}
        for sv in sweep_vals:
            test_vals = {**default_vals, sel_feat: sv}
            for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
                res = predict_single(tgt, test_vals, feature_meta, models, scalers, imputers)
                sens_results[tgt].append(res.get('prediction', 0))

        fig_sens = go.Figure()
        for tgt, cfg in TARGET_CONFIG.items():
            # Normalize to [0, 100] for comparison
            vals_arr = np.array(sens_results[tgt])
            mn, mx = vals_arr.min(), vals_arr.max()
            normalized = (vals_arr - mn) / max(mx - mn, 1e-6) * 100
            fig_sens.add_trace(go.Scatter(
                x=sweep_vals, y=normalized,
                mode='lines+markers',
                name=cfg['label'],
                line={'color': cfg['color'], 'width': 2},
                marker={'size': 5},
                customdata=vals_arr,
                hovertemplate=f"<b>{cfg['label']}</b><br>"
                               + f"{FRIENDLY_NAMES.get(sel_feat, sel_feat)}: %{{x:.2f}}<br>"
                               + f"Value: %{{customdata:.3f}} {cfg['unit']}<br>"
                               + "Normalized: %{y:.1f}%<extra></extra>",
            ))
        fig_sens.update_layout(
            **PLOT_THEME, height=380,
            title={'text': f'Sensitivity: {FRIENDLY_NAMES.get(sel_feat, sel_feat)} vs All Targets (Normalized)',
                   'font': {'size': 13}},
            margin=dict(t=60, b=40, l=40, r=20),
            xaxis={'title': FRIENDLY_NAMES.get(sel_feat, sel_feat),
                   **GRID},
            yaxis={'title': 'Normalized Prediction (%)', **GRID, 'range': [-5, 110]},
            legend={'bgcolor': 'rgba(0,0,0,0)', 'bordercolor': 'rgba(255,255,255,0.1)'},
            hovermode='x unified',
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        # Absolute values
        fig_abs = go.Figure()
        for i, (tgt, cfg) in enumerate(TARGET_CONFIG.items()):
            fig_abs.add_trace(go.Scatter(
                x=sweep_vals, y=sens_results[tgt],
                mode='lines', name=cfg['label'],
                line={'color': cfg['color'], 'width': 2},
                yaxis=f'y{i+1}' if i > 0 else 'y',
            ))
        fig_abs.update_layout(
            **PLOT_THEME, height=320,
            title={'text': 'Absolute Predicted Values', 'font': {'size': 12}},
            margin=dict(t=40, b=40, l=60, r=120),
            xaxis={'title': FRIENDLY_NAMES.get(sel_feat, sel_feat), **GRID},
            yaxis=dict(title=dict(text="Si (%)", font=dict(color='#6366f1')), **GRID),
            yaxis2=dict(title=dict(text="Temp (°C)", font=dict(color='#f97316')), overlaying='y', side='right',
                        gridcolor='rgba(255,255,255,0)', showgrid=False),
            yaxis3=dict(title=dict(text="Prod (t/hr)", font=dict(color='#22c55e')), overlaying='y', side='right',
                        position=0.92, gridcolor='rgba(255,255,255,0)', showgrid=False),
            legend={'bgcolor': 'rgba(0,0,0,0)'},
            hovermode='x unified',
        )
        st.plotly_chart(fig_abs, use_container_width=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SECTION 3: PARETO FRONT / SCENARIO COMPARISON
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🎭 Scenario Comparison</div>', unsafe_allow_html=True)
st.markdown("Compare up to **4 operating scenarios** side by side.")

n_scenarios = st.slider("Number of scenarios to compare:", 2, 4, 2, key="n_scenarios")

scenario_inputs  = {}
scenario_results = {}

scen_cols = st.columns(n_scenarios)
for i, scol in enumerate(scen_cols):
    with scol:
        st.markdown(f"**📋 Scenario {i+1}**")
        scen_vals = {}
        # Pick first 5 shared features for compact UI
        for feat in SHARED_FEATS[:5]:
            r = None
            for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
                if feat in (feature_meta['feature_ranges'].get(tgt) or {}):
                    r = feature_meta['feature_ranges'][tgt][feat]
                    break
            if r:
                val = st.number_input(
                    FRIENDLY_NAMES.get(feat, feat),
                    min_value=float(r['min']),
                    max_value=float(r['max']),
                    value=float(r.get('mean', (r['min'] + r['max']) / 2)),
                    step=float((r['max'] - r['min']) / 100),
                    key=f"scen_{i}_{feat}",
                    format="%.2f",
                )
                scen_vals[feat] = val
        scenario_inputs[f"Scenario {i+1}"] = scen_vals

if st.button("🔍 Compare Scenarios", type="primary", use_container_width=False, key="compare_btn"):
    for sname, svals in scenario_inputs.items():
        preds = {}
        for tgt in ['HM_Si', 'HM_Temp', 'Prod_Rate']:
            res = predict_single(tgt, svals, feature_meta, models, scalers, imputers)
            preds[tgt] = res.get('prediction', 0)
        scenario_results[sname] = preds

    if scenario_results:
        # Comparison table
        st.markdown("**📊 Prediction Results**")
        comp_data = {}
        for sname, preds in scenario_results.items():
            comp_data[sname] = {
                '⚗️ HM_Si (%)':       round(preds.get('HM_Si', 0), 4),
                '🌡️ HM_Temp (°C)':    round(preds.get('HM_Temp', 0), 2),
                '⚙️ Prod_Rate (t/hr)': round(preds.get('Prod_Rate', 0), 2),
            }
        comp_df = pd.DataFrame(comp_data).T
        st.dataframe(comp_df, use_container_width=True)

        # Grouped bar chart
        fig_comp = go.Figure()
        scen_names = list(scenario_results.keys())
        scenario_colors = ['#6366f1', '#f97316', '#22c55e', '#f59e0b']

        for tgt, cfg in TARGET_CONFIG.items():
            # Normalize for comparison
            vals = [scenario_results[s].get(tgt, 0) for s in scen_names]
            fig_comp.add_trace(go.Bar(
                name=cfg['label'], x=scen_names, y=vals,
                marker_color=cfg['color'], marker_opacity=0.85,
                text=[f'{v:.3f} {cfg["unit"]}' for v in vals],
                textposition='outside',
                textfont={'size': 10},
            ))
        fig_comp.update_layout(
            **PLOT_THEME, height=380, barmode='group',
            title={'text': 'Scenario Comparison — All Targets', 'font': {'size': 13}},
            margin=dict(t=60, b=40, l=40, r=20),
            xaxis={'title': 'Scenario', **GRID},
            yaxis={'title': 'Predicted Value', **GRID},
            legend={'bgcolor': 'rgba(0,0,0,0)'},
        )
        st.plotly_chart(fig_comp, use_container_width=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  SECTION 4: OPERATIONAL RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">💡 Operational Recommendations</div>', unsafe_allow_html=True)

rec_cols = st.columns(3)
recommendations = {
    'HM_Si':     {
        'icon': '⚗️', 'color': '#6366f1',
        'title': 'Silicon Control',
        'good': '0.30 – 0.60 %',
        'tips': [
            'Increase HBT to reduce Si absorption',
            'Optimize coal injection rate',
            'Monitor RAFT temperature closely',
            'Maintain stable blast volume',
        ],
        'warning': 'Si > 0.8% indicates excessive thermal state',
    },
    'HM_Temp':   {
        'icon': '🌡️', 'color': '#f97316',
        'title': 'Temperature Control',
        'good': '1480 – 1560 °C',
        'tips': [
            'Increase HBT to raise HM temperature',
            'Ensure adequate oxygen enrichment',
            'Monitor slag basicity (B2 = 1.1)',
            'Control blast moisture content',
        ],
        'warning': 'Temp < 1460°C risk of skull formation',
    },
    'Prod_Rate': {
        'icon': '⚙️', 'color': '#22c55e',
        'title': 'Production Optimization',
        'good': '140 – 200 t/hr',
        'tips': [
            'Maximize blast volume within pressure limits',
            'Optimize oxygen enrichment (target 3–5%)',
            'Maintain high burden permeability',
            'Minimize hanging and slipping',
        ],
        'warning': 'Prod > 210 t/hr may stress cooling system',
    },
}

for col, (tgt, cfg) in zip(rec_cols, recommendations.items()):
    with col:
        tips_html = "".join([f"<li style='margin-bottom:6px;'>{t}</li>" for t in cfg['tips']])
        st.markdown(f"""
        <div class="tradeoff-card" style="border-color: {cfg['color']}33;">
            <div style="font-size:1.5rem; margin-bottom:8px;">{cfg['icon']}</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:4px;">{cfg['title']}</div>
            <div style="background:{cfg['color']}22; color:{cfg['color']}; padding:4px 10px;
                        border-radius:20px; font-size:0.8rem; font-weight:600;
                        display:inline-block; margin-bottom:14px;">
                🎯 Target: {cfg['good']}
            </div>
            <ul style="color:#94a3b8; font-size:0.85rem; padding-left:16px; margin:0 0 12px;">
                {tips_html}
            </ul>
            <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.2);
                        border-radius:8px; padding:10px; font-size:0.8rem; color:#fca5a5;">
                ⚠️ {cfg['warning']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding:24px;
            border-top:1px solid rgba(255,255,255,0.05); margin-top:40px;'>
    ⚖️ Trade-Off Optimization — Blast Furnace Intelligence Platform
</div>
""", unsafe_allow_html=True)
