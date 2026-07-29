"""
pages/1_Prediction.py
Multi-Pipeline ML Prediction Engine for Blast Furnace Hot Metal Quality:
- Regression Pipeline      : PARA + BURDEN -> HM_SI & HM_TEMP (actual values)
- Classification Pipeline : PARA + BURDEN -> HM_SI Class & HM_TEMP Class (Low / Normal / High)
- Time-Series Pipeline    : Previous HM + PARA + BURDEN -> Future HM_SI & Future HM_TEMP
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.predictor import (load_all_pipelines, predict_regression,
                             predict_classification, predict_timeseries,
                             get_feature_contributions)

st.set_page_config(
    page_title="Prediction Engine — BF Intelligence",
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
    padding: 24px 30px;
    margin-bottom: 24px;
}
.page-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #f97316 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.page-subtitle { color: #94a3b8; font-size: 0.95rem; margin-top: 6px; }

.pipeline-card {
    background: rgba(15,23,42,0.85);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(99,102,241,0.25);
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(249,115,22,0.08));
    border: 2px solid rgba(99,102,241,0.4);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 16px;
}
.res-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.res-val   { font-size: 2.6rem; font-weight: 800; color: #6366f1; line-height: 1.1; margin: 8px 0; }
.res-unit  { font-size: 1rem; color: #cbd5e1; }

.badge-normal { background: #059669; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
.badge-low    { background: #2563eb; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }
.badge-high   { background: #dc2626; color: #ffffff; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 1.1rem; display: inline-block; }

.model-badge {
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}
.timestamp-box {
    background: rgba(249,115,22,0.15);
    border: 1px solid rgba(249,115,22,0.35);
    border-radius: 10px;
    padding: 10px 16px;
    color: #fbbf24;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 16px;
    text-align: center;
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
    st.markdown("- 🎯 **Prediction Engine** ← You are here")
    st.markdown("- [📊 Analysis](/Analysis)")
    st.markdown("- [⚖️ Trade-Off](/Trade_Off)")
    st.markdown("---")

# ── Load Pipelines ─────────────────────────────────────────────
artifacts = load_all_pipelines()

st.markdown("""
<div class="page-header">
    <div class="page-title">🎯 Multi-Pipeline ML Prediction Engine</div>
    <div class="page-subtitle">
        Select a prediction pipeline tab below. Adjust operating parameters and click the prediction button to execute real-time model inference.
    </div>
</div>
""", unsafe_allow_html=True)

if not artifacts or 'meta' not in artifacts:
    st.error("⚠️ Model artifacts not loaded. Please wait for model training to finish.")
    st.stop()

meta = artifacts['meta']

# FRIENDLY LABELS
FRIENDLY_NAMES = {
    'Cold Blast Volume': 'Cold Blast Volume (Nm³/hr)',
    'HBT': 'Hot Blast Temp (°C)',
    'HBP': 'Hot Blast Pressure (kPa)',
    'Oxygen Flow': 'Oxygen Flow (Nm³/hr)',
    'Steam': 'Steam Injection (kg/hr)',
    'Coal Actual': 'Coal Injection (kg/tHM)',
    'PROD_RATE': 'Production Rate (t/hr)',
    'SLAG_RATE': 'Slag Rate (kg/tHM)',
    'Permeabilty': 'Permeability Index',
    'OreCokeRatio': 'Ore / Coke Ratio',
    'SinterFrac': 'Sinter Fraction in Burden',
    'PelletFrac': 'Pellet Fraction in Burden',
    'FluxIronRatio': 'Flux / Iron Ratio',
    'Coke_kg': 'Coke Consumption (kg/h)',
    'Sinter_kg': 'Sinter Weight (kg/h)',
    'Pellet_kg': 'Pellet Weight (kg/h)',
    'TotalIron_kg': 'Total Iron Charge (kg/h)',
    'thermal_idx': 'Thermal Index (HBT × O₂)',
    'burden_thermal': 'Burden Thermal Load',
    'HM_SI_lag1': 'Previous HM_SI (Lag 1)',
    'HM_TEMP_lag1': 'Previous HM_TEMP (Lag 1)',
    'HM_SI_lag2': 'Previous HM_SI (Lag 2)',
    'HM_TEMP_lag2': 'Previous HM_TEMP (Lag 2)',
    'HM_SI_d1': 'HM_SI Rate of Change (Δ1)',
    'HM_TEMP_d1': 'HM_TEMP Rate of Change (Δ1)',
}

tab_reg, tab_cls, tab_ts = st.tabs([
    "📈 1. Regression Pipeline",
    "🏷️ 2. Classification Pipeline",
    "⏱️ 3. Time-Series Pipeline"
])

# ==============================================================================
# TAB 1: REGRESSION PIPELINE
# ==============================================================================
with tab_reg:
    st.markdown("""
    ### 📈 Regression ML Pipeline
    **Input:** `PARA` + `BURDEN` process parameters  
    **Output:** Continuous predictions for **HM_SI** (%Si) and **HM_TEMP** (°C)
    """)
    st.markdown("---")

    reg_meta = meta['regression']
    all_reg_feats = list(dict.fromkeys(reg_meta['HM_Si']['top_features'] + reg_meta['HM_Temp']['top_features']))

    col_inputs, col_outputs = st.columns([1.4, 1.0])

    input_vals_reg = {}
    with col_inputs:
        st.subheader("🎛️ Process & Burden Inputs")
        n_feats = len(all_reg_feats)
        for i in range(0, n_feats, 2):
            c1, c2 = st.columns(2)
            batch = all_reg_feats[i:i+2]
            for idx, feat in enumerate(batch):
                target_col = c1 if idx == 0 else c2
                r_info = reg_meta['HM_Si']['feature_ranges'].get(feat, reg_meta['HM_Temp']['feature_ranges'].get(feat, {'min': 0.0, 'max': 100.0, 'mean': 50.0}))
                fname = FRIENDLY_NAMES.get(feat, feat)
                with target_col:
                    val = st.slider(
                        fname,
                        min_value=float(r_info['min']),
                        max_value=float(r_info['max']),
                        value=float(r_info['mean']),
                        key=f"reg_s_{feat}"
                    )
                    input_vals_reg[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        btn_reg = st.button("🔥 Run Regression Prediction", type="primary", key="btn_run_reg", use_container_width=True)

    if btn_reg:
        st.session_state['reg_preds'] = predict_regression(artifacts, input_vals_reg)
        st.session_state['reg_inputs'] = input_vals_reg

    with col_outputs:
        st.subheader("🎯 Regression Predictions")
        
        if 'reg_preds' in st.session_state:
            reg_preds = st.session_state['reg_preds']
            last_inputs = st.session_state.get('reg_inputs', input_vals_reg)

            # HM_SI Result Card
            si_res = reg_preds['HM_Si']
            si_val = si_res['prediction']
            si_status = "Normal (In Spec)" if 0.25 <= si_val <= 0.80 else ("Low (< 0.25)" if si_val < 0.25 else "High (> 0.80)")
            si_color = "#059669" if 0.25 <= si_val <= 0.80 else ("#2563eb" if si_val < 0.25 else "#dc2626")

            st.markdown(f"""
            <div class="result-card">
                <div class="res-label">Silicon Content (HM_SI)</div>
                <div class="res-val" style="color:{si_color};">{si_val:.3f} <span style="font-size:1.2rem;">%Si</span></div>
                <div>Status: <b style="color:{si_color};">{si_status}</b></div>
                <div style="margin-top:8px;"><span class="model-badge">Best Model: {si_res['best_model']} (R²: {si_res['metrics'].get('r2', 0):.3f})</span></div>
            </div>
            """, unsafe_allow_html=True)

            # HM_TEMP Result Card
            temp_res = reg_preds['HM_Temp']
            temp_val = temp_res['prediction']
            temp_status = "Normal (In Spec)" if 1480 <= temp_val <= 1535 else ("Low (< 1480°C)" if temp_val < 1480 else "High (> 1535°C)")
            temp_color = "#059669" if 1480 <= temp_val <= 1535 else ("#2563eb" if temp_val < 1480 else "#dc2626")

            st.markdown(f"""
            <div class="result-card">
                <div class="res-label">Hot Metal Temperature (HM_TEMP)</div>
                <div class="res-val" style="color:{temp_color};">{temp_val:.1f} <span style="font-size:1.2rem;">°C</span></div>
                <div>Status: <b style="color:{temp_color};">{temp_status}</b></div>
                <div style="margin-top:8px;"><span class="model-badge">Best Model: {temp_res['best_model']} (R²: {temp_res['metrics'].get('r2', 0):.3f})</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Top Feature Contributions")
            si_contribs = get_feature_contributions(artifacts, 'regression', 'HM_Si', last_inputs)
            fig_si = px.bar(
                si_contribs.head(6), x='Contribution', y='Feature', orientation='h',
                title='HM_SI Impact Factors',
                color='Contribution', color_continuous_scale='RdBu_r'
            )
            fig_si.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
            st.plotly_chart(fig_si, use_container_width=True)
        else:
            st.info("💡 Adjust process parameters on the left and click **'Run Regression Prediction'** to calculate model predictions.")

# ==============================================================================
# TAB 2: CLASSIFICATION PIPELINE
# ==============================================================================
with tab_cls:
    st.markdown("""
    ### 🏷️ Classification Pipeline
    **Input:** `PARA` + `BURDEN` process parameters  
    **Output:** Predicted Quality Class (**Low** / **Normal** / **High**) with confidence probabilities
    """)
    st.markdown("---")

    cls_meta = meta['classification']
    all_cls_feats = list(dict.fromkeys(cls_meta['HM_Si']['top_features'] + cls_meta['HM_Temp']['top_features']))

    col_inputs_c, col_outputs_c = st.columns([1.4, 1.0])

    input_vals_cls = {}
    with col_inputs_c:
        st.subheader("🎛️ Process & Burden Inputs")
        n_feats = len(all_cls_feats)
        for i in range(0, n_feats, 2):
            c1, c2 = st.columns(2)
            batch = all_cls_feats[i:i+2]
            for idx, feat in enumerate(batch):
                target_col = c1 if idx == 0 else c2
                r_info = cls_meta['HM_Si']['feature_ranges'].get(feat, cls_meta['HM_Temp']['feature_ranges'].get(feat, {'min': 0.0, 'max': 100.0, 'mean': 50.0}))
                fname = FRIENDLY_NAMES.get(feat, feat)
                with target_col:
                    val = st.slider(
                        fname,
                        min_value=float(r_info['min']),
                        max_value=float(r_info['max']),
                        value=float(r_info['mean']),
                        key=f"cls_s_{feat}"
                    )
                    input_vals_cls[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        btn_cls = st.button("🎯 Run Classification Prediction", type="primary", key="btn_run_cls", use_container_width=True)

    if btn_cls:
        st.session_state['cls_preds'] = predict_classification(artifacts, input_vals_cls)

    with col_outputs_c:
        st.subheader("🎯 Classification Predictions")
        
        if 'cls_preds' in st.session_state:
            cls_preds = st.session_state['cls_preds']

            # HM_SI Class
            si_cls_res = cls_preds['HM_Si']
            si_class = si_cls_res['predicted_class']
            badge_cls_si = "badge-normal" if si_class == "Normal" else ("badge-low" if si_class == "Low" else "badge-high")

            st.markdown(f"""
            <div class="result-card">
                <div class="res-label">HM_SI Class</div>
                <div style="margin: 12px 0;"><span class="{badge_cls_si}">{si_class.upper()}</span></div>
                <div style="margin-top:8px;"><span class="model-badge">Model: {si_cls_res['best_model']} (Acc: {si_cls_res['metrics'].get('accuracy', 0)*100:.1f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)

            if si_cls_res['probabilities']:
                df_p = pd.DataFrame(list(si_cls_res['probabilities'].items()), columns=['Class', 'Probability'])
                fig_p_si = px.bar(df_p, x='Class', y='Probability', color='Class',
                                  color_discrete_map={'Low': '#2563eb', 'Normal': '#059669', 'High': '#dc2626'},
                                  title='HM_SI Class Probabilities')
                fig_p_si.update_layout(height=180, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
                st.plotly_chart(fig_p_si, use_container_width=True)

            # HM_TEMP Class
            temp_cls_res = cls_preds['HM_Temp']
            temp_class = temp_cls_res['predicted_class']
            badge_cls_temp = "badge-normal" if temp_class == "Normal" else ("badge-low" if temp_class == "Low" else "badge-high")

            st.markdown(f"""
            <div class="result-card">
                <div class="res-label">HM_TEMP Class</div>
                <div style="margin: 12px 0;"><span class="{badge_cls_temp}">{temp_class.upper()}</span></div>
                <div style="margin-top:8px;"><span class="model-badge">Model: {temp_cls_res['best_model']} (Acc: {temp_cls_res['metrics'].get('accuracy', 0)*100:.1f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)

            if temp_cls_res['probabilities']:
                df_pt = pd.DataFrame(list(temp_cls_res['probabilities'].items()), columns=['Class', 'Probability'])
                fig_p_t = px.bar(df_pt, x='Class', y='Probability', color='Class',
                                 color_discrete_map={'Low': '#2563eb', 'Normal': '#059669', 'High': '#dc2626'},
                                 title='HM_TEMP Class Probabilities')
                fig_p_t.update_layout(height=180, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
                st.plotly_chart(fig_p_t, use_container_width=True)
        else:
            st.info("💡 Adjust process parameters on the left and click **'Run Classification Prediction'** to classify hot metal quality.")

# ==============================================================================
# TAB 3: TIME-SERIES PIPELINE
# ==============================================================================
with tab_ts:
    st.markdown("""
    ### ⏱️ Time-Series Pipeline
    **Input:** Previous HM state (`HM_SI_lag1`, `HM_TEMP_lag1`) + `PARA` + `BURDEN` process variables  
    **Output:** Next-Step Horizon Forecast for **Future HM_SI** (%Si) and **Future HM_TEMP** (°C) with exact time horizon
    """)
    st.markdown("---")

    ts_meta = meta['timeseries']
    all_ts_feats = list(dict.fromkeys(ts_meta['HM_Si']['top_features'] + ts_meta['HM_Temp']['top_features']))

    col_inputs_ts, col_outputs_ts = st.columns([1.4, 1.0])

    input_vals_ts = {}
    with col_inputs_ts:
        st.subheader("⏱️ Forecast Time & Operational Inputs")
        
        c_time1, c_time2 = st.columns(2)
        with c_time1:
            base_date = st.date_input("📅 Current Tap Date", datetime.now().date(), key="ts_base_date")
        with c_time2:
            base_hour = st.selectbox("🕒 Current Tap Hour", [f"{h:02d}:00" for h in range(24)], index=datetime.now().hour, key="ts_base_hour")

        # Compute exact base and forecast timestamps
        base_time_str = f"{base_date} {base_hour}:00"
        try:
            base_dt = datetime.strptime(base_time_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            base_dt = datetime.now()
        
        forecast_dt = base_dt + timedelta(hours=1)
        base_str_formatted = base_dt.strftime("%Y-%m-%d %H:%M")
        forecast_str_formatted = forecast_dt.strftime("%Y-%m-%d %H:%M")

        st.markdown(f"""
        <div class="timestamp-box">
            ⏱️ <b>Base Tap Time:</b> {base_str_formatted} &nbsp; ➔ &nbsp; 🔮 <b>Forecast Horizon:</b> {forecast_str_formatted} (+1 hr)
        </div>
        """, unsafe_allow_html=True)

        n_feats = len(all_ts_feats)
        for i in range(0, n_feats, 2):
            c1, c2 = st.columns(2)
            batch = all_ts_feats[i:i+2]
            for idx, feat in enumerate(batch):
                target_col = c1 if idx == 0 else c2
                r_info = ts_meta['HM_Si']['feature_ranges'].get(feat, ts_meta['HM_Temp']['feature_ranges'].get(feat, {'min': 0.0, 'max': 100.0, 'mean': 50.0}))
                fname = FRIENDLY_NAMES.get(feat, feat)
                with target_col:
                    val = st.slider(
                        fname,
                        min_value=float(r_info['min']),
                        max_value=float(r_info['max']),
                        value=float(r_info['mean']),
                        key=f"ts_s_{feat}"
                    )
                    input_vals_ts[feat] = val

        st.markdown("<br>", unsafe_allow_html=True)
        btn_ts = st.button("⏱️ Forecast Next Tap Quality", type="primary", key="btn_run_ts", use_container_width=True)

    if btn_ts:
        st.session_state['ts_preds'] = predict_timeseries(artifacts, input_vals_ts)
        st.session_state['ts_inputs'] = input_vals_ts
        st.session_state['ts_time_base'] = base_str_formatted
        st.session_state['ts_time_forecast'] = forecast_str_formatted

    with col_outputs_ts:
        st.subheader("🔮 Next-Tap Timestamped Forecast")
        
        if 'ts_preds' in st.session_state:
            ts_preds = st.session_state['ts_preds']
            last_ts_inputs = st.session_state.get('ts_inputs', input_vals_ts)
            t_base = st.session_state.get('ts_time_base', base_str_formatted)
            t_fut = st.session_state.get('ts_time_forecast', forecast_str_formatted)

            # Future HM_SI Forecast
            fut_si = ts_preds['HM_Si']['prediction']
            st.markdown(f"""
            <div class="result-card">
                <div class="res-label">Future HM_SI (%Si)</div>
                <div class="res-val" style="color:#818cf8;">{fut_si:.3f} <span style="font-size:1.2rem;">%Si</span></div>
                <div style="font-size:0.9rem; color:#e2e8f0; margin-top:4px;">Forecast Timestamp: <b style="color:#fbbf24;">{t_fut}</b></div>
                <div style="margin-top:8px;"><span class="model-badge">Best TS Model: {ts_preds['HM_Si']['best_model']} (R²: {ts_preds['HM_Si']['metrics'].get('r2', 0):.3f})</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Future HM_TEMP Forecast
            fut_temp = ts_preds['HM_Temp']['prediction']
            st.markdown(f"""
            <div class="result-card">
                <div class="res-label">Future HM_TEMP (°C)</div>
                <div class="res-val" style="color:#f97316;">{fut_temp:.1f} <span style="font-size:1.2rem;">°C</span></div>
                <div style="font-size:0.9rem; color:#e2e8f0; margin-top:4px;">Forecast Timestamp: <b style="color:#fbbf24;">{t_fut}</b></div>
                <div style="margin-top:8px;"><span class="model-badge">Best TS Model: {ts_preds['HM_Temp']['best_model']} (R²: {ts_preds['HM_Temp']['metrics'].get('r2', 0):.3f})</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Forecast Trajectory visualization with timestamps
            prev_si = last_ts_inputs.get('HM_SI_lag1', 0.45)
            prev_temp = last_ts_inputs.get('HM_TEMP_lag1', 1505.0)

            df_traj = pd.DataFrame({
                'Time': [f"Prev Tap ({t_base})", f"Forecast Tap ({t_fut})"],
                'HM_SI': [prev_si, fut_si],
                'HM_TEMP': [prev_temp, fut_temp]
            })

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(x=df_traj['Time'], y=df_traj['HM_SI'], mode='lines+markers+text',
                                         text=[f"{prev_si:.3f}", f"{fut_si:.3f}"], textposition="top center",
                                         name='HM_SI (%Si)', line=dict(color='#818cf8', width=3),
                                         marker=dict(size=10)))
            fig_traj.update_layout(title=f'HM_SI Trajectory ({t_base} ➔ {t_fut})', height=200,
                                   margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("💡 Set current tap date/hour on the left, adjust parameters, and click **'Forecast Next Tap Quality'** to run the time-series prediction.")
