"""
pages/2_Analysis.py
EDA and Analysis page — distributions, correlations, time-series, model metrics
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.predictor import load_all_models

st.set_page_config(
    page_title="Analysis — BF Intelligence",
    page_icon="📊",
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
    background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(99,102,241,0.1) 100%);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
.page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #22c55e, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.page-subtitle { color: #64748b; font-size: 0.95rem; margin-top: 6px; }

.stat-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.stat-value { font-size: 1.8rem; font-weight: 800; color: #6366f1; }
.stat-sub   { font-size: 0.8rem; color: #475569; margin-top: 4px; }

.section-title {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 2px solid rgba(34,197,94,0.3);
    padding-bottom: 10px; margin: 28px 0 20px;
}
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34,197,94,0.3), transparent);
    margin: 32px 0;
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
    st.markdown("- 📊 **Analysis** ← You are here")
    st.markdown("- [⚖️ Trade-Off](/Trade_Off)")
    st.markdown("---")

    st.markdown("**🔍 Analysis Filters**")
    show_section = st.multiselect(
        "Show Sections",
        ["Dataset Overview", "Target Distributions", "Time Series", "Correlations",
         "Feature Importance", "Model Metrics"],
        default=["Dataset Overview", "Target Distributions", "Time Series",
                 "Correlations", "Feature Importance", "Model Metrics"],
    )

# ── Load Models & Data ────────────────────────────────────────
feature_meta, models, scalers, imputers = load_all_models()

@st.cache_data(show_spinner="Loading EDA data...")
def load_eda_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'eda_data.parquet')
    if os.path.exists(data_path):
        return pd.read_parquet(data_path)
    return None

@st.cache_data(show_spinner="Loading correlation matrix...")
def load_corr_matrix():
    corr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'correlation_matrix.csv')
    if os.path.exists(corr_path):
        return pd.read_csv(corr_path, index_col=0)
    return None

eda_df   = load_eda_data()
corr_mat = load_corr_matrix()

PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': '#e2e8f0'},
)
GRID = dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)')

# ── Page Header ───────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">📊 EDA & Analysis Dashboard</div>
    <div class="page-subtitle">
        Explore the blast furnace dataset — distributions, trends, correlations, and model performance metrics.
    </div>
</div>
""", unsafe_allow_html=True)

if not feature_meta or eda_df is None:
    st.error("⚠️ Trained model data not found. Please run `python train_models.py` first.")
    st.stop()

TARGET_COLS = {
    'HM_SI':    {'label': 'Silicon Content (HM_Si)',   'unit': '%',    'color': '#6366f1'},
    'HM_TEMP':  {'label': 'Hot Metal Temperature',      'unit': '°C',   'color': '#f97316'},
    'PROD_RATE':{'label': 'Production Rate',             'unit': 't/hr', 'color': '#22c55e'},
}

# ── 1. DATASET OVERVIEW ───────────────────────────────────────
if "Dataset Overview" in show_section:
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    stats_display = [
        ("Total Records",    f"{feature_meta['n_records']:,}",              "merged taps"),
        ("Features (Total)", f"{len(feature_meta['all_feature_names'])}",   "process vars"),
        ("Date Start",       feature_meta['date_range']['start'],            ""),
        ("Date End",         feature_meta['date_range']['end'],              ""),
        ("Prediction Models","3",                                            "trained"),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4,c5], stats_display):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{lbl}</div>
                <div class="stat-value" style='font-size:1.3rem;'>{val}</div>
                <div class="stat-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    # Missing values heatmap
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**📊 Target Variable Statistics**")
        target_stats = []
        for col_name, cfg in TARGET_COLS.items():
            if col_name in eda_df.columns:
                s = eda_df[col_name].dropna()
                target_stats.append({
                    'Target': cfg['label'],
                    'Count': len(s),
                    'Mean': round(s.mean(), 4),
                    'Std': round(s.std(), 4),
                    'Min': round(s.min(), 4),
                    'Max': round(s.max(), 4),
                    'Unit': cfg['unit'],
                })
        if target_stats:
            st.dataframe(pd.DataFrame(target_stats).set_index('Target'),
                         use_container_width=True)

    with col_right:
        st.markdown("**❓ Missing Values by Column**")
        miss_data = {c: eda_df[c].isna().sum() for c in eda_df.columns if eda_df[c].isna().sum() > 0}
        if miss_data:
            miss_df = pd.DataFrame(list(miss_data.items()), columns=['Column', 'Missing'])
            miss_df['Missing %'] = (miss_df['Missing'] / len(eda_df) * 100).round(1)
            miss_df = miss_df.sort_values('Missing %', ascending=False).head(15)
            fig_miss = go.Figure(go.Bar(
                x=miss_df['Missing %'], y=miss_df['Column'], orientation='h',
                marker_color='#f97316', marker_opacity=0.8,
                text=[f'{v:.1f}%' for v in miss_df['Missing %']],
                textposition='outside',
            ))
            fig_miss.update_layout(
                **PLOT_THEME, height=250, margin=dict(t=10, b=10, l=10, r=60),
                xaxis={'title': 'Missing %', **GRID}, yaxis={'autorange': 'reversed', **GRID}
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅ No missing values in the merged dataset!")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── 2. TARGET DISTRIBUTIONS ───────────────────────────────────
if "Target Distributions" in show_section:
    st.markdown('<div class="section-title">📈 Target Variable Distributions</div>', unsafe_allow_html=True)

    dist_cols = st.columns(3)
    for col, (col_name, cfg) in zip(dist_cols, TARGET_COLS.items()):
        if col_name not in eda_df.columns:
            continue
        data = eda_df[col_name].dropna()
        with col:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data, nbinsx=60,
                marker_color=cfg['color'], marker_opacity=0.7,
                name='Distribution',
            ))
            # Add mean and std lines
            mean_val = data.mean()
            std_val  = data.std()
            fig.add_vline(x=mean_val, line_dash='dash', line_color='#fbbf24',
                          annotation_text=f'Mean: {mean_val:.2f}',
                          annotation_font_color='#fbbf24')
            fig.add_vline(x=mean_val - std_val, line_dash='dot', line_color='#64748b', line_width=1)
            fig.add_vline(x=mean_val + std_val, line_dash='dot', line_color='#64748b', line_width=1)

            fig.update_layout(
                **PLOT_THEME, height=300,
                title={'text': f"{cfg['icon'] if 'icon' in cfg else '📊'} {cfg['label']}", 'font': {'size': 13}},
                margin=dict(t=50, b=30, l=30, r=20),
                xaxis={'title': cfg['unit'], **GRID},
                yaxis={'title': 'Count', **GRID},
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Box plot below
            fig_box = go.Figure(go.Box(
                x=data, name=cfg['label'],
                marker_color=cfg['color'],
                boxpoints='outliers',
                line={'color': cfg['color']},
            ))
            fig_box.update_layout(
                **PLOT_THEME, height=140, margin=dict(t=5, b=30, l=10, r=10),
                xaxis={'title': cfg['unit'], **GRID}, yaxis=dict(visible=False),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── 3. TIME SERIES ────────────────────────────────────────────
if "Time Series" in show_section:
    st.markdown('<div class="section-title">📅 Time Series Trends</div>', unsafe_allow_html=True)

    time_target = st.selectbox(
        "Select Target for Time Series",
        options=[k for k in TARGET_COLS if k in eda_df.columns],
        format_func=lambda x: TARGET_COLS[x]['label'],
    )

    if 'SAMPLETAKEN' in eda_df.columns:
        ts_df = eda_df[['SAMPLETAKEN', time_target]].dropna().copy()
        ts_df['SAMPLETAKEN'] = pd.to_datetime(ts_df['SAMPLETAKEN'])
        ts_df = ts_df.sort_values('SAMPLETAKEN')

        # Rolling averages
        ts_df['MA_24h'] = ts_df[time_target].rolling(24, min_periods=1).mean()
        ts_df['MA_7d']  = ts_df[time_target].rolling(168, min_periods=1).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_df['SAMPLETAKEN'], y=ts_df[time_target],
            mode='lines', name='Raw', line={'color': TARGET_COLS[time_target]['color'],
                                             'width': 1, 'dash': 'solid'},
            opacity=0.4,
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_df['SAMPLETAKEN'], y=ts_df['MA_24h'],
            mode='lines', name='24h MA',
            line={'color': '#fbbf24', 'width': 2},
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_df['SAMPLETAKEN'], y=ts_df['MA_7d'],
            mode='lines', name='7-day MA',
            line={'color': '#f97316', 'width': 2},
        ))
        fig_ts.update_layout(
            **PLOT_THEME, height=380,
            title={'text': f"{TARGET_COLS[time_target]['label']} Over Time", 'font': {'size': 14}},
            margin=dict(t=50, b=40, l=40, r=20),
            xaxis={'title': 'Date', **GRID},
            yaxis={'title': TARGET_COLS[time_target]['unit'], **GRID},
            legend={'bgcolor': 'rgba(0,0,0,0)', 'bordercolor': 'rgba(255,255,255,0.1)'},
            hovermode='x unified',
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Monthly stats
        st.markdown("**📆 Monthly Statistics**")
        ts_df['Month'] = ts_df['SAMPLETAKEN'].dt.to_period('M').astype(str)
        monthly = ts_df.groupby('Month')[time_target].agg(['mean', 'std', 'min', 'max']).reset_index()
        monthly.columns = ['Month', 'Mean', 'Std Dev', 'Min', 'Max']
        monthly = monthly.round(3)

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly['Month'], y=monthly['Mean'],
            name='Monthly Mean', marker_color=TARGET_COLS[time_target]['color'],
            marker_opacity=0.8,
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Max'],
            mode='lines+markers', name='Max',
            line={'color': '#ef4444', 'dash': 'dot'},
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Min'],
            mode='lines+markers', name='Min',
            line={'color': '#22c55e', 'dash': 'dot'},
        ))
        fig_monthly.update_layout(
            **PLOT_THEME, height=320,
            margin=dict(t=20, b=60, l=40, r=20),
            xaxis={'title': 'Month', **GRID, 'tickangle': -45},
            yaxis={'title': TARGET_COLS[time_target]['unit'], **GRID},
            legend={'bgcolor': 'rgba(0,0,0,0)'},
            barmode='group',
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── 4. CORRELATIONS ───────────────────────────────────────────
if "Correlations" in show_section:
    st.markdown('<div class="section-title">🔗 Feature Correlations</div>', unsafe_allow_html=True)

    if corr_mat is not None:
        # Heatmap
        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat.values,
            x=corr_mat.columns.tolist(),
            y=corr_mat.index.tolist(),
            colorscale=[
                [0.0, '#ef4444'], [0.25, '#f97316'], [0.5, '#1e293b'],
                [0.75, '#6366f1'], [1.0, '#22c55e']
            ],
            zmid=0,
            text=np.round(corr_mat.values, 2),
            texttemplate='%{text}',
            textfont={'size': 9},
            colorbar={'title': 'Correlation', 'tickfont': {'color': '#94a3b8'}},
        ))
        fig_corr.update_layout(
            **PLOT_THEME, height=580,
            title={'text': 'Pearson Correlation Heatmap (Top Features × Targets)', 'font': {'size': 14}},
            margin=dict(t=60, b=80, l=80, r=20),
            xaxis={'tickangle': -45, 'tickfont': {'size': 9}, **GRID},
            yaxis={'tickfont': {'size': 9}, **GRID},
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations with each target
        st.markdown("**🏆 Strongest Correlations with Each Target**")
        corr_target_cols = st.columns(3)
        for col, (tgt_col, cfg) in zip(corr_target_cols, TARGET_COLS.items()):
            if tgt_col in corr_mat.columns:
                with col:
                    top_corr = (corr_mat[tgt_col]
                                .drop([t for t in TARGET_COLS if t in corr_mat.index], errors='ignore')
                                .abs().sort_values(ascending=False).head(8))
                    fig_top = go.Figure(go.Bar(
                        x=corr_mat.loc[top_corr.index, tgt_col],
                        y=top_corr.index,
                        orientation='h',
                        marker_color=[cfg['color'] if v > 0 else '#ef4444'
                                      for v in corr_mat.loc[top_corr.index, tgt_col]],
                        text=[f'{v:.3f}' for v in corr_mat.loc[top_corr.index, tgt_col]],
                        textposition='outside',
                        textfont={'size': 10},
                    ))
                    fig_top.update_layout(
                        **PLOT_THEME, height=300,
                        title={'text': cfg['label'], 'font': {'size': 12, 'color': cfg['color']}},
                        margin=dict(t=40, b=20, l=10, r=60),
                        xaxis={'range': [-1, 1], **GRID},
                        yaxis={'autorange': 'reversed', **GRID},
                        showlegend=False,
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── 5. FEATURE IMPORTANCE ─────────────────────────────────────
if "Feature Importance" in show_section:
    st.markdown('<div class="section-title">🏆 Feature Importance Rankings</div>', unsafe_allow_html=True)

    fi_tab1, fi_tab2, fi_tab3 = st.tabs(["⚗️ HM_Si", "🌡️ HM_Temp", "⚙️ Prod_Rate"])
    fi_target_map = {
        'HM_Si': fi_tab1, 'HM_Temp': fi_tab2, 'Prod_Rate': fi_tab3,
    }
    fi_colors = {'HM_Si': '#6366f1', 'HM_Temp': '#f97316', 'Prod_Rate': '#22c55e'}

    for tgt_name, tab in fi_target_map.items():
        with tab:
            fi_dict = feature_meta['feature_importance'].get(tgt_name, {})
            if not fi_dict:
                st.info("No feature importance data. Retrain models.")
                continue

            fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
            names = [x[0] for x in fi_sorted]
            vals  = [x[1] for x in fi_sorted]

            # Color top features differently
            c_list = []
            top9 = set(feature_meta['top_features'].get(tgt_name, []))
            for n in names:
                if n == names[0]:
                    c_list.append('#fbbf24')
                elif n in top9:
                    c_list.append(fi_colors[tgt_name])
                else:
                    c_list.append('rgba(100,116,139,0.4)')

            fig_fi = go.Figure(go.Bar(
                x=vals, y=names, orientation='h',
                marker_color=c_list,
                text=[f'{v:.4f}' for v in vals],
                textposition='outside',
                textfont={'size': 10, 'color': '#94a3b8'},
            ))
            fig_fi.update_layout(
                **PLOT_THEME, height=450,
                title={'text': f'Feature Importance — {tgt_name}  (Top {len(names)} features)',
                       'font': {'size': 13}},
                margin=dict(t=50, b=20, l=20, r=80),
                xaxis={'title': 'Importance Score', **GRID},
                yaxis={'autorange': 'reversed', **GRID},
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            st.markdown(f"""
            <div style='background:rgba(15,23,42,0.6); border-radius:8px; padding:12px;
                        border:1px solid rgba(255,255,255,0.05); font-size:0.85rem; color:#64748b;'>
            💡 <b style='color:{fi_colors[tgt_name]};'>Top 9 features</b> (highlighted) are used for prediction.
            The remaining features are shown for reference only.
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ── 6. MODEL METRICS ──────────────────────────────────────────
if "Model Metrics" in show_section:
    st.markdown('<div class="section-title">🤖 Model Performance Metrics</div>', unsafe_allow_html=True)

    model_metrics = feature_meta.get('model_metrics', {})
    if model_metrics:
        # Summary table
        rows = []
        for tgt, m in model_metrics.items():
            rows.append({
                'Target': tgt,
                'Best Model': m['best_model'],
                'R² Score': m['r2'],
                'MAE': m['mae'],
                'RMSE': m['rmse'],
                'CV R² (mean)': m['cv_r2_mean'],
                'CV R² (std)': m['cv_r2_std'],
            })
        metrics_df = pd.DataFrame(rows).set_index('Target')
        st.dataframe(metrics_df, use_container_width=True)

        # Comparison bar chart
        st.markdown("**📊 All Models Comparison**")
        comp_tabs = st.tabs([f"⚗️ {t}" for t in model_metrics.keys()])
        for tab, (tgt, m) in zip(comp_tabs, model_metrics.items()):
            with tab:
                all_mods = m.get('all_models', {})
                if all_mods:
                    mod_names = list(all_mods.keys())
                    r2s   = [all_mods[mn]['r2'] for mn in mod_names]
                    maes  = [all_mods[mn]['mae'] for mn in mod_names]
                    rmses = [all_mods[mn]['rmse'] for mn in mod_names]

                    fig_comp = go.Figure()
                    colors_comp = ['#6366f1', '#f97316', '#22c55e', '#f59e0b']
                    for i, mn in enumerate(mod_names):
                        fig_comp.add_trace(go.Bar(
                            name=mn, x=['R²', 'MAE', 'RMSE'],
                            y=[all_mods[mn]['r2'], all_mods[mn]['mae'], all_mods[mn]['rmse']],
                            marker_color=colors_comp[i % len(colors_comp)],
                            text=[f'{v:.4f}' for v in [all_mods[mn]['r2'],
                                                        all_mods[mn]['mae'],
                                                        all_mods[mn]['rmse']]],
                            textposition='outside',
                        ))
                    fig_comp.update_layout(
                        **PLOT_THEME, height=320, barmode='group',
                        margin=dict(t=20, b=40, l=40, r=20),
                        xaxis={'title': 'Metric', **GRID},
                        yaxis={'title': 'Score', **GRID},
                        legend={'bgcolor': 'rgba(0,0,0,0)'},
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding:24px;
            border-top:1px solid rgba(255,255,255,0.05); margin-top:40px;'>
    📊 Analysis Dashboard — Blast Furnace Intelligence Platform
</div>
""", unsafe_allow_html=True)
