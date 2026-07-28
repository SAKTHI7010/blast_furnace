# -*- coding: utf-8 -*-
# =============================================================
#  BLAST FURNACE PREDICTION PIPELINE
#  Targets: HM_Si, HM_Temp, PROD_RATE
#  Datasets: PARA.xlsx, HM_ANALYSIS.xls, BURDEN.xlsx
#  Saves: models/*.pkl for Streamlit app
# =============================================================

import os, sys, warnings, joblib
import numpy as np
import pandas as pd
# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from xgboost                  import XGBRegressor

warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("="*65)
print("  BLAST FURNACE ML TRAINING PIPELINE")
print("="*65)

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("\n[1] Loading datasets...")

# PARA.xlsx
para_raw = pd.read_excel('PARA.xlsx', sheet_name=1, header=None)
col_names_fixed = [
    'CLOCK' if i == 0 else (str(v) if pd.notna(v) else f'col_{i}')
    for i, v in enumerate(para_raw.iloc[1].tolist())
]
para = para_raw.iloc[3:].copy()
para.columns = col_names_fixed
para = para.reset_index(drop=True)
para['CLOCK'] = pd.to_datetime(para['CLOCK'], errors='coerce')
for c in para.columns:
    if c != 'CLOCK':
        para[c] = pd.to_numeric(para[c], errors='coerce')
para = para.dropna(subset=['CLOCK']).sort_values('CLOCK').reset_index(drop=True)
print(f"  PARA   : {para.shape[0]:,} rows | Columns: {list(para.columns[1:])}")

# HM_ANALYSIS.xls
hm = pd.read_excel('HM_ANALYSIS.xls')
hm['SAMPLETAKEN'] = pd.to_datetime(hm['SAMPLETAKEN'], errors='coerce')
hm = (hm.dropna(subset=['SAMPLETAKEN', 'HM_SI', 'HM_TEMP'])
        .sort_values('SAMPLETAKEN').reset_index(drop=True))
print(f"  HM     : {hm.shape[0]:,} taps  | Targets: HM_SI, HM_TEMP")

# BURDEN.xlsx
burden = pd.read_excel('BURDEN.xlsx')
burden['CHARGETIME'] = pd.to_datetime(burden['CHARGETIME'], errors='coerce')
burden['BRANDCODE']  = burden['BRANDCODE'].str.strip()
burden['hour']       = burden['CHARGETIME'].dt.floor('h')
bp = (burden.pivot_table(index='hour', columns='BRANDCODE',
                          values='ACTWT', aggfunc='sum')
              .reset_index().sort_values('hour').reset_index(drop=True))
bp.columns.name = None
bp.columns = ['hour'] + [f'BRD_{c}' for c in bp.columns[1:]]
print(f"  Burden : {bp.shape[0]:,} hourly bundles | {len(bp.columns)-1} brand codes")

# ── 2. MERGE ─────────────────────────────────────────────────
print("\n[2] Merging datasets...")
pb = pd.merge_asof(
    para.sort_values('CLOCK'), bp.sort_values('hour'),
    left_on='CLOCK', right_on='hour', direction='backward'
).drop(columns=['hour'])

full = pd.merge_asof(
    hm.sort_values('SAMPLETAKEN'), pb.sort_values('CLOCK'),
    left_on='SAMPLETAKEN', right_on='CLOCK', direction='backward'
)
full = full.dropna(subset=['CLOCK'])
full['lag_hours'] = (full['SAMPLETAKEN'] - full['CLOCK']).dt.total_seconds() / 3600
full = full[full['lag_hours'] <= 4].reset_index(drop=True)
print(f"  Merged (<=4 h lag): {len(full):,} records")

# ── 3. OUTLIER REMOVAL ───────────────────────────────────────
def iqr_filter(df, col, k=3):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    return df[(df[col] >= q1 - k*(q3-q1)) & (df[col] <= q3 + k*(q3-q1))]

print("\n[3] Removing outliers (3xIQR)...")
n_raw = len(full)
for col in ['HM_SI', 'HM_TEMP', 'PROD_RATE']:
    if col in full.columns:
        full = iqr_filter(full, col).reset_index(drop=True)
print(f"  After cleanup: {len(full):,} records (removed {n_raw - len(full)})")

# Check PROD_RATE
if 'PROD_RATE' not in full.columns:
    # Fallback: compute from BURDEN total hourly weight
    burden['hour'] = burden['CHARGETIME'].dt.floor('h')
    hourly_prod = burden.groupby('hour')['ACTWT'].sum().reset_index()
    hourly_prod.columns = ['hour', 'PROD_RATE']
    full['hour_key'] = full['SAMPLETAKEN'].dt.floor('h')
    full = pd.merge_asof(
        full.sort_values('SAMPLETAKEN'),
        hourly_prod.rename(columns={'hour': 'SAMPLETAKEN_h'}).sort_values('SAMPLETAKEN_h'),
        left_on='hour_key', right_on='SAMPLETAKEN_h', direction='backward'
    )
    print("  PROD_RATE computed from BURDEN ACTWT (fallback)")
else:
    print(f"  PROD_RATE found in PARA: range [{full['PROD_RATE'].min():.1f}, {full['PROD_RATE'].max():.1f}]")

print(f"  HM_SI  range: [{full['HM_SI'].min():.3f}, {full['HM_SI'].max():.3f}]")
print(f"  HM_TEMP range: [{full['HM_TEMP'].min():.1f}, {full['HM_TEMP'].max():.1f}] °C")

# ── 4. FEATURE MATRIX ────────────────────────────────────────
print("\n[4] Building feature matrix...")

# Columns to always exclude from features
EXCLUDE = {
    'SAMPLETAKEN', 'TAPNO', 'TAPHOLE', 'SAMPLENO',
    'HM_C', 'HM_MN', 'HM_P', 'HM_S', 'HM_TI',
    'HM_SI', 'HM_TEMP', 'PROD_RATE',
    'CLOCK', 'lag_hours', 'hour_key', 'SAMPLETAKEN_h'
}

df_feat = full.drop(columns=[c for c in full.columns if c in EXCLUDE], errors='ignore')
df_feat = df_feat.apply(pd.to_numeric, errors='coerce')
df_feat = df_feat.dropna(axis=1, how='all')

# Remove near-zero variance columns
var = df_feat.var()
df_feat = df_feat.loc[:, var > 0.001]

all_feature_names = df_feat.columns.tolist()
print(f"  Total candidate features: {len(all_feature_names)}")

# Global imputer for all features
imputer_global = SimpleImputer(strategy='median')
X_all = imputer_global.fit_transform(df_feat)

# ── 5. TRAIN MODELS FOR EACH TARGET ──────────────────────────
print("\n[5] Training models...")

TARGETS = {
    'HM_Si':   ('HM_SI',   'regression'),
    'HM_Temp': ('HM_TEMP', 'regression'),
    'Prod_Rate': ('PROD_RATE', 'regression'),
}

TOP_N_FEATURES = 9  # features per target for Streamlit sliders

results    = {}
best_models_dict = {}
scalers_dict     = {}
imputers_dict    = {}
top_features_dict = {}
feature_importance_dict = {}
model_metrics_dict = {}
data_stats = {}

print("="*65)
for tgt_name, (ycol, task) in TARGETS.items():
    if ycol not in full.columns:
        print(f"  Skipping {tgt_name}: column {ycol} not found")
        continue

    y_all = full[ycol].dropna().values
    # Align X with non-null y
    mask = full[ycol].notna()
    X_tgt = X_all[mask]
    y_tgt = full.loc[mask, ycol].values

    print(f"\n  TARGET: {tgt_name} ({ycol})  |  {len(y_tgt):,} samples")
    print(f"  Range: [{y_tgt.min():.3f}, {y_tgt.max():.3f}]  Mean: {y_tgt.mean():.3f}")

    # ── Step A: Preliminary Random Forest for feature importance ──
    pre_rf = RandomForestRegressor(n_estimators=50, max_depth=8,
                                   random_state=42, n_jobs=-1)
    pre_rf.fit(X_tgt, y_tgt)
    fi = pd.Series(pre_rf.feature_importances_, index=all_feature_names)
    fi_sorted = fi.sort_values(ascending=False)
    top_features = fi_sorted.head(TOP_N_FEATURES).index.tolist()
    top_features_dict[tgt_name] = top_features
    feature_importance_dict[tgt_name] = fi_sorted.head(20).to_dict()
    print(f"  Top {TOP_N_FEATURES} features: {top_features}")

    # ── Step B: Train on top features only ──
    # Get indices of top features
    top_idx = [all_feature_names.index(f) for f in top_features]
    X_top = X_tgt[:, top_idx]

    # Imputer and scaler for top features
    imp_top = SimpleImputer(strategy='median')
    X_top   = imp_top.fit_transform(X_top)
    sc      = StandardScaler()
    X_sc    = sc.fit_transform(X_top)

    # Store
    imputers_dict[tgt_name] = imp_top
    scalers_dict[tgt_name]  = sc

    # Data stats for UI
    data_stats[tgt_name] = {
        'min': float(y_tgt.min()),
        'max': float(y_tgt.max()),
        'mean': float(y_tgt.mean()),
        'std': float(y_tgt.std()),
    }

    # ── Step C: Train-Test Split ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_tgt, test_size=0.2, random_state=42)

    # ── Step D: Train 3 models ──
    models = {
        'Random Forest':   RandomForestRegressor(n_estimators=200, max_depth=10,
                                                  random_state=42, n_jobs=-1),
        'Gradient Boost':  GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                                      max_depth=5, random_state=42),
        'XGBoost':         XGBRegressor(n_estimators=200, learning_rate=0.08,
                                         max_depth=6, random_state=42,
                                         eval_metric='rmse', n_jobs=-1),
    }

    model_results = {}
    print(f"  {'Model':<22} {'R2':<8} {'MAE':<10} {'RMSE':<10}")
    print(f"  {'--'*26}")

    for mname, clf in models.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        model_results[mname] = {
            'r2': round(r2, 4), 'mae': round(mae, 4),
            'rmse': round(rmse, 4), 'clf': clf
        }
        print(f"  {mname:<22} {r2:<8.4f} {mae:<10.4f} {rmse:<10.4f}")

    # Best model by R²
    best_name = max(model_results, key=lambda k: model_results[k]['r2'])
    best_clf  = model_results[best_name]['clf']
    print(f"  BEST: {best_name}  (R²={model_results[best_name]['r2']:.4f})")

    # Cross-val on best
    cv_r2 = cross_val_score(best_clf, X_sc, y_tgt, cv=5, scoring='r2')
    print(f"  CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    best_models_dict[tgt_name] = best_clf
    model_metrics_dict[tgt_name] = {
        'best_model': best_name,
        'r2':   model_results[best_name]['r2'],
        'mae':  model_results[best_name]['mae'],
        'rmse': model_results[best_name]['rmse'],
        'cv_r2_mean': round(cv_r2.mean(), 4),
        'cv_r2_std':  round(cv_r2.std(), 4),
        'all_models': {k: {m: v for m, v in vv.items() if m != 'clf'}
                       for k, vv in model_results.items()},
    }

# ── 6. SAVE ALL ARTIFACTS ─────────────────────────────────────
print("\n[6] Saving model artifacts to models/...")

# Collect feature ranges from data for slider UI
feature_ranges = {}
for tgt_name, top_feats in top_features_dict.items():
    feature_ranges[tgt_name] = {}
    for f in top_feats:
        if f in all_feature_names:
            col_data = df_feat[f].dropna()
            feature_ranges[tgt_name][f] = {
                'min':  float(col_data.quantile(0.01)),
                'max':  float(col_data.quantile(0.99)),
                'mean': float(col_data.mean()),
                'std':  float(col_data.std()),
                'p25':  float(col_data.quantile(0.25)),
                'p75':  float(col_data.quantile(0.75)),
            }

# Feature meta bundle
feature_meta = {
    'all_feature_names':      all_feature_names,
    'top_features':           top_features_dict,
    'feature_importance':     feature_importance_dict,
    'feature_ranges':         feature_ranges,
    'model_metrics':          model_metrics_dict,
    'data_stats':             data_stats,
    'target_columns':         {k: v[0] for k, v in TARGETS.items()},
    'n_records':              int(len(full)),
    'date_range':             {
        'start': str(full['SAMPLETAKEN'].min().date()),
        'end':   str(full['SAMPLETAKEN'].max().date()),
    }
}

# Save each model, scaler, imputer
for tgt_name in best_models_dict:
    joblib.dump(best_models_dict[tgt_name], f'models/model_{tgt_name}.pkl')
    joblib.dump(scalers_dict[tgt_name],     f'models/scaler_{tgt_name}.pkl')
    joblib.dump(imputers_dict[tgt_name],    f'models/imputer_{tgt_name}.pkl')
    print(f"  Saved: models/model_{tgt_name}.pkl + scaler + imputer")

# Save global imputer
joblib.dump(imputer_global, 'models/imputer_global.pkl')
# Save feature meta
joblib.dump(feature_meta,   'models/feature_meta.pkl')
print("  Saved: models/imputer_global.pkl")
print("  Saved: models/feature_meta.pkl")

# ── 7. SAVE DATA SNAPSHOT FOR EDA ─────────────────────────────
print("\n[7] Saving EDA data snapshot...")
eda_cols = ['SAMPLETAKEN', 'HM_SI', 'HM_TEMP', 'PROD_RATE'] + \
           [c for c in all_feature_names if c in full.columns][:20]
eda_df = full[[c for c in eda_cols if c in full.columns]].copy()
eda_df.to_parquet('models/eda_data.parquet', index=False)
print(f"  Saved: models/eda_data.parquet  ({len(eda_df):,} rows)")

# Save correlation matrix
numeric_full = full.select_dtypes(include='number').dropna(axis=1, how='all')
corr_targets = ['HM_SI', 'HM_TEMP', 'PROD_RATE']
top_corr_cols = []
for t in corr_targets:
    if t in numeric_full.columns:
        abs_corr = numeric_full.corr()[t].abs().drop(corr_targets, errors='ignore')
        top_corr_cols += abs_corr.nlargest(10).index.tolist()
top_corr_cols = list(set(top_corr_cols)) + corr_targets
corr_df = numeric_full[[c for c in top_corr_cols if c in numeric_full.columns]].corr()
corr_df.to_csv('models/correlation_matrix.csv')
print("  Saved: models/correlation_matrix.csv")

print("\n" + "="*65)
print("  TRAINING COMPLETE!")
print("="*65)
for tgt_name, metrics in model_metrics_dict.items():
    print(f"\n  {tgt_name}:")
    print(f"    Best Model : {metrics['best_model']}")
    print(f"    R²         : {metrics['r2']:.4f}")
    print(f"    MAE        : {metrics['mae']:.4f}")
    print(f"    RMSE       : {metrics['rmse']:.4f}")
    print(f"    CV R² mean : {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

print("\n  Run: streamlit run app.py")
print("="*65)
