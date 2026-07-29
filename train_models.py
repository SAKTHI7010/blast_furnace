# -*- coding: utf-8 -*-
"""
=============================================================
 BLAST FURNACE MULTI-PIPELINE ML TRAINING SCRIPT
 
 Pipelines Built:
 1. Regression ML Pipeline     : PARA + BURDEN -> HM_Si & HM_Temp (actual values)
 2. Classification Pipeline    : PARA + BURDEN -> HM_Si Class & HM_Temp Class (Low / Normal / High)
 3. Time-Series Pipeline       : Previous HM + PARA + BURDEN -> Future HM_Si & Future HM_Temp

 Saves model artifacts to models/ for Streamlit deployment.
 Cleanly deletes old obsolete model files.
=============================================================
"""

import os, sys, warnings, joblib, glob
import numpy as np
import pandas as pd

# Console encoding setup
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, f1_score)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("=" * 70)
print("  🔥 BLAST FURNACE MULTI-PIPELINE MODEL TRAINING")
print("=" * 70)

# Remove old obsolete model files
old_patterns = [
    'models/model_HM_Si.pkl', 'models/model_HM_Temp.pkl', 'models/model_Prod_Rate.pkl',
    'models/scaler_HM_Si.pkl', 'models/scaler_HM_Temp.pkl', 'models/scaler_Prod_Rate.pkl',
    'models/imputer_HM_Si.pkl', 'models/imputer_HM_Temp.pkl', 'models/imputer_Prod_Rate.pkl'
]
for p in old_patterns:
    if os.path.exists(p):
        try:
            os.remove(p)
            print(f"  🗑️ Removed old obsolete file: {p}")
        except Exception:
            pass

# ── 1. MATERIAL CLASSIFICATION HELPER ───────────────────────
def classify_material(brand):
    b = str(brand).upper()
    if 'COKE' in b or 'NUTCOKE' in b:   return 'Coke_kg'
    if 'SINTER' in b:                    return 'Sinter_kg'
    if 'PELLET' in b:                    return 'Pellet_kg'
    if 'ORE' in b or 'BHQ' in b:        return 'Ore_kg'
    if 'LIMESTONE' in b:                 return 'Limestone_kg'
    if 'DOLOMITE' in b:                  return 'Dolomite_kg'
    if 'DRI' in b:                       return 'DRI_kg'
    if 'MIXED' in b:                     return 'MixedMaterial_kg'
    return 'Other_kg'

# ── 2. LOAD DATASETS ─────────────────────────────────────────
print("\n[1/6] Loading datasets (PARA.xlsx, HM_ANALYSIS.xls, BURDEN.xlsx)...")

# A. PARA.xlsx
para_raw = pd.read_excel('PARA.xlsx', sheet_name=1, header=None)
col_names_fixed = [
    'CLOCK' if i == 0 else (str(v).strip() if pd.notna(v) else f'col_{i}')
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
para['CLOCK'] = para['CLOCK'].dt.floor('h')
para = para.groupby('CLOCK').mean(numeric_only=True).reset_index()

# Filter sensor faults
if 'Heat Flow Flux' in para.columns:
    para = para[para['Heat Flow Flux'] < 100]

print(f"  ✓ PARA loaded        : {len(para):,} hourly records | {len(para.columns)-1} parameters")

# B. HM_ANALYSIS.xls
hm_raw = pd.read_excel('HM_ANALYSIS.xls')
hm_raw['SAMPLETAKEN'] = pd.to_datetime(hm_raw['SAMPLETAKEN'], errors='coerce')
hm_raw = hm_raw.dropna(subset=['SAMPLETAKEN', 'HM_SI', 'HM_TEMP']).sort_values('SAMPLETAKEN').reset_index(drop=True)
hm_raw['CLOCK'] = hm_raw['SAMPLETAKEN'].dt.floor('h')
for c in [col for col in hm_raw.columns if col.startswith('HM_')]:
    if hm_raw[c].dtype == object:
        hm_raw[c] = pd.to_numeric(hm_raw[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
    else:
        hm_raw[c] = pd.to_numeric(hm_raw[c], errors='coerce')

hm_h = hm_raw.groupby('CLOCK')[['HM_SI', 'HM_TEMP']].mean().reset_index()
hm_h = hm_h.rename(columns={'HM_SI': 'HM_Si', 'HM_TEMP': 'HM_Temp'})
print(f"  ✓ HM_ANALYSIS loaded : {len(hm_raw):,} taps → {len(hm_h):,} hourly averages")

# C. BURDEN.xlsx
burden_raw = pd.read_excel('BURDEN.xlsx')
burden_raw['CHARGETIME'] = pd.to_datetime(burden_raw['CHARGETIME'], errors='coerce')
burden_raw['CLOCK'] = burden_raw['CHARGETIME'].dt.floor('h')
burden_raw['BRANDCODE'] = burden_raw['BRANDCODE'].astype(str).str.strip()
burden_raw['MaterialGroup'] = burden_raw['BRANDCODE'].apply(classify_material)

pivot_b = burden_raw.groupby(['CLOCK', 'MaterialGroup'])['ACTWT'].sum().unstack(fill_value=0).reset_index()
for col in ['Coke_kg', 'Sinter_kg', 'Pellet_kg', 'Ore_kg', 'Limestone_kg', 'Dolomite_kg', 'DRI_kg', 'MixedMaterial_kg', 'Other_kg']:
    if col not in pivot_b.columns:
        pivot_b[col] = 0.0

bh = pivot_b.copy()
bh['TotalIron_kg']   = bh['Ore_kg'] + bh['Pellet_kg'] + bh['Sinter_kg'] + bh['DRI_kg'] + bh['MixedMaterial_kg']
bh['TotalFlux_kg']   = bh['Limestone_kg'] + bh['Dolomite_kg']
bh['OreCokeRatio']   = bh['TotalIron_kg'] / bh['Coke_kg'].replace(0, np.nan)
bh['SinterFrac']     = bh['Sinter_kg'] / bh['TotalIron_kg'].replace(0, np.nan)
bh['PelletFrac']     = bh['Pellet_kg'] / bh['TotalIron_kg'].replace(0, np.nan)
bh['FluxIronRatio']  = bh['TotalFlux_kg'] / bh['TotalIron_kg'].replace(0, np.nan)
bh['TotalCharge_kg'] = bh['Coke_kg'] + bh['TotalIron_kg'] + bh['TotalFlux_kg']

# Apply 8-hour furnace descent delay
bh_lag = bh.copy()
bh_lag['CLOCK'] = bh_lag['CLOCK'] + pd.Timedelta(hours=8)
print(f"  ✓ BURDEN loaded      : {len(burden_raw):,} charges → {len(bh_lag):,} hourly records (lagged 8h)")

# ── 3. MERGE & CLEAN ─────────────────────────────────────────
print("\n[2/6] Merging PARA + BURDEN(8h lag) + HM_ANALYSIS ...")
merged = pd.merge(para, bh_lag, on='CLOCK', how='inner')
merged = pd.merge(merged, hm_h, on='CLOCK', how='inner')
merged = merged.sort_values('CLOCK').reset_index(drop=True)

def iqr_clean(df, cols, factor=3.0):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns: continue
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        mask &= (df[c] >= q1 - factor * iqr) & (df[c] <= q3 + factor * iqr)
    return df[mask].reset_index(drop=True)

target_cols = ['HM_Si', 'HM_Temp']
merged = iqr_clean(merged, target_cols)
merged = merged.dropna(subset=target_cols).reset_index(drop=True)
print(f"  ✓ Cleaned merged dataset: {len(merged):,} records ({merged['CLOCK'].min().date()} → {merged['CLOCK'].max().date()})")

# Domain Engineering
merged['thermal_idx']    = (merged['HBT'] * merged['Oxygen Flow'] / 1e6) if ('HBT' in merged and 'Oxygen Flow' in merged) else 0
merged['coal_intensity'] = (merged['Coal Actual'] / merged['PROD_RATE'].replace(0, np.nan)) if ('Coal Actual' in merged and 'PROD_RATE' in merged) else 0
merged['gas_eff']        = (merged['ETACO'] * merged['Permeabilty']) if ('ETACO' in merged and 'Permeabilty' in merged) else 0
merged['burden_thermal'] = (merged['OreCokeRatio'] * merged['HBT']) if ('OreCokeRatio' in merged and 'HBT' in merged) else 0
merged['flux_thermal']   = (merged['FluxIronRatio'] * merged['HBT']) if ('FluxIronRatio' in merged and 'HBT' in merged) else 0

# Feature lists
BF_PARAMS = [c for c in [
    'HBT', 'Cold Blast Volume', 'Oxygen Flow', 'Coal Actual', 'PROD_RATE', 'ETACO', 'Permeabilty',
    'HBP', 'FTP', 'Steam', 'B MOIST', 'SLAG_RATE', 'Radar Level',
    'Heat_Load Q1', 'Heat_Load Q2', 'Heat_Load Q3', 'Heat_Load Q4',
    'Top DP', 'Middle DP', 'Bottom DP', 'Heat Flow Flux', 'Coal Inj. SP'
] if c in merged.columns]

BURDEN_PARAMS = [c for c in [
    'Coke_kg', 'Sinter_kg', 'Pellet_kg', 'Ore_kg', 'Limestone_kg', 'Dolomite_kg',
    'DRI_kg', 'MixedMaterial_kg', 'TotalIron_kg', 'TotalFlux_kg',
    'OreCokeRatio', 'SinterFrac', 'PelletFrac', 'FluxIronRatio', 'TotalCharge_kg'
] if c in merged.columns]

DERIVED_PARAMS = [c for c in ['thermal_idx', 'coal_intensity', 'gas_eff', 'burden_thermal', 'flux_thermal'] if c in merged.columns]

BASE_FEATURES = list(set(BF_PARAMS + BURDEN_PARAMS + DERIVED_PARAMS))
print(f"  ✓ Base operational features selected: {len(BASE_FEATURES)}")

# Metadata containers
feature_meta = {
    'regression': {},
    'classification': {},
    'timeseries': {},
    'data_summary': {
        'n_records': len(merged),
        'date_start': str(merged['CLOCK'].min().date()),
        'date_end': str(merged['CLOCK'].max().date()),
        'HM_SI_mean': float(merged['HM_Si'].mean()),
        'HM_TEMP_mean': float(merged['HM_Temp'].mean()),
    }
}

# ── 4. PIPELINE 1: REGRESSION ML PIPELINE ────────────────────
print("\n" + "=" * 70)
print("  📈 [PIPELINE 1] REGRESSION ML PIPELINE (PARA + BURDEN -> HM_Si & HM_Temp)")
print("=" * 70)

X_reg_df = merged[BASE_FEATURES].copy()
imputer_reg = SimpleImputer(strategy='median')
X_reg_imp = imputer_reg.fit_transform(X_reg_df)

for tgt in ['HM_Si', 'HM_Temp']:
    y_reg = merged[tgt].values
    
    pre_rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    pre_rf.fit(X_reg_imp, y_reg)
    fi = pd.Series(pre_rf.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)
    top_feats = fi.head(9).index.tolist()
    
    top_idx = [BASE_FEATURES.index(f) for f in top_feats]
    X_top = X_reg_imp[:, top_idx]
    
    scaler_reg = StandardScaler()
    X_sc = scaler_reg.fit_transform(X_top)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_reg, test_size=0.20, random_state=42)
    
    candidate_models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1),
    }
    
    eval_results = {}
    print(f"\n  Target: {tgt} (Regression)")
    print(f"  {'Model':<20} {'R²':<8} {'MAE':<10} {'RMSE':<10}")
    print(f"  {'-'*50}")
    for mname, model in candidate_models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        eval_results[mname] = {'r2': round(r2, 4), 'mae': round(mae, 4), 'rmse': round(rmse, 4), 'model': model}
        print(f"  {mname:<20} {r2:<8.4f} {mae:<10.4f} {rmse:<10.4f}")
        
    best_mname = max(eval_results, key=lambda k: eval_results[k]['r2'])
    best_model = eval_results[best_mname]['model']
    print(f"  🏆 Best Regression Model for {tgt}: {best_mname} (R² = {eval_results[best_mname]['r2']})")
    
    joblib.dump(best_model, f'models/model_regression_{tgt}.pkl')
    joblib.dump(scaler_reg, f'models/scaler_regression_{tgt}.pkl')
    
    feat_ranges = {}
    for f in top_feats:
        col = merged[f].dropna()
        feat_ranges[f] = {
            'min': float(col.quantile(0.01)),
            'max': float(col.quantile(0.99)),
            'mean': float(col.mean()),
            'std': float(col.std()),
        }
        
    feature_meta['regression'][tgt] = {
        'top_features': top_feats,
        'feature_importance': fi.head(15).to_dict(),
        'feature_ranges': feat_ranges,
        'best_model': best_mname,
        'metrics': {k: {m: v for m, v in vv.items() if m != 'model'} for k, vv in eval_results.items()}
    }

joblib.dump(imputer_reg, 'models/imputer_regression.pkl')

# ── 5. PIPELINE 2: CLASSIFICATION PIPELINE ───────────────────
print("\n" + "=" * 70)
print("  🏷️ [PIPELINE 2] CLASSIFICATION PIPELINE (PARA + BURDEN -> HM_Si Class & HM_Temp Class)")
print("=" * 70)

merged['HM_Si_Class'] = pd.cut(merged['HM_Si'], bins=[-np.inf, 0.25, 0.80, np.inf], labels=['Low', 'Normal', 'High']).astype(str)
merged['HM_Temp_Class'] = pd.cut(merged['HM_Temp'], bins=[-np.inf, 1480, 1535, np.inf], labels=['Low', 'Normal', 'High']).astype(str)

X_cls_df = merged[BASE_FEATURES].copy()
imputer_cls = SimpleImputer(strategy='median')
X_cls_imp = imputer_cls.fit_transform(X_cls_df)

for tgt_raw in ['HM_Si', 'HM_Temp']:
    tgt_cls_col = f'{tgt_raw}_Class'
    encoder = LabelEncoder()
    encoder.fit(['Low', 'Normal', 'High'])
    y_cls = encoder.transform(merged[tgt_cls_col])
    
    pre_rf_cls = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    pre_rf_cls.fit(X_cls_imp, y_cls)
    fi = pd.Series(pre_rf_cls.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)
    top_feats = fi.head(9).index.tolist()
    
    top_idx = [BASE_FEATURES.index(f) for f in top_feats]
    X_top = X_cls_imp[:, top_idx]
    
    scaler_cls = StandardScaler()
    X_sc = scaler_cls.fit_transform(X_top)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_cls, test_size=0.20, random_state=42, stratify=y_cls)
    
    candidate_cls = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
    }
    
    eval_results = {}
    print(f"\n  Target: {tgt_raw} Class (Classification)")
    print(f"  {'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
    print(f"  {'-'*45}")
    for mname, model in candidate_cls.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average='macro')
        eval_results[mname] = {'accuracy': round(acc, 4), 'f1_score': round(f1, 4), 'model': model}
        print(f"  {mname:<20} {acc:<10.4f} {f1:<10.4f}")
        
    best_mname = max(eval_results, key=lambda k: eval_results[k]['accuracy'])
    best_model = eval_results[best_mname]['model']
    print(f"  🏆 Best Classification Model for {tgt_raw} Class: {best_mname} (Accuracy = {eval_results[best_mname]['accuracy']})")
    
    joblib.dump(best_model, f'models/model_classification_{tgt_raw}.pkl')
    joblib.dump(scaler_cls, f'models/scaler_classification_{tgt_raw}.pkl')
    joblib.dump(encoder, f'models/encoder_classification_{tgt_raw}.pkl')
    
    feat_ranges = {}
    for f in top_feats:
        col = merged[f].dropna()
        feat_ranges[f] = {
            'min': float(col.quantile(0.01)),
            'max': float(col.quantile(0.99)),
            'mean': float(col.mean()),
            'std': float(col.std()),
        }
        
    feature_meta['classification'][tgt_raw] = {
        'top_features': top_feats,
        'feature_importance': fi.head(15).to_dict(),
        'feature_ranges': feat_ranges,
        'classes': ['Low', 'Normal', 'High'],
        'best_model': best_mname,
        'metrics': {k: {m: v for m, v in vv.items() if m != 'model'} for k, vv in eval_results.items()}
    }

joblib.dump(imputer_cls, 'models/imputer_classification.pkl')

# ── 6. PIPELINE 3: TIME-SERIES PIPELINE ──────────────────────
print("\n" + "=" * 70)
print("  ⏱️ [PIPELINE 3] TIME-SERIES PIPELINE (Previous HM + PARA + BURDEN -> Future HM_Si & Future HM_Temp)")
print("=" * 70)

ts_df = merged.copy().sort_values('CLOCK').reset_index(drop=True)

for lag in [1, 2, 3]:
    ts_df[f'HM_SI_lag{lag}']   = ts_df['HM_Si'].shift(lag)
    ts_df[f'HM_TEMP_lag{lag}'] = ts_df['HM_Temp'].shift(lag)

ts_df['HM_SI_d1']   = ts_df['HM_Si'].diff(1)
ts_df['HM_TEMP_d1'] = ts_df['HM_Temp'].diff(1)

ts_df['Future_HM_Si']   = ts_df['HM_Si'].shift(-1)
ts_df['Future_HM_Temp'] = ts_df['HM_Temp'].shift(-1)

ts_clean = ts_df.dropna(subset=['HM_SI_lag1', 'HM_TEMP_lag1', 'Future_HM_Si', 'Future_HM_Temp']).reset_index(drop=True)

AR_FEATURES = ['HM_SI_lag1', 'HM_TEMP_lag1', 'HM_SI_lag2', 'HM_TEMP_lag2', 'HM_SI_lag3', 'HM_TEMP_lag3', 'HM_SI_d1', 'HM_TEMP_d1']
TS_FEATURE_POOL = list(set(BASE_FEATURES + AR_FEATURES))

X_ts_df = ts_clean[TS_FEATURE_POOL].copy()
imputer_ts = SimpleImputer(strategy='median')
X_ts_imp = imputer_ts.fit_transform(X_ts_df)

for tgt_pair in [('HM_Si', 'Future_HM_Si'), ('HM_Temp', 'Future_HM_Temp')]:
    tgt_raw, future_target = tgt_pair
    y_ts = ts_clean[future_target].values
    
    pre_rf_ts = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    pre_rf_ts.fit(X_ts_imp, y_ts)
    fi = pd.Series(pre_rf_ts.feature_importances_, index=TS_FEATURE_POOL).sort_values(ascending=False)
    top_feats = fi.head(9).index.tolist()
    
    top_idx = [TS_FEATURE_POOL.index(f) for f in top_feats]
    X_top = X_ts_imp[:, top_idx]
    
    scaler_ts = StandardScaler()
    X_sc = scaler_ts.fit_transform(X_top)
    
    split_idx = int(len(X_sc) * 0.80)
    X_tr, X_te = X_sc[:split_idx], X_sc[split_idx:]
    y_tr, y_te = y_ts[:split_idx], y_ts[split_idx:]
    
    candidate_ts = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1),
    }
    
    eval_results = {}
    print(f"\n  Target: Future {tgt_raw} (Time-Series)")
    print(f"  {'Model':<20} {'R²':<8} {'MAE':<10} {'RMSE':<10}")
    print(f"  {'-'*50}")
    for mname, model in candidate_ts.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        eval_results[mname] = {'r2': round(r2, 4), 'mae': round(mae, 4), 'rmse': round(rmse, 4), 'model': model}
        print(f"  {mname:<20} {r2:<8.4f} {mae:<10.4f} {rmse:<10.4f}")
        
    best_mname = max(eval_results, key=lambda k: eval_results[k]['r2'])
    best_model = eval_results[best_mname]['model']
    print(f"  🏆 Best Time-Series Model for Future {tgt_raw}: {best_mname} (R² = {eval_results[best_mname]['r2']})")
    
    joblib.dump(best_model, f'models/model_timeseries_{tgt_raw}.pkl')
    joblib.dump(scaler_ts, f'models/scaler_timeseries_{tgt_raw}.pkl')
    
    feat_ranges = {}
    for f in top_feats:
        col = ts_clean[f].dropna()
        feat_ranges[f] = {
            'min': float(col.quantile(0.01)),
            'max': float(col.quantile(0.99)),
            'mean': float(col.mean()),
            'std': float(col.std()),
        }
        
    feature_meta['timeseries'][tgt_raw] = {
        'top_features': top_feats,
        'feature_importance': fi.head(15).to_dict(),
        'feature_ranges': feat_ranges,
        'best_model': best_mname,
        'metrics': {k: {m: v for m, v in vv.items() if m != 'model'} for k, vv in eval_results.items()}
    }

joblib.dump(imputer_ts, 'models/imputer_timeseries.pkl')

# Save metadata
joblib.dump(feature_meta, 'models/feature_meta.pkl')

# ── PRE-COMPUTE SENSITIVITY CACHE ────────────────────────────
print("\n[6/6] Pre-computing sensitivity curves for all top regression features...")

# Gather all unique features from regression pipeline
all_reg_feats_si   = feature_meta['regression']['HM_Si']['top_features']
all_reg_feats_temp = feature_meta['regression']['HM_Temp']['top_features']
all_sens_feats = list(dict.fromkeys(all_reg_feats_si + all_reg_feats_temp))

# We need the loaded best models to compute predictions
reg_model_si   = joblib.load('models/model_regression_HM_Si.pkl')
reg_model_temp = joblib.load('models/model_regression_HM_Temp.pkl')
reg_scaler_si  = joblib.load('models/scaler_regression_HM_Si.pkl')
reg_scaler_temp= joblib.load('models/scaler_regression_HM_Temp.pkl')

def _predict_reg_fast(feat_list_si, feat_list_temp, input_dict):
    """Fast dual-target regression prediction using preloaded objects."""
    row_si   = np.array([[input_dict.get(f, feature_meta['regression']['HM_Si']['feature_ranges'].get(f, {}).get('mean', 0)) for f in feat_list_si]], dtype=float)
    row_temp = np.array([[input_dict.get(f, feature_meta['regression']['HM_Temp']['feature_ranges'].get(f, {}).get('mean', 0)) for f in feat_list_temp]], dtype=float)
    try:
        si_pred   = float(reg_model_si.predict(reg_scaler_si.transform(row_si))[0])
    except Exception:
        si_pred = np.nan
    try:
        temp_pred = float(reg_model_temp.predict(reg_scaler_temp.transform(row_temp))[0])
    except Exception:
        temp_pred = np.nan
    return si_pred, temp_pred

NUM_POINTS = 25
sensitivity_cache = {}

ranges_si   = feature_meta['regression']['HM_Si']['feature_ranges']
ranges_temp = feature_meta['regression']['HM_Temp']['feature_ranges']

# Build base inputs (means)
base_inputs = {}
for f in all_sens_feats:
    info = ranges_si.get(f, ranges_temp.get(f, {'mean': 0.0}))
    base_inputs[f] = float(info.get('mean', 0.0))

for feat in all_sens_feats:
    f_info = ranges_si.get(feat, ranges_temp.get(feat, {'min': 0.0, 'max': 1.0}))
    min_v, max_v = float(f_info.get('min', 0)), float(f_info.get('max', 1))
    grid = np.linspace(min_v, max_v, NUM_POINTS)
    records = []
    for val in grid:
        inp = {**base_inputs, feat: val}
        si_p, temp_p = _predict_reg_fast(all_reg_feats_si, all_reg_feats_temp, inp)
        records.append({'FeatureValue': round(float(val), 4),
                        'Predicted_HM_SI': round(si_p, 5) if not np.isnan(si_p) else None,
                        'Predicted_HM_TEMP': round(temp_p, 3) if not np.isnan(temp_p) else None})
    sensitivity_cache[feat] = records
    print(f"  ✓ Cached sensitivity curve for: {feat}")

joblib.dump(sensitivity_cache, 'models/sensitivity_cache.pkl')
print("  ✅ Sensitivity cache saved to models/sensitivity_cache.pkl")

print("\n" + "=" * 70)
print("  ✅ SUCCESS: ALL 3 PIPELINES TRAINED & ARTIFACTS SAVED TO models/")
print("=" * 70)

