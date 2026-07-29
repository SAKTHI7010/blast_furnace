#!/usr/bin/env python3
"""
=============================================================================
BF-4 BLAST FURNACE HOT METAL QUALITY PREDICTION
Complete ML Pipeline — 3 Dataset Integration
(PARA.xlsx + HM_ANALYSIS.xls + BURDEN.csv)
=============================================================================
Author: Generated for IIT Madras / Extractmet Pvt. Ltd.
Datasets: BF-4 Process Parameters | Hot Metal Quality | Burden Charging
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, copy, os, json, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ═══════════════════════════════════════════════════════════
# SECTION 1: LOAD & PREPROCESS BURDEN.csv
# ═══════════════════════════════════════════════════════════
print("="*60)
print("STEP 1: Loading and processing BURDEN.csv")
print("="*60)

burden = pd.read_csv('BURDEN.csv', encoding='utf-8-sig')
burden.columns = burden.columns.str.strip()
burden['BRANDCODE'] = burden['BRANDCODE'].str.strip()
burden['CHARGETIME'] = pd.to_datetime(burden['CHARGETIME'], dayfirst=True, errors='coerce')
burden['CLOCK'] = burden['CHARGETIME'].dt.floor('h')

print(f"  Raw burden records: {len(burden):,}")
print(f"  Unique materials:   {burden['BRANDCODE'].nunique()}")
print(f"  Date range: {burden['CHARGETIME'].min()} → {burden['CHARGETIME'].max()}")

def classify_material(brand):
    """Classify BF burden material into standard groups."""
    b = brand.upper()
    if 'COKE' in b or 'NUTCOKE' in b:  return 'Coke_kg'
    if 'SINTER' in b:                   return 'Sinter_kg'
    if 'PELLET' in b:                   return 'Pellet_kg'
    if 'ORE' in b or 'BHQ' in b:       return 'Ore_kg'
    if 'LIMESTONE' in b:                return 'Limestone_kg'
    if 'DOLOMITE' in b:                 return 'Dolomite_kg'
    if 'DRI' in b:                      return 'DRI_kg'
    if 'MIXED' in b:                    return 'MixedMaterial_kg'
    if 'SCRAP' in b:                    return 'Scrap_kg'
    return 'Other_kg'

burden['MaterialGroup'] = burden['BRANDCODE'].apply(classify_material)

# Pivot to wide hourly format
burden_hourly = (burden.groupby(['CLOCK', 'MaterialGroup'])['ACTWT']
                       .sum()
                       .unstack(fill_value=0)
                       .reset_index())

# Ensure all expected columns exist
for col in ['Coke_kg','Sinter_kg','Pellet_kg','Ore_kg','Limestone_kg',
            'Dolomite_kg','DRI_kg','MixedMaterial_kg','Scrap_kg','Other_kg']:
    if col not in burden_hourly.columns:
        burden_hourly[col] = 0.0

# Derived burden features
bh = burden_hourly.copy()
bh['TotalIron_kg']    = (bh['Ore_kg'] + bh['Pellet_kg'] + bh['Sinter_kg']
                         + bh['DRI_kg'] + bh['MixedMaterial_kg'])
bh['TotalFlux_kg']    = bh['Limestone_kg'] + bh['Dolomite_kg']
bh['OreCokeRatio']    = bh['TotalIron_kg'] / bh['Coke_kg'].replace(0, np.nan)
bh['SinterFrac']      = bh['Sinter_kg'] / bh['TotalIron_kg'].replace(0, np.nan)
bh['PelletFrac']      = bh['Pellet_kg'] / bh['TotalIron_kg'].replace(0, np.nan)
bh['OreFrac']         = (bh['Ore_kg'] + bh['MixedMaterial_kg']) / bh['TotalIron_kg'].replace(0, np.nan)
bh['FluxIronRatio']   = bh['TotalFlux_kg'] / bh['TotalIron_kg'].replace(0, np.nan)
bh['TotalCharge_kg']  = bh['Coke_kg'] + bh['TotalIron_kg'] + bh['TotalFlux_kg']

# Apply 8-hour lag (BF descent time) before merging with HM quality
bh_lag = bh.copy()
bh_lag['CLOCK'] = bh_lag['CLOCK'] + pd.Timedelta(hours=8)
print(f"  Hourly burden records: {len(bh):,}")

# ═══════════════════════════════════════════════════════════
# SECTION 2: LOAD HM_ANALYSIS.xls (Target variables)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Loading HM_ANALYSIS.xls")
print("="*60)

hm_raw = pd.read_excel('HM_ANALYSIS.xls', sheet_name='Sheet 1')
hm_raw['SAMPLETAKEN'] = pd.to_datetime(hm_raw['SAMPLETAKEN'])
hm_raw['CLOCK'] = hm_raw['SAMPLETAKEN'].dt.floor('h')
HM_TARGETS = ['HM_C', 'HM_MN', 'HM_P', 'HM_S', 'HM_SI', 'HM_TEMP']
hm_h = hm_raw.groupby('CLOCK')[HM_TARGETS].mean().reset_index()
print(f"  Raw HM samples: {len(hm_raw):,}")
print(f"  Hourly averages: {len(hm_h):,}")
print(f"  Date range: {hm_h['CLOCK'].min()} → {hm_h['CLOCK'].max()}")

# ═══════════════════════════════════════════════════════════
# SECTION 3: LOAD PARA.xlsx (BF-4 Process Parameters)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: Loading PARA.xlsx (BF-4 Data)")
print("="*60)

raw = pd.read_excel('PARA.xlsx', sheet_name='BF-4 Data', header=None)
var_names = raw.iloc[1].tolist()
bf = raw.iloc[3:].copy()
bf.columns = var_names
bf = bf.rename(columns={var_names[0]: 'CLOCK'}).reset_index(drop=True)
for col in bf.columns:
    if col != 'CLOCK':
        bf[col] = pd.to_numeric(bf[col], errors='coerce')
bf['CLOCK'] = pd.to_datetime(bf['CLOCK'], errors='coerce')
# Drop columns with >99% nulls (e.g., CS Moisture columns)
bf = bf[[c for c in bf.columns if c == 'CLOCK' or bf[c].isnull().mean() < 0.99]]
# Remove sensor fault spikes
bf = bf[bf['Heat Flow Flux'] < 100]
bf['CLOCK'] = bf['CLOCK'].dt.floor('h')
print(f"  BF-4 records: {len(bf):,}")
print(f"  Usable parameters: {len(bf.columns)-1}")

# ═══════════════════════════════════════════════════════════
# SECTION 4: THREE-WAY MERGE
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: Three-way merge (BF + Burden_lag8h + HM)")
print("="*60)

merged = pd.merge(bf, bh_lag, on='CLOCK', how='inner')
merged = pd.merge(merged, hm_h, on='CLOCK', how='inner')
print(f"  Merged shape: {merged.shape}")

# ═══════════════════════════════════════════════════════════
# SECTION 5: DATA CLEANING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: IQR Outlier Cleaning")
print("="*60)

BF_FEAT = [c for c in [
    'HBT', 'Oxygen Flow', 'Coal Actual', 'PROD_RATE', 'ETACO', 'Permeabilty',
    'Cold Blast Volume', 'HBP', 'FTP', 'Steam', 'B MOIST', 'SLAG_RATE',
    'Radar Level', 'Heat_Load Q1', 'Heat_Load Q2', 'Heat_Load Q3', 'Heat_Load Q4',
    'Top DP', 'Middle DP', 'Bottom DP', 'Heat Flow Flux', 'Coal Inj. SP'
] if c in merged.columns]

BRD_FEAT = [c for c in [
    'Coke_kg', 'Sinter_kg', 'Pellet_kg', 'Ore_kg', 'Limestone_kg', 'Dolomite_kg',
    'DRI_kg', 'MixedMaterial_kg', 'TotalIron_kg', 'TotalFlux_kg',
    'OreCokeRatio', 'SinterFrac', 'PelletFrac', 'OreFrac', 'FluxIronRatio', 'TotalCharge_kg'
] if c in merged.columns]

BASE_FEAT = BF_FEAT + BRD_FEAT
TARGET_COLS = ['HM_C', 'HM_SI', 'HM_S', 'HM_MN', 'HM_P', 'HM_TEMP']
TARGET_UNITS = {
    'HM_C': '%C', 'HM_SI': '%Si', 'HM_S': '%S',
    'HM_MN': '%Mn', 'HM_P': '%P', 'HM_TEMP': '°C'
}

def iqr_clean(df, cols, factor=3.0):
    mask = pd.Series([True]*len(df), index=df.index)
    for c in cols:
        if c not in df.columns: continue
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        mask &= (df[c] >= Q1 - factor*IQR) & (df[c] <= Q3 + factor*IQR)
    return df[mask]

df = merged[['CLOCK'] + BASE_FEAT + TARGET_COLS].dropna().reset_index(drop=True)
n_before = len(df)
df = iqr_clean(df, TARGET_COLS)
df = iqr_clean(df, BASE_FEAT)
df = df.dropna().sort_values('CLOCK').reset_index(drop=True)
print(f"  Records before cleaning: {n_before:,}")
print(f"  Records after cleaning:  {len(df):,}")

# ═══════════════════════════════════════════════════════════
# SECTION 6: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6: Feature Engineering (Lag + Rolling + Metallurgical)")
print("="*60)

eng_feats = list(BASE_FEAT)

# Lag features (BF thermal state delay: 2h, 4h, 8h)
lag_vars = ['HBT', 'Oxygen Flow', 'Coal Actual', 'ETACO', 'Permeabilty',
            'PROD_RATE', 'OreCokeRatio', 'SinterFrac', 'FluxIronRatio']
for var in lag_vars:
    if var in df.columns:
        for lag in [2, 4, 8]:
            cn = f'{var}_lag{lag}h'
            df[cn] = df[var].shift(lag)
            eng_feats.append(cn)

# Rolling mean features (process stability window)
roll_vars = ['HBT', 'ETACO', 'Coal Actual', 'Oxygen Flow', 'OreCokeRatio', 'SinterFrac']
for var in roll_vars:
    if var in df.columns:
        for win in [4, 8]:
            cn = f'{var}_roll{win}h'
            df[cn] = df[var].rolling(win, min_periods=2).mean()
            eng_feats.append(cn)

# Derived metallurgical interaction features
df['thermal_idx']    = df['HBT'] * df['Oxygen Flow'] / 1e6  # Thermal energy index
df['coal_intensity'] = df['Coal Actual'] / df['PROD_RATE'].replace(0, np.nan)  # PCI rate
df['gas_eff']        = df['ETACO'] * df['Permeabilty']  # Gas utilization efficiency
df['DP_ratio']       = df['Top DP'] / df['Bottom DP'].replace(0, np.nan)  # Pressure balance
df['burden_thermal'] = df['OreCokeRatio'] * df['HBT']  # Burden × thermal interaction
df['flux_thermal']   = df['FluxIronRatio'] * df['HBT']  # Desulphurisation efficiency proxy
df['pellet_thermal'] = df['PelletFrac'] * df['HBT']  # Pellet reducibility index

derived = ['thermal_idx', 'coal_intensity', 'gas_eff', 'DP_ratio',
           'burden_thermal', 'flux_thermal', 'pellet_thermal']
eng_feats.extend(derived)

df = df.dropna().reset_index(drop=True)
print(f"  Base features:         {len(BASE_FEAT)}")
print(f"  Engineered features:   {len(eng_feats) - len(BASE_FEAT)}")
print(f"  Total features:        {len(eng_feats)}")
print(f"  Final dataset shape:   {df.shape}")

# ═══════════════════════════════════════════════════════════
# SECTION 7: EDA — Correlation Analysis
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7: EDA & Correlation Analysis")
print("="*60)

key_features = ['HBT', 'OreCokeRatio', 'SinterFrac', 'FluxIronRatio',
                'Coal Actual', 'ETACO', 'Oxygen Flow', 'Permeabilty',
                'HM_C', 'HM_SI', 'HM_S', 'HM_MN', 'HM_P', 'HM_TEMP']
corr_df = df[[c for c in key_features if c in df.columns]].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, annot_kws={'size': 8})
plt.title('Correlation Heatmap: BF Process + Burden Features vs HM Quality', pad=12)
plt.tight_layout()
plt.savefig('output/01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/01_correlation_heatmap.png")

# Target distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, (tgt, unit) in enumerate(TARGET_UNITS.items()):
    if tgt in df.columns:
        axes[i].hist(df[tgt].dropna(), bins=40, edgecolor='white', alpha=0.8)
        axes[i].axvline(df[tgt].mean(), color='red', linestyle='--', label=f'Mean={df[tgt].mean():.3f}')
        axes[i].set_title(f'{tgt} ({unit})', fontweight='bold')
        axes[i].set_xlabel(unit)
        axes[i].legend(fontsize=8)
plt.suptitle('Hot Metal Quality Target Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/02_target_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/02_target_distributions.png")

# ═══════════════════════════════════════════════════════════
# SECTION 8: MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8: Model Training (5 algorithms × 6 targets)")
print("="*60)

X_raw = df[eng_feats].values
scaler = RobustScaler()
X = scaler.fit_transform(X_raw)

MODELS = {
    'Ridge':        Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(
                        n_estimators=100, max_depth=10,
                        min_samples_leaf=5, n_jobs=-1, random_state=42),
    'ExtraTrees':   ExtraTreesRegressor(
                        n_estimators=100, max_depth=10,
                        min_samples_leaf=5, n_jobs=-1, random_state=42),
    'GradBoost':    GradientBoostingRegressor(
                        n_estimators=100, max_depth=4,
                        learning_rate=0.08, subsample=0.8, random_state=42),
    'ANN':          MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32), activation='relu',
                        max_iter=300, early_stopping=True, random_state=42),
}

all_results = {}
best_models = {}
feature_importances = {}

for target in TARGET_COLS:
    y = df[target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    all_results[target] = {}
    best_r2, best_name, best_yp = -999, None, None

    print(f"\n  Target: {target} ({TARGET_UNITS[target]})")
    print(f"  {'Model':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*45}")

    for mname, model in MODELS.items():
        m = copy.deepcopy(model)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        r2   = r2_score(yte, yp)
        rmse = np.sqrt(mean_squared_error(yte, yp))
        mae  = mean_absolute_error(yte, yp)
        all_results[target][mname] = {
            'R2': round(r2, 4), 'RMSE': round(rmse, 5), 'MAE': round(mae, 5)
        }
        print(f"  {mname:<15} {r2:>8.4f} {rmse:>10.5f} {mae:>10.5f}")

        if r2 > best_r2:
            best_r2, best_name, best_yp = r2, mname, (yte, yp)
            if hasattr(m, 'feature_importances_'):
                imp = dict(zip(eng_feats, m.feature_importances_))
                feature_importances[target] = sorted(imp.items(), key=lambda x: -x[1])[:15]
            joblib.dump(m, f'models/best_{target}.pkl')

    best_models[target] = best_name
    all_results[target]['best'] = best_name
    all_results[target]['y_test'] = best_yp[0].tolist()
    all_results[target]['y_pred'] = best_yp[1].tolist()
    print(f"  >>> BEST: {best_name} | R²={best_r2:.4f}")

# ═══════════════════════════════════════════════════════════
# SECTION 9: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9: Generating Result Visualizations")
print("="*60)

# 9a: Actual vs Predicted (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, (tgt, unit) in enumerate(TARGET_UNITS.items()):
    yt = np.array(all_results[tgt]['y_test'])
    yp = np.array(all_results[tgt]['y_pred'])
    r2 = all_results[tgt][best_models[tgt]]['R2']
    ax = axes[i]
    ax.scatter(yt, yp, alpha=0.4, s=15, c='steelblue')
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Ideal')
    ax.set_xlabel(f'Actual ({unit})')
    ax.set_ylabel(f'Predicted ({unit})')
    ax.set_title(f'{tgt} — {best_models[tgt]} | R²={r2:.3f}', fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle('Actual vs Predicted: Best Model per Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/03_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/03_actual_vs_predicted.png")

# 9b: Feature Importances (HM_SI and HM_MN)
for tgt in ['HM_SI', 'HM_MN', 'HM_S']:
    if tgt in feature_importances:
        feats = [f[0] for f in feature_importances[tgt]]
        imps  = [f[1] for f in feature_importances[tgt]]
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(range(len(feats)), imps, color='steelblue', edgecolor='white')
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([f.replace('_',' ') for f in feats], fontsize=10)
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top 15 Feature Importances — {tgt} ({TARGET_UNITS[tgt]})', fontweight='bold')
        for bar, imp in zip(bars, imps):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'output/04_feat_importance_{tgt}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: output/04_feat_importance_{tgt}.png")

# 9c: R² Summary Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
model_names = ['Ridge', 'RandomForest', 'ExtraTrees', 'GradBoost', 'ANN']
colors_map = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6']
x = np.arange(len(TARGET_COLS))
w = 0.16
for k, (mname, col) in enumerate(zip(model_names, colors_map)):
    vals = [all_results[tgt].get(mname, {}).get('R2', 0) for tgt in TARGET_COLS]
    ax.bar(x + k*w, vals, w, label=mname, color=col, edgecolor='white')
ax.set_xticks(x + 2*w)
ax.set_xticklabels([f"{t}\n({TARGET_UNITS[t]})" for t in TARGET_COLS], fontsize=10)
ax.set_ylabel('R² Score')
ax.set_title('R² Score Comparison: All Models × All Targets', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 0.8)
plt.tight_layout()
plt.savefig('output/05_r2_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/05_r2_model_comparison.png")

# ═══════════════════════════════════════════════════════════
# SECTION 10: SAVE RESULTS & SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10: Saving Results")
print("="*60)

summary_rows = []
for tgt in TARGET_COLS:
    best = best_models[tgt]
    row = {'Target': tgt, 'Unit': TARGET_UNITS[tgt], 'Best_Model': best}
    row.update(all_results[tgt][best])
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('output/model_performance_summary.csv', index=False)
print("\n=== FINAL MODEL PERFORMANCE SUMMARY ===")
print(summary_df[['Target','Unit','Best_Model','R2','RMSE','MAE']].to_string(index=False))
print("\nSaved: output/model_performance_summary.csv")
print("Saved: models/best_<target>.pkl (joblib pickled models)")

# ═══════════════════════════════════════════════════════════
# SECTION 11: REAL-TIME INFERENCE FUNCTION
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 11: Real-time Inference API")
print("="*60)

def predict_hm_quality(process_params: dict, burden_params: dict) -> dict:
    """
    Predict hot metal quality from current BF process + burden conditions.

    Parameters
    ----------
    process_params : dict — BF process variables (HBT, Oxygen Flow, etc.)
    burden_params  : dict — Burden composition at charging time (8h earlier)

    Returns
    -------
    dict with predicted quality values + alarm flags
    """
    # Merge inputs
    combined = {**process_params, **burden_params}
    row = pd.DataFrame([combined])

    # Compute derived features (same as training)
    row['TotalIron_kg']   = sum(row.get(c, 0)[0] for c in ['Ore_kg','Pellet_kg','Sinter_kg','DRI_kg','MixedMaterial_kg'])
    row['TotalFlux_kg']   = (row.get('Limestone_kg', pd.Series([0]))[0] +
                              row.get('Dolomite_kg', pd.Series([0]))[0])
    row['OreCokeRatio']   = row['TotalIron_kg'] / row.get('Coke_kg', pd.Series([1]))[0]
    row['SinterFrac']     = row.get('Sinter_kg', pd.Series([0]))[0] / row['TotalIron_kg'][0]
    row['PelletFrac']     = row.get('Pellet_kg', pd.Series([0]))[0] / row['TotalIron_kg'][0]
    row['OreFrac']        = (row.get('Ore_kg', pd.Series([0]))[0] +
                              row.get('MixedMaterial_kg', pd.Series([0]))[0]) / row['TotalIron_kg'][0]
    row['FluxIronRatio']  = row['TotalFlux_kg'][0] / row['TotalIron_kg'][0]
    row['TotalCharge_kg'] = row.get('Coke_kg', pd.Series([0]))[0] + row['TotalIron_kg'][0] + row['TotalFlux_kg'][0]
    row['thermal_idx']    = row.get('HBT', pd.Series([1150]))[0] * row.get('Oxygen Flow', pd.Series([8000]))[0] / 1e6
    row['coal_intensity'] = row.get('Coal Actual', pd.Series([150]))[0] / max(row.get('PROD_RATE', pd.Series([200]))[0], 1)
    row['gas_eff']        = row.get('ETACO', pd.Series([0.47]))[0] * row.get('Permeabilty', pd.Series([1.5]))[0]
    row['DP_ratio']       = row.get('Top DP', pd.Series([1.2]))[0] / max(row.get('Bottom DP', pd.Series([0.5]))[0], 0.01)
    row['burden_thermal'] = row['OreCokeRatio'][0] * row.get('HBT', pd.Series([1150]))[0]
    row['flux_thermal']   = row['FluxIronRatio'][0] * row.get('HBT', pd.Series([1150]))[0]
    row['pellet_thermal'] = row['PelletFrac'][0] * row.get('HBT', pd.Series([1150]))[0]

    # Load models and predict
    results = {}
    spec_limits = {
        'HM_C': (4.3, 4.9), 'HM_SI': (0.25, 0.80), 'HM_S': (0.0, 0.055),
        'HM_MN': (0.8, 1.8), 'HM_P': (0.0, 0.18), 'HM_TEMP': (1470, 1540)
    }
    for tgt in TARGET_COLS:
        try:
            model = joblib.load(f'models/best_{tgt}.pkl')
            # Build feature vector (fill missing with mean)
            feat_row = np.array([[row.get(f, pd.Series([0]))[0] for f in eng_feats]])
            feat_scaled = scaler.transform(feat_row)
            pred = float(model.predict(feat_scaled)[0])
            lo, hi = spec_limits.get(tgt, (-np.inf, np.inf))
            alarm = pred < lo or pred > hi
            results[tgt] = {'predicted': round(pred, 4),
                             'unit': TARGET_UNITS[tgt],
                             'in_spec': not alarm,
                             'spec': f"{lo}–{hi}"}
        except Exception as e:
            results[tgt] = {'error': str(e)}
    return results

print("  predict_hm_quality() function ready.")
print("  Usage example:")
print("    result = predict_hm_quality(")
print("        process_params={'HBT': 1165, 'Oxygen Flow': 9200, ...},")
print("        burden_params={'Coke_kg': 155000, 'Sinter_kg': 390000, ...}")
print("    )")

print("\n" + "="*60)
print("PIPELINE COMPLETE!")
print("="*60)
print("Outputs saved in: ./output/")
print("Models saved in:  ./models/")
