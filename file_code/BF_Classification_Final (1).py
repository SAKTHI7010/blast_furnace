# =============================================================
#  BLAST FURNACE HOT METAL CLASSIFICATION PIPELINE
#  Targets : HM_Si & HM_Temp  |  Sources: PARA.xlsx, HM_ANALYSIS.xls, BURDEN.csv
# =============================================================

# ── 0. INSTALL ───────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imbalanced-learn", "xgboost"], check=True)

# ── 1. IMPORTS ───────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score,
                                      confusion_matrix, classification_report)
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)
print("Libraries loaded ✓\n")

# ── 2. LOAD DATA ─────────────────────────────────────────────
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

hm = pd.read_excel('HM_ANALYSIS.xls')
hm['SAMPLETAKEN'] = pd.to_datetime(hm['SAMPLETAKEN'], errors='coerce')
hm = (hm.dropna(subset=['SAMPLETAKEN', 'HM_SI', 'HM_TEMP'])
        .sort_values('SAMPLETAKEN').reset_index(drop=True))

burden = pd.read_csv('BURDEN.csv')
burden['CHARGETIME'] = pd.to_datetime(burden['CHARGETIME'], errors='coerce')
burden['BRANDCODE']  = burden['BRANDCODE'].str.strip()
burden['hour']       = burden['CHARGETIME'].dt.floor('h')
bp = (burden.pivot_table(index='hour', columns='BRANDCODE',
                          values='ACTWT', aggfunc='sum')
              .reset_index().sort_values('hour').reset_index(drop=True))
bp.columns.name = None
bp.columns = ['hour'] + [f'BRD_{c}' for c in bp.columns[1:]]

print(f"PARA   : {para.shape[0]:,} rows  ({para['CLOCK'].min().date()} -> {para['CLOCK'].max().date()})")
print(f"HM     : {hm.shape[0]:,} taps  ({hm['SAMPLETAKEN'].min().date()} -> {hm['SAMPLETAKEN'].max().date()})")
print(f"Burden : {bp.shape[0]:,} hourly bundles | {len(bp.columns)-1} brand codes\n")

# ── 3. MERGE ─────────────────────────────────────────────────
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
print(f"Merged (<=4 h lag): {len(full):,} records")

# ── 4. OUTLIER REMOVAL ───────────────────────────────────────
def iqr_filter(df, col, k=3):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    return df[(df[col] >= q1 - k*(q3-q1)) & (df[col] <= q3 + k*(q3-q1))]

n_raw = len(full)
full  = iqr_filter(full, 'HM_SI').reset_index(drop=True)
full  = iqr_filter(full, 'HM_TEMP').reset_index(drop=True)
print(f"After 3xIQR cleanup: {len(full):,} records (removed {n_raw - len(full)})\n")

# ── 5. LABELS ────────────────────────────────────────────────
si_q33, si_q66 = full['HM_SI'].quantile(0.33), full['HM_SI'].quantile(0.66)
t_q33,  t_q66  = full['HM_TEMP'].quantile(0.33), full['HM_TEMP'].quantile(0.66)

def tertile(x, lo, hi):
    return 0 if x < lo else (1 if x <= hi else 2)

full['Si_class']   = full['HM_SI'].apply(lambda x: tertile(x, si_q33, si_q66))
full['Temp_class'] = full['HM_TEMP'].apply(lambda x: tertile(x, t_q33, t_q66))
CLASS_NAMES = ['Low', 'Normal', 'High']

print("── Label Thresholds ────────────────────────────────────")
print(f"  HM_Si  : Low<{si_q33:.3f} | Normal {si_q33:.3f}-{si_q66:.3f} | High>{si_q66:.3f}")
print(f"  HM_Temp: Low<{t_q33:.1f}C | Normal {t_q33:.1f}-{t_q66:.1f}C | High>{t_q66:.1f}C")
print("── Class Distribution ──────────────────────────────────")
for col, name in [('Si_class', 'HM_Si'), ('Temp_class', 'HM_Temp')]:
    dist = full[col].value_counts().sort_index()
    print(f"  {name}: " + "  ".join(f"{CLASS_NAMES[k]}={v:,}" for k, v in dist.items()))
print()

# ── 6. FEATURE MATRIX ────────────────────────────────────────
# Always exclude these columns
EXCLUDE = {
    'SAMPLETAKEN', 'TAPNO', 'TAPHOLE', 'SAMPLENO',
    'HM_C', 'HM_MN', 'HM_P', 'HM_S', 'HM_SI', 'HM_TI',
    'HM_TEMP', 'Si_class', 'Temp_class', 'CLOCK', 'lag_hours'
}

# Drop all excluded columns first, then convert everything possible to numeric,
# then drop any remaining non-numeric (object/datetime) columns.
# This approach is immune to any column name surprises in the merged dataframe.

df_feat = full.drop(columns=[c for c in full.columns if c in EXCLUDE], errors='ignore')

# Force-convert to numeric where possible (object columns with numeric strings become float)
df_feat = df_feat.apply(pd.to_numeric, errors='coerce')

# Now ALL columns are numeric (or NaN). Drop columns that are entirely NaN.
df_feat = df_feat.dropna(axis=1, how='all')

# *** feature_names is set from df_feat AFTER all cleaning — guaranteed match ***
feature_names = df_feat.columns.tolist()

imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(df_feat)

# This must always be True now
print(f"Feature matrix : {X_imp.shape[0]:,} x {X_imp.shape[1]}  "
      f"(feature_names={len(feature_names)})  ✓ lengths match\n")

# ── 7. TRAIN / EVALUATE ───────────────────────────────────────
TARGETS = [('HM_Si', 'Si_class'), ('HM_Temp', 'Temp_class')]
results = {}; best_models = {}; y_tests_all = {}
y_preds_all = {}; scalers_dict = {}; feat_imps = {}

print("=" * 65)
print("  MODEL TRAINING & EVALUATION")
print("=" * 65)

for tgt, ycol in TARGETS:
    y_all = full[ycol].values
    print(f"\n  TARGET: {tgt}  ({len(y_all):,} samples)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y_all, test_size=0.2, random_state=42, stratify=y_all)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    scalers_dict[tgt] = sc

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_sc, y_tr)
    print(f"  Train {X_tr_sc.shape} -> SMOTE -> {X_tr_sm.shape} | Test {X_te_sc.shape}")

    mods = {
        'Logistic Regression':
            LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        'Random Forest':
            RandomForestClassifier(n_estimators=200, max_depth=12,
                                   class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting':
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                       max_depth=5, random_state=42),
        'XGBoost':
            XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                          eval_metric='mlogloss', random_state=42, n_jobs=-1)
    }

    results[tgt] = {}; y_tests_all[tgt] = y_te; y_preds_all[tgt] = {}

    print(f"  {'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─'*50}")

    for name, clf in mods.items():
        clf.fit(X_tr_sm, y_tr_sm)
        y_pred = clf.predict(X_te_sc)
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_te,  y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_te,    y_pred, average='macro', zero_division=0)
        results[tgt][name] = {
            'accuracy':  round(acc,  4), 'precision': round(prec, 4),
            'recall':    round(rec,  4), 'f1':        round(f1,   4),
            'cm':        confusion_matrix(y_te, y_pred).tolist(),
            'report':    classification_report(y_te, y_pred,
                             target_names=CLASS_NAMES, zero_division=0),
            'clf':       clf
        }
        y_preds_all[tgt][name] = y_pred
        print(f"  {name:<25} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f}")

    best = max(results[tgt], key=lambda k: results[tgt][k]['f1'])
    best_models[tgt] = best
    print(f"  BEST: {best}  (Macro F1={results[tgt][best]['f1']:.4f})")

    clf_b = results[tgt][best]['clf']
    if hasattr(clf_b, 'feature_importances_'):
        fi  = clf_b.feature_importances_
        # feature_names length == fi length guaranteed by construction above
        imp = pd.Series(fi, index=feature_names).sort_values(ascending=False)
        feat_imps[tgt] = imp
        imp.to_csv(f'output/feature_importance_{tgt}.csv')

# ── 8. CLASSIFICATION REPORTS ─────────────────────────────────
print("\n" + "=" * 65)
print("  CLASSIFICATION REPORTS  (best models)")
print("=" * 65)
for tgt, _ in TARGETS:
    b = best_models[tgt]
    print(f"\n  [{tgt}]  Model: {b}  | Accuracy: {results[tgt][b]['accuracy']:.4f}")
    print(results[tgt][b]['report'])

# ── 9. TOP-15 FEATURE IMPORTANCES ─────────────────────────────
print("=" * 65)
print("  TOP 15 FEATURE IMPORTANCES")
print("=" * 65)
for tgt, _ in TARGETS:
    if tgt not in feat_imps:
        continue
    print(f"\n  [{tgt}]  Model: {best_models[tgt]}")
    print(f"  {'#':<4} {'Feature':<36} {'Importance':>10}  Type")
    print(f"  {'─'*58}")
    for rank, (feat, val) in enumerate(feat_imps[tgt].head(15).items(), 1):
        ftype = 'Burden' if feat.startswith('BRD_') else 'Process'
        print(f"  {rank:<4} {feat:<36} {val:>10.5f}  {ftype}")

# ── 10. SAVE CSV OUTPUTS ───────────────────────────────────────
rows = []
for tgt, _ in TARGETS:
    for name in results[tgt]:
        r = results[tgt][name]
        rows.append({'Target': tgt, 'Model': name,
                     'Accuracy': r['accuracy'], 'Precision_Macro': r['precision'],
                     'Recall_Macro': r['recall'], 'F1_Macro': r['f1'],
                     'Best': (name == best_models[tgt])})
pd.DataFrame(rows).to_csv('output/model_summary.csv', index=False)
full[['SAMPLETAKEN', 'HM_SI', 'Si_class',
      'HM_TEMP', 'Temp_class']].to_csv('output/labels.csv', index=False)

# ── 11. CONFUSION MATRICES PLOT ───────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle('Blast Furnace Classification — Normalised Confusion Matrices\n'
             'HM_Si & HM_Temp  |  4 Algorithms',
             fontsize=14, fontweight='bold', y=1.01)
for ri, (tgt, _) in enumerate(TARGETS):
    for ci, name in enumerate(['Logistic Regression', 'Random Forest',
                                'Gradient Boosting', 'XGBoost']):
        ax   = axes[ri][ci]
        cm   = np.array(results[tgt][name]['cm'])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False, linewidths=0.5, annot_kws={'size': 11})
        is_best = (name == best_models[tgt])
        ax.set_title(('BEST: ' if is_best else '') + f'{tgt}\n{name}',
                     fontsize=9, fontweight='bold',
                     color='darkgreen' if is_best else 'black')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual',    fontsize=8)
plt.tight_layout()
plt.savefig('output/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/confusion_matrices.png")

# ── 12. MODEL COMPARISON BAR CHART ────────────────────────────
metrics = ['accuracy', 'precision', 'recall', 'f1']
mlabels = ['Accuracy', 'Precision\n(macro)', 'Recall\n(macro)', 'F1\n(macro)']
mnames  = list(results['HM_Si'].keys())
colors  = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Blast Furnace ML — Model Performance Comparison',
             fontsize=14, fontweight='bold')
for ai, (tgt, _) in enumerate(TARGETS):
    ax = axes[ai]; x = np.arange(len(metrics)); w = 0.18
    for i, (nm, col) in enumerate(zip(mnames, colors)):
        vals = [results[tgt][nm][m] for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=nm, color=col,
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_title(f'Target: {tgt}', fontsize=12, fontweight='bold')
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(mlabels, fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Score', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.7, color='gray', linestyle='--', lw=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/model_comparison.png")

# ── 13. FEATURE IMPORTANCE CHARTS ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Top 20 Feature Importances — Blast Furnace Classification',
             fontsize=13, fontweight='bold')
for ai, tgt in enumerate(['HM_Si', 'HM_Temp']):
    if tgt not in feat_imps:
        continue
    top20 = feat_imps[tgt].head(20)
    bc    = ['#C73E1D' if f.startswith('BRD_') else '#2E86AB' for f in top20.index]
    top20.plot(kind='barh', ax=axes[ai], color=bc, edgecolor='white')
    axes[ai].invert_yaxis()
    axes[ai].set_title(f'{tgt}  |  {best_models[tgt]}\n'
                       f'(F1={results[tgt][best_models[tgt]]["f1"]:.4f})',
                       fontsize=11, fontweight='bold')
    axes[ai].set_xlabel('Importance Score', fontsize=10)
    axes[ai].grid(axis='x', alpha=0.3)
    axes[ai].legend(handles=[Patch(color='#2E86AB', label='Process Feature'),
                              Patch(color='#C73E1D', label='Burden Feature')],
                    fontsize=9)
plt.tight_layout()
plt.savefig('output/feature_importances.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/feature_importances.png")

# ── 14. PREDICTION FUNCTION ───────────────────────────────────
def predict_new_tap(process_row: dict, burden_row: dict) -> dict:
    """
    Predict Si_class and Temp_class for one new tap.
    process_row : dict  e.g. {'Cold Blast Volume': 332000, 'HBT': 1162, ...}
    burden_row  : dict  e.g. {'BRD_SinterSp2Fresh': 62500, ...}
    """
    row_dict = {**process_row, **burden_row}
    df_new   = pd.DataFrame([row_dict])
    for col in feature_names:
        if col not in df_new.columns:
            df_new[col] = np.nan
    df_new = df_new[feature_names].apply(pd.to_numeric, errors='coerce')
    X_new  = imputer.transform(df_new)
    preds  = {}
    for tgt in ['HM_Si', 'HM_Temp']:
        clf_b = results[tgt][best_models[tgt]]['clf']
        cls   = clf_b.predict(scalers_dict[tgt].transform(X_new))[0]
        preds[tgt] = {'class_id': int(cls), 'label': CLASS_NAMES[cls]}
    print(f"  Predicted HM_Si  : {preds['HM_Si']['label']}")
    print(f"  Predicted HM_Temp: {preds['HM_Temp']['label']}")
    return preds

# ── 15. FILE SUMMARY ──────────────────────────────────────────
print("\n" + "=" * 65)
print("  OUTPUT FILES")
print("=" * 65)
for fn in ['output/model_summary.csv', 'output/labels.csv',
           'output/feature_importance_HM_Si.csv',
           'output/feature_importance_HM_Temp.csv',
           'output/confusion_matrices.png',
           'output/model_comparison.png',
           'output/feature_importances.png']:
    sz = os.path.getsize(fn)/1024 if os.path.exists(fn) else 0
    print(f"  {'OK' if sz > 0 else 'MISSING'}  {fn:<45} ({sz:.1f} KB)")

print("\n=== PIPELINE COMPLETE ===")
