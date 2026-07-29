#!/usr/bin/env python3
"""
BF-4 Blast Furnace — ML Classification Model (CORRECTED VERSION)
Targets : Si_Class, Temp_Class, Quality_Grade
Models  : Logistic Regression | Decision Tree | Random Forest |
          Gradient Boosting | XGBoost | SVM | MLP Neural Network
Data    : HM_ANALYSIS.xls + PARA.xlsx + BURDEN.csv
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.inspection import permutation_importance
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. pip install xgboost")

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)
os.makedirs("models",  exist_ok=True)

print("=" * 70)
print("  BF-4 BLAST FURNACE — ML CLASSIFICATION PIPELINE")
print("=" * 70)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BF_LAG_H  = 8      # burden → tap delay hours
IQR_K     = 3.0    # IQR outlier multiplier
TEST_SIZE = 0.20   # chronological test fraction
CV_FOLDS  = 5      # stratified k-fold

# Quality thresholds (operator spec)
SI_LO, SI_HI     = 0.25, 0.80   # %Si
T_LO,  T_HI      = 1480, 1535   # °C
S_MAX             = 0.055        # %S
P_MAX             = 0.180        # %P

# ─── STEP 1: LOAD ─────────────────────────────────────────────────────────────
def _clock(df):
    df.columns = (df.columns.astype(str).str.strip().str.upper()
                  .str.replace(r'\s+','_',regex=True))
    for c in ["CLOCK","DATETIME","DATE","TIME","CHARGETIME","TIMESTAMP"]:
        if c in df.columns:
            df['CLOCK'] = pd.to_datetime(df[c], dayfirst=True, errors='coerce')
            return df
    for c in df.columns:
        try:
            p = pd.to_datetime(df[c], dayfirst=True, errors='coerce')
            if p.notna().sum() > 0.5 * len(df):
                df['CLOCK'] = p
                return df
        except:
            pass
    raise ValueError("No timestamp column found")

def load_hm():
    print("[1a] HM_ANALYSIS.xls ...")
    df = pd.read_excel("HM_ANALYSIS.xls", sheet_name=0)
    df = _clock(df)
    df['CLOCK'] = df['CLOCK'].dt.floor('H')
    for c in [x for x in df.columns if x.startswith('HM_')]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['HM_SI','HM_TEMP'])
    return df.groupby('CLOCK')[['HM_SI','HM_TEMP','HM_S','HM_P',
                                  'HM_MN','HM_C']].mean().reset_index()

def load_para():
    print("[1b] PARA.xlsx ...")
    df = pd.read_excel("PARA.xlsx", sheet_name=0)
    df = _clock(df)
    df['CLOCK'] = df['CLOCK'].dt.floor('H')
    df = df.dropna(axis=1, thresh=int(0.01*len(df)))
    num = df.select_dtypes(include=np.number).columns.tolist()
    return df.groupby('CLOCK')[num].mean().reset_index()

def load_burden():
    print("[1c] BURDEN.csv ...")
    df = pd.read_csv("BURDEN.csv", low_memory=False)
    df = _clock(df)
    df['CLOCK'] = df['CLOCK'].dt.floor('H')
    bc = next((c for c in df.columns if 'BRAND' in c), None)
    wc = next((c for c in df.columns if 'ACTWT' in c or 'WEIGHT' in c), None)
    if bc and wc:
        df[wc] = pd.to_numeric(df[wc], errors='coerce').fillna(0)
        df['bl'] = df[bc].astype(str).str.lower()
        bmap = {'Coke_kg':['coke','cok'],'Sinter_kg':['sinter','sint'],
                'Pellet_kg':['pellet','pell'],'Ore_kg':['ore','bhq'],
                'Flux_kg':['lime','dolo'],'PCI_kg':['pci','coal'],'DRI_kg':['dri']}
        for col,kw in bmap.items():
            df[col] = np.where(df['bl'].str.contains('|'.join(kw),na=False),df[wc],0)
        bdf = df.groupby('CLOCK')[list(bmap)].sum().reset_index()
    else:
        num = df.select_dtypes(include=np.number).columns.tolist()
        bdf = df.groupby('CLOCK')[num].sum().reset_index()
    bdf['CLOCK'] += pd.Timedelta(hours=BF_LAG_H)
    return bdf

# ─── STEP 2: MERGE & ENGINEER ─────────────────────────────────────────────────
def build(hm, para, burden):
    print("[2] Merging ...")
    df = hm.merge(para,   on='CLOCK', how='inner', suffixes=('','_P'))
    df = df.merge(burden, on='CLOCK', how='inner', suffixes=('','_B'))

    coke = df.get('Coke_kg', pd.Series(np.ones(len(df)))).replace(0,np.nan)
    ore  = df.get('Ore_kg',  pd.Series(np.zeros(len(df))))
    flux = df.get('Flux_kg', pd.Series(np.zeros(len(df))))
    iron = (ore + df.get('Sinter_kg',0) + df.get('Pellet_kg',0)).replace(0,np.nan)
    df['OreCokeRatio']  = ore  / coke
    df['FluxIronRatio'] = flux / iron
    df['OreFrac']       = ore  / iron
    if 'HBT' in df.columns and 'O_FLOW_ACT' in df.columns:
        df['thermal_idx'] = df['HBT'] * df['O_FLOW_ACT'] / 1e5
    if 'COAL_ACT' in df.columns and 'PROD_RATE' in df.columns:
        df['coal_intensity'] = df['COAL_ACT'] / df['PROD_RATE'].replace(0,np.nan)

    num = df.select_dtypes(include=np.number).columns
    for c in num:
        q1,q3 = df[c].quantile([0.25,0.75])
        iqr   = q3-q1
        df    = df[df[c].between(q1-IQR_K*iqr, q3+IQR_K*iqr)]

    df = df.sort_values('CLOCK').reset_index(drop=True)
    print(f"    Clean rows: {len(df):,}")
    return df

# ─── STEP 3: CREATE CLASS LABELS ──────────────────────────────────────────────
def make_labels(df):
    print("[3] Creating classification targets ...")
    df = df.copy()

    # Si_Class
    df['Si_Class'] = pd.cut(df['HM_SI'],
                             bins=[-np.inf, SI_LO, SI_HI, np.inf],
                             labels=['Low','Normal','High'])

    # Temp_Class
    df['Temp_Class'] = pd.cut(df['HM_TEMP'],
                               bins=[-np.inf, T_LO, T_HI, np.inf],
                               labels=['Cold','Normal','Hot'])

    # Quality_Grade: A=all in spec, B=1 off, C=2+ off
    checks = pd.DataFrame({
        'si_ok':   df['HM_SI'].between(SI_LO, SI_HI),
        't_ok':    df['HM_TEMP'].between(T_LO, T_HI),
        's_ok':    df['HM_S']  <= S_MAX if 'HM_S'  in df.columns else True,
        'p_ok':    df['HM_P']  <= P_MAX if 'HM_P'  in df.columns else True,
    })
    n_off = (~checks).sum(axis=1)
    df['Quality_Grade'] = pd.cut(n_off, bins=[-1,0,1,np.inf],
                                  labels=['A','B','C'])

    for t in ['Si_Class','Temp_Class','Quality_Grade']:
        df[t] = df[t].astype(str)
        vc    = df[t].value_counts()
        print(f"    {t}: {dict(vc)}")

    return df.dropna(subset=['Si_Class','Temp_Class','Quality_Grade'])

# ─── STEP 4: FEATURE ENGINEERING ──────────────────────────────────────────────
def feature_eng(df):
    print("[4] Feature engineering (lag + rolling) ...")
    fd = df.copy()
    
    key_proc = ['HBT','O_FLOW_ACT','COAL_ACT','ETACO','HM_SI','HM_TEMP',
                'OreCokeRatio','FluxIronRatio','thermal_idx']
    for c in key_proc:
        if c in fd.columns:
            for lag in [1,2,4,8]:
                fd[f'{c}_lag{lag}'] = fd[c].shift(lag)
            fd[f'{c}_roll4'] = fd[c].shift(1).rolling(4).mean()
            fd[f'{c}_roll8'] = fd[c].shift(1).rolling(8).mean()

    fd = fd.dropna().reset_index(drop=True)
    print(f"    Rows after lag drop: {len(fd):,}")
    return fd

def get_feat_cols(fd):
    exclude = {'CLOCK','HM_SI','HM_TEMP','HM_S','HM_P','HM_MN','HM_C',
               'Si_Class','Temp_Class','Quality_Grade'}
    return [c for c in fd.columns
            if c not in exclude and fd[c].dtype in [np.float64,np.int64]]

# ─── STEP 5: MODEL TRAINING ────────────────────────────────────────────────────
MODELS = {
    'LogReg'  : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'DecTree' : DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
    'RandFor' : RandomForestClassifier(n_estimators=200, max_depth=12,
                                        min_samples_leaf=3, n_jobs=-1, random_state=42),
    'GradBst' : GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                             learning_rate=0.08, random_state=42),
    'SVM'     : SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'MLP'     : MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500,
                               learning_rate_init=0.001, random_state=42),
}
if HAS_XGB:
    MODELS['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=5,
                                        learning_rate=0.08, use_label_encoder=False,
                                        eval_metric='mlogloss', random_state=42,
                                        n_jobs=-1)

def train_all(X_tr, X_te, y_tr, y_te, target_name, feat_cols):
    print(f"\n  ── TARGET: {target_name} ──")
    le       = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)
    classes  = le.classes_

    results = {}
    for name, clf in MODELS.items():
        try:
            clf.fit(X_tr, y_tr_enc)
            y_pred  = clf.predict(X_te)
            acc     = accuracy_score(y_te_enc, y_pred)
            cv_acc  = cross_val_score(clf, X_tr, y_tr_enc,
                                       cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                                           random_state=42),
                                       scoring='accuracy').mean()
            # macro ROC-AUC (OvR)
            try:
                prob = clf.predict_proba(X_te)
                if len(classes) == 2:
                    auc_val = roc_auc_score(y_te_enc, prob[:,1])
                else:
                    auc_val = roc_auc_score(y_te_enc, prob,
                                             multi_class='ovr', average='macro')
            except Exception:
                auc_val = np.nan

            results[name] = {
                'clf': clf, 'le': le, 'classes': classes,
                'acc': acc, 'cv_acc': cv_acc, 'auc': auc_val,
                'y_pred': y_pred, 'y_te_enc': y_te_enc,
            }
            print(f"    {name:<10} Acc={acc:.4f}  CV={cv_acc:.4f}  AUC={auc_val:.4f}")
            joblib.dump(clf, f"models/clf_{target_name}_{name}.pkl")
        except Exception as e:
            print(f"    {name:<10} FAILED: {e}")

    return results

# ─── STEP 6: VISUALISATIONS ───────────────────────────────────────────────────
COLORS = ['#2C7BB6','#D7191C','#1A9641','#FDAE61','#762A83','#F46D43','#74ADD1']

def plot_class_dist(fd):
    fig, axes = plt.subplots(1,3, figsize=(16,5))
    fig.suptitle('BF-4 Hot Metal Quality — Class Distributions', fontsize=13, fontweight='bold')
    targets = ['Si_Class','Temp_Class','Quality_Grade']
    palette = {'Low':'#2166AC','Normal':'#1A9641','High':'#D73027',
               'Cold':'#2166AC','Hot':'#D73027',
               'A':'#1A9641','B':'#FDAE61','C':'#D73027'}
    for ax, tgt in zip(axes, targets):
        vc = fd[tgt].value_counts().sort_index()
        bars = ax.bar(vc.index, vc.values,
                      color=[palette.get(k,'#888888') for k in vc.index],
                      edgecolor='white', linewidth=1.2)
        ax.set_title(tgt, fontweight='bold', fontsize=11)
        ax.set_xlabel('Class'); ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        for bar,v in zip(bars, vc.values):
            pct = 100*v/vc.sum()
            ax.text(bar.get_x()+bar.get_width()/2, v+10,
                    f'{v:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('output/clf01_class_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf01_class_dist.png")

def plot_model_comparison(all_results):
    targets  = list(all_results.keys())
    models   = list(MODELS.keys())
    fig, axes = plt.subplots(len(targets), 2, figsize=(16, 5*len(targets)))
    if len(targets) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('BF-4 Classification — Model Comparison (All Targets)',
                 fontsize=13, fontweight='bold')
    x = np.arange(len(models))
    for r, tgt in enumerate(targets):
        res = all_results[tgt]
        accs   = [res[m]['acc']    if m in res else 0 for m in models]
        aucs   = [res[m]['auc']    if m in res else 0 for m in models]
        cvaccs = [res[m]['cv_acc'] if m in res else 0 for m in models]
        for col, (vals, lbl, ylbl) in enumerate([
            (list(zip(accs,cvaccs)), 'Accuracy vs CV Accuracy', 'Score'),
            (aucs, 'Macro ROC-AUC', 'AUC'),
        ]):
            ax = axes[r,col]
            if col == 0:
                bars1 = ax.bar(x-0.2, accs,   0.35, label='Test Acc', color=COLORS[0], alpha=0.85)
                bars2 = ax.bar(x+0.2, cvaccs, 0.35, label='CV Acc',   color=COLORS[1], alpha=0.85)
                ax.legend(fontsize=9)
                for bar,v in zip(bars1,accs):
                    if v > 0:
                        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=7)
            else:
                bars1 = ax.bar(x, aucs, 0.55, color=COLORS[2], alpha=0.85)
                for bar,v in zip(bars1,aucs):
                    if v > 0:
                        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9, rotation=15)
            ax.set_title(f'{tgt} — {lbl}', fontweight='bold', fontsize=10)
            ax.set_ylabel(ylbl); ax.set_ylim(0, 1.15)
            ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/clf02_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf02_model_comparison.png")

def plot_confusion_matrices(all_results):
    targets = list(all_results.keys())
    best_models = {}
    for tgt in targets:
        res = all_results[tgt]
        best_m = max((m for m in res), key=lambda m: res[m]['acc'])
        best_models[tgt] = best_m

    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    fig.suptitle('Confusion Matrices — Best Model per Target',
                 fontsize=13, fontweight='bold')
    for ax, tgt in zip(axes, targets):
        best_m = best_models[tgt]
        res    = all_results[tgt][best_m]
        cm     = confusion_matrix(res['y_te_enc'], res['y_pred'])
        classes = res['classes']
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{tgt}\n(Best: {best_m}, Acc={res["acc"]:.3f})',
                     fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig('output/clf03_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf03_confusion_matrices.png")

def plot_feature_importance(all_results, feat_cols, X_te, y_te_dict):
    targets = list(all_results.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(7*len(targets), 7))
    if len(targets) == 1:
        axes = [axes]
    fig.suptitle('Top 15 Feature Importances (Random Forest)',
                 fontsize=13, fontweight='bold')
    for ax, tgt in zip(axes, targets):
        res = all_results[tgt]
        if 'RandFor' not in res:
            ax.set_title(f'{tgt}\n(RF unavailable)'); continue
        clf  = res['RandFor']['clf']
        imp  = clf.feature_importances_
        idx  = np.argsort(imp)[-15:]
        colors = plt.cm.RdYlGn(np.linspace(0.3,0.9,15))
        ax.barh([feat_cols[i] for i in idx], imp[idx], color=colors, edgecolor='white')
        ax.set_title(f'{tgt}\nFeature Importances (RF)', fontweight='bold', fontsize=10)
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/clf04_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf04_feature_importance.png")

def plot_roc_curves(all_results, X_te, y_te_dict):
    targets = list(all_results.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(7*len(targets), 6))
    if len(targets) == 1:
        axes = [axes]
    fig.suptitle('ROC Curves — One-vs-Rest (macro)', fontsize=13, fontweight='bold')
    for ax, tgt in zip(axes, targets):
        res = all_results[tgt]
        for i, (name, r) in enumerate(res.items()):
            try:
                clf     = r['clf']
                le      = r['le']
                classes = r['classes']
                y_score = clf.predict_proba(X_te)
                y_bin   = np.eye(len(classes))[r['y_te_enc']]
                fpr, tpr, auc_val = {}, {}, {}
                for j in range(len(classes)):
                    fpr[j], tpr[j], _ = roc_curve(y_bin[:,j], y_score[:,j])
                    auc_val[j] = auc(fpr[j], tpr[j])
                macro_fpr = np.linspace(0,1,200)
                macro_tpr = np.mean([np.interp(macro_fpr, fpr[j], tpr[j])
                                      for j in range(len(classes))], axis=0)
                macro_auc = np.mean(list(auc_val.values()))
                ax.plot(macro_fpr, macro_tpr,
                        label=f'{name} (AUC={macro_auc:.3f})',
                        linewidth=1.8, color=COLORS[i % len(COLORS)])
            except Exception:
                pass
        ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5)
        ax.set_title(f'{tgt}', fontweight='bold', fontsize=11)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/clf05_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf05_roc_curves.png")

def plot_heatmap_summary(all_results):
    targets = list(all_results.keys())
    models  = list(MODELS.keys())
    acc_mat  = np.zeros((len(targets), len(models)))
    auc_mat  = np.zeros((len(targets), len(models)))
    for r, tgt in enumerate(targets):
        for c, m in enumerate(models):
            if m in all_results[tgt]:
                acc_mat[r,c] = all_results[tgt][m]['acc']
                auc_mat[r,c] = all_results[tgt][m]['auc']

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))
    fig.suptitle('BF-4 Classification Scorecard — All Models × All Targets',
                 fontsize=13, fontweight='bold')
    for ax, mat, lbl in [(ax1,acc_mat,'Test Accuracy'),(ax2,auc_mat,'Macro ROC-AUC')]:
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(models)));    ax.set_xticklabels(models, rotation=20, fontsize=9)
        ax.set_yticks(range(len(targets)));   ax.set_yticklabels(targets, fontsize=9)
        ax.set_title(lbl, fontweight='bold')
        for r in range(len(targets)):
            for c in range(len(models)):
                v = mat[r,c]
                ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                        fontsize=9, color='black' if v < 0.9 else 'white',
                        fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig('output/clf06_scorecard_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf06_scorecard_heatmap.png")

def plot_learning_curves(X_tr, feat_cols, y_all, all_results):
    from sklearn.model_selection import learning_curve
    targets = list(all_results.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(7*len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    fig.suptitle('Learning Curves — Random Forest (best model)',
                 fontsize=13, fontweight='bold')
    for ax, tgt in zip(axes, targets):
        res = all_results[tgt]
        if 'RandFor' not in res: continue
        le      = res['RandFor']['le']
        y_enc   = le.transform(pd.Series(y_all[tgt]).dropna().astype(str))
        X_sub   = X_tr[:len(y_enc)]
        sizes, tr_sc, cv_sc = learning_curve(
            RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            X_sub, y_enc, cv=3, scoring='accuracy',
            train_sizes=np.linspace(0.2,1.0,8), n_jobs=-1)
        tr_m = tr_sc.mean(axis=1); tr_s = tr_sc.std(axis=1)
        cv_m = cv_sc.mean(axis=1); cv_s = cv_sc.std(axis=1)
        ax.plot(sizes, tr_m, 'b-', label='Train', linewidth=2)
        ax.fill_between(sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='b')
        ax.plot(sizes, cv_m, 'r-', label='CV',    linewidth=2)
        ax.fill_between(sizes, cv_m-cv_s, cv_m+cv_s, alpha=0.15, color='r')
        ax.set_title(f'{tgt}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Training Samples'); ax.set_ylabel('Accuracy')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/clf07_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: output/clf07_learning_curves.png")

# ─── STEP 7: DECISION TREE RULES ──────────────────────────────────────────────
def print_tree_rules(all_results, feat_cols):
    print("\n[7] Decision Tree Rules (Si_Class — depth 5) ...")
    if 'Si_Class' not in all_results: return
    res = all_results['Si_Class']
    if 'DecTree' not in res: return
    clf     = res['DecTree']['clf']
    classes = res['DecTree']['classes']
    rules   = export_text(clf, feature_names=feat_cols, max_depth=5)
    # Save rules to file
    with open('output/clf_tree_rules.txt','w') as f:
        f.write(f"Decision Tree Rules — Si_Class\n{'='*60}\n")
        f.write(f"Classes: {classes}\n\n")
        f.write(rules)
    print("    Saved: output/clf_tree_rules.txt")
    print(rules[:2000])   # print first 2000 chars

# ─── STEP 8: INFERENCE FUNCTION ───────────────────────────────────────────────
def classify_tap(process_state: dict, feat_cols: list,
                 all_results: dict) -> dict:
    """Real-time quality classification for a single BF tap."""
    row    = np.array([[process_state.get(f,0.0) for f in feat_cols]])
    output = {}
    SPEC   = {
        'Si_Class':      {'Normal':1.0, 'Low':-1.0, 'High':-1.0},
        'Temp_Class':    {'Normal':1.0, 'Cold':-1.0,'Hot':-1.0},
        'Quality_Grade': {'A':1.0,      'B': 0.5,   'C':-1.0},
    }
    for tgt, res in all_results.items():
        best_m = max(res, key=lambda m: res[m]['acc'])
        clf    = res[best_m]['clf']
        le     = res[best_m]['le']
        pred_enc  = clf.predict(row)[0]
        pred_lbl  = le.inverse_transform([pred_enc])[0]
        prob      = clf.predict_proba(row)[0]
        prob_dict = {le.inverse_transform([i])[0]: round(float(p),4)
                     for i,p in enumerate(prob)}
        desirability = SPEC.get(tgt,{}).get(pred_lbl, 0.0)
        output[tgt] = {
            'prediction':    pred_lbl,
            'probabilities': prob_dict,
            'model_used':    best_m,
            'desirability':  desirability,
            'alert':        ('✓ OK' if desirability >= 0.5 else '⚠ OFF-SPEC'),
        }
    return output

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    hm     = load_hm()
    para   = load_para()
    burden = load_burden()

    df  = build(hm, para, burden)
    df  = make_labels(df)
    fd  = feature_eng(df)

    feat_cols = get_feat_cols(fd)
    X = fd[feat_cols].fillna(0).values
    sc = RobustScaler()
    X  = sc.fit_transform(X)
    joblib.dump(sc,'models/clf_scaler.pkl')

    TARGETS_CLF = ['Si_Class','Temp_Class','Quality_Grade']
    n_tr  = int(len(fd)*(1-TEST_SIZE))
    X_tr, X_te = X[:n_tr], X[n_tr:]

    all_results  = {}
    y_te_dict    = {}
    y_all_dict   = {}

    for tgt in TARGETS_CLF:
        y      = fd[tgt].astype(str).values
        y_tr   = y[:n_tr]
        y_te   = y[n_tr:]
        y_te_dict[tgt] = y_te
        y_all_dict[tgt] = y
        all_results[tgt] = train_all(X_tr, X_te, y_tr, y_te, tgt, feat_cols)

    print("\n[Generating plots ...]")
    plot_class_dist(fd)
    plot_model_comparison(all_results)
    plot_confusion_matrices(all_results)
    plot_feature_importance(all_results, feat_cols, X_te, y_te_dict)
    plot_roc_curves(all_results, X_te, y_te_dict)
    plot_heatmap_summary(all_results)
    plot_learning_curves(X_tr, feat_cols, y_all_dict, all_results)
    print_tree_rules(all_results, feat_cols)

    # Demo inference
    print("\n[Demo — classify_tap()]")
    state = {f: float(fd[f].iloc[-1]) if f in fd.columns else 0.0 for f in feat_cols}
    result = classify_tap(state, feat_cols, all_results)
    for tgt,r in result.items():
        print(f"  {tgt}: {r['prediction']}  {r['alert']}")
        print(f"    Model: {r['model_used']}  |  Confidence: {max(r['probabilities'].values()):.3f}")
        print(f"    Probabilities: {r['probabilities']}\n")

    # Save results CSV
    rows = []
    for tgt in TARGETS_CLF:
        res = all_results[tgt]
        for m,r in res.items():
            rows.append({'Target':tgt,'Model':m,'Accuracy':round(r['acc'],4),
                         'CV_Accuracy':round(r['cv_acc'],4),'ROC_AUC':round(r['auc'],4)})
    pd.DataFrame(rows).to_csv('output/clf_results_summary.csv', index=False)
    print("\n✓ Saved: output/clf_results_summary.csv")
    print("✓ All outputs in ./output/  |  Models in ./models/")
    print(f"✓ Completed: {datetime.now().strftime('%d-%b-%Y %H:%M')}")

if __name__ == "__main__":
    main()