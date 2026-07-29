#!/usr/bin/env python3
"""
=============================================================================
BF-4 BLAST FURNACE HOT METAL QUALITY PREDICTION  [FIXED VERSION v3 — FINAL]
Complete ML Pipeline with Mathematical Equations + Confidence Intervals
=============================================================================
BUG FIXED (v3):
  GradientBoostingRegressor.estimators_ is a 2D ndarray of shape (n_estimators, 1).
  Iterating "for t in clf.estimators_" yields 1D numpy arrays, NOT tree objects.
  Calling .predict() on a numpy array raises:
      AttributeError: 'numpy.ndarray' object has no attribute 'predict'
  FIX: Use isinstance(clf, (RandomForestRegressor, ExtraTreesRegressor)) to identify
  RF/ET models, then iterate clf.estimators_ (flat list of trees).
  For GBM/ANN, use ±1.96*RMSE as the prediction interval.

Datasets Required (same folder):
  PARA.xlsx  |  HM_ANALYSIS.xls  |  BURDEN.csv

Install:
    pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib openpyxl xlrd

Run:
    python BF_Full_Pipeline_v3_FINAL.py
=============================================================================
"""

import os, copy, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
TARGET_COLS  = ["HM_C","HM_SI","HM_S","HM_MN","HM_P","HM_TEMP"]
TARGET_UNITS = {"HM_C":"%C","HM_SI":"%Si","HM_S":"%S",
                "HM_MN":"%Mn","HM_P":"%P","HM_TEMP":"°C"}
SPEC_LIMITS  = {"HM_C":(4.30,4.90),"HM_SI":(0.25,0.80),"HM_S":(0.00,0.055),
                "HM_MN":(0.80,1.80),"HM_P":(0.00,0.180),"HM_TEMP":(1470,1540)}
BF_FEAT_NAMES = [
    "HBT","Oxygen Flow","Coal Actual","PROD_RATE","ETACO","Permeabilty",
    "Cold Blast Volume","HBP","FTP","Steam","B MOIST","SLAG_RATE",
    "Radar Level","Heat_Load Q1","Heat_Load Q2","Heat_Load Q3",
    "Heat_Load Q4","Top DP","Middle DP","Bottom DP","Heat Flow Flux","Coal Inj. SP",
]
BURDEN_FEAT_NAMES = [
    "Coke_kg","Sinter_kg","Pellet_kg","Ore_kg","Limestone_kg","Dolomite_kg",
    "DRI_kg","MixedMaterial_kg","TotalIron_kg","TotalFlux_kg","OreCokeRatio",
    "SinterFrac","PelletFrac","OreFrac","FluxIronRatio","TotalCharge_kg",
]
ALPHA = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD BURDEN.csv
# ─────────────────────────────────────────────────────────────────────────────
def classify_material(brand):
    b = str(brand).upper()
    if "COKE" in b or "NUTCOKE" in b: return "Coke_kg"
    if "SINTER" in b:                  return "Sinter_kg"
    if "PELLET" in b:                  return "Pellet_kg"
    if "ORE" in b or "BHQ" in b:      return "Ore_kg"
    if "LIMESTONE" in b:               return "Limestone_kg"
    if "DOLOMITE" in b:                return "Dolomite_kg"
    if "DRI" in b:                     return "DRI_kg"
    if "MIXED" in b:                   return "MixedMaterial_kg"
    return "Other_kg"

def load_burden(path="BURDEN.csv", lag_hours=8):
    print("\n[1/9] Loading BURDEN.csv ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["BRANDCODE"]  = df["BRANDCODE"].astype(str).str.strip()
    df["CHARGETIME"] = pd.to_datetime(df["CHARGETIME"], dayfirst=True, errors="coerce")
    df["CLOCK"]      = df["CHARGETIME"].dt.floor("h")
    df["MaterialGroup"] = df["BRANDCODE"].apply(classify_material)
    pivot = (df.groupby(["CLOCK","MaterialGroup"])["ACTWT"]
               .sum().unstack(fill_value=0).reset_index())
    for col in ["Coke_kg","Sinter_kg","Pellet_kg","Ore_kg","Limestone_kg",
                "Dolomite_kg","DRI_kg","MixedMaterial_kg","Other_kg"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    p = pivot.copy()
    p["TotalIron_kg"]   = (p["Ore_kg"]+p["Pellet_kg"]+p["Sinter_kg"]
                           +p["DRI_kg"]+p["MixedMaterial_kg"])
    p["TotalFlux_kg"]   = p["Limestone_kg"]+p["Dolomite_kg"]
    p["OreCokeRatio"]   = p["TotalIron_kg"] / p["Coke_kg"].replace(0,np.nan)
    p["SinterFrac"]     = p["Sinter_kg"]    / p["TotalIron_kg"].replace(0,np.nan)
    p["PelletFrac"]     = p["Pellet_kg"]    / p["TotalIron_kg"].replace(0,np.nan)
    p["OreFrac"]        = ((p["Ore_kg"]+p["MixedMaterial_kg"])
                           / p["TotalIron_kg"].replace(0,np.nan))
    p["FluxIronRatio"]  = p["TotalFlux_kg"] / p["TotalIron_kg"].replace(0,np.nan)
    p["TotalCharge_kg"] = p["Coke_kg"]+p["TotalIron_kg"]+p["TotalFlux_kg"]
    p["CLOCK"]          = p["CLOCK"] + pd.Timedelta(hours=lag_hours)
    print(f"    Raw records : {len(df):,}  |  Hourly (lag {lag_hours}h): {len(p):,}")
    return p

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD HM_ANALYSIS.xls
# ─────────────────────────────────────────────────────────────────────────────
def load_hm(path="HM_ANALYSIS.xls"):
    print("\n[2/9] Loading HM_ANALYSIS.xls ...")
    df = pd.read_excel(path, sheet_name="Sheet 1")
    df["SAMPLETAKEN"] = pd.to_datetime(df["SAMPLETAKEN"])
    df["CLOCK"]       = df["SAMPLETAKEN"].dt.floor("h")
    hm_h = df.groupby("CLOCK")[TARGET_COLS].mean().reset_index()
    print(f"    Raw taps: {len(df):,}  |  Hourly avg: {len(hm_h):,}")
    return hm_h

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LOAD PARA.xlsx
# ─────────────────────────────────────────────────────────────────────────────
def load_para(path="PARA.xlsx"):
    print("\n[3/9] Loading PARA.xlsx ...")
    raw = pd.read_excel(path, sheet_name="BF-4 Data", header=None)
    vn  = raw.iloc[1].tolist()
    bf  = raw.iloc[3:].copy()
    bf.columns = vn
    bf  = bf.rename(columns={vn[0]:"CLOCK"}).reset_index(drop=True)
    for col in bf.columns:
        if col != "CLOCK":
            bf[col] = pd.to_numeric(bf[col], errors="coerce")
    bf["CLOCK"] = pd.to_datetime(bf["CLOCK"], errors="coerce")
    bf = bf[[c for c in bf.columns
              if c=="CLOCK" or bf[c].isnull().mean()<0.99]]
    if "Heat Flow Flux" in bf.columns:
        bf = bf[bf["Heat Flow Flux"] < 100]
    bf["CLOCK"] = bf["CLOCK"].dt.floor("h")
    print(f"    Records: {len(bf):,}  |  Parameters: {len(bf.columns)-1}")
    return bf

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MERGE + CLEAN + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def iqr_clean(df, cols, factor=3.0):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns: continue
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        mask &= (df[c] >= Q1-factor*IQR) & (df[c] <= Q3+factor*IQR)
    return df[mask]

def build_dataset(bf, burden_lag, hm_h):
    print("\n[4/9] Merging + cleaning + feature engineering ...")
    merged = pd.merge(bf, burden_lag, on="CLOCK", how="inner")
    merged = pd.merge(merged, hm_h,   on="CLOCK", how="inner")
    base_feats = [c for c in (BF_FEAT_NAMES+BURDEN_FEAT_NAMES)
                  if c in merged.columns]
    df = merged[["CLOCK"]+base_feats+TARGET_COLS].copy()
    df = iqr_clean(df, TARGET_COLS)
    df = iqr_clean(df, base_feats)
    df = df.dropna().sort_values("CLOCK").reset_index(drop=True)
    print(f"    After clean : {len(df):,} rows")

    eng = list(base_feats)
    lag_vars = ["HBT","Oxygen Flow","Coal Actual","ETACO","Permeabilty",
                "PROD_RATE","OreCokeRatio","SinterFrac","FluxIronRatio"]
    for var in lag_vars:
        if var not in df.columns: continue
        for lag in [2,4,8]:
            cn = f"{var}_lag{lag}h"
            df[cn] = df[var].shift(lag)
            eng.append(cn)
    roll_vars = ["HBT","ETACO","Coal Actual","Oxygen Flow","OreCokeRatio","SinterFrac"]
    for var in roll_vars:
        if var not in df.columns: continue
        for win in [4,8]:
            cn = f"{var}_roll{win}h"
            df[cn] = df[var].rolling(win, min_periods=2).mean()
            eng.append(cn)
    df["thermal_idx"]    = df["HBT"]*df["Oxygen Flow"]/1e6
    df["coal_intensity"] = df["Coal Actual"]/df["PROD_RATE"].replace(0,np.nan)
    df["gas_eff"]        = df["ETACO"]*df["Permeabilty"]
    df["DP_ratio"]       = df["Top DP"]/df["Bottom DP"].replace(0,np.nan)
    df["burden_thermal"] = df["OreCokeRatio"]*df["HBT"]
    df["flux_thermal"]   = df["FluxIronRatio"]*df["HBT"]
    df["pellet_thermal"] = df["PelletFrac"]*df["HBT"]
    for c in ["thermal_idx","coal_intensity","gas_eff","DP_ratio",
              "burden_thermal","flux_thermal","pellet_thermal"]:
        eng.append(c)
    df = df.dropna().reset_index(drop=True)
    print(f"    Final shape : {df.shape}  |  Features: {len(eng)}")
    return df, eng

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EDA
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df):
    print("\n[5/9] EDA plots ...")
    key = ["HBT","OreCokeRatio","SinterFrac","FluxIronRatio","Coal Actual",
           "ETACO","Oxygen Flow","Permeabilty",
           "HM_C","HM_SI","HM_S","HM_MN","HM_P","HM_TEMP"]
    key = [c for c in key if c in df.columns]
    plt.figure(figsize=(13,9))
    sns.heatmap(df[key].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, annot_kws={"size":7})
    plt.title("Correlation: BF Process + Burden → HM Quality", fontsize=12)
    plt.tight_layout()
    plt.savefig("output/01_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(14,8))
    for i, tgt in enumerate(TARGET_COLS):
        ax   = axes[i//3][i%3]
        data = df[tgt].dropna()
        ax.hist(data, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red", linestyle="--", lw=1.5,
                   label=f"μ={data.mean():.3f}")
        lo, hi = SPEC_LIMITS[tgt]
        ax.axvspan(lo, hi, alpha=0.12, color="green", label="Spec")
        ax.set_title(f"{tgt}  ({TARGET_UNITS[tgt]})", fontweight="bold")
        ax.set_xlabel(TARGET_UNITS[tgt])
        ax.legend(fontsize=7)
    plt.suptitle("Hot Metal Quality Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/02_target_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/01_correlation_heatmap.png")
    print("    Saved: output/02_target_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_all(df, eng_feats):
    print("\n[6/9] Training models ...")

    X_raw    = df[eng_feats].values
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, "models/scaler.pkl")

    CLFS = {
        "Ridge":        Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10,
                                               min_samples_leaf=5, n_jobs=-1,
                                               random_state=42),
        "ExtraTrees":   ExtraTreesRegressor(n_estimators=100, max_depth=10,
                                             min_samples_leaf=5, n_jobs=-1,
                                             random_state=42),
        "GradBoost":    GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                   learning_rate=0.08,
                                                   subsample=0.8, random_state=42),
        "ANN":          MLPRegressor(hidden_layer_sizes=(128,64,32),
                                      activation="relu", max_iter=300,
                                      early_stopping=True, random_state=42),
    }

    all_results = {}
    feature_imp = {}
    ridge_coef  = {}
    hdr = f"  {'Model':<14}{'R²':>8}{'RMSE':>12}{'MAE':>12}"
    sep = "  " + "-"*46

    for target in TARGET_COLS:
        y = df[target].values
        Xtr_s, Xte_s, ytr, yte = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        all_results[target] = {}
        best_r2, best_name, best_clf = -999, None, None
        best_rmse = None

        print(f"\n  ─── {target}  ({TARGET_UNITS[target]}) ───")
        print(hdr); print(sep)

        for cname, clf_tmpl in CLFS.items():
            clf  = copy.deepcopy(clf_tmpl)
            clf.fit(Xtr_s, ytr)
            yp   = clf.predict(Xte_s)
            r2   = r2_score(yte, yp)
            rmse = float(np.sqrt(mean_squared_error(yte, yp)))
            mae  = float(mean_absolute_error(yte, yp))
            all_results[target][cname] = {
                "R2":round(r2,4), "RMSE":round(rmse,6), "MAE":round(mae,6),
                "y_test":yte.tolist(), "y_pred":yp.tolist()
            }
            print(f"  {cname:<14}{r2:>8.4f}{rmse:>12.5f}{mae:>12.5f}")
            if r2 > best_r2:
                best_r2, best_name, best_clf = r2, cname, clf
                best_rmse = rmse
                if hasattr(clf, "feature_importances_"):
                    imp = dict(zip(eng_feats, clf.feature_importances_))
                    feature_imp[target] = sorted(imp.items(),
                                                  key=lambda x: -x[1])[:15]

        joblib.dump(best_clf, f"models/best_{target}.pkl")
        all_results[target]["best"]      = best_name
        all_results[target]["best_r2"]   = round(best_r2, 4)
        all_results[target]["best_rmse"] = round(best_rmse, 6)
        all_results[target]["Xtr_s"]     = Xtr_s
        all_results[target]["Xte_s"]     = Xte_s
        all_results[target]["ytr"]       = ytr
        all_results[target]["yte"]       = yte
        print(f"  >>> BEST: {best_name}  R²={best_r2:.4f}")

        ridge_clf = Ridge(alpha=1.0)
        ridge_clf.fit(Xtr_s, ytr)
        ridge_coef[target]                     = dict(zip(eng_feats, ridge_clf.coef_))
        all_results[target]["ridge_intercept"] = float(ridge_clf.intercept_)

    return scaler, X_scaled, all_results, feature_imp, ridge_coef

# ─────────────────────────────────────────────────────────────────────────────
# CI / PI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def is_rf_type(clf):
    """True only for RF and ET — both have a flat list of tree estimators."""
    return isinstance(clf, (RandomForestRegressor, ExtraTreesRegressor))


def compute_pi_rf(clf, x_scaled_1row):
    """
    95% PI via tree-level quantile distribution.
    Only call this when is_rf_type(clf) is True.
    clf.estimators_ is a flat Python list of DecisionTreeRegressor objects.
    x_scaled_1row : ndarray shape (1, p) — already RobustScaler-transformed.
    """
    tree_preds = np.array([t.predict(x_scaled_1row)[0]
                            for t in clf.estimators_])  # clf.estimators_ is a list for RF/ET
    y_pred    = float(np.mean(tree_preds))
    pi_lo     = float(np.percentile(tree_preds, 2.5))
    pi_hi     = float(np.percentile(tree_preds, 97.5))
    std_trees = float(np.std(tree_preds))
    return {"y_pred":round(y_pred,5), "PI_lower":round(pi_lo,5),
            "PI_upper":round(pi_hi,5), "std_trees":round(std_trees,6)}


def compute_pi_rmse(clf, x_scaled_1row, rmse, n_train, p, alpha=0.05):
    """
    ±t*RMSE*sqrt(1+1/n) prediction interval for GBM, Ridge, ANN.
    """
    yp     = float(clf.predict(x_scaled_1row)[0])
    t_crit = stats.t.ppf(1-alpha/2, df=max(n_train-p-1, 1))
    hw     = t_crit * rmse * np.sqrt(1 + 1/n_train)
    return {"y_pred":round(yp,5), "PI_lower":round(yp-hw,5),
            "PI_upper":round(yp+hw,5), "std_trees":np.nan}


def predict_with_pi(clf, x_scaled_1row, rmse, n_train, p, alpha=0.05):
    """Unified PI dispatcher — uses tree quantiles for RF/ET, RMSE-based for others."""
    if is_rf_type(clf):
        return compute_pi_rf(clf, x_scaled_1row)
    else:
        return compute_pi_rmse(clf, x_scaled_1row, rmse, n_train, p, alpha)


def cv_r2_ci(clf, X_scaled, y, cv=5, alpha=0.05):
    scores  = cross_val_score(clf, X_scaled, y, cv=cv, scoring="r2", n_jobs=-1)
    mean_r2 = float(np.mean(scores))
    se_r2   = float(np.std(scores, ddof=1)/np.sqrt(cv))
    t_crit  = stats.t.ppf(1-alpha/2, df=cv-1)
    return {"R2_mean":round(mean_r2,4),
            "CI_lo":round(max(0, mean_r2-t_crit*se_r2),4),
            "CI_hi":round(min(1, mean_r2+t_crit*se_r2),4),
            "R2_std":round(float(np.std(scores,ddof=1)),4)}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PRINT RIDGE EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
def print_equations(all_results, ridge_coef):
    print("\n" + "="*72)
    print("RIDGE REGRESSION EQUATIONS  (x̃ᵢ = (xᵢ − median) / IQR)")
    print("="*72)
    for target in TARGET_COLS:
        b0   = all_results[target]["ridge_intercept"]
        coef = ridge_coef[target]
        top8 = sorted(coef.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        rmse = all_results[target]["best_rmse"]
        best = all_results[target]["best"]
        Xte  = all_results[target]["Xte_s"]
        n    = Xte.shape[0] + len(all_results[target]["ytr"])
        p    = Xte.shape[1]
        t_c  = stats.t.ppf(0.975, df=n-p-1)
        ci   = t_c * rmse / np.sqrt(n)
        pi   = t_c * rmse * np.sqrt(1 + 1/n)
        print(f"\n  {target} ({TARGET_UNITS[target]})  β₀ = {b0:.5f}")
        eq = f"  ŷ = {b0:.5f}"
        for feat, val in top8:
            eq += f" + ({val:+.5f})·{feat}"
        print(eq + " + ···")
        print(f"  Best model : {best}   R²={all_results[target]['best_r2']:.4f}"
              f"   RMSE={rmse:.5f}")
        print(f"  95% CI (mean) : ±{ci:.5f} {TARGET_UNITS[target]}")
        print(f"  95% PI (obs.) : ±{pi:.5f} {TARGET_UNITS[target]}")
    print("\n" + "="*72)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — COMPUTE PI ON TEST SET + SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
def compute_and_save_ci(all_results):
    print("\n[8/9] Computing 95% prediction intervals on test set ...")
    rows = []
    for target in TARGET_COLS:
        clf    = joblib.load(f"models/best_{target}.pkl")   # always a fitted estimator
        Xte_s  = all_results[target]["Xte_s"]
        yte    = all_results[target]["yte"]
        rmse   = all_results[target]["best_rmse"]
        n_tr   = len(all_results[target]["ytr"])
        p      = Xte_s.shape[1]

        n_samples = min(100, len(yte))
        coverage  = 0
        for i in range(n_samples):
            x_row  = Xte_s[i:i+1]          # (1, p) — already scaled
            actual = float(yte[i])
            res    = predict_with_pi(clf, x_row, rmse, n_tr, p, ALPHA)
            in_pi  = res["PI_lower"] <= actual <= res["PI_upper"]
            lo_s, hi_s = SPEC_LIMITS[target]
            in_spec = lo_s <= res["y_pred"] <= hi_s
            if in_pi: coverage += 1
            rows.append({"Target":target, "Sample":i,
                         "Actual":round(actual,5),
                         "Predicted":res["y_pred"],
                         "PI_lower":res["PI_lower"],
                         "PI_upper":res["PI_upper"],
                         "In_PI":in_pi, "In_Spec":in_spec,
                         "Spec_Lo":lo_s, "Spec_Hi":hi_s})
        print(f"    {target}: 95% PI coverage = {coverage/n_samples*100:.1f}%")

    ci_df = pd.DataFrame(rows)
    ci_df.to_csv("output/predictions_with_CI.csv", index=False)
    print("    Saved: output/predictions_with_CI.csv")
    return ci_df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(all_results, feature_imp, ci_df):
    print("\n[9/9] Generating plots ...")

    # Actual vs Predicted
    fig, axes = plt.subplots(2, 3, figsize=(16,10))
    for i, tgt in enumerate(TARGET_COLS):
        ax  = axes[i//3][i%3]
        yt  = np.array(all_results[tgt][all_results[tgt]["best"]]["y_test"])
        yp  = np.array(all_results[tgt][all_results[tgt]["best"]]["y_pred"])
        r2  = all_results[tgt]["best_r2"]
        ax.scatter(yt, yp, alpha=0.35, s=10, c="steelblue")
        mn, mx = min(yt.min(),yp.min()), max(yt.max(),yp.max())
        ax.plot([mn,mx],[mn,mx],"r--",lw=1.5,label="Ideal")
        ax.set_xlabel(f"Actual ({TARGET_UNITS[tgt]})", fontsize=9)
        ax.set_ylabel(f"Predicted ({TARGET_UNITS[tgt]})", fontsize=9)
        ax.set_title(f"{tgt}  |  {all_results[tgt]['best']}  R²={r2:.3f}",
                     fontweight="bold", fontsize=10)
    plt.suptitle("Actual vs Predicted — Best Model per Target",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/03_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importances
    for tgt in ["HM_SI","HM_MN","HM_S"]:
        if tgt not in feature_imp: continue
        feats = [f[0].replace("_"," ") for f in feature_imp[tgt]]
        imps  = [f[1] for f in feature_imp[tgt]]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(range(len(feats)), imps, color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top Feature Importances — {tgt}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"output/04_feat_imp_{tgt}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # R² bar chart
    model_names = ["Ridge","RandomForest","ExtraTrees","GradBoost","ANN"]
    colors_m    = ["#3498DB","#2ECC71","#E74C3C","#F39C12","#9B59B6"]
    x = np.arange(len(TARGET_COLS)); w = 0.16
    fig, ax = plt.subplots(figsize=(14,6))
    for k, (mn, col) in enumerate(zip(model_names, colors_m)):
        vals = [all_results[tgt].get(mn, {}).get("R2", 0) for tgt in TARGET_COLS]
        ax.bar(x+k*w, vals, w, label=mn, color=col, edgecolor="white")
    ax.set_xticks(x+2*w)
    ax.set_xticklabels([f"{t}\n({TARGET_UNITS[t]})" for t in TARGET_COLS], fontsize=10)
    ax.set_ylabel("R² Score"); ax.set_ylim(0, 0.85)
    ax.set_title("R² — All Models × All Targets", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("output/05_r2_all_models.png", dpi=150, bbox_inches="tight")
    plt.close()

    # PI band plots
    for tgt in ["HM_SI","HM_TEMP"]:
        sub = ci_df[ci_df.Target==tgt].head(50).reset_index(drop=True)
        if len(sub) == 0: continue
        xi  = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(14,5))
        ax.fill_between(xi, sub.PI_lower, sub.PI_upper,
                        alpha=0.25, color="steelblue", label="95% PI")
        ax.plot(xi, sub.Predicted, "b-o", ms=4, lw=1.5, label="Predicted")
        ax.plot(xi, sub.Actual,    "r-s", ms=4, lw=1.5, label="Actual")
        lo_s, hi_s = SPEC_LIMITS[tgt]
        ax.axhline(lo_s, color="green", linestyle="--", lw=1, label="Spec")
        ax.axhline(hi_s, color="green", linestyle="--", lw=1)
        ax.set_xlabel("Test Sample")
        ax.set_ylabel(f"{tgt} ({TARGET_UNITS[tgt]})")
        ax.set_title(f"{tgt} — Predicted vs Actual + 95% PI", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"output/06_PI_{tgt}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Residuals
    fig, axes = plt.subplots(2, 3, figsize=(14,8))
    for i, tgt in enumerate(TARGET_COLS):
        ax  = axes[i//3][i%3]
        yt  = np.array(all_results[tgt][all_results[tgt]["best"]]["y_test"])
        yp  = np.array(all_results[tgt][all_results[tgt]["best"]]["y_pred"])
        res = yt - yp
        ax.hist(res, bins=40, color="coral", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", lw=1.5)
        ax.axvline(np.mean(res), color="red", lw=1.5,
                   label=f"bias={np.mean(res):.4f}")
        ax.set_title(f"Residuals — {tgt}", fontweight="bold")
        ax.set_xlabel(f"Residual ({TARGET_UNITS[tgt]})")
        ax.legend(fontsize=7)
    plt.suptitle("Residual Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/07_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    All plots saved to output/")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — SUMMARY CSV + CV CI TABLE
# ─────────────────────────────────────────────────────────────────────────────
def save_summary(all_results, X_scaled, df):
    print("\n[10] 5-fold CV R² with 95% CI ...")
    print(f"  {'Target':<10}{'Best':>14}{'R²(CV)':>10}{'CI_lo':>10}{'CI_hi':>10}{'σ':>8}")
    print("  "+"-"*56)
    rows = []
    for tgt in TARGET_COLS:
        clf  = joblib.load(f"models/best_{tgt}.pkl")
        y    = df[tgt].values
        cv   = cv_r2_ci(clf, X_scaled, y, cv=5)
        best = all_results[tgt]["best"]
        rows.append({"Target":tgt, "Unit":TARGET_UNITS[tgt], "Best_Model":best,
                     "R2_test":all_results[tgt]["best_r2"],
                     "R2_CV":cv["R2_mean"], "R2_CI_lo":cv["CI_lo"],
                     "R2_CI_hi":cv["CI_hi"], "R2_std":cv["R2_std"],
                     "RMSE":all_results[tgt]["best_rmse"],
                     "MAE":all_results[tgt][best]["MAE"]})
        print(f"  {tgt:<10}{best:>14}{cv['R2_mean']:>10.4f}"
              f"{cv['CI_lo']:>10.4f}{cv['CI_hi']:>10.4f}{cv['R2_std']:>8.4f}")
    pd.DataFrame(rows).to_csv("output/model_summary.csv", index=False)
    print("    Saved: output/model_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — REAL-TIME INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_hm_quality(process_params: dict, burden_8h_ago: dict,
                        eng_feats: list, all_results: dict,
                        alpha: float = 0.05) -> pd.DataFrame:
    """
    Predict all 6 HM quality targets with 95% prediction intervals.
    burden_8h_ago : burden charged ~8 hours before the current tap.
    """
    scaler = joblib.load("models/scaler.pkl")
    c = {**process_params, **burden_8h_ago}
    ti = (c.get("Ore_kg",0)+c.get("Pellet_kg",0)+c.get("Sinter_kg",0)
          +c.get("DRI_kg",0)+c.get("MixedMaterial_kg",0))
    tf  = c.get("Limestone_kg",0)+c.get("Dolomite_kg",0)
    ck  = max(c.get("Coke_kg",1), 1)
    c.update({
        "TotalIron_kg":ti, "TotalFlux_kg":tf,
        "OreCokeRatio":ti/ck, "SinterFrac":c.get("Sinter_kg",0)/max(ti,1),
        "PelletFrac":c.get("Pellet_kg",0)/max(ti,1),
        "OreFrac":(c.get("Ore_kg",0)+c.get("MixedMaterial_kg",0))/max(ti,1),
        "FluxIronRatio":tf/max(ti,1), "TotalCharge_kg":ck+ti+tf,
        "thermal_idx":c.get("HBT",1150)*c.get("Oxygen Flow",8000)/1e6,
        "coal_intensity":c.get("Coal Actual",150)/max(c.get("PROD_RATE",200),1),
        "gas_eff":c.get("ETACO",0.47)*c.get("Permeabilty",1.5),
        "DP_ratio":c.get("Top DP",1.2)/max(c.get("Bottom DP",0.5),0.01),
        "burden_thermal":(ti/ck)*c.get("HBT",1150),
        "flux_thermal":(tf/max(ti,1))*c.get("HBT",1150),
        "pellet_thermal":(c.get("Pellet_kg",0)/max(ti,1))*c.get("HBT",1150),
    })
    row_raw = np.array([[c.get(f, 0.0) for f in eng_feats]])
    row_s   = scaler.transform(row_raw)    # scale exactly once

    out = []
    for tgt in TARGET_COLS:
        clf    = joblib.load(f"models/best_{tgt}.pkl")
        rmse   = all_results[tgt]["best_rmse"]
        n_tr   = len(all_results[tgt]["ytr"])
        p      = len(eng_feats)
        res    = predict_with_pi(clf, row_s, rmse, n_tr, p, alpha)
        lo_s, hi_s = SPEC_LIMITS[tgt]
        in_spec = lo_s <= res["y_pred"] <= hi_s
        out.append({"Target":tgt, "Unit":TARGET_UNITS[tgt],
                    "Predicted":res["y_pred"],
                    "PI_lower":res["PI_lower"], "PI_upper":res["PI_upper"],
                    "Spec":f"{lo_s}–{hi_s}",
                    "Status":"OK" if in_spec else "⚠ ALARM",
                    "Tree_Std":res["std_trees"]})
    return pd.DataFrame(out)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    burden_lag    = load_burden("BURDEN.csv", lag_hours=8)
    hm_h          = load_hm("HM_ANALYSIS.xls")
    bf            = load_para("PARA.xlsx")
    df, eng_feats = build_dataset(bf, burden_lag, hm_h)
    run_eda(df)
    scaler, X_scaled, all_results, feature_imp, ridge_coef = \
        train_all(df, eng_feats)
    print_equations(all_results, ridge_coef)
    ci_df = compute_and_save_ci(all_results)
    make_plots(all_results, feature_imp, ci_df)
    save_summary(all_results, X_scaled, df)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE — check output/ and models/")
    print("="*60)

    print("\n--- Example Real-Time Prediction ---")
    proc = {"HBT":1168,"Oxygen Flow":9400,"Coal Actual":155,"PROD_RATE":210,
             "ETACO":0.472,"Permeabilty":1.54,"Cold Blast Volume":285000,
             "HBP":3.62,"FTP":2.18,"Steam":12.5,"B MOIST":8.2,
             "SLAG_RATE":285,"Radar Level":1.85,"Top DP":1.28,
             "Middle DP":0.94,"Bottom DP":0.47,"Heat Flow Flux":0.38,
             "Heat_Load Q1":1.2,"Heat_Load Q2":1.4,"Heat_Load Q3":1.1,
             "Heat_Load Q4":0.9,"Coal Inj. SP":152}
    burd = {"Coke_kg":155000,"Sinter_kg":392000,"Pellet_kg":240000,
             "Ore_kg":72000,"Limestone_kg":2300,"Dolomite_kg":0,
             "DRI_kg":0,"MixedMaterial_kg":4500}
    result_df = predict_hm_quality(proc, burd, eng_feats, all_results)
    print(result_df.to_string(index=False))
