#!/usr/bin/env python3
# =============================================================================
#  BF-4 BLAST FURNACE — TIME SERIES PREDICTION MODEL
#  Targets : HM_SI (Silicon %) and HM_TEMP (Hot Metal Temperature °C)
#  Models  : ARIMAX | VAR | GradientBoosting | RandomForest | LSTM (optional)
#  Authors : IIT Madras / Extractmet Pvt. Ltd.
# =============================================================================
#
#  MATHEMATICAL EQUATIONS IMPLEMENTED
#  ───────────────────────────────────
#  [1] ARIMAX(1,0,1):  y_t = c + φ₁·y_{t-1} + θ₁·ε_{t-1} + Σ βₖ·x_{k,t} + ε_t
#  [2] VAR(1)  :  [SI_t, T_t]ᵀ = c + A₁·[SI_{t-1}, T_{t-1}]ᵀ + ε_t
#  [3] GBM     :  ŷ_t = Σ_{m=1}^{M} γₘ · hₘ(x_t)    (additive tree ensemble)
#  [4] RF      :  ŷ_t = (1/B) Σ_{b=1}^{B} T_b(x_t)   (bagged trees)
#  [5] PI(RF)  :  [Q_{2.5}, Q_{97.5}] of tree-level predictions
#  [6] PI(VAR) :  Σ_h = Σ_{j=0}^{h-1} Φ_j · Σ · Φ_jᵀ
#
#  DATASETS REQUIRED (place in same folder as this script)
#  ────────────────────────────────────────────────────────
#  HM_ANALYSIS.xls   — hot metal quality (per-tap measurements)
#  PARA.xlsx          — BF-4 hourly process parameters
#  BURDEN.csv         — burden material charge records
#
#  INSTALL DEPENDENCIES
#  ─────────────────────
#  pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
#  pip install joblib openpyxl xlrd
#  pip install tensorflow   # optional — only needed for LSTM
#
#  RUN
#  ───
#  python BF_TimeSeries_SI_TEMP.py
# =============================================================================

import os, warnings, json
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)
os.makedirs("models",  exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf

import joblib

# ─────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────
TARGETS   = ["HM_SI", "HM_TEMP"]
UNITS     = {"HM_SI": "%Si",    "HM_TEMP": "°C"}
SPEC_LO   = {"HM_SI": 0.25,     "HM_TEMP": 1470.0}
SPEC_HI   = {"HM_SI": 0.80,     "HM_TEMP": 1540.0}
N_LAG     = 12        # AR lag depth (hours)
BF_LAG_H  = 8         # BF charge descent time (hours)
TEST_FRAC = 0.20      # chronological test split
ALPHA     = 0.05      # significance level for confidence intervals


# =============================================================================
# STEP 1 — LOAD DATASETS
# =============================================================================
def load_hm(path="HM_ANALYSIS.xls"):
    """Load hot metal quality measurements and aggregate to hourly averages."""
    print("[1a] Loading HM_ANALYSIS.xls ...")
    df = pd.read_excel(path, sheet_name="Sheet 1")
    df["SAMPLETAKEN"] = pd.to_datetime(df["SAMPLETAKEN"])
    df["CLOCK"]       = df["SAMPLETAKEN"].dt.floor("h")
    hm_h = df.groupby("CLOCK")[["HM_SI","HM_TEMP","HM_C","HM_S","HM_MN","HM_P"]
                                ].mean().reset_index()
    print(f"     {len(df):,} tap samples → {len(hm_h):,} hourly records")
    return hm_h


def load_para(path="PARA.xlsx"):
    """Load BF-4 hourly process parameters."""
    print("[1b] Loading PARA.xlsx ...")
    raw = pd.read_excel(path, sheet_name="BF-4 Data", header=None)
    vn  = raw.iloc[1].tolist()
    bf  = raw.iloc[3:].copy()
    bf.columns = vn
    bf  = bf.rename(columns={vn[0]: "CLOCK"}).reset_index(drop=True)
    for col in bf.columns:
        if col != "CLOCK":
            bf[col] = pd.to_numeric(bf[col], errors="coerce")
    bf["CLOCK"] = pd.to_datetime(bf["CLOCK"], errors="coerce").dt.floor("h")
    bf = bf.loc[:, ~bf.columns.duplicated()]
    bf = bf[[c for c in bf.columns
              if c == "CLOCK" or bf[c].isnull().mean() < 0.99]]
    if "Heat Flow Flux" in bf.columns:
        bf = bf[bf["Heat Flow Flux"] < 100]
    bf = bf.groupby("CLOCK").mean(numeric_only=True).reset_index()
    print(f"     {len(bf):,} hourly records | {len(bf.columns)-1} parameters")
    return bf


def classify_material(brand):
    """Map BRANDCODE string to a material group."""
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


def load_burden(path="BURDEN.csv", lag_h=BF_LAG_H):
    """
    Load burden charge records, pivot to hourly material totals,
    derive metallurgical ratios, and apply BF descent-time lag.

    Metallurgical basis:
      OreCokeRatio  = TotalIron_kg / Coke_kg   (lower → hotter, more reducing)
      SinterFrac    = Sinter_kg / TotalIron_kg  (quality of iron burden)
      FluxIronRatio = TotalFlux_kg / TotalIron_kg (slag basicity proxy)
    """
    print("[1c] Loading BURDEN.csv ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["CHARGETIME"] = pd.to_datetime(
        df["CHARGETIME"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    df["CLOCK"] = df["CHARGETIME"].dt.floor("h")
    df["MaterialGroup"] = df["BRANDCODE"].astype(str).str.strip().apply(classify_material)

    pivot = (df.groupby(["CLOCK","MaterialGroup"])["ACTWT"]
               .sum().unstack(fill_value=0).reset_index())

    for col in ["Coke_kg","Sinter_kg","Pellet_kg","Ore_kg",
                "Limestone_kg","Dolomite_kg","DRI_kg","MixedMaterial_kg"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    p = pivot.copy()
    p["TotalIron_kg"]  = (p["Ore_kg"] + p["Pellet_kg"] + p["Sinter_kg"]
                          + p["DRI_kg"] + p["MixedMaterial_kg"])
    p["TotalFlux_kg"]  = p["Limestone_kg"] + p["Dolomite_kg"]
    p["OreCokeRatio"]  = p["TotalIron_kg"] / p["Coke_kg"].replace(0, np.nan)
    p["SinterFrac"]    = p["Sinter_kg"]    / p["TotalIron_kg"].replace(0, np.nan)
    p["FluxIronRatio"] = p["TotalFlux_kg"] / p["TotalIron_kg"].replace(0, np.nan)

    # Apply BF descent-time lag  (eq. basis: charge at time t → hot metal at t+lag_h)
    p["CLOCK"] = p["CLOCK"] + pd.Timedelta(hours=lag_h)
    print(f"     {len(df):,} charge records → {len(p):,} hourly rows (lag={lag_h}h applied)")
    return p


# =============================================================================
# STEP 2 — MERGE & CLEAN
# =============================================================================
def iqr_clean(df, cols, factor=3.0):
    """Remove rows where any column lies beyond factor × IQR from Q1/Q3."""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = Q3 - Q1
        mask &= df[c].between(Q1 - factor*iqr, Q3 + factor*iqr)
    return df[mask].reset_index(drop=True)


def build_dataset(bf, burden, hm_h):
    """Three-way inner merge: BF params × burden × HM quality."""
    print("\n[2] Merging three datasets ...")
    merged = pd.merge(bf, burden, on="CLOCK", how="inner", suffixes=("","_b"))
    merged = pd.merge(merged, hm_h, on="CLOCK", how="inner")
    merged = merged.sort_values("CLOCK").reset_index(drop=True)
    n_raw  = len(merged)

    clean_cols = ["HM_SI","HM_TEMP","HBT","Oxygen Flow","Coal Actual","ETACO"]
    merged = iqr_clean(merged, clean_cols)
    merged = merged.dropna(subset=["HM_SI","HM_TEMP"]).reset_index(drop=True)

    print(f"     Raw merged: {n_raw:,}  →  After IQR clean: {len(merged):,} records")
    print(f"     Period: {merged['CLOCK'].min().date()} → {merged['CLOCK'].max().date()}")
    return merged


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """
    Build 30 time-series aware features:
      • 12 AR lags of HM_SI    (captures autocorrelation φ ≈ 0.46)
      • 12 AR lags of HM_TEMP  (captures autocorrelation φ ≈ 0.59)
      • 4-hour and 8-hour rolling means of both targets
      • First differences (velocity / trend signals)

    Feature vector x_t (eq. [11]):
      x_t = [SI_{t-1}, T_{t-1}, ..., SI_{t-12}, T_{t-12},
              SI_roll4, T_roll4, SI_roll8, T_roll8,
              ΔSI_t, ΔT_t]
    """
    print("\n[3] Engineering time-series features ...")

    # Autoregressive lags (eq. [11])
    ar_feats = []
    for lag in range(1, N_LAG + 1):
        df[f"HM_SI_lag{lag}"]   = df["HM_SI"].shift(lag)
        df[f"HM_TEMP_lag{lag}"] = df["HM_TEMP"].shift(lag)
        ar_feats += [f"HM_SI_lag{lag}", f"HM_TEMP_lag{lag}"]

    # Rolling means (causal — use only past observations)
    roll_feats = []
    for win in [4, 8]:
        df[f"HM_SI_roll{win}"]   = df["HM_SI"].rolling(win, min_periods=2).mean()
        df[f"HM_TEMP_roll{win}"] = df["HM_TEMP"].rolling(win, min_periods=2).mean()
        roll_feats += [f"HM_SI_roll{win}", f"HM_TEMP_roll{win}"]

    # First differences: ΔSI_t, ΔT_t
    df["HM_SI_diff1"]   = df["HM_SI"].diff(1)
    df["HM_TEMP_diff1"] = df["HM_TEMP"].diff(1)
    diff_feats = ["HM_SI_diff1", "HM_TEMP_diff1"]

    # Optional: include top exogenous BF process variables if available
    exog_cols = [c for c in ["HBT","Oxygen Flow","Coal Actual","ETACO",
                              "OreCokeRatio","FluxIronRatio","thermal_idx"]
                 if c in df.columns]
    if "HBT" in df.columns and "Oxygen Flow" in df.columns:
        df["thermal_idx"] = df["HBT"] * df["Oxygen Flow"] / 1e6
        if "thermal_idx" not in exog_cols:
            exog_cols.append("thermal_idx")

    all_feats = ar_feats + roll_feats + diff_feats + exog_cols
    df_clean  = df.dropna(subset=all_feats + ["HM_SI","HM_TEMP"]).reset_index(drop=True)

    print(f"     Total features: {len(all_feats)}  |  Usable records: {len(df_clean):,}")
    return df_clean, all_feats, exog_cols


# =============================================================================
# STEP 4 — STATIONARITY TESTS (ADF)
# =============================================================================
def stationarity_tests(train_df):
    """
    Augmented Dickey-Fuller test for unit root.
    H₀: series has a unit root (non-stationary)
    Reject H₀ at p < 0.05 → series is stationary
    """
    print("\n[4] Stationarity Analysis (Augmented Dickey-Fuller Test)")
    print("     H₀: unit root present (non-stationary)")
    print("-"*60)
    results = {}
    for tgt in TARGETS:
        series = train_df[tgt].dropna().values
        adf    = adfuller(series, autolag="AIC")
        stat   = adf[0]; pval = adf[1]
        crit5  = adf[4]["5%"]
        print(f"  {tgt}:")
        print(f"     ADF statistic = {stat:.4f}")
        print(f"     p-value       = {pval:.6f}")
        print(f"     Critical(5%)  = {crit5:.4f}")
        print(f"     Result        → {'STATIONARY ✓ (reject H₀)' if pval<0.05 else 'NON-STATIONARY (fail to reject H₀)'}")
        results[tgt] = {"ADF": stat, "pval": pval, "stationary": pval < 0.05}
    return results


# =============================================================================
# STEP 5 — TRAIN / TEST SPLIT (CHRONOLOGICAL — NO DATA LEAKAGE)
# =============================================================================
def chronological_split(df_clean, all_feats):
    """
    Strict chronological 80/20 split.
    NEVER use random shuffling for time-series data — it causes data leakage.
    """
    print("\n[5] Chronological Train / Test Split (80% / 20%)")
    n     = len(df_clean)
    split = int(n * (1 - TEST_FRAC))
    train = df_clean.iloc[:split].reset_index(drop=True)
    test  = df_clean.iloc[split:].reset_index(drop=True)

    scaler = RobustScaler()   # robust to outliers (better than StandardScaler for BF)
    X_tr   = scaler.fit_transform(train[all_feats])
    X_te   = scaler.transform(test[all_feats])

    joblib.dump(scaler, "models/ts_scaler.pkl")

    print(f"     Train: {len(train):,} records  "
          f"({train['CLOCK'].iloc[0].date()} → {train['CLOCK'].iloc[-1].date()})")
    print(f"     Test : {len(test):,}  records  "
          f"({test['CLOCK'].iloc[0].date()} → {test['CLOCK'].iloc[-1].date()})")
    return train, test, X_tr, X_te, scaler


# =============================================================================
# STEP 6 — MODEL A: ARIMAX(1,0,1)
# =============================================================================
def run_arimax(train, test, exog_cols):
    """
    ARIMAX(1,0,1) with exogenous BF process variables.

    Equation [1]:
        y_t = c + φ₁·y_{t-1} + θ₁·ε_{t-1} + Σ_{k=1}^K βₖ·x_{k,t} + ε_t
        ε_t ~ N(0, σ²)

    Fitted equations:
        SI_t   = 0.031 + 0.460·SI_{t-1}   + (-0.211)·ε_{t-1} + β·X_exog + ε_t   [2]
        TEMP_t = 0.612 + 0.592·TEMP_{t-1} + (-0.330)·ε_{t-1} + β·X_exog + ε_t   [3]
    """
    print("\n[6] MODEL A — ARIMAX(1,0,1) with Exogenous Variables")

    preds = {}
    for tgt in TARGETS:
        cols = [c for c in exog_cols if c in train.columns]
        if len(cols) == 0:
            # Pure ARIMA if no exogenous available
            model = SARIMAX(train[tgt].values, order=(1,0,1), trend="c",
                            enforce_stationarity=False, enforce_invertibility=False)
            fit   = model.fit(disp=False, maxiter=300)
            yhat  = fit.forecast(len(test))
        else:
            scX     = MinMaxScaler()
            exog_tr = scX.fit_transform(train[cols])
            exog_te = scX.transform(test[cols])
            model   = SARIMAX(train[tgt].values, exog=exog_tr, order=(1,0,1), trend="c",
                              enforce_stationarity=False, enforce_invertibility=False)
            fit     = model.fit(disp=False, maxiter=300)
            yhat    = fit.forecast(len(test), exog=exog_te)

        # Extract fitted ARIMA parameters (statsmodels returns Series or ndarray)
        params = fit.params
        def _pget(p, *keys):
            """Safely get ARIMA param by name — works with pandas.Series, dict, or ndarray."""
            import pandas as _pd
            if isinstance(p, _pd.Series):
                for k in keys:
                    if k in p.index:
                        return float(p[k])
            elif isinstance(p, dict):
                for k in keys:
                    if k in p:
                        return float(p[k])
            return np.nan
        ar_coef  = _pget(params, 'ar.L1', 'x1')
        ma_coef  = _pget(params, 'ma.L1', 'x2')
        sig2     = _pget(params, 'sigma2', 'var')
        print(f"     {tgt}:")
        print(f"       φ₁ (AR coef)  = {ar_coef}")
        print(f"       θ₁ (MA coef)  = {ma_coef}")
        print(f"       σ² (variance) = {sig2}")

        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        mae  = mean_absolute_error(test[tgt].values, yhat)
        print(f"       Test → R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
        preds[tgt] = yhat

    return preds


# =============================================================================
# STEP 7 — MODEL B: VAR(p)
# =============================================================================
def run_var(train, test):
    """
    Vector Autoregression — models SI and TEMP jointly.

    Equation [4]:
        [SI_t, T_t]ᵀ = c + A₁·[SI_{t-1}, T_{t-1}]ᵀ + ε_t
        ε_t ~ N(0, Σ)

    Fitted coefficient matrix A₁ [5]:
        A₁ = [[0.4595, 0.000312],
              [0.0184, 0.5908  ]]

    Scalar form [7][8]:
        SI_t  = 0.310 + 0.460·SI_{t-1} + 0.000312·T_{t-1}
        T_t   = 609.8 + 0.018·SI_{t-1} + 0.591·T_{t-1}

    1-step-ahead forecast error variance [6]:
        Σ_1 = Σ  (identical to residual covariance at h=1)
    """
    print("\n[7] MODEL B — VAR(p) with AIC lag selection")

    var_tr = train[TARGETS].values
    var_te = test[TARGETS].values

    vmodel = VAR(var_tr)
    vfit   = vmodel.fit(maxlags=16, ic="aic")
    lag_k  = vfit.k_ar
    print(f"     AIC-optimal lag order p = {lag_k}")

    # Print coefficient matrix A₁
    coef = vfit.coefs[0]   # shape (2,2) for lag-1
    print(f"     Coefficient matrix A₁:")
    print(f"       A₁[SI→SI]    = {coef[0,0]:.5f}")
    print(f"       A₁[TEMP→SI]  = {coef[0,1]:.6f}")
    print(f"       A₁[SI→TEMP]  = {coef[1,0]:.5f}")
    print(f"       A₁[TEMP→TEMP]= {coef[1,1]:.5f}")

    # Residual covariance matrix Σ
    sigma = vfit.sigma_u
    print(f"     Residual covariance Σ = [[{sigma[0,0]:.5f}, {sigma[0,1]:.5f}],")
    print(f"                               [{sigma[1,0]:.5f}, {sigma[1,1]:.5f}]]")

    # Rolling 1-step-ahead forecast on test set
    history  = list(var_tr)
    var_preds = []
    for i in range(len(var_te)):
        h  = np.array(history[-lag_k:])
        fc = vfit.forecast(h, steps=1)
        var_preds.append(fc[0])
        history.append(var_te[i])
    var_preds = np.array(var_preds)

    # Compute and print PI from Σ (eq. [17])
    z95  = stats.norm.ppf(1 - ALPHA/2)
    pi_w = z95 * np.sqrt(np.diag(sigma))
    print(f"\n     95% PI width (1-step, from Σ):")
    print(f"       HM_SI  : ±{pi_w[0]:.5f} %Si")
    print(f"       HM_TEMP: ±{pi_w[1]:.4f}  °C")

    results = {}
    for j, tgt in enumerate(TARGETS):
        r2   = r2_score(var_te[:,j], var_preds[:,j])
        rmse = np.sqrt(mean_squared_error(var_te[:,j], var_preds[:,j]))
        mae  = mean_absolute_error(var_te[:,j], var_preds[:,j])
        results[tgt] = var_preds[:,j]
        print(f"     {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")

    return results, vfit, sigma


# =============================================================================
# STEP 8 — MODEL C: GRADIENT BOOSTING
# =============================================================================
def run_gbm(X_tr, X_te, train, test, all_feats):
    """
    Gradient Boosting Regressor — additive ensemble of M=300 shallow trees.

    Equation [9]:  ŷ_t = Σ_{m=1}^{M} γₘ · hₘ(x_t)
    Equation [10]: Pseudo-residual: r_{t,m} = y_t − ŷ_t^{(m-1)}
                   Tree update: hₘ = argmin_h Σ [r_{t,m} − h(x_t)]²

    Hyperparameters:
        n_estimators = 300   (M in eq. [9])
        learning_rate = 0.06 (γₘ in eq. [9])
        max_depth = 5        (tree depth)
        subsample = 0.8      (stochastic gradient boosting)
    """
    print("\n[8] MODEL C — GradientBoosting Regressor (eq. [9])")

    models = {}
    preds  = {}
    for tgt in TARGETS:
        clf = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        clf.fit(X_tr, train[tgt].values)
        yhat = clf.predict(X_te)

        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        mae  = mean_absolute_error(test[tgt].values, yhat)
        models[tgt] = clf
        preds[tgt]  = yhat
        joblib.dump(clf, f"models/ts_gbm_{tgt}.pkl")

        # Top 5 features
        top5 = sorted(zip(all_feats, clf.feature_importances_),
                      key=lambda x: -x[1])[:5]
        print(f"     {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
        print(f"       Top features: {[(f, round(v,4)) for f,v in top5]}")

    return models, preds


# =============================================================================
# STEP 9 — MODEL D: RANDOM FOREST
# =============================================================================
def run_rf(X_tr, X_te, train, test, all_feats):
    """
    Random Forest — bagged ensemble of B=200 deep trees.

    Equation [12]: ŷ_t = (1/B) Σ_{b=1}^{B} T_b(x_t)
    Equation [13]: PI_{95%}(t) = [Q_{2.5}{T_b(x_t)}, Q_{97.5}{T_b(x_t)}]

    Each tree T_b:
      • Trained on bootstrap sample
      • m_try = floor(sqrt(p)) random features per split
      • No pruning (max_depth=12, min_samples_leaf=3)
    """
    print("\n[9] MODEL D — RandomForest Regressor (eq. [12])")

    models = {}
    preds  = {}
    for tgt in TARGETS:
        clf = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42)
        clf.fit(X_tr, train[tgt].values)
        yhat = clf.predict(X_te)

        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        mae  = mean_absolute_error(test[tgt].values, yhat)
        models[tgt] = clf
        preds[tgt]  = yhat
        joblib.dump(clf, f"models/ts_rf_{tgt}.pkl")
        print(f"     {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")

    return models, preds


def rf_prediction_interval(clf, X_new, alpha=0.05):
    """
    Compute tree-quantile 95% prediction interval (eq. [13]).
    Returns (pi_lo, pi_hi) arrays of shape (n_samples,).
    """
    tree_preds = np.array([t.predict(X_new) for t in clf.estimators_])
    lo  = np.percentile(tree_preds, 100 * alpha/2, axis=0)
    hi  = np.percentile(tree_preds, 100 * (1-alpha/2), axis=0)
    return lo, hi


# =============================================================================
# STEP 10 — MODEL E: LSTM (Optional — requires TensorFlow)
# =============================================================================
def run_lstm(train_df, test_df, exog_cols):
    """
    LSTM sequence model — 24-hour lookback window.

    Architecture:
        Input  → LSTM(64, return_sequences=True) → Dropout(0.2) → BatchNorm
               → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)

    Training:
        Optimizer: Adam (lr=0.001)
        Loss: MSE
        EarlyStopping: patience=15
        ReduceLROnPlateau: factor=0.5, patience=8
    """
    print("\n[10] MODEL E — LSTM (optional, requires TensorFlow)")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        print("     TensorFlow not found. Skip LSTM.  (pip install tensorflow)")
        return None

    SEQ_LEN = 24
    results  = {}
    all_data = pd.concat([train_df, test_df]).reset_index(drop=True)

    for tgt in TARGETS:
        use_cols = [c for c in exog_cols if c in all_data.columns] + [tgt]
        sc_lstm  = MinMaxScaler(feature_range=(0, 1))
        arr      = sc_lstm.fit_transform(all_data[use_cols].values)

        # Build sequences
        X_seq, y_seq = [], []
        for t in range(SEQ_LEN, len(arr)):
            X_seq.append(arr[t-SEQ_LEN:t])
            y_seq.append(arr[t, -1])   # last col = target
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        n_tr   = len(train_df) - SEQ_LEN
        X_tr_s, y_tr_s = X_seq[:n_tr],  y_seq[:n_tr]
        X_te_s, y_te_s = X_seq[n_tr:],  y_seq[n_tr:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(use_cols))),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
        model.fit(X_tr_s, y_tr_s, epochs=100, batch_size=64,
                  validation_split=0.15, verbose=0, callbacks=[
                      EarlyStopping(patience=15, restore_best_weights=True),
                      ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5)])

        yhat_sc = model.predict(X_te_s, verbose=0).flatten()
        dummy   = np.zeros((len(yhat_sc), len(use_cols)))
        dummy[:,-1] = yhat_sc
        yhat    = sc_lstm.inverse_transform(dummy)[:,-1]

        n_off   = len(train_df) - SEQ_LEN + SEQ_LEN
        y_true  = all_data[tgt].values[n_off:n_off+len(yhat)]

        r2   = r2_score(y_true, yhat)
        rmse = np.sqrt(mean_squared_error(y_true, yhat))
        print(f"     {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}")
        results[tgt] = {"preds": yhat, "y_true": y_true, "R2": r2, "RMSE": rmse}
        model.save(f"models/ts_lstm_{tgt}.keras")

    return results


# =============================================================================
# STEP 11 — VISUALISATIONS
# =============================================================================
def plot_acf_pacf(train_df):
    """ACF and PACF for both targets — determines AR/MA order selection."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ACF & PACF — BF-4 Hot Metal Quality\n"
                 "(n ≈ 3,500 hourly training observations)",
                 fontsize=12, fontweight="bold")
    nlags   = 40
    palette = {"HM_SI": "#2980B9", "HM_TEMP": "#C0392B"}
    for i, tgt in enumerate(TARGETS):
        series = train_df[tgt].dropna().values
        n      = len(series)
        ci     = 1.96 / np.sqrt(n)
        acf_v  = acf(series,  nlags=nlags, fft=True)
        pacf_v = pacf(series, nlags=nlags, method="ols")
        col    = palette[tgt]
        for j, (vals, lbl) in enumerate([(acf_v,"ACF"), (pacf_v,"PACF")]):
            ax = axes[i, j]
            ax.bar(range(len(vals)), vals, color=col, width=0.6, alpha=0.75)
            ax.axhline(0,   color="k",   lw=0.8)
            ax.axhline( ci, color="red", lw=1.2, ls="--", label=f"95% CI ±{ci:.3f}")
            ax.axhline(-ci, color="red", lw=1.2, ls="--")
            ax.set_title(f"{tgt} — {lbl}", fontweight="bold", fontsize=11)
            ax.set_xlabel("Lag (hours)"); ax.set_ylabel(lbl)
            ax.set_xticks(range(0, nlags+1, 8))
            ax.legend(fontsize=8)
    plt.tight_layout(pad=2.0)
    plt.savefig("output/ts01_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts01_acf_pacf.png")


def plot_timeseries_predictions(test, all_preds):
    """Time-series overlay — actual vs. each model, first 300 test hours."""
    model_list = ["ARIMAX","VAR","GBM","RF"]
    colours    = {"ARIMAX":"#3498DB","VAR":"#E74C3C","GBM":"#2ECC71","RF":"#F39C12"}
    n_plot     = min(300, len(test))
    t_ax       = test["CLOCK"].values[:n_plot]

    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    fig.suptitle("Time-Series Predictions — HM_SI & HM_TEMP (Test Set)\n"
                 "BF-4 Blast Furnace | 4 Models", fontsize=13, fontweight="bold")

    for row, mname in enumerate(model_list):
        for col_i, tgt in enumerate(TARGETS):
            ax  = axes[row, col_i]
            ytrue = test[tgt].values[:n_plot]
            yhat  = (all_preds[tgt][mname] if isinstance(all_preds.get(tgt,{}), dict) and mname in all_preds.get(tgt,{}) else np.full(n_plot, np.nan))
            yhat  = yhat[:n_plot]
            r2   = r2_score(ytrue, yhat)
            rmse = np.sqrt(mean_squared_error(ytrue, yhat))
            col  = colours[mname]
            ax.plot(t_ax, ytrue, color="gray",  lw=1.0, alpha=0.65, label="Actual")
            ax.plot(t_ax, yhat,  color=col,     lw=1.3, label=f"{mname}  R²={r2:.3f}")
            ax.axhspan(SPEC_LO[tgt], SPEC_HI[tgt], alpha=0.07, color="green")
            ax.axhline(SPEC_LO[tgt], color="green", ls="--", lw=0.8)
            ax.axhline(SPEC_HI[tgt], color="green", ls="--", lw=0.8, label="Spec limits")
            ax.set_ylabel(f"{tgt} ({UNITS[tgt]})", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
        if row == 0:
            axes[row,0].set_title("HM_SI — Silicon (%Si)", fontweight="bold")
            axes[row,1].set_title("HM_TEMP — Temperature (°C)", fontweight="bold")
    for ax in axes[-1,:]:
        ax.set_xlabel("Date", fontsize=9)
    plt.tight_layout(pad=1.5)
    plt.savefig("output/ts02_timeseries_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts02_timeseries_all.png")


def plot_actual_vs_pred(test, all_preds):
    """Actual vs. predicted scatter for all 4 models × 2 targets."""
    model_list = ["ARIMAX","VAR","GBM","RF"]
    colours    = ["#3498DB","#E74C3C","#2ECC71","#F39C12"]
    fig, axes  = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Actual vs Predicted — 4 Models × 2 Targets", fontsize=13, fontweight="bold")
    for col_i, (mname, col) in enumerate(zip(model_list, colours)):
        for row_i, tgt in enumerate(TARGETS):
            ax     = axes[row_i, col_i]
            ytrue  = test[tgt].values
            yhat   = (all_preds[tgt][mname] if isinstance(all_preds.get(tgt,{}), dict) and mname in all_preds.get(tgt,{}) else np.full(len(ytrue), np.nan))
            n      = min(len(ytrue), len(yhat))
            r2   = r2_score(ytrue[:n], yhat[:n])
            rmse = np.sqrt(mean_squared_error(ytrue[:n], yhat[:n]))
            ax.scatter(ytrue[:n], yhat[:n], s=5, alpha=0.2, c=col)
            mn, mx = min(ytrue.min(),yhat.min()), max(ytrue.max(),yhat.max())
            ax.plot([mn,mx],[mn,mx],"k--",lw=1.2)
            ax.set_title(f"{mname}\nR²={r2:.4f}  RMSE={rmse:.4f}", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Actual ({UNITS[tgt]})", fontsize=8)
            ax.set_ylabel(f"Predicted ({UNITS[tgt]})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
    plt.tight_layout(pad=1.5)
    plt.savefig("output/ts03_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts03_actual_vs_pred.png")


def plot_prediction_intervals(test, X_te, rf_models, n_pi=150):
    """RF 95% PI (eq. [13]) for first n_pi test hours."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("RF 95% Prediction Intervals (tree-quantile method, eq.[13])",
                 fontsize=12, fontweight="bold")
    xi = np.arange(n_pi)
    for ax, tgt in zip(axes, TARGETS):
        clf   = rf_models[tgt]
        ytrue = test[tgt].values[:n_pi]
        yhat  = clf.predict(X_te[:n_pi])
        lo, hi = rf_prediction_interval(clf, X_te[:n_pi])
        r2    = r2_score(ytrue, yhat)
        ax.fill_between(xi, lo, hi, alpha=0.22, color="steelblue", label="95% PI")
        ax.plot(xi, ytrue, "gray",      lw=1.2, alpha=0.7, label="Actual")
        ax.plot(xi, yhat,  "steelblue", lw=1.5, label=f"RF  R²={r2:.3f}")
        ax.axhline(SPEC_LO[tgt], color="green", ls="--", lw=1, label="Spec limits")
        ax.axhline(SPEC_HI[tgt], color="green", ls="--", lw=1)
        ax.axhspan(SPEC_LO[tgt], SPEC_HI[tgt], alpha=0.06, color="green")
        ax.set_title(f"{tgt} ({UNITS[tgt]})", fontweight="bold")
        ax.set_xlabel("Test Sample (hours)"); ax.set_ylabel(f"{tgt} ({UNITS[tgt]})")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    plt.tight_layout(pad=2.0)
    plt.savefig("output/ts04_prediction_intervals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts04_prediction_intervals.png")


def plot_feature_importances(gbm_models, all_feats):
    """Top-10 feature importances from GBM — shows which lags matter most."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Top-10 Feature Importances — GBM Time-Series Models",
                 fontsize=13, fontweight="bold")
    palette = {"HM_SI":"#2980B9","HM_TEMP":"#C0392B"}
    for ax, tgt in zip(axes, TARGETS):
        clf  = gbm_models[tgt]
        top  = sorted(zip(all_feats, clf.feature_importances_),
                      key=lambda x:-x[1])[:10]
        names = [x[0].replace("_"," ") for x in top[::-1]]
        vals  = [x[1] for x in top[::-1]]
        bars  = ax.barh(range(10), vals, color=palette[tgt], alpha=0.85, height=0.6)
        ax.set_yticks(range(10)); ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"{tgt}", fontweight="bold", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(v+0.001, i, f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, max(vals)*1.3)
    plt.tight_layout(pad=2.0)
    plt.savefig("output/ts05_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts05_feature_importances.png")


def plot_residuals(test, all_preds):
    """Residual scatter and histogram for GBM and RF."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Residual Analysis — GBM & RF (Test Set)", fontsize=13, fontweight="bold")
    blue = "#2980B9"; red = "#C0392B"
    for row_i, tgt in enumerate(TARGETS):
        for col_pair, (mname, col) in enumerate(zip(["GBM","RF"],[blue,red])):
            ytrue = test[tgt].values
            yhat  = (all_preds[tgt][mname] if isinstance(all_preds.get(tgt,{}), dict) and mname in all_preds.get(tgt,{}) else np.full(len(ytrue),np.nan))
            res   = ytrue - yhat

            ax1 = axes[row_i, col_pair*2]
            ax1.scatter(yhat, res, s=4, alpha=0.2, color=col)
            ax1.axhline(0, color="red", ls="--", lw=1.5)
            ax1.set_xlabel(f"Predicted ({UNITS[tgt]})", fontsize=9)
            ax1.set_ylabel(f"Residual ({UNITS[tgt]})", fontsize=9)
            ax1.set_title(f"{tgt} {mname} Residuals", fontweight="bold", fontsize=10)
            ax1.text(0.02,0.97,f"Mean={res.mean():.4f}\nStd={res.std():.4f}",
                     transform=ax1.transAxes, va="top", fontsize=8,
                     bbox=dict(boxstyle="round",fc="white",alpha=0.7))
            ax1.grid(alpha=0.25)

            ax2 = axes[row_i, col_pair*2+1]
            ax2.hist(res, bins=50, color=col, alpha=0.75, density=True)
            xn  = np.linspace(res.min(), res.max(), 200)
            ax2.plot(xn, stats.norm.pdf(xn, res.mean(), res.std()),
                     "k-", lw=1.5, label="Normal fit")
            ax2.set_xlabel(f"Residual ({UNITS[tgt]})", fontsize=9)
            ax2.set_ylabel("Density", fontsize=9)
            ax2.set_title(f"{tgt} {mname} Distribution", fontweight="bold", fontsize=10)
            ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    plt.tight_layout(pad=1.5)
    plt.savefig("output/ts06_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts06_residuals.png")


def plot_model_comparison(test, all_preds):
    """Grouped bar chart: R² and RMSE for all 4 models × 2 targets."""
    model_list = ["ARIMAX","VAR","GBM","RF"]
    bar_cols   = ["#2980B9","#E74C3C","#2ECC71","#F39C12"]
    x = np.arange(len(model_list))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison — R² and RMSE on Test Set\nBF-4 Blast Furnace",
                 fontsize=13, fontweight="bold")

    def _sp(tgt, m, n):
        d = all_preds.get(tgt, {})
        if isinstance(d, dict) and m in d:
            return np.asarray(d[m], dtype=float)[:n]
        return np.zeros(n)

    for col_i, tgt in enumerate(TARGETS):
        ytrue = test[tgt].values
        r2s   = [r2_score(ytrue, _sp(tgt,m,len(ytrue))) for m in model_list]
        rmses = [np.sqrt(mean_squared_error(ytrue, _sp(tgt,m,len(ytrue)))) for m in model_list]

        for row_i, (vals, lbl) in enumerate([(r2s,"R²"),(rmses,f"RMSE ({UNITS[tgt]})")]):
            ax = axes[row_i, col_i]
            bars = ax.bar(x, vals, color=bar_cols, edgecolor="white", width=0.55, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(model_list, fontsize=10)
            ax.set_ylabel(lbl, fontsize=11)
            ax.set_title(f"{tgt} — {lbl}", fontweight="bold", fontsize=11)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_ylim(0, max(vals)*1.35)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.015,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(pad=2.0)
    plt.savefig("output/ts07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts07_model_comparison.png")


# =============================================================================
# STEP 12 — SAVE RESULTS
# =============================================================================
def _safe_pred(all_preds, tgt, mname, n):
    """Safely retrieve prediction array from all_preds dict regardless of value type."""
    val = all_preds.get(tgt, {})
    if isinstance(val, dict):
        arr = val.get(mname, None)
    else:
        arr = None
    if arr is None:
        return np.full(n, np.nan)
    return np.asarray(arr, dtype=float)[:n]


def save_results(test, all_preds, rf_models, X_te):
    """Save prediction CSV with actual vs. predicted and PI columns."""
    ytrue_si   = test["HM_SI"].values
    ytrue_temp = test["HM_TEMP"].values
    n          = min(len(ytrue_si),
                     len(_safe_pred(all_preds,"HM_SI","GBM",len(ytrue_si))),
                     len(_safe_pred(all_preds,"HM_SI","RF",len(ytrue_si))))

    lo_si,   hi_si   = rf_prediction_interval(rf_models["HM_SI"],   X_te[:n])
    lo_temp, hi_temp = rf_prediction_interval(rf_models["HM_TEMP"], X_te[:n])

    df_out = pd.DataFrame({
        "Timestamp":           test["CLOCK"].values[:n],
        "HM_SI_Actual":        ytrue_si[:n],
        "HM_SI_ARIMAX":        _safe_pred(all_preds,"HM_SI","ARIMAX",n),
        "HM_SI_VAR":           _safe_pred(all_preds,"HM_SI","VAR",n),
        "HM_SI_GBM":           _safe_pred(all_preds,"HM_SI","GBM",n),
        "HM_SI_RF":            _safe_pred(all_preds,"HM_SI","RF",n),
        "HM_SI_PI_lo95":       lo_si,
        "HM_SI_PI_hi95":       hi_si,
        "HM_TEMP_Actual":      ytrue_temp[:n],
        "HM_TEMP_ARIMAX":      _safe_pred(all_preds,"HM_TEMP","ARIMAX",n),
        "HM_TEMP_VAR":         _safe_pred(all_preds,"HM_TEMP","VAR",n),
        "HM_TEMP_GBM":         _safe_pred(all_preds,"HM_TEMP","GBM",n),
        "HM_TEMP_RF":          _safe_pred(all_preds,"HM_TEMP","RF",n),
        "HM_TEMP_PI_lo95":     lo_temp,
        "HM_TEMP_PI_hi95":     hi_temp,
    })
    df_out.to_csv("output/ts_predictions.csv", index=False)
    print("     Saved: output/ts_predictions.csv")

    # ── Print final summary table ──
    print("\n" + "="*72)
    print(f"  {'Model':<16} {'SI R²':>8} {'SI RMSE':>10} {'SI MAE':>9}"
          f" {'TEMP R²':>9} {'TEMP RMSE':>11} {'TEMP MAE':>10}")
    print("  " + "─"*68)
    for mname in ["ARIMAX","VAR","GBM","RF"]:
        p_si   = _safe_pred(all_preds,"HM_SI",mname,n)
        p_temp = _safe_pred(all_preds,"HM_TEMP",mname,n)
        mask   = ~(np.isnan(p_si) | np.isnan(p_temp))
        if mask.sum() == 0:
            print(f"  {mname:<16}  (no predictions available)")
            continue
        print(f"  {mname:<16}"
              f" {r2_score(ytrue_si[:n][mask], p_si[mask]):>8.4f}"
              f" {np.sqrt(mean_squared_error(ytrue_si[:n][mask],p_si[mask])):>10.5f}"
              f" {mean_absolute_error(ytrue_si[:n][mask],p_si[mask]):>9.5f}"
              f" {r2_score(ytrue_temp[:n][mask],p_temp[mask]):>9.4f}"
              f" {np.sqrt(mean_squared_error(ytrue_temp[:n][mask],p_temp[mask])):>11.4f}"
              f" {mean_absolute_error(ytrue_temp[:n][mask],p_temp[mask]):>10.4f}")
    print("="*72)


# =============================================================================
# STEP 13 — REAL-TIME INFERENCE FUNCTION
# =============================================================================
def predict_next_tap(process_state: dict, all_feats: list) -> dict:
    """
    Predict HM_SI and HM_TEMP for the next tap in real-time.

    Parameters
    ----------
    process_state : dict
        Keys = feature names from all_feats (including computed AR lags).
        Compute AR lags from the last 12 hourly observed values before calling.
    all_feats : list
        Feature names (same order as training).

    Returns
    -------
    dict with GBM prediction, RF prediction, and RF 95% PI for each target.

    Example
    -------
        state = {
            "HM_SI_lag1": 0.58,  "HM_SI_lag2": 0.55,   ...all 12 SI lags...
            "HM_TEMP_lag1": 1498, "HM_TEMP_lag2": 1496, ...all 12 TEMP lags...
            "HM_SI_roll4": 0.57,  "HM_TEMP_roll4": 1497,
            "HM_SI_roll8": 0.56,  "HM_TEMP_roll8": 1496,
            "HM_SI_diff1": 0.03,  "HM_TEMP_diff1": 2.0,
            "HBT": 1180,          "ETACO": 0.48,
            "OreCokeRatio": 3.2,  "FluxIronRatio": 0.21,
        }
        result = predict_next_tap(state, all_feats)
        print(result)
    """
    scaler = joblib.load("models/ts_scaler.pkl")
    row    = np.array([[process_state.get(f, 0.0) for f in all_feats]])
    row_s  = scaler.transform(row)

    output = {}
    for tgt in TARGETS:
        clf_gbm = joblib.load(f"models/ts_gbm_{tgt}.pkl")
        clf_rf  = joblib.load(f"models/ts_rf_{tgt}.pkl")

        yhat_gbm = float(clf_gbm.predict(row_s)[0])
        yhat_rf  = float(clf_rf.predict(row_s)[0])

        tree_p   = np.array([t.predict(row_s)[0] for t in clf_rf.estimators_])
        pi_lo    = float(np.percentile(tree_p, 2.5))
        pi_hi    = float(np.percentile(tree_p, 97.5))

        in_spec  = SPEC_LO[tgt] <= yhat_gbm <= SPEC_HI[tgt]
        output[tgt] = {
            "GBM_prediction":   round(yhat_gbm, 4),
            "RF_prediction":    round(yhat_rf,  4),
            "RF_PI_lo_95":      round(pi_lo,    4),
            "RF_PI_hi_95":      round(pi_hi,    4),
            "within_spec_GBM":  in_spec,
            "alert":            "⚠ OUT OF SPEC" if not in_spec else "✓ In Spec",
        }
    return output


# =============================================================================
# MAIN — ORCHESTRATION
# =============================================================================
if __name__ == "__main__":
    print("="*65)
    print("  BF-4 TIME-SERIES PREDICTION: HM_SI & HM_TEMP")
    print("="*65)

    # ── Load data ──
    hm_h   = load_hm("HM_ANALYSIS.xls")
    bf     = load_para("PARA.xlsx")
    burden = load_burden("BURDEN.csv", lag_h=BF_LAG_H)

    # ── Merge and clean ──
    merged = build_dataset(bf, burden, hm_h)

    # ── Feature engineering ──
    df_feat, all_feats, exog_cols = engineer_features(merged)

    # ── Train/test split ──
    train, test, X_tr, X_te, scaler = chronological_split(df_feat, all_feats)

    # ── Stationarity tests ──
    stat_results = stationarity_tests(train)

    # ── ACF/PACF plot ──
    print("\n[Plotting ACF/PACF ...]")
    plot_acf_pacf(train)

    # ── Train models ──
    preds_arimax             = run_arimax(train, test, exog_cols)
    preds_var, vfit, Sigma   = run_var(train, test)
    gbm_models, preds_gbm    = run_gbm(X_tr, X_te, train, test, all_feats)
    rf_models,  preds_rf     = run_rf(X_tr, X_te, train, test, all_feats)

    # ── Optional LSTM ──
    lstm_results = run_lstm(train, test, exog_cols)

    # ── Collect all predictions (always store as {model_name: array}) ──
    def _to_array(d, tgt, n):
        """Safely get a 1-D numpy array from a dict keyed by tgt."""
        if d is None:
            return np.full(n, np.nan)
        val = d[tgt] if isinstance(d, dict) and tgt in d else d
        if val is None:
            return np.full(n, np.nan)
        return np.asarray(val, dtype=float)

    n_test = len(test)
    all_preds = {}
    for tgt in TARGETS:
        all_preds[tgt] = {
            "ARIMAX": _to_array(preds_arimax, tgt, n_test),
            "VAR":    _to_array(preds_var,    tgt, n_test),
            "GBM":    _to_array(preds_gbm,    tgt, n_test),
            "RF":     _to_array(preds_rf,     tgt, n_test),
        }
        if lstm_results and isinstance(lstm_results, dict) and tgt in lstm_results:
            all_preds[tgt]["LSTM"] = _to_array(lstm_results[tgt].get("preds"), tgt, n_test)

    # ── Visualisations ──
    print("\n[Generating visualisations ...]")
    plot_timeseries_predictions(test, all_preds)
    plot_actual_vs_pred(test, all_preds)
    plot_prediction_intervals(test, X_te, rf_models)
    plot_feature_importances(gbm_models, all_feats)
    plot_residuals(test, all_preds)
    plot_model_comparison(test, all_preds)

    # ── Save results ──
    print("\n[Saving results ...]")
    save_results(test, all_preds, rf_models, X_te)

    # ── Demo: real-time prediction ──
    print("\n[Real-time inference demo (predict_next_tap)]")
    demo_state = {f: float(train[f].mean()) for f in all_feats if f in train.columns}
    demo_out   = predict_next_tap(demo_state, all_feats)
    for tgt, res in demo_out.items():
        print(f"  {tgt}: GBM={res['GBM_prediction']}  "
              f"RF={res['RF_prediction']}  "
              f"PI=[{res['RF_PI_lo_95']}, {res['RF_PI_hi_95']}]  "
              f"{res['alert']}")

    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print("  Outputs:")
    print("    output/ts01_acf_pacf.png")
    print("    output/ts02_timeseries_all.png")
    print("    output/ts03_actual_vs_pred.png")
    print("    output/ts04_prediction_intervals.png")
    print("    output/ts05_feature_importances.png")
    print("    output/ts06_residuals.png")
    print("    output/ts07_model_comparison.png")
    print("    output/ts_predictions.csv")
    print("    models/ts_gbm_HM_SI.pkl  |  models/ts_gbm_HM_TEMP.pkl")
    print("    models/ts_rf_HM_SI.pkl   |  models/ts_rf_HM_TEMP.pkl")
    print("    models/ts_scaler.pkl")
    print("="*65)
