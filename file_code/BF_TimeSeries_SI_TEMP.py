#!/usr/bin/env python3
"""
=============================================================================
BF-4 BLAST FURNACE — TIME SERIES PREDICTION MODEL
Targets: HM_SI (Silicon) and HM_TEMP (Hot Metal Temperature)
=============================================================================
Models implemented:
  1. ARIMAX(1,0,1)      — ARIMA with exogenous BF process variables
  2. VAR(p)             — Vector Autoregression (joint SI + TEMP)
  3. GradientBoosting   — Best tabular time-series ML model
  4. RandomForest       — Ensemble with AR lag features
  5. LSTM               — Deep learning sequence model (TensorFlow/Keras)

Datasets Required (same folder):
  PARA.xlsx  |  HM_ANALYSIS.xls  |  BURDEN.csv

Install:
  pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels joblib openpyxl xlrd
  pip install tensorflow  # optional, for LSTM

Run:
  python BF_TimeSeries_SI_TEMP.py
=============================================================================
"""

import os, warnings, json, copy
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
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
import joblib

TARGETS = ["HM_SI", "HM_TEMP"]
UNITS   = {"HM_SI": "%Si", "HM_TEMP": "°C"}
SPEC    = {"HM_SI": (0.25, 0.80), "HM_TEMP": (1470, 1540)}
ALPHA   = 0.05

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════
def load_hm(path="HM_ANALYSIS.xls"):
    print("[1a] Loading HM_ANALYSIS.xls ...")
    df = pd.read_excel(path, sheet_name="Sheet 1")
    df["SAMPLETAKEN"] = pd.to_datetime(df["SAMPLETAKEN"])
    df["CLOCK"]       = df["SAMPLETAKEN"].dt.floor("h")
    hm_h = df.groupby("CLOCK")[["HM_SI","HM_TEMP"]].mean().reset_index()
    print(f"     {len(df):,} taps → {len(hm_h):,} hourly averages")
    return hm_h

def load_para(path="PARA.xlsx"):
    print("[1b] Loading PARA.xlsx ...")
    raw = pd.read_excel(path, sheet_name="BF-4 Data", header=None)
    vn  = raw.iloc[1].tolist()
    bf  = raw.iloc[3:].copy(); bf.columns = vn
    bf  = bf.rename(columns={vn[0]:"CLOCK"}).reset_index(drop=True)
    for col in bf.columns:
        if col != "CLOCK": bf[col] = pd.to_numeric(bf[col], errors="coerce")
    bf["CLOCK"] = pd.to_datetime(bf["CLOCK"], errors="coerce").dt.floor("h")
    bf = bf.loc[:, ~bf.columns.duplicated()]
    bf = bf[[c for c in bf.columns if c=="CLOCK" or bf[c].isnull().mean()<0.99]]
    if "Heat Flow Flux" in bf.columns: bf = bf[bf["Heat Flow Flux"]<100]
    bf = bf.groupby("CLOCK").mean(numeric_only=True).reset_index()
    print(f"     {len(bf):,} hourly records, {len(bf.columns)-1} parameters")
    return bf

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

def load_burden(path="BURDEN.csv", lag_h=8):
    print("[1c] Loading BURDEN.csv ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["CHARGETIME"]    = pd.to_datetime(df["CHARGETIME"].astype(str).str.strip(),
                                          dayfirst=True, errors="coerce")
    df["CLOCK"]         = df["CHARGETIME"].dt.floor("h")
    df["MaterialGroup"] = df["BRANDCODE"].astype(str).str.strip().apply(classify_material)
    pivot = df.groupby(["CLOCK","MaterialGroup"])["ACTWT"].sum().unstack(fill_value=0).reset_index()
    for col in ["Coke_kg","Sinter_kg","Pellet_kg","Ore_kg","Limestone_kg",
                "Dolomite_kg","DRI_kg","MixedMaterial_kg"]:
        if col not in pivot.columns: pivot[col] = 0.0
    p = pivot.copy()
    p["TotalIron_kg"]  = (p["Ore_kg"]+p["Pellet_kg"]+p["Sinter_kg"]
                          +p["DRI_kg"]+p["MixedMaterial_kg"])
    p["TotalFlux_kg"]  = p["Limestone_kg"]+p["Dolomite_kg"]
    p["OreCokeRatio"]  = p["TotalIron_kg"]/p["Coke_kg"].replace(0,np.nan)
    p["SinterFrac"]    = p["Sinter_kg"]    /p["TotalIron_kg"].replace(0,np.nan)
    p["FluxIronRatio"] = p["TotalFlux_kg"] /p["TotalIron_kg"].replace(0,np.nan)
    # Apply BF descent-time lag
    p["CLOCK"] = p["CLOCK"] + pd.Timedelta(hours=lag_h)
    print(f"     {len(df):,} charge records → {len(p):,} hourly (lag={lag_h}h)")
    return p

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — MERGE + CLEAN
# ═══════════════════════════════════════════════════════════════════════
def iqr_clean(df, cols, factor=3.0):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns: continue
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        mask &= df[c].between(Q1-factor*(Q3-Q1), Q3+factor*(Q3-Q1))
    return df[mask].reset_index(drop=True)

def build_merged(bf, burden, hm_h):
    print("\n[2] Merging datasets ...")
    merged = pd.merge(bf, burden, on="CLOCK", how="inner", suffixes=("","_b"))
    merged = pd.merge(merged, hm_h, on="CLOCK", how="inner")
    merged = merged.sort_values("CLOCK").reset_index(drop=True)
    merged = iqr_clean(merged, ["HM_SI","HM_TEMP","HBT","Oxygen Flow","Coal Actual","ETACO"])
    merged = merged.dropna(subset=["HM_SI","HM_TEMP"]).reset_index(drop=True)
    print(f"     Merged: {len(merged):,} records  "
          f"({merged['CLOCK'].min().date()} → {merged['CLOCK'].max().date()})")
    return merged

# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING (time-series aware)
# ═══════════════════════════════════════════════════════════════════════
def engineer_features(df):
    print("\n[3] Feature engineering ...")

    # Base process features
    base = [c for c in ["HBT","Oxygen Flow","Coal Actual","ETACO","Permeabilty",
                         "PROD_RATE","SLAG_RATE","Top DP","Bottom DP","Coal Inj. SP",
                         "Coke_kg","Sinter_kg","OreCokeRatio","SinterFrac",
                         "FluxIronRatio","TotalIron_kg","TotalFlux_kg"]
             if c in df.columns]

    # Interaction / metallurgical derived features
    df["thermal_idx"]    = df["HBT"] * df["Oxygen Flow"] / 1e6
    df["burden_thermal"] = df["OreCokeRatio"] * df["HBT"]
    df["flux_thermal"]   = df["FluxIronRatio"] * df["HBT"]
    derived = ["thermal_idx","burden_thermal","flux_thermal"]

    # Rolling statistics (causal — use past data only)
    roll_feats = []
    for col in ["HBT","ETACO","OreCokeRatio","Coal Actual","Oxygen Flow"]:
        if col not in df.columns: continue
        for win in [4, 8]:
            cn = f"{col}_r{win}m"
            df[cn] = df[col].rolling(win, min_periods=2).mean()
            roll_feats.append(cn)
        cn_std = f"{col}_r4s"
        df[cn_std] = df[col].rolling(4, min_periods=2).std().fillna(0)
        roll_feats.append(cn_std)

    # Autoregressive lags of HM_SI and HM_TEMP (key for TS models)
    ar_si_feats, ar_temp_feats = [], []
    for lag in [1,2,3,6,8,12,24]:
        df[f"HM_SI_lag{lag}"]   = df["HM_SI"].shift(lag)
        df[f"HM_TEMP_lag{lag}"] = df["HM_TEMP"].shift(lag)
        ar_si_feats.append(f"HM_SI_lag{lag}")
        ar_temp_feats.append(f"HM_TEMP_lag{lag}")

    # First-difference (trend / velocity)
    df["HM_SI_d1"]   = df["HM_SI"].diff(1)
    df["HM_TEMP_d1"] = df["HM_TEMP"].diff(1)
    df["HBT_d1"]     = df["HBT"].diff(1) if "HBT" in df.columns else 0
    diff_feats = ["HM_SI_d1","HM_TEMP_d1","HBT_d1"]

    all_feats = base + derived + roll_feats + ar_si_feats + ar_temp_feats + diff_feats
    df_clean  = df.dropna(subset=all_feats+["HM_SI","HM_TEMP"]).reset_index(drop=True)
    print(f"     {len(df_clean):,} usable records  |  {len(all_feats)} features")
    return df_clean, all_feats

# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN / TEST SPLIT (CHRONOLOGICAL)
# ═══════════════════════════════════════════════════════════════════════
def split_data(df_clean, all_feats, test_frac=0.20):
    print("\n[4] Chronological train/test split ...")
    n     = len(df_clean)
    split = int(n * (1 - test_frac))
    train = df_clean.iloc[:split].reset_index(drop=True)
    test  = df_clean.iloc[split:].reset_index(drop=True)

    scaler = RobustScaler()
    X_tr   = scaler.fit_transform(train[all_feats])
    X_te   = scaler.transform(test[all_feats])

    joblib.dump(scaler, "models/ts_scaler.pkl")
    print(f"     Train: {len(train):,}  Test: {len(test):,}")
    print(f"     Test  : {test['CLOCK'].iloc[0].date()} → {test['CLOCK'].iloc[-1].date()}")
    return train, test, X_tr, X_te, scaler

# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — STATIONARITY & ACF/PACF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def stationarity_analysis(train, test):
    print("\n[5] Stationarity & ACF/PACF analysis ...")
    results = {}
    for tgt in TARGETS:
        adf = adfuller(train[tgt].dropna())
        print(f"     {tgt}: ADF={adf[0]:.4f}  p={adf[1]:.5f}  "
              f"→ {'STATIONARY' if adf[1]<0.05 else 'NON-STATIONARY'}")
        results[tgt] = {"ADF_stat":round(adf[0],4), "ADF_p":round(adf[1],5),
                         "stationary": adf[1]<0.05}

    # ACF/PACF plot
    fig, axes = plt.subplots(2, 2, figsize=(14,8))
    nlags = 48
    for i, tgt in enumerate(TARGETS):
        series = train[tgt].dropna().values
        acf_vals  = acf(series,  nlags=nlags, fft=True)
        pacf_vals = pacf(series, nlags=nlags)
        axes[i,0].bar(range(len(acf_vals)),  acf_vals,  color="steelblue", width=0.5)
        axes[i,0].axhline(0, color="black", lw=0.8)
        axes[i,0].axhline(1.96/np.sqrt(len(series)),  color="red", ls="--", lw=1)
        axes[i,0].axhline(-1.96/np.sqrt(len(series)), color="red", ls="--", lw=1)
        axes[i,0].set_title(f"{tgt} — ACF", fontweight="bold")
        axes[i,0].set_xlabel("Lag (hours)")
        axes[i,1].bar(range(len(pacf_vals)), pacf_vals, color="coral",   width=0.5)
        axes[i,1].axhline(0, color="black", lw=0.8)
        axes[i,1].axhline(1.96/np.sqrt(len(series)),  color="red", ls="--", lw=1)
        axes[i,1].axhline(-1.96/np.sqrt(len(series)), color="red", ls="--", lw=1)
        axes[i,1].set_title(f"{tgt} — PACF", fontweight="bold")
        axes[i,1].set_xlabel("Lag (hours)")
    plt.suptitle("Autocorrelation & Partial Autocorrelation — BF-4 Hot Metal Quality",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/ts01_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts01_acf_pacf.png")
    return results

# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL A: ARIMAX
# ═══════════════════════════════════════════════════════════════════════
def run_arimax(train, test):
    print("\n[6] ARIMAX(1,0,1) with exogenous BF variables ...")
    exog_cols = {
        "HM_SI":   ["HBT","thermal_idx","burden_thermal","OreCokeRatio","Coal Actual"],
        "HM_TEMP": ["HBT","thermal_idx","Oxygen Flow","Coal Actual","burden_thermal"],
    }
    preds = {}
    for tgt in TARGETS:
        cols = [c for c in exog_cols[tgt] if c in train.columns]
        scX  = MinMaxScaler()
        exog_tr = scX.fit_transform(train[cols])
        exog_te = scX.transform(test[cols])

        model = SARIMAX(train[tgt].values, exog=exog_tr,
                        order=(1,0,1), trend='c',
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=200)

        # One-step-ahead on test via filter
        res_filter = fit.apply(test[tgt].values, exog=exog_te, refit=False)
        yhat = res_filter.fittedvalues

        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        mae  = mean_absolute_error(test[tgt].values, yhat)
        preds[tgt] = yhat
        print(f"     ARIMAX {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
    return preds

# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — MODEL B: VAR
# ═══════════════════════════════════════════════════════════════════════
def run_var(train, test):
    print("\n[7] VAR(p) — joint HM_SI + HM_TEMP ...")
    var_data_tr = train[TARGETS].values
    var_data_te = test[TARGETS].values

    var_model = VAR(var_data_tr)
    var_fit   = var_model.fit(maxlags=12, ic='aic')
    lag_order = var_fit.k_ar
    print(f"     VAR optimal lag order = {lag_order}")

    history = list(var_data_tr)
    var_preds = []
    for i in range(len(var_data_te)):
        h  = np.array(history[-lag_order:])
        fc = var_fit.forecast(h, steps=1)
        var_preds.append(fc[0])
        history.append(var_data_te[i])
    var_preds = np.array(var_preds)

    results = {}
    for j, tgt in enumerate(TARGETS):
        r2   = r2_score(var_data_te[:,j], var_preds[:,j])
        rmse = np.sqrt(mean_squared_error(var_data_te[:,j], var_preds[:,j]))
        mae  = mean_absolute_error(var_data_te[:,j], var_preds[:,j])
        results[tgt] = var_preds[:,j]
        print(f"     VAR {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
    return results

# ═══════════════════════════════════════════════════════════════════════
# STEP 8 — MODEL C: GradientBoosting (TS-aware features)
# ═══════════════════════════════════════════════════════════════════════
def run_gbm(X_tr, X_te, train, test):
    print("\n[8] GradientBoosting with TS lag features ...")
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
        print(f"     GBM {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
    return models, preds

# ═══════════════════════════════════════════════════════════════════════
# STEP 9 — MODEL D: RandomForest (TS-aware)
# ═══════════════════════════════════════════════════════════════════════
def run_rf(X_tr, X_te, train, test):
    print("\n[9] RandomForest with TS lag features ...")
    models = {}
    preds  = {}
    for tgt in TARGETS:
        clf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=42)
        clf.fit(X_tr, train[tgt].values)
        yhat = clf.predict(X_te)
        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        mae  = mean_absolute_error(test[tgt].values, yhat)
        models[tgt] = clf
        preds[tgt]  = yhat
        joblib.dump(clf, f"models/ts_rf_{tgt}.pkl")
        print(f"     RF  {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
    return models, preds

# ═══════════════════════════════════════════════════════════════════════
# STEP 10 — MODEL E: LSTM (TensorFlow/Keras)
# ═══════════════════════════════════════════════════════════════════════
def build_sequences(data, seq_len=24):
    """Create (X, y) sequence arrays for LSTM input."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def run_lstm(train, test, all_feats, scaler):
    print("\n[10] LSTM sequence model ...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        print("     TensorFlow not installed. Skipping LSTM.")
        print("     Install: pip install tensorflow")
        return None

    SEQ_LEN = 24   # 24-hour lookback window
    scaler_lstm = MinMaxScaler(feature_range=(0,1))
    all_data = pd.concat([train, test]).reset_index(drop=True)

    results_lstm = {}
    for tgt in TARGETS:
        print(f"     Building LSTM for {tgt} ...")

        # Use top exogenous features + target itself
        if tgt == "HM_SI":
            use_cols = ["HBT","thermal_idx","OreCokeRatio","Coal Actual",
                        "burden_thermal","ETACO","HM_SI"]
        else:
            use_cols = ["HBT","thermal_idx","Oxygen Flow","Coal Actual",
                        "burden_thermal","ETACO","HM_TEMP"]
        use_cols = [c for c in use_cols if c in all_data.columns]

        data_arr = scaler_lstm.fit_transform(all_data[use_cols].values)
        n_tr     = len(train)

        X_all_seq, y_all_seq = build_sequences(data_arr, SEQ_LEN)
        # y is last column (target)
        y_all_seq = y_all_seq[:, -1]

        train_end = n_tr - SEQ_LEN
        X_tr_seq, y_tr_seq = X_all_seq[:train_end], y_all_seq[:train_end]
        X_te_seq, y_te_seq = X_all_seq[train_end:], y_all_seq[train_end:]

        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(use_cols))),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss="mse", metrics=["mae"])

        cb = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5)
        ]
        hist = model.fit(X_tr_seq, y_tr_seq, epochs=100, batch_size=64,
                          validation_split=0.15, callbacks=cb, verbose=0)

        # Predict and inverse-scale
        yhat_sc = model.predict(X_te_seq, verbose=0).flatten()
        # Inverse-transform only the target column
        dummy   = np.zeros((len(yhat_sc), len(use_cols)))
        dummy[:,-1] = yhat_sc
        yhat    = scaler_lstm.inverse_transform(dummy)[:,-1]

        y_true_idx = len(train) - SEQ_LEN
        y_true     = all_data[tgt].values[y_true_idx + SEQ_LEN:]
        y_true     = y_true[:len(yhat)]

        r2   = r2_score(y_true, yhat)
        rmse = np.sqrt(mean_squared_error(y_true, yhat))
        mae  = mean_absolute_error(y_true, yhat)
        results_lstm[tgt] = {"preds": yhat, "y_true": y_true,
                              "R2":round(r2,4), "RMSE":round(rmse,5), "MAE":round(mae,5),
                              "history": hist.history}
        print(f"     LSTM {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")
        model.save(f"models/ts_lstm_{tgt}.keras")

    return results_lstm

# ═══════════════════════════════════════════════════════════════════════
# STEP 11 — 95% PREDICTION INTERVALS (Bootstrap for RF/GBM)
# ═══════════════════════════════════════════════════════════════════════
def rf_pi(clf, X_new):
    """RF quantile PI — tree-level distribution (only for RandomForest)."""
    from sklearn.ensemble import RandomForestRegressor
    if not isinstance(clf, RandomForestRegressor):
        return None
    tree_preds = np.array([t.predict(X_new) for t in clf.estimators_])
    return (np.percentile(tree_preds, 2.5, axis=0),
            np.percentile(tree_preds, 97.5, axis=0))

def rmse_pi(yhat, rmse, n_tr, p, alpha=0.05):
    """RMSE-based PI for any model."""
    t_c = stats.t.ppf(1-alpha/2, df=max(n_tr-p-1,1))
    hw  = t_c * rmse * np.sqrt(1 + 1/n_tr)
    return yhat - hw, yhat + hw

# ═══════════════════════════════════════════════════════════════════════
# STEP 12 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════
def make_plots(train, test, all_preds, all_feats, rf_models, gbm_models):
    print("\n[11] Generating plots ...")
    clk = test["CLOCK"].values

    # ── 12a: Time-series plot of predictions vs actual (test set) ──
    for tgt in TARGETS:
        y_true = test[tgt].values
        fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
        model_list = ["ARIMAX","VAR","GBM","RF"]
        colors     = ["#3498DB","#E74C3C","#2ECC71","#F39C12"]
        for ax, mname, col in zip(axes, model_list, colors):
            yhat = all_preds[tgt][mname]
            if yhat is None or len(yhat)==0: continue
            n   = min(len(yhat), len(y_true), len(clk))
            r2  = r2_score(y_true[:n], yhat[:n])
            rmse= np.sqrt(mean_squared_error(y_true[:n], yhat[:n]))
            ax.plot(clk[:n], y_true[:n], "gray",    lw=1.2, alpha=0.7, label="Actual")
            ax.plot(clk[:n], yhat[:n],   color=col, lw=1.2, alpha=0.9,
                    label=f"{mname}  R²={r2:.3f}  RMSE={rmse:.4f}")
            lo_s, hi_s = SPEC[tgt]
            ax.axhspan(lo_s, hi_s, alpha=0.07, color="green", label="Spec band")
            ax.set_ylabel(f"{tgt} ({UNITS[tgt]})", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"{tgt} Prediction — 4 Time-Series Models (Test Period)",
                           fontsize=13, fontweight="bold")
        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        plt.savefig(f"output/ts02_timeseries_{tgt}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"     Saved: output/ts02_timeseries_{tgt}.png")

    # ── 12b: Actual vs Predicted scatter (GBM — best model) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tgt in zip(axes, TARGETS):
        y_true = test[tgt].values
        yhat   = all_preds[tgt]["GBM"]
        n      = min(len(yhat), len(y_true))
        r2     = r2_score(y_true[:n], yhat[:n])
        rmse   = np.sqrt(mean_squared_error(y_true[:n], yhat[:n]))
        ax.scatter(y_true[:n], yhat[:n], alpha=0.3, s=8, c="steelblue")
        mn, mx = min(y_true.min(), yhat.min()), max(y_true.max(), yhat.max())
        ax.plot([mn,mx],[mn,mx], "r--", lw=1.5, label="Ideal")
        ax.set_xlabel(f"Actual ({UNITS[tgt]})")
        ax.set_ylabel(f"Predicted ({UNITS[tgt]})")
        ax.set_title(f"{tgt} — GBM  R²={r2:.4f}  RMSE={rmse:.5f}",
                     fontweight="bold")
        ax.legend(fontsize=8)
    plt.suptitle("Actual vs Predicted — GradientBoosting (Best Model)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/ts03_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 12c: RF Prediction Interval plot ──
    for tgt in TARGETS:
        y_true = test[tgt].values
        n_plot = min(200, len(y_true))
        xi     = np.arange(n_plot)
        X_te_sub = all_preds["X_te"][:n_plot]
        clf    = rf_models[tgt]
        pi_lo, pi_hi = rf_pi(clf, X_te_sub)
        yhat         = clf.predict(X_te_sub)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(xi, pi_lo[:n_plot], pi_hi[:n_plot],
                         alpha=0.25, color="steelblue", label="95% PI")
        ax.plot(xi, y_true[:n_plot],  "gray",      lw=1.2, label="Actual")
        ax.plot(xi, yhat[:n_plot],    "steelblue", lw=1.5, label="RF Predicted")
        lo_s, hi_s = SPEC[tgt]
        ax.axhline(lo_s, color="green", ls="--", lw=1)
        ax.axhline(hi_s, color="green", ls="--", lw=1, label="Spec limits")
        ax.set_title(f"{tgt} — RF Prediction with 95% PI (first 200 test hours)",
                     fontweight="bold")
        ax.set_xlabel("Test Sample (hours)"); ax.set_ylabel(f"{tgt} ({UNITS[tgt]})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f"output/ts04_PI_{tgt}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print("     Saved: output/ts03_actual_vs_pred.png  ts04_PI_*.png")

    # ── 12d: Feature importances (GBM) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, tgt in zip(axes, TARGETS):
        clf  = gbm_models[tgt]
        imp  = sorted(zip(all_feats, clf.feature_importances_),
                       key=lambda x:-x[1])[:15]
        feats = [x[0].replace("_"," ") for x in imp]
        imps  = [x[1] for x in imp]
        ax.barh(range(len(feats)), imps, color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top Features — {tgt} (GBM)", fontweight="bold")
    plt.suptitle("Feature Importances — GradientBoosting Time-Series Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/ts05_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 12e: Model comparison bar chart ──
    model_names = ["ARIMAX","VAR","GBM","RF"]
    metrics = {tgt: {} for tgt in TARGETS}
    for tgt in TARGETS:
        y_true = test[tgt].values
        for mn in model_names:
            yhat = all_preds[tgt].get(mn)
            if yhat is None or len(yhat)==0:
                metrics[tgt][mn] = {"R2":0, "RMSE":999}
                continue
            n = min(len(yhat), len(y_true))
            metrics[tgt][mn] = {
                "R2":   round(r2_score(y_true[:n], yhat[:n]), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_true[:n], yhat[:n])), 5),
            }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tgt in zip(axes, TARGETS):
        r2s   = [metrics[tgt][mn]["R2"]   for mn in model_names]
        rmses = [metrics[tgt][mn]["RMSE"] for mn in model_names]
        x = np.arange(len(model_names)); w = 0.35
        ax.bar(x-w/2, r2s,   w, label="R²",   color="steelblue")
        ax_r = ax.twinx()
        ax_r.bar(x+w/2, rmses, w, label="RMSE", color="coral", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("R²"); ax_r.set_ylabel(f"RMSE ({UNITS[tgt]})")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{tgt} — Model Comparison", fontweight="bold")
        ax.legend(loc="upper left"); ax_r.legend(loc="upper right")
        for i, (r2, rmse) in enumerate(zip(r2s, rmses)):
            ax.text(x[i]-w/2, r2+0.01, f"{r2:.3f}", ha="center", fontsize=8)
    plt.suptitle("Time-Series Model Comparison: R² and RMSE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/ts06_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: output/ts05_feature_importances.png  ts06_model_comparison.png")
    return metrics

# ═══════════════════════════════════════════════════════════════════════
# STEP 13 — SAVE RESULTS CSV
# ═══════════════════════════════════════════════════════════════════════
def save_results(test, all_preds, metrics):
    rows = []
    y_si_true   = test["HM_SI"].values
    y_temp_true = test["HM_TEMP"].values
    n = min(len(y_si_true), len(all_preds["HM_SI"]["GBM"]))
    for i in range(n):
        rows.append({
            "Timestamp":         test["CLOCK"].iloc[i],
            "HM_SI_Actual":      y_si_true[i],
            "HM_SI_GBM_Pred":    all_preds["HM_SI"]["GBM"][i],
            "HM_SI_RF_Pred":     all_preds["HM_SI"]["RF"][i],
            "HM_TEMP_Actual":    y_temp_true[i],
            "HM_TEMP_GBM_Pred":  all_preds["HM_TEMP"]["GBM"][i],
            "HM_TEMP_RF_Pred":   all_preds["HM_TEMP"]["RF"][i],
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv("output/ts_predictions.csv", index=False)
    print("\n     Saved: output/ts_predictions.csv")

    print("\n" + "="*64)
    print(f"  {'Model':<10} {'HM_SI R²':>10} {'HM_SI RMSE':>12} {'TEMP R²':>10} {'TEMP RMSE':>12}")
    print("  " + "-"*54)
    for mn in ["ARIMAX","VAR","GBM","RF"]:
        si  = metrics["HM_SI"][mn]
        tmp = metrics["HM_TEMP"][mn]
        print(f"  {mn:<10} {si['R2']:>10.4f} {si['RMSE']:>12.5f}"
              f" {tmp['R2']:>10.4f} {tmp['RMSE']:>12.4f}")
    print("="*64)

# ═══════════════════════════════════════════════════════════════════════
# STEP 14 — REAL-TIME SINGLE-POINT PREDICTION
# ═══════════════════════════════════════════════════════════════════════
def predict_next_tap(process_state: dict, all_feats: list, alpha=0.05) -> dict:
    """
    Predict HM_SI and HM_TEMP for the next tap.
    process_state: dict with ALL feature values (including AR lags computed
                   from the last 24 hourly observations).
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

        # RF quantile PI
        tree_preds = np.array([t.predict(row_s)[0] for t in clf_rf.estimators_])
        pi_lo = float(np.percentile(tree_preds, 2.5))
        pi_hi = float(np.percentile(tree_preds, 97.5))

        lo_s, hi_s = SPEC[tgt]
        output[tgt] = {
            "GBM_pred":   round(yhat_gbm, 4),
            "RF_pred":    round(yhat_rf, 4),
            "RF_PI_lo":   round(pi_lo, 4),
            "RF_PI_hi":   round(pi_hi, 4),
            "In_Spec_GBM": lo_s <= yhat_gbm <= hi_s,
            "In_Spec_RF":  lo_s <= yhat_rf  <= hi_s,
        }
    return output

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hm_h   = load_hm("HM_ANALYSIS.xls")
    bf     = load_para("PARA.xlsx")
    burden = load_burden("BURDEN.csv", lag_h=8)

    merged   = build_merged(bf, burden, hm_h)
    df_feat, all_feats = engineer_features(merged)
    train, test, X_tr, X_te, scaler = split_data(df_feat, all_feats)

    stationarity_analysis(train, test)

    preds_arimax = run_arimax(train, test)
    preds_var    = run_var(train, test)
    gbm_models, preds_gbm = run_gbm(X_tr, X_te, train, test)
    rf_models,  preds_rf  = run_rf(X_tr, X_te, train, test)

    # TF LSTM (optional)
    lstm_results = run_lstm(train, test, all_feats, scaler)

    # Collect all predictions
    all_preds = {"X_te": X_te}
    for tgt in TARGETS:
        all_preds[tgt] = {
            "ARIMAX": preds_arimax.get(tgt, []),
            "VAR":    preds_var.get(tgt, []),
            "GBM":    preds_gbm[tgt],
            "RF":     preds_rf[tgt],
        }
        if lstm_results and tgt in lstm_results:
            all_preds[tgt]["LSTM"] = lstm_results[tgt]["preds"]

    metrics = make_plots(train, test, all_preds, all_feats, rf_models, gbm_models)
    save_results(test, all_preds, metrics)

    print("\n" + "="*56)
    print("TIME-SERIES PIPELINE COMPLETE")
    print("  Plots  : output/ts01_acf_pacf.png ... ts06_model_comparison.png")
    print("  Data   : output/ts_predictions.csv")
    print("  Models : models/ts_gbm_*.pkl  |  models/ts_rf_*.pkl")
    print("="*56)
