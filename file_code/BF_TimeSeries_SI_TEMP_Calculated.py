#!/usr/bin/env python3
"""
BF-4 Blast Furnace — Time-Series ML Prediction
Targets : HM_SI (%Si)  |  HM_TEMP (°C)
Models  : ARIMAX | VAR | GradientBoosting | RandomForest | LSTM (optional)
Files   : HM_ANALYSIS.xls | PARA.xlsx | BURDEN.csv  (same folder)

Run:
    pip install pandas numpy matplotlib scikit-learn statsmodels scipy joblib openpyxl xlrd
    python BF_TimeSeries_SI_TEMP.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, acf, pacf
import joblib

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TARGETS   = ["HM_SI", "HM_TEMP"]
UNITS     = {"HM_SI": "%Si", "HM_TEMP": "°C"}
SI_SPEC   = (0.25, 0.80)
TEMP_SPEC = (1480, 1535)
ALPHA     = 0.05
BF_LAG_H  = 8          # burden descent lag (hours)
IQR_K     = 3.5        # wider IQR band to keep more rows
TEST_FRAC = 0.20
AR_LAGS   = list(range(1, 13))

print("="*66)
print("  BF-4 HOT METAL QUALITY — TIME-SERIES PREDICTION PIPELINE")
print("="*66)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — clean comma-formatted numbers  e.g. "1,533.790" → 1533.790
# ─────────────────────────────────────────────────────────────────────────────
def _strip_commas(series):
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False),
                             errors="coerce")
    return pd.to_numeric(series, errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1A — LOAD HM_ANALYSIS.xls
#   Timestamp col : SAMPLETAKEN  (format dd/mm/yyyy HH:MM:SS)
#   HM_TEMP       : stored as "1,533.790" — strip comma
# ─────────────────────────────────────────────────────────────────────────────
def load_hm(path="HM_ANALYSIS.xls"):
    print(f"\n[1a] Loading {path} ...")
    df = pd.read_excel(path, sheet_name=0)
    df.columns = (df.columns.astype(str).str.strip().str.upper()
                  .str.replace(r"\s+", "_", regex=True))

    # Timestamp
    ts_col = next((c for c in df.columns
                   if c in ("SAMPLETAKEN","CLOCK","DATETIME","DATE","TIME","TIMESTAMP")),
                  None)
    if ts_col is None:
        # fall back: first column that parses as datetime
        for c in df.columns:
            parsed = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                ts_col = c
                break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path}. Columns: {list(df.columns)}")

    df["CLOCK"] = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["CLOCK"])
    df["CLOCK"] = df["CLOCK"].dt.floor("H")

    # Numeric conversion — handle comma-formatted HM_TEMP
    for c in [cc for cc in df.columns if cc.startswith("HM_")]:
        df[c] = _strip_commas(df[c])

    df = df.dropna(subset=["HM_SI", "HM_TEMP"])
    keep = ["CLOCK", "HM_SI", "HM_TEMP", "HM_C", "HM_S", "HM_MN", "HM_P"]
    keep = [c for c in keep if c in df.columns]
    df = df.groupby("CLOCK")[keep[1:]].mean().reset_index()
    print(f"    HM records  : {len(df):,}  |  "
          f"{df.CLOCK.min().date()} → {df.CLOCK.max().date()}")
    print(f"    HM_SI  range: {df.HM_SI.min():.3f} – {df.HM_SI.max():.3f} %Si")
    print(f"    HM_TEMP range: {df.HM_TEMP.min():.1f} – {df.HM_TEMP.max():.1f} °C")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1B — LOAD PARA.xlsx
#   Sheet "BF-4 Data"  (sheet index 1) is the data sheet.
#   Timestamp is the FIRST column (no header name — becomes column index 0).
#   Format: dd/mm/yyyy HH:MM
# ─────────────────────────────────────────────────────────────────────────────
def load_para(path="PARA.xlsx"):
    print(f"[1b] Loading {path} ...")
    xl  = pd.ExcelFile(path)
    # Pick "BF-4 Data" sheet, or the last sheet if not found
    data_sheet = xl.sheet_names[-1]
    for name in xl.sheet_names:
        if "BF" in name.upper() or "DATA" in name.upper() or "PARA" in name.upper():
            data_sheet = name
            break

    df = pd.read_excel(path, sheet_name=data_sheet, header=0)
    df.columns = (df.columns.astype(str).str.strip().str.upper()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^A-Z0-9_]", "", regex=True))

    # First column is timestamp (may be unnamed → "UNNAMED:_0")
    ts_col = df.columns[0]
    df["CLOCK"] = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["CLOCK"])
    df["CLOCK"] = df["CLOCK"].dt.floor("H")

    # Drop columns with >99% nulls (CS moisture etc.)
    thresh = max(1, int(0.01 * len(df)))
    df = df.dropna(axis=1, thresh=thresh)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df = df.groupby("CLOCK")[num_cols].mean().reset_index()

    # Rename known columns (robust partial matching)
    rename_map = {}
    for c in df.columns:
        cu = c.upper()
        if "HBT" in cu or "BLASTTEMP" in cu:    rename_map[c] = "HBT"
        elif "OXYGENFLOW" in cu or "O_FLOW" in cu or "O2FLOW" in cu: rename_map[c] = "O_FLOW_ACT"
        elif "COALACTUAL" in cu or "COAL_ACT" in cu or "COALACT" in cu: rename_map[c] = "COAL_ACT"
        elif "ETACO" in cu or "ETA_CO" in cu or ("ETA" in cu and "CO" in cu): rename_map[c] = "ETACO"
        elif "PERMEABIL" in cu or "PERM" in cu: rename_map[c] = "PERM_INDEX"
        elif "PROD_RATE" in cu or "PRODRATE" in cu: rename_map[c] = "PROD_RATE"
        elif "SLAG_RATE" in cu or "SLAGRATE" in cu: rename_map[c] = "SLAG_RATE"
    df = df.rename(columns=rename_map)

    print(f"    PARA records: {len(df):,}  |  "
          f"{df.CLOCK.min().date()} → {df.CLOCK.max().date()}")
    print(f"    PARA columns: {len(df.columns)-1}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1C — LOAD BURDEN.csv
#   Columns: CHARGETIME, TYP, BRANDCODE, ACTWT, SETWT
#   Pivot to hourly kg per material group; apply BF_LAG_H
# ─────────────────────────────────────────────────────────────────────────────
def load_burden(path="BURDEN.csv"):
    print(f"[1c] Loading {path} ...")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = (df.columns.astype(str).str.strip().str.upper()
                  .str.replace(r"\s+", "_", regex=True))

    # Timestamp
    ts_col = next((c for c in df.columns
                   if c in ("CHARGETIME","CLOCK","DATETIME","DATE","TIME","TIMESTAMP")),
                  df.columns[0])
    df["CLOCK"] = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["CLOCK"])
    df["CLOCK"] = df["CLOCK"].dt.floor("H")

    # Weight column
    wt_col = next((c for c in df.columns if "ACTWT" in c or "WEIGHT" in c), None)
    if wt_col:
        df[wt_col] = pd.to_numeric(df[wt_col], errors="coerce").fillna(0)
    brand_col = next((c for c in df.columns if "BRAND" in c), None)

    if brand_col and wt_col:
        df["bl"] = df[brand_col].astype(str).str.strip().str.lower()
        brand_map = {
            "Coke_kg"  : ["coke","cok"],
            "Sinter_kg": ["sinter","sint","smallsinter"],
            "Pellet_kg": ["pellet","pell"],
            "Ore_kg"   : ["ironore","ore","bhq"],
            "Flux_kg"  : ["limestone","lime","dolomite","dolo","quartzite","quartz"],
            "NutCoke_kg": ["nutcoke","nut"],
            "DRI_kg"   : ["dri"],
            "Mix_kg"   : ["mix","return","mixed"],
        }
        for col_name, kws in brand_map.items():
            pat = "|".join(kws)
            df[col_name] = np.where(df["bl"].str.contains(pat, na=False), df[wt_col], 0)
        piv_cols = list(brand_map.keys())
        bdf = df.groupby("CLOCK")[piv_cols].sum().reset_index()
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        bdf = df.groupby("CLOCK")[num_cols].sum().reset_index()

    # Apply BF descent time lag
    bdf["CLOCK"] = bdf["CLOCK"] + pd.Timedelta(hours=BF_LAG_H)
    print(f"    BURDEN rows : {len(bdf):,}  |  "
          f"{bdf.CLOCK.min().date()} → {bdf.CLOCK.max().date()}")
    return bdf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MERGE THREE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(hm, para, burden):
    print("\n[2] Merging datasets ...")
    print(f"    HM overlap   : {hm.CLOCK.min().date()} – {hm.CLOCK.max().date()}")
    print(f"    PARA overlap : {para.CLOCK.min().date()} – {para.CLOCK.max().date()}")
    print(f"    BURDEN+lag   : {burden.CLOCK.min().date()} – {burden.CLOCK.max().date()}")

    # First merge HM + PARA (inner)
    df = hm.merge(para, on="CLOCK", how="inner", suffixes=("", "_P"))
    print(f"    HM ∩ PARA    : {len(df):,} rows")
    if len(df) == 0:
        raise ValueError(
            "HM ∩ PARA merge is EMPTY.\n"
            "Check that PARA.xlsx 'BF-4 Data' sheet timestamps overlap with HM_ANALYSIS.xls.\n"
            f"  HM   : {hm.CLOCK.min()} → {hm.CLOCK.max()}\n"
            f"  PARA : {para.CLOCK.min()} → {para.CLOCK.max()}"
        )

    # Merge with BURDEN (left so we don't lose HM+PARA rows if burden is sparse)
    df = df.merge(burden, on="CLOCK", how="left", suffixes=("", "_B"))
    burden_cols = [c for c in burden.columns if c != "CLOCK"]
    df[burden_cols] = df[burden_cols].fillna(0)
    print(f"    After burden : {len(df):,} rows")

    # Derived metallurgical features
    coke  = df.get("Coke_kg",  pd.Series(np.ones(len(df)),  index=df.index))
    ore   = df.get("Ore_kg",   pd.Series(np.zeros(len(df)), index=df.index))
    flux  = df.get("Flux_kg",  pd.Series(np.zeros(len(df)), index=df.index))
    sinter = df.get("Sinter_kg", pd.Series(np.zeros(len(df)), index=df.index))
    pellet = df.get("Pellet_kg", pd.Series(np.zeros(len(df)), index=df.index))
    iron  = (ore + sinter + pellet).replace(0, np.nan)

    df["OreCokeRatio"]  = ore  / coke.replace(0, np.nan)
    df["FluxIronRatio"] = flux / iron
    df["OreFrac"]       = ore  / iron

    if "HBT" in df.columns and "O_FLOW_ACT" in df.columns:
        df["thermal_idx"]   = df["HBT"] * df["O_FLOW_ACT"] / 1e5
    if "COAL_ACT" in df.columns and "PROD_RATE" in df.columns:
        df["coal_intensity"] = df["COAL_ACT"] / df["PROD_RATE"].replace(0, np.nan)

    # IQR cleaning — only on process + HM columns, NOT on burden weight sums
    iqr_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ["Coke_kg","Sinter_kg","Pellet_kg","Ore_kg",
                             "Flux_kg","NutCoke_kg","DRI_kg","Mix_kg"]]
    n_before = len(df)
    for col in iqr_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            df = df[df[col].between(q1 - IQR_K*iqr, q3 + IQR_K*iqr)]
    df = df.sort_values("CLOCK").dropna(subset=TARGETS).reset_index(drop=True)
    print(f"    After IQR    : {len(df):,} rows  (removed {n_before-len(df):,})")
    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} rows after merge+clean — too few for modeling.\n"
            "Check that PARA.xlsx BF-4 Data timestamps match HM_ANALYSIS.xls dates."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
EXOG_COLS = ["HBT","O_FLOW_ACT","COAL_ACT","ETACO","PERM_INDEX","PROD_RATE",
             "OreCokeRatio","FluxIronRatio","thermal_idx","coal_intensity"]

def engineer_features(df):
    print("\n[3] Engineering autoregressive features ...")
    fd = df.copy()
    for tgt in TARGETS:
        for lag in AR_LAGS:
            fd[f"{tgt}_lag{lag}"] = fd[tgt].shift(lag)
        fd[f"{tgt}_roll4"]  = fd[tgt].shift(1).rolling(4).mean()
        fd[f"{tgt}_roll8"]  = fd[tgt].shift(1).rolling(8).mean()
        fd[f"{tgt}_diff1"]  = fd[tgt].diff(1)
    proc_cols = [c for c in EXOG_COLS if c in fd.columns]
    for col in proc_cols:
        for lag in [2, 4, 8]:
            fd[f"{col}_lag{lag}"] = fd[col].shift(lag)
    fd = fd.dropna().reset_index(drop=True)
    print(f"    Rows after lag: {len(fd):,}")
    return fd


def get_feature_cols(fd):
    excl = set(TARGETS) | {"CLOCK","HM_C","HM_S","HM_MN","HM_P"}
    return [c for c in fd.columns
            if c not in excl and pd.api.types.is_numeric_dtype(fd[c])]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — STATIONARITY TEST
# ─────────────────────────────────────────────────────────────────────────────
def stationarity_tests(fd):
    print("\n[4] Augmented Dickey-Fuller Stationarity Tests")
    print(f"    {'Target':<12} {'ADF stat':>10} {'p-value':>10} {'Crit(5%)':>10} {'Result':>14}")
    print("    " + "─"*58)
    for tgt in TARGETS:
        s = fd[tgt].dropna().values
        if len(s) < 40 or np.allclose(s.min(), s.max()):
            print(f"    {tgt:<12} {'—':>10} {'—':>10} {'—':>10} {'not enough data':>14}")
            continue
        try:
            stat, pv, _, _, cv, _ = adfuller(s, autolag="AIC")
            res = "STATIONARY ✓" if pv < 0.05 else "UNIT ROOT ✗"
            print(f"    {tgt:<12} {stat:>10.3f} {pv:>10.6f} {cv['5%']:>10.3f} {res:>14}")
        except Exception as e:
            print(f"    {tgt:<12} {'—':>10} {'—':>10} {'—':>10} {str(e)[:20]:>14}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def chronological_split(fd):
    n      = len(fd)
    n_test = max(int(n * TEST_FRAC), 100)
    n_train = n - n_test
    if n_train < 50:
        raise ValueError(
            f"Too few rows for train/test split (n={n}). "
            "Check dataset loading — see messages above for row counts.")
    train = fd.iloc[:n_train].reset_index(drop=True)
    test  = fd.iloc[n_train:].reset_index(drop=True)
    print(f"\n[5] Chronological split → Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MODEL A: ARIMAX(1,0,1)
# ─────────────────────────────────────────────────────────────────────────────
def run_arimax(train, test, exog_cols):
    print("\n[6] ARIMAX(1,0,1) ...")
    preds = {}
    avail = [c for c in exog_cols if c in train.columns and c in test.columns]
    for tgt in TARGETS:
        try:
            if avail:
                scX     = MinMaxScaler()
                ex_tr   = scX.fit_transform(train[avail].fillna(0))
                ex_te   = scX.transform(test[avail].fillna(0))
                mdl = SARIMAX(train[tgt].values, exog=ex_tr, order=(1,0,1),
                              trend="c", enforce_stationarity=False,
                              enforce_invertibility=False)
            else:
                mdl = SARIMAX(train[tgt].values, order=(1,0,1), trend="c",
                              enforce_stationarity=False, enforce_invertibility=False)
                ex_te = None
            fit  = mdl.fit(disp=False, maxiter=300)
            yhat = fit.forecast(len(test), exog=ex_te)
            r2   = r2_score(test[tgt].values, yhat)
            rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
            print(f"    {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}")
            preds[tgt] = np.asarray(yhat, dtype=float)
        except Exception as exc:
            print(f"    {tgt}: ARIMAX failed ({exc}) — naive mean fallback")
            preds[tgt] = np.full(len(test), train[tgt].mean(), dtype=float)
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — MODEL B: VAR(p)
# ─────────────────────────────────────────────────────────────────────────────
def run_var(train, test):
    print("\n[7] VAR(p) ...")
    vtr = train[TARGETS].values
    vte = test[TARGETS].values
    try:
        vfit = VAR(vtr).fit(maxlags=12, ic="aic")
        k    = vfit.k_ar
        print(f"    AIC-optimal lag p={k}")
        hist = list(vtr)
        vp   = []
        for i in range(len(vte)):
            h  = np.array(hist[-k:])
            fc = vfit.forecast(h, steps=1)
            vp.append(fc[0])
            hist.append(vte[i])
        vp = np.array(vp)
        res = {}
        for j, tgt in enumerate(TARGETS):
            r2   = r2_score(vte[:,j], vp[:,j])
            rmse = np.sqrt(mean_squared_error(vte[:,j], vp[:,j]))
            print(f"    {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}")
            res[tgt] = vp[:,j].astype(float)
        return res, vfit
    except Exception as exc:
        print(f"    VAR failed ({exc})")
        return {tgt: np.full(len(test), train[tgt].mean(), dtype=float)
                for tgt in TARGETS}, None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — MODEL C: GRADIENT BOOSTING
# ─────────────────────────────────────────────────────────────────────────────
def run_gbm(X_tr, X_te, train, test, feat_cols):
    print("\n[8] GradientBoosting ...")
    models, preds = {}, {}
    for tgt in TARGETS:
        clf = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        clf.fit(X_tr, train[tgt].values)
        yhat = clf.predict(X_te)
        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        print(f"    {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}")
        models[tgt] = clf
        preds[tgt]  = yhat.astype(float)
        joblib.dump(clf, f"models/ts_gbm_{tgt}.pkl")
    return models, preds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — MODEL D: RANDOM FOREST + PREDICTION INTERVALS
# ─────────────────────────────────────────────────────────────────────────────
def run_rf(X_tr, X_te, train, test, feat_cols):
    print("\n[9] RandomForest ...")
    models, preds = {}, {}
    for tgt in TARGETS:
        clf = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42)
        clf.fit(X_tr, train[tgt].values)
        yhat = clf.predict(X_te)
        r2   = r2_score(test[tgt].values, yhat)
        rmse = np.sqrt(mean_squared_error(test[tgt].values, yhat))
        print(f"    {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}")
        models[tgt] = clf
        preds[tgt]  = yhat.astype(float)
        joblib.dump(clf, f"models/ts_rf_{tgt}.pkl")
    return models, preds


def rf_pi(clf, X, alpha=ALPHA):
    """Tree-quantile 95% prediction interval."""
    tp  = np.array([t.predict(X) for t in clf.estimators_])
    lo  = np.percentile(tp, 100*alpha/2,     axis=0)
    hi  = np.percentile(tp, 100*(1-alpha/2), axis=0)
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — MODEL E: LSTM (optional, requires TensorFlow)
# ─────────────────────────────────────────────────────────────────────────────
def run_lstm(train_df, test_df):
    print("\n[10] LSTM (optional) ...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        print("     TensorFlow not installed — skipping. (pip install tensorflow)")
        return {}

    SEQ_LEN = 24
    results  = {}
    all_data = pd.concat([train_df, test_df]).reset_index(drop=True)
    avail    = [c for c in EXOG_COLS if c in all_data.columns]

    for tgt in TARGETS:
        use_cols = avail + [tgt]
        sc       = MinMaxScaler()
        arr      = sc.fit_transform(all_data[use_cols].fillna(0).values)
        Xs, ys   = [], []
        for t in range(SEQ_LEN, len(arr)):
            Xs.append(arr[t-SEQ_LEN:t])
            ys.append(arr[t, -1])
        Xs, ys = np.array(Xs), np.array(ys)
        n_tr   = len(train_df) - SEQ_LEN
        Xtr, ytr = Xs[:n_tr], ys[:n_tr]
        Xte, yte = Xs[n_tr:], ys[n_tr:]

        mdl = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(use_cols))),
            Dropout(0.2), BatchNormalization(),
            LSTM(32), Dropout(0.2),
            Dense(16, activation="relu"), Dense(1)
        ])
        mdl.compile(optimizer="adam", loss="mse")
        mdl.fit(Xtr, ytr, epochs=100, batch_size=64, validation_split=0.1, verbose=0,
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                           ReduceLROnPlateau(factor=0.5, patience=8)])
        yhat_sc      = mdl.predict(Xte, verbose=0).flatten()
        dummy        = np.zeros((len(yhat_sc), len(use_cols)))
        dummy[:, -1] = yhat_sc
        yhat         = sc.inverse_transform(dummy)[:, -1]

        n_off  = n_tr + SEQ_LEN
        y_true = all_data[tgt].values[n_off:n_off+len(yhat)]
        r2     = r2_score(y_true, yhat)
        rmse   = np.sqrt(mean_squared_error(y_true, yhat))
        print(f"    {tgt}: R²={r2:.4f}  RMSE={rmse:.5f}")
        results[tgt] = np.asarray(yhat, dtype=float)
        try:
            mdl.save(f"models/ts_lstm_{tgt}.keras")
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_LIST   = ["ARIMAX", "VAR", "GBM", "RF"]
MODEL_COLORS = {"ARIMAX":"#2980B9","VAR":"#E74C3C","GBM":"#2ECC71",
                "RF":"#F39C12","LSTM":"#9B59B6"}
TGT_COLOR    = {"HM_SI":"#2980B9","HM_TEMP":"#C0392B"}


def _get(ap, tgt, model, n):
    arr = ap.get(tgt, {}).get(model, None)
    if arr is None:
        return np.full(n, np.nan)
    arr = np.asarray(arr, dtype=float)
    out = np.full(n, np.nan)
    m   = min(len(arr), n)
    out[:m] = arr[:m]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_acf_pacf(train_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ACF & PACF — BF-4 Hot Metal Quality (Training Set)", fontweight="bold")
    for i, tgt in enumerate(TARGETS):
        s  = train_df[tgt].dropna().values
        ci = 1.96 / np.sqrt(len(s))
        av = acf(s,  nlags=40, fft=True)
        pv = pacf(s, nlags=40, method="ols")
        for j, (vals, lbl) in enumerate([(av,"ACF"),(pv,"PACF")]):
            ax = axes[i, j]
            ax.bar(range(len(vals)), vals, color=TGT_COLOR[tgt], width=0.6, alpha=0.75)
            ax.axhline(0,  color="k",   lw=0.8)
            ax.axhline( ci, color="red", lw=1.2, ls="--")
            ax.axhline(-ci, color="red", lw=1.2, ls="--")
            ax.set_title(f"{tgt} — {lbl}", fontweight="bold")
            ax.set_xlabel("Lag (h)"); ax.set_ylabel(lbl)
    plt.tight_layout()
    plt.savefig("output/ts01_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts01_acf_pacf.png")


def plot_timeseries(test, ap):
    n     = min(300, len(test))
    clk   = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    for ax, tgt in zip(axes, TARGETS):
        ax.plot(clk, test[tgt].values[:n], "k-", lw=1.8, label="Actual", zorder=5)
        for m in MODEL_LIST:
            y = _get(ap, tgt, m, n)
            if not np.all(np.isnan(y)):
                ax.plot(clk, y, "--", lw=1.2, color=MODEL_COLORS[m], label=m, alpha=0.85)
        ax.set_title(f"{tgt} — Actual vs Predicted (first {n} test hours)", fontweight="bold")
        ax.set_xlabel("Test Hour Index"); ax.set_ylabel(UNITS[tgt])
        ax.legend(ncol=5, fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ts02_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts02_timeseries.png")


def plot_scatter(test, ap):
    n    = len(test)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Actual vs Predicted — All Models", fontweight="bold")
    for r, tgt in enumerate(TARGETS):
        yt = test[tgt].values
        for c, m in enumerate(MODEL_LIST):
            ax   = axes[r, c]
            yhat = _get(ap, tgt, m, n)
            mask = ~np.isnan(yhat)
            if mask.sum() > 5:
                r2   = r2_score(yt[mask], yhat[mask])
                rmse = np.sqrt(mean_squared_error(yt[mask], yhat[mask]))
                ax.scatter(yt[mask], yhat[mask], s=5, alpha=0.35, color=MODEL_COLORS[m])
                lim = [min(yt[mask].min(), yhat[mask].min()),
                       max(yt[mask].max(), yhat[mask].max())]
                ax.plot(lim, lim, "k--", lw=1)
                ax.set_title(f"{tgt}|{m}\nR²={r2:.4f} RMSE={rmse:.4f}", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Actual {UNITS[tgt]}"); ax.set_ylabel(f"Pred {UNITS[tgt]}")
    plt.tight_layout()
    plt.savefig("output/ts03_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts03_scatter.png")


def plot_pi(test, X_te, rf_models):
    n     = min(200, len(test))
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    for ax, tgt in zip(axes, TARGETS):
        yt   = test[tgt].values[:n]
        yhat = rf_models[tgt].predict(X_te[:n])
        lo, hi = rf_pi(rf_models[tgt], X_te[:n])
        cov  = np.mean((yt >= lo) & (yt <= hi)) * 100
        ax.fill_between(range(n), lo, hi, alpha=0.25, color=TGT_COLOR[tgt], label="95% PI")
        ax.plot(range(n), yt,   "k-",  lw=1.8, label="Actual")
        ax.plot(range(n), yhat, "--",  lw=1.2, color=TGT_COLOR[tgt], label="RF Pred")
        ax.set_title(f"{tgt} — RF Prediction Interval (Coverage={cov:.1f}%)", fontweight="bold")
        ax.set_xlabel("Test Hour Index"); ax.set_ylabel(UNITS[tgt])
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ts04_pi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts04_pi.png")


def plot_feat_importance(gbm_models, feat_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, tgt in zip(axes, TARGETS):
        imp = gbm_models[tgt].feature_importances_
        idx = np.argsort(imp)[-15:]
        ax.barh([feat_cols[i] for i in idx], imp[idx],
                color=TGT_COLOR[tgt], edgecolor="white")
        ax.set_title(f"{tgt} — Top 15 Features (GBM)", fontweight="bold")
        ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ts05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts05_feature_importance.png")


def plot_residuals(test, ap):
    n    = len(test)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Residual Analysis — GBM Model", fontweight="bold")
    for r, tgt in enumerate(TARGETS):
        yt    = test[tgt].values
        yhat  = _get(ap, tgt, "GBM", n)
        mask  = ~np.isnan(yhat)
        resid = yt[mask] - yhat[mask]
        ax1 = axes[r, 0]; ax2 = axes[r, 1]
        ax1.scatter(yhat[mask], resid, s=5, alpha=0.3, color=TGT_COLOR[tgt])
        ax1.axhline(0, color="k", lw=1.2, ls="--")
        ax1.set_title(f"{tgt} — Residual vs Fitted", fontweight="bold")
        ax1.set_xlabel(f"Fitted {UNITS[tgt]}"); ax1.set_ylabel("Residual")
        ax1.grid(alpha=0.3)
        ax2.hist(resid, bins=40, color=TGT_COLOR[tgt], edgecolor="white", alpha=0.8)
        ax2.set_title(f"{tgt} — Residual Distribution\nmean={resid.mean():.4f} σ={resid.std():.4f}",
                      fontweight="bold")
        ax2.set_xlabel("Residual"); ax2.set_ylabel("Count"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ts06_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts06_residuals.png")


def plot_comparison(test, ap):
    n    = len(test)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison — Test Set Performance", fontsize=13, fontweight="bold")
    bc   = [MODEL_COLORS[m] for m in MODEL_LIST]
    x    = np.arange(len(MODEL_LIST))
    for ci, tgt in enumerate(TARGETS):
        yt   = test[tgt].values
        r2s  = [r2_score(yt[~np.isnan(_get(ap,tgt,m,n))],
                         _get(ap,tgt,m,n)[~np.isnan(_get(ap,tgt,m,n))])
                if (~np.isnan(_get(ap,tgt,m,n))).sum()>5 else 0 for m in MODEL_LIST]
        rmses= [np.sqrt(mean_squared_error(yt[~np.isnan(_get(ap,tgt,m,n))],
                                           _get(ap,tgt,m,n)[~np.isnan(_get(ap,tgt,m,n))]))
                if (~np.isnan(_get(ap,tgt,m,n))).sum()>5 else 0 for m in MODEL_LIST]
        for ri, (vals, lbl) in enumerate([(r2s,"R²"),(rmses,f"RMSE {UNITS[tgt]}")]):
            ax = axes[ri, ci]
            bars = ax.bar(x, vals, color=bc, width=0.55, edgecolor="white", zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(MODEL_LIST, fontsize=10)
            ax.set_ylabel(lbl); ax.set_title(f"{tgt} — {lbl}", fontweight="bold")
            ax.grid(axis="y", alpha=0.3, zorder=0)
            mx = max(vals) if max(vals)>0 else 1
            ax.set_ylim(0, mx*1.35)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+mx*0.015,
                        f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/ts07_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: output/ts07_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — SAVE PREDICTIONS CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_results(test, ap, rf_models, X_te):
    n = len(test)
    lo_si,   hi_si   = rf_pi(rf_models["HM_SI"],   X_te)
    lo_temp, hi_temp = rf_pi(rf_models["HM_TEMP"], X_te)
    pd.DataFrame({
        "Timestamp"      : test["CLOCK"].values,
        "HM_SI_Actual"   : test["HM_SI"].values,
        "HM_SI_ARIMAX"   : _get(ap,"HM_SI","ARIMAX",n),
        "HM_SI_VAR"      : _get(ap,"HM_SI","VAR",n),
        "HM_SI_GBM"      : _get(ap,"HM_SI","GBM",n),
        "HM_SI_RF"       : _get(ap,"HM_SI","RF",n),
        "HM_SI_PI_lo"    : lo_si,
        "HM_SI_PI_hi"    : hi_si,
        "HM_TEMP_Actual" : test["HM_TEMP"].values,
        "HM_TEMP_ARIMAX" : _get(ap,"HM_TEMP","ARIMAX",n),
        "HM_TEMP_VAR"    : _get(ap,"HM_TEMP","VAR",n),
        "HM_TEMP_GBM"    : _get(ap,"HM_TEMP","GBM",n),
        "HM_TEMP_RF"     : _get(ap,"HM_TEMP","RF",n),
        "HM_TEMP_PI_lo"  : lo_temp,
        "HM_TEMP_PI_hi"  : hi_temp,
    }).to_csv("output/ts_predictions.csv", index=False)
    print("\n    Saved: output/ts_predictions.csv")

    print("\n" + "="*70)
    print(f"  {'Model':<8} {'SI R²':>7} {'SI RMSE':>9} {'SI MAE':>8}"
          f" | {'T R²':>7} {'T RMSE':>9} {'T MAE':>8}")
    print("  " + "─"*66)
    for m in MODEL_LIST:
        ps = _get(ap,"HM_SI",  m,n); pt = _get(ap,"HM_TEMP",m,n)
        mk = ~(np.isnan(ps)|np.isnan(pt))
        if mk.sum()==0: continue
        ys = test["HM_SI"].values[mk]; yt2 = test["HM_TEMP"].values[mk]
        print(f"  {m:<8}"
              f" {r2_score(ys,ps[mk]):>7.4f} {np.sqrt(mean_squared_error(ys,ps[mk])):>9.5f}"
              f" {mean_absolute_error(ys,ps[mk]):>8.5f}"
              f" | {r2_score(yt2,pt[mk]):>7.4f} {np.sqrt(mean_squared_error(yt2,pt[mk])):>9.4f}"
              f" {mean_absolute_error(yt2,pt[mk]):>8.4f}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — REAL-TIME INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_next_tap(state, feat_cols, gbm_models, rf_models):
    """
    Real-time prediction from a dict of current process conditions.
    Returns GBM point prediction + RF 95% prediction interval + spec alert.
    """
    row  = np.array([[state.get(f, 0.0) for f in feat_cols]])
    spec = {"HM_SI": SI_SPEC, "HM_TEMP": TEMP_SPEC}
    out  = {}
    for tgt in TARGETS:
        pred     = float(gbm_models[tgt].predict(row)[0])
        lo, hi   = rf_pi(rf_models[tgt], row)
        lo, hi   = float(lo[0]), float(hi[0])
        lo_s, hi_s = spec[tgt]
        out[tgt] = {
            "GBM_pred"  : round(pred, 4),
            "PI_lo_95"  : round(lo, 4),
            "PI_hi_95"  : round(hi, 4),
            "spec"      : spec[tgt],
            "status"    : "✓ In Spec" if lo_s <= pred <= hi_s else "⚠ OUT OF SPEC",
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    hm     = load_hm()
    para   = load_para()
    burden = load_burden()

    df      = build_dataset(hm, para, burden)
    feat_df = engineer_features(df)
    stationarity_tests(feat_df)

    train, test = chronological_split(feat_df)
    feat_cols   = get_feature_cols(feat_df)

    sc   = RobustScaler()
    X_tr = sc.fit_transform(train[feat_cols].fillna(0))
    X_te = sc.transform(test[feat_cols].fillna(0))
    joblib.dump(sc, "models/ts_scaler.pkl")

    print("\n[Plotting ACF/PACF ...]")
    plot_acf_pacf(train)

    exog_avail = [c for c in EXOG_COLS if c in train.columns]

    p_arimax          = run_arimax(train, test, exog_avail)
    p_var, _          = run_var(train, test)
    gbm_mdls, p_gbm   = run_gbm(X_tr, X_te, train, test, feat_cols)
    rf_mdls,  p_rf    = run_rf(X_tr,  X_te, train, test, feat_cols)
    p_lstm            = run_lstm(train, test)

    ap = {}
    for tgt in TARGETS:
        ap[tgt] = {
            "ARIMAX" : np.asarray(p_arimax.get(tgt, np.full(len(test),np.nan)), dtype=float),
            "VAR"    : np.asarray(p_var.get(tgt,    np.full(len(test),np.nan)), dtype=float),
            "GBM"    : p_gbm[tgt],
            "RF"     : p_rf[tgt],
        }
        if p_lstm and tgt in p_lstm:
            pad = np.full(len(test), np.nan)
            arr = np.asarray(p_lstm[tgt], dtype=float)
            pad[:len(arr)] = arr
            ap[tgt]["LSTM"] = pad

    print("\n[Generating plots ...]")
    plot_timeseries(test, ap)
    plot_scatter(test, ap)
    plot_pi(test, X_te, rf_mdls)
    plot_feat_importance(gbm_mdls, feat_cols)
    plot_residuals(test, ap)
    plot_comparison(test, ap)
    save_results(test, ap, rf_mdls, X_te)

    print("\n[Demo: predict_next_tap()]")
    demo = {f: float(test[f].iloc[0]) if f in test.columns else 0.0
            for f in feat_cols}
    for tgt, res in predict_next_tap(demo, feat_cols, gbm_mdls, rf_mdls).items():
        print(f"  {tgt}: Pred={res['GBM_pred']}  "
              f"PI=[{res['PI_lo_95']}, {res['PI_hi_95']}]  {res['status']}")

    print("\n✓ Done.  Outputs → ./output/    Models → ./models/")


if __name__ == "__main__":
    main()