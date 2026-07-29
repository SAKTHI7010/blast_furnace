# -*- coding: utf-8 -*-
"""
generate_sensitivity_cache.py
-------------------------------------------------------
Standalone script to pre-compute sensitivity curves using
the already-trained regression models.
Run this once after training: python generate_sensitivity_cache.py
-------------------------------------------------------
"""
import os, sys, joblib, warnings
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = 'models'

print("=" * 60)
print("  PRE-COMPUTING SENSITIVITY CURVES FROM TRAINED MODELS")
print("=" * 60)

# Load metadata
meta_path = os.path.join(MODEL_DIR, 'feature_meta.pkl')
if not os.path.exists(meta_path):
    print("ERROR: feature_meta.pkl not found. Run train_models.py first.")
    sys.exit(1)

feature_meta = joblib.load(meta_path)

all_reg_feats_si   = feature_meta['regression']['HM_Si']['top_features']
all_reg_feats_temp = feature_meta['regression']['HM_Temp']['top_features']
all_sens_feats = list(dict.fromkeys(all_reg_feats_si + all_reg_feats_temp))

reg_model_si   = joblib.load(os.path.join(MODEL_DIR, 'model_regression_HM_Si.pkl'))
reg_model_temp = joblib.load(os.path.join(MODEL_DIR, 'model_regression_HM_Temp.pkl'))
reg_scaler_si  = joblib.load(os.path.join(MODEL_DIR, 'scaler_regression_HM_Si.pkl'))
reg_scaler_temp= joblib.load(os.path.join(MODEL_DIR, 'scaler_regression_HM_Temp.pkl'))

ranges_si   = feature_meta['regression']['HM_Si']['feature_ranges']
ranges_temp = feature_meta['regression']['HM_Temp']['feature_ranges']

# Base inputs (means)
base_inputs = {}
for f in all_sens_feats:
    info = ranges_si.get(f, ranges_temp.get(f, {'mean': 0.0}))
    base_inputs[f] = float(info.get('mean', 0.0))

NUM_POINTS = 25
sensitivity_cache = {}

for feat in all_sens_feats:
    f_info = ranges_si.get(feat, ranges_temp.get(feat, {'min': 0.0, 'max': 1.0}))
    min_v, max_v = float(f_info.get('min', 0)), float(f_info.get('max', 1))
    if min_v == max_v:
        max_v = min_v + 1.0
    grid = np.linspace(min_v, max_v, NUM_POINTS)
    records = []
    for val in grid:
        inp = {**base_inputs, feat: val}
        row_si   = np.array([[inp.get(f, 0) for f in all_reg_feats_si]], dtype=float)
        row_temp = np.array([[inp.get(f, 0) for f in all_reg_feats_temp]], dtype=float)
        try:
            si_p = float(reg_model_si.predict(reg_scaler_si.transform(row_si))[0])
        except Exception:
            si_p = None
        try:
            tp = float(reg_model_temp.predict(reg_scaler_temp.transform(row_temp))[0])
        except Exception:
            tp = None
        records.append({
            'FeatureValue': round(float(val), 4),
            'Predicted_HM_SI':   round(si_p, 5) if si_p is not None else None,
            'Predicted_HM_TEMP': round(tp,   3) if tp  is not None else None,
        })
    sensitivity_cache[feat] = records
    print(f"  ✓ {feat}")

out_path = os.path.join(MODEL_DIR, 'sensitivity_cache.pkl')
joblib.dump(sensitivity_cache, out_path)
print(f"\n✅ Saved sensitivity cache → {out_path}")
print(f"   Features cached: {len(sensitivity_cache)} | Points per curve: {NUM_POINTS}")
print("=" * 60)
