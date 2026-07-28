"""
utils/predictor.py
Load pre-trained models and run inference for the Blast Furnace app.
"""
import os, joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

TARGETS = ['HM_Si', 'HM_Temp', 'Prod_Rate']
TARGET_LABELS = {
    'HM_Si':     'Silicon Content (HM_Si)',
    'HM_Temp':   'Hot Metal Temperature',
    'Prod_Rate': 'Production Rate',
}
TARGET_UNITS = {
    'HM_Si':     '%',
    'HM_Temp':   '°C',
    'Prod_Rate': 't/hr',
}


@st.cache_resource(show_spinner="Loading models...")
def load_all_models():
    """Load all model artifacts from models/ directory."""
    meta_path = os.path.join(MODEL_DIR, 'feature_meta.pkl')
    if not os.path.exists(meta_path):
        return None, None, None, None, None

    feature_meta = joblib.load(meta_path)
    models  = {}
    scalers = {}
    imputers = {}

    for tgt in TARGETS:
        m_path = os.path.join(MODEL_DIR, f'model_{tgt}.pkl')
        s_path = os.path.join(MODEL_DIR, f'scaler_{tgt}.pkl')
        i_path = os.path.join(MODEL_DIR, f'imputer_{tgt}.pkl')

        if os.path.exists(m_path):
            models[tgt]   = joblib.load(m_path)
            scalers[tgt]  = joblib.load(s_path)
            imputers[tgt] = joblib.load(i_path)

    return feature_meta, models, scalers, imputers


def predict_single(tgt_name, input_values: dict,
                   feature_meta, models, scalers, imputers) -> dict:
    """
    Predict a single target given feature values.

    Parameters
    ----------
    tgt_name : str, e.g. 'HM_Si'
    input_values : dict  { feature_name: value, ... }

    Returns
    -------
    dict with keys: prediction, unit, features_used
    """
    if tgt_name not in models:
        return {'error': f'Model for {tgt_name} not found'}

    top_feats = feature_meta['top_features'][tgt_name]

    # Build input row in correct feature order
    row = np.array([[input_values.get(f, np.nan) for f in top_feats]], dtype=float)

    # Impute → scale → predict
    row_imp  = imputers[tgt_name].transform(row)
    row_sc   = scalers[tgt_name].transform(row_imp)
    pred     = models[tgt_name].predict(row_sc)[0]

    return {
        'prediction':   float(pred),
        'unit':         TARGET_UNITS.get(tgt_name, ''),
        'features_used': top_feats,
    }


def get_feature_contributions(tgt_name, input_values: dict,
                               feature_meta, models, scalers, imputers) -> pd.DataFrame:
    """
    Approximate feature contributions using permutation/delta approach.
    Returns a DataFrame with feature names and estimated contributions.
    """
    top_feats = feature_meta['top_features'].get(tgt_name, [])
    ranges    = feature_meta['feature_ranges'].get(tgt_name, {})

    baseline_vals = {f: ranges[f]['mean'] for f in top_feats if f in ranges}
    baseline_pred = predict_single(tgt_name, baseline_vals,
                                   feature_meta, models, scalers, imputers)['prediction']
    full_pred     = predict_single(tgt_name, input_values,
                                   feature_meta, models, scalers, imputers)['prediction']

    contributions = []
    for f in top_feats:
        mix = {**baseline_vals, f: input_values.get(f, baseline_vals.get(f, 0))}
        p   = predict_single(tgt_name, mix,
                             feature_meta, models, scalers, imputers)['prediction']
        contributions.append({'Feature': f, 'Contribution': p - baseline_pred})

    df = pd.DataFrame(contributions).sort_values('Contribution', key=abs, ascending=False)
    return df
