"""
utils/predictor.py
Load pre-trained models and run inference for:
1. Regression ML Pipeline (PARA + BURDEN -> HM_SI & HM_TEMP actual values)
2. Classification Pipeline (PARA + BURDEN -> HM_SI Class & HM_TEMP Class: Low / Normal / High)
3. Time-Series Pipeline (Previous HM + PARA + BURDEN -> Future HM_SI & Future HM_TEMP)
"""
import os, joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

TARGET_LABELS = {
    'HM_Si':   'Silicon Content (HM_Si)',
    'HM_Temp': 'Hot Metal Temperature (HM_Temp)',
}
TARGET_UNITS = {
    'HM_Si':   '%Si',
    'HM_Temp': '°C',
}

def _get_meta_target(meta_dict: dict, tgt: str) -> dict:
    """Helper for case-insensitive target lookup in metadata."""
    if tgt in meta_dict:
        return meta_dict[tgt]
    if tgt.upper() in meta_dict:
        return meta_dict[tgt.upper()]
    if tgt.lower() in meta_dict:
        return meta_dict[tgt.lower()]
    # Try replacing _Si with _SI or _Temp with _TEMP
    alt = tgt.replace('_Si', '_SI').replace('_Temp', '_TEMP')
    if alt in meta_dict:
        return meta_dict[alt]
    return {}

def _get_artifact(art_dict: dict, prefix: str, tgt: str):
    """Helper for case-insensitive model/scaler lookup."""
    keys_to_try = [
        f'{prefix}_{tgt}',
        f'{prefix}_{tgt.upper()}',
        f'{prefix}_{tgt.lower()}',
        f'{prefix}_{tgt.replace("_Si", "_SI").replace("_Temp", "_TEMP")}'
    ]
    for k in keys_to_try:
        if k in art_dict:
            return art_dict[k]
    return None

@st.cache_resource(show_spinner="Loading ML prediction pipelines...")
def load_all_pipelines():
    """Load all pipeline artifacts from models/ directory."""
    meta_path = os.path.join(MODEL_DIR, 'feature_meta.pkl')
    if not os.path.exists(meta_path):
        return None

    try:
        feature_meta = joblib.load(meta_path)
    except Exception:
        return None

    artifacts = {
        'meta': feature_meta,
        'imputers': {},
        'scalers': {},
        'models': {},
        'encoders': {}
    }

    # Imputers
    for pipe in ['regression', 'classification', 'timeseries']:
        imp_p = os.path.join(MODEL_DIR, f'imputer_{pipe}.pkl')
        if os.path.exists(imp_p):
            artifacts['imputers'][pipe] = joblib.load(imp_p)

    # Load all models, scalers, encoders
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith('.pkl'):
            continue
        full_p = os.path.join(MODEL_DIR, fname)
        key = fname.replace('.pkl', '')
        if fname.startswith('model_'):
            artifacts['models'][key.replace('model_', '')] = joblib.load(full_p)
        elif fname.startswith('scaler_'):
            artifacts['scalers'][key.replace('scaler_', '')] = joblib.load(full_p)
        elif fname.startswith('encoder_'):
            artifacts['encoders'][key.replace('encoder_', '')] = joblib.load(full_p)

    return artifacts


def predict_regression(artifacts: dict, input_values: dict) -> dict:
    """Run Regression ML Pipeline for HM_SI and HM_TEMP."""
    meta = artifacts['meta']['regression']
    results = {}

    for tgt in ['HM_Si', 'HM_Temp']:
        tgt_meta = _get_meta_target(meta, tgt)
        top_feats = tgt_meta.get('top_features', [])
        ranges    = tgt_meta.get('feature_ranges', {})

        row = np.array([[input_values.get(f, ranges.get(f, {}).get('mean', 0.0)) for f in top_feats]], dtype=float)

        scaler = _get_artifact(artifacts['scalers'], 'regression', tgt)
        model  = _get_artifact(artifacts['models'], 'regression', tgt)

        if scaler is None or model is None:
            continue

        row_sc = scaler.transform(row)
        pred   = float(model.predict(row_sc)[0])

        results[tgt] = {
            'prediction': pred,
            'unit': TARGET_UNITS[tgt],
            'label': TARGET_LABELS[tgt],
            'top_features': top_feats,
            'best_model': tgt_meta.get('best_model', 'Best ML Model'),
            'metrics': tgt_meta.get('metrics', {}).get(tgt_meta.get('best_model', ''), {})
        }

    return results


def predict_classification(artifacts: dict, input_values: dict) -> dict:
    """Run Classification Pipeline for HM_SI Class and HM_TEMP Class."""
    meta = artifacts['meta']['classification']
    results = {}

    for tgt in ['HM_Si', 'HM_Temp']:
        tgt_meta = _get_meta_target(meta, tgt)
        top_feats = tgt_meta.get('top_features', [])
        ranges    = tgt_meta.get('feature_ranges', {})

        row = np.array([[input_values.get(f, ranges.get(f, {}).get('mean', 0.0)) for f in top_feats]], dtype=float)

        scaler  = _get_artifact(artifacts['scalers'], 'classification', tgt)
        model   = _get_artifact(artifacts['models'], 'classification', tgt)
        encoder = _get_artifact(artifacts['encoders'], 'classification', tgt)

        if scaler is None or model is None or encoder is None:
            continue

        row_sc    = scaler.transform(row)
        pred_idx  = model.predict(row_sc)[0]
        pred_cls  = encoder.inverse_transform([pred_idx])[0]

        probs_dict = {}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(row_sc)[0]
            classes = encoder.classes_
            for c_name, p_val in zip(classes, probs):
                probs_dict[str(c_name)] = float(p_val)

        results[tgt] = {
            'predicted_class': str(pred_cls),
            'probabilities': probs_dict,
            'label': f"{TARGET_LABELS[tgt]} Class",
            'top_features': top_feats,
            'best_model': tgt_meta.get('best_model', 'Best ML Model'),
            'metrics': tgt_meta.get('metrics', {}).get(tgt_meta.get('best_model', ''), {})
        }

    return results


def predict_timeseries(artifacts: dict, input_values: dict) -> dict:
    """Run Time-Series Pipeline for Future HM_SI and Future HM_TEMP."""
    meta = artifacts['meta']['timeseries']
    results = {}

    for tgt in ['HM_Si', 'HM_Temp']:
        tgt_meta = _get_meta_target(meta, tgt)
        top_feats = tgt_meta.get('top_features', [])
        ranges    = tgt_meta.get('feature_ranges', {})

        row = np.array([[input_values.get(f, ranges.get(f, {}).get('mean', 0.0)) for f in top_feats]], dtype=float)

        scaler = _get_artifact(artifacts['scalers'], 'timeseries', tgt)
        model  = _get_artifact(artifacts['models'], 'timeseries', tgt)

        if scaler is None or model is None:
            continue

        row_sc = scaler.transform(row)
        pred   = float(model.predict(row_sc)[0])

        results[tgt] = {
            'prediction': pred,
            'unit': TARGET_UNITS[tgt],
            'label': f"Future {TARGET_LABELS[tgt]}",
            'top_features': top_feats,
            'best_model': tgt_meta.get('best_model', 'Best ML Model'),
            'metrics': tgt_meta.get('metrics', {}).get(tgt_meta.get('best_model', ''), {})
        }

    return results


def get_feature_contributions(artifacts: dict, pipeline_type: str, tgt: str, input_values: dict) -> pd.DataFrame:
    """Compute feature contribution estimates via baseline diff."""
    meta = artifacts['meta'][pipeline_type]
    tgt_meta = _get_meta_target(meta, tgt)
    top_feats = tgt_meta.get('top_features', [])
    ranges    = tgt_meta.get('feature_ranges', {})

    baseline_vals = {f: ranges[f]['mean'] for f in top_feats if f in ranges}
    
    if pipeline_type == 'regression':
        base_res = predict_regression(artifacts, baseline_vals)
        base_pred = base_res[tgt]['prediction'] if tgt in base_res else 0.0
        
        contributions = []
        for f in top_feats:
            mix = {**baseline_vals, f: input_values.get(f, baseline_vals.get(f, 0))}
            mix_res = predict_regression(artifacts, mix)
            p = mix_res[tgt]['prediction'] if tgt in mix_res else base_pred
            contributions.append({'Feature': f, 'Contribution': p - base_pred})
    elif pipeline_type == 'timeseries':
        base_res = predict_timeseries(artifacts, baseline_vals)
        base_pred = base_res[tgt]['prediction'] if tgt in base_res else 0.0
        
        contributions = []
        for f in top_feats:
            mix = {**baseline_vals, f: input_values.get(f, baseline_vals.get(f, 0))}
            mix_res = predict_timeseries(artifacts, mix)
            p = mix_res[tgt]['prediction'] if tgt in mix_res else base_pred
            contributions.append({'Feature': f, 'Contribution': p - base_pred})
    else:
        contributions = []
        for f in top_feats:
            diff = input_values.get(f, ranges.get(f, {}).get('mean', 0)) - ranges.get(f, {}).get('mean', 0)
            contributions.append({'Feature': f, 'Contribution': diff})

    df = pd.DataFrame(contributions)
    if not df.empty:
        df = df.sort_values('Contribution', key=abs, ascending=False)
    return df


def compute_feature_sensitivity(artifacts: dict, feature_name: str, num_points: int = 30) -> pd.DataFrame:
    """Compute model prediction trade-off sensitivity curves as feature_name varies."""
    meta = artifacts['meta']['regression']
    tgt_meta_si = _get_meta_target(meta, 'HM_Si')
    tgt_meta_temp = _get_meta_target(meta, 'HM_Temp')

    ranges_si = tgt_meta_si.get('feature_ranges', {})
    ranges_temp = tgt_meta_temp.get('feature_ranges', {})

    # Combine top features
    all_feats = list(dict.fromkeys(tgt_meta_si.get('top_features', []) + tgt_meta_temp.get('top_features', [])))

    # Get feature range
    f_info = ranges_si.get(feature_name, ranges_temp.get(feature_name, {'min': 0.0, 'max': 100.0, 'mean': 50.0}))
    min_val, max_val = float(f_info['min']), float(f_info['max'])

    # Baseline input values using mean values
    base_inputs = {}
    for f in all_feats:
        info = ranges_si.get(f, ranges_temp.get(f, {'mean': 50.0}))
        base_inputs[f] = float(info['mean'])

    grid_vals = np.linspace(min_val, max_val, num_points)
    records = []

    for val in grid_vals:
        curr_inputs = {**base_inputs, feature_name: val}
        preds = predict_regression(artifacts, curr_inputs)
        si_val = preds['HM_Si']['prediction'] if 'HM_Si' in preds else np.nan
        temp_val = preds['HM_Temp']['prediction'] if 'HM_Temp' in preds else np.nan
        records.append({
            'FeatureValue': val,
            'Predicted_HM_SI': si_val,
            'Predicted_HM_TEMP': temp_val,
        })

    return pd.DataFrame(records)

