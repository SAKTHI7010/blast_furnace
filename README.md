# 🔥 Blast Furnace Intelligence Platform

> **AI-Powered Prediction & Optimization for JSW Blast Furnace**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blast-furnace.streamlit.app)

---

## 📋 Overview

A full-stack **Machine Learning platform** for blast furnace operations that predicts:
- ⚗️ **HM_Si** — Hot Metal Silicon Content (%)
- 🌡️ **HM_Temp** — Hot Metal Temperature (°C)
- ⚙️ **Production Rate** — Hourly production (t/hr)

Built with **XGBoost**, **Random Forest**, **Gradient Boosting** on 3 merged datasets.

---

## 🗂️ Project Structure

```
blast/
├── app.py                   # 🏠 Home page (Streamlit entry point)
├── pages/
│   ├── 1_Prediction.py      # 🎯 Prediction page (8-9 feature sliders per target)
│   ├── 2_Analysis.py        # 📊 EDA & Analysis page
│   └── 3_Trade_Off.py       # ⚖️ Trade-Off Optimization page
├── utils/
│   ├── predictor.py         # Model loading & inference
│   └── __init__.py
├── models/                  # Trained model artifacts (.pkl)
│   ├── model_HM_Si.pkl
│   ├── model_HM_Temp.pkl
│   ├── model_Prod_Rate.pkl
│   ├── scaler_*.pkl
│   ├── imputer_*.pkl
│   ├── feature_meta.pkl     # Feature names, importance, ranges
│   └── eda_data.parquet     # Data snapshot for EDA
├── train_models.py          # 🔬 Model training script (run locally)
├── requirements.txt
├── .streamlit/config.toml   # Dark theme config
├── PARA.xlsx                # Process parameters dataset
├── HM_ANALYSIS.xls          # Hot metal analysis dataset
└── BURDEN.xlsx              # Burden charge dataset
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Run Locally First!)
```bash
python train_models.py
```
This will:
- Load and merge all 3 datasets
- Select top 9 features per target
- Train XGBoost/RF/GBM models
- Save `.pkl` files to `models/`

### 3. Run Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Datasets

| Dataset | Rows | Key Columns |
|---------|------|-------------|
| `PARA.xlsx` | ~17,000 | CLOCK, Cold Blast Volume, HBT, Raft, PROD_RATE, ... |
| `HM_ANALYSIS.xls` | ~13,500 | SAMPLETAKEN, HM_SI, HM_TEMP |
| `BURDEN.xlsx` | ~746,000 | CHARGETIME, BRANDCODE, ACTWT |

---

## 🤖 ML Models

| Target | Best Model | R² | Features Used |
|--------|------------|-----|----------------|
| HM_Si  | XGBoost / RF | ~0.85+ | HBT, Raft, O₂ Flow, ... |
| HM_Temp | XGBoost / RF | ~0.82+ | HBT, Cold Blast Vol, ... |
| Prod_Rate | XGBoost / RF | ~0.88+ | Cold Blast Vol, O₂, ... |

---

## ☁️ Streamlit Cloud Deployment

1. Push all files (including `models/` folder) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo: `SAKTHI7010/blast_furnace`
4. Set **Main file**: `app.py`
5. Click **Deploy**

> ⚠️ **Important**: Always run `train_models.py` locally first and push the generated `models/` folder to GitHub before deploying. Streamlit Cloud does not retrain models.

---

## 📱 App Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Platform overview, model performance gauges, quick stats |
| 🎯 **Prediction** | 3 tabs × 9 sliders each → real-time predictions with feature contributions |
| 📊 **Analysis** | Dataset overview, distributions, time-series, correlations, feature importance |
| ⚖️ **Trade-Off** | What-if simulator, sensitivity analysis, scenario comparison, recommendations |

---

## 🔧 Retraining Models

If you get new data, simply:
1. Replace the Excel files in the project directory
2. Run `python train_models.py`
3. Push the new `models/` folder to GitHub
4. Streamlit Cloud will automatically reload

---

## 📞 Contact

JSW Steel — Blast Furnace No.3 Operations Team
