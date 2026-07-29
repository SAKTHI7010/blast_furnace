# 🔥 Blast Furnace Intelligence Platform

> **AI-Powered Multi-Pipeline Prediction Platform for JSW Blast Furnace**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blast-furnace.streamlit.app)

---

## 📋 Overview

A full-stack **Machine Learning platform** for blast furnace operations with 3 dedicated AI prediction pipelines:

1. **📈 Regression ML Pipeline**: Predicts continuous values for **Silicon Content (`HM_SI`)** (%Si) and **Hot Metal Temperature (`HM_TEMP`)** (°C) using `PARA` + `BURDEN` inputs.
2. **🏷️ Classification Pipeline**: Classifies operational quality grades into **`Low` / `Normal` / `High`** for `HM_SI` and `HM_TEMP` with confidence probabilities.
3. **⏱️ Time-Series Pipeline**: Uses previous hot metal state (`HM_SI_lag1`, `HM_TEMP_lag1`) + `PARA` + `BURDEN` to forecast next-tap **Future `HM_SI`** and **Future `HM_TEMP`**.

---

## 🤖 ML Pipeline Architecture

| Pipeline | Inputs | Target Variables | Models Evaluated & Best Selected |
| :--- | :--- | :--- | :--- |
| **Regression** | `PARA` + `BURDEN` | `HM_SI`, `HM_TEMP` | XGBoost / Random Forest / Gradient Boosting Regressors |
| **Classification** | `PARA` + `BURDEN` | `HM_SI Class`, `HM_TEMP Class` | XGBoost / Random Forest / Gradient Boosting Classifiers |
| **Time-Series** | Previous `HM` + `PARA` + `BURDEN` | Future `HM_SI`, Future `HM_TEMP` | Autoregressive XGBoost / Random Forest / Gradient Boosting |

---

## 🗂️ Project Structure

```
blast/
├── app.py                   # 🏠 Home page (Streamlit entry point)
├── pages/
│   ├── 1_Prediction.py      # 🎯 Multi-Pipeline Engine (Regression, Classification, Time-Series tabs)
│   ├── 2_Analysis.py        # 📊 EDA & Pipeline Performance Analysis
│   └── 3_Trade_Off.py       # ⚖️ Trade-Off Optimization & What-If Simulator
├── utils/
│   ├── predictor.py         # Multi-pipeline loading & inference engine
│   └── __init__.py
├── models/                  # Trained model artifacts (.pkl)
├── train_models.py          # 🔬 Multi-pipeline training script
├── requirements.txt
├── .streamlit/config.toml   # Theme configuration
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

### 2. Train Models (Run Locally)
```bash
python train_models.py
```
This script will:
- Load and merge `PARA.xlsx`, `BURDEN.xlsx` (with 8h descent lag), and `HM_ANALYSIS.xls`.
- Train candidate models (Random Forest, Gradient Boosting, XGBoost) for each pipeline.
- Evaluate metrics (R², MAE, RMSE, Accuracy, F1-Score) and pick the best performing models.
- Save model `.pkl` artifacts into `models/`.

### 3. Run Streamlit App
```bash
streamlit run app.py
```

---

## 📞 Contact

JSW Steel — Blast Furnace No.3 Operations & Analytics Team
