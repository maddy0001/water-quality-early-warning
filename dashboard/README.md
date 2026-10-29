# Water Quality Early Warning Dashboard

Streamlit dashboard for the paper *"An Integrated Machine Learning Framework for Real-Time Water
Quality Monitoring"* (ICESC 2026, paper ID 469). **Every page runs the real trained models**: there is
no simulated or random output.

## Before you start

The dashboard loads the models created by the training script. From the project folder:

```
python main.py --train          # trains classification, forecasting and anomaly models
cd dashboard
streamlit run app.py
```

The sidebar shows **"All trained models found"** when everything is in place. If a model is missing,
the sidebar and the page say which file is missing.

## Pages and which file to upload

| Page | What it runs | Upload | Ready-made sample (`data/samples/`) |
|---|---|---|---|
| 🔍 Water Classification | Ensemble RF 0.35 + XGBoost 0.40 + SVM 0.25 | CSV with the 9 Kaggle columns | `1_classification_test_set_570.csv` (the 570 held-out test rows, with true labels) |
| ⚠️ Anomaly Detection | Isolation Forest, LOF, One-Class SVM, Autoencoder, 2-of-4 vote | CSV with the 9 Kaggle columns | `2_anomaly_spike_test.csv` (40 clean + 10 spiked samples) |
| 🚨 Alert History | Classification + anomaly → fused alert (0.40 / 0.30 / 0.30) | CSV with the 9 Kaggle columns (+ optional `timestamp`, `location`) | `3_early_warning_batch.csv` |
| 📈 pH Forecasting | BiLSTM + Attention, 30-day window → next day | Choose a built-in USGS station, **or** a CSV with the 11 USGS sensor columns (≥ 30 rows) | `4_forecasting_usgs_station_2198840.csv` |

The 9 Kaggle columns: `ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon,
Trihalomethanes, Turbidity`. Extra columns are ignored; empty cells are filled with KNN imputation (k = 5).

The 11 USGS columns: `Specific_conductance_max, pH_max, pH_min, Specific_conductance_min,
Specific_conductance_mean, Dissolved_oxygen_max, Dissolved_oxygen_mean, Dissolved_oxygen_min,
Temperature_mean, Temperature_min, Temperature_max` (+ optional `pH_actual` to compare forecasts).

## Files

```
dashboard/
├── app.py                         # navigation, theme, model status
└── dashboard_modules/
    ├── engine.py                  # loads the trained models; shared prediction functions
    ├── home.py                    # project overview
    ├── classification.py          # ensemble classification
    ├── anomaly.py                 # 4-detector anomaly detection
    ├── forecasting.py             # BiLSTM next-day pH forecast
    └── alerts.py                  # fused early warning + alert log
```

## Notes

- LOF is not saved by the training pipeline (it runs in fit-predict mode), so the dashboard re-fits it on the
  same 2,847 training samples with the same settings when it first loads. With it, the detectors reproduce the
  published 120 flagged samples (4.21%).
- The USGS forecasting data is distributed normalised to 0–1; forecasts are shown on that scale, the same
  scale as the published RMSE (0.0524).
- Requires `pyarrow==25.0.1` with `numpy==1.24.3` (pyarrow 26+ needs NumPy 2).
