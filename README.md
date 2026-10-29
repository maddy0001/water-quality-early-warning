# 🚰 Water Contamination Early Warning System

**AI-Powered Real-Time Water Quality Monitoring and Contamination Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Overview

A comprehensive AI system for water quality monitoring that integrates:
- **Classification Models** for potability prediction
- **Time Series Forecasting** for pH trend prediction
- **Anomaly Detection** for contamination events
- **Explainability Analysis** with SHAP
- **4-Level Alert System** for real-time warnings

**M.Tech Project | Published at ICESC 2026 (IEEE), Paper ID 469:**
*"An Integrated Machine Learning Framework for Real-Time Water Quality Monitoring"*

---

## 🎯 Key Features

### 🤖 Machine Learning Models
- **Classification**: Random Forest, XGBoost, SVM, Ensemble
- **Forecasting**: BiLSTM-Attention, GRU-Attention
- **Anomaly Detection**: Isolation Forest, LOF, One-Class SVM, Autoencoder

### 🚨 Early Warning System
- 4-Level alerts: NORMAL → LOW → MEDIUM → CRITICAL
- Weighted voting from multiple models
- Real-time monitoring capability (~0.5 ms per sample)
- Alert logging and notification hooks

### 📊 Explainability
- SHAP value analysis for feature importance
- Individual prediction explanations
- Feature correlation analysis
- Visual interpretation tools

### 📈 Visualization
- 20+ publication-quality plots (300 DPI)
- Confusion matrices, ROC curves
- Time series forecasts, residual analysis
- LaTeX tables for IEEE paper

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd water_contamination_early_warning_system

# Install dependencies
pip install -r requirements.txt

# Optional (for full functionality)
pip install xgboost imbalanced-learn shap
```

### Run Complete System

```bash
# Option 1: Full system (all pipelines)
python main.py --full

# Option 2: Quick demo
python main.py --demo

# Option 3: Train models only
python main.py --train

# Option 4: Generate evaluation plots only
python main.py --evaluate

# Option 5: Show system info
python main.py --info
```

---

## 📂 Project Structure

```
water_contamination_early_warning_system/
│
├── main.py                          # Main entry point ⭐
├── requirements.txt                 # Dependencies
├── README.md                        # This file
│
├── data/                            # Datasets
│   ├── raw/                        # Original data
│   │   ├── water_potability.csv   # Classification dataset
│   │   └── water_dataset.mat      # Forecasting dataset
│   └── processed/                  # Preprocessed data
│
├── models/                          # Trained models
│   ├── classification/             # RF, XGBoost, SVM
│   ├── forecasting/                # BiLSTM, GRU
│   └── anomaly/                    # IF, LOF, OCSVM, AE
│
├── outputs/                         # Results
│   ├── plots/                      # Visualizations
│   ├── metrics/                    # Performance metrics
│   └── logs/                       # Execution logs
│
├── src/                             # Source code
│   ├── config.py                   # Configuration
│   ├── data/                       # Data handling
│   ├── models/                     # Model implementations
│   ├── early_warning/              # Alert system
│   ├── explainability/             # SHAP analysis
│   ├── visualization/              # Plotting functions
│   └── utils/                      # Utilities
│
└── pipelines/                       # End-to-end workflows
    ├── pipeline_classification.py
    ├── pipeline_forecasting.py
    ├── pipeline_anomaly.py
    ├── pipeline_explainability.py
    ├── pipeline_early_warning.py
    └── pipeline_evaluation.py
```

---

## 🔬 System Components

### 1. Classification Models

**Purpose**: Predict water potability (safe/unsafe)

Held-out test set of 570 real samples (SMOTE applied to the training split only).

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 63.33% | 50.82% | 66.36% |
| XGBoost | 59.82% | 51.59% | 65.01% |
| SVM | 60.53% | 51.82% | 64.12% |
| **Ensemble (0.35 / 0.40 / 0.25)** | **61.75%** | **53.02%** | **66.62%** |

**Features**: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity

### 2. Time Series Forecasting

**Purpose**: Predict next-day pH (1 day ahead)

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| BiLSTM-Attention | 0.0524 | 0.0336 | 19.36% | 0.5251 |
| GRU-Attention | 0.0610 | 0.0396 | 20.55% | 0.3572 |

Errors are on the dataset's normalised (0–1) pH scale, over 9,324 test sequences.

**Architecture**: Bidirectional LSTM with attention mechanism (30-day lookback)

### 3. Anomaly Detection

**Purpose**: Detect contamination events

The dataset has no contamination labels, so precision and recall cannot be measured directly.

| Check | Result |
|-------|--------|
| Samples flagged (≥ 2 of 4 detectors agree) | 120 of 2,847 (4.21%), against an assumed 5% rate |
| Synthetic contamination test (2 parameters shifted ±3 SD) | 75.6% caught, 5.4% false alarms |

**Strategy**: 4 detectors with 2-of-4 quorum voting

### 4. Early Warning System

**Alert Levels**:
- 🟢 **NORMAL**: Water quality safe
- 🟡 **LOW**: Minor deviation, monitor
- 🟠 **MEDIUM**: Significant concern, investigate
- 🔴 **CRITICAL**: Immediate action required

**Aggregation**: Weighted voting (Classification: 40%, Forecasting: 30%, Anomaly: 30%)

---

## 📊 Usage Examples

### Train Individual Models

```python
# Classification
from pipelines.pipeline_classification import run_classification_pipeline
results = run_classification_pipeline(save_models=True)

# Forecasting
from pipelines.pipeline_forecasting import run_forecasting_pipeline
results = run_forecasting_pipeline(save_models=True)

# Anomaly Detection
from pipelines.pipeline_anomaly import run_anomaly_pipeline
results = run_anomaly_pipeline(save_models=True)
```

### Real-Time Monitoring

```python
from early_warning.monitoring_engine import WaterQualityMonitor

# Initialize monitor
monitor = WaterQualityMonitor()
monitor.load_models()

# Check water sample
sample = [7.2, 180.5, 15000, 6.8, 250, 420, 12.5, 60, 3.2]  # 9 features
alert = monitor.check_sample(sample, location='Site_A')

print(f"Alert: {alert.level.name}")
print(f"Message: {alert.get_message()}")
```

### Generate Explainability

```python
from pipelines.pipeline_explainability import run_explainability_pipeline
results = run_explainability_pipeline(generate_plots=True)

# Top 5 features
print(results['rf_importance'].head(5))
```

### Batch Monitoring

```python
from early_warning.monitoring_engine import WaterQualityMonitor

monitor = WaterQualityMonitor()
monitor.load_models()

# Monitor 100 samples
alerts = monitor.monitor_batch(samples_df, verbose=True)

# Summary
critical = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
print(f"Critical alerts: {critical}")
```

---

## 🖥️ Dashboard

```bash
python main.py --train        # once: creates the trained models the dashboard loads
cd dashboard
streamlit run app.py
```

Every page runs the trained models. Sample files for each page are in `data/samples/`
(see `dashboard/README.md` for which file goes on which page).

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model parameters
CLASSIFICATION_CONFIG = {
    'rf_n_estimators': 200,
    'xgb_n_estimators': 150,
    'svm_C': 10,
    # ...
}

# Alert thresholds
CLASSIFICATION_CRITICAL = 0.9    # 90%+ confidence → CRITICAL
FORECASTING_CRITICAL = 0.15      # pH deviation >0.15 → CRITICAL
ANOMALY_CRITICAL = 4             # All 4 detectors → CRITICAL

# Data paths
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
MODELS_DIR = BASE_DIR / 'models'
PLOTS_DIR = BASE_DIR / 'outputs' / 'plots'
```

---

## 📈 Performance Metrics

### Overall System
- **Classification (Ensemble)**: 61.75% accuracy, 66.62% ROC-AUC, 53.02% F1
- **Forecasting (BiLSTM-Attention)**: RMSE 0.0524, R² 0.5251
- **Anomaly Detection**: 4.21% flag rate; 75.6% of synthetic contamination caught
- **Processing Speed**: ~0.05 s per 100 samples
- **Alert Generation**: ~0.02 s

### Datasets
- **Classification**: 3,276 samples (Kaggle Water Potability dataset)
- **Forecasting**: 705 days × 37 USGS monitoring stations, Georgia (UCI Water Quality dataset)
- **Train/Test Split**: 80/20 stratified

---

## 📊 Generated Outputs

### Plots (outputs/plots/)
- **Classification**: 10 plots (confusion matrices, ROC curves, metrics comparison)
- **Forecasting**: 4 plots (predictions, time series, residuals, error analysis)
- **System**: 2 plots (performance summary, computation time)
- **Explainability**: 5 plots (SHAP, feature importance, correlations)

### Metrics (outputs/metrics/)
- `classification_metrics.json`
- `forecasting_metrics.json`
- `anomaly_metrics.json`
- `feature_importance_summary.json`
- `evaluation_summary.json`

### LaTeX Tables
- `results_tables.tex` (ready for IEEE paper)

---

## 🛠️ Dependencies

### Required
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
tensorflow>=2.10.0
```

### Optional (for full functionality)
```
xgboost>=1.5.0              # XGBoost model
imbalanced-learn>=0.9.0     # SMOTE balancing
shap>=0.40.0                # Explainability
```

---

## 🎓 For IEEE Paper

### Results Tables
Generated LaTeX tables available in:
- `outputs/plots/evaluation/results_tables.tex`

### Figures
All plots are 300 DPI, publication-ready:
- Confusion matrices
- ROC/PR curves
- Feature importance
- Time series forecasts
- System performance summary

### Key Metrics to Report
- **Best Classification Model**: Ensemble (F1 = 53.02%, AUC = 66.62%)
- **Best Forecasting Model**: BiLSTM-Attention (RMSE = 0.0524, R² = 0.5251)
- **Anomaly Detection**: 2-of-4 quorum, 4.21% flag rate
- **Real-time Capability**: ~0.5 ms per sample

---

## 🔬 System Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    WATER QUALITY DATA                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │        DATA PREPROCESSING             │
        │  • Imputation • Scaling • Balancing   │
        └───────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │CLASSIFICATION│  │ FORECASTING │  │  ANOMALY    │
    │  (RF+XGB+   │  │ (BiLSTM+GRU)│  │ (IF+LOF+    │
    │   SVM)      │  │             │  │  OCSVM+AE)  │
    └─────────────┘  └─────────────┘  └─────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
            ┌───────────────────────────────────┐
            │     ALERT AGGREGATION             │
            │  (Weighted Voting: 40/30/30)      │
            └───────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────────┐
            │    EARLY WARNING ALERT            │
            │  NORMAL/LOW/MEDIUM/CRITICAL       │
            └───────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────────┐
            │  NOTIFICATION & LOGGING           │
            │  • Alert history • Dashboard      │
            └───────────────────────────────────┘
```

---

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@inproceedings{icesc2026_469,
  title={An Integrated Machine Learning Framework for Real-Time Water Quality Monitoring},
  author={Madhusuthanan G and Vijayalakshmi V},
  booktitle={International Conference on Electronics and Sustainable Communication Systems (ICESC)},
  year={2026}
}
```

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### XGBoost issues
```bash
pip install xgboost --upgrade
```

### TensorFlow GPU not detected
```bash
pip install tensorflow-gpu
# or use CPU version: pip install tensorflow
```

### SHAP not working
```bash
pip install shap
# SHAP is optional - system works without it
```

---

## 📞 Support

For issues or questions:
- Check documentation in `/docs/`
- Review segment summaries (SEGMENT_X_SUMMARY.txt)
- See examples in main.py

---

## 🎯 Future Enhancements

- [ ] Real-time streaming from IoT sensors
- [ ] Mobile app integration
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-location correlation analysis
- [ ] Automated report generation

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- WHO Water Quality Guidelines
- UCI Machine Learning Repository
- Kaggle Water Quality Dataset
- TensorFlow & Scikit-learn Communities

---

**M.Tech Project | ICESC 2026 (IEEE), Paper ID 469**

*Ensuring safe drinking water through AI-powered monitoring* 🚰💧
