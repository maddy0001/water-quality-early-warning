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

**M.Tech Project | IEEE Paper Submission 2026**

---

## 🎯 Key Features

### 🤖 Machine Learning Models
- **Classification**: Random Forest, XGBoost, SVM, Ensemble
- **Forecasting**: BiLSTM-Attention, GRU-Attention
- **Anomaly Detection**: Isolation Forest, LOF, One-Class SVM, Autoencoder

### 🚨 Early Warning System
- 4-Level alerts: NORMAL → LOW → MEDIUM → CRITICAL
- Weighted voting from multiple models
- Real-time monitoring capability (<50ms per sample)
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

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.685 | 0.652 | 0.742 |
| XGBoost | 0.713 | 0.684 | 0.769 |
| SVM | 0.669 | 0.623 | 0.723 |
| **Ensemble** | **0.732** | **0.710** | **0.789** |

**Features**: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity

### 2. Time Series Forecasting

**Purpose**: Predict future pH values (24-hour ahead)

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| BiLSTM-Attention | 0.0182 | 0.0147 | 2.31% | 0.912 |
| GRU-Attention | 0.0202 | 0.0162 | 2.54% | 0.896 |

**Architecture**: Bidirectional LSTM with attention mechanism (30-day lookback)

### 3. Anomaly Detection

**Purpose**: Detect contamination events

| Detector | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Isolation Forest | 0.769 | 1.000 | 0.870 |
| LOF | 1.000 | 1.000 | 1.000 |
| One-Class SVM | 0.714 | 1.000 | 0.833 |
| **Ensemble (Voting)** | **0.833** | **1.000** | **0.909** |

**Strategy**: 4 detectors with majority voting (threshold: 50%)

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
- **Classification Accuracy**: 73.2%
- **Forecasting RMSE**: 0.0182
- **Anomaly Detection F1**: 90.9%
- **Processing Speed**: ~50ms per sample
- **Alert Latency**: <50ms

### Datasets
- **Classification**: 3,276 samples (WHO water quality data)
- **Forecasting**: 705 days × 37 locations (UCI repository)
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
- **Best Classification Model**: Ensemble (F1=0.710, AUC=0.789)
- **Best Forecasting Model**: BiLSTM-Attention (RMSE=0.0182, R²=0.912)
- **Anomaly Detection**: Ensemble voting (F1=0.909)
- **Real-time Capability**: <50ms per sample

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
@article{water_early_warning_2026,
  title={AI-Powered Water Contamination Early Warning System: 
         Multi-Model Ensemble for Real-Time Monitoring},
  author={[Your Name]},
  journal={IEEE Access},
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
- [ ] Advanced visualization dashboard
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

**Built for M.Tech Project | IEEE Paper Submission 2026**

*Ensuring safe drinking water through AI-powered monitoring* 🚰💧
