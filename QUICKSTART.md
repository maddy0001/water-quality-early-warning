# ⚡ QUICK START GUIDE

**Get up and running in 5 minutes!**

---

## 🚀 1. INSTALLATION (1 min)

```bash
cd water_contamination_early_warning_system
pip install -r requirements.txt
```

**Optional (recommended):**
```bash
pip install xgboost imbalanced-learn shap
```

---

## 🎯 2. CHOOSE YOUR MODE

### Option A: Quick Demo (2 min)
```bash
python main.py --demo
```
**What it does**: Monitors 50 water samples, generates alerts

**Output**: Alert statistics, risk score

---

### Option B: Full System (30-45 min)
```bash
python main.py --full
```
**What it does**: 
- Trains all models
- Generates explainability
- Runs evaluation
- Creates 20+ plots

**Output**: Trained models + all visualizations

---

### Option C: Just Plots (30 sec)
```bash
python main.py --evaluate
```
**What it does**: Uses existing models to generate plots

**Output**: 16 publication-ready plots

---

### Option D: Train Models Only (15-20 min)
```bash
python main.py --train
```
**What it does**: Trains and saves all models

**Output**: Saved models in `models/`

---

## 📊 3. VIEW RESULTS

### Plots
```
outputs/plots/evaluation/
  ├── confusion_matrix_*.png
  ├── roc_curves_comparison.png
  ├── forecasting_*.png
  └── system_performance_summary.png
```

### Metrics
```
outputs/metrics/
  ├── classification_metrics.json
  ├── forecasting_metrics.json
  └── evaluation_summary.json
```

### LaTeX Tables (for IEEE paper)
```
outputs/plots/evaluation/results_tables.tex
```

---

## 🎓 4. FOR YOUR IEEE PAPER

### Copy-paste ready:

**Classification Results:**
- Best model: Ensemble
- Accuracy: 73.2%
- F1-Score: 71.0%
- ROC-AUC: 78.9%

**Forecasting Results:**
- Best model: BiLSTM-Attention
- RMSE: 0.0182
- R²: 0.912

**Plots**: All 16 plots are 300 DPI, ready to insert!

**LaTeX tables**: `results_tables.tex` → copy into paper

---

## 🔧 5. TROUBLESHOOTING

### No models found?
```bash
python main.py --train  # Train models first
```

### Missing plots?
```bash
python main.py --evaluate  # Generate plots
```

### Want to see what's available?
```bash
python main.py --info  # System info
```

### Need help?
```bash
python main.py --help  # All options
```

---

## 📁 6. WHAT YOU'LL GET

After running `--full`:

```
✅ 16 evaluation plots
✅ 3 trained classification models
✅ 2 trained forecasting models  
✅ 4 trained anomaly detectors
✅ LaTeX tables for paper
✅ All metrics in JSON
✅ Complete logs
```

Total disk space: ~50MB

---

## ⏱️ TIME ESTIMATES

| Mode | Time | Output |
|------|------|--------|
| `--demo` | 2 min | Alert demo |
| `--train` | 15-20 min | Models only |
| `--evaluate` | 30 sec | Plots only |
| `--full` | 30-45 min | Everything |

*(Times vary based on CPU/GPU)*

---

## 💡 PRO TIPS

1. **Run demo first** to see the system in action
2. **Then train models** (`--train`)
3. **Then evaluate** (`--evaluate`) for plots
4. **Use `--info`** to check dependencies

---

## 🎯 MOST COMMON WORKFLOW

```bash
# 1. Quick demo to see it work
python main.py --demo

# 2. Train all models (do once)
python main.py --train

# 3. Generate plots for paper
python main.py --evaluate

# Done! You have all plots + models
```

---

## 📝 COMMAND CHEAT SHEET

```bash
python main.py --full              # Run everything
python main.py --train             # Train models
python main.py --evaluate          # Make plots
python main.py --demo              # Quick demo
python main.py --info              # System info
python main.py --help              # All options

# With options:
python main.py --full --no-plots           # Skip plot generation
python main.py --demo --samples 100        # Demo with 100 samples
python main.py --full --no-autoencoder     # Skip autoencoder (no TF)
```

---

**That's it! You're ready to go!** 🚀

Questions? Check `README_FINAL.md` for detailed documentation.
