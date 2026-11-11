# Water Quality Early Warning System

Machine-learning early warning for drinking-water quality, in two versions that live side by side in this
repository:

| | Version 1 | Version 2 |
|---|---|---|
| What it is | The system of paper 1, *"An Integrated Machine Learning Framework for Real-Time Water Quality Monitoring"*, ICESC 2026 (IEEE), Paper ID 469 | The follow-up: real utility data, honest evaluation, water event vs sensor fault, false-alarm control, checked explanations, fact-checked AI reports, Arduino Uno edge device, operator dashboard |
| Code | repository root: `main.py`, `src/`, `pipelines/`, `dashboard/` | `ewsv2/` (one package) |
| Python environment | `venv/` (TensorFlow 2.13, NumPy 1.24) | `venv2/` (TensorFlow 2.18, NumPy 1.26) |
| Requirements | `requirements.txt` | `ewsv2/requirements.txt` |
| Exact versions (lock files) | `ewsv2/handover/locks/requirements-lock-v1.txt` | `ewsv2/handover/locks/requirements-lock-v2.txt` |
| Data | `data/raw/water_potability.csv`, `data/raw/water_dataset.mat` | `data/raw/gecco/` (GECCO 2018/2019) |
| Trained models / results | `models/`, `outputs/` | `ewsv2/edge/model/`, `outputs_v2/` |
| Dashboard | `dashboard/app.py`, port 8501 | `python -m ewsv2 dashboard`, port 8502 (also starts version 1) |
| Detailed guide | this file, section *Version 1* | [`ewsv2/README.md`](ewsv2/README.md) and `ewsv2/docs/SEGMENT_00.md` … `SEGMENT_10.md` |
| Git tag | `v1.0` | `main` |

The two versions never import each other and use separate environments, so version 1 stays exactly as
published. Each folder has its own README: this one describes the repository and version 1; `ewsv2/README.md`
describes version 2 in detail. They are different files in different folders, so they do not conflict in Git;
GitHub shows this one on the repository's front page.

---

## Repository layout

```
water-quality-early-warning/
├── README.md                 this file (both versions, version 1 in detail)
├── requirements.txt          version 1 requirements
├── main.py                   version 1 entry point
├── src/  pipelines/          version 1 source code
├── dashboard/                version 1 dashboard (Streamlit)
├── models/  outputs/         version 1 trained models and results
├── notebooks/  reports/      version 1 notebooks and paper material
├── data/
│   ├── raw/                  version 1 datasets  (+ raw/gecco/ for version 2, not in Git)
│   ├── processed/            version 1 preprocessed data (not in Git)
│   └── samples/              version 1 dashboard sample files
├── ewsv2/                    version 2 (own README, requirements, configs, tests, docs, hardware, dashboard)
├── outputs_v2/               version 2 results (not in Git)
├── venv/                     version 1 environment (not in Git)
└── venv2/                    version 2 environment (not in Git)
```

Not in Git (too large or machine-specific): both environments, the GECCO data, `data/processed/`, the
trained version 1 models and `outputs_v2/`. They are all in the offline time capsule (see *Backup and
restore*).

---

## Quick start (Windows PowerShell)

### Check that everything works

```powershell
cd "$HOME\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1
python -m ewsv2 doctor
```

`doctor` checks both versions (environments, packages, models, data), the results, the Arduino and its
sensors, the local AI model and the dashboard ports. Every line ends in `[ OK ]`, `[ -- ]` (missing, a
replacement is used) or `[FAIL]`, with the fix.

### Start the dashboards

```powershell
venv2\Scripts\Activate.ps1
python -m ewsv2 dashboard        # version 2 on http://localhost:8502, version 1 on http://localhost:8501
```

Version 1 alone:

```powershell
cd dashboard
..\venv\Scripts\python -m streamlit run app.py --server.port 8501
```

### First-time setup on a new computer (with internet)

```powershell
py -3.11 -m venv venv
venv\Scripts\python -m pip install --no-deps -r ewsv2\handover\locks\requirements-lock-v1.txt
venv\Scripts\python main.py --train              # version 1 models (if models/ is empty)

py -3.11 -m venv venv2
venv2\Scripts\python -m pip install --no-deps -r ewsv2\handover\locks\requirements-lock-v2.txt
venv2\Scripts\python -m ewsv2 fetch all          # GECCO data for version 2
venv2\Scripts\python -m ewsv2 doctor
```

Without internet, use the time capsule instead (next section).

---

## Backup and restore (offline time capsule)

One command writes everything needed to rebuild both versions on a new Windows laptop **without internet**
to a pendrive: the repository with data, models and results, every package of both environments as Windows
wheels, the local AI model, the Arduino board files, the Python and Ollama installers, a restore script and a
SHA-256 manifest.

```powershell
venv2\Scripts\Activate.ps1
python -m ewsv2 capsule --dest H:\ewsv2_capsule --with-ollama --with-arduino --with-installers
python -m ewsv2 capsule --check H:\ewsv2_capsule
```

Restore (installs Python 3.11 first if needed, then rebuilds `venv` and `venv2` offline and runs `doctor`):

```powershell
powershell -ExecutionPolicy Bypass -File H:\ewsv2_capsule\restore.ps1
```

The environments are rebuilt rather than copied because a Python environment stores absolute paths and does
not work after being moved. `H:\ewsv2_capsule\RESTORE.md` explains every step.

---

## Version 2 in brief

Full guide: [`ewsv2/README.md`](ewsv2/README.md). Results on the public GECCO 2018/2019 data of a German
drinking-water utility:

| Question | Result |
|---|---|
| Was version 1 tested fairly? | No – shuffled minutes leak the answers. The same random forest scores 0.98 with a shuffled split and **0.64** with a time-ordered (honest) split. All version 2 numbers are time-ordered. |
| Water event or sensor fault? | Fault checks send failing sensors to maintenance: injected faults raising a public water alarm drop from 7 to **1** (2018) and from 3 to **0** (2019). |
| Can the false-alarm rate be promised? | Yes – an adaptive threshold keeps **about 1 per day** (0.97–1.01 in 2018, 0.90–1.18 in 2019); fixed thresholds range from 0.05 to 3.5. |
| Are the explanations true? | The deviation explainer passes a deletion test 0.95–1.00 of the time; version 1's TreeSHAP 0.74. |
| Can an AI write the incident report? | A 3-billion-parameter model on the laptop wrote **59 of 60** reports without a factual error; every report is fact-checked and a wrong one is never shown. |
| Does it run on cheap hardware? | The decision logic runs on an **Arduino Uno** (19,316 bytes flash, 1,140 bytes RAM, 25 ms per minute) with decisions **identical to the laptop for 10,080 minutes** of real data. |

Dashboard pages: Overview, Operations Console (replay of real utility data with operator review), Sensor Lab
(live Arduino sensors or simulation), Performance Analytics, Data Explorer, Incident Reports, System Guide,
Version 1 Baseline, System Status. Every part is optional: without the Arduino, the AI model, the data files
or the results, the dashboard uses a replacement and says so.

Hardware for the live demonstration (all optional): Arduino Uno (CH340), conductivity probe HW-080 with
HW-103 board (`AO→A0`, `VCC→A2`, `GND→GND`), touch sensor TTP223 (`SIG→D3`, `VCC→D4`, `GND→GND`), DHT11
(`DATA→A1`, `VCC→5V`, `GND→GND`). Wiring guide: `ewsv2/hardware/`.

Paper tables and figures: `python -m ewsv2 paper` writes them to `outputs_v2/paper/` (CSV, Markdown, LaTeX;
PNG and PDF). Architecture diagram: `ewsv2/docs/figures/architecture.png`.

---

## Version 1 – paper 1 (ICESC 2026)

A machine-learning framework for water quality monitoring that integrates:

- **Classification** for potability prediction (Random Forest, XGBoost, SVM, weighted ensemble)
- **Time-series forecasting** for next-day pH (BiLSTM-Attention, GRU-Attention)
- **Anomaly detection** for contamination events (Isolation Forest, LOF, One-Class SVM, Autoencoder; 2-of-4 vote)
- **Explainability** with SHAP
- **A 4-level alert system** (NORMAL, LOW, MEDIUM, CRITICAL) by weighted voting (classification 40 %,
  forecasting 30 %, anomaly 30 %)

### Run version 1

```powershell
venv\Scripts\Activate.ps1
python main.py --info        # system information
python main.py --demo        # quick demo
python main.py --train       # train all models (creates models/)
python main.py --evaluate    # evaluation plots
python main.py --full        # everything
```

### Results

**Classification** – held-out test set of 570 real samples (SMOTE applied to the training split only):

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 63.33% | 50.82% | 66.36% |
| XGBoost | 59.82% | 51.59% | 65.01% |
| SVM | 60.53% | 51.82% | 64.12% |
| **Ensemble (0.35 / 0.40 / 0.25)** | **61.75%** | **53.02%** | **66.62%** |

Features: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity.

**Forecasting** – next-day pH, 30-day look-back, 9,324 test sequences, errors on the normalised (0–1) pH scale:

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| BiLSTM-Attention | 0.0524 | 0.0336 | 19.36% | 0.5251 |
| GRU-Attention | 0.0610 | 0.0396 | 20.55% | 0.3572 |

**Anomaly detection** – the dataset has no contamination labels, so precision and recall cannot be measured
directly:

| Check | Result |
|-------|--------|
| Samples flagged (≥ 2 of 4 detectors agree) | 120 of 2,847 (4.21%), against an assumed 5% rate |
| Synthetic contamination test (2 parameters shifted ±3 SD) | 75.6% caught, 5.4% false alarms |

Processing speed: about 0.5 ms per sample.

**Datasets:** Kaggle Water Potability (3,276 samples) for classification; UCI Water Quality (705 days × 37
USGS stations, Georgia) for forecasting.

### Version 1 structure

```
main.py                         entry point
src/config.py                   configuration (model parameters, alert thresholds, paths)
src/data/                       loading and preprocessing
src/models/                     classification, forecasting and anomaly models
src/early_warning/              alert system and monitoring engine
src/explainability/             SHAP analysis
src/visualization/  src/utils/  plots and helpers
pipelines/                      classification, forecasting, anomaly, explainability, early warning, evaluation
dashboard/                      Streamlit dashboard (sample inputs in data/samples/)
outputs/plots, outputs/metrics  figures (300 DPI) and metrics (JSON); LaTeX tables for the paper
```

### Use from Python

```python
from pipelines.pipeline_classification import run_classification_pipeline
results = run_classification_pipeline(save_models=True)

from early_warning.monitoring_engine import WaterQualityMonitor
monitor = WaterQualityMonitor()
monitor.load_models()
sample = [7.2, 180.5, 15000, 6.8, 250, 420, 12.5, 60, 3.2]      # the 9 features
alert = monitor.check_sample(sample, location="Site_A")
print(alert.level.name, alert.get_message())
```

### Version 1 dashboard

```powershell
venv\Scripts\python main.py --train            # once: creates the trained models the dashboard loads
cd dashboard
..\venv\Scripts\python -m streamlit run app.py --server.port 8501
```

Every page runs the trained models. Sample files for each page are in `data/samples/` (see
`dashboard/README.md`).

### Configuration

`src/config.py`: model parameters (`CLASSIFICATION_CONFIG`), alert thresholds (`CLASSIFICATION_CRITICAL`,
`FORECASTING_CRITICAL`, `ANOMALY_CRITICAL`) and data paths.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Not sure what is wrong | `venv2\Scripts\python -m ewsv2 doctor` – it prints the fix for each part |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `venv2\Scripts\python` directly |
| Module not found | install the lock file of that version (see *First-time setup*) or restore from the capsule |
| Arduino not found | close the Arduino IDE Serial Monitor (it blocks the port), re-plug the USB cable |
| AI reports not written by the model | start Ollama, then `ollama pull qwen2.5:3b`; otherwise template reports are used |
| Port 8501 or 8502 in use | close the old dashboard window (Ctrl+C) |

---

## Citation

```bibtex
@inproceedings{icesc2026_469,
  title={An Integrated Machine Learning Framework for Real-Time Water Quality Monitoring},
  author={Madhusuthanan G and Vijayalakshmi V},
  booktitle={International Conference on Electronics and Sustainable Communication Systems (ICESC)},
  year={2026}
}
```

## License and acknowledgements

MIT License. Data: Kaggle Water Potability, UCI Water Quality, and the GECCO Industrial Challenge 2018/2019
(Thüringer Fernwasserversorgung and TH Köln, CC BY 4.0). Thanks to the WHO drinking-water guidelines and the
TensorFlow, scikit-learn and Streamlit communities.
