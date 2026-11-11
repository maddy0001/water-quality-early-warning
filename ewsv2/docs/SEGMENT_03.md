# Segment 03 – Forecast surprise (signal F)

**Status:** done · **Builds on:** S0–S2 · **Feeds:** S5 (fusion and calibration), S8 (TensorFlow for the ESP32)
**Proves:** the forecast-surprise part of contribution **C5**, and a second piece of evidence for the
threshold-transfer problem that **C4** (S5) solves

## 1. Goal

Turn paper 1's BiLSTM-attention forecaster, which predicted tomorrow's pH from 30 days, into a
minute-level early-warning signal: forecast every sensor 5 minutes ahead, and raise the score
when the water does something the forecast did not expect. Judge it honestly against simpler
forecasters and against the S1 bar.

## 2. What happens

```
 minute t-34 ... t-5  (30 minutes of all sensors)  ──►  forecaster  ──►  forecast for minute t
                                                                               │
 actual reading at minute t  ──────────────────────────────────────────►  residual per sensor
                                                                               │
                                       ÷ typical forecast error of that sensor (held-out windows)
                                                                               │
                                     surprise = max over sensors   (surprise_mean = mean)
```

1. **Four forecasters** (`signals/forecast.py`), all with the same inputs and the same score:
   - `forecast_persistence` – the forecast is simply the reading 5 minutes ago;
   - `forecast_linear` – ridge regression on the last 30 minutes of every sensor;
   - `v1_gru_attention` – paper 1's GRU(128) → GRU(64) → attention → Dense(32) → Dense(16);
   - `v1_bilstm_attention` – paper 1's BiLSTM(128) → BiLSTM(64) → attention → Dense(32) → Dense(16)
     (310,602 weights for 9 sensors).
2. **Kept from paper 1:** layer sizes, attention (Dense(1, tanh) → softmax over time → weighted
   sum), dropout 0.3 (0.15 in the dense head), Adam, learning rate 0.001, MSE loss, 30-step lookback.
   **Changed for minute data:** all sensors are forecast at once; one step = one minute; recurrent
   dropout 0.2 → 0 (it doubled training time in our tests); each model trains on at most 20,000
   randomly chosen windows for up to 10 epochs with early stopping (patience 2).
3. **Normal behaviour only.** Windows that touch a labelled event are left out of training, so
   events stay surprising.
4. **Honest error scale.** Each sensor's "typical forecast error" is measured on the last 10% of
   training windows, which the model did not train on. Two fixes found while building this:
   many sensors are coarsely quantised and do not change for minutes, so the usual robust measure
   (median absolute deviation) was 0 and turned every tiny step into a huge surprise. We use a
   winsorised RMS instead, never smaller than the sensor's own reading resolution.
5. **Causal and gap-aware.** A minute is scored only if the 35 minutes before it are all present
   and consecutive.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/signals/forecast.py` | Windows, error scale, four forecasters, paper-1 network builder |
| `ewsv2/configs/s3_forecast.yaml` | The S3 experiment |
| `ewsv2/cli.py`, `ewsv2/__init__.py` | Quieter TensorFlow start-up; version S3 |
| `ewsv2/tests/test_forecast.py` | 11 new tests (73 in total), including an exact check of paper 1's architecture |
| `ewsv2/requirements.txt` | Adds `tensorflow-cpu` and `ml-dtypes<0.5` |

## 4. Run it (PowerShell)

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S3 code
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S3.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S3)
python -m ewsv2 info

# 3. Packages: adds TensorFlow (CPU only, about 370 MB, one time)
pip install -r ewsv2\requirements.txt

python -m pytest ewsv2\tests -q                 # 73 passed

# 4. The experiment (cloud, 2 cores: 14 min for 2018, 12 min for 2019; expect 5-8 min each on a laptop)
python -m ewsv2 evaluate --config s3_forecast
python -m ewsv2 evaluate --config s3_forecast --set data.source=gecco2019

# 5. Save
git add ewsv2
git commit -m "ewsv2 S3: forecast surprise with paper-1 BiLSTM/GRU-attention"
git push
```

Reports: `outputs_v2\s3_forecast\gecco2018\report.md` and `...\gecco2019\report.md`.

The first TensorFlow import prints a few lines about "oneDNN" or CPU instructions. They are
information only.

## 5. Expected output (verified in the cloud run)

Honest protocol D from S1. 95% bootstrap intervals in brackets.

### GECCO 2018 (46 test events, 78 days of normal data)

| Signal | Events caught | False alarms / day | Event F1 |
|---|---|---|---|
| **S1 bar:** random_forest | 27/46 [0.46–0.72] | 0.15 [0.05–0.29] | **0.64** |
| squared_diff (one-liner) | 5/46 [0.02–0.20] | 0.37 [0.23–0.52] | 0.12 |
| forecast_persistence | 6/46 [0.04–0.24] | 0.33 [0.21–0.47] | 0.15 |
| forecast_persistence:surprise_mean | 10/46 [0.11–0.35] | 0.29 [0.17–0.42] | **0.25** |
| forecast_linear | 17/46 [0.24–0.50] | 3.37 [2.85–3.91] | 0.10 |
| forecast_linear:surprise_mean | 6/46 [0.04–0.24] | 0.29 [0.18–0.41] | 0.16 |
| v1_gru_attention | 5/46 [0.02–0.20] | 0.24 [0.14–0.35] | 0.14 |
| v1_gru_attention:surprise_mean | 3/46 [0.00–0.15] | 0.11 [0.05–0.19] | 0.10 |
| v1_bilstm_attention | 10/46 [0.11–0.35] | 2.16 [1.73–2.57] | 0.09 |
| v1_bilstm_attention:surprise_mean | 5/46 [0.02–0.20] | 0.00 | 0.20 |

### GECCO 2019 (12 test events, 22 days of normal data)

| Signal | Events caught | False alarms / day | Event F1 |
|---|---|---|---|
| **S1 bar:** random_forest | 10/12 [0.58–1.00] | 0.00 | **0.91** |
| squared_diff (one-liner) | 5/12 [0.17–0.67] | 0.46 [0.18–0.78] | 0.37 |
| forecast_persistence | 2/12 [0.00–0.42] | 0.18 [0.05–0.37] | 0.22 |
| forecast_persistence:surprise_mean | 2/12 [0.00–0.42] | 0.14 [0.00–0.28] | 0.24 |
| forecast_linear | 4/12 [0.08–0.58] | 0.05 [0.00–0.14] | 0.47 |
| forecast_linear:surprise_mean | 5/12 [0.17–0.67] | 0.05 [0.00–0.14] | **0.56** |
| v1_gru_attention (both parts) | 0/12 | 0.00 | 0.00 |
| v1_bilstm_attention (both parts) | 0/12 | 0.00 | 0.00 |

### What the numbers say

1. **Forecast surprise on its own is a weak detector.** The best forecaster reaches event F1 0.25
   on 2018 and 0.56 on 2019, against the supervised bar of 0.64 and 0.91. That is the level of the
   unsupervised detectors in S2 (best 0.46 and 0.67), and in line with the ICLR 2026 finding that
   forecasting-based detectors rarely beat one-line baselines.
2. **Paper 1's deep forecasters are not better than simple ones here.** On 2018 the BiLSTM-attention
   reaches 0.20 at best, the persistence forecaster 0.25. Bigger models forecast normal water a
   little better, but a better forecast of normal water does not make events more surprising.
3. **On 2019 the deep forecasters catch nothing – because of the threshold, not the model.** The
   threshold chosen honestly by cross-validation was 57.9 (BiLSTM). Refitting the final BiLSTM and
   looking at its test scores: the highest is 41.8, so no alarm could fire. Fold models forecast
   months they had not seen and made larger errors; the final model, trained on all pre-test data,
   forecasts the test weeks better, so its scores are smaller. With a threshold that fitted the final
   model (not allowed in the honest protocol, shown only to diagnose) it would catch 11/12 events at
   1.33 false alarms/day. The same failure hit the One-Class SVM in S2. Thresholds on unsupervised
   scores do not transfer between models trained on different data – this is what S5's adaptive
   conformal threshold fixes, by calibrating the final model on its own recent held-out minutes.
4. **Averaging over sensors usually beats taking the largest** (better in 5 of the 6 comparisons
   that caught anything): one noisy sensor no longer triggers the alarm alone.
5. **Why keep signal F at all?** It is a different kind of evidence from the classifiers: it needs no
   event labels and reacts to *any* unexpected change, including kinds of contamination never seen
   in training. Whether it adds anything on top of the ensemble is tested in S5, not here.

These are honest, partly negative results. They are worth reporting: paper 1 used this forecaster,
and paper 2 shows what it does and does not do on real minute data.

## 6. Check it worked

- `python -m ewsv2 info` lists `forecast_linear, forecast_persistence, v1_bilstm_attention, v1_gru_attention`.
- `random_forest` and `squared_diff` match S1 exactly (27/46 and 5/46 on 2018).
- `pytest` includes `test_v1_architecture_matches_paper_one` (310,602 weights).

| Problem | Fix |
|---|---|
| `pip` downloads NumPy 2 and things break | Run exactly `pip install -r ewsv2\requirements.txt` (it pins NumPy below 2 and `ml-dtypes` below 0.5); then `pip check` must say "No broken requirements" |
| `ImportError: ... DLL load failed` when importing TensorFlow | Install the "Microsoft Visual C++ Redistributable 2015–2022 (x64)" from Microsoft, then retry |
| Very slow or out of memory | Close other programs; lower `max_windows` in `ewsv2\configs\s3_forecast.yaml` (e.g. 10000) – numbers will change slightly |
| BiLSTM/GRU numbers differ a little from section 5 | Expected: neural network training is not bit-for-bit identical across CPUs. The persistence and linear rows must match exactly |

## 7. Viva notes

**What is forecast surprise?** The model predicts every sensor 5 minutes ahead from the last 30
minutes. If the actual reading is far from the prediction – measured in units of the model's usual
error for that sensor – the minute is surprising. Contamination is, by definition, something the
recent past did not predict.

**Why forecast 5 minutes ahead, not 1?** One minute ahead, every forecaster collapses to "same as
now", and surprise becomes the same as the squared-difference baseline. Five minutes gives the model
room to show what it has learned about normal dynamics, while staying early enough to warn.

**Why train only on normal minutes?** If the forecaster learned events too, it would start predicting
them, and they would stop being surprising.

**Your BiLSTM is worse than "the value 5 minutes ago". Is something wrong?** No – it is a known result.
A better forecast of normal water does not make abnormal water more surprising; what matters for
detection is how differently the model treats normal and abnormal minutes, and a simple forecaster
does that as well as a complex one here. The ICLR 2026 study "When foundation models are one-liners"
found the same for much larger models.

**Why did the 2019 deep models catch nothing?** Explained in section 5, point 3: the honestly chosen
threshold sat above every test score of the final model. We diagnosed it by refitting the final
BiLSTM and comparing its test scores with the threshold, and report it rather than tuning it away.

**Why remove recurrent dropout?** With it, training took about twice as long (23.5 s vs 16.1 s per
epoch on 20,000 windows, 2 cores) because TensorFlow cannot use its fast LSTM kernel. Normal dropout
(0.3) and early stopping still guard against overfitting.

**Why TensorFlow (370 MB) instead of something lighter?** Paper 1 used TensorFlow/Keras, so the
architecture is ported one-to-one, and S8 needs TensorFlow's converter to put a model on the ESP32.

**Why not a time-series foundation model (Chronos, TimesFM)?** They need PyTorch, another large
download, and the ICLR 2026 study already shows they detect anomalies no better than one-liners. We
cover that comparison with the one-liner baselines instead; adding a foundation model is an optional
extension.

## 8. Next segment

**S4 – Injection and triage.** Builds the GECCO-Triage benchmark: realistic contaminant signatures
and four sensor-fault types (spike, drift, stuck, dropout) injected into the real streams with
labels; adds the constancy and coherence fault checks; and makes the first three-way decision –
normal, water event, or sensor fault (contribution C1).
