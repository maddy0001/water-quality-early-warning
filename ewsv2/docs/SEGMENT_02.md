# Segment 02 – Paper-1 models on streaming data

**Status:** done · **Builds on:** S0 (data), S1 (honest protocol, baselines) · **Feeds:** S3 (forecast), S4 (triage), S5 (fusion)
**Proves:** part of contribution **C5** (what each v1 module adds) and the coherence features behind **C1**

## 1. Goal

Move paper 1's two machine-learning modules onto real one-minute utility data and judge
them honestly:

- the **anomaly quorum**: Isolation Forest, LOF, One-Class SVM and an autoencoder, alarm
  when 2 of 4 agree;
- the **classifier ensemble**: Random Forest, XGBoost and SVM, soft-voted 0.35 / 0.40 / 0.25.

Along the way this segment adds the **coherence features**, which measure whether
a sensor's move is shared by the sensors it normally moves with. These features are the
physical basis of the water-vs-sensor triage in S4.

## 2. What happens

```
 one-minute stream
      │
      ▼
 RichFeatures (stateless, causal)            per sensor: value, 1-min change, deviation from
      │                                      the last 5/15/60 min, spread, flat-line flag
      ▼
 FeaturePipeline.fit (fit rows only)         robust scaling + coherence model:
      │                                      predict each sensor's 15-min move from the others
      ▼
 ┌──────────────── v1_quorum ────────────────┐   ┌──────────── v1_ensemble ────────────┐
 │ IF   LOF   OC-SVM   autoencoder           │   │ RF        XGBoost        SVM        │
 │  └── calibrated scores ──┘                │   │  └── probabilities ──┘              │
 │  quorum = 2nd-highest score               │   │  ensemble = 0.35·RF + 0.40·XGB       │
 │  fixed_vote = v1 rule (2 of 4 at 95th pct)│   │             + 0.25·SVM               │
 └───────────────────────────────────────────┘   └─────────────────────────────────────┘
      │                                                 │
      └──────── every part evaluated under protocol D (S1) ────────┘
```

1. **Rich features** (`features/rich.py`). Every value at minute t uses only readings up
   to t. Short gaps (≤ 5 min) are carried forward; minutes with no data are never scored.
2. **Coherence model.** On normal training minutes, a ridge regression predicts each
   sensor's 15-minute deviation from the other sensors' deviations. The residual says how
   much of a sensor's move the others do not explain. Four summary columns follow: largest
   deviation, largest residual, *breadth* (how many sensors moved more than 3 robust
   units), and *share* (residual relative to deviation).
3. **v1 quorum** (`signals/v1_port.py`). v1's settings, two changes for streaming data:
   LOF is fitted on the past and applied to new minutes (v1 fitted and scored the same
   rows), and the kernel methods train on a 10,000-row sample. The autoencoder keeps v1's
   16-8-16 shape, built with scikit-learn so this segment needs no TensorFlow.
   "2 of 4 fire" equals "the 2nd-highest score crosses the threshold", so the quorum score
   is that order statistic and the honest protocol picks the threshold. Paper 1's own fixed
   rule is reported too (`v1_quorum:fixed_vote`).
4. **v1 ensemble.** v1's three classifiers and weights. Every event minute is kept;
   normal minutes are sampled down to 60,000 (8,000 for the SVM).
5. **Parts.** Each composite reports its parts, so one run gives every single detector and
   classifier too, without refitting. Two extra rows repeat the ensemble with the S1
   features and with rich features minus coherence (feature ablation).

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/features/rich.py` | Rich causal features, `FeaturePipeline`, coherence model |
| `ewsv2/signals/v1_port.py` | `v1_quorum` and `v1_ensemble` |
| `ewsv2/configs/s2_v1_port.yaml` | The S2 experiment |
| `ewsv2/evaluate/audit.py` | Evaluates every part of a composite; fixed thresholds; `label:` for repeated models |
| `ewsv2/cli.py`, `ewsv2/__init__.py` | Wider result table; version S2 |
| `ewsv2/tests/test_v1_port.py`, `test_evaluate.py` | 10 new tests, one extended (62 in total) |
| `ewsv2/requirements.txt` | Adds `xgboost-cpu` (2 MB CPU-only build) |

## 4. Run it (PowerShell)

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S2 code (change the path if the zip is not in Downloads)
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S2.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S2)
python -m ewsv2 info

pip install -r ewsv2\requirements.txt          # adds xgboost-cpu

python -m pytest ewsv2\tests -q                 # 62 passed

# 3. The experiment (about 10 min per dataset on 2 cloud cores; expect 3-6 min on a 12-thread laptop)
python -m ewsv2 evaluate --config s2_v1_port
python -m ewsv2 evaluate --config s2_v1_port --set data.source=gecco2019

# 4. Save
git add ewsv2
git commit -m "ewsv2 S2: paper-1 quorum and ensemble on streaming data, coherence features"
git push
```

Reports: `outputs_v2\s2_v1_port\gecco2018\report.md` and `...\gecco2019\report.md`.

## 5. Expected output (verified in the cloud run)

All rows use the honest protocol D from S1. 95% bootstrap intervals in brackets.

### GECCO 2018 (46 test events, 78 days of normal data)

| Model | Events caught | False alarms / day | Median delay, min | Event F1 |
|---|---|---|---|---|
| **S1 bar:** random_forest | 27/46 [0.46–0.72] | 0.15 [0.05–0.29] | 1 | **0.64** |
| **Paper 1's fixed rule** (v1_quorum:fixed_vote) | 46/46 | **33.28** [30.75–35.81] | 0 | 0.03 |
| v1_quorum (2-of-4, honest threshold) | 6/46 [0.04–0.24] | 0.00 | 2 | 0.23 |
| v1_quorum:iforest | 22/46 [0.33–0.61] | 0.49 [0.26–0.74] | 4 | 0.42 |
| v1_quorum:lof | 12/46 [0.13–0.39] | 0.03 [0.00–0.06] | 9.5 | 0.40 |
| v1_quorum:ocsvm | 39/46 [0.74–0.93] | 1.10 [0.86–1.38] | 0 | 0.46 |
| v1_quorum:autoencoder | 7/46 [0.07–0.26] | 0.20 [0.09–0.35] | 0 | 0.20 |
| **v1_ensemble** (RF+XGB+SVM, rich + coherence) | 37/46 [0.67–0.91] | 0.92 [0.68–1.17] | 0 | 0.48 |
| v1_ensemble:rf | 41/46 [0.80–0.96] | 0.64 [0.42–0.89] | 0 | 0.60 |
| v1_ensemble:svm | 35/46 [0.65–0.87] | 2.26 [1.76–2.83] | 0 | 0.27 |
| v1_ensemble:xgb | 4/46 [0.02–0.17] | 0.00 | 2 | 0.16 |
| v1_ensemble_nocoh (no coherence) | 41/46 [0.80–0.96] | 1.98 [1.55–2.47] | 0 | 0.34 |
| v1_ensemble_basic (S1 features) | 39/46 [0.74–0.93] | 2.61 [2.06–3.21] | 0 | 0.27 |

### GECCO 2019 (12 test events, 22 days of normal data)

| Model | Events caught | False alarms / day | Median delay, min | Event F1 |
|---|---|---|---|---|
| **S1 bar:** random_forest | 10/12 [0.58–1.00] | 0.00 | 1 | **0.91** |
| **Paper 1's fixed rule** | 12/12 | **21.50** [16.21–26.94] | 0 | 0.05 |
| v1_quorum (2-of-4, honest threshold) | 6/12 [0.25–0.75] | 0.05 [0.00–0.14] | 3 | 0.63 |
| v1_quorum:iforest | 7/12 [0.33–0.83] | 0.28 [0.00–0.74] | 1 | 0.56 |
| v1_quorum:lof | 7/12 [0.33–0.83] | 0.09 [0.00–0.23] | 2 | 0.67 |
| v1_quorum:ocsvm | 0/12 | 0.00 | – | 0.00 |
| v1_quorum:autoencoder | 6/12 [0.25–0.75] | 0.05 [0.00–0.14] | 2.5 | 0.63 |
| **v1_ensemble** | **12/12** | 0.09 [0.00–0.23] | 1 | **0.92** |
| v1_ensemble:rf | 10/12 [0.58–1.00] | 0.00 | 1 | 0.91 |
| v1_ensemble:svm | 9/12 [0.50–1.00] | 0.69 [0.37–1.05] | 1 | 0.50 |
| v1_ensemble:xgb | 9/12 [0.50–1.00] | 0.00 | 1 | 0.86 |
| v1_ensemble_nocoh | 12/12 | 0.05 [0.00–0.14] | 1 | 0.96 |
| v1_ensemble_basic | 12/12 | 0.05 [0.00–0.14] | 1 | 0.96 |

### What the numbers say

1. **Paper 1's alarm rule does not survive streaming data.** Each detector at its 95th
   percentile with a 2-of-4 vote fires on every event but raises 33 false alarms a day on
   2018 and 21 on 2019 – roughly one every 45–70 minutes. No operator would keep it on.
2. **The vote does not help.** With an honest threshold the 2-of-4 quorum is never better
   than the best single detector (2018: 0.23 vs 0.46; 2019: 0.63 vs 0.67). The detectors
   rarely agree on the same minute, so requiring two of them mostly throws events away.
3. **Unsupervised detectors are fragile.** The One-Class SVM catches 39/46 events on 2018
   and 0/12 on 2019: its threshold, chosen honestly on the pre-test period, ends up above
   every 2019 test score. Thresholds that do not transfer are the problem S5 addresses.
4. **The supervised ensemble finds the most events, but pays in false alarms on 2018.**
   2018: 37/46 events (S1 bar: 27/46) at 0.92 false alarms/day (S1: 0.15); its event F1
   (0.48) is below the S1 bar. 2019: 12/12 at 0.09/day, the best result (0.92). Neither
   model dominates on both datasets – which is why S5 fuses signals and sets the
   threshold from a false-alarm budget instead of event F1.
5. **Coherence features halve false alarms on 2018** at the same recall: 0.92/day
   [0.68–1.17] with coherence against 1.98 [1.55–2.47] without – the intervals do not
   overlap. On 2019 they make no difference (1 vs 2 false alarms in 22 days). Richer
   features also help: S1 features give 2.61/day on 2018.
6. **Within the ensemble**, the Random Forest carries the result; XGBoost with v1's
   setting (scale_pos_weight 1.5, tuned for Kaggle's 60/40 balance) is too timid on 2018's
   1% event rate, and the SVM raises the most false alarms on both datasets.

These are honest numbers, and several are worse than paper 1's. That is the point of
S1's protocol: they are what a utility would actually see.

## 6. Check it worked

- `python -m ewsv2 info` lists `feature basic, rich` and the signals `v1_ensemble, v1_quorum`.
- The console table has 13 rows per dataset; the first (`random_forest`) must match S1
  exactly (27/46 on 2018, 10/12 on 2019) because nothing about it changed.

| Problem | Fix |
|---|---|
| `No module named xgboost` | `pip install -r ewsv2\requirements.txt` (installs `xgboost-cpu`) |
| You already have the big `xgboost` package | Fine: both give the same `import xgboost` |
| Run takes much longer than stated | Normal on 4 cores or fewer; the SVM and LOF are the slow parts |
| Numbers differ slightly from section 5 | Tree and neural models use all CPU cores, so the last digit can differ between machines; the ranking should not |

## 7. Viva notes

**Why port paper 1's models instead of starting over?** They are the components of the
system we already published. Testing them honestly on real streaming data tells us which
parts survive (the Random Forest, the features) and which do not (the fixed 2-of-4 rule),
and it gives paper 2 a direct, fair comparison with paper 1.

**Why did paper 1's 2-of-4 rule fail here when it worked in paper 1?** Paper 1 scored
water samples, each judged once. Here a detector judges every minute, 1,440 times a day.
A rule that flags 5% of normal samples per detector flags dozens of normal minutes a day;
even after the vote, that is 21–33 false alarms per day.

**What is the coherence model, in one sentence?** For each sensor it learns, from normal
data, how that sensor usually moves together with the others, and reports how much of
the current move is *not* explained by them – a lone, unexplained move looks like a
faulty sensor; a shared move looks like a real change in the water.

**Does coherence help?** On GECCO 2018 it halves false alarms (0.92 vs 1.98 per day, 95%
intervals do not overlap) at the same event recall. On 2019 there is no measurable
difference. We report both; S4 tests coherence directly on the water-vs-sensor question,
where it is meant to matter most.

**Why is the One-Class SVM 0/12 on 2019?** Its threshold, chosen honestly on the earlier
months, is higher than any score it gives in the test weeks. Unsupervised scores drift with
the season; a threshold that held in autumn need not hold in winter. Segment 5's adaptive
conformal threshold is designed for exactly this.

**Why is XGBoost so weak on 2018?** It keeps paper 1's `scale_pos_weight = 1.5`, chosen for a
dataset that was 40% positive. Here events are about 1% of minutes, so XGBoost's
probabilities stay low and few cross the threshold. We kept v1's setting to make the
comparison fair; tuning it is not the contribution.

**Why the CPU-only XGBoost?** `xgboost-cpu` is the same library without the GPU code: 2 MB
instead of 125 MB on Windows. Results are identical on CPU.

**Why scikit-learn for the autoencoder, not TensorFlow?** Same architecture (16-8-16, ReLU,
Adam, MSE); avoiding a 500 MB dependency for one small network. TensorFlow arrives in S3,
where the BiLSTM needs it.

## 8. Next segment

**S3 – Forecast surprise.** Ports v1's BiLSTM-attention as a minute-level, multi-sensor
forecaster (TensorFlow), with GRU, persistence and a time-series foundation model as
baselines. The forecast error becomes signal F, judged under protocol D against the
tables above.
