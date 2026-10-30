# Segment 05 – Fusion and a false-alarm budget

**Status:** done · **Builds on:** S2 (classifier, anomaly quorum), S3 (forecast surprise), S4 (GECCO-Triage, rules triage) · **Feeds:** S6 (explanations), S7 (incident reports), S8 (ESP32)
**Proves:** contribution **C4** (an alarm threshold that keeps a stated false-alarm rate when the
season changes) and **C5** (the first honest test of paper 1's three-module fusion); experiments
**E4** (fusion), **E5** (budget), **E6** (persistence rule)

## 1. Goal

Paper 1 fused its three modules with hand-set weights, R = 0.40 C + 0.30 F + 0.30 A, and raised alerts at
fixed levels. This segment asks two questions on one real stream:

1. **Do the weights matter?** Paper 1's weights against equal weights, an "any module" rule, each
   module alone, and weights learned from data (stacking).
2. **Can we promise an operator a false-alarm rate?** "At most one false alarm a day" is what a
   utility can plan staff around. S1–S4 chose thresholds for the best F1 score, which says nothing
   about the rate, and S3 and S4 showed thresholds learned in summer behave differently in winter.

Everything runs on the GECCO-Triage stream of S4, so each answer is checked on real events,
on contaminants the models never saw, and on sensor faults.

## 2. What happens

```
 GECCO-Triage stream (S4)
      │
      ▼
 three paper-1 modules, each as 4 fold models (no refit)      C  RF + XGBoost + SVM ensemble (S2)
      │  every score → "nines": how rare among normal           A  four-detector quorum (S2)
      │  pre-test minutes (3 nines = 1 in 1,000)                F  forecast surprise (S3)
      ▼
 fusion:  C / A / F alone · hand 0.40/0.30/0.30 · equal · max · stacking (learned)
      │   main fusion chosen on pre-test data only
      ▼
 threshold:  cv_f1 (S1 rule) · budget (fixed) · adaptive (corrects itself online)
      │       + persistence rule (k of the last n minutes)
      ▼
 S4 rules triage  →  "check the water" / "check sensor X"
```

1. **Cross-conformal module scores** (`fusion/modules.py`). S3 found the *threshold-transfer*
   problem: thresholds were chosen on four fold models, then applied to a fifth, refitted model whose
   scores sat on a different scale (the 2019 BiLSTM caught 0/12 because of it). Now there is no
   refit: the four fold models also score the test period, every score is converted to *nines*
   against the same reference (out-of-fold scores of normal pre-test minutes), and a test minute
   gets the average over the four models. Thresholds and test scores come from the same models.
2. **Nines** (`calibrate/methods.py`): nines = −log10(share of normal minutes scoring at least as
   high). One scale for every module, so they can be added, compared and thresholded alike.
3. **Fusions** (`fusion/methods.py`). `hand` and `equal` average the modules on paper 1's [0, 1]
   scale; `max` takes the most alarming module; `stacking` is a logistic regression on the modules'
   nines, cross-fitted on out-of-fold pre-test scores. The **main fusion** used in E5, E6 and the full
   pipeline is the one with the best pre-test event F1 at the budget threshold (no test data used).
4. **Thresholds.**
   - `cv_f1` – the S1 rule: best event F1 on out-of-fold pre-test scores.
   - `budget` – the lowest threshold that keeps false alarms on pre-test water within the budget
     (default 1 per day). A fixed, conformal threshold.
   - `adaptive` – starts at the budget threshold and corrects itself online. Every minute lowers it a
     little; every false alarm, once the operator closes it as false, raises it by 0.25 nines. In the
     long run false alarms settle at the budget whatever the season does to the scores: adaptive
     conformal inference (Gibbs & Candès, NeurIPS 2021), with the operator's verdict as feedback. The
     verdict is read only after an alarm ends, as in a real control room.
5. **Persistence rule** (E6): alarm only when k of the last n minutes are above the threshold.
6. **Full pipeline:** main fusion → adaptive threshold → the S4 rules triage, compared with no
   triage and with the triage minus its "lone jump" rule.

Also in this segment: the GECCO-Triage injector now rounds injected values to each sensor's reading
step, as a real sensor reports them. The first S4 release gave slightly different 2018 numbers on
your laptop and in the cloud; with rounding the benchmark is identical everywhere. `SEGMENT_04.md`
section 5 has the updated S4 numbers (same conclusions).

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/fusion/methods.py` | `single`, `hand`, `equal`, `max`, `stacking` |
| `ewsv2/fusion/modules.py` | Cross-conformal module scores (fold models, nines, no refit) |
| `ewsv2/fusion/run.py` | The S5 experiment, report and chart |
| `ewsv2/calibrate/methods.py` | Nines; `cv_f1`, `budget`, `adaptive` thresholds; persistence rule |
| `ewsv2/configs/s5_fusion.yaml` | The S5 experiment |
| `ewsv2/cli.py` | New command `fuse` |
| `ewsv2/core/registry.py` | Loads `fusion` and `calibrate` |
| `ewsv2/inject/benchmark.py` | Injected values rounded to the reading step (reproducibility) |
| `ewsv2/tests/test_fusion.py` | 9 new tests (102 in total), incl. "the adaptive threshold holds the budget when scores drift" |
| `ewsv2/docs/SEGMENT_04.md` | Updated S4 numbers |

No new packages.

## 4. Run it (PowerShell)

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S5 code
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S5.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S5)
python -m ewsv2 info
python -m pytest ewsv2\tests -q                 # 102 passed

# 3. S4 again (the benchmark is now rounded; numbers in SEGMENT_04.md section 5)
python -m ewsv2 triage --config s4_triage
python -m ewsv2 triage --config s4_triage --set data.source=gecco2019

# 4. The S5 experiment (cloud, 2 cores: about 5 min per dataset)
python -m ewsv2 fuse --config s5_fusion
python -m ewsv2 fuse --config s5_fusion --set data.source=gecco2019

# 5. Save
git add ewsv2
git commit -m "ewsv2 S5: fusion of paper-1 modules, cross-conformal scores, adaptive false-alarm budget"
git push
```

Reports: `outputs_v2\s5_fusion\gecco2018\report.md` and `...\gecco2019\report.md`, with
`weekly.png`. Re-running only the report part is quick with `--set fuse.cache=true` (the module
scores are saved after the first run).

## 5. Expected output (verified in the cloud run)

### E4 + E5 – fusion × threshold (budget 1 false alarm/day)

GECCO 2018 (46 real events, 30 injected contaminants):

```
  fusion                     threshold    real  contam.  FA/day worst wk    F1
  C alone                    cv_f1      13/46     2/30     0.08     0.44  0.31
  C alone                    budget     34/46    16/30     0.31     0.87  0.67
  C alone                    adaptive   46/46    19/30     0.97     1.97  0.60
  A alone                    budget     34/46    11/30     1.01     1.92  0.45
  A alone                    adaptive   33/46    15/30     1.00     1.97  0.48
  F alone                    budget     14/46     7/30     1.50     3.11  0.20
  F alone                    adaptive   15/46     6/30     1.01     1.31  0.24
  hand (v1 0.40/0.30/0.30)   budget     34/46    16/30     0.69     1.75  0.56
  hand (v1 0.40/0.30/0.30)   adaptive   39/46    18/30     1.00     1.97  0.54
  equal (1/3 each)           adaptive   39/46    19/30     1.00     1.97  0.55
  max (any module)           adaptive   34/46    10/30     1.00     1.47  0.45
  stacking (learned)         budget     34/46    10/30     0.52     1.53  0.55
  stacking (learned)         adaptive   38/46    18/30     1.00     1.77  0.54
```

GECCO 2019 (12 real events, 18 injected contaminants):

```
  C alone                    budget     12/12     7/18     1.61     2.09  0.46
  C alone                    adaptive   12/12     7/18     1.09     1.48  0.53
  A alone                    budget     11/12    10/18     3.45     4.15  0.34
  A alone                    adaptive    9/12     6/18     1.18     1.33  0.43
  hand (v1 0.40/0.30/0.30)   budget     11/12     8/18     1.98     2.30  0.42
  hand (v1 0.40/0.30/0.30)   adaptive   10/12     6/18     1.09     1.19  0.46
  max (any module)           budget     12/12     7/18     3.54     5.19  0.31
  stacking (learned)         budget      3/12     0/18     0.05     0.35  0.18
  stacking (learned)         adaptive   10/12     7/18     0.90     1.48  0.52
```

(The console prints every row, 21 per dataset; the report has them all.)

| | 2018 | 2019 |
|---|---|---|
| False alarms/day, fixed budget threshold, over the 7 fusions | 0.31 – 1.50 | 0.05 – 3.54 |
| False alarms/day, adaptive threshold, over the 7 fusions | **0.97 – 1.01** | **0.90 – 1.18** |
| Main fusion chosen on pre-test data | stacking | C alone |
| Learned stacking weights (C, A, F) | +0.99, +3.57, +0.30 | +1.70, +0.08, +1.11 |

### E6 – persistence rule (main fusion, adaptive threshold)

| Rule | 2018 real | 2018 contaminants | 2018 FA/day | 2018 delay | 2019 real | 2019 FA/day | 2019 delay |
|---|---|---|---|---|---|---|---|
| 1 of 1 min | 38/46 | 18/30 | 1.00 | 2 | 12/12 | 1.09 | 1 |
| 2 of 3 min | 39/46 | 20/30 | 1.04 | 4 | 12/12 | 1.04 | 2 |
| 3 of 5 min | 41/46 | 20/30 | 0.99 | 5 | 12/12 | 1.04 | 3 |

### Full pipeline – main fusion → adaptive threshold → S4 triage

```
GECCO 2018 (stacking)                 real   →sensor contam.  waterFA/d  F1  | faults→sensor →water  sensorFA/d
  no triage                           38/46    0      18/30     1.00    0.54 |      0/24        14       0.00
  rules triage                        34/46    4      18/30     0.45    0.64 |     18/24         1       0.82
  rules triage without the jump rule  37/46    1      18/30     0.82    0.57 |     11/24         8       0.37

GECCO 2019 (C alone)
  no triage                           12/12    0       7/18     1.09    0.53 |      0/24         5       0.00
  rules triage                        12/12    0       7/18     0.33    0.68 |     13/24         1       1.87
  rules triage without the jump rule  12/12    0       7/18     1.04    0.54 |     10/24         4       0.72
```

What the numbers say:

1. **C4 – the adaptive threshold keeps its promise; fixed thresholds don't.** With the same budget
   (1/day) set on the same pre-test water, a fixed threshold gave anything from 0.05 to 3.54 false
   alarms a day in the test season, depending on the fusion and the year. The adaptive threshold gave
   0.90–1.18 for **every** fusion on **both** years. The guarantee is about the average: single weeks
   still vary (worst week about 2/day), as the chart shows.
2. **Holding the budget also finds more events.** In winter 2018 the fixed threshold used only a third
   of the budget (0.31/day); the adaptive one spends it and the classifier catches all 46 real events
   (34 with the fixed threshold, 13 with the S1 F1 rule).
3. **Cross-conformal scoring fixes the S3 transfer problem.** The classifier with the fixed budget
   threshold: 34/46 at 0.31/day on 2018 and 12/12 on 2019 – better than the S1 Random Forest
   (27/46 at 0.15, 10/12) with no tuning on the test period.
4. **C5 – the fusion weights hardly matter.** Paper 1's hand weights, equal weights and learned
   stacking land within one event of each other (adaptive, 2018: 39, 39, 38 of 46), and none beats the
   classifier alone. Learned stacking is also fragile: its weights change completely between years,
   and with a fixed threshold on 2019 it caught 3 of 12. With about 50 training events there is not
   enough to learn weights from; **the threshold matters far more than the weights.** This is the
   honest answer to paper 1's own limitation ("weights set by expert judgement").
5. **E6 – a short persistence rule is nearly free.** Waiting for 3 of 5 minutes costs 3 minutes of
   delay, keeps false alarms at the budget, and catches 3 more real events and 2 more contaminants on
   2018: fewer one-minute false alarms leave room for a lower threshold.
6. **The triage matters more at a sensitive threshold.** Spending the full budget means the detector
   fires on more sensor faults: 14 of 24 faults raised a water alarm without triage on 2018. The S4
   rules bring that to 1, send 18 faults to "check the sensor" and cut false water alarms from 1.00 to
   0.45 a day (2019: 1.09 → 0.33, no real event lost).
7. **The cost of the lone-jump rule, measured.** On 2018 it sends 4 real events to "check the sensor"
   instead of "check the water": three are 18-minute events where conductivity stepped once and the
   detector fired for only 1–2 minutes, one happened while the checks had flagged chlorine sensor 2
   for drift on the real data. Without the rule those events stay water alarms, but 7 more faults become false water
   alarms. Both settings are in the report; which one a utility prefers is an operating choice
   (e.g. `--set fuse.rules.jump_units=8`). The rule also raises false sensor alerts on 2019 (1.87/day).

## 6. Check it worked

- `python -m ewsv2 info` lists `fusion  equal, hand, max, single, stacking` and
  `calibrator  adaptive, budget, cv_f1`.
- `pytest` includes `test_adaptive_holds_budget_when_scores_drift`.
- Every number above must match exactly (no neural network is used in S5).

| Problem | Fix |
|---|---|
| Slow | The two paper-1 modules (ensemble, quorum) take most of the time; the second run with `--set fuse.cache=true` reuses their scores |
| `module_nines_cache.pkl` from an older run | Delete `outputs_v2\s5_fusion\<dataset>\module_nines_cache.pkl`; it is only used with `fuse.cache=true` |

## 7. Viva notes

**What is a "nine"?** How rare a score is among normal minutes: 2 nines = rarer than 1 in 100, 3 nines
= 1 in 1,000. Turning every module into nines puts a classifier probability, a detector vote and a
forecast error on one scale.

**What is conformal about the thresholds?** The threshold is set from the distribution of scores on
normal calibration minutes, with no assumption about their shape; under exchangeability the false
alarm rate then matches the calibration rate. The adaptive version drops the exchangeability
assumption (seasons change) and corrects itself from feedback, with a long-run guarantee (Gibbs &
Candès 2021).

**Isn't using the operator's verdict cheating?** It uses a label only after an alarm has ended, which
is how a control room works: every alarm is investigated and logged as real or false. It never looks
at a minute's label before deciding that minute.

**Why doesn't the adaptive threshold keep every week under budget?** It promises the long-run average.
A week with a burst of false alarms raises the threshold, and the following quiet weeks lower it again.
Keeping every single week under the budget would need a much higher threshold and miss events.

**Why did learned stacking not win?** It learns from out-of-fold scores of about 50 real events; on
2018 it put most weight on the anomaly module, on 2019 on the classifier and forecast. Fitting three
weights is easy; fitting them to something that transfers to the next season is not.

**So were paper 1's weights wrong?** No: on real data they perform as well as learned or equal weights.
The finding is that the weights were never the important choice; the threshold is.

**Why one false alarm per day?** It is the example the project plan used and a rate a small utility
can investigate. It is one config value (`fuse.budget`).

## 8. Next segment

**S6 – Explanations and transfer.** For every alarm, which sensors drove it (SHAP for the classifier,
per-sensor residuals for the forecast), checked against the injected episodes where we *know* which
sensors changed (E9); and training on 2018 to run on 2019 (E10).
