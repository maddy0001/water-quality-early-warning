# Segment 06 – Explanations that are checked, and moving to a new station

**Status:** done · **Builds on:** S2 (classifier), S3 (forecast), S4 (GECCO-Triage), S5 (pipeline alarms) · **Feeds:** S7 (incident reports), S8 (ESP32), S9 (dashboard)
**Proves:** contribution **C6** (explanations measured against known answers, not just shown);
experiments **E9** (are the explanations right?) and **E10** (train on 2018, run on 2019)

## 1. Goal

An alarm is only useful if the operator knows where to look: "pH dropped" or "turbidity rose". Paper 1
showed SHAP bar charts; nobody checked whether they were right. This segment

1. explains every alarm of the S5 pipeline with four methods, and **checks** them: on injected
   episodes the right answer is known (the sensors the signature or the fault changed), and a
   deletion test asks whether the blamed sensor really carries the alarm;
2. moves the system to a station it was not trained on (GECCO 2018 → 2019) and measures what
   survives.

## 2. What happens

```
 S5 pipeline alarms (outputs_v2/s5_fusion/<dataset>/pipeline_minutes.csv.gz)
      │  first alarm minute of every alarmed test episode
      ▼
 four explainers → one score per sensor
   deviation          how far each sensor moved in 15 min (in its own normal units)      no model
   forecast residual  each sensor's 5-min forecast error (module F)
   TreeSHAP           exact SHAP of the classifier's XGBoost part, summed per sensor     module C
   occlusion          hold one sensor at its pre-alarm level, re-score the full ensemble  module C
      │
      ▼
 checks:  injected contaminant → sensors the signature moved      hit@1, lead@1, coverage
          injected fault       → the faulty sensor
          deletion test        → does holding the top sensor remove ≥ 50% of the score?   faithful@1
```

1. **Explainers** (`explain/methods.py`). TreeSHAP uses XGBoost's built-in exact TreeSHAP
   (`pred_contribs`), so no new package is needed. Occlusion is model-agnostic and explains the whole
   RF + XGBoost + SVM ensemble; it is slow (one re-score per sensor) but faithful by construction, so
   it is also the reference for the deletion test.
2. **E9 runner** (`explain/run.py`). Reads the S5 alarms (before triage), rebuilds GECCO-Triage,
   fits the explainers on normal pre-test data, explains the first alarm minute of each alarmed test
   episode, and scores the rankings.
3. **E10 runner** (`explain/transfer.py`). Both stations measure temperature, pH, conductivity and
   turbidity. Each station's four sensors are renamed to these roles and put on the station's own
   normal scale (centre = pre-test median, unit = normal variability; no labels). The classifier is
   trained on the 2018 training period. The anomaly and forecast modules are trained either on 2018
   too, or on the new station's own pre-test data, which needs no labels. Three thresholds: the 2018
   threshold unchanged (zero-shot), a budget threshold from 2019 pre-test minutes treated as normal
   (label-free), and the S5 adaptive threshold.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/explain/methods.py` | `deviation`, `forecast_residual` explainers; `ClassifierExplainer` (TreeSHAP, occlusion) |
| `ewsv2/explain/run.py` | E9: explanations checked against known answers; deletion test |
| `ewsv2/explain/transfer.py` | E10: station normalisation by sensor role, 2018 → 2019 |
| `ewsv2/configs/s6_explain.yaml` | Both experiments |
| `ewsv2/cli.py` | New commands `explain` and `transfer` |
| `ewsv2/core/registry.py` | Loads `explain` |
| `ewsv2/tests/test_explain.py` | 6 new tests (108 in total) |

No new packages.

## 4. Run it (PowerShell)

S6 reads the alarms written by your S5 run (`outputs_v2\s5_fusion\...\pipeline_minutes.csv.gz`), so
S5 does not need to run again.

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S6 code
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S6.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S6)
python -m ewsv2 info
python -m pytest ewsv2\tests -q                 # 108 passed

# 3. E9 – explanations (about 1-2 min each)
python -m ewsv2 explain --config s6_explain
python -m ewsv2 explain --config s6_explain --set data.source=gecco2019

# 4. E10 – train on 2018, run on 2019 (about 3 min)
python -m ewsv2 transfer --config s6_explain

# 5. Save
git add ewsv2
git commit -m "ewsv2 S6: checked explanations, transfer 2018 to 2019"
git push
```

Reports: `outputs_v2\s6_explain\gecco2018\report.md`, `...\gecco2019\report.md` (with
`explanations.png`) and `...\transfer_gecco2018_to_gecco2019\report.md`.

## 5. Expected output (verified in the cloud run)

### E9 – are the explanations right?

`hit@1` = the top sensor is one that really changed; `lead@1` = it is the one that changed most;
`faithful@1` = holding it at its pre-alarm level removes at least half of the classifier's score.
"Random sensor" is the expected value of picking a sensor at random.

GECCO 2018:

```
  injected contaminants (18)        hit@1  lead@1  cover  faithful@1
  deviation (model-free)             0.94    0.50   0.86        0.61
  forecast residual (F)              0.94    0.50   0.90        0.44
  TreeSHAP (classifier C)            1.00    0.50   0.61        0.56
  occlusion (classifier C)           1.00    0.50   0.73        0.72
  random sensor                      0.47                       0.09

  injected faults (14)
  deviation (model-free)             1.00    1.00   1.00        1.00
  forecast residual (F)              1.00    1.00   1.00        1.00
  TreeSHAP (classifier C)            0.64    0.64   0.64        0.64
  occlusion (classifier C)           1.00    1.00   1.00        1.00
  random sensor                      0.11                       0.11

  real events (38)                                         faithful@1
  deviation 0.95 · forecast residual 0.95 · TreeSHAP 0.74 · occlusion 0.95 · random 0.11
```

GECCO 2019:

```
  injected contaminants (7)         hit@1  lead@1  cover  faithful@1
  deviation (model-free)             1.00    0.43   0.83        0.43
  forecast residual (F)              0.71    0.14   0.88        0.14
  TreeSHAP (classifier C)            1.00    0.71   0.73        0.71
  occlusion (classifier C)           1.00    0.71   0.80        1.00
  random sensor                      0.60                       0.26

  injected faults (5)               all four explainers 0.80; random 0.17
  real events (12)                  faithful@1: deviation 1.00 · forecast residual 1.00 · TreeSHAP 0.75 · occlusion 1.00
```

Which sensor the explainers blame first for the **real** events (no ground truth):

| | 2018 (38 alarmed events) | 2019 (12 alarmed events) |
|---|---|---|
| deviation | pH 21, redox 7, conductivity 5, chlorine 1 + 4 (two sensors) | turbidity 8, SAC 4 |
| TreeSHAP | pH 19, redox 6, chlorine 5 + 3, conductivity 2, flow 2, turbidity 1 | turbidity 9, SAC 3 |
| agreement of the top sensor with occlusion | deviation 0.97, forecast 0.95, TreeSHAP 0.74 | 1.00, 1.00, 0.75 |

### E10 – train on 2018, run on 2019 (12 real test events, budget 1/day)

```
  system                                 threshold                    caught  FA/day    F1
  C alone (from 2018)                    zero-shot (2018 threshold)    0/12     1.65  0.00
  C alone (from 2018)                    adaptive                      0/12     1.10  0.00
  A alone (from 2018)                    zero-shot (2018 threshold)    7/12     2.57  0.19
  A alone (fitted at 2019)               adaptive                      7/12     1.05  0.33
  F alone (from 2018)                    label-free (2019 pre-test)    9/12     2.52  0.24
  F alone (fitted at 2019)               adaptive                      5/12     1.05  0.25
  hand 0.40/0.30/0.30 (all from 2018)    zero-shot (2018 threshold)    6/12     1.01  0.30
  hand 0.40/0.30/0.30 (all from 2018)    adaptive                      6/12     1.05  0.29
```

(The console prints all 18 rows.) For comparison, trained on 2019 itself (S5): 12/12 at 1.09/day.

What the numbers say:

1. **C6 – explanations can be checked, and they are mostly right.** For injected contaminants the top
   sensor is one that really changed 94–100% of the time on 2018 (random: 47%) and 71–100% on 2019. For
   faults that slipped into a water alarm, three explainers name the broken sensor every time on 2018.
2. **The simplest explainer is as good as the expensive ones.** "Which sensor moved most, in its own
   normal units" agrees with occlusion on 97% (2018) and 100% (2019) of real events, and passes the
   deletion test as often (0.95, 1.00). It needs no model, so it can run on the ESP32 (S8).
3. **TreeSHAP – paper 1's method – is the weakest.** It explains only the XGBoost part of the ensemble,
   so it names the faulty sensor in 64% of 2018 faults, and its top sensor passes the deletion test on
   74–75% of real events against 95–100% for the others. A SHAP chart is not evidence on its own;
   checking it is.
4. **"Lead sensor" is harder than "a changed sensor".** Contaminants move 3–6 sensors together, and
   several respond by similar amounts (two chlorine sensors in 2018), so naming *the* biggest mover is
   right about half the time on 2018.
5. **The real events have a clear signature.** The explainers blame pH first in about half of the 2018
   events and turbidity or organics (SAC) in every alarmed 2019 event – the same picture S0 found by
   inspection (pH moves in 64% of 2018 events, turbidity in 78% of 2019 events).
6. **E10 – the classifier does not move; the paper-1 fusion does.** Trained on 2018's pH events, the
   classifier catches 0 of 2019's turbidity events. The unsupervised modules still catch up to 7 of 12
   at about one false alarm a day (adaptive), and up to 10 at a higher rate; and
   paper 1's hand-weighted fusion of all three, moved unchanged with its 2018 threshold, catches 6/12
   at 1.01 false alarms/day. The fusion that did not beat the classifier at home (S5) is what keeps
   the system useful at a new station.
7. **Label-free calibration alone overshoots; adaptive holds the budget.** Thresholds set on 2019
   pre-test minutes (no labels) gave mostly 2–4 false alarms/day in the test month; the adaptive threshold
   brought every system back to 0.96–1.19/day. A new station still needs its own labelled events to
   reach S5's 12/12.

## 6. Check it worked

- `python -m ewsv2 info` lists `explainer  deviation, forecast_residual`.
- The summary tables must match section 5 exactly. In `explanations.csv` the classifier score is
  rounded to 6 decimals; the rankings are identical everywhere.

| Problem | Fix |
|---|---|
| `FileNotFoundError: ...pipeline_minutes.csv.gz not found` | Run S5 first for that dataset: `python -m ewsv2 fuse --config s5_fusion [--set data.source=gecco2019]` |
| `transfer` says a station has no sensor with a role | Both GECCO datasets must be in `data\raw\gecco\` |

## 7. Viva notes

**How do you know an explanation is right?** On injected episodes we know which sensors were changed,
so we count how often the top-ranked sensor is one of them. On real events we use a deletion test:
put the blamed sensor back to its pre-alarm level and see whether the alarm score collapses.

**Why is TreeSHAP worse than a simple deviation?** TreeSHAP is exact for the model it explains, but
here that is only one of three classifiers in the ensemble, and its per-feature credit is spread over
many correlated features (deviations, spreads, coherence residuals). The deviation explainer asks the
operator's question directly: which sensor did something unusual?

**Then why keep TreeSHAP?** Paper 1 used it, and the comparison is the point: the measured result is
that a SHAP chart alone can mislead 25% of the time.

**Why does the classifier fail at the new station?** It learned what 2018's events look like (pH moves).
2019's events are turbidity and organics events. No amount of normalisation fixes a different kind
of event; the unsupervised modules, which learn what *normal* looks like, transfer better.

**Is "label-free" really label-free?** The label-free thresholds and the "fitted at 2019" modules use
no 2019 labels. The adaptive threshold reads the operator's verdict after each alarm, as in S5.

## 8. Next segment

**S7 – Local LLM incident reports.** Each alarm (time, triage decision, level, top sensors from the
deviation explainer, values) is turned into a short incident report by a free local language model
(Ollama), grounded on a small index of WHO drinking-water guidance. Every sensor, number and time in
the report is checked automatically against the alarm record (contribution C8).
