# Segment 01 – Evaluation and leakage audit

**Status:** done · **Builds on:** S0 (datasets, registry) · **Feeds:** every later segment
**Proves:** contribution **C2** (leakage and metric audit) and experiment **E2** (honest baselines)

## 1. Goal

Decide, once and for all, *how* every v2 result is measured, and show with numbers how
much the usual shortcuts inflate scores on GECCO. After this segment:

- every model is judged the way a utility would judge it: **events caught, minutes of
  warning, false alarms per day**, with 95% intervals;
- there is an honest protocol that later segments reuse unchanged;
- there is a baseline table that every v2 signal must beat.

## 2. What happens

```
 Dataset (S0)
     │
     ▼
 Splitter ──► fit rows ──► Signal.transform (whole stream, causal)
     │                         │
     │                         ▼
     │                    Signal.fit (fit rows only)
     │                         │
     ▼                         ▼
 val rows ──► out-of-fold scores ──► threshold chosen here (honest)
 test rows ─► final scores ───────► alarms ──► event metrics + bootstrap ──► report
```

1. **Metrics** (`evaluate/metrics.py`). A labelled event is *caught* if an alarm fires
   inside it. *Delay* is minutes from event start to the first alarm. A *false alarm* is a
   run of alarm minutes outside every event; runs closer than 5 minutes merge, and an alarm
   left on counts again every 60 minutes (an operator re-checks it hourly). Minutes with no
   data are never scored. Point F1 and point-adjusted F1 are also computed, only so we can
   compare with papers.
2. **Splits** (`evaluate/splits.py`):
   - `official` (honest): test = the official test period, untouched. The threshold is chosen
     by 4-fold *blocked* cross-validation over everything before the test period: each time
     block is scored by a model trained on the other blocks, with 1 day removed on both sides.
     Then the model is refitted on all pre-test data and applied once to the test period.
   - `chrono`: time-ordered fractions (used to mimic the NIPS-TS-GECCO 50/50 setup).
   - `random` (leaky): minutes shuffled at random; kept only for the audit.
3. **Thresholds** (`evaluate/thresholds.py`): search over score quantiles on the chosen rows
   (validation for honest runs, test for the audit's shortcut protocols).
4. **Baselines** (`signals/baselines.py`): random scores (a control), squared difference and
   moving variance (the "one-liners" from the ICLR 2026 study), rolling z-score, static limits,
   Isolation Forest, and the supervised Random Forest from the pilot. All use causal features
   (`features/basic.py`): every value at time t uses only readings up to t.
5. **Audit** (`evaluate/audit.py`, config `s1_audit`): the same seven signals under four protocols:

| | Protocol | Seen in |
|---|---|---|
| A | Random 70:30 split of minutes, threshold tuned on test, point F1 | random-split GECCO papers (e.g. GRU-PINN, 2025) |
| B | First vs second half of the 2018 training file, threshold tuned on test, point-adjusted F1 | NIPS-TS-GECCO benchmark papers (e.g. DCdetector, MFAM-AD) |
| C | Official test period, threshold tuned on test | isolates the threshold shortcut |
| D | Official test period, threshold from blocked cross-validation | **honest – used from now on** |

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/evaluate/metrics.py` | Point, point-adjusted and event metrics; bootstrap intervals |
| `ewsv2/evaluate/splits.py` | `official`, `chrono`, `random` splitters |
| `ewsv2/evaluate/thresholds.py` | Threshold search |
| `ewsv2/evaluate/audit.py` | Runs protocols × signals; writes `results.csv`, `per_event.csv`, `report.md`, `audit.png` |
| `ewsv2/features/basic.py` | Causal per-sensor features |
| `ewsv2/signals/baselines.py` | Seven baseline signals |
| `ewsv2/configs/s1_audit.yaml`, `s1_baselines.yaml` | The two S1 experiments |
| `ewsv2/core/types.py`, `core/interfaces.py` | `Split` type; Signal contract is now transform → fit → score |
| `ewsv2/core/registry.py` | Loads the new plug-in packages |
| `ewsv2/cli.py` | New command: `evaluate` |
| `ewsv2/tests/test_metrics.py`, `test_evaluate.py` | 24 new tests (52 in total) |
| `ewsv2/requirements.txt` | Adds scikit-learn |

## 4. Run it (PowerShell)

Replace the `ewsv2` folder with the one from the zip. Use PowerShell's own unzip, not
Explorer's "Extract All" (which can nest folders or leave old files behind). Deleting
`ewsv2\` is safe: S0 is committed in Git, and your data (`data\raw\gecco\`) and
results (`outputs_v2\`) live outside it.

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S1 code (change the path if the zip is not in Downloads)
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S1.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S1)
python -m ewsv2 info

pip install -r ewsv2\requirements.txt          # adds scikit-learn

python -m pytest ewsv2\tests -q                 # 52 passed

# Leakage audit on GECCO 2018 (about 4 minutes)
python -m ewsv2 evaluate --config s1_audit

# Honest baselines (about 2-3 minutes each)
python -m ewsv2 evaluate --config s1_baselines
python -m ewsv2 evaluate --config s1_baselines --set data.source=gecco2019

git add ewsv2
git commit -m "ewsv2 S1: event metrics, honest protocol, leakage audit, baselines"
git push
```

Reports: `outputs_v2\s1_audit\gecco2018\report.md` and `outputs_v2\s1_baselines\<dataset>\report.md`
(open in VS Code with Ctrl+Shift+V to see the chart).

## 5. Expected output (verified in the cloud run)

### Leakage audit, GECCO 2018

Each cell is the score that protocol would report for the same signal on the same data.

| Signal | **Honest event F1 (D)** | A: random split, point F1 | B: half split, point-adjusted F1 | C: threshold on test, event F1 | Largest inflation |
|---|---|---|---|---|---|
| random_score | **0.04** | 0.03 | 0.37 | 0.05 | 9.7× |
| squared_diff | **0.12** | 0.18 | 0.52 | 0.20 | 4.1× |
| moving_variance | **0.12** | 0.25 | 0.51 | 0.20 | 4.1× |
| rolling_zscore | **0.12** | 0.07 | 0.78 | 0.23 | 6.5× |
| static_limits | **0.20** | 0.41 | 0.55 | 0.23 | 2.8× |
| isolation_forest | **0.22** | 0.54 | 0.73 | 0.40 | 3.3× |
| random_forest | **0.64** | 0.79 | 0.50 | 0.68 | 1.3× |

What it shows:

- **Point adjustment** is the biggest inflator: random scores reach 0.37 and the rolling
  z-score 0.78, against honest event F1 of 0.04 and 0.12.
- **A random split** makes the Random Forest look near-perfect: event F1 0.98 under A
  (93 of 97 events, no false alarms) against 0.64 honestly. Point F1 drops from 0.79 to 0.49.
- **Tuning the threshold on test** alone nearly doubles Isolation Forest (0.22 → 0.40) and
  lifts its event recall from 13% to 43%.

### Honest baselines (protocol D) – the bar to beat

GECCO 2018 (46 test events, 78 days of normal data):

| Signal | Events caught | Event recall [95% CI] | False alarms / day [95% CI] | Median delay, min | Event F1 |
|---|---|---|---|---|---|
| random_forest (supervised) | 27/46 | 0.59 [0.46–0.72] | 0.15 [0.05–0.29] | 1 | **0.64** |
| isolation_forest | 6/46 | 0.13 [0.04–0.24] | 0.03 [0.00–0.06] | 8 | 0.22 |
| static_limits | 5/46 | 0.11 [0.02–0.20] | 0.00 [0.00–0.00] | 2 | 0.20 |
| squared_diff | 5/46 | 0.11 [0.02–0.20] | 0.37 [0.23–0.52] | 2 | 0.12 |
| moving_variance | 5/46 | 0.11 [0.02–0.20] | 0.37 [0.24–0.50] | 1 | 0.12 |
| rolling_zscore | 3/46 | 0.07 [0.00–0.13] | 0.01 [0.00–0.04] | 0 | 0.12 |
| random_score (control) | 46/46 | 1.00 | 29.45 | 0 | 0.04 |

GECCO 2019 (12 test events, 22 days of normal data):

| Signal | Events caught | Event recall [95% CI] | False alarms / day [95% CI] | Median delay, min | Event F1 |
|---|---|---|---|---|---|
| random_forest (supervised) | 10/12 | 0.83 [0.58–1.00] | 0.00 [0.00–0.00] | 1 | **0.91** |
| static_limits | 4/12 | 0.33 [0.08–0.58] | 0.14 [0.00–0.28] | 2 | 0.42 |
| moving_variance | 5/12 | 0.42 [0.17–0.67] | 0.32 [0.09–0.60] | 2 | 0.42 |
| isolation_forest | 5/12 | 0.42 [0.17–0.67] | 0.37 [0.09–0.73] | 2 | 0.40 |
| squared_diff | 5/12 | 0.42 [0.17–0.67] | 0.46 [0.18–0.78] | 1 | 0.37 |
| rolling_zscore | 2/12 | 0.17 [0.00–0.42] | 0.18 [0.00–0.41] | 0 | 0.22 |
| random_score (control) | 12/12 | 1.00 | 29.52 | 0 | 0.04 |

Your laptop should give the same numbers. If one differs by an event or so, that is
floating-point noise between machines in the forest models; the conclusions do not change.

Run times (cloud, 2–4 cores): audit 3.5 min, each baseline run 2–2.5 min.

## 6. Check it worked

- `python -m ewsv2 info` lists `splitter chrono, official, random` and seven signals.
- The audit console ends with protocol D, where `random_forest` shows `27/46` caught.
- `report.md` starts with the inflation table above; `audit.png` shows black triangles
  (honest) to the left of the coloured markers (shortcuts).

| Problem | Fix |
|---|---|
| `No module named sklearn` | `pip install -r ewsv2\requirements.txt` |
| `cannot import name 'SEGMENT'` (or any other import error) | The folder is a mix of old and new files: redo step 1 (delete `ewsv2`, `Expand-Archive`) |
| `write threshold.choose_on` error | In your own configs, use `choose_on:` (YAML reads a bare `on:` as true/false) |
| Audit takes much longer than 5 min | Close other programs; the Random Forest uses all CPU cores |
| Numbers differ a lot from section 5 | Check S0 first: `python -m ewsv2 inspect --config s0_inspect` must show 51 + 46 events |

## 7. Viva notes

**What is data leakage here?** With a random split, minute *t* is in training and minute
*t+1* in test. They are almost identical, so the model has effectively seen the answer.
Our audit shows a Random Forest going from event F1 0.98 (random split) to 0.64 (time-ordered).

**What is point adjustment and why is it a problem?** If any one minute of an event is
flagged, the whole event counts as correctly found. A detector that fires randomly hits
part of most long events, so it looks good: random scores reach 0.37 on GECCO, against 0.04
honestly. Kim et al. (AAAI 2022) showed the same effect on other benchmarks.

**Why choose the threshold by cross-validation instead of a validation block?** GECCO
2018's training events cluster in September; the last quarter of the training period has
only a couple of events, so a threshold tuned there just memorises them. Blocked
cross-validation uses every pre-test event, with a one-day gap so neighbouring minutes never
straddle a boundary, and never touches the test labels.

**Why count a long false alarm more than once?** An alarm that stays on for hours keeps
demanding attention. Without this rule, an alarm switched on permanently would count as a
handful of "episodes" and look perfect – we found and closed exactly that loophole while
building this segment (unit test `test_always_on_alarm_cannot_score_well`).

**Why are the unsupervised baselines so weak?** They see one sensor at a time or treat all
deviations equally. GECCO events are specific (mostly pH in 2018, turbidity in 2019); the
supervised Random Forest learns which changes matter. Segments 2–5 build signals that combine
both, and add the water-vs-sensor triage.

**Are these numbers final results?** They are the baselines. Every v2 signal is judged under
protocol D and must beat the Random Forest's 27/46 events at 0.15 false alarms/day on 2018
and 10/12 on 2019.

**Why are the 2019 intervals so wide?** Only 12 test events. With 12 events, recall can only
move in steps of 8%; the bootstrap interval (0.58–1.00) says so honestly.

## 8. Next segment

**S2 – Detectors and classifier.** Ports the v1 four-detector 2-of-4 quorum and the
RF + XGBoost + SVM weighted ensemble into the Signal interface, adds richer rolling and
coherence features, and evaluates them under protocol D against the table above.
