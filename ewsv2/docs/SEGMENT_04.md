# Segment 04 – Water event or sensor fault? (GECCO-Triage)

**Status:** done · **Builds on:** S0 (data), S1 (honest protocol, event metrics), S2 (coherence idea) · **Feeds:** S5 (fusion and calibration), S6 (explanations), S7 (incident reports)
**Proves:** contribution **C1** (station-level triage: "check the water" vs "check the sensor") and
**C3** (the GECCO-Triage benchmark); experiments **E7** (contaminants by type and dose) and **E8**
(sensor faults by type)

## 1. Goal

Every published GECCO detector, and paper 1, raises one kind of alarm. In a real utility many alarms
come from the instrument, not the water: a frozen transmitter, a lost 4–20 mA signal, a fouled probe.
An operator who is sent to sample the water for a broken sensor stops trusting the system.

This segment makes the alarm say **which**: *check the water* or *check sensor X*. To measure that,
it builds a benchmark with known answers – the real GECCO stream with realistic contaminant slugs and
sensor faults injected at known times – and compares three station-level systems on it.

## 2. What happens

```
 real GECCO stream ──► inject/ ──► GECCO-Triage stream + episode table
                        │  contaminants (5 types × 3 doses)  → labelled water events  (class 1)
                        │  sensor faults (4 types × 3 levels) → labelled sensor faults (class 2)
                        │  one set before the test period (training), one inside it (scoring)
                        ▼
 ┌──────────────── three systems, same honest protocol as S1 ────────────────┐
 │ detector_only   S1 Random Forest → every alarm is a water alarm            │
 │ rules           same detector + 4 fault checks (no fault labels used):     │
 │                 withdraw alarms that only suspect sensors explain;         │
 │                 raise "check sensor X" alerts                              │
 │ learned         one 3-class Random Forest (normal / water / fault)         │
 └────────────────────────────────────────────────────────────────────────────┘
                        ▼
  every test episode → water alarm / sensor alert only / both / missed      (+ real-data scan)
```

1. **Contaminant signatures** (`inject/signatures.py`). The *pattern* – which sensors move and in
   which direction – follows the US EPA pipe-loop injection studies (Hall et al., J. AWWA 2007; EPA
   report 600/R-09/076, 2009): chlorine and organic carbon respond to most contaminants, ORP follows
   chlorine, conductivity responds to salts and sewage, turbidity to sewage and biological material,
   pH to acids and bases, temperature and flow do not respond.

   | Type | Example | Main responses |
   |---|---|---|
   | pesticide | aldicarb, glyphosate | chlorine ↓, ORP ↓, organics ↑ |
   | biological | E. coli in growth broth | chlorine ↓, organics ↑, turbidity ↑, ORP ↓ |
   | inorganic | arsenite, ferricyanide | conductivity ↑, chlorine ↓, ORP ↓ |
   | wastewater | sewage intrusion | turbidity ↑, conductivity ↑, chlorine ↓, organics ↑ |
   | alkaline | caustic over-dose | pH ↑, ORP ↓, conductivity ↑ |

   The *size* is set in units of each sensor's normal 60-minute variability (`features/scale.py`)
   and swept over three doses: 4, 8 and 15 units for the lead sensor. A slug passes the station with
   a smooth 4–10 min rise, a 10–60 min plateau and a slower fall; each sensor gets a 0–3 min lag and
   ±25% jitter. Chlorine, conductivity, turbidity and organics never go below 0.

   A signature needs at least two responding sensors at the station. GECCO 2019 has no chlorine and
   no ORP sensor, so **pesticide and inorganic contaminants cannot be seen there** and are not injected.
   This is a result in its own right: which contaminants a station can see depends on its sensors.

2. **Sensor faults** (`inject/faults.py`), on one quality sensor at a time:
   `spike` (1–4 one-minute outliers, 8–40 units), `drift` (an offset growing to 6–30 units over
   2–8 h), `stuck` (frozen at the last value for 1–24 h), `dropout` (reads 0 for 10 min–6 h).

3. **Benchmark** (`inject/benchmark.py`, injector `gecco_triage`). Two copies of every (type, level)
   in the test period and two in the pre-test period, placed at random (seeded) where at least 90% of
   minutes have data, and at least 4 h from real events and from each other. 2018: 54 + 54 episodes;
   2019: 42 + 42. Real events stay in the data as water events.

4. **Fault checks** (`signals/faults.py`). Learned from normal pre-test minutes only – no fault labels:

   | Check | Fires when |
   |---|---|
   | stuck | the reading has not changed for longer than its limit **and** the partner sensors say it should have |
   | dropout | the reading falls in one step to (near) zero |
   | spike | a one- or two-minute excursion comes straight back (flagged on return) |
   | drift | the sensor's 4-hour change disagrees with what its partner sensors predict, for 90 min in a row |

   One parameter sets all data-driven limits: a **false-flag budget** of 0.5 sensor flags per day on
   normal pre-test water, shared by the checks and sensors. Whether the budget still holds in the test
   season is measured, not assumed.

5. **Rules triage** (`triage/systems.py`). A sensor becomes a *suspect* when a check flags it, or when,
   during an alarm, it jumps alone (≥ 6 units in one minute, most of its 15-minute move, no other
   sensor moving). It stays a suspect for an hour, unless it keeps moving the same way (the front of a
   real change, e.g. organics rising). A water alarm is **withdrawn** when the suspects explain it: no
   other sensor moved more than it does in normal water. Sensor alerts name the sensor.

6. **Learned triage.** One 3-class Random Forest on the basic features plus all triage evidence,
   trained on the pre-test half of GECCO-Triage (which includes injected faults). Two thresholds
   (water, fault) are chosen by blocked cross-validation, like every threshold since S1.

7. **Scoring** (`triage/evaluate.py`). Each test episode gets one outcome per system. Rates use the S1
   event evaluator: a water alarm during a sensor fault is a false water alarm. A false sensor alert is
   counted once however long it lasts (it is a maintenance ticket).

8. **Real-data scan.** The fault checks also run on the untouched GECCO stream, to see whether they
   find the sensor problems spotted by eye in S0.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/inject/signatures.py` | Contaminant response patterns, roles of each sensor, doses, slug shape |
| `ewsv2/inject/faults.py` | The four fault types |
| `ewsv2/inject/benchmark.py` | `gecco_triage` injector: plans, places and labels episodes |
| `ewsv2/features/scale.py` | One "normal variability" unit per sensor |
| `ewsv2/signals/faults.py` | The four fault checks; also registered as signal `fault_checks` |
| `ewsv2/triage/features.py` | Per-minute water and sensor evidence |
| `ewsv2/triage/systems.py` | `detector_only`, `rules`, `learned` |
| `ewsv2/triage/evaluate.py` | Episode outcomes and rates |
| `ewsv2/triage/run.py` | The S4 experiment, report and chart |
| `ewsv2/configs/s4_triage.yaml` | The S4 experiment |
| `ewsv2/cli.py` | New commands `triage` and `inject` |
| `ewsv2/core/registry.py` | New plug-in kind `injector`; loads `inject` and `triage` |
| `ewsv2/tests/test_inject.py`, `test_triage.py` | 20 new tests (93 in total) |
| `.gitignore` (repository root) | One line added so the v2 test files are saved to GitHub (see below) |

**Important – test files were never on GitHub.** The v1 `.gitignore` contains the rule `test_*.py`,
so every `ewsv2\tests\test_*.py` from S0–S3 was silently left out of your commits (only `conftest.py`
and `__init__.py` are on GitHub). Step 3 below adds one exception line; after that `git add` picks
up all nine test files.

No new packages: `requirements.txt` is unchanged.

## 4. Run it (PowerShell)

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S4 code
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S4.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S4)
python -m ewsv2 info

# 3. One time only: let Git save the v2 test files (v1's .gitignore hides test_*.py)
Add-Content -Path .gitignore -Value "", "!ewsv2/tests/test_*.py"

python -m pytest ewsv2\tests -q                 # 93 passed

# 4. The experiment (cloud, 2 cores: about 4 min for 2018, 7 min for 2019)
python -m ewsv2 triage --config s4_triage
python -m ewsv2 triage --config s4_triage --set data.source=gecco2019

# 5. Save
git add .gitignore ewsv2
git status                                      # should list ewsv2/tests/test_*.py as new files
git commit -m "ewsv2 S4: GECCO-Triage benchmark, fault checks, water-vs-sensor triage"
git push
```

Reports: `outputs_v2\s4_triage\gecco2018\report.md` and `...\gecco2019\report.md`, each with
`triage.png`, `episodes.csv` (the benchmark), `outcomes.csv` (every episode × system),
`summary.csv` and `real_fault_scan.csv`.

Optional: `python -m ewsv2 inject --config s4_triage` writes the whole GECCO-Triage stream as a CSV to
`outputs_v2\benchmarks\` (about 30 MB; this is the file a benchmark release would share).

## 5. Expected output (verified in the cloud run)

*Updated in S5.* The first S4 release gave slightly different numbers on different computers
(tiny last-digit differences in the injected values reached the models). Since S5 the injected
values are rounded to each sensor's reading step, as a real sensor reports them, and the numbers
below are identical everywhere. The conclusions did not change.

### GECCO 2018 – official test period (46 real events, 30 injected contaminants, 24 injected faults)

```
  system                             real  contam. waterFA/d | fault->sensor fault->water missed sensorFA/d
  Detector only (v1-style)         28/46    17/30       0.45 |         0/24         7/24      17       0.00
  Detector + fault checks (rules)  28/46    17/30       0.21 |        14/24         1/24       9       0.44
  Learned three-class triage       21/46    27/30       0.21 |        11/24         1/24      12       0.33
```

### GECCO 2019 – official test period (12 real events, 18 injected contaminants, 24 injected faults)

```
  system                             real  contam. waterFA/d | fault->sensor fault->water missed sensorFA/d
  Detector only (v1-style)          9/12     3/18       0.14 |         0/24         3/24      21       0.00
  Detector + fault checks (rules)   9/12     3/18       0.00 |        14/24         0/24      10       0.99
  Learned three-class triage        8/12    10/18       0.00 |        11/24         0/24      13       0.00
```

`waterFA/d` = false water alarms per day (including alarms raised by faulty sensors);
`fault->sensor` = faults that got "check the sensor" and no water alarm; `fault->water` = faults that
raised a water alarm; `sensorFA/d` = false sensor alerts per day.

| | 2018 | 2019 |
|---|---|---|
| Water event F1, detector only → rules | 0.58 → 0.66 | 0.53 → 0.57 |
| Faulty sensors that raised a water alarm, detector only → rules | 7 → 1 | 3 → 0 |
| Real events lost by the triage | 0 | 0 |
| Sensor alerts that name the right sensor (rules) | 100% | 93% |
| False sensor alerts / day (budget 0.5, set on pre-test water) | 0.44 | 0.99 |

What the numbers say:

1. **C1 – the triage works, at no cost to detection.** With the same detector and threshold, the
   rules turn 14 of 24 faults into "check the sensor" on both datasets, stop faulty sensors from
   raising water alarms (7 → 1 and 3 → 0), cut false water alarms by more than half (2018: 0.45 →
   0.21 per day) and to zero (2019),
   and lose no real event. No fault label is used anywhere in the rules.
2. **Rules beat learning for faults.** The learned triage saw 24 injected faults in training and
   still gets fewer right (11 and 11) and catches fewer real events (21/46, 8/12): with few examples a
   classifier learns the training faults, while the checks encode what a fault *is*.
3. **E7 – a detector trained on past events only knows those events.** The S1 detector learned the
   2018 events, which are mostly pH changes. It catches alkaline and high-dose slugs but only 1 of 6
   inorganic slugs and no low-dose slug of any type. On 2019 (turbidity-led events) it catches 3 of 18
   and no alkaline slug at all. The learned triage, trained on injected signatures, catches 27/30 on
   2018 and 10/18 on 2019 – but only for signature types it has seen. This is the case for S5: fuse the supervised detector with
   the unsupervised and forecast signals of S2–S3, which do not depend on past event types.
4. **E8 – which faults are easy.** Dropouts: 6/6 on both datasets, at the first minute. Spikes and
   stuck sensors: 2–3 of 6. Slow drifts: 2/6 (2018) and 4/6 (2019), usually found only after 4–7 h.
   Stuck readings on coarse sensors (temperature at 0.1 °C) are only caught when partner sensors prove
   the reading should have moved – the check refuses to guess.
5. **The budget held in 2018 (0.44/day) but not in 2019 (0.99/day).** Limits learned in summer
   over-fire in a different season. Holding a stated rate under drift is exactly what the adaptive
   calibration of S5 (C4) is for.
6. **Real data, nothing injected.** On 2018 the longest flag is conductivity (`Leit`) stuck at 198.0
   from 2017-02-07 18:19 for 1,838 min: the frozen sensor found by eye in S0, now found automatically
   (322 labelled event minutes fall inside it). On 2019, the telemetry glitch of 2017-08-24 (all
   sensors read 0) is flagged as a spike on five of the six sensors. The 7-day temperature freeze of October 2017 is **not**
   flagged: its partners never implied a change of more than 3 units, so the stuck check stays silent.

## 6. Check it worked

- `python -m ewsv2 info` shows `injector  gecco_triage`, `signal ... fault_checks ...` and
  `triage  detector_only, learned, rules`.
- The detector-only and rules rows must match section 5 exactly (Random Forests are deterministic).
- `git status` after step 5's `git add` lists the nine `ewsv2/tests/test_*.py` files.

| Problem | Fix |
|---|---|
| `Could not place ... skipped` warning | A free stretch could not be found; it only happens if you raise `reps_test`. Lower it or `buffer_minutes` |
| `git status` still does not show the test files | Open `.gitignore` in VS Code and check that the last line is exactly `!ewsv2/tests/test_*.py` |
| Numbers differ | Make sure the zip replaced the whole `ewsv2` folder (step 1) and that the data files are the same as in S0 |

## 7. Viva notes

**Why inject contaminants instead of using real ones?** Nobody may contaminate a drinking-water
network, and the GECCO events are not labelled with a cause. Injecting events into real data gives
known answers while keeping real background noise; EPA's own event-detection-system challenge evaluated tools the same way.
We keep the real events too, and report them separately.

**How realistic are the signatures?** The direction pattern follows EPA pipe-loop experiments. The
size is swept (three doses in units of normal variability) instead of copied, because EPA's absolute
changes were measured on a much calmer background than a real utility.

**Why can't 2019 see pesticides?** Its station has no chlorine or ORP sensor, and those are the sensors
chlorine-demanding contaminants move. The benchmark reports it instead of injecting invisible events.

**Why do rules beat the learned model for faults?** The learned model needs examples of every fault
on every sensor; it had 24. The rules encode physics: a broken sensor moves alone, too fast, or not
at all while its partners move. They also need no fault labels, so they work at a new station on day one.

**Isn't withdrawing alarms dangerous?** An alarm is withdrawn only when every sensor that is not a
suspect stayed within its normal 15-minute range. A real slug moves several sensors, and a single
sensor that keeps changing over minutes (the 2019 organics events) is not treated as a suspect. The
result: no real event was lost on either dataset.

**Why is the false-sensor-alert rate higher in 2019?** The limits were learned in summer and the test
is in winter. We report the achieved rate against the budget instead of tuning on the test period;
S5 adds adaptive calibration to hold it.

**What about the real frozen conductivity in 2018?** Found automatically: stuck at 198.0, flagged at
18:19 on 7 Feb 2017 and held for 1,838 minutes. It was flagged about 13 h after the freeze began
because conductivity is quantised to 1 µS/cm and legitimately stays constant for hours; the check
waits for the partner sensors to prove it should have changed.

## 8. Next segment

**S5 – Fusion and calibration.** Combines the signals built so far – the S1/S2 classifiers, the S2
quorum, S3 forecast surprise and the S4 triage evidence – with paper 1's hand weights, equal weights
and learned stacking; then sets the alarm threshold with adaptive conformal calibration, so the
false-alarm rate holds at a stated budget across seasons (contributions C4 and C5; E4–E6).
