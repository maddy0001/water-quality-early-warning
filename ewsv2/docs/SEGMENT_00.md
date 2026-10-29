# Segment 00 – Foundation and data

**Status:** done · **Builds on:** nothing (first v2 segment) · **Feeds:** every later segment

## 1. Goal

Set up the `ewsv2/` package that every later segment plugs into, download the real
GECCO drinking-water data, and produce a data inspection report. After this segment
you can load GECCO 2018 or 2019 with one line, and you know exactly what the data
contains before any model touches it.

No contribution is proven yet; this segment makes C1–C8 possible. The inspection
already gives two pieces of evidence for C1 (water event vs sensor fault): which
sensors react during real events, and a real frozen sensor.

## 2. What happens

```
 python -m ewsv2 fetch gecco2018          python -m ewsv2 inspect --config s0_inspect
            │                                          │
            ▼                                          ▼
  GitHub (EventDetectR .rda files)          configs/s0_inspect.yaml
            │  converted to CSV                        │  "data.source: gecco2018"
            ▼                                          ▼
  data/raw/gecco/gecco2018_*.csv  ───────►  registry.build("source", "gecco2018")
                                                       │  Gecco2018Source.load()
                                                       ▼
                                             Dataset (one common format)
                                                       │
                                                       ▼
                                   outputs_v2/s0_inspection/gecco2018/
                                   report.md, summary.json, events.csv, figures/
```

1. **Plug-in core.** `core/registry.py` keeps a list of plug-ins by kind and name
   (`source`, `signal`, `fusion`, ...). A class becomes available by adding
   `@register("source", "gecco2018")` above it. Configs pick plug-ins by name.
2. **One data format.** Whatever the file looks like (CSV with `,` or `;`, R data file,
   `TRUE/FALSE` or `0/1` labels, German decimal commas), `data/base.py` turns it into
   the same table: a time index, one float column per sensor, `EVENT` (0/1) and
   `LABEL_CLASS` (0 normal, 1 water event; 2 sensor fault is added in Segment 4).
3. **Data sources.** `gecco2018`, `gecco2019` and a generic `csv` source. GECCO 2018
   keeps its official train/test periods (test starts 2016-11-08 07:55).
4. **Fetch.** GECCO 2018 comes from the EventDetectR GitHub repository (same data as
   Zenodo). GECCO 2019 comes from Zenodo record 4304080 as one file,
   `5_gecco2019_all_water_quality.csv`, which the loader splits at the official
   boundaries (validation starts 2017-10-01 00:00, test starts 2017-11-08 00:30). Both are free.
5. **Inspection.** Sampling and gaps, per-sensor statistics, zero readings, frozen
   stretches, the event list, which sensors move during events, and sensor correlations.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/__init__.py`, `ewsv2/__main__.py` | Package and `python -m ewsv2` entry point |
| `ewsv2/cli.py` | Commands: `info`, `fetch`, `inspect`, `run` |
| `ewsv2/core/types.py` | `Dataset`, `SignalOutput`, `Alarm`, `TriageClass`, `AlertLevel` |
| `ewsv2/core/interfaces.py` | Plug-in contracts for all six layers |
| `ewsv2/core/registry.py` | `register`, `build`, `available` |
| `ewsv2/core/config.py` | YAML configs with `extends:` and `--set` overrides |
| `ewsv2/core/paths.py` | Data folder `data/raw/gecco/`, output folder `outputs_v2/` |
| `ewsv2/data/base.py` | Turns any raw table into the common format |
| `ewsv2/data/gecco.py` | `gecco2018` and `gecco2019` sources, sensor descriptions |
| `ewsv2/data/csv_file.py` | `csv` source for any new station file |
| `ewsv2/data/fetch.py` | Downloads GECCO 2018 and 2019 |
| `ewsv2/data/inspect.py` | The inspection report and charts |
| `ewsv2/configs/base.yaml`, `s0_inspect.yaml`, `full_pipeline.yaml` | Configs |
| `ewsv2/tests/` | 28 tests (synthetic data + checks against the real GECCO 2018 and 2019 counts) |
| `ewsv2/requirements.txt` | v2 packages (separate from v1's pinned file) |
| `.gitignore` | Adds `venv2/`, `data/raw/gecco/`, `outputs_v2/` |

Nothing in v1 (`main.py`, `src/`, `pipelines/`, `dashboard/`, `requirements.txt`) is changed.

## 4. Run it (PowerShell)

Copy the `ewsv2` folder and `.gitignore` from the zip into your project folder, so
`ewsv2\` sits next to `main.py`. Then:

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"

# 0. Freeze paper 1 forever (once)
git tag v1.0 144014c
git push origin v1.0

# 1. A separate environment for v2 (v1 keeps using venv\)
py -3.11 -m venv venv2
venv2\Scripts\Activate.ps1
#    If PowerShell blocks the script, run this once, then activate again:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r ewsv2\requirements.txt

# 2. Check the package
python -m ewsv2 info

# 3. Download the data (about 20 MB for 2018)
python -m ewsv2 fetch gecco2018
python -m ewsv2 fetch gecco2019

# 4. Inspect
python -m ewsv2 inspect --config s0_inspect
python -m ewsv2 inspect --config s0_inspect --set data.source=gecco2019

# 5. Tests
python -m pytest ewsv2\tests -q

# 6. Pipeline skeleton (shows which steps arrive in later segments)
python -m ewsv2 run --config full_pipeline

# 7. Save to GitHub (first push of this branch needs -u once; later just: git push)
git add ewsv2 .gitignore
git commit -m "ewsv2 S0: plug-in core, GECCO loaders, data inspection"
git push -u origin main
```

To go back to v1 at any time: `deactivate`, then `venv\Scripts\activate`, then `python main.py ...` as before.

## 5. Expected output

### GECCO 2018 (verified in the cloud run)

```
Dataset      : gecco2018
Rows         : 279,132  (2016-08-03 09:49:00 -> 2017-02-13 06:00:00)
  train : 139,566 rows, 51 events, 1,726 event minutes (1.237%)
  test  : 139,566 rows, 46 events, 2,329 event minutes (1.669%)
Events total : 97  (median length 22.0 min)
Top sensors during events: pH, Cl_2, Cl
```

| Fact | Value |
|---|---|
| Sampling | 1 minute, no irregular steps, no missing timestamps |
| Rows with every sensor empty | 25,524 (1,044 in train, 24,480 in test) |
| Longest outage | 13,128 minutes (about 9 days) from 2016-12-12 06:01 |
| Event length | median 22 min, shortest 10 min, longest 450 min; 14 events last an hour or more |
| Sensor that moves most in events | pH: shifts by more than 1 SD in 64% of events; temperature in 0% |
| Strongest sensor coupling (normal data) | pH–redox −0.73, chlorine–redox −0.71, temperature–pH 0.75 |
| Frozen-sensor candidate | conductivity (`Leit`) stuck at exactly 198.0 for 2,654 minutes, 2017-02-07 04:43 to 2017-02-09 00:56 |
| Zero readings | chlorine dioxide 2: 199, chlorine dioxide 1: 152; temperature, conductivity, turbidity, flows: 4–7 each |

Files in `outputs_v2\s0_inspection\gecco2018\`: `report.md`, `summary.json`, `events.csv`
(one row per event), `sensors.csv`, `event_response.csv`, `correlations.csv`, and
`figures\overview.png`, `event_lengths.png`, `event_response.png`, `correlations.png`.

### GECCO 2019 (verified with the files downloaded by hand)

```
Dataset      : gecco2019
Rows         : 218,880  (2017-07-01 00:00:00 -> 2017-11-29 23:59:00)
  train : 132,480 rows, 13 events, 212 event minutes (0.16%)
  valid : 54,750 rows, 12 events, 225 event minutes (0.411%)
  test  : 31,650 rows, 12 events, 191 event minutes (0.603%)
Events total : 37  (median length 16.0 min)
Top sensors during events: Turb, Cond, pH
```

| Fact | Value |
|---|---|
| Sensors | 6: `Tp`, `pH`, `Cond` (conductivity), `Turb` (turbidity), `SAC` (spectral absorption, organics), `PFM` |
| Sensor that moves most in events | turbidity: shifts by more than 1 SD in 78% of events; temperature and `PFM` in 0% |
| Strongest coupling (normal data) | temperature–SAC 0.88, conductivity–SAC −0.77, temperature–conductivity −0.74 |
| Frozen-sensor candidate | temperature stuck at exactly 12.31 from 2017-10-19 06:49 to 2017-10-26 04:46 (about 7 days; no events inside) |
| Telemetry glitches | 3 minutes where all sensors read 0 at once (2017-08-24 12:45 and 13:20, 2017-10-10 13:03) |
| Longest outage | 2,146 minutes from 2017-10-09 01:17 |

The two years show different kinds of events: 2018 events are mostly **pH** changes,
2019 events mostly **turbidity** changes. A method that works on both is not tuned
to one sensor, which is why GECCO 2019 is the transfer test in Segment 6.

Which 2019 files you need: only `5_gecco2019_all_water_quality.csv`. If `1_...` and
`6_..._test...` are also in the folder, the loader still uses the "all" file. If only
part files are present, it loads them and prints a warning that periods are missing.

### Test run

```
28 passed
```

Two of these tests check the real data (2018: 51 + 46 events; 2019: 212 / 225 / 191 event
minutes). If a dataset is not downloaded, its test is skipped (e.g. `27 passed, 1 skipped`).

## 6. Check it worked

- `python -m ewsv2 info` lists `source  csv, gecco2018, gecco2019`.
- The GECCO 2018 console numbers match section 5 exactly (51 + 46 events). These match
  the published challenge data; any difference means the download is incomplete.
- Open `report.md` in VS Code (Ctrl+Shift+V shows it with the charts).

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `No module named ewsv2` | Run commands from the project folder (the one containing `ewsv2\`) |
| `No gecco2018 files found` | Run `python -m ewsv2 fetch gecco2018`, or put the Zenodo CSV in `data\raw\gecco\` |
| `pyreadr` fails to install | `pip install --upgrade pip`, then retry; or download the 2018 CSV from Zenodo by hand |
| Fetch fails (firewall) | The error prints the exact link; download in a browser into `data\raw\gecco\` |
| `pip` tries to build numpy | You are on Python 3.13; recreate the venv with `py -3.11` |
| `fetch gecco2019` gives HTTP 404 | Fixed in this update (it now downloads `5_gecco2019_all_water_quality.csv`); or download that file by hand |
| `git push`: no upstream branch | `git push -u origin main` once; plain `git push` afterwards |
| `LF will be replaced by CRLF` warnings | Harmless: Git converts line endings on Windows; nothing to do |

## 7. Viva notes

**Why GECCO instead of the Kaggle potability data?** GECCO is real one-minute data from
a German utility with expert-labelled events, so detection delay and false alarms per
day can be measured. The Kaggle labels behave like noise (all features correlate with
the label below 0.035), as shown in paper 1.

**Why keep v1 untouched?** Paper 1 must stay reproducible. It is tagged `v1.0`; v2 lives
in `ewsv2/` with its own environment, so neither can break the other.

**Why one common data format?** It is what makes the system modular: any detector,
fusion method or report works on any source, including a future ESP32 stream or a new
utility's CSV, without code changes.

**Why does the test period have 24,480 empty minutes?** The utility's data has a long
outage (about 9 days from 12 December 2016). We keep those rows as missing rather than
filling them, so no model learns from made-up values; later segments skip them when
scoring.

**What does the event-response table show?** Real events are not random: pH moves in
64% of them and temperature never does. And pH, chlorine and redox move together in
normal operation (correlations around ±0.7). A real water change should therefore
move several coupled sensors at once, while a faulty sensor moves alone. That is the
physical basis of the water-vs-sensor triage (C1).

**Is the frozen conductivity a real fault?** It is a strong candidate: conductivity
normally varies (SD 16.5), yet it reads exactly 198.0 for 44 hours, and 717 labelled
event minutes fall inside that window. We treat it as a candidate, not a proven fault,
and Segment 4 checks it with the constancy rule.

**Licence?** GECCO data is CC BY 4.0 (Thüringer Fernwasserversorgung, TH Köln); we cite
it and do not redistribute it (it is downloaded by the user and git-ignored).

## 8. Next segment

**S1 – Evaluation and audit.** Uses the `Dataset` and its official periods from S0 to build:
leakage-free splits (blocked, with a gap), event-level metrics (events caught, detection
delay, false alarms per day), bootstrap intervals, and the re-run of published GECCO
protocols that shows how much random splits inflate scores (contribution C2).
