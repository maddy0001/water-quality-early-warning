# Segment 10 – Robustness and hand-over

**Status:** done and checked in the cloud (164 tests, including 17 that break parts on purpose); you run it
on your laptop · **Builds on:** S0–S9 · **Feeds:** paper 2, the viva, and every future run of the project
**Delivers:** a system check, failure tests, exact lock files, an offline pendrive capsule, and the paper's
tables and figures

## 1. Goal

From your requirement: *"if I execute this project again after 5 months or more, nothing should be
broken."* Three things can break a project after months:

1. **Something is missing** – the board, a sensor, Ollama, data files, results, version 1.
   → `python -m ewsv2 doctor` names every part, says what replaces it, and how to fix it.
2. **Packages change** – a new pip release, a new Python, no internet in the exam hall.
   → exact lock files of *your* laptop's environments, and an offline **time capsule** on the pendrive that
   rebuilds both environments without internet.
3. **Damaged files** – a half-copied CSV, a corrupted result file.
   → the system falls back instead of crashing; tests prove it.

And for the paper: every table and figure is generated from the results on your laptop by one command.

## 2. What is in it

| Command | What it does |
|---|---|
| `python -m ewsv2 doctor` | Checks Python, packages (against the lock file), plug-ins, data, edge models, results, Arduino and its three sensors, Ollama, version 1, dashboard ports, disk space. Each line: `[ OK ]`, `[ -- ]` (replaced, still works) or `[FAIL]`, with the fix. Writes `outputs_v2/doctor/doctor.md`. Exit code 1 only if something required is missing |
| `python -m ewsv2 paper` | Writes `outputs_v2/paper/`: 9 tables (CSV, `tables.md` for Word, `tables.tex` for LaTeX) and 7 figures (PNG 300 dpi + PDF, IEEE column widths), with `index.md` naming the source file of each |
| `python -m ewsv2 capsule --dest E:\ewsv2_capsule` | Writes the offline time capsule to the pendrive (section 4c) |
| `python -m ewsv2 capsule --check E:\ewsv2_capsule` | Verifies every file of a capsule against its SHA-256 manifest |
| Dashboard → System Status → **Run the full system check** | The same check inside the dashboard |

### Lock files (`ewsv2/handover/locks/`)

Read directly from your laptop's environments on 11 Nov 2026 (Windows, Python 3.11, 64-bit):

- `requirements-lock-v2.txt` – the 87 packages of `venv2` (e.g. numpy 1.26.4, pandas 2.2.3,
  tensorflow-cpu 2.18.1, streamlit 1.49.1, plotly 5.24.1, streamlit-echarts 0.4.0).
- `requirements-lock-v1.txt` – the 100 packages of version 1's `venv` (tensorflow 2.13.0, numpy 1.24.3 …).
  Version 1 itself is not touched.

All of them were checked to exist as Windows wheels for Python 3.11, except `fpdf` and `lime` (version 1),
which only exist as source archives; the capsule builds wheels for those two automatically.

### Failure tests (`ewsv2/tests/test_robustness.py`)

| Broken on purpose | Expected and tested behaviour |
|---|---|
| Data files missing | Synthetic station, with a printed notice |
| Data files damaged (garbage CSV) | Synthetic station, notice says "unreadable" |
| Data missing and no fallback configured | Clear error, no crash |
| Unknown config name | Exit code 2 with a message |
| Ollama unreachable | Template report |
| AI report with a wrong time in it | Fact check catches it; template shown instead |
| Arduino port that does not exist | Gives up in under 5 s; laptop path used |
| No result files at all | Overview, Analytics, Reports, Status pages still render |
| Damaged result files (wrong columns, broken JSON) | Pages still render; the affected tab says which file to regenerate |
| Everything missing at once | `doctor` still runs and lists each fix |
| Capsule file changed or deleted | `capsule --check` reports it (exit code 1) |
| No result files for the paper | Tables and the architecture figure are still written from the bundled results |

(Already covered earlier: board unplugged mid-run, serial noise, missing chart component, no board in the
dashboard.)

### Architecture diagram

`ewsv2/docs/figures/architecture.png` (and `.pdf`), also `F1` in the paper folder: the utility and demo
stations feed four stages (sensor validation, change features, anomaly score, adaptive threshold) that run
bit-identically on the laptop and the Uno, then the decision, operator review (feedback to the threshold)
and the fact-checked incident report.

## 3. Files added or changed

```
ewsv2/handover/__init__.py
ewsv2/handover/doctor.py            system check
ewsv2/handover/capsule.py           time capsule + restore.ps1 + RESTORE.md + manifest check
ewsv2/handover/paper.py             tables and figures
ewsv2/handover/locks/requirements-lock-v2.txt
ewsv2/handover/locks/requirements-lock-v1.txt
ewsv2/tests/test_robustness.py      17 failure tests
ewsv2/docs/figures/architecture.png / .pdf
ewsv2/docs/SEGMENT_10.md            this file
ewsv2/cli.py                        doctor, capsule, paper commands; damaged data → fallback; Ctrl+C exits cleanly
ewsv2/app/views/system.py           "Run the full system check" button
ewsv2/app/views/analytics.py        a damaged result file only affects its own tab
ewsv2/README.md                     hand-over section
ewsv2/__init__.py                   SEGMENT = "S10"
```

## 4. Run it (PowerShell)

### 4a. Install and test

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S10.zip" -DestinationPath . -Force

python -m ewsv2 info                         # built up to segment S10
python -m pytest ewsv2\tests -q              # about 160 passed, 4 skipped (cloud: 164 passed)
```

No new packages.

### 4b. System check and paper material

```powershell
python -m ewsv2 doctor                       # Arduino plugged in: also checks the three sensors
python -m ewsv2 paper
explorer outputs_v2\paper
```

Expected from `doctor` on your laptop: Python 3.11, all packages matching the lock file, both datasets, 10 of
10 analyses, version 1 found. Ollama and the Arduino show `[ OK ]` when running/plugged in, otherwise
`[ -- ]` with the fix. Send me the doctor output.

`paper` uses *your* laptop's results, so T7 includes your AI-model report runs.

### 4c. The pendrive time capsule (do this once, while you have internet)

1. Plug in the 32 GB pendrive and look up its drive letter in File Explorer (below: `E:`).
2. Run (about 15–30 minutes; it downloads about 2 GB of packages and copies the rest):

```powershell
python -m ewsv2 capsule --dest E:\ewsv2_capsule --dry-run          # shows sizes, writes nothing
python -m ewsv2 capsule --dest E:\ewsv2_capsule --with-ollama --with-arduino --with-installers
python -m ewsv2 capsule --check E:\ewsv2_capsule
```

| Part | Size (about) |
|---|---|
| Project: version 1 + 2, data, results, trained models (no venvs) | 0.2–0.5 GB |
| Wheelhouse v2 + v1 (every package, Windows, Python 3.11) | 2 GB |
| Ollama models (qwen2.5:3b and any others you pulled) | 2–4 GB |
| Arduino IDE board files | 0.3–1 GB |
| Installers (Python 3.11, Ollama) | 1.2 GB |

The CH340 driver and the Arduino IDE cannot be downloaded automatically; their links are in `RESTORE.md`.
If you want them on the pendrive too, download them by hand into `E:\ewsv2_capsule\installers`.

3. Optional but recommended – **prove that the restore works offline**. Turn Wi-Fi off, then:

```powershell
powershell -ExecutionPolicy Bypass -File E:\ewsv2_capsule\restore.ps1 -Target "$HOME\Desktop\ewsv2_restore_test" -SkipV1
```

It copies the project, builds `venv2` only from the pendrive, and ends with the system check. Then delete
the test folder and turn Wi-Fi on again.

### 4d. Commit

```powershell
git add -A ewsv2
git commit -m "ewsv2 S10: doctor, failure tests, lock files, offline capsule, paper tables and figures"
git push
```

## 5. Check it worked

- [ ] tests: about 160 passed, 4 skipped
- [ ] `doctor`: no `[FAIL]`; the `[ -- ]` lines are only parts that are really absent
- [ ] `outputs_v2\paper`: `tables.md`, `tables.tex`, 7 figures (PNG + PDF)
- [ ] `capsule --check`: "Capsule intact"
- [ ] (optional) offline restore test ends with the system check

## 6. Viva notes

**How do you know it still works in five months?** Run `python -m ewsv2 doctor`: it checks every layer and
prints the fix for anything missing. The failure tests break each part on purpose (missing or damaged data,
damaged results, no AI model, a lying AI model, a missing board) and prove the system degrades instead of
crashing.

**What if pip or Python changes?** The exact versions that worked are frozen in the lock files, and the
pendrive holds every package as a Windows wheel plus the Python installer. `restore.ps1` rebuilds both
environments without internet. `capsule --check` proves the pendrive copy is intact (SHA-256 per file).

**Where do the paper's numbers come from?** `python -m ewsv2 paper` writes every table and figure from
the result files on the laptop, and `index.md` names the source file of each, so every number in the
paper can be traced to a file and re-generated.

## 7. Next

That completes the planned segments S0–S10. Next steps, when you are ready:

- **Paper 2:** draft the paper from `outputs_v2/paper` (tables, figures, architecture diagram).
- **Dashboard:** any changes after you have gone through it fully.
