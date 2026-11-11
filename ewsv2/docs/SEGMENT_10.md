# Segment 10 – Robustness and hand-over

**Status:** done – checked in the cloud (166 tests, 19 of them break parts on purpose) and on your laptop (10.0); update 10.1 below · **Builds on:** S0–S9 · **Feeds:** paper 2, the viva, and every future run of the project
**Delivers:** a system check, failure tests, exact lock files, an offline pendrive capsule, and the paper's
tables and figures

## 0. Update 10.1 – both versions in the capsule, pendrive on H:, one README per folder

Your run of 10.0: tests 160/4, `doctor` 19/19 OK, capsule 7.56 GB intact, offline restore of version 2 worked.
Your questions, answered:

| You said | What it was | Now |
|---|---|---|
| "Version 1 is not working" | The test restore used `-SkipV1` (my instruction), so the test folder had no `venv`. Your real project's version 1 was fine (doctor: OK, dashboard started it on 8501) | The restore test below rebuilds **both** environments. `doctor` now really tests version 1: it starts version 1's Python, imports TensorFlow, scikit-learn, XGBoost, Streamlit, SHAP …, loads a trained model, and checks the 7 model files and both datasets |
| "All the files in the capsule" | Everything was in it except `venv` and `venv2` (and two junk files) | Same, on purpose: an environment stores absolute paths and breaks when moved, so the capsule holds every package and **rebuilds** both environments, which is what your restore test proved. Exact installs now use `--no-deps`, so the rebuilt environment is the locked one, package for package |
| "Remove the unwanted files" | `ewsv2_S2\` (old copy from Segment 2), `structure.txt`, `fix_pipeline.py` (one-off, already applied), `__pycache__` folders | Removed with the commands in 4a; never copied into the capsule; re-running `capsule` deletes from the capsule whatever you deleted from the project |
| "The capsule went to E: (hard disk), the pendrive is H:" | | 4c refreshes E:, copies it to H: and checks it there, then deletes E: |
| "Two READMEs will conflict" | Different folders = different files; Git never mixes them. GitHub shows the root one | Root `README.md` rewritten for the **whole repository**: both versions side by side (folders, environments, dashboards, setup, backup), then version 1 in full. `ewsv2/README.md` is the detailed version 2 guide and points back to it |

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

### 4a. Install, clean up, test

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S10_1.zip" -DestinationPath . -Force

# unwanted files (not in Git; not needed by either version)
Remove-Item -Recurse -Force ewsv2_S2, structure.txt, fix_pipeline.py
Get-ChildItem -Recurse -Force -Directory -Include __pycache__, .pytest_cache |
    Where-Object { $_.FullName -notmatch '\\venv2?\\' } | Remove-Item -Recurse -Force

python -m pytest ewsv2\tests -q              # about 161 passed, 5 skipped (cloud: 166 passed)
python -m ewsv2 doctor                       # now also tests version 1 (takes ~20 s longer)
```

The root `README.md` was already written into your project folder by me (you will see it in `git status`).

### 4b. Paper material (unchanged)

```powershell
python -m ewsv2 paper
```

### 4c. Move the capsule to the pendrive (H:)

The capsule on E: is complete; refresh it with the new code (fast: unchanged files are skipped, the removed
junk is deleted from it), copy it to the pendrive, check it there, then delete the E: copy:

```powershell
python -m ewsv2 capsule --dest E:\ewsv2_capsule --with-ollama --with-arduino --with-installers
robocopy E:\ewsv2_capsule H:\ewsv2_capsule /MIR /NFL /NDL /NP
python -m ewsv2 capsule --check H:\ewsv2_capsule
Remove-Item -Recurse -Force E:\ewsv2_capsule
```

Copying 7.5 GB to a USB pendrive takes about 10–30 minutes. If the pendrive is FAT32, files of 4 GB or more
cannot be stored; none of yours is that big (the largest is the AI model, under 2 GB), and `capsule` warns
if one ever is.

### 4d. Prove both versions restore offline

Turn Wi-Fi off. Delete the old test folder, then restore **both versions** from the pendrive:

```powershell
cd $HOME
Remove-Item -Recurse -Force "$HOME\Desktop\ewsv2_restore_test"
powershell -ExecutionPolicy Bypass -File H:\ewsv2_capsule\restore.ps1 -Target "$HOME\Desktop\ewsv2_restore_test"
```

Expected at the end: `doctor` with **Version 1 environment (venv): all packages import (TensorFlow 2.13.0 …)**,
**Version 1 trained models: all 7 model files present**, and every version 2 line OK. Then start both
dashboards from the restored copy:

```powershell
cd "$HOME\Desktop\ewsv2_restore_test"
venv2\Scripts\python -m ewsv2 dashboard      # must print: version 1 dashboard on http://localhost:8501
```

Open http://localhost:8501 (version 1) and http://localhost:8502 (version 2), click through a page of each,
press Ctrl+C, turn Wi-Fi on, and delete the test folder:

```powershell
cd $HOME
Remove-Item -Recurse -Force "$HOME\Desktop\ewsv2_restore_test"
```

### 4e. Commit

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
git add -A ewsv2 README.md
git commit -m "ewsv2 S10.1: version 1 checks, capsule refresh/mirror, both versions restored, repository README"
git push
```

## 5. Check it worked

- [ ] tests: about 161 passed, 5 skipped
- [ ] `doctor` on the project: no `[FAIL]`; three version 1 lines OK
- [ ] `capsule --check H:\ewsv2_capsule`: "Capsule intact"
- [ ] offline restore of both versions: doctor OK for version 1 and 2; both dashboards open
- [ ] GitHub front page shows the new README with both versions

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
