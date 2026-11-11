# Segment 09 – The operator dashboard and live demo (version 1 alongside)

**Status:** done and checked in the cloud (every page runs headless in the tests, the board path was
checked against the real firmware on a simulated Uno); you run it on your laptop · **Builds on:**
S4–S8 · **Feeds:** S10 (hand-over, final diagram, paper tables)
**Shows:** all contributions in one place, live; experiment **E12** (the Uno answers minute by minute
next to the laptop in the dashboard)

## 0. Update 9.3 – professional operator dashboard (complete redesign)

Your feedback on 9.2: still confusing, emojis, unclear buttons and toggles, unclear names ("gecco 18/19",
"who answers", "answer key"), a missing report download, too few charts, the stored experiment results not
shown, and a request for a professional, vibrant look with 3D and animation. Everything was redesigned:

| Topic | Now |
|---|---|
| Look | Deep-indigo sidebar with logo, animated gradient page headers with moving waves, gradient KPI tiles that tilt in 3D on hover, Inter font, no emojis anywhere |
| Pages (9) | **Monitor:** Overview, Operations Console, Sensor Lab · **Analyze:** Performance Analytics, Data Explorer, Incident Reports · **Reference:** System Guide, Version 1 Baseline, System Status |
| Every control | Has a one-line caption under it saying exactly what it does |
| Data names | "Utility data A – 2018 test period (9 sensors)", "Utility data B – 2019 test period (6 sensors)", "Synthetic station (generated, no data files needed)" |
| "Who answers the alarms" | Removed from the main screen. The replay always asks you; *Automatic review* is an advanced option with an explanation |
| "Answer key" | Now **Recorded outcome**: what really happened in the data, never seen by the detection system |
| Report download | Fixed and extended: **.txt** and printable **.html** (Print → Save as PDF) in the Operations Console, the Sensor Lab and the Incident Reports page; alarm log and all reports as **.csv** |
| Charts | Live gauge (ECharts) for the anomaly score, area timeline, polar sensor chart, liquid-level ball and gauges in the Sensor Lab, Sankey flow diagrams, heatmaps, bubble charts, radar, sunburst, rotating 3D sensor space |
| Experiment results | **Performance Analytics** reads every result file in `outputs_v2` (7 tabs: evaluation method, detection methods, water event or sensor fault, false-alarm control, explanations, incident reports, edge device); each tab opens with a one-paragraph finding and every table can be downloaded |
| Data | **Data Explorer**: time series with zoom strip, 3D sensor space with a Rotate button, correlation heatmap, which sensors react to events, data quality |
| Removed | Build history (segment list), RFID mentions, the help-box toggle, emojis |

New packages (pinned, with their JavaScript bundled inside, so they work offline):
`streamlit-echarts==0.4.0`, `pyecharts==2.0.6`, `simplejson==3.19.3`. If they are ever missing, every gauge
automatically falls back to a Plotly chart – nothing breaks.

## 0a. Update 9.2 – first friendly version (superseded by 9.3)

Replaced by 9.3 above.

## 0b. Update 9.1 – calmer and clearer (after your first run)

You said the dashboard was confusing and "glitched" every time the data changed. Both are fixed:

| You saw | Why | Now |
|---|---|---|
| The whole page faded and flashed every second while playing | Streamlit greys out ("stale") every element while it refreshes, and the charts were rebuilt each time | Nothing fades; charts keep their place and update in place; the question box has a fixed size, so nothing jumps (measured: 0 layout shifts, opacity never below 100 % during 8 s of play) |
| "I don't know what is what" | Words like *log-odds*, *θ*, *triage* | Plain words everywhere: **alarm score**, **alarm line**, **answer key**; a numbered *How to use this page* box on every page, and one line under every chart. Sidebar switch **Show explanations** hides them for a clean demo screen |
| "Skip to next alarm" stopped before the question | It stopped at the start of the alarm, the question comes at its end | **⏭ Next alarm** runs until the alarm has ended and the question is waiting |
| You confirmed an alarm on normal water (your CSV, 08:36 pH) | The truth was a separate toggle, far from the question | The question box shows the **answer key** next to the buttons: *correct answer: FALSE ALARM – nothing happened, normal water*. A **Your answers** card scores you against it |
| Live lab chart looked dramatic on tiny noise | The y-axis zoomed into a 4-count wiggle | Fixed minimum scale, plus a green **normal band**: leaving it = water alarm. A demo checklist ticks off calibrated / water alarm / sensor fault / false alarm |

Also: the overview has a **Start here** row with three buttons; the incident report sits below the live
area and changes only when you click (pause to write one). The answer is now stored on the exact alarm it
was asked about.

## 0b. Segment 8 – your result

Your run on the real Uno: **10,080 minutes, every alarm, sensor alert, fault mask, top sensor and log-odds
bit-identical to the laptop**, 25.2 ms per minute (max 26.1 ms), 423 bytes of RAM never touched,
19,316 bytes of program space. That is the C9 result for the paper. The stop in the earlier run (at minute
7,895) happened because the Serial Monitor took the port. The run ended cleanly, as designed.

## 1. Goal

One dashboard for the demo and the viva that:

- shows the system working **live**, on the real GECCO station and on your own sensors;
- lets **you** be the operator: answer alarms and watch the threshold adapt;
- puts the **evidence** of S1–S8 in front of the examiners;
- brings **version 1** in, so the examiners see both versions side by side, without touching version 1's
  code or environment;
- **never breaks**: no board, no sensor, no LLM, no data files, no outputs – every page still works.

## 2. What is in it

```
 python -m ewsv2 dashboard ──► http://localhost:8502   (version 2, dark "control-room" theme)
          │                    └─► also starts version 1 on http://localhost:8501 with version 1's own venv (if found)
          ▼
 Demo       Overview        what it does, headline results, what is connected right now
            Control room    GECCO station replayed minute by minute (+ injected contaminants and faults);
                            you answer the alarms; optional: the Uno computes every minute too
            Live sensor lab the probe in a glass of water (salt = event, pulled wire = fault, touch = "false alarm");
                            simulated probe with the same buttons when the board is missing
 Evidence   Results         S1–S8 in charts and tables (from your output files, or the bundled snapshot)
            Incident reports S7 reports next to the facts they were checked against
            Version 1       paper 1's results, its own dashboard embedded, and how its parts did on real data
 Station    System check    every part: ready or fallback, and how to fix it
```

**Control room.**
- **Controls:**
  - choose the station (GECCO 2018, GECCO 2019, simulated);
  - choose who answers alarms: you, or the labels;
  - speed 1–240 minutes per second;
  - Play, Step, *Skip to next alarm*, Restart.
- **On screen:**
  - a status banner: NORMAL, WATER QUALITY ALARM (pulsing red), or SENSOR FAULT ALERT;
  - the detector and the adaptive threshold θ;
  - all nine sensors, with flagged readings marked;
  - the *why* chart: each sensor's move against its normal range;
  - the alarm log, with a CSV download;
  - the incident report: template, or the local LLM with every fact checked.
- **When an alarm ends** the dashboard asks *Real event / False alarm*. A "false alarm" raises θ.
- **"Show what really happened"** reveals what each episode truly was (real event, injected contaminant,
  fault). The system never sees it.
- **"Also run on the Uno"**: the board computes every minute too. The dashboard shows *N/N minutes
  identical to the laptop*. If the cable is pulled, the laptop simply continues.

**Live sensor lab.**
- **Shows:** the probe's ion index (1023 − reading), its move in units of normal noise, the alarm limit,
  and DHT11 temperature and humidity.
- **How it works:**
  - it calibrates on the first ~40 s in water;
  - a fast step is a water alarm, slow drift is normal;
  - a pulled fork wire or a dry probe is a sensor alert;
  - a touch on the pad (or the button) is the operator's "false alarm" and raises the limit.
- **Without the board** the same page runs a simulated probe, with *Add salt / Fresh water / Pull probe
  out / Put it back* buttons.

**How version 1 and 2 come together.** The two use different Python packages: version 1 pins
TensorFlow 2.13 and NumPy 1.24, version 2 has its own. Putting them in one program would risk breaking
version 1. So each runs in its own environment, unchanged, and the v2 dashboard **embeds** the v1
dashboard on its *Version 1* page (`python -m ewsv2 dashboard` starts it for you). Next to it, a table
shows what happened to each of paper 1's parts on real data in v2: the 2-of-4 rule, the ensemble, the
BiLSTM, the 0.40/0.30/0.30 weights, TreeSHAP and the 4 levels.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/app/main.py` | Dashboard entry, navigation |
| `ewsv2/app/ui.py` | Theme, CSS, chart template, cards, banners |
| `ewsv2/app/services.py` | Status checks, replay engine, alarm records, reports, result files |
| `ewsv2/app/views/*.py` | The seven pages |
| `ewsv2/app/snapshot.json` | Measured headline results (used when output files are missing) |
| `ewsv2/edge/stream.py` | The edge logic one minute at a time in Python, bit-identical to the Uno |
| `ewsv2/live/station.py` | Live demo station + simulated probe |
| `ewsv2/live/board_service.py` | Board connection for the dashboard: sensor polling, live Uno check, reconnects |
| `ewsv2/edge/firmware/ewsv2_edge/edge_sensors.h` | DHT11: waits the 1 s it needs between readings |
| `ewsv2/edge/hil.py` | Timing from the answers themselves when a run is cut short |
| `ewsv2/configs/s9_dashboard.yaml` | Port, version-1 settings, LLM settings |
| `ewsv2/cli.py` | Command `dashboard` |
| `ewsv2/requirements.txt` | `streamlit`, `plotly`, `pyarrow` (kept below the versions that need NumPy 2) |
| `ewsv2/tests/test_dashboard.py` | 14 tests (143 in total): engine bit-identical, live station, replay, every page without hardware |

## 4. Run it (PowerShell)

### 4a. Install and check

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S9_3.zip" -DestinationPath . -Force
pip install -r ewsv2\requirements.txt          # 9.3 adds streamlit-echarts, pyecharts, simplejson

python -m ewsv2 info                            # built up to segment S9
python -m pytest ewsv2\tests -q                 # 142 passed, 4 skipped (cloud: 146 passed)
```

### 4b. Upload the updated sketch (one small change: DHT11 timing)

Arduino IDE: open `ewsv2\edge\firmware\ewsv2_edge\ewsv2_edge.ino` → **Upload** → **close the Serial
Monitor** (and preferably Arduino IDE). Wiring unchanged.

### 4c. Start the dashboard

```powershell
python -m ewsv2 dashboard
```

The browser opens at **http://localhost:8502**. Version 1 starts in the background on 8501 if `venv`
and `dashboard\app.py` are found, and appears on the *Version 1* page. Stop everything with **Ctrl+C**
in the PowerShell window.

```powershell
python -m ewsv2 dashboard --set dashboard.with_v1=false     # only version 2
```

### 4d. A 5-minute demo script

1. **Overview** (30 s): read the header; point at the five result tiles and the five decision stages.
2. **Operations Console** (2 min), data source *Utility data A – 2018*:
   - **Skip to next alarm**; read the status bar, the gauge and the polar chart (which sensor changed);
   - in **Alarm review** read the *Recorded outcome*, select **Mark as false alarm**, and show the
     *Alarm threshold* tile and the dashed line stepping up;
   - **Skip to next alarm** again until an amber **Sensor-fault alert** appears;
   - optional: *Advanced options → Verify on Arduino*;
   - scroll to **Incident report and downloads** → **Download printable report (.html)**.
3. **Sensor Lab** (1.5 min), input *Arduino sensors*, probe in tap water: wait for *Calibrate* to complete,
   add salt (red alarm), lift the probe (amber sensor fault), touch the pad (false alarm). All four steps
   turn green.
4. **Performance Analytics** (1 min): *Evaluation method* (0.98 vs 0.64), *Water event or sensor fault*
   (Sankey diagram), *Edge device* (gauges).
5. **Data Explorer** (30 s): press **Rotate** on the 3D sensor space.

Take a few screenshots for the report and send them to me if anything looks wrong.

## 5. Expected behaviour

| Situation | What you see |
|---|---|
| Uno plugged in, sketch uploaded, Serial Monitor closed | System Status: Arduino Uno *ready*, probe/touch/DHT11 *ready*; Sensor Lab uses "Arduino sensors" |
| No Uno | Sensor Lab uses the simulated probe; Operations Console's Uno toggle is off; nothing else changes |
| Probe wire pulled in the Sensor Lab | Yellow Sensor problem within one reading (~2 s) |
| Salt in the glass | Red Water alarm within 2–4 s; it clears once the new level is "normal" |
| Ollama not running | Reports use the template; System Status says how to start Ollama |
| Version 1 not running | Version 1 page shows its published results and how to start it |
| GECCO files missing | Station list offers "simulated" only; every page still works |
| Playing in the control room | Numbers and charts change in place once a second; nothing fades or jumps |
| An alarm ends (You mode) | The replay stops; the question box shows two buttons and the answer key |

In the cloud, every page ran headless in the tests with no board, no LLM and no browser.
The control room's Uno check ran against the real firmware on a simulated Uno: 300 of 300 minutes identical.

## 6. Check it worked

- `python -m ewsv2 info` prints `built up to segment S9`.
- All seven pages open without a red error box.
- With the Uno: *System Status* shows the board and the three sensors as ready.

| Problem | Fix |
|---|---|
| `Streamlit is missing` | `pip install -r ewsv2\requirements.txt` |
| `pyarrow requires NumPy 2.0` | `pip install "pyarrow>=19,<23"` (the requirements pin this) |
| Sensor Lab says "No probe on the board" | Close the Serial Monitor; *System Status* → *Look for the board again* |
| Port busy / Access denied | Only one program may use COM7: close Arduino IDE's Serial Monitor; do not run `python -m ewsv2 edge` while the dashboard uses the board |
| Version 1 page empty | Start version 1 by hand (the page shows the commands); it needs its models (`python main.py --train`) |
| Port 8502 in use | `python -m ewsv2 dashboard --set dashboard.port=8503` |

## 7. Viva notes

**Why not merge version 1 into version 2's code?** Version 1 is published and pinned to TensorFlow 2.13
and NumPy 1.24. Version 2 needs other versions. Merging would risk changing the published results. Running
both side by side, each in its own environment, keeps v1 exactly reproducible and still shows both in one
place.

**Is the replay "live"?** Each minute is computed when it is shown, by the same algorithm that runs on the
Uno (bit-identical), and your answers change the threshold for the following minutes. Only the data
come from a recording, the real GECCO station.

**What does the live lab prove?** Not chemistry: the probe is not a calibrated conductivity sensor. It shows
the decision logic on a physical sensor. A real change is an alarm, a broken sensor is a maintenance ticket,
an operator's "false alarm" makes the system less jumpy. Every case can be repeated in front of the examiners.

**What if something fails during the demo?** That is the design: unplug the Uno and the laptop continues;
stop Ollama and the template report appears; the System Status page says what changed.

## 8. Next segment

**S10 – Robustness and hand-over.**
- `python -m ewsv2 doctor` checks every part from the command line.
- Tests that break each part on purpose.
- A pinned lock file and an offline "time capsule" on your pendrive, so everything reinstalls in 5 months
  without internet.
- The final architecture diagram and the paper's tables and figures.
