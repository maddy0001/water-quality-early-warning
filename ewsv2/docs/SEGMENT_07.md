# Segment 07 – Incident reports by a local LLM, fact-checked

**Status:** code done and verified; the LLM numbers come from **your laptop** (see section 5) · **Builds on:** S4 (triage), S5 (pipeline alarms), S6 (deviation explainer) · **Feeds:** S9 (dashboard)
**Proves:** contribution **C8** (an LLM in the control room whose every fact is checked)

## 1. Goal

When an alarm fires, the operator needs a short note: what changed, what it probably is, what to
check first. A language model writes such notes well – and also invents numbers, times and causes.
In a drinking-water control room an invented number is worse than no note.

This segment adds an LLM **as the operator's assistant, never as the alarm**:

- it only reads a fixed *alarm record* (decision, level, times, sensors and their values) and a few
  retrieved guidance snippets (WHO and project notes);
- every sensor, number, time and citation in its report is checked automatically against that
  record;
- the operator sees the LLM report only when it passes the check; otherwise a template report built
  directly from the record (the *guarded* output). If the LLM is off, slow or wrong, nothing breaks.

Everything is free and offline: the model runs on your laptop with **Ollama**.

## 2. What happens

```
 S5 pipeline alarms + S4 triage + S6 deviation explainer
      │
      ▼
 alarm record (JSON)          decision · level · start · end · duration · top sensors (before → now,
      │                       times their normal 15-min move) · sensor alerts        [truth kept apart]
      ├──► retrieve guidance  by sensor role and alarm kind: W1–W4 (WHO), N1–N5, P1–P2 (project notes)
      ▼
 reporters:  template (fixed sentences)  ·  ollama:<model> (local LLM, temperature 0, seed 42)
      │
      ▼
 fact check:  decision line · times/dates · numbers · sensors · citations
      │
      ▼
 guarded output: LLM report if it has no error, else the template report
```

1. **Alarm records** (`llm/records.py`). One per test-period alarm episode: water alarms after the
   S4 triage, and sensor alerts that are not part of a water alarm. Sensors = top three of the S6
   deviation explainer at the first alarm minute (S6: as accurate as the costly explainers). Level,
   paper 1's names applied to sensors: one sensor beyond its normal 15-minute range → LOW, two →
   MEDIUM, three or more → CRITICAL; sensor alerts → MAINTENANCE. What the episode really was (real
   event, injected contaminant, fault, false alarm) is stored for evaluation and never shown to a
   reporter.
2. **Guidance** (`llm/guidance.yaml`), a small hand-made index rather than a vector database, so every
   snippet is visible and has a source:

   | Id | Source | Says |
   |---|---|---|
   | W1 | WHO chlorine fact sheet (GDWQ 2022) | guideline value 5 mg/l free chlorine; residual of a few tenths of a mg/l; 0.2–1 mg/l in most disinfected water |
   | W2 | WHO chlorine dioxide / chlorite / chlorate fact sheet | no guideline value for chlorine dioxide itself; chlorite and chlorate 0.7 mg/l each (provisional); taste/odour threshold 0.2–0.4 mg/l |
   | W3 | WHO GDWQ (2017), as summarised by USAID (2018) | turbidity below 5 NTU, if possible below 1 NTU |
   | W4 | same | optimum pH 6.5 to 8 for most systems |
   | N1–N5 | project notes | no WHO guideline value for conductivity, redox, SAC, temperature; flow is operational |
   | P1–P2 | project notes | first checks for a water alarm / a sensor alert |

   GECCO 2018's "Cl" sensors measure chlorine **dioxide**, so W2 is the relevant WHO text for them.
3. **Reporters** (`llm/reporters.py`). `template` and `ollama` (Ollama's HTTP API through Python's
   standard library: no new package). The prompt fixes the first line (`<DECISION> - level <LEVEL>`),
   three sections (What happened / Likely cause / Check first), citations in brackets, at most 120
   words, and "use only facts from the record".
4. **Fact check** (`llm/faithfulness.py`): five error kinds, found mechanically – wrong decision line,
   a time or date that is not the alarm's, a number that is neither in the record (rounding allowed)
   nor in the cited guidance (small counts 1–10 allowed), a station sensor that is not part of the
   alarm, a citation id that was not provided. Tested on reports with planted errors.
5. **Sample** (`llm.max_reports`, default 40): a balanced, fixed set of real events, contaminants,
   faults, false alarms and sensor alerts – 33 alarms on 2018, 27 on 2019.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `ewsv2/llm/records.py` | Alarm records, levels, balanced sample |
| `ewsv2/llm/guidance.yaml` | Guidance snippets with sources |
| `ewsv2/llm/reporters.py` | `template` and `ollama` reporters; retrieval |
| `ewsv2/llm/faithfulness.py` | The fact checker |
| `ewsv2/llm/run.py` | The S7 experiment |
| `ewsv2/configs/s7_reports.yaml` | Model, host, sample size |
| `ewsv2/cli.py` | New command `report` |
| `ewsv2/core/registry.py` | Loads `llm` |
| `ewsv2/tests/test_llm.py` | 9 new tests (117 in total), incl. planted errors and a stand-in Ollama server |

No new Python packages. New program: Ollama (free).

## 4. Run it (PowerShell)

### 4a. One-time: install Ollama and a model (about 2 GB)

1. Download the Windows installer from https://ollama.com/download and install it. Ollama then runs
   in the background (an icon in the system tray).
2. In a **new** PowerShell window:

```powershell
ollama pull qwen2.5:3b          # about 1.9 GB, one time
ollama list                     # must show qwen2.5:3b
ollama run qwen2.5:3b "Say hello in five words."   # quick test; type /bye if it waits for input
```

Optional second model for a comparison: `ollama pull llama3.2:3b` (about 2 GB).

### 4b. The segment

```powershell
cd "C:\Users\THE GOD\Documents\Github\water-quality-early-warning"
venv2\Scripts\Activate.ps1

# 1. Clean install of the S7 code
Remove-Item -Recurse -Force ewsv2
Expand-Archive -Path "$HOME\Downloads\ewsv2_S7.zip" -DestinationPath . -Force

# 2. Check: must print  ewsv2 2.0.0.dev0  (built up to segment S7)
python -m ewsv2 info
python -m pytest ewsv2\tests -q                 # 117 passed

# 3. Reports (needs your S5 output; Ollama must be running)
python -m ewsv2 report --config s7_reports
python -m ewsv2 report --config s7_reports --set data.source=gecco2019

# optional: compare two models (after  ollama pull llama3.2:3b)
python -m ewsv2 report --config s7_reports --set "llm.models=[qwen2.5:3b,llama3.2:3b]"

# 4. Save
git add ewsv2
git commit -m "ewsv2 S7: local LLM incident reports with automatic fact check"
git push
```

Time: each LLM report takes a few seconds with a GPU and roughly 10–30 s on a laptop CPU, so about
5–15 minutes per dataset for one model. Without Ollama the command still runs and writes the
template reports, and prints why the LLM was skipped.

Reports: `outputs_v2\s7_reports\gecco2018\report.md` (summary table and one example per reporter),
`reports.json` (every report with its record and errors).

## 5. Expected output

**The cloud machine cannot download models** (its network only reaches package registries), so the
LLM rows exist only on your laptop. What is verified in the cloud:

```
Incident reports on gecco2018: 166 alarm records, 33 reported
  reporter               faithful  err/rep decision  time number sensor  cite top sensor
  template                   1.00     0.00     0.00  0.00   0.00   0.00  0.00       1.00
Incident reports on gecco2019: 61 alarm records, 27 reported
  template                   1.00     0.00     0.00  0.00   0.00   0.00  0.00       1.00
```

The template is faithful by construction – the check that the checker does not raise false errors on
correct text. On your laptop a line `ollama:qwen2.5:3b` follows. What to read in it:

- **faithful** – share of LLM reports with no error at all. This is the headline number of C8; the
  guarded output shows exactly this share of LLM reports and the template for the rest.
- the error columns – which kind of mistake the model makes (numbers it computed itself, times it
  rewrote, sensors it guessed, citations it made up);
- **top sensor** – does the report name the sensor that moved most;
- **sec** – median seconds per report on your machine.

With temperature 0 and a fixed seed the same model on the same laptop gives the same reports each time.
Different hardware or Ollama versions can change some words, so LLM numbers are reported as measured
on your laptop, together with the model name and Ollama version (`ollama --version`).

## 6. Check it worked

- `python -m ewsv2 info` lists `reporter  ollama, template`.
- The template line must read 1.00 faithful on both datasets.
- If the LLM line is missing, the console says why (`Ollama is not running ...` or
  `model 'qwen2.5:3b' not pulled ...`).

| Problem | Fix |
|---|---|
| `Ollama is not running` | Start the Ollama app (Start menu), or run `ollama serve` in a separate PowerShell window |
| `model 'qwen2.5:3b' not pulled` | `ollama pull qwen2.5:3b` |
| Very slow | Close other programs; or `--set llm.max_reports=15`; or try the smaller `qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b`, then `--set "llm.models=[qwen2.5:1.5b]"`) |
| `FileNotFoundError ... pipeline_minutes.csv.gz` | Run S5 first for that dataset |

## 7. Viva notes

**Why add an LLM at all?** Operators read alarms in a hurry. A short plain-language note with the
changed sensors, their values and the first check saves time – if it is correct. The novelty is not
the LLM but the measurement: we report how often its notes are factually exact, and we never show
one that is not.

**Can the LLM cause or hide an alarm?** No. The decision, level and times are made by the S5/S4
pipeline and stored in the record; the LLM only writes text about it, and its first line must repeat
the decision exactly or the report is rejected.

**What is "grounding" here?** The LLM may quote only the guidance snippets retrieved for this alarm,
each with an id and a source. The retrieval is a simple lookup by sensor type, because the guidance
fits on one page; a vector database would add nothing but opacity.

**How do you check facts automatically?** Every number, time, sensor name and citation in the text is
extracted with patterns and compared with the record and the snippets. The checker is tested on
reports with planted errors of each kind, and on correct paraphrases.

**Why not a paid API like GPT?** ₹0 budget, and water-utility data should not leave the utility. A 3B
model runs on a laptop CPU.

**Are the WHO values right?** W1 and W2 are quoted from WHO fact sheets. W3 and W4 are WHO statements
as summarised in a USAID technical annex; check them against the WHO guidelines document before the
paper is submitted. The project notes are labelled as not WHO.

## 8. Next segment

**S8 – ESP32 edge.** Puts the alarm logic that needs no heavy model (deviation explainer, fault checks,
a small int8 detector) on your friend's ESP32, and replays the GECCO stream to it over USB
(hardware-in-the-loop): memory, time per minute, and agreement with the laptop.
