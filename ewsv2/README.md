# ewsv2 – Water Quality Early Warning, version 2

*Is it the water or the sensor?* Calibrated, leakage-free triage of drinking-water
quality alarms on public utility data (GECCO 2018/2019), with explanations, a local
LLM incident reporter, and an ESP32 edge version.

Version 1 (paper 1, ICESC 2026) is the code at the repository root and is tagged `v1.0`.
Nothing here imports or changes it.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv venv2
venv2\Scripts\Activate.ps1
pip install -r ewsv2\requirements.txt
python -m ewsv2 fetch gecco2018
python -m ewsv2 inspect --config s0_inspect
python -m pytest ewsv2\tests -q
```

## Layout

| Folder | Layer | From segment |
|---|---|---|
| `core/` | registry, config, types, interfaces | S0 |
| `data/` | data sources, fetch, inspection | S0 |
| `evaluate/` | splits, event metrics, thresholds, bootstrap, leakage audit | S1 |
| `features/` | causal feature builders | S1 |
| `signals/` | baselines (S1); detectors, classifier, forecast, fault checks (S2–S4) | S1–S4 |
| `inject/`, `triage/` | GECCO-Triage benchmark, water-vs-sensor decision | S4 |
| `fusion/`, `calibrate/` | learned fusion, conformal alarm budget | S5 |
| `explain/` | attribution | S6 |
| `llm/` | local LLM reporter and assistant | S7 |
| `edge/` | ESP32 export and firmware | S8 |
| `app/` | v2 dashboard pages | S9 |
| `configs/` | experiment configs | all |
| `docs/` | `SEGMENT_XX.md`, one per segment | all |

## Adding a plug-in

1. Create a file in the right folder following the interface in `core/interfaces.py`.
2. Put `@register("<kind>", "<name>")` above the class.
3. Add the package to `_PLUGIN_PACKAGES` in `core/registry.py` if it is a new folder.
4. Use `<name>` in a config, and add a test in `tests/`.

## Commands

```
python -m ewsv2 info
python -m ewsv2 fetch {gecco2018|gecco2019|all}
python -m ewsv2 inspect --config s0_inspect [--set data.source=gecco2019]
python -m ewsv2 evaluate --config s1_audit          # leakage audit (GECCO 2018)
python -m ewsv2 evaluate --config s1_baselines [--set data.source=gecco2019]
python -m ewsv2 run --config full_pipeline
```

## Data

GECCO Industrial Challenge 2018 and 2019, Thüringer Fernwasserversorgung and TH Köln,
CC BY 4.0 (Zenodo 10.5281/zenodo.3884398 and record 4304080). Downloaded on demand into
`data/raw/gecco/` (git-ignored).
