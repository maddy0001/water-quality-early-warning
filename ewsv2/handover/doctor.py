"""System check: ``python -m ewsv2 doctor`` (Segment 10).

Checks every layer of the system and says, for each part, whether it works, what the system uses instead
if it does not, and how to fix it. Nothing here changes anything; it only looks.

Status values
  ok        – the part works
  degraded  – the part is missing or differs; the system uses its replacement and keeps working
  fail      – something required is missing; some commands will not run until it is fixed
  skipped   – the check was not asked for (e.g. ``--no-board``)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import shutil
import socket
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

from ewsv2 import SEGMENT, __version__
from ewsv2.core.paths import DATA_DIR, OUTPUT_DIR, REPO_ROOT

LOCK_DIR = Path(__file__).with_name("locks")
REQUIRED = ["numpy", "pandas", "pyyaml", "scipy", "scikit-learn", "matplotlib", "streamlit", "plotly", "pyarrow"]
OPTIONAL = {"tensorflow-cpu": "training new edge models and the version-1 forecasters",
            "xgboost-cpu": "the version-1 ensemble re-tests", "pyserial": "talking to the Arduino",
            "streamlit-echarts": "dashboard gauges (Plotly gauges are used instead)",
            "pyreadr": "reading GECCO 2019 .rds files", "pytest": "running the tests"}
ANALYSES = [("Data inspection", "s0_inspection"), ("Evaluation audit", "s1_audit"),
            ("Detection methods", "s1_baselines"), ("Version 1 components on real data", "s2_v1_port"),
            ("Forecasting", "s3_forecast"), ("Water event or sensor fault", "s4_triage"),
            ("False-alarm control", "s5_fusion"), ("Explanations", "s6_explain"), ("Incident reports", "s7_reports"),
            ("Edge device", "s8_edge")]


@dataclass
class Check:
    area: str
    name: str
    status: str
    detail: str
    fix: str = ""


def _canon(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def read_lock(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            n, v = line.split("==", 1)
            out[_canon(n)] = v.strip()
    return out


def installed() -> dict[str, str]:
    out = {}
    for d in metadata.distributions():
        try:
            out[_canon(d.metadata["Name"])] = d.version
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------------------------- checks

def check_python() -> list[Check]:
    v = sys.version_info
    ok = v[:2] == (3, 11)
    st = "ok" if ok else ("degraded" if v >= (3, 10) else "fail")
    return [Check("Software", "Python", st, f"{platform.python_version()} ({platform.architecture()[0]}, "
                  f"{platform.system()} {platform.release()})",
                  "" if ok else "Python 3.11 (64-bit) is the tested version; install it from the capsule's "
                                "installers folder or python.org")]


def check_packages() -> list[Check]:
    lock = read_lock(LOCK_DIR / "requirements-lock-v2.txt")
    have = installed()
    out = []
    missing_req = [p for p in REQUIRED if _canon(p) not in have]
    out.append(Check("Software", "Required packages", "fail" if missing_req else "ok",
                     "missing: " + ", ".join(missing_req) if missing_req else f"all {len(REQUIRED)} present",
                     "pip install -r ewsv2\\requirements.txt" if missing_req else ""))
    for p, why in OPTIONAL.items():
        if _canon(p) not in have:
            out.append(Check("Software", f"Optional package {p}", "degraded", f"not installed – needed for {why}",
                             f"pip install {p}=={lock.get(_canon(p), '')}".rstrip("=")))
    diff = [f"{n} {have[n]} (lock {v})" for n, v in sorted(lock.items()) if n in have and have[n] != v]
    gone = [n for n in lock if n not in have]
    if lock:
        if diff or gone:
            out.append(Check("Software", "Versions vs lock file", "degraded",
                             f"{len(diff)} differ, {len(gone)} not installed"
                             + (": " + "; ".join((diff + gone)[:6]) + (" …" if len(diff) + len(gone) > 6 else "")),
                             "pip install --no-deps -r ewsv2\\handover\\locks\\requirements-lock-v2.txt "
                             "(or offline from the capsule)"))
        else:
            out.append(Check("Software", "Versions vs lock file", "ok",
                             f"all {len(lock)} packages match the tested versions"))
    return out


def check_plugins() -> list[Check]:
    try:
        importlib.import_module("ewsv2.cli")
        from ewsv2.core.registry import available
        kinds = available()
        n = sum(len(v) for v in kinds.values())
        return [Check("Software", "ewsv2 plug-ins", "ok", f"{n} plug-ins registered in {len(kinds)} groups")]
    except Exception as exc:  # noqa: BLE001
        return [Check("Software", "ewsv2 plug-ins", "fail", f"import failed: {exc}",
                      "reinstall the packages; re-extract the ewsv2 folder")]


def check_data() -> list[Check]:
    out = []
    files = sorted(p.name for p in DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    for name, needles, cmd in (("Utility data A (GECCO 2018)", ("gecco2018",), "gecco2018"),
                               ("Utility data B (GECCO 2019)", ("gecco2019",), "gecco2019")):
        hit = [f for f in files if any(n in f.lower() for n in needles)]
        ok = len(hit) > 0 and all((DATA_DIR / f).stat().st_size > 1000 for f in hit)
        out.append(Check("Data", name, "ok" if ok else "degraded",
                         f"{len(hit)} file(s) in {DATA_DIR}" if ok else "files not found – the synthetic station is "
                                                                         "used instead",
                         "" if ok else f"python -m ewsv2 fetch {cmd}  (or copy data\\raw\\gecco from the capsule)"))
    from ewsv2.edge import export as E
    missing = [d for d in ("gecco2018", "gecco2019", "simulated") if not E.spec_path(d).exists()]
    out.append(Check("Data", "Edge models", "ok" if not missing else "degraded",
                     "shipped for 2018, 2019 and synthetic" if not missing else
                     f"missing for {', '.join(missing)} – trained automatically on first use (needs TensorFlow)",
                     "" if not missing else "re-extract the ewsv2 folder"))
    return out


def check_outputs() -> list[Check]:
    def has(d: Path) -> bool:
        return d.exists() and (any(d.rglob("*.csv")) or any(d.rglob("*.json")))
    have = [n for n, sub in ANALYSES if has(OUTPUT_DIR / sub)]
    missing = [n for n, _ in ANALYSES if n not in have]
    st = "ok" if not missing else "degraded"
    return [Check("Results", "Experiment results", st, f"{len(have)} of {len(ANALYSES)} analyses on this computer"
                  + (f"; missing: {', '.join(missing)}" if missing else ""),
                  "" if not missing else "the dashboard shows the bundled results; re-run the segment commands or "
                                         "copy outputs_v2 from the capsule")]


def check_board(enabled: bool) -> list[Check]:
    if not enabled:
        return [Check("Hardware", "Arduino Uno", "skipped", "not checked (--no-board)")]
    try:
        from ewsv2.edge import board as B
        r = B.detect("auto", readings=1)
    except Exception as exc:  # noqa: BLE001
        return [Check("Hardware", "Arduino Uno", "degraded", f"check failed: {exc}", "plug in the board and retry")]
    if not r["board"]:
        return [Check("Hardware", "Arduino Uno", "degraded", f"{r['message']} – the laptop runs the same logic",
                      "plug in the Uno, upload ewsv2_edge.ino, close the Arduino Serial Monitor")]
    out = [Check("Hardware", "Arduino Uno", "ok", f"answering on {r.get('port', '?')}")]
    names = {"probe": ("Conductivity probe", "AO→A0, VCC→A2, GND→GND", "the simulation"),
             "touch": ("Touch sensor", "SIG→D3, VCC→D4, GND→GND", "the on-screen button"),
             "dht": ("Temperature/humidity sensor", "DATA→A1, VCC→5V, GND→GND", "values shown as unavailable")}
    for k, (n, wires, alt) in names.items():
        ok = r["parts"].get(k, False)
        out.append(Check("Hardware", n, "ok" if ok else "degraded", "answering" if ok else f"no answer – {alt} is used",
                         "" if ok else f"check the wires: {wires}" + (" (touch the pad once during the check)"
                                                                       if k == "touch" else "")))
    return out


def check_llm() -> list[Check]:
    try:
        from ewsv2.core.config import load_config
        from ewsv2.core.registry import build
        cfg = load_config("s9_dashboard")
        model, host = cfg["llm"]["models"][0], cfg["llm"]["host"]
        ok, why = build("reporter", "ollama", model=model, host=host).available()
    except Exception as exc:  # noqa: BLE001
        ok, why, model = False, str(exc), "qwen2.5:3b"
    detail = f"{model} ready" if ok else ("Ollama is not running" if "refused" in str(why).lower() else str(why)[:120])
    return [Check("AI report writer", "Ollama + model", "ok" if ok else "degraded",
                  detail + ("" if ok else " – template reports are used (always correct)"),
                  "" if ok else f"start Ollama, then: ollama pull {model}")]


V1_PROBE = r"""
import json, importlib, joblib
mods = ["numpy", "pandas", "sklearn", "xgboost", "tensorflow", "streamlit", "plotly", "shap", "imblearn", "seaborn"]
out = {"versions": {}, "errors": {}}
for m in mods:
    try:
        out["versions"][m] = getattr(importlib.import_module(m), "__version__", "?")
    except Exception as exc:
        out["errors"][m] = f"{type(exc).__name__}: {exc}"[:160]
try:
    joblib.load("models/classification/random_forest.pkl")
    out["model_load"] = "ok"
except Exception as exc:
    out["model_load"] = f"{type(exc).__name__}: {exc}"[:160]
print("V1PROBE" + json.dumps(out))
"""
V1_MODELS = ["classification/random_forest.pkl", "classification/xgboost.pkl", "classification/svm.pkl",
             "classification/scaler.pkl", "forecasting/bilstm_attention.keras", "forecasting/scaler.pkl",
             "anomaly/isolation_forest.pkl"]
V1_DATA = ["data/raw/water_potability.csv", "data/raw/water_dataset.mat"]


def check_version1(probe: bool = True) -> list[Check]:
    """Version 1 lives in the same repository with its own environment (venv). It is checked, never changed."""
    out = []
    py = REPO_ROOT / "venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
    if not py.exists():
        out.append(Check("Version 1", "Version 1 environment (venv)", "degraded",
                         "not found – the Version 1 page shows the published results instead",
                         "restore it: powershell -ExecutionPolicy Bypass -File <capsule>\\restore.ps1 "
                         "(without -SkipV1), or: py -3.11 -m venv venv; venv\\Scripts\\pip install -r "
                         "ewsv2\\handover\\locks\\requirements-lock-v1.txt"))
    elif not probe:
        out.append(Check("Version 1", "Version 1 environment (venv)", "ok", "found (packages not tested: --quick)"))
    else:
        import subprocess
        try:
            r = subprocess.run([str(py), "-c", V1_PROBE], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("V1PROBE")), None)
            res = json.loads(line[7:]) if line else {"errors": {"python": (r.stderr or "no output")[-160:]}}
        except Exception as exc:  # noqa: BLE001
            res = {"errors": {"python": f"{type(exc).__name__}: {exc}"}}
        errs = res.get("errors", {})
        if errs:
            out.append(Check("Version 1", "Version 1 environment (venv)", "fail",
                             "cannot import " + "; ".join(f"{k} ({v})" for k, v in list(errs.items())[:3]),
                             "venv\\Scripts\\pip install --no-deps -r ewsv2\\handover\\locks\\requirements-lock-v1.txt "
                             "(or restore from the capsule)"))
        else:
            v = res.get("versions", {})
            lock = read_lock(LOCK_DIR / "requirements-lock-v1.txt")
            diff = [f"{n} {v[m]} (lock {lock[n]})" for m, n in (("numpy", "numpy"), ("tensorflow", "tensorflow"),
                                                                 ("streamlit", "streamlit"), ("sklearn", "scikit-learn"))
                    if m in v and n in lock and v[m] != lock[n]]
            out.append(Check("Version 1", "Version 1 environment (venv)", "degraded" if diff else "ok",
                             f"all packages import (TensorFlow {v.get('tensorflow')}, numpy {v.get('numpy')}, "
                             f"Streamlit {v.get('streamlit')})" + (f"; differs from lock: {', '.join(diff)}"
                                                                    if diff else ""),
                             "venv\\Scripts\\pip install --no-deps -r ewsv2\\handover\\locks\\requirements-lock-v1.txt"
                             if diff else ""))
            if res.get("model_load") != "ok":
                out.append(Check("Version 1", "Version 1 model loading", "fail",
                                 f"random_forest.pkl does not load: {res.get('model_load')}",
                                 "venv\\Scripts\\python main.py --train"))
    missing = [m for m in V1_MODELS if not (REPO_ROOT / "models" / m).exists()]
    out.append(Check("Version 1", "Version 1 trained models", "ok" if not missing else "degraded",
                     f"all {len(V1_MODELS)} model files present" if not missing else "missing: " + ", ".join(missing),
                     "" if not missing else "venv\\Scripts\\python main.py --train"))
    gone = [d for d in V1_DATA if not (REPO_ROOT / d).exists()]
    app = (REPO_ROOT / "dashboard" / "app.py").exists()
    out.append(Check("Version 1", "Version 1 data and dashboard", "ok" if (app and not gone) else "degraded",
                     "datasets and dashboard/app.py present" if (app and not gone) else
                     "missing: " + ", ".join(gone + ([] if app else ["dashboard/app.py"])),
                     "" if (app and not gone) else "copy them from the capsule's project folder"))
    return out


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def check_dashboard() -> list[Check]:
    out = []
    for port, what in ((8502, "version 2 dashboard"), (8501, "version 1 dashboard")):
        free = _port_free(port)
        out.append(Check("Dashboard", f"Port {port}", "ok" if free else "degraded",
                         f"free for the {what}" if free else f"in use – the {what} may already be running, or another "
                                                             "program uses the port",
                         "" if free else "close the other program or the old dashboard window (Ctrl+C)"))
    if importlib.util.find_spec("streamlit_echarts") is not None:
        out.append(Check("Dashboard", "Gauge charts", "ok", "ECharts component available (works offline)"))
    else:
        out.append(Check("Dashboard", "Gauge charts", "degraded", "ECharts component missing – Plotly gauges are used",
                         "pip install streamlit-echarts==0.4.0 pyecharts==2.0.6 simplejson==3.19.3"))
    return out


def check_disk() -> list[Check]:
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    st = "ok" if free > 5 else ("degraded" if free > 1 else "fail")
    return [Check("Computer", "Free disk space", st, f"{free:.1f} GB free on the project drive",
                  "" if st == "ok" else "free some space; outputs and the capsule need a few GB")]


# ---------------------------------------------------------------------------------------------- run

def run(board: bool = True, llm: bool = True, v1_probe: bool = True) -> list[Check]:
    checks: list[Check] = []
    for fn in (check_python, check_packages, check_plugins, check_data, check_outputs):
        checks += fn()
    checks += check_board(board)
    checks += check_llm() if llm else [Check("AI report writer", "Ollama + model", "skipped", "not checked (--no-llm)")]
    checks += check_version1(v1_probe)
    for fn in (check_dashboard, check_disk):
        checks += fn()
    return checks


MARK = {"ok": "[ OK ]", "degraded": "[ -- ]", "fail": "[FAIL]", "skipped": "[skip]"}


def render(checks: list[Check]) -> str:
    lines, area = [], None
    for c in checks:
        if c.area != area:
            lines.append(f"\n{c.area}")
            area = c.area
        lines.append(f"  {MARK[c.status]} {c.name}: {c.detail}")
        if c.fix and c.status in ("degraded", "fail"):
            lines.append(f"         fix: {c.fix}")
    n = {s: sum(c.status == s for c in checks) for s in MARK}
    verdict = ("READY – every part works." if n["fail"] == 0 and n["degraded"] == 0 else
               "READY WITH REPLACEMENTS – the system runs; the parts marked [ -- ] use their replacement."
               if n["fail"] == 0 else "NOT READY – fix the parts marked [FAIL] first.")
    lines.append(f"\n{n['ok']} ok, {n['degraded']} replaced, {n['fail']} failed, {n['skipped']} skipped.  {verdict}")
    return "\n".join(lines)


def write_report(checks: list[Check], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    (out_dir / "doctor.json").write_text(json.dumps({"time": stamp, "version": __version__, "segment": SEGMENT,
                                                     "checks": [asdict(c) for c in checks]}, indent=2),
                                         encoding="utf-8")
    md = [f"# System check – {stamp}", "", f"ewsv2 {__version__} · Python {platform.python_version()} · "
          f"{platform.system()} {platform.release()}", "", "| Area | Part | Status | Detail | Fix |",
          "|---|---|---|---|---|"]
    md += [f"| {c.area} | {c.name} | {c.status} | {c.detail} | {c.fix} |" for c in checks]
    (out_dir / "doctor.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out_dir / "doctor.md"
