"""Segment 7 experiment: incident reports and their fact check (contribution C8).

    python -m ewsv2 report --config s7_reports [--set data.source=gecco2019]
                                              [--set llm.models=[qwen2.5:3b,llama3.2:3b]]

1. Rebuild GECCO-Triage and read the S5 pipeline alarms (run S5 first).
2. Build an alarm record for every test-period alarm; take a balanced sample
   (``llm.max_reports``: real events, contaminants, faults, false alarms, sensor alerts).
3. Retrieve guidance snippets for each record (by sensor role and alarm kind).
4. Write a report with the template and with each local LLM (Ollama), if it is running.
5. Fact-check every report; the *guarded* output shows the LLM report only when it passes,
   otherwise the template.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ewsv2.core.registry import build
from ewsv2.core.types import EVENT_COL, Dataset
from ewsv2.explain.run import load_alarms
from ewsv2.llm import faithfulness as FC
from ewsv2.llm.records import build_records, public, select
from ewsv2.llm.reporters import load_guidance, retrieve
from ewsv2.triage.run import build_benchmark

log = logging.getLogger(__name__)

STATION = {"gecco2018": "GECCO 2018 station (Thueringer Fernwasserversorgung)",
           "gecco2019": "GECCO 2019 station (Thueringer Fernwasserversorgung)"}


def make_records(ds: Dataset, cfg: dict) -> tuple[list[dict], Dataset]:
    lcfg = cfg.get("llm") or {}
    inj, episodes = build_benchmark(ds, cfg)
    minutes = load_alarms(ds.name, inj.frame.index, lcfg.get("alarms_from", "s5_fusion"))
    split = build("splitter", "official", cv_folds=4, gap_minutes=1440).split(inj)
    f = inj.frame
    real = ds.frame[EVENT_COL].to_numpy().astype(int)
    has = ~f[inj.sensors].isna().all(axis=1).to_numpy()
    cls = f["LABEL_CLASS"].to_numpy()
    normal = split.fit & has & (cls == 0) & (real == 0)
    dev = build("explainer", "deviation", sensors=inj.sensors).fit(f, normal)
    recs = build_records(ds, inj, episodes, minutes, dev, STATION.get(ds.name, ds.name))
    return recs, inj


def run_reports(ds: Dataset, cfg: dict, out_dir: Path) -> dict:
    lcfg = cfg.get("llm") or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    records, inj = make_records(ds, cfg)
    chosen = select(records, int(lcfg.get("max_reports", 40)))
    log.info("%d alarm records in the test period; reporting on %d (%s)", len(records), len(chosen),
             pd.Series([r["truth"].split(" on ")[0] for r in chosen]).value_counts().to_dict())
    guidance = load_guidance(lcfg.get("guidance_file"))
    (out_dir / "alarm_records.json").write_text(json.dumps(records, indent=1, default=str), encoding="utf-8")

    reporters = [build("reporter", "template")]
    status = {}
    for model in lcfg.get("models", ["qwen2.5:3b"]) if lcfg.get("enabled", True) else []:
        rep = build("reporter", "ollama", model=model, host=lcfg.get("host", "http://localhost:11434"),
                    seed=int(cfg.get("seed", 42)), max_tokens=int(lcfg.get("max_tokens", 300)))
        ok, why = rep.available()
        status[rep.name] = why
        if ok:
            reporters.append(rep)
        else:
            log.warning("Skipping %s: %s", rep.name, why)

    rows, texts = [], []
    for rep in reporters:
        t1 = time.perf_counter()
        for i, r in enumerate(chosen, 1):
            snippets = retrieve(r, guidance)
            out = rep.report(r, snippets)
            errs = FC.check(out["text"], r, snippets, inj.sensors) if out["ok"] else [{"type": "decision", "found": "no report"}]
            rows.append({"reporter": rep.name, "id": r["id"], "kind": r["kind"], "truth": r["truth"],
                         "seconds": round(out["seconds"], 2), "ok": out["ok"], "errors": len(errs),
                         "error_types": ",".join(sorted({e["type"] for e in errs})),
                         "error_tokens": "; ".join(f"{e['type']}:{e['found']}" for e in errs)[:300],
                         "mentions_top_sensor": FC.mentions_top_sensor(out["text"], r, inj.sensors)})
            texts.append({"reporter": rep.name, "id": r["id"], "text": out["text"], "errors": errs,
                          "record": public(r), "guidance_ids": [g["id"] for g in snippets]})
            if rep.name != "template" and i % 5 == 0:
                log.info("  %s: %d/%d reports (%.0f s)", rep.name, i, len(chosen), time.perf_counter() - t1)
        log.info("  %-22s %d reports in %.0f s", rep.name, len(chosen), time.perf_counter() - t1)
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "report_checks.csv", index=False)
    (out_dir / "reports.json").write_text(json.dumps(texts, indent=1, default=str, ensure_ascii=False),
                                          encoding="utf-8")

    summary = []
    for name, g in per.groupby("reporter", sort=False):
        errs = [t["errors"] for t in texts if t["reporter"] == name]
        s = FC.summarise(errs)
        s.update({"reporter": name, "mentions_top_sensor": float(g["mentions_top_sensor"].mean()),
                  "median_seconds": float(g["seconds"].median()), "failed_calls": int((~g["ok"]).sum())})
        if name != "template":
            s["guarded_shows_llm"] = s["faithful_share"]        # the rest fall back to the template
        summary.append(s)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "report_summary.csv", index=False)
    write_report(ds, cfg, records, chosen, summary, texts, status, out_dir, time.perf_counter() - t0)
    return {"summary": summary, "per_report": per, "status": status, "records": records}


def _f(v, nd=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_report(ds, cfg, records, chosen, summary, texts, status, out_dir: Path, seconds: float) -> None:
    kinds = pd.Series([r["kind"] for r in records]).value_counts().to_dict()
    L = [f"# Incident reports and their fact check – {ds.name}", "",
         f"{len(records)} alarm records in the test period ({kinds}); {len(chosen)} reported "
         "(balanced sample).", ""]
    if status:
        L += ["LLM status: " + "; ".join(f"{k}: {v}" for k, v in status.items()), ""]
    L += ["| Reporter | Reports | Faithful (no error) | Errors / report | decision | time | number | sensor | "
          "citation | Names the top sensor | Median seconds |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, s in summary.iterrows():
        L.append(f"| {s['reporter']} | {int(s['reports'])} | {_f(s['faithful_share'])} | {_f(s['errors_per_report'])} | "
                 + " | ".join(_f(s[f"with_{t}_error"]) for t in FC.ERROR_TYPES)
                 + f" | {_f(s['mentions_top_sensor'])} | {_f(s['median_seconds'], 1)} |")
    L += ["", "Error columns: share of reports with at least one error of that kind. The guarded output shows "
          "an LLM report only if it has no error, otherwise the template report.", ""]
    L += ["## Examples", ""]
    shown = set()
    for t in texts:
        key = (t["reporter"], t["record"]["decision"])
        if key in shown:
            continue
        shown.add(key)
        L += [f"**{t['reporter']} – {t['id']}** ({t['record']['decision']})", "", "```", t["text"], "```",
              ("Errors: " + "; ".join(f"{e['type']}: {e['found']}" for e in t["errors"])) if t["errors"] else "No errors.",
              ""]
    L += ["Files: `alarm_records.json` (every alarm, with what it really was), `reports.json` (every report with "
          "its record and errors), `report_checks.csv`, `report_summary.csv`.",
          f"", f"Run time: {seconds:.0f} s."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
