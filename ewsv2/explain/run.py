"""Segment 6, experiment E9: are the explanations right?

    python -m ewsv2 explain --config s6_explain [--set data.source=gecco2019]

Reads the alarms of the S5 pipeline (``outputs_v2/s5_fusion/<dataset>/pipeline_minutes.csv.gz``;
run S5 first), takes the first alarm minute of every test episode that raised one, and asks
each explainer which sensors drove it.

* Injected contaminants: the right answer is known – the sensors the signature moved
  (``hit@1`` = the top sensor is one of them; ``lead@1`` = it is the sensor that moved most).
* Injected faults that raised a water alarm: the right answer is the faulty sensor.
* Real events: no ground truth; we report which sensors are blamed and how often the
  explainers agree.
* Deletion test (faithfulness): hold the explainer's top sensor at its pre-alarm level and
  re-score the classifier; ``faithful@1`` = share of alarms where that removes at least half
  of the classifier's score. A random sensor sets the floor.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ewsv2.core.registry import build  # noqa: E402
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset  # noqa: E402
from ewsv2.core.paths import OUTPUT_DIR  # noqa: E402
from ewsv2.evaluate.metrics import segments  # noqa: E402
from ewsv2.explain.methods import ClassifierExplainer, ranking  # noqa: E402
from ewsv2.inject.benchmark import test_mask  # noqa: E402
from ewsv2.triage.run import build_benchmark  # noqa: E402

log = logging.getLogger(__name__)

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
EXPLAINERS = ("deviation", "forecast_residual", "tree_shap", "occlusion")
LABELS = {"deviation": "deviation (model-free)", "forecast_residual": "forecast residual (F)",
          "tree_shap": "TreeSHAP (classifier C)", "occlusion": "occlusion (classifier C)", "random": "random sensor"}


def load_alarms(ds_name: str, index: pd.DatetimeIndex, source_run: str = "s5_fusion") -> pd.DataFrame:
    path = OUTPUT_DIR / source_run / ds_name / "pipeline_minutes.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run  python -m ewsv2 fuse --config s5_fusion"
                                + ("" if ds_name == "gecco2018" else f" --set data.source={ds_name}") + "  first")
    m = pd.read_csv(path, index_col=0, parse_dates=True)
    return m.reindex(index)


def explained_episodes(ds: Dataset, inj: Dataset, episodes: pd.DataFrame, alarm: np.ndarray, grace: int) -> pd.DataFrame:
    """Test episodes with an alarm inside, the first alarm minute, and the known answer."""
    idx = inj.frame.index
    tst = test_mask(inj)
    rows = []
    for s, e in segments(ds.frame[EVENT_COL].to_numpy().astype(bool) & tst):
        hit = np.flatnonzero(alarm[s:e + 1])
        if len(hit):
            rows.append({"source": "real event", "type": "real", "level": "", "start": idx[s], "onset": s + hit[0],
                         "truth": "", "lead": ""})
    for _, r in episodes[episodes["period"] == "test"].iterrows():
        s, e = int(r["start_pos"]), int(r["end_pos"]) + (grace if r["kind"] == "fault" else 0)
        hit = np.flatnonzero(alarm[s:e + 1])
        if not len(hit):
            continue
        if r["kind"] == "contaminant":
            resp = json.loads(r["info"])["response"]
            truth = [k for k in r["sensors"].split(",") if k]
            lead = max(resp, key=lambda k: abs(resp[k]))
            src = "injected contaminant"
        else:
            truth, lead, src = [r["lead_sensor"]], r["lead_sensor"], "injected fault"
        rows.append({"source": src, "type": r["type"], "level": r["level"], "start": idx[s], "onset": s + hit[0],
                     "truth": ",".join(truth), "lead": lead})
    return pd.DataFrame(rows)


def run_explain(ds: Dataset, cfg: dict, out_dir: Path) -> dict:
    ecfg = cfg.get("explain") or {}
    seed = int(cfg.get("seed", 42))
    grace = int(ecfg.get("fault_grace", 10))
    drop = float(ecfg.get("faithful_drop", 0.5))
    out_dir.mkdir(parents=True, exist_ok=True)

    inj, episodes = build_benchmark(ds, cfg)
    frame, sensors = inj.frame, inj.sensors
    alarms = load_alarms(ds.name, frame.index, ecfg.get("alarms_from", "s5_fusion"))
    alarm = alarms["alarm_before_triage"].eq(True).to_numpy()
    split = build("splitter", "official", cv_folds=4, gap_minutes=1440).split(inj)
    cls = frame[CLASS_COL].to_numpy().astype(int)
    real = ds.frame[EVENT_COL].to_numpy().astype(int)
    has = ~frame[sensors].isna().all(axis=1).to_numpy()
    usable = has & ~(((cls == 1) & (real == 0)) | (cls == 2)) & split.fit
    normal = usable & (real == 0)

    eps = explained_episodes(ds, inj, episodes, alarm, grace)
    log.info("Explaining %d alarmed test episodes on %s (%s)", len(eps), inj.name,
             eps["source"].value_counts().to_dict())
    t0 = time.perf_counter()
    dev = build("explainer", "deviation", sensors=sensors).fit(frame, normal)
    fr = build("explainer", "forecast_residual", sensors=sensors, seed=seed).fit(frame, normal)
    clf = ClassifierExplainer(sensors, seed=seed).fit(frame, usable, real)
    log.info("  explainers fitted in %.0f s", time.perf_counter() - t0)

    pos = eps["onset"].to_numpy(dtype=int)
    t1 = time.perf_counter()
    att = {"deviation": dev.explain(frame, pos), "forecast_residual": fr.explain(frame, pos),
           "tree_shap": clf.tree_shap(frame, pos)}
    occ, base = clf.occlusion(frame, pos, pos)
    att["occlusion"] = occ
    log.info("  attributions for %d alarms in %.0f s", len(pos), time.perf_counter() - t1)

    rows = []
    for i, (_, e) in enumerate(eps.iterrows()):
        truth = [x for x in str(e["truth"]).split(",") if x]
        row = {**e.to_dict(), "classifier_score": round(float(base[i]), 6)}
        for name in EXPLAINERS:
            rank = ranking(att[name].iloc[i])
            row[f"{name}_top3"] = ",".join(rank[:3])
            top = rank[0]
            row[f"{name}_hit1"] = (top in truth) if truth else np.nan
            row[f"{name}_lead1"] = (top == e["lead"]) if truth else np.nan
            k = max(1, len(truth))
            row[f"{name}_cover"] = (len(set(rank[:k]) & set(truth)) / len(truth)) if truth else np.nan
            row[f"{name}_faithful1"] = (occ.iloc[i][top] >= drop * base[i]) if base[i] > 0 else np.nan
        # random sensor: the expected value over all sensors (no lucky draw)
        row["random_hit1"] = (len(set(truth) & set(sensors)) / len(sensors)) if truth else np.nan
        row["random_faithful1"] = float((occ.iloc[i] >= drop * base[i]).mean()) if base[i] > 0 else np.nan
        rows.append(row)
    per = pd.DataFrame(rows)
    per.drop(columns=["onset"]).to_csv(out_dir / "explanations.csv", index=False)

    summary = []
    for src in ("injected contaminant", "injected fault", "real event"):
        sub = per[per["source"] == src]
        if not len(sub):
            continue
        for name in (*EXPLAINERS, "random"):
            rec = {"source": src, "explainer": name, "episodes": len(sub)}
            for m in ("hit1", "lead1", "cover", "faithful1"):
                col = f"{name}_{m}"
                rec[m] = float(pd.to_numeric(sub[col], errors="coerce").mean()) if col in sub else np.nan
            summary.append(rec)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "explanation_summary.csv", index=False)
    write_report(ds, cfg, per, summary, sensors, out_dir)
    return {"summary": summary, "per_episode": per}


def _f(v, nd=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def plot_hits(summary: pd.DataFrame, path: Path, title: str) -> None:
    srcs = [s for s in ("injected contaminant", "injected fault") if s in set(summary["source"])]
    names = [*EXPLAINERS, "random"]
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    width = 0.8 / max(1, len(srcs))
    x = np.arange(len(names))
    for k, src in enumerate(srcs):
        sub = summary[summary["source"] == src].set_index("explainer").reindex(names)
        vals = sub["hit1"].to_numpy(dtype=float)
        bars = ax.bar(x + (k - (len(srcs) - 1) / 2) * width, vals, width * 0.9, color=SERIES[k],
                      label=f"{src}s ({int(sub['episodes'].iloc[0])})")
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, [LABELS[n] for n in names], fontsize=8, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("top sensor is one that really changed", color=INK_2, fontsize=8.5)
    ax.tick_params(axis="y", colors=INK_2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_report(ds, cfg, per, summary, sensors, out_dir: Path) -> None:
    L = [f"# Explanations (E9) – {ds.name}", "",
         "First alarm minute of every alarmed test episode (alarms of the S5 pipeline, before triage).", "",
         "| Explainer | What it looks at |", "|---|---|",
         "| deviation | each sensor's 15-minute move, in units of its largest normal move (no model) |",
         "| forecast residual | each sensor's 5-minute-ahead forecast error (module F) |",
         "| TreeSHAP | exact SHAP contributions of the classifier's XGBoost part, summed per sensor (module C) |",
         "| occlusion | drop of the full classifier score when one sensor is held at its pre-alarm level (module C) |",
         "| random sensor | a sensor picked at random: expected value (the floor) |", ""]
    for src, what in (("injected contaminant", "Right answer = the sensors the injected signature moved."),
                      ("injected fault", "Right answer = the faulty sensor (faults that raised a water alarm "
                                         "before triage).")):
        sub = summary[summary["source"] == src]
        if not len(sub):
            continue
        L += [f"## {src.capitalize()}s ({int(sub['episodes'].iloc[0])} alarmed)", "", what, "",
              "| Explainer | hit@1 | lead@1 | coverage | faithful@1 |", "|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            L.append(f"| {LABELS[r['explainer']]} | {_f(r['hit1'])} | {_f(r['lead1'])} | {_f(r['cover'])} | "
                     f"{_f(r['faithful1'])} |")
        L.append("")
    real = per[per["source"] == "real event"]
    if len(real):
        L += [f"## Real events ({len(real)} alarmed) – no ground truth", "",
              "Which sensor each explainer blames first (number of events):", "",
              "| Sensor | " + " | ".join(LABELS[n] for n in EXPLAINERS) + " |", "|---|" + "---|" * len(EXPLAINERS)]
        tops = {n: real[f"{n}_top3"].str.split(",").str[0].value_counts() for n in EXPLAINERS}
        for s in sensors:
            L.append(f"| {s} | " + " | ".join(str(int(tops[n].get(s, 0))) for n in EXPLAINERS) + " |")
        agree = {n: float((real[f"{n}_top3"].str.split(",").str[0] == real["occlusion_top3"].str.split(",").str[0]).mean())
                 for n in EXPLAINERS if n != "occlusion"}
        fa = summary[(summary["source"] == "real event")].set_index("explainer")["faithful1"]
        L += ["", "Agreement of the top sensor with occlusion: " +
              ", ".join(f"{LABELS[k]} {v:.2f}" for k, v in agree.items()) + ".",
              "faithful@1 on real events: " + ", ".join(f"{LABELS[k]} {_f(v)}" for k, v in fa.items()) + ".", ""]
    L += ["## Definitions", "",
          "- **hit@1**: the top-ranked sensor is one of the sensors that really changed.",
          "- **lead@1**: the top-ranked sensor is the one that changed most.",
          "- **coverage**: share of the changed sensors found in the top k (k = number of changed sensors).",
          f"- **faithful@1**: holding the top sensor at its pre-alarm level removes at least "
          f"{int(100 * float(cfg.get('explain', {}).get('faithful_drop', 0.5)))}% of the classifier's score.",
          "", "![hit@1](explanations.png)", "",
          "Files: `explanations.csv` (every episode: top-3 sensors per explainer – used by the S7 reports), "
          "`explanation_summary.csv`."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot_hits(summary, out_dir / "explanations.png",
              f"Does the explanation blame a sensor that really changed? ({ds.name}, test period)")
