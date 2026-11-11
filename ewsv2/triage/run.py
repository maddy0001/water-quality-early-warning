"""Segment 4 experiment: GECCO-Triage benchmark + three station systems.

    python -m ewsv2 triage --config s4_triage [--set data.source=gecco2019]

Steps
1. Build GECCO-Triage from the real dataset (``inject:`` section).
2. Honest split (official test period; 4 blocked folds before it).
3. Run each system in ``triage.systems`` and score every test episode.
4. Scan the *real* (uninjected) data with the fault checks: do they find the sensor
   problems spotted in S0, and how often do they fire on normal water?
5. Write results, ``report.md`` and ``triage.png``.
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
from ewsv2.evaluate.metrics import _merge, segments  # noqa: E402
from ewsv2.inject.benchmark import pretest_mask, test_mask  # noqa: E402
from ewsv2.signals.faults import CHECKS, FaultChecks  # noqa: E402
from ewsv2.triage import evaluate as TE  # noqa: E402
from ewsv2.triage.systems import Context, WaterDetector, build_system  # noqa: E402

log = logging.getLogger(__name__)

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUTCOME_COLOURS = {
    "water alarm": "#2a78d6", "sensor alert only": "#1baf7a", "both": "#eda100",
    "water alarm only": "#eb6834", "missed": "#c9c8c3",
}


def build_benchmark(ds: Dataset, cfg: dict) -> tuple[Dataset, pd.DataFrame]:
    inj_cfg = cfg.get("inject") or {}
    injector = build("injector", inj_cfg.get("name", "gecco_triage"), **(inj_cfg.get("params") or {}))
    return injector.build(ds)


def scan_real_faults(ds: Dataset, fault_params: dict, min_minutes: int = 30) -> pd.DataFrame:
    """Fault checks on the real, uninjected stream (limits from pre-test normal minutes)."""
    f = ds.frame
    has = ~f[ds.sensors].isna().all(axis=1).to_numpy()
    y = f[EVENT_COL].to_numpy().astype(bool)
    normal = pretest_mask(ds) & ~y & has
    fc = FaultChecks(ds.sensors, **fault_params).fit(f, normal)
    fl = fc.flags(f)
    tst = test_mask(ds)
    rows = []
    for check in CHECKS:
        for s in ds.sensors:
            m = fl[check][s].to_numpy()
            for a, b in _merge(segments(m), 60):
                dur = int(b - a + 1)
                if check in ("stuck", "drift") and dur < min_minutes:
                    continue
                rows.append({"check": check, "sensor": s, "start": f.index[a], "end": f.index[b],
                             "flagged_minutes": int(m[a:b + 1].sum()), "span_minutes": dur,
                             "period": "test" if tst[a] else "pre-test",
                             "event_minutes_inside": int(y[a:b + 1].sum()),
                             "value_at_start": float(f[s].iloc[a]) if np.isfinite(f[s].iloc[a]) else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["period", "span_minutes"], ascending=[True, False]).reset_index(drop=True)
    days = {"pre-test": (normal.sum()) / 1440, "test": (tst & has).sum() / 1440}
    out.attrs["days"] = days
    out.attrs["limits"] = {"stuck_limit_min": fc.stuck_limit, "spike_limit_units": {k: round(v, 1) for k, v in fc.spike_limit.items()},
                           "units": {k: float(v) for k, v in fc.units.items()}}
    return out


def run_triage(ds: Dataset, cfg: dict, out_dir: Path) -> dict:
    tcfg = cfg.get("triage") or {}
    seed = int(cfg.get("seed", 42))
    tol, gap = int(tcfg.get("tolerance_after", 0)), int(tcfg.get("merge_gap", 5))
    max_ep, grace = int(tcfg.get("max_episode_minutes", 60)), int(tcfg.get("fault_grace", 10))
    fault_params = dict(tcfg.get("fault_checks") or {})
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    inj, episodes = build_benchmark(ds, cfg)
    episodes.to_csv(out_dir / "episodes.csv", index=False)
    sp_cfg = tcfg.get("splitter", {"name": "official", "params": {"cv_folds": 4, "gap_minutes": 1440}})
    split = build("splitter", sp_cfg["name"], **(sp_cfg.get("params") or {})).split(inj)
    basic = build("feature", "basic").transform(inj.frame, inj.sensors)
    ctx = Context(inj, split, inj.frame[CLASS_COL].to_numpy(), basic, seed, ds.frame[EVENT_COL].to_numpy(),
                  tol, gap, max_ep, fault_params, grace)
    log.info("GECCO-Triage ready in %.1f s: %d test episodes injected, %d pre-test",
             time.perf_counter() - t0, int((episodes["period"] == "test").sum()),
             int((episodes["period"] == "pretest").sum()))

    eps = TE.test_episodes(ds.frame[EVENT_COL].to_numpy(), inj.frame.index, test_mask(inj), ctx.has_data, episodes)
    valid = split.test & ctx.has_data
    detector = WaterDetector(**(tcfg.get("detector") or {}))
    summaries, all_outcomes, infos = [], [], {}
    for item in tcfg.get("systems", ["detector_only", "rules", "learned"]):
        name, params = (item, {}) if isinstance(item, str) else (item["name"], item.get("params") or {})
        t1 = time.perf_counter()
        system = build_system(name, detector, params)
        out = system.run(ctx)
        secs = time.perf_counter() - t1
        oc = TE.episode_outcomes(eps, out.water, out.sensor, out.sensor_names, grace, tol)
        oc.insert(0, "system", name)
        all_outcomes.append(oc)
        row = {"system": name, "label": getattr(system, "label", name), "seconds": round(secs, 1),
               **TE.summarise(oc), **TE.rates(ctx.cls, out.water, out.sensor, inj.frame.index, valid,
                                               grace, tol, gap, max_ep)}
        summaries.append(row)
        infos[name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in out.info.items()}
        log.info("  %-14s %5.1fs  water caught %d/%d real, %d/%d injected, %.2f false water alarms/day | "
                 "faults: %d/%d sensor alert only, %d raised a water alarm, %.2f false sensor alerts/day",
                 name, secs, row["real_caught"], row["real_n"], row["contaminant_caught"], row["contaminant_n"],
                 row["water_false_alarms_per_day"], row["fault_correct"], row["fault_n"],
                 row["fault_raised_water_alarm"], row["false_sensor_alerts_per_day"])

    summary = pd.DataFrame(summaries)
    outcomes = pd.concat(all_outcomes, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    outcomes.drop(columns=["start_pos", "end_pos"]).to_csv(out_dir / "outcomes.csv", index=False)
    scan = scan_real_faults(ds, fault_params)
    scan.to_csv(out_dir / "real_fault_scan.csv", index=False)
    (out_dir / "thresholds.json").write_text(json.dumps(infos, indent=2, default=str), encoding="utf-8")
    write_report(ds, inj, cfg, episodes, summary, outcomes, scan, infos, out_dir)
    return {"summary": summary, "outcomes": outcomes, "episodes": episodes, "scan": scan}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _f(v, nd=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _med(series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return f"{s.median():.0f}" if len(s) else ""


def plot_outcomes(outcomes: pd.DataFrame, summary: pd.DataFrame, path: Path, title: str) -> None:
    water_order, fault_order = list(TE.WATER_OUTCOMES), list(TE.FAULT_OUTCOMES)
    groups = [("real event", water_order), ("injected contaminant", water_order), ("injected fault", fault_order)]
    systems = list(summary["system"])
    labels = dict(zip(summary["system"], summary["label"]))
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 1.0 + 0.62 * len(systems) * len(groups)),
                             facecolor=SURFACE, sharex=True)
    for ax, (src, order) in zip(np.atleast_1d(axes), groups):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        for i, sysname in enumerate(reversed(systems)):
            sub = outcomes[(outcomes["system"] == sysname) & (outcomes["source"] == src)]
            n = len(sub)
            left = 0.0
            for k in order:
                share = float((sub["outcome"] == k).mean()) if n else 0.0
                if share > 0:
                    ax.barh(i, share, left=left, color=OUTCOME_COLOURS[k], edgecolor=SURFACE, height=0.62,
                            label=k)
                    if share >= 0.08:
                        ax.text(left + share / 2, i, f"{int(round(share * n))}", ha="center", va="center",
                                fontsize=8, color=SURFACE if k != "missed" else INK)
                left += share
        ax.set_yticks(range(len(systems)), [labels[s] for s in reversed(systems)], fontsize=8.5, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=INK_2, labelsize=8)
        n_src = int(((outcomes["system"] == systems[0]) & (outcomes["source"] == src)).sum())
        ax.set_title(f"{src}s ({n_src} in the test period)", loc="left", fontsize=9.5, color=INK)
        ax.set_xlim(0, 1)
    handles, labs = [], []
    for ax in np.atleast_1d(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labs:
                handles.append(h)
                labs.append(lab)
    np.atleast_1d(axes)[-1].set_xlabel("share of episodes (numbers inside bars = episodes)", color=INK_2, fontsize=9)
    fig.legend(handles, labs, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_report(ds, inj, cfg, episodes, summary, outcomes, scan, infos, out_dir: Path) -> None:
    L = [f"# {cfg.get('title', 'Triage')} – {ds.name}", ""]
    test_eps = episodes[episodes["period"] == "test"]
    pre_eps = episodes[episodes["period"] == "pretest"]
    L += ["## GECCO-Triage benchmark", "",
          f"Real {ds.name} stream with injected episodes. Test period: {len(test_eps)} injected "
          f"({int((test_eps['kind'] == 'contaminant').sum())} contaminants, {int((test_eps['kind'] == 'fault').sum())} faults) "
          f"plus the real labelled events. Pre-test period: {len(pre_eps)} injected, used only for training "
          "and for choosing thresholds.", ""]
    unobs = inj.meta.get("unobservable_contaminants") or []
    if unobs:
        L += [f"Not observable at this station (fewer than two of its sensors respond): {', '.join(unobs)}. "
              "These contaminants are not injected.", ""]
    L += ["| Kind | Type | Levels | Test episodes | Median length (min) |", "|---|---|---|---|---|"]
    for (kind, typ), g in test_eps.groupby(["kind", "type"]):
        L.append(f"| {kind} | {typ} | {', '.join(sorted(g['level'].unique()))} | {len(g)} | {g['minutes'].median():.0f} |")
    L.append("")
    units = inj.meta.get("units", {})
    L += ["One unit of change per sensor (normal 60-minute variability, floored at the reading step): "
          + ", ".join(f"{k} {v:.4g}" for k, v in units.items()) + ".", ""]

    L += ["## Results (official test period)", "",
          "| System | Real events caught | Contaminants caught | Water false alarms / day | Water event F1 | "
          "Faults → sensor alert only | Faults → water alarm | Faults missed | False sensor alerts / day |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        L.append(f"| {r['label']} | {r['real_caught']}/{r['real_n']} | {r['contaminant_caught']}/{r['contaminant_n']} | "
                 f"{_f(r['water_false_alarms_per_day'])} | {_f(r['water_event_f1'])} | "
                 f"{r['fault_correct']}/{r['fault_n']} | {r['fault_raised_water_alarm']}/{r['fault_n']} | "
                 f"{r['fault_missed']}/{r['fault_n']} | {_f(r['false_sensor_alerts_per_day'])} |")
    L += ["", "![Episode outcomes](triage.png)", "",
          "Water false alarms include water alarms raised during a sensor fault. "
          "A fault 'raises a water alarm' if a water alarm sounds while it lasts (plus "
          f"{cfg.get('triage', {}).get('fault_grace', 10)} min).", ""]

    sys_names = list(summary["system"])
    labels = dict(zip(summary["system"], summary["label"]))
    # E7
    L += ["## E7 – Contaminants by type and dose", "",
          "Events caught as water alarms (median delay in minutes in brackets).", "",
          "| Contaminant | Dose | " + " | ".join(labels[s] for s in sys_names) + " |",
          "|---|---|" + "---|" * len(sys_names)]
    inj_out = outcomes[outcomes["source"] == "injected contaminant"]
    for (typ, lvl), _ in inj_out[inj_out["system"] == sys_names[0]].groupby(["type", "level"]):
        cells = []
        for s in sys_names:
            g = inj_out[(inj_out["system"] == s) & (inj_out["type"] == typ) & (inj_out["level"] == lvl)]
            caught = g[g["outcome"] == "water alarm"]
            d = _med(caught["delay_min"])
            cells.append(f"{len(caught)}/{len(g)}" + (f" ({d})" if d else ""))
        L.append(f"| {typ} | {lvl} | " + " | ".join(cells) + " |")
    L.append("")
    # E8
    L += ["## E8 – Sensor faults by type", "",
          "Each cell: sensor alert only / raised a water alarm / missed (median minutes to the first "
          "sensor alert).", "",
          "| Fault | Severity | " + " | ".join(labels[s] for s in sys_names) + " |",
          "|---|---|" + "---|" * len(sys_names)]
    f_out = outcomes[outcomes["source"] == "injected fault"]
    for (typ, lvl), _ in f_out[f_out["system"] == sys_names[0]].groupby(["type", "level"]):
        cells = []
        for s in sys_names:
            g = f_out[(f_out["system"] == s) & (f_out["type"] == typ) & (f_out["level"] == lvl)]
            ok = int((g["outcome"] == "sensor alert only").sum())
            wa = int(g["outcome"].isin(["both", "water alarm only"]).sum())
            mi = int((g["outcome"] == "missed").sum())
            d = _med(g.loc[g["outcome"].isin(["sensor alert only", "both"]), "delay_min"])
            cells.append(f"{ok} / {wa} / {mi}" + (f" ({d})" if d else ""))
        L.append(f"| {typ} | {lvl} | " + " | ".join(cells) + " |")
    L.append("")
    named = summary[["label", "fault_named_right_sensor"]].dropna()
    if len(named):
        L += ["Sensor alerts that name the faulty sensor: " + "; ".join(
            f"{r['label']} {r['fault_named_right_sensor'] * 100:.0f}%" for _, r in named.iterrows()) + ".", ""]

    # Real-data scan
    L += ["## Fault checks on the real data (nothing injected)", "",
          "Limits learned from normal pre-test minutes only. Flags are merged when less than 60 min apart; "
          "stuck and drift flags shorter than 30 min are dropped.", ""]
    days = scan.attrs.get("days", {})
    if len(scan):
        cnt = scan.groupby(["period", "check"]).size().unstack(fill_value=0)
        L += ["| Period | Days | " + " | ".join(CHECKS) + " |", "|---|---|" + "---|" * len(CHECKS)]
        for p in ("pre-test", "test"):
            if p in cnt.index:
                L.append(f"| {p} | {days.get(p, 0):.0f} | " + " | ".join(str(int(cnt.loc[p].get(c, 0))) for c in CHECKS) + " |")
        L += ["", "Longest flagged stretches:", "",
              "| Check | Sensor | Start | Length (min) | Period | Event minutes inside | Value |",
              "|---|---|---|---|---|---|---|"]
        top = scan.sort_values("span_minutes", ascending=False).head(12)
        for _, r in top.iterrows():
            L.append(f"| {r['check']} | {r['sensor']} | {r['start']:%Y-%m-%d %H:%M} | {r['span_minutes']} | "
                     f"{r['period']} | {r['event_minutes_inside']} | {_f(r['value_at_start'], 3)} |")
    else:
        L.append("No flags.")
    lim = scan.attrs.get("limits", {})
    if lim:
        L += ["", "Stuck limits (min): " + ", ".join(f"{k} {v}" for k, v in lim["stuck_limit_min"].items()) + ".",
              "Spike limits (units): " + ", ".join(f"{k} {v}" for k, v in lim["spike_limit_units"].items()) + "."]
    L += ["", "## Thresholds (chosen on pre-test data only)", "", "```", json.dumps(infos, indent=2, default=str), "```",
          "", "Files: `episodes.csv` (the benchmark), `outcomes.csv` (every episode × system), "
          "`summary.csv`, `real_fault_scan.csv`."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    best = summary.loc[summary["fault_correct"].idxmax(), "label"]
    plot_outcomes(outcomes, summary, out_dir / "triage.png",
                  f"Water event or sensor fault? ({ds.name}, test period) – most faults sent to the sensor: {best}")
