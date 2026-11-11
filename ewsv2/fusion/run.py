"""Segment 5 experiment: fusion (E4), calibration under a false-alarm budget (E5),
persistence (E6), and the full pipeline with the S4 triage on top.

    python -m ewsv2 fuse --config s5_fusion [--set data.source=gecco2019]

Runs on the GECCO-Triage stream of S4, so each configuration is scored on the real events,
on injected contaminants it never saw, and on injected sensor faults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ewsv2.calibrate.methods import Nines, budget_threshold, persistence, to_alarm  # noqa: E402
from ewsv2.core.registry import build  # noqa: E402
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset  # noqa: E402
from ewsv2.evaluate.metrics import EventEvaluator  # noqa: E402
from ewsv2.fusion.methods import Stacking  # noqa: E402
from ewsv2.fusion.modules import module_nines  # noqa: E402
from ewsv2.inject.benchmark import test_mask  # noqa: E402
from ewsv2.triage import evaluate as TE  # noqa: E402
from ewsv2.triage.run import build_benchmark  # noqa: E402
from ewsv2.triage.systems import Context, RuleTriage  # noqa: E402

log = logging.getLogger(__name__)

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]


def _spec(item, kind_default: str) -> tuple[str, dict]:
    if isinstance(item, str):
        return item, {}
    return item["name"], dict(item.get("params") or {})


def _fusion_label(name: str, params: dict, obj) -> str:
    return getattr(obj, "name", name)


class _FixedAlarm:
    """Lets the S4 rules run on an alarm series produced here."""

    def __init__(self, alarm: np.ndarray):
        self._alarm = alarm
        self.final, self.threshold, self.cv_value = True, float("nan"), float("nan")

    def alarm(self):
        return self._alarm


def fused_scores(fusion_name: str, params: dict, N: pd.DataFrame, real: np.ndarray, split, usable: np.ndarray):
    """Fused score: out-of-fold over pre-test rows (cross-fitted if the fusion learns), and on test rows."""
    fusion = build("fusion", fusion_name, **params)
    n = len(N)
    out = np.full(n, np.nan)
    if getattr(fusion, "trainable", False):
        val_rows = split.val & usable & N.notna().all(axis=1).to_numpy()
        for _, val_k in split.folds:
            train = val_rows & ~val_k
            if real[train].sum() == 0 or not val_k.any():
                continue
            f = build("fusion", fusion_name, **params).fit(N, real, train)
            out[val_k] = f.combine(N[val_k])
        fusion.fit(N, real, val_rows)
        out[split.test] = fusion.combine(N[split.test])
    else:
        out[split.val | split.test] = fusion.combine(N[split.val | split.test])
    return out, fusion


def week_rates(result, evaluator, freq: str = "W-MON") -> pd.DataFrame:
    """False alarms per normal day, week by week."""
    df = pd.DataFrame({"false": result.false_episode_days,
                       "minutes": evaluator.normal_minutes_by_day.reindex(result.false_episode_days.index, fill_value=0)})
    wk = df.resample(freq, label="left", closed="left").sum()
    wk = wk[wk["minutes"] >= 1440]                       # at least one normal day of data
    wk["per_day"] = wk["false"] / (wk["minutes"] / 1440)
    return wk


def run_fusion(ds: Dataset, cfg: dict, out_dir: Path) -> dict:
    fcfg = cfg.get("fuse") or {}
    seed = int(cfg.get("seed", 42))
    tol, gap = int(fcfg.get("tolerance_after", 0)), int(fcfg.get("merge_gap", 5))
    max_ep, grace = int(fcfg.get("max_episode_minutes", 60)), int(fcfg.get("fault_grace", 10))
    out_dir.mkdir(parents=True, exist_ok=True)

    inj, episodes = build_benchmark(ds, cfg)
    frame = inj.frame
    sp_cfg = fcfg.get("splitter", {"name": "official", "params": {"cv_folds": 4, "gap_minutes": 1440}})
    split = build("splitter", sp_cfg["name"], **(sp_cfg.get("params") or {})).split(inj)
    cls = frame[CLASS_COL].to_numpy().astype(int)
    real = ds.frame[EVENT_COL].to_numpy().astype(int)
    has = ~frame[inj.sensors].isna().all(axis=1).to_numpy()
    injected = ((cls == 1) & (real == 0)) | (cls == 2)
    usable = has & ~injected
    water = (cls == 1).astype(int)
    index = frame.index
    valid_test = split.test & has
    cal_rows = split.val & usable
    log.info("Fusion on %s: %d test episodes injected", inj.name, int((episodes["period"] == "test").sum()))

    # 1. module nines (cross-conformal); optional cache for re-running the report quickly
    key = json.dumps({"modules": fcfg["modules"], "inject": cfg.get("inject"), "split": sp_cfg, "seed": seed},
                     sort_keys=True, default=str)
    cache = out_dir / "module_nines_cache.pkl"
    N = None
    if fcfg.get("cache", False) and cache.exists():
        stored = pd.read_pickle(cache)
        if stored.get("key") == key:
            N, minfo = stored["N"], stored["info"]
            log.info("  module scores loaded from %s", cache.name)
    if N is None:
        cols, minfo = {}, {}
        for spec in fcfg["modules"]:
            s, info = module_nines(frame, inj.sensors, split, real, usable, spec, seed)
            cols[spec["role"]] = s
            minfo[spec["role"]] = info
        N = pd.DataFrame(cols)
        if fcfg.get("cache", False):
            pd.to_pickle({"key": key, "N": N, "info": minfo}, cache)

    eps = TE.test_episodes(real, index, test_mask(inj), has, episodes)
    evaluator = EventEvaluator(water, index, valid_test, tol, gap, max_ep)

    def score_alarm(alarm, label, fusion_label, calib_label, extra=None):
        res = evaluator.evaluate(alarm)
        oc = TE.episode_outcomes(eps, alarm, np.zeros(len(alarm), dtype=bool), None, grace, tol)
        s = TE.summarise(oc)
        wk = week_rates(res, evaluator)
        row = {"config": label, "fusion": fusion_label, "calibrator": calib_label,
               "real_caught": s["real_caught"], "real_n": s["real_n"],
               "contaminant_caught": s["contaminant_caught"], "contaminant_n": s["contaminant_n"],
               "faults_raising_water_alarm": s["fault_raised_water_alarm"], "fault_n": s["fault_n"],
               "water_recall": res.event_recall, "false_alarms_per_day": res.false_alarms_per_day,
               "median_delay_min": res.median_delay_min, "event_f1": res.event_f1,
               "worst_week_fa_per_day": float(wk["per_day"].max()) if len(wk) else np.nan,
               "weeks": len(wk), **(extra or {})}
        return row, wk, oc

    # 2. fusions x calibrators
    budget = float(fcfg.get("budget", 1.0))
    rows, weekly, fused_store, selection = [], {}, {}, []
    for item in fcfg["fusions"]:
        fname, fparams = _spec(item, "fusion")
        fused, fobj = fused_scores(fname, fparams, N, real, split, usable)
        flabel = _fusion_label(fname, fparams, fobj)
        nines = Nines(fused[cal_rows & np.isfinite(fused) & (real == 0)])
        z = nines(fused)
        fused_store[flabel] = (z, fobj)
        # pre-test (out-of-fold) quality at the budget threshold: used to pick the main fusion
        thr, _ = budget_threshold(z, real, index, cal_rows, budget, gap, max_ep)
        pre = EventEvaluator(real, index, cal_rows, tol, gap, max_ep).evaluate(to_alarm(z, thr))
        selection.append({"fusion": flabel, "pretest_event_f1": pre.event_f1, "pretest_recall": pre.event_recall,
                          "pretest_false_alarms_per_day": pre.false_alarms_per_day})
        for citem in fcfg["calibrators"]:
            cname, cparams = _spec(citem, "calibrator")
            cparams.setdefault("budget", budget)
            cal = build("calibrator", cname, **cparams).fit(z, real, index, cal_rows, tol, gap, max_ep)
            alarm = cal.alarm(z, water, index, valid_test) if cname == "adaptive" else cal.alarm(z)
            alarm &= valid_test
            row, wk, _ = score_alarm(alarm, f"{flabel} | {cname}", flabel, cname,
                                     {"threshold_nines": float(cal.threshold)})
            rows.append(row)
            weekly[(flabel, cname)] = wk
            log.info("  %-28s %-9s real %2d/%-2d contam. %2d/%-2d  FA/day %.2f (worst week %.2f)  F1 %.2f",
                     flabel, cname, row["real_caught"], row["real_n"], row["contaminant_caught"], row["contaminant_n"],
                     row["false_alarms_per_day"], row["worst_week_fa_per_day"], row["event_f1"])
    results = pd.DataFrame(rows)

    # 3. persistence (E6) and the full pipeline with triage, on the main configuration
    selection = pd.DataFrame(selection)
    main = fcfg.get("main", {"fusion": "auto", "calibrator": "adaptive"})
    if main["fusion"] == "auto":       # chosen on pre-test data only: best out-of-fold event F1 at the budget
        main_f = selection.loc[selection["pretest_event_f1"].idxmax(), "fusion"]
    else:
        main_f = _fusion_label(main["fusion"], {}, build("fusion", main["fusion"]))
    z, _ = fused_store[main_f]
    mc_name, mc_params = _spec(main["calibrator"], "calibrator")
    mc_params.setdefault("budget", budget)
    pers_rows = []
    for k, m in fcfg.get("persistence", [[1, 1], [2, 3], [3, 5]]):
        if mc_name == "adaptive":       # the rule sits inside the online loop
            cal = build("calibrator", mc_name, **{**mc_params, "persist": (k, m)}).fit(z, real, index, cal_rows, tol, gap, max_ep)
            alarm = cal.alarm(z, water, index, valid_test) & valid_test
        else:
            cal = build("calibrator", mc_name, **mc_params).fit(z, real, index, cal_rows, tol, gap, max_ep)
            alarm = persistence(cal.alarm(z) & valid_test, k, m) & valid_test
        row, _, _ = score_alarm(alarm, f"{k} of {m}", main_f, mc_name, {"k": k, "n": m})
        pers_rows.append(row)
    pers = pd.DataFrame(pers_rows)

    cal = build("calibrator", mc_name, **mc_params).fit(z, real, index, cal_rows, tol, gap, max_ep)
    main_alarm = (cal.alarm(z, water, index, valid_test) if mc_name == "adaptive" else cal.alarm(z)) & valid_test
    trace = getattr(cal, "trace", None)
    ctx = Context(inj, split, cls, None, seed, real, tol, gap, max_ep, dict(fcfg.get("fault_checks") or {}), grace)
    no_sensor = np.zeros(len(main_alarm), dtype=bool)
    variants = [("no triage", main_alarm, no_sensor, None)]
    rules = dict(fcfg.get("rules") or {})
    tri = RuleTriage(detector=_FixedAlarm(main_alarm), **rules).run(ctx)
    variants.append(("rules triage", tri.water & valid_test, tri.sensor & valid_test, tri.sensor_names))
    tri_nj = RuleTriage(detector=_FixedAlarm(main_alarm), **{**rules, "jump_units": float("inf")}).run(ctx)
    variants.append(("rules triage without the jump rule", tri_nj.water & valid_test, tri_nj.sensor & valid_test,
                     tri_nj.sensor_names))
    pipelines, oc_all = [], []
    for vlabel, w_, s_, names in variants:
        oc = TE.episode_outcomes(eps, w_, s_, names, grace, tol)
        row = {"label": f"{main_f} + {mc_name} + {vlabel}", "variant": vlabel, **TE.summarise(oc),
               **TE.rates(cls, w_, s_, index, valid_test, grace, tol, gap, max_ep)}
        pipelines.append(row)
        oc.insert(0, "system", row["label"])
        oc_all.append(oc)
    pipeline = pipelines[1]
    oc = pd.concat(oc_all, ignore_index=True)
    pd.DataFrame({"score_nines": z, "alarm_before_triage": main_alarm, "water_alarm": tri.water & valid_test,
                  "sensor_alert": tri.sensor & valid_test, "sensor_names": tri.sensor_names,
                  "label_class": cls}, index=index)[valid_test].to_csv(out_dir / "pipeline_minutes.csv.gz")

    results.to_csv(out_dir / "fusion_results.csv", index=False)
    pers.to_csv(out_dir / "persistence.csv", index=False)
    oc.drop(columns=["start_pos", "end_pos"]).to_csv(out_dir / "pipeline_outcomes.csv", index=False)
    pd.DataFrame(pipelines).to_csv(out_dir / "pipeline_summary.csv", index=False)
    wk_all = pd.concat({f"{a} | {b}": w["per_day"] for (a, b), w in weekly.items()}, axis=1)
    wk_all.to_csv(out_dir / "weekly_false_alarms.csv")
    stack = [obj for (z_, obj) in fused_store.values() if isinstance(obj, Stacking)]
    selection.to_csv(out_dir / "fusion_selection.csv", index=False)
    info = {"modules": minfo, "budget": budget, "stacking_weights": stack[0].weights() if stack else None,
            "main_fusion": main_f, "selection": selection.to_dict(orient="records")}
    (out_dir / "fusion_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    write_report(ds, cfg, results, pers, pipelines, weekly, info, main_f, mc_name, trace, index, out_dir)
    return {"results": results, "persistence": pers, "pipeline": pipeline, "pipelines": pipelines, "info": info}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _f(v, nd=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def plot_weekly(weekly: dict, fusion_label: str, budget: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    names = {"cv_f1": "best F1 on pre-test (S1 rule)", "budget": "fixed budget threshold", "adaptive": "adaptive"}
    k = 0
    last = None
    for (f, c), wk in weekly.items():
        if f != fusion_label or not len(wk):
            continue
        ax.plot(wk.index, wk["per_day"], marker="o", markersize=4, linewidth=2, color=SERIES[k % len(SERIES)],
                label=names.get(c, c))
        last = wk
        k += 1
    ax.axhline(budget, color=INK, linestyle="--", linewidth=1)
    if last is not None:
        ax.text(last.index[0], budget, f" budget {budget:g}/day", va="bottom", fontsize=8, color=INK)
    ax.set_ylabel("false alarms per day", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_report(ds, cfg, results, pers, pipelines, weekly, info, main_f, mc_name, trace, index, out_dir: Path) -> None:
    b = info["budget"]
    L = [f"# {cfg.get('title', 'Fusion and calibration')} – {ds.name}", "",
         "Run on the GECCO-Triage stream of S4 (real events + injected contaminants and faults). "
         "Modules learn from real events only; every injected episode is unseen.", "",
         "## Modules (cross-conformal: four fold models, no refit)", "",
         "| Role | Module | Fold models | Reference normal minutes | Seconds |", "|---|---|---|---|---|"]
    for role, m in info["modules"].items():
        L.append(f"| {role} | `{m['name']}`{(':' + m['part']) if m['part'] else ''} | {m['folds_used']} | "
                 f"{m['reference_minutes']:,} | {m['seconds']} |")
    if info.get("stacking_weights"):
        L += ["", "Learned stacking weights (logistic regression on nines): " +
              ", ".join(f"{k} {v:+.2f}" for k, v in info["stacking_weights"].items()) + "."]
    if info.get("selection"):
        L += ["", "## Choosing the main fusion (pre-test data only)", "",
              "Out-of-fold pre-test scores, threshold at the budget. The best event F1 here becomes the main "
              f"fusion for E5, E6 and the full pipeline: **{info['main_fusion']}**.", "",
              "| Fusion | Pre-test event F1 | Pre-test recall | Pre-test false alarms / day |", "|---|---|---|---|"]
        for r in info["selection"]:
            L.append(f"| {r['fusion']} | {_f(r['pretest_event_f1'])} | {_f(r['pretest_recall'])} | "
                     f"{_f(r['pretest_false_alarms_per_day'])} |")
    for cname, title in (("cv_f1", "E4 – fusion, threshold by best pre-test F1 (S1 rule)"),
                         ("budget", f"E4 – fusion, threshold held to {b:g} false alarm/day on pre-test water"),
                         ("adaptive", f"E5 – fusion with the adaptive threshold (budget {b:g}/day)")):
        sub = results[results["calibrator"] == cname]
        if not len(sub):
            continue
        L += ["", f"## {title}", "",
              "| Fusion | Real events | Contaminants | Faults → water alarm | False alarms / day | Worst week | "
              "Median delay (min) | Event F1 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            L.append(f"| {r['fusion']} | {r['real_caught']}/{r['real_n']} | {r['contaminant_caught']}/{r['contaminant_n']} | "
                     f"{r['faults_raising_water_alarm']}/{r['fault_n']} | {_f(r['false_alarms_per_day'])} | "
                     f"{_f(r['worst_week_fa_per_day'])} | {_f(r['median_delay_min'], 0)} | {_f(r['event_f1'])} |")
    L += ["", f"## E5 – does the false-alarm rate hold? ({main_f})", "",
          "| Threshold | False alarms / day | Worst week | Weeks within budget |", "|---|---|---|---|"]
    for (f, c), wk in weekly.items():
        if f != main_f or not len(wk):
            continue
        r = results[(results["fusion"] == f) & (results["calibrator"] == c)].iloc[0]
        L.append(f"| {c} | {_f(r['false_alarms_per_day'])} | {_f(wk['per_day'].max())} | "
                 f"{int((wk['per_day'] <= b).sum())}/{len(wk)} |")
    L += ["", "![Weekly false alarms](weekly.png)", "",
          f"## E6 – persistence rule ({main_f}, {mc_name})", "",
          "| Rule | Real events | Contaminants | False alarms / day | Median delay (min) | Event F1 |",
          "|---|---|---|---|---|---|"]
    for _, r in pers.iterrows():
        L.append(f"| {r['config']} minutes | {r['real_caught']}/{r['real_n']} | {r['contaminant_caught']}/{r['contaminant_n']} | "
                 f"{_f(r['false_alarms_per_day'])} | {_f(r['median_delay_min'], 0)} | {_f(r['event_f1'])} |")
    L += ["", f"## Full pipeline: {main_f} → {mc_name} threshold → S4 rules triage", "",
          "| Triage | Real events → water alarm | Real events → sensor alert only | Contaminants | "
          "Water false alarms / day | Water event F1 | Faults → sensor alert only | Faults → water alarm | "
          "False sensor alerts / day |", "|---|---|---|---|---|---|---|---|---|"]
    for p in pipelines:
        L.append(f"| {p['variant']} | {p['real_caught']}/{p['real_n']} | {p['real_sent_to_sensor']} | "
                 f"{p['contaminant_caught']}/{p['contaminant_n']} | {_f(p['water_false_alarms_per_day'])} | "
                 f"{_f(p['water_event_f1'])} | {p['fault_correct']}/{p['fault_n']} | "
                 f"{p['fault_raised_water_alarm']}/{p['fault_n']} | {_f(p['false_sensor_alerts_per_day'])} |")
    L += [
          "", "Compare with the S4 table (`outputs_v2/s4_triage/<dataset>/report.md`), which used the S1 Random Forest alone.",
          "", "Files: `fusion_results.csv`, `persistence.csv`, `weekly_false_alarms.csv`, `pipeline_summary.csv`, "
          "`pipeline_outcomes.csv`, `fusion_info.json`."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    plot_weekly(weekly, main_f, b, out_dir / "weekly.png",
                f"False alarms per day, week by week ({ds.name}, {main_f}) – dashed line = budget")
