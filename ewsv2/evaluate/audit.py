"""Run signals under one or more evaluation protocols, and write the report (Segment 1).

A *protocol* = a splitter + a rule for choosing the threshold + the metric a paper
would headline. The leakage audit (contribution C2) runs the same signals under the
protocols seen in published GECCO work and under the honest protocol, and puts the
numbers side by side. The baseline run (E2) is the honest protocol alone.

Config (``evaluate:`` section)::

    evaluate:
      tolerance_after: 0       # minutes after an event that still count as a hit
      merge_gap: 5             # alarm runs closer than this form one episode
      max_episode_minutes: 60  # a false alarm left on counts again every 60 min
      bootstrap: 1000          # resamples for 95% intervals (honest protocols only)
      signals:
        - rolling_zscore
        - {name: random_forest, params: {n_estimators: 100}}
        - {name: v1_ensemble, label: v1_ensemble_basic, params: {features: basic}, parts: false}
      protocols:
        - name: D_honest
          label: "Official test period, threshold from cross-validation"
          splitter: {name: official, params: {cv_folds: 4, gap_minutes: 1440}}
          threshold: {choose_on: val, objective: event_f1}
          headline: event_f1
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
from ewsv2.core.types import EVENT_COL, Dataset  # noqa: E402
from ewsv2.evaluate.metrics import EventEvaluator, bootstrap_ci, evaluate_all  # noqa: E402
from ewsv2.evaluate.thresholds import select_threshold, to_alarm  # noqa: E402

log = logging.getLogger(__name__)

HEADLINE_NAMES = {
    "point_f1": "point F1",
    "pa_f1": "point-adjusted F1",
    "event_f1": "event F1",
}

# chart colours (validated reference palette; text stays in ink colours)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
MARKERS = ["o", "s", "D", "^", "v"]


def _signal_specs(items) -> list[tuple[str, dict, str, bool]]:
    """(name, params, label, report_parts) for each configured signal."""
    specs = []
    for it in items:
        if isinstance(it, str):
            specs.append((it, {}, it, True))
        else:
            specs.append((it["name"], dict(it.get("params") or {}), it.get("label", it["name"]),
                          bool(it.get("parts", True))))
    labels = [s[2] for s in specs]
    dup = {x for x in labels if labels.count(x) > 1}
    if dup:
        raise ValueError(f"signal labels must be unique; give each a 'label:' ({sorted(dup)})")
    return specs


def _parts_of(model, Z: pd.DataFrame) -> pd.DataFrame:
    """All scores a signal reports: its own first, then its components (if any)."""
    if hasattr(model, "score_parts"):
        return model.score_parts(Z)
    return model.score(Z).to_frame(name="main")


def _scores_for(ds: Dataset, split, name: str, params: dict, Z: pd.DataFrame, y: np.ndarray, seed: int,
                use_folds: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final test scores (model fitted on ``split.fit``) and threshold-selection scores.

    Both are DataFrames with one column per reported part. With cross-validation folds
    (and ``use_folds``), selection scores are out-of-fold: every pre-test minute is
    scored by a model that never saw it.
    """
    def fresh():
        return build("signal", name, sensors=ds.sensors, seed=seed, **params)

    final = fresh().fit(Z[split.fit], y[split.fit])
    scores = _parts_of(final, Z)
    if split.folds and use_folds:
        oof = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
        for fit_k, val_k in split.folds:
            if not val_k.any():
                continue
            model = fresh()
            if getattr(model, "supervised", False) and y[fit_k].sum() == 0:
                continue  # no events to learn from in this fold
            model.fit(Z[fit_k], y[fit_k])
            part = _parts_of(model, Z[val_k])
            oof.loc[val_k, part.columns] = part.to_numpy()
        return scores, oof
    return scores, scores


def run_protocols(ds: Dataset, cfg: dict, out_dir: Path) -> pd.DataFrame:
    ev_cfg = cfg["evaluate"]
    tol, gap = int(ev_cfg.get("tolerance_after", 0)), int(ev_cfg.get("merge_gap", 5))
    max_ep = int(ev_cfg.get("max_episode_minutes", 60))
    n_boot, seed = int(ev_cfg.get("bootstrap", 1000)), int(cfg.get("seed", 42))
    specs = _signal_specs(ev_cfg["signals"])
    y = ds.frame[EVENT_COL].to_numpy()
    idx = ds.frame.index

    cache: dict[str, pd.DataFrame] = {}
    rows, per_event_frames = [], []
    for proto in ev_cfg["protocols"]:
        sp_cfg = proto["splitter"]
        split = build("splitter", sp_cfg["name"], **(sp_cfg.get("params") or {})).split(ds)
        th = proto.get("threshold", {"choose_on": "val", "objective": "event_f1"})
        if True in th or "on" in th:
            raise ValueError(f"protocol {proto['name']}: write threshold.choose_on (YAML reads a bare `on` as true)")
        on = th.get("choose_on", "val")
        if on == "val" and not split.val.any():
            raise ValueError(f"protocol {proto['name']}: split has no validation rows; use threshold.choose_on: test")
        log.info("Protocol %s: %s %s", proto["name"], split.description, split.sizes())
        honest = not split.leaky and on == "val"

        for name, params, label, report_parts in specs:
            t0 = time.perf_counter()
            key = f"{name}:{json.dumps(params, sort_keys=True)}"
            probe = build("signal", name, sensors=ds.sensors, seed=seed, **params)
            if key not in cache:
                cache[key] = probe.transform(ds.frame)
            Z = cache[key]
            if getattr(probe, "supervised", False) and y[split.fit].sum() == 0:
                log.warning("Skipping %s under %s: no labelled events in the fit rows", label, proto["name"])
                continue
            scores, select_scores = _scores_for(ds, split, name, params, Z, y, seed, use_folds=(on == "val"))
            fit_seconds = round(time.perf_counter() - t0, 1)
            fixed = getattr(probe, "fixed_thresholds", {}) or {}
            columns = list(scores.columns) if report_parts else list(scores.columns[:1])
            for j, part in enumerate(columns):
                row_name = label if j == 0 else f"{label}:{part}"
                s_final = scores[part].to_numpy()
                if part in fixed:
                    thr, sel_value, how = float(fixed[part]), float("nan"), "fixed"
                else:
                    rows_for_t = split.test if on == "test" else split.val
                    thr, sel_value = select_threshold(select_scores[part].to_numpy(), y, idx, rows_for_t,
                                                      th.get("objective", "event_f1"), tol, gap, max_ep)
                    how = on
                alarm = to_alarm(s_final, thr)
                metrics = evaluate_all(y, alarm, idx, split.test, tol, gap, max_ep)
                row = {
                    "protocol": proto["name"], "protocol_label": proto.get("label", proto["name"]),
                    "leaky_split": split.leaky, "threshold_on": how, "honest": honest,
                    "headline_metric": proto.get("headline", "event_f1"),
                    "signal": row_name, "model": name, "part": part, "threshold": thr,
                    "selection_value": sel_value, **metrics, "seconds": fit_seconds if j == 0 else 0.0,
                }
                row["headline_value"] = row[row["headline_metric"]]
                if honest and n_boot > 0:
                    evaluator = EventEvaluator(y, idx, split.test, tol, gap, max_ep)
                    res = evaluator.evaluate(alarm)
                    ci = bootstrap_ci(res, evaluator.normal_minutes_by_day, n_boot, seed)
                    for k, (lo, hi) in ci.items():
                        row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
                    per_event_frames.append(res.per_event.assign(protocol=proto["name"], signal=row_name))
                rows.append(row)
                log.info("  %-28s %5.1fs  %s=%.3f  event recall=%.2f  false alarms/day=%.2f",
                         row_name, row["seconds"], row["headline_metric"], row["headline_value"],
                         row["event_recall"], row["false_alarms_per_day"])

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    if per_event_frames:
        pd.concat(per_event_frames).to_csv(out_dir / "per_event.csv", index=False)
    write_report(ds, cfg, results, out_dir)
    return results


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _fmt(v, nd=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _ci(row, key, nd=2):
    lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if lo is None or (isinstance(lo, float) and not np.isfinite(lo)):
        return _fmt(row[key], nd)
    return f"{_fmt(row[key], nd)} [{_fmt(lo, nd)}–{_fmt(hi, nd)}]"


def plot_single(results: pd.DataFrame, path: Path, title: str) -> None:
    """One protocol: models sorted by score, value printed beside each mark."""
    r = results.sort_values("headline_value")
    metric = HEADLINE_NAMES.get(r["headline_metric"].iloc[0], r["headline_metric"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(r) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    y = np.arange(len(r))
    ax.hlines(y, 0, r["headline_value"], color=GRID, linewidth=2, zorder=1)
    ax.scatter(r["headline_value"], y, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, (_, row) in zip(y, r.iterrows()):
        ax.text(row["headline_value"] + 0.015, yi, f"{row['headline_value']:.2f}  "
                f"({int(row['detected'])}/{int(row['events'])} events, {row['false_alarms_per_day']:.2f} FA/day)",
                va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, r["signal"], fontsize=9, color=INK)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel(f"{metric}, honest protocol (0–1)", color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_audit(results: pd.DataFrame, path: Path, title: str) -> None:
    protos = list(dict.fromkeys(results["protocol"]))
    if len(protos) == 1:
        plot_single(results, path, title)
        return
    signals = list(dict.fromkeys(results["signal"]))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(signals) + 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ypos = {s: i for i, s in enumerate(reversed(signals))}
    k = 0
    for p in protos:
        sub = results[results["protocol"] == p]
        label = sub["protocol_label"].iloc[0]
        metric = HEADLINE_NAMES.get(sub["headline_metric"].iloc[0], sub["headline_metric"].iloc[0])
        if bool(sub["honest"].iloc[0]):   # the number that matters: dark, larger, on top
            color, marker, size, z = INK, "^", 90, 4
        else:
            color, marker, size, z = SERIES[k % len(SERIES)], MARKERS[k % len(MARKERS)], 60, 3
            k += 1
        ax.scatter(sub["headline_value"], [ypos[s] for s in sub["signal"]], s=size,
                   color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=z,
                   label=f"{label} ({metric})")
    for s, yv in ypos.items():
        vals = results.loc[results["signal"] == s, "headline_value"]
        ax.plot([vals.min(), vals.max()], [yv, yv], color=GRID, linewidth=2, zorder=1)
    ax.set_yticks(list(ypos.values()), list(ypos.keys()), fontsize=9, color=INK)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("score each protocol would report (0–1)", color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1,
              frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def write_report(ds: Dataset, cfg: dict, results: pd.DataFrame, out_dir: Path) -> None:
    ev = cfg["evaluate"]
    protos = list(dict.fromkeys(results["protocol"]))
    lines = [f"# {cfg.get('title', 'Evaluation')} – {ds.name}", ""]

    honest = results[results["honest"]]
    leaky = results[~results["honest"]]
    if len(leaky) and len(honest):
        hp = honest["protocol"].iloc[0]
        lines += [
            "## How much each shortcut inflates the score",
            "",
            f"Honest = event F1 under \"{honest['protocol_label'].iloc[0]}\". Each other column is the "
            "score that protocol would report for the same signal on the same data.",
            "",
        ]
        others = [p for p in protos if p != hp]
        lines.append("| Signal | Honest event F1 | " + " | ".join(
            f"{p} ({HEADLINE_NAMES.get(results.loc[results['protocol'] == p, 'headline_metric'].iloc[0])})"
            for p in others) + " | Largest inflation |")
        lines.append("|---|---|" + "---|" * len(others) + "---|")
        for sig in dict.fromkeys(results["signal"]):
            h = honest[honest["signal"] == sig]
            if not len(h):
                continue
            hv = float(h["event_f1"].iloc[0])
            vals = []
            for p in others:
                v = results[(results["protocol"] == p) & (results["signal"] == sig)]["headline_value"]
                vals.append(float(v.iloc[0]) if len(v) else np.nan)
            best = np.nanmax(vals) if np.isfinite(vals).any() else np.nan
            ratio = f"{best / hv:.1f}×" if hv > 0 and np.isfinite(best) else ("∞" if np.isfinite(best) and best > 0 else "")
            lines.append(f"| {sig} | {_fmt(hv)} | " + " | ".join(_fmt(v) for v in vals) + f" | {ratio} |")
        lines.append("")

    lines += ["## Protocols", "", "| Protocol | Split | Threshold chosen on | Headline metric | Honest? |",
              "|---|---|---|---|---|"]
    for p in ev["protocols"]:
        sub = results[results["protocol"] == p["name"]]
        if not len(sub):
            continue
        sp = p["splitter"]
        lines.append(f"| {p.get('label', p['name'])} | `{sp['name']}` {json.dumps(sp.get('params') or {})} | "
                     f"{p.get('threshold', {}).get('choose_on', 'val')} | {HEADLINE_NAMES.get(p.get('headline', 'event_f1'))} | "
                     f"{'yes' if bool(sub['honest'].iloc[0]) else 'no'} |")

    lines += ["", "## Headline score by protocol", "",
              "Each cell is the metric that protocol would report.", ""]
    head = results.pivot(index="signal", columns="protocol", values="headline_value")[protos]
    lines.append("| Signal | " + " | ".join(protos) + " |")
    lines.append("|---|" + "---|" * len(protos))
    for s, r in head.iterrows():
        lines.append(f"| {s} | " + " | ".join(_fmt(v) for v in r) + " |")

    lines += ["", "![Headline scores](audit.png)", ""]

    for p in protos:
        sub = results[results["protocol"] == p]
        honest_p = bool(sub["honest"].iloc[0])
        lines += ["", f"## Operator view – {sub['protocol_label'].iloc[0]}", ""]
        if honest_p:
            lines.append("95% bootstrap intervals in brackets (events resampled for recall and delay, days for false alarms).")
            lines.append("")
        lines += ["| Signal | Events caught | Event recall | False alarms / day | False-alarm hours / day | "
                  "Median delay (min) | Event precision | Event F1 | Point F1 | Point-adjusted F1 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.sort_values("event_f1", ascending=False).iterrows():
            lines.append(
                f"| {r['signal']} | {int(r['detected'])}/{int(r['events'])} | {_ci(r, 'event_recall')} | "
                f"{_ci(r, 'false_alarms_per_day')} | {_fmt(r['false_alarm_hours_per_day'])} | "
                f"{_ci(r, 'median_delay_min', 0)} | {_fmt(r['event_precision'])} | "
                f"{_fmt(r['event_f1'])} | {_fmt(r['point_f1'])} | {_fmt(r['pa_f1'])} |")

    lines += ["", "## Definitions", "",
              "- **Event recall**: share of labelled test events with at least one alarm inside the event"
              + (f" (or up to {ev.get('tolerance_after', 0)} min after it)." if int(ev.get("tolerance_after", 0)) else "."),
              f"- **False alarms / day**: runs of alarm minutes outside every event (merged if at most "
              f"{ev.get('merge_gap', 5)} min apart), per day of normal data. An alarm left on between "
              "events counts as false.",
              f"  A false alarm left on counts again every {ev.get('max_episode_minutes', 60)} min.",
              "- **Event precision**: events caught ÷ (events caught + false alarms).",
              "- **Median delay**: minutes from event start to the first alarm, for caught events.",
              "- **Point F1**: every minute is a sample. **Point-adjusted F1**: an event counts as fully "
              "found if any one of its minutes is flagged (common in papers; inflates scores).",
              "- Minutes with no data are never scored and never count as normal time.",
              "", "Raw numbers: `results.csv`; per-event hits and delays: `per_event.csv`."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    title = cfg.get("title", "Evaluation")
    if results["honest"].any() and (~results["honest"]).any():
        title = f"{title} ({ds.name}): honest scores (black) sit well below what shortcuts report"
    else:
        best = results.loc[results["headline_value"].idxmax()]
        title = f"{title} ({ds.name}): best is {best['signal']} at {best['headline_value']:.2f}"
    plot_audit(results, out_dir / "audit.png", title)
