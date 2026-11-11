"""Data inspection report (Segment 0).

Answers, for any loaded Dataset, the questions we must settle before modelling:

1. How big is it, how is it sampled, where are the gaps?
2. What do the sensors look like (ranges, missing values, frozen stretches)?
3. How many labelled events are there, how long, how are they split?
4. Which sensors move during events?  (the basis of the water-vs-sensor triage)
5. How strongly are sensors correlated?  (the basis of the coherence check)

Writes ``report.md``, ``summary.json``, ``events.csv`` and charts to an output folder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ewsv2.core.types import EVENT_COL, Dataset  # noqa: E402

log = logging.getLogger(__name__)

# Chart colours (validated reference palette; text never takes series colours).
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LINE, EVENT_FILL, BAR = "#2a78d6", "#eb6834", "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", ["#2a78d6", "#f1f0ec", "#eb6834"])


# ----------------------------------------------------------------------------
# Computations
# ----------------------------------------------------------------------------

def runs(mask: pd.Series) -> pd.DataFrame:
    """Contiguous runs of True rows: start, end (timestamps) and length (rows)."""
    m = mask.fillna(False).astype(bool).to_numpy()
    if not m.any():
        return pd.DataFrame(columns=["start", "end", "length"])
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2] - 1
    idx = mask.index
    return pd.DataFrame({"start": idx[starts], "end": idx[ends], "length": ends - starts + 1})


def event_table(ds: Dataset) -> pd.DataFrame:
    """One row per labelled event, with the official period it falls in."""
    ev = runs(ds.frame[EVENT_COL] == 1)
    if ev.empty:
        return ev.assign(period=pd.Series(dtype=str))

    def which(ts: pd.Timestamp) -> str:
        for name, (a, b) in ds.periods.items():
            if a <= ts <= b:
                return name
        return "outside"

    ev["period"] = ev["start"].map(which)
    ev.insert(0, "event_id", range(1, len(ev) + 1))
    return ev


def sampling_summary(frame: pd.DataFrame, sensors: list[str]) -> dict:
    diffs = frame.index.to_series().diff().dropna()
    step = diffs.mode().iloc[0] if len(diffs) else pd.Timedelta(0)
    all_nan = frame[sensors].isna().all(axis=1)
    nan_runs = runs(all_nan)
    expected = int((frame.index.max() - frame.index.min()) / step) + 1 if step > pd.Timedelta(0) else len(frame)
    return {
        "rows": int(len(frame)),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "days": round((frame.index.max() - frame.index.min()).total_seconds() / 86400, 1),
        "step": str(step),
        "irregular_steps": int((diffs != step).sum()),
        "missing_timestamps": max(expected - len(frame), 0),
        "rows_all_sensors_missing": int(all_nan.sum()),
        "longest_all_missing_run_minutes": int(nan_runs["length"].max()) if len(nan_runs) else 0,
        "longest_all_missing_run_start": str(nan_runs.loc[nan_runs["length"].idxmax(), "start"]) if len(nan_runs) else None,
    }


def longest_constant_run(s: pd.Series) -> int:
    """Longest stretch of identical consecutive non-missing values (rows)."""
    v = s.dropna()
    if v.empty:
        return 0
    change = v.ne(v.shift()).cumsum()
    return int(change.value_counts().max())


def sensor_table(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    rows = []
    for s in sensors:
        x = frame[s]
        rows.append({
            "sensor": s,
            "missing_pct": round(100 * x.isna().mean(), 2),
            "mean": x.mean(), "std": x.std(),
            "min": x.min(), "p01": x.quantile(0.01), "median": x.median(),
            "p99": x.quantile(0.99), "max": x.max(),
            "distinct_values": int(x.nunique()),
            "zero_readings": int((x == 0).sum()),
            "longest_constant_run_min": longest_constant_run(x),
        })
    return pd.DataFrame(rows).set_index("sensor")


def event_response(frame: pd.DataFrame, sensors: list[str], events: pd.DataFrame,
                   baseline_minutes: int = 60, shift_threshold: float = 1.0) -> pd.DataFrame:
    """How much each sensor moves during events, in units of its normal spread.

    For each event: |mean during event − mean of the ``baseline_minutes`` before|,
    divided by the sensor's standard deviation on normal (non-event) rows.
    Reported per sensor: median shift across events, and the share of events
    where the shift exceeds ``shift_threshold`` standard deviations.
    """
    normal = frame.loc[frame[EVENT_COL] == 0, sensors]
    spread = normal.std().replace(0, np.nan)
    shifts = []
    for _, e in events.iterrows():
        during = frame.loc[e["start"]:e["end"], sensors].mean()
        before = frame.loc[e["start"] - pd.Timedelta(minutes=baseline_minutes):
                           e["start"] - pd.Timedelta(minutes=1), sensors].mean()
        shifts.append(((during - before).abs() / spread).rename(e["event_id"]))
    if not shifts:
        return pd.DataFrame(columns=["median_shift_sd", "share_events_shift_gt_threshold", "events_with_data"])
    table = pd.DataFrame(shifts)
    return pd.DataFrame({
        "median_shift_sd": table.median(),
        "share_events_shift_gt_threshold": (table > shift_threshold).mean(),
        "events_with_data": table.notna().sum(),
    }).sort_values("median_shift_sd", ascending=False)


def correlations(frame: pd.DataFrame, sensors: list[str], resample: str = "10min") -> pd.DataFrame:
    """Spearman correlation between sensors on normal rows (10-minute means for speed)."""
    normal = frame.loc[frame[EVENT_COL] == 0, sensors]
    return normal.resample(resample).mean().corr(method="spearman")


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_overview(ds: Dataset, events: pd.DataFrame, path: Path, resample: str = "30min") -> None:
    sensors = ds.sensors
    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 1.25 * len(sensors) + 1),
                             sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    smooth = ds.frame[sensors].resample(resample).mean()
    for ax, s in zip(axes, sensors):
        _style(ax)
        ax.plot(smooth.index, smooth[s], color=LINE, linewidth=0.9)
        for _, e in events.iterrows():
            ax.axvspan(e["start"], e["end"], color=EVENT_FILL, alpha=0.35, linewidth=0)
        for name, (a, _) in list(ds.periods.items())[1:]:
            ax.axvline(a, color=INK_2, linewidth=0.8, linestyle="--")
        ax.set_ylabel(s, rotation=0, ha="right", va="center", color=INK, fontsize=9)
    first = list(ds.periods.items())
    title = f"{ds.name}: {len(events)} labelled events (orange) across {len(sensors)} sensors"
    if len(first) > 1:
        title += "; dashed line = start of " + ", ".join(n for n, _ in first[1:])
    axes[0].set_title(title, loc="left", color=INK, fontsize=11)
    axes[-1].set_xlabel(f"time ({resample} means)", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_event_lengths(events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    _style(ax)
    lengths = events["length"].clip(lower=1)
    bins = np.logspace(0, np.log10(max(lengths.max(), 10)) + 0.05, 20)
    ax.hist(lengths, bins=bins, color=BAR, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xscale("log")
    med = lengths.median()
    ax.set_title(f"Event length: median {med:.0f} min, longest {lengths.max():.0f} min "
                 f"({len(events)} events)", loc="left", color=INK, fontsize=10)
    ax.set_xlabel("event length (minutes, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("events", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_event_response(resp: pd.DataFrame, path: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    _style(ax)
    r = resp.sort_values("share_events_shift_gt_threshold")
    ax.barh(r.index, 100 * r["share_events_shift_gt_threshold"], color=BAR, height=0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(100 * r["share_events_shift_gt_threshold"]):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", color=INK_2, fontsize=8)
    top = r.index[-1] if len(r) else "-"
    ax.set_title(f"Share of events where each sensor shifts by >{threshold:g} SD (top: {top})",
                 loc="left", color=INK, fontsize=10)
    ax.set_xlabel("% of events", color=INK_2, fontsize=9)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_correlations(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8, color=INK_2)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Sensor correlation on normal data (Spearman)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _fmt_step(step: str) -> str:
    td = pd.Timedelta(step)
    minutes = td.total_seconds() / 60
    return f"{minutes:g} min" if minutes >= 1 else f"{td.total_seconds():g} s"


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inspect_dataset(ds: Dataset, out_dir: Path, baseline_minutes: int = 60,
                    shift_threshold: float = 1.0, make_plots: bool = True) -> dict:
    """Compute everything, write the report files, return the summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    frame, sensors = ds.frame, ds.sensors

    events = event_table(ds)
    sampling = sampling_summary(frame, sensors)
    sensors_df = sensor_table(frame, sensors)
    response = event_response(frame, sensors, events, baseline_minutes, shift_threshold)
    corr = correlations(frame, sensors)

    per_period = {}
    for name, (a, b) in ds.periods.items():
        part = frame.loc[a:b]
        ev = events[events["period"] == name]
        per_period[name] = {
            "start": str(a), "end": str(b), "rows": int(len(part)),
            "event_minutes": int(part[EVENT_COL].sum()),
            "event_minute_pct": round(100 * part[EVENT_COL].mean(), 3) if len(part) else 0.0,
            "events": int(len(ev)),
            "median_event_length_min": float(ev["length"].median()) if len(ev) else None,
            "rows_all_sensors_missing": int(part[sensors].isna().all(axis=1).sum()),
        }

    summary = {
        "dataset": ds.name,
        "sensors": sensors,
        "sensor_info": ds.meta.get("sensor_info", {}),
        "files": ds.meta.get("files", []),
        "licence": ds.meta.get("licence"),
        "load_notes": ds.meta.get("load_notes", {}),
        "sampling": sampling,
        "periods": per_period,
        "events_total": int(len(events)),
        "event_minutes_total": int(frame[EVENT_COL].sum()),
        "event_length_min": {
            "median": float(events["length"].median()) if len(events) else None,
            "max": int(events["length"].max()) if len(events) else None,
            "min": int(events["length"].min()) if len(events) else None,
        },
        "event_response_top_sensors": response.head(3).index.tolist(),
    }

    events.to_csv(out_dir / "events.csv", index=False)
    sensors_df.to_csv(out_dir / "sensors.csv")
    response.to_csv(out_dir / "event_response.csv")
    corr.to_csv(out_dir / "correlations.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    figures = {}
    if make_plots:
        figures = {
            "overview": "figures/overview.png",
            "event_lengths": "figures/event_lengths.png",
            "event_response": "figures/event_response.png",
            "correlations": "figures/correlations.png",
        }
        plot_overview(ds, events, out_dir / figures["overview"])
        if len(events):
            plot_event_lengths(events, out_dir / figures["event_lengths"])
            plot_event_response(response, out_dir / figures["event_response"], shift_threshold)
        else:
            figures.pop("event_lengths")
            figures.pop("event_response")
        plot_correlations(corr, out_dir / figures["correlations"])

    _write_markdown(ds, summary, sensors_df, response, events, figures, out_dir, shift_threshold, baseline_minutes)
    return summary


def _write_markdown(ds, summary, sensors_df, response, events, figures, out_dir,
                    shift_threshold, baseline_minutes) -> None:
    s = summary["sampling"]
    lines = [
        f"# Data inspection – {ds.name}",
        "",
        f"{s['rows']:,} rows from {s['start']} to {s['end']} ({s['days']} days), "
        f"one reading every {_fmt_step(s['step'])}. {summary['events_total']} labelled events "
        f"covering {summary['event_minutes_total']:,} minutes.",
        "",
        "## Periods",
        "",
        "| Period | From | To | Rows | Events | Event minutes | % of minutes | Rows with all sensors missing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, p in summary["periods"].items():
        lines.append(f"| {name} | {p['start']} | {p['end']} | {p['rows']:,} | {p['events']} | "
                     f"{p['event_minutes']:,} | {p['event_minute_pct']} | {p['rows_all_sensors_missing']:,} |")
    lines += [
        "",
        "## Sampling and gaps",
        "",
        f"- Irregular time steps: {s['irregular_steps']:,}; missing timestamps: {s['missing_timestamps']:,}.",
        f"- Rows where every sensor is empty: {s['rows_all_sensors_missing']:,}; longest such stretch: "
        f"{s['longest_all_missing_run_minutes']:,} minutes, starting {s['longest_all_missing_run_start']}.",
        "",
        "## Sensors",
        "",
        _md_table(sensors_df.drop(columns=["p01", "p99"])),
        "",
        "`zero_readings` = readings of exactly 0; for temperature, conductivity or flow these are "
        "almost certainly sensor or telemetry faults, not water. `longest_constant_run_min` = longest "
        "stretch of identical readings; long runs on normally varying sensors are candidates for frozen "
        "sensors. Both feed the fault catalogue in Segment 4.",
        "",
        "## Events",
        "",
    ]
    el = summary["event_length_min"]
    if summary["events_total"]:
        lines.append(f"Event length: median {el['median']:.0f} min, shortest {el['min']} min, longest {el['max']} min. "
                     "Full list in `events.csv`.")
        if "event_lengths" in figures:
            lines += ["", f"![Event lengths]({figures['event_lengths']})"]
        lines += [
            "",
            "## Which sensors react during events",
            "",
            f"Shift = |mean during event − mean of the {baseline_minutes} minutes before| ÷ the sensor's "
            f"normal standard deviation. A real water change should move several coupled sensors together; "
            f"this table is the starting point for the coherence check in Segment 4.",
            "",
            _md_table(response),
        ]
        if "event_response" in figures:
            lines += ["", f"![Event response]({figures['event_response']})"]
    else:
        lines.append("No labelled events in this dataset.")
    lines += ["", "## Overview", "", f"![Overview]({figures.get('overview', '')})" if figures else "",
              "", "## Sensor correlation", ""]
    if "correlations" in figures:
        lines += [f"![Correlations]({figures['correlations']})", ""]
    lines += ["Values in `correlations.csv`.", "", "## Source", ""]
    for f in summary["files"]:
        lines.append(f"- `{f}`")
    if summary.get("licence"):
        lines.append(f"- Licence: {summary['licence']}")
    if summary.get("load_notes"):
        lines.append(f"- Load notes: `{json.dumps(summary['load_notes'])}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
