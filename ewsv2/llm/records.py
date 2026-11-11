"""Alarm records: the facts a report may use (Segment 7).

An *alarm record* is everything the operator should know about one alarm, as plain data:
decision (water quality alarm or sensor fault alert), alert level, start, end, duration,
and the sensors involved with their values before and at the alarm. Reports – template or
LLM – are written *from* the record, and the faithfulness checker compares every report
with it. The LLM never sees anything else, and it never changes the decision.

Built from the S5 pipeline output (``outputs_v2/s5_fusion/<dataset>/pipeline_minutes.csv.gz``):

* water alarm episodes (after the S4 triage) → ``kind = water``; the sensors are the top
  three of the S6 ``deviation`` explainer at the first alarm minute (the S6 result:
  as accurate as the expensive explainers, and needs no model);
* sensor alerts that do not overlap a water alarm → ``kind = sensor``; the sensors are the
  ones the alert names.

Alert level (paper 1's levels, applied to sensors instead of modules): one sensor beyond
its normal 15-minute range → LOW, two → MEDIUM, three or more → CRITICAL. Sensor alerts are
level MAINTENANCE.

``truth`` (what the episode really was) is kept for evaluation only and never shown to a
reporter.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ewsv2.core.types import EVENT_COL, Dataset
from ewsv2.data.gecco import SENSOR_INFO
from ewsv2.evaluate.metrics import _merge, segments
from ewsv2.explain.methods import ranking
from ewsv2.features.scale import reading_step  # noqa: F401  (fallback)
from ewsv2.inject.benchmark import test_mask

DECISION = {"water": "WATER QUALITY ALARM", "sensor": "SENSOR FAULT ALERT"}


def _decimals(step: float) -> int:
    if step <= 0:
        return 3
    return int(min(4, max(0, math.ceil(-math.log10(step) - 1e-9))))


def _resolution(x: pd.Series) -> float:
    """Typical reading step: 5th percentile of the non-zero minute-to-minute changes."""
    d = x.astype(float).diff().abs().to_numpy()
    d = d[np.isfinite(d) & (d > 1e-12)]
    return float(np.percentile(d, 5)) if len(d) else reading_step(x)


def _level(moved: int) -> str:
    return "CRITICAL" if moved >= 3 else ("MEDIUM" if moved == 2 else "LOW")


def build_records(ds: Dataset, inj: Dataset, episodes: pd.DataFrame, minutes: pd.DataFrame,
                  dev_explainer, station: str, merge_gap: int = 5, fault_grace: int = 10) -> list[dict]:
    """One record per alarm episode in the test period (see module docstring)."""
    f = inj.frame
    sensors = inj.sensors
    idx = f.index
    n = len(f)
    tst = test_mask(inj)
    water = minutes["water_alarm"].eq(True).to_numpy() & tst
    sensor = minutes["sensor_alert"].eq(True).to_numpy() & tst
    names = minutes["sensor_names"].fillna("").astype(str).to_numpy()
    pre = f.index < inj.periods["test"][0]
    decimals = {s: _decimals(_resolution(f[s][pre])) for s in sensors}

    # what each minute really was (evaluation only)
    truth = np.full(n, "", dtype=object)
    for s, e in segments(ds.frame[EVENT_COL].to_numpy().astype(bool)):
        truth[s:e + 1] = "real event"
    for _, r in episodes.iterrows():
        a, b = int(r["start_pos"]), int(r["end_pos"]) + (fault_grace if r["kind"] == "fault" else 0)
        truth[a:b + 1] = (f"injected {r['type']} contaminant" if r["kind"] == "contaminant"
                          else f"injected {r['type']} fault on {r['lead_sensor']}")

    def values(sensor_code: str, onset: int) -> dict:
        x = f[sensor_code].astype(float)
        before = x.iloc[max(0, onset - 35):max(1, onset - 5)].median()
        now = x.iloc[onset]
        d = decimals[sensor_code]
        rec = {"code": sensor_code, "description": SENSOR_INFO.get(sensor_code, sensor_code),
               "before": None if not np.isfinite(before) else round(float(before), d),
               "now": None if not np.isfinite(now) else round(float(now), d)}
        if rec["before"] is not None and rec["now"] is not None:
            rec["change"] = round(rec["now"] - rec["before"], d)
        return rec

    records = []
    water_eps = _merge(segments(water), merge_gap)
    for a, b in water_eps:
        dev = dev_explainer.explain(f, [a]).iloc[0]
        top = [s for s in ranking(dev)[:3] if dev[s] >= 0.5] or ranking(dev)[:1]   # sensors that moved
        moved = int((dev >= 1.0).sum())
        sens = []
        for s in top:
            v = values(s, a)
            v["normal_units"] = round(float(dev[s]), 1)
            sens.append(v)
        overlapping = [names[i] for i in range(a, b + 1) if names[i]]
        records.append({"kind": "water", "start_pos": int(a), "end_pos": int(b), "sensors": sens,
                        "level": _level(moved), "sensors_beyond_normal": moved,
                        "sensor_alerts_during": sorted({x for s in overlapping for x in s.split(",") if x}),
                        "truth": _truth_of(truth, a, b)})
    covered = np.zeros(n, dtype=bool)
    for a, b in water_eps:
        covered[a:b + 1] = True
    for a, b in _merge(segments(sensor & ~covered), merge_gap):
        flagged = sorted({x for i in range(a, b + 1) for x in names[i].split(",") if x})
        records.append({"kind": "sensor", "start_pos": int(a), "end_pos": int(b),
                        "sensors": [values(s, a) for s in flagged[:3]], "level": "MAINTENANCE",
                        "sensors_beyond_normal": None, "sensor_alerts_during": flagged,
                        "truth": _truth_of(truth, a, b)})
    records.sort(key=lambda r: r["start_pos"])
    for i, r in enumerate(records, 1):
        a, b = r["start_pos"], r["end_pos"]
        r.update({"id": f"A{i:03d}", "station": station, "decision": DECISION[r["kind"]],
                  "move_window_min": 15,
                  "start": idx[a].strftime("%Y-%m-%d %H:%M"), "end": idx[b].strftime("%Y-%m-%d %H:%M"),
                  "duration_min": int(b - a + 1)})
    return records


def _truth_of(truth: np.ndarray, a: int, b: int) -> str:
    vals = [t for t in truth[a:b + 1] if t]
    return max(set(vals), key=vals.count) if vals else "false alarm (nothing labelled)"


PUBLIC_FIELDS = ("id", "station", "decision", "level", "start", "end", "duration_min", "sensors",
                 "sensors_beyond_normal", "sensor_alerts_during", "move_window_min")


def public(record: dict) -> dict:
    """The part of a record a reporter may see (no truth, no positions)."""
    return {k: record[k] for k in PUBLIC_FIELDS if k in record and record[k] is not None}


def select(records: list[dict], max_reports: int) -> list[dict]:
    """A balanced, deterministic sample: real events, contaminants, faults, false alarms, sensor alerts."""
    if max_reports <= 0 or len(records) <= max_reports:
        return list(records)

    def group(r):
        t = r["truth"]
        if r["kind"] == "sensor":
            return "sensor alert"
        if t.startswith("real"):
            return "real event"
        if "contaminant" in t:
            return "contaminant"
        if "fault" in t:
            return "fault"
        return "false alarm"

    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(group(r), []).append(r)
    share = max(1, max_reports // len(groups))
    chosen = []
    for g in groups.values():
        k = min(share, len(g))
        pick = np.linspace(0, len(g) - 1, k).round().astype(int) if k else []
        chosen += [g[i] for i in sorted(set(pick))]
    chosen.sort(key=lambda r: r["start_pos"])
    return chosen[:max_reports]
