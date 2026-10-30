"""Scoring the station outputs on GECCO-Triage (Segment 4).

Every test-period episode gets one outcome per system:

Water episodes (real GECCO events and injected contaminants), window = the labelled event:
    ``water alarm``        a water alarm inside the event (caught)
    ``sensor alert only``  only a sensor alert – the operator is sent to the sensor (dangerous)
    ``missed``             nothing

Fault episodes, window = the fault plus ``grace`` minutes (a spike is provable only once it
comes back):
    ``sensor alert only``  correct: "check the sensor", and no water alarm
    ``both``               a sensor alert and a water alarm
    ``water alarm only``   a false water alarm caused by a faulty sensor
    ``missed``             nothing

Rates over the test period use the S1 event evaluator: water alarms are scored against
water labels (so a water alarm during a sensor fault is a false alarm), sensor alerts
against fault labels. A false sensor alert is counted once however long it stays on (it is
a maintenance ticket), while a false water alarm is re-counted every hour as in S1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.evaluate.metrics import EventEvaluator, segments

WATER_OUTCOMES = ("water alarm", "sensor alert only", "missed")
FAULT_OUTCOMES = ("sensor alert only", "both", "water alarm only", "missed")


def test_episodes(original_event: np.ndarray, index: pd.DatetimeIndex, test: np.ndarray,
                  has_data: np.ndarray, episodes: pd.DataFrame) -> pd.DataFrame:
    """Real test events plus injected test episodes, one row each, sorted by time."""
    rows = []
    for s, e in segments(np.asarray(original_event, dtype=bool) & test):
        if has_data[s:e + 1].any():
            rows.append({"source": "real event", "kind": "water", "type": "real", "level": "",
                         "lead_sensor": "", "start_pos": s, "end_pos": e})
    for _, r in episodes[episodes["period"] == "test"].iterrows():
        rows.append({"source": "injected contaminant" if r["kind"] == "contaminant" else "injected fault",
                     "kind": "water" if r["kind"] == "contaminant" else "fault",
                     "type": r["type"], "level": r["level"], "lead_sensor": r["lead_sensor"],
                     "start_pos": int(r["start_pos"]), "end_pos": int(r["end_pos"])})
    out = pd.DataFrame(rows).sort_values("start_pos").reset_index(drop=True)
    out["start"] = index[out["start_pos"]]
    out["minutes"] = out["end_pos"] - out["start_pos"] + 1
    return out


def episode_outcomes(eps: pd.DataFrame, water: np.ndarray, sensor: np.ndarray,
                     sensor_names: pd.Series | None, grace: int = 10, tolerance_after: int = 0) -> pd.DataFrame:
    n = len(water)
    names = sensor_names.to_numpy() if sensor_names is not None else None
    rows = []
    for _, e in eps.iterrows():
        s = int(e["start_pos"])
        end = int(e["end_pos"]) + (grace if e["kind"] == "fault" else tolerance_after)
        end = min(end, n - 1)
        w = np.flatnonzero(water[s:end + 1])
        a = np.flatnonzero(sensor[s:end + 1])
        if e["kind"] == "water":
            outcome = "water alarm" if len(w) else ("sensor alert only" if len(a) else "missed")
            delay = float(w[0]) if len(w) else np.nan
        else:
            outcome = ("both" if len(w) else "sensor alert only") if len(a) else \
                ("water alarm only" if len(w) else "missed")
            delay = float(a[0]) if len(a) else np.nan
        right_sensor = np.nan
        if e["kind"] == "fault" and len(a) and names is not None:
            named = ",".join(str(x) for x in names[s + a])
            right_sensor = e["lead_sensor"] in named.split(",")
        rows.append({"outcome": outcome, "delay_min": delay, "names_right_sensor": right_sensor})
    return pd.concat([eps.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def rates(cls: np.ndarray, water: np.ndarray, sensor: np.ndarray, index, valid: np.ndarray,
          grace: int, tol: int, gap: int, max_ep: int) -> dict:
    from ewsv2.triage.systems import _with_grace
    w = EventEvaluator((cls == 1).astype(int), index, valid, tol, gap, max_ep).evaluate(water)
    # A sensor alert is a maintenance ticket: it stays open until the sensor is fixed and is
    # counted once, however long it lasts (no re-count every hour as for water alarms).
    f = EventEvaluator(_with_grace((cls == 2).astype(int), grace), index, valid, 0, gap, 10**9).evaluate(sensor)
    return {
        "water_event_f1": w.event_f1, "water_recall": w.event_recall,
        "water_false_alarms_per_day": w.false_alarms_per_day, "water_median_delay": w.median_delay_min,
        "fault_recall": f.event_recall, "false_sensor_alerts_per_day": f.false_alarms_per_day,
        "fault_median_delay": f.median_delay_min,
    }


def summarise(outcomes: pd.DataFrame) -> dict:
    o = outcomes
    real = o[o["source"] == "real event"]
    inj = o[o["source"] == "injected contaminant"]
    flt = o[o["source"] == "injected fault"]
    share = lambda df, k: float((df["outcome"] == k).mean()) if len(df) else np.nan  # noqa: E731
    return {
        "real_caught": int((real["outcome"] == "water alarm").sum()), "real_n": len(real),
        "real_sent_to_sensor": int((real["outcome"] == "sensor alert only").sum()),
        "contaminant_caught": int((inj["outcome"] == "water alarm").sum()), "contaminant_n": len(inj),
        "contaminant_sent_to_sensor": int((inj["outcome"] == "sensor alert only").sum()),
        "fault_correct": int((flt["outcome"] == "sensor alert only").sum()), "fault_n": len(flt),
        "fault_both": int((flt["outcome"] == "both").sum()),
        "fault_false_water": int((flt["outcome"] == "water alarm only").sum()),
        "fault_missed": int((flt["outcome"] == "missed").sum()),
        "fault_correct_share": share(flt, "sensor alert only"),
        "fault_raised_water_alarm": int(flt["outcome"].isin(["both", "water alarm only"]).sum()),
        "fault_named_right_sensor": float(flt["names_right_sensor"].dropna().astype(float).mean())
        if flt["names_right_sensor"].notna().any() else np.nan,
    }
