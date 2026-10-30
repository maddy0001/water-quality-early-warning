"""GECCO-Triage: real utility data with injected contaminants and sensor faults (Segment 4).

Contribution C3. The real GECCO stream is kept as it is (real events stay labelled as
water events); on top of it we inject, at random but reproducible times,

* contaminant slugs (``signatures.py``) – labelled water events (LABEL_CLASS 1, EVENT 1)
* single-sensor faults (``faults.py``)   – labelled sensor faults (LABEL_CLASS 2, EVENT 0)

in two separate sets: one inside the pre-test period (may be used for training and for
choosing thresholds) and one inside the official test period (used only for the final
numbers), with different random draws. Every episode keeps a ``buffer_minutes`` distance
from real events and from other episodes, so outcomes can be attributed cleanly, and is
placed only where at least ``min_data`` of its minutes have data.

Output: the injected :class:`Dataset` (same sensors, same periods) and the episode table
(one row per injected episode), which the triage evaluation reads.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ewsv2.core.registry import register
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset
from ewsv2.evaluate.metrics import segments
from ewsv2.features.scale import sensor_units
from ewsv2.inject import faults as F
from ewsv2.inject import signatures as S

log = logging.getLogger(__name__)


def pretest_mask(ds: Dataset) -> np.ndarray:
    """Minutes before the official test period."""
    start = ds.periods["test"][0]
    return np.asarray(ds.frame.index < start)


def test_mask(ds: Dataset) -> np.ndarray:
    a, b = ds.periods["test"]
    idx = ds.frame.index
    return np.asarray((idx >= a) & (idx <= b))


@register("injector", "gecco_triage")
class TriageBenchmark:
    """Builds the GECCO-Triage version of a dataset.

    Parameters (config ``inject:`` section)
    ---------------------------------------
    contaminants, doses       which signatures and dose levels (default: all)
    faults, severities        which fault types and severities (default: all)
    reps_test, reps_pretest   copies of each (type, level) in each period
    buffer_minutes            minimum distance to real events and other episodes
    min_data                  share of an episode's minutes that must have data
    seed                      the test and pre-test sets use seed and seed + 1
    roles                     extra sensor -> role mapping (see signatures.ROLES)
    """

    def __init__(self, contaminants=None, doses=None, faults=None, severities=None,
                 reps_test: int = 2, reps_pretest: int = 2, buffer_minutes: int = 240,
                 min_data: float = 0.9, seed: int = 2026, roles: dict | None = None):
        self.contaminants = list(contaminants or S.CONTAMINANTS)
        self.doses = list(doses or S.DOSES)
        self.faults = list(faults or F.FAULT_TYPES)
        self.severities = list(severities or ("low", "medium", "high"))
        self.reps = {"test": int(reps_test), "pretest": int(reps_pretest)}
        self.buffer = int(buffer_minutes)
        self.min_data = float(min_data)
        self.seed = int(seed)
        self.roles = {**S.ROLES, **(roles or {})}
        for c in self.contaminants:
            if c not in S.CONTAMINANTS:
                raise ValueError(f"unknown contaminant {c!r}; known: {list(S.CONTAMINANTS)}")
        for f in self.faults:
            if f not in F.FAULT_TYPES:
                raise ValueError(f"unknown fault type {f!r}; known: {list(F.FAULT_TYPES)}")

    # ------------------------------------------------------------------------
    def plan(self, ds: Dataset) -> list[dict]:
        """Which episodes to inject, before placing them."""
        plan = []
        for period in ("test", "pretest"):
            for _ in range(self.reps[period]):
                for c in self.contaminants:
                    resp = S.station_response(c, ds.sensors, self.roles)
                    if not resp:
                        continue
                    for d in self.doses:
                        plan.append({"period": period, "kind": "contaminant", "type": c, "level": d})
                for f in self.faults:
                    for sev in self.severities:
                        plan.append({"period": period, "kind": "fault", "type": f, "level": sev})
        return plan

    def unobservable(self, ds: Dataset) -> list[str]:
        return [c for c in self.contaminants if not S.station_response(c, ds.sensors, self.roles)]

    def build(self, ds: Dataset) -> tuple[Dataset, pd.DataFrame]:
        frame = ds.frame.copy()
        n = len(frame)
        sensors = list(ds.sensors)
        has_data = ~frame[sensors].isna().all(axis=1).to_numpy()
        y_real = frame[EVENT_COL].to_numpy().astype(bool)
        pre, tst = pretest_mask(ds), test_mask(ds)
        units = sensor_units(frame, sensors, pre & ~y_real & has_data)
        fault_sensors = [s for s in sensors if self.roles.get(s) in S.QUALITY_ROLES]
        if not fault_sensors:
            raise ValueError("no quality sensors to inject faults into; check the roles mapping")

        # minutes where an episode may not sit: near real events, and later near other episodes
        blocked = np.zeros(n, dtype=bool)
        for s, e in segments(y_real):
            blocked[max(0, s - self.buffer): min(n, e + self.buffer + 1)] = True
        data_cum = np.concatenate([[0], np.cumsum(has_data)])
        blocked_cum = None

        cls = frame[CLASS_COL].to_numpy().astype(int).copy()
        ev = frame[EVENT_COL].to_numpy().astype(int).copy()
        values = {s: frame[s].to_numpy(dtype=float).copy() for s in sensors}
        original = {s: frame[s].to_numpy(dtype=float) for s in sensors}

        rows = []
        plan = self.plan(ds)
        rngs = {"test": np.random.default_rng(self.seed), "pretest": np.random.default_rng(self.seed + 1)}
        # draw the concrete episode (length, shape, sensor) first, then place the longest first
        drawn = []
        for item in plan:
            rng = rngs[item["period"]]
            if item["kind"] == "contaminant":
                resp = S.station_response(item["type"], sensors, self.roles)
                rise = int(rng.integers(4, 11))
                shape = S.slug_shape(rise, int(rng.integers(10, 61)), int(rng.integers(rise, 3 * rise + 1)))
                lags = {s: int(rng.integers(0, 4)) for s in resp}
                jitter = {s: float(rng.uniform(0.75, 1.25)) for s in resp}
                length = len(shape) + max(lags.values())
                drawn.append({**item, "length": length, "resp": resp, "shape": shape, "lags": lags,
                              "jitter": jitter, "rng": rng})
            else:
                sensor = str(rng.choice(fault_sensors))
                length = F.fault_length(item["type"], item["level"], rng)
                drawn.append({**item, "length": length, "sensor": sensor, "rng": rng})
        order = sorted(range(len(drawn)), key=lambda i: -drawn[i]["length"])

        placed = 0
        for i in order:
            d = drawn[i]
            rng, length = d["rng"], d["length"]
            allowed = (tst if d["period"] == "test" else pre)
            cand = np.flatnonzero(allowed[: n - length])
            start = None
            blocked_cum = np.concatenate([[0], np.cumsum(blocked)])
            for _ in range(4000):
                s0 = int(rng.choice(cand))
                e0 = s0 + length
                if not allowed[e0 - 1]:
                    continue
                if blocked_cum[e0] - blocked_cum[s0] > 0:
                    continue
                if (data_cum[e0] - data_cum[s0]) < self.min_data * length:
                    continue
                start = s0
                break
            if start is None:
                log.warning("Could not place %s %s/%s in the %s period (no free stretch of %d min); skipped",
                            d["kind"], d["type"], d["level"], d["period"], length)
                continue
            end = start + length - 1
            blocked[max(0, start - self.buffer): min(n, end + self.buffer + 1)] = True
            placed += 1

            if d["kind"] == "contaminant":
                dose = S.DOSES[d["level"]]
                changed = []
                for s, r in d["resp"].items():
                    lag = d["lags"][s]
                    delta = np.zeros(length)
                    delta[lag: lag + len(d["shape"])] = d["shape"]
                    seg = slice(start, start + length)
                    new = values[s][seg] + r * d["jitter"][s] * dose * units[s] * delta
                    if self.roles.get(s) in S.NON_NEGATIVE:
                        new = np.maximum(new, 0.0)
                    values[s][seg] = np.where(np.isfinite(original[s][seg]), new, np.nan)
                    if abs(r) >= S.RESPONDING_LEVEL:
                        changed.append(s)
                ev[start:end + 1] = 1
                cls[start:end + 1] = 1
                lead = max(d["resp"], key=lambda s: abs(d["resp"][s]))
                info = {"dose_units": dose,
                        "response": {s: round(r * d["jitter"][s], 3) for s, r in d["resp"].items()}}
                sensors_hit = changed
            else:
                lead = d["sensor"]
                new, info = F.apply_fault(values[lead], start, length, d["type"], d["level"], units[lead], rng)
                if "length" in info:          # spikes: the episode ends at the last spike
                    end = start + info.pop("length") - 1
                values[lead] = new
                cls[start:end + 1] = 2
                sensors_hit = [lead]
            rows.append({
                "period": d["period"], "kind": d["kind"], "type": d["type"], "level": d["level"],
                "lead_sensor": lead, "sensors": ",".join(sensors_hit),
                "start": frame.index[start], "end": frame.index[end], "minutes": end - start + 1,
                "start_pos": start, "end_pos": end, "info": json.dumps(info),
            })

        for s in sensors:
            frame[s] = values[s]
        frame[EVENT_COL] = ev
        frame[CLASS_COL] = cls
        episodes = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
        episodes.insert(0, "episode", np.arange(1, len(episodes) + 1))
        log.info("GECCO-Triage on %s: placed %d of %d episodes (%s)", ds.name, placed, len(plan),
                 episodes.groupby(["period", "kind"]).size().to_dict())
        meta = {**ds.meta, "injected": True, "units": units.to_dict(),
                "unobservable_contaminants": self.unobservable(ds)}
        out = Dataset(name=f"{ds.name}_triage", frame=frame, sensors=sensors, periods=dict(ds.periods), meta=meta)
        return out, episodes
