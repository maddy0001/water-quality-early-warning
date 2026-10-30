"""Per-minute evidence for the water-vs-sensor decision (Segment 4).

Two kinds of evidence, both causal and both in normal-variability units:

* **water evidence** – how far each sensor moved in the last 15 minutes (``z15``: value
  minus the mean of the previous 15 minutes), how many sensors moved together, and
  whether a sensor that is *not* flagged moved more than it ever does in normal water
  (``max_move_unflagged`` >= 1, limit = 99.9th percentile of its normal 15-minute moves);
* **sensor evidence** – the four fault checks of ``signals/faults.py`` per sensor, and
  the size of the last one-minute step (``jump``), since a slug of water changes sensors
  over minutes while a faulty sensor can jump in one.

``TriageFeatures.fit`` learns only from normal minutes (no fault labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.signals.faults import CHECKS, FaultChecks


class TriageFeatures:
    def __init__(self, sensors: list[str], fault_params: dict | None = None, z_move: float = 3.0,
                 hold_minutes: int = 60):
        self.sensors = list(sensors)
        self.fault_params = dict(fault_params or {})
        self.z_move = float(z_move)
        self.hold = int(hold_minutes)

    def _z15(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.sensors].astype(float).ffill(limit=5)
        z = (x - x.shift(1).rolling(15, min_periods=5).mean()) / self.units[self.sensors]
        return z.fillna(0.0).clip(-200, 200)

    def fit(self, frame: pd.DataFrame, normal_rows) -> TriageFeatures:
        normal_rows = np.asarray(normal_rows, dtype=bool)
        self.checks = FaultChecks(self.sensors, **self.fault_params).fit(frame, normal_rows)
        self.units = self.checks.units
        # how far each sensor moves in 15 minutes in normal water (99.9th percentile, at least z_move)
        q = self._z15(frame)[normal_rows].abs().quantile(0.999)
        self.move_limit = q.clip(lower=self.z_move)
        return self

    def compute(self, frame: pd.DataFrame) -> dict:
        """Everything the triage systems need, aligned with ``frame``.

        Returns a dict with
            evidence : {check: DataFrame sensor -> evidence (>= 1 fires)}
            flags    : {check: bool DataFrame} plus 'any'
            z15      : DataFrame of signed 15-min moves (units)
            jump     : DataFrame of |1-min step| (units)
            table    : one row per minute of summary features (for the learned triage)
        """
        x = frame[self.sensors].astype(float).ffill(limit=5)
        u = self.units[self.sensors]
        z15 = self._z15(frame)
        jump = (x.diff().abs() / u).fillna(0.0).clip(0, 200)
        ev = self.checks.evidence(frame)
        flags = {k: v >= 1.0 for k, v in ev.items()}
        now = flags["stuck"] | flags["dropout"] | flags["spike"] | flags["drift"]
        # A flagged sensor stays under suspicion for ``hold`` minutes after its flag clears
        # (a sensor alert is a ticket; the jump back after a dropout or a drift reset is
        # still the sensor, not the water).
        anyf = now.astype(float).rolling(self.hold + 1, min_periods=1).max().astype(bool) if self.hold else now
        flags["now"] = now
        flags["any"] = anyf

        a = z15.abs().to_numpy()
        top2 = -np.sort(-a, axis=1)[:, :2] if a.shape[1] >= 2 else np.column_stack([a[:, 0], np.zeros(len(a))])
        moved = a / self.move_limit[self.sensors].to_numpy()             # >= 1: moved more than in normal water
        clean = np.where(anyf.to_numpy(), 0.0, moved)                   # ... among sensors not flagged
        lead = a.argmax(axis=1)
        table = {
            "max_z15": top2[:, 0],
            "second_z15": top2[:, 1],
            "breadth": (a >= self.z_move).sum(axis=1),
            "isolation": top2[:, 0] / (top2[:, 1] + 1.0),
            "max_move_unflagged": clean.max(axis=1),
            "lead_jump": jump.to_numpy()[np.arange(len(a)), lead],
            "lead_jump_share": jump.to_numpy()[np.arange(len(a)), lead] / (top2[:, 0] + 1.0),
            "n_flagged": anyf.sum(axis=1).to_numpy(),
        }
        for k in CHECKS:
            table[f"max_{k}"] = ev[k].max(axis=1).to_numpy()
        for s in self.sensors:
            table[f"{s}__z15"] = z15[s].to_numpy()
            table[f"{s}__jump"] = jump[s].to_numpy()
            for k in CHECKS:
                table[f"{s}__{k}"] = ev[k][s].to_numpy()
        tab = pd.DataFrame(table, index=frame.index)
        return {"evidence": ev, "flags": flags, "z15": z15, "jump": jump, "table": tab}
