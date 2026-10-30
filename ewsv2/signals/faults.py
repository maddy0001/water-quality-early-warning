"""Sensor fault checks (Segment 4): is one sensor misbehaving?

Four checks, one per fault type, each run on every sensor, each causal (a minute is
flagged using only readings up to that minute). All limits come from *normal* minutes
of the fit period – no fault labels are needed, so the checks work on a new station
from day one.

One parameter sets every data-driven limit: the station's **false-flag budget**
``flag_rate`` (sensor flags per day on normal water; default 0.5). It is shared equally
by the three calibrated checks (stuck, spike, drift) and the sensors, and each limit is
the smallest value that its check reaches no more than its share of the budget in the
normal fit data, never below a physical floor (``min_stuck``, ``spike_units``,
``drift_z``). Whether the budget still holds in a new season is measured, not assumed.

``stuck``    the reading has not changed for longer than its limit **and** the partner
             sensors say it should have moved: over the run, the partner model (see
             ``drift``) predicts a change of at least ``stuck_partner_units``. A coarse
             sensor in very stable water (temperature in winter, 0.1 °C steps) can
             legitimately stay constant for days; without evidence from its partners we
             cannot call that a fault.
``dropout``  the reading fell in one step (at least ``dropout_units`` and at least half its
             previous value) to (near) zero: below
             min(lowest normal reading − ``dropout_units``, half the lowest normal
             reading), or to exactly 0 for a sensor that is normally positive
``spike``    a one- or two-minute excursion from the median of the previous 15 minutes
             that comes straight back (flagged on return), larger than ``spike_units``
             and above its budget limit (turbidity has natural one-minute spikes from
             particles, so its limit is much higher)
``drift``    the sensor's 4-hour change disagrees with what its partner sensors predict:
             a ridge regression on normal data predicts each sensor's 4-hour change from
             the others' (like the 15-minute coherence model of S2); a robust z-score of
             the residual above its limit (at least ``drift_z``) for ``drift_persist``
             minutes in a row

Sizes are in each sensor's normal-variability unit (``features/scale.py``).

The registered signal ``fault_checks`` turns the flags into a score (largest evidence
over sensors and checks), so the checks can also be evaluated like any other signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.core.registry import register
from ewsv2.evaluate.metrics import segments
from ewsv2.features.scale import sensor_units

CHECKS = ("stuck", "dropout", "spike", "drift")


def run_lengths(x: pd.Series) -> pd.Series:
    """For each minute, how many minutes the reading has been exactly unchanged (1 = it just changed).

    Missing minutes break a run.
    """
    v = x.to_numpy(dtype=float)
    n = len(v)
    if n == 0:
        return pd.Series(np.zeros(0), index=x.index)
    ok = np.isfinite(v)
    same = np.zeros(n, dtype=bool)
    same[1:] = ok[1:] & ok[:-1] & (v[1:] == v[:-1])
    start_pos = np.where(~same, np.arange(n), 0)
    np.maximum.accumulate(start_pos, out=start_pos)   # position where the current run began
    out = (np.arange(n) - start_pos + 1).astype(float)
    out[~ok] = 0.0
    return pd.Series(out, index=x.index)


def _rolling_dev(x: pd.DataFrame, window: int, lag: int) -> pd.DataFrame:
    """Value minus the mean of a past window that ends ``lag`` minutes ago."""
    base = x.shift(lag).rolling(window, min_periods=window // 3).mean()
    return x - base


class FaultChecks:
    """Fitted per-sensor fault checks. ``flags(frame)`` returns one boolean frame per check."""

    def __init__(self, sensors: list[str], flag_rate: float = 0.5, min_stuck: int = 30,
                 stuck_partner_units: float = 3.0,
                 spike_units: float = 8.0, dropout_units: float = 10.0,
                 drift_window: int = 240, drift_lag: int = 30, drift_z: float = 6.0,
                 drift_persist: int = 90, ridge_alpha: float = 1.0):
        self.sensors = list(sensors)
        self.flag_rate, self.min_stuck = float(flag_rate), int(min_stuck)
        self.stuck_partner_units = float(stuck_partner_units)
        self.spike_units = float(spike_units)
        self.dropout_units = float(dropout_units)
        self.drift_window, self.drift_lag = int(drift_window), int(drift_lag)
        self.drift_z, self.drift_persist = float(drift_z), int(drift_persist)
        self.ridge_alpha = float(ridge_alpha)

    # ------------------------------------------------------------------------
    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.sensors].astype(float)

    def _drift_dev(self, x: pd.DataFrame) -> np.ndarray:
        d = _rolling_dev(x.ffill(limit=5), self.drift_window, self.drift_lag) / self.units
        return np.clip(np.nan_to_num(d.to_numpy(dtype=float), nan=0.0), -200, 200)

    def fit(self, frame: pd.DataFrame, rows=None) -> FaultChecks:
        """Learn the limits from ``rows`` (should be normal, pre-test minutes)."""
        rows = np.ones(len(frame), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        x = self._x(frame)
        self.units = sensor_units(frame, self.sensors, rows)
        days = rows.sum() / 1440
        share = self.flag_rate / (3 * len(self.sensors))            # per calibrated check and sensor
        self.budget = max(1, int(np.ceil(share * days)))            # natural flags allowed in the fit data

        # drift / partner model of the 4-hour change (also used by ``stuck``)
        D_all = self._drift_dev(x)
        D = D_all[rows]
        k = len(self.sensors)
        self.drift_coef = []
        for i in range(k):
            others = [j for j in range(k) if j != i]
            A, b = D[:, others], D[:, i]
            w = np.linalg.solve(A.T @ A + self.ridge_alpha * np.eye(len(others)), A.T @ b) if others else np.zeros(0)
            self.drift_coef.append((others, w))
        R = self._drift_resid(D)
        med = np.median(R, axis=0)
        self.drift_center = med
        self.drift_scale = np.maximum(np.median(np.abs(R - med), axis=0) * 1.4826, 0.5)
        P_all = self._drift_pred(D_all)
        Z_all = np.abs((D_all - P_all - med) / self.drift_scale)
        persist = pd.DataFrame(Z_all).rolling(self.drift_persist, min_periods=self.drift_persist).min().to_numpy()

        self.stuck_limit, self.low_limit, self.spike_limit, self.drift_limit = {}, {}, {}, {}
        for i, s in enumerate(self.sensors):
            v = x[s].to_numpy(dtype=float)
            # stuck: lengths of normal runs during which the partners predicted a change
            rl = run_lengths(x[s]).to_numpy()
            ends = np.flatnonzero((rl > 0) & (np.append(rl[1:], 0) <= rl) & rows)
            sizes = []
            for e in ends:
                if rl[e] < self.min_stuck:
                    continue
                p = P_all[int(e - rl[e] + 1): e + 1, i]
                if np.abs(p - p[0]).max() >= self.stuck_partner_units:
                    sizes.append(rl[e])
            self.stuck_limit[s] = int(max(self.min_stuck, self._kth(sizes) + 1))
            # dropout: physical, not calibrated
            q = float(x[s][rows].quantile(0.001))
            low = q - self.dropout_units * self.units[s]
            if q > 0:
                low = max(min(low, 0.5 * q), 0.0)    # exactly 0 always counts for a positive sensor
            self.low_limit[s] = low
            # spike: sizes of natural one/two-minute excursions
            size = self._excursions(v, self.units[s], 0.0)
            self.spike_limit[s] = max(self.spike_units, self._kth(size[rows & (size > 0)]))
            # drift: peak of each natural run of persistent partner disagreement
            pz = np.nan_to_num(persist[:, i], nan=0.0)
            peaks = [pz[a:b + 1].max() for a, b in segments(rows & (pz >= self.drift_z))]
            self.drift_limit[s] = max(self.drift_z, self._kth(peaks))
        return self

    def _kth(self, sizes) -> float:
        """The budget-th largest natural size (0 if there are fewer)."""
        sizes = np.sort(np.asarray(sizes, dtype=float))[::-1]
        return float(sizes[self.budget - 1]) if len(sizes) >= self.budget else 0.0

    def _drift_pred(self, D: np.ndarray) -> np.ndarray:
        P = np.zeros_like(D)
        for i, (others, w) in enumerate(self.drift_coef):
            if len(others):
                P[:, i] = D[:, others] @ w
        return P

    def _drift_resid(self, D: np.ndarray) -> np.ndarray:
        return D - self._drift_pred(D)

    @staticmethod
    def _excursions(v: np.ndarray, unit: float, limit: float) -> np.ndarray:
        """Size (in units) of one- or two-minute excursions that came straight back,
        reported at the minute of return; 0 elsewhere. Only excursions above ``limit``."""
        ref = pd.Series(v).shift(1).rolling(15, min_periods=5).median().to_numpy()
        dist = np.abs(v - ref) / unit
        out = np.zeros(len(v))
        out_mask = np.nan_to_num(dist, nan=0.0) > max(limit, 3.0)
        for a, b in segments(out_mask):
            r = b + 1
            if b - a + 1 > 2 or r >= len(v) or not np.isfinite(v[r]) or not np.isfinite(ref[a]):
                continue
            if abs(v[r] - ref[a]) / unit < 3.0:
                out[r] = float(np.nanmax(dist[a:b + 1]))
        return out

    # ------------------------------------------------------------------------
    def evidence(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Continuous evidence per check and sensor (>= 1 means the check fires)."""
        x = self._x(frame)
        idx = frame.index
        out = {}

        D = self._drift_dev(x)
        P = self._drift_pred(D)                      # what the partners say each sensor did
        stuck = pd.DataFrame(0.0, index=idx, columns=self.sensors)
        drop = pd.DataFrame(0.0, index=idx, columns=self.sensors)
        spike = pd.DataFrame(0.0, index=idx, columns=self.sensors)
        for i, s in enumerate(self.sensors):
            v = x[s].to_numpy(dtype=float)
            u = self.units[s]
            # stuck: long run of identical readings while the partners predict a change
            rl = run_lengths(x[s]).to_numpy()
            ev = np.zeros(len(v))
            long_run = rl >= self.stuck_limit[s]
            for a, b in segments(long_run):
                r0 = int(a - rl[a] + 1)              # where the run began
                p = P[r0:b + 1, i]
                moved = np.maximum.accumulate(np.abs(p - p[0]))[a - r0:]
                ok = moved >= self.stuck_partner_units
                ev[a:b + 1] = np.where(ok, rl[a:b + 1] / self.stuck_limit[s], 0.0)
            stuck[s] = ev
            # dropout: one big step down to (near) zero
            low = np.nan_to_num(v, nan=np.inf) <= self.low_limit[s]
            ev = np.zeros(len(v))
            for a, b in segments(low):
                prev = v[a - 1] if a > 0 else np.nan
                if np.isfinite(prev) and prev > self.low_limit[s] and \
                        (prev - v[a]) >= max(self.dropout_units * u, 0.5 * abs(prev)):
                    ev[a:b + 1] = 1.0
            drop[s] = ev
            # spike: short excursion that comes straight back, larger than normal ones
            spike[s] = self._excursions(v, u, self.spike_limit[s]) / self.spike_limit[s]
        out["stuck"] = stuck
        out["dropout"] = drop
        out["spike"] = spike

        R = (D - P - self.drift_center) / self.drift_scale
        z = pd.DataFrame(np.abs(R), index=idx, columns=self.sensors)
        # persistent: the smallest |z| over the last ``drift_persist`` minutes
        lim = pd.Series(self.drift_limit)[self.sensors]
        out["drift"] = z.rolling(self.drift_persist, min_periods=self.drift_persist).min().fillna(0) / lim
        empty = x.isna().all(axis=1).to_numpy()
        for k in out:
            out[k] = out[k].fillna(0.0)
            out[k].loc[empty] = 0.0
        return out

    def flags(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Boolean flags per check and sensor, plus ``any`` (any check)."""
        ev = self.evidence(frame)
        fl = {k: v >= 1.0 for k, v in ev.items()}
        anyf = fl["stuck"].copy()
        for k in CHECKS[1:]:
            anyf |= fl[k]
        fl["any"] = anyf
        return fl


@register("signal", "fault_checks")
class FaultCheckSignal:
    """The four checks as one signal: score = largest evidence over sensors and checks.

    Fitted on the normal fit minutes (labels are used only to leave out event minutes).
    """

    name = "fault_checks"
    supervised = False

    def __init__(self, sensors: list[str] | None = None, seed: int = 42, **params):
        self.sensors = sensors
        self.params = params

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        sensors = self.sensors or [c for c in frame.columns if c not in ("EVENT", "LABEL_CLASS")]
        self.sensors = sensors
        return frame[sensors].astype(float)

    def fit(self, Z: pd.DataFrame, y=None) -> FaultCheckSignal:
        rows = ~Z.isna().all(axis=1).to_numpy()
        if y is not None:
            rows &= np.asarray(y) == 0
        self.checks = FaultChecks(list(Z.columns), **self.params).fit(Z, rows)
        return self

    def score_parts(self, Z: pd.DataFrame) -> pd.DataFrame:
        ev = self.checks.evidence(Z)
        parts = {k: v.max(axis=1) for k, v in ev.items()}
        main = pd.concat(parts, axis=1).max(axis=1)
        df = pd.DataFrame({"main": main, **parts})
        df.loc[Z.isna().all(axis=1).to_numpy()] = np.nan
        return df

    def score(self, Z: pd.DataFrame) -> pd.Series:
        return self.score_parts(Z)["main"].rename(self.name)
