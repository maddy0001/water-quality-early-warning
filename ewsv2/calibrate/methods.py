"""Turning a score into alarms (Segment 5, contribution C4).

All calibrators work on *nines*: a score is first mapped to how unusual it is among
normal pre-test minutes, ``nines = -log10(share of normal minutes scoring at least as
high)``. 3 nines = rarer than 1 in 1,000 normal minutes. Above the largest normal score
the scale keeps growing linearly, so extreme minutes are not tied.

``cv_f1``     the S1 rule: the threshold with the best event F1 on out-of-fold pre-test
              scores. Optimises a score; says nothing about the false-alarm rate.
``budget``    static conformal threshold: the lowest threshold whose false alarms on
              normal pre-test minutes stay within ``budget`` per day. A stated rate,
              valid as long as the test season looks like the pre-test one.
``adaptive``  starts at the ``budget`` threshold and corrects itself online. Every
              normal minute lowers it a little (``step × budget / 1440``); every false
              alarm, once the operator has closed it as false (when the alarm episode
              ends and no water event was confirmed), raises it by ``step``. In the long
              run false alarms settle at the budget, whatever the season does to the
              scores. This is adaptive conformal inference (Gibbs & Candès, NeurIPS
              2021) applied to alarm episodes, with the operator's verdict as feedback.

``persistence(alarm, k, n)`` (E6): alarm only when at least k of the last n minutes are
above the threshold (inside the online loop for ``adaptive``: ``persist=(k, n)``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.core.registry import register
from ewsv2.evaluate.metrics import EventEvaluator
from ewsv2.evaluate.thresholds import candidate_thresholds, select_threshold

MINUTES_PER_DAY = 1440


class Nines:
    """Map scores to nines of rarity among reference (normal) scores."""

    def __init__(self, reference: np.ndarray):
        ref = np.asarray(reference, dtype=float)
        ref = np.sort(ref[np.isfinite(ref)])
        if len(ref) == 0:
            raise ValueError("no reference scores")
        self.ref = ref
        self.n = len(ref)
        self.top = ref[-1]
        spread = np.quantile(ref, 0.999) - np.quantile(ref, 0.5)
        self.spread = spread if spread > 1e-12 else 1.0
        self.max_nines = np.log10(self.n + 1)

    def __call__(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        out = np.full(s.shape, np.nan)
        ok = np.isfinite(s)
        above = self.n - np.searchsorted(self.ref, s[ok], side="left")   # normal minutes scoring >= s
        nines = -np.log10((above + 1) / (self.n + 1))
        extra = np.maximum(s[ok] - self.top, 0.0) / self.spread
        out[ok] = nines + extra
        return out


def to_alarm(nines: np.ndarray, threshold: float) -> np.ndarray:
    return np.nan_to_num(nines, nan=-np.inf) >= threshold


def persistence(alarm: np.ndarray, k: int, n: int) -> np.ndarray:
    """At least ``k`` of the last ``n`` minutes in alarm (causal)."""
    a = np.asarray(alarm, dtype=float)
    if k <= 1 and n <= 1:
        return a.astype(bool)
    run = pd.Series(a).rolling(n, min_periods=1).sum().to_numpy()
    return run >= k


class Calibrator:
    """Common interface: ``fit`` on pre-test (out-of-fold) nines, ``alarm`` on test nines."""

    name = "calibrator"

    def fit(self, nines, y, index, rows, tol=0, gap=5, max_ep=60) -> Calibrator:
        raise NotImplementedError

    def alarm(self, nines, y=None, index=None, rows=None) -> np.ndarray:
        return to_alarm(nines, self.threshold)


@register("calibrator", "cv_f1")
class CvF1(Calibrator):
    name = "cv_f1"

    def __init__(self, **_):
        pass

    def fit(self, nines, y, index, rows, tol=0, gap=5, max_ep=60):
        self.threshold, self.value = select_threshold(nines, y, index, rows, "event_f1", tol, gap, max_ep)
        return self


def _false_episodes(alarm: np.ndarray, y: np.ndarray, gap: int, max_ep: int) -> int:
    """False alarms in a stretch, as the S1 evaluator counts them: alarm minutes without a
    water event, merged when at most ``gap`` apart, one more for every ``max_ep`` minutes."""
    from ewsv2.evaluate.metrics import _merge, segments
    runs = _merge(segments(np.asarray(alarm, bool) & ~np.asarray(y, bool)), gap)
    return int(sum(max(1, int(np.ceil((b - a + 1) / max_ep))) for a, b in runs))


def budget_threshold(nines, y, index, rows, budget: float, gap: int = 5, max_ep: int = 60) -> tuple[float, float]:
    """Lowest threshold whose false alarms per day on ``rows`` stay within ``budget``."""
    ev = EventEvaluator(y, index, rows, 0, gap, max_ep)
    cands = np.sort(candidate_thresholds(np.asarray(nines)[np.asarray(rows, bool) & np.isfinite(nines)], 60, 160))
    # false alarms fall as the threshold rises: binary search for the first one within budget
    lo, hi = 0, len(cands) - 1
    best = (float(cands[-1]) + 1e-9, 0.0)
    while lo <= hi:
        mid = (lo + hi) // 2
        fa = ev.evaluate(to_alarm(nines, cands[mid])).false_alarms_per_day
        if fa <= budget:
            best = (float(cands[mid]), float(fa))
            hi = mid - 1
        else:
            lo = mid + 1
    return best


@register("calibrator", "budget")
class Budget(Calibrator):
    name = "budget"

    def __init__(self, budget: float = 1.0, **_):
        self.budget = float(budget)

    def fit(self, nines, y, index, rows, tol=0, gap=5, max_ep=60):
        self.threshold, self.value = budget_threshold(nines, y, index, rows, self.budget, gap, max_ep)
        return self


@register("calibrator", "adaptive")
class Adaptive(Calibrator):
    """Budget threshold, then corrected online from the operator's verdict on each alarm."""

    name = "adaptive"

    def __init__(self, budget: float = 1.0, step: float = 0.25, low: float = 1.0, high: float = 8.0,
                 merge_gap: int = 5, max_episode_minutes: int = 60, persist: tuple = (1, 1), **_):
        self.budget, self.step = float(budget), float(step)
        self.k, self.n = int(persist[0]), int(persist[1])
        self.low, self.high = float(low), float(high)
        self.merge_gap, self.max_ep = int(merge_gap), int(max_episode_minutes)

    def fit(self, nines, y, index, rows, tol=0, gap=5, max_ep=60):
        self.threshold, self.value = budget_threshold(nines, y, index, rows, self.budget, gap, max_ep)
        self.merge_gap, self.max_ep = int(gap), int(max_ep)
        return self

    def _false_count(self, alarm, y) -> int:
        return _false_episodes(alarm, y, self.merge_gap, self.max_ep)

    def alarm(self, nines, y=None, index=None, rows=None) -> np.ndarray:
        """Online pass over ``rows`` in time order. ``y`` (water-event labels) plays the operator:
        it is read only at the end of each alarm episode, to say whether it was false."""
        s = np.nan_to_num(np.asarray(nines, dtype=float), nan=-np.inf)
        n = len(s)
        rows = np.ones(n, dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        y = np.zeros(n, dtype=bool) if y is None else np.asarray(y, dtype=bool)
        decay = self.step * self.budget / MINUTES_PER_DAY
        theta = self.threshold
        alarm = np.zeros(n, dtype=bool)
        self.trace = np.full(n, np.nan)
        ep_start, last_on = -1, -10**9
        above = np.zeros(n, dtype=bool)
        for t in range(n):
            if not rows[t]:
                continue
            above[t] = s[t] >= theta
            on = above[max(0, t - self.n + 1): t + 1].sum() >= self.k if self.n > 1 else above[t]
            alarm[t] = on
            self.trace[t] = theta
            if on:
                if t - last_on > self.merge_gap:          # a new episode starts
                    ep_start = t
                last_on = t
            elif ep_start >= 0 and t - last_on > self.merge_gap:
                # the episode is closed; the operator marks which of its minutes had no water
                # event. Those are false alarms, counted exactly as the evaluator counts them.
                theta += self.step * self._false_count(alarm[ep_start:last_on + 1], y[ep_start:last_on + 1])
                ep_start = -1
            theta -= decay                                 # the clock: no label needed
            theta = min(max(theta, self.low), self.high)
        return alarm
