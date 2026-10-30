"""Choosing the alarm threshold (Segment 1).

A score becomes an alarm when it reaches a threshold. *Where* the threshold is
chosen matters as much as the model:

* chosen on the **test** rows to maximise the reported metric – common in papers,
  and a leak: the test labels pick the operating point (the audit measures this)
* chosen on the **validation** rows, then frozen – honest

Segment 5 replaces this simple search with a conformal threshold that holds a
false-alarm budget.
"""

from __future__ import annotations

import numpy as np

from ewsv2.evaluate.metrics import EventEvaluator, point_adjusted_f1, point_prf

OBJECTIVES = ("point_f1", "pa_f1", "event_f1")


def candidate_thresholds(scores: np.ndarray, n_coarse: int = 40, n_fine: int = 80) -> np.ndarray:
    """Score quantiles, dense near the top where rare-event thresholds live."""
    s = scores[np.isfinite(scores)]
    if s.size == 0:
        return np.array([np.inf])
    q = np.concatenate([np.linspace(0.5, 0.99, n_coarse), 1 - np.logspace(-2, -5, n_fine)])
    return np.unique(np.quantile(s, np.clip(q, 0, 1)))


def select_threshold(scores, y, index, rows, objective: str = "event_f1",
                     tolerance_after: int = 0, merge_gap: int = 5,
                     max_episode_minutes: int = 60) -> tuple[float, float]:
    """Threshold that maximises ``objective`` on the given rows. Returns (threshold, value)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    scores = np.asarray(scores, dtype=float)
    rows = np.asarray(rows, dtype=bool) & np.isfinite(scores)
    if not rows.any():
        raise ValueError("no rows to choose a threshold on")
    y = np.asarray(y)
    evaluator = (EventEvaluator(y, index, rows, tolerance_after, merge_gap, max_episode_minutes)
                 if objective == "event_f1" else None)

    best_t, best_v = np.inf, -1.0
    for t in candidate_thresholds(scores[rows]):
        alarm = np.nan_to_num(scores, nan=-np.inf) >= t
        if objective == "point_f1":
            v = point_prf(y, alarm, rows)["f1"]
        elif objective == "pa_f1":
            v = point_adjusted_f1(y, alarm, rows)
        else:
            v = evaluator.evaluate(alarm).event_f1
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


def to_alarm(scores, threshold: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.nan_to_num(s, nan=-np.inf) >= threshold

