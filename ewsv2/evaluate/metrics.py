"""Evaluation metrics (Segment 1).

Two families:

**Point metrics** (what most GECCO papers report) treat every minute as a sample:
point precision / recall / F1, and the *point-adjusted* F1, where a whole labelled
event counts as found if any one of its minutes is flagged. Point adjustment is known
to inflate scores (Kim et al., AAAI 2022); we compute it to show by how much.

**Event metrics** (what a utility cares about), computed by :class:`EventEvaluator`:

* event recall      – share of labelled events with at least one alarm inside the event
                      (optionally extended by ``tolerance_after`` minutes)
* detection delay   – minutes from event start to the first alarm (detected events only)
* false alarms/day  – false-alarm episodes per day of normal data. A false-alarm
                      episode is a run of alarm minutes *outside* every event window
                      (runs at most ``merge_gap`` minutes apart are merged). Alarms that
                      stay on between events therefore count as false, however many
                      events they also touch. A false alarm that stays on counts once per
                      ``max_episode_minutes`` (default 60): an operator re-checks it every
                      hour, so a permanent alarm cannot score well.
* false-alarm hours/day – alarm time outside events, per day of normal data
* event precision   – events caught / (events caught + false-alarm episodes)
* event F1          – harmonic mean of event recall and event precision

Minutes with no data (all sensors missing) are excluded everywhere: they cannot
raise an alarm, and they do not count as normal time for the false-alarm rate.
Events with no data at all are excluded from recall (reported as ``events_without_data``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def segments(mask: np.ndarray) -> np.ndarray:
    """(start, end) row positions of runs of True, end inclusive. Shape (n, 2)."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return np.empty((0, 2), dtype=int)
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return np.column_stack([edges[0::2], edges[1::2] - 1])


def _merge(segs: np.ndarray, max_gap: int) -> np.ndarray:
    """Merge runs whose gap (rows strictly between them) is <= max_gap."""
    if len(segs) <= 1 or max_gap <= 0:
        return segs
    gaps = segs[1:, 0] - segs[:-1, 1] - 1
    new_group = np.concatenate([[True], gaps > max_gap])
    group = np.cumsum(new_group) - 1
    starts = segs[new_group, 0]
    ends = np.zeros(group[-1] + 1, dtype=int)
    np.maximum.at(ends, group, segs[:, 1])
    return np.column_stack([starts, ends])


# ----------------------------------------------------------------------------
# Point metrics
# ----------------------------------------------------------------------------

def point_prf(y_true, y_pred, valid=None) -> dict[str, float]:
    """Precision, recall and F1 over individual minutes."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        y, p = y[v], p[v]
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def point_adjust(y_true, y_pred) -> np.ndarray:
    """If any minute of a labelled event is flagged, flag the whole event (Xu et al. 2018)."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool).copy()
    for s, e in segments(y):
        if p[s:e + 1].any():
            p[s:e + 1] = True
    return p


def point_adjusted_f1(y_true, y_pred, valid=None) -> float:
    return point_prf(y_true, point_adjust(y_true, y_pred), valid)["f1"]


# ----------------------------------------------------------------------------
# Event metrics
# ----------------------------------------------------------------------------

@dataclass
class EventResult:
    n_events: int
    detected: int
    event_recall: float
    median_delay_min: float
    mean_delay_min: float
    alarm_episodes: int
    false_episodes: int
    normal_days: float
    false_alarms_per_day: float
    false_alarm_minutes_per_day: float
    event_precision: float
    event_f1: float
    events_without_data: int
    per_event: pd.DataFrame        # one row per scorable event: start, end, detected, delay_min
    false_episode_days: pd.Series  # number of false episodes per calendar day

    def summary(self) -> dict[str, float]:
        return {
            "events": self.n_events,
            "detected": self.detected,
            "event_recall": self.event_recall,
            "median_delay_min": self.median_delay_min,
            "false_alarms_per_day": self.false_alarms_per_day,
            "false_alarm_hours_per_day": self.false_alarm_minutes_per_day / 60,
            "event_precision": self.event_precision,
            "event_f1": self.event_f1,
            "alarm_episodes": self.alarm_episodes,
            "false_episodes": self.false_episodes,
            "normal_days": self.normal_days,
        }


class EventEvaluator:
    """Precomputes the event structure once, then scores many alarm series quickly.

    Parameters
    ----------
    y_true          : 0/1 per minute (labelled event minutes)
    index           : DatetimeIndex of the same length (regular 1-minute steps assumed)
    valid           : True where data exist (default: all True)
    tolerance_after : minutes after an event's end in which an alarm still counts as a hit
    merge_gap       : alarm runs separated by at most this many minutes form one episode
    max_episode_minutes : a false alarm that stays on counts again for every this many minutes
    """

    def __init__(self, y_true, index: pd.DatetimeIndex, valid=None,
                 tolerance_after: int = 0, merge_gap: int = 5, max_episode_minutes: int = 60):
        self.y = np.asarray(y_true, dtype=bool)
        self.index = pd.DatetimeIndex(index)
        n = len(self.y)
        if len(self.index) != n:
            raise ValueError("y_true and index must have the same length")
        self.valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self.tolerance_after = int(tolerance_after)
        self.merge_gap = int(merge_gap)
        self.max_episode_minutes = max(1, int(max_episode_minutes))

        all_events = segments(self.y)
        has_data = np.array([self.valid[s:e + 1].any() for s, e in all_events], dtype=bool) \
            if len(all_events) else np.empty(0, dtype=bool)
        self.events = all_events[has_data]
        self.events_without_data = int((~has_data).sum())

        # Event windows (event + tolerance) – used both for hits and to define "normal" time.
        self.windows = self.events.copy()
        if len(self.windows):
            self.windows[:, 1] = np.minimum(self.windows[:, 1] + self.tolerance_after, n - 1)
        self.window_mask = np.zeros(n, dtype=bool)
        for s, e in all_events:  # every labelled minute is "not normal", even without data
            self.window_mask[s:min(e + self.tolerance_after, n - 1) + 1] = True
        self._window_cum = np.concatenate([[0], np.cumsum(self.window_mask)])

        normal = self.valid & ~self.window_mask
        self.normal_minutes = int(normal.sum())
        self.normal_days = self.normal_minutes / MINUTES_PER_DAY
        days = self.index.normalize()
        self.normal_minutes_by_day = pd.Series(normal.astype(int), index=days).groupby(level=0).sum()

    # -- core ---------------------------------------------------------------
    def evaluate(self, alarm) -> EventResult:
        a = np.asarray(alarm, dtype=bool) & self.valid
        if len(a) != len(self.y):
            raise ValueError("alarm length does not match y_true")

        # hits and delays
        rows = []
        for (s, e), (_, we) in zip(self.events, self.windows):
            hits = np.flatnonzero(a[s:we + 1])
            detected = hits.size > 0
            delay = (self.index[s + hits[0]] - self.index[s]).total_seconds() / 60 if detected else np.nan
            rows.append((self.index[s], self.index[e], detected, delay))
        per_event = pd.DataFrame(rows, columns=["start", "end", "detected", "delay_min"])
        n_events = len(per_event)
        detected = int(per_event["detected"].sum()) if n_events else 0
        recall = detected / n_events if n_events else float("nan")
        delays = per_event.loc[per_event["detected"], "delay_min"] if n_events else pd.Series(dtype=float)

        # alarm episodes (all alarms, for information)
        episodes = _merge(segments(a), self.merge_gap)
        n_eps = len(episodes)
        # false alarms: alarm minutes OUTSIDE every event window, grouped into episodes.
        # An alarm that stays on between events therefore produces many false episodes;
        # it cannot hide by touching an event somewhere.
        outside = a & ~self.window_mask
        runs = _merge(segments(outside), self.merge_gap)
        # A false alarm left on keeps costing attention: every ``max_episode_minutes``
        # it stays on counts as one more false alarm.
        if len(runs):
            lengths = runs[:, 1] - runs[:, 0] + 1
            chunks = np.maximum(1, np.ceil(lengths / self.max_episode_minutes).astype(int))
            starts = np.concatenate([r0 + self.max_episode_minutes * np.arange(c) for r0, c in zip(runs[:, 0], chunks)])
            false_eps = np.column_stack([starts, starts])
        else:
            false_eps = np.empty((0, 2), dtype=int)
        n_false = int(len(false_eps))
        false_minutes = int(outside.sum())

        fa_day = n_false / self.normal_days if self.normal_days > 0 else float("nan")
        fa_min_day = false_minutes / self.normal_days if self.normal_days > 0 else float("nan")
        precision = detected / (detected + n_false) if (detected + n_false) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if n_false:
            fdays = self.index[false_eps[:, 0]].normalize()
            false_by_day = pd.Series(1, index=fdays).groupby(level=0).sum()
        else:
            false_by_day = pd.Series(dtype=int)
        false_by_day = false_by_day.reindex(self.normal_minutes_by_day.index, fill_value=0)

        return EventResult(
            n_events=n_events,
            detected=detected,
            event_recall=recall,
            median_delay_min=float(delays.median()) if len(delays) else float("nan"),
            mean_delay_min=float(delays.mean()) if len(delays) else float("nan"),
            alarm_episodes=n_eps,
            false_episodes=n_false,
            normal_days=self.normal_days,
            false_alarms_per_day=fa_day,
            false_alarm_minutes_per_day=fa_min_day,
            event_precision=precision,
            event_f1=f1,
            events_without_data=self.events_without_data,
            per_event=per_event,
            false_episode_days=false_by_day,
        )


def evaluate_all(y_true, alarm, index, valid=None, tolerance_after: int = 0, merge_gap: int = 5,
                 max_episode_minutes: int = 60) -> dict:
    """Every metric in one dict: point, point-adjusted and event metrics."""
    ev = EventEvaluator(y_true, index, valid, tolerance_after, merge_gap, max_episode_minutes).evaluate(alarm)
    pt = point_prf(y_true, alarm, valid)
    out = {f"point_{k}": v for k, v in pt.items()}
    out["pa_f1"] = point_adjusted_f1(y_true, alarm, valid)
    out.update(ev.summary())
    return out


# ----------------------------------------------------------------------------
# Bootstrap confidence intervals
# ----------------------------------------------------------------------------

def bootstrap_ci(result: EventResult, normal_minutes_by_day: pd.Series, n_boot: int = 1000,
                 seed: int = 42, level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Percentile intervals for event recall, median delay and false alarms per day.

    Events are resampled for recall and delay; days are resampled for the false-alarm
    rate. With ~50 test events the intervals are wide, and reporting them is the point.
    """
    rng = np.random.default_rng(seed)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out: dict[str, tuple[float, float]] = {}

    det = result.per_event["detected"].to_numpy(dtype=float)
    delay = result.per_event["delay_min"].to_numpy(dtype=float)
    if len(det):
        idx = rng.integers(0, len(det), size=(n_boot, len(det)))
        recalls = det[idx].mean(axis=1)
        with np.errstate(all="ignore"):
            med = np.array([np.nanmedian(delay[i]) if np.isfinite(delay[i]).any() else np.nan for i in idx])
        out["event_recall"] = (float(np.percentile(recalls, lo)), float(np.percentile(recalls, hi)))
        med = med[np.isfinite(med)]
        out["median_delay_min"] = (float(np.percentile(med, lo)), float(np.percentile(med, hi))) if len(med) else (np.nan, np.nan)

    minutes = normal_minutes_by_day.reindex(result.false_episode_days.index, fill_value=0).to_numpy(dtype=float)
    falses = result.false_episode_days.to_numpy(dtype=float)
    keep = minutes > 0
    minutes, falses = minutes[keep], falses[keep]
    if len(minutes):
        idx = rng.integers(0, len(minutes), size=(n_boot, len(minutes)))
        rates = falses[idx].sum(axis=1) / (minutes[idx].sum(axis=1) / MINUTES_PER_DAY)
        out["false_alarms_per_day"] = (float(np.percentile(rates, lo)), float(np.percentile(rates, hi)))
    return out
