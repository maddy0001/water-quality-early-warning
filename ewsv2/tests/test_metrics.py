"""Metrics give the exact numbers worked out by hand, and cannot be gamed."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.evaluate.metrics import (EventEvaluator, bootstrap_ci, evaluate_all, point_adjust,
                                    point_adjusted_f1, point_prf, segments)

IDX = pd.date_range("2020-01-01", periods=3000, freq="min")


@pytest.fixture
def case():
    y = np.zeros(3000, int)
    y[100:110] = 1      # event 1 (10 min)
    y[500:520] = 1      # event 2 (20 min)
    y[2000:2005] = 1    # event 3 (5 min)
    a = np.zeros(3000, bool)
    a[[105, 106]] = True            # hits event 1 after 5 minutes
    a[1000:1003] = True             # false run ...
    a[1006] = True                  # ... merged with it (gap of 3 <= 5)
    a[2500] = True                  # second false alarm
    return y, a


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1])).tolist() == [[1, 2], [4, 4]]
    assert segments(np.zeros(3)).shape == (0, 2)


def test_event_metrics_by_hand(case):
    y, a = case
    r = EventEvaluator(y, IDX).evaluate(a)
    assert (r.n_events, r.detected) == (3, 1)
    assert r.event_recall == pytest.approx(1 / 3)
    assert r.median_delay_min == 5
    assert r.false_episodes == 2
    assert r.normal_days == pytest.approx((3000 - 35) / 1440)
    assert r.false_alarms_per_day == pytest.approx(2 / ((3000 - 35) / 1440))
    assert r.event_precision == pytest.approx(1 / 3)   # 1 caught / (1 caught + 2 false)
    assert r.event_f1 == pytest.approx(1 / 3)
    assert r.per_event["detected"].tolist() == [True, False, False]


def test_point_metrics_by_hand(case):
    y, a = case
    p = point_prf(y, a)
    assert p["precision"] == pytest.approx(2 / 7) and p["recall"] == pytest.approx(2 / 35)
    # point adjustment turns 2 flagged minutes of event 1 into all 10
    assert point_adjust(y, a)[100:110].all()
    assert point_adjusted_f1(y, a) == pytest.approx(0.4)


def test_always_on_alarm_cannot_score_well(case):
    y, _ = case
    r = EventEvaluator(y, IDX).evaluate(np.ones(3000, bool))
    assert r.event_recall == 1.0
    assert r.false_alarms_per_day > 20          # ~24 per day: one per hour left on
    assert r.event_f1 < 0.15


def test_long_false_alarm_counts_once_per_hour():
    y = np.zeros(3000, int)
    a = np.zeros(3000, bool)
    a[0:150] = True                              # 2.5 hours on, no event
    r = EventEvaluator(y, IDX, max_episode_minutes=60).evaluate(a)
    assert r.false_episodes == 3


def test_missing_minutes_are_excluded(case):
    y, a = case
    valid = np.ones(3000, bool)
    valid[500:520] = False                       # event 2 has no data at all
    valid[2500] = False                          # the lone false alarm falls in a gap
    r = EventEvaluator(y, IDX, valid).evaluate(a)
    assert r.n_events == 2 and r.events_without_data == 1
    assert r.false_episodes == 1


def test_tolerance_after_event_end():
    y = np.zeros(300, int)
    y[100:110] = 1
    a = np.zeros(300, bool)
    a[112] = True                                # 2 minutes after the event ends
    idx = IDX[:300]
    assert EventEvaluator(y, idx, tolerance_after=0).evaluate(a).detected == 0
    assert EventEvaluator(y, idx, tolerance_after=5).evaluate(a).detected == 1


def test_evaluate_all_keys(case):
    y, a = case
    out = evaluate_all(y, a, IDX)
    for k in ("point_f1", "pa_f1", "event_recall", "false_alarms_per_day", "event_f1", "median_delay_min"):
        assert k in out


def test_bootstrap_intervals_contain_estimate(case):
    y, a = case
    ev = EventEvaluator(y, IDX)
    r = ev.evaluate(a)
    ci = bootstrap_ci(r, ev.normal_minutes_by_day, n_boot=300, seed=1)
    lo, hi = ci["event_recall"]
    assert lo <= r.event_recall <= hi
    lo, hi = ci["false_alarms_per_day"]
    assert lo <= r.false_alarms_per_day <= hi
