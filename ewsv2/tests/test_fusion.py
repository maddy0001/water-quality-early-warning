"""Fusion, calibration under a budget, cross-conformal module scores (Segment 5)."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.calibrate.methods import Adaptive, Budget, Nines, budget_threshold, persistence, to_alarm
from ewsv2.cli import build_parser
from ewsv2.core.registry import available, build
from ewsv2.core.types import EVENT_COL
from ewsv2.evaluate.metrics import EventEvaluator
from ewsv2.fusion.methods import Stacking, unit_scale
from ewsv2.fusion.modules import module_nines

from .test_inject import station


def test_registered():
    assert {"single", "hand", "equal", "max", "stacking"} <= set(available("fusion")["fusion"])
    assert {"cv_f1", "budget", "adaptive"} <= set(available("calibrator")["calibrator"])


def test_nines_monotone_and_open_ended():
    ref = np.arange(1000.0)
    f = Nines(ref)
    z = f(np.array([-1.0, 500.0, 998.0, 999.0, 1500.0, np.nan]))
    assert z[0] == pytest.approx(0.0, abs=1e-3)
    assert z[1] == pytest.approx(-np.log10(501 / 1001), abs=1e-6)
    assert z[2] < z[3] < z[4] and np.isnan(z[5])
    assert z[3] == pytest.approx(np.log10(1001 / 2), abs=1e-6)       # top score: 1 of 1000 at least as high


def test_persistence():
    a = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=bool)
    assert persistence(a, 2, 3).tolist() == [False, False, False, True, True, True, False, False]
    assert persistence(a, 1, 1).tolist() == a.tolist()


def _stream(days=20, shift_after=None, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    s = rng.standard_normal(n)
    s = pd.Series(s).rolling(5, min_periods=1).mean().to_numpy() * 2      # some autocorrelation
    if shift_after is not None:
        s[shift_after:] += shift
    return idx, s


def test_budget_threshold_holds_on_fit_rows():
    idx, s = _stream()
    y = np.zeros(len(s), dtype=int)
    rows = np.ones(len(s), dtype=bool)
    z = Nines(s)(s)
    thr, fa = budget_threshold(z, y, idx, rows, budget=1.0)
    assert fa <= 1.0
    got = EventEvaluator(y, idx, rows).evaluate(to_alarm(z, thr)).false_alarms_per_day
    assert got <= 1.0 and got > 0.2                         # close to the budget, not far below


def test_adaptive_holds_budget_when_scores_drift():
    idx, s = _stream(days=40, shift_after=20 * 1440, shift=0.6)
    y = np.zeros(len(s), dtype=int)
    fit = np.zeros(len(s), dtype=bool)
    fit[: 20 * 1440] = True
    test = ~fit
    z = Nines(s[fit])(s)
    ev = EventEvaluator(y, idx, test)
    static = Budget(budget=1.0).fit(z, y, idx, fit)
    fa_static = ev.evaluate(static.alarm(z) & test).false_alarms_per_day
    adapt = Adaptive(budget=1.0).fit(z, y, idx, fit)
    fa_adapt = ev.evaluate(adapt.alarm(z, y, idx, test)).false_alarms_per_day
    assert fa_static > 3.0                                   # the drift breaks the fixed threshold
    assert fa_adapt < 2.0 and fa_adapt < fa_static / 2       # the adaptive one comes back to the budget
    assert np.nanmax(adapt.trace) > static.threshold         # by raising its threshold


def test_adaptive_ignores_confirmed_events():
    idx, s = _stream(days=10)
    y = np.zeros(len(s), dtype=int)
    s[5000:5030] += 50
    y[5000:5030] = 1
    z = Nines(s)(s)
    a = Adaptive(budget=1.0).fit(z, y, idx, np.ones(len(s), bool))
    a.alarm(z, y, idx, np.ones(len(s), bool))
    assert abs(a.trace[5100] - a.trace[4990]) < 0.05         # a confirmed water event does not raise it


def test_fusions():
    N = pd.DataFrame({"C": [0.0, 3.0, 0.0, 3.0], "A": [0.0, 0.0, 3.0, 3.0], "F": [0.0, 0.0, 0.0, 3.0]})
    hand = build("fusion", "hand").combine(N)
    equal = build("fusion", "equal").combine(N)
    assert hand[1] > hand[2] and equal[1] == pytest.approx(equal[2])  # v1 trusts the classifier more
    assert build("fusion", "max").combine(N).tolist() == [0, 3, 3, 3]
    assert unit_scale(N).max().max() < 1.0
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    M = pd.DataFrame({"C": y * 4 + rng.random(3000), "A": rng.random(3000) * 4, "F": rng.random(3000) * 4})
    w = Stacking().fit(M, y).weights()
    assert w["C"] > 1.0 and abs(w["A"]) < w["C"] / 3          # it learns which module carries the signal


def test_module_nines_cross_conformal():
    ds = station(days=8, seed=4)
    split = build("splitter", "official", cv_folds=2, gap_minutes=60).split(ds)
    real = ds.frame[EVENT_COL].to_numpy()
    has = ~ds.frame[ds.sensors].isna().all(axis=1).to_numpy()
    s, info = module_nines(ds.frame, ds.sensors, split, real, has, {"role": "A", "name": "squared_diff"})
    assert info["folds_used"] == 2
    assert s[split.test].notna().mean() > 0.95 and s[split.val].notna().mean() > 0.95
    assert s[~(split.test | split.val)].isna().all()
    assert (s.dropna() >= 0).all()


def test_cli_has_fuse():
    assert build_parser().parse_args(["fuse", "--config", "s5_fusion"]).command == "fuse"
