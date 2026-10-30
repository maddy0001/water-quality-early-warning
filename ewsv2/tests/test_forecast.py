"""Forecast surprise: windows, error scale, causality, and the paper-1 networks."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.core.registry import available, build
from ewsv2.signals.forecast import error_scale, make_windows, reading_resolution, window_positions


def _frame(n=4000, seed=0, sensors=("a", "b")):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cols = {}
    for k, s in enumerate(sensors):
        cols[s] = np.sin(2 * np.pi * t / 240 + k) + 0.02 * rng.standard_normal(n)
    df = pd.DataFrame(cols, index=pd.date_range("2024-01-01", periods=n, freq="min"))
    df["EVENT"] = 0
    return df


def test_registered():
    names = set(available("signal")["signal"])
    assert {"forecast_persistence", "forecast_linear", "v1_gru_attention", "v1_bilstm_attention"} <= names


def test_window_positions_need_consecutive_complete_minutes():
    idx = pd.date_range("2024", periods=10, freq="min").delete(5)     # one missing minute
    v = np.ones((9, 2))
    v[7, 1] = np.nan
    pos = window_positions(idx, v, span=3)
    # rows 2-4 are fine; rows 5-6 span the missing minute; rows 7-8 include the NaN
    assert pos.tolist() == [2, 3, 4]


def test_make_windows_alignment():
    v = np.arange(20, dtype=float).reshape(10, 2)          # row r = [2r, 2r+1]
    X, Y = make_windows(v, np.array([9]), lookback=3, horizon=2)
    # target row 9; inputs end at row 9-2 = 7 -> rows 5, 6, 7
    assert X.shape == (1, 3, 2) and Y.shape == (1, 2)
    assert X[0, :, 0].tolist() == [10, 12, 14] and Y[0].tolist() == [18, 19]


def test_error_scale_handles_quantised_sensors():
    res = np.zeros((1000, 2))
    res[::50, 0] = 0.1                                       # mostly exactly zero
    scale = error_scale(res, resolution=np.array([0.1, 0.05]))
    assert scale[0] >= 0.1 and scale[1] == pytest.approx(0.05)
    v = np.repeat(np.arange(0, 1, 0.1), 10)[:, None]        # steps of 0.1
    assert reading_resolution(v)[0] == pytest.approx(0.1)


def test_persistence_flags_step_and_is_causal():
    df = _frame()
    df.iloc[3000:, 0] += 2.0                                  # step change at row 3000
    sig = build("signal", "forecast_persistence", sensors=["a", "b"], lookback=10, horizon=5)
    Z = sig.transform(df)
    sig.fit(Z.iloc[:2500])
    s = sig.score(Z)
    assert s.iloc[3000:3005].max() > 5 * s.iloc[2600:2900].median()
    later = df.copy()
    later.iloc[3500:, 1] += 100.0                             # changing the future ...
    s2 = sig.score(sig.transform(later))
    pd.testing.assert_series_equal(s.iloc[:3500], s2.iloc[:3500])   # ... leaves the past unchanged


def test_linear_forecast_beats_persistence_on_smooth_signal():
    df = _frame(seed=1)
    scores = {}
    for name in ("forecast_persistence", "forecast_linear"):
        sig = build("signal", name, sensors=["a", "b"], lookback=20, horizon=10)
        sig.fit(sig.transform(df).iloc[:3000])
        scores[name] = sig.err_scale.mean()
    assert scores["forecast_linear"] < scores["forecast_persistence"]


def test_training_skips_event_windows():
    df = _frame(seed=2)
    df.loc[df.index[1000:1200], "EVENT"] = 1
    sig = build("signal", "forecast_linear", sensors=["a", "b"], lookback=20, horizon=5, max_windows=10**6)
    Z = sig.transform(df)
    sig.fit(Z.iloc[:3000], df["EVENT"].to_numpy()[:3000])
    clean = build("signal", "forecast_linear", sensors=["a", "b"], lookback=20, horizon=5, max_windows=10**6)
    clean.fit(Z.iloc[:3000], None)
    assert not np.allclose(sig.W, clean.W)                   # event windows really were left out


def test_no_score_before_a_full_window_or_after_a_gap():
    df = _frame()
    df.iloc[2000:2010, :2] = np.nan
    sig = build("signal", "forecast_persistence", sensors=["a", "b"], lookback=10, horizon=5)
    Z = sig.transform(df)
    sig.fit(Z.iloc[:1500])
    s = sig.score(Z)
    assert s.iloc[:14].isna().all() and s.iloc[14:20].notna().all()
    assert s.iloc[2000:2010 + 14].isna().all()


@pytest.mark.parametrize("name,layer", [("v1_bilstm_attention", "bilstm_1"), ("v1_gru_attention", "gru_1")])
def test_v1_neural_forecasters_train_and_score(name, layer):
    pytest.importorskip("keras")
    df = _frame(n=3000)
    sig = build("signal", name, sensors=["a", "b"], lookback=12, horizon=3, max_windows=800,
                epochs=2, units=[8, 4], dense=[4, 4])
    Z = sig.transform(df)
    sig.fit(Z.iloc[:2000])
    names = [lyr.name for lyr in sig.model.layers]
    assert layer in names and "attention_weights" in names and "sum_over_time" in names
    assert sig.model.output_shape == (None, 2)
    parts = sig.score_parts(Z)
    assert list(parts.columns) == ["surprise", "surprise_mean"]
    assert parts.iloc[20:].notna().all().all()


def test_v1_architecture_matches_paper_one():
    pytest.importorskip("keras")
    from ewsv2.signals.forecast import build_v1_forecaster

    m = build_v1_forecaster(30, 9, cell="bilstm")
    assert m.get_layer("bilstm_1").forward_layer.units == 128
    assert m.get_layer("bilstm_2").forward_layer.units == 64
    assert m.get_layer("dense_1").units == 32 and m.get_layer("dense_2").units == 16
    assert m.count_params() == 310_602
