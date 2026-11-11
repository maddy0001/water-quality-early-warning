"""GECCO-Triage injection toolkit (Segment 4)."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.core.registry import available, build
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset
from ewsv2.evaluate.metrics import segments
from ewsv2.features.scale import reading_step, sensor_units
from ewsv2.inject import faults as F
from ewsv2.inject import signatures as S

S18 = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]
S19 = ["Tp", "pH", "Cond", "Turb", "SAC", "PFM"]


def station(days: int = 12, seed: int = 0, events=((2000, 20), (12000, 15))) -> Dataset:
    """Quantised, partly coupled synthetic station with train/test periods and two real events."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = pd.date_range("2024-01-01", periods=n, freq="1min")
    wave = np.sin(np.arange(n) * 2 * np.pi / 1440)                       # daily cycle shared by sensors
    df = pd.DataFrame({
        "Tp": np.round(8 + 0.3 * wave + 0.02 * rng.standard_normal(n), 1),
        "Cl": np.round(0.17 + 0.02 * wave + 0.004 * rng.standard_normal(n), 2),
        "pH": np.round(8.35 + 0.05 * rng.standard_normal(n), 2),
        "Redox": np.round(755 + 8 * wave + rng.standard_normal(n), 0),
        "Leit": np.round(210 + 3 * wave + 0.6 * rng.standard_normal(n), 0),
        "Trueb": np.round(0.02 + 0.0015 * rng.standard_normal(n), 3),
        "Cl_2": np.round(0.11 + 0.013 * wave + 0.002 * rng.standard_normal(n), 3),
        "Fm": np.round(1500 + 300 * wave + 30 * rng.standard_normal(n), 0),
        "Fm_2": np.round(900 + 150 * wave + 20 * rng.standard_normal(n), 0),
    }, index=pd.DatetimeIndex(t, name="time"))
    ev = np.zeros(n, dtype=int)
    for a, length in events:
        ev[a:a + length] = 1
        df.iloc[a:a + length, S18.index("pH")] -= 0.4
        df.iloc[a:a + length, S18.index("Redox")] -= 30
    df[EVENT_COL] = ev
    df[CLASS_COL] = ev
    split_at = t[int(n * 0.6)]
    periods = {"train": (t[0], split_at - pd.Timedelta(minutes=1)), "test": (split_at, t[-1])}
    return Dataset("synthetic", df, list(S18), periods)


def test_registered():
    assert "gecco_triage" in available("injector")["injector"]


def test_station_response_renormalised_and_observability():
    r = S.station_response("biological", S18)
    assert max(abs(v) for v in r.values()) == pytest.approx(1.0)
    assert r["Cl"] < 0 and r["Trueb"] > 0 and "Tp" not in r and "Fm" not in r
    # no chlorine, no ORP at the 2019 station: chlorine-demand signatures cannot be seen there
    assert S.station_response("pesticide", S19) == {}
    assert S.station_response("inorganic", S19) == {}
    assert set(S.station_response("wastewater", S19)) == {"pH", "Cond", "Turb", "SAC"}


def test_slug_shape():
    sh = S.slug_shape(5, 10, 8)
    assert len(sh) == 23 and sh.max() == pytest.approx(1.0) and sh.min() >= 0
    assert np.all(np.diff(sh[:5]) > 0) and np.all(sh[5:15] == 1)


def test_units_and_reading_step():
    ds = station()
    u = sensor_units(ds.frame, ds.sensors)
    assert reading_step(ds.frame["Cl"]) == pytest.approx(0.01)
    assert u["Cl"] >= 0.01 and (u > 0).all()


@pytest.mark.parametrize("kind", F.FAULT_TYPES)
def test_each_fault(kind):
    rng = np.random.default_rng(1)
    v = np.full(300, 5.0) + 0.01 * np.arange(300)
    v[250] = np.nan
    out, info = F.apply_fault(v, 100, 60, kind, "high", unit=0.1, rng=rng)
    assert np.isnan(out[250])                                    # missing stays missing
    assert np.array_equal(out[:100], v[:100]) and np.array_equal(out[160:250], v[160:250])
    seg = out[100:160] - v[100:160]
    if kind == "stuck":
        assert np.all(out[100:160] == v[99])
    elif kind == "dropout":
        assert np.all(out[100:160] == 0)
    elif kind == "drift":
        assert seg[0] == 0 and abs(seg[-1]) == pytest.approx(abs(info["magnitude_units"]) * 0.1, abs=1e-3)
    else:
        changed = np.flatnonzero(seg != 0)
        assert changed[0] == 0 and 1 <= len(changed) <= 4 and np.all(np.diff(changed) >= 2)


def test_benchmark_labels_buffers_and_periods():
    ds = station()
    inj, ep = build("injector", "gecco_triage", reps_test=1, reps_pretest=1, buffer_minutes=60, seed=3).build(ds)
    assert len(ep) > 0 and set(ep["period"]) == {"test", "pretest"}
    test_start = ds.periods["test"][0]
    assert (ep.loc[ep["period"] == "test", "start"] >= test_start).all()
    assert (ep.loc[ep["period"] == "pretest", "end"] < test_start).all()
    cls = inj.frame[CLASS_COL].to_numpy()
    for _, e in ep.iterrows():
        a, b = e["start_pos"], e["end_pos"]
        expect = 1 if e["kind"] == "contaminant" else 2
        assert (cls[a:b + 1] == expect).all()
        assert (inj.frame[EVENT_COL].to_numpy()[a:b + 1] == (1 if expect == 1 else 0)).all()
    # episodes keep the buffer from each other and from real events
    spans = sorted([(e["start_pos"], e["end_pos"]) for _, e in ep.iterrows()] +
                   [tuple(x) for x in segments(ds.frame[EVENT_COL].to_numpy().astype(bool))])
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 - b1 > 60
    # untouched outside the episodes; the original dataset is not modified
    mask = np.zeros(len(cls), dtype=bool)
    for _, e in ep.iterrows():
        mask[e["start_pos"]:e["end_pos"] + 1] = True
    pd.testing.assert_frame_equal(inj.frame.loc[~mask, ds.sensors], ds.frame.loc[~mask, ds.sensors])
    assert ds.frame[CLASS_COL].max() == 1


def test_benchmark_reproducible_and_seed_dependent():
    ds = station()
    a = build("injector", "gecco_triage", reps_test=1, reps_pretest=1, buffer_minutes=60, seed=5).build(ds)[1]
    b = build("injector", "gecco_triage", reps_test=1, reps_pretest=1, buffer_minutes=60, seed=5).build(ds)[1]
    c = build("injector", "gecco_triage", reps_test=1, reps_pretest=1, buffer_minutes=60, seed=6).build(ds)[1]
    pd.testing.assert_frame_equal(a, b)
    assert not a["start"].equals(c["start"])


def test_benchmark_rejects_unknown_types():
    with pytest.raises(ValueError):
        build("injector", "gecco_triage", contaminants=["plutonium"])
    with pytest.raises(ValueError):
        build("injector", "gecco_triage", faults=["melted"])
