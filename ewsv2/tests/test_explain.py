"""Explainers and station transfer (Segment 6)."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.cli import build_parser
from ewsv2.core.registry import available, build
from ewsv2.core.types import EVENT_COL
from ewsv2.explain.methods import ClassifierExplainer, ranking
from ewsv2.explain.transfer import normalise

from .test_inject import station


@pytest.fixture(scope="module")
def ds():
    return station(days=6, seed=7, events=[(i, 20) for i in range(600, 8000, 700)])


def _normal(ds):
    f = ds.frame
    return (f[EVENT_COL] == 0).to_numpy() & (f.index < ds.periods["test"][0])


def test_registered():
    assert {"deviation", "forecast_residual"} <= set(available("explainer")["explainer"])


def test_ranking_puts_missing_last():
    r = ranking(pd.Series({"a": 0.1, "b": np.nan, "c": 2.0}))
    assert r == ["c", "a", "b"]


def _bump(ds, sensor, at, size_units, units):
    f = ds.frame.copy()
    f.iloc[at:at + 20, ds.sensors.index(sensor)] += size_units * units[sensor]
    return f


def test_deviation_and_forecast_blame_the_sensor_that_moved(ds):
    normal = _normal(ds)
    dev = build("explainer", "deviation", sensors=ds.sensors).fit(ds.frame, normal)
    fr = build("explainer", "forecast_residual", sensors=ds.sensors).fit(ds.frame, normal)
    at = 7000 + 300
    f = _bump(ds, "Leit", at, 40, dev.tf.units)
    p = at + 5
    assert ranking(dev.explain(f, [p]).iloc[0])[0] == "Leit"
    assert ranking(fr.explain(f, [p]).iloc[0])[0] == "Leit"


def test_classifier_explainers_find_the_event_sensor(ds):
    f = ds.frame
    rows = (f.index < ds.periods["test"][0]) & f[ds.sensors].notna().all(axis=1).to_numpy()
    clf = ClassifierExplainer(ds.sensors, seed=0).fit(f, rows, f[EVENT_COL].to_numpy())
    # a test-period event (pH and redox drop, as in the synthetic events)
    test_events = np.flatnonzero(f[EVENT_COL].to_numpy() & (f.index >= ds.periods["test"][0]))
    p = int(test_events[3])
    onset = int(test_events[0])
    shap = clf.tree_shap(f, [p]).iloc[0]
    assert ranking(shap)[0] in ("pH", "Redox")
    occ, base = clf.occlusion(f, [p], [onset])
    assert base[0] > 0.5
    assert ranking(occ.iloc[0])[0] in ("pH", "Redox")
    assert occ.iloc[0][["Tp", "Fm"]].abs().max() < occ.iloc[0][["pH", "Redox"]].max()


def test_normalise_maps_roles_and_scales(ds):
    n = normalise(ds, ["temperature", "ph", "conductivity", "turbidity"])
    assert n.sensors == ["temperature", "ph", "conductivity", "turbidity"]
    assert n.meta["transfer_sensors"]["conductivity"] == "Leit"
    pre = n.frame.index < n.periods["test"][0]
    med = n.frame.loc[pre, n.sensors].median()
    assert (med.abs() < 1e-9).all()                         # centred on the pre-test median
    with pytest.raises(ValueError):
        normalise(ds, ["organics"])                          # this station has no organics sensor


def test_cli_has_explain_and_transfer():
    p = build_parser()
    assert p.parse_args(["explain", "--config", "s6_explain"]).command == "explain"
    assert p.parse_args(["transfer", "--config", "s6_explain"]).command == "transfer"
