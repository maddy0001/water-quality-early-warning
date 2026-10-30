"""Splits, baseline signals, threshold choice and the evaluate command."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.cli import main
from ewsv2.core.registry import available, build
from ewsv2.evaluate.thresholds import select_threshold, to_alarm

from .conftest import make_stream

DAYS = 6


@pytest.fixture
def station(tmp_path):
    """6 days of 1-minute data, an event every ~5 hours, last 2 days = test period."""
    minutes = DAYS * 1440
    events = [(i, 15) for i in range(200, minutes - 100, 300)]
    df = make_stream("2024-01-01", minutes, events, seed=3)
    df.loc[3000:3059, ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]] = np.nan  # outage
    path = tmp_path / "station.csv"
    df.to_csv(path, index=False)
    return build("source", "csv", path=str(path), test_start="2024-01-05").load(), path


def test_splitters_registered():
    assert {"official", "chrono", "random"} <= set(available("splitter")["splitter"])


def test_official_split_is_time_ordered_with_gap(station):
    ds, _ = station
    sp = build("splitter", "official", cv_folds=0, gap_minutes=60).split(ds)
    idx = ds.frame.index
    assert not (sp.fit & sp.val).any() and not (sp.fit & sp.test).any() and not (sp.val & sp.test).any()
    assert idx[sp.fit].max() < idx[sp.val].min() < idx[sp.val].max() < idx[sp.test].min()
    assert (idx[sp.val].min() - idx[sp.fit].max()) > pd.Timedelta(minutes=60)   # gap respected
    assert not sp.leaky
    assert not (sp.fit | sp.val | sp.test)[3000:3060].any()                      # outage never used


def test_official_cv_folds_purge_around_each_block(station):
    ds, _ = station
    sp = build("splitter", "official", cv_folds=4, gap_minutes=60).split(ds)
    assert len(sp.folds) == 4
    for fit_k, val_k in sp.folds:
        assert not (fit_k & val_k).any()
        vpos = np.flatnonzero(val_k)
        fpos = np.flatnonzero(fit_k)
        nearest = np.min(np.abs(fpos[:, None] - np.array([vpos[0], vpos[-1]])[None, :]))
        assert nearest > 60
        assert not (fit_k & sp.test).any() and not (val_k & sp.test).any()


def test_random_split_is_marked_leaky(station):
    ds, _ = station
    sp = build("splitter", "random", test_frac=0.3, seed=0).split(ds)
    assert sp.leaky
    assert abs(sp.test.sum() / (sp.fit.sum() + sp.test.sum()) - 0.3) < 0.01


@pytest.mark.parametrize("name", ["random_score", "squared_diff", "moving_variance", "rolling_zscore",
                                  "static_limits", "isolation_forest", "random_forest"])
def test_baseline_signal_contract(station, name):
    ds, _ = station
    y = ds.frame["EVENT"].to_numpy()
    sp = build("splitter", "official", cv_folds=0, gap_minutes=60).split(ds)
    params = {"n_estimators": 20} if name in ("isolation_forest", "random_forest") else {}
    sig = build("signal", name, sensors=ds.sensors, seed=0, **params)
    Z = sig.transform(ds.frame)
    assert len(Z) == len(ds.frame)
    s = sig.fit(Z[sp.fit], y[sp.fit]).score(Z)
    assert len(s) == len(ds.frame)
    assert s.iloc[3000:3060].isna().all()             # no score where there is no data
    finite = s.dropna()
    assert ((finite >= 0) & (finite <= 1)).all()


def test_supervised_signal_needs_labels(station):
    ds, _ = station
    sig = build("signal", "random_forest", sensors=ds.sensors, n_estimators=5)
    Z = sig.transform(ds.frame)
    with pytest.raises(ValueError, match="labels"):
        sig.fit(Z, None)


def test_threshold_selection_finds_obvious_cut():
    idx = pd.date_range("2020-01-01", periods=2000, freq="min")
    y = np.zeros(2000, int)
    y[500:520] = 1
    y[1500:1520] = 1
    scores = np.where(y == 1, 0.9, 0.1) + np.random.default_rng(0).normal(0, 0.01, 2000)
    t, v = select_threshold(scores, y, idx, np.ones(2000, bool), "event_f1")
    alarm = to_alarm(scores, t)
    assert v == 1.0                                   # both events caught, no false alarm
    assert alarm[500:520].any() and alarm[1500:1520].any()
    assert not alarm[y == 0].any()


def test_evaluate_command_end_to_end(station, tmp_path, monkeypatch, capsys):
    import ewsv2.core.paths as paths

    _, path = station
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    cfg = tmp_path / "mini.yaml"
    cfg.write_text(f"""
extends: base
run_name: mini_audit
title: Mini audit
data: {{source: csv, params: {{path: "{path.as_posix()}", test_start: "2024-01-05"}}}}
evaluate:
  bootstrap: 50
  signals:
    - random_score
    - rolling_zscore
    - {{name: random_forest, params: {{n_estimators: 20}}}}
    - {{name: v1_quorum, params: {{max_kernel_train: 1000}}}}
  protocols:
    - name: A_random
      splitter: {{name: random, params: {{test_frac: 0.3}}}}
      threshold: {{choose_on: test, objective: point_f1}}
      headline: point_f1
    - name: D_honest
      splitter: {{name: official, params: {{cv_folds: 3, gap_minutes: 60}}}}
      threshold: {{choose_on: val, objective: event_f1}}
      headline: event_f1
""")
    assert main(["evaluate", "--config", str(cfg)]) == 0
    out = tmp_path / "out" / "mini_audit" / "csv"
    for f in ("results.csv", "per_event.csv", "report.md", "audit.png"):
        assert (out / f).is_file(), f
    res = pd.read_csv(out / "results.csv")
    # 3 single signals + v1_quorum with 6 parts, under 2 protocols
    assert len(res) == 2 * (3 + 6) and set(res["protocol"]) == {"A_random", "D_honest"}
    fixed = res[res["signal"] == "v1_quorum:fixed_vote"]
    assert (fixed["threshold_on"] == "fixed").all() and (fixed["threshold"] == 0.5).all()
    assert {"v1_quorum:lof", "v1_quorum:autoencoder"} <= set(res["signal"])
    assert res.loc[res["protocol"] == "D_honest", "honest"].all()
    assert "event_recall_lo" in res.columns
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "How much each shortcut inflates the score" in text
    assert "Mini audit on csv" in capsys.readouterr().out


def test_bare_on_key_is_rejected(station, tmp_path, monkeypatch):
    import ewsv2.core.paths as paths

    _, path = station
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(f"""
extends: base
run_name: bad
data: {{source: csv, params: {{path: "{path.as_posix()}", test_start: "2024-01-05"}}}}
evaluate:
  signals: [random_score]
  protocols:
    - name: X
      splitter: {{name: official, params: {{cv_folds: 0, gap_minutes: 60}}}}
      threshold: {{on: val, objective: event_f1}}
""")
    with pytest.raises(ValueError, match="choose_on"):
        main(["evaluate", "--config", str(cfg)])
