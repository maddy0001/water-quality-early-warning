"""Fault checks, triage evidence, station systems and episode scoring (Segment 4)."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.cli import build_parser
from ewsv2.core.registry import available, build
from ewsv2.core.types import CLASS_COL, EVENT_COL
from ewsv2.signals.faults import FaultChecks, run_lengths
from ewsv2.triage import evaluate as TE
from ewsv2.triage.features import TriageFeatures
from ewsv2.triage.systems import Context, RuleTriage, WaterDetector, _with_grace

from .test_inject import station


@pytest.fixture(scope="module")
def ds():
    return station(days=10, seed=2)


@pytest.fixture(scope="module")
def checks(ds):
    f = ds.frame
    normal = (f[EVENT_COL] == 0).to_numpy() & (f.index < ds.periods["test"][0])
    return FaultChecks(ds.sensors).fit(f, normal)


def test_registered():
    assert {"detector_only", "rules", "learned"} <= set(available("triage")["triage"])
    assert "fault_checks" in available("signal")["signal"]


def test_run_lengths():
    x = pd.Series([1.0, 1.0, 1.0, 2.0, np.nan, 2.0, 2.0])
    assert run_lengths(x).tolist() == [1, 2, 3, 1, 0, 1, 2]


def _flag_after(checks, frame, sensor, check, a, b):
    return bool(checks.flags(frame)[check][sensor].iloc[a:b].any())


def test_each_check_fires_on_its_fault_and_is_quiet_on_normal_water(ds, checks):
    f = ds.frame
    test = np.asarray(f.index >= ds.periods["test"][0])
    base = checks.flags(f)
    for k in ("stuck", "dropout", "spike", "drift"):
        assert base[k].to_numpy()[test].mean() < 0.01, k            # rarely fires on normal water

    g = f.copy()
    a = int(np.flatnonzero(test)[0]) + 500
    g.iloc[a:a + 900, ds.sensors.index("Redox")] = g["Redox"].iloc[a - 1]     # frozen while Cl, Cl_2 cycle
    assert _flag_after(checks, g, "Redox", "stuck", a, a + 900)

    g = f.copy()
    g.iloc[a:a + 30, ds.sensors.index("Cl")] = 0.0
    assert _flag_after(checks, g, "Cl", "dropout", a, a + 1)                 # at the first minute

    g = f.copy()
    g.iloc[a, ds.sensors.index("pH")] += 30 * checks.units["pH"]
    assert _flag_after(checks, g, "pH", "spike", a + 1, a + 2)                # on return

    g = f.copy()
    g.iloc[a:a + 400, ds.sensors.index("Cl_2")] += np.linspace(0, 40, 400) * checks.units["Cl_2"]
    assert _flag_after(checks, g, "Cl_2", "drift", a, a + 400)


def test_checks_are_causal(ds, checks):
    f = ds.frame
    cut = 9000
    g = f.copy()
    g.iloc[cut:, :len(ds.sensors)] += 5.0                                    # change only the future
    e1, e2 = checks.evidence(f), checks.evidence(g)
    for k in e1:
        pd.testing.assert_frame_equal(e1[k].iloc[:cut], e2[k].iloc[:cut], check_exact=False)


def test_triage_features_hold_suspicion(ds):
    f = ds.frame.copy()
    normal = (f[EVENT_COL] == 0).to_numpy() & (f.index < ds.periods["test"][0])
    tf = TriageFeatures(ds.sensors, hold_minutes=60).fit(f, normal)
    a = 10000
    f.iloc[a:a + 20, ds.sensors.index("Cl")] = 0.0
    out = tf.compute(f)
    assert out["flags"]["now"]["Cl"].iloc[a] and not out["flags"]["now"]["Cl"].iloc[a + 30]
    assert out["flags"]["any"]["Cl"].iloc[a + 30]                            # still a suspect
    assert not out["flags"]["any"]["Cl"].iloc[a + 20 + 61]


def test_with_grace():
    y = np.array([0, 1, 1, 0, 0, 0, 0])
    assert _with_grace(y, 2).tolist() == [0, 1, 1, 1, 1, 0, 0]


def test_episode_outcomes():
    eps = pd.DataFrame([
        {"source": "real event", "kind": "water", "type": "real", "level": "", "lead_sensor": "", "start_pos": 10, "end_pos": 14},
        {"source": "injected fault", "kind": "fault", "type": "dropout", "level": "low", "lead_sensor": "Cl", "start_pos": 30, "end_pos": 34},
        {"source": "injected fault", "kind": "fault", "type": "stuck", "level": "low", "lead_sensor": "pH", "start_pos": 50, "end_pos": 54},
        {"source": "injected contaminant", "kind": "water", "type": "biological", "level": "low", "lead_sensor": "Cl", "start_pos": 70, "end_pos": 74},
    ])
    water = np.zeros(100, dtype=bool)
    sensor = np.zeros(100, dtype=bool)
    water[12] = True                       # real event caught after 2 min
    sensor[30] = True                      # dropout: sensor alert only
    water[52] = True                       # stuck: false water alarm only
    sensor[71] = True                      # contaminant sent to the sensor
    names = pd.Series([""] * 100)
    names[30] = "Cl"
    out = TE.episode_outcomes(eps, water, sensor, names, grace=5)
    assert out["outcome"].tolist() == ["water alarm", "sensor alert only", "water alarm only", "sensor alert only"]
    assert out.loc[0, "delay_min"] == 2 and out.loc[1, "names_right_sensor"] is True
    s = TE.summarise(out)
    assert (s["real_caught"], s["fault_correct"], s["fault_raised_water_alarm"], s["contaminant_sent_to_sensor"]) == (1, 1, 1, 1)


def test_rules_withdraw_alarm_explained_by_a_dropout(ds):
    inj, episodes = build("injector", "gecco_triage", contaminants=["biological"], doses=["high"],
                          faults=["dropout"], severities=["high"], reps_test=1, reps_pretest=1,
                          buffer_minutes=60, seed=1).build(ds)
    split = build("splitter", "official", cv_folds=2, gap_minutes=60).split(inj)
    basic = build("feature", "basic").transform(inj.frame, inj.sensors)
    ctx = Context(inj, split, inj.frame[CLASS_COL].to_numpy(), basic, 0, ds.frame[EVENT_COL].to_numpy())
    det = WaterDetector(n_estimators=10)
    det.fit(ctx)
    det.scores = np.zeros(len(inj.frame))                 # force: the detector fires as the reading drops
    d = episodes[(episodes["period"] == "test") & (episodes["kind"] == "fault")].iloc[0]
    a, b = d["start_pos"], d["end_pos"]
    det.scores[a:a + 10] = 1.0
    det.threshold = 0.5
    out = RuleTriage(detector=det).run(ctx)
    assert out.sensor[a:b + 1].all()                      # "check the sensor" from the first minute
    assert not out.water[a:a + 10].any()                  # and the detector's alarm is withdrawn
    assert d["lead_sensor"] in out.sensor_names.iloc[a]


def test_cli_has_triage_and_inject():
    p = build_parser()
    assert p.parse_args(["triage", "--config", "s4_triage"]).command == "triage"
    assert p.parse_args(["inject", "--config", "s4_triage"]).command == "inject"
