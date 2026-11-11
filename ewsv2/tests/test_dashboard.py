"""Dashboard (Segment 9): the minute-by-minute engine, the live station, the replay, and every page.

Pages run headless with Streamlit's test runner, with the board switched off (EWSV2_NO_BOARD=1):
the dashboard must work with no hardware, no LLM and – via the simulated station – no data files.
"""

import time

import numpy as np
import pytest

from ewsv2.cli import build_parser
from ewsv2.core.types import EVENT_COL
from ewsv2.edge import reference as R
from ewsv2.edge.stream import EdgeStream
from ewsv2.live.station import LiveStation, Reading, SimProbe

from .test_edge import stream, tiny_spec


def _run_stream(C, spec, y, int8=True, adaptive=True):
    es = EdgeStream(spec, int8, adaptive)
    alarm = np.zeros(len(C), dtype=bool)
    out = []
    for t in range(len(C)):
        m = es.push([None if not np.isfinite(v) else int(v) for v in C[t]])
        alarm[t] = m.alarm
        out.append(m)
        if m.ask:
            es.verdict(R.false_count(alarm[m.ep_start:m.last_on + 1], y[m.ep_start:m.last_on + 1]))
    return out


@pytest.mark.parametrize("int8,adaptive", [(True, True), (False, False)])
def test_edge_stream_is_bit_identical_to_batch_reference(int8, adaptive):
    ds, f = stream()
    spec = tiny_spec(ds.sensors)
    C = R.to_counts(f, spec)
    y = f[EVENT_COL].to_numpy().astype(bool)
    out = _run_stream(C, spec, y, int8, adaptive)
    ref = R.run_counts(C, spec, int8, y, adaptive)
    lg = np.array([m.logit for m in out], dtype=np.float32)
    assert (lg.view(np.uint32) == ref["logit"].to_numpy().view(np.uint32)).all()
    th = np.array([m.theta for m in out], dtype=np.float32)
    rt = ref["theta"].to_numpy()
    assert np.array_equal(np.isnan(th), np.isnan(rt)) and (th[~np.isnan(th)] == rt[~np.isnan(rt)]).all()
    for c in ("alarm", "water", "sensor", "mask", "top"):
        assert (np.array([getattr(m, c) for m in out]) == ref[c].to_numpy()).all(), c


def test_live_station_water_event_sensor_fault_and_false_alarm():
    st, sp, t = LiveStation(), SimProbe(seed=3), 1000.0
    outs = []
    for i in range(110):
        if i == 50:
            sp.add_salt()
        if i == 80:
            sp.pull_out()
        if i == 90:
            sp.put_back()
        if i == 100:
            sp.touch()
        r = sp.read()
        r.t = t + 2 * i
        outs.append(st.push(r))
    kinds = [e.kind for e in st.events]
    assert kinds[0] == "info" and "water" in kinds and "sensor" in kinds and kinds[-1] == "verdict"
    assert not any(o["water"] for o in outs[20:50])                     # calm water: no alarm
    assert any(o["water"] for o in outs[50:56])                         # salt: water alarm within seconds
    assert all(o["sensor"] and not o["water"] for o in outs[80:90])     # probe out: sensor alert only
    assert st.limit > st.limit0                                        # the touch raised the limit


def test_live_station_without_probe_reports_a_sensor_fault():
    st = LiveStation()
    st.push(Reading(1.0, 600))
    out = st.push(Reading(3.0, None))
    assert out["sensor"] and "not answering" in st.events[-1].text


def test_replay_on_the_simulated_station_with_an_operator():
    from ewsv2.app import services as S
    cfg = S.config()
    base = S.load_base("simulated", cfg)
    rs = S.ReplayState(**base)
    for _ in range(20000):
        if rs.step("you") is None:
            break
    assert rs.pending is not None or rs.done
    if rs.pending is not None:
        rs.answer(True)
        assert rs.pending is None and rs.step("you") is not None
    assert rs.episodes
    rec = S.episode_record(rs, rs.episodes[0], "test station")
    rep = S.write_report(rec, rs.sensors, use_llm=False, model="", host="")
    assert rep["shown"] == "template" and rep["template"].splitlines()[0].startswith(rec["decision"])


def test_board_service_without_a_board(monkeypatch):
    from ewsv2.edge import hil as H
    from ewsv2.live.board_service import BoardService
    monkeypatch.setattr(H, "candidate_ports", lambda: [])
    b = BoardService("auto")
    t0 = time.monotonic()
    assert b.connect() is False and b.read_once() is None and b.edge_minute([1] * 9) is None
    assert time.monotonic() - t0 < 2 and "no Arduino" in b.message


PAGES = ["overview", "how_it_works", "control_room", "live_lab", "results", "reports", "version1", "system"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_hardware(page, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("EWSV2_NO_BOARD", "1")
    script = tmp_path / f"{page}.py"
    script.write_text(
        "import streamlit as st\n"
        "from ewsv2.app import ui\n"
        f"from ewsv2.app.views import {page}\n"
        "ui.setup_page()\n"
        f"{page}.render()\n", encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "control_room":
        next(b for b in at.button if "Next alarm" in b.label).click()
        at.run()
        assert not at.exception
        rs = at.session_state["cr_state"]
        assert rs.pos > 0
        if rs.pending is not None:                         # the question box asks, with the answer key
            assert rs.pending["key"] in ("real event", "false alarm")
            assert any("ANSWER KEY" in m.value for m in at.markdown)
            next(b for b in at.button if "False alarm" in b.label).click()
            at.run()
            assert not at.exception
            rs = at.session_state["cr_state"]
            assert rs.pending is None and len(rs.answers) == 1


def test_next_alarm_answers_and_verdicts_line_up():
    """The operator's answer is stored on the alarm it was asked about, and scored against the answer key."""
    from ewsv2.app import services as S
    rs = S.ReplayState(**S.load_base("simulated", S.config()))
    for _ in range(3):
        n = 0
        while rs.pending is None and not rs.done and n < 30000:
            rs.step("you")
            n += 1
        if rs.pending is None:
            break
        p = rs.pending
        rs.answer(p["key"] == "false alarm")               # answer exactly like the key
        assert rs.answers[-1]["right"]
        assert (rs.answers[-1]["lift"] > 0) == (p["key"] == "false alarm")   # a false alarm lifts the line
        hit = [e for e in rs.episodes if e["kind"] == "water" and e["start"] <= p["last_on"] + 5
               and e["end"] >= p["ep_start"] - 5]
        assert all(e["verdict"] for e in hit)
    assert rs.asks == len(rs.answers) >= 1


def test_cli_has_dashboard():
    assert build_parser().parse_args(["dashboard"]).config == "s9_dashboard"
