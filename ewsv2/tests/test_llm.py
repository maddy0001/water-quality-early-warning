"""Alarm records, report writers and the fact checker (Segment 7). No LLM needed."""

import pandas as pd

from ewsv2.cli import build_parser
from ewsv2.core.registry import available, build
from ewsv2.core.types import EVENT_COL
from ewsv2.llm import faithfulness as FC
from ewsv2.llm.records import build_records, public, select
from ewsv2.llm.reporters import FunctionReporter, load_guidance, retrieve

from .test_inject import station

SENS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]

RECORD = {
    "id": "A001", "kind": "water", "station": "GECCO 2018 station", "decision": "WATER QUALITY ALARM",
    "level": "MEDIUM", "start": "2016-11-09 00:33", "end": "2016-11-09 00:56", "duration_min": 24,
    "move_window_min": 15, "sensors_beyond_normal": 2, "sensor_alerts_during": [],
    "sensors": [{"code": "pH", "description": "pH", "before": 8.38, "now": 8.12, "change": -0.26, "normal_units": 9.6},
                {"code": "Redox", "description": "redox potential", "before": 755.0, "now": 731.0, "change": -24.0,
                 "normal_units": 4.1}],
}


def _snips():
    return retrieve(RECORD, load_guidance())


def test_registered_and_guidance():
    assert {"template", "ollama"} <= set(available("reporter")["reporter"])
    g = load_guidance()
    assert {"W1", "W2", "W3", "W4", "P1", "P2"} <= {x["id"] for x in g}
    assert all("source" in x and "text" in x for x in g)
    ids = {x["id"] for x in _snips()}
    assert {"W4", "N2", "P1"} <= ids and "W3" not in ids and "P2" not in ids    # pH, redox, water alarm only


def test_template_is_faithful():
    out = build("reporter", "template").report(RECORD, _snips())
    assert FC.check(out["text"], RECORD, _snips(), SENS) == []
    assert out["text"].splitlines()[0] == "WATER QUALITY ALARM - level MEDIUM"


def test_checker_accepts_a_faithful_paraphrase():
    text = ("WATER QUALITY ALARM - level MEDIUM\n"
            "What happened: from 00:33 to 00:56 on 2016-11-09 (24 minutes) pH fell from 8.38 to 8.12 and the "
            "redox potential from 755 to 731 mV, two sensors beyond their normal 15-minute range.\n"
            "Likely cause: a change in the water, seen on two sensors at once.\n"
            "Check first: take a grab sample and compare it with the analysers [P1]; the optimum pH range is "
            "6.5 to 8 [W4].")
    assert FC.check(text, RECORD, _snips(), SENS) == []
    assert FC.mentions_top_sensor(text, RECORD, SENS)


def test_checker_catches_planted_errors():
    text = ("Water event detected\n"                                      # decision line missing
            "What happened: at 01:15 pH fell by 3.2% to 8.1 while turbidity rose.\n"   # wrong time, new number, other sensor
            "Check first: follow procedure [W9].")                        # citation not provided
    types = {e["type"] for e in FC.check(text, RECORD, _snips(), SENS)}
    assert types == {"decision", "time", "number", "sensor", "citation"}


def test_rounded_numbers_are_accepted_new_ones_are_not():
    ok = "WATER QUALITY ALARM - level MEDIUM\npH went from 8.4 to 8.1 (about 10 times its normal move)."
    errs = FC.check(ok, RECORD, _snips(), SENS)
    assert [e["found"] for e in errs] == []                  # 8.4, 8.1 are roundings; 10 is a small count
    bad = "WATER QUALITY ALARM - level MEDIUM\npH went from 8.38 to 7.95."
    assert [e["found"] for e in FC.check(bad, RECORD, _snips(), SENS)] == ["7.95"]


def test_function_reporter_and_summary():
    rep = FunctionReporter(lambda messages: "WATER QUALITY ALARM - level MEDIUM\npH changed at 00:33 [P1].")
    out = rep.report(RECORD, _snips())
    assert out["ok"] and FC.check(out["text"], RECORD, _snips(), SENS) == []
    msgs = rep._ollama.messages(RECORD, _snips())
    assert "truth" not in msgs[1]["content"] and "WATER QUALITY ALARM - level MEDIUM" in msgs[0]["content"]
    s = FC.summarise([[], [{"type": "number", "found": "3"}]])
    assert s["faithful_share"] == 0.5 and s["with_number_error"] == 0.5


def test_build_records_and_select():
    ds = station(days=8, seed=3)
    inj, episodes = build("injector", "gecco_triage", reps_test=1, reps_pretest=1, buffer_minutes=60,
                          contaminants=["biological"], doses=["high"], faults=["dropout"], severities=["high"],
                          seed=2).build(ds)
    n = len(inj.frame)
    minutes = pd.DataFrame({"water_alarm": False, "sensor_alert": False, "sensor_names": ""}, index=inj.frame.index)
    c = episodes[(episodes["period"] == "test") & (episodes["kind"] == "contaminant")].iloc[0]
    flt = episodes[(episodes["period"] == "test") & (episodes["kind"] == "fault")].iloc[0]
    minutes.iloc[c["start_pos"] + 5:c["start_pos"] + 20, 0] = True
    minutes.iloc[flt["start_pos"]:flt["start_pos"] + 30, 1] = True
    minutes.iloc[flt["start_pos"]:flt["start_pos"] + 30, 2] = flt["lead_sensor"]
    normal = (inj.frame.index < inj.periods["test"][0]) & (ds.frame[EVENT_COL].to_numpy() == 0)
    dev = build("explainer", "deviation", sensors=inj.sensors).fit(inj.frame, normal)
    recs = build_records(ds, inj, episodes, minutes, dev, "test station")
    kinds = [r["kind"] for r in recs]
    assert kinds.count("water") == 1 and kinds.count("sensor") == 1
    w = [r for r in recs if r["kind"] == "water"][0]
    assert "biological" in w["truth"] and w["sensors"][0]["code"] in c["sensors"].split(",")
    s = [r for r in recs if r["kind"] == "sensor"][0]
    assert s["sensors"][0]["code"] == flt["lead_sensor"] and s["level"] == "MAINTENANCE"
    assert "truth" not in public(w) and n > 0
    assert len(select(recs, 1)) == 1


def test_cli_has_report():
    assert build_parser().parse_args(["report", "--config", "s7_reports"]).command == "report"


def test_ollama_client_against_a_fake_server():
    """The HTTP calls match Ollama's API (/api/tags, /api/chat) – checked with a stand-in server."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(req)
            body = json.dumps({"message": {"role": "assistant",
                                           "content": "WATER QUALITY ALARM - level MEDIUM\npH fell [W4]."}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rep = build("reporter", "ollama", model="qwen2.5:3b", host=f"http://127.0.0.1:{srv.server_port}")
        assert rep.available() == (True, "ok")
        out = rep.report(RECORD, _snips())
        assert out["ok"] and FC.check(out["text"], RECORD, _snips(), SENS) == []
        assert seen["model"] == "qwen2.5:3b" and seen["stream"] is False
        assert seen["options"]["temperature"] == 0.0 and seen["options"]["seed"] == 42
        assert [m["role"] for m in seen["messages"]] == ["system", "user"]
        missing = build("reporter", "ollama", model="llama3.2:3b", host=f"http://127.0.0.1:{srv.server_port}")
        assert missing.available()[0] is False
    finally:
        srv.shutdown()
