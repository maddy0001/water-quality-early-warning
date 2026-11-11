"""Segment 10 – break each part on purpose and check that the system degrades cleanly instead of failing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewsv2.core.config import load_config


# ---------------------------------------------------------------------------------------------- data

def _cfg(source: str, data_dir: Path) -> dict:
    cfg = load_config("s8_edge")
    cfg["data"]["source"] = source
    cfg["data"]["params"] = {"data_dir": str(data_dir)}
    cfg["data"]["fallback"] = "simulated"
    return cfg


def test_missing_data_falls_back_to_the_synthetic_station(tmp_path, capsys):
    from ewsv2.cli import _load_dataset
    ds = _load_dataset(_cfg("gecco2018", tmp_path))
    assert ds.name == "simulated"
    assert "using the 'simulated' source instead" in capsys.readouterr().out


def test_damaged_data_falls_back_too(tmp_path, capsys):
    from ewsv2.cli import _load_dataset
    for name in ("gecco2018_train.csv", "gecco2018_test.csv"):
        (tmp_path / name).write_text("this is not,a sensor file\n\x00\x01garbage\n", encoding="utf-8")
    ds = _load_dataset(_cfg("gecco2018", tmp_path))
    assert ds.name == "simulated"
    assert "unreadable" in capsys.readouterr().out


def test_missing_data_without_fallback_is_a_clear_error(tmp_path):
    from ewsv2.cli import _load_dataset
    cfg = _cfg("gecco2018", tmp_path)
    cfg["data"]["fallback"] = None
    with pytest.raises(FileNotFoundError):
        _load_dataset(cfg)


def test_unknown_config_gives_exit_code_2_not_a_crash():
    from ewsv2.cli import main
    assert main(["inspect", "--config", "no_such_config_anywhere"]) == 2


# ---------------------------------------------------------------------------------------------- AI writer

def _record():
    from ewsv2.app import services as S
    rs = S.ReplayState(**S.load_base("simulated", S.config()))
    for _ in range(20000):
        if rs.step("auto") is None or any(e["kind"] == "water" for e in rs.episodes):
            break
    ep = next(e for e in rs.episodes if e["kind"] == "water")
    return S.episode_record(rs, ep, "test station"), rs.sensors


def test_unreachable_llm_gives_the_template_report():
    from ewsv2.app import services as S
    rec, sensors = _record()
    rep = S.write_report(rec, sensors, use_llm=True, model="qwen2.5:3b", host="http://127.0.0.1:9")
    assert rep["shown"] == "template" and rep["template"] and rep["why"]


def test_llm_report_with_a_wrong_fact_is_never_shown(monkeypatch):
    from ewsv2.app import services as S
    rec, sensors = _record()
    good = S.write_report(rec, sensors, False, "", "")["template"]
    wrong = good.replace(rec["start"][-5:], "23:59") if rec["start"][-5:] != "23:59" else good.replace("23:59", "00:01")

    class Liar:
        def available(self):
            return True, ""

        def report(self, record, snippets):
            return {"ok": True, "text": wrong}

    real_build = S.build
    monkeypatch.setattr(S, "build", lambda kind, name, **kw: Liar() if name == "ollama" else real_build(kind, name, **kw))
    rep = S.write_report(rec, sensors, True, "fake", "fake")
    assert rep["errors"], "the fact check should catch the changed time"
    assert rep["shown"] == "template"


# ---------------------------------------------------------------------------------------------- board

def test_board_on_a_port_that_does_not_exist_fails_fast():
    import time
    from ewsv2.live.board_service import BoardService
    b = BoardService("COM_DOES_NOT_EXIST_99")
    t0 = time.monotonic()
    assert b.connect() is False
    assert b.read_once() is None and b.edge_minute([1] * 9) is None
    assert time.monotonic() - t0 < 5


# ---------------------------------------------------------------------------------------------- dashboard

def _render(page: str, tmp_path: Path, monkeypatch, outputs: Path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from ewsv2.app import services as S
    monkeypatch.setenv("EWSV2_NO_BOARD", "1")
    monkeypatch.setattr(S, "OUTPUT_DIR", outputs)
    script = tmp_path / f"{page}.py"
    script.write_text("import streamlit as st\nfrom ewsv2.app import ui\n"
                      f"from ewsv2.app.views import {page}\nui.setup_page()\n{page}.render()\n", encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=300)
    at.run()
    return at


@pytest.mark.parametrize("page", ["overview", "analytics", "reports", "system"])
def test_pages_work_without_any_result_files(page, tmp_path, monkeypatch):
    empty = tmp_path / "no_outputs"
    empty.mkdir()
    at = _render(page, tmp_path, monkeypatch, empty)
    assert not at.exception, [e.value for e in at.exception]


def test_analytics_survives_damaged_result_files(tmp_path, monkeypatch):
    bad = tmp_path / "bad_outputs"
    for sub in ("s1_audit/gecco2018", "s1_baselines/gecco2018", "s4_triage/gecco2018", "s5_fusion/gecco2018",
                "s6_explain/gecco2018", "s7_reports/gecco2018", "s8_edge/gecco2018"):
        d = bad / sub
        d.mkdir(parents=True)
        for f in ("results.csv", "summary.csv", "outcomes.csv", "fusion_results.csv", "weekly_false_alarms.csv",
                  "explanation_summary.csv", "report_checks.csv", "edge_results.csv", "per_event.csv"):
            (d / f).write_text("wrong,columns\n1,2\n", encoding="utf-8")
    (bad / "s7_reports" / "gecco2018" / "reports.json").write_text("{not json", encoding="utf-8")
    for page in ("analytics", "reports"):
        at = _render(page, tmp_path, monkeypatch, bad)
        assert not at.exception, (page, [e.value for e in at.exception])


# ---------------------------------------------------------------------------------------------- hand-over tools

def test_doctor_runs_in_a_broken_environment(tmp_path, monkeypatch):
    from ewsv2.handover import doctor as D
    monkeypatch.setattr(D, "DATA_DIR", tmp_path / "no_data")
    monkeypatch.setattr(D, "OUTPUT_DIR", tmp_path / "no_outputs")
    monkeypatch.setattr(D, "REPO_ROOT", tmp_path)
    checks = D.run(board=False, llm=False)
    by = {c.name: c for c in checks}
    assert by["Utility data A (GECCO 2018)"].status == "degraded"
    assert by["Experiment results"].status == "degraded"
    assert by["Version 1 environment (venv)"].status == "degraded"
    assert all(c.fix for c in checks if c.status == "degraded")
    text = D.render(checks)
    assert "READY" in text
    D.write_report(checks, tmp_path / "rep")
    assert json.loads((tmp_path / "rep" / "doctor.json").read_text())["checks"]


def test_doctor_command_exit_code():
    from ewsv2.cli import main
    assert main(["doctor", "--no-board", "--no-llm"]) in (0, 1)


def test_capsule_build_check_and_tamper_detection(tmp_path, monkeypatch):
    from ewsv2.handover import capsule as K
    repo = tmp_path / "repo"
    (repo / "ewsv2").mkdir(parents=True)
    (repo / "ewsv2" / "a.py").write_text("print(1)\n")
    (repo / "venv2" / "Lib").mkdir(parents=True)                    # must be left out
    (repo / "venv2" / "Lib" / "big.bin").write_bytes(b"0" * 1000)
    monkeypatch.setattr(K, "REPO_ROOT", repo)
    dest = tmp_path / "capsule"
    assert K.build(dest, wheels=False) == 0
    assert (dest / "project" / "ewsv2" / "a.py").exists() and not (dest / "project" / "venv2").exists()
    for f in ("RESTORE.md", "restore.ps1", "manifest.json", "locks/requirements-lock-v2.txt",
              "locks/requirements-lock-v1.txt"):
        assert (dest / f).exists(), f
    assert K.check(dest) == 0
    (dest / "project" / "ewsv2" / "a.py").write_text("print(2)\n")
    assert K.check(dest) == 1
    (dest / "project" / "ewsv2" / "a.py").unlink()
    assert K.check(dest) == 1


def test_lock_files_are_complete_and_pinned():
    from ewsv2.handover import doctor as D
    for name in ("requirements-lock-v2.txt", "requirements-lock-v1.txt"):
        lock = D.read_lock(D.LOCK_DIR / name)
        assert len(lock) > 50 and all(v for v in lock.values())
    v2 = D.read_lock(D.LOCK_DIR / "requirements-lock-v2.txt")
    for req in ("numpy", "pandas", "streamlit", "plotly", "tensorflow-cpu", "streamlit-echarts", "pyserial"):
        assert req in v2


def test_paper_tables_and_figures_without_result_files(tmp_path, monkeypatch):
    from ewsv2.handover import paper as P
    monkeypatch.setattr(P, "OUTPUT_DIR", tmp_path / "none")
    out = P.build(tmp_path / "paper")
    assert (out / "tables.md").exists() and (out / "tables.tex").exists()
    assert (out / "figures" / "F1_architecture.png").exists()
    assert (out / "tables" / "T3.csv").exists() and (out / "tables" / "T9.csv").exists()
