"""Inspection numbers are right on data where we know the answer."""

import json

import pandas as pd

from ewsv2.cli import main
from ewsv2.core.registry import build
from ewsv2.data.inspect import event_response, event_table, inspect_dataset, longest_constant_run, runs


def test_runs_finds_contiguous_blocks():
    s = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 1], index=pd.date_range("2020", periods=9, freq="min")).astype(bool)
    r = runs(s)
    assert list(r["length"]) == [2, 1, 3]
    assert r["start"].iloc[2] == s.index[6]


def test_runs_empty():
    assert runs(pd.Series([False, False])).empty


def test_longest_constant_run():
    assert longest_constant_run(pd.Series([1, 1, 2, 2, 2, None, 2, 3])) == 4  # NaN skipped
    assert longest_constant_run(pd.Series([], dtype=float)) == 0


def test_event_table_and_response(gecco_dir):
    ds = build("source", "gecco2018", data_dir=str(gecco_dir)).load()
    ev = event_table(ds)
    assert list(ev["length"]) == [10, 5, 20]
    assert list(ev["period"]) == ["train", "train", "test"]
    resp = event_response(ds.frame, ds.sensors, ev, baseline_minutes=20)
    # the synthetic events move pH and Redox only
    assert set(resp.index[:2]) == {"pH", "Redox"}
    assert resp.loc["pH", "share_events_shift_gt_threshold"] == 1.0
    assert resp.loc["Tp", "share_events_shift_gt_threshold"] == 0.0


def test_inspect_writes_all_outputs(gecco_dir, tmp_path):
    ds = build("source", "gecco2018", data_dir=str(gecco_dir)).load()
    out = tmp_path / "report"
    summary = inspect_dataset(ds, out, baseline_minutes=20)
    for f in ("report.md", "summary.json", "events.csv", "sensors.csv", "event_response.csv",
              "correlations.csv", "figures/overview.png", "figures/event_lengths.png",
              "figures/event_response.png", "figures/correlations.png"):
        assert (out / f).is_file(), f
    assert summary["events_total"] == 3
    assert summary["sampling"]["rows_all_sensors_missing"] == 10
    assert summary["sampling"]["longest_all_missing_run_minutes"] == 10
    assert json.loads((out / "summary.json").read_text())["periods"]["test"]["events"] == 1
    assert "## Which sensors react during events" in (out / "report.md").read_text(encoding="utf-8")


def test_cli_inspect_end_to_end(gecco_dir, tmp_path, monkeypatch, capsys):
    import ewsv2.core.paths as paths

    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "outputs_v2")
    code = main(["inspect", "--config", "s0_inspect",
                 "--set", f"data.params.data_dir={gecco_dir}", "--set", "inspect.plots=false"])
    assert code == 0
    assert "Events total : 3" in capsys.readouterr().out
    assert (tmp_path / "outputs_v2" / "s0_inspection" / "gecco2018" / "report.md").is_file()


def test_cli_reports_missing_data_cleanly(tmp_path, capsys):
    code = main(["inspect", "--config", "s0_inspect", "--set", f"data.params.data_dir={tmp_path}"])
    assert code == 2
    assert "fetch gecco2018" in capsys.readouterr().err
