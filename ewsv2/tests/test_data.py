"""Loaders turn raw files into the common format correctly."""

import numpy as np
import pandas as pd
import pytest

from ewsv2.core.paths import DATA_DIR
from ewsv2.core.registry import build
from ewsv2.core.types import CLASS_COL, EVENT_COL
from ewsv2.data.base import DataFormatError, standardise
from ewsv2.data.gecco import GECCO2018_SENSORS, GECCO2018_TEST_START

from .conftest import SENSORS, make_stream


def test_gecco2018_loads_and_splits(gecco_dir):
    ds = build("source", "gecco2018", data_dir=str(gecco_dir)).load()
    assert ds.sensors == GECCO2018_SENSORS
    assert len(ds.frame) == 240
    assert ds.frame.index.is_monotonic_increasing and ds.frame.index.is_unique
    assert ds.periods["test"][0] == GECCO2018_TEST_START
    assert int(ds.period("train")[EVENT_COL].sum()) == 15
    assert int(ds.period("test")[EVENT_COL].sum()) == 20
    assert (ds.frame[CLASS_COL] == ds.frame[EVENT_COL]).all()
    assert ds.frame[EVENT_COL].dtype.kind == "i"


def test_merged_file_is_split_at_official_boundary(tmp_path):
    a = make_stream("2016-11-08 06:55:00", 120, [(10, 5)])
    a.to_csv(tmp_path / "1_gecco2018_water_quality.csv", sep=";", index=False)  # Zenodo-style name, ; separator
    ds = build("source", "gecco2018", data_dir=str(tmp_path)).load()
    assert ds.periods["train"][1] == GECCO2018_TEST_START - pd.Timedelta(minutes=1)
    assert len(ds.period("test")) == 60


def test_missing_data_message_tells_user_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m ewsv2 fetch gecco2018"):
        build("source", "gecco2018", data_dir=str(tmp_path)).load()


def _gecco2019_like(start: str, minutes: int) -> pd.DataFrame:
    df = make_stream(start, minutes, [(5, 5)])
    df = df.rename(columns={"Leit": "Cond", "Trueb": "Turb", "Fm": "PFM", "Cl": "SAC", "EVENT": "Event"})
    df.insert(0, "", range(1, len(df) + 1))          # Zenodo files start with an R row-number column
    return df[["", "Time", "Tp", "pH", "Cond", "Turb", "SAC", "PFM", "Event"]]


def test_gecco2019_all_file_split_at_official_boundaries(tmp_path):
    from ewsv2.data.gecco import GECCO2019_TEST_START, GECCO2019_VALID_START

    # 2017-09-30 23:00 to 2017-11-08 01:29 crosses both boundaries
    minutes = int((pd.Timestamp("2017-11-08 01:30") - pd.Timestamp("2017-09-30 23:00")) / pd.Timedelta("1min"))
    _gecco2019_like("2017-09-30 23:00", minutes).to_csv(tmp_path / "5_gecco2019_all_water_quality.csv", index=False)
    ds = build("source", "gecco2019", data_dir=str(tmp_path)).load()
    assert ds.sensors == ["Tp", "pH", "Cond", "Turb", "SAC", "PFM"]
    assert list(ds.periods) == ["train", "valid", "test"]
    assert ds.periods["valid"][0] == GECCO2019_VALID_START
    assert ds.periods["test"][0] == GECCO2019_TEST_START
    assert len(ds.period("train")) == 60 and len(ds.period("test")) == 60
    assert "incomplete" not in ds.meta["load_notes"]


def test_gecco2019_prefers_all_file_over_parts(tmp_path):
    """The exact set downloaded by hand: 1_ (train), 5_ (all), 6_ (test)."""
    full = _gecco2019_like("2017-11-08 00:00", 120)
    full.to_csv(tmp_path / "5_gecco2019_all_water_quality.csv", index=False)
    full.iloc[:30].to_csv(tmp_path / "1_gecco2019_water_quality.csv", index=False)
    full.iloc[30:].to_csv(tmp_path / "6_gecco2019_test_water_quality.csv", index=False)
    ds = build("source", "gecco2019", data_dir=str(tmp_path)).load()
    assert len(ds.frame) == 120
    assert ds.meta["files"][0].endswith("5_gecco2019_all_water_quality.csv")
    assert set(ds.periods) == {"valid", "test"}


def test_gecco2019_test_file_only_is_flagged_incomplete(tmp_path):
    _gecco2019_like("2017-11-08 00:30", 60).to_csv(tmp_path / "6_gecco2019_test_water_quality.csv", index=False)
    ds = build("source", "gecco2019", data_dir=str(tmp_path)).load()
    assert list(ds.periods) == ["test"]
    assert "incomplete" in ds.meta["load_notes"]


def test_csv_source_with_test_start(tmp_path):
    make_stream("2024-01-01", 100, [(40, 10)]).to_csv(tmp_path / "station.csv", index=False)
    ds = build("source", "csv", path=str(tmp_path / "station.csv"), test_start="2024-01-01 01:00").load()
    assert len(ds.period("train")) == 60 and len(ds.period("test")) == 40


def test_standardise_handles_messy_input():
    raw = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3],
        "time": ["2020-01-01 00:01", "2020-01-01 00:00", "2020-01-01 00:01", "bad"],
        "pH": ["8.1", "8.0", "8.2", "8.3"],
        "event": ["TRUE", "FALSE", "true", "0"],
    })
    frame, sensors, notes = standardise(raw)
    assert sensors == ["pH"]                         # index column ignored
    assert list(frame[EVENT_COL]) == [0, 1]          # sorted, duplicate dropped
    assert notes["duplicate_timestamps_dropped"] == 1
    assert notes["rows_with_unparseable_time_dropped"] == 1
    assert frame["pH"].dtype == float


def test_standardise_rejects_bad_event_values():
    raw = pd.DataFrame({"time": ["2020-01-01"], "pH": [8.0], "EVENT": ["maybe"]})
    with pytest.raises(DataFormatError):
        standardise(raw)


@pytest.mark.skipif(not (DATA_DIR / "gecco2018_train.csv").is_file(), reason="real GECCO 2018 not downloaded")
def test_real_gecco2018_matches_published_counts():
    """The official data: 51 train events, 46 test events, 279,132 minutes."""
    from ewsv2.data.inspect import event_table

    ds = build("source", "gecco2018").load()
    ev = event_table(ds)
    assert len(ds.frame) == 279_132
    assert (ev["period"] == "train").sum() == 51
    assert (ev["period"] == "test").sum() == 46
    assert int(ds.frame[EVENT_COL].sum()) == 1_726 + 2_329
    assert set(SENSORS) == set(ds.sensors)
    assert np.isclose(ds.frame["pH"].median(), 8.34, atol=0.05)


_ALL_2019 = next(iter(DATA_DIR.glob("*gecco2019_all*")), None) if DATA_DIR.is_dir() else None


@pytest.mark.skipif(_ALL_2019 is None, reason="real GECCO 2019 'all' file not downloaded")
def test_real_gecco2019_matches_zenodo_files():
    """Checked against the separate Zenodo files: train 212, valid 225, test 191 event minutes."""
    ds = build("source", "gecco2019").load()
    assert len(ds.frame) == 218_880
    assert [int(ds.period(p)[EVENT_COL].sum()) for p in ("train", "valid", "test")] == [212, 225, 191]
    assert [len(ds.period(p)) for p in ("train", "valid", "test")] == [132_480, 54_750, 31_650]
