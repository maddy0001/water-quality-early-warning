"""GECCO Industrial Challenge drinking-water datasets (2018 and 2019).

Real one-minute sensor data from Thüringer Fernwasserversorgung (Germany),
with expert-labelled water-quality events. Licence: CC BY 4.0.

* 2018: Zenodo DOI 10.5281/zenodo.3884398 (also bundled in the R package
  EventDetectR, which is where ``python -m ewsv2 fetch gecco2018`` takes it from)
* 2019: Zenodo record 4304080 (train / validation / test files)

Sensor names are the utility's German abbreviations; ``SENSOR_INFO`` gives the
meaning. Check units against the Zenodo description before quoting them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ewsv2.core.paths import DATA_DIR, resolve
from ewsv2.core.registry import register
from ewsv2.core.types import Dataset
from ewsv2.data.base import DataFormatError, read_table, standardise

log = logging.getLogger(__name__)

SENSOR_INFO = {
    "Tp": "water temperature",
    "Cl": "chlorine dioxide, analyser 1",
    "Cl_2": "chlorine dioxide, analyser 2",
    "pH": "pH",
    "Redox": "redox potential",
    "Leit": "electrical conductivity (Leitfähigkeit)",
    "Trueb": "turbidity (Trübung)",
    "Fm": "flow rate, line 1",
    "Fm_2": "flow rate, line 2",
}

GECCO2018_SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]
# First minute of the official 2018 test period (end of train is one minute earlier).
GECCO2018_TEST_START = pd.Timestamp("2016-11-08 07:55:00")

LICENCE = "CC BY 4.0 (Thüringer Fernwasserversorgung / TH Köln SPOTSeven Lab)"


def _match_files(data_dir: Path, year: str) -> list[Path]:
    files = sorted(
        p for p in data_dir.glob("*")
        if p.is_file() and year in p.name and p.suffix.lower() in (".csv", ".rda", ".rds")
    )
    return files


def _missing_data_message(name: str, data_dir: Path) -> str:
    return (
        f"No {name} files found in {data_dir}.\n"
        f"Run:  python -m ewsv2 fetch {name}\n"
        f"or download the CSV files from Zenodo into that folder (see ewsv2/docs/SEGMENT_00.md)."
    )


def _concat(parts: list[tuple[str, pd.DataFrame]], sensors: list[str] | None):
    frames, notes_all, sensor_list = [], {}, None
    for label, raw in parts:
        frame, found, notes = standardise(raw, sensors)
        if sensor_list is None:
            sensor_list = found
        elif found != sensor_list:
            raise DataFormatError(f"Sensor columns differ between files: {sensor_list} vs {found}")
        frames.append((label, frame))
        if notes:
            notes_all[label] = notes
    return frames, sensor_list, notes_all


@register("source", "gecco2018")
class Gecco2018Source:
    """GECCO 2018: official train (Aug–Nov 2016) and test (Nov 2016–Feb 2017) periods.

    Looks in ``data_dir`` for, in order:
      1. ``gecco2018_train.csv`` + ``gecco2018_test.csv`` (written by ``ewsv2 fetch``)
      2. any other files with ``2018`` in the name (e.g. the Zenodo CSV or the
         EventDetectR ``.rda`` files); a single merged file is split at the
         official test start.
    """

    name = "gecco2018"

    def __init__(self, data_dir: str | Path | None = None, sensors: list[str] | None = None):
        self.data_dir = resolve(data_dir) if data_dir else DATA_DIR
        self.sensors = sensors or GECCO2018_SENSORS

    def load(self) -> Dataset:
        d = self.data_dir
        train_f, test_f = d / "gecco2018_train.csv", d / "gecco2018_test.csv"
        if train_f.is_file() and test_f.is_file():
            sources = [("train", train_f), ("test", test_f)]
        else:
            files = _match_files(d, "2018")
            if not files:
                raise FileNotFoundError(_missing_data_message(self.name, d))
            train = [f for f in files if "train" in f.name.lower()]
            test = [f for f in files if "test" in f.name.lower()]
            if len(train) == 1 and len(test) == 1:
                sources = [("train", train[0]), ("test", test[0])]
            else:
                sources = [("all", f) for f in files]

        parts = [(label, read_table(path)) for label, path in sources]
        frames, sensors, notes = _concat(parts, self.sensors)
        frame = pd.concat([f for _, f in frames]).sort_index()
        frame = frame[~frame.index.duplicated(keep="first")]

        train_end = frame.index[frame.index < GECCO2018_TEST_START].max()
        periods = {
            "train": (frame.index.min(), train_end),
            "test": (GECCO2018_TEST_START, frame.index.max()),
        }
        labelled = {label for label, _ in frames}
        if labelled == {"train", "test"}:
            # Trust the files, but confirm they agree with the documented boundary.
            first_test = frames[1][1].index.min()
            if first_test != GECCO2018_TEST_START:
                log.warning("Test file starts at %s, not %s", first_test, GECCO2018_TEST_START)
                periods["test"] = (first_test, frame.index.max())

        return Dataset(
            name=self.name,
            frame=frame,
            sensors=sensors,
            periods=periods,
            meta={
                "files": [str(p) for _, p in sources],
                "licence": LICENCE,
                "sensor_info": {s: SENSOR_INFO.get(s, "") for s in sensors},
                "sampling": "1 minute",
                "load_notes": notes,
            },
        )


@register("source", "gecco2019")
class Gecco2019Source:
    """GECCO 2019: separate train, validation and test files from Zenodo record 4304080.

    Expects files containing ``2019`` and ``train`` / ``valid`` / ``test`` in their
    names (as on Zenodo). A file with ``all`` in the name is ignored when the three
    split files are present.
    """

    name = "gecco2019"

    def __init__(self, data_dir: str | Path | None = None, sensors: list[str] | None = None):
        self.data_dir = resolve(data_dir) if data_dir else DATA_DIR
        self.sensors = sensors

    def load(self) -> Dataset:
        files = _match_files(self.data_dir, "2019")
        if not files:
            raise FileNotFoundError(_missing_data_message(self.name, self.data_dir))
        by_split = {}
        for split in ("train", "valid", "test"):
            hits = [f for f in files if split in f.name.lower() and "all" not in f.name.lower()]
            if len(hits) > 1:
                raise DataFormatError(f"More than one GECCO 2019 {split} file: {hits}")
            if hits:
                by_split[split] = hits[0]
        if not by_split:
            raise DataFormatError(
                f"Found {files} but none has train/valid/test in its name; rename them to match Zenodo."
            )

        parts = [(split, read_table(path)) for split, path in by_split.items()]
        frames, sensors, notes = _concat(parts, self.sensors)
        periods = {split: (f.index.min(), f.index.max()) for split, f in frames}
        frame = pd.concat([f for _, f in frames]).sort_index()
        dup = int(frame.index.duplicated().sum())
        if dup:
            notes["overlap_rows_between_files_dropped"] = dup
            frame = frame[~frame.index.duplicated(keep="first")]

        return Dataset(
            name=self.name,
            frame=frame,
            sensors=sensors,
            periods=periods,
            meta={
                "files": [str(p) for p in by_split.values()],
                "licence": LICENCE,
                "sensor_info": {s: SENSOR_INFO.get(s, "") for s in sensors},
                "sampling": "1 minute",
                "load_notes": notes,
            },
        )
