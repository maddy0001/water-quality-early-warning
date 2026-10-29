"""Generic CSV source: any file with a time column and numeric sensor columns.

Use it for new data (another utility, a USGS export, an ESP32 log) without
writing code::

    data:
      source: csv
      params:
        path: data/my_station.csv
        sensors: [pH, Leit, Trueb]      # optional; default = all numeric columns
        test_start: "2024-06-01"        # optional; creates train/test periods
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ewsv2.core.paths import resolve
from ewsv2.core.registry import register
from ewsv2.core.types import Dataset
from ewsv2.data.base import read_table, standardise


@register("source", "csv")
class CsvSource:
    name = "csv"

    def __init__(
        self,
        path: str | Path,
        sensors: list[str] | None = None,
        test_start: str | None = None,
        name: str | None = None,
    ):
        self.path = resolve(path)
        self.sensors = sensors
        self.test_start = pd.Timestamp(test_start) if test_start else None
        if name:
            self.name = name

    def load(self) -> Dataset:
        if not self.path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.path}")
        frame, sensors, notes = standardise(read_table(self.path), self.sensors)
        periods = {"all": (frame.index.min(), frame.index.max())}
        if self.test_start is not None:
            before = frame.index[frame.index < self.test_start]
            if len(before) == 0 or self.test_start > frame.index.max():
                raise ValueError(f"test_start {self.test_start} is outside the data range")
            periods = {
                "train": (frame.index.min(), before.max()),
                "test": (self.test_start, frame.index.max()),
            }
        return Dataset(
            name=self.name,
            frame=frame,
            sensors=sensors,
            periods=periods,
            meta={"files": [str(self.path)], "load_notes": notes},
        )
