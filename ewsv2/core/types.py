"""Data containers used across ewsv2.

A sensor stream is a pandas DataFrame:

* index   : ``DatetimeIndex`` named ``time`` (sorted, unique)
* columns : one float column per sensor, plus label columns:

  - ``EVENT``        1 if the minute belongs to a labelled water-quality event, else 0
  - ``LABEL_CLASS``  0 = normal, 1 = water event, 2 = sensor fault
                     (added in Segment 4 by the injection toolkit; until then it
                     mirrors EVENT)

Keeping one simple, shared format is what lets every plug-in be swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import pandas as pd

TIME_COL = "time"
EVENT_COL = "EVENT"
CLASS_COL = "LABEL_CLASS"
LABEL_COLS = (EVENT_COL, CLASS_COL)


class TriageClass(IntEnum):
    """The three answers the v2 alarm can give."""

    NORMAL = 0
    WATER_EVENT = 1
    SENSOR_FAULT = 2


class AlertLevel(IntEnum):
    """Four operational levels, kept from paper 1."""

    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    CRITICAL = 3


@dataclass
class Dataset:
    """A loaded sensor stream plus what we know about it.

    Attributes
    ----------
    name     : short id, e.g. ``"gecco2018"``
    frame    : the stream (see module docstring)
    sensors  : sensor column names, in a fixed order
    periods  : official periods, e.g. ``{"train": (start, end), "test": (start, end)}``
    meta     : free-form facts (source file, licence, sensor descriptions, ...)
    """

    name: str
    frame: pd.DataFrame
    sensors: list[str]
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def period(self, name: str) -> pd.DataFrame:
        """Rows of one official period (inclusive on both ends)."""
        start, end = self.periods[name]
        return self.frame.loc[start:end]

    @property
    def labels(self) -> pd.Series:
        return self.frame[EVENT_COL]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        f = self.frame
        return (
            f"Dataset(name={self.name!r}, rows={len(f):,}, sensors={self.sensors}, "
            f"from={f.index.min()}, to={f.index.max()}, periods={list(self.periods)})"
        )


@dataclass
class SignalOutput:
    """What every signal (forecast, detector, classifier, fault check) returns.

    score      : one value per timestamp in [0, 1]; higher = more abnormal
    per_sensor : optional per-sensor contribution (same index, one column per sensor)
    meta       : anything the signal wants to report (thresholds, model name, ...)
    """

    name: str
    score: pd.Series
    per_sensor: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alarm:
    """One alarm raised by the decision layer (filled in from Segment 5 on)."""

    start: pd.Timestamp
    end: pd.Timestamp | None
    triage: TriageClass
    level: AlertLevel
    score: float
    sensors: list[str] = field(default_factory=list)
    values: dict[str, float] = field(default_factory=dict)
    explanation: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
