"""Plug-in contracts.

Each layer of the architecture has one small interface. A new module only has
to follow its interface and register itself; nothing else in the pipeline
changes. These are ``typing.Protocol`` classes, so a plug-in does not need to
inherit from them – it just needs the same methods.

Segment where each is first implemented:

    DataSource  S0   Splitter  S1   Metric     S1
    Signal      S1 (baselines), S2-S4   Fusion    S5   Calibrator S5
    Triage      S4   Explainer S6   Reporter   S7   Output S8/S9
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ewsv2.core.types import Alarm, Dataset, SignalOutput, Split


@runtime_checkable
class DataSource(Protocol):
    """Loads a sensor stream into the common :class:`Dataset` format."""

    name: str

    def load(self) -> Dataset: ...


@runtime_checkable
class Splitter(Protocol):
    """Decides which rows fit the model, set thresholds, and test it.

    Honest splitters keep time order and leave a gap; leaky ones (random) exist
    only so the audit can show how much they inflate scores.
    """

    def split(self, dataset: Dataset) -> Split: ...


@runtime_checkable
class Signal(Protocol):
    """Forecast surprise, detector vote, event classifier or fault check.

    Three steps, so that time-based features are always computed on the full,
    unbroken stream (causally: each row only sees the past), whatever the split:

    1. ``transform(frame)`` – per-row representation of the whole stream
    2. ``fit(Z_fit, y_fit)`` – learn from the fit rows only
    3. ``score(Z)``          – abnormality per row (higher = more abnormal)
    """

    name: str

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def fit(self, Z: pd.DataFrame, y: pd.Series | None = None) -> Signal: ...

    def score(self, Z: pd.DataFrame) -> pd.Series: ...


@runtime_checkable
class Fusion(Protocol):
    """Combines several signal outputs into one score in [0, 1]."""

    def fit(self, signals: list[SignalOutput], labels: pd.Series) -> Fusion: ...

    def combine(self, signals: list[SignalOutput]) -> pd.Series: ...


@runtime_checkable
class Calibrator(Protocol):
    """Turns a score into alarm / no alarm under a false-alarm budget."""

    def fit(self, scores: pd.Series) -> Calibrator: ...

    def alarm(self, scores: pd.Series) -> pd.Series: ...


@runtime_checkable
class Triage(Protocol):
    """Decides Normal / Water event / Sensor fault and the alert level."""

    def decide(
        self, frame: pd.DataFrame, fused: pd.Series, signals: list[SignalOutput]
    ) -> list[Alarm]: ...


@runtime_checkable
class Explainer(Protocol):
    """Says which sensors drove an alarm."""

    def explain(self, alarm: Alarm, frame: pd.DataFrame) -> dict[str, float]: ...


@runtime_checkable
class Reporter(Protocol):
    """Writes an incident report (template or LLM). Reads alarms, never sets them."""

    def report(self, alarm: Alarm) -> dict[str, Any]: ...


@runtime_checkable
class Metric(Protocol):
    """Scores predictions against labels (event recall, delay, false alarms/day...)."""

    name: str

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float: ...
