"""Baseline signals (Segment 1).

Every signal follows the three-step contract in ``core/interfaces.py``:
``transform`` the whole stream causally, ``fit`` on the fit rows, ``score`` any rows.
Scores are mapped to [0, 1] by the empirical distribution of the fit-period scores
(0.99 = more abnormal than 99% of fit minutes), so thresholds are comparable across
signals. Rows with no data score NaN and can never raise an alarm.

Why these baselines:

* ``random_score``    – a control. Under point adjustment even random scores look good
                        (Kim et al., AAAI 2022); the audit shows it on GECCO.
* ``squared_diff``,
  ``moving_variance`` – the "one-liner" baselines that matched time-series foundation
                        models in an ICLR 2026 study.
* ``rolling_zscore``  – the classic utility rule: how far is each sensor from its recent
                        mean, in recent standard deviations.
* ``static_limits``   – fixed per-sensor limits learned from the fit period (median ± MAD).
* ``isolation_forest``– a standard unsupervised detector on the basic features.
* ``random_forest``   – the supervised model from the leakage pilot (needs labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ewsv2.core.registry import build, register


class BaseSignal:
    """Shared plumbing: feature filling, empirical scaling, missing rows."""

    name = "base"
    supervised = False
    scale_scores = True

    def __init__(self, sensors: list[str] | None = None, seed: int = 42):
        self.sensors = sensors
        self.seed = seed
        self._ref: np.ndarray | None = None

    # -- override these ------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self._sensors(frame)].astype(float)

    def _fit(self, Z: pd.DataFrame, y: np.ndarray | None) -> None:
        pass

    def _raw(self, Z: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # -- shared ----------------------------------------------------------------
    def _sensors(self, frame: pd.DataFrame) -> list[str]:
        if self.sensors:
            return list(self.sensors)
        return [c for c in frame.columns if c not in ("EVENT", "LABEL_CLASS")]

    def fit(self, Z: pd.DataFrame, y=None) -> BaseSignal:
        rows = ~Z.isna().all(axis=1)
        Zf = Z[rows]
        self._fill = Zf.median()
        yv = None if y is None else np.asarray(y)[rows.to_numpy()]
        self._fit(Zf.fillna(self._fill), yv)
        if self.scale_scores:
            raw = self._raw(Zf.fillna(self._fill))
            self._ref = np.sort(raw[np.isfinite(raw)])
        return self

    def score(self, Z: pd.DataFrame) -> pd.Series:
        rows = ~Z.isna().all(axis=1).to_numpy()
        out = np.full(len(Z), np.nan)
        if rows.any():
            raw = self._raw(Z[rows].fillna(self._fill))
            if self.scale_scores and self._ref is not None and len(self._ref):
                raw = np.searchsorted(self._ref, raw, side="right") / len(self._ref)
            out[rows] = raw
        return pd.Series(out, index=Z.index, name=self.name)


def _robust_scale(values: pd.DataFrame) -> pd.Series:
    """Per-column typical size, robust to outliers (median absolute value, floored)."""
    s = values.abs().median()
    return s.where(s > 1e-12, values.abs().mean()).replace(0, 1.0)


@register("signal", "random_score")
class RandomScore(BaseSignal):
    name = "random_score"
    scale_scores = False

    def _raw(self, Z):
        return np.random.default_rng(self.seed).random(len(Z))


@register("signal", "squared_diff")
class SquaredDiff(BaseSignal):
    """max over sensors of (x_t − x_{t−1})², scaled per sensor."""

    name = "squared_diff"

    def transform(self, frame):
        x = frame[self._sensors(frame)].astype(float)
        empty = x.isna().all(axis=1).to_numpy()
        z = x.ffill(limit=5).diff() ** 2
        z.loc[empty] = np.nan
        return z

    def _fit(self, Z, y):
        self._scale = _robust_scale(Z) + 1e-12

    def _raw(self, Z):
        return (Z / self._scale).max(axis=1).to_numpy()


@register("signal", "moving_variance")
class MovingVariance(BaseSignal):
    """max over sensors of the variance in the last ``window`` minutes, scaled per sensor."""

    name = "moving_variance"

    def __init__(self, window: int = 10, **kw):
        super().__init__(**kw)
        self.window = int(window)

    def transform(self, frame):
        x = frame[self._sensors(frame)].astype(float)
        empty = x.isna().all(axis=1).to_numpy()
        z = x.ffill(limit=5).rolling(self.window, min_periods=2).var()
        z.loc[empty] = np.nan
        return z

    def _fit(self, Z, y):
        self._scale = _robust_scale(Z) + 1e-12

    def _raw(self, Z):
        return (Z / self._scale).max(axis=1).to_numpy()


@register("signal", "rolling_zscore")
class RollingZScore(BaseSignal):
    """max over sensors of |x − mean of previous ``window`` min| / their SD."""

    name = "rolling_zscore"

    def __init__(self, window: int = 60, **kw):
        super().__init__(**kw)
        self.window = int(window)

    def transform(self, frame):
        x = frame[self._sensors(frame)].astype(float)
        empty = x.isna().all(axis=1).to_numpy()
        x = x.ffill(limit=5)
        past = x.shift(1).rolling(self.window, min_periods=self.window // 3)
        sd = past.std()
        floor = sd.median() * 0.1 + 1e-9          # avoid huge z when a sensor is flat
        z = (x - past.mean()).abs() / sd.clip(lower=floor, axis=1)
        z.loc[empty] = np.nan
        return z

    def _raw(self, Z):
        return Z.max(axis=1).to_numpy()


@register("signal", "static_limits")
class StaticLimits(BaseSignal):
    """max over sensors of |x − fit median| / fit MAD (fixed limits, no time context)."""

    name = "static_limits"

    def _fit(self, Z, y):
        self._med = Z.median()
        mad = (Z - self._med).abs().median()
        self._mad = mad.where(mad > 1e-12, (Z - self._med).abs().mean()).replace(0, 1.0)

    def _raw(self, Z):
        return ((Z - self._med).abs() / self._mad).max(axis=1).to_numpy()


@register("signal", "isolation_forest")
class IForest(BaseSignal):
    """Isolation Forest on the basic causal features (unsupervised)."""

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, max_samples: int | str = 256, **kw):
        super().__init__(**kw)
        self.n_estimators, self.max_samples = int(n_estimators), max_samples

    def transform(self, frame):
        return build("feature", "basic").transform(frame, self._sensors(frame))

    def _fit(self, Z, y):
        self._model = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                      random_state=self.seed, n_jobs=-1).fit(Z.to_numpy())

    def _raw(self, Z):
        return -self._model.score_samples(Z.to_numpy())


@register("signal", "random_forest")
class RForest(BaseSignal):
    """Random Forest classifier on the basic features (supervised; needs event labels)."""

    name = "random_forest"
    supervised = True
    scale_scores = False  # already a probability

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 1, max_features: str = "sqrt", **kw):
        super().__init__(**kw)
        self.n_estimators, self.min_samples_leaf, self.max_features = int(n_estimators), int(min_samples_leaf), max_features

    def transform(self, frame):
        return build("feature", "basic").transform(frame, self._sensors(frame))

    def _fit(self, Z, y):
        if y is None:
            raise ValueError("random_forest is supervised: it needs event labels")
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, class_weight="balanced_subsample",
            random_state=self.seed, n_jobs=-1,
        ).fit(Z.to_numpy(), y.astype(int))

    def _raw(self, Z):
        return self._model.predict_proba(Z.to_numpy())[:, 1]
