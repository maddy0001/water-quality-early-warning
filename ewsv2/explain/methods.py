"""Which sensors drove an alarm? Four explainers (Segment 6, contribution C6).

Every explainer returns, for chosen minutes, one score per sensor (higher = more
responsible). They answer the operator's first question – "what changed?" – and they are
checked, not just shown: against injected episodes where we *know* which sensors changed
(E9), and against each other with a deletion test.

``deviation``          model-free: how far each sensor moved in the last 15 minutes, in units
                       of its own largest normal 15-minute move (S4 triage evidence)
``forecast_residual``  the forecast-surprise module (S3): each sensor's 5-minute-ahead
                       forecast error, in units of its normal forecast error
``tree_shap``          the classifier module (S2): exact TreeSHAP contributions of its XGBoost
                       part (XGBoost's built-in ``pred_contribs``; paper 1 also used TreeSHAP),
                       summed over each sensor's features
``occlusion``          the classifier module again, model-agnostic: hold one sensor at its
                       level from before the alarm, recompute its features, re-score the full
                       RF + XGBoost + SVM ensemble; responsibility = how much the score drops.
                       Faithful by construction, and slow, so it also serves as the check of
                       the other three (deletion test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.core.registry import build, register
from ewsv2.signals.forecast import make_windows, window_positions
from ewsv2.triage.features import TriageFeatures


def _sensor_of(feature: str, sensors: list[str]) -> str | None:
    head = feature.split("__", 1)[0]
    return head if head in sensors else None


@register("explainer", "deviation")
class DeviationExplainer:
    name = "deviation"

    def __init__(self, sensors: list[str], **_):
        self.sensors = list(sensors)

    def fit(self, frame: pd.DataFrame, normal_rows, y=None) -> DeviationExplainer:
        self.tf = TriageFeatures(self.sensors).fit(frame, normal_rows)
        return self

    def explain(self, frame: pd.DataFrame, positions) -> pd.DataFrame:
        positions = np.asarray(positions, dtype=int)
        out = []
        for p in positions:
            lo = max(0, p - 30)
            z = self.tf._z15(frame.iloc[lo:p + 1]).iloc[-1].abs()
            out.append((z / self.tf.move_limit[self.sensors]).to_numpy())
        return pd.DataFrame(out, columns=self.sensors, index=positions)


@register("explainer", "forecast_residual")
class ForecastResidualExplainer:
    name = "forecast_residual"

    def __init__(self, sensors: list[str], lookback: int = 30, horizon: int = 5, seed: int = 42, **_):
        self.sensors, self.lookback, self.horizon, self.seed = list(sensors), int(lookback), int(horizon), seed

    def fit(self, frame, normal_rows, y=None):
        self.sig = build("signal", "forecast_linear", sensors=self.sensors, seed=self.seed,
                         lookback=self.lookback, horizon=self.horizon)
        Z = self.sig.transform(frame)
        rows = np.asarray(normal_rows, dtype=bool)
        self.sig.fit(Z[rows], np.zeros(int(rows.sum()), dtype=int))
        return self

    def explain(self, frame, positions) -> pd.DataFrame:
        positions = np.asarray(positions, dtype=int)
        out = np.full((len(positions), len(self.sensors)), np.nan)
        span = self.sig.span
        for i, p in enumerate(positions):
            lo = max(0, p - span - 5)
            Z = self.sig.transform(frame.iloc[lo:p + 1])
            V = self.sig._scaled(Z[self.sig.columns])
            pos = window_positions(Z.index, V, span)
            if len(pos) and pos[-1] == len(Z) - 1:
                X, Y = make_windows(V, pos[-1:], self.lookback, self.horizon)
                out[i] = (np.abs(Y - self.sig._predict(X)) / self.sig.err_scale)[0]
        return pd.DataFrame(out, columns=self.sensors, index=positions)


class ClassifierExplainer:
    """The classifier module (paper 1's ensemble), fitted once on pre-test data, explained two ways."""

    def __init__(self, sensors: list[str], seed: int = 42, hold_back: int = 15, window: int = 240):
        self.sensors, self.seed = list(sensors), seed
        self.hold_back, self.window = int(hold_back), int(window)

    def fit(self, frame, rows, y) -> ClassifierExplainer:
        self.model = build("signal", "v1_ensemble", sensors=self.sensors, seed=self.seed)
        Z = self.model.transform(frame)
        rows = np.asarray(rows, dtype=bool)
        self.model.fit(Z[rows], np.asarray(y)[rows])
        self.names = self.model.pipe.feature_names()
        self.owner = [_sensor_of(f, self.sensors) for f in self.names]
        return self

    def _score_slice(self, sl: pd.DataFrame) -> float:
        Z = self.model.transform(sl)
        return float(self.model.score_parts(Z.iloc[-1:])["ensemble"].iloc[0])

    def score_at(self, frame, p: int) -> float:
        return self._score_slice(frame.iloc[max(0, p - self.window):p + 1])

    def tree_shap(self, frame, positions) -> pd.DataFrame:
        import xgboost as xgb

        positions = np.asarray(positions, dtype=int)
        out = np.zeros((len(positions), len(self.sensors)))
        if getattr(self.model, "xgb", None) is None:
            return pd.DataFrame(np.nan, index=positions, columns=self.sensors)
        booster = self.model.xgb.get_booster()
        for i, p in enumerate(positions):
            Z = self.model.transform(frame.iloc[max(0, p - self.window):p + 1])
            X = self.model.pipe.apply(Z.iloc[-1:])
            contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]     # last column = bias
            for c, owner in zip(contrib, self.owner):
                if owner is not None:
                    out[i, self.sensors.index(owner)] += c
        return pd.DataFrame(out, columns=self.sensors, index=positions)

    def occluded(self, frame, p: int, onset: int, sensor: str) -> float:
        """Score at ``p`` if ``sensor`` had stayed at its level from before the alarm onset."""
        lo = max(0, p - self.window)
        sl = frame.iloc[lo:p + 1].copy()
        start = max(0, onset - self.hold_back) - lo
        before = frame[sensor].iloc[max(0, onset - self.hold_back - 15):max(1, onset - self.hold_back)]
        level = before.dropna().median() if before.notna().any() else np.nan
        if np.isfinite(level):
            col = sl.columns.get_loc(sensor)
            keep = sl.iloc[start:, col].notna()
            sl.iloc[start:, col] = np.where(keep, level, np.nan)
        return self._score_slice(sl)

    def occlusion(self, frame, positions, onsets) -> tuple[pd.DataFrame, np.ndarray]:
        positions = np.asarray(positions, dtype=int)
        out = np.zeros((len(positions), len(self.sensors)))
        base = np.zeros(len(positions))
        for i, (p, o) in enumerate(zip(positions, onsets)):
            base[i] = self.score_at(frame, p)
            for j, s in enumerate(self.sensors):
                out[i, j] = base[i] - self.occluded(frame, p, int(o), s)
        return pd.DataFrame(out, columns=self.sensors, index=positions), base


def ranking(scores: pd.Series) -> list[str]:
    """Sensors ordered from most to least responsible (missing scores last)."""
    s = scores.astype(float).fillna(-np.inf)
    return list(s.sort_values(ascending=False, kind="mergesort").index)
