"""Rich causal features and the coherence model (Segment 2).

Two parts, so that nothing learned from data leaks into the stateless step:

1. :class:`RichFeatures` (stateless, causal) – for each sensor ``s``:

   ``s``            current value (short gaps carried forward up to 5 min)
   ``s__d1``        change over the last minute
   ``s__dev{w}``    value minus the mean of the previous ``w`` minutes  (w = 5, 15, 60)
   ``s__sd{w}``     standard deviation of the previous ``w`` minutes    (w = 15, 60)
   ``s__flat{w}``   1 if the sensor has not changed at all in the last ``w`` minutes (w = 15)

2. :class:`FeaturePipeline` (fitted on the fit rows only):

   * robust scaling of every column (median / IQR of the fit rows), clipped to ±``clip``
   * the **coherence model**: for each sensor, a ridge regression that predicts its
     15-minute deviation from the other sensors' 15-minute deviations, trained on normal
     fit minutes. Its residual ``s__incoh`` is the part of a sensor's move that the other
     sensors do not explain. Summary columns:

     ``coh__max_dev``    largest scaled 15-min deviation across sensors
     ``coh__max_incoh``  largest scaled residual across sensors
     ``coh__breadth``    number of sensors whose scaled deviation exceeds ``breadth_z``
     ``coh__share``      max_incoh / (max_dev + 1): near 1 = one sensor moves alone
                         (fault-like), near 0 = the move is shared (water-like)

   A real change in the water should move coupled sensors together (small residual,
   large breadth); a failing sensor moves alone (large residual, breadth 1). These
   columns are the raw material for the triage in Segment 4.

Feature sets (``features=`` parameter of the S2 signals):
    basic        the S1 features
    rich         RichFeatures + coherence columns
    rich_nocoh   RichFeatures only (ablation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.core.registry import build, register

COH_WINDOW = 15


@register("feature", "rich")
class RichFeatures:
    def __init__(self, dev_windows=(5, 15, 60), sd_windows=(15, 60), flat_windows=(15,), ffill_limit: int = 5):
        self.dev_windows, self.sd_windows = tuple(dev_windows), tuple(sd_windows)
        self.flat_windows, self.ffill_limit = tuple(flat_windows), int(ffill_limit)

    def transform(self, frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        x = frame[sensors].astype(float)
        empty = x.isna().all(axis=1).to_numpy()
        x = x.ffill(limit=self.ffill_limit)
        cols = {}
        for s in sensors:
            v = x[s]
            past = v.shift(1)
            cols[s] = v
            cols[f"{s}__d1"] = v.diff()
            for w in self.dev_windows:
                cols[f"{s}__dev{w}"] = v - past.rolling(w, min_periods=max(2, w // 3)).mean()
            for w in self.sd_windows:
                cols[f"{s}__sd{w}"] = past.rolling(w, min_periods=max(2, w // 3)).std()
            for w in self.flat_windows:
                changed = v.diff().abs().rolling(w, min_periods=w).max()
                cols[f"{s}__flat{w}"] = (changed == 0).astype(float).where(changed.notna())
        out = pd.DataFrame(cols, index=frame.index)
        out.loc[empty] = np.nan
        return out


class FeaturePipeline:
    """Stateless transform + fitted scaling (+ optional coherence model)."""

    def __init__(self, features: str = "rich", clip: float = 50.0, ridge_alpha: float = 1.0,
                 breadth_z: float = 3.0, sensors: list[str] | None = None):
        if features not in ("basic", "rich", "rich_nocoh"):
            raise ValueError("features must be basic, rich or rich_nocoh")
        self.features, self.clip = features, float(clip)
        self.ridge_alpha, self.breadth_z = float(ridge_alpha), float(breadth_z)
        self.sensors: list[str] = list(sensors or [])

    # -- stateless --------------------------------------------------------------
    def transform(self, frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        self.sensors = list(sensors)
        name = "basic" if self.features == "basic" else "rich"
        return build("feature", name).transform(frame, sensors)

    # -- fitted -----------------------------------------------------------------
    def fit(self, Z: pd.DataFrame, y=None) -> FeaturePipeline:
        if not self.sensors:  # fitted copies may never have run transform(): read sensors from the columns
            self.sensors = [c[: -len(f"__dev{COH_WINDOW}")] for c in Z.columns if c.endswith(f"__dev{COH_WINDOW}")]
        rows = ~Z.isna().all(axis=1).to_numpy()
        Zf = Z[rows]
        self.columns = list(Z.columns)
        self.center = Zf.median()
        q75, q25 = Zf.quantile(0.75), Zf.quantile(0.25)
        iqr = (q75 - q25)
        fallback = (Zf - self.center).abs().mean()
        self.scale = iqr.where(iqr > 1e-12, fallback).where(lambda s: s > 1e-12, 1.0)
        self.use_coh = self.features == "rich"
        if self.use_coh and len(self.sensors) < 2:
            raise ValueError("the coherence model needs at least two sensors")
        if self.use_coh:
            normal = np.ones(len(Zf), dtype=bool) if y is None else (np.asarray(y)[rows] == 0)
            D = self._dev(Zf[normal])
            self.coef = {}
            for i, s in enumerate(self.sensors):
                others = [j for j in range(len(self.sensors)) if j != i]
                A, b = D[:, others], D[:, i]
                self.coef[s] = (others, np.linalg.solve(A.T @ A + self.ridge_alpha * np.eye(len(others)), A.T @ b))
            R = self._incoh(D)
            self.incoh_scale = np.maximum(np.median(np.abs(R), axis=0) * 1.4826, 1e-6)
        return self

    def _dev(self, Z: pd.DataFrame) -> np.ndarray:
        cols = [f"{s}__dev{COH_WINDOW}" for s in self.sensors]
        D = ((Z[cols] - self.center[cols]) / self.scale[cols]).to_numpy(dtype=float)
        return np.clip(np.nan_to_num(D, nan=0.0), -self.clip, self.clip)

    def _incoh(self, D: np.ndarray) -> np.ndarray:
        R = np.empty_like(D)
        for i, s in enumerate(self.sensors):
            others, w = self.coef[s]
            R[:, i] = D[:, i] - D[:, others] @ w
        return R

    def apply(self, Z: pd.DataFrame) -> np.ndarray:
        """Scaled feature matrix (rows with no data must be removed by the caller)."""
        X = ((Z[self.columns] - self.center) / self.scale).to_numpy(dtype=float)
        X = np.clip(np.nan_to_num(X, nan=0.0), -self.clip, self.clip)
        if not self.use_coh:
            return X
        D = self._dev(Z)
        R = np.clip(self._incoh(D) / self.incoh_scale, -self.clip, self.clip)
        absD, absR = np.abs(D), np.abs(R)
        max_dev, max_incoh = absD.max(axis=1), absR.max(axis=1)
        summary = np.column_stack([
            max_dev, max_incoh, (absD > self.breadth_z).sum(axis=1), max_incoh / (max_dev + 1.0),
        ])
        return np.hstack([X, R, summary])

    def feature_names(self) -> list[str]:
        names = list(self.columns)
        if self.use_coh:
            names += [f"{s}__incoh" for s in self.sensors]
            names += ["coh__max_dev", "coh__max_incoh", "coh__breadth", "coh__share"]
        return names
