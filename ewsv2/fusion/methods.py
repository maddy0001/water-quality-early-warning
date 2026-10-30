"""Combining the three paper-1 modules into one score (Segment 5, contribution C5).

Inputs: one column per module, already calibrated to *nines* of rarity among normal
pre-test minutes (see ``calibrate/methods.py``), so the modules are on one scale:

    C  classifier   v1 RF + XGBoost + SVM ensemble (supervised, S2)
    A  anomaly      v1 four-detector quorum (unsupervised, S2)
    F  forecast     forecast surprise (S3)

``single``    one module alone (the baseline each fusion must beat)
``hand``      paper 1's published weights, R = 0.40 C + 0.30 F + 0.30 A, on the modules'
              [0, 1] scale (share of normal minutes below the score)
``equal``     the same with 1/3 each
``max``       the most alarming module (an OR rule)
``stacking``  logistic regression on the modules' nines, learned on out-of-fold
              pre-test scores (the weights are learned, not set by hand)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ewsv2.core.registry import register

V1_WEIGHTS = {"C": 0.40, "F": 0.30, "A": 0.30}


def unit_scale(nines: pd.DataFrame) -> pd.DataFrame:
    """Nines back to the [0, 1) scale: share of normal minutes scoring lower."""
    return 1.0 - np.power(10.0, -nines.clip(lower=0))


class Fusion:
    name = "fusion"
    trainable = False

    def fit(self, N: pd.DataFrame, y=None, rows=None) -> Fusion:
        return self

    def combine(self, N: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@register("fusion", "single")
class Single(Fusion):
    def __init__(self, role: str = "C", **_):
        self.role = role
        self.name = f"{role} alone"

    def combine(self, N):
        return N[self.role].to_numpy(dtype=float)


@register("fusion", "hand")
class HandWeights(Fusion):
    name = "hand (v1 0.40/0.30/0.30)"

    def __init__(self, weights: dict | None = None, **_):
        self.weights = dict(weights or V1_WEIGHTS)

    def combine(self, N):
        U = unit_scale(N).fillna(0.0)            # a module with no score yet counts as normal
        w = {k: v for k, v in self.weights.items() if k in U.columns}
        out = (sum(w[k] * U[k] for k in w) / sum(w.values())).to_numpy(dtype=float)
        out[~N.notna().any(axis=1).to_numpy()] = np.nan
        return out


@register("fusion", "equal")
class EqualWeights(HandWeights):
    name = "equal (1/3 each)"

    def __init__(self, **_):
        super().__init__({"C": 1.0, "F": 1.0, "A": 1.0})


@register("fusion", "max")
class MaxFusion(Fusion):
    name = "max (any module)"

    def combine(self, N):
        return N.max(axis=1, skipna=True).to_numpy(dtype=float)


@register("fusion", "stacking")
class Stacking(Fusion):
    """Logistic regression on the modules' nines (clipped to [0, 12])."""

    name = "stacking (learned)"
    trainable = True

    def __init__(self, C: float = 1.0, **_):
        self.C = float(C)

    @staticmethod
    def _X(N):
        return N.clip(0, 12).fillna(0.0).to_numpy(dtype=float)

    def fit(self, N, y=None, rows=None):
        rows = np.ones(len(N), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        rows &= N.notna().all(axis=1).to_numpy()
        self.model = LogisticRegression(C=self.C, class_weight="balanced", max_iter=1000)
        self.model.fit(self._X(N[rows]), np.asarray(y)[rows].astype(int))
        self.columns = list(N.columns)
        return self

    def combine(self, N):
        out = self.model.decision_function(self._X(N[self.columns]))
        out[~N.notna().any(axis=1).to_numpy()] = np.nan
        return out

    def weights(self) -> dict:
        return {c: float(w) for c, w in zip(self.columns, self.model.coef_[0])}
