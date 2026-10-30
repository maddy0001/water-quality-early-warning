"""Paper-1 (v1) models, ported to the streaming Signal interface (Segment 2).

Settings are copied from v1's ``src/config.py`` and model files:

Anomaly quorum (v1 ``EnsembleAnomalyDetector``, 2-of-4 vote = vote share ≥ 0.5):
    Isolation Forest  150 trees, max_samples auto, contamination 0.05
    LOF               20 neighbours, contamination 0.05
    One-Class SVM     RBF kernel, nu 0.05, gamma "scale"
    Autoencoder       Dense 16 → 8 → 16, ReLU, Adam, MSE; anomaly above the 95th
                      percentile of training error

Classifier ensemble (v1 ``EnsembleClassifier``, weighted soft voting 0.35 / 0.40 / 0.25):
    Random Forest     200 trees, max_depth 15, min_samples_split 5, min_samples_leaf 2, balanced
    XGBoost           150 trees, max_depth 7, learning rate 0.1, scale_pos_weight 1.5
    SVM               RBF, C 10, gamma "scale", balanced, probability outputs

What changed for one-minute streaming data, and why:

* **Input**: v1 saw one water sample (9 values). Here each minute is described by the
  causal rich features + coherence columns (``features/rich.py``); ``features: basic``
  and ``rich_nocoh`` exist for the ablation.
* **No transductive fitting**: v1's LOF used ``fit_predict`` on the data it scored.
  Here LOF uses ``novelty=True``: fitted on past minutes, applied to new ones.
* **Training rows**: the unsupervised detectors learn "normal" from normal fit minutes
  (labels of the fit period are known). Kernel and neighbour methods are trained on a
  random subsample (default 10,000 rows) because their cost grows with the square of
  the data size; the supervised models keep every event minute and subsample normal
  minutes to ``max_train`` rows.
* **Autoencoder**: same architecture, implemented with scikit-learn's MLPRegressor so
  this segment does not need TensorFlow (TensorFlow arrives with the BiLSTM in S3).
* **Continuous quorum**: "at least 2 of 4 detectors fire" is equivalent to "the
  2nd-highest detector score exceeds the threshold". Using that order statistic as the
  score lets the honest protocol choose the threshold. The original fixed operating
  point (each detector at its own 95th percentile, alarm on 2 votes) is reported too,
  as ``v1_quorum:fixed_vote``.

Each composite signal reports its parts (``score_parts``), so the evaluation also gets
every single detector and every single classifier without fitting anything twice.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, OneClassSVM

from ewsv2.core.registry import register
from ewsv2.features.rich import FeaturePipeline

log = logging.getLogger(__name__)

try:  # the small CPU-only wheel is "xgboost-cpu"; both install the module "xgboost"
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on install
    XGBClassifier = None


class PipelineSignal:
    """Common plumbing for signals built on a FeaturePipeline.

    Subclasses implement ``_fit_model(X, y)`` and ``_parts(X) -> dict[str, ndarray]``.
    The first part is the signal's own score; scores are higher = more abnormal.
    """

    name = "pipeline"
    supervised = False
    main_part = "main"
    fixed_thresholds: dict[str, float] = {}

    def __init__(self, sensors: list[str] | None = None, seed: int = 42, features: str = "rich",
                 train_on: str = "normal", max_train: int = 60000, max_kernel_train: int = 10000):
        self.sensors, self.seed = sensors, int(seed)
        self.features, self.train_on = features, train_on
        self.max_train, self.max_kernel_train = int(max_train), int(max_kernel_train)
        self.pipe = FeaturePipeline(features, sensors=sensors)
        self.rng = np.random.default_rng(self.seed)

    # -- contract ---------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        sensors = self.sensors or [c for c in frame.columns if c not in ("EVENT", "LABEL_CLASS")]
        return self.pipe.transform(frame, sensors)

    def fit(self, Z: pd.DataFrame, y=None):
        if self.supervised and y is None:
            raise ValueError(f"{self.name} is supervised: it needs event labels")
        rows = ~Z.isna().all(axis=1).to_numpy()
        Zf = Z[rows]
        yv = None if y is None else np.asarray(y)[rows].astype(int)
        self.pipe.fit(Zf, yv)
        X = self.pipe.apply(Zf)
        if not self.supervised and self.train_on == "normal" and yv is not None:
            X = X[yv == 0]
            yv = None
        self._fit_model(X, yv)
        return self

    def score(self, Z: pd.DataFrame) -> pd.Series:
        return self.score_parts(Z).iloc[:, 0].rename(self.name)

    def score_parts(self, Z: pd.DataFrame) -> pd.DataFrame:
        rows = ~Z.isna().all(axis=1).to_numpy()
        names = None
        out = None
        if rows.any():
            parts = self._parts(self.pipe.apply(Z[rows]))
            names = list(parts)
            out = np.full((len(Z), len(names)), np.nan)
            for j, n in enumerate(names):
                out[rows, j] = parts[n]
        else:
            names = [self.main_part]
            out = np.full((len(Z), 1), np.nan)
        return pd.DataFrame(out, index=Z.index, columns=names)

    # -- helpers ------------------------------------------------------------------
    def _subsample(self, X, y=None, cap: int | None = None):
        cap = cap or self.max_train
        if len(X) <= cap:
            return X, y
        if y is None:
            idx = self.rng.choice(len(X), cap, replace=False)
            return X[idx], None
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        keep_neg = self.rng.choice(neg, max(cap - len(pos), 1), replace=False) if len(neg) > cap - len(pos) else neg
        idx = np.sort(np.concatenate([pos, keep_neg]))
        return X[idx], y[idx]

    @staticmethod
    def _ecdf(ref: np.ndarray):
        """Share of reference scores below v, extended past the reference maximum.

        Inside the reference range this is the empirical CDF (0..1). Above the largest
        reference score it keeps growing linearly (1 + distance / reference spread), so
        very abnormal minutes are not all tied at exactly 1.
        """
        ref = np.sort(ref[np.isfinite(ref)])
        top = ref[-1] if len(ref) else 0.0
        spread = (np.quantile(ref, 0.99) - np.quantile(ref, 0.5)) if len(ref) else 1.0
        spread = spread if spread > 1e-12 else 1.0

        def f(v):
            v = np.asarray(v, dtype=float)
            base = np.searchsorted(ref, v, side="right") / max(len(ref), 1)
            return base + np.maximum(v - top, 0.0) / spread
        return f


# ----------------------------------------------------------------------------
# Unsupervised detectors (v1 anomaly module)
# ----------------------------------------------------------------------------

class _Detectors:
    """The four v1 detectors on one feature matrix."""

    def __init__(self, seed: int, kernel_cap: int, rng):
        self.seed, self.kernel_cap, self.rng = seed, kernel_cap, rng

    def fit(self, X: np.ndarray):
        Xk = X if len(X) <= self.kernel_cap else X[self.rng.choice(len(X), self.kernel_cap, replace=False)]
        Xae = X if len(X) <= 50000 else X[self.rng.choice(len(X), 50000, replace=False)]
        self.iforest = IsolationForest(n_estimators=150, max_samples="auto", contamination=0.05,
                                       random_state=self.seed, n_jobs=-1).fit(X)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1).fit(Xk)
        self.ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Xk)
        with warnings.catch_warnings():  # 60 epochs is the budget (v1: 50); not converging fully is expected
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu", solver="adam",
                                   learning_rate_init=1e-3, batch_size=256, max_iter=60, early_stopping=True,
                                   validation_fraction=0.1, n_iter_no_change=10,
                                   random_state=self.seed).fit(Xae, Xae)
        # reference distributions on (a sample of) training rows -> calibrated scores
        Xref = Xk
        self.cal = {n: PipelineSignal._ecdf(f(Xref)) for n, f in self._raw_fns().items()}
        return self

    def _raw_fns(self):
        return {
            "iforest": lambda X: -self.iforest.score_samples(X),
            "lof": lambda X: -self.lof.score_samples(X),
            "ocsvm": lambda X: -self.ocsvm.score_samples(X),
            "autoencoder": lambda X: np.mean((self.ae.predict(X) - X) ** 2, axis=1),
        }

    def calibrated(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {n: self.cal[n](f(X)) for n, f in self._raw_fns().items()}


@register("signal", "v1_quorum")
class V1Quorum(PipelineSignal):
    """v1's four-detector vote. Score = k-th highest calibrated detector score."""

    name = "v1_quorum"
    main_part = "quorum"

    def __init__(self, k: int = 2, vote_level: float = 0.95, **kw):
        super().__init__(**kw)
        self.k, self.vote_level = int(k), float(vote_level)
        # v1 rule: vote share >= 0.5 (2 of 4); not tuned
        self.fixed_thresholds = {"fixed_vote": 0.5}

    def _fit_model(self, X, y):
        self.det = _Detectors(self.seed, self.max_kernel_train, self.rng).fit(X)

    def _parts(self, X):
        cal = self.det.calibrated(X)
        S = np.column_stack(list(cal.values()))
        kth = np.sort(S, axis=1)[:, -self.k]
        votes = (S > self.vote_level).mean(axis=1)
        return {"quorum": kth, "fixed_vote": votes, **cal}


# ----------------------------------------------------------------------------
# Supervised ensemble (v1 classification module)
# ----------------------------------------------------------------------------

V1_WEIGHTS = {"rf": 0.35, "xgb": 0.40, "svm": 0.25}


@register("signal", "v1_ensemble")
class V1Ensemble(PipelineSignal):
    """v1's RF + XGBoost + SVM, weighted soft vote 0.35 / 0.40 / 0.25."""

    name = "v1_ensemble"
    supervised = True
    main_part = "ensemble"

    def __init__(self, weights: dict | None = None, svm_normal_rows: int = 8000, **kw):
        super().__init__(**kw)
        self.weights = dict(weights or V1_WEIGHTS)
        self.svm_normal_rows = int(svm_normal_rows)

    def _fit_model(self, X, y):
        if y.sum() == 0:
            raise ValueError("no event minutes in the training rows")
        Xs, ys = self._subsample(X, y, self.max_train)
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
                                         class_weight="balanced", random_state=self.seed, n_jobs=-1).fit(Xs, ys)
        if XGBClassifier is not None:
            self.xgb = XGBClassifier(n_estimators=150, max_depth=7, learning_rate=0.1, scale_pos_weight=1.5,
                                     tree_method="hist", eval_metric="logloss", random_state=self.seed,
                                     n_jobs=-1).fit(Xs, ys)
        else:
            log.warning("xgboost not installed: v1_ensemble runs without it (weights renormalised)")
            self.xgb = None
        Xv, yv = self._subsample(Xs, ys, int(ys.sum()) + self.svm_normal_rows)
        self.svm = SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced", probability=True,
                       random_state=self.seed).fit(Xv, yv)

    def _parts(self, X):
        p = {"rf": self.rf.predict_proba(X)[:, 1], "svm": self.svm.predict_proba(X)[:, 1]}
        if self.xgb is not None:
            p["xgb"] = self.xgb.predict_proba(X)[:, 1]
        w = {k: v for k, v in self.weights.items() if k in p}
        total = sum(w.values())
        ens = sum(w[k] * p[k] for k in w) / total
        return {"ensemble": ens, **p}
