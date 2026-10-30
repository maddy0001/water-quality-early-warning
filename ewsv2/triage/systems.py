"""Three station-level systems compared on GECCO-Triage (Segment 4, contribution C1).

Every system gives the operator two outputs per minute:

* ``water``  – a water-quality alarm ("check the water")
* ``sensor`` – a sensor alert, naming the sensor ("check the sensor")

``detector_only``  (what v1 and most published systems do) a water-event detector – here
                   the S1 Random Forest on the basic features, trained on water events;
                   every alarm is a water alarm; no sensor alerts.
``rules``          the same detector, plus the four fault checks (no fault labels needed).
                   A sensor becomes a *suspect* when a check flags it or when, during an
                   alarm, it jumps alone (>= ``jump_units`` in one minute, most of its
                   15-minute move, no other sensor moving); it stays a suspect for an
                   hour, unless it keeps moving the same way (the front of a real change
                   in the water, e.g. organics rising over several minutes). A water alarm is withdrawn when it is explained by the suspects:
                   no other sensor moved more than it does in normal water (99.9th pct. of
                   its normal 15-minute moves). A sensor alert is raised while a check
                   fires, and at a lone jump.
``learned``        one three-class Random Forest (normal / water event / sensor fault) on
                   the basic features plus the triage evidence, trained on the pre-test
                   part of GECCO-Triage (which includes injected faults).

Thresholds are chosen honestly, as in S1: out-of-fold scores over the pre-test period
(4 blocked folds, 1440 min purged), objective = event F1. By default the detector is the
S1 detector exactly: trained and thresholded on real events, never shown an injected
minute. The same detector and threshold are used by ``detector_only`` and ``rules``, so
the difference between them is the triage alone.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ewsv2.core.registry import build, register
from ewsv2.evaluate.thresholds import select_threshold
from ewsv2.triage.features import TriageFeatures

log = logging.getLogger(__name__)


class Context:
    """What every system gets: the injected stream, labels, split and shared features."""

    def __init__(self, ds, split, cls: np.ndarray, basic: pd.DataFrame, seed: int, real_event: np.ndarray,
                 tolerance_after: int = 0, merge_gap: int = 5, max_episode_minutes: int = 60,
                 fault_params: dict | None = None, fault_grace: int = 10):
        self.ds, self.split, self.cls = ds, split, np.asarray(cls, dtype=int)
        self.frame, self.sensors, self.index = ds.frame, ds.sensors, ds.frame.index
        self.basic, self.seed = basic, int(seed)
        self.tol, self.gap, self.max_ep = int(tolerance_after), int(merge_gap), int(max_episode_minutes)
        self.fault_params = dict(fault_params or {})
        self.fault_grace = int(fault_grace)
        self.has_data = ~self.frame[self.sensors].isna().all(axis=1).to_numpy()
        self.water = (self.cls == 1).astype(int)
        self.fault = (self.cls == 2).astype(int)
        self.real = np.asarray(real_event, dtype=int)                 # labels of the original data
        self.injected = ((self.cls == 1) & (self.real == 0)) | (self.cls == 2)
        self._triage_cache: dict = {}

    def triage_features(self, fit_rows: np.ndarray, key: str) -> dict:
        """Triage evidence with fault-check limits learned from the normal minutes of ``fit_rows``."""
        if key not in self._triage_cache:
            normal = fit_rows & (self.cls == 0) & self.has_data
            tf = TriageFeatures(self.sensors, self.fault_params).fit(self.frame, normal)
            self._triage_cache[key] = (tf, tf.compute(self.frame))
        return self._triage_cache[key]


def _rf(seed: int, n_estimators: int, min_samples_leaf: int = 1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  random_state=seed, n_jobs=-1)


def _fill(X: pd.DataFrame, med: pd.Series) -> np.ndarray:
    return X.fillna(med).to_numpy(dtype=float)


# ----------------------------------------------------------------------------
# Water-event detector shared by detector_only and rules
# ----------------------------------------------------------------------------

class WaterDetector:
    """The S1 Random Forest on the basic features.

    ``train_on="real"`` (default): trained and thresholded on the real GECCO events only,
    with every injected minute left out – exactly the S1 detector, which has never seen an
    injected contaminant or a fault. Injected contaminants then test whether it generalises
    to contaminants it was not trained on (E7).
    ``train_on="all"``: also learns from the injected pre-test contaminants (a utility with
    a library of contaminant signatures); faults are still left out.
    """

    def __init__(self, n_estimators: int = 100, train_on: str = "real"):
        if train_on not in ("real", "all"):
            raise ValueError("detector train_on must be 'real' or 'all'")
        self.n_estimators, self.train_on = int(n_estimators), train_on

    def _labels(self, ctx: Context) -> np.ndarray:
        return ctx.real if self.train_on == "real" else ctx.water

    def _usable(self, ctx: Context) -> np.ndarray:
        return ~ctx.injected if self.train_on == "real" else (ctx.cls != 2)

    def _fit(self, ctx: Context, rows: np.ndarray):
        rows = rows & ctx.has_data & self._usable(ctx)
        X = ctx.basic[rows]
        med = X.median()
        model = _rf(ctx.seed, self.n_estimators).fit(_fill(X, med), self._labels(ctx)[rows])
        return model, med

    def _predict(self, model_med, ctx: Context, rows: np.ndarray) -> np.ndarray:
        model, med = model_med
        out = np.full(len(ctx.frame), np.nan)
        r = rows & ctx.has_data
        if r.any():
            out[r] = model.predict_proba(_fill(ctx.basic[r], med))[:, 1]
        return out

    def fit(self, ctx: Context) -> WaterDetector:
        sp = ctx.split
        oof = np.full(len(ctx.frame), np.nan)
        y = self._labels(ctx)
        for fit_k, val_k in sp.folds:
            if not val_k.any() or y[fit_k].sum() == 0:
                continue
            oof[val_k] = self._predict(self._fit(ctx, fit_k), ctx, val_k)[val_k]
        rows = sp.val & self._usable(ctx)
        self.threshold, self.cv_value = select_threshold(oof, y, ctx.index, rows, "event_f1",
                                                         ctx.tol, ctx.gap, ctx.max_ep)
        self.final = self._fit(ctx, sp.fit)
        self.scores = self._predict(self.final, ctx, np.ones(len(ctx.frame), dtype=bool))
        return self

    def alarm(self) -> np.ndarray:
        return np.nan_to_num(self.scores, nan=-np.inf) >= self.threshold


# ----------------------------------------------------------------------------
# Systems
# ----------------------------------------------------------------------------

class StationOutput:
    def __init__(self, water: np.ndarray, sensor: np.ndarray, sensor_names: pd.Series | None = None,
                 info: dict | None = None):
        self.water = np.asarray(water, dtype=bool)
        self.sensor = np.asarray(sensor, dtype=bool)
        self.sensor_names = sensor_names          # which sensor(s) each sensor alert names
        self.info = info or {}


@register("triage", "detector_only")
class DetectorOnly:
    label = "Detector only (v1-style)"

    def __init__(self, detector: WaterDetector | None = None, n_estimators: int = 100):
        self.detector = detector or WaterDetector(n_estimators)

    def run(self, ctx: Context) -> StationOutput:
        if not hasattr(self.detector, "final"):
            self.detector.fit(ctx)
        d = self.detector
        return StationOutput(d.alarm(), np.zeros(len(ctx.frame), dtype=bool),
                             info={"water_threshold": d.threshold, "cv_event_f1": d.cv_value})


@register("triage", "rules")
class RuleTriage:
    """Detector + physics checks. No fault labels are used anywhere."""

    label = "Detector + fault checks (rules)"

    def __init__(self, detector: WaterDetector | None = None, n_estimators: int = 100,
                 z_move: float = 3.0, jump_units: float = 6.0, jump_share: float = 0.7,
                 continue_units: float = 2.0):
        self.detector = detector or WaterDetector(n_estimators)
        self.z_move, self.jump_units, self.jump_share = float(z_move), float(jump_units), float(jump_share)
        self.continue_units = float(continue_units)

    def run(self, ctx: Context) -> StationOutput:
        if not hasattr(self.detector, "final"):
            self.detector.fit(ctx)
        D = self.detector.alarm()
        tf, feats = ctx.triage_features(ctx.split.fit, "fit")
        tab = feats["table"]
        sens = np.array(ctx.sensors)
        n = len(ctx.frame)
        # one sensor jumped alone, most of its 15-minute move in a single minute: a slug of
        # water changes a sensor over several minutes and moves its partners too
        z = feats["z15"].abs().to_numpy()
        lead = z.argmax(axis=1)
        lone_jump = D & (tab["lead_jump"].to_numpy() >= self.jump_units) & \
            (tab["second_z15"].to_numpy() < self.z_move) & \
            (tab["lead_jump_share"].to_numpy() >= self.jump_share)
        # A lone jump makes its sensor a suspect for ``hold`` minutes – unless the sensor keeps
        # moving the same way (a further step of >= ``continue_units`` within the hold): then
        # it is the front of a real change in the water, and the suspicion is dropped.
        hold = tf.hold
        x = ctx.frame[ctx.sensors].astype(float).ffill(limit=5)
        step = (x.diff() / tf.units[ctx.sensors]).fillna(0.0).to_numpy()
        jumped = np.zeros((n, len(sens)), dtype=bool)
        for t in np.flatnonzero(lone_jump):
            j = lead[t]
            sign = np.sign(step[t, j])
            end = min(n, t + hold + 1)
            cont = np.flatnonzero(step[t + 1:end, j] * sign >= self.continue_units)
            stop = t + 1 + cont[0] if len(cont) else end
            jumped[t:stop, j] = True            # causal: the suspicion ends when the next step is seen
        suspect = feats["flags"]["any"].to_numpy() | jumped
        any_suspect = suspect.any(axis=1)
        # the alarm is explained by the suspects: no other sensor moved more than it does in normal water
        moved = z / tf.move_limit[ctx.sensors].to_numpy()
        clean = np.where(suspect, 0.0, moved).max(axis=1)
        explained = any_suspect & (clean < 1.0)
        water = D & ~explained & ~lone_jump
        # sensor alerts follow the current evidence (the one-hour suspicion only explains alarms)
        now = feats["flags"]["now"].to_numpy().copy()
        now[np.flatnonzero(lone_jump), lead[lone_jump]] = True
        sensor = now.any(axis=1)
        names = np.full(n, "", dtype=object)
        for i in np.flatnonzero(sensor):
            names[i] = ",".join(sens[now[i]])
        names = pd.Series(names, index=ctx.index)
        return StationOutput(water, sensor, names, info={
            "water_threshold": self.detector.threshold, "cv_event_f1": self.detector.cv_value,
            "withdrawn_minutes": int((D & ~water).sum())})


@register("triage", "learned")
class LearnedTriage:
    """Three-class Random Forest (normal / water / fault) on basic + triage features."""

    label = "Learned three-class triage"

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 2):
        self.n_estimators, self.min_samples_leaf = int(n_estimators), int(min_samples_leaf)

    def _X(self, ctx: Context, feats: dict) -> pd.DataFrame:
        return pd.concat([ctx.basic, feats["table"]], axis=1)

    def _fit(self, ctx: Context, rows: np.ndarray, key: str):
        _, feats = ctx.triage_features(rows, key)
        X = self._X(ctx, feats)
        r = rows & ctx.has_data
        med = X[r].median()
        model = _rf(ctx.seed, self.n_estimators, self.min_samples_leaf).fit(_fill(X[r], med), ctx.cls[r])
        return model, med, X

    @staticmethod
    def _proba(model_med_X, rows: np.ndarray, has_data: np.ndarray) -> np.ndarray:
        model, med, X = model_med_X
        out = np.full((len(X), 3), np.nan)
        r = rows & has_data
        if r.any():
            p = model.predict_proba(_fill(X[r], med))
            full = np.zeros((p.shape[0], 3))
            full[:, model.classes_.astype(int)] = p
            out[r] = full
        return out

    def run(self, ctx: Context) -> StationOutput:
        sp = ctx.split
        oof = np.full((len(ctx.frame), 3), np.nan)
        for k, (fit_k, val_k) in enumerate(sp.folds):
            if not val_k.any() or ctx.water[fit_k].sum() == 0:
                continue
            oof[val_k] = self._proba(self._fit(ctx, fit_k, f"fold{k}"), val_k, ctx.has_data)[val_k]
        tw, vw = select_threshold(oof[:, 1], ctx.water, ctx.index, sp.val, "event_f1", ctx.tol, ctx.gap, ctx.max_ep)
        fault_y = _with_grace(ctx.fault, ctx.fault_grace)
        tf_, vf = select_threshold(oof[:, 2], fault_y, ctx.index, sp.val, "event_f1", 0, ctx.gap, ctx.max_ep)
        final = self._fit(ctx, sp.fit, "fit")
        p = self._proba(final, np.ones(len(ctx.frame), dtype=bool), ctx.has_data)
        water = np.nan_to_num(p[:, 1], nan=-np.inf) >= tw
        sensor = (np.nan_to_num(p[:, 2], nan=-np.inf) >= tf_) & ~water
        _, feats = ctx.triage_features(sp.fit, "fit")
        ev = pd.concat({k: feats["evidence"][k] for k in ("stuck", "dropout", "spike", "drift")}, axis=1)
        per_sensor = ev.T.groupby(level=1).max().T[ctx.sensors]
        z = feats["z15"].abs()
        score = per_sensor + 0.01 * z                       # strongest fault evidence, else largest move
        names = pd.Series(score.idxmax(axis=1).to_numpy(), index=ctx.index)
        self.model = final[0]
        self.feature_names = list(final[2].columns)
        return StationOutput(water, sensor, names, info={
            "water_threshold": tw, "cv_water_event_f1": vw, "fault_threshold": tf_, "cv_fault_event_f1": vf})


def _with_grace(y: np.ndarray, grace: int) -> np.ndarray:
    """Extend each labelled run by ``grace`` minutes (a fault is often only provable just after it)."""
    y = np.asarray(y, dtype=bool).copy()
    if grace <= 0:
        return y.astype(int)
    from ewsv2.evaluate.metrics import segments
    for s, e in segments(y):
        y[e + 1: e + 1 + grace] = True
    return y.astype(int)


def build_system(name: str, shared_detector: WaterDetector, params: dict | None = None):
    params = dict(params or {})
    if name in ("detector_only", "rules"):
        return build("triage", name, detector=shared_detector, **params)
    return build("triage", name, **params)
