"""Forecast surprise – signal F (Segment 3).

Idea: learn how the water normally evolves, forecast every sensor ``horizon`` minutes
ahead, and raise the score when reality departs from the forecast. A sudden change
in the water is, by definition, something the recent past did not predict.

For every minute t (all inputs causal):

    inputs  = readings from minute t - horizon - lookback + 1  to  t - horizon
    target  = readings at minute t
    residual_s = (actual_s - forecast_s) / typical forecast error of sensor s
    surprise   = max over sensors of |residual_s|          (part "surprise")
    surprise_mean = mean over sensors of |residual_s|      (part "surprise_mean")

The "typical forecast error" is measured on held-out training windows, so it is not
flattered by the model's own fit, and is never smaller than the sensor's reading
resolution (see ``error_scale``). Forecasters are trained on normal minutes only
(windows that touch a labelled event are left out), so events stay "surprising".

Forecasters (registered signal names):

    forecast_persistence   tomorrow = today: the value ``horizon`` minutes ago (no training)
    forecast_linear        ridge regression on the flattened window (fast, strong baseline)
    v1_gru_attention       paper 1's GRU(128) → GRU(64) → attention → Dense(32, 16)
    v1_bilstm_attention    paper 1's BiLSTM(128) → BiLSTM(64) → attention → Dense(32, 16)

Paper-1 settings kept: layer sizes, attention (Dense(1, tanh) → softmax over time →
weighted sum), dropout 0.3 (0.15 before the dense head), Adam, learning rate 1e-3, MSE,
lookback 30 steps. Changed for one-minute data: outputs all sensors instead of one; the
step is one minute instead of one day; recurrent dropout 0.2 → 0 (it disables the fast
LSTM kernel and doubled training time in our tests); training uses at most ``max_windows``
randomly chosen windows and stops early (patience 2) within ``epochs`` epochs.

TensorFlow is imported only when a neural forecaster is used.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ewsv2.core.registry import register

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------

def window_positions(index: pd.DatetimeIndex, values: np.ndarray, span: int) -> np.ndarray:
    """Row positions p where rows p-span+1 .. p are consecutive minutes with no missing value."""
    n = len(index)
    if n < span:
        return np.empty(0, dtype=int)
    t = index.asi8 // 60_000_000_000                       # minutes since epoch
    ok_row = np.isfinite(values).all(axis=1)
    p = np.arange(span - 1, n)
    contiguous = (t[p] - t[p - span + 1]) == span - 1
    bad = np.concatenate([[0], np.cumsum(~ok_row)])         # missing rows before each position
    complete = (bad[p + 1] - bad[p - span + 1]) == 0
    return p[contiguous & complete]


def error_scale(res: np.ndarray, resolution: np.ndarray | None = None, floor: float = 1e-3) -> np.ndarray:
    """Typical forecast error per sensor: RMS of residuals winsorised at their 0.5/99.5 percentiles,
    never smaller than the sensor's reading resolution.

    The median absolute deviation is not used: many water sensors are coarsely quantised
    and do not change for minutes, so most residuals are exactly zero and the MAD is 0 –
    which would turn every one-step change into an enormous "surprise". For the same
    reason a sensor that almost never moves (water temperature in winter) is floored at
    the smallest step it actually reports.
    """
    lo, hi = np.percentile(res, 0.5, axis=0), np.percentile(res, 99.5, axis=0)
    clipped = np.clip(res, lo, hi)
    scale = np.sqrt(np.mean(clipped ** 2, axis=0))
    if resolution is not None:
        scale = np.maximum(scale, resolution)
    return np.maximum(scale, floor)


def reading_resolution(values: np.ndarray) -> np.ndarray:
    """Smallest step each sensor reports (5th percentile of non-zero minute-to-minute changes)."""
    d = np.abs(np.diff(values, axis=0))
    out = np.empty(values.shape[1])
    for j in range(values.shape[1]):
        nz = d[:, j][np.isfinite(d[:, j]) & (d[:, j] > 1e-12)]
        out[j] = np.percentile(nz, 5) if len(nz) else 0.0
    return out


def make_windows(values: np.ndarray, positions: np.ndarray, lookback: int, horizon: int):
    """Inputs (n, lookback, sensors) and targets (n, sensors) for target rows ``positions``."""
    offsets = np.arange(lookback) - (lookback + horizon - 1)
    X = values[positions[:, None] + offsets[None, :]]
    return X.astype(np.float32), values[positions].astype(np.float32)


# ----------------------------------------------------------------------------
# Base signal
# ----------------------------------------------------------------------------

class ForecastSignal:
    name = "forecast"
    supervised = False
    main_part = "surprise"
    trainable = True

    def __init__(self, sensors: list[str] | None = None, seed: int = 42, lookback: int = 30, horizon: int = 5,
                 max_windows: int = 20000, train_on: str = "normal", chunk: int = 50000, **model_kw):
        self.sensors, self.seed = sensors, int(seed)
        self.lookback, self.horizon = int(lookback), int(horizon)
        self.max_windows, self.train_on, self.chunk = int(max_windows), train_on, int(chunk)
        self.model_kw = model_kw
        self.span = self.lookback + self.horizon
        self.rng = np.random.default_rng(self.seed)

    # -- contract ---------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        sensors = self.sensors or [c for c in frame.columns if c not in ("EVENT", "LABEL_CLASS")]
        x = frame[sensors].astype(float)
        empty = x.isna().all(axis=1).to_numpy()
        x = x.ffill(limit=5)
        x.loc[empty] = np.nan
        return x

    def fit(self, Z: pd.DataFrame, y=None):
        self.columns = list(Z.columns)
        rows = ~Z.isna().all(axis=1).to_numpy()
        med = Z[rows].median()
        iqr = Z[rows].quantile(0.75) - Z[rows].quantile(0.25)
        self.center = med.to_numpy(dtype=float)
        self.scale = np.where(iqr.to_numpy() > 1e-12, iqr.to_numpy(), 1.0)
        V = self._scaled(Z)

        pos = window_positions(Z.index, V, self.span)
        if y is not None and self.train_on == "normal":
            yv = np.asarray(y).astype(int)
            ev = np.concatenate([[0], np.cumsum(yv)])
            pos = pos[(ev[pos + 1] - ev[pos - self.span + 1]) == 0]   # no event inside the window
        if len(pos) < 200:
            raise ValueError(f"{self.name}: only {len(pos)} complete training windows")
        if len(pos) > self.max_windows:
            pos = np.sort(self.rng.choice(pos, self.max_windows, replace=False))
        cut = int(len(pos) * 0.9)                    # last 10% in time: early stopping + error scale
        tr, va = pos[:cut], pos[cut:]
        Xtr, Ytr = make_windows(V, tr, self.lookback, self.horizon)
        Xva, Yva = make_windows(V, va, self.lookback, self.horizon)
        if self.trainable:
            self._train(Xtr, Ytr, Xva, Yva)
        res = Yva - self._predict(Xva)
        self.resolution = reading_resolution(V)
        self.err_scale = error_scale(res, self.resolution)
        return self

    def score(self, Z: pd.DataFrame) -> pd.Series:
        return self.score_parts(Z).iloc[:, 0].rename(self.name)

    def score_parts(self, Z: pd.DataFrame) -> pd.DataFrame:
        V = self._scaled(Z[self.columns])
        pos = window_positions(Z.index, V, self.span)
        out = np.full((len(Z), 2), np.nan)
        for start in range(0, len(pos), self.chunk):
            p = pos[start:start + self.chunk]
            X, Y = make_windows(V, p, self.lookback, self.horizon)
            r = np.abs(Y - self._predict(X)) / self.err_scale
            out[p, 0] = r.max(axis=1)
            out[p, 1] = r.mean(axis=1)
        return pd.DataFrame(out, index=Z.index, columns=["surprise", "surprise_mean"])

    # -- helpers ------------------------------------------------------------------
    def _scaled(self, Z: pd.DataFrame) -> np.ndarray:
        V = (Z.to_numpy(dtype=float) - self.center) / self.scale
        return np.clip(V, -50, 50)

    def _train(self, Xtr, Ytr, Xva, Yva) -> None:
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@register("signal", "forecast_persistence")
class PersistenceForecast(ForecastSignal):
    """Forecast = last value before the horizon. Surprise = scaled change over ``horizon`` minutes."""

    name = "forecast_persistence"
    trainable = False

    def _predict(self, X):
        return X[:, -1, :]


@register("signal", "forecast_linear")
class LinearForecast(ForecastSignal):
    """Ridge regression from the flattened window (plus a constant) to every sensor."""

    name = "forecast_linear"

    def _train(self, Xtr, Ytr, Xva, Yva):
        alpha = float(self.model_kw.get("alpha", 1.0))
        A = np.hstack([Xtr.reshape(len(Xtr), -1), np.ones((len(Xtr), 1), np.float32)]).astype(np.float64)
        reg = alpha * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ Ytr.astype(np.float64))

    def _predict(self, X):
        A = np.hstack([X.reshape(len(X), -1), np.ones((len(X), 1), np.float32)])
        return A @ self.W


# ----------------------------------------------------------------------------
# Paper-1 neural forecasters (TensorFlow / Keras)
# ----------------------------------------------------------------------------

def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import keras
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError("v1 neural forecasters need TensorFlow: pip install -r ewsv2\\requirements.txt") from exc
    return keras


def build_v1_forecaster(lookback: int, n_sensors: int, cell: str = "bilstm", units=(128, 64), dense=(32, 16),
                        dropout: float = 0.3, recurrent_dropout: float = 0.0, learning_rate: float = 1e-3):
    """Paper 1's recurrent + attention architecture, with one output per sensor."""
    keras = _keras()
    from keras import layers

    inputs = keras.Input(shape=(lookback, n_sensors))
    x = inputs
    for i, u in enumerate(units):
        if cell == "bilstm":
            x = layers.Bidirectional(layers.LSTM(u, return_sequences=True, dropout=dropout,
                                                 recurrent_dropout=recurrent_dropout), name=f"bilstm_{i + 1}")(x)
        else:
            x = layers.GRU(u, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout,
                           name=f"gru_{i + 1}")(x)
    scores = layers.Dense(1, activation="tanh", name="attention_scores")(x)
    weights = layers.Softmax(axis=1, name="attention_weights")(scores)
    context = layers.Multiply(name="context")([x, weights])
    attended = layers.Lambda(lambda t: keras.ops.sum(t, axis=1), name="sum_over_time")(context)
    h = layers.Dense(dense[0], activation="relu", name="dense_1")(attended)
    h = layers.Dropout(dropout * 0.5, name="dropout_dense")(h)
    h = layers.Dense(dense[1], activation="relu", name="dense_2")(h)
    outputs = layers.Dense(n_sensors, activation="linear", name="output")(h)
    model = keras.Model(inputs, outputs, name=f"v1_{cell}_attention")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


class KerasForecast(ForecastSignal):
    cell = "bilstm"

    def _train(self, Xtr, Ytr, Xva, Yva):
        keras = _keras()
        keras.utils.set_random_seed(self.seed)
        kw = dict(self.model_kw)
        epochs, batch = int(kw.pop("epochs", 10)), int(kw.pop("batch_size", 256))
        patience = int(kw.pop("patience", 2))
        self.model = build_v1_forecaster(self.lookback, Xtr.shape[2], cell=self.cell, **kw)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        hist = self.model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs, batch_size=batch,
                              shuffle=True, callbacks=[stop], verbose=0)
        self.epochs_run = len(hist.history["loss"])
        log.debug("%s trained %d epochs, best val_loss %.4f", self.name, self.epochs_run, min(hist.history["val_loss"]))

    def _predict(self, X):
        return self.model.predict(X, batch_size=2048, verbose=0)


@register("signal", "v1_bilstm_attention")
class BiLSTMForecast(KerasForecast):
    name = "v1_bilstm_attention"
    cell = "bilstm"


@register("signal", "v1_gru_attention")
class GRUForecast(KerasForecast):
    name = "v1_gru_attention"
    cell = "gru"
