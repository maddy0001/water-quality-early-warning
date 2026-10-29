"""
Shared model engine for the dashboard.

Loads the REAL trained models produced by `python main.py --train` and exposes
small helper functions that every page uses, so all pages give consistent results:

  * Classification : weighted soft-voting ensemble RF 0.35 + XGBoost 0.40 + SVM 0.25
  * Anomaly        : Isolation Forest, LOF, One-Class SVM, Autoencoder (2-of-4 quorum)
  * Forecasting    : BiLSTM + Attention (30-day window -> next-day normalised pH)
  * Alert fusion   : the project's own alert engine (src/early_warning/alert_system.py),
                     weights classification 0.40 / forecasting 0.30 / anomaly 0.30
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]          # project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from early_warning.alert_system import AlertLevel, AlertThresholds, AlertAggregator, Alert  # noqa: E402

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

FORECAST_FEATURES = ['Specific_conductance_max', 'pH_max', 'pH_min',
                     'Specific_conductance_min', 'Specific_conductance_mean',
                     'Dissolved_oxygen_max', 'Dissolved_oxygen_mean', 'Dissolved_oxygen_min',
                     'Temperature_mean', 'Temperature_min', 'Temperature_max']
LOOKBACK = 30

ENSEMBLE_WEIGHTS = {'random_forest': 0.35, 'xgboost': 0.40, 'svm': 0.25}
LEVEL_NAMES = ['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL']
LEVEL_EMOJI = {'NORMAL': '🟢', 'LOW': '🟡', 'MEDIUM': '🟠', 'CRITICAL': '🔴'}


def _full_width_kwargs():
    """Full-width argument that works on both old and new Streamlit (avoids deprecation warnings)."""
    import inspect
    p = inspect.signature(st.button).parameters.get('width')
    if p is not None and isinstance(p.default, str):      # Streamlit >= 1.46
        return {'width': 'stretch'}
    return {'use_container_width': True}


FULL = _full_width_kwargs()

P = {
    'rf': ROOT / 'models' / 'classification' / 'random_forest.pkl',
    'xgb': ROOT / 'models' / 'classification' / 'xgboost.pkl',
    'svm': ROOT / 'models' / 'classification' / 'svm.pkl',
    'scaler': ROOT / 'models' / 'classification' / 'scaler.pkl',
    'scaler_old': ROOT / 'data' / 'processed' / 'scaler.pkl',
    'imputer': ROOT / 'data' / 'processed' / 'imputer.pkl',
    'raw': ROOT / 'data' / 'raw' / 'water_potability.csv',
    'if': ROOT / 'models' / 'anomaly' / 'isolation_forest.pkl',
    'ocsvm': ROOT / 'models' / 'anomaly' / 'ocsvm.pkl',
    'ae': ROOT / 'models' / 'anomaly' / 'autoencoder.h5',
    'ae_thr': ROOT / 'models' / 'anomaly' / 'autoencoder_threshold.json',
    'bilstm': (ROOT / 'models' / 'forecasting' / 'bilstm_attention.keras'
               if (ROOT / 'models' / 'forecasting' / 'bilstm_attention.keras').exists()
               or not (ROOT / 'models' / 'forecasting' / 'bilstm_attention.h5').exists()
               else ROOT / 'models' / 'forecasting' / 'bilstm_attention.h5'),   # pipeline saves .keras
    'f_scaler': ROOT / 'models' / 'forecasting' / 'scaler.pkl',
    't_scaler': ROOT / 'data' / 'processed' / 'target_scaler.pkl',
}


def missing_files(keys):
    """Return the list of required model files that do not exist yet."""
    return [str(P[k].relative_to(ROOT)) for k in keys if not P[k].exists()]


def show_missing(keys, what):
    """Show a clear error if trained models are missing. Returns True if all present."""
    miss = missing_files(keys)
    if miss:
        st.error(
            f"❌ The trained {what} model files were not found:\n\n"
            + "\n".join(f"- `{m}`" for m in miss)
            + "\n\nRun **`python main.py --train`** from the project folder first, then refresh this page."
        )
        return False
    return True


# --------------------------------------------------------------------------
# Input checking
# --------------------------------------------------------------------------
def read_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    # accept 'pH' / 'PH' for 'ph'
    for c in list(df.columns):
        if c.lower() == 'ph' and c != 'ph':
            df = df.rename(columns={c: 'ph'})
    return df


def check_water_columns(df):
    """Return (ok, message). Extra columns are allowed and ignored."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return False, ("Missing columns: " + ", ".join(missing) +
                       ". This page needs the 9 water-quality columns: " + ", ".join(FEATURES))
    bad = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        return False, "These columns contain non-numeric values: " + ", ".join(bad)
    return True, ""


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifiers():
    models = {
        'random_forest': joblib.load(P['rf']),
        'xgboost': joblib.load(P['xgb']),
        'svm': joblib.load(P['svm']),
    }
    scaler_path = P['scaler'] if P['scaler'].exists() else P['scaler_old']
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(P['imputer']) if P['imputer'].exists() else None
    return models, scaler, imputer


def prepare(df):
    """Raw 9-parameter DataFrame -> (imputed raw DataFrame, scaled DataFrame)."""
    _, scaler, imputer = load_classifiers()
    X = df[FEATURES].astype(float).copy()
    if X.isna().any().any():
        if imputer is not None:
            X = pd.DataFrame(imputer.transform(X), columns=FEATURES, index=X.index)
        else:
            X = X.fillna(X.median())
    Xs = pd.DataFrame(scaler.transform(X), columns=FEATURES, index=X.index)
    return X, Xs


def classify(df):
    """Returns DataFrame with per-model and ensemble probability of POTABLE, prediction, confidence."""
    models, _, _ = load_classifiers()
    _, Xs = prepare(df)
    out = pd.DataFrame(index=df.index)
    p = np.zeros(len(Xs))
    for name, m in models.items():
        pm = m.predict_proba(Xs)[:, 1]
        out[f'p_{name}'] = pm
        p += ENSEMBLE_WEIGHTS[name] * pm
    out['p_potable'] = p
    out['prediction'] = (p >= 0.5).astype(int)                 # 1 = potable, 0 = not potable
    out['confidence'] = np.where(out['prediction'] == 1, p, 1 - p)
    out['cls_level'] = [
        AlertThresholds.get_alert_level_from_classification(c, pr).value
        for c, pr in zip(out['confidence'], out['prediction'])
    ]
    return out


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def reference_data():
    """The 2,847 cleaned + scaled samples the detectors were trained on (same steps as the pipeline)."""
    from data.data_loader import WaterPotabilityLoader
    from data.preprocessing_classification import ClassificationPreprocessor
    loader = WaterPotabilityLoader()
    loader.load(verbose=False)
    X, y = loader.get_features_and_target()
    Xp, _ = ClassificationPreprocessor().fit_transform(X, y, verbose=False)
    return Xp.values.astype(np.float32)


@st.cache_resource(show_spinner=False)
def load_detectors():
    from sklearn.neighbors import LocalOutlierFactor
    from tensorflow import keras
    ref = reference_data()
    iso = joblib.load(P['if'])
    oc = joblib.load(P['ocsvm'])
    from config import ANOMALY_CONFIG
    # LOF is not saved by the pipeline (it runs in fit_predict mode), so it is re-fitted here on the
    # same training data with the project's settings (k = 20, contamination from config) in novelty
    # mode, which lets it score new samples.
    lof = LocalOutlierFactor(n_neighbors=ANOMALY_CONFIG.get('lof_n_neighbors', 20),
                             contamination=ANOMALY_CONFIG.get('lof_contamination', 'auto'),
                             novelty=True).fit(ref)
    ae = keras.models.load_model(str(P['ae']), compile=False)
    with open(P['ae_thr']) as f:
        thr = json.load(f)['threshold']
    return iso, lof, oc, ae, thr


def detect(df):
    """Returns DataFrame with each detector's flag (1 = anomaly), vote count, quorum flag, level."""
    iso, lof, oc, ae, thr = load_detectors()
    _, Xs = prepare(df)
    X = Xs.values.astype(np.float32)
    err = np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)
    out = pd.DataFrame(index=df.index)
    out['Isolation Forest'] = (iso.predict(X) == -1).astype(int)
    out['LOF'] = (lof.predict(X) == -1).astype(int)
    out['One-Class SVM'] = (oc.predict(X) == -1).astype(int)
    out['Autoencoder'] = (err > thr).astype(int)
    out['votes'] = out[['Isolation Forest', 'LOF', 'One-Class SVM', 'Autoencoder']].sum(axis=1)
    out['anomaly'] = (out['votes'] >= 2).astype(int)
    out['an_level'] = [AlertThresholds.get_alert_level_from_anomaly(v).value for v in out['votes']]
    return out


# --------------------------------------------------------------------------
# Alert fusion (project's alert engine)
# --------------------------------------------------------------------------
def fuse(levels_by_source):
    """levels_by_source: dict source -> level int. Returns (level int, risk 0-1)."""
    from datetime import datetime
    alerts = [Alert(level=AlertLevel(int(v)), timestamp=datetime.now(), source=s)
              for s, v in levels_by_source.items()]
    a = AlertAggregator.aggregate(alerts)
    return a.level.value, float(a.confidence)


def early_warning(df):
    """Run classification + anomaly detection and fuse into a final alert for every row."""
    c = classify(df)
    a = detect(df)
    res = pd.concat([c, a], axis=1)
    fused = [fuse({'classification': cl, 'anomaly': al}) for cl, al in zip(res['cls_level'], res['an_level'])]
    res['level'] = [f[0] for f in fused]
    res['risk'] = [f[1] for f in fused]
    res['Alert Level'] = [LEVEL_NAMES[v] for v in res['level']]
    return res


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_forecaster():
    from tensorflow import keras
    from models.forecasting.bilstm_attention import SumOverTime
    model = keras.models.load_model(str(P['bilstm']), custom_objects={'SumOverTime': SumOverTime},
                                    compile=False)
    f_scaler = joblib.load(P['f_scaler'])
    t_scaler = joblib.load(P['t_scaler']) if P['t_scaler'].exists() else None
    return model, f_scaler, t_scaler


def check_forecast_columns(df):
    missing = [c for c in FORECAST_FEATURES if c not in df.columns]
    if missing:
        return False, ("Missing columns: " + ", ".join(missing) +
                       ". The BiLSTM needs the 11 USGS sensor columns: " + ", ".join(FORECAST_FEATURES))
    if len(df) < LOOKBACK:
        return False, f"The BiLSTM needs at least {LOOKBACK} consecutive days; this file has {len(df)} rows."
    return True, ""


def forecast(df):
    """
    Rolling one-day-ahead forecasts.
    For each day t >= 30 (and one extra step past the end): uses days t-30 .. t-1 to predict day t.
    Returns (predictions array of length n-30+1, index of the day each prediction is for).
    Values are on the model's normalised pH scale (0-1), the same scale as the paper's RMSE (0.0524).
    """
    model, f_scaler, _ = load_forecaster()
    X = f_scaler.transform(df[FORECAST_FEATURES].astype(float).values)
    n = len(X)
    windows = np.stack([X[i - LOOKBACK:i] for i in range(LOOKBACK, n + 1)]).astype(np.float32)
    y = model.predict(windows, verbose=0).ravel()
    return y, np.arange(LOOKBACK, n + 1)


def scale_actual_ph(values):
    """Put the file's actual pH column on the same normalised scale as the model output."""
    _, _, t_scaler = load_forecaster()
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    return t_scaler.transform(v).ravel() if t_scaler is not None else v.ravel()
