"""Rich features, coherence model, and the ported paper-1 models."""

import numpy as np
import pandas as pd
import pytest

import ewsv2.signals.v1_port as v1
from ewsv2.core.registry import available, build
from ewsv2.evaluate.audit import _signal_specs
from ewsv2.features.rich import FeaturePipeline, RichFeatures

from .conftest import make_stream

SENS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]


@pytest.fixture
def stream():
    df = make_stream("2024-01-01", 3 * 1440, [(i, 15) for i in range(300, 4000, 400)], seed=5)
    return df.set_index("Time").assign(EVENT=lambda d: d["EVENT"].astype(int))


def test_registered():
    assert {"v1_quorum", "v1_ensemble"} <= set(available("signal")["signal"])
    assert "rich" in available("feature")["feature"]


def test_rich_features_are_causal(stream):
    a = RichFeatures().transform(stream, SENS)
    changed = stream.copy()
    changed.iloc[2000:, :len(SENS)] += 100.0            # change the future only
    b = RichFeatures().transform(changed, SENS)
    pd.testing.assert_frame_equal(a.iloc[:2000], b.iloc[:2000])


def test_flat_flag_marks_frozen_sensor(stream):
    s = stream.copy()
    s.iloc[1000:1100, SENS.index("Leit")] = 210.0       # stuck for 100 minutes
    f = RichFeatures().transform(s, SENS)
    assert f["Leit__flat15"].iloc[1050] == 1.0 and f["Leit__flat15"].iloc[900] == 0.0


def test_coherence_separates_lone_sensor_from_shared_move():
    rng = np.random.default_rng(0)
    n = 6000
    base = np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame({"a": base + rng.normal(0, 0.01, n), "b": 2 * base + rng.normal(0, 0.01, n),
                       "c": rng.normal(0, 1, n)}, index=pd.date_range("2024", periods=n, freq="min"))
    shared, lone = df.copy(), df.copy()
    shared.iloc[5000:5030, 0] += 3.0
    shared.iloc[5000:5030, 1] += 6.0                     # a and b move together, as they always do
    lone.iloc[5000:5030, 0] += 3.0                       # only a moves: fault-like
    out = {}
    for name, d in (("shared", shared), ("lone", lone)):
        fp = FeaturePipeline("rich", sensors=["a", "b", "c"])
        Z = fp.transform(d, ["a", "b", "c"])
        fp.fit(Z.iloc[:4000])
        X = fp.apply(Z.iloc[5005:5020])
        names = fp.feature_names()
        out[name] = (X[:, names.index("a__incoh")].mean(), X[:, names.index("coh__breadth")].mean())
    assert abs(out["lone"][0]) > 3 * abs(out["shared"][0])     # lone move is incoherent
    assert out["shared"][1] > out["lone"][1]                     # shared move is broad


def test_fresh_pipeline_uses_coherence_without_transform(stream):
    """Regression: copies fitted on cached features must still build the coherence model."""
    Z = FeaturePipeline("rich").transform(stream, SENS)
    fresh = FeaturePipeline("rich")
    fresh.fit(Z, stream["EVENT"].to_numpy())
    assert fresh.use_coh and fresh.sensors == SENS
    assert fresh.apply(Z.iloc[:3]).shape[1] == Z.shape[1] + len(SENS) + 4
    nocoh = FeaturePipeline("rich_nocoh").fit(Z)
    assert nocoh.apply(Z.iloc[:3]).shape[1] == Z.shape[1]


def test_v1_quorum_parts_and_fixed_rule(stream):
    sig = build("signal", "v1_quorum", sensors=SENS, max_kernel_train=1500)
    Z = sig.transform(stream)
    y = stream["EVENT"].to_numpy()
    sig.fit(Z.iloc[:3000], y[:3000])
    parts = sig.score_parts(Z)
    assert list(parts.columns) == ["quorum", "fixed_vote", "iforest", "lof", "ocsvm", "autoencoder"]
    assert set(np.unique(parts["fixed_vote"].dropna())) <= {0, 0.25, 0.5, 0.75, 1.0}
    assert sig.fixed_thresholds == {"fixed_vote": 0.5}
    # quorum = 2nd highest calibrated detector score
    det = parts[["iforest", "lof", "ocsvm", "autoencoder"]].to_numpy()
    np.testing.assert_allclose(parts["quorum"], np.sort(det, axis=1)[:, -2])
    # calibrated scores do not saturate at 1
    assert parts[["iforest", "lof", "ocsvm", "autoencoder"]].to_numpy().max() > 1.0


def test_v1_ensemble_weights(stream, monkeypatch):
    y = stream["EVENT"].to_numpy()
    sig = build("signal", "v1_ensemble", sensors=SENS, svm_normal_rows=800)
    Z = sig.transform(stream)
    sig.fit(Z.iloc[:3000], y[:3000])
    p = sig.score_parts(Z.iloc[3000:])
    assert list(p.columns) == ["ensemble", "rf", "svm", "xgb"]
    np.testing.assert_allclose(p["ensemble"], 0.35 * p["rf"] + 0.40 * p["xgb"] + 0.25 * p["svm"])

    monkeypatch.setattr(v1, "XGBClassifier", None)        # without xgboost: RF and SVM renormalised
    sig2 = build("signal", "v1_ensemble", sensors=SENS, svm_normal_rows=800)
    sig2.fit(Z.iloc[:3000], y[:3000])
    p2 = sig2.score_parts(Z.iloc[3000:])
    assert "xgb" not in p2.columns
    np.testing.assert_allclose(p2["ensemble"], (0.35 * p2["rf"] + 0.25 * p2["svm"]) / 0.60)


def test_v1_ensemble_needs_labels(stream):
    sig = build("signal", "v1_ensemble", sensors=SENS)
    with pytest.raises(ValueError, match="labels"):
        sig.fit(sig.transform(stream), None)


def test_no_scores_where_no_data(stream):
    s = stream.copy()
    s.iloc[3500:3560, :len(SENS)] = np.nan
    sig = build("signal", "v1_quorum", sensors=SENS, max_kernel_train=1000)
    Z = sig.transform(s)
    sig.fit(Z.iloc[:3000], s["EVENT"].to_numpy()[:3000])
    assert sig.score_parts(Z).iloc[3500:3560].isna().all().all()


def test_signal_labels_must_be_unique():
    specs = _signal_specs(["random_score", {"name": "v1_ensemble", "label": "x", "parts": False}])
    assert specs[1] == ("v1_ensemble", {}, "x", False)
    with pytest.raises(ValueError, match="unique"):
        _signal_specs(["random_score", "random_score"])
