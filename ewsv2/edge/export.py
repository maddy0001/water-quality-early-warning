"""Build the edge model and everything the Arduino needs (Segment 8).

    python -m ewsv2 edge --config s8_edge --set edge.retrain=true   (the shipped models are used otherwise)

1. Counts: q = unit / 16 per sensor (S4 units), offset O = median of normal pre-test readings in q.
2. Limits from normal pre-test minutes, no labels: move limits (S4 triage), dropout and spike
   limits (S4 fault checks), and an edge stuck limit on counts without the partner model (the S4
   false-flag budget of 0.5 per day, shared by sensors and checks).
3. A small network (3k+2 → 16 → 8 → 1) trained with Keras on the first 75% of the pre-test period
   (real events), threshold on the last 25% for 1 false alarm/day (S5 budget).
4. Exports: ``edge_spec_<dataset>.json`` (read by the reference and the replay), ``edge_model.h``
   (the same numbers as C arrays in flash) and a TensorFlow Lite int8 file for the size comparison.

The model is trained once and shipped, so every computer uses identical weights. A station without
a shipped model (e.g. the simulated one) is trained automatically the first time.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ewsv2.calibrate.methods import budget_threshold
from ewsv2.core.types import EVENT_COL, Dataset
from ewsv2.edge import reference as R
from ewsv2.evaluate.metrics import EventEvaluator
from ewsv2.features.scale import sensor_units
from ewsv2.inject.benchmark import pretest_mask
from ewsv2.signals.faults import FaultChecks
from ewsv2.triage.features import TriageFeatures

log = logging.getLogger(__name__)

EDGE_DIR = Path(__file__).parent
MODEL_DIR = EDGE_DIR / "model"
FIRMWARE_DIR = EDGE_DIR / "firmware" / "ewsv2_edge"
f32 = np.float32


def spec_path(ds_name: str) -> Path:
    return MODEL_DIR / f"edge_spec_{ds_name}.json"


def _stuck_limits(C: np.ndarray, sensors, rows: np.ndarray, flag_rate: float = 0.5, min_stuck: int = 30) -> dict:
    """Edge stuck limit: one more than the budget-th longest natural run of identical counts."""
    k = len(sensors)
    days = rows.sum() / 1440
    budget = max(1, int(np.ceil(flag_rate / (3 * k) * days)))
    rl = R.run_lengths(C)
    out = {}
    for j, s in enumerate(sensors):
        r = rl[:, j]
        ends = np.flatnonzero((r > 0) & (np.append(r[1:], 0) <= r) & rows)
        sizes = np.sort(r[ends])[::-1]
        kth = sizes[budget - 1] if len(sizes) >= budget else 0
        out[s] = int(min(30000, max(min_stuck, kth + 1)))
    return out


def build_spec(ds: Dataset, seed: int = 42, epochs: int = 15, budget: float = 1.0) -> tuple[dict, dict]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    sensors = list(ds.sensors)
    f = ds.frame
    y = f[EVENT_COL].to_numpy().astype(int)
    pre = pretest_mask(ds)
    has = ~f[sensors].isna().all(axis=1).to_numpy()
    normal = pre & has & (y == 0)

    tf_ = TriageFeatures(sensors).fit(f, normal)
    fc = FaultChecks(sensors).fit(f, normal)
    units = sensor_units(f, sensors, normal)
    q = {s: float(units[s]) / R.K for s in sensors}
    offset = {s: int(np.round(np.nanmedian(f[s].to_numpy(dtype=float)[normal]) / q[s])) for s in sensors}
    spec = {"dataset": ds.name, "sensors": sensors, "q": q, "offset": offset,
            "move_limit": {s: float(f32(tf_.move_limit[s])) for s in sensors},
            "low_count": {s: int(np.floor(fc.low_limit[s] / q[s])) - offset[s] for s in sensors},
            "spike2": {s: float(f32(2 * fc.spike_limit[s] * R.K)) for s in sensors}}
    C = R.carry(R.to_counts(f, spec))
    spec["stuck_limit"] = _stuck_limits(C, sensors, normal)
    F, _, _ = R.features(C, spec)

    pos = np.flatnonzero(pre & has)
    cut = pos[int(len(pos) * 0.75)]
    train = pre & has & (np.arange(len(f)) < cut - 1440)          # one-day gap before validation
    val = pre & has & (np.arange(len(f)) >= cut)
    mu = F[train].mean(axis=0).astype(f32)
    sd = np.maximum(F[train].std(axis=0), 1e-3).astype(f32)

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - depends on the TF build
        pass
    net = tf.keras.Sequential([tf.keras.layers.Input((F.shape[1],)),
                               tf.keras.layers.Dense(16, activation="relu"),
                               tf.keras.layers.Dense(8, activation="relu"),
                               tf.keras.layers.Dense(1)])                  # output = log-odds
    net.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    Xtr = ((F[train] - mu) / sd).astype(f32)
    ytr = y[train].astype(f32)
    w1 = len(ytr) / (2.0 * max(ytr.sum(), 1))
    w0 = len(ytr) / (2.0 * max((1 - ytr).sum(), 1))
    net.fit(Xtr, ytr, epochs=epochs, batch_size=512, verbose=0, class_weight={0: w0, 1: w1}, shuffle=True)

    layers = []
    for d in net.layers:
        W, b = d.get_weights()
        scale = float(f32(np.abs(W).max() / 127.0)) or 1.0
        layers.append({"W": W.astype(f32).tolist(), "b": b.astype(f32).tolist(),
                       "Wq": np.clip(np.round(W / scale), -127, 127).astype(int).tolist(), "scale": scale})
    spec["model"] = {"mu": mu.tolist(), "sd": sd.tolist(), "layers": layers,
                     "inputs": [f"{s}__z15" for s in sensors] + [f"{s}__d60" for s in sensors]
                     + [f"{s}__r15" for s in sensors] + ["breadth", "max_move"]}
    lp = R.mlp(F, spec["model"], int8=True).astype(float)
    thr, _ = budget_threshold(np.where(val, lp, np.nan), y, f.index, val, budget)
    spec["theta0"] = float(f32(thr))
    spec["budget"] = float(budget)
    r = EventEvaluator(y, f.index, val).evaluate((lp >= spec["theta0"]) & val)
    n_params = sum(np.size(L["W"]) + np.size(L["b"]) for L in layers)
    spec["info"] = {"params": int(n_params), "float32_bytes": int(4 * n_params),
                    "int8_bytes": int(sum(np.size(L["W"]) for L in layers) + 4 * sum(np.size(L["b"]) + 1 for L in layers)),
                    "validation": {"events": r.n_events, "caught": r.detected,
                                   "false_alarms_per_day": r.false_alarms_per_day},
                    "train_minutes": int(train.sum()), "validation_minutes": int(val.sum())}
    spec["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return spec, {"tflite": _tflite_int8(net, Xtr, seed)}


def _tflite_int8(net, Xtr: np.ndarray, seed: int) -> bytes | None:
    """Full-integer TensorFlow Lite model, only for the size comparison."""
    try:
        import tensorflow as tf

        rng = np.random.default_rng(seed)
        sample = Xtr[rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False)]

        def rep():
            for row in sample:
                yield [row[None, :].astype(f32)]

        conv = tf.lite.TFLiteConverter.from_keras_model(net)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        return conv.convert()
    except Exception as exc:  # pragma: no cover - converter availability varies
        log.warning("TFLite conversion skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------- C header
def _num(v: float) -> str:
    """A float32 as a C literal that converts back to exactly the same float32."""
    t = f"{float(f32(v)):.9g}"
    if not any(c in t for c in ".eEn"):
        t += ".0"
    return t + "f"


def _arr(name: str, values, ctype: str, per_line: int = 8) -> str:
    flat = np.asarray(values).ravel()
    items = [_num(v) for v in flat] if ctype == "float" else [str(int(v)) for v in flat]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return f"static const {ctype} {name}[{len(flat)}] EDGE_FLASH = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def write_header(spec: dict, path: Path) -> None:
    """The model and limits as C arrays (in flash on the Arduino), bit-exact with the reference."""
    s = spec["sensors"]
    m = spec["model"]
    L = m["layers"]
    info = spec.get("info", {})
    decay = f32(f32(R.STEP) * f32(spec.get("budget", 1.0)) / f32(1440))
    out = [f"// Generated by ewsv2 S8 from {spec['dataset']} ({spec.get('created', '')}). Do not edit by hand:",
           "// rebuild with  python -m ewsv2 edge --config s8_edge --set edge.retrain=true",
           "#pragma once", "#include <stdint.h>", "#ifndef EDGE_FLASH", "#define EDGE_FLASH", "#endif", "",
           f'#define EDGE_DATASET "{spec["dataset"]}"',
           f"#define EDGE_K {len(s)}", f"#define EDGE_IN {len(m['mu'])}",
           f"#define EDGE_H1 {len(L[0]['b'])}", f"#define EDGE_H2 {len(L[1]['b'])}",
           f"#define EDGE_PARAMS {info.get('params', 0)}",
           f"#define EDGE_THETA0 {_num(spec['theta0'])}      // start threshold, log-odds",
           f"#define EDGE_DECAY {_num(decay)}       // per minute: step x budget / 1440", "",
           "// sensors: " + ", ".join(s),
           _arr("EDGE_MOVE_LIMIT", [spec["move_limit"][x] for x in s], "float"),
           _arr("EDGE_SPIKE2", [spec["spike2"][x] for x in s], "float"),
           _arr("EDGE_STUCK", [spec["stuck_limit"][x] for x in s], "int16_t"),
           _arr("EDGE_LOW", [spec["low_count"][x] for x in s], "int16_t"),
           _arr("EDGE_BASE", [spec["offset"][x] for x in s], "int32_t"),
           _arr("EDGE_MU", m["mu"], "float"), _arr("EDGE_SD", m["sd"], "float")]
    for i, layer in enumerate(L, 1):
        out += [_arr(f"EDGE_W{i}", layer["W"], "float"), _arr(f"EDGE_B{i}", layer["b"], "float"),
                _arr(f"EDGE_W{i}Q", layer["Wq"], "int8_t"),
                f"#define EDGE_W{i}S {_num(layer['scale'])}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")


def export(ds: Dataset, seed: int = 42, epochs: int = 15, budget: float = 1.0, sketch: bool | None = None) -> dict:
    """Train and write the spec, the TFLite file and (for the sketch's station) the sketch header."""
    spec, extra = build_spec(ds, seed, epochs, budget)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if extra.get("tflite"):
        (MODEL_DIR / f"edge_model_int8_{ds.name}.tflite").write_bytes(extra["tflite"])
        spec["info"]["tflite_int8_bytes"] = len(extra["tflite"])
    spec_path(ds.name).write_text(json.dumps(spec), encoding="utf-8")
    if sketch if sketch is not None else ds.name == "gecco2018":
        write_header(spec, FIRMWARE_DIR / "edge_model.h")
    log.info("Edge model for %s: %d weights, threshold %.4f (log-odds), validation %s", ds.name,
             spec["info"]["params"], spec["theta0"], spec["info"]["validation"])
    return spec


def load_spec(ds_name: str) -> dict | None:
    p = spec_path(ds_name)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
