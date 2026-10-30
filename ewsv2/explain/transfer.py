"""Segment 6, experiment E10: train at one station, run at another (GECCO 2018 → 2019).

    python -m ewsv2 transfer --config s6_explain

A utility will not have labelled events for every new monitoring station. How much of
the system still works when it is moved?

1. **Common sensors.** Both stations measure temperature, pH, conductivity and turbidity
   (``transfer.roles``); each station's sensors are renamed to these roles.
2. **Station normalisation, no labels.** Every sensor is centred on its pre-test median and
   divided by its normal-variability unit (``features/scale.py``), computed from pre-test
   minutes of that station. 2018 conductivity (µS/cm, ~210) and 2019 conductivity (~0.02)
   end up on the same scale.
3. **Modules.** The classifier (C) can only be trained where there are labels: on the 2018
   training period. The unsupervised modules (A quorum, F forecast) are trained either on
   2018 too ("from 2018") or on the new station's own pre-test data, which needs no labels
   ("fitted at 2019").
4. **Thresholds** (budget 1 false alarm/day):
   - ``zero-shot``: nines and budget threshold from the 2018 test period, used unchanged;
   - ``label-free``: nines and budget threshold from 2019 pre-test minutes, treating them all
     as normal (no 2019 labels used);
   - ``adaptive``: the label-free threshold, then corrected online from the operator's verdicts.

Scored on the 2019 official test period (12 real events), without injection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ewsv2.calibrate.methods import Adaptive, Nines, budget_threshold, to_alarm
from ewsv2.core.paths import OUTPUT_DIR
from ewsv2.core.registry import build
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset
from ewsv2.evaluate.audit import _parts_of
from ewsv2.evaluate.metrics import EventEvaluator
from ewsv2.features.scale import sensor_units
from ewsv2.inject.benchmark import pretest_mask, test_mask
from ewsv2.inject.signatures import ROLES

log = logging.getLogger(__name__)

MODULES = {
    "C": {"name": "v1_ensemble", "part": "ensemble", "params": {}},
    "A": {"name": "v1_quorum", "part": "quorum", "params": {}},
    "F": {"name": "forecast_linear", "part": "surprise_mean", "params": {"lookback": 30, "horizon": 5}},
}


def normalise(ds: Dataset, roles: list[str]) -> Dataset:
    """Rename sensors to their roles and put each on its own station's normal scale (no labels used)."""
    pick = {}
    for r in roles:
        found = [s for s in ds.sensors if ROLES.get(s) == r]
        if not found:
            raise ValueError(f"{ds.name} has no sensor with role {r!r}")
        pick[r] = found[0]
    f = ds.frame
    pre = pretest_mask(ds) & ~f[ds.sensors].isna().all(axis=1).to_numpy()
    units = sensor_units(f, list(pick.values()), pre)
    out = pd.DataFrame(index=f.index)
    for r, s in pick.items():
        x = f[s].astype(float)
        out[r] = (x - x[pre].median()) / units[s]
    out[EVENT_COL] = f[EVENT_COL].to_numpy()
    out[CLASS_COL] = f[CLASS_COL].to_numpy()
    meta = {**ds.meta, "transfer_sensors": pick, "transfer_units": units.to_dict()}
    return Dataset(f"{ds.name}_norm", out, list(roles), dict(ds.periods), meta)


def _fit(spec: dict, ds: Dataset, rows: np.ndarray, seed: int, use_labels: bool = True):
    model = build("signal", spec["name"], sensors=ds.sensors, seed=seed, **spec["params"])
    Z = model.transform(ds.frame)
    y = ds.frame[EVENT_COL].to_numpy().astype(int) if use_labels else np.zeros(len(ds.frame), dtype=int)
    return model.fit(Z[rows], y[rows])


def _score(model, spec: dict, ds: Dataset) -> np.ndarray:
    Z = model.transform(ds.frame)
    parts = _parts_of(model, Z)
    col = spec["part"] if spec["part"] in parts.columns else parts.columns[0]
    return parts[col].to_numpy(dtype=float)


def _unit(n: np.ndarray) -> np.ndarray:
    return 1.0 - np.power(10.0, -np.clip(n, 0, None))


def run_transfer(cfg: dict, out_dir: Path) -> pd.DataFrame:
    tcfg = cfg.get("transfer") or {}
    seed = int(cfg.get("seed", 42))
    roles = list(tcfg.get("roles", ["temperature", "ph", "conductivity", "turbidity"]))
    budget = float(tcfg.get("budget", 1.0))
    gap, max_ep = 5, 60
    out_dir.mkdir(parents=True, exist_ok=True)

    src_raw = build("source", tcfg.get("source", "gecco2018")).load()
    tgt_raw = build("source", tcfg.get("target", "gecco2019")).load()
    src, tgt = normalise(src_raw, roles), normalise(tgt_raw, roles)
    log.info("Transfer %s -> %s on %s (source sensors %s, target sensors %s)", src_raw.name, tgt_raw.name, roles,
             src.meta["transfer_sensors"], tgt.meta["transfer_sensors"])

    def has(ds):
        return ~ds.frame[ds.sensors].isna().all(axis=1).to_numpy()

    s_has, t_has = has(src), has(tgt)
    s_pre, s_test = pretest_mask(src) & s_has, test_mask(src) & s_has
    t_pre, t_test = pretest_mask(tgt) & t_has, test_mask(tgt) & t_has
    s_y = src.frame[EVENT_COL].to_numpy().astype(int)
    t_y = tgt.frame[EVENT_COL].to_numpy().astype(int)
    idx = tgt.frame.index

    # module scores on the target: trained on the source, and (A, F) trained on the target's own pre-test data
    scores = {}
    for role, spec in MODULES.items():
        t0 = time.perf_counter()
        m = _fit(spec, src, s_pre, seed)
        scores[(role, "from 2018")] = (_score(m, spec, tgt), _score(m, spec, src))
        if role != "C":
            mt = _fit(spec, tgt, t_pre, seed, use_labels=False)    # no 2019 labels at all
            scores[(role, "fitted at 2019")] = (_score(mt, spec, tgt), None)
        log.info("  module %s %-16s %5.0fs", role, spec["name"], time.perf_counter() - t0)

    ev = EventEvaluator(t_y, idx, t_test, 0, gap, max_ep)

    def evaluate(label, alarm, threshold_kind):
        r = ev.evaluate(alarm & t_test)
        return {"system": label, "threshold": threshold_kind, "caught": r.detected, "events": r.n_events,
                "false_alarms_per_day": r.false_alarms_per_day, "median_delay_min": r.median_delay_min,
                "event_f1": r.event_f1}

    # nines of each module: label-free (reference = target pre-test minutes, all treated as normal)
    # and zero-shot (reference = normal minutes of the source test period)
    def label_free(role, origin):
        t_scores = scores[(role, origin)][0]
        return Nines(t_scores[t_pre])(t_scores)

    def zero_shot(role):
        t_scores, s_scores = scores[(role, "from 2018")]
        ref = Nines(s_scores[s_test & (s_y == 0)])
        return ref(t_scores), ref(s_scores)

    systems = {}
    for role in MODULES:
        systems[f"{role} alone (from 2018)"] = (label_free(role, "from 2018"), zero_shot(role))
        if role != "C":
            systems[f"{role} alone (fitted at 2019)"] = (label_free(role, "fitted at 2019"), None)
    # paper-1 hand weights 0.40 C + 0.30 F + 0.30 A, on the [0, 1] scale
    w = {"C": 0.4, "F": 0.3, "A": 0.3}

    def hand(parts):
        return sum(w[r] * _unit(np.nan_to_num(parts[r], nan=0.0)) for r in MODULES)

    zs = {r: zero_shot(r) for r in MODULES}
    t_h, s_h = hand({r: zs[r][0] for r in MODULES}), hand({r: zs[r][1] for r in MODULES})
    ref = Nines(s_h[s_test & (s_y == 0)])
    h18 = hand({r: label_free(r, "from 2018") for r in MODULES})
    systems["hand 0.40/0.30/0.30 (all from 2018)"] = (Nines(h18[t_pre])(h18), (ref(t_h), ref(s_h)))
    hmix = hand({"C": label_free("C", "from 2018"), "A": label_free("A", "fitted at 2019"),
                 "F": label_free("F", "fitted at 2019")})
    systems["hand (C from 2018, A+F fitted at 2019)"] = (Nines(hmix[t_pre])(hmix), None)

    rows = []
    zeros = np.zeros(len(idx), dtype=int)
    for label, (lf, zs) in systems.items():
        if zs is not None:
            t_n, s_n = zs
            thr, _ = budget_threshold(s_n, s_y, src.frame.index, s_test, budget, gap, max_ep)
            rows.append(evaluate(label, to_alarm(t_n, thr), "zero-shot (2018 threshold)"))
        thr, _ = budget_threshold(lf, zeros, idx, t_pre, budget, gap, max_ep)
        rows.append(evaluate(label, to_alarm(lf, thr), "label-free (2019 pre-test)"))
        ad = Adaptive(budget=budget)
        ad.threshold = thr
        rows.append(evaluate(label, ad.alarm(lf, t_y, idx, t_test), "adaptive"))
        log.info("  %-42s %s", label, " | ".join(f"{r['threshold'].split()[0]} {r['caught']}/{r['events']} "
                                                  f"{r['false_alarms_per_day']:.2f}/d" for r in rows[-3:]
                                                  if r["system"] == label))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "transfer_results.csv", index=False)
    write_report(src, tgt, res, budget, out_dir)
    return res


def _f(v, nd=2):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_report(src, tgt, res, budget, out_dir: Path) -> None:
    L = ["# Transfer GECCO 2018 → GECCO 2019 (E10)", "",
         f"Common sensors: " + ", ".join(f"{r} ({src.meta['transfer_sensors'][r]} → {tgt.meta['transfer_sensors'][r]})"
                                         for r in src.sensors) + ".",
         "Each station is put on its own normal scale using its pre-test minutes only (no labels).", "",
         f"Scored on the 2019 official test period ({int(res['events'].iloc[0])} real events), budget "
         f"{budget:g} false alarm/day.", "",
         "| System | Threshold | Events caught | False alarms / day | Median delay (min) | Event F1 |",
         "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['system']} | {r['threshold']} | {r['caught']}/{r['events']} | {_f(r['false_alarms_per_day'])} | "
                 f"{_f(r['median_delay_min'], 0)} | {_f(r['event_f1'])} |")
    s5 = OUTPUT_DIR / "s5_fusion" / "gecco2019" / "fusion_results.csv"
    if s5.exists():
        f = pd.read_csv(s5)
        f = f[f["calibrator"] == "adaptive"]
        L += ["", "For comparison, trained and calibrated on 2019 itself (S5, adaptive threshold, all six sensors):", "",
              "| Fusion | Real events | False alarms / day |", "|---|---|---|"]
        for _, r in f.iterrows():
            L.append(f"| {r['fusion']} | {r['real_caught']}/{r['real_n']} | {_f(r['false_alarms_per_day'])} |")
    L += ["", "Thresholds: **zero-shot** = budget threshold from the 2018 test period, used unchanged; "
          "**label-free** = budget threshold from 2019 pre-test minutes, all treated as normal; "
          "**adaptive** = label-free start, corrected online from the operator's verdicts.",
          "", "No 2019 label is used except by the adaptive threshold, which reads the operator's verdict "
          "after each alarm (as in S5).",
          "", "File: `transfer_results.csv`."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
