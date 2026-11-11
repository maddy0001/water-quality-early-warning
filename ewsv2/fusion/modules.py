"""Module scores for fusion, scored the cross-conformal way (Segment 5).

S3 showed the *threshold-transfer* problem: thresholds were chosen on out-of-fold scores
of four fold models, then applied to a fifth model refitted on all pre-test data, whose
scores can sit on a different scale (the 2019 BiLSTM caught 0/12 because of it).

Here there is no refit. The four fold models score the test period too, and every score
is turned into *nines* against the same reference: the out-of-fold scores of normal
pre-test minutes. A test minute's nines is the average over the four fold models
(cross-conformal prediction, Vovk 2015). Thresholds and test scores therefore come from
the same models and sit on the same scale.

Supervised modules learn from the real events only; every injected GECCO-Triage minute
is left out of training and of the reference, so injected episodes stay unseen.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ewsv2.calibrate.methods import Nines
from ewsv2.core.registry import build
from ewsv2.evaluate.audit import _parts_of

log = logging.getLogger(__name__)


def _with_lead_in(mask: np.ndarray, minutes: int) -> np.ndarray:
    """Extend every block of ``mask`` backwards (forecasters need the previous minutes)."""
    if minutes <= 0:
        return mask
    m = pd.Series(mask[::-1].astype(float)).rolling(minutes + 1, min_periods=1).max().to_numpy()[::-1]
    return m.astype(bool)


def module_nines(frame: pd.DataFrame, sensors: list[str], split, real: np.ndarray, usable: np.ndarray,
                 spec: dict, seed: int = 42) -> tuple[pd.Series, dict]:
    """Nines of one module: out-of-fold on pre-test rows, fold-averaged on test rows.

    ``usable`` = rows the module may learn from and calibrate on (data present, not injected).
    """
    name, params, part = spec["name"], dict(spec.get("params") or {}), spec.get("part")
    t0 = time.perf_counter()
    probe = build("signal", name, sensors=sensors, seed=seed, **params)
    Z = probe.transform(frame)
    supervised = getattr(probe, "supervised", False)
    lead = int(params.get("lookback", 0)) + int(params.get("horizon", 0)) + 5
    n = len(frame)
    oof = np.full(n, np.nan)
    test_scores = []
    for fit_k, val_k in split.folds:
        rows = fit_k & usable
        if not val_k.any() or (supervised and real[rows].sum() == 0):
            continue
        model = build("signal", name, sensors=sensors, seed=seed, **params).fit(Z[rows], real[rows])
        want = val_k | split.test
        score_rows = _with_lead_in(want, lead)
        parts = _parts_of(model, Z[score_rows])
        col = part if part in parts.columns else parts.columns[0]
        s = np.full(n, np.nan)
        s[score_rows] = parts[col].to_numpy(dtype=float)
        oof[val_k] = s[val_k]
        t = np.full(n, np.nan)
        t[split.test] = s[split.test]
        test_scores.append(t)
    if not test_scores:
        raise ValueError(f"module {name}: no fold could be fitted")
    reference = oof[split.val & usable & (real == 0)]
    nines = Nines(reference)
    out = np.full(n, np.nan)
    out[split.val] = nines(oof[split.val])
    stacked = np.vstack([nines(t) for t in test_scores])[:, split.test]
    seen = np.isfinite(stacked).any(axis=0)                    # minutes some fold model could score
    avg = np.full(stacked.shape[1], np.nan)
    avg[seen] = np.nanmean(stacked[:, seen], axis=0)
    out[split.test] = avg
    info = {"name": name, "part": part, "folds_used": len(test_scores), "seconds": round(time.perf_counter() - t0, 1),
            "reference_minutes": int(np.isfinite(reference).sum())}
    log.info("  module %-2s %-18s %6.1fs  (%d fold models, reference %d normal minutes)", spec["role"],
             f"{name}:{part}" if part else name, info["seconds"], len(test_scores), info["reference_minutes"])
    return pd.Series(out, index=frame.index, name=spec["role"]), info
