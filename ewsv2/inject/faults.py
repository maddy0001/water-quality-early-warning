"""Sensor faults: one sensor misbehaves while the water is fine (Segment 4).

The four types are the standard ones in the sensor-fault literature (e.g. Ni et al.,
ACM TOSN 2009; Sharma, Golubchik and Govindan, ACM TOSN 2010) and are the faults
utilities report most for online water-quality probes:

``spike``    1–4 isolated one-minute outliers within 10 minutes (electrical noise, air bubble)
``drift``    a slowly growing offset over 2–8 hours (fouling, calibration loss), removed at
             the end of the episode (the maintenance visit)
``stuck``    the reading freezes at its last value (frozen transmitter, failed analyser)
``dropout``  the reading falls to 0 (lost 4–20 mA signal, power or cable fault)

Severity sets the size (spike, drift: in normal-variability units) or the duration
(stuck, dropout: minutes).
"""

from __future__ import annotations

import numpy as np

FAULT_TYPES = ("spike", "drift", "stuck", "dropout")

SEVERITY: dict[str, dict[str, tuple[float, float]]] = {
    # spike/drift: magnitude range (units); stuck/dropout: duration range (minutes)
    "spike":   {"low": (8, 12), "medium": (12, 20), "high": (20, 40)},
    "drift":   {"low": (6, 10), "medium": (10, 18), "high": (18, 30)},
    "stuck":   {"low": (60, 180), "medium": (180, 480), "high": (480, 1440)},
    "dropout": {"low": (10, 30), "medium": (30, 120), "high": (120, 360)},
}


def fault_length(kind: str, severity: str, rng: np.random.Generator) -> int:
    """Episode length in minutes."""
    if kind == "spike":
        return int(rng.integers(1, 11))
    if kind == "drift":
        return int(rng.integers(120, 481))
    lo, hi = SEVERITY[kind][severity]
    return int(rng.integers(int(lo), int(hi) + 1))


def apply_fault(values: np.ndarray, start: int, length: int, kind: str, severity: str,
                unit: float, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Return a faulty copy of one sensor's values and a description of what was done.

    ``values`` is the sensor's full series; positions ``start .. start+length-1`` are changed.
    """
    v = values.astype(float).copy()
    end = start + length
    info: dict = {}
    if kind == "spike":
        n = int(rng.integers(1, 5))
        mag = rng.uniform(*SEVERITY["spike"][severity])
        sign = rng.choice([-1.0, 1.0])
        # first spike at ``start``; the others at distinct later minutes of the window,
        # never adjacent, so each is a one-minute outlier
        later = np.arange(start + 2, end, 2)
        extra = rng.choice(later, size=min(n - 1, len(later)), replace=False) if len(later) and n > 1 else []
        pos = np.sort(np.concatenate([[start], np.asarray(extra, dtype=int)])).astype(int)
        for p in pos:
            v[p] = values[p] + sign * mag * unit * rng.uniform(0.8, 1.2)
        info = {"magnitude_units": round(float(sign * mag), 2), "spikes": int(len(pos)),
                "length": int(pos[-1] - start + 1)}
    elif kind == "drift":
        final = rng.uniform(*SEVERITY["drift"][severity]) * rng.choice([-1.0, 1.0])
        v[start:end] = values[start:end] + np.linspace(0, final, length) * unit
        info = {"magnitude_units": round(float(final), 2)}
    elif kind == "stuck":
        before = values[:start][np.isfinite(values[:start])]
        frozen = before[-1] if len(before) else np.nanmedian(values)
        v[start:end] = frozen
        info = {"frozen_at": float(frozen)}
    elif kind == "dropout":
        v[start:end] = 0.0
        info = {"value": 0.0}
    else:
        raise ValueError(f"unknown fault type {kind!r}; use one of {FAULT_TYPES}")
    v[~np.isfinite(values)] = np.nan          # missing stays missing
    return v, info
