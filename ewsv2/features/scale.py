"""How much does each sensor normally move? (Segment 4)

Injected changes and fault checks are both measured in units of a sensor's normal
short-term variability, so one rule works for chlorine (0.01 mg/L steps) and for
flow (thousands of m³/h).

``unit`` = winsorised RMS of (value − mean of the previous 60 min) over normal minutes,
never smaller than the sensor's reading resolution (the smallest step it reports).
GECCO sensors are coarsely quantised; a plain median-based scale is 0 for most of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reading_step(values: pd.Series) -> float:
    """Smallest non-zero difference between distinct readings (the sensor's resolution)."""
    v = np.unique(np.round(values.dropna().to_numpy(dtype=float), 9))
    if len(v) < 2:
        return 0.0
    d = np.diff(v)
    d = d[d > 1e-9]
    return float(np.round(d.min(), 9)) if len(d) else 0.0


def sensor_units(frame: pd.DataFrame, sensors: list[str], rows=None, window: int = 60) -> pd.Series:
    """One 'normal variability' unit per sensor, from the given rows (default: all)."""
    x = frame[sensors].astype(float).ffill(limit=5)
    rows = np.ones(len(frame), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    out = {}
    for s in sensors:
        dev = (x[s] - x[s].shift(1).rolling(window, min_periods=window // 3).mean())[rows].dropna()
        if len(dev):
            lo, hi = dev.quantile([0.005, 0.995])
            rms = float(np.sqrt((dev.clip(lo, hi) ** 2).mean()))
        else:
            rms = 0.0
        step = reading_step(x[s][rows])
        unit = max(rms, step)
        if unit <= 0:
            unit = float(x[s][rows].abs().median() * 0.01) or 1.0
        out[s] = unit
    return pd.Series(out, name="unit")
