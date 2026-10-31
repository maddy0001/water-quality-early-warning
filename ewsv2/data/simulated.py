"""A simulated station: the stream the laptop makes up when no data files or sensors are there.

    data: {source: simulated, params: {days: 60, seed: 7}}

Nine sensors with the GECCO 2018 names and typical levels, daily cycles, slow drift, noise and the
reading resolution of the real station; labelled "real" water events (pH and redox drop, chlorine
and conductivity react, as in the GECCO events), a few short data gaps, and official-style
train/test periods (first 60% / last 40%). Fully reproducible from the seed.

It exists so that every command still runs on a day without the GECCO files (``data.fallback``),
in tests, and in the live demo. Numbers measured on it say nothing about real water.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ewsv2.core.registry import register
from ewsv2.core.types import CLASS_COL, EVENT_COL, Dataset

#            level   daily  drift  noise   resolution
_SENSORS = {
    "Tp":    (8.0,   0.30,  0.20,  0.02,   0.1),
    "Cl":    (0.17,  0.02,  0.01,  0.004,  0.0001),
    "pH":    (8.35,  0.03,  0.03,  0.012,  0.001),
    "Redox": (755.0, 6.0,   4.0,   0.8,    0.1),
    "Leit":  (210.0, 3.0,   2.0,   0.5,    0.1),
    "Trueb": (0.020, 0.002, 0.002, 0.0012, 0.0001),
    "Cl_2":  (0.11,  0.010, 0.006, 0.002,  0.00001),
    "Fm":    (1500., 300.,  60.,   25.,    1.0),
    "Fm_2":  (900.,  150.,  40.,   15.,    1.0),
}
#            direction during an event, in units of the sensor's noise × size
_EVENT = {"pH": -1.0, "Redox": -1.0, "Cl": -0.6, "Cl_2": -0.6, "Leit": 0.5, "Trueb": 0.4}


@register("source", "simulated")
class SimulatedSource:
    name = "simulated"

    def __init__(self, days: int = 60, seed: int = 7, events_per_day: float = 0.5, start: str = "2024-01-01"):
        self.days, self.seed, self.rate, self.start = int(days), int(seed), float(events_per_day), start

    def load(self) -> Dataset:
        rng = np.random.default_rng(self.seed)
        n = self.days * 1440
        t = pd.date_range(self.start, periods=n, freq="1min")
        day = np.sin(np.arange(n) * 2 * np.pi / 1440)
        cols = {}
        for s, (level, daily, drift, noise, res) in _SENSORS.items():
            walk = np.cumsum(rng.standard_normal(n)) / np.sqrt(n) * drift * 4      # slow drift
            ar = lfilter([1.0], [1.0, -0.7], rng.standard_normal(n) * noise)     # correlated noise
            cols[s] = level + daily * day + walk + ar
        ev = np.zeros(n, dtype=int)
        n_events = rng.poisson(self.rate * self.days)
        for _ in range(n_events):
            a = int(rng.integers(1440, n - 300))
            length = int(rng.integers(10, 120))
            size = rng.uniform(4, 12)
            ramp = np.minimum(1.0, np.arange(length) / 5.0)
            for s, sign in _EVENT.items():
                noise = _SENSORS[s][3]
                cols[s][a:a + length] += sign * size * noise * 3 * ramp
            ev[a:a + length] = 1
        df = pd.DataFrame({s: np.round(v / _SENSORS[s][4]) * _SENSORS[s][4] for s, v in cols.items()},
                          index=pd.DatetimeIndex(t, name="time"))
        for _ in range(max(1, self.days // 10)):                                # short data gaps
            a = int(rng.integers(0, n - 30))
            df.iloc[a:a + int(rng.integers(2, 20))] = np.nan
        df[EVENT_COL] = ev
        df[CLASS_COL] = ev
        split = t[int(n * 0.6)]
        periods = {"train": (t[0], split - pd.Timedelta(minutes=1)), "test": (split, t[-1])}
        return Dataset(self.name, df, list(_SENSORS), periods,
                       meta={"note": "simulated station, not real water", "seed": self.seed, "days": self.days,
                             "events": int(n_events)})
