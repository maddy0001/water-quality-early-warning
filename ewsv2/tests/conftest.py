"""Shared test fixtures: small synthetic GECCO-like files, no real data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SENSORS = ["Tp", "Cl", "pH", "Redox", "Leit", "Trueb", "Cl_2", "Fm", "Fm_2"]


def make_stream(start: str, minutes: int, events: list[tuple[int, int]], seed: int = 0) -> pd.DataFrame:
    """Synthetic 1-minute stream. ``events`` = list of (first_row, length)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=minutes, freq="1min")
    df = pd.DataFrame({
        "Time": t,
        "Tp": 7 + 0.01 * rng.standard_normal(minutes),
        "Cl": 0.16 + 0.005 * rng.standard_normal(minutes),
        "pH": 8.3 + 0.02 * rng.standard_normal(minutes),
        "Redox": 760 + 2 * rng.standard_normal(minutes),
        "Leit": 210 + rng.standard_normal(minutes),
        "Trueb": 0.02 + 0.001 * rng.standard_normal(minutes),
        "Cl_2": 0.11 + 0.003 * rng.standard_normal(minutes),
        "Fm": 1500 + 50 * rng.standard_normal(minutes),
        "Fm_2": 900 + 40 * rng.standard_normal(minutes),
        "EVENT": False,
    })
    for first, length in events:
        df.loc[first:first + length - 1, "EVENT"] = True
        df.loc[first:first + length - 1, "pH"] -= 0.5      # events move pH ...
        df.loc[first:first + length - 1, "Redox"] -= 40    # ... and redox
    return df


@pytest.fixture
def gecco_dir(tmp_path):
    """Folder with gecco2018_train.csv / gecco2018_test.csv in the fetch format."""
    train = make_stream("2016-11-08 05:55:00", 120, [(30, 10), (80, 5)], seed=1)
    test = make_stream("2016-11-08 07:55:00", 120, [(50, 20)], seed=2)
    test.loc[100:109, SENSORS] = np.nan        # a 10-minute outage
    for name, df in (("train", train), ("test", test)):
        out = df.rename(columns={"Time": "time"}).assign(EVENT=df["EVENT"].astype(int))
        out.to_csv(tmp_path / f"gecco2018_{name}.csv", index=False)
    return tmp_path
