"""Basic causal features: every value at time t uses only readings up to t.

For each sensor ``s``:

    s               current value
    s__d1, s__d5    change over the last 1 and 5 minutes
    s__dev15/60     value minus its mean over the previous 15 / 60 minutes
    s__sd15/60      standard deviation over the previous 15 / 60 minutes

Short gaps (up to ``ffill_limit`` minutes) are carried forward; rows where every
sensor is missing stay missing, so they are never scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ewsv2.core.registry import register


@register("feature", "basic")
class BasicFeatures:
    def __init__(self, windows: tuple[int, ...] = (15, 60), diffs: tuple[int, ...] = (1, 5), ffill_limit: int = 5):
        self.windows, self.diffs, self.ffill_limit = tuple(windows), tuple(diffs), int(ffill_limit)

    def transform(self, frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        x = frame[sensors].astype(float)
        empty = x.isna().all(axis=1)
        x = x.ffill(limit=self.ffill_limit)
        cols = {s: x[s] for s in sensors}
        for s in sensors:
            for d in self.diffs:
                cols[f"{s}__d{d}"] = x[s].diff(d)
            for w in self.windows:
                past = x[s].shift(1).rolling(w, min_periods=max(2, w // 3))
                cols[f"{s}__dev{w}"] = x[s] - past.mean()
                cols[f"{s}__sd{w}"] = past.std()
        out = pd.DataFrame(cols, index=frame.index)
        out.loc[empty.to_numpy()] = np.nan
        return out
