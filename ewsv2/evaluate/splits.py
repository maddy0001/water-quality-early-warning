"""Train / validation / test splits (Segment 1).

Three splitters, registered by name:

``official``  (honest) – the dataset's own periods. Test = the official test period,
               untouched. Thresholds come from blocked cross-validation over the
               pre-test period, with a ``gap_minutes`` buffer so neighbouring minutes
               never sit on both sides of a boundary.
``chrono``    – time-ordered fractions of one period (e.g. the first half of the
               2018 training file vs the second half, as in the NIPS-TS-GECCO benchmark).
``random``    (leaky)  – minutes shuffled at random into fit / val / test. Neighbouring,
               almost identical minutes land on both sides; kept only for the audit.

Rows where every sensor is missing are never assigned to any block.
"""

from __future__ import annotations

import numpy as np

from ewsv2.core.registry import register
from ewsv2.core.types import Dataset, Split


def has_data(ds: Dataset) -> np.ndarray:
    return ~ds.frame[ds.sensors].isna().all(axis=1).to_numpy()


def _period_mask(ds: Dataset, name: str) -> np.ndarray:
    a, b = ds.periods[name]
    idx = ds.frame.index
    return np.asarray((idx >= a) & (idx <= b))


def _trim_tail(mask: np.ndarray, rows: int) -> np.ndarray:
    """Drop the last ``rows`` True positions of a contiguous block."""
    if rows <= 0 or not mask.any():
        return mask
    out = mask.copy()
    pos = np.flatnonzero(out)
    out[pos[-min(rows, len(pos)):]] = False
    return out


@register("splitter", "official")
class OfficialSplit:
    """Honest split on the dataset's own periods.

    * test  = the official test period (untouched)
    * fit   = everything before it (train, plus valid if the dataset has one), minus a
              ``gap_minutes`` buffer right before the test period
    * val   = the same pre-test rows, scored **out of fold**: the pre-test period is cut
              into ``cv_folds`` consecutive blocks; each block is scored by a model fitted
              on the other blocks, with ``gap_minutes`` purged on both sides of the block.
              Thresholds are chosen on these out-of-fold scores, then the model is refitted
              on all of ``fit`` and applied once to the test period.

    With ``cv_folds=0`` the old single split is used instead: validation = the official
    valid period, or the last ``val_frac`` of train.
    """

    def __init__(self, cv_folds: int = 4, gap_minutes: int = 1440, val_frac: float = 0.25):
        self.cv_folds = int(cv_folds)
        self.gap_minutes = int(gap_minutes)
        self.val_frac = float(val_frac)

    def split(self, ds: Dataset) -> Split:
        data = has_data(ds)
        if "test" not in ds.periods:
            raise ValueError(f"{ds.name} has no official test period; use the 'chrono' splitter")
        test = _period_mask(ds, "test") & data
        pre = np.zeros(len(data), dtype=bool)
        for name in ("train", "valid"):
            if name in ds.periods:
                pre |= _period_mask(ds, name)
        n = len(data)

        if self.cv_folds >= 2:
            fit = _trim_tail(pre, self.gap_minutes) & data
            pos = np.flatnonzero(pre)
            blocks = np.array_split(pos, self.cv_folds)
            folds = []
            for blk in blocks:
                val_k = np.zeros(n, dtype=bool)
                val_k[blk] = True
                lo, hi = max(blk[0] - self.gap_minutes, 0), min(blk[-1] + self.gap_minutes, n - 1)
                purge = np.zeros(n, dtype=bool)
                purge[lo:hi + 1] = True
                folds.append((pre & ~purge & data, val_k & data))
            return Split(
                name="official", fit=fit, val=pre & data, test=test, leaky=False, folds=folds,
                description=(f"official test period; threshold from {self.cv_folds}-fold blocked "
                             f"cross-validation over the pre-test period ({self.gap_minutes} min purged "
                             f"around each block); model refitted on all pre-test data"),
            )

        if "valid" in ds.periods:
            train_all, val = _period_mask(ds, "train"), _period_mask(ds, "valid")
        else:
            train_all = _period_mask(ds, "train")
            pos = np.flatnonzero(train_all)
            cut = pos[int(len(pos) * (1 - self.val_frac))]
            val = train_all & (np.arange(n) >= cut)
            train_all = train_all & ~val
        fit = _trim_tail(train_all, self.gap_minutes) & data
        val = _trim_tail(val, self.gap_minutes) & data
        return Split(
            name="official", fit=fit, val=val, test=test, leaky=False,
            description=(f"official test period; validation = "
                         f"{'official valid period' if 'valid' in ds.periods else f'last {self.val_frac:.0%} of train'}; "
                         f"{self.gap_minutes} min gap before each later block"),
        )


@register("splitter", "chrono")
class ChronoSplit:
    def __init__(self, period: str | None = None, fit_frac: float = 0.5, val_frac: float = 0.0,
                 gap_minutes: int = 0):
        self.period = period
        self.fit_frac, self.val_frac = float(fit_frac), float(val_frac)
        self.gap_minutes = int(gap_minutes)

    def split(self, ds: Dataset) -> Split:
        data = has_data(ds)
        scope = _period_mask(ds, self.period) if self.period else np.ones(len(ds.frame), dtype=bool)
        pos = np.flatnonzero(scope)
        n = len(pos)
        i_val, i_test = int(n * self.fit_frac), int(n * (self.fit_frac + self.val_frac))
        order = np.zeros(len(scope), dtype=int)  # 0 outside, 1 fit, 2 val, 3 test
        order[pos[:i_val]], order[pos[i_val:i_test]], order[pos[i_test:]] = 1, 2, 3
        fit = _trim_tail(order == 1, self.gap_minutes) & data
        val = _trim_tail(order == 2, self.gap_minutes) & data
        test = (order == 3) & data
        where = f"period '{self.period}'" if self.period else "whole dataset"
        return Split(
            name="chrono", fit=fit, val=val, test=test, leaky=False,
            description=f"time-ordered {self.fit_frac:.0%}/{self.val_frac:.0%}/rest of {where}",
        )


@register("splitter", "random")
class RandomSplit:
    def __init__(self, test_frac: float = 0.3, val_frac: float = 0.0, seed: int = 42):
        self.test_frac, self.val_frac, self.seed = float(test_frac), float(val_frac), int(seed)

    def split(self, ds: Dataset) -> Split:
        data = has_data(ds)
        pos = np.flatnonzero(data)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(pos)
        n_test, n_val = int(len(pos) * self.test_frac), int(len(pos) * self.val_frac)
        masks = [np.zeros(len(data), dtype=bool) for _ in range(3)]
        masks[2][pos[:n_test]] = True
        masks[1][pos[n_test:n_test + n_val]] = True
        masks[0][pos[n_test + n_val:]] = True
        return Split(
            name="random", fit=masks[0], val=masks[1], test=masks[2], leaky=True,
            description=f"minutes shuffled at random: {1 - self.test_frac - self.val_frac:.0%} fit, "
                        f"{self.val_frac:.0%} val, {self.test_frac:.0%} test (LEAKY)",
        )
