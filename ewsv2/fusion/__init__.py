"""Fusion (layer 4, Segment 5): one score from the three paper-1 modules.

    methods  single, hand (v1 0.40/0.30/0.30), equal, max, stacking
    modules  cross-conformal module scores (fold models, no refit)
    run      the S5 experiment and its report
"""

from ewsv2.fusion import methods  # noqa: F401
