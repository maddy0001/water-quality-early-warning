"""Evaluation (Segment 1): splits, event metrics, thresholds, bootstrap, leakage audit.

Importing this package registers the splitters (``random``, ``chrono``, ``official``)
with the registry.
"""

from ewsv2.evaluate import splits  # noqa: F401  (registers splitters)
