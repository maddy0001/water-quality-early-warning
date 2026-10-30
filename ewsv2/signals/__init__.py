"""Signals (layer 3). Importing registers them.

Segment 1 – baselines, the bar every later signal must clear:
    random_score, squared_diff, moving_variance, rolling_zscore, static_limits,
    isolation_forest, random_forest

Segments 2–4 add the ported v1 modules (four-detector quorum, RF+XGBoost+SVM
ensemble, BiLSTM-attention forecast surprise) and the fault checks.
"""

from ewsv2.signals import baselines  # noqa: F401
