"""Signals (layer 3). Importing registers them.

Segment 1 – baselines, the bar every later signal must clear:
    random_score, squared_diff, moving_variance, rolling_zscore, static_limits,
    isolation_forest, random_forest

Segment 2 – the paper-1 models, ported to one-minute streams:
    v1_quorum   (Isolation Forest, LOF, One-Class SVM, autoencoder; 2-of-4 vote)
    v1_ensemble (Random Forest + XGBoost + SVM; weighted soft vote)

Segments 3–4 add the BiLSTM-attention forecast surprise and the fault checks.
"""

from ewsv2.signals import baselines, v1_port  # noqa: F401
