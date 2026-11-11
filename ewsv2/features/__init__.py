"""Feature builders (layer 2). Importing registers them.

    basic – per-sensor value, short differences, rolling deviations and spreads (S1)
    rich  – multi-window deviations, spreads and flat-line flags (S2)

``rich.FeaturePipeline`` adds fitted scaling and the coherence model (S2).
"""

from ewsv2.features import basic, rich  # noqa: F401
