"""Feature builders (layer 2). Importing registers them.

    basic – per-sensor value, short differences, rolling deviations and spreads (S1)

Segment 2 adds the coherence features used by the water-vs-sensor triage.
"""

from ewsv2.features import basic  # noqa: F401
