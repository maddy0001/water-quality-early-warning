"""Triage (layer 4, Segment 4): is it the water or the sensor?

    features  per-minute water and sensor evidence (fault checks, 15-min moves, jumps)
    systems   detector_only, rules, learned (registered as kind "triage")
    evaluate  episode outcomes and rates on GECCO-Triage
    run       the S4 experiment and its report
"""

from ewsv2.triage import systems  # noqa: F401
