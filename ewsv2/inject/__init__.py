"""Injection toolkit (layer 0b, Segment 4) – builds the GECCO-Triage benchmark.

    signatures  contaminant response patterns (EPA pipe-loop studies), dose levels
    faults      spike, drift, stuck, dropout
    benchmark   ``gecco_triage`` injector: places episodes, labels them, returns the episode table
"""

from ewsv2.inject import benchmark, faults, signatures  # noqa: F401
