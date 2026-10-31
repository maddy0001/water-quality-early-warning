"""Data sources (layer 1 of the architecture).

Importing this package registers every source with the registry:

    gecco2018  – GECCO Industrial Challenge 2018 (main dataset)
    gecco2019  – GECCO Industrial Challenge 2019 (transfer test)
    csv        – any CSV with a time column, sensor columns and optional EVENT
    simulated  – a made-up station for days without data files (Segment 8; ``data.fallback``)
"""

from ewsv2.data import csv_file, gecco, simulated  # noqa: F401  (imported to register sources)
