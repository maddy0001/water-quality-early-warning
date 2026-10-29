"""Data sources (layer 1 of the architecture).

Importing this package registers every source with the registry:

    gecco2018  – GECCO Industrial Challenge 2018 (main dataset)
    gecco2019  – GECCO Industrial Challenge 2019 (transfer test)
    csv        – any CSV with a time column, sensor columns and optional EVENT

Planned slots: ``esp32_serial`` (Segment 8).
"""

from ewsv2.data import csv_file, gecco  # noqa: F401  (imported to register sources)
