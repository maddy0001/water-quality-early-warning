"""ewsv2 – Water Quality Early Warning System, version 2.

Station-level, calibrated triage of drinking-water quality alarms
(Normal / Water event / Sensor fault) on real utility sensor data.

Every part of the system is a plug-in registered by name and chosen in a
YAML config. See ``ewsv2/docs/SEGMENT_00.md`` for the overview.

Version 1 of the project (paper 1, ICESC 2026) lives at the repository root
(``main.py``, ``src/``, ``pipelines/``) and is never imported or modified here.
"""

__version__ = "2.0.0.dev0"
SEGMENT = "S5"
