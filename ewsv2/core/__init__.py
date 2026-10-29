"""Core building blocks shared by every ewsv2 module.

- ``types``      : data containers (Dataset, SignalOutput, Alarm) and column names
- ``interfaces`` : the plug-in contracts (DataSource, Signal, Fusion, ...)
- ``registry``   : register plug-ins by name and build them from a config
- ``config``     : load and validate YAML configs
- ``paths``      : where data and outputs live
"""

from ewsv2.core.registry import available, build, get, register  # noqa: F401
