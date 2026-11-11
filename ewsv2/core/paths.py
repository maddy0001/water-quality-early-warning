"""Where things live.

All paths are resolved from the repository root (the folder that contains
``ewsv2/``), so commands work no matter which folder you run them from.

* ``data/raw/gecco/``  – GECCO CSV files (downloaded by ``python -m ewsv2 fetch``)
* ``outputs_v2/``      – everything v2 produces (v1's ``outputs/`` is untouched)
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent

# Environment variables let you move data or outputs without editing configs.
DATA_DIR = Path(os.environ.get("EWSV2_DATA_DIR", REPO_ROOT / "data" / "raw" / "gecco"))
OUTPUT_DIR = Path(os.environ.get("EWSV2_OUTPUT_DIR", REPO_ROOT / "outputs_v2"))
CONFIG_DIR = PACKAGE_DIR / "configs"


def resolve(path: str | os.PathLike[str]) -> Path:
    """Absolute path; relative paths are taken from the repository root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def output_dir(*parts: str) -> Path:
    """A folder under ``outputs_v2/``, created if missing."""
    d = OUTPUT_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
