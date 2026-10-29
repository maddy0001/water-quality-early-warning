"""YAML configs.

A config file picks plug-ins by name and gives their parameters, for example::

    run_name: s0_inspect
    data:
      source: gecco2018
      params: {data_dir: data/raw/gecco}

Configs may ``extends:`` another config file; keys in the child override the
parent (nested dictionaries are merged, lists are replaced).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ewsv2.core.paths import CONFIG_DIR, resolve

REQUIRED_KEYS = ("run_name", "data")


class ConfigError(ValueError):
    """Raised for a missing or malformed config."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _find(path: str | Path) -> Path:
    """Accept a full path, a repo-relative path, or a bare name in ewsv2/configs."""
    p = Path(path)
    candidates = [resolve(p), CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml"]
    for c in candidates:
        if c.is_file():
            return c
    raise ConfigError(f"Config not found: {path}. Looked in: " + ", ".join(map(str, candidates)))


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config (following ``extends:``), apply overrides, validate."""
    file = _find(path)
    with file.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{file} must contain a mapping at the top level")

    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(parent), cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"{file}: missing required key(s): {', '.join(missing)}")
    if "source" not in cfg["data"]:
        raise ConfigError(f"{file}: data.source is required (e.g. gecco2018)")

    cfg.setdefault("seed", 42)
    cfg["_file"] = str(file)
    return cfg


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Turn CLI ``--set a.b=1`` items into a nested dict (values parsed as YAML)."""
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return result
