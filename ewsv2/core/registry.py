"""Plug-in registry.

Every swappable part of ewsv2 registers itself under a *kind* and a *name*::

    from ewsv2.core.registry import register

    @register("source", "gecco2018")
    class Gecco2018Source:
        ...

A config then picks it by name, and :func:`build` creates it::

    source = build("source", "gecco2018", data_dir="data/raw/gecco")

Kinds used in the project (one per layer of the architecture):

    source, injector, feature, splitter, signal, fusion, calibrator, triage,
    explainer, reporter, metric, output
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

KINDS = (
    "source",
    "injector",
    "feature",
    "splitter",
    "signal",
    "fusion",
    "calibrator",
    "triage",
    "explainer",
    "reporter",
    "metric",
    "output",
)

# Modules imported on first lookup so that their @register decorators run.
# Add a line here when a new plug-in package is created.
_PLUGIN_PACKAGES = (
    "ewsv2.data",
    "ewsv2.features",
    "ewsv2.evaluate",
    "ewsv2.signals",
    "ewsv2.inject",
    "ewsv2.triage",
)

_REGISTRY: dict[str, dict[str, Any]] = {kind: {} for kind in KINDS}
_loaded = False


class RegistryError(KeyError):
    """Raised when a kind or name is unknown or registered twice."""

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable
        return str(self.args[0]) if self.args else ""


def register(kind: str, name: str) -> Callable[[Any], Any]:
    """Class or function decorator that adds a plug-in to the registry."""
    if kind not in _REGISTRY:
        raise RegistryError(f"Unknown kind {kind!r}. Known kinds: {', '.join(KINDS)}")

    def decorator(obj: Any) -> Any:
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not obj:
            raise RegistryError(f"{kind} {name!r} is already registered by {existing!r}")
        _REGISTRY[kind][name] = obj
        obj.registry_kind = kind
        obj.registry_name = name
        return obj

    return decorator


def _load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    for package in _PLUGIN_PACKAGES:
        importlib.import_module(package)
    _loaded = True


def get(kind: str, name: str) -> Any:
    """Return the registered class or function."""
    _load_plugins()
    if kind not in _REGISTRY:
        raise RegistryError(f"Unknown kind {kind!r}. Known kinds: {', '.join(KINDS)}")
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY[kind])) or "(none yet)"
        raise RegistryError(f"No {kind} named {name!r}. Available: {known}") from None


def build(kind: str, name: str, **params: Any) -> Any:
    """Create a plug-in instance with keyword parameters from the config."""
    return get(kind, name)(**params)


def available(kind: str | None = None) -> dict[str, list[str]]:
    """Names registered per kind (all kinds if ``kind`` is None)."""
    _load_plugins()
    kinds = [kind] if kind else list(KINDS)
    return {k: sorted(_REGISTRY[k]) for k in kinds}
