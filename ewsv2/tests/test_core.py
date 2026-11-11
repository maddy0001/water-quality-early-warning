"""Registry and config behave as documented."""

import pytest

from ewsv2.core.config import ConfigError, load_config, parse_overrides
from ewsv2.core.registry import KINDS, RegistryError, available, build, get, register


def test_sources_are_registered():
    assert {"gecco2018", "gecco2019", "csv"} <= set(available("source")["source"])


def test_every_kind_listed():
    assert set(available()) == set(KINDS)


def test_unknown_name_lists_alternatives():
    with pytest.raises(RegistryError, match="Available: .*gecco2018"):
        get("source", "does_not_exist")


def test_unknown_kind_rejected():
    with pytest.raises(RegistryError):
        register("not_a_kind", "x")


def test_register_and_build_new_plugin():
    @register("metric", "_test_metric")
    class Dummy:
        def __init__(self, scale=1):
            self.scale = scale

    m = build("metric", "_test_metric", scale=3)
    assert m.scale == 3 and Dummy.registry_name == "_test_metric"


def test_duplicate_registration_rejected():
    @register("metric", "_dup")
    class A: ...

    with pytest.raises(RegistryError):
        @register("metric", "_dup")
        class B: ...


def test_config_extends_and_overrides():
    cfg = load_config("s0_inspect", parse_overrides(["data.source=gecco2019", "inspect.plots=false"]))
    assert cfg["run_name"] == "s0_inspection"
    assert cfg["data"]["source"] == "gecco2019"
    assert cfg["inspect"]["plots"] is False
    assert cfg["inspect"]["baseline_minutes"] == 60        # kept from file
    assert cfg["pipeline"] == ["load"]                     # inherited from base
    assert cfg["seed"] == 42


def test_config_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("no_such_config")


def test_config_requires_source(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_name: x\ndata: {}\n")
    with pytest.raises(ConfigError, match="data.source"):
        load_config(p)


def test_bad_override_syntax():
    with pytest.raises(ConfigError):
        parse_overrides(["no_equals_sign"])
