"""Arduino Uno edge (Segment 8): reference algorithm, header, C++ builds, replay and fallbacks.

The C++ tests compile the firmware logic for this computer (g++/clang++); the simulated-Uno test
needs avr-gcc, the Arduino AVR core and simavr (Linux). Each is skipped where its tools are missing.
"""

import json
import os
import sys

import numpy as np
import pytest

from ewsv2.cli import _load_dataset, build_parser
from ewsv2.core.registry import build
from ewsv2.core.types import EVENT_COL
from ewsv2.edge import export as E
from ewsv2.edge import hil as H
from ewsv2.edge import reference as R

from .test_inject import station

no_cxx = pytest.mark.skipif(H.compiler() is None, reason="no C++ compiler")
UNITS = {"Tp": 0.1, "Cl": 0.01, "pH": 0.02, "Redox": 1.0, "Leit": 1.0, "Trueb": 0.001, "Cl_2": 0.002,
         "Fm": 30.0, "Fm_2": 20.0}


def tiny_spec(sensors, seed=0, theta0=-1.0):
    """A random small network and simple limits: exercises every branch, not a trained model."""
    rng = np.random.default_rng(seed)
    k = len(sensors)
    sizes = [3 * k + 2, 16, 8, 1]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0, 0.6, (a, b)).astype(np.float32)
        scale = float(np.float32(np.abs(W).max() / 127.0))
        layers.append({"W": W.tolist(), "b": rng.normal(0, 0.2, b).astype(np.float32).tolist(),
                       "Wq": np.clip(np.round(W / scale), -127, 127).astype(int).tolist(), "scale": scale})
    q = {s: UNITS[s] * (1.0137 + 0.0173 * i) / R.K for i, s in enumerate(sensors)}
    return {"dataset": "synthetic", "sensors": list(sensors), "created": "test", "budget": 1.0, "theta0": theta0,
            "q": q, "offset": {s: 0 for s in sensors}, "move_limit": {s: float(np.float32(3.07)) for s in sensors},
            "stuck_limit": {s: 40 for s in sensors}, "low_count": {s: 8 for s in sensors},
            "spike2": {s: float(np.float32(2 * 8 * R.K)) for s in sensors},
            "model": {"mu": [0.0] * (3 * k + 2), "sd": [1.0] * (3 * k + 2), "layers": layers},
            "info": {"params": sum(np.size(L["W"]) + np.size(L["b"]) for L in layers)}}


def stream(days=3, seed=1):
    """A synthetic station with a stuck sensor, a dropout, a spike, a lone jump, gaps and a water event."""
    ds = station(days=days, seed=seed, events=((1500, 30),))
    f = ds.frame.copy()
    c = f.columns.get_loc
    f.iloc[300:420, c("Leit")] = f["Leit"].iloc[300]                 # stuck
    f.iloc[800:860, c("Fm")] = 0.0                                     # dropout
    f.iloc[1000, c("Redox")] += 40                                     # spike
    f.iloc[2200:, c("Tp")] += 3.0                                      # lone jump (stays)
    f.iloc[2600:2610, :9] = np.nan                                     # gap in the data
    f.iloc[3000:3003, c("pH")] = np.nan                                # short gap, carried
    return ds, f


def test_counts_are_integers_and_reference_is_causal():
    ds, f = stream()
    spec = tiny_spec(ds.sensors)
    C = R.to_counts(f, spec)
    ok = np.isfinite(C)
    assert np.all(C[ok] == np.round(C[ok])) and np.abs(C[ok]).max() <= R.COUNT_MAX
    y = f[EVENT_COL].to_numpy()
    out = R.run_counts(C, spec, y_water=y)
    C2 = C.copy()
    C2[2000:] = C2[2000:] * 2 + 7                                       # change only the future
    out2 = R.run_counts(C2, spec, y_water=y)
    cols = ["alarm", "water", "sensor", "mask", "top", "logit"]
    assert out.iloc[:2000][cols].equals(out2.iloc[:2000][cols])
    j = ds.sensors.index
    mask = out["mask"].to_numpy()
    assert (mask[300 + 40:420] >> j("Leit") & 1).all()          # stuck after the stuck limit
    assert (mask[800:860] >> j("Fm") & 1).all()                 # dropout held while low
    assert mask[1001] >> j("Redox") & 1                         # spike seen one minute later
    assert not out["has_data"].iloc[2606] and out["has_data"].iloc[3001]
    assert out["alarm"].sum() > 20 and out["sensor"].sum() > 50


def test_adaptive_threshold_moves_both_ways_in_float32():
    n = 3000
    lp = np.full(n, -5.0, dtype=np.float32)
    lp[100:110] = 5.0                                        # an alarm that is false
    has = np.ones(n, dtype=bool)
    alarm, theta, asks = R.adaptive_alarm(lp, has, 1.0, np.zeros(n, dtype=bool))
    assert alarm[100:110].all() and not alarm[:100].any() and theta.dtype == np.float32
    assert theta[200] > theta[99] and theta[2999] < theta[200] and asks[0][3] == 1
    _, theta2, asks2 = R.adaptive_alarm(lp, has, 1.0, np.r_[np.zeros(100), np.ones(10), np.zeros(n - 110)].astype(bool))
    assert theta2[200] < theta[200] and asks2[0][3] == 0     # a true alarm does not raise it


def test_shipped_header_matches_shipped_spec(tmp_path):
    assert E._num(3) == "3.0f" and E._num(0.5) == "0.5f"
    spec = E.load_spec("gecco2018")
    p = tmp_path / "edge_model.h"
    E.write_header(spec, p)
    shipped = (E.FIRMWARE_DIR / "edge_model.h").read_text(encoding="utf-8")
    assert p.read_text(encoding="utf-8") == shipped
    assert "#define EDGE_K 9" in shipped and "EDGE_FLASH" in shipped
    assert spec["info"]["params"] == 625 and spec["info"]["int8_bytes"] < spec["info"]["float32_bytes"] / 3
    sizes = json.loads((E.FIRMWARE_DIR / "uno_build_info.json").read_text())["gecco2018"]
    assert sizes["flash_bytes"] < 32256 and sizes["ram_static_bytes"] < 2048


def test_int8_close_to_float():
    ds, f = stream()
    spec = tiny_spec(ds.sensors)
    C = R.to_counts(f, spec)
    a = R.run_counts(C, spec, int8=False, adaptive=False)
    b = R.run_counts(C, spec, int8=True, adaptive=False)
    assert (a["alarm"] == b["alarm"]).mean() > 0.97


def _desktop(tmp_path, spec):
    header = tmp_path / "edge_model_test.h"
    E.write_header(spec, header)
    return H.build_desktop(tmp_path / "build", header)


@no_cxx
@pytest.mark.parametrize("int8,adaptive", [(True, True), (False, False)])
def test_cpp_is_bit_identical_to_reference(tmp_path, int8, adaptive):
    ds, f = stream()
    spec = tiny_spec(ds.sensors)
    C = R.to_counts(f, spec)
    y = f[EVENT_COL].to_numpy().astype(bool)
    link = H.ProcessLink([str(_desktop(tmp_path, spec))], "desktop")
    try:
        dev, info = H.replay(link, C, y, int8=int8, adaptive=adaptive)
    finally:
        link.close()
    ref = R.run_counts(C, spec, int8, y, adaptive)
    c = H.compare(dev, ref)
    for k in ("alarm_agreement", "water_agreement", "sensor_agreement", "mask_agreement", "top_agreement",
              "logit_bit_identical", "theta_bit_identical"):
        assert c[k] == 1.0, (k, c)
    assert info["error"] is None and info["dataset"] == "synthetic" and info["int8"] == int8


def _sim_available():
    try:
        return H.arduino_core() is not None and H.build_sim is not None and os.name != "nt" and \
            __import__("shutil").which("avr-gcc") is not None and os.path.exists("/usr/include/simavr/sim_avr.h")
    except Exception:
        return False


@pytest.mark.skipif(not _sim_available(), reason="needs avr-gcc, the Arduino AVR core and simavr")
def test_real_uno_firmware_on_simulated_uno_is_bit_identical(tmp_path):
    ds, f = stream()
    spec = tiny_spec(ds.sensors)
    header = tmp_path / "edge_model_test.h"
    E.write_header(spec, header)
    elf, sizes = H.build_uno_elf(tmp_path / "uno", header)
    assert sizes["flash_bytes"] < 32256 and sizes["ram_static_bytes"] < 2048
    C = R.to_counts(f, spec)[:400]
    y = f[EVENT_COL].to_numpy().astype(bool)[:400]
    link = H.ProcessLink([str(H.build_sim(tmp_path / "sim")), str(elf)], "sim")
    try:
        dev, info = H.replay(link, C, y)
        hw = link.ask("H", "HW,", timeout=10)
        s = link.ask("S", "S,", timeout=10)
    finally:
        link.close()
    c = H.compare(dev, R.run_counts(C, spec, True, y, True))
    assert c["logit_bit_identical"] == 1.0 and c["alarm_agreement"] == 1.0 and c["mask_agreement"] == 1.0
    assert info["free_ram_bytes"] > 300 and info["mean_us"] > 1000          # a real 16 MHz AVR
    assert "dht=1" in hw and "probe=1" in hw and s.split(",")[4] == "27"     # simulated sensors


FAKE = r"""
import sys
n = 0
print("READY,ewsv2-edge,synthetic,9,10,1,1,500,700", flush=True)
for line in sys.stdin:
    if line.startswith("C,"):
        print("READY,ewsv2-edge,synthetic,9,10,1,1,500,700", flush=True)
    elif line.startswith("D,"):
        n += 1
        if n > 20:
            sys.exit(3)                       # the board is "unplugged"
        print(f"R,{n-1},c0000000,3f800000,0,0,0,0,0,10,0,0,0", flush=True)
"""


def test_replay_stops_cleanly_when_the_device_disappears():
    link = H.ProcessLink([sys.executable, "-c", FAKE], "fake")
    C = np.zeros((50, 9))
    try:
        dev, info = H.replay(link, C, np.zeros(50, dtype=bool))
    finally:
        link.close()
    assert len(dev) == 20 and info["minutes_done"] == 20 and "stopped after 20 of 50" in info["error"]


def test_board_detection_without_a_board(monkeypatch):
    from ewsv2.edge import board
    monkeypatch.setattr(H, "candidate_ports", lambda: [])
    res = board.detect("auto")
    assert res["board"] is False and "no Arduino" in res["message"]
    assert "laptop runs the same logic" in res["choice"]["board"] and "simulated" in res["choice"]["probe"]


def test_simulated_source_and_data_fallback(tmp_path):
    a = build("source", "simulated", days=6, seed=3).load()
    b = build("source", "simulated", days=6, seed=3).load()
    assert a.frame.equals(b.frame) and a.frame[EVENT_COL].sum() > 0 and set(a.periods) == {"train", "test"}
    cfg = {"data": {"source": "gecco2018", "params": {"data_dir": str(tmp_path)}, "fallback": "simulated",
                    "fallback_params": {"days": 5}}}
    ds = _load_dataset(cfg)
    assert ds.name == "simulated" and len(ds.frame) == 5 * 1440
    with pytest.raises(FileNotFoundError):
        _load_dataset({"data": {"source": "gecco2018", "params": {"data_dir": str(tmp_path)}}})


def test_cli_has_edge_and_board():
    p = build_parser()
    assert p.parse_args(["edge", "--config", "s8_edge"]).command == "edge"
    assert p.parse_args(["board"]).config == "s8_edge"


@no_cxx
@pytest.mark.skipif(sys.platform.startswith("win"), reason="pseudo-terminals are POSIX only")
def test_serial_link_over_a_pseudo_terminal(tmp_path):
    """The pyserial path, with the PC build standing in for the board."""
    pytest.importorskip("serial")
    import subprocess
    import tty

    ds, f = stream()
    spec = tiny_spec(ds.sensors)
    exe = _desktop(tmp_path, spec)
    master, slave = os.openpty()
    tty.setraw(slave)
    proc = subprocess.Popen([str(exe)], stdin=master, stdout=master)
    try:
        link = H.SerialLink(os.ttyname(slave), 115200, reset=False)
        C = R.to_counts(f, spec)[:300]
        dev, info = H.replay(link, C, np.zeros(300, dtype=bool))
        link.close()
    finally:
        proc.kill()
    ref = R.run_counts(C, spec, True, np.zeros(300, dtype=bool))
    assert H.compare(dev, ref)["logit_bit_identical"] == 1.0 and info["minutes"] == 300
