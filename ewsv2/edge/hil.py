"""Hardware-in-the-loop: the laptop feeds a recorded stream to the edge code, minute by minute.

Targets, all speaking the same line protocol (``firmware/ewsv2_edge/edge_protocol.h``):

* ``uno``     – the real Arduino Uno over USB (pyserial); the port is found automatically;
* ``sim``     – the real Uno firmware running on a simulated ATmega328P (simavr, Linux only):
                cycle-exact timing without a board;
* ``desktop`` – the same C++ compiled for this computer (g++/clang++);
* ``python``  – the reference itself, always available (``reference.py``).

The laptop also plays the operator: when the device says an alarm episode has ended, it answers
with the number of false alarms in it (from the labels), as the reference does.
Every answer is compared with the reference. Nothing here is required: a target that is not
available is skipped with one line saying why.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ewsv2.edge import reference as R
from ewsv2.edge.export import FIRMWARE_DIR

log = logging.getLogger(__name__)

EDGE_ROOT = Path(__file__).parent
DESKTOP_SRC = FIRMWARE_DIR.parent / "desktop" / "desktop_main.cpp"
SIM_SRC = EDGE_ROOT / "tools" / "uno_sim.c"
OUT_COLS = ["t", "logit", "theta", "alarm", "water", "sensor", "mask", "top", "us", "ask", "ep_start", "last_on"]
BOARD_IDS = {0x1A86, 0x2341, 0x2A03, 0x0403, 0x10C4}      # CH340, Arduino, Arduino.org, FTDI, CP210x


class DeviceError(RuntimeError):
    """The device stopped answering or answered nonsense."""


# ---------------------------------------------------------------------------- links
class Link:
    name = "link"

    def send(self, line: str) -> None:
        raise NotImplementedError

    def recv(self, timeout: float = 5.0) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def ask(self, line: str, prefix: str, timeout: float = 5.0) -> str:
        """Send a line and return the first answer starting with ``prefix`` (other lines are skipped)."""
        self.send(line)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            got = self.recv(max(0.05, end - time.monotonic()))
            if got.startswith(prefix):
                return got
            if got:
                log.debug("device: %s", got)
        raise DeviceError(f"no '{prefix}' answer to '{line[:30]}' within {timeout:.0f} s")


class SerialLink(Link):
    name = "uno"

    def __init__(self, port: str, baud: int = 115200, reset: bool = True):
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise DeviceError("pyserial is missing:  pip install pyserial") from exc
        self.port = port
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.1
        self.ser.dtr = False             # open without resetting ...
        self.ser.rts = False
        self.ser.open()
        if reset:                        # ... then one clean reset (Uno: DTR; boards wired to RTS: RTS)
            try:
                self.ser.dtr = True
                self.ser.rts = True
                time.sleep(0.1)
                self.ser.dtr = False
                self.ser.rts = False
            except Exception:  # pragma: no cover - some drivers have no modem lines
                pass
        self._buf = b""

    def send(self, line: str) -> None:
        try:
            self.ser.write((line + "\n").encode("ascii"))
        except Exception as exc:  # unplugged
            raise DeviceError(f"cannot write to {self.port}: {exc}") from exc

    def recv(self, timeout: float = 5.0) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                return raw.decode("ascii", "replace").strip()
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except Exception as exc:  # unplugged
                raise DeviceError(f"cannot read from {self.port}: {exc}") from exc
            if chunk:
                self._buf += chunk
        return ""

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:  # pragma: no cover
            pass


class ProcessLink(Link):
    def __init__(self, cmd: list[str], name: str):
        self.name = name
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for got in self.proc.stdout:
            self.lines.put(got.strip())
        self.lines.put(None)

    def send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DeviceError(f"{self.name} stopped: {exc}") from exc

    def recv(self, timeout: float = 5.0) -> str:
        try:
            got = self.lines.get(timeout=timeout)
        except queue.Empty:
            return ""
        if got is None:
            self.lines.put(None)
            raise DeviceError(f"{self.name} stopped (exit code {self.proc.poll()})")
        return got

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            self.proc.kill()


# ---------------------------------------------------------------------------- builds
def compiler() -> str | None:
    for c in ("g++", "clang++"):
        if shutil.which(c):
            return c
    return None


def _header_flag(header: Path | None) -> list[str]:
    return [f'-DEDGE_MODEL_FILE="{Path(header).resolve().as_posix()}"'] if header else []


def build_desktop(out_dir: Path, header: Path | None = None) -> Path:
    """The firmware logic compiled for this computer."""
    cxx = compiler()
    if cxx is None:
        raise DeviceError("no C++ compiler (g++ or clang++) on this computer")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / ("ewsv2_edge_desktop" + (f"_{header.stem}" if header else "") + (".exe" if os.name == "nt" else ""))
    cmd = [cxx, "-O2", "-std=c++17", "-ffp-contract=off", *_header_flag(header), "-o", str(exe), str(DESKTOP_SRC)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise DeviceError(f"compiling the desktop build failed:\n{res.stderr[-1500:]}")
    return exe


def arduino_core() -> Path | None:
    """The Arduino AVR core (cores/arduino + variants/standard), wherever it is installed."""
    cands = []
    if os.environ.get("EWSV2_ARDUINO_CORE"):
        cands.append(Path(os.environ["EWSV2_ARDUINO_CORE"]))
    for base in (Path.home() / ".arduino15", Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "Arduino15"):
        cands += sorted((base / "packages" / "arduino" / "hardware" / "avr").glob("*"), reverse=True)
    cands.append(Path("/tmp/ArduinoCore-avr"))
    for c in cands:
        if (c / "cores" / "arduino" / "Arduino.h").exists() and (c / "variants" / "standard").exists():
            return c
    return None


def build_uno_elf(out_dir: Path, header: Path | None = None) -> tuple[Path, dict]:
    """Compile the Uno sketch with avr-gcc (as Arduino IDE does). Returns the ELF and its sizes."""
    core = arduino_core()
    gcc, gxx = shutil.which("avr-gcc"), shutil.which("avr-g++")
    if core is None or gcc is None or gxx is None:
        raise DeviceError("avr-gcc and the Arduino AVR core are needed to build the Uno firmware here")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = out_dir / "obj"
    obj.mkdir(exist_ok=True)
    cdir, vdir, sdir = core / "cores" / "arduino", core / "variants" / "standard", FIRMWARE_DIR
    flags = ["-mmcu=atmega328p", "-DF_CPU=16000000L", "-DARDUINO=10819", "-DARDUINO_AVR_UNO", "-DARDUINO_ARCH_AVR",
             "-Os", "-ffunction-sections", "-fdata-sections", f"-I{cdir}", f"-I{vdir}", f"-I{sdir}"]
    cxxf = ["-std=gnu++11", "-fno-exceptions", "-fno-threadsafe-statics"]
    jobs = []
    for f in cdir.glob("*.c"):
        jobs.append([gcc, *flags, "-c", str(f), "-o", str(obj / (f.name + ".o"))])
    for f in cdir.glob("*.S"):
        jobs.append([gcc, *flags, "-x", "assembler-with-cpp", "-c", str(f), "-o", str(obj / (f.name + ".o"))])
    for f in cdir.glob("*.cpp"):
        jobs.append([gxx, *flags, *cxxf, "-c", str(f), "-o", str(obj / (f.name + ".o"))])
    sketch = out_dir / "sketch.cpp"
    sketch.write_text(f'#include <Arduino.h>\n#include "{(sdir / "ewsv2_edge.ino").as_posix()}"\n', encoding="utf-8")
    jobs.append([gxx, *flags, *cxxf, *_header_flag(header), "-c", str(sketch), "-o", str(obj / "sketch.o")])
    for cmd in jobs:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise DeviceError(f"avr build failed: {' '.join(cmd[-3:])}\n{res.stderr[-1500:]}")
    elf = out_dir / ("ewsv2_edge" + (f"_{header.stem}" if header else "") + ".elf")
    res = subprocess.run([gcc, "-mmcu=atmega328p", "-Os", "-Wl,--gc-sections", *map(str, obj.glob("*.o")), "-o",
                          str(elf), "-lm"], capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise DeviceError(f"avr link failed:\n{res.stderr[-1500:]}")
    sizes = {}
    size_tool = shutil.which("avr-size")
    if size_tool:
        out = subprocess.run([size_tool, "-A", str(elf)], capture_output=True, text=True, check=False).stdout
        sec = {ln.split()[0]: int(ln.split()[1]) for ln in out.splitlines() if ln.startswith(".")}
        sizes = {"flash_bytes": sec.get(".text", 0) + sec.get(".data", 0),
                 "ram_static_bytes": sec.get(".data", 0) + sec.get(".bss", 0)}
    return elf, sizes


def build_sim(out_dir: Path) -> Path:
    """The simulated Uno (needs simavr's library: apt install libsimavr-dev libelf-dev)."""
    cc = shutil.which("gcc")
    if cc is None or os.name == "nt":
        raise DeviceError("the simulated Uno needs Linux with gcc and simavr")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "uno_sim"
    res = subprocess.run([cc, "-O2", "-o", str(exe), str(SIM_SRC), "-lsimavr", "-lelf"], capture_output=True,
                         text=True, check=False)
    if res.returncode != 0:
        raise DeviceError(f"simavr is not installed (apt install libsimavr-dev libelf-dev): {res.stderr[-300:]}")
    return exe


# ---------------------------------------------------------------------------- finding the board
def candidate_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    out = []
    for p in list_ports.comports():
        desc = f"{p.description} {p.manufacturer or ''}".lower()
        if (p.vid in BOARD_IDS) or any(w in desc for w in ("ch340", "arduino", "usb-serial", "usb serial", "cp210")):
            out.append(p.device)
    return out


def open_board(port: str | None = "auto", baud: int = 115200, reset: bool = True) -> tuple[SerialLink | None, str]:
    """Find the Uno running the ewsv2 sketch. Returns (link, message); link is None if none answers."""
    try:
        import serial  # noqa: F401
    except ImportError:
        return None, "pyserial is missing (pip install pyserial), so the board cannot be used"
    ports = candidate_ports() if port in (None, "", "auto") else [port]
    if not ports:
        return None, "no Arduino found on any USB port"
    notes = []
    for p in ports:
        try:
            link = SerialLink(p, baud, reset)
        except Exception as exc:
            notes.append(f"{p}: cannot open ({str(exc).splitlines()[0]}; is the Serial Monitor open?)")
            continue
        first = ""
        end = time.monotonic() + 4.0
        while time.monotonic() < end:
            got = link.recv(0.3)
            if got:
                first = first or got
            if got.startswith("READY,ewsv2-edge"):
                return link, f"board on {p}"
        try:
            got = link.ask("X", "READY", timeout=2.5)
            if got.startswith("READY,ewsv2-edge"):
                return link, f"board on {p}"
        except DeviceError:
            pass
        link.close()
        other = " (it runs another program, e.g. the hardware check: upload ewsv2_edge)" if first else ""
        notes.append(f"{p}: no answer from the ewsv2_edge sketch{other}")
    return None, "; ".join(notes)


# ---------------------------------------------------------------------------- replay
def handshake(link: Link, int8: bool, adaptive: bool, wait: float = 3.0) -> dict:
    end = time.monotonic() + wait          # the start-up READY (boards restart when the port opens)
    while time.monotonic() < end:
        if link.recv(0.3).startswith("READY"):
            break
    while link.recv(0.05):                  # anything else still waiting is old
        pass
    f = link.ask(f"C,{int(int8)},{int(adaptive)}", "READY", timeout=5.0).split(",")
    return {"firmware": f[1], "dataset": f[2], "sensors": int(f[3]), "params": int(f[4]), "int8": f[5] == "1",
            "adaptive": f[6] == "1", "free_ram_bytes": int(f[7]), "state_bytes": int(f[8])}


def _f32(h: str) -> float:
    return struct.unpack(">f", bytes.fromhex(h))[0]


def _fmt(c: float) -> str:
    return "" if not np.isfinite(c) else str(int(c))


def replay(link: Link, C: np.ndarray, y_water: np.ndarray, int8: bool = True, adaptive: bool = True,
           progress_every: int = 0) -> tuple[pd.DataFrame, dict]:
    """Feed every minute of counts to the device. On failure, returns what was done and the reason."""
    info = handshake(link, int8, adaptive)
    y = np.asarray(y_water, dtype=bool)
    n = len(C)
    rows, alarm = [], np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    info["error"] = None
    try:
        for t in range(n):
            got = link.ask("D," + ",".join(_fmt(v) for v in C[t]), "R,", timeout=5.0).split(",")[1:]
            r = [int(got[0]), _f32(got[1]), _f32(got[2])] + [int(v) for v in got[3:]]
            rows.append(r)
            alarm[t] = bool(r[3])
            if r[9]:                                            # episode ended: the operator's verdict
                nf = R.false_count(alarm[r[10]:r[11] + 1], y[r[10]:r[11] + 1])
                link.ask(f"V,{nf}", "OK,V", timeout=5.0)
            if progress_every and (t + 1) % progress_every == 0:
                el = time.perf_counter() - t0
                log.info("  %s: %d/%d minutes (%.0f s, about %.0f s left)", link.name, t + 1, n, el,
                         el / (t + 1) * (n - t - 1))
        inf = link.ask("I", "INFO", timeout=5.0)
        for kv in inf.split(",")[1:]:
            k, v = kv.split("=")
            info[k] = int(v)
    except DeviceError as exc:
        info["error"] = f"stopped after {len(rows)} of {n} minutes: {exc}"
        log.warning("%s %s", link.name, info["error"])
    wall = time.perf_counter() - t0
    info["minutes_done"] = len(rows)
    info["wall_seconds"] = wall
    info["round_trip_ms"] = 1000 * wall / max(len(rows), 1)
    if info.get("sum_us") is not None and len(rows):
        info["mean_us"] = info["sum_us"] / len(rows)
    return pd.DataFrame(rows, columns=OUT_COLS), info


def compare(device: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Minute-by-minute agreement (over the minutes the device answered)."""
    n = len(device)
    ref = ref.iloc[:n].reset_index(drop=True)
    d = {"minutes": n}
    if n == 0:
        return d
    for c in ("alarm", "water", "sensor", "mask", "top"):
        a, b = device[c].to_numpy().astype(int), ref[c].to_numpy().astype(int)
        d[f"{c}_agreement"] = float((a == b).mean())
    for c in ("alarm", "water", "sensor"):
        d[f"{c}_minutes_device"] = int(device[c].sum())
        d[f"{c}_minutes_reference"] = int(ref[c].sum())
    dl = device["logit"].to_numpy(dtype=np.float32)
    rl = ref["logit"].to_numpy(dtype=np.float32)
    d["logit_bit_identical"] = float((dl.view(np.uint32) == rl.view(np.uint32)).mean())
    d["max_logit_diff"] = float(np.max(np.abs(dl.astype(float) - rl.astype(float))))
    dt, rt = device["theta"].to_numpy(dtype=np.float32), ref["theta"].to_numpy(dtype=np.float32)
    both = np.isfinite(dt) & np.isfinite(rt)
    d["theta_bit_identical"] = float(((dt.view(np.uint32) == rt.view(np.uint32)) | (~np.isfinite(dt) & ~np.isfinite(rt))).mean())
    d["max_theta_diff"] = float(np.max(np.abs(dt[both].astype(float) - rt[both].astype(float)))) if both.any() else 0.0
    return d
