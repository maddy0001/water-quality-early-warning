"""Keeps the Arduino connection for the dashboard: finds the board, reads its sensors every ~2 s in the
background, sends it GECCO minutes for the live hardware check, and reconnects when it comes back.
Never blocks the dashboard; if there is no board, it simply says so (Segment 9).
"""

from __future__ import annotations

import threading
import time
from collections import deque

from ewsv2.edge import hil as H
from ewsv2.live.station import Reading


class BoardService:
    def __init__(self, port: str | None = "auto", baud: int = 115200, period: float = 2.0):
        self.port, self.baud, self.period = port, baud, period
        self.lock = threading.Lock()
        self.link = None
        self.message = "not started"
        self.parts = {"probe": False, "touch": False, "dht": False, "rfid": False}
        self.readings: deque = deque(maxlen=2000)
        self.polling = False
        self._thread = None
        self._next_try = 0.0

    # ---------------------------------------------------------------- connection
    @property
    def connected(self) -> bool:
        return self.link is not None

    def connect(self) -> bool:
        with self.lock:
            if self.link is not None:
                return True
            if time.monotonic() < self._next_try:
                return False
            link, msg = H.open_board(self.port, self.baud)
            self.message = msg
            if link is None:
                self._next_try = time.monotonic() + 10.0         # try again later, never in a tight loop
                return False
            self.link = link
            try:
                hw = link.ask("H", "HW,", timeout=6.0)
                for kv in hw.split(",")[1:]:
                    k, v = kv.split("=")
                    self.parts[k] = v == "1"
            except H.DeviceError as exc:
                self._drop(f"board stopped answering: {exc}")
                return False
            return True

    def _drop(self, why: str) -> None:
        try:
            if self.link is not None:
                self.link.close()
        finally:
            self.link = None
            self.message = why
            self._next_try = time.monotonic() + 5.0

    def close(self) -> None:
        self.polling = False
        with self.lock:
            self._drop("closed")

    # ---------------------------------------------------------------- sensors
    def read_once(self) -> Reading | None:
        if not self.connect():
            return None
        with self.lock:
            try:
                f = self.link.ask("S", "S,", timeout=6.0).split(",")[1:]
            except H.DeviceError as exc:
                self._drop(f"board stopped answering: {exc}")
                return None
        on, off, dht, t, h, touches = int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])
        r = Reading(time.time(), on if on - off >= 50 else None,
                    float(t) if dht == 0 else None, float(h) if dht == 0 else None, touches)
        self.parts["probe"] = r.probe is not None
        self.parts["dht"] = dht == 0
        self.readings.append(r)
        return r

    def start_polling(self) -> None:
        if self.polling:
            return
        self.polling = True

        def loop():
            while self.polling:
                t0 = time.monotonic()
                self.read_once()
                time.sleep(max(0.2, self.period - (time.monotonic() - t0)))

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_polling(self) -> None:
        self.polling = False

    # ---------------------------------------------------------------- edge minutes (live HIL)
    def configure_edge(self, int8: bool = True, adaptive: bool = True) -> bool:
        if not self.connect():
            return False
        with self.lock:
            try:
                self.link.ask(f"C,{int(int8)},{int(adaptive)}", "READY", timeout=5.0)
                return True
            except H.DeviceError as exc:
                self._drop(f"board stopped answering: {exc}")
                return False

    def edge_minute(self, counts) -> list | None:
        """Send one minute of counts; returns the parsed R line or None (board gone)."""
        if self.link is None:
            return None
        with self.lock:
            try:
                got = self.link.ask("D," + ",".join(H._fmt(v) for v in counts), "R,", timeout=5.0).split(",")[1:]
                return [int(got[0]), H._f32(got[1]), H._f32(got[2])] + [int(v) for v in got[3:]]
            except (H.DeviceError, Exception) as exc:  # noqa: BLE001 - any failure: fall back to the laptop
                self._drop(f"board stopped answering: {exc}")
                return None

    def edge_verdict(self, n_false: int) -> None:
        if self.link is None:
            return
        with self.lock:
            try:
                self.link.ask(f"V,{n_false}", "OK,V", timeout=5.0)
            except H.DeviceError as exc:
                self._drop(f"board stopped answering: {exc}")
