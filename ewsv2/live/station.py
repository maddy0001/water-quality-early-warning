"""The live demo station: the probe in a glass of water (+ DHT11, touch), or a simulated one (Segment 9).

Readings arrive every ~2 seconds. The probe board reads 0..1023: high = dry, lower = more ions in
the water. We show its complement, the **ion index** = 1023 − reading (tap water ≈ 450–570, salt
water higher). It is not a calibrated conductivity – it is a demo of the decision logic.

Same ideas as the edge algorithm, on one live sensor:
* **calibration**: the first readings in water set the normal noise ("unit");
* **move**: z = (reading − mean of the previous 15 readings) / unit;
* **water alarm**: |z| ≥ limit on 2 readings in a row (a slow drift is normal, a step is an event);
* **sensor checks**: probe out of water / wire off (jumps to dry), probe board not answering,
  DHT11 not answering – each a sensor alert, never a water alarm;
* **adaptive limit**: the operator's "false alarm" (touch sensor or button) raises the limit by 25%;
  it slowly returns towards the starting limit (S5 in miniature).
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field

DRY = 980                 # probe reading above this: out of water / wire off


@dataclass
class Reading:
    t: float                           # seconds (time.time())
    probe: int | None                  # 0..1023, None = probe board not answering
    temperature: float | None = None
    humidity: float | None = None
    touches: int = 0


@dataclass
class LiveEvent:
    kind: str                          # "water" | "sensor" | "verdict" | "info"
    start: float
    end: float | None
    text: str
    level: str = ""
    peak_z: float = 0.0
    before: float | None = None
    now: float | None = None


@dataclass
class LiveStation:
    limit0: float = 8.0
    calib_n: int = 20
    window: int = 15
    min_unit: float = 3.0
    limit: float = 8.0
    history: deque = field(default_factory=lambda: deque(maxlen=1800))
    events: list = field(default_factory=list)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.limit = self.limit0
        self.calib: list[float] = []
        self.unit: float | None = None
        self.recent: deque = deque(maxlen=self.window)
        self.over = 0
        self.alarm: LiveEvent | None = None
        self.fault: LiveEvent | None = None
        self.last_touches: int | None = None
        self.last_t: float | None = None
        self.status = "calibrating"

    # ------------------------------------------------------------------
    def _open(self, kind, t, text, level="", **kw) -> LiveEvent:
        e = LiveEvent(kind, t, None, text, level, **kw)
        self.events.append(e)
        return e

    def verdict_false_alarm(self, t: float | None = None) -> None:
        """The operator says the last water alarm was false: raise the limit by 25 %."""
        t = t or time.time()
        self.limit = min(30.0, self.limit * 1.25)
        self.events.append(LiveEvent("verdict", t, t, f"Operator: false alarm – limit raised to {self.limit:.1f} units"))

    def push(self, r: Reading) -> dict:
        t = r.t
        out = {"t": t, "probe": r.probe, "ion": None if r.probe is None else 1023 - r.probe, "z": 0.0,
               "water": False, "sensor": False, "limit": self.limit, "status": self.status,
               "temperature": r.temperature, "humidity": r.humidity}
        # touch sensor = operator's "false alarm" button
        if self.last_touches is not None and r.touches > self.last_touches:
            self.verdict_false_alarm(t)
        self.last_touches = r.touches
        # slow return of the limit: half-way back to the starting limit in 10 minutes
        if self.last_t is not None:
            self.limit = self.limit0 + (self.limit - self.limit0) * 0.5 ** (max(0.0, t - self.last_t) / 600.0)
        self.last_t = t

        problem = None
        if r.probe is None:
            problem = "probe board not answering (check AO, VCC, GND wires)"
        elif r.probe >= DRY and (self.unit is not None or self.calib):
            problem = "probe out of water or a fork wire is off"
        if problem:
            out["sensor"] = True
            out["status"] = "sensor alert"
            if self.fault is None:
                self.fault = self._open("sensor", t, f"SENSOR FAULT ALERT – {problem}", "MAINTENANCE")
            if self.alarm is not None:
                self.alarm.end = t
                self.alarm = None
            self.recent.clear()
            self.over = 0
            self.history.append(out)
            return out
        if self.fault is not None:
            self.fault.end = t
            self.fault = None
        if r.probe >= DRY:                       # dry, never calibrated: waiting for water
            self.status = out["status"] = "waiting for water"
            self.history.append(out)
            return out

        x = 1023.0 - r.probe
        if self.unit is None:                    # calibration in water
            self.calib.append(x)
            self.status = out["status"] = f"calibrating ({len(self.calib)}/{self.calib_n})"
            if len(self.calib) >= self.calib_n:
                d = [abs(b - a) for a, b in zip(self.calib[:-1], self.calib[1:])]
                d.sort()
                mad = d[len(d) // 2] if d else 0.0
                self.unit = max(self.min_unit, 1.4826 * mad)
                self.recent.extend(self.calib[-self.window:])
                self.status = "watching"
                self.events.append(LiveEvent("info", t, t, f"Calibrated: normal noise {self.unit:.1f} counts per reading"))
            self.history.append(out)
            return out

        mean = sum(self.recent) / len(self.recent) if self.recent else x
        z = (x - mean) / self.unit
        out["z"] = z
        out["ref"], out["band"] = mean, self.limit * self.unit      # for the chart: the "normal" band
        self.over = self.over + 1 if abs(z) >= self.limit else 0
        if self.over >= 2 and self.alarm is None:
            level = "CRITICAL" if abs(z) >= 3 * self.limit else ("MEDIUM" if abs(z) >= 2 * self.limit else "LOW")
            self.alarm = self._open("water", t, "WATER QUALITY ALARM – the ion index jumped "
                                    + ("up" if z > 0 else "down"), level, peak_z=z, before=mean, now=x)
        if self.alarm is not None:
            self.alarm.peak_z = max(self.alarm.peak_z, z, key=abs)
            self.alarm.now = x
            if abs(z) < self.limit / 2:
                self.alarm.end = t
                self.alarm = None
        out["water"] = self.alarm is not None
        self.status = out["status"] = "WATER ALARM" if out["water"] else "watching"
        self.recent.append(x)
        self.history.append(out)
        return out


class SimProbe:
    """A made-up probe, DHT11 and touch sensor for demos without the board."""

    def __init__(self, seed: int = 1):
        self.rng = random.Random(seed)
        self.level = 520.0                  # probe reading in tap water
        self.target = self.level
        self.in_water = True
        self.touches = 0
        self.temperature, self.humidity = 31.0, 86.0

    def add_salt(self) -> None:
        self.target = max(150.0, self.target - 280.0)

    def fresh_water(self) -> None:
        self.target = 520.0

    def pull_out(self) -> None:
        self.in_water = False

    def put_back(self) -> None:
        self.in_water = True

    def touch(self) -> None:
        self.touches += 1

    def read(self) -> Reading:
        self.level += (self.target - self.level) * 0.5 + self.rng.gauss(0.6, 1.5)   # quick step, slow drift up
        probe = int(round(self.rng.gauss(1009, 1))) if not self.in_water else int(round(self.level))
        self.humidity = min(99.0, max(40.0, self.humidity + self.rng.gauss(0, 0.3)))
        return Reading(time.time(), max(0, min(1023, probe)), round(self.temperature + self.rng.gauss(0, 0.2), 1),
                       round(self.humidity), self.touches)
