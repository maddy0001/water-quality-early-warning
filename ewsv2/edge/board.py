"""Find the Arduino and say which parts answer (Segment 8). Every part is optional.

    python -m ewsv2 board                     (port found automatically)
    python -m ewsv2 board --set edge.port=COM7 --set board.readings=10

For each part the system then uses the hardware path when it is detected and the computer path
otherwise – the rule from the plan: detect, then choose; never wait for a missing part.
"""

from __future__ import annotations

import time

from ewsv2.edge import hil as H

FALLBACK = {
    "board": ("Uno runs the edge logic", "the laptop runs the same logic (Python reference, identical answers)"),
    "probe": ("live water readings from the probe", "replayed GECCO data or the simulated station"),
    "touch": ("touch sensor = operator's 'false alarm' button", "keyboard / dashboard button"),
    "dht": ("station temperature and humidity", "left out"),
    "rfid": ("operator taps a tag", "not used (needs soldered pins; optional)"),
}


def detect(port: str | None = "auto", baud: int = 115200, readings: int = 3) -> dict:
    """{'board': bool, 'port': str, 'message': str, 'parts': {name: bool}, 'readings': [dict], 'choice': {...}}"""
    link, msg = H.open_board(port, baud)
    out = {"board": link is not None, "message": msg, "parts": {k: False for k in ("probe", "touch", "dht", "rfid")},
           "readings": []}
    if link is not None:
        out["port"] = link.port
        try:
            hw = link.ask("H", "HW,", timeout=6.0)
            for kv in hw.split(",")[1:]:
                k, v = kv.split("=")
                out["parts"][k] = v == "1"
            for _ in range(readings):
                f = link.ask("S", "S,", timeout=6.0).split(",")[1:]
                out["readings"].append({"probe": int(f[0]), "probe_off": int(f[1]), "dht_status": int(f[2]),
                                        "temperature_c": int(f[3]), "humidity_pct": int(f[4]),
                                        "touches": int(f[5]), "touch_now": f[6] == "1"})
                time.sleep(1.2)                           # the DHT11 needs 1 s between reads
            hw = link.ask("H", "HW,", timeout=6.0)       # touches seen during the readings count too
            for kv in hw.split(",")[1:]:
                k, v = kv.split("=")
                out["parts"][k] = out["parts"][k] or v == "1"
        except H.DeviceError as exc:
            out["message"] += f"; stopped answering: {exc}"
        finally:
            link.close()
    have = {"board": out["board"], **out["parts"]}
    out["choice"] = {k: (FALLBACK[k][0] if have.get(k) else FALLBACK[k][1]) for k in FALLBACK}
    return out
