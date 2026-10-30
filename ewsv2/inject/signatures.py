"""Contaminant signatures: how a contaminant slug changes conventional sensors (Segment 4).

Source of the *pattern* (which sensors move, and in which direction): US EPA pipe-loop
and pipe-flow injection studies at the EPA Test & Evaluation facility –
Hall et al. (2007), J. AWWA 99(1):66-77, and the EPA report 600/R-09/076 (Hall, Szabo,
Panguluri, Meiners, 2009). Their main findings, used here:

* chlorine residual and organic carbon (TOC, here the SAC254 proxy) respond to most
  contaminants; ORP follows chlorine; conductivity responds to salts and wastewater;
  turbidity to wastewater and biological material; pH to acids and bases;
* temperature and flow do not respond.

The *size* of each change is set in units of the sensor's normal variability (see
``features/scale.py``) and swept over three doses. The EPA tests ran on a very stable
background, so their absolute changes do not transfer to a different utility; a dose
sweep in normal-variability units says instead "how big must a change be before the
system notices", which is what an operator needs.

A signature is a list of role -> relative response (lead response = ±1). Roles are
mapped to the station's sensors with :data:`ROLES`; roles the station does not measure
are dropped, and a signature that leaves fewer than two responding sensors is not
observable at that station and is skipped (reported as such).
"""

from __future__ import annotations

import numpy as np

# Sensor name -> physical role. Extend this for a new station (or pass ``roles=`` in the config).
ROLES: dict[str, str] = {
    # GECCO 2018
    "Cl": "chlorine", "Cl_2": "chlorine", "Redox": "orp", "pH": "ph", "Leit": "conductivity",
    "Trueb": "turbidity", "Tp": "temperature", "Fm": "flow", "Fm_2": "flow",
    # GECCO 2019
    "Cond": "conductivity", "Turb": "turbidity", "SAC": "organics", "PFM": "flow",
    # ESP32 node (S8)
    "TDS": "conductivity", "Temp": "temperature",
}

# Roles that a sensor fault may hit (flows are process, not quality, sensors).
QUALITY_ROLES = ("chlorine", "orp", "ph", "conductivity", "turbidity", "organics", "temperature")

# Readings that cannot go below zero.
NON_NEGATIVE = ("chlorine", "conductivity", "turbidity", "organics")

CONTAMINANTS: dict[str, dict] = {
    "pesticide": {
        "examples": "aldicarb, glyphosate, malathion (organic, chlorine-demanding)",
        "response": {"chlorine": -1.0, "orp": -0.6, "organics": +0.8, "conductivity": +0.1, "ph": -0.1},
    },
    "biological": {
        "examples": "E. coli in growth broth",
        "response": {"chlorine": -1.0, "orp": -0.6, "organics": +1.0, "turbidity": +0.5, "conductivity": +0.2},
    },
    "inorganic": {
        "examples": "sodium arsenite, potassium ferricyanide, metal salts",
        "response": {"conductivity": +1.0, "chlorine": -0.5, "orp": -0.4},
    },
    "wastewater": {
        "examples": "sewage or back-flow intrusion",
        "response": {"turbidity": +1.0, "conductivity": +0.8, "chlorine": -0.8, "orp": -0.5,
                     "organics": +0.8, "ph": -0.2},
    },
    "alkaline": {
        "examples": "caustic over-dose, alkaline chemicals",
        "response": {"ph": +1.0, "orp": -0.5, "conductivity": +0.4},
    },
}

# Lead-sensor change in normal-variability units.
DOSES: dict[str, float] = {"low": 4.0, "medium": 8.0, "high": 15.0}

MIN_RESPONDING = 2       # sensors with |response| >= RESPONDING_LEVEL needed to be observable
RESPONDING_LEVEL = 0.2


def role_of(sensor: str, roles: dict[str, str] | None = None) -> str | None:
    return (roles or ROLES).get(sensor)


def station_response(kind: str, sensors: list[str], roles: dict[str, str] | None = None) -> dict[str, float]:
    """Relative response of each of the station's sensors (largest |value| = 1).

    Empty dict if the contaminant is not observable with these sensors.
    """
    pattern = CONTAMINANTS[kind]["response"]
    resp = {s: pattern[r] for s in sensors if (r := role_of(s, roles)) in pattern}
    if not resp:
        return {}
    top = max(abs(v) for v in resp.values())
    resp = {s: v / top for s, v in resp.items()}
    if sum(abs(v) >= RESPONDING_LEVEL for v in resp.values()) < MIN_RESPONDING:
        return {}
    return resp


def slug_shape(rise: int, plateau: int, fall: int) -> np.ndarray:
    """Passage of a contaminant slug past a sensor: smooth rise, plateau, smooth fall (0..1)."""
    up = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, rise + 1)[1:])
    down = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fall + 1)[1:])
    return np.concatenate([up, np.ones(plateau), down])
