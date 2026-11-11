"""Automatic fact check of an incident report against its alarm record (Segment 7, C8).

Five kinds of error, each found mechanically:

``decision``  the first line does not state the record's decision and level
``time``      a clock time or date that is not the alarm's start or end
``number``    a number that is not in the record or in the cited guidance (and is not a small
              count from 1 to 10); a number counts as correct if it is the record's number
              rounded to the decimals written
``sensor``    a sensor of the station named that is not part of this alarm
``citation``  a guidance id in square brackets that was not provided

A report is *faithful* when it has no error. The checker is itself tested on reports with
planted errors (``tests/test_llm.py``).
"""

from __future__ import annotations

import re

import numpy as np

from ewsv2.data.gecco import SENSOR_INFO
from ewsv2.inject.signatures import ROLES

ROLE_WORDS = {
    "chlorine": ["chlorine dioxide", "chlorine", "disinfectant residual"],
    "orp": ["redox", "orp", "oxidation-reduction"],
    "ph": ["ph"],
    "conductivity": ["conductivity", "leitfähigkeit"],
    "turbidity": ["turbidity", "trübung"],
    "organics": ["spectral absorption", "organic", "sac"],
    "temperature": ["temperature"],
    "flow": ["flow rate", "flowmeter", "flow meter"],
}

_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_CITE = re.compile(r"\[([A-Za-z]\d+)\]")
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")


def _record_numbers(record: dict) -> list[float]:
    nums = [record.get("duration_min"), record.get("sensors_beyond_normal"), record.get("move_window_min")]
    for s in record.get("sensors", []):
        for k in ("before", "now", "change", "normal_units"):
            v = s.get(k)
            if v is not None:
                nums += [v, abs(v)]
    nums += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(record.get("station", "")))]
    return [float(x) for x in nums if x is not None]


def _snippet_numbers(snippets: list[dict]) -> list[float]:
    out = []
    for g in snippets:
        out += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", g["text"])]
    return out


def _matches(token: str, allowed: list[float]) -> bool:
    try:
        x = float(token)
    except ValueError:
        return True
    decimals = len(token.split(".")[1]) if "." in token else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(x) - abs(a)) <= tol for a in allowed)


def _sensor_mentions(text: str, station_sensors: list[str], roles: dict) -> set[str]:
    """Sensors of the station named in ``text`` (by code or by what they measure)."""
    low = text.lower()
    found = set()
    for s in station_sensors:
        if re.search(rf"(?<![\w]){re.escape(s.lower())}(?![\w])", low):
            found.add(s)
    for s in station_sensors:
        for w in ROLE_WORDS.get(roles.get(s, ""), []):
            if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", low):
                found.add(s)
    return found


def check(text: str, record: dict, snippets: list[dict], station_sensors: list[str],
          roles: dict | None = None) -> list[dict]:
    roles = roles or ROLES
    errors = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0].lower() if lines else ""
    if record["decision"].lower() not in first or record["level"].lower() not in first:
        errors.append({"type": "decision", "found": lines[0] if lines else ""})

    # times and dates
    times = {record["start"][11:16], record["end"][11:16]}
    dates = {record["start"][:10], record["end"][:10]}
    for m in _TIME.finditer(text):
        hhmm = f"{int(m.group(1)):02d}:{m.group(2)}"
        if hhmm not in times:
            errors.append({"type": "time", "found": m.group(0)})
    for m in _DATE.finditer(text):
        if m.group(0) not in dates:
            errors.append({"type": "time", "found": m.group(0)})

    # citations
    given = {g["id"] for g in snippets}
    for m in _CITE.finditer(text):
        if m.group(1) not in given:
            errors.append({"type": "citation", "found": m.group(0)})

    # numbers: remove what is not a free number (times, dates, citations, sensor codes like Cl_2)
    scrub = _DATE.sub(" ", _TIME.sub(" ", _CITE.sub(" ", text)))
    for s in sorted(station_sensors, key=len, reverse=True):
        scrub = re.sub(rf"(?<![\w]){re.escape(s)}(?![\w])", " ", scrub)
    allowed = _record_numbers(record) + _snippet_numbers([g for g in snippets])
    for m in _NUM.finditer(scrub):
        tok = m.group(0).lstrip("+")
        val = abs(float(tok))
        if val.is_integer() and 1 <= val <= 10 and "." not in tok:
            continue                                   # small counts ("two sensors", "3 checks")
        if not _matches(tok, allowed):
            errors.append({"type": "number", "found": m.group(0)})

    # sensors: anything of the station named outside the guidance quotes must be in the record
    in_record = {s["code"] for s in record.get("sensors", [])} | set(record.get("sensor_alerts_during", []))
    record_roles = {roles.get(s) for s in in_record}
    body = text
    for g in snippets:                                 # quoted guidance may name other parameters
        body = body.replace(g["text"], " ")
    for s in _sensor_mentions(body, station_sensors, roles):
        if s not in in_record and roles.get(s) not in record_roles:
            errors.append({"type": "sensor", "found": s})
    return errors


def mentions_top_sensor(text: str, record: dict, station_sensors: list[str], roles: dict | None = None) -> bool:
    if not record.get("sensors"):
        return True
    top = record["sensors"][0]["code"]
    return top in _sensor_mentions(text, station_sensors, roles or ROLES)


ERROR_TYPES = ("decision", "time", "number", "sensor", "citation")


def summarise(errors_per_report: list[list[dict]]) -> dict:
    n = len(errors_per_report)
    out = {"reports": n, "faithful_share": float(np.mean([len(e) == 0 for e in errors_per_report])) if n else np.nan,
           "errors_per_report": float(np.mean([len(e) for e in errors_per_report])) if n else np.nan}
    for t in ERROR_TYPES:
        out[f"with_{t}_error"] = float(np.mean([any(x["type"] == t for x in e) for e in errors_per_report])) if n else np.nan
    return out


__all__ = ["check", "summarise", "mentions_top_sensor", "ERROR_TYPES", "SENSOR_INFO"]
