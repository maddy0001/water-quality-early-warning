"""Everything the dashboard pages share: status checks, the replay engine, reports, results (Segment 9).

Every optional part is checked, never assumed: no data files → simulated station; no board → the
laptop runs the edge logic; no Ollama → template reports; no outputs → bundled snapshot results.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ewsv2.core.config import load_config
from ewsv2.core.paths import OUTPUT_DIR
from ewsv2.core.registry import build
from ewsv2.core.types import CLASS_COL, EVENT_COL

APP_DIR = Path(__file__).parent
SNAPSHOT = APP_DIR / "snapshot.json"
SENSOR_NAMES = {
    "Tp": "temperature", "Cl": "chlorine dioxide 1", "pH": "pH", "Redox": "redox potential",
    "Leit": "conductivity", "Trueb": "turbidity", "Cl_2": "chlorine dioxide 2", "Fm": "flow 1", "Fm_2": "flow 2",
    "Cond": "conductivity", "Turb": "turbidity", "SAC": "organics (SAC)", "PFM": "flow",
}


SOURCE_LABELS = {
    "gecco2018": "Utility data A – 2018 test period (9 sensors)",
    "gecco2019": "Utility data B – 2019 test period (6 sensors)",
    "simulated": "Synthetic station (generated, no data files needed)",
}
SHORT_LABELS = {"gecco2018": "Utility data A (2018)", "gecco2019": "Utility data B (2019)",
                "simulated": "Synthetic station"}


def source_label(s: str, short: bool = False) -> str:
    return (SHORT_LABELS if short else SOURCE_LABELS).get(s, s)


def config(set_: dict | None = None) -> dict:
    return load_config("s9_dashboard", set_ or {})


def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ status
def data_status() -> dict:
    from ewsv2.data.gecco import DATA_DIR
    files = sorted(p.name for p in DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    return {"gecco2018": any("2018" in f for f in files), "gecco2019": any("2019" in f for f in files),
            "folder": str(DATA_DIR), "files": files}


def ollama_status(model: str = "qwen2.5:3b", host: str = "http://localhost:11434") -> tuple[bool, str]:
    try:
        rep = build("reporter", "ollama", model=model, host=host)
        return rep.available()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def url_alive(url: str, timeout: float = 0.6) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 500
    except Exception:  # noqa: BLE001
        return False


def outputs_status() -> dict:
    out = {}
    for seg, sub in (("S1", "s1_baselines"), ("S4", "s4_triage"), ("S5", "s5_fusion"), ("S6", "s6_explain"),
                     ("S7", "s7_reports"), ("S8", "s8_edge")):
        d = OUTPUT_DIR / sub
        out[seg] = d.exists() and any(d.rglob("report.md"))
    return out


# ------------------------------------------------------------------------------ replay engine
@dataclass
class ReplayState:
    """One station replayed minute by minute through the edge logic (laptop; optionally also the Uno)."""

    name: str
    frame: pd.DataFrame
    sensors: list[str]
    counts: np.ndarray
    cls: np.ndarray                    # 0 normal, 1 water event, 2 sensor fault (truth, hidden from the system)
    truth: np.ndarray                  # text per minute
    spec: dict
    start: int
    pos: int = 0                       # minutes done (relative to start)
    rows: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    pending: dict | None = None
    uno_checked: int = 0
    uno_same: int = 0
    uno_note: str = ""
    asks: int = 0                      # alarm cycles that ended (the system asked for a verdict)
    answers: list = field(default_factory=list)   # operator answers: {"start", "said", "key", "right"}

    WARMUP = 120                       # minutes fed before the shown period, so the history is warm

    def __post_init__(self):
        from ewsv2.edge.stream import EdgeStream
        self.engine = EdgeStream(self.spec)
        self.alarm = np.zeros(len(self.counts), dtype=bool)
        self.first = max(0, self.start - self.WARMUP)      # engine minute 0 = stream position `first`
        self.warm = False

    def warm_up(self, board=None) -> None:
        """Feed the minutes before the start silently (laptop, and the Uno when it runs too)."""
        if self.warm:
            return
        from ewsv2.edge import reference as R
        y = self.cls == 1
        for i in range(self.first, self.start):
            c = self.counts[i]
            m = self.engine.push([None if not np.isfinite(v) else int(v) for v in c])
            if board is not None and board.connected:
                board.edge_minute(c)
            self.alarm[i] = m.alarm
            if m.ask:
                n = R.false_count(self.alarm[self.first + m.ep_start:self.first + m.last_on + 1],
                                  y[self.first + m.ep_start:self.first + m.last_on + 1])
                self.engine.verdict(n)
                if board is not None and board.connected:
                    board.edge_verdict(n)
        self.warm = True

    @property
    def t(self) -> int:
        return self.start + self.pos

    @property
    def done(self) -> bool:
        return self.t >= len(self.counts)

    def step(self, verdict_mode: str = "auto", board=None) -> dict | None:
        """Process one minute. Returns the row, or None when waiting for the operator or finished."""
        if self.done or self.pending is not None:
            return None
        from ewsv2.edge import reference as R
        self.warm_up(board)
        i = self.t
        c = self.counts[i]
        raw = [None if not np.isfinite(v) else int(v) for v in c]
        m = self.engine.push(raw)
        uno = None
        if board is not None and board.connected:
            uno = board.edge_minute(c)
            if uno is None:
                self.uno_note = "the Uno stopped answering – the laptop continues alone"
            else:
                self.uno_checked += 1
                same = (np.float32(uno[1]).view(np.uint32) == np.float32(m.logit).view(np.uint32)
                        and bool(uno[3]) == m.alarm and bool(uno[4]) == m.water and int(uno[6]) == m.mask)
                self.uno_same += int(same)
        self.alarm[i] = m.alarm
        row = {"i": i, "time": self.frame.index[i], "logit": m.logit, "theta": m.theta, "alarm": m.alarm,
               "water": m.water, "sensor": m.sensor, "mask": m.mask, "top": m.top, "cls": int(self.cls[i]),
               "m": m.m.copy(), "z": m.z.copy(), "uno": uno is not None}
        self.rows.append(row)
        self._track(row)
        self.pos += 1
        if m.ask:
            y = self.cls == 1
            a, z = self.first + m.ep_start, self.first + m.last_on       # engine minutes -> stream positions
            auto = R.false_count(self.alarm[a:z + 1], y[a:z + 1])
            self.asks += 1
            if verdict_mode == "auto":
                self._verdict(auto, board, a, z)
            else:
                self.pending = {"ep_start": a, "last_on": z, "auto": auto, "truth": self.span_truth(a, z),
                                "key": "false alarm" if auto > 0 else "real event"}
        return row

    def span_truth(self, a: int, z: int) -> str:
        """What really happened during minutes a..z (the answer key – never seen by the system)."""
        seen = [str(t) for t in dict.fromkeys(self.truth[a:z + 1]) if str(t)]
        return "; ".join(seen) if seen else ""

    def answer(self, false_alarm: bool, board=None) -> None:
        if self.pending is None:
            return
        p = self.pending
        n = max(1, p["auto"]) if false_alarm else 0
        said = "false alarm" if false_alarm else "real event"
        before = float(self.engine.theta)
        self._verdict(n, board, p["ep_start"], p["last_on"])
        self.answers.append({"start": p["ep_start"], "said": said, "key": p["key"], "right": said == p["key"],
                             "lift": float(self.engine.theta) - before})
        self.pending = None

    def _verdict(self, n: int, board, a: int | None = None, z: int | None = None) -> None:
        self.engine.verdict(n)
        if board is not None and board.connected:
            board.edge_verdict(n)
        word = "false alarm" if n > 0 else "confirmed"
        hit = [e for e in self.episodes if e["kind"] == "water" and a is not None
               and e["start"] <= z + 5 and e["end"] >= a - 5]
        if not hit and self.episodes:                       # older callers: the latest water alarm
            hit = [e for e in reversed(self.episodes) if e["kind"] == "water"][:1]
        for e in hit:
            e["verdict"] = word

    def _track(self, row: dict) -> None:
        """Group minutes into alarm episodes (water / sensor), 5-minute merge gap."""
        i = row["i"]
        for kind in ("water", "sensor"):
            if not row[kind]:
                continue
            last = next((e for e in reversed(self.episodes) if e["kind"] == kind), None)
            if last is not None and i - last["end"] <= 5:
                last["end"] = i
                last["mask"] |= row["mask"]
                if kind == "water":
                    last["peak_m"] = np.maximum(last["peak_m"], row["m"])
                if not last["truth"] and self.truth[i]:
                    last["truth"] = self.truth[i]
            else:
                self.episodes.append({"kind": kind, "start": i, "end": i, "mask": row["mask"], "top": row["top"],
                                      "peak_m": row["m"].copy(), "truth": self.truth[i], "verdict": ""})


def load_replay(source: str, cfg: dict) -> ReplayState:
    """A fresh replay of a station (see ``load_base``)."""
    return ReplayState(**load_base(source, cfg))


def load_base(source: str, cfg: dict) -> dict:
    """GECCO-Triage stream (real data + injected contaminants and faults), or the simulated station."""
    from ewsv2.cli import _load_dataset
    from ewsv2.edge import reference as R
    from ewsv2.edge.run import get_spec
    from ewsv2.triage.run import build_benchmark

    c = json.loads(json.dumps(cfg))
    c["data"]["source"] = source
    ds = _load_dataset(c)
    inj, eps = build_benchmark(ds, c)
    f = inj.frame
    spec = get_spec(ds, c)
    cls = f[CLASS_COL].to_numpy().astype(int)
    truth = np.full(len(f), "", dtype=object)
    real = ds.frame[EVENT_COL].to_numpy().astype(bool)
    truth[real] = "real water event (labelled by the utility)"
    for _, r in eps.iterrows():
        a, b = int(r["start_pos"]), int(r["end_pos"])
        truth[a:b + 1] = (f"injected {r['type']} contaminant ({r['level']} dose)" if r["kind"] == "contaminant"
                          else f"injected {r['type']} fault on {r['lead_sensor']}")
    start = int(np.flatnonzero(f.index >= inj.periods["test"][0])[0])
    return {"name": ds.name, "frame": f, "sensors": list(inj.sensors), "counts": R.to_counts(f, spec), "cls": cls,
            "truth": truth, "spec": spec, "start": start}


# ------------------------------------------------------------------------------ reports
def episode_record(state: ReplayState, ep: dict, station: str) -> dict:
    """An S7 alarm record for an episode of the replay."""
    a, b = ep["start"], ep["end"]
    f = state.frame
    sens = []
    order = np.argsort(-ep["peak_m"])
    picks = [state.sensors[j] for j in order[:3] if ep["peak_m"][j] >= 0.5] if ep["kind"] == "water" else \
        [s for j, s in enumerate(state.sensors) if ep["mask"] >> j & 1][:3]
    picks = picks or [state.sensors[int(order[0])]]
    for s in picks:
        x = f[s].astype(float)
        before = x.iloc[max(0, a - 35):max(1, a - 5)].median()
        now = x.iloc[a]
        rec = {"code": s, "description": SENSOR_NAMES.get(s, s),
               "before": None if not np.isfinite(before) else round(float(before), 3),
               "now": None if not np.isfinite(now) else round(float(now), 3)}
        if rec["before"] is not None and rec["now"] is not None:
            rec["change"] = round(rec["now"] - rec["before"], 3)
        if ep["kind"] == "water":
            rec["normal_units"] = round(float(ep["peak_m"][state.sensors.index(s)]), 1)
        sens.append(rec)
    moved = int((ep["peak_m"] >= 1).sum())
    level = "MAINTENANCE" if ep["kind"] == "sensor" else ("CRITICAL" if moved >= 3 else ("MEDIUM" if moved == 2 else "LOW"))
    return {"id": f"E{a}", "kind": ep["kind"], "station": station,
            "decision": "WATER QUALITY ALARM" if ep["kind"] == "water" else "SENSOR FAULT ALERT", "level": level,
            "start": f.index[a].strftime("%Y-%m-%d %H:%M"), "end": f.index[b].strftime("%Y-%m-%d %H:%M"),
            "duration_min": int(b - a + 1), "move_window_min": 15,
            "sensors_beyond_normal": moved if ep["kind"] == "water" else None,
            "sensor_alerts_during": [s for j, s in enumerate(state.sensors) if ep["mask"] >> j & 1], "sensors": sens}


def write_report(record: dict, station_sensors: list[str], use_llm: bool, model: str, host: str) -> dict:
    """Template always; the local LLM if asked and available. The fact check decides what is shown."""
    from ewsv2.llm import faithfulness as FC
    from ewsv2.llm.reporters import load_guidance, retrieve

    snips = retrieve(record, load_guidance())
    tpl = build("reporter", "template").report(record, snips)
    out = {"template": tpl["text"], "llm": None, "errors": [], "shown": "template", "why": "", "seconds": 0.0}
    if not use_llm:
        out["why"] = "local LLM switched off"
        return out
    rep = build("reporter", "ollama", model=model, host=host, timeout=120)
    ok, why = rep.available()
    if not ok:
        out["why"] = why
        return out
    t0 = time.perf_counter()
    res = rep.report(record, snips)
    out["seconds"] = time.perf_counter() - t0
    if not res.get("ok"):
        out["why"] = res.get("error", "no answer")
        return out
    out["llm"] = res["text"]
    out["errors"] = FC.check(res["text"], record, snips, station_sensors)
    if out["errors"]:
        out["why"] = f"the LLM report had {len(out['errors'])} fact error(s), so the template is shown"
    else:
        out["shown"] = "llm"
    return out


# ------------------------------------------------------------------------------ results files
def read_csv(*parts: str) -> pd.DataFrame | None:
    p = OUTPUT_DIR.joinpath(*parts)
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def read_json(*parts: str):
    p = OUTPUT_DIR.joinpath(*parts)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None
