"""Incident report writers (Segment 7, contribution C8).

``template``  fixed sentences filled from the alarm record. Always faithful; the baseline.
``ollama``    a free open-weight model running locally with Ollama (http://localhost:11434).
              It gets the public alarm record and the retrieved guidance snippets, with
              instructions to use nothing else. Temperature 0 and a fixed seed.

Safety rule (from the project plan): the reporter reads the alarm, it never writes it.
The decision, level, times and sensors come from the record; if the LLM is off, slow or
wrong, the template report is used instead (``guarded`` output, see ``run.py``).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from ewsv2.core.registry import register
from ewsv2.inject.signatures import ROLES
from ewsv2.llm.records import public

GUIDANCE_FILE = Path(__file__).with_name("guidance.yaml")


def load_guidance(path: Path | None = None) -> list[dict]:
    return yaml.safe_load(Path(path or GUIDANCE_FILE).read_text(encoding="utf-8"))


def retrieve(record: dict, guidance: list[dict], roles: dict | None = None) -> list[dict]:
    """Snippets for the roles of the record's sensors and for its kind (water / sensor)."""
    roles = roles or ROLES
    rs = {roles.get(s["code"]) for s in record.get("sensors", [])}
    out = []
    for g in guidance:
        if set(g.get("roles", [])) & rs or record["kind"] in g.get("kinds", []):
            out.append(g)
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:+g}" if isinstance(v, float) and v != 0 else f"{v:g}")


@register("reporter", "template")
class TemplateReporter:
    name = "template"

    def report(self, record: dict, snippets: list[dict]) -> dict:
        t0 = time.perf_counter()
        r = record
        lines = [f"{r['decision']} - level {r['level']}",
                 f"{r['station']}, {r['start']} to {r['end']} ({r['duration_min']} min)."]
        parts = []
        for s in r["sensors"]:
            txt = f"{s['code']} ({s['description']}) from {s['before']:g} to {s['now']:g}" \
                if s.get("before") is not None and s.get("now") is not None else f"{s['code']} ({s['description']})"
            if "normal_units" in s:
                txt += f", {s['normal_units']:g} times its largest normal {r.get('move_window_min', 15)}-minute move"
            parts.append(txt)
        ids = [g["id"] for g in snippets]
        if r["kind"] == "water":
            lines.append("What happened: " + "; ".join(parts) + ".")
            lines.append("Likely cause: a change in the water reaching the station"
                         + (", seen on several sensors at once." if (r.get("sensors_beyond_normal") or 0) >= 2
                            else ", seen mainly on one sensor."))
            act = [i for i in ids if i.startswith("P")]
            lines.append("Check first: take a grab sample and compare it with the analysers"
                         + (f" [{act[0]}]." if act else "."))
        else:
            lines.append("What happened: the checks flagged " + ", ".join(s["code"] for s in r["sensors"])
                         + ("; " + "; ".join(parts) if parts else "") + ".")
            lines.append("Likely cause: the sensor, not the water; no other sensor moved beyond its normal range.")
            act = [i for i in ids if i.startswith("P")]
            lines.append("Check first: inspect the named analyser on site" + (f" [{act[0]}]." if act else "."))
        who = [g for g in snippets if g["id"].startswith("W")]
        if who:
            lines.append("Guidance: " + " ".join(f"[{g['id']}] {g['text']}" for g in who[:2]))
        return {"text": "\n".join(lines), "seconds": time.perf_counter() - t0, "ok": True}


SYSTEM_PROMPT = """You write short incident reports for operators in a drinking-water control room.
Rules:
1. Use ONLY facts from the ALARM RECORD and the GUIDANCE below. Do not invent numbers, times,
   sensors, causes or actions. Copy numbers exactly as written; do not compute new ones.
2. The first line must be exactly: {decision} - level {level}
3. Then three short sections, each starting on a new line with its label:
   What happened: ...
   Likely cause: ...
   Check first: ...
4. When you use guidance, cite it with its id in square brackets, e.g. [W1].
5. At most 120 words. Plain text, no markdown."""


@register("reporter", "ollama")
class OllamaReporter:
    """Local LLM through Ollama's HTTP API (no Python package needed)."""

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                 temperature: float = 0.0, seed: int = 42, max_tokens: int = 300, timeout: int = 300, **_):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.seed = float(temperature), int(seed)
        self.max_tokens, self.timeout = int(max_tokens), int(timeout)
        self.name = f"ollama:{model}"

    def available(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            return False, f"Ollama is not running at {self.host} ({exc})"
        models = [m.get("name", "") for m in tags.get("models", [])]
        if not any(m == self.model or m.split(":")[0] == self.model for m in models):
            return False, f"model {self.model!r} not pulled (have: {', '.join(models) or 'none'}); run: ollama pull {self.model}"
        return True, "ok"

    def messages(self, record: dict, snippets: list[dict]) -> list[dict]:
        guidance = "\n".join(f"[{g['id']}] {g['text']} (source: {g['source']})" for g in snippets) or "(none)"
        user = ("ALARM RECORD (JSON):\n" + json.dumps(public(record), indent=1, ensure_ascii=False)
                + "\n\nGUIDANCE:\n" + guidance + "\n\nWrite the report.")
        return [{"role": "system", "content": SYSTEM_PROMPT.format(decision=record["decision"], level=record["level"])},
                {"role": "user", "content": user}]

    def report(self, record: dict, snippets: list[dict]) -> dict:
        body = json.dumps({"model": self.model, "messages": self.messages(record, snippets), "stream": False,
                           "options": {"temperature": self.temperature, "seed": self.seed,
                                       "num_predict": self.max_tokens}}).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            text = out.get("message", {}).get("content", "").strip()
            return {"text": text, "seconds": time.perf_counter() - t0, "ok": bool(text)}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"text": "", "seconds": time.perf_counter() - t0, "ok": False, "error": str(exc)}


class FunctionReporter:
    """Wraps any ``f(messages) -> text`` (tests use it to stand in for an LLM)."""

    def __init__(self, fn, name: str = "function"):
        self.fn, self.name = fn, name
        self._ollama = OllamaReporter()

    def report(self, record: dict, snippets: list[dict]) -> dict:
        t0 = time.perf_counter()
        text = self.fn(self._ollama.messages(record, snippets))
        return {"text": text, "seconds": time.perf_counter() - t0, "ok": bool(text)}
