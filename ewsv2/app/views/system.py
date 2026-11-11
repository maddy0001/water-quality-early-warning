"""System check: what is connected, what falls back to the computer, and how to fix it."""

from __future__ import annotations

import platform
import sys

import streamlit as st

from ewsv2 import SEGMENT, __version__
from ewsv2.app import services as S
from ewsv2.app import ui
from ewsv2.app.views.common import board


def _row(name: str, ok: bool, now: str, fallback: str, fix: str = "") -> str:
    p = ui.pill("ready", "ok") if ok else ui.pill("fallback", "warn")
    extra = "" if ok else f'<br><span class="muted">Instead: {ui.esc(fallback)}' + (f" · Fix: {ui.esc(fix)}" if fix else "") + "</span>"
    return f'<div class="logrow">{p} <b>{ui.esc(name)}</b> – {ui.esc(now)}{extra}</div>'


def render() -> None:
    ui.hero("What is connected", "Every part is optional. This page shows which ones are ready on this computer, "
            "what the system uses instead of the others, and how to fix them.", eyebrow="Setup", color="slate")
    ui.guide('How to read this page', [
        '<b>ready</b> = this part works on this computer now. <b>fallback</b> = it is missing, and the line says what the computer uses instead – the demo still works.',
        'Plug the Uno in or start Ollama, then press “Look for the board again” or “Check the local LLM again”.',
    ])
    cfg = S.config()
    c1, c2 = st.columns([1, 1])
    if c1.button("🔌 Look for the board again", width="stretch"):
        b = board(connect=False)
        if b is not None:
            b._next_try = 0.0
            b.connect()
    if c2.button("🤖 Check the local LLM again", width="stretch"):
        st.session_state.pop("_llm_status", None)
    data = S.data_status()
    b = board()
    llm_ok, llm_why = st.session_state.get("_llm_status") or S.ollama_status(cfg["llm"]["models"][0], cfg["llm"]["host"])
    st.session_state["_llm_status"] = (llm_ok, llm_why)
    outs = S.outputs_status()
    parts = b.parts if (b is not None and b.connected) else {}
    html = "".join([
        _row("GECCO 2018 data", data["gecco2018"], "files found" if data["gecco2018"] else "no files",
             "the simulated station", "python -m ewsv2 fetch gecco2018"),
        _row("GECCO 2019 data", data["gecco2019"], "files found" if data["gecco2019"] else "no files",
             "the simulated station", "python -m ewsv2 fetch gecco2019"),
        _row("Arduino Uno", b is not None and b.connected, (b.message if b is not None else "switched off"),
             "the laptop runs the same logic (identical answers)",
             "plug it in, upload ewsv2_edge.ino, close the Serial Monitor"),
        _row("Probe (water)", parts.get("probe", False), "answers" if parts.get("probe") else "not detected",
             "the simulated probe", "wires AO→A0, VCC→A2, GND→GND"),
        _row("Touch sensor", parts.get("touch", False), "answers" if parts.get("touch") else "not detected",
             "the 'False alarm' button on screen", "wires SIG→D3, VCC→D4, GND→GND"),
        _row("DHT11", parts.get("dht", False), "answers" if parts.get("dht") else "not detected",
             "left out", "wires DATA→A1, VCC→5V, GND→GND"),
        _row("RFID reader (not used)", parts.get("rfid", False), "not part of this project", "nothing – it is not needed",
             "optional: needs soldered pins"),
        _row("AI writer (Ollama)", llm_ok, "ready" if llm_ok else
             ("Ollama is not running" if "refused" in str(llm_why).lower() else llm_why), "template reports",
             "start Ollama; ollama pull " + cfg["llm"]["models"][0]),
        _row("Version 1 dashboard", S.url_alive(cfg["dashboard"]["v1_url"]), cfg["dashboard"]["v1_url"],
             "published results on the Version 1 page", "python -m ewsv2 dashboard starts it if venv exists"),
    ])
    ui.card("Parts", html)
    seg = "".join(_row(f"{k} results", v, "on this computer" if v else "not run here", "bundled snapshot of the measured results",
                       "run that segment's command") for k, v in outs.items())
    ui.card("Experiment outputs", seg)
    ui.card("Software", f'ewsv2 {ui.esc(__version__)} (built up to {ui.esc(SEGMENT)}) · Python {ui.esc(sys.version.split()[0])} · '
            f'{ui.esc(platform.system())} {ui.esc(platform.release())}')
