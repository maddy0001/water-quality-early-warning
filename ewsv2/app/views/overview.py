"""Overview: what the system does, the headline results, and what is connected right now."""

from __future__ import annotations

import streamlit as st

from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    snap = S.snapshot()
    e = snap["edge"]
    ui.hero("Water quality early warning – is it the water or the sensor?",
            "Version 2 of the ICESC 2026 system: tested honestly on real drinking-water data (GECCO 2018/2019), "
            "it separates water events from broken sensors, keeps a promised false-alarm rate, explains every "
            "alarm, writes fact-checked reports with a local LLM, and runs on a ₹350 Arduino Uno.")
    ui.kpi_row([
        ("Leakage exposed", f"{snap['leakage']['random_split_event_f1']:.2f} → {snap['leakage']['honest_event_f1']:.2f}",
         "event F1: random split vs honest"),
        ("Faults kept out of water alarms", "7 → 1", "GECCO 2018, same detector, with the triage"),
        ("False alarms / day", "0.97–1.01", "adaptive threshold for a 1/day budget"),
        ("LLM reports fact-correct", "59 / 60", "qwen2.5:3b on a laptop CPU; checked automatically"),
        ("Arduino Uno = laptop", f"{e['uno_minutes']:,} min", "bit-identical answers, 25 ms per minute"),
    ])
    _start_here()
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("How an alarm is decided")
        ui.flow([("1 · Sensors", "9 analysers, every minute"), ("2 · Detect", "small network + 3 paper-1 modules"),
                 ("3 · Budget", "adaptive threshold, ≤ 1 false alarm/day"),
                 ("4 · Triage", "water event or broken sensor?"), ("5 · Explain", "which sensor moved, how much"),
                 ("6 · Report", "template or local LLM, fact-checked")])
        st.write("")
        st.markdown(
            "- **Control room** – replay a real station minute by minute, with injected contaminants and sensor faults; "
            "answer the alarms yourself and watch the threshold adapt. If the Uno is plugged in it computes every "
            "minute too, and the dashboard shows that its answers match.\n"
            "- **Live sensor lab** – the probe in a glass of water: add salt for a water event, pull the probe for a "
            "sensor fault, touch the pad to say *false alarm*. Without the board the same lab runs on a simulated probe.\n"
            "- **Results**, **Incident reports**, **Version 1** – the evidence, and how paper 1's parts behaved on real data.")
    with right:
        st.subheader("Right now")
        data = S.data_status()
        ok_llm, why_llm = _llm_status()
        board = _board_status()
        rows = [
            ("GECCO 2018 data", data["gecco2018"], "simulated station is used instead"),
            ("GECCO 2019 data", data["gecco2019"], "simulated station is used instead"),
            ("Arduino Uno", board[0], "the laptop runs the same logic"),
            ("Local LLM (Ollama)", ok_llm, "template reports"),
            ("Version 1 dashboard", S.url_alive(S.config()["dashboard"]["v1_url"]), "published results shown instead"),
        ]
        html = ""
        for name, ok, fallback in rows:
            html += (f'<div class="logrow">{ui.pill("ready" if ok else "fallback", "ok" if ok else "warn")} '
                     f'<b>{ui.esc(name)}</b>' + ("" if ok else f' <span class="muted">– {ui.esc(fallback)}</span>') + "</div>")
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Details and fixes: *System check*.")


def _start_here() -> None:
    """Three big entry points, so a first-time viewer knows where to click."""
    from ewsv2.app import nav
    st.write("")
    st.markdown("#### Start here")
    items = [
        ("control-room", "1 · Control room", "A real station replayed. Press ⏭ Next alarm, then answer: real or false?"),
        ("live-lab", "2 · Live sensor lab", "The probe in a glass of water (or simulated): salt → water alarm, "
                                            "probe out → sensor fault."),
        ("results", "3 · Results", "The numbers for the report: v1 vs v2, honest testing, the Uno, the LLM."),
    ]
    cols = st.columns(len(items))
    for c, (path, title, text) in zip(cols, items):
        with c.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
            page = nav.PAGES.get(path)
            if page is not None:
                st.page_link(page, label="Open", icon=":material/arrow_forward:")
    ui.explain("Words used on every page: <b>water alarm</b> (red) = the water itself changed · <b>sensor alert</b> "
               "(yellow) = a sensor looks broken, maintenance only · <b>alarm line</b> = how high the alarm score must "
               "go before an alarm; it rises when the operator says “false alarm” · <b>answer key</b> = what really "
               "happened in the data; shown to you, never to the system.")


def _llm_status():
    key = "_llm_status"
    if key not in st.session_state:
        cfg = S.config()
        st.session_state[key] = S.ollama_status(cfg["llm"]["models"][0], cfg["llm"]["host"])
    return st.session_state[key]


def _board_status():
    from ewsv2.app.views.common import board
    b = board()
    return (b is not None and b.connected, b.message if b is not None else "")
