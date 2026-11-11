"""Start here: the project in one sentence, where to click, the headline results and what is connected."""

from __future__ import annotations

import streamlit as st

from ewsv2.app import nav
from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    snap = S.snapshot()
    e = snap["edge"]
    ui.hero("Is it the water, or the sensor?",
            "Sensors in a drinking-water network report every minute. When a reading suddenly changes there are two "
            "very different reasons: the water really changed (warn people!) or the sensor broke (send a technician). "
            "This system tells the two apart, learns from the operator, runs on a ₹350 Arduino and writes the "
            "incident report.", eyebrow="Water quality early warning · version 2")

    st.markdown("### What would you like to do?")
    tiles = [
        ("how-it-works", "💡", "How it works", "The idea in 5 simple steps, what each sensor does, the AI report "
                                              "writer, and what is new in version 2."),
        ("control-room", "🎬", "Watch a real station", "Replay real data from a water utility. The system raises "
                                                      "alarms – you decide whether each one was real."),
        ("live-lab", "🧪", "Try the sensors", "A probe in a glass of water. Salt = pollution, probe out = broken "
                                             "sensor, touch pad = “false alarm”."),
        ("results", "📊", "See the results", "The proof in numbers: version 1 vs version 2, the Arduino test, the "
                                            "AI reports."),
    ]
    for c, (path, emoji, title, text) in zip(st.columns(4), tiles):
        with c:
            ui.page_tile(nav.PAGES.get(path), emoji, title, text)

    st.markdown("### Version 2 in 30 seconds")
    lk = snap["leakage"]
    tri = snap["triage"]["2018"]
    ui.kpi_row([
        ("Honest testing", f"{lk['random_split_event_f1']:.2f} → {lk['honest_event_f1']:.2f}",
         "Score with the easy test vs the honest test (train on the past, test on the future)."),
        ("Broken sensors → false water alarms", f"{tri['detector_only']['faults_to_water']} → "
                                                 f"{tri['rules']['faults_to_water']}",
         "Faulty sensors are now sent to maintenance instead of alarming the city."),
        ("False alarms per day", "≈ 1", "Promised and kept: 0.90–1.18 a day on both real test periods."),
        ("Arduino = laptop", f"{e['uno_minutes']:,} min", "One week of real data: the ₹350 Uno gave exactly the same "
                                                          "answer every minute."),
        ("AI reports with no mistake", "59 / 60", "Every number is checked; a report with a mistake is never shown."),
    ])
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### From version 1 to version 2")
        ui.compare(
            "Version 1 (paper 1, ICESC 2026)",
            ["Three separate models on three different datasets",
             "Tested on shuffled minutes – an easy test",
             "Every strange reading = “bad water”",
             "Fixed alarm threshold",
             "Explanations nobody checked",
             "Runs on a laptop only"],
            "Version 2 (this project)",
            ["One pipeline on real utility data, minute by minute",
             "Honest test: learn from the past, test on the future",
             "<b>Water or sensor?</b> Broken sensors go to maintenance",
             "Alarm line adapts – about 1 false alarm a day, and the operator teaches it",
             "Explanations tested; AI reports fact-checked",
             "Same decisions on a ₹350 Arduino Uno"])
    with right:
        st.markdown("### Right now on this computer")
        data = S.data_status()
        ok_llm, _ = _llm_status()
        b_ok, _ = _board_status()
        ui.chips([
            ("Real station data", data["gecco2018"] or data["gecco2019"],
             "ready" if (data["gecco2018"] or data["gecco2019"]) else "using a simulated station"),
            ("Arduino Uno", b_ok, "connected" if b_ok else "the laptop does its job"),
            ("AI writer (Ollama)", ok_llm, "ready" if ok_llm else "simple template reports"),
            ("Version 1 dashboard", S.url_alive(S.config()["dashboard"]["v1_url"]), "on port 8501"
             if S.url_alive(S.config()["dashboard"]["v1_url"]) else "not running – results shown instead"),
        ])
        ui.explain("🟢 = ready. 🟡 = not here, and the computer uses the replacement written next to it. "
                   "<b>Nothing is required</b> – every page works with or without the board, the sensors, the AI "
                   "and the data files.")
        ui.page_tile(nav.PAGES.get("system"), "🔌", "What is connected", "Every part, what replaces it when it is "
                     "missing, and how to fix it.", label="Check")


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
