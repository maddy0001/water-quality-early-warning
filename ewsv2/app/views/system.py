"""System Status: which parts are available on this computer, what replaces the others, and how to fix them."""

from __future__ import annotations

import platform
import sys

import pandas as pd
import streamlit as st

from ewsv2 import __version__
from ewsv2.app import charts as C
from ewsv2.app import services as S
from ewsv2.app import ui
from ewsv2.app.views.common import board

TOPICS = [("Detection methods", "s1_baselines"), ("Version 1 components on real data", "s2_v1_port"),
          ("Forecasting", "s3_forecast"), ("Water event or sensor fault", "s4_triage"),
          ("False-alarm control", "s5_fusion"), ("Explanations", "s6_explain"), ("Incident reports", "s7_reports"),
          ("Edge device", "s8_edge")]


def render() -> None:
    ui.header("System Status",
              "Which parts are available on this computer right now, what the system uses instead of the missing "
              "ones, and how to connect them.", kicker="Reference · status", color="slate")
    cfg = S.config()
    with st.container(border=True):
        ui.panel_title("Actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Search for the Arduino again", width="stretch", icon=":material/usb:", key="sy_board"):
                b = board(connect=False)
                if b is not None:
                    b._next_try = 0.0
                    b.connect()
            ui.cap("Use after plugging in the board. Close the Arduino Serial Monitor first – it blocks the port.")
        with c2:
            if st.button("Check the AI model again", width="stretch", icon=":material/refresh:", key="sy_llm"):
                st.session_state.pop("_llm_status", None)
            ui.cap("Use after starting Ollama.")
    data = S.data_status()
    b = board()
    llm_ok, llm_why = st.session_state.get("_llm_status") or S.ollama_status(cfg["llm"]["models"][0], cfg["llm"]["host"])
    st.session_state["_llm_status"] = (llm_ok, llm_why)
    parts = b.parts if (b is not None and b.connected) else {}
    v1_ok = S.url_alive(cfg["dashboard"]["v1_url"])
    rows = [
        ("Utility data A (2018)", data["gecco2018"], "files found" if data["gecco2018"] else "files missing",
         "synthetic station", "python -m ewsv2 fetch gecco2018"),
        ("Utility data B (2019)", data["gecco2019"], "files found" if data["gecco2019"] else "files missing",
         "synthetic station", "python -m ewsv2 fetch gecco2019"),
        ("Arduino Uno", b is not None and b.connected, b.message if b is not None else "hardware access switched off",
         "laptop runs the same logic", "plug in, upload ewsv2_edge.ino, close the Serial Monitor"),
        ("Conductivity probe", parts.get("probe", False), "answering" if parts.get("probe") else "not detected",
         "simulation", "AO → A0, VCC → A2, GND → GND"),
        ("Touch sensor", parts.get("touch", False), "answering" if parts.get("touch") else "not detected",
         "on-screen Report false alarm button", "SIG → D3, VCC → D4, GND → GND"),
        ("Temperature/humidity sensor", parts.get("dht", False), "answering" if parts.get("dht") else "not detected",
         "values shown as unavailable", "DATA → A1, VCC → 5V, GND → GND"),
        ("AI report writer (Ollama)", llm_ok, "ready" if llm_ok else
         ("Ollama is not running" if "refused" in str(llm_why).lower() else str(llm_why)[:80]),
         "standard template reports", "start Ollama; ollama pull " + cfg["llm"]["models"][0]),
        ("Version 1 dashboard", v1_ok, cfg["dashboard"]["v1_url"], "published results on the Version 1 page",
         "started automatically when venv exists"),
    ]
    ready = sum(1 for r in rows if r[1])
    a, bcol = st.columns([1, 2.2])
    with a:
        C.plot(C.donut(["available", "replaced"], [ready, len(rows) - ready], [ui.OK, "#fbbf24"], "Components",
                       f"{ready}/{len(rows)}", height=300), key="sy_donut")
        ui.cap("Replaced components do not stop the system; the dashboard uses the alternative listed in the table.")
    with bcol:
        st.dataframe(pd.DataFrame([{"Component": n, "Status": "Available" if ok else "Replaced", "Details": d,
                                    "Used instead": "–" if ok else alt, "How to connect": "–" if ok else fix}
                                   for n, ok, d, alt, fix in rows]), hide_index=True, width="stretch", height=330)
    ui.section("Analysis results on this computer",
               "Result files produced by the experiment commands. Missing ones are replaced by the bundled results.")
    out = []
    for name, sub in TOPICS:
        d = S.OUTPUT_DIR / sub
        files = list(d.rglob("*.csv")) + list(d.rglob("*.json")) if d.exists() else []
        out.append({"Analysis": name, "Status": "Available" if files else "Not run on this computer",
                    "Files": len(files)})
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
    ui.cap(f"Software version {__version__} · Python {sys.version.split()[0]} · {platform.system()} "
           f"{platform.release()}")
