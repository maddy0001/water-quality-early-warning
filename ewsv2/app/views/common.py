"""Shared, cached objects for the pages: the board connection, the station data and the sidebar status."""

from __future__ import annotations

import os

import streamlit as st

from ewsv2.app import services as S
from ewsv2.app import ui


@st.cache_resource(show_spinner=False)
def _board_service():
    from ewsv2.live.board_service import BoardService
    cfg = S.config()
    e = cfg.get("edge") or {}
    return BoardService(e.get("port", "auto"), int(e.get("baud", 115200)))


def board(connect: bool = True):
    """The board service, or None when hardware is switched off (EWSV2_NO_BOARD=1, e.g. in tests)."""
    if os.environ.get("EWSV2_NO_BOARD") == "1":
        return None
    b = _board_service()
    if connect and not b.connected:
        b.connect()                     # throttled inside: at most one attempt every 10 s
    return b


@st.cache_resource(show_spinner="Loading the station data (first time only, about 10 seconds) …")
def base(source: str) -> dict:
    return S.load_base(source, S.config())


def llm_status() -> tuple[bool, str]:
    key = "_llm_status"
    if key not in st.session_state:
        cfg = S.config()
        st.session_state[key] = S.ollama_status(cfg["llm"]["models"][0], cfg["llm"]["host"])
    return st.session_state[key]


def sources() -> list[str]:
    cfg = S.config()
    data = S.data_status()
    return [s for s in cfg["dashboard"]["sources"] if s == "simulated" or data.get(s)] or ["simulated"]


def sidebar_status() -> None:
    data = S.data_status()
    b = board(connect=False)
    ok_llm, _ = llm_status()
    real = data.get("gecco2018") or data.get("gecco2019")
    st.markdown('<div style="font-size:.7rem;font-weight:800;letter-spacing:.12em;color:#818cf8">CONNECTIONS</div>',
                unsafe_allow_html=True)
    ui.side_status([
        ("Utility data", bool(real), "loaded" if real else "synthetic only"),
        ("Arduino", b is not None and b.connected, "connected" if (b is not None and b.connected) else "laptop mode"),
        ("AI report writer", ok_llm, "ready" if ok_llm else "template mode"),
    ])
