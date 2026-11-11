"""Shared, cached objects for the pages: the board connection and the station data."""

from __future__ import annotations

import os

import streamlit as st

from ewsv2.app import services as S


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


@st.cache_resource(show_spinner="Loading the station (real data + injected contaminants and faults) …")
def base(source: str) -> dict:
    return S.load_base(source, S.config())
