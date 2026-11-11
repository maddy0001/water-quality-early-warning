"""ewsv2 dashboard – entry point (Segment 9).

    python -m ewsv2 dashboard                               (recommended: sets the theme and the port)
    python -m streamlit run ewsv2/app/main.py --server.port 8502
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                     # so "streamlit run ewsv2/app/main.py" finds the package
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ewsv2.app import ui  # noqa: E402
from ewsv2.app.views import control_room, live_lab, overview, reports, results, system, version1  # noqa: E402

st.set_page_config(page_title="Water EWS v2", page_icon="💧", layout="wide", initial_sidebar_state="expanded")
ui.setup_page()

pages = {
    "Demo": [
        st.Page(overview.render, title="Overview", icon=":material/water_drop:", url_path="overview", default=True),
        st.Page(control_room.render, title="Control room", icon=":material/monitor_heart:", url_path="control-room"),
        st.Page(live_lab.render, title="Live sensor lab", icon=":material/sensors:", url_path="live-lab"),
    ],
    "Evidence": [
        st.Page(results.render, title="Results", icon=":material/insights:", url_path="results"),
        st.Page(reports.render, title="Incident reports", icon=":material/description:", url_path="reports"),
        st.Page(version1.render, title="Version 1 (paper 1)", icon=":material/history_edu:", url_path="version-1"),
    ],
    "Station": [
        st.Page(system.render, title="System check", icon=":material/health_and_safety:", url_path="system"),
    ],
}
from ewsv2.app import nav as NAV  # noqa: E402

NAV.PAGES.update({p.url_path or "overview": p for group in pages.values() for p in group})
nav = st.navigation(pages)
with st.sidebar:
    st.toggle("Show explanations", value=st.session_state.get("explain", True), key="explain",
              help="Plain-language help on every page. Switch off for a cleaner screen during a demo.")
    st.caption("💧 **Water EWS v2** – is it the water or the sensor? Every part is optional: a missing board, "
               "sensor, LLM or data file is replaced by the computer.")
nav.run()
