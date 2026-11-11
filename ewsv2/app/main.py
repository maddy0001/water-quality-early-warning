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
from ewsv2.app.views import (control_room, how_it_works, live_lab, overview, reports, results,  # noqa: E402
                             system, version1)

st.set_page_config(page_title="Water EWS v2", page_icon="💧", layout="wide", initial_sidebar_state="expanded")
ui.setup_page()

pages = {
    "Demo": [
        st.Page(overview.render, title="Start here", icon=":material/home:", url_path="overview", default=True),
        st.Page(how_it_works.render, title="How it works", icon=":material/lightbulb:", url_path="how-it-works"),
        st.Page(control_room.render, title="Watch a real station", icon=":material/play_circle:",
                url_path="control-room"),
        st.Page(live_lab.render, title="Try the sensors", icon=":material/science:", url_path="live-lab"),
    ],
    "Proof": [
        st.Page(results.render, title="Results", icon=":material/insights:", url_path="results"),
        st.Page(reports.render, title="AI incident reports", icon=":material/description:", url_path="reports"),
        st.Page(version1.render, title="Version 1 (paper 1)", icon=":material/history_edu:", url_path="version-1"),
    ],
    "Setup": [
        st.Page(system.render, title="What is connected", icon=":material/settings_input_component:",
                url_path="system"),
    ],
}
from ewsv2.app import nav as NAV  # noqa: E402

NAV.PAGES.update({p.url_path or "overview": p for group in pages.values() for p in group})
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("### 💧 Water EWS v2")
    st.caption("Is it the **water** or the **sensor**?")
    st.toggle("Show help boxes", value=st.session_state.get("explain", True), key="explain",
              help="The blue 'how to use this page' boxes and the grey lines under charts. "
                   "Switch off for a cleaner screen.")
    st.markdown(
        '<div style="font-size:.82rem;line-height:1.7;margin-top:6px">'
        '<b>Colours mean the same everywhere</b><br>'
        '🟢 all good &nbsp; 🔴 the <b>water</b> changed<br>🟡 a <b>sensor</b> looks broken &nbsp; 🟣 you / the AI</div>',
        unsafe_allow_html=True)
nav.run()
