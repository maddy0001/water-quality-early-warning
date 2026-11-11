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
from ewsv2.app.views import (analytics, control_room, explorer, guide, live_lab, overview, reports,  # noqa: E402
                             system, version1)

st.set_page_config(page_title="Water Quality Early Warning", page_icon=":material/water_drop:", layout="wide",
                   initial_sidebar_state="expanded")
ui.setup_page()

pages = {
    "Monitor": [
        st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(control_room.render, title="Operations Console", icon=":material/monitor_heart:",
                url_path="control-room"),
        st.Page(live_lab.render, title="Sensor Lab", icon=":material/sensors:", url_path="live-lab"),
    ],
    "Analyze": [
        st.Page(analytics.render, title="Performance Analytics", icon=":material/analytics:", url_path="results"),
        st.Page(explorer.render, title="Data Explorer", icon=":material/query_stats:", url_path="data"),
        st.Page(reports.render, title="Incident Reports", icon=":material/description:", url_path="reports"),
    ],
    "Reference": [
        st.Page(guide.render, title="System Guide", icon=":material/menu_book:", url_path="guide"),
        st.Page(version1.render, title="Version 1 Baseline", icon=":material/history:", url_path="version-1"),
        st.Page(system.render, title="System Status", icon=":material/settings:", url_path="system"),
    ],
}
from ewsv2.app import nav as NAV  # noqa: E402

NAV.PAGES.update({p.url_path or "overview": p for group in pages.values() for p in group})
st.logo(str(Path(__file__).with_name("assets") / "logo.svg"), size="large")
nav = st.navigation(pages)
with st.sidebar:
    st.divider()
    from ewsv2.app.views.common import sidebar_status  # noqa: E402
    sidebar_status()
nav.run()
