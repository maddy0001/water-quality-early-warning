"""Version 1 Baseline: the published system (paper 1), its own dashboard if running, and how its parts did in v2."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ewsv2.app import charts as C
from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    v1 = S.snapshot()["v1"]
    cfg = S.config()["dashboard"]
    ui.header("Version 1 Baseline",
              f"\"An Integrated Machine Learning Framework for Real-Time Water Quality Monitoring\", {v1['venue']}. "
              "Version 1 is kept exactly as published; version 2 builds on it and re-tests its parts on real data.",
              kicker="Reference · published system", color="slate")
    ui.tiles([("Potability classifier", f"{v1['classification_accuracy']:.1%}",
               f"accuracy · ROC-AUC {v1['roc_auc']:.1%} · F1 {v1['f1']:.1%}", "indigo"),
              ("BiLSTM pH forecast", f"RMSE {v1['bilstm_rmse']}", f"R² {v1['bilstm_r2']}", "ocean"),
              ("Anomaly vote (2 of 4)", f"{v1['anomaly_rate']:.1%}", "of samples flagged", "amber"),
              ("Risk fusion", "0.40 / 0.30 / 0.30", "classifier / forecast / anomaly → 4 risk levels", "violet")])
    t1, t2 = st.tabs(["Version 1 dashboard", "Version 1 parts on real utility data"])
    with t1:
        if S.url_alive(cfg["v1_url"]):
            st.markdown(ui.badge("Version 1 dashboard is running", "ok") + f' <span class="muted">{ui.esc(cfg["v1_url"])}'
                        '</span>', unsafe_allow_html=True)
            st.link_button("Open it in a separate tab", cfg["v1_url"], icon=":material/open_in_new:")
            components.iframe(cfg["v1_url"], height=900, scrolling=True)
        else:
            st.markdown(ui.badge("Version 1 dashboard is not running", "warn"), unsafe_allow_html=True)
            st.markdown("Version 1 has its own environment (`venv`) and dashboard. `python -m ewsv2 dashboard` starts it "
                        "automatically when `venv` and `dashboard\\app.py` exist in the project folder. To start it by "
                        "hand, in a second PowerShell window:")
            st.code("cd \"C:\\Users\\THE GOD\\Documents\\Github\\water-quality-early-warning\"\nvenv\\Scripts\\activate\n"
                    "cd dashboard\nstreamlit run app.py --server.port 8501", language="powershell")
    with t2:
        rows = S.snapshot()["paper1_on_real_data"]
        st.dataframe(pd.DataFrame([{"Version 1 component": r["part"], "Result on real utility data": r["result"]}
                                   for r in rows]), hide_index=True, width="stretch")
        p = S.snapshot()["pipeline_2018"]
        fig = go.Figure()
        cats = ["False water alarms per day", "Real events detected (of 46)"]
        fig.add_bar(name="Version 1 rule (2 of 4)", x=cats, y=[33.3, 46], marker_color="#f43f5e",
                    text=["33.3", "46"], textposition="outside")
        fig.add_bar(name="Version 2 pipeline", x=cats, y=[float(p["water_fa_day"]), int(p["real"].split("/")[0])],
                    marker_color=ui.ACCENT, text=[str(p["water_fa_day"]), p["real"].split("/")[0]],
                    textposition="outside")
        fig.update_layout(barmode="group", height=360, title="Version 1's fixed rule vs version 2 on utility data A",
                          yaxis=dict(type="log", title="log scale"), legend=dict(orientation="h", y=-0.15))
        C.plot(fig, key="v1_cmp")
        ui.cap("Version 1's rule catches every event but raises an alarm about every 45 minutes; version 2 detects "
               "most events with less than one false alarm per day.")
