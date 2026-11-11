"""Version 1 (paper 1): its published results, its own dashboard (if running), and how its parts did in v2."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    v1 = S.snapshot()["v1"]
    cfg = S.config()["dashboard"]
    ui.hero("Version 1 – the published system", f"\"An Integrated Machine Learning Framework for Real-Time Water Quality "
            f"Monitoring\", {v1['venue']}. Version 1 is kept exactly as published; version 2 builds on it.",
            eyebrow="Paper 1", color="slate")
    ui.guide('How to read this page', [
        'This is paper 1 (ICESC 2026), unchanged. If its own dashboard is running (port 8501) it is shown inside this page.',
        "Below it: how paper 1's parts behaved when tested honestly on real utility data in version 2.",
    ])
    ui.kpi_row([("Potability classifier", f"{v1['classification_accuracy']:.2%}", f"ROC-AUC {v1['roc_auc']:.2%} · F1 {v1['f1']:.2%}"),
                ("BiLSTM pH forecast", f"RMSE {v1['bilstm_rmse']}", f"R² {v1['bilstm_r2']}"),
                ("2-of-4 anomaly vote", f"{v1['anomaly_rate']:.2%}", "of samples flagged (expected ~5%)"),
                ("Risk fusion", "0.40 / 0.30 / 0.30", "classifier / forecast / anomaly → 4 levels")])
    st.write("")
    t1, t2 = st.tabs(["Version 1 dashboard", "From version 1 to version 2"])
    with t1:
        alive = S.url_alive(cfg["v1_url"])
        if alive:
            st.markdown(ui.pill("version 1 dashboard is running", "ok") + f' <span class="muted">{ui.esc(cfg["v1_url"])}</span>',
                        unsafe_allow_html=True)
            st.link_button("Open it in its own tab", cfg["v1_url"])
            components.iframe(cfg["v1_url"], height=900, scrolling=True)
        else:
            st.markdown(ui.pill("version 1 dashboard is not running", "warn"), unsafe_allow_html=True)
            st.markdown(
                "Version 1 has its own environment (`venv`) and its own dashboard. `python -m ewsv2 dashboard` starts it "
                "automatically when it finds `venv` and `dashboard\\app.py` in the project folder. To start it by hand, "
                "in a second PowerShell window:")
            st.code("cd \"C:\\Users\\THE GOD\\Documents\\Github\\water-quality-early-warning\"\nvenv\\Scripts\\activate\n"
                    "cd dashboard\nstreamlit run app.py --server.port 8501", language="powershell")
            st.caption("It needs the version-1 models (python main.py --train, once). Then reload this page.")
    with t2:
        st.markdown("**What version 2 kept, tested and changed**")
        st.dataframe(pd.DataFrame(S.snapshot()["paper1_on_real_data"]).rename(
            columns={"part": "Paper-1 part", "v2_test": "Tested in", "result": "Result on real utility data"}),
            hide_index=True, width="stretch")
        st.markdown(
            "- **Same modules, one stream.** Paper 1 ran its three modules on three different datasets; v2 runs all of "
            "them on the same real station, minute by minute.\n"
            "- **Honest test.** Official time-ordered test periods; the leakage audit shows how much a random split hides.\n"
            "- **New questions.** Water or sensor? A promised false-alarm rate. Explanations that are checked. Reports "
            "that cannot invent facts. The whole decision on a ₹350 microcontroller.")
