"""Results: the evidence of every segment – from this computer's output files, or the bundled snapshot."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    ui.hero("Results", "What each part of version 2 showed on real utility data – honest protocol, official test "
            "periods, unseen contaminants and sensor faults.")
    ui.guide('How to read this page', [
        'Each tab is one question the project answers, with the number to quote in the report.',
        '<b>Event F1</b> = how well alarms line up with the real events (1.0 is perfect). <b>False alarms/day</b> = how often the operator is disturbed for nothing.',
        '“Honest” means trained on the past and tested on the future, never on shuffled minutes.',
    ])
    snap = S.snapshot()
    tabs = st.tabs(["Leakage audit", "Paper 1 on real data", "Water or sensor?", "False-alarm budget",
                    "Explanations", "LLM reports", "Arduino Uno"])
    with tabs[0]:
        _leakage(snap)
    with tabs[1]:
        _paper1(snap)
    with tabs[2]:
        _triage(snap)
    with tabs[3]:
        _budget(snap)
    with tabs[4]:
        _explain(snap)
    with tabs[5]:
        _llm(snap)
    with tabs[6]:
        _edge(snap)


def _bars(values: dict, title: str, highlight: str | None = None, fmt: str = "{:.2f}", height: int = 320) -> go.Figure:
    names = list(values)
    v = [values[k] for k in names]
    colors = [ui.ACCENT if k == highlight else "#3b4d66" for k in names]
    fig = go.Figure(go.Bar(x=v, y=names, orientation="h", marker_color=colors, text=[fmt.format(x) for x in v],
                           textposition="outside"))
    fig.update_layout(title=title, height=height, yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(v) * 1.2]))
    return fig


def _leakage(snap) -> None:
    lk = snap["leakage"]
    ui.kpi_row([("Random split (leaky)", f"{lk['random_split_event_f1']:.2f}", "event F1, Random Forest"),
                ("Time-ordered (honest)", f"{lk['honest_event_f1']:.2f}", "same model, GECCO 2018"),
                ("Inflation", f"{lk['random_split_event_f1'] / lk['honest_event_f1']:.1f}×", "what a random split hides")])
    c1, c2 = st.columns(2)
    c1.plotly_chart(_bars(lk["baselines_2018"], "Honest event F1 – GECCO 2018", "random_forest"), width="stretch")
    c2.plotly_chart(_bars(lk["baselines_2019"], "Honest event F1 – GECCO 2019", "random_forest"), width="stretch")
    st.caption("S1: neighbouring minutes are almost identical, so a random split puts the answer in the training data. "
               "Every number in this dashboard uses the official, time-ordered test periods.")


def _paper1(snap) -> None:
    st.markdown("Every part of paper 1, moved onto the same real stream and tested the same honest way:")
    st.dataframe(pd.DataFrame(snap["paper1_on_real_data"]).rename(
        columns={"part": "Paper-1 part", "v2_test": "Tested in", "result": "Result"}), hide_index=True,
        width="stretch")
    st.caption("Version 1 itself is unchanged – see the *Version 1* page.")


def _triage(snap) -> None:
    tri = snap["triage"]
    df = S.read_csv("s4_triage", "gecco2018", "summary.csv")
    ui.kpi_row([("Faulty sensors that raised a water alarm", f"{tri['2018']['detector_only']['faults_to_water']} → "
                 f"{tri['2018']['rules']['faults_to_water']}", "GECCO 2018, 24 injected faults"),
                ("Faults sent to maintenance", f"{tri['2018']['rules']['faults_to_sensor']}/24", "with the rule triage"),
                ("False water alarms / day", f"{tri['2018']['detector_only']['water_fa_day']:.2f} → "
                 f"{tri['2018']['rules']['water_fa_day']:.2f}", "no real event lost")])
    if df is not None:
        st.dataframe(df, hide_index=True, width="stretch")
    st.caption("S4: a broken sensor moves alone, too fast, or not at all while its partners move. The rules use no fault labels.")


def _budget(snap) -> None:
    b = snap["budget"]
    df = S.read_csv("s5_fusion", "gecco2018", "fusion_results.csv")
    ui.kpi_row([("Fixed threshold (2018)", f"{b['fixed_fa_day_range_2018'][0]:.2f}–{b['fixed_fa_day_range_2018'][1]:.2f}",
                 "false alarms/day across 7 fusions"),
                ("Adaptive threshold (2018)", f"{b['adaptive_fa_day_range_2018'][0]:.2f}–{b['adaptive_fa_day_range_2018'][1]:.2f}",
                 "the promise: 1/day"),
                ("Adaptive threshold (2019)", f"{b['adaptive_fa_day_range_2019'][0]:.2f}–{b['adaptive_fa_day_range_2019'][1]:.2f}",
                 "fixed: 0.05–3.54")])
    if df is not None:
        d = df[["fusion", "calibrator", "real_caught", "real_n", "contaminant_caught", "contaminant_n",
                "false_alarms_per_day", "event_f1"]].copy()
        fig = go.Figure()
        for cal, color in (("budget", "#3b4d66"), ("adaptive", ui.ACCENT), ("cv_f1", ui.VIOLET)):
            x = d[d["calibrator"] == cal]
            fig.add_trace(go.Bar(x=x["fusion"], y=x["false_alarms_per_day"], name=cal, marker_color=color))
        fig.add_hline(y=1.0, line=dict(color=ui.WARN, dash="dash"), annotation_text="budget 1/day")
        fig.update_layout(barmode="group", height=360, title="False alarms per day on the test period (GECCO 2018)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(d, hide_index=True, width="stretch")
    else:
        st.info("Run `python -m ewsv2 fuse --config s5_fusion` to see the full table here.")


def _explain(snap) -> None:
    e = snap["explanations"]
    ui.kpi_row([("Simple deviation explainer", f"{e['deviation_hit1_2018'][0]:.2f}–{e['deviation_hit1_2018'][1]:.2f}",
                 "top sensor really changed (contaminants, 2018)"),
                ("Deletion test – deviation", f"{e['deviation_faithful'][0]:.2f}–{e['deviation_faithful'][1]:.2f}",
                 "holding the top sensor removes the alarm"),
                ("Deletion test – TreeSHAP (paper 1)", f"{e['treeshap_faithful'][0]:.2f}–{e['treeshap_faithful'][1]:.2f}",
                 "weakest; explains only one model")])
    df = S.read_csv("s6_explain", "gecco2018", "explanation_summary.csv")
    if df is not None:
        st.dataframe(df, hide_index=True, width="stretch")


def _llm(snap) -> None:
    df = pd.DataFrame(snap["llm"])
    fig = go.Figure(go.Bar(x=df["model"] + " · " + df["station"], y=df["faithful"],
                           marker_color=[ui.OK if v >= 0.99 else ui.WARN for v in df["faithful"]],
                           text=[f"{v:.2f}" for v in df["faithful"]], textposition="outside"))
    fig.update_layout(height=320, title="Share of reports with no fact error (checked automatically)", yaxis=dict(range=[0, 1.15]))
    st.plotly_chart(fig, width="stretch")
    st.caption("S7: the LLM only writes text about an alarm the pipeline already decided; a report with any wrong "
               "number, time, sensor or citation is replaced by the template, so the operator never sees an error.")


def _edge(snap) -> None:
    e = snap["edge"]
    ui.kpi_row([("Minutes on the real Uno", f"{e['uno_minutes']:,}", f"bit-identical: {e['bit_identical']:.5f}"),
                ("Time per minute of data", f"{e['us_per_minute'] / 1000:.1f} ms", f"max {e['max_us'] / 1000:.1f} ms; 0.04% busy"),
                ("Program / memory", f"{e['flash_bytes'] / 1024:.1f} KB / {e['ram_bytes']} B", "of 32 KB / 2,048 B"),
                ("Weights", f"{e['weights']}", f"{e['bilstm_weights'] / e['weights']:.0f}× fewer than paper 1's BiLSTM")])
    df = S.read_csv("s8_edge", "gecco2018", "edge_results.csv")
    if df is not None:
        st.dataframe(df[["system", "real_caught", "real_n", "contaminant_caught", "contaminant_n", "fault_correct",
                         "fault_n", "fault_raised_water_alarm", "water_false_alarms_per_day"]], hide_index=True,
                     width="stretch")
    else:
        st.markdown(f"Uno chain on GECCO 2018: real events **{e['real']}**, contaminants **{e['contaminants']}**, faults to "
                    f"maintenance **{e['faults_to_sensor']}**, **{e['water_fa_day']}** false alarms/day.")
