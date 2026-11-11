"""Results: the evidence of every segment – from this computer's output files, or the bundled snapshot."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    ui.hero("Results", "What version 2 found, tested the honest way on real utility data. Each tab answers one "
            "question – the first line is the answer in plain words, the numbers below are the proof.",
            eyebrow="The proof", color="orange")
    ui.guide("How to read this page", [
        "Pick a question (tab). The <b>bold sentence</b> is the answer; the cards and charts are the evidence.",
        "<b>Event F1</b> = one score from 0 to 1 for “did the alarms match the real events?” (1 = perfect). "
        "<b>False alarms per day</b> = how often the operator is disturbed for nothing.",
        "Every number uses the <b>honest test</b>: learn from the past, test on the future.",
    ])
    snap = S.snapshot()
    tabs = st.tabs(["✅ Honest testing", "📜 Version 1 on real data", "🔧 Water or sensor?", "📏 False-alarm promise",
                    "🔍 Explanations", "🤖 AI reports", "🧠 Arduino"])
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


NICE = {"random_forest": "Random Forest (paper 1 style)", "isolation_forest": "Isolation Forest",
        "static_limits": "Fixed limits", "rolling_zscore": "Rolling z-score", "moving_variance": "Moving variance",
        "squared_diff": "Squared difference", "random_score": "Random guess"}


def _answer(text: str) -> None:
    st.markdown(f'<div class="big" style="margin:4px 0 12px 0"><b>{ui.esc(text)}</b></div>', unsafe_allow_html=True)


def _bars(values: dict, title: str, highlight: str | None = None, fmt: str = "{:.2f}", height: int = 320) -> go.Figure:
    names = list(values)
    v = [values[k] for k in names]
    colors = [ui.ACCENT if k == highlight else "#cbd5e1" for k in names]
    fig = go.Figure(go.Bar(x=v, y=[NICE.get(k, k) for k in names], orientation="h", marker_color=colors, text=[fmt.format(x) for x in v],
                           textposition="outside"))
    fig.update_layout(title=title, height=height, yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(v) * 1.2]))
    return fig


def _leakage(snap) -> None:
    lk = snap["leakage"]
    _answer("Version 1's test was too easy. Tested honestly, the same model scores 0.64 instead of 0.98 – "
            "so every number in version 2 uses the honest test.")
    ui.kpi_row([("Random split (leaky)", f"{lk['random_split_event_f1']:.2f}", "event F1, Random Forest"),
                ("Time-ordered (honest)", f"{lk['honest_event_f1']:.2f}", "same model, GECCO 2018"),
                ("Inflation", f"{lk['random_split_event_f1'] / lk['honest_event_f1']:.1f}×", "what a random split hides")])
    c1, c2 = st.columns(2)
    c1.plotly_chart(_bars(lk["baselines_2018"], "Honest event F1 – GECCO 2018", "random_forest"), width="stretch")
    c2.plotly_chart(_bars(lk["baselines_2019"], "Honest event F1 – GECCO 2019", "random_forest"), width="stretch")
    st.caption("S1: neighbouring minutes are almost identical, so a random split puts the answer in the training data. "
               "Every number in this dashboard uses the official, time-ordered test periods.")


def _paper1(snap) -> None:
    _answer("Every part of paper 1 was moved onto the same real station and tested honestly. Some held up (the "
            "ensemble, the hand-set weights), some did not (the fixed 2-of-4 rule: 33 false alarms a day).")
    st.dataframe(pd.DataFrame(snap["paper1_on_real_data"]).rename(
        columns={"part": "Paper-1 part", "v2_test": "Tested in", "result": "Result"}), hide_index=True,
        width="stretch")
    st.caption("Version 1 itself is unchanged – see the *Version 1* page.")


def _triage(snap) -> None:
    tri = snap["triage"]
    _answer("Broken sensors no longer cause water alarms: 7 → 1 on GECCO 2018, 3 → 0 on 2019 – and no real event "
            "was lost.")
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
    _answer("With a fixed threshold the false-alarm rate is luck (0.05 to 3.5 a day). The adaptive alarm line keeps "
            "the promise of about one a day on both real test periods.")
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
        for cal, color in (("budget", "#94a3b8"), ("adaptive", ui.ACCENT), ("cv_f1", ui.VIOLET)):
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
    _answer("The simple explanation (“which sensor moved most compared with normal”) is the right one 95–100% of "
            "the time; paper 1's TreeSHAP is weaker (74%).")
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
    _answer("A small AI running on the laptop wrote 59 of 60 reports with no mistake – and the one with a mistake "
            "was caught and replaced by the template.")
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
    _answer("The whole decision fits on a ₹350 Arduino Uno and gives exactly the same answer as the laptop – "
            "checked for one full week of real data.")
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
