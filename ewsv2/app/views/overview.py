"""Overview: what the system does, its headline results and where to go next."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import charts as C
from ewsv2.app import nav
from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    snap = S.snapshot()
    e, lk, p18, v1 = snap["edge"], snap["leakage"], snap["pipeline_2018"], snap["paper1_on_real_data"]
    ui.header("Water Quality Early Warning System",
              "Monitors drinking-water sensors minute by minute. For every unusual reading it decides whether the "
              "water itself changed or a sensor failed, explains the decision, learns from the operator's review "
              "and produces a fact-checked incident report.", kicker="Version 2 · Overview")

    real_ok, real_n = (int(x) for x in p18["faults_to_sensor"].split("/"))
    ui.tiles([
        ("Detection on unseen data", f"{lk['honest_event_f1']:.2f}", "event score (0–1) when trained on the past and "
                                                                       "tested on later months", "indigo"),
        ("Sensor faults routed correctly", f"{real_ok} / {real_n}", "injected sensor faults sent to maintenance "
                                                                     "instead of raising a water alarm", "amber"),
        ("False alarms per day", "1.0", "kept at the promised rate on both real test periods (0.90–1.18)", "emerald"),
        ("Edge device agreement", "100 %", f"{e['uno_minutes']:,} minutes: Arduino and laptop gave identical "
                                           "decisions", "ocean"),
        ("Incident reports correct", "59 / 60", "AI-written reports with no factual error; errors are blocked",
         "violet"),
    ])

    ui.section("Where to go", "Each page has one purpose. Start with the Operations Console for a live demonstration.")
    cols = st.columns(4)
    links = [
        ("control-room", "Operations Console", "Replay recorded utility data as if it were live, watch alarms being "
                                               "raised and review each one as the operator.", "Open the console"),
        ("live-lab", "Sensor Lab", "Live readings from the Arduino sensors (or a simulation): contaminate the water, "
                                   "disconnect the probe and see the system respond.", "Open the lab"),
        ("results", "Performance Analytics", "Every experiment result as interactive charts: detection, fault "
                                             "routing, false-alarm control, explanations, reports, edge device.",
         "View the analytics"),
        ("guide", "System Guide", "How the system works, stage by stage, including the hardware, the AI report "
                                  "writer and the terms used in this dashboard.", "Read the guide"),
    ]
    for c, (path, title, text, label) in zip(cols, links):
        with c:
            ui.link_card(nav.PAGES.get(path), title, text, label)

    ui.section("How a decision is made", "The same five stages run on the laptop and on the Arduino.")
    ui.pipeline([
        ("Acquire", "Every sensor reports once per minute: pH, chlorine, conductivity, turbidity, temperature, flow."),
        ("Validate sensors", "Health checks flag sensors that are stuck, dropping out or spiking. Their readings are "
                             "excluded from the water assessment."),
        ("Score the water", "A compact neural network rates how unusual the water is right now: the "
                            "<b>anomaly score</b>."),
        ("Decide", "Score above the <b>alarm threshold</b> → water-quality alarm. A problem confined to one "
                   "failing sensor → sensor-fault alert for maintenance."),
        ("Review and report", "The operator confirms or rejects each alarm, the threshold adapts, and an incident "
                              "report is generated and fact-checked."),
    ])

    ui.section("Key results", "Measured on public data from a German drinking-water utility (GECCO 2018 and 2019).")
    a, b, c = st.columns([1.1, 1, 1])
    with a:
        fig = go.Figure()
        fig.add_bar(x=["Shuffled test<br>(version 1)", "Time-ordered test<br>(version 2)"],
                    y=[lk["random_split_event_f1"], lk["honest_event_f1"]],
                    marker=dict(color=["#94a3b8", ui.ACCENT]), text=[f"{lk['random_split_event_f1']:.2f}",
                                                                     f"{lk['honest_event_f1']:.2f}"],
                    textposition="outside", width=0.5)
        fig.update_layout(title="Same model, two ways of testing", height=320, yaxis=dict(range=[0, 1.15],
                          title="event score"))
        C.plot(fig, key="ov_leak")
        ui.cap("Shuffling minutes lets the model see the answers during training. Version 2 reports only "
               "time-ordered results.")
    with b:
        miss = real_n - real_ok - int(p18["faults_to_water"])
        C.plot(C.donut(["Sent to maintenance", "Raised a water alarm", "Not detected"],
                       [real_ok, int(p18["faults_to_water"]), miss], [ui.OK, ui.BAD, "#cbd5e1"],
                       "Where 24 injected sensor faults went", f"{real_ok}/{real_n}"), key="ov_faults")
        ui.cap("A failing sensor should create a maintenance task, not a public water alarm.")
    with c:
        rule = next((r for r in v1 if r["part"].startswith("Fixed 2-of-4")), None)
        fig = go.Figure(go.Bar(x=["Version 1 rule", "Version 2"], y=[33.3, float(p18["water_fa_day"])],
                               marker=dict(color=["#f43f5e", ui.OK]), text=["33.3", f"{p18['water_fa_day']}"],
                               textposition="outside", width=0.5))
        fig.update_layout(title="False water alarms per day", height=320,
                          yaxis=dict(type="log", title="per day (log scale)", range=[-1, 2],
                                     tickvals=[0.1, 1, 10, 100], ticktext=["0.1", "1", "10", "100"]))
        C.plot(fig, key="ov_fa")
        ui.cap("Version 1's fixed 2-of-4 rule on real data" + (" (" + rule["result"].split(" but ")[-1] + ")"
                                                              if rule else "") + " versus the complete version 2 "
               "pipeline.")
