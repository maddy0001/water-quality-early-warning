"""Performance Analytics: every experiment result stored on this computer, as interactive charts.

Reads the result files in ``outputs_v2``. Where a file is missing, the measured headline values bundled with the
code are shown instead, so the page always works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import charts as C
from ewsv2.app import services as S
from ewsv2.app import ui

METHOD = {
    "random_score": "Random guess (reference)", "squared_diff": "Squared difference", "moving_variance": "Moving variance",
    "rolling_zscore": "Rolling z-score", "static_limits": "Fixed limits", "isolation_forest": "Isolation forest",
    "random_forest": "Random forest", "v1_quorum": "V1 anomaly vote (2 of 4)", "v1_ensemble": "V1 classifier ensemble",
    "v1_ensemble_nocoh": "V1 ensemble, no coherence features", "v1_ensemble_basic": "V1 ensemble, basic features",
    "forecast_persistence": "Forecast: persistence", "forecast_linear": "Forecast: linear",
    "v1_gru_attention": "Forecast: GRU-attention", "v1_bilstm_attention": "Forecast: BiLSTM-attention (V1)",
}
FUSION = {"C alone": "Classifier only", "A alone": "Anomaly detectors only", "F alone": "Forecaster only",
          "hand (v1 0.40/0.30/0.30)": "Version 1 weights", "equal (1/3 each)": "Equal weights",
          "max (any module)": "Any module", "stacking (learned)": "Learned combination"}
CALIB = {"cv_f1": "Fixed (tuned for score)", "budget": "Fixed (alarm budget)", "adaptive": "Adaptive"}
EXPLAINER = {"deviation": "Deviation from normal", "forecast_residual": "Forecast error", "tree_shap": "TreeSHAP (V1)",
             "occlusion": "Occlusion", "random": "Random (reference)"}
OUTCOME_COLORS = {"water alarm": ui.BAD, "water alarm only": ui.BAD, "sensor alert only": ui.WARN, "both": ui.VIOLET,
                  "missed": "#94a3b8", "real event": ui.ACCENT, "injected contaminant": ui.CYAN,
                  "injected fault": "#ea580c"}


def _method(s: str) -> str:
    base, _, part = s.partition(":")
    name = METHOD.get(base, base)
    return f"{name} – {part.replace('_', ' ')}" if part else name


def render() -> None:
    ui.header("Performance Analytics",
              "All experiment results stored on this computer, as interactive charts: how well the system detects "
              "water events, routes sensor faults, controls false alarms, explains itself, writes reports and runs "
              "on the edge device.", kicker="Analyze · results", color="violet")
    with st.container(border=True):
        c1, c2 = st.columns([1.4, 2])
        with c1:
            ds = st.segmented_control("Dataset", ["gecco2018", "gecco2019"], default="gecco2018", key="an_ds",
                                      format_func=lambda s: S.source_label(s, short=True)) or "gecco2018"
        with c2:
            ui.cap("<br>Results are computed separately for each real dataset, always trained on earlier months and "
                   "tested on later months. Hover over any chart for exact values; drag to zoom; double-click to "
                   "reset.")
    tabs = st.tabs(["Evaluation method", "Detection methods", "Water event or sensor fault", "False-alarm control",
                    "Explanations", "Incident reports", "Edge device"])
    with tabs[0]:
        _evaluation(ds)
    with tabs[1]:
        _detection(ds)
    with tabs[2]:
        _routing(ds)
    with tabs[3]:
        _false_alarms(ds)
    with tabs[4]:
        _explanations(ds)
    with tabs[5]:
        _reports(ds)
    with tabs[6]:
        _edge(ds)


def _finding(text: str) -> None:
    st.markdown(f'<div class="card" style="border-left:5px solid {ui.ACCENT}"><div class="h">Finding</div>'
                f'{ui.esc(text)}</div>', unsafe_allow_html=True)


def _missing(what: str) -> None:
    st.info(f"The detailed result file for {what} is not on this computer for this dataset. The headline values "
            "bundled with the software are shown.", icon=":material/info:")


def _data(df: pd.DataFrame, name: str, key: str) -> None:
    with st.expander("Show the data behind these charts"):
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv",
                           on_click="ignore", key=key, icon=":material/download:")


# --------------------------------------------------------------------------------------------- 1. evaluation

def _evaluation(ds: str) -> None:
    lk = S.snapshot()["leakage"]
    _finding(f"How a model is tested changes its score dramatically. The same random forest scores "
             f"{lk['random_split_event_f1']:.2f} when minutes are shuffled (the version 1 method) but "
             f"{lk['honest_event_f1']:.2f} when it is trained on the past and tested on later months. All other "
             "results in this dashboard use the time-ordered test.")
    ui.tiles([("Shuffled test", f"{lk['random_split_event_f1']:.2f}", "event score, random forest", "slate"),
              ("Time-ordered test", f"{lk['honest_event_f1']:.2f}", "same model, honest evaluation", "indigo"),
              ("Over-estimation", f"{lk['random_split_event_f1'] / lk['honest_event_f1']:.1f}×",
               "how much shuffling inflates the score", "rose")])
    df = S.read_csv("s1_audit", "gecco2018", "results.csv")
    if df is None:
        _missing("the evaluation audit")
        return
    df = df.copy()
    df["Method"] = df["signal"].map(_method)
    df["Protocol"] = df["protocol_label"].str.replace(r"^[A-D]\. ", "", regex=True)
    fig = px.bar(df, x="Method", y="event_f1", color="Protocol", barmode="group",
                 color_discrete_sequence=["#94a3b8", "#f59e0b", "#06b6d4", ui.ACCENT],
                 labels={"event_f1": "event score (0–1)"})
    fig.update_layout(title="Event score of each method under four testing protocols (utility data A)", height=430,
                      legend=dict(orientation="h", y=-0.35, x=0, title=None), xaxis_title=None)
    C.plot(fig, key="an_audit")
    ui.cap("The first three protocols tune the alarm threshold on the test data itself or mix past and future; only "
           "the last one (indigo) is a fair test. This audit was run on utility data A.")
    _data(df[["Protocol", "Method", "event_f1", "event_recall", "false_alarms_per_day", "point_f1"]],
          "evaluation_audit.csv", "an_dl_audit")


# --------------------------------------------------------------------------------------------- 2. detection

def _detection(ds: str) -> None:
    frames = [S.read_csv(seg, ds, "results.csv") for seg in ("s1_baselines", "s2_v1_port", "s3_forecast")]
    frames = [f for f in frames if f is not None]
    if not frames:
        _missing("detection methods")
        base = S.snapshot()["leakage"]["baselines_2018" if ds == "gecco2018" else "baselines_2019"]
        C.plot(C.hbar([_method(k) for k in base], list(base.values()), "Event score per method"), key="an_det_snap")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df[df["honest"].astype(str) == "True"].drop_duplicates("signal")
    df = df[~df["signal"].str.contains(":")].copy()
    df["Method"] = df["signal"].map(_method)
    best = df.sort_values("event_f1", ascending=False).iloc[0]
    _finding(f"On {S.source_label(ds, short=True)} the strongest single method is {best['Method']} (event score "
             f"{best['event_f1']:.2f}, {int(best['detected'])} of {int(best['events'])} events, "
             f"{best['false_alarms_per_day']:.2f} false alarms per day). Version 1's fixed vote catches events "
             "only at the cost of many false alarms or misses most events once its threshold is set fairly.")
    a, b = st.columns([1.25, 1])
    with a:
        d = df.copy()
        d["fa"] = d["false_alarms_per_day"].clip(lower=0.01)
        d["size"] = d["event_f1"].fillna(0) * 50 + 8
        fig = px.scatter(d, x="fa", y="event_recall", size="size", color="Method", log_x=True, size_max=36,
                         color_discrete_sequence=ui.SERIES, hover_data={"event_f1": ":.2f", "size": False, "fa": ":.2f"},
                         labels={"fa": "false alarms per day (log scale)", "event_recall": "share of events detected"})
        fig.update_layout(title="Detection vs disturbance (bubble size = event score)", height=430,
                          legend=dict(orientation="v", x=1.02, y=1, title=None, font=dict(size=10)))
        fig.add_vrect(x0=0.01, x1=1, fillcolor="rgba(16,185,129,.07)", line_width=0)
        C.plot(fig, key="an_bubble")
        ui.cap("Top-left is best: most events found with few false alarms. The green area is one false alarm per "
               "day or less.")
    with b:
        d = df.sort_values("event_f1", ascending=True)
        fig = go.Figure(go.Bar(x=d["event_f1"], y=d["Method"], orientation="h",
                               marker=dict(color=d["event_f1"], colorscale="Plasma", cmin=0, cmax=1),
                               text=[f"{v:.2f}" for v in d["event_f1"]], textposition="outside"))
        fig.update_layout(title="Event score (0–1)", height=430, xaxis=dict(range=[0, 1.1]))
        C.plot(fig, key="an_rank")
    pe = [S.read_csv(seg, ds, "per_event.csv") for seg in ("s1_baselines", "s2_v1_port", "s3_forecast")]
    pe = [p for p in pe if p is not None]
    if pe:
        p = pd.concat(pe, ignore_index=True).drop_duplicates(["signal", "start"])
        p = p[~p["signal"].str.contains(":")]
        p["Method"] = p["signal"].map(_method)
        p["hit"] = p["detected"].astype(str).eq("True").astype(int)
        piv = p.pivot_table(index="Method", columns="start", values="hit", aggfunc="max").fillna(0)
        piv = piv.loc[piv.sum(axis=1).sort_values().index]
        fig = go.Figure(go.Heatmap(z=piv.values, x=[str(c)[5:16] for c in piv.columns], y=piv.index,
                                   colorscale=[[0, "#eef2ff"], [1, ui.ACCENT]], showscale=False, xgap=1, ygap=1,
                                   hovertemplate="%{y}<br>event %{x}<br>detected: %{z}<extra></extra>"))
        fig.update_layout(title="Which real events each method detected (dark = detected)", height=120 + 28 * len(piv),
                          xaxis=dict(tickangle=-60, tickfont=dict(size=9)))
        C.plot(fig, key="an_heat")
        ui.cap("One column per real event in the test period. Events missed by every method are hard cases.")
    _data(df[["Method", "event_f1", "event_recall", "detected", "events", "false_alarms_per_day", "median_delay_min"]],
          f"detection_{ds}.csv", "an_dl_det")


# --------------------------------------------------------------------------------------------- 3. routing

def _routing(ds: str) -> None:
    summ = S.read_csv("s4_triage", ds, "summary.csv")
    tri = S.snapshot()["triage"]["2018" if ds == "gecco2018" else "2019"]
    _finding(f"Without fault checks, {tri['detector_only']['faults_to_water']} injected sensor faults raised a public "
             f"water alarm; with the rule-based checks only {tri['rules']['faults_to_water']} did, while "
             f"{tri['rules']['faults_to_sensor']} of 24 were sent to maintenance – and no real water event was lost.")
    if summ is None:
        _missing("fault routing")
        return
    s = summ.copy()
    s["System"] = s["label"]
    long = pd.DataFrame({
        "System": np.repeat(s["System"].values, 4),
        "Measure": ["Real events caught", "Contaminations caught", "Faults sent to maintenance",
                    "Faults raising water alarm"] * len(s),
        "Count": np.ravel(s[["real_caught", "contaminant_caught", "fault_correct", "fault_raised_water_alarm"]].values),
    })
    fig = px.bar(long, x="Measure", y="Count", color="System", barmode="group", text="Count",
                 color_discrete_sequence=["#94a3b8", ui.ACCENT, ui.VIOLET])
    fig.update_layout(title=f"Outcome counts per system (of {int(s['real_n'][0])} real events, "
                            f"{int(s['contaminant_n'][0])} contaminations, {int(s['fault_n'][0])} sensor faults)",
                      height=400, legend=dict(orientation="h", y=-0.2, title=None), xaxis_title=None)
    C.plot(fig, key="an_route_bar")
    outs = [S.read_csv("s4_triage", ds, "outcomes.csv"), S.read_csv("s5_fusion", ds, "pipeline_outcomes.csv")]
    outs = pd.concat([o for o in outs if o is not None], ignore_index=True) if any(o is not None for o in outs) else None
    if outs is not None:
        names = {"detector_only": "Detector only (version 1 style)", "rules": "Detector + rule-based fault checks",
                 "learned": "Detector + learned fault classifier"}
        systems = list(dict.fromkeys(outs["system"]))
        pick = st.selectbox("System shown in the flow diagram", systems,
                            index=systems.index("rules") if "rules" in systems else 0,
                            format_func=lambda x: names.get(x, x.replace("stacking (learned)", "Complete pipeline")),
                            key="an_route_sys")
        ui.cap("Choose a configuration to see where every event and fault ended up.")
        o = outs[outs["system"] == pick].groupby(["source", "outcome"]).size().reset_index(name="n")
        C.plot(C.sankey(o["source"].tolist(), o["outcome"].tolist(), o["n"].astype(float).tolist(),
                        "From what happened (left) to what the system did (right)", height=420,
                        node_colors=OUTCOME_COLORS), key="an_sankey")
        ui.cap("Left: real events, injected contaminations and injected sensor faults. Right: water alarm, "
               "sensor alert, both, or missed. Ideally faults flow to 'sensor alert only' and water events to "
               "'water alarm'.")
    _data(s.drop(columns=["System"]), f"fault_routing_{ds}.csv", "an_dl_route")


# --------------------------------------------------------------------------------------------- 4. false alarms

def _false_alarms(ds: str) -> None:
    b = S.snapshot()["budget"]
    y = "2018" if ds == "gecco2018" else "2019"
    _finding(f"A fixed threshold gives anywhere from {b[f'fixed_fa_day_range_{y}'][0]:.2f} to "
             f"{b[f'fixed_fa_day_range_{y}'][1]:.2f} false alarms per day depending on the model. The adaptive "
             f"threshold keeps every model at the promised one per day ({b[f'adaptive_fa_day_range_{y}'][0]:.2f}–"
             f"{b[f'adaptive_fa_day_range_{y}'][1]:.2f}).")
    w = S.read_csv("s5_fusion", ds, "weekly_false_alarms.csv")
    fr = S.read_csv("s5_fusion", ds, "fusion_results.csv")
    if w is None or fr is None:
        _missing("false-alarm control")
        return
    fusions = list(dict.fromkeys(c.split(" | ")[0] for c in w.columns if " | " in c))
    pick = st.selectbox("Model combination", fusions, index=fusions.index("stacking (learned)")
                        if "stacking (learned)" in fusions else 0, format_func=lambda f: FUSION.get(f, f),
                        key="an_fa_pick")
    ui.cap("How the module outputs are combined into one anomaly score. The chart compares three ways of setting "
           "the alarm threshold for that combination.")
    a, b2 = st.columns([1.5, 1])
    with a:
        fig = go.Figure()
        for cal, col in (("cv_f1", "#94a3b8"), ("budget", "#f59e0b"), ("adaptive", ui.ACCENT)):
            c = f"{pick} | {cal}"
            if c in w.columns:
                fig.add_trace(go.Scatter(x=w["time"], y=w[c], name=CALIB[cal], mode="lines+markers",
                                         line=dict(width=3 if cal == "adaptive" else 2, color=col, shape="spline"),
                                         fill="tozeroy" if cal == "adaptive" else None,
                                         fillcolor=C.alpha(ui.ACCENT, 0.08)))
        fig.add_hline(y=1, line=dict(color=ui.BAD, dash="dash"), annotation_text="promise: 1 per day")
        fig.update_layout(title="False alarms per day, week by week", height=380, yaxis_title="per day",
                          legend=dict(orientation="h", y=-0.2))
        C.plot(fig, key="an_weekly")
    with b2:
        f = fr.copy()
        f["Combination"] = f["fusion"].map(lambda x: FUSION.get(x, x))
        f["Threshold"] = f["calibrator"].map(lambda x: CALIB.get(x, x))
        piv = f.pivot_table(index="Combination", columns="Threshold", values="false_alarms_per_day")
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Turbo", zmin=0, zmax=2,
                                   text=np.round(piv.values, 2), texttemplate="%{text}", xgap=2, ygap=2,
                                   colorbar=dict(title="per day")))
        fig.update_layout(title="False alarms per day (target 1.0)", height=380)
        C.plot(fig, key="an_fa_heat")
    f = fr.copy()
    f["Combination"] = f["fusion"].map(lambda x: FUSION.get(x, x))
    f["Threshold"] = f["calibrator"].map(lambda x: CALIB.get(x, x))
    fig = px.scatter(f, x="false_alarms_per_day", y="water_recall", color="Combination", symbol="Threshold",
                     size=[14] * len(f), color_discrete_sequence=ui.SERIES,
                     labels={"false_alarms_per_day": "false alarms per day", "water_recall": "share of water events "
                                                                                               "detected"})
    fig.add_vline(x=1, line=dict(color=ui.BAD, dash="dot"))
    fig.update_layout(title="Every combination and threshold method: detection vs false alarms", height=420,
                      legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)))
    C.plot(fig, key="an_fa_scatter")
    ui.cap("Shapes are the threshold methods; adaptive points line up on the dotted 1-per-day line.")
    _data(fr, f"false_alarm_control_{ds}.csv", "an_dl_fa")


# --------------------------------------------------------------------------------------------- 5. explanations

def _explanations(ds: str) -> None:
    e = S.snapshot()["explanations"]
    _finding(f"Naming the sensor that moved most compared with its normal behaviour is a faithful explanation "
             f"({e['deviation_faithful'][0]:.2f}–{e['deviation_faithful'][1]:.2f} of alarms disappear when that sensor "
             f"is held still). Version 1's TreeSHAP is weaker ({e['treeshap_faithful'][0]:.2f}) and only explains "
             "one model.")
    df = S.read_csv("s6_explain", ds, "explanation_summary.csv")
    if df is None:
        _missing("explanations")
        return
    d = df.copy()
    d["Explainer"] = d["explainer"].map(lambda x: EXPLAINER.get(x, x))
    a, b = st.columns(2)
    with a:
        fig = px.bar(d, x="source", y="faithful1", color="Explainer", barmode="group",
                     color_discrete_sequence=[ui.ACCENT, ui.CYAN, "#f59e0b", ui.VIOLET, "#cbd5e1"],
                     labels={"faithful1": "faithfulness (0–1)", "source": ""})
        fig.update_layout(title="Faithfulness: removing the named sensor removes the alarm", height=400,
                          legend=dict(orientation="h", y=-0.25, title=None), yaxis=dict(range=[0, 1.05]))
        C.plot(fig, key="an_faith")
    with b:
        fig = go.Figure()
        for i, (src, g) in enumerate(d.groupby("source")):
            fig.add_trace(go.Scatterpolar(r=g["faithful1"].fillna(0).tolist() + [g["faithful1"].fillna(0).iloc[0]],
                                          theta=g["Explainer"].tolist() + [g["Explainer"].iloc[0]], name=src,
                                          fill="toself", line=dict(color=[ui.ACCENT, ui.CYAN, "#ea580c"][i % 3]),
                                          fillcolor=C.alpha([ui.ACCENT, ui.CYAN, "#ea580c"][i % 3], 0.18)))
        fig.update_layout(title="Faithfulness profile per explainer", height=400,
                          polar=dict(radialaxis=dict(range=[0, 1])), legend=dict(orientation="h", y=-0.15))
        C.plot(fig, key="an_radar")
    tr = S.read_csv("s6_explain", "transfer_gecco2018_to_gecco2019", "transfer_results.csv")
    if tr is not None:
        t = tr.copy()
        t["Setting"] = t["system"] + " · " + t["threshold"]
        fig = px.bar(t, y="Setting", x="event_f1", color="false_alarms_per_day", orientation="h",
                     color_continuous_scale="Sunsetdark", labels={"event_f1": "event score", "false_alarms_per_day":
                                                                  "false alarms/day"})
        fig.update_layout(title="Moving a 2018 model to the 2019 station without retraining", height=520,
                          yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
        C.plot(fig, key="an_transfer")
        ui.cap("Colour shows false alarms per day. Models moved to a new station need their threshold refitted on "
               "local data.")
    _data(d.drop(columns=["Explainer"]), f"explanations_{ds}.csv", "an_dl_exp")


# --------------------------------------------------------------------------------------------- 6. reports

def _reports(ds: str) -> None:
    llm = pd.DataFrame(S.snapshot()["llm"])
    _finding("A small language model running on the laptop wrote 59 of 60 incident reports without a single factual "
             "error. Every report is checked automatically against the alarm record; a report with an error is never "
             "shown – the standard template is shown instead.")
    a, b = st.columns(2)
    with a:
        lab = llm["model"] + " · " + llm["station"]
        fig = go.Figure(go.Bar(x=lab, y=llm["faithful"], marker=dict(color=[ui.VIOLET, "#c026d3", ui.PINK, ui.CYAN]),
                               text=[f"{v:.0%}" for v in llm["faithful"]], textposition="outside"))
        fig.update_layout(title="Share of reports without factual error", height=360, yaxis=dict(range=[0, 1.15],
                          tickformat=".0%"))
        C.plot(fig, key="an_llm_share")
    with b:
        fig = go.Figure(go.Bar(x=lab, y=llm["seconds"], marker=dict(color=llm["seconds"], colorscale="Purp"),
                               text=[f"{v:.1f} s" for v in llm["seconds"]], textposition="outside"))
        fig.update_layout(title="Time to write one report (laptop CPU)", height=360, yaxis_title="seconds")
        C.plot(fig, key="an_llm_time")
    chk = S.read_csv("s7_reports", ds, "report_checks.csv")
    if chk is not None:
        c = chk.copy()
        c["result"] = np.where(c["ok"].astype(str) == "True", "no error", "had an error")
        g = c.groupby(["reporter", "kind", "result"]).size().reset_index(name="reports")
        fig = px.sunburst(g, path=["reporter", "kind", "result"], values="reports", color="result",
                          color_discrete_map={"no error": ui.OK, "had an error": ui.BAD, "(?)": ui.VIOLET})
        fig.update_layout(title=f"Reports checked on this computer – {S.source_label(ds, short=True)}", height=420)
        C.plot(fig, key="an_sun")
        ui.cap("Inner ring: who wrote the report. Middle: alarm type. Outer: result of the automatic fact check. "
               "Click a segment to zoom in.")
        _data(chk, f"report_checks_{ds}.csv", "an_dl_rep")


# --------------------------------------------------------------------------------------------- 7. edge

def _edge(ds: str) -> None:
    e = S.snapshot()["edge"]
    _finding(f"The complete decision logic runs on an Arduino Uno (32 KB program memory, 2 KB RAM). Over "
             f"{e['uno_minutes']:,} minutes of real data its decisions were identical to the laptop's, at "
             f"{e['us_per_minute'] / 1000:.1f} ms per minute.")
    g1, g2, g3 = st.columns(3)
    with g1:
        C.gauge(e["flash_bytes"] / 1024, 0, 32, "program memory used", key="an_g_flash", unit=" KB",
                stops=[(20, ui.OK), (28, ui.WARN), (32, ui.BAD)], height=250)
    with g2:
        C.gauge(e["ram_bytes"], 0, 2048, "RAM used", key="an_g_ram", unit=" B",
                stops=[(1400, ui.OK), (1800, ui.WARN), (2048, ui.BAD)], height=250, decimals=0)
    with g3:
        C.gauge(e["us_per_minute"] / 1000, 0, 100, "time per minute of data", key="an_g_time", unit=" ms",
                stops=[(50, ui.OK), (80, ui.WARN), (100, ui.BAD)], height=250)
    ui.cap(f"The model has {e['weights']} weights – about {e['bilstm_weights'] // e['weights']} times fewer than "
           "version 1's BiLSTM forecaster.")
    df = S.read_csv("s8_edge", ds, "edge_results.csv")
    if df is None:
        _missing("the edge device")
        return
    d = df.copy()
    d["System"] = d["system"].str.replace("S5 laptop pipeline (stacking (learned))", "Laptop pipeline", regex=False)
    a, b = st.columns([1.3, 1])
    with a:
        long = d.melt(id_vars="System", value_vars=["real_caught", "contaminant_caught", "fault_correct"],
                      var_name="Measure", value_name="Count")
        long["Measure"] = long["Measure"].map({"real_caught": "Real events caught", "contaminant_caught":
                                               "Contaminations caught", "fault_correct": "Faults to maintenance"})
        fig = px.bar(long, y="System", x="Count", color="Measure", orientation="h", barmode="group",
                     color_discrete_sequence=[ui.ACCENT, ui.CYAN, ui.WARN])
        fig.update_layout(title="Edge configurations compared with the laptop pipeline", height=470,
                          yaxis=dict(autorange="reversed", tickfont=dict(size=10)), legend=dict(orientation="h", y=-0.15))
        C.plot(fig, key="an_edge_bar")
    with b:
        fig = go.Figure(go.Bar(x=d["water_false_alarms_per_day"], y=d["System"], orientation="h",
                               marker=dict(color=d["water_false_alarms_per_day"], colorscale="RdYlGn_r", cmin=0, cmax=2.8),
                               text=[f"{v:.2f}" for v in d["water_false_alarms_per_day"]], textposition="outside"))
        fig.update_layout(title="False water alarms per day", height=470, yaxis=dict(autorange="reversed",
                          tickfont=dict(size=10)), xaxis=dict(range=[0, 3.2]))
        C.plot(fig, key="an_edge_fa")
    ui.cap("int8 = weights stored as 8-bit integers to fit the Uno; float32 = full precision. 'Edge triage' adds the "
           "sensor-fault checks on the board.")
    _data(d.drop(columns=["System"]), f"edge_{ds}.csv", "an_dl_edge")
