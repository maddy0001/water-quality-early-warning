"""Control room: a real station replayed minute by minute through the edge logic, with you as the operator.

Layout rules that keep the page calm while it updates (Segment 9.1):
  * every chart has a fixed key and size, so it is updated in place instead of being rebuilt;
  * the question box is always there (empty when there is no question), so nothing jumps;
  * the slow parts (incident report, download) live outside the live area and only change on a click.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ewsv2.app import services as S
from ewsv2.app import ui
from ewsv2.app.views.common import base, board

WINDOW = 720           # minutes shown in the charts (12 hours)
SPEEDS = [1, 5, 15, 60, 240]
MAX_JUMP = 30000       # "Next alarm" gives up after this many minutes (about 3 weeks)
CHART = {"displayModeBar": False}


def _state() -> S.ReplayState | None:
    return st.session_state.get("cr_state")


def _restart(source: str) -> None:
    st.session_state["cr_state"] = S.ReplayState(**base(source))
    st.session_state["cr_reports"] = {}
    st.session_state["cr_play"] = False
    b = board(connect=False)
    if st.session_state.get("cr_uno") and b is not None and b.connected:
        b.configure_edge(True, True)


def _board_if(uno: bool):
    return board(connect=False) if uno else None


def render() -> None:
    ui.hero("Watch a real station", "Real minute-by-minute data from a drinking-water utility, replayed fast. "
            "The system watches the sensors, raises alarms, and asks you – the operator – whether each alarm was real.",
            eyebrow="Demo 1 · real data")
    ui.guide("Try it in 3 clicks", [
        "Press <b>⏭ Next alarm</b>. The replay runs by itself until an alarm has happened and finished.",
        "Read the coloured banner and the <b>Why?</b> chart: 🚨 red = the water changed, 🔧 yellow = a sensor looks "
        "broken. Then answer the question: <b>Real event</b> or <b>False alarm</b>? The <b>answer key</b> next to "
        "the buttons tells you what really happened.",
        "Say <b>False alarm</b> and watch the dotted <b>alarm line</b> step up – the system just learned from you "
        "and became a little less sensitive. Press ⏭ again for the next one.",
    ])
    cfg = S.config()
    data = S.data_status()
    sources = [s for s in cfg["dashboard"]["sources"] if s == "simulated" or data.get(s)] or ["simulated"]
    b = board()
    uno_ok = b is not None and b.connected

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.5, 0.9, 1.0, 0.8], vertical_alignment="bottom")
    source = c1.selectbox("Station", sources, key="cr_source_pick",
                          help="gecco2018 / gecco2019: real utility data (needs the data files). "
                               "simulated: made by the computer, always works.")
    verdict_mode = c2.segmented_control("Who answers the alarms", ["you", "auto"], default="you", key="cr_mode",
                                        format_func=lambda x: "You (operator)" if x == "you" else "Automatic (labels)",
                                        help="You: the replay waits for your answer after each alarm. "
                                             "Automatic: the utility's labels answer, the replay never waits.")
    verdict_mode = verdict_mode or "you"
    with st.expander("⚙ Settings – speed, Arduino, answer key, single steps"):
        s1, s2, s3, s4 = st.columns([1.4, 1.2, 1.2, 1], vertical_alignment="bottom")
        speed = s1.select_slider("Play speed", options=SPEEDS, value=5, key="cr_speed",
                                 format_func=lambda v: f"{v} min/s",
                                 help="How many minutes of station time pass per second while playing.")
        uno = s2.toggle("Also run on the Arduino", value=False, disabled=not uno_ok, key="cr_uno_toggle",
                        help="The Uno computes every minute too and the page checks it gives exactly the same answer. "
                             + ("" if uno_ok else f"Not available now: {b.message if b else 'board switched off'}"))
        reveal = s3.toggle("Show the answer key", value=True, key="cr_reveal",
                           help="What really happened (real event, injected contaminant or sensor fault). "
                                "Only you see it – the system never does.")
        step1 = s4.button("Step 1 minute", width="stretch")
    if uno != st.session_state.get("cr_uno", False):
        st.session_state["cr_uno"] = uno
        st.session_state.pop("cr_state", None)              # both start fresh so their states match
    if _state() is None or st.session_state.get("cr_source") != source:
        st.session_state["cr_source"] = source
        _restart(source)
    if uno:
        speed = min(speed, 15)

    playing = st.session_state.get("cr_play", False)
    if c3.button("⏸ Pause" if playing else "▶ Play", width="stretch", type="secondary" if playing else "primary",
                 help="Run the replay continuously."):
        st.session_state["cr_play"] = not playing
        st.rerun()
    if c4.button("⏭ Next alarm", width="stretch", type="primary" if not playing else "secondary",
                 help="Jump ahead until the next alarm has happened and ended."):
        st.session_state["cr_play"] = False
        with st.spinner("Looking for the next alarm …"):
            st.session_state["cr_jump"] = _next_alarm(verdict_mode, uno)
    if c5.button("↺ Restart", width="stretch", help="Back to the start of the test period."):
        _restart(source)
    if step1:
        _advance(1, verdict_mode, uno)

    run_every = 1.0 if st.session_state.get("cr_play") else None

    @st.fragment(run_every=run_every)
    def live() -> None:
        state = _state()
        if state is None:
            return
        if st.session_state.get("cr_play") and state.pending is None and not state.done:
            _advance(speed, verdict_mode, uno)
        _view(state, reveal, uno, verdict_mode, speed)

    live()
    _below(_state())


# --------------------------------------------------------------------------------------------- moving in time

def _advance(n: int, mode: str, uno: bool) -> None:
    state = _state()
    if state is None:
        return
    b = _board_if(uno)
    for _ in range(n):
        if state.step(mode, b) is None:
            break


def _next_alarm(mode: str, uno: bool) -> str:
    """Run until an alarm (water alarm or sensor alert) has happened and ended. Returns a short note."""
    state = _state()
    if state is None or state.pending is not None:
        return ""
    b = _board_if(uno)
    asks0, n0 = state.asks, len(state.episodes)
    limit = 3000 if uno else MAX_JUMP
    for k in range(limit):
        row = state.step(mode, b)
        if row is None:                                   # waiting for the operator, or the data ended
            break
        if state.asks > asks0:                            # an alarm cycle ended (auto mode answered it)
            break
        new = state.episodes[n0:]
        if new and all(e["kind"] == "sensor" for e in new) and row["i"] - new[-1]["end"] > 5:
            break                                         # a sensor alert happened and ended
    else:
        return f"No alarm in the next {limit:,} minutes – press Next alarm again to keep looking."
    if state.done and state.pending is None:
        return "The end of the test data has been reached. Press Restart to begin again."
    return ""


# --------------------------------------------------------------------------------------------- the live area

def _view(state: S.ReplayState, reveal: bool, uno: bool, mode: str, speed: int) -> None:
    rows = state.rows[-WINDOW:]
    if st.session_state.get("cr_play"):
        doing = "▶ playing" if state.pending is None else "⏸ waiting for your answer"
    else:
        doing = "⏸ paused" if state.pending is None else "waiting for your answer"
    if state.done:
        doing = "finished"
    now = rows[-1]["time"] if rows else state.frame.index[state.start]
    ui.status([("Station", state.name), ("Station clock", f"{now:%d %b %Y · %H:%M}"),
               ("Replayed", f"{state.pos:,} min ({state.pos / 1440:.1f} days)"), ("Status", doing),
               ("Speed", f"{speed} min/s")])
    note = st.session_state.pop("cr_jump", "")
    if note:
        st.info(note)

    if not rows:
        ui.banner("idle", "Ready", "Press ⏭ Next alarm (fastest) or ▶ Play. The replay starts at the first minute of "
                  f"the official test period ({state.frame.index[state.start]:%d %b %Y %H:%M}).")
    else:
        _banner(state, rows[-1])
    _kpis(state, rows, reveal, uno)
    _question(state, reveal, uno, mode)

    left, right = st.columns([2.2, 1])
    with left:
        ui.explain("<b>Alarm score</b> (blue) = how unusual the water looks right now. <b>Alarm line</b> (dotted) = "
                   "the score needed for an alarm. Score above the line → alarm. Each “false alarm” answer lifts the "
                   "line a little; it slowly comes back down over the next day."
                   + (" Shaded = the answer key (red: real water event, yellow: broken sensor)." if reveal else ""))
        st.plotly_chart(_detector_chart(rows, reveal), width="stretch", config=CHART, key="cr_det")
        with st.expander(f"📈 See all {len(state.sensors)} sensors (raw readings, last 12 hours)"):
            ui.explain("Each small chart is one sensor. ◆ = this sensor failed a health check (stuck, dropped out or "
                       "spiking) – its reading is not trusted.")
            st.plotly_chart(_sensor_chart(rows, state), width="stretch", config=CHART, key="cr_sens")
    with right:
        ui.explain("<b>Why?</b> How far each sensor moved in 15 minutes, compared with its largest normal move. "
                   "A red bar (≥ 1×) is beyond anything seen in normal water.")
        st.plotly_chart(_move_chart(rows[-1] if rows else None, state.sensors), width="stretch", config=CHART,
                        key="cr_move")
        if reveal:
            truth = state.truth[rows[-1]["i"]] if rows else ""
            ui.card("🔑 Answer key – what is really happening now", f'<span class="muted">{ui.esc(truth or "normal water, healthy sensors")}</span>')
        _episode_log(state, reveal)


def _banner(state: S.ReplayState, last: dict) -> None:
    names = state.sensors
    flagged = [names[j] for j in range(len(names)) if last["mask"] >> j & 1]
    if last["water"]:
        top = names[last["top"]]
        ui.banner("water", "Water alarm – the water changed",
                  f"{S.SENSOR_NAMES.get(top, top)} moved {last['m'][last['top']]:.1f}× its largest normal 15-minute "
                  "move – the water itself changed"
                  + (f" (a sensor alert is on at the same time: {', '.join(flagged)})" if flagged else "") + ".")
    elif last["sensor"]:
        ui.banner("sensor", "Sensor problem – check " + ", ".join(flagged),
                  "This sensor is behaving like a broken sensor, and the other sensors say the water is normal. "
                  "This is a maintenance job, not a water alarm.")
    elif last["alarm"]:
        ui.banner("sensor", "Alarm cancelled – only a suspect sensor moved",
                  "The score crossed the line, but the movement came from a sensor that failed its health check.")
    else:
        ui.banner("normal", "All normal", f"All {len(names)} sensors are healthy and the water looks normal. "
                  "Nothing to do.")


def _kpis(state: S.ReplayState, rows: list, reveal: bool, uno: bool) -> None:
    eps = state.episodes
    water = [e for e in eps if e["kind"] == "water"]
    sensor = [e for e in eps if e["kind"] == "sensor"]
    real = sum(1 for e in water if e["truth"])
    theta0 = float(state.spec["theta0"])
    theta = float(state.engine.theta)                 # the line now (includes your last answer)
    lifted = theta - theta0 if np.isfinite(theta) else 0.0
    right = sum(a["right"] for a in state.answers)
    ui.kpi_row([
        ("🚨 Water alarms", str(len(water)), (f"{real} real · {len(water) - real} false (answer key)" if reveal
                                              else "red banner")),
        ("🔧 Sensor alerts", str(len(sensor)), "yellow banner · sent to maintenance"),
        ("🟣 Your answers", f"{right}/{len(state.answers)}" if state.answers else "–",
         "matching the answer key" if state.answers else "none yet"),
        ("📏 Alarm line", "–" if not np.isfinite(theta) else f"{theta:+.2f}",
         "at its starting level" if abs(lifted) < 0.05 else
         (f"{lifted:+.2f} vs start · less sensitive" if lifted > 0 else f"{lifted:+.2f} vs start · more sensitive")),
    ] + ([("Arduino check", f"{state.uno_same:,}/{state.uno_checked:,}",
           state.uno_note or "minutes where the Uno = the laptop")] if uno else []))


def _question(state: S.ReplayState, reveal: bool, uno: bool, mode: str) -> None:
    """The question box. Always the same size, so the page below does not jump when a question appears."""
    p = state.pending
    with st.container(border=True, height=150, key="cr_qbox_ask" if p is not None else "cr_qbox"):
        if p is None:
            if mode == "auto":
                st.markdown("**Question box** · Automatic mode: the utility's labels answer every alarm, so the "
                            "replay never stops to ask. Switch *Who answers the alarms* to **You** to answer yourself.")
            else:
                last = state.answers[-1] if state.answers else None
                st.markdown("**Question box** · No question right now. When an alarm ends, the replay pauses and asks "
                            "you here whether it was real.")
                if last:
                    tick = "✅ matched the answer key" if last["right"] else "❌ did not match the answer key"
                    lift = last.get("lift", 0.0)
                    eff = (f"The alarm line went up by {lift:.2f} – the system is now a little less sensitive."
                           if lift > 0 else "The alarm line stays where it was – the system keeps its sensitivity.")
                    st.caption(f"Your last answer: **{last['said']}** for the alarm at "
                               f"{state.frame.index[last['start']]:%d %b %H:%M} – {tick}. {eff}")
            return
        a, z = p["ep_start"], p["last_on"]
        st.markdown(f"**❓ The alarm from {state.frame.index[a]:%d %b %H:%M} to {state.frame.index[z]:%H:%M} has ended. "
                    "Was it a real water problem?**")
        q1, q2, q3 = st.columns([1, 1, 2.6], vertical_alignment="center")
        q1.button("✅ Real event", width="stretch", key="cr_real", on_click=_answer, args=(False, uno))
        q2.button("❌ False alarm", width="stretch", key="cr_false", on_click=_answer, args=(True, uno))
        if reveal:
            what = p["truth"] or "nothing happened – normal water"
            q3.markdown(f'{ui.pill("ANSWER KEY", "info")} correct answer: <b>{p["key"].upper()}</b><br>'
                        f'<span class="muted">what really happened: {ui.esc(what)}</span>', unsafe_allow_html=True)
        else:
            q3.caption("Answer key hidden (Settings → Show the answer key).")


def _answer(false_alarm: bool, uno: bool) -> None:
    state = _state()
    if state is not None:
        state.answer(false_alarm, _board_if(uno))


# --------------------------------------------------------------------------------------------- charts

def _detector_chart(rows, reveal) -> go.Figure:
    t = [r["time"] for r in rows]
    fig = go.Figure()
    if reveal and rows:
        cls = np.array([r["cls"] for r in rows])
        for code, color, name in ((1, "rgba(248,113,113,.18)", "real water event (key)"),
                                  (2, "rgba(251,191,36,.18)", "broken sensor (key)")):
            y = np.where(cls == code, 16, None)
            fig.add_trace(go.Scatter(x=t, y=y, fill="tozeroy", mode="none", fillcolor=color, name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=[r["logit"] for r in rows], name="alarm score", line=dict(width=1.3, color=ui.ACCENT)))
    fig.add_trace(go.Scatter(x=t, y=[r["theta"] for r in rows], name="alarm line",
                             line=dict(width=2, color=ui.INK, dash="dot")))
    wt = [r["time"] for r in rows if r["water"]]
    st_ = [r["time"] for r in rows if r["sensor"]]
    fig.add_trace(go.Scatter(x=wt, y=[14] * len(wt), mode="markers", name="water alarm", marker=dict(color=ui.BAD, size=6)))
    fig.add_trace(go.Scatter(x=st_, y=[-9] * len(st_), mode="markers", name="sensor alert",
                             marker=dict(color=ui.WARN, size=6, symbol="diamond")))
    fig.update_layout(height=300, title="Alarm score and alarm line (last 12 hours)",
                      yaxis=dict(range=[-11, 16], title="score"),
                      legend=dict(orientation="h", y=-0.18, x=0))
    return ui.steady(fig)


def _sensor_chart(rows, state) -> go.Figure:
    k = len(state.sensors)
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, vertical_spacing=0.09, horizontal_spacing=0.05,
                        subplot_titles=[f"{s} · {S.SENSOR_NAMES.get(s, s)}" for s in state.sensors])
    f = state.frame.iloc[rows[0]["i"]:rows[-1]["i"] + 1] if rows else state.frame.iloc[0:0]
    for j, s in enumerate(state.sensors):
        rr, cc = j // ncol + 1, j % ncol + 1
        fig.add_trace(go.Scatter(x=f.index, y=f[s], line=dict(width=1, color=ui.SERIES[j % len(ui.SERIES)]),
                                 showlegend=False, name=s), row=rr, col=cc)
        flagged = [r["time"] for r in rows if r["mask"] >> j & 1]
        fig.add_trace(go.Scatter(x=flagged, y=f[s].reindex(flagged) if flagged else [], mode="markers",
                                 marker=dict(color=ui.WARN, size=5, symbol="diamond"),
                                 showlegend=False, name=f"{s} flagged"), row=rr, col=cc)
    fig.update_layout(height=120 * nrow + 60, title="Sensors (◆ = failed a health check)",
                      margin=dict(l=10, r=10, t=60, b=10))
    fig.update_annotations(font_size=11)
    fig.update_xaxes(nticks=4, tickangle=0, tickformat="%H:%M")
    return ui.steady(fig)


def _move_chart(last, names) -> go.Figure:
    m = last["m"] if last is not None else np.zeros(len(names))
    colors = [ui.BAD if v >= 1 else ui.ACCENT for v in m]
    fig = go.Figure(go.Bar(x=m, y=list(names), orientation="h", marker_color=colors,
                           text=[f"{v:.1f}×" for v in m], textposition="outside", cliponaxis=False))
    fig.add_vline(x=1, line=dict(color=ui.WARN, dash="dash"))
    top = float(np.max(m)) if len(m) else 0.0
    xmax = 2.5 if top < 2 else (6 if top < 5 else (12 if top < 10 else top * 1.2))   # few, stable scales
    fig.update_layout(height=300, title="Why? Move vs normal",
                      xaxis=dict(range=[0, xmax]), yaxis=dict(autorange="reversed"))
    return ui.steady(fig)


def _episode_log(state, reveal) -> None:
    eps = state.episodes[-6:][::-1]
    if not eps:
        ui.card("Alarm log", '<span class="muted">No alarms yet. Press ⏭ Next alarm.</span>')
        return
    html = ""
    for e in eps:
        kind = ui.pill("WATER", "bad") if e["kind"] == "water" else ui.pill("SENSOR", "warn")
        sens = ", ".join(s for j, s in enumerate(state.sensors) if e["mask"] >> j & 1) if e["kind"] == "sensor" \
            else state.sensors[int(np.argmax(e["peak_m"]))]
        verdict = ""
        if e["verdict"]:
            said = "real event" if e["verdict"] == "confirmed" else "false alarm"
            verdict = " " + ui.pill(f"answer: {said}", "ok" if e["verdict"] == "confirmed" else "info")
        truth = ""
        if reveal:
            key = e["truth"] or ("normal water" if e["kind"] == "water" else "no fault in the key")
            truth = f'<br><span class="muted">key: {ui.esc(key)}</span>'
        html += (f'<div class="logrow">{kind}{verdict} <b>{state.frame.index[e["start"]]:%d %b %H:%M}</b> · '
                 f'{e["end"] - e["start"] + 1} min · {ui.esc(sens)}{truth}</div>')
    ui.card("Alarm log (latest first)", html)


# --------------------------------------------------------------------------------------------- below the live area

def _log_frame(state) -> pd.DataFrame:
    out = []
    for e in state.episodes:
        out.append({"start": state.frame.index[e["start"]], "end": state.frame.index[e["end"]],
                    "minutes": e["end"] - e["start"] + 1, "kind": e["kind"],
                    "sensors": ", ".join(s for j, s in enumerate(state.sensors) if e["mask"] >> j & 1)
                    or state.sensors[int(np.argmax(e["peak_m"]))],
                    "operator_verdict": e["verdict"], "truth": e["truth"] or "normal water"})
    return pd.DataFrame(out)


def _below(state) -> None:
    """Incident report + download. Outside the live area: it changes only when you click something."""
    if state is None:
        return
    st.markdown("### 📝 Incident report")
    eps = state.episodes
    if not eps:
        st.caption("After the first alarm you can write an incident report for it here and download the alarm log.")
        return
    if st.session_state.get("cr_play"):
        st.caption("Pause the replay to write an incident report (the list of alarms is updated when you pause).")
        return
    ui.explain("Pick an alarm. The report is written from the alarm record only. With the local LLM on, every number "
               "and sensor name in its text is checked against the record; a report with any mistake is replaced by "
               "the template report.")
    cfg = S.config()
    labels = {i: f"{state.frame.index[e['start']]:%d %b %H:%M} · {'water alarm' if e['kind'] == 'water' else 'sensor alert'}"
              for i, e in enumerate(eps)}
    c1, c2, c3 = st.columns([2, 1.2, 1.2], vertical_alignment="bottom")
    pick = c1.selectbox("Alarm", list(labels)[::-1], format_func=labels.get)
    use_llm = c2.toggle("Let the AI write it (Ollama)", value=False,
                        help="Off: the template report (always works). On: the local AI writes it and every fact is "
                             "checked; any mistake → the template is shown instead.")
    c3.download_button("⬇ Alarm log (CSV)", _log_frame(state).to_csv(index=False).encode("utf-8"),
                       file_name=f"alarm_log_{state.name}.csv", mime="text/csv", width="stretch")
    rec = S.episode_record(state, eps[pick], f"GECCO station ({state.name})")
    key = (pick, eps[pick]["end"], use_llm)
    cache = st.session_state.setdefault("cr_reports", {})
    if key not in cache:
        with st.spinner("Writing the report …"):
            cache[key] = S.write_report(rec, state.sensors, use_llm, cfg["llm"]["models"][0], cfg["llm"]["host"])
    rep = cache[key]
    a, b = st.columns([1.4, 1])
    with a:
        shown = rep["llm"] if rep["shown"] == "llm" else rep["template"]
        st.markdown(f'<div class="report">{ui.esc(shown)}</div>', unsafe_allow_html=True)
    with b:
        if rep["shown"] == "llm":
            st.markdown(ui.pill("AI report · every fact checked", "violet"), unsafe_allow_html=True)
            st.caption(f"{rep['seconds']:.1f} s on this computer")
        else:
            st.markdown(ui.pill("template report", "info"), unsafe_allow_html=True)
            if rep["why"]:
                st.caption(rep["why"])
        if rep["errors"]:
            st.dataframe(pd.DataFrame(rep["errors"]), hide_index=True, width="stretch")
        with st.expander("The facts the report may use (the alarm record)"):
            st.json(rec, expanded=False)
