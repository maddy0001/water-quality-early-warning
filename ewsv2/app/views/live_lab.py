"""Live sensor lab: the probe in a glass of water on the Uno – or a simulated probe when the board is missing."""

from __future__ import annotations

import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import services as S
from ewsv2.app import ui
from ewsv2.app.views.common import board
from ewsv2.live.station import LiveStation, SimProbe


def _lab() -> dict:
    if "lab" not in st.session_state:
        st.session_state["lab"] = {"station": LiveStation(), "sim": SimProbe(), "source": None, "reports": {}}
    return st.session_state["lab"]


def render() -> None:
    ui.hero("Live sensor lab", "The decision logic on a real sensor: the probe in a glass of water. Salt is a "
            "'contaminant', a pulled fork wire is a 'broken sensor', the touch pad is the operator's 'false alarm' button.")
    ui.guide("How to run the live demo", [
        "<b>Uno sensors</b> needs the board on USB with the probe on A0/A2/GND. Without it, pick <b>Simulated probe</b> "
        "– the buttons then play the part of the glass, the salt and the wires.",
        "Put the fork in plain water and wait about 40 seconds: the lab first learns what <b>normal</b> looks like "
        "(<i>calibrating</i>).",
        "Add a pinch of salt and stir (or press <b>🧂 Add salt</b>): the reading jumps → red <b>WATER QUALITY ALARM</b>.",
        "Lift the fork out of the water (or <b>✋ Pull probe out</b>): the reading goes to “dry” → yellow "
        "<b>SENSOR FAULT</b>, not a water alarm. Put it back.",
        "Touch the touch pad (or press <b>❌ False alarm</b>): the alarm limit goes up – the lab becomes less sensitive, "
        "like the control room. Press <b>🔄 Re-calibrate</b> after changing the water.",
    ])
    lab = _lab()
    b = board()
    hw = b is not None and b.connected and b.parts.get("probe", False)
    c1, c2, c3 = st.columns([1.4, 1, 1.6])
    choice = c1.segmented_control("Sensor source", ["board", "sim"], default="board" if hw else "sim",
                                  format_func=lambda x: "Uno sensors" if x == "board" else "Simulated probe")
    if choice == "board" and not hw:
        c3.warning("No probe on the board right now – using the simulated probe. "
                   + (b.message if b is not None and not b.connected else "Check the probe wires (A0, A2, GND)."), icon="⚠️")
        choice = "sim"
    if lab["source"] != choice:
        lab["source"] = choice
        lab["station"] = LiveStation()
        if choice == "board":
            b.start_polling()
        elif b is not None:
            b.stop_polling()
    running = c2.toggle("Running", value=True)
    st.session_state["lab_running"] = running

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    if a1.button("🔄 Re-calibrate", width="stretch", help="Learn 'normal' again (new glass of water)."):
        lab["station"] = LiveStation()
    if a2.button("❌ False alarm", width="stretch", help="Same as touching the touch pad."):
        lab["station"].verdict_false_alarm()
    sim = lab["sim"]
    if choice == "sim":
        if a3.button("🧂 Add salt", width="stretch"):
            sim.add_salt()
        if a4.button("💧 Fresh water", width="stretch"):
            sim.fresh_water()
        if a5.button("✋ Pull probe out", width="stretch"):
            sim.pull_out()
        if a6.button("↩ Put it back", width="stretch"):
            sim.put_back()
    else:
        a3.caption("On the real probe: add salt to the glass, pull a fork wire, touch the pad.")

    @st.fragment(run_every=2.0 if running else None)
    def live() -> None:
        station: LiveStation = lab["station"]
        if st.session_state.get("lab_running", True):
            if lab["source"] == "board" and b is not None:
                r = b.readings[-1] if b.readings else None
                if r is not None and (not station.history or station.history[-1]["t"] < r.t):
                    station.push(r)
                elif not b.connected:
                    ui.banner("sensor", "Board disconnected", b.message + " – switch to the simulated probe or plug it in.")
            else:
                station.push(sim.read())
        _view(station, lab)

    live()


def _checklist(station: LiveStation) -> None:
    kinds = {e.kind for e in station.events}
    ui.checklist([("calibrated", bool(station.unit)), ("water alarm shown", "water" in kinds),
                  ("sensor fault shown", "sensor" in kinds), ("false alarm given", "verdict" in kinds)])


def _view(station: LiveStation, lab: dict) -> None:
    hist = list(station.history)
    if not hist:
        ui.banner("idle", "Waiting for the first reading …")
        return
    last = hist[-1]
    _checklist(station)
    if last["sensor"]:
        open_fault = next((e for e in reversed(station.events) if e.kind == "sensor"), None)
        ui.banner("sensor", "SENSOR FAULT ALERT", open_fault.text.split("– ", 1)[-1] if open_fault else "")
    elif last["water"]:
        ui.banner("water", "WATER QUALITY ALARM", f"The ion index moved {last['z']:+.1f} units (limit {station.limit:.1f}).")
    elif str(last["status"]).startswith("calibrating") or last["status"] == "waiting for water":
        ui.banner("idle", last["status"].capitalize(), "Put the fork in the glass and wait about 40 seconds.")
    else:
        ui.banner("normal", "NORMAL", "The water reading is steady.")
    unit = f"{station.unit:.1f}" if station.unit else "–"
    ui.kpi_row([
        ("Ion index", "–" if last["ion"] is None else f"{last['ion']}", "1023 − probe reading; salt raises it"),
        ("Change", f"{last['z']:+.1f}", (f"× normal noise ({unit} counts); alarm at ±{station.limit:.1f}"
                                         if station.unit else "learning the normal noise …")),
        ("Alarm limit", f"{station.limit:.1f}", "goes up after a false alarm"),
        ("Temperature", "–" if last["temperature"] is None else f"{last['temperature']:.0f} °C", "DHT11 (air)"),
        ("Humidity", "–" if last["humidity"] is None else f"{last['humidity']:.0f} %", "DHT11 (air)"),
    ])
    left, right = st.columns([2.2, 1])
    with left:
        ui.explain("The ion index goes <b>up</b> when the water conducts better (salt). Green band = normal; "
                   "the line leaving the band for 2 readings = water alarm (red dots). Yellow ◆ = sensor fault "
                   "(probe dry or wire pulled).")
        st.plotly_chart(_chart(hist, station), width="stretch", config={"displayModeBar": False}, key="lab_chart")
    with right:
        html = ""
        for e in reversed(station.events[-10:]):
            kind = {"water": ui.pill("WATER", "bad"), "sensor": ui.pill("SENSOR", "warn"),
                    "verdict": ui.pill("OPERATOR", "info"), "info": ui.pill("INFO", "ok")}[e.kind]
            dur = "" if e.end is None else f" · {e.end - e.start:.0f} s"
            html += (f'<div class="logrow">{kind} <b>{time.strftime("%H:%M:%S", time.localtime(e.start))}</b>{dur}<br>'
                     f'<span class="muted">{ui.esc(e.text)}</span></div>')
        ui.card("Events (latest first)", html or '<span class="muted">No events yet.</span>')
    water = [e for e in station.events if e.kind in ("water", "sensor")]
    if water:
        e = water[-1]
        rec = _record(e, station)
        key = (id(e), e.end)
        if key not in lab["reports"]:
            lab["reports"][key] = S.write_report(rec, ["Probe"], False, "", "")
        st.markdown("**Incident report (template)**")
        st.markdown(f'<div class="report">{ui.esc(lab["reports"][key]["template"])}</div>', unsafe_allow_html=True)


def _chart(hist, station) -> go.Figure:
    t = [datetime.fromtimestamp(h["t"]) for h in hist[-300:]]
    h = hist[-300:]
    fig = go.Figure()
    lo = [x["ref"] - x["band"] if x.get("ref") is not None else None for x in h]
    hi = [x["ref"] + x["band"] if x.get("ref") is not None else None for x in h]
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(52,211,153,.18)", name="normal band (outside = alarm)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=[x["ion"] for x in h], name="ion index", line=dict(color=ui.ACCENT, width=2)))
    wt = [ti for ti, x in zip(t, h) if x["water"]]
    fig.add_trace(go.Scatter(x=wt, y=[x["ion"] for x in h if x["water"]], mode="markers", name="water alarm",
                             marker=dict(color=ui.BAD, size=8)))
    vals = [v for v in [x["ion"] for x in h] + lo + hi if v is not None]
    floor = min(vals) if vals else 0
    stt = [ti for ti, x in zip(t, h) if x["sensor"]]
    fig.add_trace(go.Scatter(x=stt, y=[x["ion"] if x["ion"] is not None else floor for x in h if x["sensor"]],
                             mode="markers", name="sensor alert", marker=dict(color=ui.WARN, size=8, symbol="diamond")))
    if vals:                                   # at least 120 counts tall, so normal noise looks like noise
        mid, span = (max(vals) + min(vals)) / 2, max(120.0, (max(vals) - min(vals)) * 1.15)
        fig.update_yaxes(range=[mid - span / 2, mid + span / 2])
    fig.update_xaxes(tickformat="%H:%M:%S", nticks=8, tickangle=0)
    fig.update_layout(height=360, title="Probe in the glass (last 10 minutes)", yaxis_title="ion index",
                      legend=dict(orientation="h", y=-0.2, x=0))
    return ui.steady(fig)


def _record(e, station) -> dict:
    t0 = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.start))
    t1 = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.end or time.time()))
    sens = {"code": "Probe", "description": "probe in the water (ion index)"}
    if e.kind == "water" and e.before is not None and e.now is not None:
        sens.update({"before": round(e.before), "now": round(e.now), "change": round(e.now - e.before)})
    return {"id": "L1", "kind": e.kind, "station": "Live lab", "level": e.level or "LOW",
            "decision": "WATER QUALITY ALARM" if e.kind == "water" else "SENSOR FAULT ALERT",
            "start": t0, "end": t1, "duration_min": max(1, int(((e.end or time.time()) - e.start) // 60) + 1),
            "move_window_min": 15, "sensors_beyond_normal": 1 if e.kind == "water" else None,
            "sensor_alerts_during": [] if e.kind == "water" else ["Probe"], "sensors": [sens]}
