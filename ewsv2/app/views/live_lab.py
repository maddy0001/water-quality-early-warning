"""Sensor Lab: live readings from the Arduino sensors – or a simulation of them when no board is connected."""

from __future__ import annotations

import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import charts as C
from ewsv2.app import export as X
from ewsv2.app import services as S
from ewsv2.app import ui
from ewsv2.app.views.common import board
from ewsv2.live.station import LiveStation, SimProbe


def _lab() -> dict:
    if "lab" not in st.session_state:
        st.session_state["lab"] = {"station": LiveStation(), "sim": SimProbe(), "source": None, "reports": {}}
    return st.session_state["lab"]


def render() -> None:
    ui.header("Sensor Lab",
              "A small physical monitoring station: a conductivity probe in a glass of water, a touch sensor for the "
              "operator and a temperature/humidity sensor, read by the Arduino every two seconds. Contaminate the "
              "water or disconnect the probe and watch the system tell the two situations apart.",
              kicker="Monitor · live sensors", color="emerald")
    lab = _lab()
    b = board()
    hw = b is not None and b.connected and b.parts.get("probe", False)

    with st.container(border=True):
        ui.panel_title("Control panel")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            choice = st.segmented_control("Input", ["board", "sim"], default="board" if hw else "sim", key="lab_src",
                                          format_func=lambda x: "Arduino sensors" if x == "board" else "Simulation")
            choice = choice or ("board" if hw else "sim")
            ui.cap("Arduino sensors: real readings over USB. Simulation: the computer imitates the same sensors, "
                   "for demonstrations without hardware.")
        with c2:
            running = st.toggle("Live updates", value=True, key="lab_running_t")
            ui.cap("Reads a new value every two seconds. Switch off to freeze the screen.")
        if choice == "board" and not hw:
            st.warning("No probe is answering on the Arduino, so the simulation is used. "
                       + (b.message if b is not None and not b.connected else
                          "Check the probe wiring: AO to A0, VCC to A2, GND to GND."), icon=":material/warning:")
            choice = "sim"
        if lab["source"] != choice:
            lab["source"] = choice
            lab["station"] = LiveStation()
            if choice == "board":
                b.start_polling()
            elif b is not None:
                b.stop_polling()
        st.session_state["lab_running"] = running

        k1, k2 = st.columns(2)
        with k1:
            if st.button("Recalibrate", width="stretch", icon=":material/tune:", key="lab_recal"):
                lab["station"] = LiveStation()
            ui.cap("Learns the normal reading again (about 40 seconds). Use after changing the water.")
        with k2:
            if st.button("Report false alarm", width="stretch", icon=":material/cancel:", key="lab_false"):
                lab["station"].verdict_false_alarm()
            ui.cap("Same as touching the touch sensor: tells the system the last alarm was not real, which makes it "
                   "less sensitive.")
        sim = lab["sim"]
        if choice == "sim":
            ui.panel_title("Simulation actions – stand-ins for physical actions")
            a3, a4, a5, a6 = st.columns(4)
            with a3:
                if st.button("Add salt", width="stretch", key="lab_salt"):
                    sim.add_salt()
                ui.cap("Contaminates the water: conductivity rises.")
            with a4:
                if st.button("Replace water", width="stretch", key="lab_fresh"):
                    sim.fresh_water()
                ui.cap("Back to clean water.")
            with a5:
                if st.button("Remove probe", width="stretch", key="lab_out"):
                    sim.pull_out()
                ui.cap("Simulates a broken or disconnected sensor.")
            with a6:
                if st.button("Reinsert probe", width="stretch", key="lab_in"):
                    sim.put_back()
                ui.cap("Puts the probe back into the water.")
        else:
            ui.cap("With the Arduino: add a pinch of salt to the glass, lift the probe out of the water, touch the "
                   "touch sensor. The parts that are answering are listed below.")
            _parts(b)

    @st.fragment(run_every=2.0 if running else None)
    def live() -> None:
        station: LiveStation = lab["station"]
        if st.session_state.get("lab_running", True):
            if lab["source"] == "board" and b is not None:
                r = b.readings[-1] if b.readings else None
                if r is not None and (not station.history or station.history[-1]["t"] < r.t):
                    station.push(r)
                elif not b.connected:
                    ui.state("sensor", "Arduino disconnected", b.message + " Switch to Simulation or reconnect it.")
            else:
                station.push(sim.read())
        _view(station, lab)

    live()


def _parts(b) -> None:
    p = b.parts if (b is not None and b.connected) else {}
    st.markdown(" ".join([
        ui.badge("Arduino connected" if (b is not None and b.connected) else "Arduino not found",
                 "ok" if (b is not None and b.connected) else "warn"),
        ui.badge("Conductivity probe: " + ("answering" if p.get("probe") else "no answer"), "ok" if p.get("probe") else "warn"),
        ui.badge("Touch sensor: " + ("answering" if p.get("touch") else "no answer"), "ok" if p.get("touch") else "warn"),
        ui.badge("Temperature/humidity: " + ("answering" if p.get("dht") else "no answer"), "ok" if p.get("dht") else "warn"),
    ]), unsafe_allow_html=True)


def _view(station: LiveStation, lab: dict) -> None:
    hist = list(station.history)
    kinds = {e.kind for e in station.events}
    ui.steps([
        ("Calibrate", "Probe in clean water; wait ~40 s.", bool(station.unit)),
        ("Contaminate", "Add salt → water-quality alarm.", "water" in kinds),
        ("Break the sensor", "Remove the probe → sensor-fault alert.", "sensor" in kinds),
        ("Review", "Report a false alarm → less sensitive.", "verdict" in kinds),
    ])
    if not hist:
        ui.state("idle", "Waiting for the first reading", "The first value arrives within two seconds.")
        return
    last = hist[-1]
    if last["sensor"]:
        open_fault = next((e for e in reversed(station.events) if e.kind == "sensor"), None)
        ui.state("sensor", "Sensor-fault alert – no water alarm",
                 open_fault.text.split("– ", 1)[-1] if open_fault else "The probe reading is not trustworthy.")
    elif last["water"]:
        ui.state("water", "Water-quality alarm",
                 f"Conductivity changed {abs(last['z']):.1f} times the normal noise (alarm limit "
                 f"{station.limit:.1f}). The water composition changed.")
    elif str(last["status"]).startswith("calibrating") or last["status"] == "waiting for water":
        ui.state("idle", "Calibrating" if str(last["status"]).startswith("calibrating") else "Waiting for water",
                 "Keep the probe in clean water for about 40 seconds while the normal reading is learned.")
    else:
        ui.state("normal", "Normal operation", "The water reading is stable and the sensors are healthy.")

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        ion = last["ion"] if last["ion"] is not None else 0
        C.liquid(ion / 1023, f"{ion}", key="lab_liquid")
        ui.cap("Conductivity index (0–1023). Higher means more dissolved salt.")
    with g2:
        z = abs(last["z"]) if last["z"] is not None else 0
        lim = station.limit
        C.gauge(min(z, lim * 2), 0, lim * 2, "change vs normal", key="lab_z",
                stops=[(lim * 0.6, ui.OK), (lim, ui.WARN), (lim * 2, ui.BAD)], unit="×", height=250)
        ui.cap(f"Current change in units of normal noise. Alarm at {lim:.1f}×; raised by every false-alarm report.")
    with g3:
        tval = last["temperature"]
        C.gauge(tval if tval is not None else 0, 0, 50, "air temperature", key="lab_t", unit=" °C",
                stops=[(18, "#38bdf8"), (32, ui.OK), (50, "#f97316")], height=250, decimals=0)
        ui.cap("Temperature next to the station (DHT11)." + (" No reading." if tval is None else ""))
    with g4:
        hval = last["humidity"]
        C.gauge(hval if hval is not None else 0, 0, 100, "air humidity", key="lab_h", unit=" %",
                stops=[(30, "#f59e0b"), (70, ui.OK), (100, "#0284c7")], height=250, decimals=0)
        ui.cap("Humidity next to the station (DHT11)." + (" No reading." if hval is None else ""))

    left, right = st.columns([2.2, 1])
    with left:
        C.plot(_chart(hist), key="lab_chart", steady=True)
        ui.cap("Green band: the normal range learned during calibration. Leaving it for two readings in a row raises a "
               "water-quality alarm (red). A dry or disconnected probe raises a sensor-fault alert (amber).")
    with right:
        h = ""
        for e in reversed(station.events[-10:]):
            kind = {"water": ui.badge("WATER ALARM", "bad"), "sensor": ui.badge("SENSOR FAULT", "warn"),
                    "verdict": ui.badge("OPERATOR", "violet"), "info": ui.badge("INFO", "ok")}[e.kind]
            dur = "" if e.end is None else f" · {e.end - e.start:.0f} s"
            h += (f'<div class="logrow">{kind} <b>{time.strftime("%H:%M:%S", time.localtime(e.start))}</b>{dur}<br>'
                  f'<span class="muted">{ui.esc(e.text)}</span></div>')
        ui.card("Event log (newest first)", h or '<span class="muted">No events yet.</span>')

    alarms = [e for e in station.events if e.kind in ("water", "sensor")]
    if alarms:
        e = alarms[-1]
        rec = _record(e)
        key = (id(e), e.end)
        if key not in lab["reports"]:
            lab["reports"][key] = S.write_report(rec, ["Probe"], False, "", "")
        text = lab["reports"][key]["template"]
        ui.section("Incident report – latest alarm", "Standard template report, generated from the alarm record.")
        a, b = st.columns([2, 1])
        a.markdown(f'<div class="report">{ui.esc(text)}</div>', unsafe_allow_html=True)
        with b:
            stem = f"lab_incident_{time.strftime('%Y%m%d_%H%M%S', time.localtime(e.start))}"
            st.download_button("Download report (.txt)", X.report_txt(text, rec, "standard template"),
                               file_name=stem + ".txt", mime="text/plain", width="stretch",
                               icon=":material/download:", on_click="ignore", key="lab_dl_txt")
            st.download_button("Download printable report (.html)", X.report_html(text, rec, "standard template"),
                               file_name=stem + ".html", mime="text/html", width="stretch",
                               icon=":material/print:", on_click="ignore", key="lab_dl_html")


def _chart(hist) -> go.Figure:
    h = hist[-300:]
    t = [datetime.fromtimestamp(x["t"]) for x in h]
    lo = [x["ref"] - x["band"] if x.get("ref") is not None else None for x in h]
    hi = [x["ref"] + x["band"] if x.get("ref") is not None else None for x in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(16,185,129,.18)", name="normal range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=[x["ion"] for x in h], name="conductivity index",
                             line=dict(color=ui.CYAN, width=2.4)))
    wt = [ti for ti, x in zip(t, h) if x["water"]]
    fig.add_trace(go.Scatter(x=wt, y=[x["ion"] for x in h if x["water"]], mode="markers", name="water-quality alarm",
                             marker=dict(color=ui.BAD, size=9)))
    vals = [v for v in [x["ion"] for x in h] + lo + hi if v is not None]
    floor = min(vals) if vals else 0
    stt = [ti for ti, x in zip(t, h) if x["sensor"]]
    fig.add_trace(go.Scatter(x=stt, y=[x["ion"] if x["ion"] is not None else floor for x in h if x["sensor"]],
                             mode="markers", name="sensor-fault alert",
                             marker=dict(color=ui.WARN, size=9, symbol="diamond")))
    if vals:                                   # at least 120 counts tall, so normal noise looks like noise
        mid, span = (max(vals) + min(vals)) / 2, max(120.0, (max(vals) - min(vals)) * 1.15)
        fig.update_yaxes(range=[mid - span / 2, mid + span / 2])
    fig.update_xaxes(tickformat="%H:%M:%S", nticks=8, tickangle=0)
    fig.update_layout(height=360, title="Conductivity – last 10 minutes", yaxis_title="conductivity index",
                      legend=dict(orientation="h", y=-0.2, x=0))
    return fig


def _record(e) -> dict:
    t0 = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.start))
    t1 = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.end or time.time()))
    sens = {"code": "Probe", "description": "conductivity probe in the water (index)"}
    if e.kind == "water" and e.before is not None and e.now is not None:
        sens.update({"before": round(e.before), "now": round(e.now), "change": round(e.now - e.before)})
    return {"id": "L1", "kind": e.kind, "station": "Sensor Lab", "level": e.level or "LOW",
            "decision": "WATER QUALITY ALARM" if e.kind == "water" else "SENSOR FAULT ALERT",
            "start": t0, "end": t1, "duration_min": max(1, int(((e.end or time.time()) - e.start) // 60) + 1),
            "move_window_min": 15, "sensors_beyond_normal": 1 if e.kind == "water" else None,
            "sensor_alerts_during": [] if e.kind == "water" else ["Probe"], "sensors": [sens]}
