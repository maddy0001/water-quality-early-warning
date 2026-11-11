"""System Guide: how the system works, for someone seeing it for the first time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ewsv2.app import charts as C
from ewsv2.app import nav
from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    ui.header("System Guide",
              "How the early warning system works – the problem it solves, the decision process, the data, the "
              "hardware, the report writer and the terms used throughout this dashboard.",
              kicker="Reference", color="slate")
    tabs = st.tabs(["The problem", "Decision process", "Data", "Hardware", "Operator review", "Incident reports",
                    "Reliability", "Terms"])
    with tabs[0]:
        _problem()
    with tabs[1]:
        _process()
    with tabs[2]:
        _data()
    with tabs[3]:
        _hardware()
    with tabs[4]:
        _review()
    with tabs[5]:
        _reports()
    with tabs[6]:
        _reliability()
    with tabs[7]:
        _terms()


def _problem() -> None:
    ui.section("Two very different reasons for an unusual reading")
    st.markdown("A water utility monitors its network with stations that measure pH, chlorine, conductivity, "
                "turbidity, temperature and flow every minute. When a reading suddenly changes, there are two "
                "possible causes, and they require opposite responses:")
    a, b = st.columns(2)
    a.markdown(ui.component("H2O", "The water changed", "water-quality alarm", ui.BAD, [
        ("Typical pattern", "Several related sensors change together in a way never seen in normal operation – for "
                            "example pH and conductivity after a contamination."),
        ("Required response", "Alert the operator immediately; the public may be affected."),
    ]), unsafe_allow_html=True)
    b.markdown(ui.component("FLT", "A sensor failed", "sensor-fault alert", ui.WARN, [
        ("Typical pattern", "One sensor behaves on its own: stuck at a value, dropping towards zero or producing a "
                            "single spike, while related sensors show normal water."),
        ("Required response", "Send a technician. Do not raise a public water alarm."),
    ]), unsafe_allow_html=True)
    ui.section("What version 2 adds to version 1")
    st.dataframe(pd.DataFrame([
        ("Testing", "Shuffled minutes (optimistic)", "Trained on earlier months, tested on later months"),
        ("Data", "Three separate public datasets", "Real minute-by-minute utility data from one station"),
        ("Sensor faults", "Treated as bad water", "Recognised and routed to maintenance"),
        ("Alarm threshold", "Fixed", "Adapts to keep about one false alarm per day; learns from operator reviews"),
        ("Explanations", "TreeSHAP, not verified", "Named sensor and size of change, verified by a removal test"),
        ("Reports", "None", "Generated per alarm, every fact checked automatically"),
        ("Hardware", "Laptop", "Also runs on an Arduino Uno with identical decisions"),
    ], columns=["Aspect", "Version 1", "Version 2"]), hide_index=True, width="stretch")


def _process() -> None:
    ui.section("Every minute, five stages")
    ui.pipeline([
        ("Acquire", "One value per sensor per minute is received from the station."),
        ("Validate sensors", "Each sensor is checked for being stuck (no change for too long), dropping out (falling "
                             "towards zero) or spiking (one isolated jump). A failing sensor is excluded and a "
                             "sensor-fault alert is raised."),
        ("Score the water", "For each healthy sensor the change over the last 15 and 60 minutes is compared with its "
                            "normal behaviour. A compact neural network (625 weights) turns these into one "
                            "<b>anomaly score</b>."),
        ("Decide", "If the anomaly score exceeds the <b>alarm threshold</b>, a water-quality alarm is raised. The "
                   "threshold adapts so that about one false alarm per day occurs."),
        ("Review and report", "When the alarm ends, the operator confirms it or marks it as false; a false alarm "
                              "raises the threshold slightly. An incident report is generated and fact-checked."),
    ])
    fig = C.sankey(
        ["pH", "Chlorine", "Conductivity", "Turbidity", "Temperature", "Flow", "Sensor validation", "Sensor validation",
         "Anomaly score", "Anomaly score", "Decision", "Decision", "Water-quality alarm", "Water-quality alarm"],
        ["Sensor validation", "Sensor validation", "Sensor validation", "Sensor validation", "Sensor validation",
         "Sensor validation", "Anomaly score", "Sensor-fault alert", "Decision", "Normal operation",
         "Water-quality alarm", "Normal operation", "Operator review", "Incident report"],
        [1, 1, 1, 1, 1, 1, 5, 1, 3, 2, 2, 1, 1, 1], "Information flow through the system", height=420,
        node_colors={"Sensor-fault alert": ui.WARN, "Water-quality alarm": ui.BAD, "Normal operation": ui.OK,
                     "Operator review": ui.VIOLET, "Incident report": "#c026d3", "Anomaly score": ui.ACCENT,
                     "Decision": "#0f172a", "Sensor validation": ui.CYAN})
    C.plot(fig, key="gd_flow")
    t = nav.PAGES.get("control-room")
    if t is not None:
        st.page_link(t, label="See the five stages on real data in the Operations Console",
                     icon=":material/monitor_heart:")


def _data() -> None:
    ui.section("Data sources available in the dashboard")
    st.dataframe(pd.DataFrame([
        (S.source_label("gecco2018"), "Real", "Public GECCO 2018 dataset of a German drinking-water utility "
         "(Thüringer Fernwasserversorgung): temperature, two chlorine sensors, pH, redox, conductivity, turbidity, "
         "two flow meters; one row per minute; real events labelled by the utility."),
        (S.source_label("gecco2019"), "Real", "Public GECCO 2019 dataset from the same utility: temperature, pH, "
         "conductivity, turbidity, organics (SAC) and flow."),
        (S.source_label("simulated"), "Generated", "Nine sensors generated by the computer with realistic noise, "
         "daily cycles, events and gaps. Always available, needs no files."),
    ], columns=["Data source", "Type", "Description"]), hide_index=True, width="stretch")
    ui.section("Recorded outcome")
    st.markdown("To measure whether the system is right, every minute carries a **recorded outcome** that the "
                "detection system never sees:\n"
                "- **Real water event** – marked by the utility in the original data.\n"
                "- **Injected contamination** – a contamination added to the real data at a known time "
                "(e.g. a chemical or biological signature over several sensors).\n"
                "- **Injected sensor fault** – a fault added to one sensor at a known time (stuck, drop-out, spike).\n"
                "- **No event** – normal water and healthy sensors.\n\n"
                "In the Operations Console the recorded outcome can be shown next to each alarm, so you can see "
                "whether the alarm – and your own review – was correct.")


def _hardware() -> None:
    ui.section("The physical demonstration station",
               "A low-cost station built for live demonstrations. Every component is optional; if one is missing "
               "the computer takes over its role.")
    cards = [
        ui.component("UNO", "Arduino Uno", "microcontroller · USB connection to the laptop", ui.ACCENT, [
            ("Role", "The station computer. Reads the three sensors every two seconds and sends the values to the "
                     "laptop. It can also run the complete decision logic on recorded utility data to prove that "
                     "it fits on inexpensive hardware."),
            ("Capacity", "32 KB program memory, 2 KB RAM. The decision logic uses about 19 KB and 1.1 KB."),
            ("Verified", "10,080 minutes of real data: identical decisions to the laptop, 25 ms per minute."),
            ("If not connected", "The laptop performs the same logic; the Sensor Lab offers a simulation."),
        ]),
        ui.component("EC", "Conductivity probe", "two-pronged probe with comparator board", ui.CYAN, [
            ("Measures", "How easily current flows through the water between the prongs. Dissolved salts raise it. "
                         "Shown as the conductivity index (0–1023)."),
            ("Role in the demonstration", "Stands in for the utility's conductivity sensor. Salt in the water → "
                                          "water-quality alarm. Probe lifted out → sensor-fault alert."),
            ("Wiring", "<code>AO → A0</code> <code>VCC → A2</code> <code>GND → GND</code>. Powered only during a "
                       "reading to slow corrosion."),
            ("If not connected", "Simulation with the actions Add salt, Remove probe and Reinsert probe."),
        ]),
        ui.component("TCH", "Touch sensor", "capacitive touch pad", ui.VIOLET, [
            ("Measures", "Whether a finger touches the pad."),
            ("Role in the demonstration", "The operator's \"false alarm\" button: tells the system that the last "
                                          "alarm was not real, which raises the alarm limit."),
            ("Wiring", "<code>SIG → D3</code> <code>VCC → D4</code> <code>GND → GND</code>"),
            ("If not connected", "The on-screen Report false alarm button does the same."),
        ]),
        ui.component("T/H", "Temperature and humidity sensor", "DHT11", ui.OK, [
            ("Measures", "Air temperature (°C) and relative humidity (%) around the station."),
            ("Role in the demonstration", "Shows the conditions at the station and is a second monitored device: if "
                                          "it stops answering, a sensor-fault alert is raised. It does not affect "
                                          "water-quality alarms."),
            ("Wiring", "<code>DATA → A1</code> <code>VCC → 5V</code> <code>GND → GND</code>"),
            ("If not connected", "Temperature and humidity are shown as unavailable; nothing else changes."),
        ]),
    ]
    for i in range(0, 4, 2):
        a, b = st.columns(2)
        a.markdown(cards[i], unsafe_allow_html=True)
        b.markdown(cards[i + 1], unsafe_allow_html=True)
        st.write("")


def _review() -> None:
    ui.section("Why the operator reviews alarms")
    st.markdown("Too many false alarms and operators stop reacting. The system therefore promises a false-alarm "
                "rate – about one per day – and keeps it with an **adaptive threshold**:\n"
                "- When an alarm ends, the replay pauses and asks the operator: *Confirm as real event* or "
                "*Mark as false alarm*.\n"
                "- *Mark as false alarm* raises the threshold by 0.5: the system becomes slightly less sensitive.\n"
                "- Over the following day the threshold slowly returns towards its starting value.\n"
                "- *Automatic review* (Operations Console → Advanced options) answers from the recorded outcome "
                "instead, so the replay never pauses.\n\n"
                "The **review accuracy** tile counts how many of your reviews match the recorded outcome.")


def _reports() -> None:
    cfg = S.config()
    ui.section("How incident reports are produced")
    ui.pipeline([
        ("Alarm record", "The system collects the facts of the alarm: type, times, sensors involved, values before "
                         "and during, size of change."),
        ("Writing", f"Either the standard template fills in the facts, or the local AI model "
                    f"(<b>{ui.esc(cfg['llm']['models'][0])}</b>, run by Ollama on this computer, no internet) writes a "
                    "readable report from the same record."),
        ("Fact check", "Every number, time and sensor name in the text is compared with the record."),
        ("Release", "A report with any error is withheld and the template report is used instead."),
    ], colors=[ui.ACCENT, ui.VIOLET, ui.WARN, ui.OK])
    st.markdown("Reports can be downloaded as plain text or as a printable page (use the browser's *Print → Save as "
                "PDF*) from the Operations Console, the Sensor Lab and the Incident Reports page.")


def _reliability() -> None:
    ui.section("Designed to keep working")
    st.markdown("Every external part is checked when the dashboard starts. If one is missing, the system "
                "continues with a replacement:")
    st.dataframe(pd.DataFrame([
        ("Arduino not connected", "The laptop runs the same decision logic; the Sensor Lab uses the simulation."),
        ("A sensor not answering", "Simulation for that sensor, or the value is shown as unavailable."),
        ("AI model (Ollama) not running", "Standard template reports, which are always correct."),
        ("Utility data files missing", "The synthetic station."),
        ("Version 1 dashboard not running", "Its published results are shown on the Version 1 Baseline page."),
        ("Experiment result files missing", "The measured headline results bundled with the software."),
        ("Chart component unavailable", "Equivalent standard charts."),
    ], columns=["Situation", "What the system does"]), hide_index=True, width="stretch")


def _terms() -> None:
    ui.section("Terms used in this dashboard")
    ui.glossary([
        ("Water-quality alarm", "The water itself behaves abnormally. Shown in red."),
        ("Sensor-fault alert", "One sensor behaves like a failing instrument. Maintenance task, shown in amber."),
        ("Anomaly score", "How unusual the water looks in the current minute. Higher means more unusual."),
        ("Alarm threshold", "The anomaly score at which an alarm is raised. Adapts over time and after reviews."),
        ("False alarm", "An alarm raised although the water was normal."),
        ("Recorded outcome", "What actually happened in the data: real event, injected contamination, injected "
                             "fault, or nothing. Never visible to the detection system."),
        ("Review accuracy", "Share of your alarm reviews that match the recorded outcome."),
        ("Event score", "A 0–1 measure combining how many events were detected and how many alarms were correct."),
        ("Time-ordered test", "Training on earlier months and testing on later months, as in real use."),
        ("Change vs normal", "A sensor's change over 15 minutes divided by its largest change in normal operation."),
        ("Conductivity index", "The Sensor Lab probe reading, 0–1023; more dissolved salt gives a higher value."),
        ("Edge device", "The small computer next to the sensors – here the Arduino Uno."),
        ("Ollama", "Free software that runs AI language models locally on the laptop."),
        ("GECCO 2018 / 2019", "Public datasets of a German drinking-water utility used for evaluation."),
    ])
