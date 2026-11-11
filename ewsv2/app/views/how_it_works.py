"""How it works: the whole project explained for someone who has never seen it (Segment 9.2).

No numbers that need a statistics course; every word that matters is explained on the "Words" tab.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ewsv2.app import nav
from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    ui.hero("How it works", "Everything this project does, in plain words: the idea, the five steps, what each "
            "sensor is for, the AI report writer, what version 2 added, and what happens when a part is missing.",
            eyebrow="Explained simply", color="violet")
    tabs = st.tabs(["💡 The idea", "🔢 5 steps", "🔌 The hardware", "🤖 The AI writer", "🆕 What's new in v2",
                    "🛟 If something is missing", "📖 Words"])
    with tabs[0]:
        _idea()
    with tabs[1]:
        _steps()
    with tabs[2]:
        _hardware()
    with tabs[3]:
        _llm()
    with tabs[4]:
        _new()
    with tabs[5]:
        _missing()
    with tabs[6]:
        _words()


# --------------------------------------------------------------------------------------------- the idea

def _idea() -> None:
    st.markdown("### The problem")
    st.markdown(
        '<div class="big">A water utility has measuring stations in its pipes. Each station has several sensors – '
        'pH, chlorine, conductivity, turbidity, temperature, flow – and reports <b>every minute</b>.<br>'
        'When a reading suddenly jumps there are two very different explanations:</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(ui.part_card("🚨", "The water changed", "red = water alarm", ui.BAD, [
        ("Example", "Something got into the pipe: pH and conductivity move together."),
        ("Typical signs", "Several sensors move at the same time, in a way never seen in normal water."),
        ("What to do", "Warn the operator immediately – people drink this water."),
    ]), unsafe_allow_html=True)
    c2.markdown(ui.part_card("🔧", "The sensor broke", "yellow = sensor alert", ui.WARN, [
        ("Example", "The chlorine sensor is stuck on one value, or drops to zero, or spikes once."),
        ("Typical signs", "One sensor moves <b>alone</b>, too fast or not at all, while its partners say "
                          "the water is normal."),
        ("What to do", "Send a technician. Do <b>not</b> alarm the city."),
    ]), unsafe_allow_html=True)
    st.write("")
    st.markdown("### Like a smoke alarm")
    st.markdown(
        '<div class="big">A smoke alarm that beeps because its battery is low is not a fire. Version 1 treated '
        'every strange reading as “bad water”. Version 2 first asks <b>“is the sensor healthy?”</b> and only then '
        '<b>“is the water unusual?”</b>. Too many false alarms and the operator stops listening – so the system '
        'also <b>promises</b> how many false alarms it will give (about one a day) and learns from the operator '
        'every time they say “that was a false alarm”.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("### The data")
    st.markdown(
        "The real data comes from a German drinking-water utility, published for the **GECCO 2018 and 2019** "
        "industrial challenges: one minute per row, nine sensors (2018) or six (2019), with the minutes when real "
        "events happened marked by the utility. On top of that we added **made-up pollution** (salt, chemicals) and "
        "**made-up broken sensors** (stuck, drop-out, spikes) whose times we know – that is the **answer key** the "
        "dashboard can show you. The system itself never sees the answer key.")


# --------------------------------------------------------------------------------------------- 5 steps

def _steps() -> None:
    st.markdown("### What happens every minute")
    ui.steps([
        ("Read", "The 9 sensors of the station send one value each.",
         "pH 8.31 · chlorine 0.14 · conductivity 742 …"),
        ("Check each sensor's health", "Is a sensor <b>stuck</b> (same value too long), <b>dropped out</b> (falls "
         "towards zero) or <b>spiking</b> (one wild value)? Then it is marked 🔧 and its value is not trusted.",
         "Cl_2 stuck for 40 min → 🔧 sensor alert"),
        ("Score the water", "A tiny neural network (625 numbers – small enough for an Arduino) looks at how far each "
         "healthy sensor moved in 15 and 60 minutes and gives an <b>alarm score</b>: how unusual the water looks.",
         "score −4 = very normal · score +6 = very unusual"),
        ("Decide", "If the score is above the <b>alarm line</b> → 🚨 water alarm. The line moves: it goes up a bit "
         "after each false alarm and slowly comes back down, so the system gives about <b>one false alarm a day</b>.",
         "score 6 > line 2 → alarm"),
        ("Explain and report", "Show <b>which sensor moved and how far</b>. When the alarm ends, ask the operator: "
         "real or false? Then write the incident report – a template, or the local AI writer, always fact-checked.",
         "“pH moved 3.1× its largest normal move”"),
    ])
    ui.explain("You can watch all five steps happen on <b>Watch a real station</b>: the yellow/red banner is step 4, "
               "the <i>Why?</i> chart is step 5, the question box is the operator teaching the alarm line.")
    st.markdown("### Where each step runs")
    ui.flow([("Sensors", "station or our probe"), ("Arduino Uno", "reads sensors · can run steps 2–4"),
             ("USB cable", "one text line per minute"), ("Laptop", "runs steps 1–5, shows this dashboard"),
             ("AI writer", "local, no internet, optional")])
    t = nav.PAGES.get("control-room")
    if t is not None:
        st.page_link(t, label="See the five steps on real data", icon=":material/play_circle:")


# --------------------------------------------------------------------------------------------- hardware

def _hardware() -> None:
    st.markdown("### The parts on the table")
    st.markdown("A real utility station costs lakhs. To show the idea live, we built a tiny one for under ₹1,000: "
                "one water sensor, one button for the operator, one room sensor, and an Arduino as the station "
                "computer. **Every part is optional** – unplug anything and the laptop takes over its job.")
    cols = st.columns(2)
    cards = [
        ui.part_card("🧠", "Arduino Uno", "CH340 clone · USB to the laptop (COM7)", ui.ACCENT, [
            ("What it is", "A ₹350 microcontroller: 2 KB of memory, 32 KB for the program. The “station computer”."),
            ("Its job here", "1) Reads the three sensors every 2 seconds and sends them to the laptop.<br>"
                             "2) Runs the <b>same alarm logic as the laptop</b> (steps 2–4) on the real station data "
                             "– to prove it fits on cheap hardware."),
            ("Proof", "One full week of real data (10,080 minutes): the Uno's answer was identical to the laptop's "
                      "<b>every single minute</b>, 25 ms per minute."),
            ("If unplugged", "The laptop runs the same logic itself. Nothing breaks."),
        ]),
        ui.part_card("💧", "Water probe", "HW-080 fork + HW-103 (LM393) board", "#0891b2", [
            ("What it measures", "How easily electricity flows through the water between the two prongs. More "
                                 "dissolved salt = flows more easily. We show it as the <b>ion index</b> (higher = "
                                 "more salt)."),
            ("Its job here", "Plays the part of the utility's conductivity sensor. <b>Salt in the glass</b> = "
                             "pollution → 🚨 water alarm. <b>Probe lifted out</b> = broken sensor → 🔧 sensor alert."),
            ("Wires", "<code>AO → A0</code> · <code>VCC → A2</code> · <code>GND → GND</code>. "
                      "VCC is switched on only while reading, so the prongs corrode slower."),
            ("If missing", "The <b>Simulated probe</b> buttons do the same: Add salt, Pull probe out, Put it back."),
        ]),
        ui.part_card("👆", "Touch pad", "TTP223 capacitive touch", ui.VIOLET, [
            ("What it does", "Gives a signal when a finger touches it – like a button, with no moving parts."),
            ("Its job here", "The <b>operator's “false alarm” button</b>. A touch tells the system “that alarm was "
                             "not real”, so the alarm limit goes up and the system becomes a bit less sensitive."),
            ("Wires", "<code>SIG → D3</code> · <code>VCC → D4</code> · <code>GND → GND</code>"),
            ("If missing", "The on-screen <b>❌ False alarm</b> button does the same."),
        ]),
        ui.part_card("🌡️", "DHT11", "air temperature + humidity", ui.OK, [
            ("What it measures", "Temperature (°C) and humidity (%) of the air around the station."),
            ("Its job here", "Shows the conditions next to the station on the live page, and is a second sensor whose "
                             "<b>health</b> is checked: if it stops answering that is a 🔧 sensor alert – never a "
                             "water alarm. It does <b>not</b> decide water alarms."),
            ("Wires", "<code>DATA → A1</code> · <code>VCC → 5V</code> · <code>GND → GND</code>"),
            ("If missing", "Temperature and humidity show “–”. Everything else works."),
        ]),
    ]
    for i, c in enumerate(cards):
        cols[i % 2].markdown(c, unsafe_allow_html=True)
        if i % 2 == 1:
            cols = st.columns(2)
    st.caption("The RFID reader (RC522) is not used: its pins are not soldered. It is not needed for anything.")
    st.markdown("### Which page uses what")
    st.dataframe(pd.DataFrame([
        {"Page": "Try the sensors", "Uses": "probe + touch pad + DHT11 through the Uno",
         "Who decides": "the laptop (a simple version of steps 2–4 for one sensor)"},
        {"Page": "Watch a real station", "Uses": "real utility data from the laptop; the Uno optionally",
         "Who decides": "the laptop – and, if switched on, the Uno too, checked minute by minute"},
    ]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------------- LLM

def _llm() -> None:
    cfg = S.config()
    st.markdown("### The AI report writer (LLM)")
    st.markdown(
        '<div class="big">After an alarm someone has to write an <b>incident report</b>. A small AI language model '
        f'(<b>{ui.esc(cfg["llm"]["models"][0])}</b>, run by the free program <b>Ollama</b>) writes it on this laptop – '
        'no internet, no cost, no data leaves the computer.</div>', unsafe_allow_html=True)
    ui.steps([
        ("The system decides", "The alarm, its times and the sensors are decided by steps 1–5 – never by the AI.",
         "Water alarm · 00:33–00:47 · pH"),
        ("The AI writes", "It gets only the <b>alarm record</b> (the facts) and writes a readable report.",
         "“At 00:33 the pH rose sharply …”"),
        ("Every fact is checked", "Every number, time and sensor name in the text is compared with the record.",
         "“00:35” in the text but 00:33 in the record → mistake"),
        ("Mistake? Not shown", "A report with any mistake is replaced by the plain <b>template</b> report, so the "
         "operator never reads a wrong fact.", "59 of 60 AI reports had no mistake"),
    ], colors=[ui.ACCENT, ui.VIOLET, ui.WARN, ui.OK])
    ui.explain("If Ollama is not running, every report is the template – it is always correct, just plainer. "
               "See real examples on <b>AI incident reports</b>.")
    t = nav.PAGES.get("reports")
    if t is not None:
        st.page_link(t, label="Read real AI reports and their fact checks", icon=":material/description:")


# --------------------------------------------------------------------------------------------- what's new

def _new() -> None:
    st.markdown("### Version 2, segment by segment")
    st.markdown("Version 2 was built in ten steps (“segments”). Each one asks one question:")
    rows = [
        ("S0", "Get real data", "Load the real utility data (GECCO 2018/2019) and add pollution and broken sensors "
                                "whose times we know."),
        ("S1", "Was version 1 tested fairly?", "No: shuffled minutes make the test too easy. Honest score 0.64 instead "
                                               "of 0.98. Every later number uses the honest test."),
        ("S2", "Do version 1's parts work on real data?", "Moved onto the same real stream; the ensemble works (37/46 "
                                                          "events), the fixed 2-of-4 rule gives 33 false alarms a day."),
        ("S3", "Does forecasting help?", "Paper 1's BiLSTM, used as “surprise”: no better than a simple method."),
        ("S4", "Water or sensor?", "Health checks send broken sensors to maintenance: faulty-sensor water alarms 7 → 1."),
        ("S5", "Can we promise the false-alarm rate?", "Yes – the adaptive alarm line keeps about 1 a day (0.90–1.18)."),
        ("S6", "Are the explanations true?", "Tested by removing the named sensor: the simple explainer passes "
                                             "0.95–1.00, paper 1's TreeSHAP 0.74."),
        ("S7", "Can an AI write the report safely?", "Yes, with fact checking: 59/60 reports with no mistake."),
        ("S8", "Does it fit on a ₹350 Arduino?", "Yes: identical answers for 10,080 minutes, 25 ms per minute."),
        ("S9", "Can anyone use it?", "This dashboard, the live sensor demo, and version 1 alongside."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Segment", "Question", "Answer"]), hide_index=True, width="stretch",
                 column_config={"Segment": st.column_config.TextColumn(width="small"),
                                "Question": st.column_config.TextColumn(width="medium"),
                                "Answer": st.column_config.TextColumn(width="large")})
    t = nav.PAGES.get("results")
    if t is not None:
        st.page_link(t, label="See the numbers behind each answer", icon=":material/insights:")


# --------------------------------------------------------------------------------------------- fallbacks

def _missing() -> None:
    st.markdown("### Nothing is required")
    st.markdown("The project was designed so that it still runs in five months on a different laptop. "
                "Each part is checked when the dashboard starts; if it is missing, the computer replaces it:")
    st.dataframe(pd.DataFrame([
        ("Arduino Uno not plugged in", "The laptop runs the same logic. “Try the sensors” uses the simulated probe."),
        ("Water probe loose or missing", "Simulated probe (buttons: Add salt, Pull probe out …)."),
        ("Touch pad missing", "The on-screen ❌ False alarm button."),
        ("DHT11 missing", "Temperature and humidity show “–”."),
        ("Ollama / AI model not running", "Template reports (always correct)."),
        ("Real data files missing", "A simulated station with the same nine sensors."),
        ("Version 1 not running", "Its published results are shown on the Version 1 page."),
        ("Experiment results not on this computer", "The measured results bundled with the code are shown."),
    ], columns=["If this is missing …", "… the computer does this instead"]), hide_index=True, width="stretch")
    t = nav.PAGES.get("system")
    if t is not None:
        st.page_link(t, label="See what is connected right now", icon=":material/settings_input_component:")


# --------------------------------------------------------------------------------------------- words

def _words() -> None:
    st.markdown("### Words used in this dashboard")
    ui.glossary([
        ("🚨 Water alarm", "The water itself changed in an unusual way. Red everywhere."),
        ("🔧 Sensor alert", "A sensor looks broken (stuck, dropped out, spiking, not answering). Maintenance, not "
                           "a water alarm. Yellow everywhere."),
        ("Alarm score", "How unusual the water looks right now. Higher = more unusual."),
        ("Alarm line", "How high the score must go for an alarm. It rises after a false alarm and slowly returns."),
        ("False alarm", "An alarm when nothing was wrong with the water."),
        ("Operator", "The person in the control room – in the demo, you."),
        ("Answer key", "What really happened (from the utility and from our own injected events). Shown to you, "
                       "never to the system."),
        ("Move (×)", "How far a sensor moved in 15 minutes compared with its largest move in normal water. "
                     "1× = the largest normal move."),
        ("Ion index", "The live probe's reading turned around so that more salt = higher number."),
        ("Calibration", "The first ~40 seconds of the live lab, when it learns what “normal” looks like."),
        ("Honest test", "Train on the past, test on the future – like real life. The opposite of shuffling minutes."),
        ("Event F1", "One score (0 to 1) for “did the alarms match the real events?”. 1 is perfect."),
        ("LLM", "Large language model – the AI that writes text. Here a small one running on the laptop."),
        ("Template report", "A fill-in-the-blanks report. Plain, but always correct."),
        ("GECCO 2018/2019", "Public real data from a German drinking-water utility, one row per minute."),
        ("Edge / Uno", "Running the decision on the small board next to the sensors instead of on a big computer."),
    ])
