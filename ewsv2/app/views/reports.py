"""Incident reports from S7: every report next to the facts it was checked against."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ewsv2.app import services as S
from ewsv2.app import ui


def render() -> None:
    ui.hero("AI incident reports", "Reports for real alarms, written by the plain template and by a small AI on the "
            "laptop – each shown next to the facts it was checked against.", eyebrow="The proof", color="violet")
    ui.guide('How to read this page', [
        'Each real alarm has two reports: the <b>template</b> (always correct, plain) and the <b>local LLM</b> (nicer to read).',
        'Every number and sensor name in the LLM text is checked against the alarm record; mistakes are listed in red, and such a report is never shown to the operator – the template is shown instead.',
    ])
    data = {ds: S.read_json("s7_reports", ds, "reports.json") for ds in ("gecco2018", "gecco2019")}
    data = {k: v for k, v in data.items() if v}
    if not data:
        st.info("No S7 output on this computer yet. Run `python -m ewsv2 report --config s7_reports` (Ollama optional). "
                "Reports for new alarms are written live in the *Control room*.")
        return
    c1, c2, c3 = st.columns([1, 1.2, 2])
    ds = c1.selectbox("Station", list(data))
    reps = data[ds]
    reporters = sorted({r["reporter"] for r in reps})
    who = c2.selectbox("Writer", reporters, index=reporters.index(next((x for x in reporters if x != "template"), reporters[0])))
    rows = [r for r in reps if r["reporter"] == who]
    ok = sum(1 for r in rows if not r["errors"])
    c3.markdown(ui.kpi(f"{who} on {ds}", f"{ok}/{len(rows)}", "reports with no fact error"), unsafe_allow_html=True)
    ids = [r["id"] for r in rows]
    pick = st.select_slider("Report", options=ids, value=ids[0])
    r = next(x for x in rows if x["id"] == pick)
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown(ui.pill("no errors", "ok") if not r["errors"] else ui.pill(f"{len(r['errors'])} error(s)", "bad"),
                    unsafe_allow_html=True)
        st.markdown(f'<div class="report">{ui.esc(r["text"])}</div>', unsafe_allow_html=True)
        if r["errors"]:
            st.dataframe(pd.DataFrame(r["errors"]), hide_index=True, width="stretch")
        tpl = next((x for x in reps if x["reporter"] == "template" and x["id"] == pick), None)
        if tpl is not None and who != "template":
            with st.expander("The template report for the same alarm"):
                st.markdown(f'<div class="report">{ui.esc(tpl["text"])}</div>', unsafe_allow_html=True)
    with b:
        rec = r["record"]
        ui.card("Alarm record (the only facts allowed)",
                f'<b>{ui.esc(rec.get("decision"))}</b> · level {ui.esc(rec.get("level"))}<br>'
                f'<span class="muted">{ui.esc(rec.get("start"))} → {ui.esc(rec.get("end"))} '
                f'({ui.esc(rec.get("duration_min"))} min)</span>')
        st.json(rec, expanded=False)
        st.caption("Guidance used: " + ", ".join(r.get("guidance_ids", [])))
