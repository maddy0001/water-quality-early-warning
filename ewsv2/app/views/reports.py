"""Incident Reports: every report generated for the real alarms, its fact check, and downloads."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ewsv2.app import charts as C
from ewsv2.app import export as X
from ewsv2.app import services as S
from ewsv2.app import ui


def _writer(r: str) -> str:
    return "Standard template" if r == "template" else f"AI model {r}"


def render() -> None:
    ui.header("Incident Reports",
              "Reports written for every alarm in the real test periods – by the standard template and, where it "
              "was run, by the local AI model – each shown with the automatic fact check and available for download.",
              kicker="Analyze · reports", color="violet")
    data = {ds: S.read_json("s7_reports", ds, "reports.json") for ds in ("gecco2018", "gecco2019")}
    data = {k: v for k, v in data.items() if v}
    if not data:
        st.info("No generated reports are stored on this computer yet. Run `python -m ewsv2 report --config "
                "s7_reports` once (the AI model is optional). Reports for new alarms can be prepared and downloaded in "
                "the Operations Console.", icon=":material/info:")
        return
    with st.container(border=True):
        ui.panel_title("Report selection")
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        with c1:
            ds = st.selectbox("Dataset", list(data), format_func=lambda s: S.source_label(s, short=True), key="rp_ds")
            ui.cap("Which real test period the alarms come from.")
        reps = data[ds]
        writers = sorted({r["reporter"] for r in reps}, key=lambda x: (x == "template", x))
        with c2:
            who = st.selectbox("Written by", writers, format_func=_writer, key="rp_who")
            ui.cap("The standard template is always correct; AI reports read better and are fact-checked.")
        rows = [r for r in reps if r["reporter"] == who]
        ids = [r["id"] for r in rows]
        with c3:
            pick = st.selectbox("Alarm", ids, key="rp_pick",
                                format_func=lambda i: _label(next(x for x in rows if x["id"] == i)))
            ui.cap("One report per alarm. The list shows the start time and the decision.")

    ok = sum(1 for r in rows if not r["errors"])
    secs = [r.get("seconds", 0) or 0 for r in rows]
    ui.tiles([("Reports", str(len(rows)), f"{_writer(who)} · {S.source_label(ds, short=True)}", "violet"),
              ("Without factual error", f"{ok} / {len(rows)}", "checked automatically against the alarm record",
               "emerald" if ok == len(rows) else "amber"),
              ("Average writing time", f"{sum(secs) / max(1, len(secs)):.1f} s", "per report on this computer", "ocean")])

    r = next(x for x in rows if x["id"] == pick)
    a, b = st.columns([1.45, 1])
    with a:
        st.markdown(ui.badge("No factual errors", "ok") if not r["errors"] else
                    ui.badge(f"{len(r['errors'])} factual error(s) – would be replaced by the template", "bad"),
                    unsafe_allow_html=True)
        st.markdown(f'<div class="report">{ui.esc(r["text"])}</div>', unsafe_allow_html=True)
        if r["errors"]:
            st.dataframe(pd.DataFrame(r["errors"]), hide_index=True, width="stretch")
        tpl = next((x for x in reps if x["reporter"] == "template" and x["id"] == pick), None)
        if tpl is not None and who != "template":
            with st.expander("Standard template report for the same alarm"):
                st.markdown(f'<div class="report">{ui.esc(tpl["text"])}</div>', unsafe_allow_html=True)
    with b:
        rec = r["record"]
        ui.card("Alarm record – the only facts the report may use",
                f'<b>{ui.esc(rec.get("decision"))}</b> · level {ui.esc(rec.get("level"))}<br>'
                f'<span class="muted">{ui.esc(rec.get("start"))} to {ui.esc(rec.get("end"))} '
                f'({ui.esc(rec.get("duration_min"))} min)</span>')
        stem = f"incident_{ds}_{pick}_{who.replace(':', '-')}"
        st.download_button("Download this report (.txt)", X.report_txt(r["text"], rec, _writer(who)),
                           file_name=stem + ".txt", mime="text/plain", width="stretch", icon=":material/download:",
                           on_click="ignore", key="rp_dl_txt")
        st.download_button("Download printable report (.html)", X.report_html(r["text"], rec, _writer(who)),
                           file_name=stem + ".html", mime="text/html", width="stretch", icon=":material/print:",
                           on_click="ignore", key="rp_dl_html")
        allrows = pd.DataFrame([{"id": x["id"], "written_by": _writer(x["reporter"]), "decision": x["record"].get("decision"),
                                 "start": x["record"].get("start"), "end": x["record"].get("end"),
                                 "factual_errors": len(x["errors"]), "report": x["text"]} for x in reps])
        st.download_button("Download all reports for this dataset (.csv)", allrows.to_csv(index=False).encode("utf-8"),
                           file_name=f"incident_reports_{ds}.csv", mime="text/csv", width="stretch",
                           icon=":material/table_view:", on_click="ignore", key="rp_dl_all")
        with st.expander("Full alarm record"):
            st.json(rec, expanded=False)

    ui.section("All reports in this dataset", "Each bar is one writer; colour shows the result of the fact check.")
    g = pd.DataFrame([{"Written by": _writer(x["reporter"]),
                       "Alarm type": "water-quality alarm" if x["record"].get("kind") == "water" else "sensor fault",
                       "Fact check": "no error" if not x["errors"] else "error found"} for x in reps])
    g = g.groupby(["Written by", "Alarm type", "Fact check"]).size().reset_index(name="Reports")
    fig = px.bar(g, x="Written by", y="Reports", color="Fact check", pattern_shape="Alarm type", barmode="stack",
                 color_discrete_map={"no error": ui.OK, "error found": ui.BAD})
    fig.update_layout(height=340, legend=dict(orientation="h", y=-0.25, title=None), xaxis_title=None)
    C.plot(fig, key="rp_bar")


def _label(r: dict) -> str:
    rec = r["record"]
    kind = "Water-quality alarm" if rec.get("kind") == "water" else "Sensor-fault alert"
    return f"{rec.get('start', '')} · {kind} ({r['id']})"
