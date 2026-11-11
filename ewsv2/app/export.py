"""Downloadable files: incident reports (text and printable HTML) and alarm logs."""

from __future__ import annotations

import html
import json
from datetime import datetime


def report_txt(text: str, record: dict, writer: str) -> bytes:
    head = (f"INCIDENT REPORT\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\nWritten by: {writer}\n"
            f"{'=' * 60}\n\n")
    facts = "\n\nALARM RECORD (facts the report was checked against)\n" + json.dumps(record, indent=2, default=str)
    return (head + text.strip() + "\n" + facts + "\n").encode("utf-8")


def report_html(text: str, record: dict, writer: str) -> bytes:
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(_fmt(v))}</td></tr>"
                   for k, v in record.items() if k != "sensors")
    sens = "".join(f"<tr><td>{html.escape(str(s.get('code', '')))}</td><td>{html.escape(str(s.get('description', '')))}</td>"
                   f"<td>{html.escape(_fmt(s.get('before')))}</td><td>{html.escape(_fmt(s.get('now')))}</td>"
                   f"<td>{html.escape(_fmt(s.get('change')))}</td></tr>" for s in record.get("sensors", []))
    kind = record.get("decision", "Incident")
    color = "#dc2626" if record.get("kind") == "water" else "#d97706"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Incident report {html.escape(str(record.get('id', '')))}</title>
<style>
body {{ font-family: Inter, Segoe UI, Arial, sans-serif; color:#0b1324; max-width:820px; margin:32px auto; padding:0 20px; }}
.top {{ border-radius:14px; padding:18px 22px; color:#fff; background:{color}; }}
.top h1 {{ margin:0; font-size:22px; }} .top p {{ margin:4px 0 0 0; opacity:.9; }}
pre {{ white-space:pre-wrap; font-family: Consolas, monospace; background:#f8fafc; border:1px solid #e2e7f1; border-radius:10px; padding:14px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 18px 0; font-size:14px; }}
th, td {{ border:1px solid #e2e7f1; padding:6px 10px; text-align:left; }} th {{ background:#f4f6fb; width:34%; }}
.foot {{ color:#5a6782; font-size:12px; margin-top:24px; }}
@media print {{ body {{ margin:0; }} }}
</style></head><body>
<div class="top"><h1>{html.escape(str(kind))}</h1><p>{html.escape(str(record.get('station', '')))} · {html.escape(str(record.get('start', '')))} to {html.escape(str(record.get('end', '')))}</p></div>
<h2>Report</h2><pre>{html.escape(text.strip())}</pre>
<h2>Alarm record</h2><table>{rows}</table>
<h2>Sensors involved</h2><table><tr><th>Sensor</th><th>Measures</th><th>Before</th><th>At alarm</th><th>Change</th></tr>{sens}</table>
<div class="foot">Written by: {html.escape(writer)} · generated {datetime.now():%Y-%m-%d %H:%M} · Water Quality Early Warning System v2.
Use the browser's Print → Save as PDF to make a PDF.</div>
</body></html>"""
    return doc.encode("utf-8")


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) or "–"
    return str(v)
