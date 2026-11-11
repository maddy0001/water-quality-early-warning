"""Look and feel: colours, CSS, chart template and small building blocks (Segment 9)."""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#e8eef6"
MUTED = "#8ea3b8"
PANEL = "#111c2b"
LINE = "#22324a"
ACCENT = "#38bdf8"          # water blue
OK = "#34d399"
WARN = "#fbbf24"
BAD = "#f87171"
VIOLET = "#a78bfa"
SERIES = [ACCENT, OK, WARN, VIOLET, "#f472b6", "#22d3ee", "#fb923c", "#a3e635", "#e879f9"]

CSS = f"""
<style>
:root {{ --ink:{INK}; --muted:{MUTED}; --panel:{PANEL}; --line:{LINE}; --accent:{ACCENT}; }}
html, body, [class*="css"] {{ font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-weight: 750; }}
.hero {{ padding: 26px 28px; border-radius: 18px; margin-bottom: 18px;
        background: radial-gradient(1200px 300px at 0% 0%, rgba(56,189,248,.20), transparent 60%),
                    radial-gradient(800px 260px at 100% 0%, rgba(167,139,250,.16), transparent 60%), {PANEL};
        border: 1px solid {LINE}; }}
.hero h1 {{ margin: 0 0 6px 0; font-size: 2.0rem; }}
.hero p {{ margin: 0; color: {MUTED}; font-size: 1.02rem; }}
.kpi {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; height:100%; }}
.kpi .label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }}
.kpi .value {{ color:{INK}; font-size:1.65rem; font-weight:700; margin-top:2px; line-height:1.2; }}
.kpi .sub {{ color:{MUTED}; font-size:.82rem; margin-top:2px; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
        border:1px solid; margin:2px 4px 2px 0; }}
.pill.ok {{ color:{OK}; border-color:rgba(52,211,153,.45); background:rgba(52,211,153,.10); }}
.pill.warn {{ color:{WARN}; border-color:rgba(251,191,36,.45); background:rgba(251,191,36,.10); }}
.pill.bad {{ color:{BAD}; border-color:rgba(248,113,113,.45); background:rgba(248,113,113,.10); }}
.pill.info {{ color:{ACCENT}; border-color:rgba(56,189,248,.45); background:rgba(56,189,248,.10); }}
.banner {{ border-radius:16px; padding:16px 20px; margin:6px 0 12px 0; border:1px solid; }}
.banner .title {{ font-size:1.35rem; font-weight:750; }}
.banner .text {{ color:{INK}; opacity:.9; margin-top:2px; }}
.banner.normal {{ border-color:rgba(52,211,153,.4); background:linear-gradient(90deg, rgba(52,211,153,.14), rgba(52,211,153,.03)); }}
.banner.normal .title {{ color:{OK}; }}
.banner.water {{ border-color:rgba(248,113,113,.55); background:linear-gradient(90deg, rgba(248,113,113,.22), rgba(248,113,113,.04));
                 animation: pulse 1.6s ease-in-out infinite; }}
.banner.water .title {{ color:{BAD}; }}
.banner.sensor {{ border-color:rgba(251,191,36,.55); background:linear-gradient(90deg, rgba(251,191,36,.18), rgba(251,191,36,.03)); }}
.banner.sensor .title {{ color:{WARN}; }}
.banner.idle {{ border-color:{LINE}; background:{PANEL}; }}
.banner.idle .title {{ color:{MUTED}; }}
@keyframes pulse {{ 0%,100% {{ box-shadow:0 0 0 0 rgba(248,113,113,.0); }} 50% {{ box-shadow:0 0 22px 2px rgba(248,113,113,.28); }} }}
.card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; margin-bottom:10px; }}
.card h4 {{ margin:0 0 6px 0; font-size:1.0rem; }}
.muted {{ color:{MUTED}; }}
.report {{ white-space:pre-wrap; font-family: Consolas, "Cascadia Mono", monospace; font-size:.86rem;
          background:#0b1420; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; }}
.logrow {{ border-bottom:1px solid {LINE}; padding:6px 2px; font-size:.88rem; }}
.flow {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.flow .node {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:10px 12px; min-width:120px; }}
.flow .node b {{ color:{ACCENT}; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
.flow .arrow {{ color:{MUTED}; font-size:1.2rem; }}
section[data-testid="stSidebar"] {{ background: #0b1420; border-right:1px solid {LINE}; }}
/* Smooth live updates: Streamlit dims ("stale") every element while a page refreshes – that was the
   flicker. Keep everything fully visible and hide the running indicator. */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"] {{ opacity: 1 !important; transition: none !important; }}
[data-testid="stStatusWidget"], [data-testid="stDecoration"] {{ visibility: hidden; }}
.js-plotly-plot, .plot-container {{ transition: none !important; }}
.guide {{ background: rgba(56,189,248,.07); border:1px solid rgba(56,189,248,.30); border-radius:14px;
          padding:12px 16px; margin:4px 0 12px 0; }}
.guide .t {{ color:{ACCENT}; font-weight:700; font-size:.9rem; margin-bottom:4px; }}
.guide ol {{ margin:0 0 0 18px; padding:0; }} .guide li {{ margin:2px 0; color:{INK}; }}
.explain {{ color:{MUTED}; font-size:.86rem; margin:-4px 0 10px 2px; }}
.status {{ display:flex; gap:18px; flex-wrap:wrap; align-items:center; padding:8px 14px; border-radius:12px;
           background:{PANEL}; border:1px solid {LINE}; margin:6px 0 10px 0; font-size:.92rem; }}
.status b {{ color:{INK}; }} .status span {{ color:{MUTED}; }}
.slot {{ min-height: 150px; }}
.big {{ font-size:1.05rem; }}
.check {{ display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 12px 0; }}
.check .item {{ border:1px solid {LINE}; border-radius:10px; padding:6px 10px; color:{MUTED}; font-size:.88rem; }}
.check .item.done {{ border-color:rgba(52,211,153,.5); color:{OK}; background:rgba(52,211,153,.08); }}
div[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:10px 14px; }}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    tmpl = go.layout.Template(pio.templates["plotly_dark"])
    tmpl.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
                       font=dict(color=INK, size=12), margin=dict(l=10, r=10, t=36, b=10),
                       xaxis=dict(gridcolor=LINE, zerolinecolor=LINE), yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
                       legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18))
    pio.templates["ewsv2"] = tmpl
    pio.templates.default = "ewsv2"


def esc(x) -> str:
    return html.escape(str(x))


def hero(title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(text)}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>')


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (a, b, d) in zip(cols, items):
        c.markdown(kpi(a, b, d), unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def banner(kind: str, title: str, text: str = "") -> None:
    st.markdown(f'<div class="banner {kind}"><div class="title">{esc(title)}</div>'
                f'<div class="text">{esc(text)}</div></div>', unsafe_allow_html=True)


def explanations_on() -> bool:
    return st.session_state.get("explain", True)


def guide(title: str, steps: list[str]) -> None:
    """A numbered "how to use this page" box (hidden when explanations are switched off)."""
    if not explanations_on():
        return
    items = "".join(f"<li>{s}</li>" for s in steps)          # steps may contain simple HTML (bold)
    st.markdown(f'<div class="guide"><div class="t">{esc(title)}</div><ol>{items}</ol></div>', unsafe_allow_html=True)


def explain(text: str) -> None:
    """One plain-language line under a chart or panel (hidden when explanations are switched off)."""
    if explanations_on():
        st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def status(parts: list[tuple[str, str]]) -> None:
    html = "".join(f"<div><span>{esc(a)}</span> <b>{esc(b)}</b></div>" for a, b in parts)
    st.markdown(f'<div class="status">{html}</div>', unsafe_allow_html=True)


def checklist(items: list[tuple[str, bool]]) -> None:
    html = "".join(f'<div class="item{" done" if ok else ""}">{"✓" if ok else "○"} {esc(t)}</div>' for t, ok in items)
    st.markdown(f'<div class="check">{html}</div>', unsafe_allow_html=True)


def steady(fig):
    """Charts that update in place: keep zoom, no animation."""
    fig.update_layout(uirevision="keep", transition={"duration": 0})
    return fig


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="card"><h4>{esc(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def flow(nodes: list[tuple[str, str]]) -> None:
    parts = []
    for i, (a, b) in enumerate(nodes):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f'<div class="node"><b>{esc(a)}</b><br>{esc(b)}</div>')
    st.markdown('<div class="flow">' + "".join(parts) + "</div>", unsafe_allow_html=True)
