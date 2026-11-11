"""Look and feel: a bright, simple theme, the chart template and small building blocks (Segment 9.2).

Design rules (so every page feels the same):
  * one colour = one meaning everywhere: blue = information, green = all good, red = the WATER changed,
    amber = a SENSOR looks broken, violet = the operator / the AI writer;
  * every page starts with a coloured header that says in one sentence what the page is for;
  * plain words first, numbers second, details hidden in "more" boxes.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- colours (bright, light background) ----
BG = "#f5f7fb"
PANEL = "#ffffff"
INK = "#0f172a"
MUTED = "#5b6b82"
LINE = "#e3e8f0"
ACCENT = "#2563eb"          # blue  – information, the alarm score
CYAN = "#06b6d4"            # water
OK = "#16a34a"              # green – all good
WARN = "#d97706"            # amber – a sensor looks broken
BAD = "#dc2626"             # red   – the water changed
VIOLET = "#7c3aed"          # the operator, the AI writer
PINK = "#db2777"
SERIES = [ACCENT, "#0891b2", "#16a34a", "#d97706", VIOLET, PINK, "#ea580c", "#0d9488", "#4f46e5"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --panel:{PANEL}; --line:{LINE}; --accent:{ACCENT}; }}
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {{
    font-family: "Plus Jakarta Sans", "Segoe UI", system-ui, -apple-system, sans-serif; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.015em; color:{INK}; }}
h3 {{ font-weight: 800; }}
a {{ color:{ACCENT}; }}

/* header of every page */
.hero {{ padding: 26px 30px; border-radius: 22px; margin-bottom: 18px; color: #fff;
        background: linear-gradient(120deg, #2563eb 0%, #0ea5e9 55%, #06b6d4 100%);
        box-shadow: 0 10px 30px -12px rgba(37,99,235,.55); position: relative; overflow: hidden; }}
.hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
              background: rgba(255,255,255,.12); }}
.hero .eyebrow {{ font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; opacity:.85; }}
.hero h1 {{ margin: 2px 0 6px 0; font-size: 2.05rem; font-weight: 800; color:#fff; }}
.hero p {{ margin: 0; font-size: 1.05rem; opacity:.95; max-width: 920px; line-height:1.5; }}
.hero.violet {{ background: linear-gradient(120deg, #7c3aed 0%, #6366f1 55%, #3b82f6 100%);
               box-shadow: 0 10px 30px -12px rgba(124,58,237,.55); }}
.hero.green {{ background: linear-gradient(120deg, #059669 0%, #10b981 55%, #06b6d4 100%);
              box-shadow: 0 10px 30px -12px rgba(5,150,105,.5); }}
.hero.orange {{ background: linear-gradient(120deg, #ea580c 0%, #f59e0b 60%, #eab308 100%);
               box-shadow: 0 10px 30px -12px rgba(234,88,12,.5); }}
.hero.slate {{ background: linear-gradient(120deg, #334155 0%, #475569 60%, #64748b 100%);
              box-shadow: 0 10px 30px -12px rgba(51,65,85,.5); }}

/* number cards */
.kpi {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:14px 16px; height:100%;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.kpi .label {{ color:{MUTED}; font-size:.76rem; text-transform:uppercase; letter-spacing:.07em; font-weight:700; }}
.kpi .value {{ color:{INK}; font-size:1.6rem; font-weight:800; margin-top:2px; line-height:1.2; }}
.kpi .sub {{ color:{MUTED}; font-size:.83rem; margin-top:3px; line-height:1.35; }}

.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.76rem; font-weight:700;
        margin:2px 4px 2px 0; }}
.pill.ok {{ color:#15803d; background:#dcfce7; }}
.pill.warn {{ color:#b45309; background:#fef3c7; }}
.pill.bad {{ color:#b91c1c; background:#fee2e2; }}
.pill.info {{ color:#1d4ed8; background:#dbeafe; }}
.pill.violet {{ color:#6d28d9; background:#ede9fe; }}

/* the big state banner */
.banner {{ border-radius:18px; padding:16px 20px; margin:6px 0 12px 0; border:2px solid; display:flex; gap:14px;
          align-items:center; }}
.banner .icon {{ font-size:2rem; line-height:1; }}
.banner .title {{ font-size:1.3rem; font-weight:800; }}
.banner .text {{ color:{INK}; margin-top:2px; }}
.banner.normal {{ border-color:#86efac; background:#f0fdf4; }} .banner.normal .title {{ color:#15803d; }}
.banner.water {{ border-color:#fca5a5; background:#fef2f2; animation: pulse 1.6s ease-in-out infinite; }}
.banner.water .title {{ color:{BAD}; }}
.banner.sensor {{ border-color:#fcd34d; background:#fffbeb; }} .banner.sensor .title {{ color:#b45309; }}
.banner.idle {{ border-color:{LINE}; background:{PANEL}; }} .banner.idle .title {{ color:{MUTED}; }}
@keyframes pulse {{ 0%,100% {{ box-shadow:0 0 0 0 rgba(220,38,38,0); }} 50% {{ box-shadow:0 0 0 6px rgba(220,38,38,.12); }} }}

.card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:14px 18px; margin-bottom:12px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.card .h {{ margin:0 0 6px 0; font-size:1.0rem; font-weight:800; color:{INK}; }}
.muted {{ color:{MUTED}; }}
.report {{ white-space:pre-wrap; font-family: "Cascadia Mono", Consolas, monospace; font-size:.85rem; color:{INK};
          background:#f8fafc; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; }}
.logrow {{ border-bottom:1px solid {LINE}; padding:7px 2px; font-size:.9rem; }}
.logrow:last-child {{ border-bottom:none; }}

/* pipeline / flow */
.flow {{ display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; }}
.flow .node {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:10px 12px; min-width:130px; flex:1; }}
.flow .node b {{ color:{ACCENT}; font-size:.76rem; text-transform:uppercase; letter-spacing:.06em; }}
.flow .arrow {{ color:{MUTED}; font-size:1.3rem; align-self:center; }}

/* numbered steps */
.steps {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap:12px; margin:6px 0 14px 0; }}
.step {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px; padding:16px; position:relative;
        box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.step .n {{ width:34px; height:34px; border-radius:50%; color:#fff; font-weight:800; display:flex; align-items:center;
           justify-content:center; margin-bottom:8px; }}
.step .t {{ font-weight:800; font-size:1.02rem; margin-bottom:4px; color:{INK}; }}
.step .d {{ color:{MUTED}; font-size:.9rem; line-height:1.45; }}
.step .ex {{ margin-top:8px; font-size:.84rem; color:{INK}; background:#f8fafc; border-radius:10px; padding:6px 9px; }}

/* hardware / part cards */
.part {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px; padding:16px 18px; height:100%;
        border-top:6px solid var(--c); box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.part .head {{ display:flex; gap:12px; align-items:center; margin-bottom:8px; }}
.part .emoji {{ font-size:2rem; }}
.part .name {{ font-weight:800; font-size:1.1rem; color:{INK}; }}
.part .model {{ color:{MUTED}; font-size:.82rem; }}
.part .row {{ margin:7px 0; font-size:.92rem; line-height:1.45; color:{INK}; }}
.part .row .lab {{ display:block; color:var(--c); font-size:.72rem; text-transform:uppercase; letter-spacing:.07em;
                   font-weight:800; }}
.part code {{ background:#f1f5f9; color:{INK}; padding:1px 5px; border-radius:6px; font-size:.82rem; }}

/* comparison */
.vs {{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }}
.vs .col {{ border-radius:18px; padding:16px 18px; border:1px solid {LINE}; background:{PANEL}; }}
.vs .col.old {{ border-top:6px solid #94a3b8; }} .vs .col.new {{ border-top:6px solid {ACCENT}; }}
.vs .col .h {{ margin:0 0 8px 0; font-weight:800; font-size:1.05rem; color:{INK}; }}
.vs .col li {{ margin:4px 0; }}
@media (max-width: 800px) {{ .vs {{ grid-template-columns: 1fr; }} }}

/* status chips */
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin:4px 0 12px 0; }}
.chip {{ border-radius:999px; padding:6px 12px; font-size:.86rem; font-weight:600; background:{PANEL};
        border:1px solid {LINE}; color:{INK}; }}
.chip.on {{ border-color:#86efac; background:#f0fdf4; }} .chip.off {{ border-color:#fde68a; background:#fffbeb; }}
.chip .s {{ color:{MUTED}; font-weight:500; }}

.guide {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:16px; padding:12px 18px; margin:4px 0 14px 0; }}
.guide .t {{ color:#1d4ed8; font-weight:800; font-size:.92rem; margin-bottom:4px; }}
.guide ol {{ margin:0 0 0 18px; padding:0; }} .guide li {{ margin:3px 0; color:{INK}; }}
.explain {{ color:{MUTED}; font-size:.87rem; margin:-2px 0 8px 2px; line-height:1.45; }}
.status {{ display:flex; gap:20px; flex-wrap:wrap; align-items:center; padding:9px 16px; border-radius:14px;
           background:{PANEL}; border:1px solid {LINE}; margin:6px 0 10px 0; font-size:.92rem; }}
.status b {{ color:{INK}; }} .status span {{ color:{MUTED}; }}
.check {{ display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 12px 0; }}
.check .item {{ border:1px solid {LINE}; border-radius:999px; padding:6px 12px; color:{MUTED}; font-size:.88rem;
               background:{PANEL}; font-weight:600; }}
.check .item.done {{ border-color:#86efac; color:#15803d; background:#f0fdf4; }}
.gloss {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }}
.gloss div {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:10px 14px; font-size:.9rem; }}
.gloss b {{ color:{INK}; }}
.big {{ font-size:1.08rem; line-height:1.6; }}

section[data-testid="stSidebar"] {{ background: #ffffff; border-right:1px solid {LINE}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:18px; }}
div[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:10px 14px; }}
button[kind="primary"] {{ font-weight:700; }}
.st-key-cr_qbox_ask {{ background:#f5f3ff; border-radius:12px; padding:4px 8px; }}
.st-key-cr_qbox_ask p {{ font-size:1.02rem; }}
div[data-testid="stExpander"] details {{ background:{PANEL}; border-radius:14px; }}

/* Smooth live updates: Streamlit dims ("stale") every element while a page refreshes – that was the
   flicker. Keep everything fully visible and hide the running indicator. */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"] {{ opacity: 1 !important; transition: none !important; }}
[data-testid="stStatusWidget"], [data-testid="stDecoration"] {{ visibility: hidden; }}
.js-plotly-plot, .plot-container {{ transition: none !important; }}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    tmpl = go.layout.Template(pio.templates["plotly_white"])
    tmpl.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
                       font=dict(color=INK, size=12, family="Plus Jakarta Sans, Segoe UI, sans-serif"),
                       title=dict(font=dict(size=14)), margin=dict(l=10, r=10, t=40, b=10),
                       xaxis=dict(gridcolor="#edf1f7", zerolinecolor=LINE, linecolor=LINE),
                       yaxis=dict(gridcolor="#edf1f7", zerolinecolor=LINE, linecolor=LINE),
                       legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18))
    pio.templates["ewsv2"] = tmpl
    pio.templates.default = "ewsv2"


def esc(x) -> str:
    return html.escape(str(x))


def hero(title: str, text: str, eyebrow: str = "", color: str = "") -> None:
    eb = f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    st.markdown(f'<div class="hero {color}">{eb}<h1>{esc(title)}</h1><p>{esc(text)}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>')


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for c, (a, b, d) in zip(cols, items):
        c.markdown(kpi(a, b, d), unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


ICONS = {"normal": "✅", "water": "🚨", "sensor": "🔧", "idle": "⏳"}


def banner(kind: str, title: str, text: str = "") -> None:
    st.markdown(f'<div class="banner {kind}"><div class="icon">{ICONS.get(kind, "")}</div><div>'
                f'<div class="title">{esc(title)}</div><div class="text">{esc(text)}</div></div></div>',
                unsafe_allow_html=True)


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
    h = "".join(f"<div><span>{esc(a)}</span> <b>{esc(b)}</b></div>" for a, b in parts)
    st.markdown(f'<div class="status">{h}</div>', unsafe_allow_html=True)


def checklist(items: list[tuple[str, bool]]) -> None:
    h = "".join(f'<div class="item{" done" if ok else ""}">{"✓" if ok else "○"} {esc(t)}</div>' for t, ok in items)
    st.markdown(f'<div class="check">{h}</div>', unsafe_allow_html=True)


def chips(items: list[tuple[str, bool, str]]) -> None:
    """Status chips: (name, ready?, short note)."""
    h = "".join(f'<div class="chip {"on" if ok else "off"}">{"🟢" if ok else "🟡"} {esc(n)} '
                f'<span class="s">· {esc(note)}</span></div>' for n, ok, note in items)
    st.markdown(f'<div class="chips">{h}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str, str]], colors: list[str] | None = None) -> None:
    """Numbered step cards: (title, text, example). Text may contain simple HTML."""
    colors = colors or [ACCENT, "#0891b2", OK, WARN, VIOLET, PINK]
    h = ""
    for i, (t, d, ex) in enumerate(items):
        c = colors[i % len(colors)]
        h += (f'<div class="step"><div class="n" style="background:{c}">{i + 1}</div><div class="t">{esc(t)}</div>'
              f'<div class="d">{d}</div>' + (f'<div class="ex">{ex}</div>' if ex else "") + "</div>")
    st.markdown(f'<div class="steps">{h}</div>', unsafe_allow_html=True)


def part_card(emoji: str, name: str, model: str, color: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<div class="row"><span class="lab">{esc(a)}</span>{b}</div>' for a, b in rows)
    return (f'<div class="part" style="--c:{color}"><div class="head"><div class="emoji">{emoji}</div><div>'
            f'<div class="name">{esc(name)}</div><div class="model">{esc(model)}</div></div></div>{body}</div>')


def compare(old_title: str, old: list[str], new_title: str, new: list[str]) -> None:
    o = "".join(f"<li>{x}</li>" for x in old)
    n = "".join(f"<li>{x}</li>" for x in new)
    st.markdown(f'<div class="vs"><div class="col old"><div class="h">{esc(old_title)}</div><ul>{o}</ul></div>'
                f'<div class="col new"><div class="h">{esc(new_title)}</div><ul>{n}</ul></div></div>',
                unsafe_allow_html=True)


def glossary(items: list[tuple[str, str]]) -> None:
    h = "".join(f"<div><b>{esc(a)}</b><br><span class='muted'>{b}</span></div>" for a, b in items)
    st.markdown(f'<div class="gloss">{h}</div>', unsafe_allow_html=True)


def steady(fig):
    """Charts that update in place: keep zoom, no animation."""
    fig.update_layout(uirevision="keep", transition={"duration": 0})
    return fig


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="card"><div class="h">{esc(title)}</div>{body_html}</div>', unsafe_allow_html=True)


def flow(nodes: list[tuple[str, str]]) -> None:
    parts = []
    for i, (a, b) in enumerate(nodes):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f'<div class="node"><b>{esc(a)}</b><br>{esc(b)}</div>')
    st.markdown('<div class="flow">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def page_tile(page, emoji: str, title: str, text: str, label: str = "Open") -> None:
    """A big clickable entry point to another page (falls back to text when the page is unknown, e.g. in tests)."""
    with st.container(border=True):
        st.markdown(f'<div style="font-size:2rem;line-height:1">{emoji}</div>'
                    f'<div style="font-weight:800;font-size:1.08rem;margin:6px 0 2px 0">{esc(title)}</div>'
                    f'<div class="muted" style="font-size:.9rem;min-height:5.2em">{esc(text)}</div>',
                    unsafe_allow_html=True)
        if page is not None:
            st.page_link(page, label=label, icon=":material/arrow_forward:")
