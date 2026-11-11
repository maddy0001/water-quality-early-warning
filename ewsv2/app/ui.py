"""Design system of the dashboard: colours, CSS, chart template and page building blocks.

Rules every page follows
  * one colour = one meaning: indigo = information, emerald = normal, red = water-quality alarm,
    amber = sensor fault, violet = operator review / report generation;
  * every page opens with a header that states its purpose in one sentence;
  * every control has a short caption under it that says what it does;
  * numbers come with a one-line interpretation.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------------------------- palette
BG = "#f4f6fb"
PANEL = "#ffffff"
INK = "#0b1324"
MUTED = "#5a6782"
LINE = "#e2e7f1"
ACCENT = "#4f46e5"          # indigo   – information, anomaly score
CYAN = "#0891b2"
OK = "#059669"              # emerald  – normal
WARN = "#d97706"            # amber    – sensor fault
BAD = "#dc2626"             # red      – water-quality alarm
VIOLET = "#7c3aed"          # violet   – operator review, AI reports
PINK = "#db2777"
SERIES = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#db2777", "#7c3aed", "#ea580c", "#0d9488", "#2563eb",
          "#c026d3", "#65a30d", "#e11d48"]
GRAD = {
    "indigo": "linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%)",
    "ocean": "linear-gradient(135deg,#0284c7 0%,#06b6d4 100%)",
    "emerald": "linear-gradient(135deg,#059669 0%,#14b8a6 100%)",
    "amber": "linear-gradient(135deg,#ea580c 0%,#f59e0b 100%)",
    "rose": "linear-gradient(135deg,#db2777 0%,#f43f5e 100%)",
    "violet": "linear-gradient(135deg,#7c3aed 0%,#c026d3 100%)",
    "slate": "linear-gradient(135deg,#334155 0%,#64748b 100%)",
    "red": "linear-gradient(135deg,#b91c1c 0%,#ef4444 100%)",
}

WAVES = ("<svg class='waves' viewBox='0 24 150 28' preserveAspectRatio='none'><defs>"
         "<path id='w' d='M-160 44c30 0 58-18 88-18s58 18 88 18 58-18 88-18 58 18 88 18v44h-352z'/></defs>"
         "<g class='p'><use href='#w' x='48' y='0' fill='rgba(255,255,255,.22)'/>"
         "<use href='#w' x='48' y='3' fill='rgba(255,255,255,.14)'/>"
         "<use href='#w' x='48' y='5' fill='rgba(255,255,255,.10)'/></g></svg>")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select, label {{
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1360px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.02em; color:{INK}; }}

/* ---- sidebar: deep indigo ---- */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0b1324 0%,#1e1b4b 100%); }}
section[data-testid="stSidebar"] * {{ color: #cbd5e1; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{ border-radius: 10px; margin: 1px 0; }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{ background: rgba(255,255,255,.06); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: linear-gradient(90deg, rgba(99,102,241,.45), rgba(124,58,237,.25)); }}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {{ color:#fff; font-weight:700; }}
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] span,
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {{ color:#818cf8 !important; font-size:.72rem;
    letter-spacing:.12em; text-transform:uppercase; font-weight:800; }}
.brand {{ display:flex; gap:12px; align-items:center; padding:4px 2px 10px 2px; }}
.brand .logo {{ width:40px; height:40px; border-radius:12px; background:{GRAD['ocean']}; display:flex;
               align-items:center; justify-content:center; box-shadow:0 6px 18px -6px rgba(6,182,212,.7); }}
.brand .t1 {{ color:#fff !important; font-weight:800; font-size:.98rem; line-height:1.15; }}
.brand .t2 {{ color:#94a3b8 !important; font-size:.74rem; }}
.side-status {{ font-size:.8rem; line-height:1.9; }}
.side-status .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }}

/* ---- page header with animated gradient and waves ---- */
.hdr {{ position:relative; overflow:hidden; border-radius:22px; padding:26px 30px 44px 30px; margin-bottom:18px; color:#fff;
       background: linear-gradient(120deg,#312e81,#4f46e5,#0891b2,#7c3aed); background-size: 300% 300%;
       animation: hdrflow 14s ease infinite; box-shadow: 0 18px 40px -22px rgba(79,70,229,.8); }}
.hdr.ocean {{ background: linear-gradient(120deg,#0c4a6e,#0284c7,#06b6d4,#0d9488); background-size:300% 300%; }}
.hdr.emerald {{ background: linear-gradient(120deg,#064e3b,#059669,#14b8a6,#0891b2); background-size:300% 300%; }}
.hdr.violet {{ background: linear-gradient(120deg,#4c1d95,#7c3aed,#c026d3,#db2777); background-size:300% 300%; }}
.hdr.amber {{ background: linear-gradient(120deg,#7c2d12,#ea580c,#f59e0b,#db2777); background-size:300% 300%; }}
.hdr.slate {{ background: linear-gradient(120deg,#0f172a,#334155,#475569,#4f46e5); background-size:300% 300%; }}
@keyframes hdrflow {{ 0% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} 100% {{ background-position:0% 50%; }} }}
.hdr .kicker {{ font-size:.74rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; opacity:.85; }}
.hdr h1 {{ color:#fff; margin:4px 0 6px 0; font-size:2.0rem; font-weight:800; }}
.hdr p {{ margin:0; max-width:900px; font-size:1.02rem; line-height:1.55; opacity:.95; }}
.hdr .waves {{ position:absolute; left:0; bottom:0; width:100%; height:46px; }}
.hdr .waves .p use {{ animation: wave 9s cubic-bezier(.55,.5,.45,.5) infinite; }}
.hdr .waves .p use:nth-child(2) {{ animation-delay:-3s; animation-duration:12s; }}
.hdr .waves .p use:nth-child(3) {{ animation-delay:-5s; animation-duration:16s; }}
@keyframes wave {{ 0% {{ transform: translate3d(-90px,0,0); }} 100% {{ transform: translate3d(85px,0,0); }} }}

/* ---- section titles ---- */
.sec {{ margin: 18px 0 8px 0; }}
.sec .t {{ font-size:1.15rem; font-weight:800; color:{INK}; }}
.sec .s {{ font-size:.9rem; color:{MUTED}; margin-top:2px; }}

/* ---- gradient KPI tiles with 3D hover ---- */
.tiles {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:14px; margin:4px 0 8px 0;
         perspective: 900px; }}
.tile {{ border-radius:18px; padding:16px 18px; color:#fff; min-height:118px; position:relative; overflow:hidden;
        transform-style:preserve-3d; transition: transform .35s ease, box-shadow .35s ease;
        box-shadow: 0 12px 28px -18px rgba(15,23,42,.65); }}
.tile:hover {{ transform: rotateX(7deg) rotateY(-7deg) translateY(-3px); box-shadow: 0 22px 36px -18px rgba(15,23,42,.7); }}
.tile:after {{ content:""; position:absolute; right:-40px; top:-40px; width:130px; height:130px; border-radius:50%;
              background: rgba(255,255,255,.14); }}
.tile .l {{ font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; opacity:.9; }}
.tile .v {{ font-size:1.85rem; font-weight:800; margin:4px 0 2px 0; line-height:1.15; }}
.tile .n {{ font-size:.82rem; opacity:.92; line-height:1.35; }}

/* ---- white cards ---- */
.card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px; padding:16px 18px; margin-bottom:12px;
        box-shadow: 0 1px 3px rgba(15,23,42,.05); }}
.card .h {{ font-weight:800; font-size:1rem; color:{INK}; margin-bottom:6px; }}
.muted {{ color:{MUTED}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:18px; background:{PANEL}; }}
div[data-testid="stExpander"] details {{ background:{PANEL}; border-radius:14px; }}

/* ---- control captions ---- */
.cap {{ color:{MUTED}; font-size:.78rem; line-height:1.35; margin-top:-6px; }}
.panel-title {{ font-size:.74rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:{ACCENT};
               margin-bottom:4px; }}

/* ---- badges ---- */
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:800;
         letter-spacing:.04em; margin:2px 4px 2px 0; }}
.badge.ok {{ color:#065f46; background:#d1fae5; }} .badge.warn {{ color:#92400e; background:#fef3c7; }}
.badge.bad {{ color:#991b1b; background:#fee2e2; }} .badge.info {{ color:#3730a3; background:#e0e7ff; }}
.badge.violet {{ color:#5b21b6; background:#ede9fe; }} .badge.grey {{ color:#334155; background:#e2e8f0; }}

/* ---- the system state bar ---- */
.state {{ display:flex; align-items:center; gap:16px; border-radius:18px; padding:14px 20px; margin:8px 0 12px 0;
         color:#fff; box-shadow:0 14px 30px -20px rgba(15,23,42,.7); }}
.state .lbl {{ font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; opacity:.85; }}
.state .ttl {{ font-size:1.25rem; font-weight:800; }}
.state .txt {{ font-size:.92rem; opacity:.95; }}
.state .pulse {{ width:14px; height:14px; border-radius:50%; background:#fff; flex:none; box-shadow:0 0 0 0 rgba(255,255,255,.7);
                animation: ping 1.8s infinite; }}
.state.normal {{ background:{GRAD['emerald']}; }} .state.water {{ background:{GRAD['red']}; }}
.state.sensor {{ background:{GRAD['amber']}; }} .state.idle {{ background:{GRAD['slate']}; }}
.state.water .pulse {{ animation-duration: .9s; }}
@keyframes ping {{ 0% {{ box-shadow:0 0 0 0 rgba(255,255,255,.7); }} 100% {{ box-shadow:0 0 0 14px rgba(255,255,255,0); }} }}

/* ---- meta strip ---- */
.meta {{ display:flex; gap:22px; flex-wrap:wrap; align-items:center; padding:10px 16px; border-radius:14px;
        background:{PANEL}; border:1px solid {LINE}; font-size:.88rem; }}
.meta span {{ color:{MUTED}; }} .meta b {{ color:{INK}; }}

/* ---- pipeline ---- */
.pipe {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap:12px; margin:6px 0 10px 0; }}
.pipe .st {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:14px; position:relative;
            border-top:5px solid var(--c); box-shadow: 0 1px 3px rgba(15,23,42,.05); }}
.pipe .n {{ font-size:.7rem; font-weight:800; letter-spacing:.12em; color:var(--c); }}
.pipe .t {{ font-weight:800; margin:2px 0 4px 0; color:{INK}; }}
.pipe .d {{ font-size:.86rem; color:{MUTED}; line-height:1.45; }}

/* ---- component cards ---- */
.comp {{ background:{PANEL}; border:1px solid {LINE}; border-radius:18px; padding:16px 18px; height:100%;
        box-shadow: 0 1px 3px rgba(15,23,42,.05); }}
.comp .top {{ display:flex; gap:12px; align-items:center; margin-bottom:8px; }}
.comp .mono {{ width:46px; height:46px; border-radius:14px; color:#fff; font-weight:800; font-size:.8rem; display:flex;
              align-items:center; justify-content:center; flex:none; }}
.comp .nm {{ font-weight:800; font-size:1.05rem; color:{INK}; }}
.comp .md {{ font-size:.8rem; color:{MUTED}; }}
.comp .row {{ font-size:.9rem; line-height:1.45; margin:6px 0; color:{INK}; }}
.comp .row .k {{ display:block; font-size:.68rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--c); }}
.comp code {{ background:#eef2ff; color:#3730a3; padding:1px 6px; border-radius:6px; font-size:.8rem; }}

/* ---- glossary ---- */
.gloss {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(270px,1fr)); gap:10px; }}
.gloss div {{ background:{PANEL}; border:1px solid {LINE}; border-left:4px solid {ACCENT}; border-radius:12px;
             padding:10px 14px; font-size:.88rem; }}

/* ---- steps checklist ---- */
.steps {{ display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 10px 0; }}
.steps .s {{ flex:1; min-width:170px; border-radius:14px; padding:10px 12px; border:1px solid {LINE}; background:{PANEL};
            font-size:.86rem; color:{MUTED}; }}
.steps .s b {{ display:block; color:{INK}; }}
.steps .s.done {{ border-color:#6ee7b7; background:#ecfdf5; }}
.steps .s .k {{ font-size:.68rem; font-weight:800; letter-spacing:.1em; color:{ACCENT}; }}
.steps .s.done .k {{ color:{OK}; }}

.report {{ white-space:pre-wrap; font-family: "Cascadia Mono", Consolas, monospace; font-size:.84rem; color:{INK};
          background:#f8fafc; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; max-height:460px;
          overflow:auto; }}
.logrow {{ border-bottom:1px solid {LINE}; padding:7px 2px; font-size:.88rem; }}
.logrow:last-child {{ border-bottom:none; }}
.review {{ font-size:1.02rem; font-weight:700; color:{INK}; }}
.st-key-review_open {{ background: linear-gradient(90deg,#f5f3ff,#eef2ff); border-radius:12px; padding:2px 8px; }}

button[kind="primary"] {{ background:{GRAD['indigo']} !important; border:none !important; font-weight:700; }}
button[kind="primary"]:hover {{ filter:brightness(1.08); }}
button {{ font-weight:600 !important; }}

/* Live updates without flicker: never dim elements while a fragment refreshes. */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"] {{ opacity: 1 !important; transition: none !important; }}
[data-testid="stStatusWidget"], [data-testid="stDecoration"] {{ visibility: hidden; }}
.js-plotly-plot, .plot-container {{ transition: none !important; }}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    tmpl = go.layout.Template(pio.templates["plotly_white"])
    tmpl.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
                       font=dict(color=INK, size=12, family="Inter, Segoe UI, sans-serif"),
                       title=dict(font=dict(size=14, color=INK), x=0.01), margin=dict(l=10, r=10, t=44, b=10),
                       xaxis=dict(gridcolor="#edf1f8", zerolinecolor=LINE, linecolor=LINE),
                       yaxis=dict(gridcolor="#edf1f8", zerolinecolor=LINE, linecolor=LINE),
                       hoverlabel=dict(bgcolor="#0b1324", font=dict(color="#fff")),
                       legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2))
    pio.templates["ewsv2"] = tmpl
    pio.templates.default = "ewsv2"


def esc(x) -> str:
    return html.escape(str(x))


# ---------------------------------------------------------------------------------------------- blocks

def header(title: str, purpose: str, kicker: str = "", color: str = "") -> None:
    k = f'<div class="kicker">{esc(kicker)}</div>' if kicker else ""
    st.markdown(f'<div class="hdr {color}">{k}<h1>{esc(title)}</h1><p>{esc(purpose)}</p>{WAVES}</div>',
                unsafe_allow_html=True)


hero = header          # older name


def section(title: str, sub: str = "") -> None:
    s = f'<div class="s">{sub}</div>' if sub else ""
    st.markdown(f'<div class="sec"><div class="t">{esc(title)}</div>{s}</div>', unsafe_allow_html=True)


def tiles(items: list[tuple]) -> None:
    """Gradient KPI tiles: (label, value, note, gradient-name)."""
    h = ""
    for it in items:
        label, value, note = it[0], it[1], it[2]
        g = GRAD.get(it[3] if len(it) > 3 else "indigo", GRAD["indigo"])
        h += (f'<div class="tile" style="background:{g}"><div class="l">{esc(label)}</div>'
              f'<div class="v">{esc(value)}</div><div class="n">{esc(note)}</div></div>')
    st.markdown(f'<div class="tiles">{h}</div>', unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """Older helper: same as tiles with rotating colours."""
    names = ["indigo", "ocean", "emerald", "amber", "violet", "rose"]
    tiles([(a, b, c, names[i % len(names)]) for i, (a, b, c) in enumerate(items)])


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile" style="background:{GRAD["indigo"]}"><div class="l">{esc(label)}</div>'
            f'<div class="v">{esc(value)}</div><div class="n">{esc(sub)}</div></div>')


def cap(text: str) -> None:
    """Caption under a control: what it does."""
    st.markdown(f'<div class="cap">{text}</div>', unsafe_allow_html=True)


def panel_title(text: str) -> None:
    st.markdown(f'<div class="panel-title">{esc(text)}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


pill = badge           # older name


def state(kind: str, title: str, text: str = "", label: str = "System status") -> None:
    st.markdown(f'<div class="state {kind}"><div class="pulse"></div><div><div class="lbl">{esc(label)}</div>'
                f'<div class="ttl">{esc(title)}</div><div class="txt">{esc(text)}</div></div></div>',
                unsafe_allow_html=True)


def banner(kind: str, title: str, text: str = "") -> None:
    state(kind, title, text)


def meta(parts: list[tuple[str, str]]) -> None:
    h = "".join(f"<div><span>{esc(a)}</span>&nbsp; <b>{esc(b)}</b></div>" for a, b in parts)
    st.markdown(f'<div class="meta">{h}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="card"><div class="h">{esc(title)}</div>{body_html}</div>', unsafe_allow_html=True)


def pipeline(stages: list[tuple[str, str]], colors: list[str] | None = None) -> None:
    colors = colors or [ACCENT, CYAN, OK, WARN, VIOLET, PINK]
    h = "".join(f'<div class="st" style="--c:{colors[i % len(colors)]}"><div class="n">STAGE {i + 1}</div>'
                f'<div class="t">{esc(t)}</div><div class="d">{d}</div></div>' for i, (t, d) in enumerate(stages))
    st.markdown(f'<div class="pipe">{h}</div>', unsafe_allow_html=True)


def component(mono: str, name: str, model: str, color: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<div class="row"><span class="k">{esc(k)}</span>{v}</div>' for k, v in rows)
    return (f'<div class="comp" style="--c:{color}"><div class="top"><div class="mono" style="background:{color}">'
            f'{esc(mono)}</div><div><div class="nm">{esc(name)}</div><div class="md">{esc(model)}</div></div></div>'
            f'{body}</div>')


def glossary(items: list[tuple[str, str]]) -> None:
    h = "".join(f"<div><b>{esc(a)}</b><br><span class='muted'>{b}</span></div>" for a, b in items)
    st.markdown(f'<div class="gloss">{h}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str, bool]]) -> None:
    """Checklist of demonstration steps: (title, text, done)."""
    h = "".join(f'<div class="s{" done" if ok else ""}"><span class="k">{"COMPLETED" if ok else f"STEP {i + 1}"}</span>'
                f'<b>{esc(t)}</b>{esc(d)}</div>' for i, (t, d, ok) in enumerate(items))
    st.markdown(f'<div class="steps">{h}</div>', unsafe_allow_html=True)


def side_status(items: list[tuple[str, bool, str]]) -> None:
    h = "".join(f'<div><span class="dot" style="background:{"#34d399" if ok else "#fbbf24"}"></span>{esc(n)}'
                f' <span style="color:#64748b">· {esc(note)}</span></div>' for n, ok, note in items)
    st.markdown(f'<div class="side-status">{h}</div>', unsafe_allow_html=True)


def steady(fig):
    """Charts that update in place: keep zoom, no animation between refreshes."""
    fig.update_layout(uirevision="keep", transition={"duration": 0})
    return fig


def link_card(page, title: str, text: str, label: str) -> None:
    """A described entry point to another page."""
    with st.container(border=True):
        st.markdown(f'<div style="font-weight:800;font-size:1.05rem;color:{INK}">{esc(title)}</div>'
                    f'<div class="muted" style="font-size:.88rem;min-height:4.4em;margin:4px 0 6px 0">{esc(text)}</div>',
                    unsafe_allow_html=True)
        if page is not None:
            st.page_link(page, label=label, icon=":material/arrow_forward:")
