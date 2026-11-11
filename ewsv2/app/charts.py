"""Charts used across the dashboard.

Gauges and the liquid-level chart use ECharts through the ``streamlit-echarts`` component (its JavaScript is
bundled inside the Python package, so it works offline). If that package is missing or fails, every such
chart falls back to an equivalent Plotly chart – the page never breaks.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import ui

try:                                                     # optional – see module docstring
    from streamlit_echarts import st_echarts as _st_echarts
except Exception:  # noqa: BLE001
    _st_echarts = None

CONFIG = {"displayModeBar": False}


def echarts_ok() -> bool:
    return _st_echarts is not None and not st.session_state.get("_no_echarts", False)


def _echart(option: dict, height: int, key: str) -> bool:
    if not echarts_ok():
        return False
    try:
        _st_echarts(options=option, height=f"{height}px", key=key)
        return True
    except Exception:  # noqa: BLE001
        st.session_state["_no_echarts"] = True
        return False


def plot(fig: go.Figure, key: str | None = None, steady: bool = False) -> None:
    if steady:
        ui.steady(fig)
    st.plotly_chart(fig, width="stretch", config=CONFIG, key=key)


# --------------------------------------------------------------------------------------------- gauges

def gauge(value: float, lo: float, hi: float, title: str, key: str, unit: str = "",
          stops: list[tuple[float, str]] | None = None, height: int = 230, decimals: int = 1) -> None:
    """A dial. ``stops`` = [(upper value, colour), ...] in increasing order."""
    value = float(np.clip(value if np.isfinite(value) else lo, lo, hi))
    stops = stops or [(hi, ui.ACCENT)]
    span = hi - lo
    line = [[min(1.0, max(0.0, (v - lo) / span)), c] for v, c in stops]
    opt = {
        "series": [{
            "type": "gauge", "min": lo, "max": hi, "startAngle": 210, "endAngle": -30, "radius": "92%",
            "axisLine": {"lineStyle": {"width": 16, "color": line}},
            "pointer": {"length": "62%", "width": 5, "itemStyle": {"color": "#0b1324"}},
            "anchor": {"show": True, "size": 12, "itemStyle": {"color": "#0b1324"}},
            "axisTick": {"distance": -16, "length": 5, "lineStyle": {"color": "#fff", "width": 1}},
            "splitLine": {"distance": -16, "length": 16, "lineStyle": {"color": "#fff", "width": 2}},
            "axisLabel": {"distance": 22, "color": "#5a6782", "fontSize": 10},
            "title": {"offsetCenter": [0, "72%"], "fontSize": 12, "color": "#5a6782"},
            "detail": {"valueAnimation": True, "offsetCenter": [0, "42%"], "fontSize": 24, "fontWeight": 800,
                       "color": "#0b1324", "formatter": "{value}" + unit},
            "data": [{"value": round(value, decimals), "name": title}],
        }],
        "animationDurationUpdate": 400,
    }
    if _echart(opt, height, key):
        return
    steps, prev = [], lo
    for v, c in stops:
        steps.append({"range": [prev, v], "color": c})
        prev = v
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number={"suffix": unit},
                                 title={"text": title, "font": {"size": 12}},
                                 gauge={"axis": {"range": [lo, hi]}, "bar": {"color": "#0b1324", "thickness": 0.18},
                                        "steps": steps}))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=30, b=10))
    plot(fig, key=key + "_p", steady=True)


def liquid(fraction: float, label: str, key: str, color: str = "#0891b2", height: int = 240) -> None:
    """A liquid-level ball (fraction 0..1)."""
    f = float(np.clip(fraction if np.isfinite(fraction) else 0, 0, 1))
    opt = {"series": [{
        "type": "liquidFill", "data": [f, max(0, f - 0.05), max(0, f - 0.1)], "radius": "86%",
        "color": [color, "#06b6d4", "#4f46e5"], "backgroundStyle": {"color": "#eef6fb"},
        "outline": {"borderDistance": 4, "itemStyle": {"borderWidth": 4, "borderColor": color}},
        "label": {"fontSize": 26, "fontWeight": 800, "color": "#0b1324", "insideColor": "#fff",
                  "formatter": label},
    }]}
    if _echart(opt, height, key):
        return
    fig = go.Figure(go.Indicator(mode="gauge+number", value=f * 100, number={"suffix": " %"},
                                 gauge={"shape": "bullet", "axis": {"range": [0, 100]}, "bar": {"color": color}}))
    fig.update_layout(height=120, margin=dict(l=20, r=20, t=20, b=10))
    plot(fig, key=key + "_p", steady=True)


# --------------------------------------------------------------------------------------------- plotly helpers

def donut(labels: list[str], values: list[float], colors: list[str], title: str, center: str = "",
          height: int = 300) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, marker=dict(colors=colors, line=dict(color="#fff", width=2)),
                           textinfo="value", sort=False))
    fig.update_layout(title=title, height=height, showlegend=True, legend=dict(orientation="v", y=0.5, x=1.02),
                      annotations=[dict(text=center, showarrow=False, font=dict(size=15, color=ui.INK))])
    return fig


def hbar(names: list[str], values: list[float], title: str, colors=None, fmt: str = "{:.2f}",
         height: int | None = None, xmax: float | None = None) -> go.Figure:
    colors = colors or [ui.SERIES[i % len(ui.SERIES)] for i in range(len(names))]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           text=[fmt.format(v) for v in values], textposition="outside", cliponaxis=False))
    top = xmax if xmax is not None else (max(values) * 1.18 if len(values) and max(values) > 0 else 1)
    fig.update_layout(title=title, height=height or 70 + 34 * len(names), yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, top]), bargap=0.32)
    return fig


def sankey(sources: list[str], targets: list[str], values: list[float], title: str, height: int = 380,
           node_colors: dict | None = None) -> go.Figure:
    labels = list(dict.fromkeys(sources + targets))
    idx = {l: i for i, l in enumerate(labels)}
    nc = node_colors or {}
    colors = [nc.get(l, ui.SERIES[i % len(ui.SERIES)]) for i, l in enumerate(labels)]
    link_colors = [_alpha(colors[idx[t]], 0.35) for t in targets]
    fig = go.Figure(go.Sankey(arrangement="snap",
                              node=dict(label=labels, color=colors, pad=18, thickness=18, line=dict(width=0)),
                              textfont=dict(color=ui.INK, size=12),
                              link=dict(source=[idx[s] for s in sources], target=[idx[t] for t in targets],
                                        value=values, color=link_colors)))
    fig.update_layout(title=title, height=height, font=dict(size=12))
    return fig


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


alpha = _alpha
