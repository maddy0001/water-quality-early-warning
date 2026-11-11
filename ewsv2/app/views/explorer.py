"""Data Explorer: the sensor data itself – time series, a 3D view of the sensor space, correlations and data quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ewsv2.app import charts as C
from ewsv2.app import services as S
from ewsv2.app import ui
from ewsv2.app.views.common import base, sources

CLASS = {0: ("normal operation", "#93c5fd"), 1: ("water event", ui.BAD), 2: ("sensor fault", "#f59e0b")}


def _runs(cls: np.ndarray, code: int) -> list[tuple[int, int]]:
    on = np.flatnonzero(np.diff(np.r_[0, (cls == code).astype(int), 0]))
    return list(zip(on[::2], on[1::2] - 1))


def render() -> None:
    ui.header("Data Explorer",
              "The measurements behind every result: browse each sensor over time, rotate the sensor space in 3D to "
              "see how events stand apart from normal water, and check how the sensors relate to each other.",
              kicker="Analyze · data", color="amber")
    with st.container(border=True):
        ui.panel_title("Data selection")
        c1, c2 = st.columns([1.4, 2])
        with c1:
            src = st.selectbox("Data source", sources(), format_func=S.source_label, key="ex_src")
            ui.cap("Real utility data (A, B) includes the utility's own event labels plus injected contaminations and "
                   "sensor faults with known times; the synthetic station is generated.")
        b = base(src)
        f, sens, cls = b["frame"], b["sensors"], b["cls"]
        with c2:
            pick = st.multiselect("Sensors to plot", sens, default=sens[:4], key="ex_sens",
                                  format_func=lambda s: f"{s} – {S.SENSOR_NAMES.get(s, s)}")
            ui.cap("Choose one or more sensors for the time-series chart. Each is scaled to its own range so they "
                   "can be compared.")
        days = pd.to_datetime(f.index.normalize().unique())
        start_day = days[min(len(days) - 1, max(0, int(np.searchsorted(days, f.index[b["start"]].normalize()))))]
        default_end = days[min(len(days) - 1, int(np.searchsorted(days, start_day)) + 3)]
        win = st.select_slider("Time window", options=list(days), value=(start_day, default_end), key="ex_win",
                               format_func=lambda d: f"{d:%d %b %Y}")
        ui.cap("Days shown in the time-series chart. Shorter windows show more detail.")

    a, z = f.index.searchsorted(win[0]), f.index.searchsorted(win[1] + pd.Timedelta(days=1))
    w, wc = f.iloc[a:z], cls[a:z]
    ui.tiles([
        ("Recorded minutes", f"{len(f):,}", f"{f.index[0]:%d %b %Y} to {f.index[-1]:%d %b %Y}", "amber"),
        ("Sensors", str(len(sens)), ", ".join(sens), "ocean"),
        ("Water-event minutes", f"{int((cls == 1).sum()):,}", f"{len(_runs(cls, 1))} events (labelled or injected)",
         "red"),
        ("Sensor-fault minutes", f"{int((cls == 2).sum()):,}", f"{len(_runs(cls, 2))} injected faults", "violet"),
    ])

    ui.section("Sensor time series", "Shaded areas: red = water event, amber = sensor fault. Drag on the lower strip "
                                      "to zoom.")
    fig = go.Figure()
    for code in (1, 2):
        for r0, r1 in _runs(wc, code):
            fig.add_vrect(x0=w.index[r0], x1=w.index[r1], fillcolor=CLASS[code][1], opacity=0.13, line_width=0)
    for j, s in enumerate(pick or sens[:1]):
        x = w[s].astype(float)
        lo, hi = np.nanpercentile(f[s].astype(float), [1, 99])
        fig.add_trace(go.Scatter(x=w.index, y=(x - lo) / (hi - lo + 1e-9), name=f"{s} ({S.SENSOR_NAMES.get(s, s)})",
                                 line=dict(width=1.4, color=ui.SERIES[j % len(ui.SERIES)]),
                                 customdata=x, hovertemplate="%{customdata:.3f}<extra>" + s + "</extra>"))
    fig.update_layout(height=430, title="Selected sensors (each scaled 0–1 to its normal range)",
                      xaxis=dict(rangeslider=dict(visible=True, thickness=0.08)), yaxis_title="scaled value",
                      yaxis=dict(range=[-0.3, 1.4]),
                      legend=dict(orientation="h", y=-0.35))
    C.plot(fig, key="ex_ts")

    ui.section("Sensor space in 3D", "Each dot is one minute. Normal minutes form a cloud; events and faults fall "
                                     "outside it. Drag to rotate, scroll to zoom, or press Rotate.")
    k1, k2, k3 = st.columns(3)
    defaults = [s for s in ("pH", "Leit", "Redox", "Cond", "Turb", "Tp") if s in sens] + sens
    ax = [k1.selectbox("X axis", sens, index=sens.index(defaults[0]), key="ex_x"),
          k2.selectbox("Y axis", sens, index=sens.index(defaults[1]), key="ex_y"),
          k3.selectbox("Z axis", sens, index=sens.index(defaults[2]), key="ex_z")]
    C.plot(_cube(f, cls, ax), key="ex_3d")

    a1, a2 = st.columns(2)
    with a1:
        corr = S.read_csv("s0_inspection", src, "correlations.csv")
        if corr is not None:
            corr = corr.set_index(corr.columns[0])
        else:
            corr = f[sens].astype(float).corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale="RdBu_r",
                                   zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}", xgap=2, ygap=2))
        fig.update_layout(title="How strongly sensors move together (correlation)", height=430,
                          yaxis=dict(autorange="reversed"))
        C.plot(fig, key="ex_corr")
        ui.cap("Red: rise together. Blue: one rises when the other falls. A real water event usually moves several "
               "related sensors at once; a failing sensor moves alone.")
    with a2:
        resp = S.read_csv("s0_inspection", src, "event_response.csv")
        if resp is not None:
            resp = resp.set_index(resp.columns[0]).sort_values("share_events_shift_gt_threshold")
            fig = go.Figure(go.Bar(x=resp["share_events_shift_gt_threshold"], y=resp.index, orientation="h",
                                   marker=dict(color=resp["share_events_shift_gt_threshold"], colorscale="Viridis"),
                                   text=[f"{v:.0%}" for v in resp["share_events_shift_gt_threshold"]],
                                   textposition="outside"))
            fig.update_layout(title="Share of real events in which each sensor clearly changed", height=430,
                              xaxis=dict(range=[0, 1.1], tickformat=".0%"))
            C.plot(fig, key="ex_resp")
            ui.cap("Shows which measurements carry the signal of a real water-quality event at this station.")
        else:
            counts = pd.Series({CLASS[k][0]: int((cls == k).sum()) for k in CLASS})
            C.plot(C.donut(list(counts.index), list(counts.values), [CLASS[k][1] for k in CLASS], "Minutes by state"),
                   key="ex_donut")

    q = S.read_csv("s0_inspection", src, "sensors.csv")
    if q is not None:
        ui.section("Data quality per sensor", "Missing values and long flat stretches are what makes real data hard.")
        b1, b2 = st.columns(2)
        with b1:
            C.plot(C.hbar(q["sensor"].tolist(), q["missing_pct"].tolist(), "Missing values (% of minutes)",
                          fmt="{:.1f} %"), key="ex_missing")
        with b2:
            C.plot(C.hbar(q["sensor"].tolist(), (q["longest_constant_run_min"] / 60).tolist(),
                          "Longest unchanged stretch (hours)", fmt="{:.0f} h",
                          colors=[ui.SERIES[(i + 3) % len(ui.SERIES)] for i in range(len(q))]), key="ex_flat")
        with st.expander("Sensor statistics table"):
            st.dataframe(q, hide_index=True, width="stretch")
            st.download_button("Download as CSV", q.to_csv(index=False).encode("utf-8"),
                               file_name=f"sensor_statistics_{src}.csv", mime="text/csv", on_click="ignore",
                               key="ex_dl_q", icon=":material/download:")


def _cube(f: pd.DataFrame, cls: np.ndarray, ax: list[str]) -> go.Figure:
    rng = np.random.default_rng(7)
    x = f[ax].astype(float)
    ok = x.notna().all(axis=1).to_numpy()
    normal = x[ok & (cls == 0)]
    ranges = []
    for c in ax:                               # robust axes: the normal cloud fills the cube, outliers sit on the walls
        lo, hi = np.nanpercentile(normal[c] if len(normal) else x[c], [0.5, 99.5])
        pad = (hi - lo) * 0.6 + 1e-9
        ranges.append((lo - pad, hi + pad))
        x[c] = x[c].clip(lo - pad, hi + pad)
    fig = go.Figure()
    for code, n in ((0, 6000), (1, 700), (2, 700)):
        idx = np.flatnonzero(ok & (cls == code))
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
        name, col = CLASS[code]
        fig.add_trace(go.Scatter3d(x=x.iloc[idx, 0], y=x.iloc[idx, 1], z=x.iloc[idx, 2], mode="markers", name=name,
                                   marker=dict(size=2.4 if code == 0 else 3.4, color="#3b82f6" if code == 0 else col,
                                               opacity=0.30 if code == 0 else 0.9)))
    frames = []
    for k in range(36):
        ang = 2 * np.pi * k / 36
        frames.append(go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.8 * np.cos(ang), y=1.8 * np.sin(ang), z=0.9)))))
    fig.frames = frames
    fig.update_layout(
        height=560, title=f"{ax[0]} × {ax[1]} × {ax[2]}", legend=dict(orientation="h", y=0.02, x=0.02),
        scene=dict(xaxis_title=ax[0], yaxis_title=ax[1], zaxis_title=ax[2], camera=dict(eye=dict(x=1.8, y=0, z=0.9)),
                   xaxis=dict(backgroundcolor="#f8fafc", range=list(ranges[0])),
                   yaxis=dict(backgroundcolor="#f1f5f9", range=list(ranges[1])),
                   zaxis=dict(backgroundcolor="#eef2ff", range=list(ranges[2])), aspectmode="cube"),
        updatemenus=[dict(type="buttons", x=0.98, y=0.98, xanchor="right", showactive=False, buttons=[
            dict(label="Rotate", method="animate",
                 args=[None, dict(frame=dict(duration=120, redraw=True), transition=dict(duration=0), fromcurrent=True,
                                  mode="immediate")]),
            dict(label="Stop", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        margin=dict(l=0, r=0, t=40, b=0))
    return fig
