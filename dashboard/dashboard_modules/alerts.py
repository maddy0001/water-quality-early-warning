"""
Early Warning & Alert History Page - real alert fusion using the project's alert engine
(src/early_warning/alert_system.py: weights classification 0.40 / forecasting 0.30 / anomaly 0.30)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from . import engine as E

COLORS = {'NORMAL': '#2ECC71', 'LOW': '#E9B949', 'MEDIUM': '#E67E22', 'CRITICAL': '#E74C3C'}


def show():
    st.markdown("## 🚨 Early Warning & Alert History")

    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0;'>🎯 What This Does</h3>
        <p>Runs the <b>full early-warning system</b> on every sample and combines the modules into one alert:</p>
        <ol>
            <li><b>Classification</b> (ensemble) gives a level: potable → NORMAL; unsafe with confidence
            50–70% → LOW, 70–90% → MEDIUM, ≥ 90% → CRITICAL</li>
            <li><b>Anomaly detection</b> gives a level from the votes: 2 → LOW, 3 → MEDIUM, 4 → CRITICAL</li>
            <li>The levels are combined with weights <b>classification 0.40 · forecasting 0.30 · anomaly 0.30</b>
            and rounded to the nearest level. If any module is CRITICAL, the final alert is at least MEDIUM.</li>
        </ol>
        <p><b>Risk score</b> = the weighted average level ÷ 3, so it runs from 0 (all NORMAL) to 1 (all CRITICAL).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='honest-box'>
        <b>Note:</b> the 9-parameter samples are single measurements with no time history, so the forecasting
        module cannot score them. As in the paper's fusion run, the alert here combines classification and
        anomaly detection; forecasting joins when a station has co-located sensor history.
    </div>
    """, unsafe_allow_html=True)

    if not E.show_missing(['rf', 'xgb', 'svm', 'scaler', 'if', 'ocsvm', 'ae', 'ae_thr', 'raw'], "early-warning"):
        return

    st.markdown("---")
    st.markdown("### 📤 Upload Samples to Check")
    st.info("💡 The CSV needs the 9 water-quality columns. Optional columns **timestamp** and **location** "
            "are shown in the log if present.")
    up = st.file_uploader("Choose CSV file", type=['csv'], key='al_upload')

    if up:
        try:
            df = E.read_csv(up)
        except Exception as e:
            st.error(f"❌ Could not read the file: {e}")
            df = None
        if df is not None:
            ok, msg = E.check_water_columns(df)
            if not ok:
                st.error("❌ " + msg)
            else:
                st.success(f"✅ Loaded {len(df)} samples.")
                if st.button("🚨 Run Early Warning", type="primary", **E.FULL):
                    with st.spinner("Running classification + 4 anomaly detectors + alert fusion..."):
                        res = E.early_warning(df)
                    add_to_log(df, res, up.name)
                    st.session_state['al_last'] = (up.name, len(df))

    log = st.session_state.get('alert_log')
    if log is None or log.empty:
        st.markdown("---")
        st.info("No alerts yet. Upload a file and click **Run Early Warning**. Results are kept here "
                "for this session so you can compare several files.")
        return

    show_log(log)


def add_to_log(df, res, fname):
    run_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n = len(df)
    ts = df['timestamp'].astype(str).values if 'timestamp' in df.columns else [run_time] * n
    loc = df['location'].astype(str).values if 'location' in df.columns else [fname] * n

    def trig(row):
        parts = []
        if row['cls_level'] > 0:
            parts.append(f"classification:{E.LEVEL_NAMES[row['cls_level']]}")
        if row['an_level'] > 0:
            parts.append(f"anomaly:{E.LEVEL_NAMES[row['an_level']]} ({row['votes']}/4)")
        return ", ".join(parts) if parts else "—"

    entry = pd.DataFrame({
        'Run': run_time,
        'Timestamp': ts,
        'Location': loc,
        'Sample #': np.arange(1, n + 1),
        'pH': df['ph'].round(2).values,
        'P(potable)': res['p_potable'].round(3).values,
        'Classification': [E.LEVEL_NAMES[v] for v in res['cls_level']],
        'Anomaly votes': res['votes'].values,
        'Anomaly': [E.LEVEL_NAMES[v] for v in res['an_level']],
        'Risk score': res['risk'].round(3).values,
        'Level': res['Alert Level'].values,
        'Triggered by': [trig(r) for _, r in res.iterrows()],
    })
    old = st.session_state.get('alert_log')
    st.session_state['alert_log'] = entry if old is None else pd.concat([entry, old], ignore_index=True)


def show_log(log):
    st.markdown("---")
    last = st.session_state.get('al_last')
    runs = log['Run'].unique()

    c_sel, c_clear = st.columns([3, 1])
    with c_sel:
        which = st.selectbox("Show", ["Latest run"] + (["All runs this session"] if len(runs) > 1 else []))
    with c_clear:
        st.write("")
        if st.button("🗑️ Clear history", **E.FULL):
            st.session_state.pop('alert_log', None)
            st.rerun()

    df = log[log['Run'] == runs[0]] if which == "Latest run" else log
    if which == "Latest run" and last:
        st.caption(f"File: **{last[0]}** · {last[1]} samples · run at {runs[0]}")

    st.markdown("### 📊 Final Alert Levels")
    n = len(df)
    cols = st.columns(4)
    for col, name in zip(cols, E.LEVEL_NAMES):
        k = int((df['Level'] == name).sum())
        col.metric(f"{E.LEVEL_EMOJI[name]} {name}", k, f"{k / n:.1%}" if n else "0%", delta_color="off")

    left, right = st.columns(2)
    with left:
        counts = [int((df['Level'] == L).sum()) for L in E.LEVEL_NAMES]
        fig = go.Figure(go.Bar(x=E.LEVEL_NAMES, y=counts, marker_color=[COLORS[L] for L in E.LEVEL_NAMES],
                               text=counts, textposition='outside'))
        fig.update_layout(title={'text': "Samples per alert level", 'font': {'color': '#E8871E'}}, height=360,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                          yaxis=dict(gridcolor='rgba(255,255,255,0.1)'), margin=dict(t=50))
        st.plotly_chart(fig, **E.FULL)
    with right:
        fig2 = go.Figure(go.Histogram(x=df['Risk score'], nbinsx=12, marker_color='#1C7293'))
        fig2.update_layout(title={'text': "Risk score distribution", 'font': {'color': '#E8871E'}}, height=360,
                           xaxis=dict(title="Risk score (0 = safe, 1 = all modules critical)", range=[0, 1],
                                      gridcolor='rgba(255,255,255,0.1)'),
                           yaxis=dict(title="Samples", gridcolor='rgba(255,255,255,0.1)'),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                           margin=dict(t=50))
        st.plotly_chart(fig2, **E.FULL)

    st.markdown("### 📋 Alert Log")
    flt = st.multiselect("Filter by level", E.LEVEL_NAMES, default=E.LEVEL_NAMES)
    view = df[df['Level'].isin(flt)]

    def color_rows(row):
        c = {'CRITICAL': 'rgba(231, 76, 60, 0.3)', 'MEDIUM': 'rgba(230, 126, 34, 0.3)',
             'LOW': 'rgba(233, 185, 73, 0.3)', 'NORMAL': 'rgba(46, 204, 113, 0.15)'}[row['Level']]
        return [f'background: {c}'] * len(row)

    fmt = {'pH': '{:.2f}', 'P(potable)': '{:.3f}', 'Risk score': '{:.3f}'}
    st.dataframe(view.style.apply(color_rows, axis=1).format(fmt, na_rep='missing'),
                 height=420, hide_index=True, **E.FULL)
    st.caption("Read across a row: what the classifier said, how many anomaly detectors agreed, "
               "and the final fused level. 'Triggered by' lists the modules that raised a concern.")
    st.download_button("📥 Download Alert Log", data=view.to_csv(index=False),
                       file_name=f"alert_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv", **E.FULL)


if __name__ == "__main__":
    show()
