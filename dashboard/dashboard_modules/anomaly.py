"""
Anomaly Detection Page - real trained detectors with 2-of-4 quorum voting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from . import engine as E

DETECTORS = ['Isolation Forest', 'LOF', 'One-Class SVM', 'Autoencoder']


def show():
    st.markdown("## ⚠️ Anomaly Detection")

    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0;'>🎯 What This Does</h3>
        <p>Four structurally different detectors look for <b>unusual</b> water samples, i.e. samples that do not
        look like the 2,847 cleaned samples the detectors were trained on:</p>
        <ul>
            <li><b>Isolation Forest</b> (contamination = 0.05): unusual samples are isolated by random splits quickly</li>
            <li><b>LOF</b> (k = 20): sample sits in a much emptier region than its neighbours</li>
            <li><b>One-Class SVM</b> (RBF, ν = 0.05): sample falls outside the learned boundary of normal data</li>
            <li><b>Autoencoder</b>: sample is rebuilt badly (error above the 95th-percentile threshold)</li>
        </ul>
        <p><b>Quorum voting:</b> a sample is an <b>anomaly</b> only if <b>at least 2 of the 4 detectors</b> agree.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='honest-box'>
        <b>Note:</b> there are no ground-truth contamination labels for this dataset, so precision and recall
        cannot be measured directly. The published check is the <b>flag rate: 4.21%</b> (120 of 2,847 samples)
        against an assumed 5% contamination rate, plus the synthetic contamination test (75.6% of injected
        spikes caught, 5.4% false alarms).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔍 Detection Method
        - 4 independent detectors, each votes Normal (0) or Anomaly (1)
        - **≥ 2 votes → Anomaly** (quorum)
        - Alert level from votes: 2 → LOW · 3 → MEDIUM · 4 → CRITICAL

        **Why:** each method has blind spots; two different methods rarely agree by chance,
        so requiring agreement halves false alarms.
        """)
    with col2:
        st.markdown("""
        ### 📊 Published Results *(2,847 samples)*
        - Flagged as anomaly (≥ 2 votes): **120 (4.21%)**
        - 3-of-4 agreement: 25 (0.88%)
        - 4-of-4 (unanimous): 4 (0.14%)
        - No detector flagged: 2,570 (90.27%)
        """)

    st.markdown("---")
    if not E.show_missing(['if', 'ocsvm', 'ae', 'ae_thr', 'scaler', 'raw'], "anomaly detection"):
        return

    st.markdown("### 📤 Step 1: Upload Your Data")
    st.info("💡 The CSV needs the 9 water-quality columns (extra columns are ignored): " + ", ".join(E.FEATURES))
    uploaded = st.file_uploader("Choose CSV file", type=['csv'], key='an_upload')

    if not uploaded:
        st.session_state.pop('an_results', None)
        st.markdown("---")
        show_illustration()
        return

    try:
        df = E.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")
        return
    ok, msg = E.check_water_columns(df)
    if not ok:
        st.error("❌ " + msg)
        if any(c in df.columns for c in E.FORECAST_FEATURES):
            st.info("This looks like a forecasting file. Upload it on the **📈 pH Forecasting** page instead.")
        return

    st.success(f"✅ Loaded {len(df)} samples.")
    with st.expander("📋 View your data (first 10 rows)"):
        st.dataframe(df.head(10), **E.FULL)

    st.markdown("### 📊 Step 2: Run Detection")
    key = (uploaded.name, uploaded.size)
    if st.button("🔍 Detect Anomalies", type="primary", **E.FULL):
        with st.spinner("Loading the 4 detectors and scoring every sample... (first run takes ~10 s)"):
            st.session_state['an_results'] = (key, E.detect(df))

    stored = st.session_state.get('an_results')
    if stored and stored[0] == key:
        display_results(stored[1], df)


def display_results(res, df):
    votes = res['votes'].values
    pred = res['anomaly'].values
    n = len(df)

    st.markdown("---")
    st.markdown("## 📊 Detection Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Normal (0–1 votes)", int((pred == 0).sum()), f"{(pred == 0).mean():.1%}", delta_color="off")
    c2.metric("🔴 Anomalies (≥ 2 votes)", int(pred.sum()), f"{pred.mean():.1%}", delta_color="off")
    c3.metric("⚠️ Strong (3 votes)", int((votes == 3).sum()), f"{(votes == 3).mean():.1%}", delta_color="off")
    c4.metric("🚨 Unanimous (4 votes)", int((votes == 4).sum()), f"{(votes == 4).mean():.1%}", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.markdown("### 📈 How many detectors flagged each sample?")
        counts = [int((votes == k).sum()) for k in range(5)]
        colors = ['#2ECC71', '#E9B949', '#E67E22', '#E74C3C', '#7f1d1d']
        fig = go.Figure(go.Bar(x=[f"{k} votes" for k in range(5)], y=counts, marker_color=colors,
                               text=counts, textposition='outside'))
        fig.add_vline(x=1.5, line_dash='dash', line_color='white',
                      annotation_text='quorum: anomaly from here →', annotation_position='top right')
        fig.update_layout(height=380, xaxis_title="Detectors that flagged the sample", yaxis_title="Samples",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                          yaxis=dict(gridcolor='rgba(255,255,255,0.1)'), margin=dict(t=40))
        st.plotly_chart(fig, **E.FULL)
        st.caption("Bars right of the dashed line are anomalies. Most samples should sit at 0 votes.")
    with right:
        st.markdown("### 🔎 Flags by each detector")
        per = [int(res[d].sum()) for d in DETECTORS]
        fig2 = go.Figure(go.Bar(x=per, y=DETECTORS, orientation='h', marker_color='#1C7293',
                                text=[f"{p} ({p / n:.0%})" for p in per], textposition='outside'))
        fig2.update_layout(height=380, xaxis_title="Samples flagged", plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                           xaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, max(per + [1]) * 1.35]),
                           margin=dict(t=40))
        st.plotly_chart(fig2, **E.FULL)
        st.caption("Each detector alone flags roughly 5% of normal data, by design.")

    if pred.sum() > 0:
        st.markdown(f"""
        <div class='alert-medium'>
            <h4>⚠️ Interpretation</h4>
            <p><b>{int(pred.sum())} of {n}</b> samples ({pred.mean():.1%}) were flagged by at least 2 detectors.
            These samples look unlike the normal training data. Check them first.
            This is a screening signal, not a confirmed contamination.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='alert-normal'>
            <h4>✅ All clear</h4>
            <p>No sample reached the 2-of-4 quorum. All samples look consistent with normal water.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 All Samples")
    table = pd.DataFrame({'Sample #': np.arange(1, n + 1)})
    for d in DETECTORS:
        table[d] = np.where(res[d].values == 1, '🚩', '·')
    table['Votes'] = votes
    table['Result'] = np.where(pred == 1, '🔴 ANOMALY', '🟢 normal')
    table['Alert Level'] = [E.LEVEL_NAMES[v] for v in res['an_level']]
    table = pd.concat([table, df[E.FEATURES].reset_index(drop=True).round(2)], axis=1)

    only = st.checkbox("Show only anomalies (≥ 2 votes)", value=False)
    view = table[table['Votes'] >= 2] if only else table

    def color_by_votes(row):
        v = row['Votes']
        c = ('rgba(127, 29, 29, 0.5)' if v == 4 else 'rgba(231, 76, 60, 0.4)' if v == 3
             else 'rgba(230, 126, 34, 0.3)' if v == 2 else 'rgba(0,0,0,0)')
        return [f'background: {c}'] * len(row)

    st.dataframe(view.style.apply(color_by_votes, axis=1).format(precision=2, na_rep='missing'),
                 height=420, hide_index=True, **E.FULL)
    st.caption("🚩 = that detector flagged the sample. Votes = number of 🚩. ≥ 2 votes = anomaly.")
    st.download_button("📥 Download Results CSV", data=table.to_csv(index=False),
                       file_name="anomaly_results.csv", mime="text/csv", **E.FULL)


def show_illustration():
    st.markdown("### 📊 Illustration: How Anomaly Detection Works")
    st.markdown("""
    <div class='info-box'>
        <p>👆 <b>Upload a CSV file above to run the real detectors on your data.</b></p>
        <p>Below is a simplified 2-D picture of the idea (illustration only, not real data).</p>
    </div>
    """, unsafe_allow_html=True)

    rng = np.random.default_rng(42)
    nx, ny = rng.normal(0, 0.5, 150), rng.normal(0, 0.5, 150)
    ax = np.concatenate([rng.normal(2.5, 0.3, 8), rng.normal(-2.5, 0.3, 7)])
    ay = np.concatenate([rng.normal(2.5, 0.3, 8), rng.normal(-2.5, 0.3, 7)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nx, y=ny, mode='markers', name='Normal samples',
                             marker=dict(color='#2ECC71', size=9, opacity=0.75)))
    fig.add_trace(go.Scatter(x=ax, y=ay, mode='markers', name='Anomalies',
                             marker=dict(color='#E74C3C', size=14, symbol='x')))
    fig.update_layout(title={'text': "Simplified 2-D view (illustration)", 'font': {'size': 18, 'color': '#E8871E'}},
                      xaxis_title="Parameter 1", yaxis_title="Parameter 2", height=460,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=False),
                      yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=False))
    st.plotly_chart(fig, **E.FULL)
    st.caption("Green = the dense normal cluster. Red = isolated points far from it. "
               "The real system does this in all 9 dimensions at once.")


if __name__ == "__main__":
    show()
