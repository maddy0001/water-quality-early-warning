"""
Classification Page - real weighted soft-voting ensemble (RF 0.35 + XGBoost 0.40 + SVM 0.25)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from . import engine as E


def show():
    st.markdown("## 🔍 Water Quality Classification")

    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0;'>🎯 What This Does</h3>
        <p>Classifies water samples as <b>POTABLE (safe)</b> or <b>NOT POTABLE (unsafe)</b> using the trained
        weighted soft-voting ensemble.</p>
        <ul>
            <li><b>Model:</b> Random Forest (0.35) + XGBoost (0.40) + SVM (0.25). Each model gives a probability of
            "potable"; the weighted average is the ensemble probability. 0.5 or more = POTABLE.</li>
            <li><b>Published performance:</b> 61.75% accuracy · 66.62% ROC-AUC · 53.02% F1 (570 test samples)</li>
            <li><b>Input:</b> the 9 parameters of the Kaggle Water Potability dataset</li>
            <li><b>Output:</b> potability, confidence, and the classification alert level</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if not E.show_missing(['rf', 'xgb', 'svm', 'scaler'], "classification"):
        return

    st.markdown("---")
    tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Upload CSV File"])
    with tab1:
        show_manual_input()
    with tab2:
        show_csv_upload()


# --------------------------------------------------------------------------
def show_manual_input():
    st.markdown("### Enter Water Quality Parameters")
    st.info("💡 Defaults are typical values from the dataset. Change any value and click Classify.")

    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("💧 pH", 0.0, 14.0, 7.0, 0.1)
        hardness = st.number_input("🏔️ Hardness (mg/L)", 0.0, 500.0, 196.0, 1.0)
        solids = st.number_input("🌊 Solids (ppm)", 0.0, 70000.0, 22000.0, 100.0)
    with col2:
        chloramines = st.number_input("🧪 Chloramines (ppm)", 0.0, 15.0, 7.1, 0.1)
        sulfate = st.number_input("⚗️ Sulfate (mg/L)", 0.0, 600.0, 333.0, 1.0)
        conductivity = st.number_input("⚡ Conductivity (μS/cm)", 0.0, 1000.0, 426.0, 1.0)
    with col3:
        organic_carbon = st.number_input("🍃 Organic Carbon (ppm)", 0.0, 30.0, 14.3, 0.1)
        trihalomethanes = st.number_input("🔬 Trihalomethanes (μg/L)", 0.0, 130.0, 66.4, 0.1)
        turbidity = st.number_input("🌫️ Turbidity (NTU)", 0.0, 10.0, 4.0, 0.1)

    if st.button("🔍 Classify Water Sample", type="primary", **E.FULL):
        df = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                            organic_carbon, trihalomethanes, turbidity]], columns=E.FEATURES)
        with st.spinner("Running the ensemble..."):
            res = E.classify(df).iloc[0]
        display_single_result(res, df.iloc[0])


def display_single_result(res, sample):
    level = E.LEVEL_NAMES[int(res['cls_level'])]
    potable = int(res['prediction']) == 1
    css = {'NORMAL': 'alert-normal', 'LOW': 'alert-low', 'MEDIUM': 'alert-medium', 'CRITICAL': 'alert-critical'}[level]
    status = "POTABLE" if potable else "NOT POTABLE"
    msg = {
        'NORMAL': "The ensemble considers this sample safe.",
        'LOW': "The ensemble leans towards unsafe, but with low confidence (50–70%). Monitor.",
        'MEDIUM': "The ensemble is fairly confident (70–90%) the sample is unsafe. Investigate.",
        'CRITICAL': "The ensemble is very confident (90%+) the sample is unsafe. Act now.",
    }[level]

    st.markdown(f"""
    <div class='{css}'>
        <h2 style='margin: 0;'>{E.LEVEL_EMOJI[level]} {status} · Classification alert: {level}</h2>
        <p style='font-size: 1.15rem; margin: 0.8rem 0;'>{msg}</p>
        <p style='margin: 0;'><b>Ensemble probability of potable:</b> {res['p_potable']:.1%}
        &nbsp;·&nbsp; <b>Confidence:</b> {res['confidence']:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📊 What each model says")
        names = ['Random Forest (0.35)', 'XGBoost (0.40)', 'SVM (0.25)', 'Ensemble']
        vals = [res['p_random_forest'], res['p_xgboost'], res['p_svm'], res['p_potable']]
        colors = ['#1C7293', '#1C7293', '#1C7293', '#E8871E']
        fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=colors,
                               text=[f"{v:.1%}" for v in vals], textposition='outside'))
        fig.add_vline(x=0.5, line_dash='dash', line_color='white',
                      annotation_text='0.5 decision line', annotation_position='top')
        fig.update_layout(xaxis=dict(range=[0, 1], title='Probability of POTABLE', tickformat='.0%',
                                     gridcolor='rgba(255,255,255,0.1)'),
                          height=300, margin=dict(l=10, r=40, t=30, b=40),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, **E.FULL)
        st.caption("Bars right of the dashed line vote POTABLE; left of it vote NOT POTABLE.")
    with col2:
        st.markdown("### 🚦 How the alert level is set")
        st.markdown("""
        - Potable → **NORMAL**
        - Not potable, confidence 50–70% → **LOW**
        - Not potable, confidence 70–90% → **MEDIUM**
        - Not potable, confidence ≥ 90% → **CRITICAL**

        The final alert combines this with anomaly detection (see **Alert History**).
        """)

    with st.expander("📋 Entered parameters with WHO reference values"):
        params_df = pd.DataFrame({
            'Parameter': ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                          'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'],
            'Value': sample.values,
            'Unit': ['', 'mg/L', 'ppm', 'ppm', 'mg/L', 'μS/cm', 'ppm', 'μg/L', 'NTU'],
            'WHO Reference': ['6.5-8.5', '<500', '<500', '<4', '<250', '<400', '<2', '<80', '<5'],
        })
        st.dataframe(params_df, hide_index=True, **E.FULL)
        st.caption("WHO limits are shown for context only. The model learns from the dataset's labels, "
                   "which do not follow these limits (e.g. Solids in this dataset are typically ~22,000 ppm).")


# --------------------------------------------------------------------------
def show_csv_upload():
    st.markdown("### Upload Multiple Water Samples")
    st.info("💡 The CSV needs these 9 columns (extra columns are ignored): " + ", ".join(E.FEATURES))

    uploaded = st.file_uploader("Choose CSV file", type=['csv'], key='cls_upload')
    if not uploaded:
        st.session_state.pop('cls_results', None)
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
    n_missing = int(df[E.FEATURES].isna().sum().sum())
    if n_missing:
        st.warning(f"⚠️ {n_missing} empty cells found. They will be filled with KNN imputation (k = 5), "
                   "the same method used in training.")
    with st.expander("📋 View your data (first 10 rows)"):
        st.dataframe(df.head(10), **E.FULL)

    key = (uploaded.name, uploaded.size)
    if st.button("🔍 Classify All Samples", type="primary", **E.FULL):
        with st.spinner(f"Running the ensemble on {len(df)} samples..."):
            st.session_state['cls_results'] = (key, E.classify(df))

    stored = st.session_state.get('cls_results')
    if stored and stored[0] == key:
        display_batch_results(stored[1], df)


def display_batch_results(res, df):
    st.markdown("---")
    st.markdown("## 📊 Batch Analysis Results")

    n = len(res)
    n_pot = int(res['prediction'].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Potable", n_pot, f"{n_pot / n:.0%} of samples", delta_color="off")
    c2.metric("❌ Not potable", n - n_pot, f"{(n - n_pot) / n:.0%} of samples", delta_color="off")
    c3.metric("Average P(potable)", f"{res['p_potable'].mean():.1%}")

    levels = [E.LEVEL_NAMES[v] for v in res['cls_level']]
    st.markdown("#### Classification alert levels")
    cols = st.columns(4)
    for col, name in zip(cols, E.LEVEL_NAMES):
        k = levels.count(name)
        col.metric(f"{E.LEVEL_EMOJI[name]} {name}", k, f"{k / n:.0%}", delta_color="off")

    st.markdown("### 📋 Detailed Results")
    table = pd.DataFrame({
        'Sample': [f"Sample {i + 1}" for i in range(n)],
        'pH': df['ph'].round(2).values,
        'Result': ["✅ POTABLE" if p == 1 else "❌ NOT POTABLE" for p in res['prediction']],
        'P(potable) ensemble': res['p_potable'].round(3).values,
        'RF': res['p_random_forest'].round(3).values,
        'XGBoost': res['p_xgboost'].round(3).values,
        'SVM': res['p_svm'].round(3).values,
        'Confidence': [f"{c:.1%}" for c in res['confidence']],
        'Alert Level': levels,
    })

    # If the file carries the dataset's real label, show how many the model got right.
    if 'Potability' in df.columns and df['Potability'].isin([0, 1]).all():
        truth = df['Potability'].astype(int).values
        table['True label'] = ["POTABLE" if t == 1 else "NOT POTABLE" for t in truth]
        acc = float((truth == res['prediction'].values).mean())
        st.info(f"🎯 This file has the true 'Potability' label: the ensemble is correct on "
                f"{acc:.1%} of these {n} samples (published test accuracy: 61.75%).")

    def color_rows(row):
        c = {'CRITICAL': 'rgba(231, 76, 60, 0.3)', 'MEDIUM': 'rgba(230, 126, 34, 0.3)',
             'LOW': 'rgba(233, 185, 73, 0.3)', 'NORMAL': 'rgba(46, 204, 113, 0.18)'}[row['Alert Level']]
        return [f'background: {c}'] * len(row)

    fmt = {'pH': '{:.2f}', 'P(potable) ensemble': '{:.3f}', 'RF': '{:.3f}', 'XGBoost': '{:.3f}', 'SVM': '{:.3f}'}
    st.dataframe(table.style.apply(color_rows, axis=1).format(fmt, na_rep='missing'),
                 height=420, hide_index=True, **E.FULL)
    st.caption("P(potable) is the probability of SAFE water. Below 0.5 → NOT POTABLE. "
               "RF / XGBoost / SVM columns show each model's own probability before weighting.")

    st.download_button("📥 Download Results", data=table.to_csv(index=False),
                       file_name="classification_results.csv", mime="text/csv", **E.FULL)


if __name__ == "__main__":
    show()
