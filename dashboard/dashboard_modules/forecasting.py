"""
pH Forecasting Page - real trained BiLSTM + Attention model
(30-day window of 11 USGS sensor readings -> next-day normalised pH)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from . import engine as E


def show():
    st.markdown("## 📈 pH Forecasting & Trend Analysis")

    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0;'>🎯 What This Does</h3>
        <p>Predicts <b>next-day pH</b> at a monitoring station from the <b>last 30 days</b> of its sensor readings,
        using the trained <b>BiLSTM + Attention</b> model.</p>
        <ul>
            <li><b>Data:</b> USGS monitoring stations, Georgia (37 stations, 11 daily sensor features)</li>
            <li><b>Input:</b> 30 consecutive days × 11 features (conductance, pH max/min, dissolved oxygen, temperature)</li>
            <li><b>Output:</b> next-day pH on the dataset's <b>normalised 0–1 scale</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🤖 Published Performance *(9,324 test sequences)*
        **BiLSTM + Attention:** RMSE 0.0524 · MAE 0.0336 · R² 0.5251 · MAPE 19.36%

        **GRU baseline:** RMSE 0.0610 · R² 0.3572 → BiLSTM improves RMSE by 14.0% and R² by 47.0%.

        **R² = 0.525** means the model explains about half of the day-to-day change in pH.
        """)
    with col2:
        st.markdown("""
        ### 📏 About the scale
        The USGS dataset is distributed **already normalised to 0–1**, and the paper reports
        errors on this scale (RMSE 0.0524). The dashboard therefore shows pH on the same 0–1 scale.
        WHO limits (6.5–8.5) apply to raw pH, so they are not drawn here; sudden jumps are the job of
        the anomaly module.
        """)

    st.markdown("---")
    if not E.show_missing(['bilstm', 'f_scaler'], "forecasting"):
        return

    st.markdown("### 📤 Choose Data")
    source = st.radio("Data source", ["Use a USGS station from the test set", "Upload my own CSV"],
                      horizontal=True, label_visibility="collapsed")

    df = None
    if source.startswith("Use"):
        stations = load_test_stations()
        if stations is None:
            st.error("❌ `data/raw/water_dataset.mat` not found. Upload a CSV instead.")
            return
        ids = list(stations.keys())
        sid = st.selectbox("Station", ids, format_func=lambda s: f"USGS station {s}")
        days = st.slider("Days to use (test period)", 31, len(stations[sid]), len(stations[sid]))
        df = stations[sid].iloc[:days].reset_index(drop=True)
        with st.expander("📏 Check against the paper: run the model on all 37 stations"):
            st.caption("Runs every 30-day window of the whole test period for all 37 stations "
                       "(9,324 forecasts), the same test set as the published results.")
            if st.button("Run on all 37 stations", **E.FULL):
                with st.spinner("Forecasting 9,324 windows..."):
                    st.session_state['fc_all'] = evaluate_all(stations)
            if 'fc_all' in st.session_state:
                m = st.session_state['fc_all']
                a, b, c, d = st.columns(4)
                a.metric("Forecasts", f"{m['n']:,}")
                b.metric("RMSE", f"{m['rmse']:.4f}", f"paper 0.0524", delta_color="off")
                c.metric("MAE", f"{m['mae']:.4f}", f"paper 0.0336", delta_color="off")
                d.metric("R²", f"{m['r2']:.3f}", f"paper 0.5251", delta_color="off")
    else:
        st.info("💡 The CSV needs at least 30 rows (one per day, oldest first) and these 11 columns: "
                + ", ".join(E.FORECAST_FEATURES)
                + ". An optional **pH_actual** column lets the page compare forecasts with real values.")
        up = st.file_uploader("Choose CSV file", type=['csv'], key='fc_upload')
        if not up:
            return
        try:
            df = E.read_csv(up)
        except Exception as e:
            st.error(f"❌ Could not read the file: {e}")
            return
        ok, msg = E.check_forecast_columns(df)
        if not ok:
            st.error("❌ " + msg)
            if all(c in df.columns for c in E.FEATURES):
                st.info("This file has the 9 Kaggle water-quality columns. Those samples have no time order "
                        "and no sensor history, so they go on the **🔍 Classification** or "
                        "**⚠️ Anomaly Detection** pages. The forecaster needs daily USGS sensor readings.")
            return
        st.success(f"✅ Loaded {len(df)} days.")
        with st.expander("📋 View your data (first 10 rows)"):
            st.dataframe(df.head(10), **E.FULL)

    with st.spinner("Running the BiLSTM on every 30-day window..."):
        pred, day_idx = E.forecast(df)
    show_results(df, pred, day_idx)


@st.cache_data(show_spinner=False)
def load_test_stations():
    """Test-period data (282 days) for each of the 37 USGS stations."""
    path = E.ROOT / 'data' / 'raw' / 'water_dataset.mat'
    if not path.exists():
        return None
    from scipy.io import loadmat
    d = loadmat(str(path))
    X = np.stack(d['X_te'][0])          # (days, stations, 11)
    Y = d['Y_te']                       # (stations, days)
    ids = [str(int(v[0])) for v in d['location_ids']]
    out = {}
    for s, sid in enumerate(ids):
        df = pd.DataFrame(X[:, s, :], columns=E.FORECAST_FEATURES)
        df.insert(0, 'day', np.arange(1, len(df) + 1))
        df['pH_actual'] = Y[s]
        out[sid] = df
    return out


def evaluate_all(stations):
    errs, ys = [], []
    for df in stations.values():
        pred, _ = E.forecast(df)
        y = E.scale_actual_ph(df['pH_actual'].values)[E.LOOKBACK:]
        errs.append(pred[:-1] - y)
        ys.append(y)
    e, y = np.concatenate(errs), np.concatenate(ys)
    return {'n': len(e), 'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
            'r2': float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


def show_results(df, pred, day_idx):
    n = len(df)
    has_actual = 'pH_actual' in df.columns and df['pH_actual'].notna().all()
    next_day = pred[-1]                 # forecast for the day after the last row
    fc_days = day_idx + 1               # 1-based day numbers the forecasts are for

    st.markdown("---")
    st.markdown("## 🔮 Forecast Results")

    fig = go.Figure()
    if has_actual:
        actual = E.scale_actual_ph(df['pH_actual'].values)
        fig.add_trace(go.Scatter(x=np.arange(1, n + 1), y=actual, mode='lines+markers', name='Actual pH',
                                 line=dict(color='#1C7293', width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=fc_days[:-1], y=pred[:-1], mode='lines', name='BiLSTM forecast (1 day ahead)',
                             line=dict(color='#E8871E', width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(x=[fc_days[-1]], y=[next_day], mode='markers', name='Next-day forecast',
                             marker=dict(color='#E8871E', size=16, symbol='diamond',
                                         line=dict(color='white', width=2))))
    fig.add_vrect(x0=0.5, x1=E.LOOKBACK + 0.5, fillcolor='white', opacity=0.05, line_width=0,
                  annotation_text="first 30 days: history only", annotation_position="top left")
    fig.update_layout(
        title={'text': "Actual vs Forecast pH (normalised 0–1 scale)", 'font': {'size': 20, 'color': '#E8871E'}},
        xaxis_title="Day", yaxis_title="pH (normalised)", height=500, hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', y=-0.18))
    st.plotly_chart(fig, **E.FULL)
    st.caption("How to read: blue = what really happened; orange dashes = what the model predicted for each day "
               "using only the 30 days before it. The closer the two lines, the better. "
               "The diamond is the forecast for the day after your data ends.")

    last = df.iloc[-E.LOOKBACK:]
    if has_actual:
        last_actual = E.scale_actual_ph(last['pH_actual'].values)
        lo, hi, cur = float(last_actual.min()), float(last_actual.max()), float(last_actual[-1])
    else:
        cur, lo, hi = float(pred[-2]) if len(pred) > 1 else float(next_day), None, None

    c1, c2, c3 = st.columns(3)
    c1.metric("Next-day forecast", f"{next_day:.3f}", f"{next_day - cur:+.3f} vs last day")
    c2.metric("Forecasts made", len(pred) - 1 if has_actual else len(pred))
    c3.metric("Published RMSE", "0.0524", help="On 9,324 held-out test sequences")

    if has_actual and len(pred) > 1:
        y_true = E.scale_actual_ph(df['pH_actual'].values)[E.LOOKBACK:]
        y_hat = pred[:-1]
        err = y_hat - y_true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - float(np.sum(err ** 2)) / ss if ss > 0 else float('nan')
        st.markdown("### 🎯 Accuracy on this data")
        d1, d2, d3 = st.columns(3)
        d1.metric("RMSE (lower is better)", f"{rmse:.4f}", f"{rmse - 0.0524:+.4f} vs paper", delta_color="inverse")
        d2.metric("MAE (lower is better)", f"{mae:.4f}", f"{mae - 0.0336:+.4f} vs paper", delta_color="inverse")
        d3.metric("R² (higher is better)", f"{r2:.3f}" if r2 == r2 else "n/a")
        st.caption("Judge one station by RMSE and MAE: they are on the same scale as the paper. "
                   "R² compares the error with how much pH varies; within one station pH varies very little, "
                   "so R² is usually low here (median ≈ 0.05 across stations) even when the error is small. "
                   "The published R² (0.525) is over all 37 stations together; use the 'all 37 stations' check above.")

    if lo is not None:
        if next_day < lo - 0.02 or next_day > hi + 0.02:
            st.markdown(f"""
            <div class='alert-medium'><h4>🟠 Trend note</h4>
            <p>Tomorrow's forecast (<b>{next_day:.3f}</b>) is outside this station's range over the last 30 days
            ({lo:.3f} – {hi:.3f}). pH may be drifting; check the next readings.</p></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-normal'><h4>✅ Trend note</h4>
            <p>Tomorrow's forecast (<b>{next_day:.3f}</b>) is within this station's last-30-day range
            ({lo:.3f} – {hi:.3f}). No drift expected.</p></div>""", unsafe_allow_html=True)

    out = pd.DataFrame({'day': fc_days, 'forecast_pH_normalised': np.round(pred, 4)})
    if has_actual:
        act = E.scale_actual_ph(df['pH_actual'].values)
        out['actual_pH_normalised'] = [round(float(act[d - 1]), 4) if d <= n else None for d in fc_days]
    st.download_button("📥 Download Forecasts", data=out.to_csv(index=False),
                       file_name="ph_forecasts.csv", mime="text/csv", **E.FULL)


if __name__ == "__main__":
    show()
