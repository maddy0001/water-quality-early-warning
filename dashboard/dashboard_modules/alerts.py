"""
Alert History Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show():
    """Display alert history"""
    
    st.markdown("## 🚨 Alert History & Analysis")
    
    st.markdown("""
    <div class='info-box'>
        <p>Track water quality alerts over time and analyze patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        time_range = st.selectbox(
            "📅 Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=1
        )
    
    with col2:
        alert_filter = st.multiselect(
            "⚠️ Filter Alerts",
            ["NORMAL", "LOW", "MEDIUM", "CRITICAL"],
            default=["LOW", "MEDIUM", "CRITICAL"]
        )
    
    # Generate demo data
    df = generate_demo_alerts(time_range)
    
    if alert_filter:
        df = df[df['Level'].isin(alert_filter)]
    
    # Summary
    st.markdown("### 📊 Summary")
    
    level_counts = df['Level'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        count = level_counts.get('NORMAL', 0)
        st.metric("🟢 NORMAL", count, f"{count/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    
    with col2:
        count = level_counts.get('LOW', 0)
        st.metric("🟡 LOW", count, f"{count/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    
    with col3:
        count = level_counts.get('MEDIUM', 0)
        st.metric("🟠 MEDIUM", count, f"{count/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    
    with col4:
        count = level_counts.get('CRITICAL', 0)
        st.metric("🔴 CRITICAL", count, f"{count/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    
    # Timeline
    st.markdown("---")
    st.markdown("### 📅 Alert Timeline")
    
    fig = go.Figure()
    
    colors = {'NORMAL': '#10b981', 'LOW': '#fbbf24', 'MEDIUM': '#f97316', 'CRITICAL': '#ef4444'}
    
    for level in ['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL']:
        level_df = df[df['Level'] == level]
        if len(level_df) > 0:
            fig.add_trace(go.Scatter(
                x=level_df['Timestamp'],
                y=[level] * len(level_df),
                mode='markers',
                name=level,
                marker=dict(size=12, color=colors[level]),
                text=level_df['Location'],
                hovertemplate='<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>'
            ))
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Alert Level",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            categoryorder='array',
            categoryarray=['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL'],
            gridcolor='rgba(255,255,255,0.1)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed log
    st.markdown("---")
    st.markdown("### 📋 Detailed Log")
    
    display_df = df.copy()
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    def color_rows(row):
        level = row['Level']
        if level == 'CRITICAL':
            return ['background: rgba(239, 68, 68, 0.3)'] * len(row)
        elif level == 'MEDIUM':
            return ['background: rgba(249, 115, 22, 0.3)'] * len(row)
        elif level == 'LOW':
            return ['background: rgba(251, 191, 36, 0.3)'] * len(row)
        else:
            return ['background: rgba(16, 185, 129, 0.3)'] * len(row)
    
    st.dataframe(
        display_df.style.apply(color_rows, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download
    csv = display_df.to_csv(index=False)
    st.download_button(
        "📥 Download Alert History",
        data=csv,
        file_name=f"alert_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def generate_demo_alerts(time_range):
    """Generate demo alerts"""
    hours = {"Last 24 Hours": 24, "Last 7 Days": 168, "Last 30 Days": 720}[time_range]
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    np.random.seed(42)
    n_alerts = min(80, int(hours / 3))
    
    timestamps = [start_time + timedelta(hours=np.random.uniform(0, hours)) for _ in range(n_alerts)]
    timestamps.sort()
    
    levels = np.random.choice(
        ['NORMAL', 'LOW', 'MEDIUM', 'CRITICAL'],
        size=n_alerts,
        p=[0.50, 0.30, 0.15, 0.05]
    )
    
    locations = [f"Site_{chr(65 + i)}" for i in range(5)]
    location_list = np.random.choice(locations, size=n_alerts)
    
    messages = {
        'NORMAL': 'Water quality normal',
        'LOW': 'Minor deviation',
        'MEDIUM': 'Investigate now',
        'CRITICAL': 'Immediate action required'
    }
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Level': levels,
        'Location': location_list,
        'Message': [messages[l] for l in levels]
    })
    
    return df


if __name__ == "__main__":
    show()
