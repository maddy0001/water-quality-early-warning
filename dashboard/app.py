"""
Water Quality Monitoring Dashboard - M.Tech Project | IEEE 2026
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%); border-right: 2px solid #00d4ff; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: white; }
    .main-title { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 2rem 0; margin-bottom: 2rem; }
    h1, h2, h3 { color: #00d4ff !important; }
    .alert-normal { background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1)); border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .alert-low { background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.1)); border-left: 5px solid #fbbf24; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .alert-medium { background: linear-gradient(135deg, rgba(249, 115, 22, 0.2), rgba(249, 115, 22, 0.1)); border-left: 5px solid #f97316; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .alert-critical { background: linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.1)); border-left: 5px solid #ef4444; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); } 50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.6); } }
    .info-box { background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; color: white; }
    .explain-box { background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: #e9d5ff; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #00d4ff; }
    .stButton > button { background: linear-gradient(90deg, #00d4ff, #00ff88); color: #000; border: none; border-radius: 8px; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4); }
    [data-testid="stFileUploader"] { background: rgba(255, 255, 255, 0.05); border: 2px dashed rgba(0, 212, 255, 0.5); border-radius: 12px; padding: 2rem; }
    .stSuccess { background: rgba(16, 185, 129, 0.2); border-left: 4px solid #10b981; }
    .stError { background: rgba(239, 68, 68, 0.2); border-left: 4px solid #ef4444; }
    .stWarning { background: rgba(251, 191, 36, 0.2); border-left: 4px solid #fbbf24; }
    .stInfo { background: rgba(0, 212, 255, 0.2); border-left: 4px solid #00d4ff; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stTabs [data-baseweb="tab"] { background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px 20px; color: white; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg, #00d4ff, #00ff88); color: black; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💧 Water Contamination Early Warning System</div>', unsafe_allow_html=True)

# Clean sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Dashboard Home", "🔍 Water Classification", "📈 pH Forecasting", "⚠️ Anomaly Detection", "🚨 Alert History"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.success("✅ **All Systems Online**")
st.sidebar.info("""
**Performance:**
- Classification: 72.5%
- Forecasting: R² 0.52
- Anomaly Rate: 4.2%
- Processing: <100ms
""")

# Import modules
from dashboard_modules import home, classification, forecasting, anomaly, alerts

# Route
if page == "🏠 Dashboard Home":
    home.show()
elif page == "🔍 Water Classification":
    classification.show()
elif page == "📈 pH Forecasting":
    forecasting.show()
elif page == "⚠️ Anomaly Detection":
    anomaly.show()
elif page == "🚨 Alert History":
    alerts.show()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #00d4ff;'>
    <p style='font-weight: 600;'>AI-Powered Monitoring</p>
    <p style='font-size: 0.85rem; color: #888;'>M.Tech Project | IEEE 2026</p>
</div>
""", unsafe_allow_html=True)
