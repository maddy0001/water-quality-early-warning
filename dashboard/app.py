"""
Water Quality Monitoring Dashboard - M.Tech Project | IEEE ICESC 2026
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

# ============================================================
# BRAND PALETTE — same identity as the paper and slide deck
# Primary: deep ocean navy #0B3D5C · Secondary: teal #1C7293 · Accent: amber #E8871E
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Lora:wght@600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Lora', serif !important; }

    .stApp { background: linear-gradient(160deg, #0B3D5C 0%, #0A2E45 55%, #081F2F 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #081F2F 0%, #0B3D5C 100%); border-right: 2px solid #1C7293; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #E8F1F4; }

    .main-title { font-size: 2.3rem; font-weight: 700; font-family: 'Lora', serif;
                   color: #FFFFFF; text-align: center; padding: 1.6rem 0 0.4rem 0; }
    .main-subtitle { text-align: center; color: #E8871E; font-weight: 600; letter-spacing: 1.5px;
                      font-size: 0.85rem; text-transform: uppercase; margin-bottom: 1.6rem; }

    h1, h2, h3 { color: #FFFFFF !important; }
    h4, h5 { color: #CFE3ED !important; }

    .alert-normal   { background: linear-gradient(135deg, rgba(46, 204, 113, 0.18), rgba(46, 204, 113, 0.06)); border-left: 5px solid #2ECC71; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .alert-low      { background: linear-gradient(135deg, rgba(233, 185, 73, 0.20), rgba(233, 185, 73, 0.06)); border-left: 5px solid #E9B949; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .alert-medium   { background: linear-gradient(135deg, rgba(230, 126, 34, 0.22), rgba(230, 126, 34, 0.06)); border-left: 5px solid #E67E22; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .alert-critical { background: linear-gradient(135deg, rgba(231, 76, 60, 0.28), rgba(231, 76, 60, 0.08)); border-left: 5px solid #E74C3C; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { box-shadow: 0 0 18px rgba(231, 76, 60, 0.25); } 50% { box-shadow: 0 0 34px rgba(231, 76, 60, 0.5); } }

    .info-box    { background: rgba(28, 114, 147, 0.16); border-left: 4px solid #1C7293; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; color: #E8F1F4; }
    .explain-box { background: rgba(232, 135, 30, 0.10); border: 1px solid rgba(232, 135, 30, 0.35); padding: 1.2rem; border-radius: 10px; margin: 1rem 0; color: #F5DCC0; }
    .honest-box  { background: rgba(255, 255, 255, 0.05); border: 1px dashed rgba(207, 227, 237, 0.4); padding: 1.2rem; border-radius: 10px; margin: 1rem 0; color: #CFE3ED; }

    [data-testid="stMetricValue"] { font-size: 1.9rem; color: #E8871E; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #CFE3ED !important; }

    .stButton > button { background: linear-gradient(90deg, #1C7293, #0B3D5C); color: #FFFFFF; border: 1px solid #E8871E; border-radius: 8px; padding: 0.75rem 2rem; font-weight: 600; transition: all 0.25s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(232, 135, 30, 0.35); border-color: #FFFFFF; }

    [data-testid="stFileUploader"] { background: rgba(255, 255, 255, 0.04); border: 2px dashed rgba(28, 114, 147, 0.6); border-radius: 12px; padding: 2rem; }

    .stSuccess { background: rgba(46, 204, 113, 0.15); border-left: 4px solid #2ECC71; }
    .stError   { background: rgba(231, 76, 60, 0.15);  border-left: 4px solid #E74C3C; }
    .stWarning { background: rgba(233, 185, 73, 0.18); border-left: 4px solid #E9B949; }
    .stInfo    { background: rgba(28, 114, 147, 0.20); border-left: 4px solid #1C7293; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stTabs [data-baseweb="tab"] { background-color: rgba(255, 255, 255, 0.04); border-radius: 8px; padding: 10px 20px; color: #E8F1F4; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg, #1C7293, #0B3D5C); color: #FFFFFF; border: 1px solid #E8871E; }

    hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💧 Water Contamination Early Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Ensemble Classification · BiLSTM-Attention Forecasting · Multi-Detector Anomaly Detection</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Dashboard Home", "🔍 Water Classification", "📈 pH Forecasting", "⚠️ Anomaly Detection", "🚨 Alert History"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
from dashboard_modules import engine as _engine
_status = {
    "Classification": ['rf', 'xgb', 'svm', 'scaler'],
    "Forecasting": ['bilstm', 'f_scaler'],
    "Anomaly detection": ['if', 'ocsvm', 'ae', 'ae_thr', 'raw'],
}
_missing = [name for name, keys in _status.items() if _engine.missing_files(keys)]
if not _missing:
    st.sidebar.success("✅ **All trained models found**")
else:
    st.sidebar.error("❌ **Models missing:** " + ", ".join(_missing) +
                     "\n\nRun `python main.py --train` in the project folder.")
st.sidebar.info("""
**Verified Performance** *(n = 570 test set)*
- Classification (Ensemble): **61.75%** acc · **66.62%** AUC
- Forecasting (BiLSTM): **R² 0.5251**
- Anomaly flag rate: **4.21%** (120 / 2,847)
- Processing: **< 100 ms**/sample
""")
st.sidebar.caption("Anomaly rate is a consistency check against an assumed 5% contamination prior — not a validated precision/recall claim (no ground-truth labels exist for this module).")

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
<div style='text-align: center; color: #CFE3ED;'>
    <p style='font-weight: 600; color: #E8871E;'>AI-Powered Monitoring</p>
    <p style='font-size: 0.82rem; color: #7FA8BD;'>M.Tech Project · ICESC 2026 (IEEE)</p>
    <p style='font-size: 0.7rem; color: #5C7D8F;'>Paper ID: ICESC_469</p>
</div>
""", unsafe_allow_html=True)
