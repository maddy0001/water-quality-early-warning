# 🚀 DASHBOARD - QUICK INSTALLATION GUIDE

## ⚡ 3-STEP SETUP:

### Step 1: Add Dashboard Files
```
water_contamination_early_warning_system/
├── dashboard/                     ← CREATE THIS FOLDER
│   ├── __init__.py               ← Copy all files here
│   ├── app.py                    ← Main dashboard
│   ├── requirements_dashboard.txt
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── home.py
│   │   ├── classification.py
│   │   ├── forecasting.py
│   │   ├── anomaly.py
│   │   └── alerts.py
│
├── run_dashboard.bat             ← Windows launch script (root)
└── run_dashboard.sh              ← Linux/Mac launch script (root)
```

### Step 2: Install Dependencies
```bash
pip install streamlit plotly
```

### Step 3: Launch!
```bash
# Windows: Double-click
run_dashboard.bat

# Linux/Mac:
chmod +x run_dashboard.sh
./run_dashboard.sh

# Or manually:
streamlit run dashboard/app.py
```

## 🌐 Access Dashboard:
Opens automatically at: **http://localhost:8501**

## ✅ What You'll See:
- 🏠 Home: System overview
- 🔍 Classification: Upload samples → Get 4-level alerts
- 📈 Forecasting: pH predictions
- ⚠️ Anomaly: Detect contamination
- 🚨 History: Track past alerts

## 🎯 Try It:
1. Go to "🔍 Classification"
2. Enter water quality parameters manually
3. Click "Classify"
4. See instant alert: 🟢 NORMAL / 🟡 LOW / 🟠 MEDIUM / 🔴 CRITICAL

## 📊 Features:
✅ Real-time predictions
✅ Color-coded alerts
✅ CSV batch upload
✅ Interactive charts
✅ Alert history
✅ Location tracking
✅ Export results

## ⏱️ Performance:
- Model loading: ~2-3 seconds (once per session)
- Prediction: <100ms per sample
- Batch (100 samples): ~2 seconds

Done! 🎉
