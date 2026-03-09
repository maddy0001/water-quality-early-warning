#!/bin/bash
################################################################################
#     Water Quality Monitoring Dashboard - Launch Script (Linux/Mac)
################################################################################

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║        WATER CONTAMINATION EARLY WARNING SYSTEM - DASHBOARD         ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Streamlit is installed
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] Streamlit is not installed!"
    echo ""
    echo "Installing dashboard requirements..."
    pip3 install -r dashboard/requirements_dashboard.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Installation failed!"
        exit 1
    fi
fi

echo "[INFO] Starting dashboard..."
echo "[INFO] Dashboard will open in your default browser"
echo "[INFO] Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run dashboard/app.py
