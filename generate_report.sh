#!/bin/bash
################################################################################
#     Water Quality PDF Report Generator - Linux/Mac
################################################################################

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║              WATER QUALITY PDF REPORT GENERATOR                      ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if reportlab is installed
python3 -c "import reportlab" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] ReportLab is not installed!"
    echo ""
    echo "Installing PDF report dependencies..."
    pip3 install reportlab Pillow
    if [ $? -ne 0 ]; then
        echo "[ERROR] Installation failed!"
        exit 1
    fi
fi

echo "[INFO] Generating PDF report..."
echo ""

# Generate report
python3 -c "import sys; sys.path.insert(0, 'src'); from reports.generate_report import generate_report; generate_report()"

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Report generated successfully!"
    echo "[INFO] Check outputs/reports/ folder"
    echo ""
else
    echo ""
    echo "[ERROR] Report generation failed!"
    echo ""
fi
