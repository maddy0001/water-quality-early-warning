# 📄 PDF REPORTS - QUICK START GUIDE

## ⚡ 3-STEP SETUP:

### Step 1: Add Report Files
```
water_contamination_early_warning_system/
├── reports/                      ← CREATE THIS FOLDER
│   ├── __init__.py              ← Copy all files here
│   ├── pdf_generator.py         ← Main generator
│   ├── generate_report.py       ← Automated pipeline
│   └── requirements_reports.txt
│
├── generate_report.bat           ← Windows script (root)
└── generate_report.sh            ← Linux/Mac script (root)
```

### Step 2: Install Dependencies
```bash
pip install reportlab
```

### Step 3: Generate Report!
```bash
# Windows: Double-click
generate_report.bat

# Linux/Mac:
chmod +x generate_report.sh
./generate_report.sh

# Or manually:
python -c "import sys; sys.path.insert(0, 'src'); from reports.generate_report import generate_report; generate_report()"
```

## 📊 What You'll Get:

**Professional PDF Report with:**
- ✅ Cover page with title & date
- ✅ Executive summary
- ✅ Classification results table
- ✅ Forecasting metrics table
- ✅ Alert system explanation
- ✅ Recommendations
- ✅ Conclusion
- ✅ Charts (if available)

**Saved to:** `outputs/reports/water_quality_report_YYYYMMDD_HHMMSS.pdf`

## 🎯 Usage Examples:

### Quick Report:
```python
from reports import generate_report
report_path = generate_report()
```

### Custom Title:
```python
report_path = generate_report(
    title="Monthly Water Quality Report - February 2026"
)
```

### Batch Reports:
```python
from reports import generate_batch_reports
reports = generate_batch_reports(['Site_A', 'Site_B', 'Site_C'])
```

### Executive Summary:
```python
from reports import generate_summary_report
summary = generate_summary_report()
```

## 📋 Before Generating:

Make sure you have metrics:
```bash
# Run evaluation first to generate metrics
python main.py --evaluate

# This creates: outputs/metrics/evaluation_summary.json
# Report generator uses these metrics automatically
```

## ✅ Verify Report:

1. Check file was created:
   - Location: `outputs/reports/`
   - File name: `water_quality_report_*.pdf`

2. Open PDF and verify:
   - Cover page appears
   - All sections present
   - Tables have data
   - Charts embedded (if you ran evaluation)

## 🐛 Quick Fixes:

**"No module named 'reportlab'"**
```bash
pip install reportlab
```

**"Cannot find metrics file"**
```bash
python main.py --evaluate  # Generate metrics first
```

**"Charts not appearing"**
```bash
python main.py --evaluate  # Regenerate charts
```

## 📊 Report Structure:

```
┌─────────────────────────────────────┐
│  COVER PAGE                         │
│  - Title, date, author              │
├─────────────────────────────────────┤
│  EXECUTIVE SUMMARY                  │
│  - Key findings                     │
│  - Performance overview             │
├─────────────────────────────────────┤
│  CLASSIFICATION RESULTS             │
│  - Model comparison table           │
│  - Performance charts               │
├─────────────────────────────────────┤
│  FORECASTING RESULTS                │
│  - Metrics table                    │
│  - Prediction charts                │
├─────────────────────────────────────┤
│  ALERT STATISTICS                   │
│  - 4-level system explanation       │
├─────────────────────────────────────┤
│  RECOMMENDATIONS                    │
│  - Best practices                   │
│  - Maintenance tips                 │
├─────────────────────────────────────┤
│  CONCLUSION                         │
│  - Summary & future work            │
└─────────────────────────────────────┘
```

## 🎓 For IEEE Paper:

Include report generation in your paper:
> "An automated PDF report generation system produces comprehensive 
> documentation of analysis results, including performance metrics,
> visualizations, and actionable recommendations for stakeholders."

Done! 🎉
