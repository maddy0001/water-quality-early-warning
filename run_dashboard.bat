@echo off
REM ============================================================================
REM     Water Quality Monitoring Dashboard - Launch Script (Windows)
REM ============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║        WATER CONTAMINATION EARLY WARNING SYSTEM - DASHBOARD         ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Streamlit is not installed!
    echo.
    echo Installing dashboard requirements...
    pip install -r dashboard\requirements_dashboard.txt
    echo.
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed!
        pause
        exit /b 1
    )
)

echo [INFO] Starting dashboard...
echo [INFO] Dashboard will open in your default browser
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Run Streamlit
streamlit run dashboard\app.py

pause
