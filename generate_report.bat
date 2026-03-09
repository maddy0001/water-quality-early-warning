@echo off
REM ============================================================================
REM     Water Quality PDF Report Generator - Windows
REM ============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║              WATER QUALITY PDF REPORT GENERATOR                      ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

REM Check if reportlab is installed
python -c "import reportlab" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] ReportLab is not installed!
    echo.
    echo Installing PDF report dependencies...
    pip install reportlab Pillow
    echo.
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed!
        pause
        exit /b 1
    )
)

echo [INFO] Generating PDF report...
echo.

REM Generate report
python -c "import sys; sys.path.insert(0, 'src'); from reports.generate_report import generate_report; generate_report()"

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Report generated successfully!
    echo [INFO] Check outputs\reports\ folder
    echo.
) else (
    echo.
    echo [ERROR] Report generation failed!
    echo.
)

pause
