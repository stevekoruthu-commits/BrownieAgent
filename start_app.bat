@echo off
echo ========================================
echo    Law Firm Document Analysis System
echo ========================================
echo.

cd /d "C:\Users\helna\OneDrive\Documents\BrownieAgent"
call .venv\Scripts\activate.bat

echo [1/3] Starting background services for optimal performance...
start "Legal Services" /MIN cmd /k "python service_manager.py"

echo [2/3] Initializing legal document system...
timeout /t 8 /nobreak > nul

echo [3/3] Launching law firm application...
echo.
echo ========================================
echo    Professional Legal Document Analysis
echo    Optimized for comprehensive case research
echo ========================================
echo.
streamlit run app_law_firm.py --server.port 8502

pause
