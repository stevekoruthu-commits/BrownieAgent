@echo off
echo ⚡ Ultra-Fast RAG Launcher
cd /d "C:\Users\helna\OneDrive\Documents\BrownieAgent"
call .venv\Scripts\activate.bat

:: Start minimal services in background
start "Services" /MIN cmd /k "python service_manager.py"

:: Wait briefly and launch ultra-fast app
timeout /t 5 /nobreak > nul
streamlit run app_speed_test.py --server.headless true --server.runOnSave false --browser.gatherUsageStats false
