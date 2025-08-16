@echo off
echo Starting Law Firm RAG App...

cd /d "C:\Users\helna\OneDrive\Documents\BrownieAgent"
call .venv\Scripts\activate.bat

:: Check if services are already running
tasklist /FI "WINDOWTITLE eq RAG Services*" 2>NUL | find /I /N "cmd.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Services already running. Launching app...
    streamlit run app.py
) else (
    echo Starting services and app...
    start "RAG Services" /MIN cmd /k "python service_manager.py"
    timeout /t 8 /nobreak > nul
    streamlit run app.py
)
