@echo off
echo ================================================================
echo 🚀 LAW FIRM RAG - DOCUMENT PREPROCESSING SYSTEM
echo ================================================================
echo.
echo This script will preprocess all your PDF documents for instant
echo app startup. After completion, your Streamlit app will start
echo in just 1-2 seconds instead of 30-60 seconds!
echo.
echo 📂 Documents to process:
echo    - data/general folder/*.pdf  (General Legal Knowledge)
echo    - data/*.pdf                 (Company Documents)
echo    - data/companydocs/*.pdf     (Company Documents)
echo.
echo ⏱️  Estimated time: 2-5 minutes
echo.
echo Press any key to start preprocessing...
pause >nul

echo.
echo ================================================
echo 🔧 ACTIVATING PYTHON ENVIRONMENT
echo ================================================
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Error: Could not activate Python environment
    echo 💡 Make sure you have a .venv folder with the required packages
    pause
    exit /b 1
)

echo.
echo ================================================
echo 📚 STARTING DOCUMENT PREPROCESSING
echo ================================================
python preprocessing_script.py

if errorlevel 1 (
    echo.
    echo ❌ PREPROCESSING FAILED!
    echo 💡 Common issues:
    echo    - Ollama not running: ollama serve
    echo    - Missing model: ollama pull nomic-embed-text
    echo    - No PDF files in data folders
    pause
    exit /b 1
)

echo.
echo ================================================================
echo 🎉 PREPROCESSING COMPLETED SUCCESSFULLY!
echo ================================================================
echo.
echo ✅ All documents have been preprocessed and stored in ChromaDB
echo ⚡ Your app will now start instantly!
echo.
echo 🚀 Ready to run: .\lightning_start.bat
echo.
echo Press any key to exit...
pause >nul
