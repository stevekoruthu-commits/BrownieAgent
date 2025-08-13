@echo off
echo 🚀 Setting up Git repository for BrownieAgent...

REM Initialize Git repository
git init                    # Creates a new Git repository in your folder

REM Add all files (respecting .gitignore)
git add .                   # Adds ALL files (except those in .gitignore)

REM Commit everything
git commit -m "Initial commit: Adaptive RAG Chatbot with offline LLM support"  # Saves all files with a commit message

echo ✅ Local Git repository ready!
echo.
echo 📋 Next steps:
echo 1. Go to github.com and create a new repository named 'BrownieAgent'
echo 2. Copy the repository URL
echo 3. Run: git remote add origin YOUR_REPO_URL
echo 4. Run: git push -u origin main
echo.
pause
