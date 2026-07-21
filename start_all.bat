@echo off
TITLE Synapse AI - Self-Healing RAG Launcher
COLOR 0A

echo ===================================================
echo   Starting Synapse AI Self-Healing RAG Stack...
echo ===================================================
echo.

:: 1. Check Python Venv
IF NOT EXIST "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    pause
    exit /b 1
)

:: 2. Check and start Ollama if not running on port 11434
echo Checking Ollama server (Port 11434)...
powershell -Command "$conn = Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue; if (-not $conn) { Start-Process '%LOCALAPPDATA%\Programs\Ollama\ollama.exe' -ArgumentList 'serve' -WindowStyle Hidden; Write-Host '[Ollama] Started background server.' } else { Write-Host '[Ollama] Server is already running.' }"

:: 3. Free port 8000 if in use by an old backend instance
echo Checking backend port 8000...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

:: 4. Check for npm / Node.js
where npm >nul 2>nul
if %errorlevel% neq 0 goto NO_NPM

echo [1/2] Launching FastAPI Backend (Port 8000)...
start "Synapse AI Backend (FastAPI)" cmd /k "cd /d "%~dp0" && venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Launching React Frontend (Port 5173)...
start "Synapse AI Frontend (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ===================================================
echo   Services are starting up!
echo   - Backend API: http://127.0.0.1:8000/docs
echo   - Frontend UI:  http://localhost:5173
echo ===================================================
echo.
pause
exit /b 0

:NO_NPM
echo ---------------------------------------------------
echo [NOTICE] 'npm' (Node.js) is not installed on PATH.
echo Node.js is required for the React Frontend.
echo Download Node.js from: https://nodejs.org/
echo ---------------------------------------------------
echo.
echo Launching FastAPI Backend Server...
echo API Docs: http://127.0.0.1:8000/docs
echo.
venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
pause
