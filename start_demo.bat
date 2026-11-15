@echo off
setlocal enabledelayedexpansion
title OptiSent-XAI Live Telemetry & Kinetic Visual Observatory (22ITI02)

echo ===============================================================================
echo   OptiSent-XAI: Explainable Multi-Class Product Review Triage & Calibration
echo   Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad (22ITI02)
echo   Navodita Infotech, Pune (Intern ID: 10110, Certificate ID: CNI2606698)
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Python not found in PATH. Opening standalone visual observatory...
    start "" "%~dp0index.html"
    pause
    exit /b 0
)

echo [2/3] Starting FastAPI Live Inference Kernel on http://127.0.0.1:8000 ...
start "OptiSent-XAI FastAPI Kernel (:8000)" cmd /k "title OptiSent-XAI Live Kernel & python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload"

echo [3/3] Opening Kinetic Visual Observatory in browser...
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000"

echo.
echo ===============================================================================
echo   OptiSent-XAI Observatory is LIVE!
echo   - Web Observatory:   http://127.0.0.1:8000
echo   - API Documentation: http://127.0.0.1:8000/docs
echo   - SSE Review Stream: http://127.0.0.1:8000/api/stream
echo   - Static Fallback:   %~dp0index.html
echo ===============================================================================
echo Press any key to exit this launcher window (backend will keep running).
pause >nul
