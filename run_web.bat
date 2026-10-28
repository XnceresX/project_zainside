@echo off
title "SAFEPASS CCTV AI Command Center - Web Dashboard"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS CCTV AI COMMAND CENTER - WEB DASHBOARD
echo ================================================================
echo Starting AI Surveillance Server on http://localhost:8000
echo.
echo Features:
echo   - Dynamic Multi-CCTV Playback (Native Smooth FPS)
echo   - Independent Controls: Play, Pause, Step +/- 5f / +/- 1f on any camera
echo   - Add, Upload, or Delete Camera Footages Dynamically
echo   - Multimodal Re-ID: Confirmed Matches (Green) vs Unknown Intruder (Red)
echo   - Live Cross-Camera Person Verification Evidence Report
echo ================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
) else (
    python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Web dashboard exited with error code %ERRORLEVEL%.
    pause
)
