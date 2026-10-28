@echo off
title "SAFEPASS Autonomous Person Re-ID Studio"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS Autonomous CCTV and Webcam Person Re-ID Studio
echo ================================================================
echo Features:
echo   - Autonomous Person ID Identification (No Face/Cloth Input Needed)
echo   - Multi-Camera CCTV Video Files (.mp4, .avi) and Webcams
echo   - Persistent Cross-Camera Global Re-ID (ReID-Framework)
echo   - OSNet Body Appearance and Topological Handoff Zones
echo ================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" safepass_studio.py %*
) else (
    python safepass_studio.py %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
