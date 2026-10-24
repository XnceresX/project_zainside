@echo off
title "SAFEPASS Multimodal Intake and Live Identification Studio"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS Multimodal Person Intake and Identification Studio
echo ================================================================
echo Features:
echo   - Input Person Name and Height
echo   - Multi-Angle Facial Structure Photos (Browse or Live Snapshots)
echo   - Clothing Appearance Profile (Upper and Lower Colours)
echo   - Reset / Clear Old Test Records (Zain, gracious, etc.)
echo   - Full Profile Review Screen
echo   - Direct Live Webcam Identification (Tracks P1 and P2)
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
