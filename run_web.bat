@echo off
title "SAFEPASS Live Webcam Multimodal Surveillance Web Dashboard"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS - Live Multimodal Web Surveillance Dashboard
echo ================================================================
echo Starting live web streaming server at: http://localhost:8000
echo Open http://localhost:8000 in your browser to view the demo!
echo Press Ctrl+C in this terminal window to stop the server.
echo ================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m face.webcam_live --web --port 8000 %*
) else (
    py -m face.webcam_live --web --port 8000 %* 2>nul || python -m face.webcam_live --web --port 8000 %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Web dashboard exited with error code %ERRORLEVEL%.
    pause
)
