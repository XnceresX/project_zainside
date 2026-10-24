@echo off
title "SAFEPASS Live Webcam Multimodal Face & Appearance Re-ID"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS Tracks P1 and P2 - Multimodal Face and Appearance Re-ID
echo ================================================================
echo Features:
echo   - 3D Multi-View Face Biometrics (ArcFace 512-d)
echo   - Body Appearance Re-ID (OSNet 512-d)
echo   - Upper and Lower Garment HSV Clothing Color Extraction
echo   - Quality-Adaptive Dynamic Multimodal Score Fusion
echo.
echo Controls:
echo   [E]   Enroll Detected Face + Clothing Appearance with Name
echo   [S]   Save Snapshot Frame
echo   [M]   Toggle Selfie Mirror Mode
echo   [L]   Toggle Landmarks
echo   [C]   Clear Registered Identities
echo   [Q]   Exit
echo ================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m face.webcam_live %*
) else (
    python -m face.webcam_live %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
