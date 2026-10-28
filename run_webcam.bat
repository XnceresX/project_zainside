@echo off
title "SAFEPASS Autonomous Live Person Re-ID (Webcam)"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS Autonomous Live Person Re-ID Stream
echo ================================================================
echo Mode:
echo   - Autonomous Person ID Assignment (Person_0001, Person_0002...)
echo   - No Face or Clothing Input Needed
echo   - Real-Time Bounding Box & Re-ID Tracking
echo.
echo Controls:
echo   [SPACE] Pause / Resume Feed
echo   [S]     Save Snapshot Frame
echo   [Q]     Exit
echo ================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m cctv.reid_cctv --sources 0 %*
) else (
    python -m cctv.reid_cctv --sources 0 %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
