@echo off
title "SAFEPASS Sequential CCTV Person Re-ID - Pure Model"
set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe

echo ================================================================
echo   SAFEPASS Sequential CCTV Re-ID  ^|  Pure Model-Driven
echo ================================================================
echo.
echo  Models:    OSNet (full-body 512-d)  +  ArcFace (face, if visible)
echo  Decision:  100%% model cosine similarity  -  NO manual rules
echo  Persons:   Auto-assigned IDs: PERSON_001, PERSON_002, ...
echo             Unknown persons get NEW IDs automatically
echo.
echo  Controls:
echo    [SPACE]     Pause / Resume
echo    [N / ENTER] Next Camera
echo    [S]         Save Snapshot
echo    [Q]         Quit
echo.
echo  Threshold:  --body-thresh 0.60  (tweak if needed)
echo ================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m cctv.sequential_reid --body-thresh 0.60 --face-thresh 0.45 %*
) else (
    python -m cctv.sequential_reid --body-thresh 0.60 --face-thresh 0.45 %*
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
