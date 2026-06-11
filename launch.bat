@echo off
title Blade Sharpening Robot Simulator
cd /d "%~dp0"

echo Checking for latest version...
git pull
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Failed to fetch updates. You may be offline.
    echo Starting with the currently installed version.
    echo.
)

python -m robot_sim.main
if %ERRORLEVEL% neq 0 (
    echo.
    echo.
    echo [ERROR] An error occurred. If libraries are missing, run:
    echo   pip install -r requirements.txt
    pause
)
