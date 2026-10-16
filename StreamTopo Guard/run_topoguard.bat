@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
title TopoGuard Intelligence Probe v1.142
color 0A
cd /d "%~dp0"
cls

echo ========================================================
echo   ___  ___  ____  ___
echo  / _ \/ _ \/ __ \/ _ \
echo / .__/\___/ /_/ / __/
echo /_/       /____/_/
echo   TOPOGUARD SECURITY PROBE - LIVE MONITORING MODE
echo   Hard Bottom, Evolving Top ^| Read-Only SIEM
echo ========================================================
echo.

set "PYTHON_EXE="
if exist "%~dp0.venv-build\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv-build\Scripts\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
)

if not defined PYTHON_EXE (
    echo [ERROR] Python was not found.
    echo Install Python or run setup_and_build.bat first.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" topoguard_cli.py monitor --source system --ticks 100
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ========================================================
if "%EXIT_CODE%"=="0" (
    echo   Monitoring process ended.
) else (
    echo   Monitoring ended with exit code %EXIT_CODE%.
)
echo   Press any key to close.
echo ========================================================
pause >nul
exit /b %EXIT_CODE%
