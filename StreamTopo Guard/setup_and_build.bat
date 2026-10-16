@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON_EXE="
where py >nul 2>&1
if not errorlevel 1 for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
if defined PYTHON_EXE goto python_ready

echo Python 3.12 was not found. Installing it...
where winget >nul 2>&1
if errorlevel 1 goto download_python
winget install --id Python.Python.3.12 --exact --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto download_python
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if defined PYTHON_EXE goto python_ready
where py >nul 2>&1
if not errorlevel 1 for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
if defined PYTHON_EXE goto python_ready

:download_python
echo Downloading the official Python 3.12.10 installer...
set "PYTHON_INSTALLER=%TEMP%\python-3.12.10-amd64.exe"
curl.exe --fail --location --silent --show-error "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" --output "%PYTHON_INSTALLER%"
if errorlevel 1 goto python_install_failed
start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 TargetDir="%LocalAppData%\Programs\Python\Python312" PrependPath=0 Include_pip=1 Include_tcltk=1 Include_launcher=1
set "INSTALL_RESULT=%ERRORLEVEL%"
del /q "%PYTHON_INSTALLER%" >nul 2>&1
if not "%INSTALL_RESULT%"=="0" goto python_install_failed
if not exist "%LocalAppData%\Programs\Python\Python312\python.exe" goto python_install_failed
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"

:python_ready
echo Base Python: "%PYTHON_EXE%"
if not exist "%~dp0.venv-build\Scripts\python.exe" "%PYTHON_EXE%" -m venv "%~dp0.venv-build"
if errorlevel 1 goto setup_failed
set "PYTHON_EXE=%~dp0.venv-build\Scripts\python.exe"
echo Using isolated build environment: "%PYTHON_EXE%"
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements-build.txt"
if errorlevel 1 goto setup_failed

echo Running core tests...
"%PYTHON_EXE%" -m tests.unit.core.test_tda_guard
if errorlevel 1 goto setup_failed

echo Building the standalone GUI executable...
"%PYTHON_EXE%" "%~dp0build_download.py"
if errorlevel 1 goto setup_failed

echo.
echo Build complete: "%~dp0downloads\Windows\TopoGuard.exe"
pause
exit /b 0

:python_install_failed
echo ERROR: Could not install Python 3.12. Install it from https://www.python.org/downloads/ and run this file again.
pause
exit /b 1

:setup_failed
echo.
echo ERROR: Dependency installation, tests, or EXE build failed. Review the output above.
pause
exit /b 1
