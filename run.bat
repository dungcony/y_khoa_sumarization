@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>&1
        if errorlevel 1 goto :python_missing
        python -m venv .venv
    )
    if errorlevel 1 goto :setup_failed
)

"%VENV_PYTHON%" -c "import streamlit, torch, transformers, sentencepiece" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies. This may take several minutes...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :setup_failed
    "%VENV_PYTHON%" -m pip install -e .
    if errorlevel 1 goto :setup_failed
)

"%VENV_PYTHON%" -m streamlit run app.py
if errorlevel 1 goto :run_failed
exit /b 0

:python_missing
echo Python 3.10 or newer is required.
echo Download it from https://www.python.org/downloads/
pause
exit /b 1

:setup_failed
echo Failed to create the environment or install dependencies.
echo Check the Internet connection and Python installation, then try again.
pause
exit /b 1

:run_failed
echo The application stopped with an error.
pause
exit /b 1
