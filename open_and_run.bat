@echo off
echo ==============================
echo   Material Detector Launcher
echo ==============================

REM Change to the folder where this script is located
cd /d %~dp0

REM Check if VS Code is installed and available in PATH
where code >nul 2>nul
IF ERRORLEVEL 1 (
    echo.
    echo VS Code is not in your PATH.
    echo Please add VS Code to PATH or manually open this folder in VS Code.
    pause
    exit /b
)

REM Open VS Code in this folder
start code .

REM Create virtual environment if not exists
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Run the app
python material_detector_gui.py

pause
