@echo off
setlocal
title FaceAttend SERVER
color 0b

echo ============================================================
echo           FACEATTEND SERVER - QUAN LY TRUNG TAM
echo ============================================================
echo.

:: 1. Kiem tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python! Vui long cai dat Python 3.10+.
    pause
    exit /b
)

:: 2. Thiet lap Python Executable
set PY_EXE=python
if exist .venv\Scripts\python.exe (
    set PY_EXE=.venv\Scripts\python.exe
    echo [INFO] Su dung moi truong ao: .venv
) else if exist venv\Scripts\python.exe (
    set PY_EXE=venv\Scripts\python.exe
    echo [INFO] Su dung moi truong ao: venv
) else (
    echo [INFO] Su dung Python he thong.
)

:: 3. Tao cac thu muc can thiet
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "assets\snapshots" mkdir assets\snapshots
if not exist "reports\output" mkdir reports\output

:: 4. Load bien moi truong tu .env.server
if exist ".env.server" (
    echo [INFO] Dang load cau hinh tu .env.server
)

:: 5. Khoi dong API Server (Chay trong cua so moi)
echo.
echo [1/2] Dang khoi dong API Server (Port: 8000)...
echo [HINT] Cua so nay se hien thi log ket noi tu cac Edge Box.
start "FaceAttend API Server" cmd /k "title [SERVER] API Server && %PY_EXE% api_server.py"

:: 6. Cho Server khoi dong (3 giay)
timeout /t 3 /nobreak >nul

:: 7. Khoi dong Giao dien chinh (UI)
echo.
echo [2/2] Dang khoi dong Giao dien quan ly (PyQt6)...
%PY_EXE% main.py

echo.
echo ============================================================
echo [DONE] Server da dung. Cam on ban da su dung!
echo ============================================================
pause
