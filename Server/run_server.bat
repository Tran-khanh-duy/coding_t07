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
) else if exist venv\Scripts\python.exe (
    set PY_EXE=venv\Scripts\python.exe
)

:: 3. Tao cac thu muc can thiet
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "assets\snapshots" mkdir assets\snapshots
if not exist "reports\output" mkdir reports\output

:menu
cls
color 0b
echo ==============================================================================
echo                 FACEATTEND SERVER - TRUNG TAM ĐIEU KHIEN
echo ==============================================================================
echo.
echo [THONG TIN CAU HINH HIEN TAI]
if exist ".env.server" (
    echo   Cau hinh: .env.server tim thay.
) else (
    echo   [CANH BAO] Khong tim thay file .env.server! 
    echo   Dang dung cau hinh mac dinh hoac ban nen tao file de chay an toan nhat.
)
echo.

:: TU DONG KHOI DONG SERVER BO QUA MENU
goto run_server

:run_test
cls
color 0a
echo ============================================================
echo   DANG CHAY TIEU TRINH QUET CAMERA IP...
echo ============================================================
%PY_EXE% test_ip_camera.py
echo.
pause
goto menu

:run_server
cls
color 0b
echo ============================================================
echo   DANG KHOI DONG HE THONG SERVER...
echo ============================================================
echo.

echo [1/2] Dang khoi dong API Server (Port: 8000)...
echo [HINT] Cua so moi se xuat hien de hien loi hoac log ket noi tu Edge Box.
start "FaceAttend API Server" cmd /k "title [SERVER] API Lua Chon && %PY_EXE% api_server.py"

:: Cho API Server khoi dong (3 giay)
timeout /t 3 /nobreak >nul

echo.
echo [2/2] Dang khoi dong Giao dien quan ly chinh (PyQt6)...
%PY_EXE% main.py

echo.
echo ============================================================
echo [DONE] Server da dung hoat dong.
echo ============================================================
pause
goto menu
