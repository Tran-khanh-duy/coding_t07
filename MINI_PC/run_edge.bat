@echo off
setlocal
title FaceAttend EDGE BOX
color 0b

echo ============================================================
echo           FACEATTEND EDGE BOX - DIEM DANH KHUON MAT
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
    rem echo [INFO] Su dung moi truong ao: .venv
) else if exist venv\Scripts\python.exe (
    set PY_EXE=venv\Scripts\python.exe
    rem echo [INFO] Su dung moi truong ao: venv
) else (
    rem echo [INFO] Su dung Python he thong.
)

:: 3. Tao cac thu muc can thiet
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "database" mkdir database

:menu
cls
color 0b
echo ==============================================================================
echo                 FACEATTEND EDGE BOX - TRUNG TAM ĐIEU KHIEN
echo ==============================================================================
echo.
echo [THONG TIN CAU HINH HIEN TAI]
if exist ".env.edge" (
    type .env.edge | findstr "EDGE_SERVER_URL EDGE_CAMERA_ID EDGE_CAMERA_SOURCE"
) else (
    echo   [CANH BAO] Khong tim thay file .env.edge!
)
echo.

:: TU DONG MO HE THONG NGAY LAP TUC QUA YEU CAU CUA USER
goto run_edge

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

:run_edge
cls
color 0a
echo ============================================================
echo   DANG KHOI DONG EDGE BOX (Nhan Ctrl+C de dung)...
echo ============================================================
echo.
%PY_EXE% main_edge.py
echo.
echo ============================================================
echo [DONE] Edge Box da dung.
echo ============================================================
pause
goto menu
