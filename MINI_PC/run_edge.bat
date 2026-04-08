@echo off
setlocal
title FaceAttend EDGE BOX
color 0a

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
if not exist "database" mkdir database

:: 4. Hien thi cau hinh Edge
echo.
echo [THONG TIN EDGE BOX]
if exist ".env.edge" (
    echo   Cau hinh: .env.edge
    type .env.edge | findstr "EDGE_SERVER_URL EDGE_CAMERA_ID EDGE_DEVICE_NAME"
) else (
    echo   [CANH BAO] Khong tim thay file .env.edge!
    echo   Dang su dung cau hinh mac dinh.
    echo   Hay tao file .env.edge de cau hinh ket noi Server.
)

:: 5. Khoi dong Edge Kiosk
echo.
echo ============================================================
echo   DANG KHOI DONG EDGE BOX...
echo   Nhan Ctrl+C hoac dong cua so de dung.
echo ============================================================
echo.

%PY_EXE% main_edge.py

echo.
echo ============================================================
echo [DONE] Edge Box da dung.
echo ============================================================
pause
