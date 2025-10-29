@echo off
title Iniciando Servidores do Projeto TCC
color 0A
echo ===========================================
echo Iniciando Servidores do Projeto TCC
echo ===========================================

REM === Caminho base do projeto ===
set PROJECT_DIR=C:\Users\Administrator\Desktop\main-xraymachine\main-xraymachine\src
set BACKEND_DIR=%PROJECT_DIR%\backend
set PYTHON_UVICORN=C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Scripts\uvicorn.exe
set LOG_FILE=%PROJECT_DIR%\startup_log.txt

echo [%date% %time%] Iniciando Nginx... >> "%LOG_FILE%"
cd "C:\Users\Administrator\Desktop\main-xraymachine\main-xraymachine\src\nginx"
start nginx.exe

REM === Aguarda alguns segundos para garantir inicialização ===
timeout /t 5 >nul

echo [%date% %time%] Iniciando API FastAPI (Uvicorn)... >> "%LOG_FILE%"
cd /d "%BACKEND_DIR%"
start cmd /k "%PYTHON_UVICORN% main:app --host 0.0.0.0 --port 8000"

if %errorlevel% neq 0 (
    echo ERRO ao iniciar Uvicorn! Código: %errorlevel%
    echo [%date% %time%] ERRO ao iniciar Uvicorn! Código: %errorlevel% >> "%LOG_FILE%"
    pause
    exit /b %errorlevel%
)

echo ===========================================
echo Servidores iniciados com sucesso!
echo [%date% %time%] Servidores iniciados com sucesso! >> "%LOG_FILE%"
echo ===========================================

pause
exit
