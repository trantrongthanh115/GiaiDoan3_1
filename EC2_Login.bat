@echo off
title Ket noi den AWS EC2 - Fashion Backend
echo Dang ket noi den may chu AWS EC2 (IP: 54.169.107.29)...
cd /d "%~dp0"
ssh -i "EC2.pem" ubuntu@54.169.244.75
if %errorlevel% neq 0 (
    echo.
    echo [LOI] Khong the ket noi den EC2. Vui lau kiem tra lai mang hoac trang thai Instance.
    pause
)
