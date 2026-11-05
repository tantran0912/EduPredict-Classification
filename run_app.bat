@echo off
chcp 65001 >nul
title EduPredict AI - Khởi Chạy
color 0b

echo [*] Dang kiem tra va cai dat thu vien...
python -m pip install -r requirements.txt

echo.
echo [*] Dang khoi chay server tai http://localhost:8000
start "" http://localhost:8000

python -m uvicorn web_classifier:app --host 127.0.0.1 --port 8000 --reload
pause
