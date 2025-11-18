::@echo off
::echo Starting Redis...
::start /B C:\Users\callippus\Downloads\Redis-x64-5.0.14.1\redis-server.exe
::timeout /t 3 /nobreak >nul
::echo Starting Flask app...
::python app.py


@echo off
echo ========================================
echo Starting Commission Dashboard with ML
echo ========================================

echo [1/3] Cleaning up old Redis processes...
taskkill /F /IM redis-server.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Starting Redis...
start /B C:\Users\callippus\Downloads\Redis-x64-5.0.14.1\redis-server.exe
timeout /t 3 /nobreak >nul

echo [3/3] Checking ML System...
python -c "from ml_advanced_predictor import tenant_predictor; print('ML System OK')"

echo [4/4] Starting Flask app with ML...
python app.py

pause
