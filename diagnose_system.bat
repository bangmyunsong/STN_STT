@echo off
echo =====================================
echo STN STT 시스템 진단 시작
echo =====================================

echo Python 버전 확인...
python --version
echo.

echo 시스템 진단 실행 중...
python test_system_health.py

echo.
echo 진단 완료!
echo.
pause