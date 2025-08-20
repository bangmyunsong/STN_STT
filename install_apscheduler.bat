@echo off
echo ===============================================
echo STN STT 시스템 - APScheduler 설치
echo ===============================================
echo.

echo [1/2] APScheduler 패키지 설치 중...
pip install APScheduler>=3.10.0

echo.
echo [2/2] 설치 확인 중...
python -c "import apscheduler; print('✅ APScheduler 설치 완료!')"

echo.
echo ===============================================
echo 설치가 완료되었습니다!
echo API 서버를 재시작하면 스케줄러가 활성화됩니다.
echo ===============================================
pause 