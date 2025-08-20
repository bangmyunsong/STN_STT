@echo off
echo =====================================
echo STN STT 시스템 API 서버 시작
echo =====================================

echo 가상환경 활성화 중...
call venv\Scripts\activate

echo API 서버 시작 중...
echo 서버가 시작되면 다음 주소에서 접근 가능합니다:
echo - API 문서: http://localhost:8000/docs
echo - 헬스 체크: http://localhost:8000/health
echo.

python api_server.py

pause 