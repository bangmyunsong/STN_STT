@echo off
echo =====================================
echo STN STT 시스템 API 서버 시작 (직접 실행)
echo =====================================

echo Python 경로 확인 중...
python --version
echo.

echo 현재 디렉토리: %CD%
echo.

echo API 서버 시작 중...
echo 서버가 시작되면 다음 주소에서 접근 가능합니다:
echo - API 문서: http://localhost:8000/docs
echo - 헬스 체크: http://localhost:8000/health
echo.

echo 문제 발생 시 Ctrl+C로 중단하세요.
echo.

python api_server.py

pause