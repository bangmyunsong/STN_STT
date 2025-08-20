@echo off
echo =====================================
echo STN STT 시스템 전체 실행
echo =====================================

echo 시스템을 시작합니다...
echo - API 서버: http://localhost:8000
echo - 관리자 UI: http://localhost:3001
echo.

echo 가상환경 활성화 중...
call venv\Scripts\activate

echo API 서버 시작 중...
start "STN API Server" cmd /c "python api_server.py"

echo 잠시 대기 중... (API 서버 로딩 시간)
timeout /t 10 /nobreak

echo React 관리자 UI 시작 중...
start "STN Admin UI" cmd /c "cd stn-admin-react && npm start"

echo.
echo =====================================
echo 시스템이 시작되었습니다!
echo =====================================
echo.
echo 접근 주소:
echo - API 문서: http://localhost:8000/docs
echo - 관리자 UI: http://localhost:3001
echo.
echo 시스템을 종료하려면 각 창을 닫아주세요.
echo.

pause 