@echo off
echo =====================================
echo 가상환경 설정
echo =====================================

echo Python 버전 확인...
python --version
echo.

echo 가상환경 생성 중... (시간이 걸릴 수 있습니다)
python -m venv venv

echo.
echo 가상환경 활성화 중...
call venv\Scripts\activate

echo.
echo 패키지 설치 중... (시간이 걸릴 수 있습니다)
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo =====================================
echo 가상환경 설정 완료!
echo =====================================
echo.
echo 이제 run_api_server.bat를 실행할 수 있습니다.
echo.
pause