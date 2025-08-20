@echo off
echo ==========================================
echo      WhisperX 발화자 구분 STT 설치 도구
echo ==========================================
echo.
echo 이 스크립트는 WhisperX와 발화자 구분 기능을 설치합니다.
echo 설치에는 몇 분이 소요될 수 있습니다.
echo.

echo 1. Python 환경 확인...
echo.
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python이 설치되지 않았습니다.
    echo Python 3.8 이상을 설치해주세요.
    pause
    exit
)

echo.
echo 2. 기존 패키지 업그레이드...
echo.
python -m pip install --upgrade pip setuptools wheel

echo.
echo 3. PyTorch 설치 (CPU 버전)...
echo.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo 4. WhisperX 및 관련 패키지 설치...
echo.
pip install git+https://github.com/m-bain/whisperx.git

echo.
echo 5. 발화자 구분을 위한 추가 패키지 설치...
echo.
pip install pyannote-audio

echo.
echo 6. UI 관련 패키지 설치...
echo.
pip install -r requirements.txt

echo.
echo 7. 설치 확인...
echo.
python -c "import whisperx; print('✅ WhisperX 설치 성공!')"
python -c "import pyannote.audio; print('✅ pyannote-audio 설치 성공!')"
python -c "import streamlit; print('✅ Streamlit 설치 성공!')"

echo.
echo ==========================================
echo 🎉 WhisperX 발화자 구분 STT 설치 완료!
echo.
echo 이제 다음 명령으로 앱을 실행할 수 있습니다:
echo   run_app.bat
echo.
echo 참고사항:
echo - 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다
echo - 발화자 구분 기능은 2명 이상의 화자가 있을 때 효과적입니다
echo - GPU가 있다면 코드에서 device="cuda"로 변경하면 더 빠릅니다
echo ==========================================
echo.
pause 