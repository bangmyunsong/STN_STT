@echo off
echo ==========================================
echo      FFmpeg 설치 도구
echo ==========================================
echo.
echo 이 스크립트는 관리자 권한이 필요합니다.
echo FFmpeg를 설치하여 STT 애플리케이션을 사용할 수 있도록 합니다.
echo.

echo 1. Chocolatey 설치 중...
echo.
powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

echo.
echo 2. FFmpeg 설치 중...
echo.
choco install ffmpeg -y

echo.
echo 3. 설치 확인...
echo.
ffmpeg -version

echo.
echo ==========================================
echo FFmpeg 설치가 완료되었습니다!
echo 이제 STT 애플리케이션을 사용할 수 있습니다.
echo ==========================================
echo.
pause 