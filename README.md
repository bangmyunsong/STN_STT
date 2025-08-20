# STN 고객센터 STT 시스템

STN 고객센터 STT (Speech-to-Text) 시스템은 Whisper STT와 GPT-3.5-turbo를 기반으로 한 음성 인식 및 ERP 항목 추출 시스템입니다.

## 시스템 구성

### 백엔드 (Python/FastAPI)
- **api_server.py**: 메인 FastAPI 서버
- **gpt_extractor.py**: GPT 기반 ERP 정보 추출
- **supabase_client.py**: Supabase 데이터베이스 연동
- **check_sessions.py**: 세션 관리 유틸리티

### 프론트엔드 (React)
- **stn-admin-react/**: React 기반 관리자 대시보드
  - 실시간 STT 처리 모니터링
  - 파일 업로드 및 관리
  - 결과 확인 및 통계

## 주요 기능

### STT (Speech-to-Text)
- Whisper 모델을 사용한 고정밀도 음성 인식
- 다양한 오디오 포맷 지원 (MP3, WAV, M4A 등)
- 실시간 처리 상태 모니터링

### ERP 정보 추출
- GPT-3.5-turbo를 활용한 고객 정보 자동 추출
- 고객명, 전화번호, 주소, 상담 내용 등 구조화된 데이터 생성
- 신뢰도 점수 기반 품질 관리

### 웹 대시보드
- 직관적인 관리자 인터페이스
- 실시간 처리 현황 모니터링
- 결과 검색 및 필터링
- 시스템 상태 확인

## 기술 스택

### 백엔드
- **Python 3.8+**
- **FastAPI**: REST API 프레임워크
- **Whisper**: OpenAI 음성 인식 모델
- **OpenAI GPT-3.5-turbo**: 자연어 처리
- **Supabase**: 데이터베이스 및 스토리지
- **APScheduler**: 백그라운드 작업 스케줄링

### 프론트엔드
- **React 18**: UI 프레임워크
- **TypeScript**: 타입 안전성
- **Modern CSS**: 반응형 디자인

## 설치 및 실행

### 1. 환경 설정

```bash
# Python 가상환경 생성 및 활성화
python -m venv venv
venv\\Scripts\\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

`config.env` 파일을 생성하고 다음 정보를 입력:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# Supabase
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# 기타 설정
STT_MODEL=base  # Whisper 모델 크기
MAX_FILE_SIZE=25MB
```

### 3. 백엔드 실행

```bash
python api_server.py
```

서버는 기본적으로 `http://localhost:8000`에서 실행됩니다.

### 4. 프론트엔드 실행

```bash
cd stn-admin-react
npm install
npm start
```

프론트엔드는 `http://localhost:3000`에서 실행됩니다.

## API 엔드포인트

### STT 처리
- `POST /upload-file`: 오디오 파일 업로드 및 STT 처리
- `GET /sessions`: 세션 목록 조회
- `GET /sessions/{session_id}`: 특정 세션 상세 정보

### 시스템 관리
- `GET /health`: 시스템 상태 확인
- `GET /system-info`: 시스템 정보 조회
- `GET /statistics`: 처리 통계

## 디렉토리 구조

```
STN_STT_POC/
├── api_server.py              # 메인 API 서버
├── gpt_extractor.py           # GPT 기반 정보 추출
├── supabase_client.py         # 데이터베이스 연동
├── check_sessions.py          # 세션 관리
├── config.env                 # 환경설정 (생성 필요)
├── src_record/                # 오디오 파일 저장소
├── stn-admin-react/           # React 프론트엔드
│   ├── src/
│   │   ├── components/        # React 컴포넌트
│   │   ├── services/          # API 서비스
│   │   ├── store/             # 상태 관리
│   │   └── types/             # TypeScript 타입
│   └── public/
└── venv/                      # Python 가상환경
```

## 개발 정보

- **프로젝트 시작**: 2024
- **개발자**: STN 개발팀
- **라이센스**: Private

## 문의사항

시스템 관련 문의사항이나 버그 리포트는 개발팀에 문의하시기 바랍니다.

---

> 이 시스템은 STN 고객센터의 업무 효율성 향상을 위해 개발되었습니다.