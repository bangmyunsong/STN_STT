"""
STN 고객센터 STT 시스템 API 서버
FastAPI 기반 REST API 서버 - ERP 연동 및 STT 처리
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid
import json
import os
import tempfile
import whisper  # Render 메모리 절약을 위한 극한 최적화 버전
from datetime import datetime, timedelta
import logging
import threading

# 스케줄러 관련 import (선택적)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    BackgroundScheduler = None
    CronTrigger = None
    SCHEDULER_AVAILABLE = False
    print("⚠️ APScheduler가 설치되지 않았습니다. 스케줄러 기능이 비활성화됩니다.")
    print("⚠️ 설치하려면: pip install APScheduler>=3.10.0")

# 로컬 모듈 import
from gpt_extractor import ERPExtractor, extract_erp_from_segments
from supabase_client import get_supabase_manager, save_stt_result, save_erp_result
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv('config.env')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="STN 고객센터 STT 시스템 API",
    description="Whisper STT + GPT-3.5-turbo 기반 ERP 항목 추출 및 연동 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 422 오류 디버깅을 위한 예외 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"422 검증 오류 발생 - URL: {request.url}")
    logger.error(f"요청 메소드: {request.method}")
    logger.error(f"요청 헤더: {dict(request.headers)}")
    
    # 요청 본문 로깅
    try:
        body = await request.body()
        logger.error(f"요청 본문: {body.decode('utf-8')}")
    except Exception as e:
        logger.error(f"요청 본문 읽기 실패: {e}")
    
    # 쿼리 파라미터 로깅
    logger.error(f"쿼리 파라미터: {dict(request.query_params)}")
    
    # 상세한 검증 오류 로깅
    logger.error(f"검증 오류 상세:")
    for error in exc.errors():
        logger.error(f"  - 필드: {error.get('loc')}")
        logger.error(f"  - 오류: {error.get('msg')}")
        logger.error(f"  - 타입: {error.get('type')}")
        logger.error(f"  - 입력값: {error.get('input', 'N/A')}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "요청 데이터 검증 실패",
            "debug_info": {
                "url": str(request.url),
                "method": request.method,
                "errors": exc.errors()
            }
        }
    )

# 전역 변수
whisper_model = None
erp_extractor = None
supabase_manager = None

# 모델 캐싱용 딕셔너리 (성능 최적화)
cached_whisper_models = {}

def clear_model_cache():
    """메모리 관리를 위한 모델 캐시 정리 함수"""
    global cached_whisper_models
    cached_whisper_models.clear()
    logger.info("모델 캐시가 정리되었습니다.")

def clear_whisper_file_cache():
    """손상된 Whisper 파일 캐시를 정리하는 함수"""
    import os
    import shutil
    from pathlib import Path
    
    try:
        # Windows 환경에서 Whisper 캐시 경로
        cache_paths = [
            Path.home() / ".cache" / "whisper",  # Linux/Mac
            Path(os.getenv('LOCALAPPDATA', '')) / "whisper",  # Windows
            Path(os.getenv('APPDATA', '')) / "whisper",  # Windows 대안
        ]
        
        cleared_paths = []
        for cache_path in cache_paths:
            if cache_path.exists() and cache_path.is_dir():
                try:
                    shutil.rmtree(cache_path)
                    cleared_paths.append(str(cache_path))
                    logger.info(f"Whisper 캐시 폴더 삭제됨: {cache_path}")
                except Exception as e:
                    logger.warning(f"캐시 폴더 삭제 실패 ({cache_path}): {e}")
        
        if cleared_paths:
            logger.info(f"총 {len(cleared_paths)}개의 캐시 폴더가 정리되었습니다.")
            return True, cleared_paths
        else:
            logger.info("정리할 Whisper 캐시 폴더를 찾을 수 없습니다.")
            return False, []
            
    except Exception as e:
        logger.error(f"Whisper 캐시 정리 중 오류: {e}")
        return False, []

# 상수 정의
AUDIO_DIRECTORY = "src_record"  # 음성 파일 디렉토리

# 지원되는 오디오 파일 확장자
SUPPORTED_AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']

# 일자별 폴더 관리 함수
def create_daily_directory():
    """
    오늘 날짜 기준으로 src_record 하위에 YYYY-MM-DD 형식의 폴더 생성
    """
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        daily_path = os.path.join(AUDIO_DIRECTORY, today)
        
        # 기본 src_record 디렉토리가 없으면 생성
        if not os.path.exists(AUDIO_DIRECTORY):
            os.makedirs(AUDIO_DIRECTORY)
            logger.info(f"기본 음성 파일 디렉토리 생성: {AUDIO_DIRECTORY}")
        
        # 오늘 날짜 폴더가 없으면 생성
        if not os.path.exists(daily_path):
            os.makedirs(daily_path)
            logger.info(f"일자별 폴더 생성: {daily_path}")
        else:
            logger.info(f"일자별 폴더 이미 존재: {daily_path}")
            
        return daily_path
        
    except Exception as e:
        logger.error(f"일자별 폴더 생성 실패: {e}")
        return None

# 스케줄러 관련 변수
scheduler = None

def create_daily_directory_with_date(target_date=None, auto_create=True):
    """
    특정 날짜의 폴더를 생성 (스케줄러용)
    
    Args:
        target_date: 생성할 날짜 (기본값: 오늘)
        auto_create: 자동 생성 여부
    """
    try:
        if target_date is None:
            target_date = datetime.now()
        
        date_str = target_date.strftime('%Y-%m-%d')
        daily_path = os.path.join(AUDIO_DIRECTORY, date_str)
        
        # 기본 src_record 디렉토리가 없으면 생성
        if not os.path.exists(AUDIO_DIRECTORY):
            os.makedirs(AUDIO_DIRECTORY)
            logger.info(f"기본 음성 파일 디렉토리 생성: {AUDIO_DIRECTORY}")
        
        # 해당 날짜 폴더가 없으면 생성
        if not os.path.exists(daily_path):
            if auto_create:
                os.makedirs(daily_path)
                logger.info(f"스케줄러: 일자별 폴더 생성 완료 - {daily_path}")
            else:
                logger.info(f"스케줄러: 폴더 생성 필요 - {daily_path} (auto_create=False)")
        else:
            logger.info(f"스케줄러: 일자별 폴더 이미 존재 - {daily_path}")
            
        return daily_path
        
    except Exception as e:
        logger.error(f"스케줄러: 일자별 폴더 생성 실패 - {e}")
        return None

def ensure_today_folder_exists():
    """
    오늘 날짜 폴더가 존재하는지 확인하고 없으면 생성
    """
    return create_daily_directory_with_date(datetime.now(), auto_create=True)

def scheduled_daily_folder_creation():
    """
    매일 0시에 실행되는 일별 폴더 생성 함수
    """
    try:
        today = datetime.now()
        daily_path = create_daily_directory_with_date(today, auto_create=True)
        
        if daily_path:
            logger.info(f"✅ 스케줄러: {today.strftime('%Y-%m-%d')} 폴더 생성 완료")
        else:
            logger.error(f"❌ 스케줄러: {today.strftime('%Y-%m-%d')} 폴더 생성 실패")
            
    except Exception as e:
        logger.error(f"❌ 스케줄러 실행 중 오류: {e}")

def get_daily_directory_path(date_str=None):
    """
    특정 날짜의 폴더 경로를 반환 (기본값: 오늘)
    
    Args:
        date_str (str): YYYY-MM-DD 형식의 날짜 문자열
    
    Returns:
        str: 일자별 폴더 경로
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    return os.path.join(AUDIO_DIRECTORY, date_str)

# Pydantic 모델들
class ERPData(BaseModel):
    """ERP 등록 데이터 모델"""
    as_support: str = Field("", alias="AS 및 지원", description="지원 방식 (방문기술지원, 원격기술지원 등)")
    request_org: str = Field("", alias="요청기관", description="고객사 또는 기관명")
    work_location: str = Field("", alias="작업국소", description="지역 또는 위치")
    request_date: str = Field("", alias="요청일", description="고객이 요청한 날짜 (YYYY-MM-DD)")
    request_time: str = Field("", alias="요청시간", description="고객이 요청한 시간 (24시간 형식)")
    requester: str = Field("", alias="요청자", description="고객 담당자 이름")
    support_count: str = Field("", alias="지원인원수", description="필요한 지원 인원 수")
    support_staff: str = Field("", alias="지원요원", description="투입 예정 기술자 이름")
    equipment_name: str = Field("", alias="장비명", description="장비 종류")
    model_name: str = Field("", alias="기종명", description="구체적인 장비 모델명")
    as_period_status: str = Field("", alias="A/S기간만료여부", description="A/S 기간 상태 (무상, 유상)")
    system_name: str = Field("", alias="시스템명(고객사명)", description="고객사 시스템명")
    request_content: str = Field("", alias="요청 사항", description="고객 요청 내용 요약")
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "AS 및 지원": "원격기술지원",
                "요청기관": "수자원공사 FA망",
                "작업국소": "대전",
                "요청일": "2025-04-18",
                "요청시간": "15",
                "요청자": "이정순",
                "지원인원수": "1명",
                "지원요원": "임선묵",
                "장비명": "MSPP",
                "기종명": "1646SMC",
                "A/S기간만료여부": "유상",
                "시스템명(고객사명)": "수자원공사 FA망",
                "요청 사항": "수자원 회선 문의건"
            }
        }

class ERPRegisterResponse(BaseModel):
    """ERP 등록 응답 모델"""
    status: str = Field(..., description="처리 상태")
    erp_id: str = Field(..., description="ERP 등록 ID")
    message: Optional[str] = Field(None, description="처리 메시지")

class STTRequest(BaseModel):
    """STT 처리 요청 모델"""
    model_name: Optional[str] = Field("tiny", description="Whisper 모델명")
    language: Optional[str] = Field(None, description="언어 코드")
    enable_diarization: Optional[bool] = Field(True, description="화자 분리 활성화")

class STTResponse(BaseModel):
    """STT 처리 응답 모델"""
    status: str = Field(..., description="처리 상태")
    transcript: str = Field(..., description="전체 텍스트")
    segments: List[Dict] = Field(..., description="세그먼트별 결과")
    erp_data: Optional[ERPData] = Field(None, description="추출된 ERP 데이터")
    processing_time: float = Field(..., description="처리 시간(초)")
    file_id: str = Field(..., description="파일 처리 ID")
    session_id: Optional[int] = Field(None, description="데이터베이스 세션 ID")
    extraction_id: Optional[int] = Field(None, description="ERP 추출 결과 ID")

# 초기화 함수들
def initialize_models():
    """모델들을 초기화하는 함수 (안전한 단계별 초기화)"""
    global whisper_model, erp_extractor, supabase_manager
    
    logger.info("🚀 모델 초기화 시작...")

    # Render 512MB 극한 메모리 최적화 모드
    import os
    import gc
    import psutil
    
    # 극한 메모리 최적화 설정
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_JIT"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 메모리 상태 체크 함수
    def log_memory(stage="Unknown"):
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"🧠 [{stage}] 메모리: {memory_mb:.1f}MB / 512MB")
            if memory_mb > 400:
                logger.warning(f"⚠️ 메모리 위험 수준: {memory_mb:.1f}MB")
                gc.collect()
            return memory_mb
        except Exception as e:
            logger.warning(f"메모리 체크 실패: {e}")
            return 0
    
    # 가비지 컬렉션 강제 실행
    gc.collect()
    log_memory("초기화 시작")
    
    logger.info("💾 극한 메모리 최적화 모드 (Whisper tiny 모델)")
    
    # 1. Whisper tiny 모델 로드 (극한 최적화)
    try:
        logger.info("1️⃣ Whisper tiny 모델 로딩 중... (메모리 최적화)")
        
        # 모델 로딩 전 메모리 체크
        log_memory("모델 로딩 전")
        
        # tiny 모델만 로드 (39MB) - 가장 작은 모델
        model_size = os.getenv("STT_MODEL", "tiny")
        logger.info(f"🔧 환경변수 STT_MODEL: {model_size}")
        
        # tiny로 강제 설정 (메모리 절약)
        if model_size != "tiny":
            logger.warning(f"⚠️ Render 512MB 제한으로 인해 {model_size} → tiny로 변경")
            model_size = "tiny"
        
        whisper_model = whisper.load_model(model_size)
        logger.info("✅ Whisper tiny 모델 로딩 완료")
        
        # 모델 로딩 후 메모리 체크
        log_memory("모델 로딩 후")
        
        # 캐시에 저장
        cached_whisper_models[model_size] = whisper_model
        
    except Exception as e:
        logger.error(f"❌ Whisper 모델 로딩 실패: {e}")
        raise
    
    # 2. ERP Extractor 초기화 (선택적)
    try:
        logger.info("2️⃣ ERP Extractor 초기화 중...")
        erp_extractor = ERPExtractor()
        logger.info("✅ ERP Extractor 초기화 완료")
    except Exception as e:
        logger.warning(f"⚠️ ERP Extractor 초기화 실패 (계속 진행): {e}")
        logger.warning("💡 해결방법: config.env에서 OPENAI_API_KEY 확인")
        erp_extractor = None
    
    # 3. Supabase 매니저 초기화 (선택적)
    try:
        logger.info("3️⃣ Supabase 매니저 초기화 중...")
        supabase_manager = get_supabase_manager()
        logger.info("✅ Supabase 매니저 초기화 완료")
    except Exception as e:
        logger.warning(f"⚠️ Supabase 초기화 실패 (계속 진행): {e}")
        logger.warning("💡 해결방법: config.env에서 Supabase 설정 확인")
        supabase_manager = None
    
    logger.info("🎉 모델 초기화 완료!")

# 의존성 함수
def get_whisper_model():
    """Whisper 모델 의존성 (극한 메모리 최적화)"""
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper 모델이 초기화되지 않았습니다")
    return whisper_model

def get_erp_extractor():
    """ERP Extractor 의존성 (선택적)"""
    if erp_extractor is None:
        logger.warning("ERP Extractor가 초기화되지 않았습니다. ERP 추출 기능이 비활성화됩니다.")
        # 기본 객체 반환 또는 None 반환하여 처리 로직에서 확인하게 함
        return None
    return erp_extractor

def get_supabase_manager_dep():
    """Supabase 매니저 의존성 (선택사항)"""
    return supabase_manager  # None일 수 있음

# API 엔드포인트들

@app.get("/")
async def root():
    """API 서버 상태 확인"""
    return {
        "message": "STN 고객센터 STT 시스템 API 서버",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    supabase_status = False
    if supabase_manager:
        try:
            supabase_status = supabase_manager.health_check()
        except:
            supabase_status = False
    
    scheduler_status = False
    if SCHEDULER_AVAILABLE and scheduler:
        try:
            scheduler_status = scheduler.running
        except:
            scheduler_status = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "whisper": whisper_model is not None,
            "erp_extractor": erp_extractor is not None,
            "supabase": supabase_status
        },
        "scheduler": {
            "available": SCHEDULER_AVAILABLE,
            "running": scheduler_status
        }
    }

@app.get("/test")
async def test_endpoint():
    """간단한 테스트 엔드포인트"""
    return {
        "status": "ok",
        "message": "API 서버가 정상적으로 동작하고 있습니다",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/erp-sample-register", response_model=ERPRegisterResponse)
async def register_erp_sample(
    erp_data: ERPData, 
    extraction_id: Optional[int] = None,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """
    ERP 연동용 샘플 등록 API
    PRD 요구사항에 따른 테스트용 인터페이스
    """
    try:
        # Mock ERP ID 생성
        erp_id = f"mock{uuid.uuid4().hex[:8]}"
        
        logger.info(f"ERP 샘플 등록 요청 - ID: {erp_id}")
        logger.info(f"등록 데이터: {erp_data.dict()}")
        
        # 실제 ERP 시스템 연동 시뮬레이션 (여기서는 단순히 성공 응답)
        response_data = {
            "status": "success",
            "erp_id": erp_id,
            "message": "ERP 시스템에 정상적으로 등록되었습니다"
        }
        
        # Supabase에 등록 로그 저장
        if extraction_id and supabase_mgr:
            try:
                supabase_mgr.save_erp_register_log(
                    extraction_id=extraction_id,
                    erp_id=erp_id,
                    status="success",
                    response_data=response_data
                )
                logger.info(f"ERP 등록 로그 저장 완료 - 추출 ID: {extraction_id}")
            except Exception as e:
                logger.warning(f"ERP 등록 로그 저장 실패: {e}")
        
        response = ERPRegisterResponse(**response_data)
        return response
        
    except Exception as e:
        logger.error(f"ERP 등록 실패: {e}")
        
        # 실패 로그도 저장
        if extraction_id and supabase_mgr:
            try:
                supabase_mgr.save_erp_register_log(
                    extraction_id=extraction_id,
                    erp_id="",
                    status="failed",
                    response_data={"error": str(e)}
                )
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"ERP 등록 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/stt-process", response_model=STTResponse)
async def process_audio_file(
    file: UploadFile = File(..., description="업로드할 음성 파일"),
    model_name: str = "tiny",
    language: Optional[str] = None,
    enable_diarization: bool = True,
    extract_erp: bool = True,
    save_to_db: bool = True,
    whisper_model=Depends(get_whisper_model),
    erp_extractor=Depends(get_erp_extractor),
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """
    음성 파일 STT 처리 및 ERP 항목 추출 API
    """
    start_time = datetime.now()
    file_id = f"stt_{uuid.uuid4().hex[:8]}"
    
    try:
        logger.info(f"STT 처리 시작 - File ID: {file_id}, 파일명: {file.filename}")
        
        # 파일 형식 검증
        allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
            )
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Whisper tiny 모델 STT 처리 (극한 메모리 최적화)
            logger.info(f"Whisper STT 처리 중 - 모델: tiny (메모리 최적화)")
            
            try:
                # 메모리 사용량 체크
                import psutil
                import gc
                
                def check_memory():
                    try:
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        logger.info(f"🧠 STT 처리 중 메모리: {memory_mb:.1f}MB")
                        if memory_mb > 450:
                            logger.warning("⚠️ 메모리 임계점 근접, 가비지 컬렉션 실행")
                            gc.collect()
                        return memory_mb
                    except:
                        return 0
                
                check_memory()
                
                # tiny 모델 강제 사용 (메모리 절약)
                current_model = whisper_model
                if current_model is None:
                    raise HTTPException(status_code=500, detail="Whisper 모델이 초기화되지 않았습니다")
                
                logger.info(f"📁 처리할 파일: {temp_file_path}")
                
                # STT 실행 (메모리 최적화 옵션)
                result = current_model.transcribe(
                    temp_file_path,
                    language=language,
                    verbose=False,  # 메모리 절약
                    fp16=False,  # CPU에서는 fp16 비활성화
                )
                
                logger.info("✅ Whisper STT 처리 완료")
                check_memory()
                
            except Exception as stt_error:
                logger.error(f"❌ Whisper STT 처리 실패: {stt_error}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"음성 인식 처리 중 오류가 발생했습니다: {str(stt_error)}"
                )
            
            # 세그먼트 데이터 처리
            segments = []
            for i, segment in enumerate(result.get("segments", [])):
                segment_data = {
                    "id": i,
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"Speaker_{i % 2}"  # 간단한 화자 분리 시뮬레이션
                }
                segments.append(segment_data)
            
            # ERP 데이터 추출 (타임아웃 처리 개선)
            erp_data = None
            if extract_erp and segments and erp_extractor is not None:
                try:
                    logger.info("ERP 데이터 추출 중... (30초 타임아웃)")
                    erp_dict = erp_extractor.extract_from_segments(segments)
                    erp_data = ERPData(**erp_dict)
                    logger.info(f"ERP 데이터 추출 완료: {erp_dict}")
                except TimeoutError as e:
                    logger.warning(f"ERP 데이터 추출 타임아웃: {e}")
                    logger.info("ERP 추출을 건너뛰고 STT 결과만 반환합니다.")
                except Exception as e:
                    logger.warning(f"ERP 데이터 추출 실패: {e}")
                    logger.info("ERP 추출을 건너뛰고 STT 결과만 반환합니다.")
            elif extract_erp and erp_extractor is None:
                logger.info("⚠️ ERP Extractor가 비활성화되어 있습니다. STT 결과만 반환합니다.")
            
            # 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Supabase에 결과 저장 (옵션)
            session_id = None
            extraction_id = None
            
            if save_to_db and supabase_mgr:
                try:
                    logger.info("Supabase에 STT 결과 저장 중...")
                    
                    # STT 세션 생성 및 업데이트
                    session = supabase_mgr.create_stt_session(
                        file_name=file.filename,
                        file_id=file_id,
                        model_name=model_name,
                        language=language
                    )
                    session_id = session['id']
                    
                    # STT 결과 업데이트
                    supabase_mgr.update_stt_session(
                        session_id=session_id,
                        transcript=result["text"],
                        segments=segments,
                        processing_time=processing_time,
                        status="completed"
                    )
                    
                    # ERP 추출 결과 저장
                    if erp_data:
                        erp_dict = erp_data.dict(by_alias=True)
                        extraction = supabase_mgr.save_erp_extraction(
                            session_id=session_id,
                            erp_data=erp_dict
                        )
                        extraction_id = extraction['id']
                        logger.info(f"ERP 추출 결과 저장 완료 - 추출 ID: {extraction_id}")
                        
                        # ERP 시스템에 자동 등록 (DB 저장 옵션이 활성화된 경우)
                        try:
                            logger.info("ERP 시스템에 자동 등록 중...")
                            
                            # Mock ERP ID 생성
                            erp_id = f"auto{uuid.uuid4().hex[:8]}"
                            
                            # ERP 등록 시뮬레이션 (실제 ERP 시스템 연동 시 이 부분을 수정)
                            erp_response_data = {
                                "status": "success",
                                "erp_id": erp_id,
                                "message": "STT 처리 중 ERP 시스템에 자동 등록되었습니다"
                            }
                            
                            # ERP 등록 로그 저장
                            supabase_mgr.save_erp_register_log(
                                extraction_id=extraction_id,
                                erp_id=erp_id,
                                status="success",
                                response_data=erp_response_data
                            )
                            
                            logger.info(f"ERP 자동 등록 완료 - ERP ID: {erp_id}, 추출 ID: {extraction_id}")
                            
                        except Exception as e:
                            logger.warning(f"ERP 자동 등록 실패 (계속 진행): {e}")
                            # 실패 로그도 저장
                            try:
                                supabase_mgr.save_erp_register_log(
                                    extraction_id=extraction_id,
                                    erp_id="",
                                    status="failed",
                                    response_data={"error": str(e)}
                                )
                            except:
                                pass
                    
                    logger.info(f"Supabase 저장 완료 - 세션 ID: {session_id}")
                    
                except Exception as e:
                    logger.warning(f"Supabase 저장 실패 (계속 진행): {e}")
            
            # 응답 생성
            response = STTResponse(
                status="success",
                transcript=result["text"],
                segments=segments,
                erp_data=erp_data,
                processing_time=processing_time,
                file_id=file_id
            )
            
            # 응답에 DB 저장 정보 추가 (동적 필드)
            if session_id:
                response.session_id = session_id
            if extraction_id:
                response.extraction_id = extraction_id
            
            logger.info(f"STT 처리 완료 - File ID: {file_id}, 처리시간: {processing_time:.2f}초")
            return response
            
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT 처리 실패 - File ID: {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"STT 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/stt-process-file", response_model=STTResponse)
async def process_audio_file_from_directory(
    filename: str,
    model_name: str = "tiny",
    language: Optional[str] = None,
    enable_diarization: bool = True,
    extract_erp: bool = True,
    save_to_db: bool = True,
    whisper_model=Depends(get_whisper_model),
    erp_extractor=Depends(get_erp_extractor),
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """
    src_record 디렉토리의 음성 파일 STT 처리 및 ERP 항목 추출 API
    """
    start_time = datetime.now()
    file_id = f"stt_{uuid.uuid4().hex[:8]}"
    
    try:
        # 파일 경로 검증 (일자별 폴더 구조 지원)
        # filename이 "날짜폴더/파일명" 형식이거나 단순히 "파일명"일 수 있음
        file_path = os.path.join(AUDIO_DIRECTORY, filename)
        
        # Windows 경로 정규화
        file_path = os.path.normpath(file_path)
        
        # 절대 경로로 변환 (Whisper가 상대 경로에서 문제가 있을 수 있음)
        file_path = os.path.abspath(file_path)
        
        logger.info(f"파일 경로 확인 - 요청된 파일명: {filename}")
        logger.info(f"파일 경로 확인 - 구성된 경로: {file_path}")
        logger.info(f"파일 경로 확인 - 파일 존재 여부: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail=f"파일을 찾을 수 없습니다: {filename} (경로: {file_path})"
            )
        
        if not os.path.isfile(file_path):
            raise HTTPException(
                status_code=400, 
                detail=f"유효한 파일이 아닙니다: {filename} (경로: {file_path})"
            )
        
        # 파일 형식 검증 (실제 파일명에서 확장자 추출)
        actual_filename = os.path.basename(filename)  # 경로에서 파일명만 추출
        file_extension = os.path.splitext(actual_filename)[1].lower()
        
        if file_extension not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
            )
        
        logger.info(f"STT 처리 시작 - File ID: {file_id}, 파일경로: {file_path}")
        
        # Whisper tiny 모델 STT 처리 (극한 메모리 최적화)
        logger.info(f"Whisper STT 처리 중 - 모델: tiny (메모리 최적화)")
        
        try:
            # 메모리 최적화: tiny 모델만 강제 사용
            if model_name != "tiny":
                logger.warning(f"⚠️ Render 512MB 제한으로 인해 {model_name} → tiny로 변경")
                model_name = "tiny"
            
            # 기본 tiny 모델 사용 (메모리 절약)
            current_model = whisper_model
            if current_model is None:
                raise HTTPException(status_code=500, detail="Whisper 모델이 초기화되지 않았습니다")
            
            logger.info("✅ Whisper tiny 모델 사용 (메모리 최적화)")
        
            # STT 실행 (메모리 최적화)
            logger.info(f"📁 처리할 파일: {file_path}")
            logger.info(f"🌍 언어 설정: {language}")
            
            # 메모리 사용량 체크
            import psutil
            import gc
            
            def check_memory():
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"🧠 STT 처리 중 메모리: {memory_mb:.1f}MB")
                    if memory_mb > 450:
                        logger.warning("⚠️ 메모리 임계점 근접, 가비지 컬렉션 실행")
                        gc.collect()
                    return memory_mb
                except:
                    return 0
            
            check_memory()
            
            # STT 실행 (메모리 최적화 옵션)
            result = current_model.transcribe(
                file_path,
                language=language,
                verbose=False,  # 메모리 절약
                fp16=False,  # CPU에서는 fp16 비활성화
            )
            
            logger.info(f"✅ Whisper transcribe 완료 - 텍스트 길이: {len(result.get('text', ''))}")
            check_memory()
            
        except Exception as transcribe_error:
            logger.error(f"❌ Whisper transcribe 실패 - 파일: {file_path}")
            logger.error(f"❌ 오류 내용: {transcribe_error}")
            
            # FFmpeg 관련 오류 감지
            error_msg = str(transcribe_error)
            if "WinError 2" in error_msg or "CreateProcess" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="FFmpeg가 설치되지 않았습니다. Whisper는 오디오 처리를 위해 FFmpeg가 필요합니다."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"음성 인식 처리 중 오류가 발생했습니다: {str(transcribe_error)}"
                )
        
        # 세그먼트 데이터 처리
        segments = []
        for i, segment in enumerate(result.get("segments", [])):
            segment_data = {
                "id": i,
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"],
                "speaker": f"Speaker_{i % 2}"  # 간단한 화자 분리 시뮬레이션
            }
            segments.append(segment_data)
        
        # ERP 데이터 추출 (타임아웃 처리 개선)
        erp_data = None
        if extract_erp and segments and erp_extractor is not None:
            try:
                logger.info("ERP 데이터 추출 중... (30초 타임아웃)")
                erp_dict = erp_extractor.extract_from_segments(segments)
                erp_data = ERPData(**erp_dict)
                logger.info(f"ERP 데이터 추출 완료: {erp_dict}")
            except TimeoutError as e:
                logger.warning(f"ERP 데이터 추출 타임아웃: {e}")
                logger.info("ERP 추출을 건너뛰고 STT 결과만 반환합니다.")
            except Exception as e:
                logger.warning(f"ERP 데이터 추출 실패: {e}")
                logger.info("ERP 추출을 건너뛰고 STT 결과만 반환합니다.")
        elif extract_erp and erp_extractor is None:
            logger.info("⚠️ ERP Extractor가 비활성화되어 있습니다. STT 결과만 반환합니다.")
        
        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Supabase에 결과 저장 (옵션)
        session_id = None
        extraction_id = None
        
        if save_to_db and supabase_mgr:
            try:
                logger.info("Supabase에 STT 결과 저장 중...")
                
                # STT 세션 생성 및 업데이트
                session = supabase_mgr.create_stt_session(
                    file_name=filename,
                    file_id=file_id,
                    model_name=model_name,
                    language=language
                )
                session_id = session['id']
                
                # STT 결과 업데이트
                supabase_mgr.update_stt_session(
                    session_id=session_id,
                    transcript=result["text"],
                    segments=segments,
                    processing_time=processing_time,
                    status="completed"
                )
                
                # ERP 추출 결과 저장
                if erp_data:
                    erp_dict = erp_data.dict(by_alias=True)
                    extraction = supabase_mgr.save_erp_extraction(
                        session_id=session_id,
                        erp_data=erp_dict
                    )
                    extraction_id = extraction['id']
                    logger.info(f"ERP 추출 결과 저장 완료 - 추출 ID: {extraction_id}")
                    
                    # ERP 시스템에 자동 등록 (DB 저장 옵션이 활성화된 경우)
                    try:
                        logger.info("ERP 시스템에 자동 등록 중...")
                        
                        # Mock ERP ID 생성
                        erp_id = f"auto{uuid.uuid4().hex[:8]}"
                        
                        # ERP 등록 시뮬레이션 (실제 ERP 시스템 연동 시 이 부분을 수정)
                        erp_response_data = {
                            "status": "success",
                            "erp_id": erp_id,
                            "message": "STT 처리 중 ERP 시스템에 자동 등록되었습니다"
                        }
                        
                        # ERP 등록 로그 저장
                        supabase_mgr.save_erp_register_log(
                            extraction_id=extraction_id,
                            erp_id=erp_id,
                            status="success",
                            response_data=erp_response_data
                        )
                        
                        logger.info(f"ERP 자동 등록 완료 - ERP ID: {erp_id}, 추출 ID: {extraction_id}")
                        
                    except Exception as e:
                        logger.warning(f"ERP 자동 등록 실패 (계속 진행): {e}")
                        # 실패 로그도 저장
                        try:
                            supabase_mgr.save_erp_register_log(
                                extraction_id=extraction_id,
                                erp_id="",
                                status="failed",
                                response_data={"error": str(e)}
                            )
                        except:
                            pass
                
                logger.info(f"Supabase 저장 완료 - 세션 ID: {session_id}")
                
            except Exception as e:
                logger.warning(f"Supabase 저장 실패 (계속 진행): {e}")
        
        # 응답 생성
        response = STTResponse(
            status="success",
            transcript=result["text"],
            segments=segments,
            erp_data=erp_data,
            processing_time=processing_time,
            file_id=file_id
        )
        
        # 응답에 DB 저장 정보 추가 (동적 필드)
        if session_id:
            response.session_id = session_id
        if extraction_id:
            response.extraction_id = extraction_id
        
        logger.info(f"STT 처리 완료 - File ID: {file_id}, 처리시간: {processing_time:.2f}초")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT 처리 실패 - File ID: {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"STT 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/extract-erp")
async def extract_erp_from_text(
    conversation_text: str,
    erp_extractor=Depends(get_erp_extractor)
):
    """
    텍스트에서 직접 ERP 항목을 추출하는 API
    """
    try:
        logger.info("텍스트에서 ERP 데이터 추출 중...")
        
        erp_dict = erp_extractor.extract_erp_data(conversation_text)
        erp_data = ERPData(**erp_dict)
        
        return {
            "status": "success",
            "erp_data": erp_data,
            "message": "ERP 데이터 추출 완료"
        }
        
    except Exception as e:
        logger.error(f"ERP 데이터 추출 실패: {e}")
        raise HTTPException(status_code=500, detail=f"ERP 데이터 추출 중 오류가 발생했습니다: {str(e)}")

# 데이터 관리 엔드포인트들

@app.get("/api/sessions")
async def get_stt_sessions(
    limit: int = 50, 
    offset: int = 0,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """STT 세션 목록 조회"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        sessions = supabase_mgr.get_stt_sessions(limit=limit, offset=offset)
        return {
            "status": "success",
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        logger.error(f"세션 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 실패: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_stt_session(
    session_id: int,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """특정 STT 세션 상세 조회"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        session = supabase_mgr.get_stt_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # ERP 추출 결과도 함께 조회
        erp_extraction = supabase_mgr.get_erp_extraction(session_id)
        
        return {
            "status": "success",
            "session": session,
            "erp_extraction": erp_extraction
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"세션 조회 실패: {str(e)}")

@app.post("/api/sessions/{session_id}/extract-erp")
async def extract_erp_for_session(
    session_id: int,
    erp_extractor=Depends(get_erp_extractor),
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """기존 STT 세션에 대한 ERP 재추출"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        logger.info(f"세션 {session_id}에 대한 ERP 재추출 시작")
        
        # 세션 정보 조회
        session = supabase_mgr.get_stt_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # transcript 또는 segments 확인
        transcript = session.get('transcript')
        segments = session.get('segments')
        
        if not transcript and not segments:
            raise HTTPException(status_code=400, detail="세션에 텍스트 데이터가 없습니다")
        
        # ERP 데이터 추출
        erp_data = None
        try:
            if segments:
                # 세그먼트가 있으면 세그먼트에서 추출
                logger.info("세그먼트에서 ERP 데이터 추출 중...")
                
                # segments가 문자열인 경우 JSON으로 파싱
                if isinstance(segments, str):
                    try:
                        segments = json.loads(segments)
                        logger.info("세그먼트 JSON 파싱 완료")
                    except json.JSONDecodeError as e:
                        logger.warning(f"세그먼트 JSON 파싱 실패: {e}")
                        # 파싱 실패 시 전체 텍스트 사용
                        segments = None
                
                if segments and isinstance(segments, list):
                    erp_dict = erp_extractor.extract_from_segments(segments)
                else:
                    logger.info("세그먼트 데이터가 유효하지 않아 전체 텍스트 사용")
                    erp_dict = erp_extractor.extract_erp_data(transcript)
            else:
                # 세그먼트가 없으면 전체 텍스트에서 추출
                logger.info("전체 텍스트에서 ERP 데이터 추출 중...")
                erp_dict = erp_extractor.extract_erp_data(transcript)
            
            erp_data = ERPData(**erp_dict)
            logger.info(f"ERP 데이터 추출 완료: {erp_dict}")
            
        except Exception as e:
            logger.error(f"ERP 데이터 추출 실패: {e}")
            raise HTTPException(status_code=500, detail=f"ERP 데이터 추출 실패: {str(e)}")
        
        # 기존 ERP 추출 결과 확인
        existing_extraction = supabase_mgr.get_erp_extraction(session_id)
        
        extraction_id = None
        if existing_extraction:
            # 기존 추출 결과 업데이트
            logger.info(f"기존 ERP 추출 결과 업데이트 - 추출 ID: {existing_extraction['id']}")
            updated_extraction = supabase_mgr.update_erp_extraction(
                extraction_id=existing_extraction['id'],
                erp_data=erp_data.dict(by_alias=True)
            )
            extraction_id = updated_extraction['id']
        else:
            # 새로운 ERP 추출 결과 저장
            logger.info("새로운 ERP 추출 결과 저장")
            new_extraction = supabase_mgr.save_erp_extraction(
                session_id=session_id,
                erp_data=erp_data.dict(by_alias=True)
            )
            extraction_id = new_extraction['id']
        
        logger.info(f"ERP 재추출 완료 - 세션 ID: {session_id}, 추출 ID: {extraction_id}")
        
        return {
            "status": "success",
            "message": "ERP 재추출이 완료되었습니다",
            "session_id": session_id,
            "extraction_id": extraction_id,
            "erp_data": erp_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERP 재추출 실패 - 세션 ID: {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"ERP 재추출 중 오류가 발생했습니다: {str(e)}")

@app.get("/api/extractions")
async def get_erp_extractions(
    limit: int = 50, 
    offset: int = 0,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """ERP 추출 결과 목록 조회"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        extractions = supabase_mgr.get_erp_extractions(limit=limit, offset=offset)
        return {
            "status": "success",
            "extractions": extractions,
            "total": len(extractions)
        }
    except Exception as e:
        logger.error(f"추출 결과 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"추출 결과 목록 조회 실패: {str(e)}")

@app.get("/api/statistics")
async def get_system_statistics(
    date_filter: Optional[str] = None,
    month_filter: Optional[str] = None,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """
    시스템 통계 조회
    
    Args:
        date_filter: YYYY-MM-DD 형식의 특정 날짜 필터
        month_filter: YYYY-MM 형식의 월별 필터
    """
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        # 날짜 필터링 파라미터 결정
        filter_params = {}
        if date_filter:
            filter_params['date_filter'] = date_filter
        elif month_filter:
            filter_params['month_filter'] = month_filter
        
        stats = supabase_mgr.get_statistics(**filter_params)
        return {
            "status": "success",
            "statistics": stats,
            "applied_filter": {
                "date_filter": date_filter,
                "month_filter": month_filter
            }
        }
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@app.get("/api/audio-files")
async def get_audio_files():
    """
    src_record 디렉토리에서 사용 가능한 음성 파일 목록을 조회
    - 기존 src_record 직접 하위 파일들
    - 일자별 폴더(YYYY-MM-DD) 내의 파일들
    """
    try:
        if not os.path.exists(AUDIO_DIRECTORY):
            return {
                "status": "error",
                "message": f"음성 파일 디렉토리({AUDIO_DIRECTORY})가 존재하지 않습니다.",
                "files": [],
                "daily_files": {}
            }
        
        # 기존 src_record 직접 하위 음성 파일들 검색
        audio_files = []
        daily_files = {}
        
        for item in os.listdir(AUDIO_DIRECTORY):
            item_path = os.path.join(AUDIO_DIRECTORY, item)
            
            # 파일인 경우 (기존 방식)
            if os.path.isfile(item_path):
                file_extension = os.path.splitext(item)[1].lower()
                if file_extension in SUPPORTED_AUDIO_EXTENSIONS:
                    # 파일 정보 수집
                    file_stat = os.stat(item_path)
                    file_info = {
                        "filename": item,
                        "path": item,  # 기존 파일은 파일명만
                        "size": file_stat.st_size,
                        "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "extension": file_extension,
                        "location": "root"  # 루트 디렉토리 표시
                    }
                    audio_files.append(file_info)
            
            # 디렉토리인 경우 (일자별 폴더 확인)
            elif os.path.isdir(item_path):
                # YYYY-MM-DD 형식인지 확인
                try:
                    # 날짜 형식 검증
                    datetime.strptime(item, '%Y-%m-%d')
                    
                    # 일자별 폴더 내 음성 파일들 검색
                    daily_audio_files = []
                    for daily_filename in os.listdir(item_path):
                        daily_file_path = os.path.join(item_path, daily_filename)
                        
                        if os.path.isfile(daily_file_path):
                            file_extension = os.path.splitext(daily_filename)[1].lower()
                            if file_extension in SUPPORTED_AUDIO_EXTENSIONS:
                                # 파일 정보 수집
                                file_stat = os.stat(daily_file_path)
                                file_info = {
                                    "filename": daily_filename,
                                    "path": f"{item}/{daily_filename}",  # 날짜폴더/파일명
                                    "size": file_stat.st_size,
                                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                                    "extension": file_extension,
                                    "location": item  # 날짜 폴더명
                                }
                                daily_audio_files.append(file_info)
                    
                    if daily_audio_files:
                        daily_files[item] = daily_audio_files
                        
                except ValueError:
                    # 날짜 형식이 아닌 디렉토리는 무시
                    continue
        
        # 전체 파일 수 계산
        total_files = len(audio_files) + sum(len(files) for files in daily_files.values())
        
        # 파일명으로 정렬
        audio_files.sort(key=lambda x: x['filename'])
        for date_folder in daily_files:
            daily_files[date_folder].sort(key=lambda x: x['filename'])
        
        logger.info(f"발견된 음성 파일 수: 루트 {len(audio_files)}개, 일자별 {sum(len(files) for files in daily_files.values())}개 (총 {total_files}개)")
        
        return {
            "status": "success",
            "message": f"{total_files}개의 음성 파일을 발견했습니다.",
            "files": audio_files,  # 기존 루트 파일들
            "daily_files": daily_files,  # 일자별 폴더의 파일들
            "directory": AUDIO_DIRECTORY,
            "today_folder": datetime.now().strftime('%Y-%m-%d')  # 오늘 날짜 폴더명
        }
        
    except Exception as e:
        logger.error(f"음성 파일 목록 조회 실패: {e}")
        return {
            "status": "error",
            "message": f"음성 파일 목록 조회 중 오류가 발생했습니다: {str(e)}",
            "files": [],
            "daily_files": {}
        }

@app.get("/api/register-logs")
async def get_register_logs(
    limit: int = 50, 
    offset: int = 0,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """ERP 등록 로그 목록 조회"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        register_logs = supabase_mgr.get_erp_register_logs(limit=limit, offset=offset)
        return {
            "status": "success",
            "register_logs": register_logs,
            "total": len(register_logs)
        }
    except Exception as e:
        logger.error(f"등록 로그 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"등록 로그 목록 조회 실패: {str(e)}")

# 디렉토리별 파일 처리 상태 관련 API

@app.get("/api/directory-summary")
async def get_directory_summary(folder: str = None, supabase_mgr=Depends(get_supabase_manager_dep)):
    """디렉토리별 처리 현황 요약 조회"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        summary = supabase_mgr.get_directory_processing_summary(folder=folder)
        return {
            "status": "success",
            "summary": summary,
            "total_directories": len(summary),
            "folder_filter": folder
        }
    except Exception as e:
        logger.error(f"디렉토리별 요약 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"디렉토리별 요약 조회 실패: {str(e)}")

@app.get("/api/file-processing-status")
async def get_file_processing_status(
    directory: str = None,
    limit: int = 200,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """파일 처리 상태 조회 (디렉토리별 필터링 지원)"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        if directory:
            files = supabase_mgr.get_file_processing_status_by_directory(directory=directory, limit=limit)
        else:
            files = supabase_mgr.get_file_processing_status(limit=limit)
        
        return {
            "status": "success",
            "files": files,
            "total": len(files),
            "directory": directory if directory else "전체"
        }
    except Exception as e:
        logger.error(f"파일 처리 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 처리 상태 조회 실패: {str(e)}")

@app.get("/api/check-file-processed")
async def check_file_processed(
    file_path: str,
    supabase_mgr=Depends(get_supabase_manager_dep)
):
    """특정 파일의 처리 여부 확인"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        result = supabase_mgr.check_file_processed(file_path)
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        logger.error(f"파일 처리 상태 확인 실패 ({file_path}): {e}")
        raise HTTPException(status_code=500, detail=f"파일 처리 상태 확인 실패: {str(e)}")

@app.get("/api/processing-summary-enhanced")
async def get_processing_summary_enhanced(supabase_mgr=Depends(get_supabase_manager_dep)):
    """향상된 전체 처리 상태 요약 (디렉토리별 포함)"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        summary = supabase_mgr.get_processing_summary_enhanced()
        return {
            "status": "success",
            **summary
        }
    except Exception as e:
        logger.error(f"향상된 처리 요약 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"향상된 처리 요약 조회 실패: {str(e)}")

@app.post("/api/update-directory-view")
async def update_directory_view(supabase_mgr=Depends(get_supabase_manager_dep)):
    """디렉토리별 처리 현황 뷰를 업데이트합니다"""
    if not supabase_mgr:
        raise HTTPException(status_code=503, detail="Supabase가 설정되지 않았습니다")
    
    try:
        success = supabase_mgr.update_directory_view()
        if success:
            return {
                "status": "success",
                "message": "directory_processing_summary 뷰가 성공적으로 업데이트되었습니다"
            }
        else:
            raise HTTPException(status_code=500, detail="뷰 업데이트에 실패했습니다")
    except Exception as e:
        logger.error(f"뷰 업데이트 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"뷰 업데이트 오류: {str(e)}")

@app.post("/api/ensure-daily-folder")
async def ensure_daily_folder():
    """
    수동으로 오늘 날짜 폴더 생성
    스케줄러와 별개로 필요시 수동으로 폴더를 생성할 수 있습니다.
    """
    try:
        today = datetime.now()
        daily_path = ensure_today_folder_exists()
        
        if daily_path:
            return {
                "success": True,
                "message": "일별 폴더 생성 완료",
                "path": daily_path,
                "date": today.strftime('%Y-%m-%d'),
                "created_at": today.isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="폴더 생성에 실패했습니다")
            
    except Exception as e:
        logger.error(f"수동 폴더 생성 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"폴더 생성 실패: {str(e)}")

@app.get("/api/check-daily-folders")
async def check_daily_folders():
    """
    현재 생성된 일별 폴더들의 목록을 확인
    """
    try:
        if not os.path.exists(AUDIO_DIRECTORY):
            return {
                "success": True,
                "folders": [],
                "total_count": 0,
                "message": "src_record 디렉토리가 존재하지 않습니다"
            }
        
        # YYYY-MM-DD 형식의 폴더들만 필터링
        all_items = os.listdir(AUDIO_DIRECTORY)
        date_folders = []
        
        for item in all_items:
            item_path = os.path.join(AUDIO_DIRECTORY, item)
            if os.path.isdir(item_path):
                # YYYY-MM-DD 형식 검증
                try:
                    datetime.strptime(item, '%Y-%m-%d')
                    date_folders.append(item)
                except ValueError:
                    continue  # 날짜 형식이 아닌 폴더는 제외
        
        date_folders.sort(reverse=True)  # 최신 날짜부터 정렬
        
        return {
            "success": True,
            "folders": date_folders,
            "total_count": len(date_folders),
            "latest_folder": date_folders[0] if date_folders else None,
            "today_exists": datetime.now().strftime('%Y-%m-%d') in date_folders
        }
        
    except Exception as e:
        logger.error(f"일별 폴더 확인 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"폴더 확인 실패: {str(e)}")

@app.get("/api/environment-status")
async def get_environment_status():
    """환경변수 설정 상태 확인"""
    env_status = {}
    
    # OpenAI API Key 확인
    openai_key = os.getenv('OPENAI_API_KEY')
    env_status['OPENAI_API_KEY'] = bool(openai_key and openai_key not in ['your_openai_api_key_here', ''])
    
    # Supabase 설정 확인
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    env_status['SUPABASE_URL'] = bool(supabase_url and supabase_url not in ['your_supabase_url_here', ''])
    env_status['SUPABASE_ANON_KEY'] = bool(supabase_key and supabase_key not in ['your_supabase_anon_key_here', ''])
    
    # HuggingFace Token 확인
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    env_status['HUGGINGFACE_HUB_TOKEN'] = bool(hf_token and hf_token not in ['your_huggingface_token_here', ''])
    
    return {
        "status": "success",
        "environment_variables": env_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model-status")
async def get_model_status():
    """모델 로딩 상태 확인"""
    try:
        model_status = {
            "whisper_base_loaded": whisper_model is not None,
            "cached_models": list(cached_whisper_models.keys()),
            "erp_extractor_loaded": erp_extractor is not None,
            "supabase_connected": supabase_manager is not None
        }
        
        # 캐시된 모델들의 상세 정보
        model_details = {}
        for model_name, model in cached_whisper_models.items():
            model_details[model_name] = {
                "loaded": model is not None,
                "type": str(type(model).__name__) if model else None
            }
        
        return {
            "status": "success",
            "model_status": model_status,
            "model_details": model_details,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"모델 상태 확인 중 오류: {e}")
        return {
            "status": "error",
            "message": f"모델 상태 확인 실패: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/clear-whisper-cache")
async def clear_whisper_cache():
    """손상된 Whisper 모델 캐시를 정리합니다"""
    try:
        # 메모리 캐시 정리
        clear_model_cache()
        
        # 파일 캐시 정리
        success, cleared_paths = clear_whisper_file_cache()
        
        if success:
            return {
                "status": "success",
                "message": "Whisper 캐시가 성공적으로 정리되었습니다.",
                "cleared_paths": cleared_paths,
                "action_required": "API 서버를 재시작하거나 새 모델을 로딩해주세요.",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "warning",
                "message": "정리할 캐시 파일이 없거나 일부 정리에 실패했습니다.",
                "cleared_paths": cleared_paths,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"캐시 정리 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"캐시 정리 실패: {str(e)}")

@app.post("/api/reload-base-model")
async def reload_base_model():
    """Whisper tiny 모델을 다시 로딩합니다 (메모리 최적화)"""
    global whisper_model
    
    try:
        logger.info("Whisper tiny 모델 재로딩 시작... (메모리 최적화)")
        
        # 기존 모델 정리
        if whisper_model is not None:
            del whisper_model
        cached_whisper_models.clear()
        
        # 가비지 컬렉션
        import gc
        gc.collect()
        
        # 새로 로딩 (tiny 모델 강제)
        import time
        start_time = time.time()
        whisper_model = whisper.load_model("tiny")
        loading_time = time.time() - start_time
        
        # 캐시에 저장
        cached_whisper_models["tiny"] = whisper_model
        
        logger.info(f"Whisper tiny 모델 재로딩 완료 (소요시간: {loading_time:.2f}초)")
        
        return {
            "status": "success",
            "message": "Whisper tiny 모델이 성공적으로 재로딩되었습니다.",
            "loading_time": round(loading_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"모델 재로딩 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 재로딩 실패: {str(e)}")

# 서버 시작 시 모델 초기화
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트"""
    global scheduler
    logger.info("API 서버 시작 중...")
    
    # FFmpeg 경로 설정 (Render/Linux 환경 호환)
    try:
        # Render 환경에서는 FFmpeg가 이미 설치되어 있으므로 Windows 특정 경로 설정 건너뛰기
        import platform
        if platform.system() == "Windows":
            # Windows 환경에서만 특정 경로 확인
            potential_paths = [
                r"C:\Users\bangm\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build\bin",
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin"
            ]
            
            for ffmpeg_path in potential_paths:
                if os.path.exists(ffmpeg_path):
                    current_path = os.environ.get('PATH', '')
                    if ffmpeg_path not in current_path:
                        os.environ['PATH'] = current_path + os.pathsep + ffmpeg_path
                        logger.info(f"FFmpeg 경로 추가됨: {ffmpeg_path}")
                    else:
                        logger.info("FFmpeg 경로가 이미 PATH에 있습니다.")
                    break
            else:
                logger.info("Windows에서 FFmpeg 경로를 찾을 수 없지만 계속 진행합니다.")
        else:
            # Linux/Render 환경에서는 FFmpeg가 일반적으로 시스템에 설치되어 있음
            logger.info("Linux 환경: 시스템 FFmpeg 사용")
    except Exception as e:
        logger.warning(f"FFmpeg 경로 설정 실패 (계속 진행): {e}")
    
    try:
        # 모델 초기화
        initialize_models()
        
        # 일자별 폴더 생성
        daily_path = create_daily_directory()
        if daily_path:
            logger.info(f"일자별 폴더 설정 완료: {daily_path}")
        
        # 스케줄러 시작 (오류가 있어도 API 서버는 계속 실행)
        if SCHEDULER_AVAILABLE:
            try:
                scheduler = BackgroundScheduler()
                scheduler.add_job(
                    scheduled_daily_folder_creation,
                    CronTrigger(hour=0, minute=0),  # 매일 0시 실행
                    id='daily_folder_creation',
                    name='일별 폴더 자동 생성'
                )
                scheduler.start()
                logger.info("✅ 일별 폴더 생성 스케줄러 시작 완료 (매일 0시 실행)")
            except ImportError as e:
                logger.warning(f"⚠️ APScheduler 패키지가 설치되지 않았습니다: {e}")
                logger.warning("⚠️ 수동으로 설치하세요: pip install APScheduler>=3.10.0")
            except Exception as e:
                logger.error(f"⚠️ 스케줄러 시작 실패 (API 서버는 계속 실행): {e}")
        else:
            logger.warning("⚠️ APScheduler가 사용 불가능합니다. 일별 폴더 생성 스케줄러가 비활성화됩니다.")
        
        logger.info("API 서버 시작 완료")
    except Exception as e:
        logger.error(f"API 서버 시작 중 오류: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 이벤트"""
    global scheduler
    logger.info("API 서버 종료 중...")
    try:
        if SCHEDULER_AVAILABLE and scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("✅ 스케줄러 종료 완료")
        logger.info("API 서버 종료 완료")
    except Exception as e:
        logger.error(f"API 서버 종료 중 오류: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # 개발 서버 실행
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 