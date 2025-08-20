#!/usr/bin/env python3
"""
STN STT 시스템 건강 상태 진단 스크립트
각 컴포넌트별로 문제점을 진단하고 해결책을 제시합니다.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_environment_variables():
    """환경변수 설정 확인"""
    logger.info("🔍 환경변수 설정 확인 중...")
    
    load_dotenv('config.env')
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API 키',
        'SUPABASE_URL': 'Supabase URL',
        'SUPABASE_ANON_KEY': 'Supabase Anonymous Key'
    }
    
    issues = []
    
    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if not value or value.startswith('your_'):
            issues.append(f"❌ {description} ({var_name}) 미설정")
        else:
            logger.info(f"✅ {description} 설정됨")
    
    return issues

def test_python_dependencies():
    """Python 의존성 설치 확인"""
    logger.info("🔍 Python 의존성 설치 확인 중...")
    
    required_packages = [
        ('whisper', 'OpenAI Whisper'),
        ('openai', 'OpenAI API'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('supabase', 'Supabase'),
        ('torch', 'PyTorch'),
        ('pydantic', 'Pydantic')
    ]
    
    issues = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {description} 설치됨")
        except ImportError:
            issues.append(f"❌ {description} ({package}) 미설치")
    
    return issues

def test_whisper_model():
    """Whisper 모델 로딩 테스트"""
    logger.info("🔍 Whisper 모델 로딩 테스트 중...")
    
    try:
        import whisper
        logger.info("Whisper base 모델 다운로드 시도... (시간이 걸릴 수 있습니다)")
        model = whisper.load_model("base")
        logger.info("✅ Whisper 모델 로딩 성공")
        return []
    except Exception as e:
        return [f"❌ Whisper 모델 로딩 실패: {e}"]

def test_openai_api():
    """OpenAI API 연결 테스트"""
    logger.info("🔍 OpenAI API 연결 테스트 중...")
    
    try:
        import openai
        from dotenv import load_dotenv
        
        load_dotenv('config.env')
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or api_key.startswith('your_'):
            return ["❌ OpenAI API 키가 설정되지 않음"]
        
        client = openai.OpenAI(api_key=api_key)
        
        # 간단한 테스트 요청
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        logger.info("✅ OpenAI API 연결 성공")
        return []
        
    except Exception as e:
        return [f"❌ OpenAI API 연결 실패: {e}"]

def test_supabase_connection():
    """Supabase 연결 테스트"""
    logger.info("🔍 Supabase 연결 테스트 중...")
    
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        
        load_dotenv('config.env')
        
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_ANON_KEY')
        
        if not url or url.startswith('your_'):
            return ["❌ Supabase URL이 설정되지 않음"]
            
        if not key or key.startswith('your_'):
            return ["❌ Supabase Anonymous Key가 설정되지 않음"]
        
        client = create_client(url, key)
        
        # 간단한 연결 테스트
        response = client.table('stt_sessions').select('*').limit(1).execute()
        
        logger.info("✅ Supabase 연결 성공")
        return []
        
    except Exception as e:
        return [f"❌ Supabase 연결 실패: {e}"]

def test_port_availability():
    """포트 사용 가능성 확인"""
    logger.info("🔍 포트 8000 사용 가능성 확인 중...")
    
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 8000))
        logger.info("✅ 포트 8000 사용 가능")
        return []
    except OSError:
        return ["❌ 포트 8000이 이미 사용 중 (다른 서버가 실행 중일 수 있음)"]

def main():
    """메인 진단 함수"""
    logger.info("🏥 STN STT 시스템 건강 상태 진단 시작")
    logger.info("=" * 60)
    
    all_issues = []
    
    # 각 테스트 실행
    tests = [
        ("환경변수", test_environment_variables),
        ("Python 의존성", test_python_dependencies), 
        ("포트 가용성", test_port_availability),
        ("Whisper 모델", test_whisper_model),
        ("OpenAI API", test_openai_api),
        ("Supabase", test_supabase_connection)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name} 테스트")
        logger.info("-" * 30)
        
        try:
            issues = test_func()
            all_issues.extend(issues)
            
            if not issues:
                logger.info(f"✅ {test_name} 테스트 통과")
                
        except Exception as e:
            issue = f"❌ {test_name} 테스트 중 오류: {e}"
            all_issues.append(issue)
            logger.error(issue)
    
    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("📋 진단 결과 요약")
    logger.info("=" * 60)
    
    if not all_issues:
        logger.info("🎉 모든 테스트 통과! 시스템이 정상적으로 작동할 것으로 예상됩니다.")
        logger.info("💡 이제 'python api_server.py'로 서버를 시작해보세요.")
    else:
        logger.info(f"⚠️  총 {len(all_issues)}개의 문제가 발견되었습니다:")
        for i, issue in enumerate(all_issues, 1):
            logger.info(f"   {i}. {issue}")
        
        logger.info("\n💡 해결 방법:")
        logger.info("   1. config.env 파일에서 API 키들을 확인하세요")
        logger.info("   2. pip install -r requirements.txt로 의존성을 재설치하세요")
        logger.info("   3. 인터넷 연결을 확인하세요")
        logger.info("   4. 이미 실행 중인 서버가 있다면 종료하세요")

if __name__ == "__main__":
    main()