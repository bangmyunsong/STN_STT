#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.env 설정 및 발화자 구분 기능 진단 스크립트
"""

import os
from dotenv import load_dotenv

def test_env_loading():
    print("🔍 config.env 파일 테스트")
    print("=" * 50)
    
    # config.env 파일 존재 확인
    if os.path.exists('config.env'):
        print("✅ config.env 파일이 존재합니다.")
    else:
        print("❌ config.env 파일이 없습니다.")
        return False
    
    # dotenv 로드
    load_dotenv('config.env')
    
    # 토큰 확인
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if token:
        print(f"✅ HuggingFace 토큰이 로드되었습니다: {token[:8]}...{token[-4:]}")
    else:
        print("❌ HuggingFace 토큰을 찾을 수 없습니다.")
        return False
    
    return True

def test_pyannote_import():
    print("\n🔍 pyannote-audio 패키지 테스트")
    print("=" * 50)
    
    try:
        from pyannote.audio import Pipeline
        print("✅ pyannote-audio 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ pyannote-audio 패키지가 설치되지 않았습니다: {e}")
        return False

def test_token_validity():
    print("\n🔍 토큰 유효성 테스트")
    print("=" * 50)
    
    try:
        from pyannote.audio import Pipeline
        token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        
        if not token:
            print("❌ 토큰이 없어 테스트를 건너뜁니다.")
            return False
        
        # 공개 모델로 테스트
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", token=token)
            print("✅ 최신 API (token 파라미터)로 모델 로드 성공!")
            return True
        except Exception as e1:
            try:
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
                print("✅ 구버전 API (use_auth_token 파라미터)로 모델 로드 성공!")
                return True
            except Exception as e2:
                print(f"❌ 모델 로드 실패:")
                print(f"   - 최신 API 오류: {e1}")
                print(f"   - 구버전 API 오류: {e2}")
                return False
                
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False

def main():
    print("🎙️ STT 발화자 구분 진단 도구")
    print("=" * 50)
    
    # 각 테스트 실행
    tests = [
        ("환경변수 로딩", test_env_loading),
        ("pyannote-audio 패키지", test_pyannote_import),
        ("토큰 유효성", test_token_validity)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류 발생: {e}")
            results.append((test_name, False))
    
    # 최종 결과
    print("\n📊 최종 진단 결과")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n전체 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 발화자 구분 기능을 사용할 수 있습니다.")
    else:
        print("⚠️ 일부 테스트 실패. 발화자 구분 기능에 제한이 있을 수 있습니다.")
        print("\n💡 해결 방안:")
        
        if not results[0][1]:  # 환경변수 로딩 실패
            print("- config.env 파일을 확인하세요.")
            
        if not results[1][1]:  # pyannote-audio 설치 실패
            print("- pip install pyannote-audio 명령으로 패키지를 설치하세요.")
            
        if not results[2][1]:  # 토큰 유효성 실패
            print("- HuggingFace 토큰이 유효한지 확인하세요.")
            print("- https://hf.co/settings/tokens 에서 새 토큰을 발급받으세요.")

if __name__ == "__main__":
    main() 