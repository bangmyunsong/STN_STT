#!/usr/bin/env python3
"""
Whisper 동작 테스트 스크립트
"""

try:
    print("🔄 Whisper 임포트 중...")
    import whisper
    print("✅ Whisper 임포트 성공!")
    
    print("🔄 PyTorch 임포트 중...")
    import torch
    print(f"✅ PyTorch 버전: {torch.__version__}")
    
    print("🔄 Tiny 모델 로드 중...")
    model = whisper.load_model("tiny")
    print("✅ Tiny 모델 로드 성공!")
    
    print("🎉 모든 테스트 통과! Whisper가 정상적으로 작동합니다.")
    
except Exception as e:
    print(f"❌ 오류 발생: {str(e)}")
    print(f"오류 타입: {type(e).__name__}")
    import traceback
    traceback.print_exc() 