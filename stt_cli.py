#!/usr/bin/env python3
"""
음성-텍스트 변환기 (CLI 버전)
Command Line Interface for Speech-to-Text using Whisper
"""

import whisper
import argparse
import os
import json
from datetime import datetime

def convert_audio_to_text(audio_file, model_name="base", language=None, output_format="text"):
    """
    오디오 파일을 텍스트로 변환하는 함수
    
    Args:
        audio_file (str): 오디오 파일 경로
        model_name (str): Whisper 모델 이름
        language (str): 언어 코드 (None이면 자동 감지)
        output_format (str): 출력 형식 ("text", "json", "both")
    
    Returns:
        dict: 변환 결과
    """
    
    print(f"🎙️  음성 파일 처리 중: {audio_file}")
    print(f"📋 모델: {model_name}")
    print(f"🌍 언어: {language if language else '자동 감지'}")
    print("-" * 50)
    
    # 파일 존재 확인
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_file}")
    
    # 모델 로드
    print("🔄 Whisper 모델 로딩 중...")
    model = whisper.load_model(model_name)
    
    # 음성 인식 실행
    print("🎯 음성 인식 실행 중...")
    result = model.transcribe(audio_file, language=language)
    
    print("✅ 변환 완료!")
    print("-" * 50)
    
    return result

def save_results(result, audio_file, output_format):
    """
    결과를 파일로 저장하는 함수
    
    Args:
        result (dict): Whisper 변환 결과
        audio_file (str): 원본 오디오 파일명
        output_format (str): 출력 형식
    """
    
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format in ["text", "both"]:
        # 텍스트 파일 저장
        txt_filename = f"{base_name}_transcript_{timestamp}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        print(f"📄 텍스트 파일 저장됨: {txt_filename}")
    
    if output_format in ["json", "both"]:
        # JSON 파일 저장
        json_filename = f"{base_name}_detail_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📋 JSON 파일 저장됨: {json_filename}")

def display_results(result):
    """
    결과를 화면에 출력하는 함수
    
    Args:
        result (dict): Whisper 변환 결과
    """
    
    print("\n" + "="*60)
    print("📝 변환된 텍스트:")
    print("="*60)
    print(result["text"])
    print("="*60)
    
    print(f"\n📊 감지된 언어: {result.get('language', 'Unknown')}")
    
    # 세그먼트 정보 표시
    if "segments" in result and result["segments"]:
        print(f"🕒 총 세그먼트 수: {len(result['segments'])}")
        print("\n📍 시간별 세그먼트 (처음 3개):")
        print("-" * 60)
        
        for i, segment in enumerate(result["segments"][:3]):
            start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
            end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
            print(f"[{start_time}-{end_time}] {segment['text'].strip()}")
        
        if len(result["segments"]) > 3:
            print(f"... 및 {len(result['segments']) - 3}개 세그먼트 더")

def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Whisper AI를 사용한 음성-텍스트 변환기 (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python stt_cli.py audio.mp3
  python stt_cli.py audio.wav --model medium --language ko
  python stt_cli.py audio.m4a --output json --save
        """
    )
    
    parser.add_argument("audio_file", help="변환할 오디오 파일 경로")
    
    parser.add_argument("--model", "-m", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       default="base",
                       help="Whisper 모델 선택 (기본값: base)")
    
    parser.add_argument("--language", "-l",
                       help="언어 코드 (예: ko, en, ja). 미지정시 자동 감지")
    
    parser.add_argument("--output", "-o",
                       choices=["text", "json", "both"],
                       default="text",
                       help="출력 형식 (기본값: text)")
    
    parser.add_argument("--save", "-s",
                       action="store_true",
                       help="결과를 파일로 저장")
    
    parser.add_argument("--quiet", "-q",
                       action="store_true",
                       help="텍스트 결과만 출력 (조용한 모드)")
    
    args = parser.parse_args()
    
    try:
        # 음성 변환 실행
        result = convert_audio_to_text(
            args.audio_file, 
            args.model, 
            args.language, 
            args.output
        )
        
        # 결과 출력
        if args.quiet:
            print(result["text"])
        else:
            display_results(result)
        
        # 파일 저장
        if args.save:
            save_results(result, args.audio_file, args.output)
            
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 