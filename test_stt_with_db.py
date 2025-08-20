#!/usr/bin/env python3
"""DB 저장을 포함한 STT 테스트"""

import requests
import os

def test_stt_with_db():
    # 실제로 존재하는 파일로 테스트
    filename = "2025-08-11/025006013-028981660 20250529163932-0.mp3"
    
    # 파일 존재 확인
    file_path = os.path.join("src_record", filename)
    print(f"테스트 파일: {filename}")
    print(f"전체 경로: {file_path}")
    print(f"파일 존재 여부: {os.path.exists(file_path)}")
    
    # STT 처리 요청 (DB 저장 포함)
    params = {
        'filename': filename,
        'model_name': 'base',
        'extract_erp': 'true',   # ERP 추출 활성화
        'save_to_db': 'true'     # DB 저장 활성화
    }
    
    print(f"\n=== STT 처리 요청 (DB 저장 포함) ===")
    print(f"URL: http://localhost:8000/api/stt-process-file")
    print(f"Parameters: {params}")
    
    try:
        response = requests.post('http://localhost:8000/api/stt-process-file', 
                               params=params, 
                               timeout=300)
        
        print(f"응답 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ STT 처리 성공!")
            print(f"상태: {result.get('status')}")
            transcript = result.get('transcript', '')
            print(f"텍스트 (처음 100자): {transcript[:100]}...")
            print(f"처리 시간: {result.get('processing_time', 0):.2f}초")
            print(f"세션 ID: {result.get('session_id')}")
            print(f"추출 ID: {result.get('extraction_id')}")
            
            # ERP 데이터 확인
            erp_data = result.get('erp_data')
            if erp_data:
                print(f"ERP 추출 완료:")
                print(f"  요청기관: {erp_data.get('요청기관', 'N/A')}")
                print(f"  요청자: {erp_data.get('요청자', 'N/A')}")
                print(f"  요청 사항: {erp_data.get('요청 사항', 'N/A')[:50]}...")
            else:
                print("ERP 데이터 없음")
                
        else:
            print("❌ STT 처리 실패")
            print(f"응답 텍스트: {response.text}")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    test_stt_with_db()

