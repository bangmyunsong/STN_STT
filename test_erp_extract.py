#!/usr/bin/env python3
"""
ERP 추출 엔드포인트 테스트 스크립트
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_health():
    """API 서버 상태 확인"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"Health Check - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health Check Error: {e}")
        return False

def test_sessions_list():
    """세션 목록 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/sessions", timeout=10)
        print(f"Sessions List - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            sessions = data.get('sessions', [])
            print(f"Found {len(sessions)} sessions")
            return sessions
        else:
            print(f"Error: {response.text}")
            return []
    except Exception as e:
        print(f"Sessions List Error: {e}")
        return []

def test_erp_extract(session_id):
    """ERP 추출 테스트"""
    try:
        print(f"\n=== Testing ERP Extract for Session {session_id} ===")
        response = requests.post(f"{API_BASE_URL}/api/sessions/{session_id}/extract-erp", timeout=30)
        print(f"ERP Extract - Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message')}")
            return True
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"ERP Extract Error: {e}")
        return False

if __name__ == "__main__":
    print("=== ERP 추출 엔드포인트 테스트 ===\n")
    
    # 1. Health Check
    if not test_health():
        print("❌ API 서버가 응답하지 않습니다.")
        exit(1)
    
    print("\n" + "="*50)
    
    # 2. 세션 목록 조회
    sessions = test_sessions_list()
    if not sessions:
        print("❌ 세션이 없거나 조회할 수 없습니다.")
        exit(1)
    
    print("\n" + "="*50)
    
    # 3. 첫 번째 세션으로 ERP 추출 테스트
    test_session_id = sessions[0].get('id')
    print(f"Testing with Session ID: {test_session_id}")
    
    success = test_erp_extract(test_session_id)
    
    if success:
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n❌ 테스트 실패!") 