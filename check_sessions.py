#!/usr/bin/env python3
"""STT 세션 확인 스크립트"""

import requests
import json

def check_sessions():
    # STT 세션 목록 조회
    try:
        response = requests.get('http://localhost:8000/api/sessions?limit=10')
        print('=== STT 세션 목록 ===')
        print(f'상태 코드: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            sessions = data.get('sessions', [])
            print(f'총 세션 수: {len(sessions)}')
            
            if sessions:
                print('\n최근 세션들:')
                for i, session in enumerate(sessions[:5]):
                    print(f'{i+1}. ID: {session.get("id")}, 파일: {session.get("file_name")}, 상태: {session.get("status")}, 생성일: {session.get("created_at")}')
                    print(f'    텍스트 길이: {len(session.get("transcript", ""))} 문자')
            else:
                print('저장된 세션이 없습니다.')
        else:
            print(f'오류: {response.text}')
            
    except Exception as e:
        print(f'API 호출 실패: {e}')

    # ERP 추출 결과 조회
    try:
        response = requests.get('http://localhost:8000/api/extractions?limit=10')
        print('\n=== ERP 추출 결과 ===')
        print(f'상태 코드: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            extractions = data.get('extractions', [])
            print(f'총 추출 결과 수: {len(extractions)}')
            
            if extractions:
                print('\n최근 추출 결과들:')
                for i, extraction in enumerate(extractions[:3]):
                    print(f'{i+1}. ID: {extraction.get("id")}, 세션 ID: {extraction.get("session_id")}, 생성일: {extraction.get("created_at")}')
            else:
                print('추출 결과가 없습니다.')
        else:
            print(f'오류: {response.text}')
            
    except Exception as e:
        print(f'ERP 추출 결과 조회 실패: {e}')

if __name__ == "__main__":
    check_sessions()

