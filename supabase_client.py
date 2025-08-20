"""
Supabase 클라이언트 모듈
STT 시스템 데이터 저장 및 관리를 위한 Supabase 연동
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
import json

# 환경변수 로드
load_dotenv('config.env')

# 로깅 설정
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Supabase 데이터베이스 관리 클래스"""
    
    def __init__(self):
        """Supabase 클라이언트 초기화"""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or self.supabase_url == 'your_supabase_url_here':
            raise ValueError("Supabase URL이 설정되지 않았습니다. config.env 파일을 확인하세요.")
        
        if not self.supabase_key or self.supabase_key == 'your_supabase_anon_key_here':
            raise ValueError("Supabase Anonymous Key가 설정되지 않았습니다. config.env 파일을 확인하세요.")
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"Supabase 클라이언트 초기화 실패: {e}")
            raise
    
    # STT 세션 관련 메소드들
    
    def create_stt_session(self, file_name: str, file_id: str, model_name: str = "base", 
                          language: Optional[str] = None) -> Dict[str, Any]:
        """STT 처리 세션을 생성합니다"""
        try:
            session_data = {
                "file_id": file_id,
                "file_name": file_name,
                "model_name": model_name,
                "language": language,
                "status": "processing",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.client.table('stt_sessions').insert(session_data).execute()
            
            if result.data:
                logger.info(f"STT 세션 생성 완료 - ID: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("STT 세션 생성 실패")
                
        except Exception as e:
            logger.error(f"STT 세션 생성 실패: {e}")
            raise
    
    def update_stt_session(self, session_id: int, transcript: str, segments: List[Dict],
                          processing_time: float, status: str = "completed") -> Dict[str, Any]:
        """STT 처리 결과로 세션을 업데이트합니다"""
        try:
            update_data = {
                "transcript": transcript,
                "segments": json.dumps(segments, ensure_ascii=False),
                "processing_time": processing_time,
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.client.table('stt_sessions').update(update_data).eq('id', session_id).execute()
            
            if result.data:
                logger.info(f"STT 세션 업데이트 완료 - ID: {session_id}")
                return result.data[0]
            else:
                raise Exception("STT 세션 업데이트 실패")
                
        except Exception as e:
            logger.error(f"STT 세션 업데이트 실패: {e}")
            raise
    
    def get_stt_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """STT 세션 정보를 조회합니다"""
        try:
            result = self.client.table('stt_sessions').select('*').eq('id', session_id).execute()
            
            if result.data:
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"STT 세션 조회 실패: {e}")
            return None
    
    def get_stt_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """STT 세션 목록을 조회합니다"""
        try:
            result = self.client.table('stt_sessions')\
                .select('*')\
                .order('created_at', desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"STT 세션 목록 조회 실패: {e}")
            return []
    
    # ERP 추출 관련 메소드들
    
    def save_erp_extraction(self, session_id: int, erp_data: Dict[str, str],
                           confidence_score: Optional[float] = None) -> Dict[str, Any]:
        """ERP 추출 결과를 저장합니다"""
        try:
            # 날짜 형식 변환 (YYYY-MM-DD 문자열로 저장)
            요청일_str = erp_data.get("요청일", "")
            요청일 = None
            if 요청일_str and 요청일_str != "정보 없음":
                try:
                    # 날짜 유효성 검증 후 문자열로 저장
                    datetime.strptime(요청일_str, "%Y-%m-%d")
                    요청일 = 요청일_str  # 문자열로 저장
                except ValueError:
                    logger.warning(f"날짜 형식 오류: {요청일_str}")
                    요청일 = None
            
            # 시간 형식 변환 (문자열 -> INTEGER)
            요청시간_str = erp_data.get("요청시간", "")
            요청시간 = None
            if 요청시간_str and 요청시간_str != "정보 없음":
                try:
                    요청시간 = int(요청시간_str)
                except ValueError:
                    logger.warning(f"시간 형식 오류: {요청시간_str}")
                    요청시간 = None
            
            extraction_data = {
                "session_id": session_id,
                "as_지원": erp_data.get("AS 및 지원", ""),
                "요청기관": erp_data.get("요청기관", ""),
                "작업국소": erp_data.get("작업국소", ""),
                "요청일": 요청일,
                "요청시간": 요청시간,
                "요청자": erp_data.get("요청자", ""),
                "지원인원수": erp_data.get("지원인원수", ""),
                "지원요원": erp_data.get("지원요원", ""),
                "장비명": erp_data.get("장비명", ""),
                "기종명": erp_data.get("기종명", ""),
                "as_기간만료여부": erp_data.get("A/S기간만료여부", ""),
                "시스템명": erp_data.get("시스템명(고객사명)", ""),
                "요청사항": erp_data.get("요청 사항", ""),
                "confidence_score": confidence_score,
                "raw_extraction": json.dumps(erp_data, ensure_ascii=False),
                "created_at": datetime.now().isoformat()
            }
            
            result = self.client.table('erp_extractions').insert(extraction_data).execute()
            
            if result.data:
                logger.info(f"ERP 추출 결과 저장 완료 - ID: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("ERP 추출 결과 저장 실패")
                
        except Exception as e:
            logger.error(f"ERP 추출 결과 저장 실패: {e}")
            raise
    
    def get_erp_extraction(self, session_id: int) -> Optional[Dict[str, Any]]:
        """세션의 ERP 추출 결과를 조회합니다"""
        try:
            result = self.client.table('erp_extractions')\
                .select('*')\
                .eq('session_id', session_id)\
                .execute()
            
            if result.data:
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"ERP 추출 결과 조회 실패: {e}")
            return None
    
    def update_erp_extraction(self, extraction_id: int, erp_data: Dict[str, str],
                             confidence_score: Optional[float] = None) -> Dict[str, Any]:
        """ERP 추출 결과를 업데이트합니다"""
        try:
            # 날짜 형식 변환 (YYYY-MM-DD 문자열로 저장)
            요청일_str = erp_data.get("요청일", "")
            요청일 = None
            if 요청일_str and 요청일_str != "정보 없음":
                try:
                    # 날짜 유효성 검증 후 문자열로 저장
                    datetime.strptime(요청일_str, "%Y-%m-%d")
                    요청일 = 요청일_str  # 문자열로 저장
                except ValueError:
                    logger.warning(f"날짜 형식 오류: {요청일_str}")
                    요청일 = None
            
            # 시간 형식 변환 (문자열 -> INTEGER)
            요청시간_str = erp_data.get("요청시간", "")
            요청시간 = None
            if 요청시간_str and 요청시간_str != "정보 없음":
                try:
                    요청시간 = int(요청시간_str)
                except ValueError:
                    logger.warning(f"시간 형식 오류: {요청시간_str}")
                    요청시간 = None
            
            update_data = {
                "as_지원": erp_data.get("AS 및 지원", ""),
                "요청기관": erp_data.get("요청기관", ""),
                "작업국소": erp_data.get("작업국소", ""),
                "요청일": 요청일,
                "요청시간": 요청시간,
                "요청자": erp_data.get("요청자", ""),
                "지원인원수": erp_data.get("지원인원수", ""),
                "지원요원": erp_data.get("지원요원", ""),
                "장비명": erp_data.get("장비명", ""),
                "기종명": erp_data.get("기종명", ""),
                "as_기간만료여부": erp_data.get("A/S기간만료여부", ""),
                "시스템명": erp_data.get("시스템명(고객사명)", ""),
                "요청사항": erp_data.get("요청 사항", ""),
                "confidence_score": confidence_score,
                "raw_extraction": json.dumps(erp_data, ensure_ascii=False),
                "updated_at": datetime.now().isoformat()
            }
            
            result = self.client.table('erp_extractions').update(update_data).eq('id', extraction_id).execute()
            
            if result.data:
                logger.info(f"ERP 추출 결과 업데이트 완료 - ID: {extraction_id}")
                return result.data[0]
            else:
                raise Exception("ERP 추출 결과 업데이트 실패")
                
        except Exception as e:
            logger.error(f"ERP 추출 결과 업데이트 실패: {e}")
            raise
    
    def get_erp_extractions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """ERP 추출 결과 목록을 조회합니다"""
        try:
            result = self.client.table('erp_extractions')\
                .select('*, stt_sessions!inner(file_name, created_at)')\
                .order('created_at', desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"ERP 추출 결과 목록 조회 실패: {e}")
            return []
    
    # ERP 등록 로그 관련 메소드들
    
    def save_erp_register_log(self, extraction_id: int, erp_id: str,
                             status: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """ERP 등록 로그를 저장합니다"""
        try:
            log_data = {
                "extraction_id": extraction_id,
                "erp_id": erp_id,
                "status": status,
                "response_data": json.dumps(response_data, ensure_ascii=False),
                "registered_at": datetime.now().isoformat()
            }
            
            result = self.client.table('erp_register_logs').insert(log_data).execute()
            
            if result.data:
                logger.info(f"ERP 등록 로그 저장 완료 - ID: {result.data[0]['id']}")
                return result.data[0]
            else:
                raise Exception("ERP 등록 로그 저장 실패")
                
        except Exception as e:
            logger.error(f"ERP 등록 로그 저장 실패: {e}")
            raise
    
    def get_erp_register_logs(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """ERP 등록 로그 목록을 조회합니다"""
        try:
            result = self.client.table('erp_register_logs')\
                .select('*')\
                .order('registered_at', desc=True)\
                .range(offset, offset + limit - 1)\
                .execute()
            
            if result.data:
                # JSON 응답 데이터 파싱
                for log in result.data:
                    if log.get('response_data') and isinstance(log['response_data'], str):
                        try:
                            log['response_data'] = json.loads(log['response_data'])
                        except json.JSONDecodeError:
                            pass
                
                logger.info(f"ERP 등록 로그 {len(result.data)}건 조회 완료")
                return result.data
            else:
                return []
                
        except Exception as e:
            logger.error(f"ERP 등록 로그 조회 실패: {e}")
            return []
    
    # 통계 및 분석 메소드들
    
    def get_statistics(self, date_filter: Optional[str] = None, month_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        시스템 통계를 조회합니다
        
        Args:
            date_filter: YYYY-MM-DD 형식의 특정 날짜 필터
            month_filter: YYYY-MM 형식의 월별 필터
        """
        try:
            # 날짜 필터링 조건 설정
            date_conditions = {}
            filter_info = "전체"
            
            if date_filter:
                # 특정 날짜 필터링 (YYYY-MM-DD)
                start_date = f"{date_filter}T00:00:00"
                end_date = f"{date_filter}T23:59:59"
                date_conditions = {
                    'start': start_date,
                    'end': end_date
                }
                filter_info = f"날짜: {date_filter}"
            elif month_filter:
                # 월별 필터링 (YYYY-MM)
                start_date = f"{month_filter}-01T00:00:00"
                # 다음 달 첫날 구하기
                year, month = map(int, month_filter.split('-'))
                if month == 12:
                    next_year, next_month = year + 1, 1
                else:
                    next_year, next_month = year, month + 1
                end_date = f"{next_year:04d}-{next_month:02d}-01T00:00:00"
                date_conditions = {
                    'start': start_date,
                    'end': end_date
                }
                filter_info = f"월: {month_filter}"
            
            # STT 세션 통계
            stt_query = self.client.table('stt_sessions').select('id, processing_time, status', count='exact')
            if date_conditions:
                stt_query = stt_query.gte('created_at', date_conditions['start']).lt('created_at', date_conditions['end'])
            stt_result = stt_query.execute()
            
            # 완료된 세션 수 계산
            completed_sessions = 0
            total_processing_time = 0
            processing_count = 0
            
            if stt_result.data:
                for session in stt_result.data:
                    if session.get('status') == 'completed':
                        completed_sessions += 1
                        if session.get('processing_time'):
                            total_processing_time += session['processing_time']
                            processing_count += 1
            
            # 평균 처리 시간 계산
            avg_processing_time = total_processing_time / processing_count if processing_count > 0 else 0
            
            # ERP 추출 통계
            erp_query = self.client.table('erp_extractions').select('id', count='exact')
            if date_conditions:
                erp_query = erp_query.gte('created_at', date_conditions['start']).lt('created_at', date_conditions['end'])
            erp_result = erp_query.execute()
            
            # ERP 등록 통계 (전체)
            register_query = self.client.table('erp_register_logs').select('id, status', count='exact')
            if date_conditions:
                register_query = register_query.gte('registered_at', date_conditions['start']).lt('registered_at', date_conditions['end'])
            register_result = register_query.execute()
            
            # 성공한 등록 수 계산
            success_registers = 0
            if register_result.data:
                success_registers = sum(1 for log in register_result.data if log.get('status') == 'success')
            
            # 최근 7일 통계 (날짜 필터와 별개로 항상 계산)
            seven_days_ago = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) 
                            - timedelta(days=7)).isoformat()
            
            recent_sessions = self.client.table('stt_sessions')\
                .select('id', count='exact')\
                .gte('created_at', seven_days_ago)\
                .execute()
            
            return {
                "total_sessions": stt_result.count or 0,
                "completed_sessions": completed_sessions,
                "total_extractions": erp_result.count or 0,
                "total_registers": register_result.count or 0,
                "success_registers": success_registers,
                "avg_processing_time": avg_processing_time,
                "recent_sessions_7days": recent_sessions.count or 0,
                "filter_applied": filter_info,
                "date_filter": date_filter,
                "month_filter": month_filter,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {
                "total_sessions": 0,
                "completed_sessions": 0,
                "total_extractions": 0,
                "total_registers": 0,
                "success_registers": 0,
                "avg_processing_time": 0,
                "recent_sessions_7days": 0,
                "filter_applied": "오류",
                "date_filter": date_filter,
                "month_filter": month_filter,
                "last_updated": datetime.now().isoformat(),
                "error": str(e)
            }
    
    # 헬스 체크
    
    def health_check(self) -> bool:
        """Supabase 연결 상태를 확인합니다"""
        try:
            # 간단한 쿼리로 연결 테스트
            result = self.client.table('stt_sessions').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase 연결 상태 확인 실패: {e}")
            return False
    
    # 디렉토리별 파일 처리 상태 관련 메소드들
    
    def get_file_processing_status(self, file_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """음성파일 처리 상태 조회 (디렉토리 구조 고려)"""
        try:
            query = self.client.table('audio_file_processing_status').select('*')
            
            if file_name:
                query = query.eq('전체파일경로', file_name)
            
            result = query.limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"파일 처리 상태 조회 실패: {e}")
            return []
    
    def get_file_processing_status_by_directory(self, directory: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """디렉토리별 음성파일 처리 상태 조회"""
        try:
            query = self.client.table('audio_file_processing_status').select('*')
            
            if directory:
                if directory == "루트":
                    # 루트 파일만 조회 (/ 포함하지 않는 파일명)
                    query = query.eq('디렉토리', '루트')
                else:
                    # 특정 일자별 폴더 조회
                    query = query.eq('디렉토리', directory)
            
            result = query.limit(limit).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"디렉토리별 파일 처리 상태 조회 실패: {e}")
            return []
    
    def get_directory_processing_summary(self, folder: str = None) -> List[Dict[str, Any]]:
        """디렉토리별 처리 현황 요약"""
        try:
            query = self.client.table('directory_processing_summary').select('*')
            
            if folder:
                if folder == "루트 폴더":
                    # 루트 폴더: 디렉토리가 '루트'인 것만
                    query = query.eq('디렉토리', '루트')
                elif folder != "전체 폴더":
                    # 특정 날짜 폴더: 디렉토리에 해당 날짜가 포함된 것
                    query = query.ilike('디렉토리', f'%{folder}%')
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"디렉토리별 요약 조회 실패: {e}")
            return []
    
    def check_file_processed(self, file_path: str) -> Dict[str, Any]:
        """특정 파일의 처리 여부 확인"""
        try:
            result = self.client.table('audio_file_processing_status')\
                .select('*')\
                .eq('전체파일경로', file_path)\
                .limit(1)\
                .execute()
            
            if result.data:
                return {
                    'processed': True,
                    'status': result.data[0].get('전체_처리상태', '미처리'),
                    'progress': result.data[0].get('처리_진행률', 0),
                    'session_id': result.data[0].get('session_id'),
                    'extraction_id': result.data[0].get('extraction_id'),
                    'directory': result.data[0].get('디렉토리', '루트'),
                    'filename': result.data[0].get('파일명', file_path)
                }
            else:
                return {'processed': False, 'status': '미처리', 'progress': 0}
                
        except Exception as e:
            logger.error(f"파일 처리 상태 확인 실패 ({file_path}): {e}")
            return {'processed': False, 'status': '오류', 'progress': 0}
    
    def get_processing_summary_enhanced(self) -> Dict[str, Any]:
        """향상된 전체 처리 상태 요약 통계 (디렉토리별 포함)"""
        try:
            # 전체 상태 조회
            result = self.client.table('audio_file_processing_status').select('전체_처리상태, 디렉토리').execute()
            
            if not result.data:
                return {}
            
            status_counts = {}
            directory_counts = {}
            
            for row in result.data:
                status = row.get('전체_처리상태', '미처리')
                directory = row.get('디렉토리', '루트')
                
                # 전체 상태 집계
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # 디렉토리별 집계
                if directory not in directory_counts:
                    directory_counts[directory] = {'total': 0, 'completed': 0}
                directory_counts[directory]['total'] += 1
                if status == '완료':
                    directory_counts[directory]['completed'] += 1
            
            total = len(result.data)
            completion_rate = (status_counts.get('완료', 0) / total * 100) if total > 0 else 0
            
            return {
                'total_files': total,
                'completed': status_counts.get('완료', 0),
                'stt_only': status_counts.get('STT완료', 0),
                'extracted': status_counts.get('추출완료', 0),
                'processing': status_counts.get('처리중', 0),
                'pending': status_counts.get('미처리', 0),
                'completion_rate': completion_rate,
                'directory_breakdown': directory_counts
            }
            
        except Exception as e:
            logger.error(f"향상된 처리 요약 통계 조회 실패: {e}")
            return {}

    def update_directory_view(self):
        """디렉토리별 처리 현황 뷰를 업데이트합니다"""
        try:
            # 날짜별 폴더를 지원하는 뷰로 업데이트
            view_sql = """
            CREATE OR REPLACE VIEW directory_processing_summary AS
            SELECT 
                CASE 
                    WHEN s.file_name LIKE 'src_record/%/%' 
                    THEN SPLIT_PART(s.file_name, '/', 2)  -- src_record/2025-07-16/파일명 -> 2025-07-16
                    WHEN s.file_name LIKE '%/%' 
                    THEN SPLIT_PART(s.file_name, '/', 1)  -- 기타 폴더/파일명 -> 폴더
                    ELSE '루트'
                END as 디렉토리,
                COUNT(*) as 총_파일수,
                COUNT(CASE WHEN s.status = 'completed' THEN 1 END) as stt_완료수,
                COUNT(e.id) as erp_추출수,
                COUNT(CASE WHEN r.status = 'success' THEN 1 END) as erp_등록수,
                ROUND(
                    COUNT(CASE WHEN r.status = 'success' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 1
                ) as 완료율,
                MIN(s.created_at) as 최초_처리일시,
                MAX(s.created_at) as 최근_처리일시
            FROM stt_sessions s
            LEFT JOIN erp_extractions e ON s.id = e.session_id
            LEFT JOIN erp_register_logs r ON e.id = r.extraction_id
            GROUP BY 디렉토리
            ORDER BY 디렉토리 DESC;
            """
            
            self.client.rpc('sql', {'query': view_sql}).execute()
            logger.info("directory_processing_summary 뷰가 성공적으로 업데이트되었습니다")
            return True
            
        except Exception as e:
            logger.error(f"뷰 업데이트 실패: {e}")
            return False


# 전역 Supabase 매니저 인스턴스
_supabase_manager: Optional[SupabaseManager] = None

def get_supabase_manager() -> SupabaseManager:
    """Supabase 매니저 싱글톤 인스턴스를 반환합니다"""
    global _supabase_manager
    
    if _supabase_manager is None:
        _supabase_manager = SupabaseManager()
    
    return _supabase_manager

# 편의 함수들
def save_stt_result(file_name: str, file_id: str, transcript: str, 
                   segments: List[Dict], processing_time: float,
                   model_name: str = "base", language: Optional[str] = None) -> int:
    """STT 결과를 Supabase에 저장하는 편의 함수"""
    manager = get_supabase_manager()
    
    # 세션 생성
    session = manager.create_stt_session(file_name, file_id, model_name, language)
    
    # 결과 업데이트
    manager.update_stt_session(session['id'], transcript, segments, processing_time)
    
    return session['id']

def save_erp_result(session_id: int, erp_data: Dict[str, str]) -> int:
    """ERP 추출 결과를 Supabase에 저장하는 편의 함수"""
    manager = get_supabase_manager()
    
    extraction = manager.save_erp_extraction(session_id, erp_data)
    return extraction['id']

# 데이터베이스 스키마 생성 SQL (참고용)
DATABASE_SCHEMA = """
-- STT 처리 세션 테이블
CREATE TABLE IF NOT EXISTS stt_sessions (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(100) UNIQUE NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    model_name VARCHAR(50) DEFAULT 'base',
    language VARCHAR(10),
    transcript TEXT,
    segments JSONB,
    processing_time FLOAT,
    status VARCHAR(20) DEFAULT 'processing',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ERP 추출 결과 테이블
CREATE TABLE IF NOT EXISTS erp_extractions (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES stt_sessions(id) ON DELETE CASCADE,
    as_지원 VARCHAR(50),
    요청기관 VARCHAR(200),
    작업국소 VARCHAR(100),
    요청일 DATE,
    요청시간 INTEGER,
    요청자 VARCHAR(100),
    지원인원수 VARCHAR(20),
    지원요원 VARCHAR(100),
    장비명 VARCHAR(100),
    기종명 VARCHAR(100),
    as_기간만료여부 VARCHAR(20),
    시스템명 VARCHAR(200),
    요청사항 TEXT,
    confidence_score FLOAT,
    raw_extraction JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ERP 등록 로그 테이블
CREATE TABLE IF NOT EXISTS erp_register_logs (
    id SERIAL PRIMARY KEY,
    extraction_id INTEGER REFERENCES erp_extractions(id) ON DELETE CASCADE,
    erp_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_data JSONB,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_stt_sessions_file_id ON stt_sessions(file_id);
CREATE INDEX IF NOT EXISTS idx_stt_sessions_created_at ON stt_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_stt_sessions_file_name ON stt_sessions(file_name);
CREATE INDEX IF NOT EXISTS idx_stt_sessions_file_name_pattern ON stt_sessions USING btree (file_name text_pattern_ops);
CREATE INDEX IF NOT EXISTS idx_erp_extractions_session_id ON erp_extractions(session_id);
CREATE INDEX IF NOT EXISTS idx_erp_extractions_created_at ON erp_extractions(created_at);
CREATE INDEX IF NOT EXISTS idx_erp_register_logs_extraction_id ON erp_register_logs(extraction_id);
CREATE INDEX IF NOT EXISTS idx_erp_register_logs_status ON erp_register_logs(status);

-- 디렉토리 구조를 고려한 음성파일 처리 상태 뷰
CREATE OR REPLACE VIEW audio_file_processing_status AS
SELECT 
    -- 파일 정보 (디렉토리 구조 고려)
    s.file_name as 전체파일경로,
    CASE 
        WHEN s.file_name LIKE '%/%' 
        THEN SPLIT_PART(s.file_name, '/', 1)
        ELSE '루트'
    END as 디렉토리,
    CASE 
        WHEN s.file_name LIKE '%/%' 
        THEN SPLIT_PART(s.file_name, '/', -1)
        ELSE s.file_name
    END as 파일명,
    
    -- 기존 정보
    s.id as session_id,
    s.file_id,
    s.model_name as stt_모델,
    s.status as stt_상태,
    s.processing_time as stt_처리시간,
    
    -- ERP 추출 상태
    CASE WHEN e.id IS NOT NULL THEN 'Y' ELSE 'N' END as erp_추출여부,
    e.id as extraction_id,
    e.confidence_score as erp_신뢰도,
    
    -- ERP 등록 상태  
    CASE WHEN r.id IS NOT NULL AND r.status = 'success' THEN 'Y' ELSE 'N' END as erp_등록여부,
    r.erp_id,
    r.status as 등록_상태,
    
    -- 전체 처리 상태
    CASE 
        WHEN r.id IS NOT NULL AND r.status = 'success' THEN '완료'
        WHEN e.id IS NOT NULL THEN '추출완료'
        WHEN s.status = 'completed' THEN 'STT완료'
        WHEN s.status = 'processing' THEN '처리중'
        ELSE '미처리'
    END as 전체_처리상태,
    
    -- 처리 진행률 (0-100)
    CASE 
        WHEN r.id IS NOT NULL AND r.status = 'success' THEN 100
        WHEN e.id IS NOT NULL THEN 66
        WHEN s.status = 'completed' THEN 33
        ELSE 0
    END as 처리_진행률,
    
    -- 시간 정보
    s.created_at as stt_처리일시,
    e.created_at as erp_추출일시,
    r.registered_at as erp_등록일시,
    
    -- 최종 업데이트 시간
    GREATEST(
        s.updated_at, 
        COALESCE(e.created_at, '1970-01-01'::timestamp),
        COALESCE(r.registered_at, '1970-01-01'::timestamp)
    ) as 최종_업데이트

FROM stt_sessions s
LEFT JOIN erp_extractions e ON s.id = e.session_id
LEFT JOIN erp_register_logs r ON e.id = r.extraction_id AND r.status = 'success'
ORDER BY s.created_at DESC;

-- 디렉토리별 처리 현황 요약 뷰 (날짜별 폴더 지원)
CREATE OR REPLACE VIEW directory_processing_summary AS
SELECT 
    CASE 
        WHEN s.file_name LIKE 'src_record/%/%' 
        THEN SPLIT_PART(s.file_name, '/', 2)  -- src_record/2025-07-16/파일명 -> 2025-07-16
        WHEN s.file_name LIKE '%/%' 
        THEN SPLIT_PART(s.file_name, '/', 1)  -- 기타 폴더/파일명 -> 폴더
        ELSE '루트'
    END as 디렉토리,
    COUNT(*) as 총_파일수,
    COUNT(CASE WHEN s.status = 'completed' THEN 1 END) as stt_완료수,
    COUNT(e.id) as erp_추출수,
    COUNT(CASE WHEN r.status = 'success' THEN 1 END) as erp_등록수,
    ROUND(
        COUNT(CASE WHEN r.status = 'success' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 1
    ) as 완료율,
    MIN(s.created_at) as 최초_처리일시,
    MAX(s.created_at) as 최근_처리일시
FROM stt_sessions s
LEFT JOIN erp_extractions e ON s.id = e.session_id
LEFT JOIN erp_register_logs r ON e.id = r.extraction_id
GROUP BY 디렉토리
ORDER BY 디렉토리 DESC;
"""