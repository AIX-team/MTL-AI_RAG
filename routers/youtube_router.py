from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Response
from typing import List
from models.youtube_schemas import ContentRequest, YouTubeResponse
from services.youtube_service import YouTubeService
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json

router = APIRouter()
youtube_service = YouTubeService()
youtube_repository = YouTubeRepository()

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    content: str
    metadata: dict

@router.post("/contentanalysis", 
            response_model=YouTubeResponse,
            summary="콘텐츠 분석",
            description="YouTube 영상, 네이버 블로그, 티스토리 등의 URL을 받아 내용을 분석하고 요약합니다.")
async def process_content(request: ContentRequest):
    try:
        print("\n" + "="*50)
        print("[디버그] 요청 데이터:")
        print(f"요청 본문: {request.dict()}")
        print("="*50 + "\n")

        print("[1단계] 요청 데이터 수신")
        if not isinstance(request.urls, list):
            print("❌ URLs가 리스트 형식이 아님")
            raise ValueError("URLs must be a list")
        print(f"입력 URL 목록: {json.dumps(request.urls, indent=2, ensure_ascii=False)}")
        print("="*50 + "\n")
        
        # URL 유효성 검사
        if not request.urls:
            print("❌ URL 리스트가 비어있음")
            raise ValueError("URL 리스트가 비어있습니다")
            
        print("[2단계] URL 유효성 검사")
        print("2.1 URL 형식 검증")
        for url in request.urls:
            if not isinstance(url, str):
                print(f"❌ 잘못된 URL 타입: {type(url)}")
                raise ValueError(f"URL must be string, not {type(url)}")
            if not url.startswith(('http://', 'https://')):
                print(f"❌ 잘못된 URL 형식: {url}")
                raise ValueError(f"유효하지 않은 URL 형식입니다: {url}")
            print(f"✓ 유효한 URL 확인: {url}")
        print("="*50 + "\n")
                
        print("[3단계] 콘텐츠 처리 시작")
        try:
            result = await youtube_service.process_urls(request.urls)
            print(f"처리된 콘텐츠 정보:")
            print(f"- 요약 개수: {len(result.get('summary', {}))}")
            print(f"- 콘텐츠 정보 개수: {len(result.get('content_infos', []))}")
            print(f"- 장소 정보 개수: {len(result.get('place_details', []))}")
            print(f"- 처리 시간: {result.get('processing_time_seconds', 0.0):.2f}초")
        except Exception as process_error:
            print(f"❌ 콘텐츠 처리 중 오류 발생:")
            print(f"오류 타입: {type(process_error).__name__}")
            print(f"오류 메시지: {str(process_error)}")
            raise
        print("="*50 + "\n")
        
        print("[4단계] 응답 데이터 검증")
        print("4.1 결과 데이터 타입 검증")
        if not isinstance(result, dict):
            print(f"❌ 잘못된 결과 타입: {type(result)}")
            raise ValueError("결과가 딕셔너리 형식이 아닙니다")
            
        print("4.2 필수 필드 존재 여부 확인")
        required_fields = ["summary", "content_infos", "processing_time_seconds", "place_details"]
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            print(f"❌ 누락된 필드: {missing_fields}")
            raise ValueError(f"결과에 필수 필드가 누락되었습니다: {missing_fields}")
        print("✓ 모든 필수 필드 확인 완료")
        print("="*50 + "\n")
            
        print("[5단계] 최종 응답 생성")
        print("5.1 기본값 처리 및 응답 데이터 구성")
        try:
            response_data = {
                "summary": result.get("summary", {}),
                "content_infos": result.get("content_infos", []),
                "processing_time_seconds": result.get("processing_time_seconds", 0.0),
                "place_details": result.get("place_details", [])
            }
            
            print("5.2 YouTubeResponse 객체 생성 및 검증")
            youtube_response = YouTubeResponse(**response_data)
            print("✓ 응답 객체 생성 완료")
            print("응답 데이터 미리보기:")
            print(json.dumps({
                "summary_count": len(response_data["summary"]),
                "content_infos_count": len(response_data["content_infos"]),
                "place_details_count": len(response_data["place_details"])
            }, indent=2))
            print("="*50 + "\n")
            
            return youtube_response
            
        except Exception as response_error:
            print(f"❌ 응답 생성 중 오류:")
            print(f"오류 타입: {type(response_error).__name__}")
            print(f"오류 메시지: {str(response_error)}")
            raise
        
    except ValueError as ve:
        print("\n[오류 발생] 유효성 검사 실패")
        error_detail = {
            "error": "유효성 검사 오류",
            "message": str(ve),
            "type": "ValidationError",
            "request_data": request.dict() if hasattr(request, 'dict') else str(request)
        }
        print(f"오류 상세: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_detail
        )
    except Exception as e:
        print("\n[오류 발생] 내부 처리 실패")
        error_detail = {
            "error": "내부 서버 오류",
            "message": str(e),
            "type": type(e).__name__,
            "stack_trace": str(e.__traceback__),
            "request_data": request.dict() if hasattr(request, 'dict') else str(request)
        }
        print(f"오류 상세: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail
        )

@router.post("/vectorsearch", response_model=List[SearchResponse])
async def search_content(request: SearchRequest):
    """벡터 DB에서 콘텐츠 검색"""
    try:
        results = await youtube_service.search_content(request.query)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        )