from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import List
from models.youtube_schemas import ContentRequest, YouTubeResponse
from services.youtube_service import YouTubeService
from pydantic import BaseModel

router = APIRouter()
youtube_service = YouTubeService()

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
        print(f"Received request: {request}")
        
        # URL 유효성 검사
        for url in request.urls:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"유효하지 않은 URL 형식입니다: {url}")
                
        # process_urls 메서드 호출
        result = await youtube_service.process_urls(request.urls)
        
        # 응답 형식 검증
        if not isinstance(result, dict):
            raise ValueError("결과가 딕셔너리 형식이 아닙니다")
            
        # 필수 필드 확인
        required_fields = ["summary", "content_infos", "processing_time_seconds", "place_details"]
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(f"결과에 필수 필드가 누락되었습니다: {missing_fields}")
            
        # None 값 처리
        response_data = {
            "summary": result.get("summary", {}),
            "content_infos": result.get("content_infos", []),
            "processing_time_seconds": result.get("processing_time_seconds", 0.0),
            "place_details": result.get("place_details", [])
        }
        
        # 응답 생성
        response = YouTubeResponse(**response_data)
        
        print(f"Sending response: {response}")
        return response
        
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        print(f"Processing error: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "내부 서버 오류",
                "message": str(e),
                "type": str(type(e))
            }
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