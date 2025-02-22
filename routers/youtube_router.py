from fastapi import APIRouter, HTTPException, status
from typing import List
from models.youtube_schemas import ContentRequest, YouTubeResponse
from services.youtube_service import YouTubeService
from pydantic import BaseModel
from repository.youtube_repository import YouTubeRepository  # repository로 수정
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
        # URL 검증
        if not request.urls:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL 목록이 비어있습니다."
            )

        urls = [str(url) for url in request.urls]
        
        # URL 개수 검증
        if not (1 <= len(urls) <= 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL의 개수는 최소 1개에서 최대 5개여야 합니다."
            )
        
        try:
            # 콘텐츠 처리 및 분석
            result = youtube_service.process_urls(urls)
            
            # 결과 검증
            if not result:
                raise ValueError("처리된 데이터가 없습니다")
                
            if not isinstance(result.get("summary", {}), dict):
                raise ValueError("최종 요약이 딕셔너리 형식이 아닙니다.")
            
            # 장소 정보 필터링 및 저장
            filtered_places = youtube_repository.save_place_details(result.get("place_details", []))
            
            # 필터링된 장소가 있는지 확인
            if not filtered_places:
                raise ValueError("유효한 일본 장소 데이터가 없습니다. 모든 장소가 필터링 조건을 만족하지 않습니다.")
            
            # 응답 구조화 (필터링된 장소 정보 사용)
            response = YouTubeResponse(
                summary=result["summary"],
                content_infos=result.get("content_infos", []),
                processing_time_seconds=result.get("processing_time_seconds", 0.0),
                place_details=filtered_places  # 필터링된 장소 목록만 사용
            )
            
            return response
            
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve)
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"처리 중 오류가 발생했습니다: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"예기치 않은 오류가 발생했습니다: {str(e)}"
        )

@router.post("/vectorsearch", response_model=List[SearchResponse])
async def search_content(request: SearchRequest):
    """벡터 DB에서 콘텐츠 검색"""
    try:
        results = youtube_service.search_content(request.query)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        ) 
    

