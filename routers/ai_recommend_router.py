from fastapi import APIRouter, HTTPException, Request
import logging
from models.ai_recommend_model import AIRecommendRequest, AIRecommendResponse
from services.ai_recommend_service import AIRecommendService
import json

router = APIRouter(
    prefix="/ai/recommend",
    tags=["AI Recommendations"]
)

ai_recommend_service = AIRecommendService()
logger = logging.getLogger(__name__)

@router.post("/places", response_model=AIRecommendResponse)
async def recommend_places(request: AIRecommendRequest):
    """
    여행 일정에 맞는 최적의 장소 조합을 추천합니다.
    """
    try:
        # 요청 데이터 로깅
        logger.info("=== AI 추천 요청 시작 ===")
        logger.info(f"Travel Info ID: {request.travelInfoId}")
        logger.info(f"Travel Days: {request.travelDays}")
        logger.info(f"Places Count: {len(request.places)}")
        
        # 장소 데이터 상세 로깅
        for i, place in enumerate(request.places):
            logger.info(f"Place {i+1}:")
            logger.info(f"  ID: {place.placeId}")
            logger.info(f"  Type: {place.placeType}")
            logger.info(f"  Name: {place.placeName}")
            logger.info(f"  Coordinates: ({place.latitude}, {place.longitude})")

        # 서비스 호출
        response = await ai_recommend_service.recommend_places(request)
        
        # 응답 데이터 로깅
        logger.info("=== AI 추천 응답 ===")
        logger.info(f"Success: {response.success}")
        logger.info(f"Message: {response.message}")
        logger.info(f"Recommended Places Count: {len(response.content)}")
        
        if response.success == "error":
            logger.error(f"Service returned error: {response.message}")
            raise HTTPException(status_code=400, detail=response.message)
            
        return response
        
    except Exception as e:
        logger.error("=== AI 추천 에러 발생 ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.exception("상세 에러 스택:")
        raise HTTPException(status_code=500, detail=str(e)) 