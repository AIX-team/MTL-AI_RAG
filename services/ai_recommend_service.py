from repository.ai_recommend_repository import AIRecommendRepository
from models.ai_recommend_model import AIRecommendRequest, AIRecommendResponse
import logging
import traceback

class AIRecommendService:
    def __init__(self):
        self.repository = AIRecommendRepository()
        self.logger = logging.getLogger(__name__)

    async def recommend_places(self, request: AIRecommendRequest) -> AIRecommendResponse:
        """
        여행 일정에 맞는 최적의 장소 조합을 추천합니다.
        
        Args:
            request (AIRecommendRequest): 여행 정보 ID, 여행 일수, 장소 목록을 포함한 요청 객체
            
        Returns:
            AIRecommendResponse: 추천된 장소 목록과 처리 결과를 포함한 응답 객체
        """
        try:
            self.logger.info("=== AI 추천 서비스 시작 ===")
            self.logger.info(f"요청 정보: travelInfoId={request.travelInfoId}, travelDays={request.travelDays}")
            
            # 1. 입력값 검증
            if not request.places:
                self.logger.error("장소 목록이 비어있습니다.")
                return AIRecommendResponse(
                    success="error",
                    message="장소 목록이 비어있어 추천을 할 수 없습니다.",
                    content=[]
                )
            
            if request.travelDays <= 0:
                self.logger.error(f"잘못된 여행 일수: {request.travelDays}")
                return AIRecommendResponse(
                    success="error",
                    message="여행 일수는 1일 이상이어야 합니다.",
                    content=[]
                )
            
            self.logger.info(f"여행 일수: {request.travelDays}")
            self.logger.info(f"장소 수: {len(request.places)}")
            
            # 2. 리포지토리에 추천 요청
            self.logger.info("AI 추천 리포지토리 호출 시작")
            response = await self.repository.recommend_places(request)
            self.logger.info(f"AI 추천 리포지토리 응답: success={response.success}, message={response.message}")
            
            # 3. 결과 검증 및 반환
            if response.success != "success":
                self.logger.warning(f"추천 실패: {response.message}")
                return response
            
            if not response.content:
                self.logger.warning("추천된 장소가 없습니다.")
                return AIRecommendResponse(
                    success="error",
                    message="추천 장소를 생성하지 못했습니다.",
                    content=[]
                )
            
            self.logger.info(f"추천된 장소 수: {len(response.content)}")
            self.logger.info("=== AI 추천 서비스 완료 ===")
            
            return response
            
        except Exception as e:
            self.logger.error("=== AI 추천 서비스 에러 ===")
            self.logger.error(f"에러 타입: {type(e).__name__}")
            self.logger.error(f"에러 메시지: {str(e)}")
            self.logger.error("상세 스택 트레이스:")
            self.logger.error(traceback.format_exc())
            
            return AIRecommendResponse(
                success="error",
                message=f"AI 추천 서비스 처리 중 오류 발생: {str(e)}",
                content=[]
            ) 