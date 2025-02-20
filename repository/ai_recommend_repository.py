from typing import List
import os
from openai import AsyncOpenAI
from models.ai_recommend_model import PlaceBase, AIRecommendRequest, AIRecommendResponse
import logging
import json
import traceback

class AIRecommendRepository:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.logger = logging.getLogger(__name__)
        self.place_categories = {
            "landmark": ["tourist_attraction", "museum", "art_gallery", "park", "landmark"],
            "restaurant": ["restaurant", "cafe", "bakery", "bar", "food"],
            "shopping": ["shopping_mall", "store", "market"],
            "entertainment": ["amusement_park", "zoo", "aquarium", "theater"]
        }

    async def recommend_places(self, request: AIRecommendRequest) -> AIRecommendResponse:
        try:
            self.logger.info("=== AI 추천 리포지토리 시작 ===")
            
            # 1. 장소 정보를 문자열로 변환
            places_info = self._format_places_for_prompt(request.places)
            self.logger.info("장소 정보 포맷팅 완료")
            
            # 2. OpenAI에 보낼 프롬프트 생성
            prompt = self._create_recommendation_prompt(
                places_info,
                request.travelDays
            )
            self.logger.info("프롬프트 생성 완료")
            
            # 3. OpenAI API 호출
            self.logger.info("OpenAI API 호출 시작")
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """당신은 여행 계획 전문가입니다. 
                    당신은 일본 여행 전문가입니다. 사용자가 아래와 같이 일본 내의 여행지를 입력하면, 해당 목록과 연관되어 여행 경험을 더욱 풍부하게 만들어 줄 추천 장소들을 도출해 주세요. 추천 장소를 선정할 때는 아래 기준들을 반드시 고려합니다.

                    문화 및 역사적 중요성

                    입력된 여행지가 역사적, 문화적으로 의미 있는 지역이라면, 비슷한 역사적 배경이나 전통을 가진 장소를 추천합니다.
                    자연 경관 및 풍경

                    자연경관이나 풍경이 뛰어난 지역이 포함되어 있다면, 유사하게 아름다운 자연을 즐길 수 있는 지역이나 독특한 자연 명소가 있는 곳을 제안합니다.
                    미식 경험

                    지역 특유의 음식, 전통 요리, 또는 미식 체험이 중요한 요소라면, 미식 문화가 발달한 다른 지역을 추천합니다.
                    지리적 접근성 및 인근 관광지

                    입력된 장소들이 특정 지역에 몰려 있다면, 가까운 거리에서 이동 가능한 연계 여행지를 고려하여 추천합니다.
                    계절별 특색

                    특정 계절(예: 벚꽃, 단풍, 눈 축제 등)에 맞춘 여행지가 있다면, 계절의 매력을 함께 즐길 수 있는 다른 장소를 제안합니다.
                    독특한 현지 체험

                    지역 축제, 전통 공예, 혹은 특색 있는 체험 활동이 중요한 경우, 그러한 경험을 할 수 있는 장소를 함께 고려합니다.
                    관광객 평판 및 숨은 명소

                    잘 알려진 명소와 더불어, 관광객 평판이 우수하지만 상대적으로 덜 알려진 숨은 보석 같은 장소도 균형 있게 포함합니다. """},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens= 10000
            )
            self.logger.info("OpenAI API 호출 완료")
            
            # 4. AI 응답 파싱 및 장소 목록 생성
            self.logger.info("AI 응답 파싱 시작")
            recommended_places = self._parse_ai_response(
                response.choices[0].message.content,
                request.places
            )
            self.logger.info(f"파싱된 추천 장소 수: {len(recommended_places)}")
            
            # OpenAI 응답 로깅
            self.logger.info("=== OpenAI 응답 ===")
            self.logger.info(f"Raw response: {response.choices[0].message.content[:200]}...")
            
            # 파싱 결과 로깅
            self.logger.info("=== 파싱 결과 ===")
            self.logger.info(f"Parsed places: {len(recommended_places)}")
            
            return AIRecommendResponse(
                success="success",
                message="Successfully recommended places using AI",
                content=recommended_places
            )
        except Exception as e:
            self.logger.error("=== AI 추천 리포지토리 에러 ===")
            self.logger.error(f"에러 타입: {type(e).__name__}")
            self.logger.error(f"에러 메시지: {str(e)}")
            self.logger.error("상세 스택 트레이스:")
            self.logger.error(traceback.format_exc())
            
            return AIRecommendResponse(
                success="error",
                message=str(e),
                content=[]
            )

    def _format_places_for_prompt(self, places: List[PlaceBase]) -> str:
        try:
            formatted = "Available Places:\n\n"
            for i, place in enumerate(places, 1):
                formatted += f"{i}. {place.placeName}\n"
                formatted += f"   Type: {place.placeType}\n"
                formatted += f"   Address: {place.placeAddress or 'N/A'}\n"
                formatted += f"   Description: {place.placeDescription or 'N/A'}\n"
                formatted += f"   Location: ({place.latitude}, {place.longitude})\n\n"
            return formatted
        except Exception as e:
            self.logger.error(f"장소 정보 포맷팅 중 에러: {str(e)}")
            raise

    def _create_recommendation_prompt(self, places_info: str, travel_days: int) -> str:
        return f"""여행객을 위한 {travel_days}일 일정을 계획해주세요.
        
아래는 사용 가능한 장소 목록입니다:

{places_info}

다음 JSON 형식으로 응답해주세요:

{{
    "content": [
        {{
            "placeId": "장소ID",
            "placeType": "장소타입",
            "placeName": "장소이름",
            "placeAddress": "주소",
            "placeImage": "이미지URL",
            "placeDescription": "상세설명",
            "intro": "간단소개",
            "latitude": 위도,
            "longitude": 경도
        }},
        ...더 많은 장소들
    ]
}}

각 장소에 대해 상세한 설명과 방문 이유를 placeDescription에 포함해주세요."""

    def _parse_ai_response(self, ai_response: str, available_places: List[PlaceBase]) -> List[PlaceBase]:
        try:
            self.logger.info("AI 응답 파싱 시작")
            self.logger.info(f"AI 응답 내용: {ai_response[:200]}...")  # 처음 200자만 로깅
            
            # AI 응답을 JSON으로 파싱
            response_data = json.loads(ai_response)
            self.logger.info("JSON 파싱 완료")
            
            # 사용 가능한 장소들의 매핑 생성
            place_map = {place.placeName: place for place in available_places}
            self.logger.info(f"매핑된 장소 수: {len(place_map)}")
            
            recommended = []
            for place_data in response_data.get("content", []):
                place_name = place_data.get("placeName")
                if place_name in place_map:
                    recommended.append(place_map[place_name])
                else:
                    self.logger.warning(f"매칭되지 않은 장소 발견: {place_name}")
            
            self.logger.info(f"최종 추천 장소 수: {len(recommended)}")
            return recommended
        except Exception as e:
            self.logger.error(f"AI 응답 파싱 중 에러: {str(e)}")
            self.logger.error(f"원본 응답: {ai_response}")
            return [] 