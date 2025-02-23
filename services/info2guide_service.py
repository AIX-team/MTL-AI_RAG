from typing import List, Dict
import openai
from models.info2guide_model import PlaceInfo, PlaceDetail, DayPlan, TravelPlan
from repository import info2guide_repository
import os

class TravelPlannerService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def generate_travel_plans(self, places: List[PlaceInfo], days: int, plan_type: str) -> List[TravelPlan]:
        try:
            # places를 딕셔너리로 변환
            places_dict = [{
                'id': place.id,
                'title': place.title,
                'address': place.address,
                'description': place.description,
                'intro': place.intro,
                'type': place.type,
                'image': place.image,
                'latitude': place.latitude,
                'longitude': place.longitude,
                'open_hours': place.open_hours if hasattr(place, 'open_hours') else None,
                'phone': place.phone if hasattr(place, 'phone') else None,
                'rating': place.rating if hasattr(place, 'rating') else None
            } for place in places]

            prompt = info2guide_repository.create_travel_prompt(places_dict, plan_type, days)
            response = await info2guide_repository.get_gpt_response(prompt)
            
            if not response or 'days' not in response:
                print(f"No valid response for {plan_type} plan")
                return [TravelPlan(plan_type=plan_type, daily_plans=[])]
            
            # 각 스타일에 따른 필수 장소 수 설정
            required_places = {'busy': 4, 'normal': 3, 'relaxed': 2}[plan_type.lower()]
            
            daily_plans = []
            for day_data in response['days']:
                try:
                    if day_data['day_number'] > days:
                        continue
                    # GPT 응답에서 추출한 장소 리스트
                    places_list = [
                        PlaceDetail(
                            id=place.get('id', ''),
                            name=place.get('name', '알 수 없는 장소'),
                            address=place.get('address', '주소 정보 없음'),
                            official_description=place.get('official_description', '설명 없음'),
                            reviewer_description=place.get('reviewer_description', '리뷰 없음'),
                            place_type=place.get('place_type', '기타'),
                            rating=self._parse_rating(place.get('rating', '0')),
                            image_url=place.get('image_url', ''),
                            business_hours=place.get('business_hours', '영업시간 정보 없음'),
                            website=place.get('website', ''),
                            latitude=place.get('latitude', ''),
                            longitude=place.get('longitude', '')
                        )
                        for place in day_data.get('places', [])
                    ]
                    # 만약 해당 날에 필요한 장소 수가 부족하면 원본 장소 목록에서 보충
                    if len(places_list) < required_places:
                        additional_places = []
                        existing_ids = {p.id for p in places_list}
                        # 평점이 높은 순서대로 정렬
                        sorted_places = sorted(places, key=lambda p: p.rating, reverse=True)
                        for p in sorted_places:
                            if p.id not in existing_ids and (len(places_list) + len(additional_places)) < required_places:
                                additional_places.append(
                                    PlaceDetail(
                                        id=p.id,
                                        name=p.title,
                                        address=p.address,
                                        official_description=p.description,
                                        reviewer_description="추가된 장소",
                                        place_type=p.type,
                                        rating=p.rating,
                                        image_url=p.image,
                                        business_hours=p.open_hours,
                                        website="",
                                        latitude=str(p.latitude),
                                        longitude=str(p.longitude)
                                    )
                                )
                        places_list.extend(additional_places)
                    
                    daily_plans.append(DayPlan(
                        day_number=day_data['day_number'],
                        places=places_list
                    ))
                    print(f"Added day {day_data['day_number']} with {len(places_list)} places")
                except Exception as e:
                    print(f"Error processing day {day_data.get('day_number', '?')}: {e}")
            
            return [TravelPlan(
                plan_type=plan_type,
                daily_plans=daily_plans[:days]
            )]
        except Exception as e:
            print(f"Error generating travel plans: {e}")
            return [TravelPlan(plan_type=plan_type, daily_plans=[])]
    
    def _parse_rating(self, rating_str: str) -> float:
        try:
            if rating_str in ['N/A', '', None]:
                return 0.0
            return float(rating_str)
        except (ValueError, TypeError):
            return 0.0
