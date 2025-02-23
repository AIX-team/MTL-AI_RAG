from typing import List
import openai
from models.info2guide_model import PlaceInfo, PlaceDetail, DayPlan, TravelPlan
from repository import info2guide_repository
import os

class TravelPlannerService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def generate_travel_plans(self, places: List[PlaceInfo], days: int, plan_type: str) -> List[TravelPlan]:
        try:
            # 한글 plan_type을 영문으로 매핑
            plan_type_mapping = {
                '빼곡한 일정 선호': 'busy',
                '적당한 일정 선호': 'normal',
                '널널한 일정 선호': 'relaxed'
            }
            
            if plan_type in plan_type_mapping:
                plan_type = plan_type_mapping[plan_type]
            
            # places를 딕셔너리로 변환하고 ID 매핑 정보 저장
            places_dict = {}
            for place in places:
                place_dict = {
                    'id': place.id,  # 원본 ID 보존
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
                }
                places_dict[place.title] = place_dict  # 장소 이름으로 매핑

            prompt = info2guide_repository.create_travel_prompt(list(places_dict.values()), plan_type, days)
            response = await info2guide_repository.get_gpt_response(prompt)
            
            if not response or 'days' not in response:
                print(f"No valid response for {plan_type} plan")
                return [TravelPlan(plan_type=plan_type, daily_plans=[])]
            
            daily_plans = []
            for day_data in response['days']:
                try:
                    if day_data['day_number'] > days:
                        continue
                    
                    # GPT 응답에서 추출한 장소 리스트
                    places_list = []
                    for place in day_data.get('places', []):
                        # 장소 이름으로 원본 데이터 찾기
                        original_place = places_dict.get(place.get('name'))
                        if original_place:
                            places_list.append(
                                PlaceDetail(
                                    id=original_place['id'],  # 원본 ID 사용
                                    name=original_place['title'],
                                    address=original_place['address'],
                                    official_description=original_place['description'],
                                    reviewer_description=place.get('reviewer_description', '리뷰 없음'),
                                    place_type=original_place['type'],
                                    rating=self._parse_rating(str(original_place['rating'])),
                                    image_url=original_place['image'],
                                    business_hours=original_place['open_hours'] or '영업시간 정보 없음',
                                    website='',
                                    latitude=str(original_place['latitude']),
                                    longitude=str(original_place['longitude'])
                                )
                            )
                    
                    if places_list:  # 빈 날짜는 건너뛰기
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
