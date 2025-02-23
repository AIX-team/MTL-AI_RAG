from typing import List, Union
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
                places_dict[place.id] = place_dict  # ID로 매핑

            prompt = info2guide_repository.create_travel_prompt(list(places_dict.values()), plan_type, days)
            response = await info2guide_repository.get_gpt_response(prompt)
            
            if not response or 'days' not in response:
                print(f"No valid response for {plan_type} plan")
                return self._create_default_plan(places, days, plan_type)
            
            daily_plans = []
            total_places = len(places)
            places_per_day = total_places // days
            if places_per_day == 0:
                places_per_day = 1
            
            remaining_places = list(places)
            
            # GPT가 빈 응답을 주면 기본 일정 생성
            if not response['days']:
                return self._create_default_plan(places, days, plan_type)
            
            for day_num in range(1, days + 1):
                try:
                    day_data = next((d for d in response['days'] if d['day_number'] == day_num), None)
                    if not day_data:
                        # 해당 일자의 계획이 없으면 남은 장소들로 생성
                        places_for_day = remaining_places[:places_per_day]
                        remaining_places = remaining_places[places_per_day:]
                        
                        daily_plans.append(DayPlan(
                            day_number=day_num,
                            places=[self._create_place_detail(p) for p in places_for_day]
                        ))
                        continue
                    
                    places_list = []
                    for place in day_data.get('places', []):
                        # ID로 원본 데이터 찾기
                        place_id = place.get('id')
                        if place_id in places_dict:
                            original_place = places_dict[place_id]
                            if original_place:
                                places_list.append(self._create_place_detail(original_place))
                                if original_place in remaining_places:
                                    remaining_places.remove(original_place)
                    
                    if places_list:
                        daily_plans.append(DayPlan(
                            day_number=day_num,
                            places=places_list
                        ))
                        print(f"Added day {day_num} with {len(places_list)} places")
                
                except Exception as e:
                    print(f"Error processing day {day_num}: {e}")
                    continue
            
            # 남은 장소들 처리
            if remaining_places:
                for day_plan in daily_plans:
                    if len(remaining_places) == 0:
                        break
                    if len(day_plan.places) < places_per_day:
                        day_plan.places.append(self._create_place_detail(remaining_places.pop(0)))
            
            return [TravelPlan(
                plan_type=plan_type,
                daily_plans=daily_plans
            )]
        except Exception as e:
            print(f"Error generating travel plans: {e}")
            return self._create_default_plan(places, days, plan_type)

    def _create_default_plan(self, places: List[PlaceInfo], days: int, plan_type: str) -> List[TravelPlan]:
        """GPT가 실패하거나 빈 응답을 줄 경우 기본 일정 생성"""
        total_places = len(places)
        places_per_day = total_places // days
        if places_per_day == 0:
            places_per_day = 1
        
        daily_plans = []
        remaining_places = list(places)
        
        for day_num in range(1, days + 1):
            places_for_day = remaining_places[:places_per_day]
            remaining_places = remaining_places[places_per_day:]
            
            if places_for_day:  # 해당 날짜에 배정할 장소가 있는 경우만 일정 추가
                daily_plans.append(DayPlan(
                    day_number=day_num,
                    places=[self._create_place_detail(p) for p in places_for_day]
                ))
        
        return [TravelPlan(
            plan_type=plan_type,
            daily_plans=daily_plans
        )]

    def _create_place_detail(self, place: Union[PlaceInfo, dict]) -> PlaceDetail:
        """PlaceInfo 또는 place_dict에서 PlaceDetail 생성"""
        if isinstance(place, dict):
            return PlaceDetail(
                id=place['id'],
                name=place['title'],
                address=place['address'],
                official_description=place['description'],
                reviewer_description="기본 일정으로 추가된 장소",
                place_type=place['type'],
                rating=self._parse_rating(str(place['rating'])),
                image_url=place['image'],
                business_hours=place['open_hours'] or '영업시간 정보 없음',
                website='',
                latitude=str(place['latitude']),
                longitude=str(place['longitude'])
            )
        else:
            return PlaceDetail(
                id=place.id,
                name=place.title,
                address=place.address,
                official_description=place.description,
                reviewer_description="기본 일정으로 추가된 장소",
                place_type=place.type,
                rating=self._parse_rating(str(place.rating)),
                image_url=place.image,
                business_hours=place.open_hours if hasattr(place, 'open_hours') else '영업시간 정보 없음',
                website='',
                latitude=str(place.latitude),
                longitude=str(place.longitude)
            )

    def _parse_rating(self, rating_str: str) -> float:
        try:
            if rating_str in ['N/A', '', None]:
                return 0.0
            return float(rating_str)
        except (ValueError, TypeError):
            return 0.0
