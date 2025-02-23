from typing import List
import openai
from models.info2guide_model import PlaceInfo, PlaceDetail, DayPlan, TravelPlan
from repository import info2guide_repository
import os
from decimal import Decimal

class TravelPlannerService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def generate_travel_plans(self, places: List[PlaceInfo], days: int, travelTaste: str) -> TravelPlan:
        # Map the travelTaste to a plan type
        type_mapping = {
            '빼곡한 일정 선호': 'busy',
            '적당한 일정 선호': 'normal',
            '널널한 일정 선호': 'relaxed'
        }
        plan_type = type_mapping.get(travelTaste)
        if not plan_type:
            plan_type = 'normal'
        try:
            plan = await self._create_plan(places, days, plan_type)
            print(f"Generated {plan_type} plan with {len(plan.daily_plans)} days")
            
            # 각 장소를 PlaceDetail 객체로 변환
            daily_plans = []
            for day in plan.daily_plans:
                places_list = []
                for place in day.places:
                    place_detail = PlaceDetail(
                        id=place.get('id', ''),
                        title=place.get('title', ''),
                        address=place.get('address', ''),
                        description=place.get('description', ''),
                        intro=place.get('intro', ''),
                        type=place.get('type', ''),
                        rating=Decimal(str(place.get('rating', 0))),
                        image=place.get('image', ''),
                        open_hours=place.get('open_hours', ''),
                        phone=place.get('phone', ''),
                        latitude=float(place.get('latitude', 0)),
                        longitude=float(place.get('longitude', 0))
                    )
                    places_list.append(place_detail)
                
                day_plan = DayPlan(
                    day_number=day['day_number'],
                    places=places_list
                )
                daily_plans.append(day_plan)
            
            return TravelPlan(
                plan_type=plan_type,
                daily_plans=daily_plans
            )
        except Exception as e:
            print(f"Error generating {plan_type} plan: {e}")
            return TravelPlan(plan_type=plan_type, daily_plans=[])
    
    async def _create_plan(self, places: List[PlaceInfo], days: int, plan_type: str) -> TravelPlan:
        # 원본 PlaceInfo 객체들을 딕셔너리 리스트로 변환
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
            'open_hours': place.open_hours,
            'phone': place.phone,
            'rating': place.rating
        } for place in places]
        
        prompt = info2guide_repository.create_travel_prompt(places_dict, plan_type, days)
        response = await info2guide_repository.get_gpt_response(prompt)
        if not response or 'days' not in response:
            print(f"No valid response for {plan_type} plan")
            return TravelPlan(plan_type=plan_type, daily_plans=[])
        
        # 각 스타일에 따른 필수 장소 수 설정 (busy: 4, normal: 3, relaxed: 2)
        required_places = {'busy': 4, 'normal': 3, 'relaxed': 2}[plan_type.lower()]
        
        daily_plans = []
        for day_data in response['days']:
            try:
                if day_data['day_number'] > days:
                    continue
                # GPT 응답에서 추출한 장소 리스트
                places_list = []
                for place in day_data.get('places', []):
                    try:
                        place_detail = PlaceDetail(
                            id=place.get('id', ''),
                            title=place.get('title', '알 수 없는 장소'),
                            address=place.get('address', '주소 정보 없음'),
                            description=place.get('description', '설명 없음'),
                            intro=place.get('intro', '리뷰 없음'),
                            type=place.get('type', '기타'),
                            rating=Decimal(str(place.get('rating', '0'))),
                            image=place.get('image', ''),
                            open_hours=place.get('open_hours', '영업시간 정보 없음'),
                            phone=place.get('phone', ''),
                            latitude=float(place.get('latitude', 0)),
                            longitude=float(place.get('longitude', 0))
                        )
                        places_list.append(place_detail)
                    except Exception as e:
                        print(f"Error creating PlaceDetail: {e}")
                        continue

                day_plan = DayPlan(
                    day_number=day_data['day_number'],
                    places=places_list
                )
                daily_plans.append(day_plan)
            except Exception as e:
                print(f"Error processing day {day_data.get('day_number')}: {e}")
                continue

        return TravelPlan(
            plan_type=plan_type,
            daily_plans=daily_plans
        )
    
    def _parse_rating(self, rating_str: str) -> float:
        try:
            if rating_str in ['N/A', '', None]:
                return 0.0
            return float(rating_str)
        except (ValueError, TypeError):
            return 0.0
