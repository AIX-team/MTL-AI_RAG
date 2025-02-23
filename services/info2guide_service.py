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
                'rating': float(place.rating if place.rating else 0)
            } for place in places]

            plan = await self._create_plan(places_dict, days, plan_type)
            print(f"Generated {plan_type} plan with {len(plan.daily_plans)} days")
            
            # 각 장소를 PlaceDetail 객체로 변환
            daily_plans = []
            for day in plan.daily_plans:
                places_list = []
                for place in day.places:
                    place_detail = PlaceDetail(
                        id=place.id,
                        title=place.title,
                        address=place.address,
                        description=place.description,
                        intro=place.intro,
                        type=place.type,
                        rating=Decimal(str(place.rating if place.rating else 0)),
                        image=place.image,
                        open_hours=place.open_hours or '',
                        phone=place.phone,
                        latitude=float(place.latitude),
                        longitude=float(place.longitude)
                    )
                    places_list.append(place_detail)
                
                day_plan = DayPlan(
                    day_number=day.day_number,
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
    
    async def _create_plan(self, places: List[dict], days: int, plan_type: str) -> TravelPlan:
        prompt = info2guide_repository.create_travel_prompt(places, plan_type, days)
        response = await info2guide_repository.get_gpt_response(prompt)
        if not response or 'days' not in response:
            print(f"No valid response for {plan_type} plan")
            return TravelPlan(plan_type=plan_type, daily_plans=[])
        
        daily_plans = []
        for day_data in response['days']:
            try:
                if day_data['day_number'] > days:
                    continue
                
                places_list = []
                for place_data in day_data.get('places', []):
                    try:
                        # 원본 places 리스트에서 일치하는 장소 찾기
                        original_place = next(
                            (p for p in places if p['id'] == place_data.get('id')),
                            None
                        )
                        
                        if original_place:
                            place_detail = PlaceDetail(
                                id=original_place['id'],
                                title=original_place['title'],
                                address=original_place['address'],
                                description=original_place['description'],
                                intro=original_place['intro'],
                                type=original_place['type'],
                                rating=Decimal(str(original_place['rating'])),
                                image=original_place['image'],
                                open_hours=original_place['open_hours'] or '',
                                phone=original_place['phone'],
                                latitude=float(original_place['latitude']),
                                longitude=float(original_place['longitude'])
                            )
                            places_list.append(place_detail)
                    except Exception as e:
                        print(f"Error creating PlaceDetail: {e}")
                        continue

                if places_list:  # 최소한 하나의 장소가 있는 경우에만 일정 추가
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
