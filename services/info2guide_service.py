from typing import List
import openai
from models.info2guide_model import PlaceInfo, PlaceDetail, DayPlan, TravelPlan
from repository import info2guide_repository
import os

class TravelPlannerService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def generate_travel_plans(self, places: List[PlaceInfo], days: int) -> List[TravelPlan]:
        plan_types = ['busy', 'normal', 'relaxed']
        plans = []
        for plan_type in plan_types:
            try:
                plan = await self._create_plan(places, days, plan_type)
                plans.append(plan)
                print(f"Generated {plan_type} plan with {len(plan.daily_plans)} days")
            except Exception as e:
                print(f"Error generating {plan_type} plan: {e}")
                plans.append(TravelPlan(plan_type=plan_type, daily_plans=[]))
        return plans
    
    async def _create_plan(self, places: List[PlaceInfo], days: int, plan_type: str) -> TravelPlan:
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
        
        daily_plans = []
        for day_data in response['days']:
            try:
                if day_data['day_number'] > days:
                    continue
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
                if places_list:
                    daily_plans.append(DayPlan(
                        day_number=day_data['day_number'],
                        places=places_list
                    ))
                    print(f"Added day {day_data['day_number']} with {len(places_list)} places")
            except Exception as e:
                print(f"Error processing day {day_data.get('day_number', '?')}: {e}")
        
        return TravelPlan(
            plan_type=plan_type,
            daily_plans=daily_plans[:days]
        )
    
    def _parse_rating(self, rating_str: str) -> float:
        try:
            if rating_str in ['N/A', '', None]:
                return 0.0
            return float(rating_str)
        except (ValueError, TypeError):
            return 0.0
