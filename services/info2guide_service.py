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
            for day_data in plan.daily_plans:
                places_list = []
                for place_data in day_data.places:
                    try:
                        # 원본 places 리스트에서 일치하는 장소 찾기
                        original_place = next(
                            (p for p in places if p.id == place_data.id),
                            None
                        )
                        
                        # 원본 장소를 찾은 경우 해당 데이터 사용, 아니면 GPT 응답 데이터 사용
                        if original_place:
                            place_detail = PlaceDetail(
                                id=original_place.id,
                                title=original_place.title,
                                address=original_place.address,
                                description=original_place.description,
                                intro=original_place.intro,
                                type=original_place.type,
                                rating=Decimal(str(original_place.rating if original_place.rating else 0)),
                                image=original_place.image,
                                open_hours=original_place.open_hours or '',
                                phone=original_place.phone,
                                latitude=float(original_place.latitude),
                                longitude=float(original_place.longitude)
                            )
                        else:
                            place_detail = PlaceDetail(
                                id=place_data.id,
                                title=place_data.title,
                                address=place_data.address,
                                description=place_data.description,
                                intro=place_data.intro,
                                type=place_data.type,
                                rating=Decimal(str(place_data.rating if place_data.rating else 0)),
                                image=place_data.image,
                                open_hours=place_data.open_hours or '',
                                phone=place_data.phone,
                                latitude=float(place_data.latitude),
                                longitude=float(place_data.longitude)
                            )
                        places_list.append(place_detail)
                    except Exception as e:
                        print(f"Error creating PlaceDetail: {e}")
                        continue
                
                if places_list:  # 최소한 하나의 장소가 있는 경우에만 일정 추가
                    day_plan = DayPlan(
                        day_number=day_data.day_number,
                        places=places_list
                    )
                    daily_plans.append(day_plan)
            
            # 일정이 비어있는 경우 기본 일정 생성
            if not daily_plans:
                print("Creating default plan from original places")
                places_per_day = min(len(places), {'busy': 4, 'normal': 3, 'relaxed': 2}[plan_type.lower()])
                
                for day in range(1, days + 1):
                    start_idx = (day - 1) * places_per_day
                    end_idx = start_idx + places_per_day
                    day_places = places[start_idx:end_idx]
                    
                    if not day_places:
                        break
                        
                    places_list = []
                    for place in day_places:
                        try:
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
                        except Exception as e:
                            print(f"Error creating default PlaceDetail: {e}")
                            continue
                    
                    if places_list:
                        day_plan = DayPlan(
                            day_number=day,
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
                        
                        # 원본 장소를 찾지 못한 경우 GPT 응답의 데이터 사용
                        if not original_place:
                            original_place = place_data
                        
                        # 필수 필드가 없는 경우 기본값 설정
                        place_detail = PlaceDetail(
                            id=original_place.get('id', ''),
                            title=original_place.get('title', original_place.get('name', '알 수 없는 장소')),
                            address=original_place.get('address', '주소 정보 없음'),
                            description=original_place.get('description', original_place.get('official_description', '설명 없음')),
                            intro=original_place.get('intro', original_place.get('reviewer_description', '리뷰 없음')),
                            type=original_place.get('type', original_place.get('place_type', '기타')),
                            rating=Decimal(str(original_place.get('rating', 0))),
                            image=original_place.get('image', original_place.get('image_url', '')),
                            open_hours=original_place.get('open_hours', original_place.get('business_hours', '영업시간 정보 없음')),
                            phone=original_place.get('phone', ''),
                            latitude=float(original_place.get('latitude', 0)),
                            longitude=float(original_place.get('longitude', 0))
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

        # 일정이 비어있는 경우 기본 일정 생성
        if not daily_plans:
            print("Creating default plan from original places")
            places_per_day = min(len(places), {'busy': 4, 'normal': 3, 'relaxed': 2}[plan_type.lower()])
            
            for day in range(1, days + 1):
                start_idx = (day - 1) * places_per_day
                end_idx = start_idx + places_per_day
                day_places = places[start_idx:end_idx]
                
                if not day_places:
                    break
                    
                places_list = []
                for place in day_places:
                    try:
                        place_detail = PlaceDetail(
                            id=place['id'],
                            title=place['title'],
                            address=place['address'],
                            description=place['description'],
                            intro=place['intro'],
                            type=place['type'],
                            rating=Decimal(str(place['rating'])),
                            image=place['image'],
                            open_hours=place['open_hours'] or '',
                            phone=place['phone'],
                            latitude=float(place['latitude']),
                            longitude=float(place['longitude'])
                        )
                        places_list.append(place_detail)
                    except Exception as e:
                        print(f"Error creating default PlaceDetail: {e}")
                        continue
                
                if places_list:
                    day_plan = DayPlan(
                        day_number=day,
                        places=places_list
                    )
                    daily_plans.append(day_plan)

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
