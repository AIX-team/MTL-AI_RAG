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
                try:
                    places_list = []
                    day_places = day_data.get('places', []) if isinstance(day_data, dict) else []
                    
                    for place_data in day_places:
                        try:
                            if not isinstance(place_data, dict):
                                print(f"Invalid place data format: {place_data}")
                                continue
                                
                            # 원본 places 리스트에서 일치하는 장소 찾기
                            place_id = place_data.get('id')
                            if not place_id:
                                print(f"Place data missing ID: {place_data}")
                                continue
                                
                            original_place = next(
                                (p for p in places if p.id == place_id),
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
                                # GPT 응답 데이터를 안전하게 처리
                                place_detail = PlaceDetail(
                                    id=place_data.get('id', ''),
                                    title=place_data.get('title', place_data.get('name', '알 수 없는 장소')),
                                    address=place_data.get('address', '주소 정보 없음'),
                                    description=place_data.get('description', place_data.get('official_description', '설명 없음')),
                                    intro=place_data.get('intro', place_data.get('reviewer_description', '리뷰 없음')),
                                    type=place_data.get('type', place_data.get('place_type', '기타')),
                                    rating=Decimal(str(place_data.get('rating', 0))),
                                    image=place_data.get('image', place_data.get('image_url', '')),
                                    open_hours=place_data.get('open_hours', place_data.get('business_hours', '영업시간 정보 없음')),
                                    phone=place_data.get('phone', ''),
                                    latitude=float(place_data.get('latitude', 0)),
                                    longitude=float(place_data.get('longitude', 0))
                                )
                            places_list.append(place_detail)
                        except Exception as e:
                            print(f"Error creating PlaceDetail: {e}")
                            continue
                    
                    if places_list:  # 최소한 하나의 장소가 있는 경우에만 일정 추가
                        day_plan = DayPlan(
                            day_number=day_data.get('day_number', len(daily_plans) + 1) if isinstance(day_data, dict) else len(daily_plans) + 1,
                            places=places_list
                        )
                        daily_plans.append(day_plan)
                except Exception as e:
                    print(f"Error processing day: {e}")
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
        try:
            prompt = info2guide_repository.create_travel_prompt(places, plan_type, days)
            response = await info2guide_repository.get_gpt_response(prompt)
            
            if not response or not isinstance(response, dict) or 'days' not in response:
                print(f"Invalid GPT response format: {response}")
                return TravelPlan(plan_type=plan_type, daily_plans=[])
            
            daily_plans = []
            for day_data in response['days']:
                try:
                    if not isinstance(day_data, dict):
                        print(f"Invalid day data format: {day_data}")
                        continue
                        
                    day_number = day_data.get('day_number')
                    if not day_number or day_number > days:
                        continue
                    
                    places_list = []
                    for place_data in day_data.get('places', []):
                        try:
                            if not isinstance(place_data, dict):
                                print(f"Invalid place data format: {place_data}")
                                continue
                            
                            # 원본 places 리스트에서 일치하는 장소 찾기
                            place_id = place_data.get('id')
                            if not place_id:
                                print(f"Place data missing ID: {place_data}")
                                continue
                                
                            original_place = next(
                                (p for p in places if p['id'] == place_id),
                                None
                            )
                            
                            # 원본 장소를 찾은 경우 해당 데이터 사용, 아니면 GPT 응답 데이터 사용
                            place_dict = original_place if original_place else place_data
                            
                            # 필수 필드가 없는 경우 기본값 설정
                            place_detail = PlaceDetail(
                                id=place_dict.get('id', ''),
                                title=place_dict.get('title', place_dict.get('name', '알 수 없는 장소')),
                                address=place_dict.get('address', '주소 정보 없음'),
                                description=place_dict.get('description', place_dict.get('official_description', '설명 없음')),
                                intro=place_dict.get('intro', place_dict.get('reviewer_description', '리뷰 없음')),
                                type=place_dict.get('type', place_dict.get('place_type', '기타')),
                                rating=Decimal(str(place_dict.get('rating', 0))),
                                image=place_dict.get('image', place_dict.get('image_url', '')),
                                open_hours=place_dict.get('open_hours', place_dict.get('business_hours', '영업시간 정보 없음')),
                                phone=place_dict.get('phone', ''),
                                latitude=float(place_dict.get('latitude', 0)),
                                longitude=float(place_dict.get('longitude', 0))
                            )
                            places_list.append(place_detail)
                        except Exception as e:
                            print(f"Error creating PlaceDetail: {e}")
                            continue

                    if places_list:  # 최소한 하나의 장소가 있는 경우에만 일정 추가
                        day_plan = DayPlan(
                            day_number=day_number,
                            places=places_list
                        )
                        daily_plans.append(day_plan)
                except Exception as e:
                    print(f"Error processing day {day_data.get('day_number', 'unknown')}: {e}")
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
                                id=place.get('id', ''),
                                title=place.get('title', '알 수 없는 장소'),
                                address=place.get('address', '주소 정보 없음'),
                                description=place.get('description', '설명 없음'),
                                intro=place.get('intro', '리뷰 없음'),
                                type=place.get('type', '기타'),
                                rating=Decimal(str(place.get('rating', 0))),
                                image=place.get('image', ''),
                                open_hours=place.get('open_hours', '') or '',
                                phone=place.get('phone', ''),
                                latitude=float(place.get('latitude', 0)),
                                longitude=float(place.get('longitude', 0))
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
            print(f"Error in _create_plan: {e}")
            return TravelPlan(plan_type=plan_type, daily_plans=[])
    
    def _parse_rating(self, rating_str: str) -> float:
        try:
            if rating_str in ['N/A', '', None]:
                return 0.0
            return float(rating_str)
        except (ValueError, TypeError):
            return 0.0
