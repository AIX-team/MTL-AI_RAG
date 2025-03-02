from typing import List, Union
import openai
from models.info2guide_model import PlaceInfo, PlaceDetail, DayPlan, TravelPlan
from repository import info2guide_repository
import os
from sklearn.cluster import DBSCAN
import numpy as np
import random
from math import ceil

# 상수 정의
CHUNK_SIZE = 2048  # 텍스트 청크 크기
MODEL = "gpt-4o-mini"    # 사용할 GPT 모델
MAX_TOKENS = 1500  # 최대 토큰 수

class TravelPlannerService:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def generate_travel_plans(self, places: List[PlaceInfo], days: int, plan_type: str) -> List[TravelPlan]:
        try:
            plan_type_mapping = {
                '빼곡한 일정 선호': 'busy',
                '적당한 일정 선호': 'normal',
                '널널한 일정 선호': 'relaxed'
            }
            
            if plan_type in plan_type_mapping:
                plan_type = plan_type_mapping[plan_type]
            
            # 지역 기반으로 장소 그룹핑
            places_by_area = self._group_places_by_area(places)
            
            # places를 딕셔너리로 변환
            places_dict = {place.id: {
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
                'rating': place.rating if hasattr(place, 'rating') else None,
                'area_group': self._get_area_group(place, places_by_area)
            } for place in places}

            prompt = info2guide_repository.create_travel_prompt(
                list(places_dict.values()), 
                plan_type, 
                days,
                places_by_area
            )
            
            response = await info2guide_repository.get_gpt_response(prompt)
            
            if not response or 'days' not in response:
                print(f"No valid response for {plan_type} plan")
                return self._create_default_plan(places, days, plan_type, places_by_area)
            
            # GPT 응답 검증
            valid_response = self._validate_and_fix_gpt_response(
                response, 
                places_dict, 
                days, 
                plan_type,
                places_by_area
            )
            
            if not valid_response['days']:
                return self._create_default_plan(places, days, plan_type, places_by_area)
            
            return [TravelPlan(
                plan_type=plan_type,
                daily_plans=valid_response['days']
            )]
            
        except Exception as e:
            print(f"Error generating travel plans: {e}")
            return self._create_default_plan(places, days, plan_type, places_by_area)

    def _group_places_by_area(self, places: List[PlaceInfo]) -> dict:
        """위도/경도 기반으로 근접한 장소들을 그룹화"""
        coords = np.array([[p.latitude, p.longitude] for p in places])
        clustering = DBSCAN(eps=0.01, min_samples=1).fit(coords)
        
        groups = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in groups:
                groups[label] = []
            groups[label].append(places[idx])
        
        return groups

    def _get_area_group(self, place: PlaceInfo, places_by_area: dict) -> int:
        """장소가 속한 지역 그룹 찾기"""
        for area_id, area_places in places_by_area.items():
            if any(p.id == place.id for p in area_places):
                return area_id
        return -1

    def _validate_and_fix_gpt_response(self, response: dict, places_dict: dict, days: int, plan_type: str, places_by_area: dict) -> dict:
        """GPT 응답을 검증하고 필요한 경우 수정"""
        ranges = {
            'busy': (4, 5),
            'normal': (3, 4),
            'relaxed': (2, 3)
        }
        min_places, max_places = ranges.get(plan_type.lower(), (3, 4))
        
        fixed_days = []
        used_places = set()
        
        for day_num in range(1, days + 1):
            day_data = next((d for d in response['days'] if d['day_number'] == day_num), None)
            if not day_data or not day_data.get('places'):
                # 해당 일자 계획이 없거나 장소가 비어있으면 새로 생성
                new_places = self._select_places_for_day(
                    places_by_area,
                    used_places,
                    min_places,
                    max_places
                )
                if new_places:
                    fixed_days.append(DayPlan(
                        day_number=day_num,
                        places=[self._create_place_detail(p) for p in new_places]
                    ))
                    used_places.update(p.id for p in new_places)
                continue
            
            # 기존 일정의 장소 수가 범위를 벗어나면 조정
            day_places = []
            current_area = None
            
            for place in day_data['places']:
                place_id = place.get('id')
                if place_id in places_dict and place_id not in used_places:
                    place_info = places_dict[place_id]
                    if not current_area:
                        current_area = place_info['area_group']
                        day_places.append(place_info)
                    elif place_info['area_group'] == current_area:
                        day_places.append(place_info)
                    used_places.add(place_id)
            
            # 장소 수가 부족하면 같은 지역의 장소로 보충
            while len(day_places) < min_places:
                additional = self._select_places_for_day(
                    places_by_area,
                    used_places,
                    1,
                    1,
                    preferred_area=current_area
                )
                if not additional:
                    break
                day_places.extend(additional)
                used_places.update(p.id for p in additional)
            
            if day_places:
                fixed_days.append(DayPlan(
                    day_number=day_num,
                    places=[self._create_place_detail(p) for p in day_places[:max_places]]
                ))
        
        return {'days': fixed_days}

    def _select_places_for_day(
        self, 
        places_by_area: dict, 
        used_places: set, 
        min_places: int, 
        max_places: int,
        preferred_area: int = None
    ) -> List[PlaceInfo]:
        """하루 일정에 들어갈 장소들을 선택"""
        if preferred_area is not None and preferred_area in places_by_area:
            available = [p for p in places_by_area[preferred_area] if p.id not in used_places]
            if available:
                count = random.randint(min_places, min(max_places, len(available)))
                return sorted(available, key=lambda p: p.rating if hasattr(p, 'rating') else 0, reverse=True)[:count]
        
        all_available = []
        for area_places in places_by_area.values():
            all_available.extend([p for p in area_places if p.id not in used_places])
        
        if not all_available:
            return []
        
        count = random.randint(min_places, min(max_places, len(all_available)))
        return sorted(all_available, key=lambda p: p.rating if hasattr(p, 'rating') else 0, reverse=True)[:count]

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

class TextProcessingService:
    """텍스트 처리 서비스"""
    
    @staticmethod
    def split_text(text: str, max_chunk_size: int = CHUNK_SIZE) -> List[str]:
        """텍스트를 청크로 분할"""
        try:
            words = text.split()
            total_words = len(words)
            num_chunks = ceil(total_words / (max_chunk_size // 5))
            chunks = []
            
            for i in range(num_chunks):
                start = i * (max_chunk_size // 5)
                end = start + (max_chunk_size // 5)
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
            
            print(f"[split_text] 총 단어 수: {total_words}, 청크 수: {num_chunks}")
            return chunks
            
        except Exception as e:
            print(f"[split_text] 오류 발생: {str(e)}")
            raise ValueError(f"텍스트 분할 중 오류 발생: {str(e)}")

    @staticmethod
    def _generate_prompt(text: str) -> str:
        """GPT 프롬프트 생성"""
        return f"""다음 텍스트를 분석하여 여행 정보를 요약해주세요. 
특히 다음 사항에 중점을 두어 요약해주세요:

1. 방문한 장소들 (위치 정보 포함)
2. 각 장소의 특징과 설명
3. 추천 사항이나 주의 사항
4. 시간대별 방문 정보 (있는 경우)

텍스트:
{text}

다음 형식으로 응답해주세요:

방문한 장소: [장소명] ([지역명])
- 설명: [장소에 대한 설명]
- 추천 사항: [있는 경우]
- 주의 사항: [있는 경우]
- 방문 시간: [언급된 경우]
"""

    @staticmethod
    def summarize_text(transcript_chunks: List[str], model: str = MODEL) -> str:
        """텍스트 청크들을 요약"""
        try:
            summaries = []
            for idx, chunk in enumerate(transcript_chunks):
                print(f"[summarize_text] 청크 {idx+1}/{len(transcript_chunks)} 처리 중...")
                
                prompt = TextProcessingService._generate_prompt(chunk)
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "당신은 여행 전문가로서 여행 컨텐츠를 분석하고 유용한 정보를 추출하는 AI입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=MAX_TOKENS
                )
                
                summary = response.choices[0].message.content
                summaries.append(summary)
                print(f"[summarize_text] 청크 {idx+1} 요약 완료")
            
            return "\n\n".join(summaries)
            
        except Exception as e:
            print(f"[summarize_text] 오류 발생: {str(e)}")
            raise ValueError(f"텍스트 요약 중 오류 발생: {str(e)}")
