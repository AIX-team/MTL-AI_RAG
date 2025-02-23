# repository/info2guide_repository.py

import openai
from typing import List, Dict
from models.info2guide_model import PlaceInfo, PlaceDetail, DayPlan, TravelPlan
import json

def create_travel_prompt(places: List[Dict], plan_type: str, days: int, places_by_area: dict = None) -> str:
    """GPT 프롬프트 생성 (ID와 Address 정보 포함)"""
    total_places = len(places)
    places_per_day = {
        'busy': (4, 5),
        'normal': (3, 4),
        'relaxed': (2, 3)
    }
    min_places, max_places = places_per_day[plan_type.lower()]
    required_places = days * min_places
    
    # 지역 정보 문자열 생성
    area_info = ""
    if places_by_area:
        area_info = "\n[지역 그룹 정보]\n"
        for area_id, area_places in places_by_area.items():
            center = area_places[0]
            area_info += (
                f"지역 {area_id}:\n"
                f"중심 위치: {center.address}\n"
                f"포함 장소: {', '.join(p.title for p in area_places)}\n"
                f"좌표: ({center.latitude}, {center.longitude})\n\n"
            )
    
    balance_guide = ""
    if total_places < required_places:
        balance_guide = f"""
[장소 부족 상황 가이드]
- 전체 필요 장소 수: {required_places}개
- 현재 가용 장소 수: {total_places}개
- 처리 방법:
  1. 중요도/평점 순으로 장소 우선순위 지정
  2. 각 날짜별로 가용한 장소 균등 분배
  3. 부족한 날짜는 반일 일정으로 구성
  4. 빈 시간대는 자유시간으로 할당"""
    elif total_places > required_places:
        balance_guide = f"""
[장소 초과 상황 가이드]
- 전체 필요 장소 수: {required_places}개
- 현재 가용 장소 수: {total_places}개
- 처리 방법:
  1. 장소 선정 우선순위:
     - 평점 4.5 이상 우선
     - 주요 관광지(랜드마크) 우선
     - 계절/날씨 적합성
     - 교통 접근성
  2. 선정 제외 기준:
     - 유사한 성격의 중복 장소
     - 이동 시간이 긴 원거리 장소
     - 방문 소요 시간이 너무 짧은 장소"""
    
    place_details = "\n".join([
        f"Place {i+1}:\n"
        f"ID: {place.get('id', 'N/A')}\n"
        f"Name: {place['title']}\n"
        f"Address: {place['address']}\n"
        f"Description: {place['description']}\n"
        f"Type: {place['type']}\n"
        f"Opening Hours: {place['open_hours']}\n"
        f"Image: {place['image']}\n"
        f"Location: {place['latitude']}, {place['longitude']}\n"
        for i, place in enumerate(places)
    ])
    
    style_description = {
        'busy': """
[빼곡한 일정 선호 상세]
- 하루 4곳 방문
- 장소당 체류시간: 1-1.5시간
- 이동시간: 30분 이내
- 효율적인 동선 중시
- 주요 관광지 위주""",
        'normal': """
[적당한 일정 선호 상세]
- 하루 3곳 방문
- 장소당 체류시간: 1.5-2시간
- 이동시간: 40분 이내
- 관광과 휴식 균형
- 대중적인 코스""",
        'relaxed': """
[널널한 일정 선호 상세]
- 하루 2곳 방문
- 장소당 체류시간: 2-3시간
- 이동시간: 제한 없음
- 여유로운 일정
- 문화체험 중심"""
    }
    
    return f"""당신은 전문 여행 플래너입니다. 현재 요청받은 {plan_type.upper()} 스타일의 여행 일정을 반드시 생성해주세요.

[필수 규칙]
1. 동선 최적화 (최우선 규칙)
   - 같은 지역의 장소들을 하루에 묶어서 배치
   - 이동 시간 최소화를 위해 근접한 장소끼리 순서 배치
   - 지하철/버스 환승 횟수 최소화
   - 하루 이동 거리 3시간 이내로 제한

2. 일정 수 (스타일별)
   - BUSY: 하루 4-5곳 방문
   - NORMAL: 하루 3-4곳 방문
   - RELAXED: 하루 2-3곳 방문

{area_info}
{balance_guide}

[일정 수립 규칙]
1. 시간대별 최적화:
   - 오전(9-12시): 주요 관광지, 혼잡한 장소 우선
   - 점심(12-14시): 식당 또는 가벼운 관광
   - 오후(14-17시): 박물관, 쇼핑, 공원 등
   - 저녁(17시 이후): 야경 명소, 식당가

2. 장소 조합 필수 규칙:
   - 연속된 두 장소는 같은 유형이 될 수 없음
   - 실내/실외 장소 교차 배치
   - 활동적/휴식 장소 균형
   - 유사 성격의 장소 연속 배치 지양
   - 식사 시간 고려한 레스토랑 배치

3. 동선 필수 규칙:
   - 각 장소의 영업시간 준수
   - 이동거리 최적화
   - 혼잡시간 회피
   - 날씨 영향 고려

주어진 장소 목록:
{place_details}

다음 형식으로 정확히 응답해주세요:

Day 1:
ID: [장소 ID]
Place Name: [장소명]
Address: [주소]
Official Description: [공식 설명]
Reviewer's Description: [방문 포인트]
Place Type: [장소 유형]
Rating: [평점]
Place Image URL: [이미지 URL]
Business Time: [영업 시간]
Website: [웹사이트 URL]
Location: [위도, 경도]

[다음 장소...]

Day 2:
[같은 형식 반복]

주의사항:
1. {plan_type.upper()} 스타일에 맞는 하루 방문 장소 수를 지켜주세요.
2. 각 장소의 실제 위치와 영업시간을 고려해 현실적인 일정을 작성해주세요.
3. 이동 시간과 체류 시간을 고려하여 하루 일정이 무리하지 않도록 해주세요.
4. 주어진 장소들의 특성을 고려하여 최적의 방문 순서를 정해주세요.
5. 최대한 같은 지역에 있는 장소들을 방문하도록 해주세요."""

async def get_gpt_response(prompt: str) -> Dict:
    try:
        print("Sending request to GPT...")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # 모델 변경
            messages=[
                {
                    "role": "system",
                    "content": "당신은 일본 여행 전문가입니다. 주어진 장소들을 효율적으로 연결하여 최적의 여행 일정을 만드는 것이 특기입니다."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8192,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        print("Received response from GPT")
        content = response.choices[0].message.content
        print(f"GPT Response Content: {content[:200]}...")
        
        parsed_response = parse_gpt_response(content)
        print(f"Parsed Response: {parsed_response}")
        return parsed_response
    except Exception as e:
        print(f"Error in get_gpt_response: {str(e)}")
        return {'days': []}

def parse_gpt_response(response_text: str) -> Dict:
    """GPT 응답을 파싱하여 구조화된 데이터로 변환"""
    try:
        print("Starting to parse response...")
        days = []
        current_day = None
        current_place = None
        
        # 불필요한 기호 제거
        response_text = (response_text.replace('###', '')
                                    .replace('**', '')
                                    .replace('- ', '')
                                    .replace('*', '')
                                    .replace('`', ''))
        
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            if line.lower().startswith('day'):
                if current_place and current_day:
                    current_day['places'].append(current_place)
                if current_day:
                    days.append(current_day)
                try:
                    day_num = int(''.join(filter(str.isdigit, line)))
                    current_day = {'day_number': day_num, 'places': []}
                except Exception as e:
                    print(f"Error parsing day number: {e}")
                current_place = None
                continue
            
            if ':' in line:
                key, value = [x.strip() for x in line.split(':', 1)]
                key = key.lower().replace(' ', '_')
                
                if current_place is None:
                    current_place = {
                        'id': '',
                        'name': '',
                        'address': '',
                        'official_description': '',
                        'reviewer_description': '',
                        'place_type': '',
                        'rating': '0',
                        'image_url': '',
                        'business_hours': '',
                        'website': '',
                        'latitude': '',
                        'longitude': ''
                    }
                
                if key == 'id':
                    current_place['id'] = value
                elif key == 'place_name':
                    current_place['name'] = value
                elif key == 'address':
                    current_place['address'] = value
                elif key == 'official_description':
                    current_place['official_description'] = value
                elif key == 'reviewer_description' or key == "reviewer's_description":
                    current_place['reviewer_description'] = value
                elif key == 'place_type':
                    current_place['place_type'] = value
                elif key == 'rating':
                    current_place['rating'] = value
                elif key in ['place_image_url', 'image_url']:
                    current_place['image_url'] = value
                elif key in ['business_time', 'business_hours']:
                    current_place['business_hours'] = value
                elif key == 'website':
                    current_place['website'] = value
                elif key == 'location':
                    try:
                        lat, lon = value.split(',')
                        current_place['latitude'] = lat.strip()
                        current_place['longitude'] = lon.strip()
                    except Exception as e:
                        print(f"Error parsing location: {e}")
                        current_place['latitude'] = ''
                        current_place['longitude'] = ''
        
        if current_place and current_day:
            current_day['places'].append(current_place)
        if current_day:
            days.append(current_day)
            
        print(f"Final parsed days: {len(days)}")
        return {'days': days}
    except Exception as e:
        print(f"Error parsing GPT response: {str(e)}")
        print(f"Response text: {response_text}")
        return {'days': []}


