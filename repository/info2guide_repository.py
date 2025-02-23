# repository/info2guide_repository.py

import openai
from typing import List, Dict
import json

def create_travel_prompt(places: List[Dict], plan_type: str, days: int) -> str:
    """GPT 프롬프트 생성 (ID와 Address 정보 포함)"""
    total_places = len(places)
    places_per_day = {
        'busy': 4,
        'normal': 3,
        'relaxed': 2
    }
    required_places = days * places_per_day[plan_type.lower()]
    
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
     - 방문 소요 시간이 너무 짧은 장소
  3. 제외된 장소는 '추천 대체 장소' 목록으로 별도 제공"""
    
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
    
    return f"""당신은 전문 여행 플래너입니다. 현재 요청받은 {plan_type.upper()} 스타일의 여행 일정을 반드시 생성해주세요.

{balance_guide}

[여행 스타일 정의]
1. BUSY 스타일:
   - 목적: 제한된 시간에 최대한 많은 곳을 경험
   - 하루 방문 장소: 4곳
   - 장소당 체류 시간: 1-1.5시간
   - 이동 시간: 장소 간 30분 이내
   - 특징: 효율적인 동선, 주요 관광지 위주

2. NORMAL 스타일:
   - 목적: 여유있게 주요 관광지 방문
   - 하루 방문 장소: 3곳
   - 장소당 체류 시간: 1.5-2시간
   - 이동 시간: 장소 간 40분 이내
   - 특징: 관광과 휴식 균형, 대중적인 코스, 효율적인 동선

3. RELAXED 스타일:
   - 목적: 각 장소를 충분히 음미
   - 하루 방문 장소: 2곳
   - 장소당 체류 시간: 2-3시간
   - 이동 시간: 제한 없음
   - 특징: 여유로운 일정, 문화 체험 중심

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
Recommended Visit Time: [권장 방문 시간]
Expected Duration: [예상 소요 시간]

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
        # GPT 모델 이름 수정 및 에러 처리 개선
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 일본 여행 전문가입니다. 주어진 장소들을 효율적으로 연결하여 최적의 여행 일정을 만드는 것이 특기입니다. 응답은 반드시 Day 1:, ID:, Place Name: 등의 형식으로 작성해주세요."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        if not response or not response.choices:
            raise Exception("GPT API returned invalid response")
            
        print("Received response from GPT")
        content = response.choices[0].message.content
        if not content:
            raise Exception("GPT response content is empty")
            
        print(f"GPT Response Content: {content[:200]}...")
        
        parsed_response = parse_gpt_response(content)
        if not parsed_response.get('days'):
            raise Exception("Failed to parse GPT response into valid day plans")
            
        print(f"Parsed Response: {parsed_response}")
        return parsed_response
        
    except Exception as e:
        print(f"Error in get_gpt_response: {str(e)}")
        print(f"Full error details: {e.__class__.__name__}: {str(e)}")
        raise Exception(f"GPT API 호출 실패: {str(e)}")  # 에러를 상위로 전파
def parse_gpt_response(response_text: str) -> Dict:
    """GPT 응답을 파싱하여 구조화된 데이터로 변환"""
    try:
        print("Starting to parse response...")
        days = []
        current_day = None
        current_place = None
        existing_days = set()  # 이미 처리된 날짜를 추적
        
        # 불필요한 마크다운 기호 제거
        response_text = (response_text.replace('###', '')
                                    .replace('##', '')
                                    .replace('**', '')
                                    .replace('- ', '')
                                    .replace('*', '')
                                    .replace('`', ''))
        
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for line in lines:
            print(f"Processing line: {line}")
            
            # Day 시작 감지
            if line.lower().startswith('day'):
                if current_place and current_day:
                    current_day['places'].append(current_place)
                    current_place = None
                
                try:
                    day_num = int(''.join(filter(str.isdigit, line)))
                    
                    # 이미 처리된 날짜인 경우 해당 날짜의 데이터에 추가
                    if day_num in existing_days:
                        current_day = next(day for day in days if day['day_number'] == day_num)
                    else:
                        if current_day:
                            days.append(current_day)
                        current_day = {'day_number': day_num, 'places': []}
                        existing_days.add(day_num)
                    
                    print(f"Processing day: {day_num}")
                except Exception as e:
                    print(f"Error parsing day number: {e}")
                continue
            
            # 키:값 형식의 라인 처리
            if ':' in line:
                key, value = [x.strip() for x in line.split(':', 1)]
                key = key.lower().replace(' ', '_')
                
                # ID로 새로운 장소 시작 감지
                if key == 'id':
                    if current_place and current_day:
                        current_day['places'].append(current_place)
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
                
                # 키에 따른 값 매핑
                if current_place is not None:
                    if key == 'id':
                        current_place['id'] = value
                    elif key == 'place_name':
                        current_place['name'] = value
                    elif key == 'address':
                        current_place['address'] = value
                    elif key == 'official_description':
                        current_place['official_description'] = value
                    elif key in ['reviewer_description', "reviewer's_description"]:
                        current_place['reviewer_description'] = value
                    elif key == 'place_type':
                        current_place['place_type'] = value
                    elif key == 'rating':
                        current_place['rating'] = value
                    elif key in ['place_image_url', 'image_url']:
                        if '(' in value and ')' in value:
                            url = value[value.find('(')+1:value.find(')')]
                            current_place['image_url'] = url
                        else:
                            current_place['image_url'] = value
                    elif key in ['business_time', 'business_hours']:
                        current_place['business_hours'] = value
                    elif key == 'website':
                        if '(' in value and ')' in value:
                            url = value[value.find('(')+1:value.find(')')]
                            current_place['website'] = url
                        else:
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
        
        # 마지막 장소와 날짜 처리
        if current_place and current_day:
            current_day['places'].append(current_place)
        if current_day and current_day not in days:
            days.append(current_day)
            
        print(f"Final parsed days: {len(days)}")
        for day in days:
            print(f"Added day {day['day_number']} with {len(day['places'])} places")
            
        return {'days': days}
    except Exception as e:
        print(f"Error parsing GPT response: {str(e)}")
        print(f"Response text: {response_text}")
        return {'days': []}


