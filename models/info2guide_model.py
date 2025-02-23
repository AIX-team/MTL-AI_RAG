# models/info2guide_model.py

from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic import validator

class PlaceInfo(BaseModel):
    id: str
    address: str
    title: str
    description: str
    intro: str
    type: str
    image: str
    latitude: float
    longitude: float
    open_hours:  Optional[str] = None  # Optional 필드로 변경
    phone: Optional[str] = None  # Optional 필드로 변경
    rating: Optional[float] = None  # rating도 Optional로 변경 가능

class PlaceSelectRequest(BaseModel):
    places: List[PlaceInfo]
    travel_days: int
    travel_taste: str
    
    @validator('travel_taste')
    def validate_travel_taste(cls, v):
        # 한글-영문 매핑 정의
        taste_mapping = {
            '빼곡한 일정 선호': 'busy',
            '적당한 일정 선호': 'normal',
            '널널한 일정 선호': 'relaxed'
        }
        
        # 이미 영문이면 그대로 검증
        if v.lower() in ['busy', 'normal', 'relaxed']:
            return v.lower()
            
        # 한글이면 매핑된 영문으로 변환
        if v in taste_mapping:
            return taste_mapping[v]
            
        # 둘 다 아니면 에러
        valid_tastes = list(taste_mapping.keys()) + ['busy', 'normal', 'relaxed']
        raise ValueError(f'travel_taste must be one of {valid_tastes}')

class PlaceDetail(BaseModel):
    id: str
    name: str
    address: str
    official_description: str
    reviewer_description: str
    place_type: str
    rating: float
    image_url: str
    business_hours: str
    website: str
    latitude: Optional[str] = ''
    longitude: Optional[str] = ''

class DayPlan(BaseModel):
    day_number: int
    places: List[PlaceDetail]

class TravelPlan(BaseModel):
    plan_type: str  # 'busy', 'normal', 'relaxed'
    daily_plans: List[DayPlan]
