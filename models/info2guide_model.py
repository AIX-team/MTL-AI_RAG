# models/info2guide_model.py

from pydantic import BaseModel, Field
from typing import List, Optional

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
    travel_days: int = Field(..., alias="travelDays")
    places: List[PlaceInfo]
    travelTaste: str = Field(default="적당한 일정 선호", alias="travelTaste")  # 기본값 설정

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
