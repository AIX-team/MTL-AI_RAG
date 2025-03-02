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
    places: List[PlaceInfo]
    travel_days: int = Field(..., description="Number of travel days")
    travel_taste: str = Field(..., description="Travel style preference")

    class Config:
        allow_population_by_field_name = True
        json_schema_extra = {
            "example": {
                "places": [],
                "travel_days": 3,
                "travel_taste": "빼곡한 일정 선호"
            }
        }

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
