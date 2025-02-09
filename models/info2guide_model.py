from pydantic import BaseModel
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
    open_hours: str
    phone: str
    rating: float

class PlaceSelectRequest(BaseModel):
    travel_days: int
    places: List[PlaceInfo]

class PlaceDetail(BaseModel):
    name: str
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