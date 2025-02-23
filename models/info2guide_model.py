# models/info2guide_model.py

from pydantic import BaseModel, Field
from typing import List, Optional
from decimal import Decimal

class CoursePlaceResponse(BaseModel):
    num: int
    id: str
    name: str
    type: str
    description: str
    image: str
    address: str
    hours: str
    intro: str
    latitude: str
    longitude: str

class CourseList(BaseModel):
    courseId: str
    courseNum: int
    coursePlaces: List[CoursePlaceResponse]

class GuideBookResponse(BaseModel):
    success: str = "success"
    message: str = "success"
    guideBookTitle: str
    travelInfoTitle: str
    travelInfoId: str
    courseCnt: int
    courses: List[CourseList]

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
    open_hours: Optional[str] = None
    phone: Optional[str] = None
    rating: Optional[Decimal] = None

class PlaceSelectRequest(BaseModel):
    travel_days: int = Field(..., alias="travelDays")
    places: List[PlaceInfo]
    travelTaste: str = Field(default="적당한 일정 선호", alias="travelTaste")

class PlaceDetail(BaseModel):
    id: str
    title: str
    address: str
    description: str
    intro: str
    type: str
    rating: Decimal
    image: str
    open_hours: str
    phone: Optional[str] = None
    latitude: float
    longitude: float

class DayPlan(BaseModel):
    day_number: int
    places: List[PlaceDetail]

class TravelPlan(BaseModel):
    plan_type: str
    daily_plans: List[DayPlan]
