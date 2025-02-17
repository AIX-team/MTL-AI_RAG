from pydantic import BaseModel
from typing import List

class PlacePhoto(BaseModel):
    url: str

    class Config:
        from_attributes = True

class MapLocation(BaseModel):
    name: str
    lat: float
    lng: float
   
class AIPlace(BaseModel):
    id: str
    address: str
    title: str
    description: str
    intro: str = ""
    type: str
    image: List[PlacePhoto]
    latitude: float
    longitude: float
    openHours: List[str]
    phone: str
    rating: float

    class Config:
        from_attributes = True
   