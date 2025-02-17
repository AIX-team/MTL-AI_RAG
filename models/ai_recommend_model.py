from pydantic import BaseModel, Field
from typing import List, Optional
from decimal import Decimal

class PlaceBase(BaseModel):
    placeId: str = Field(alias="placeId")
    placeType: Optional[str] = Field(alias="placeType", default="unknown")
    placeName: str = Field(alias="placeName")
    placeAddress: Optional[str] = Field(alias="placeAddress", default="")
    placeImage: Optional[str] = Field(alias="placeImage", default="")
    placeDescription: Optional[str] = Field(alias="placeDescription", default="")
    intro: Optional[str] = Field(alias="intro", default="")
    latitude: Optional[Decimal] = Field(alias="latitude", default=None)
    longitude: Optional[Decimal] = Field(alias="longitude", default=None)

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        alias_generator = None

class AIRecommendRequest(BaseModel):
    travelInfoId: str = Field(alias="travelInfoId")
    travelDays: int = Field(alias="travelDays")
    places: List[PlaceBase]

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        alias_generator = None

class AIRecommendResponse(BaseModel):
    success: str
    message: str
    content: List[PlaceBase]

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        alias_generator = None 