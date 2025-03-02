from typing import List, Optional, Dict, Union, Any
from enum import Enum
from pydantic import BaseModel, HttpUrl, Field

class ContentType(str, Enum):
    YOUTUBE = "youtube"
    NAVER_BLOG = "naver_blog"
    TISTORY = "tistory"
    TEXT_FILE = "text_file"
    WEBPAGE = "webpage"
    UNKNOWN = "unknown"

class ContentRequest(BaseModel):
    urls: List[str]

class ContentInfo(BaseModel):
    url: str
    title: Optional[str] = ""
    author: Optional[str] = ""
    platform: ContentType
    published_date: Optional[str] = ""

class VideoInfo(BaseModel):
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None

class PlacePhoto(BaseModel):
    url: str

class PlaceGeometry(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class PlaceInfo(BaseModel):
    name: str
    source_url: str
    type: str = "unknown"
    geometry: PlaceGeometry = PlaceGeometry(latitude=None, longitude=None)
    description: Optional[str] = ""
    official_description: Optional[str] = ""
    formatted_address: Optional[str] = ""
    rating: Optional[float] = 0.0
    phone: Optional[str] = ""
    website: Optional[str] = ""
    price_level: Optional[int] = 0
    opening_hours: Optional[List[str]] = None
    photos: Optional[List[PlacePhoto]] = None
    best_review: Optional[str] = ""
    google_info: Dict = {}
    types: Optional[List[str]] = None
    precautions: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None

class YouTubeResponse(BaseModel):
    summary: Dict[str, str] = Field(default_factory=dict)
    content_infos: List[Dict] = Field(default_factory=list)
    processing_time_seconds: float = Field(default=0.0)
    place_details: List[Dict] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "summary": {"url": "요약 내용"},
                "content_infos": [],
                "processing_time_seconds": 1.23,
                "place_details": []
            }
        }

class SearchResponse(BaseModel):
    """검색 결과 응답 모델"""
    content: str
    metadata: dict
