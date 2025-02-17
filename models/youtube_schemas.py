from typing import List, Optional, Dict, Union, Any
from enum import Enum
from pydantic import BaseModel, HttpUrl

class ContentType(str, Enum):
    YOUTUBE = "youtube"
    NAVER_BLOG = "naver_blog"
    TISTORY = "tistory"
    TEXT_FILE = "text_file"
    WEBPAGE = "webpage"
    UNKNOWN = "unknown"

class ContentRequest(BaseModel):
    urls: List[HttpUrl]

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
    summary: Dict[str, Any]
    content_infos: List[ContentInfo]
    processing_time_seconds: float
    place_details: List[PlaceInfo]

    

class SearchResponse(BaseModel):
    """검색 결과 응답 모델"""
    content: str
    metadata: dict
