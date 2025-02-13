from typing import List, Optional, Dict, Union
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
    published_date: Optional[str] = None

class VideoInfo(BaseModel):
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None

class PlacePhoto(BaseModel):
    url: str

class PlaceInfo(BaseModel):
    name: str  # 장소 이름
    source_url: str  # 장소 정보의 출처 URL
    timestamp: Optional[str] = None  # 영상에서의 타임스탬프
    description: Optional[str] = None  # 유튜버의 장소 설명
    official_description: Optional[str] = None  # 공식/일반적인 장소 설명
    formatted_address: Optional[str] = None  # 주소
    coordinates: Optional[Dict[str, float]] = None  # 위도/경도 정보
    rating: Optional[float] = None  # 평점
    phone: Optional[str] = None  # 전화번호
    website: Optional[str] = None  # 웹사이트
    price_level: Optional[int] = None  # 가격대
    opening_hours: Optional[List[str]] = None  # 영업시간
    photos: Optional[List[PlacePhoto]] = None  # 사진 목록
    best_review: Optional[str] = None  # 베스트 리뷰
    google_info: Dict = {}  # 구글 장소 정보
    types: Optional[List[str]] = None  # 장소 유형
    precautions: Optional[List[str]] = None  # 유의사항 목록
    recommendations: Optional[List[str]] = None  # 추천사항 목록

class YouTubeResponse(BaseModel):
    summary: Dict[str, str]
    content_infos: List[ContentInfo]
    processing_time_seconds: float
    place_details: List[PlaceInfo]

    

class SearchResponse(BaseModel):
    """검색 결과 응답 모델"""
    content: str
    metadata: dict
