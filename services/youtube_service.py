import time
import os
import requests
import datetime
import tiktoken
from cachetools import TTLCache
from math import ceil
from langdetect import detect
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
import openai
from googleapiclient.discovery import build
import googlemaps
from typing import List, Dict, Tuple, Any
from models.youtube_schemas import YouTubeResponse, VideoInfo, PlaceInfo, PlacePhoto, ContentType, ContentInfo, PlaceGeometry
from repository.youtube_repository import YouTubeRepository
from langchain.schema import Document
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fastapi import APIRouter, HTTPException, status



from ai_api.youtube_subtitle import (
    get_video_info, process_link, split_text, summarize_text,
    extract_place_names, search_place_details, get_place_photo_google
)

# 환경 변수 및 상수 설정
load_dotenv(dotenv_path=".env")

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# 상수 설정
MAX_URLS = 5
CHUNK_SIZE = 2048
MODEL = "gpt-4o-mini"
FINAL_SUMMARY_MAX_TOKENS = 1500

class ContentService:
    """컨텐츠 처리 서비스"""
    
    @staticmethod
    def get_content_info(url: str) -> Tuple[str, str, ContentType]:
        """URL에서 컨텐츠 정보 추출"""
        content_type = ContentService._detect_content_type(url)
        
        if content_type == ContentType.YOUTUBE:
            title, author = YouTubeSubtitleService.get_video_info(url)
        elif content_type == ContentType.NAVER_BLOG:
            title, author = ContentService._get_naver_blog_info(url)
        elif content_type == ContentType.TISTORY:
            title, author = ContentService._get_tistory_blog_info(url)
        else:
            title, author = ContentService._get_webpage_info(url)
            
        return title, author, content_type

    @staticmethod
    def _detect_content_type(url: str) -> ContentType:
        """URL 유형 감지"""
        domain = urlparse(url).netloc.lower()
        
        if "youtube.com" in domain or "youtu.be" in domain:
            return ContentType.YOUTUBE
        elif "blog.naver.com" in domain:
            return ContentType.NAVER_BLOG
        elif ".tistory.com" in domain:
            return ContentType.TISTORY
        elif url.endswith(".txt"):
            return ContentType.TEXT_FILE
        elif url.startswith("http"):
            return ContentType.WEBPAGE
        return ContentType.UNKNOWN

    @staticmethod
    def _get_naver_blog_info(url: str) -> Tuple[str, str]:
        """네이버 블로그 정보 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # iframe 내부의 실제 컨텐츠 URL 찾기
            if 'blog.naver.com' in url:
                iframe = soup.find('iframe', id='mainFrame')
                if iframe and iframe.get('src'):
                    real_url = f"https://blog.naver.com{iframe['src']}"
                    response = requests.get(real_url, headers=headers)
                    soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.find('meta', property='og:title')
            title = title['content'] if title else "제목 없음"
            
            author = soup.find('meta', property='og:article:author')
            author = author['content'] if author else "작성자 없음"
            
            return title, author
        except Exception as e:
            print(f"네이버 블로그 정보 추출 실패: {e}")
            return None, None

    @staticmethod
    def _get_tistory_blog_info(url: str) -> Tuple[str, str]:
        """티스토리 블로그 정보 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.find('meta', property='og:title')
            title = title['content'] if title else "제목 없음"
            
            author = soup.find('meta', property='og:article:author')
            author = author['content'] if author else "작성자 없음"
            
            return title, author
        except Exception as e:
            print(f"티스토리 블로그 정보 추출 실패: {e}")
            return None, None

    @staticmethod
    def process_content(url: str) -> str:
        """URL에서 컨텐츠 추출"""
        content_type = ContentService._detect_content_type(url)
        
        if content_type == ContentType.YOUTUBE:
            return YouTubeSubtitleService.process_link(url)
        elif content_type == ContentType.NAVER_BLOG:
            return ContentService._get_naver_blog_content(url)
        elif content_type == ContentType.TISTORY:
            return ContentService._get_tistory_blog_content(url)
        elif content_type == ContentType.TEXT_FILE:
            return YouTubeSubtitleService._get_text_from_file(url)
        else:
            return YouTubeSubtitleService._get_text_from_webpage(url)

    @staticmethod
    def _get_naver_blog_content(url: str) -> str:
        """네이버 블로그에서 본문을 가져오는 함수 (불필요한 개행 및 공백, 광고 제거 포함)"""
        cache = TTLCache(maxsize=10, ttl=300)

        if url in cache:
            return cache[url]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        def clean_text(text: str) -> str:
            import re
            text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로 변환
            text = re.sub(r'[^\S\r\n]+', ' ', text)  # 유니코드 공백 제거
            text = re.sub(r'©.*?(?= )', '', text)   # © 등 불필요한 문구 제거
            text = re.sub(r'\[바로가기\]', '', text)
            return text.strip()

        try:
            # 첫 번째 요청으로 iframe URL 가져오기
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # iframe 찾기 (새 버전 블로그)
            iframe = soup.find('iframe', id='mainFrame')
            if iframe and iframe.get('src'):
                real_url = f"https://blog.naver.com{iframe['src']}"
                response = requests.get(real_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

            # 새로운 버전 블로그 영역
            content = soup.find('div', {'class': 'se-main-container'})
            if not content:
                # 구버전 블로그 영역
                content = soup.find('div', {'class': 'post-view'})
            if not content:
                raise HTTPException(status_code=404, detail="본문을 찾을 수 없습니다.")

            # 불필요한 태그 제거
            for tag in content.find_all(['script', 'style']):
                tag.decompose()

            text = clean_text(content.get_text(separator='\n'))
            cache[url] = text
            return text

        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"블로그 데이터를 가져오는 중 오류 발생: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

    @staticmethod
    def _get_tistory_blog_content(url: str) -> str:
        """티스토리 블로그 컨텐츠 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 본문 컨텐츠 찾기
            content = soup.find('div', {'class': 'entry-content'})
            if not content:
                content = soup.find('div', {'class': 'article'})
            
            if content:
                # 불필요한 요소 제거
                for element in content.find_all(['script', 'style']):
                    element.decompose()
                
                text = content.get_text(separator='\n').strip()
                return text
            
            return "컨텐츠를 찾을 수 없습니다."
        except Exception as e:
            print(f"티스토리 블로그 컨텐츠 추출 실패: {e}")
            return "컨텐츠 추출 실패"

class YouTubeService:
    """메인 YouTube 서비스"""
    
    def __init__(self):
        """YouTubeService 초기화"""
        from dotenv import load_dotenv
        load_dotenv()  # .env 파일 로드
        
        self.repository = YouTubeRepository()
        self.content_service = ContentService()
        self.text_service = TextProcessingService()
        self.place_service = PlaceService()
        
        # 청크 크기 제한 설정
        self.MAX_CHUNK_SIZE = 50000  # 50KB
        self.MAX_TOTAL_SIZE = 200000  # 200KB
        
        # Google Maps API 키 확인 및 설정
        google_maps_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not google_maps_api_key:
            raise ValueError("GOOGLE_PLACES_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        try:
            self.gmaps = googlemaps.Client(key=google_maps_api_key)
        except Exception as e:
            raise ValueError(f"Google Maps 클라이언트 초기화 실패: {str(e)}")

    async def process_urls(self, urls: List[str]) -> Dict:
        """URL 목록을 처리하여 각각의 요약을 생성"""
        try:
            content_infos = []
            place_details = []
            final_summaries = {}  # 최종 요약을 저장할 딕셔너리
            start_time = time.time()

            for url in urls:
                # URL 처리 크기 제한 확인
                if len(url.encode()) > self.MAX_CHUNK_SIZE:
                    print(f"Warning: URL too long, skipping: {url[:100]}...")
                    continue

                parsed_url = urlparse(url)
                if 'youtube.com' in parsed_url.netloc:
                    await self._process_youtube_url(url, content_infos, place_details)
                elif 'blog.naver.com' in parsed_url.netloc:
                    await self._process_naver_blog_url(url, content_infos, place_details)

            processing_time = time.time() - start_time

            # 결과 데이터 크기 제한
            result = self._create_limited_result(content_infos, place_details, processing_time)
                    
            await self.repository.save_to_vectordb(final_summaries, content_infos, place_details)
            
            return result

        except Exception as e:
            print(f"Error in process_urls: {str(e)}")
            raise ValueError(f"URL 처리 중 오류 발생: {str(e)}")


    def _create_limited_result(self, content_infos, place_details, processing_time):
        """결과 데이터 크기를 제한하여 생성"""
        summaries = {}
        limited_place_details = []
        
        total_size = 0
        
        # 컨텐츠 정보 제한
        for info in content_infos:
            if total_size >= self.MAX_TOTAL_SIZE:
                break
            
            summary = self._format_final_result(
                content_infos=[info],
                place_details=[p for p in place_details if p.source_url == info.url],
                processing_time=processing_time,
                urls=[info.url]
            )
            
            summary_size = len(str(summary).encode())
            if total_size + summary_size <= self.MAX_TOTAL_SIZE:
                summaries[info.url] = summary
                total_size += summary_size
        
        # 장소 상세 정보 제한
        for place in place_details:
            place_size = len(str(place.dict()).encode())
            if total_size + place_size <= self.MAX_TOTAL_SIZE:
                limited_place_details.append(place)
                total_size += place_size
            else:
                break
        
        return {
            "summary": summaries,
            "content_infos": [info.dict() for info in content_infos],
            "processing_time_seconds": processing_time,
            "place_details": [place.dict() for place in limited_place_details]
        }

    async def _process_youtube_url(self, url, content_infos, place_details):
        """YouTube URL 처리"""
        video_id = parse_qs(urlparse(url).query).get('v', [None])[0]
        if video_id:
            video_info = self._get_video_info(video_id)
            content_info = ContentInfo(
                url=url,
                title=video_info.title,
                author=video_info.channel,
                platform=ContentType.YOUTUBE
            )
            content_infos.append(content_info)
            
            video_places = self._process_youtube_video(video_id, url)
            place_details.extend(video_places)

    async def _process_naver_blog_url(self, url, content_infos, place_details):
        """네이버 블로그 URL 처리"""
        title, author = self.content_service._get_naver_blog_info(url)
        content = self.content_service.process_content(url)
        
        content_info = ContentInfo(
            url=url,
            title=title,
            author=author,
            platform=ContentType.NAVER_BLOG
        )
        content_infos.append(content_info)
        
        blog_places = self._process_naver_blog(url)
        place_details.extend(blog_places)

    def _process_youtube_video(self, video_id: str, source_url: str) -> List[PlaceInfo]:
        """YouTube 영상을 처리하여 장소 정보를 수집"""
        try:
            # YouTube 자막 가져오기
            transcript_text = self._get_youtube_transcript(video_id)
            
            # 텍스트 분할 및 요약
            chunks = self.text_service.split_text(transcript_text)
            summary = self.text_service.summarize_text(chunks)
            
            # 장소 추출 및 정보 수집
            place_names = self.place_service.extract_place_names(summary)
            print(f"추출된 장소: {place_names}")
            
            # 장소 정보 수집
            place_details = []
            for place_info in place_names:
                try:
                    # 장소명과 지역명 분리
                    if " (" in place_info and ")" in place_info:
                        place_name, area = place_info.split(" (")
                        area = area.rstrip(")")
                    else:
                        place_name = place_info
                        area = "일본"
                    
                    # Google Places API로 장소 정보 검색
                    places_result = self.gmaps.places(f"{place_name} {area}")
                    if not places_result['results']:
                        print(f"검색 결과 없음: {place_name} {area}")
                        continue
                    
                    # 검색 결과가 있으면 첫 번째 결과로 진행
                    place = places_result['results'][0]
                    place_id = place['place_id']
                    details = self.gmaps.place(place_id, language='ko')['result']
                    
                    # 장소 타입과 좌표 정보 추출
                    place_type = details.get('types', ['unknown'])[0]
                    location = details.get('geometry', {}).get('location', {})
                    geometry = PlaceGeometry(
                        latitude=location.get('lat'),
                        longitude=location.get('lng')
                    )
                    
                    # 새로운 PlaceInfo 객체 생성
                    place_info_obj = PlaceInfo(
                        name=place_name,
                        source_url=source_url,
                        type=place_type,
                        geometry=geometry,
                        description=self._extract_place_description(summary, place_name),
                        official_description=self._get_place_description_from_openai(place_name, place_type),
                        formatted_address=details.get('formatted_address'),
                        rating=details.get('rating'),
                        phone=details.get('formatted_phone_number'),
                        website=details.get('website'),
                        price_level=details.get('price_level'),
                        opening_hours=details.get('opening_hours', {}).get('weekday_text'),
                        google_info=details
                    )
                    
                    # 사진 URL 추가
                    if 'photos' in details:
                        photo_ref = details['photos'][0]['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={os.getenv('GOOGLE_PLACES_API_KEY')}"
                        place_info_obj.photos = [PlacePhoto(url=photo_url)]
                    
                    # 베스트 리뷰 추가
                    if 'reviews' in details:
                        best_review = max(details['reviews'], key=lambda x: x.get('rating', 0))
                        place_info_obj.best_review = best_review.get('text')
                    
                    place_details.append(place_info_obj)
                    print(f"장소 정보 추가 완료: {place_name}")
                except Exception as e:
                    print(f"장소 정보 처리 중 오류 발생 ({place_info}): {str(e)}")
                    continue
            
            return place_details
            
        except Exception as e:
            raise Exception(f"URL 처리 중 오류 발생: {str(e)}")

    def _process_naver_blog(self, url: str) -> List[PlaceInfo]:
        """네이버 블로그 글을 처리하여 요약을 생성, YouTube와 동일한 흐름으로 처리"""
        try:
            # 네이버 블로그 컨텐츠 추출 (전용 메서드 사용)
            content = self.content_service._get_naver_blog_content(url)

            # 텍스트 분할 및 요약
            chunks = self.text_service.split_text(content)
            summary = self.text_service.summarize_text(chunks)

            # 요약 텍스트에서 장소 추출
            place_names = self.place_service.extract_place_names(summary)
            print(f"추출된 장소: {place_names}")

            # 장소 정보 수집
            place_details = []
            for place_name in place_names:
                try:
                    # 장소명과 지역명 분리
                    if " (" in place_name and ")" in place_name:
                        place_name, area = place_name.split(" (")
                        area = area.rstrip(")")
                    else:
                        place_name = place_name
                        area = "일본"

                    # Google Places API로 장소 정보 검색
                    places_result = self.gmaps.places(f"{place_name} {area}")
                    if not places_result['results']:
                        print(f"검색 결과 없음: {place_name} {area}")
                        continue

                    # 검색 결과가 있으면 첫 번째 결과로 진행
                    place = places_result['results'][0]
                    place_id = place['place_id']
                    details = self.gmaps.place(place_id, language='ko')['result']

                    # 장소 타입 및 좌표 정보 추출
                    place_type = details.get('types', ['unknown'])[0]
                    location = details.get('geometry', {}).get('location', {})
                    geometry = PlaceGeometry(
                        latitude=location.get('lat'),
                        longitude=location.get('lng')
                    )

                    # 새로운 PlaceInfo 객체 생성
                    place_info_obj = PlaceInfo(
                        name=place_name,
                        source_url=url,
                        type=place_type,
                        geometry=geometry,
                        description=self._extract_place_description(summary, place_name),
                        official_description=self._get_place_description_from_openai(place_name, place_type),
                        formatted_address=details.get('formatted_address'),
                        rating=details.get('rating'),
                        phone=details.get('formatted_phone_number'),
                        website=details.get('website'),
                        price_level=details.get('price_level'),
                        opening_hours=details.get('opening_hours', {}).get('weekday_text'),
                        google_info=details
                    )

                    # 사진 URL 추가
                    if 'photos' in details:
                        photo_ref = details['photos'][0]['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={os.getenv('GOOGLE_PLACES_API_KEY')}"
                        place_info_obj.photos = [PlacePhoto(url=photo_url)]

                    # 베스트 리뷰 추가
                    if 'reviews' in details:
                        best_review = max(details['reviews'], key=lambda x: x.get('rating', 0))
                        place_info_obj.best_review = best_review.get('text')

                    place_details.append(place_info_obj)
                    print(f"장소 정보 추가 완료: {place_name}")
                except Exception as e:
                    print(f"장소 정보 처리 중 오류 발생 ({place_name}): {str(e)}")
                    # 에러 발생 시 기본 정보로 PlaceInfo 추가
                    place_details.append(PlaceInfo(
                        name=place_name,
                        source_url=url,
                        description=self._extract_place_description(summary, place_name),
                        google_info={}
                    ))
                    continue

            return place_details
        except Exception as e:
            raise Exception(f"URL 처리 중 오류 발생: {str(e)}")

    def _extract_place_description(self, summary: str, place_name: str) -> str:
        """요약 텍스트에서 특정 장소에 대한 설명을 추출"""
        try:
            lines = summary.split('\n')
            description = ""
            
            for i, line in enumerate(lines):
                if place_name in line:
                    # 현재 줄과 다음 몇 줄을 포함하여 설명 추출
                    description = ' '.join(lines[i:i+3])
                    break
            
            return description.strip() or "장소 설명을 찾을 수 없습니다."
            
        except Exception as e:
            print(f"장소 설명 추출 중 오류 발생: {str(e)}")
            return "장소 설명을 찾을 수 없습니다."

    def _format_final_result(self, content_infos: List[ContentInfo], place_details: List[PlaceInfo], processing_time: float, urls: List[str]) -> str:
        """최종 결과 문자열을 포맷팅하는 메서드"""
        
        # 1. 기본 정보 헤더
        final_result = f"""
=== 여행 정보 요약 ===
처리 시간: {processing_time:.2f}초

분석한 영상:
{'='*50}"""
        
        # 2. 비디오 정보
        if content_infos:
            for info in content_infos:
                final_result += f"""
제목: {info.title}
채널: {info.author}
URL: {info.url}"""
        else:
            final_result += f"\nURL: {chr(10).join(urls)}"
        
        final_result += f"\n{'='*50}\n\n=== 장소별 상세 정보 ===\n\n"

        # 장소 필터링 조건 수정
        def is_valid_place(p):
            # 1. 일본 주소 확인
            if not p.formatted_address or not any(keyword in p.formatted_address for keyword in ["日本", "Japan", "일본"]):
                return False
            
            # 2. 사진 URL 존재 확인 (null이 아님)
            if not p.photos or len(p.photos) == 0:
                return False
            
            # 3. 위도/경도 필수 확인
            if not p.geometry or p.geometry.latitude is None or p.geometry.longitude is None:
                return False
            
            return True
        
        valid_places = [p for p in place_details if is_valid_place(p)]
        
        for idx, place in enumerate(valid_places, 1):
            final_result += f"{idx}. {place.name}\n"
            final_result += "=" * 50 + "\n\n"
            final_result += f"주소: {place.formatted_address}\n"
            if place.geometry and place.geometry.latitude is not None and place.geometry.longitude is not None:
                final_result += f"위도: {place.geometry.latitude}\n"
                final_result += f"경도: {place.geometry.longitude}\n"
            if place.photos and len(place.photos) > 0:
                final_result += f"사진 URL: {place.photos[0].url}\n"
            final_result += "=" * 50 + "\n\n"

        return final_result
    
    def search_content(self, query: str) -> List[Dict]:
        """벡터 DB에서 콘텐츠 검색"""
        try:
            results = self.repository.query_vectordb(query)
            filtered_results = []

            for doc in results:
                if isinstance(doc, Document) and hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                    filtered_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    })
                else:
                    print(f"⚠️ 잘못된 데이터 타입 감지: {type(doc)} - {doc}")

            return filtered_results
        except Exception as e:
            raise Exception(f"검색 중 오류 발생: {str(e)}")

    @staticmethod
    def _get_youtube_transcript(video_id: str) -> str:
        """YouTube 영상의 자막을 가져옴"""
        try:
            print(f"\n=== 자막 추출 시작: {video_id} ===")
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # 1. 한국어 자막 시도
            try:
                transcript = transcripts.find_transcript(['ko'])
                transcript_list = transcript.fetch()
                print("✅ 한국어 자막 찾음")
                
                # 자막 텍스트 구성
                transcript_text = []
                for entry in transcript_list:
                    timestamp = YouTubeService._format_timestamp(entry['start'])
                    text = entry['text'].strip()
                    if text:  # 빈 텍스트가 아닌 경우만 추가
                        transcript_text.append(f"[{timestamp}] {text}")
                
                result = "\n".join(transcript_text)
                print(f"📝 추출된 한국어 자막 길이: {len(result)} 자")
                print("=== 자막 일부 ===")
                print(result[:500])  # 처음 500자만 출력
                return result
                
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                print(f"⚠️ 한국어 자막 없음: {str(e)}")

            # 2. 자동 생성된 한국어 자막 시도
            try:
                transcript = transcripts.find_generated_transcript(['ko'])
                transcript_list = transcript.fetch()
                print("✅ 자동 생성된 한국어 자막 찾음")
                
                transcript_text = []
                for entry in transcript_list:
                    timestamp = YouTubeService._format_timestamp(entry['start'])
                    text = entry['text'].strip()
                    if text:
                        transcript_text.append(f"[{timestamp}] {text}")
                
                result = "\n".join(transcript_text)
                print(f"📝 추출된 자동 생성 한국어 자막 길이: {len(result)} 자")
                print("=== 자막 일부 ===")
                print(result[:500])
                return result
                
            except Exception as e:
                print(f"⚠️ 자동 생성된 한국어 자막 없음: {str(e)}")

            # 3. 영어 자막을 한국어로 번역
            try:
                transcript = transcripts.find_transcript(['en'])
                transcript_list = transcript.fetch()
                print("✅ 영어 자막 찾음")
                
                # 영어 자막 텍스트 구성
                en_texts = []
                timestamps = []
                for entry in transcript_list:
                    timestamp = YouTubeService._format_timestamp(entry['start'])
                    text = entry['text'].strip()
                    if text:
                        en_texts.append(text)
                        timestamps.append(timestamp)
                
                # OpenAI를 사용하여 한국어로 번역
                combined_text = " ".join(en_texts)
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional translator specializing in Korean travel content."},
                        {"role": "user", "content": f"Translate the following English text to Korean, maintaining the travel-related context:\n\n{combined_text}"}
                    ],
                    temperature=0.3
                )
                
                translated_text = response.choices[0].message.content
                
                # 번역된 텍스트를 타임스탬프와 결합
                result = f"[번역된 자막]\n"
                sentences = translated_text.split('. ')
                for i, (timestamp, sentence) in enumerate(zip(timestamps, sentences)):
                    if sentence.strip():
                        result += f"[{timestamp}] {sentence.strip()}\n"
                
                print(f"📝 번역된 자막 길이: {len(result)} 자")
                print("=== 번역된 자막 일부 ===")
                print(result[:500])
                return result
                
            except Exception as e:
                print(f"⚠️ 영어 자막 변환 실패: {str(e)}")

            raise ValueError("사용 가능한 자막을 찾을 수 없습니다.")

        except Exception as e:
            print(f"❌ 자막 추출 실패: {str(e)}")
            raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """초를 시:분:초 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_video_info(self, video_id: str) -> VideoInfo:
        """YouTube 비디오 정보를 가져옴"""
        try:
            import requests
            
            # noembed API를 사용하여 비디오 정보 가져오기
            api_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
            response = requests.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                return VideoInfo(
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    title=data.get('title'),
                    channel=data.get('author_name')
                )
            else:
                print(f"[get_video_info] API 응답 상태 코드: {response.status_code}")
                return VideoInfo(
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    title="제목을 가져올 수 없음",
                    channel="채널 정보를 가져올 수 없음"
                )
            
        except Exception as e:
            print(f"비디오 정보를 가져오는데 실패했습니다: {e}")
            return VideoInfo(
                url=f"https://www.youtube.com/watch?v={video_id}",
                title="제목을 가져올 수 없음",
                channel="채널 정보를 가져올 수 없음"
            )

    def _get_blog_info(self, url: str) -> Dict[str, str]:
        """네이버 블로그 정보를 가져옴"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 블로그 제목 추출 시도
            title = None
            title_tag = soup.find('meta', property='og:title')
            if title_tag:
                title = title_tag.get('content')
            
            if not title:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.text
            
            # 작성자 정보 추출 시도
            author = None
            author_tag = soup.find('meta', property='og:article:author')
            if author_tag:
                author = author_tag.get('content')
            
            if not author:
                # 블로그 URL에서 작성자 ID 추출
                try:
                    blog_id = url.split('blog.naver.com/')[1].split('/')[0]
                    author = f"네이버 블로그 | {blog_id}"
                except:
                    author = "네이버 블로그 작성자"
            
            return {
                'title': title or "제목을 가져올 수 없음",
                'author': author or "작성자 정보를 가져올 수 없음"
            }
            
        except Exception as e:
            print(f"블로그 정보를 가져오는데 실패했습니다: {e}")
            return {
                'title': "제목을 가져올 수 없음",
                'author': "작성자 정보를 가져올 수 없음"
            }

    def generate_final_summary(self, content_infos: List[ContentInfo], processing_time: float, place_details: List[PlaceInfo]) -> Dict[str, str]:
        """최종 요약을 생성"""
        summaries = {}
        
        for content in content_infos:
            summary = f"=== 여행 정보 요약 ===\n"
            summary += f"처리 시간: {processing_time:.2f}초\n\n"
            
            # 분석한 콘텐츠 정보
            summary += "분석한 콘텐츠:\n"
            summary += "=" * 50 + "\n"
            for idx, info in enumerate(content_infos, 1):
                summary += f"{idx}. {info.platform.value.upper()}\n"
                summary += f"제목: {info.title}\n"
                summary += f"작성자: {info.author}\n"
                summary += f"URL: {info.url}\n\n"
            
            summary += "=" * 50 + "\n\n"
            
            # 장소별 상세 정보
            summary += "=== 장소별 상세 정보 ===\n\n"
            
            # Filtering valid places according to the criteria
            def is_valid_place(p):
                # 1. 일본 주소 확인
                if not p.formatted_address or not any(keyword in p.formatted_address for keyword in ["日本", "Japan", "일본"]):
                    return False
                
                # 2. 사진 URL 존재 확인 (null이 아님)
                if not p.photos or len(p.photos) == 0:
                    return False
                
                # 3. 위도/경도 필수 확인
                if not p.geometry or p.geometry.latitude is None or p.geometry.longitude is None:
                    return False
                
                return True
            
            valid_places = [p for p in place_details if is_valid_place(p)]
            
            for idx, place in enumerate(valid_places, 1):
                summary += f"{idx}. {place.name}\n"
                summary += "=" * 50 + "\n\n"
                summary += f"주소: {place.formatted_address}\n"
                if place.geometry and place.geometry.latitude is not None and place.geometry.longitude is not None:
                    summary += f"위도: {place.geometry.latitude}\n"
                    summary += f"경도: {place.geometry.longitude}\n"
                if place.photos and len(place.photos) > 0:
                    summary += f"사진 URL: {place.photos[0].url}\n"
                summary += "=" * 50 + "\n\n"
            
            # 유효한 장소가 없는 경우 메시지 추가
            if not valid_places:
                summary += "※ 유효한 장소 정보가 없습니다. (사진, 좌표, 일본 주소 중 하나 이상 누락)\n"
            
            summaries[content.url] = summary
        
        return summaries

    def _get_place_description_from_openai(self, place_name: str, place_type: str) -> str:
        """OpenAI를 사용하여 장소에 대한 일반적인 설명 생성"""
        try:
            prompt = f"""다음 장소에 대한 정확하고 간결한 설명을 제공하세요.  
설명은 10자로 제한되며, 핵심 정보만 포함해야 합니다.
장소: {place_name}
타입: {place_type}
반드시 짧고 명확한 한 문장으로 작성하세요."""

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 일본 전문 여행 가이드입니다. 장소에 대한 객관적이고 정확한 정보를 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=30
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"장소 설명 생성 중 오류 발생: {str(e)}")
            return None

    def process_place_info(self, place_name: str, timestamp: str, description: str) -> PlaceInfo:
        """PlaceService의 process_place_info를 호출"""
        return self.place_service.process_place_info(place_name, timestamp, description)

class YouTubeSubtitleService:
    """YouTube 자막 및 비디오 정보 처리 서비스"""
    
    @staticmethod
    def get_video_info(video_url: str) -> Tuple[str, str]:
        try:
            video_id = video_url.split("v=")[-1].split("&")[0]
            api_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                title = data.get('title')
                author_name = data.get('author_name')
                print(f"[get_video_info] 제목: {title}, 채널: {author_name}")
                return title, author_name
            print(f"[get_video_info] API 응답 상태 코드: {response.status_code}")
            return None, None
        except Exception as e:
            print(f"영상 정보를 가져오는데 실패했습니다: {e}")
            return None, None

    @staticmethod
    def process_link(url: str) -> str:
        link_type = YouTubeSubtitleService._detect_link_type(url)
        print(f"[process_link] 링크 유형 감지: {link_type}")
        
        if link_type == "youtube":
            text = YouTubeSubtitleService._get_youtube_transcript(url)
        elif link_type == "text_file":
            text = YouTubeSubtitleService._get_text_from_file(url)
        else:
            text = YouTubeSubtitleService._get_text_from_webpage(url)
        
        print(f"[process_link] 추출된 텍스트 길이: {len(text)}")
        return text

    @staticmethod
    def _detect_link_type(url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif url.endswith(".txt"):
            return "text_file"
        elif url.startswith("http"):
            return "webpage"
        else:
            return "unknown"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _get_youtube_transcript(video_url: str) -> str:
        video_id = video_url.split("v=")[-1].split("&")[0]
        print(f"[get_youtube_transcript] 비디오 ID: {video_id}")
        
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # 1. 먼저 한국어 자막 시도
            try:
                transcript = transcripts.find_transcript(['ko'])
                transcript_text = "\n".join([f"[{YouTubeSubtitleService._format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript.fetch()])
                print(f"[get_youtube_transcript] 한국어 자막 추출 완료. 길이: {len(transcript_text)}")
                return transcript_text
            except (TranscriptsDisabled, NoTranscriptFound):
                print("[get_youtube_transcript] 한국어 자막 없음.")

            # 2. 영어 자막 시도
            try:
                transcript = transcripts.find_transcript(['en'])
                transcript_text = "\n".join([f"[{YouTubeSubtitleService._format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript.fetch()])
                print(f"[get_youtube_transcript] 영어 자막 추출 완료. 길이: {len(transcript_text)}")
                return transcript_text
            except (TranscriptsDisabled, NoTranscriptFound):
                print("[get_youtube_transcript] 영어 자막 없음.")

            # 3. 사용 가능한 첫 번째 자막 시도
            try:
                transcript = transcripts.find_generated_transcript()
                transcript_text = "\n".join([f"[{YouTubeSubtitleService._format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript.fetch()])
                print(f"[get_youtube_transcript] 생성된 자막 추출 완료. 길이: {len(transcript_text)}")
                return transcript_text
            except Exception as e:
                print(f"[get_youtube_transcript] 생성된 자막 추출 실패: {e}")

            raise ValueError("사용 가능한 자막을 찾을 수 없습니다.")

        except Exception as e:
            raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    @staticmethod
    def _get_text_from_file(url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text.strip()
            print(f"[get_text_from_file] 텍스트 파일 추출 완료. 길이: {len(text)}")
            return text
        except Exception as e:
            raise ValueError(f"텍스트 파일 내용을 가져오는데 오류가 발생했습니다: {e}")

    @staticmethod
    def _get_text_from_webpage(url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n").strip()
            text = text[:10000]  # 길이 제한 10000자
            print(f"[get_text_from_webpage] 웹페이지 텍스트 추출 완료. 길이: {len(text)}")
            return text
        except Exception as e:
            raise ValueError(f"웹페이지 내용을 가져오는데 오류가 발생했습니다: {e}")

class TextProcessingService:
    """텍스트 처리 서비스"""
    
    def split_text(self, text: str, max_chunk_size: int = 2048) -> List[str]:
        """텍스트를 청크로 분할"""
        try:
            words = text.split()
            total_words = len(words)
            num_chunks = ceil(total_words / (max_chunk_size // 5))
            chunks = []
            
            for i in range(num_chunks):
                start = i * (max_chunk_size // 5)
                end = start + (max_chunk_size // 5)
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
            
            print(f"[split_text] 총 단어 수: {total_words}, 청크 수: {num_chunks}")
            return chunks
            
        except Exception as e:
            print(f"[split_text] 오류 발생: {str(e)}")
            raise ValueError(f"텍스트 분할 중 오류 발생: {str(e)}")

    def _generate_prompt(self, text: str) -> str:
        """GPT 프롬프트 생성"""
        return f"""다음 텍스트를 분석하여 여행 정보를 요약해주세요. 
특히 다음 사항에 중점을 두어 요약해주세요:

1. 방문한 장소들 (위치 정보 포함)
2. 각 장소의 특징과 설명
3. 추천 사항이나 주의 사항
4. 시간대별 방문 정보 (있는 경우)

텍스트:
{text}

다음 형식으로 응답해주세요:

방문한 장소: [장소명] ([지역명])
- 설명: [장소에 대한 설명]
- 추천 사항: [있는 경우]
- 주의 사항: [있는 경우]
- 방문 시간: [언급된 경우]
"""

    def summarize_text(self, transcript_chunks: List[str], model: str = "gpt-4o-mini") -> str:
        """텍스트 청크들을 요약"""
        try:
            summaries = []
            for idx, chunk in enumerate(transcript_chunks):
                print(f"[summarize_text] 청크 {idx+1}/{len(transcript_chunks)} 처리 중...")
                
                prompt = self._generate_prompt(chunk)
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "당신은 여행 전문가로서 여행 컨텐츠를 분석하고 유용한 정보를 추출하는 AI입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                
                summary = response.choices[0].message.content
                summaries.append(summary)
                print(f"[summarize_text] 청크 {idx+1} 요약 완료")
            
            return "\n\n".join(summaries)
            
        except Exception as e:
            print(f"[summarize_text] 오류 발생: {str(e)}")
            raise ValueError(f"텍스트 요약 중 오류 발생: {str(e)}")

class PlaceService:
    """장소 정보 처리 서비스"""
    
    def __init__(self):
        self.video_url = None

    def set_video_url(self, url: str):
        self.video_url = url

    def extract_place_names(self, summary: str) -> List[str]:
        """요약 텍스트에서 장소 이름을 추출"""
        place_names = set()  # 중복 방지를 위해 set 사용
        
        # 모든 청크의 요약에서 장소 추출
        chunks = summary.split("방문한 장소:")
        for chunk in chunks[1:]:  # 첫 번째는 건너뛰기
            try:
                place_name = chunk.split("(")[0].strip()
                if place_name:
                    place_names.add(place_name)
                    print(f"장소 추출: {place_name}")
            except Exception as e:
                print(f"장소 추출 오류: {e}")
                continue
        
        result = list(place_names)
        print(f"총 추출된 장소 목록: {result}")
        return result

    @staticmethod
    def search_place_details(place_name: str, area: str = None) -> Dict[str, Any]:
        """Google Places API를 사용하여 장소 정보를 검색"""
        try:
            gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))
            
            # 지역명이 있으면 장소명과 함께 검색, 없으면 '일본'을 추가
            search_query = f"{place_name} {area if area else '일본'}"
            print(f"[search_place_details] 검색어: {search_query}")
            
            # 장소 검색
            places_result = gmaps.places(search_query)
            
            if not places_result['results']:
                print(f"[search_place_details] 장소를 찾을 수 없음: {search_query}")
                return None
                
            place = places_result['results'][0]
            place_id = place['place_id']
            
            # 상세 정보 검색
            details_result = gmaps.place(place_id, language='ko')
            if not details_result.get('result'):
                return None
                
            details = details_result['result']
            
            # 리뷰 정보 가져오기
            reviews = details.get('reviews', [])
            best_review = reviews[0]['text'] if reviews else None
            
            # 결과 딕셔너리 생성
            return {
                'name': details.get('name', ''),
                'formatted_address': details.get('formatted_address', ''),
                'rating': details.get('rating'),
                'formatted_phone_number': details.get('formatted_phone_number', ''),
                'website': details.get('website', ''),
                'price_level': details.get('price_level'),
                'opening_hours': details.get('opening_hours', {}).get('weekday_text', []),
                'photos': details.get('photos', []),
                'best_review': bestreview
            }
            
        except Exception as e:
            print(f"[search_place_details] 장소 정보 검색 중 오류 발생 ({search_query}): {str(e)}")
            return None

    @staticmethod
    def get_place_photo_google(place_name: str) -> str:
        """Google Places API를 사용하여 장소 사진 URL을 가져옴"""
        try:
            gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))
            places_result = gmaps.places(place_name)
            
            if not places_result['results']:
                print(f"[get_place_photo_google] 사진을 찾을 수 없음: {place_name}")
                return None
                
            place = places_result['results'][0]
            if not place.get('photos'):
                return None
                
            photo_reference = place['photos'][0]['photo_reference']
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={os.getenv('GOOGLE_PLACES_API_KEY')}"
            
            print(f"[get_place_photo_google] 사진 URL 생성 완료: {photo_url}")
            return photo_url
            
        except Exception as e:
            print(f"[get_place_photo_google] 사진 URL 생성 중 오류 발생: {str(e)}")
            return None

    def process_place_info(self, place_name: str, timestamp: str, description: str) -> PlaceInfo:
        """장소 정보를 처리하고 PlaceInfo 객체를 반환"""
        try:
            # Google Places API로 장소 정보 가져오기
            gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))
            places_result = gmaps.places(place_name)
            
            if not places_result['results']:
                return None
            
            google_place_info = places_result['results'][0]
            
            # 사진 URL 가져오기
            photo_url = self.get_place_photo_google(place_name)
            
            # 장소 타입 확인
            place_type = google_place_info.get('types', ['unknown'])[0]
            
            # OpenAI로 공식 설명 생성
            official_description = self._get_place_description_from_openai(place_name, place_type)
            
            # 영업시간 포맷팅
            opening_hours = None
            if google_place_info.get('opening_hours'):
                opening_hours = google_place_info['opening_hours'].get('weekday_text')

            # PlaceInfo 객체 생성
            place_info = PlaceInfo(
                name=place_name,
                source_url=self.video_url,
                timestamp=timestamp,
                description=description,
                official_description=official_description,
                formatted_address=google_place_info.get('formatted_address'),
                coordinates={
                    'lat': google_place_info['geometry']['location']['lat'],
                    'lng': google_place_info['geometry']['location']['lng']
                } if 'geometry' in google_place_info else None,
                rating=google_place_info.get('rating'),
                phone=google_place_info.get('formatted_phone_number'),
                website=google_place_info.get('website'),
                price_level=google_place_info.get('price_level'),
                opening_hours=opening_hours,
                photos=[PlacePhoto(url=photo_url)] if photo_url else None,
                best_review=google_place_info.get('reviews', [{}])[0].get('text') if google_place_info.get('reviews') else None,
                google_info=google_place_info,
                types=google_place_info.get('types')
            )
            
            return place_info
            
        except Exception as e:
            print(f"장소 정보 처리 중 오류 발생: {str(e)}")
            return None
