import os
import datetime
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from models.youtube_schemas import VideoInfo, PlaceInfo, ContentInfo
import aiofiles
import asyncio

class YouTubeRepository:
    def __init__(self, summary_dir: str = "./summaries"):
        """
        YouTubeRepository 초기화
        Args:
            summary_dir: 요약본을 저장할 디렉토리 경로 (기본값: "./summaries")
        """
        self.summary_dir = summary_dir
        os.makedirs(summary_dir, exist_ok=True)
        
        try:
            # OpenAI API 키 확인
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

            # LLM 모델 초기화
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )
            
            # Embeddings 초기화
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # ChromaDB 초기화
            self.vectordb = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings,
                collection_name="summaries"
            )
            
            print("✅ 벡터 DB 초기화 완료")
            
            # 변경 후 코드:
            self.MAX_TOTAL_SIZE = 10485760  # 10MB
            
        except Exception as e:
            print(f"⚠️ 벡터 DB 초기화 실패: {str(e)}")
            self.vectordb = None

    async def save_to_vectordb(self, final_summaries: Dict[str, str], content_infos: List[ContentInfo], place_details: List[PlaceInfo]) -> None:
        """벡터 DB에 최종 요약 저장"""
        try:
            documents = []
            for content in content_infos:
                summary = final_summaries.get(content.url, "요약 정보 없음")
                metadata = {
                    "url": content.url,
                    "title": content.title,
                    "author": content.author,
                    "platform": content.platform.value,
                    "type": "summary"
                }
                # None 값과 복잡한 객체 필터링
                filtered_metadata = filter_complex_metadata(metadata)
                documents.append(Document(page_content=summary, metadata=filtered_metadata))
            
            # 장소 정보도 저장
            for place in place_details:
                if place and place.description:  # None이 아닌 경우만 처리
                    metadata = {
                        "name": place.name,
                        "type": place.type if place.type else "unknown",
                        "address": place.formatted_address if place.formatted_address else "",
                        "rating": float(place.rating) if place.rating else 0.0,
                        "source_url": place.source_url if place.source_url else ""
                    }
                    # None 값과 복잡한 객체 필터링
                    filtered_metadata = filter_complex_metadata(metadata)
                    documents.append(Document(
                        page_content=place.description,
                        metadata=filtered_metadata
                    ))
            
            if documents:  # 문서가 있는 경우만 저장
                # 비동기로 문서 추가
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.vectordb.add_documents(documents)
                )
                print(f"✅ 벡터 DB 저장 완료: {len(documents)}개 문서")
                print("벡터 DB에 문서 저장이 성공적으로 완료되었습니다!")
        except Exception as e:
            print(f"벡터 DB 저장 중 오류 발생: {str(e)}")
            raise

    async def save_final_summary(self, final_summaries: Dict[str, str], content_infos: List[ContentInfo]) -> List[str]:
        """URL별로 최종 요약을 파일로 저장하고 파일 경로 리스트 반환"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        
        for content in content_infos:
            try:
                file_name = f"final_summary_{content.platform.value}_{timestamp}.txt"
                file_path = os.path.join(self.summary_dir, file_name)
                
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    summary = final_summaries.get(content.url, "요약 정보 없음")
                    await f.write(summary)
                
                saved_paths.append(file_path)
                print(f"✅ 요약본 저장 완료: {file_path}")
                
            except Exception as e:
                print(f"파일 저장 중 오류 발생: {str(e)}")
                continue
        
        return saved_paths

    def query_vectordb(self, query: str, k: int = 3) -> List[Document]:
        """벡터 DB에서 검색"""
        try:
            results = self.vectordb.similarity_search(query, k=k)
            return [doc for doc in results if isinstance(doc, Document)]
        except Exception as e:
            print(f"벡터 DB 검색 중 오류 발생: {str(e)}")
            return []

    def save_chunks(self, chunks: List[str]) -> None:
        """텍스트 청크들을 파일로 저장"""
        for idx, chunk in enumerate(chunks, 1):
            file_path = os.path.join(self.chunks_dir, f"chunk_{idx}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)

    def save_place_details(self, place_details: List[PlaceInfo]) -> List[Document]:
        """장소 정보를 벡터 DB에 저장"""
        documents = []
        
        for place in place_details:
            # 위치 정보 추출 및 구조화
            geometry = place.google_info.get("geometry", {})
            location = geometry.get("location", {})
            
            # 설명에서 유의사항과 추천사항 추출
            precautions = []
            recommendations = []
            if place.description:
                if "유의 사항:" in place.description:
                    precautions_part = place.description.split("유의 사항:")[-1].split("-")[0].strip()
                    precautions.append(precautions_part)
                if "추천 사항:" in place.description:
                    recommendations_part = place.description.split("추천 사항:")[-1].split("-")[0].strip()
                    recommendations.append(recommendations_part)
            
            coordinates = None
            if location and isinstance(location, dict):
                coordinates = {
                    "lat": location.get("lat"),
                    "lng": location.get("lng")
                }
            
            metadata = {
                # 기본 정보
                "name": place.name,
                "source_url": place.source_url,
                "types": place.types,
                "main_type": place.types[0] if place.types else "unknown",
                
                # 위치 정보
                "address": place.formatted_address,
                "coordinates": coordinates,
                
                # 장소 설명
                "creator_review": place.description,  # 크리에이터의 원본 리뷰
                "official_description": place.official_description,
                "precautions": precautions,  # 유의사항 목록
                "recommendations": recommendations,  # 추천사항 목록
                
                # 시설 정보
                "rating": place.rating,
                "phone": place.phone,
                "website": place.website,
                "price_level": place.price_level,
                "opening_hours": place.opening_hours,
                
                # 미디어/리뷰
                "photos": [photo.url for photo in place.photos] if place.photos else [],
                "best_review": place.best_review
            }
            
            # 검색 가능한 텍스트 생성
            searchable_text = f"""
{place.name}
{place.official_description or ''}
{place.formatted_address or ''}
"""
            
            doc = Document(
                page_content=searchable_text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Chroma DB에 저장
        self.vectordb.add_documents(documents)
        
        return documents 

    def search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """벡터 DB에서 콘텐츠 검색"""
        results = self.vectordb.similarity_search_with_score(query, k=limit)
        
        search_results = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            }
            search_results.append(result)
        
        return search_results 

    def _create_limited_result(self, summaries: Dict[str, str], content_infos: List[ContentInfo], place_details: List[PlaceInfo], processing_time: float) -> Dict:
        """
        Create a limited result dictionary from the summaries, content infos, and place details.
        Args:
            summaries: A dictionary of summaries keyed by URL.
            content_infos: A list of ContentInfo objects.
            place_details: A list of PlaceInfo objects.
            processing_time: The processing time in seconds.
        Returns:
            A dictionary containing the summaries, content infos, processing time, and place details.
        """
        limited_place_details = place_details[:5]  # Assuming the first 5 place details are returned
        return {
            "summary": summaries,
            "content_infos": [info.dict() for info in content_infos],
            "processing_time_seconds": processing_time,
            "place_details": [place.dict() for place in limited_place_details]
        } 