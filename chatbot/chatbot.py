from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_chroma import Chroma
import os
import openai
from dotenv import load_dotenv
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from config import path_config


# .env 파일 로드
load_dotenv()

# OpenAI API Key를 환경 변수에서 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# Chroma DB 경로
CHROMA_DB_PATH = path_config.CHROMA_DB_PATH

class ChatMessage(BaseModel):
    message: str
    chat_history: List[Tuple[str, str]] = []

class ChatBot:
    def __init__(self):
        try:
            print(f"Trying to connect to ChromaDB at: {CHROMA_DB_PATH}")
            
            self.embeddings = OpenAIEmbeddings()
            self.vectordb = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
            )
            
            # DB 내용 확인
            print("\n=== ChromaDB 상태 ===")
            results = self.vectordb._collection.get()
            if results and results['metadatas']:
                print(f"Total documents: {len(results['ids'])}")
                print(f"Sample metadata: {results['metadatas'][0]}")
                print(f"Sample document: {results['documents'][0][:200]}...")
            
            # 필터 없이 사용
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 1}  # 검색 결과 1개만 반환         #63개발자_모드
            )
            
            # QA 체인 설정
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0.0, model_name='gpt-4o-mini'),
                retriever=self.retriever,
                chain_type="stuff",
                return_source_documents=True
            )
            
            # 기존 대화형 체인도 유지
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.0, model_name='gpt-4o-mini'),
                retriever=self.vectordb.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            # OpenAI API 설정
            openai.api_key = openai_api_key
            
        except Exception as e:
            print(f"초기화 중 오류가 발생했습니다: {str(e)}")
    
    async def get_openai_response(self, message: str) -> str:
        """OpenAI API를 직접 호출하여 응답을 받습니다."""
        try:
            system_prompt = """당신은 일본 여행 전문가입니다. 다음 규칙을 따라 답변해주세요:
            - 일본의 문화, 역사, 관광지, 음식, 교통, 숙박 등에 대한 깊이 있는 지식을 가지고 있습니다.
            - 사용자의 질문에 대해 친절하고 상세하게 답변합니다. 필요한 경우 추가 정보를 제공하여 사용자가 이해할 수 있도록 돕습니다.
            - 모든 답변은 존댓말로 작성합니다. 사용자가 편안하게 느낄 수 있도록 배려합니다.
            - 일본 여행에 대한 전문적인 조언을 제공합니다. 예를 들어, 특정 지역의 추천 관광지, 맛집, 교통편, 여행 팁 등을 구체적으로 안내합니다.
            - 사용자의 질문이나 요구에 따라 맞춤형 응답을 제공합니다. 예를 들어, 사용자가 특정 도시를 언급하면 그 도시에 대한 정보를 집중적으로 제공합니다.
            - 일본의 문화적 배경이나 전통에 대한 설명을 포함하여, 사용자가 일본을 더 잘 이해할 수 있도록 돕습니다.
            - 여행 중 유의해야 할 안전 사항이나 문화적 예절에 대한 정보를 제공합니다. 예를 들어, 공공장소에서의 행동 규범이나 교통 규칙 등을 안내합니다.
            - 일본의 최신 여행 정보, 이벤트, 축제 등을 반영하여, 사용자가 현재 상황에 맞는 정보를 얻을 수 있도록 합니다.
            - 사용자가 추가 질문을 할 수 있도록 유도합니다. 예를 들어, '더 궁금한 점이 있으신가요?'와 같은 문구를 사용합니다.
            - 일본의 다양한 여행 일정을 추천합니다. 예를 들어, 3일, 5일, 7일 여행 코스를 제안하고 각 일정에 맞는 주요 관광지를 안내합니다.
            - 여행 비용에 대한 정보를 제공하고, 예산에 맞는 여행 계획을 세울 수 있도록 돕습니다.
            - 일본어, 영어, 한국어 등 다양한 언어로 정보를 제공할 수 있도록 합니다. 사용자가 원하는 언어로 질문할 수 있도록 유도합니다.
            - 일반적으로 한국어로 답변합니다.
            - "그에 대한 정보는 제공되지 않았습니다. 일본 여행 전에 준비해야 할 것에 대한 구체적인 내용은 알 수 없습니다."이렇게 답변하지 말고 openapi에 물어 답변을 이끌어내시오.
            - "제공된 정보에는 오사카에 대한 숙소 관련 내용이 없습니다. 따라서 오사카 숙소에 대한 정보를 제공할 수 없습니다."이렇게 답변하지 말고 openapi에 물어 답변을 이끌어내시오.
            - "죄송하지만, 제공된 정보에는 오사카에 대한 숙소 정보가 포함되어 있지 않습니다. 다른 질문이 있으시면 도와드리겠습니다."이렇게 답변하지 말고 openapi에 물어 답변을 이끌어내시오.
            - "죄송하지만, " 이렇게 시작하는 답변은 하지 않습니다.
            - 모든 질문에 대해 구체적이고 실용적인 정보를 제공합니다.
            - 정보가 부족한 경우에도 일반적인 조언이나 대안을 제시합니다.
            """

            # 새로운 OpenAI API 호출 방식 사용
            client = openai.AsyncOpenAI(api_key=openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            if not response or not response.choices:
                raise ValueError("API 응답이 유효하지 않습니다")
            
            return response.choices[0].message.content.strip()
        except openai.error.APIError as e:
            print(f"OpenAI API 오류: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API 오류: {str(e)}")
        except Exception as e:
            print(f"예상치 못한 오류: {str(e)}")
            raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다")
    
    async def get_db_info(self, query_type: str = None, search_term: str = None):
        """ChromaDB에서 정보 조회"""
        try:
            print("\n=== ChromaDB 조회 시작 ===")
            results = self.vectordb._collection.get()
            
            if not results or not results['metadatas']:
                return "데이터베이스에 저장된 정보가 없습니다."

            # 개발자 모드 - URL 기준 전체 데이터 조회
            if query_type == "dev_all":
                all_data = []
                for idx, meta in enumerate(results['metadatas']):
                    data = {
                        'id': results['ids'][idx],
                        'url': meta.get('url', 'unknown'),
                        'platform': meta.get('platform', 'unknown'),
                        'author': meta.get('author', 'unknown'),
                        'type': meta.get('type', 'unknown'),
                        'title': meta.get('title', 'unknown'),
                        'content': results['documents'][idx]  # 전체 내용
                    }
                    all_data.append(data)

                # 터미널에 출력
                print("\n=== 개발자 모드: 전체 데이터 조회 ===")
                for data in all_data:
                    print("\n" + "="*50)
                    print(f"ID: {data['id']}")
                    print(f"URL: {data['url']}")
                    print(f"Platform: {data['platform']}")
                    print(f"Author: {data['author']}")
                    print(f"Type: {data['type']}")
                    print(f"Title: {data['title']}")
                    print("-"*30 + " Content " + "-"*30)
                    print(data['content'][:500] + "...")  # 내용은 500자까지만 출력
                    print("="*50)

                # 사용자에게는 요약 정보만 반환
                return f"""
=== 데이터베이스 전체 정보 ===
총 문서 수: {len(all_data)}
플랫폼별 문서 수:
{self._count_by_field(all_data, 'platform')}
"""

            # 전체 메타데이터 정보 수집
            all_metadata = []
            for idx, meta in enumerate(results['metadatas']):
                platform = meta.get('platform', 'unknown')
                doc_info = {
                    'id': results['ids'][idx],
                    'platform': platform,
                    'author': meta.get('author', 'unknown'),
                    'channel': meta.get('channel', 'unknown'),  # 채널 정보 추가
                    'title': meta.get('title', 'unknown'),
                    'type': meta.get('type', 'unknown'),
                    'url': meta.get('url', 'unknown'),
                    'content': results['documents'][idx][:200] + "..." if results['documents'][idx] else "내용 없음"
                }
                all_metadata.append(doc_info)

            if query_type == "youtuber" or query_type == "channel":
                # 유튜브 채널 정보만 필터링
                youtube_channels = []
                for meta in all_metadata:
                    if meta['platform'].lower() == 'youtube':
                        channel_info = {
                            'channel': meta.get('channel', 'unknown'),
                            'author': meta.get('author', 'unknown'),
                            'url': meta.get('url', 'unknown')
                        }
                        if channel_info not in youtube_channels:
                            youtube_channels.append(channel_info)
                
                if not youtube_channels:
                    return "저장된 유튜브 채널 정보가 없습니다."
                    
                result = "=== 저장된 유튜브 채널 목록 ===\n\n"
                for idx, info in enumerate(youtube_channels, 1):
                    result += f"{idx}. 채널명: {info['channel']}\n"
                    result += f"   작성자: {info['author']}\n"
                    result += f"   URL: {info['url']}\n"
                    result += "=" * 30 + "\n"
                return jsonable_encoder(result)


            
            elif query_type == "platform":
                platforms = set(meta.get('platform', 'unknown') for meta in results['metadatas'])
                return f"저장된 플랫폼: {', '.join(platforms)}"
            
            elif query_type == "author":
                authors = set(meta.get('author', 'unknown') for meta in results['metadatas'])
                return f"저장된 작성자: {', '.join(authors)}"
            
            elif query_type == "title":
                titles = set(meta.get('title', 'unknown') for meta in results['metadatas'])
                return f"저장된 제목: {', '.join(titles)}"
            
            elif query_type == "type":
                types = set(meta.get('type', 'unknown') for meta in results['metadatas'])
                return f"저장된 타입: {', '.join(types)}"
            
            elif query_type == "url":
                urls = set(meta.get('url', 'unknown') for meta in results['metadatas'])
                return f"저장된 URL: {', '.join(urls)}"
            
            elif query_type == "search" and search_term:
                # 특정 키워드로 검색
                matched_docs = []
                for doc in all_metadata:
                    if any(search_term.lower() in str(value).lower() for value in doc.values()):
                        matched_docs.append(doc)
                
                if not matched_docs:
                    return f"'{search_term}' 검색 결과가 없습니다."
                    
                result = "=== 검색 결과 ===\n"
                for doc in matched_docs:
                    result += f"\n제목: {doc['title']}\n"
                    result += f"작성자: {doc['author']}\n"
                    result += f"플랫폼: {doc['platform']}\n"
                    result += f"타입: {doc['type']}\n"
                    result += f"URL: {doc['url']}\n"
                    result += f"내용 미리보기: {doc['content']}\n"
                    result += "=" * 50 + "\n"
                return jsonable_encoder(result)


            
            elif query_type == "urls_only":
                url_list = []
                for idx, meta in enumerate(results['metadatas']):
                    url_info = {
                        'url': meta.get('url', 'unknown'),
                        'title': meta.get('title', 'unknown'),
                        'author': meta.get('author', 'unknown'),
                        'platform': meta.get('platform', 'unknown'),
                        'type': meta.get('type', 'unknown')
                    }
                    url_list.append(url_info)
                
                # 터미널에 자세한 정보 출력
                print("\n=== 전체 URL 목록 ===")
                for idx, info in enumerate(url_list, 1):
                    print(f"\n{idx}. {info['platform'].upper()}")
                    print(f"제목: {info['title']}")
                    print(f"URL: {info['url']}")
                    print(f"작성자: {info['author']}")
                    print(f"데이터 유형: {info['type']}")
                    print("-" * 50)
                
                # 사용자에게 반환할 요약 정보
                platform_counts = {}
                for info in url_list:
                    platform = info['platform']
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
                
                summary = f"\n=== URL 통계 ===\n"
                summary += f"총 URL 수: {len(url_list)}\n\n"
                summary += "플랫폼별 URL 수:\n"
                for platform, count in platform_counts.items():
                    summary += f"- {platform}: {count}개\n"
                
                return summary
            
            elif query_type == "url_search":
                url_list = []
                for meta in results['metadatas']:
                    url_info = {
                        'url': meta.get('url', 'unknown'),
                        'title': meta.get('title', 'unknown'),
                        'author': meta.get('author', 'unknown'),
                        'platform': meta.get('platform', 'unknown'),
                        'type': meta.get('type', 'unknown')
                    }
                    url_list.append(url_info)
                
                # URL 목록만 반환
                return {
                    'total_count': len(url_list),
                    'urls': url_list
                }
            
            elif query_type == "url_list":
                urls = []
                for meta in results['metadatas']:
                    if 'url' in meta:
                        urls.append(meta['url'])
                return sorted(urls)  # URL 목록을 정렬하여 반환
            
            elif query_type == "url_with_title":
                url_dict = {}  # 중복 제거를 위한 딕셔너리
                
                for meta in results['metadatas']:
                    url = meta.get('url', 'unknown')
                    if url not in url_dict:
                        url_dict[url] = {
                            'url': url,
                            'title': meta.get('title', 'unknown'),
                            'author': meta.get('author', 'unknown'),
                            'platform': meta.get('platform', 'unknown'),
                            'type': meta.get('type', 'unknown')
                        }
                
                # 정렬된 목록 반환
                return sorted(url_dict.values(), key=lambda x: x['url'])
            
            elif query_type == "url_content":
                if not search_term:
                    return "검색할 URL이 제공되지 않았습니다."
                
                search_url = self._normalize_url(search_term.strip())

                for meta, content in zip(results['metadatas'], results['documents']):
                    stored_url = self._normalize_url(meta.get('url', '').strip())

                    if stored_url == search_url:
                        return {
                            'url': meta.get('url', 'unknown'),
                            'title': meta.get('title', 'unknown'),
                            'author': meta.get('author', 'unknown'),
                            'platform': meta.get('platform', 'unknown'),
                            'type': meta.get('type', 'unknown'),
                            'content': content
                        }

                return f"'{search_term}'에 해당하는 데이터가 없습니다."

            elif query_type == "type_info":
                # 데이터 유형별 개수 집계
                type_counts = {}
                for meta in results['metadatas']:
                    doc_type = meta.get('type', 'unknown')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                # 결과 포맷팅
                result = "\n=== 데이터 유형별 문서 수 ===\n"
                for doc_type, count in type_counts.items():
                    result += f"- {doc_type}: {count}건\n"
                
                # 데이터 유형 설명 추가
                result += "\n=== 데이터 유형 설명 ===\n"
                result += "- summary: 콘텐츠 요약본\n"
                result += "- transcript: 전체 자막/본문\n"
                result += "- metadata: 메타 정보\n"
                result += "- place_info: 장소 관련 정보\n"
                result += "- review: 리뷰 내용\n"
                
                return result
            
            else:
                # 전체 정보 요약
                total_docs = len(results['ids'])
                summary = f"""
=== 데이터베이스 전체 정보 ===
총 문서 수: {total_docs}

플랫폼별 문서 수:
{self._count_by_field(all_metadata, 'platform')}

타입별 문서 수:
{self._count_by_field(all_metadata, 'type')}

작성자별 문서 수:
{self._count_by_field(all_metadata, 'author')}

=== 최근 문서 5개 ===
"""
                for doc in all_metadata[:5]:
                    summary += f"\n제목: {doc['title']}\n"
                    summary += f"작성자: {doc['author']}\n"
                    summary += f"플랫폼: {doc['platform']}\n"
                    summary += f"타입: {doc['type']}\n"
                    summary += "=" * 30 + "\n"
                
                return summary
            
        except Exception as e:
            print(f"DB 조회 오류: {str(e)}")
            return f"데이터베이스 조회 중 오류가 발생했습니다: {str(e)}"

    def _count_by_field(self, metadata_list: list, field: str) -> str:
        """특정 필드별 문서 수 집계"""
        counts = {}
        for meta in metadata_list:
            value = meta.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        
        return "\n".join([f"- {k}: {v}건" for k, v in counts.items()])

    def _normalize_url(self, url: str) -> str:
        """URL을 정규화하여 비교 가능하게 만듦"""
        try:
            # 앞뒤 공백 제거
            url = url.strip()
            # 소문자로 변환
            url = url.lower()
            # @ 기호 제거
            if url.startswith('@'):
                url = url[1:]
            # 마지막 슬래시 제거
            url = url.rstrip('/')
            # 시간 파라미터(&t=) 제거
            if '&t=' in url:
                url = url.split('&t=')[0]
            # 추가 파라미터 제거
            if '?' in url:
                base_url = url.split('?')[0]
                params = url.split('?')[1].split('&')
                video_id = next((p.split('=')[1] for p in params if p.startswith('v=')), None)
                if video_id:
                    url = f"{base_url}?v={video_id}"
            # URL 앞에 붙은 "URL:" 문자열 제거
            if url.startswith('url:'):
                url = url.replace('url:', '').strip()
            return url
        except Exception as e:
            print(f"URL 정규화 중 오류 발생: {e}")
            return url
    
    async def chat(self, chat_data: ChatMessage):
        try:
            message = chat_data.message.lower()
            print("\n=== 채팅 요청 시작 ===")
            print(f"입력 메시지: {message}")
            
            # 예외처리: 쿼리 내용에 DB, 데이터베이스 등이 포함되어 있으면 '접근 거부' 처리
            if any(keyword in message.lower() for keyword in ["db", "데이터베이스", "database"]):
                print("\n=== DB 접근 시도 감지 ===")
                return {
                    "response": "죄송합니다. 데이터베이스 접근이 거부되었습니다. 다른 질문을 해주세요.",
                    "source": "access_denied",
                    "require_new_input": True  # 새로운 입력 요청 플래그
                }
            
            # 1단계: QA 체인으로 시도
            try:
                print("\n=== 1단계: QA 체인 시도 중... ===")
                qa_result = self.qa_chain.invoke({
                    "query": chat_data.message
                })
                print("\nQA 체인 결과:")
                print(f"응답: {qa_result.get('result', '없음')}")
                print(f"소스 문서: {qa_result.get('source_documents', '없음')}")
                
                # 부정적인 응답 패턴 확인
                negative_patterns = [
                    "죄송",
                    "제공된 정보에는",
                    "관련 정보가 없습니다",
                    "찾을 수 없습니다",
                    "알 수 없습니다",
                    "정보가 부족합니다"
                ]
                
                result = qa_result["result"].strip()
                
                # 응답이 있고 길이가 충분하며 부정적이지 않은 경우에만 반환
                if (result and 
                    len(result.split()) >= 10 and 
                    not any(pattern in result for pattern in negative_patterns)):
                    print("\n=== QA 체인 응답 반환 ===")
                    return {"response": result, "source": "qa_chain"}
                else:
                    print("\n=== QA 체인 부정적 응답 감지, 2단계로 진행 ===")
                    # 2단계로 자연스럽게 진행됨
            except Exception as e:
                print(f"\nQA 체인 오류: {str(e)}")
            
            # 2단계: 대화형 체인 시도
            try:
                print("\n=== 2단계: 대화형 체인 시도 중... ===")
                conv_result = self.chain.invoke({
                    "question": chat_data.message,
                    "chat_history": chat_data.chat_history
                })
                print("\n대화형 체인 결과:")
                print(f"응답: {conv_result.get('answer', '없음')}")
                print(f"소스 문서: {conv_result.get('source_documents', '없음')}")
                
                if conv_result["answer"].strip() and len(conv_result["answer"].split()) >= 10:
                    print("\n=== 대화형 체인 응답 반환 ===")
                    return {"response": conv_result["answer"], "source": "conversational_chain"}
            except Exception as e:
                print(f"\n대화형 체인 오류: {str(e)}")
            
            # 3단계: OpenAI API 직접 호출
            print("\n=== 3단계: OpenAI API 호출 중... ===")
            answer = await self.get_openai_response(chat_data.message)
            print(f"\nOpenAI API 응답: {answer}")
            print("\n=== OpenAI API 응답 반환 ===")
            return {"response": answer, "source": "openai"}
            
        except Exception as e:
            print(f"\n=== 채팅 처리 중 오류 발생 ===")
            print(f"오류 내용: {str(e)}")
            raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

    async def search_content(self, query: str):
        """ChromaDB에서 관련 콘텐츠 검색"""
        try:
            # 검색 실행
            search_results = self.retriever.get_relevant_documents(query)
            
            # 검색 결과 포맷팅
            formatted_results = []
            for doc in search_results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            return []

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (실제 운영에서는 특정 도메인만 허용 권장)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = ChatBot()

@app.post("/api/chat")
async def chat(chat_data: ChatMessage):
    try:
        # ChromaDB에서 관련 정보 검색
        search_results = await chatbot.search_content(chat_data.message)
        
        # OpenAI에 검색 결과와 함께 질문 전달
        response = await chatbot.chat(chat_data)
        
        return {
            "response": response["response"],
            "source": response["source"],
            "search_results": search_results  # 검색 결과도 함께 반환
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))