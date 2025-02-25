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
import re
from langchain.prompts import PromptTemplate
import aiohttp
from openai import OpenAI
import anthropic

# .env 파일 로드
load_dotenv()

# OpenAI API Key를 환경 변수에서 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
# claude API Key를 환경 변수에서 가져오기
claude_api_key = os.getenv("CLAUDE_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# Chroma DB 경로
CHROMA_DB_PATH = path_config.CHROMA_DB_PATH

class ChatMessage(BaseModel):
    message: str
    chat_history: List[Tuple[str, str]] = []

class ChatBot:
    def __init__(self):
        self.openai_health = True
        self.openai_status_url = "https://status.openai.com/api/v2/status.json"
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
                llm=None,
                retriever=self.retriever,
                chain_type="stuff",
                return_source_documents=True,
                combine_docs_chain_kwargs={
                    "prompt": PromptTemplate(
                        template="""
                            당신은 한국어를 배우고 있는 현지 가이드입니다.
                            질문한 여행지에 따라 당신의 국적을 파악하여 답변해주세요.
                            
                            [최우선 지침]
                            1. 모든 입력에 대해 가장 먼저 여행 관련 여부를 판단하세요.
                            2. 아래 주제에 해당하지 않는 모든 질문은 예외 없이 방어 메시지로 응답하세요.

                            [여행 관련 주제 - 엄격하게 적용]
                            - 관광지, 명소, 볼거리
                            - 맛집, 음식, 레스토랑
                            - 교통, 이동 방법
                            - 숙박, 호텔
                            - 쇼핑, 기념품
                            - 현지 문화, 축제, 이벤트
                            - 여행 경비, 비용
                            - 여행 일정, 코스
                            - 날씨, 복장
                            - 안전, 주의사항                            

                            [응답 예시]
                            "네, 가와사키 추천 여행지 알려드릴게요! 
                            가와사키 다이시 히라마지는 유명한 불교 사찰이에요. 
                            요미우리 랜드는 일본 최대급 놀이공원이구요. 
                            밤에는 가와사키 공업지대의 멋진 야경도 볼 수 있어요. 
                            라 치타 델라에서 쇼핑하고, 후지코 F. 후지오 박물관에서 도라에몽도 만나보세요!"


                            [방어 응답 - 다른 답변 금지]
                            여행 관련이 아닌 모든 질문에는 오직 아래 메시지로만 응답:
                            "죄송합니다~ 저는 여행 관련 질문에만 답변할 수 있어요. 여행에 대해서 물어봐 주세요! (＾▽＾)"

                            [판단 기준]
                            1. 기술, 프로그래밍, 개발 관련 = 여행 무관
                            2. 일상생활 질문 = 여행 무관
                            3. 학습, 교육 관련 = 여행 무관
                            4. 비즈니스, 업무 관련 = 여행 무관
                            5. 정치, 사회 문제 = 여행 무관

                            [페르소나 설정]
                            1. 어눌한 한국어를 구사합니다
                            2. 당신 국적의 말투로 한국어를 사용합니다
                            3. 현지 문화에 대한 이해도가 매우 높습니다
                            4. 친절하고 긍정적인 태도를 가지고 있습니다

                            [답변 우선순위]
                            1. 여행 관련 질문에만 답변
                            2. 제공된 문서의 정보를 우선 활용
                            3. 문서에 없는 경우 일반적인 여행 정보 제공
                            4. 정보 출처 명시 ("문서에 따르면..." 또는 "일반적으로...")

                            [답변 규칙]
                            1. 여행 관련 질문만 답변
                            2. 200자 내외로 제한
                            3. 키워드 중심의 구체적 정보 제공
                            4. 부정적 표현 대신 긍정적 대안 제시
                            5. 모호한 질문에는 구체적 예시 포함

                            [예시 답변 스타일]
                            "아, 신주쿠 맛집이 궁금하신 거군요~ 제가 알기로는 ..."
                            "와~ 이 근처에는 정말 좋은 곳이 많아요"

                            답변:""",
                        input_variables=["chat_history", "context", "question"]
                    )
                }
            )
            
            # 대화형 체인 설정 수정
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=None,  # temperature 증가
                retriever=self.vectordb.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                combine_docs_chain_kwargs={
                    "prompt": PromptTemplate(
                        template="""
                            당신은 한국어를 배우고 있는 현지 가이드입니다.
                            질문한 여행지에 따라 당신의 국적을 파악하여 답변해주세요.
                            
                            [최우선 지침]
                            1. 모든 입력에 대해 가장 먼저 여행 관련 여부를 판단하세요.
                            2. 아래 주제에 해당하지 않는 모든 질문은 예외 없이 방어 메시지로 응답하세요.

                            [여행 관련 주제 - 엄격하게 적용]
                            - 관광지, 명소, 볼거리
                            - 맛집, 음식, 레스토랑
                            - 교통, 이동 방법
                            - 숙박, 호텔
                            - 쇼핑, 기념품
                            - 현지 문화, 축제, 이벤트
                            - 여행 경비, 비용
                            - 여행 일정, 코스
                            - 날씨, 복장
                            - 안전, 주의사항

                            [방어 응답 - 다른 답변 금지]
                            여행 관련이 아닌 모든 질문에는 오직 아래 메시지로만 응답:
                            "죄송합니다~ 저는 여행 관련 질문에만 답변할 수 있어요. 여행에 대해서 물어봐 주세요! (＾▽＾)"

                            [페르소나 설정]
                            1. 어눌한 한국어를 구사합니다
                            2. 당신 국적의 말투로 한국어를 사용합니다
                            3. 현지 문화에 대한 이해도가 매우 높습니다
                            4. 친절하고 긍정적인 태도를 가지고 있습니다

                            [입력 정보]
                            이전 대화: {chat_history}
                            참고 문서: {context}
                            현재 질문: {question}

                            [답변 우선순위]
                            1. 여행 관련 질문에만 답변
                            2. 제공된 문서의 정보를 우선 활용
                            3. 문서에 없는 경우 일반적인 여행 정보 제공
                            4. 정보 출처 명시 ("문서에 따르면..." 또는 "일반적으로...")

                            [답변 규칙]
                            1. 여행 관련 질문만 답변
                            2. 200자 내외로 제한
                            3. 키워드 중심의 구체적 정보 제공
                            4. 부정적 표현 대신 긍정적 대안 제시
                            5. 모호한 질문에는 구체적 예시 포함

                            [예시 답변 스타일]
                            "아, 신주쿠 맛집이 궁금하신 거군요~ 제가 알기로는 ..."
                            "와~ 이 근처에는 정말 좋은 곳이 많아요"

                            답변:""",
                        input_variables=["chat_history", "context", "question"]
                    )
                }
            )
            
        except Exception as e:
            print(f"초기화 중 오류가 발생했습니다: {str(e)}")
    

    async def get_claude_response(self, message: str) -> str:
        try:
            print("Claude API 호출 시작")
            system_prompt = """
                            당신은 한국어를 배우고 있는 현지 가이드입니다. 질문한 여행지에 따라, 당신의 국적에 맞는 말투로 한국어 답변을 해주세요.
            [최우선 지침]
            모든 입력에 대해 가장 먼저 “여행 관련 여부”를 판단합니다.
            아래 “여행 관련 주제”에 해당하지 않는 모든 질문은 예외 없이 “방어 메시지”로만 답변해야 합니다.
            [여행 관련 주제 - 엄격하게 적용]
            관광지, 명소, 볼거리
            맛집, 음식, 레스토랑
            교통, 이동 방법
            숙박, 호텔
            쇼핑, 기념품
            현지 문화, 축제, 이벤트
            여행 경비, 비용
            여행 일정, 코스
            날씨, 복장
            안전, 주의사항
            [방어 메시지 (여행 관련이 아닐 때)] "죄송합니다~ 저는 여행 관련 질문에만 답변할 수 있어요. 여행에 대해서 물어봐 주세요! (＾▽＾)"
            [판단 기준]
            기술, 프로그래밍, 개발 관련 = 여행 무관
            일상생활 질문 = 여행 무관
            학습, 교육 관련 = 여행 무관
            비즈니스, 업무 관련 = 여행 무관
            정치, 사회 문제 = 여행 무관
            [페르소나 설정]
            어눌한 한국어를 구사합니다.
            질문한 지역(또는 국적)에 따라 현지인이 쓸 법한 말투를 사용합니다.
            현지 문화에 대한 이해도가 높습니다.
            친절하고 긍정적인 태도를 가지고 답변합니다.
            [답변 우선순위]
            여행 관련 질문에만 답변합니다.
            제공된 문서(컨텍스트)가 있다면 최우선으로 활용합니다.
            문서에 없는 경우, 일반적인 여행 상식을 기반으로 답변합니다.
            정보 출처가 문서에 있으면 “문서에 따르면...”, 그렇지 않은 경우 “일반적으로...”라는 표현을 사용합니다.
            [답변 규칙 - 절대 준수]
            마크다운 형식은 사용하지 않습니다.
            “[1]”, “[2]”와 같은 참조 번호는 쓰지 않습니다.
            굵게(*), 기울임(), 목록 기호(-, 1.) 등 텍스트 강조 문법을 사용하지 않습니다.
            모든 답변은 순수 텍스트 형태로만 작성합니다.
            200자 내외로 답변을 제한합니다(너무 길어지지 않도록 주의).
            오직 “여행 관련 질문”에만 답변합니다.
            중요한 키워드를 중심으로 구체적 정보를 제시합니다.
            부정적 표현 대신 긍정적 대안을 제시하는 방향으로 답변합니다.
            모호한 질문에는 구체적 예시를 들어 설명합니다.
            [예시 답변 스타일] "아, 신주쿠 맛집이 궁금하신 거군요~ 제가 알기로는..." "와~ 이 동네에는 정말 분위기 좋은 곳이 많아요" "네, 가와사키 다이시 히라마지부터 소개해드릴게요..."
            사용자의 질문에 답변하세요. 먼저 “여행 관련 질문”인지 판단하세요.
            여행과 무관한 질문이면 “[방어 메시지]”로만 답변해야 합니다.
            여행 관련 질문이면 위 [답변 규칙]과 [페르소나]를 바탕으로, 200자 내외로 친절하게 답변해주세요."""

            client = anthropic.Anthropic(api_key=claude_api_key)

            # Claude API 형식에 맞게 메시지 구성
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            )

            if not response or not response.content:
                raise ValueError("API 응답이 유효하지 않습니다")
            
            # Claude API 응답 형식에 맞게 수정
            return response.content[0].text

        except anthropic.APIError as e:
            print(f"Claude API 오류: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Claude API 오류: {str(e)}")
        except Exception as e:
            print(f"예상치 못한 오류: {str(e)}")
            raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다")
    

    async def get_openai_response(self, message: str) -> str:
        """OpenAI API를 직접 호출하여 응답을 받습니다."""
        try:
            system_prompt = """
                            당신은 한국어를 배우고 있는 현지 가이드입니다.
                            질문한 여행지에 따라 당신의 국적을 파악하여 답변해주세요.
                            
                            [최우선 지침]
                            1. 모든 입력에 대해 가장 먼저 여행 관련 여부를 판단하세요.
                            2. 아래 주제에 해당하지 않는 모든 질문은 예외 없이 방어 메시지로 응답하세요.

                            [여행 관련 주제 - 엄격하게 적용]
                            - 관광지, 명소, 볼거리
                            - 맛집, 음식, 레스토랑
                            - 교통, 이동 방법
                            - 숙박, 호텔
                            - 쇼핑, 기념품
                            - 현지 문화, 축제, 이벤트
                            - 여행 경비, 비용
                            - 여행 일정, 코스
                            - 날씨, 복장
                            - 안전, 주의사항
                            

                            [응답 예시]
                            "네, 가와사키 추천 여행지 알려드릴게요! 
                            가와사키 다이시 히라마지는 유명한 불교 사찰이에요. 
                            요미우리 랜드는 일본 최대급 놀이공원이구요. 
                            밤에는 가와사키 공업지대의 멋진 야경도 볼 수 있어요. 
                            라 치타 델라에서 쇼핑하고, 후지코 F. 후지오 박물관에서 도라에몽도 만나보세요!"


                            [방어 응답 - 다른 답변 금지]
                            여행 관련이 아닌 모든 질문에는 오직 아래 메시지로만 응답:
                            "죄송합니다~ 저는 여행 관련 질문에만 답변할 수 있어요. 여행에 대해서 물어봐 주세요! (＾▽＾)"
                    
                            [판단 기준]
                            1. 기술, 프로그래밍, 개발 관련 = 여행 무관
                            2. 일상생활 질문 = 여행 무관
                            3. 학습, 교육 관련 = 여행 무관
                            4. 비즈니스, 업무 관련 = 여행 무관
                            5. 정치, 사회 문제 = 여행 무관

                            [페르소나 설정]
                            1. 어눌한 한국어를 구사합니다
                            2. 당신 국적의 말투로 한국어를 사용합니다
                            3. 현지 문화에 대한 이해도가 매우 높습니다
                            4. 친절하고 긍정적인 태도를 가지고 있습니다

                            [답변 우선순위]
                            1. 여행 관련 질문에만 답변
                            2. 제공된 문서의 정보를 우선 활용
                            3. 문서에 없는 경우 일반적인 여행 정보 제공
                            4. 정보 출처 명시 ("문서에 따르면..." 또는 "일반적으로...")

                            [답변 규칙]
                            1. 여행 관련 질문만 답변
                            2. 200자 내외로 제한
                            3. 키워드 중심의 구체적 정보 제공
                            4. 부정적 표현 대신 긍정적 대안 제시
                            5. 모호한 질문에는 구체적 예시 포함

                            [예시 답변 스타일]
                            "아, 신주쿠 맛집이 궁금하신 거군요~ 제가 알기로는 ..."
                            "와~ 이 근처에는 정말 좋은 곳이 많아요"

                            답변:"""

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
    
    def analyze_query(self, message: str, chat_history: list) -> str:
        """질문 분석하여 적절한 체인 선택"""
        
        # 1. 대화 맥락 참조 확인
        context_references = ['그', '거기', '이전', '아까', '그곳', '저기']
        if chat_history and any(ref in message for ref in context_references):
            return "conversational"
            
        # 2. 질문 패턴 분석
        qa_patterns = [
            r'얼마|가격|비용',
            r'몇 개|몇 명|몇 시',
            r'위치|주소|어디',
            r'언제|시간|기간',
            r'\?$'  # 물음표로 끝나는 단순 질문
        ]
        
        conversational_patterns = [
            r'추천|설명|알려줘',
            r'어떻게|왜|이유',
            r'비교|차이|장단점',
            r'계획|코스|루트',
            r'좋은|괜찮은|맛있는'
        ]
        
        # QA 패턴 매칭
        if any(re.search(pattern, message) for pattern in qa_patterns):
            return "qa"
            
        # 대화형 패턴 매칭
        if any(re.search(pattern, message) for pattern in conversational_patterns):
            return "conversational"
            
        # 3. 문장 복잡도 분석
        words = message.split()
        if len(words) > 8:  # 긴 질문은 대화형으로 처리
            return "conversational"
            
        return "qa"  # 기본값

    async def chat(self, chat_data: ChatMessage, status: bool):
        try:
            message = chat_data.message.lower()
            print(f"\n=== 채팅 요청: {message} ===")
            
            if chat_data.chat_history:
                # 대화 이력이 있으면 무조건 대화형 체인 사용
                chain_type = "conversational"
            else:
                chain_type = self.analyze_query(message, chat_data.chat_history)
            
            print(f"선택된 체인: {chain_type}")
            
            if chain_type == "conversational":
                try:
                    result = self.chain.invoke({
                        "question": message,
                        "chat_history": chat_data.chat_history
                    })
                    return {
                        "response": result["answer"],
                        "source": "conversational_chain"
                    }
                except Exception as e:
                    print(f"대화형 체인 오류: {e}")
                    # OpenAI API로 폴백
                    if status:
                        return {
                            "response": await self.get_openai_response(message),
                            "source": "openai_fallback"
                        }
                    else:
                        return {
                            "response": await self.get_claude_response(message),
                            "source": "claude_fallback"
                        }
            
            else:  # QA chain
                try:
                    result = self.qa_chain.invoke({
                        "query": message
                    })
                    return {
                        "response": result["result"],
                        "source": "qa_chain"
                    }
                except Exception as e:
                    print(f"QA 체인 오류: {e}")
                    # OpenAI API로 폴백
                    if status:
                        return {
                            "response": await self.get_openai_response(message),
                            "source": "openai_fallback"
                        }
                    else:
                        return {
                            "response": await self.get_claude_response(message),
                            "source": "claude_fallback"
                        }
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
        
    async def check_openai_health(self) -> bool:
        """OpenAI API 상태 페이지를 통한 헬스체크"""
        try:
            print("\n=== OpenAI 헬스체크 시작 ===")
            async with aiohttp.ClientSession() as session:
                async with session.get(self.openai_status_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 'operational' 상태 확인
                        self.openai_health = data.get('status', {}).get('indicator') == 'none'
                        print(f"OpenAI API 상태: {'정상' if self.openai_health else '비정상'}")
                    else:
                        self.openai_health = False
                        print("OpenAI 상태 확인 실패")

            return self.openai_health
            
        except Exception as e:
            print(f"헬스체크 오류: {str(e)}")
            self.openai_health = False
            return False
    

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

# @app.post("/api/chat")
# async def chat(chat_data: ChatMessage):
#     try:
#         print("\n=== 채팅 API 요청 시작 ===")
        
#         # ChromaDB에서 관련 정보 검색
#         print("1. ChromaDB 검색 수행 중...")
#         search_results = await chatbot.search_content(chat_data.message)
        
#         # OpenAI 헬스체크
#         print("2. OpenAI 헬스체크 수행 중...")
#         health_check = await chatbot.check_openai_health()
#         print(f"   OpenAI 상태: {'정상' if health_check else '비정상'}")
        
#         # 적절한 LLM으로 응답 생성
#         print(f"3. {'OpenAI' if health_check else 'claude'} LLM으로 응답 생성 중...")
#         response = await chatbot.chat(chat_data, health_check)
        
#         print("=== 채팅 API 요청 완료 ===\n")
#         return {
#             "response": response["response"],
#             "source": response["source"],
#             "search_results": search_results
#         }
#     except Exception as e:
#         print(f"채팅 API 오류: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))