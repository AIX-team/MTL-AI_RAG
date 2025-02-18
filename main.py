# app/main.py
from dotenv import load_dotenv #############
import os #############

from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers.youtube_router import router as youtube_router
from routers.testrouters import router as test_router  
from routers.info2guide_router import router as info2guide_router
from routers.youtube_subtitle_router import router as youtube_subtitle_router
from routers.ai_recommend_router import router as ai_recommend_router
from chatbot.chatbot import ChatBot, ChatMessage    #############
from routers.ai_recommend_router import router as ai_recommend_router
load_dotenv() #############

openai_api_key = os.getenv("OPENAI_API_KEY") #############
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")


app = FastAPI(
    title="YouTube Info Extractor API",
    description="Extracts information from YouTube or blog URLs and summarizes travel information.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

chatbot = ChatBot() #############


# 라우터 설정
app.include_router(youtube_router, prefix="/api/v1", tags=["YouTube"])
app.include_router(test_router, prefix="/api/v1", tags=["Test"])  # test 라우터 추가
app.include_router(info2guide_router, prefix="/api/v1", tags=["Info2Guide"])
app.include_router(youtube_subtitle_router, prefix="/api/v1/youtube", tags=["YouTube Subtitle"])
app.include_router(ai_recommend_router, prefix="/api/v1", tags=["AI Recommendations"])

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")  #############
async def chat(chat_data: ChatMessage):
    try:
        print("\n=== 채팅 API 요청 시작 ===")
        
        # ChromaDB에서 관련 정보 검색
        print("1. ChromaDB 검색 수행 중...")
        search_results = await chatbot.search_content(chat_data.message)
        
        # OpenAI 헬스체크
        print("2. OpenAI 헬스체크 수행 중...")
        health_check = await chatbot.check_openai_health()
        print(f"   OpenAI 상태: {'정상' if health_check else '비정상'}")
        
        # 적절한 LLM으로 응답 생성
        print(f"3. {'OpenAI' if health_check else 'Perplexity'} LLM으로 응답 생성 중...")
        response = await chatbot.chat(chat_data, health_check)
        
        print("=== 채팅 API 요청 완료 ===\n")
        return {
            "response": response["response"],
            "source": response["source"],
            "search_results": search_results
        }
    except Exception as e:
        print(f"채팅 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 실행 환경 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
