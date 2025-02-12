from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai_api.youtube_subtitle import get_youtube_transcript
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

router = APIRouter()

class SubtitleCheckRequest(BaseModel):
    video_url: str

@router.post("/check_subtitles", summary="유튜브 자막 체크", description="유튜브 영상 URL을 받아 자막 존재 여부를 확인합니다.")
async def check_youtube_subtitles(request: SubtitleCheckRequest):
    """
    주어진 YouTube 영상 URL에서 자막을 추출하여 존재 여부와 자막 길이를 반환합니다.
    """
    video_url = request.video_url
    try:
        transcript = get_youtube_transcript(video_url)
        return {
            "video_url": video_url,
            "has_subtitles": True,
            "transcript_length": len(transcript)
        }
    except (TranscriptsDisabled, NoTranscriptFound):
        return {
            "video_url": video_url,
            "has_subtitles": False,
            "error": "자막을 찾을 수 없습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 