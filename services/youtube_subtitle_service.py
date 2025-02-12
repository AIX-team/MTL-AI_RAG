from ai_api.youtube_subtitle import get_youtube_transcript
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

class YouTubeSubtitleService:
    @staticmethod
    def check_subtitles(video_url: str) -> dict:
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
            raise Exception(str(e)) 