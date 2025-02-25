# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY2")
    GOOGLE_GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")
    GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
    MAIN_DB_PATH = os.getenv("MAIN_DB_PATH", "main_db")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")  # Define this if not already
    CHUNK_SIZE = 2048
    MODEL = "gpt-4o"
    FINAL_SUMMARY_MAX_TOKENS = 1500
    MAX_URLS = 5
    SUMMARY_DIR = "summaries"
    
class PathConfig:
    CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db", "chroma_db")

settings = Settings()
path_config = PathConfig()
