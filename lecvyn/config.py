import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
ASR_BACKEND = os.getenv("ASR_BACKEND", "groq")
LOCAL_WHISPER_SIZE = os.getenv("LOCAL_WHISPER_SIZE", "small")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/LaBSE")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
CHUNK_SECONDS = 60
