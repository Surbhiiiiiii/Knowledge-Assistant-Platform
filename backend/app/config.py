# app/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # paths
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    VECTOR_PATH: str = os.getenv("VECTOR_INDEX_PATH", "./vectorstore/faiss_index.bin")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "./vectorstore/metadata.pkl")

    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    DB_NAME: str = os.getenv("DB_NAME", "idp_db")

    # API keys - must be in .env, do NOT hardcode in code
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    # chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K: int = int(os.getenv("TOP_K", 5))

settings = Settings()
