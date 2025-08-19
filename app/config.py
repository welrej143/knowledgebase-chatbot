from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # LLM
    LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'groq')  # 'groq' or 'openai'
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile')

    # Embeddings & Vector DB
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    CHROMA_DIR: str = os.getenv('CHROMA_DIR', './storage/chroma')

    # Server
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '8000'))
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')

    # Generation parameters
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', '800'))
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', '0.2'))
    TOP_K: int = int(os.getenv('TOP_K', '4'))

settings = Settings()
