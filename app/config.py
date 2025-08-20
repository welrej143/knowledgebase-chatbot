from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central app settings. Reads from environment variables (and optional .env file).
    Also ensures important folders exist at import time.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider & models ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "groq"

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Groq
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    # --- Embeddings & Vector DB ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Where uploaded/raw files live
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    # Where Chroma persists its index
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./storage/chroma")

    # Optional: LibreOffice binary for DOC->PDF conversion (if available)
    SOFFICE_PATH: str | None = os.getenv("SOFFICE_PATH")

    # --- Server / API ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

    # --- Generation params ---
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "800"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    TOP_K: int = int(os.getenv("TOP_K", "8"))

    # --- Misc (quiet noisy libs by default) ---
    CHROMA_TELEMETRY_DISABLED: str = os.getenv("CHROMA_TELEMETRY_DISABLED", "1")
    TOKENIZERS_PARALLELISM: str = os.getenv("TOKENIZERS_PARALLELISM", "false")

    def ensure_dirs(self) -> None:
        """Create important folders on first run (ok if they already exist)."""
        for path in (self.DATA_DIR, self.CHROMA_DIR):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception:
                # Non-fatal: container may not have perms yet, or path is mounted later.
                pass


settings = Settings()
settings.ensure_dirs()
