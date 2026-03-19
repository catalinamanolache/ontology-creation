import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Google Gemini config (used when LLM_BACKEND=gemini)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

    # LLM Backend selection: "ollama" (local) or "gemini" (cloud)
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Determinism & Caching
    TEMPERATURE: float = 0.0
    TOP_P: float = 0.1
    SEED: int = 42
    USE_CACHE: bool = os.getenv("USE_CACHE", "true").lower() == "true"

    # Hugging Face config
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN", ""))
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    HF_USE_API: bool = os.getenv("HF_USE_API", "true").lower() == "true"

    # Chunking
    CHUNK_SIZE: int = 2500
    CHUNK_OVERLAP: int = 200

    # Seed ontology (Phase 2)
    # SEED_CHUNKS: int = int(os.getenv("SEED_CHUNKS", "8"))
    SEED_CHUNKS: int = 5

    # Rate limiting (only used for Gemini backend)
    RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "8"))
    RATE_LIMIT_RPD: int = int(os.getenv("RATE_LIMIT_RPD", "18"))
    MIN_DELAY_BETWEEN_REQUESTS: float = float(os.getenv("MIN_DELAY", "3.0"))

    # Paths
    INPUT_DIR: str = "data/input"
    OUTPUT_DIR: str = "data/output"

    # Legacy fields (kept for backward compatibility)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    BOOTSTRAP_CHUNKS: int = int(os.getenv("BOOTSTRAP_CHUNKS", "6"))
    BOOTSTRAP_DISTRIBUTED: bool = os.getenv("BOOTSTRAP_DISTRIBUTED", "true").lower() == "true"
    EXTRACT_ALL_CHUNKS: bool = os.getenv("EXTRACT_ALL_CHUNKS", "true").lower() == "true"
    EXTRACT_MAX_CHUNKS: int = int(os.getenv("EXTRACT_MAX_CHUNKS", "0"))

    class Config:
        env_file = ".env"


settings = Settings()
