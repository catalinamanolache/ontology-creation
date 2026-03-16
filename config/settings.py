import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Keep OpenAI fields in case you use them elsewhere
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Hugging Face Inference configuration (open-source LLM)
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))
    # Legacy keys kept for compatibility with existing .env files.
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    FALLBACK_MODEL_NAME: str = os.getenv("FALLBACK_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    TEMPERATURE: float = 0.0
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "1800"))
    SEED: int = 42
    USE_CACHE: bool = os.getenv("USE_CACHE", "true").lower() == "true"
    DET_TEST_RUNS: int = int(os.getenv("DET_TEST_RUNS", "1"))
    CLEAR_CACHE_EACH_RUN: bool = os.getenv("CLEAR_CACHE_EACH_RUN", "false").lower() == "true"
    DET_FAST_MODE: bool = os.getenv("DET_FAST_MODE", "false").lower() == "true"
    DET_CHUNK_LIMIT: int = int(os.getenv("DET_CHUNK_LIMIT", "6"))
    DET_BATCH_SIZE: int = int(os.getenv("DET_BATCH_SIZE", "6"))

    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 250

    # Bootstrap strategy
    BOOTSTRAP_CHUNKS: int = int(os.getenv("BOOTSTRAP_CHUNKS", "6"))
    BOOTSTRAP_DISTRIBUTED: bool = os.getenv("BOOTSTRAP_DISTRIBUTED", "true").lower() == "true"

    # Extraction coverage
    EXTRACT_ALL_CHUNKS: bool = os.getenv("EXTRACT_ALL_CHUNKS", "true").lower() == "true"
    EXTRACT_MAX_CHUNKS: int = int(os.getenv("EXTRACT_MAX_CHUNKS", "0"))  # 0 = no limit

    INPUT_DIR: str = "data/input"
    OUTPUT_DIR: str = "data/output"
    ONTOLOGY_OUTPUT_FILE: str = os.getenv("ONTOLOGY_OUTPUT_FILE", "ontology.ttl")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
