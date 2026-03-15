import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Keep OpenAI fields in case you use them elsewhere
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Google Gemini config (used by OntologyExtractor)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")

    TEMPERATURE: float = 0.0
    SEED: int = 42

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

    class Config:
        env_file = ".env"

settings = Settings()
