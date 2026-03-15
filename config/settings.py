import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv() # Force load from .env file

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    TEMPERATURE: float = 0.0 # Crucial for determinism
    SEED: int = 42 # For deterministic sampling

    # Chunking
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 250

    # Paths
    INPUT_DIR: str = "data/input"
    OUTPUT_DIR: str = "data/output"

    class Config:
        env_file = ".env"

settings = Settings()
