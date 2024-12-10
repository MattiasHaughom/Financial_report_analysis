import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Get the directory of the current script (settings.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, "..", ".env")  # Adjust to point to the project root

# Load the .env file
load_dotenv(dotenv_path=DOTENV_PATH)


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")


class CohereSettings(BaseModel):
    """Cohere-specific settings."""

    api_key: str = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    reports_table: str = "reports"
    analysis_table: str = "analysis"
    company_table: str = "company"
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)


class AIAnalysisSettings(BaseModel):
    """Settings for AI analysis."""

    summary_model: str = Field(default="gpt-4o-mini")
    analysis_model: str = Field(default="gpt-4o-mini")


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    cohere: CohereSettings = Field(default_factory=CohereSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    ai_analysis: AIAnalysisSettings = Field(default_factory=AIAnalysisSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings