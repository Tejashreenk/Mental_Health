"""
Central configuration for the Mental Health AI Recommendation Engine.
All settings are loaded from environment variables with safe defaults.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # ── Application ─────────────────────────────────────────────────────────
    APP_NAME: str = "Mental Health AI Recommendation Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="production")

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = Field(default="change-me-in-production-use-secrets-manager")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./mental_health.db")

    # ── Redis (session cache) ────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    SESSION_TTL_SECONDS: int = 3600  # 1 hour

    # ── NLP Models ───────────────────────────────────────────────────────────
    # HuggingFace model IDs — override for custom fine-tuned models
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    INTENT_MODEL: str = "facebook/bart-large-mnli"          # zero-shot intent
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Recommendation Engine ────────────────────────────────────────────────
    CONTENT_WEIGHT: float = 0.6          # weight for content-based score
    COLLABORATIVE_WEIGHT: float = 0.4   # weight for collaborative score
    TOP_K_RECOMMENDATIONS: int = 5
    MIN_INTERACTIONS_FOR_CF: int = 5    # min interactions before CF kicks in

    # ── Crisis Detection ─────────────────────────────────────────────────────
    CRISIS_SCORE_THRESHOLD: float = 0.75
    CRISIS_HOTLINE: str = "988"          # Suicide & Crisis Lifeline (US)
    CRISIS_TEXT_LINE: str = "Text HOME to 741741"

    # ── Conversation ─────────────────────────────────────────────────────────
    MAX_HISTORY_TURNS: int = 20
    CONTEXT_WINDOW_TURNS: int = 5        # turns fed to the recommendation engine

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
