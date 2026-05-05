"""
Pydantic request / response schemas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Auth ─────────────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    age_group: Optional[str] = None
    preferred_language: str = "en"

    @field_validator("username")
    @classmethod
    def no_special_chars(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Username may only contain letters, numbers, underscores, and hyphens.")
        return v


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None   # None → start a new session


class NLPResultSchema(BaseModel):
    sentiment_label: str
    sentiment_score: float
    emotion: str
    intent: str
    crisis_detected: bool
    crisis_severity: str
    crisis_score: float


class ResourceSchema(BaseModel):
    resource_id: str
    title: str
    description: str
    resource_type: str
    url: Optional[str]
    topics: List[str]
    duration_minutes: Optional[int]
    hybrid_score: float
    reason: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    nlp_analysis: NLPResultSchema
    recommendations: List[ResourceSchema]
    is_crisis: bool


# ── Session ───────────────────────────────────────────────────────────────────

class SessionEndRequest(BaseModel):
    session_id: str


class SessionSummaryResponse(BaseModel):
    session_id: str
    turn_count: int
    session_sentiment: Optional[float]
    crisis_flagged: bool
    started_at: datetime
    ended_at: Optional[datetime]


# ── Resource Interaction ──────────────────────────────────────────────────────

class InteractionRequest(BaseModel):
    resource_id: str
    session_id: str
    interaction_type: str = Field(pattern="^(viewed|completed|saved|dismissed|rated)$")
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    helpful: Optional[bool] = None


# ── Profile ───────────────────────────────────────────────────────────────────

class UserProfileResponse(BaseModel):
    username: str
    age_group: Optional[str]
    dominant_emotion: str
    stress_level: int
    risk_level: str
    frequent_topics: List[str]
    avg_sentiment_score: float
