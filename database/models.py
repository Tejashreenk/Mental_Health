"""
SQLAlchemy ORM models for the Mental Health AI platform.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, JSON, Index
)
from sqlalchemy.dialects.sqlite import TEXT as SQLITE_TEXT
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# User
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    created_at = Column(DateTime, default=_now, nullable=False)
    updated_at = Column(DateTime, default=_now, onupdate=_now, nullable=False)

    # Stored as hashed value — never plaintext
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)

    # De-identified demographic info (optional, aids collaborative filtering)
    age_group = Column(String(16), nullable=True)  # e.g. "18-24"
    preferred_language = Column(String(8), default="en")

    # Aggregated preferences (updated incrementally)
    topic_weights = Column(JSON, default=dict)       # {"anxiety": 0.8, "sleep": 0.4, ...}
    modality_prefs = Column(JSON, default=dict)      # {"article": 0.6, "exercise": 0.9, ...}

    is_active = Column(Boolean, default=True, nullable=False)

    sessions = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("ResourceInteraction", back_populates="user", cascade="all, delete-orphan")
    profiles = relationship("UserMentalHealthProfile", back_populates="user", uselist=False,
                            cascade="all, delete-orphan")


# ─────────────────────────────────────────────────────────────────────────────
# User Mental-Health Profile  (living document, updated each session)
# ─────────────────────────────────────────────────────────────────────────────

class UserMentalHealthProfile(Base):
    __tablename__ = "user_mental_health_profiles"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), unique=True, nullable=False)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    # Rolling emotional state (exponential moving average)
    avg_sentiment_score = Column(Float, default=0.0)   # -1 (negative) → +1 (positive)
    dominant_emotion = Column(String(32), default="neutral")
    stress_level = Column(Integer, default=3)           # 1–10
    risk_level = Column(String(16), default="low")      # low | moderate | high | crisis

    # Topics the user frequently discusses
    frequent_topics = Column(JSON, default=list)

    user = relationship("User", back_populates="profiles")


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Session
# ─────────────────────────────────────────────────────────────────────────────

class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=_now, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    # Aggregate analytics for the session
    turn_count = Column(Integer, default=0)
    session_sentiment = Column(Float, nullable=True)
    crisis_flagged = Column(Boolean, default=False)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan",
                            order_by="Message.timestamp")


# ─────────────────────────────────────────────────────────────────────────────
# Message
# ─────────────────────────────────────────────────────────────────────────────

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("conversation_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=_now, nullable=False)

    role = Column(Enum("user", "assistant", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)

    # NLP analysis results (stored for audit / model improvement)
    sentiment_label = Column(String(16), nullable=True)   # positive | neutral | negative
    sentiment_score = Column(Float, nullable=True)
    emotion_label = Column(String(32), nullable=True)
    intent_label = Column(String(32), nullable=True)
    crisis_score = Column(Float, default=0.0)

    session = relationship("ConversationSession", back_populates="messages")

    __table_args__ = (Index("ix_messages_session_ts", "session_id", "timestamp"),)


# ─────────────────────────────────────────────────────────────────────────────
# Mental Health Resource
# ─────────────────────────────────────────────────────────────────────────────

class Resource(Base):
    __tablename__ = "resources"

    id = Column(String(36), primary_key=True, default=_uuid)
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=False)
    content = Column(Text, nullable=True)           # full text (articles / exercises)
    url = Column(String(512), nullable=True)

    resource_type = Column(
        Enum("article", "exercise", "technique", "video", "hotline", "self_assessment",
             name="resource_type"),
        nullable=False
    )

    # Semantic tags for content-based filtering
    topics = Column(JSON, default=list)             # ["anxiety", "depression", "sleep"]
    emotions = Column(JSON, default=list)           # ["sadness", "fear", "anger"]
    difficulty = Column(String(16), default="easy") # easy | medium | hard
    duration_minutes = Column(Integer, nullable=True)

    # Embedding vector stored as JSON list (float32)
    embedding = Column(JSON, nullable=True)

    is_crisis_resource = Column(Boolean, default=False)
    language = Column(String(8), default="en")

    interactions = relationship("ResourceInteraction", back_populates="resource")


# ─────────────────────────────────────────────────────────────────────────────
# Resource Interaction  (for collaborative filtering)
# ─────────────────────────────────────────────────────────────────────────────

class ResourceInteraction(Base):
    __tablename__ = "resource_interactions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    resource_id = Column(String(36), ForeignKey("resources.id"), nullable=False)
    timestamp = Column(DateTime, default=_now, nullable=False)

    interaction_type = Column(
        Enum("viewed", "completed", "saved", "dismissed", "rated", name="interaction_type"),
        nullable=False
    )
    rating = Column(Integer, nullable=True)     # 1–5 stars (optional)
    helpful = Column(Boolean, nullable=True)    # thumbs up/down

    # Implicit feedback signal (0.0 – 1.0)
    implicit_score = Column(Float, default=0.5)

    user = relationship("User", back_populates="interactions")
    resource = relationship("Resource", back_populates="interactions")

    __table_args__ = (
        Index("ix_interactions_user_resource", "user_id", "resource_id"),
    )
