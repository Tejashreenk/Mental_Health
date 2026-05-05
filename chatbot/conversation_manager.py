"""
Conversation Manager
─────────────────────
Stateful manager for a single conversation session.

Responsibilities:
  • Maintains a sliding window of recent turns (in-memory + DB-persisted)
  • Extracts running topic list from conversation history
  • Manages session lifecycle (start, update, end)
  • Provides context snapshots to the recommendation engine
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import ConversationSession, Message, User, UserMentalHealthProfile
from nlp.pipeline import NLPAnalysis
from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

# Topic extraction: map emotion/intent pairs to likely topics
_TOPIC_MAP: Dict[str, List[str]] = {
    "anxiety":    ["anxiety", "worry", "panic", "nervous", "overwhelmed"],
    "depression": ["depression", "hopeless", "sad", "empty", "worthless", "low"],
    "sleep":      ["sleep", "insomnia", "tired", "exhausted", "rest", "nightmare"],
    "stress":     ["stress", "pressure", "deadline", "work", "burnout"],
    "anger":      ["anger", "angry", "rage", "frustrated", "furious"],
    "grief":      ["grief", "loss", "death", "mourning", "bereaved"],
    "loneliness": ["lonely", "alone", "isolated", "nobody", "no friends"],
    "trauma":     ["trauma", "abuse", "ptsd", "flashback", "trigger"],
    "addiction":  ["addiction", "substance", "alcohol", "drug", "compulsion"],
    "self_esteem":["worthless", "ugly", "hate myself", "failure", "useless"],
}


@dataclass
class Turn:
    role: str               # "user" | "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    analysis: Optional[NLPAnalysis] = None


@dataclass
class SessionContext:
    session_id: str
    user_id: str
    turns: List[Turn] = field(default_factory=list)
    recent_topics: List[str] = field(default_factory=list)
    running_sentiment: float = 0.0
    stress_level: int = 5
    crisis_flagged: bool = False


def _extract_topics(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for topic, keywords in _TOPIC_MAP.items():
        if any(kw in text_lower for kw in keywords):
            found.append(topic)
    return found


def _update_stress_estimate(current: int, sentiment_score: float) -> int:
    """Incrementally update stress level based on sentiment."""
    delta = -1 if sentiment_score > 0.3 else (1 if sentiment_score < -0.3 else 0)
    return max(1, min(10, current + delta))


class ConversationManager:
    """
    Manages the state of a single active conversation session.
    One instance per active session.
    """

    def __init__(self, context: SessionContext):
        self._ctx = context

    @property
    def context(self) -> SessionContext:
        return self._ctx

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    async def create_session(
        cls,
        user_id: str,
        db: AsyncSession,
    ) -> "ConversationManager":
        """Start a new conversation session and persist it to the DB."""
        session = ConversationSession(user_id=user_id)
        db.add(session)
        await db.flush()   # get the generated ID
        log.info(f"[ConversationManager] New session {session.id} for user {user_id}")
        ctx = SessionContext(session_id=session.id, user_id=user_id)
        return cls(ctx)

    @classmethod
    async def resume_session(
        cls,
        session_id: str,
        db: AsyncSession,
    ) -> Optional["ConversationManager"]:
        """Load an existing active session from the DB."""
        result = await db.execute(
            select(ConversationSession).where(
                ConversationSession.id == session_id,
                ConversationSession.is_active == True,  # noqa: E712
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            return None

        # Load the last N messages for context
        msgs_result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.desc())
            .limit(settings.MAX_HISTORY_TURNS)
        )
        messages = msgs_result.scalars().all()[::-1]  # chronological order

        turns = [
            Turn(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in messages
        ]

        ctx = SessionContext(
            session_id=session_id,
            user_id=session.user_id,
            turns=turns,
            crisis_flagged=session.crisis_flagged,
        )
        return cls(ctx)

    # ── Turn management ──────────────────────────────────────────────────────

    async def add_user_turn(
        self,
        text: str,
        analysis: NLPAnalysis,
        db: AsyncSession,
    ) -> None:
        """Record a user message, update context, and persist to DB."""
        turn = Turn(role="user", content=text, analysis=analysis)
        self._ctx.turns.append(turn)

        # Update running state
        new_topics = _extract_topics(text)
        for t in new_topics:
            if t not in self._ctx.recent_topics:
                self._ctx.recent_topics.append(t)
        self._ctx.recent_topics = self._ctx.recent_topics[-10:]  # keep last 10

        self._ctx.running_sentiment = (
            self._ctx.running_sentiment * 0.7 + analysis.sentiment.score * 0.3
        )
        self._ctx.stress_level = _update_stress_estimate(
            self._ctx.stress_level, analysis.sentiment.score
        )

        if analysis.crisis.is_crisis:
            self._ctx.crisis_flagged = True

        # Trim to window
        if len(self._ctx.turns) > settings.MAX_HISTORY_TURNS:
            self._ctx.turns = self._ctx.turns[-settings.MAX_HISTORY_TURNS:]

        # Persist message
        msg = Message(
            session_id=self._ctx.session_id,
            role="user",
            content=text,
            sentiment_label=analysis.sentiment.label,
            sentiment_score=analysis.sentiment.score,
            emotion_label=analysis.emotion.dominant,
            intent_label=analysis.intent.label,
            crisis_score=analysis.crisis.score,
        )
        db.add(msg)

        # Update session aggregate
        await db.execute(
            ConversationSession.__table__.update()
            .where(ConversationSession.__table__.c.id == self._ctx.session_id)
            .values(
                turn_count=ConversationSession.__table__.c.turn_count + 1,
                session_sentiment=self._ctx.running_sentiment,
                crisis_flagged=self._ctx.crisis_flagged,
            )
        )

    async def add_assistant_turn(self, text: str, db: AsyncSession) -> None:
        """Record an assistant response and persist it."""
        turn = Turn(role="assistant", content=text)
        self._ctx.turns.append(turn)

        msg = Message(
            session_id=self._ctx.session_id,
            role="assistant",
            content=text,
        )
        db.add(msg)

    async def end_session(self, db: AsyncSession) -> None:
        """Mark the session as ended in the DB."""
        await db.execute(
            ConversationSession.__table__.update()
            .where(ConversationSession.__table__.c.id == self._ctx.session_id)
            .values(is_active=False, ended_at=datetime.utcnow())
        )
        log.info(f"[ConversationManager] Session {self._ctx.session_id} ended.")

    # ── Context access ───────────────────────────────────────────────────────

    def get_recent_context(self, n: int | None = None) -> List[Turn]:
        """Return the last n turns (defaults to CONTEXT_WINDOW_TURNS)."""
        n = n or settings.CONTEXT_WINDOW_TURNS
        return self._ctx.turns[-n:]

    def get_recent_topics(self) -> List[str]:
        return list(self._ctx.recent_topics)
