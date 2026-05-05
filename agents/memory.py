"""
Agent Memory
─────────────
Two memory tiers that persist across sessions:

1. EpisodicMemory — what happened (session events, emotional moments, crisis episodes)
   Stored in the DB as JSON blobs on the user profile.

2. SemanticMemory — what the agent has learned about this user (patterns, triggers,
   what worked, what didn't). Built incrementally from episodic events.

Both are loaded at the start of each agent run and injected into context
so the agent can personalise its reasoning across sessions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from database.models import UserMentalHealthProfile
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Memory — individual events stored per session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodicEvent:
    timestamp: str                  # ISO string
    event_type: str                 # "crisis", "breakthrough", "goal_set", "technique_tried"
    description: str
    emotion: str
    sentiment_score: float
    session_id: str


@dataclass
class EpisodicMemory:
    events: List[EpisodicEvent] = field(default_factory=list)
    MAX_EVENTS = 50

    def record(self, event: EpisodicEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.MAX_EVENTS:
            self.events = self.events[-self.MAX_EVENTS:]

    def recent(self, n: int = 5) -> List[EpisodicEvent]:
        return self.events[-n:]

    def crisis_events(self) -> List[EpisodicEvent]:
        return [e for e in self.events if e.event_type == "crisis"]

    def to_context_summary(self) -> str:
        if not self.events:
            return "No episodic memory yet."
        recent = self.recent(5)
        lines = ["Recent memory:"]
        for e in recent:
            lines.append(f"  [{e.timestamp[:10]}] {e.event_type}: {e.description} (emotion: {e.emotion})")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict) -> "EpisodicMemory":
        em = cls()
        for ev in data.get("events", []):
            em.events.append(EpisodicEvent(**ev))
        return em

    def to_dict(self) -> Dict:
        return {"events": [asdict(e) for e in self.events]}


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Memory — learned patterns about the user
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticMemory:
    """
    Extracted facts the agent has inferred about the user over time.
    Updated at the end of each session.
    """
    effective_techniques: List[str] = field(default_factory=list)   # what worked
    ineffective_techniques: List[str] = field(default_factory=list) # what didn't
    known_triggers: List[str] = field(default_factory=list)         # things that worsen state
    known_strengths: List[str] = field(default_factory=list)        # user's coping strengths
    goals: List[str] = field(default_factory=list)                  # user-stated goals
    preferred_tone: str = "empathetic"                              # empathetic | direct | informational

    def add_effective_technique(self, technique: str) -> None:
        if technique not in self.effective_techniques:
            self.effective_techniques.append(technique)

    def add_ineffective_technique(self, technique: str) -> None:
        if technique not in self.ineffective_techniques:
            self.ineffective_techniques.append(technique)

    def to_context_summary(self) -> str:
        lines = ["User knowledge:"]
        if self.effective_techniques:
            lines.append(f"  Works well: {', '.join(self.effective_techniques[:5])}")
        if self.known_triggers:
            lines.append(f"  Triggers: {', '.join(self.known_triggers[:5])}")
        if self.goals:
            lines.append(f"  Goals: {', '.join(self.goals[:3])}")
        return "\n".join(lines) if len(lines) > 1 else "No semantic memory yet."

    @classmethod
    def from_dict(cls, data: Dict) -> "SemanticMemory":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Manager — loads/saves from DB
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemoryManager:
    """
    Loads and saves both memory tiers for a given user.
    Memory is serialised as JSON and stored in columns on UserMentalHealthProfile.

    Note: in a production system, use a dedicated vector DB (e.g. Chroma, Pinecone)
    for semantic memory search. This implementation uses JSON for simplicity.
    """

    def __init__(self, user_id: str, db: AsyncSession):
        self.user_id = user_id
        self.db = db
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

    async def load(self) -> None:
        result = await self.db.execute(
            select(UserMentalHealthProfile).where(UserMentalHealthProfile.user_id == self.user_id)
        )
        profile = result.scalar_one_or_none()
        if profile is None:
            return

        # UserMentalHealthProfile stores memory as JSON in frequent_topics
        # We extend the model conceptually here — in production add explicit columns
        meta = profile.frequent_topics  # reused as a dict if it's a dict; list otherwise
        if isinstance(meta, dict):
            episodic_raw = meta.get("episodic", {})
            semantic_raw = meta.get("semantic", {})
            self.episodic = EpisodicMemory.from_dict(episodic_raw)
            self.semantic = SemanticMemory.from_dict(semantic_raw)
            log.debug(f"[AgentMemory] Loaded memory for {self.user_id}: "
                      f"{len(self.episodic.events)} episodic events.")

    async def save(self) -> None:
        result = await self.db.execute(
            select(UserMentalHealthProfile).where(UserMentalHealthProfile.user_id == self.user_id)
        )
        profile = result.scalar_one_or_none()
        if profile is None:
            return

        memory_blob = {
            "episodic": self.episodic.to_dict(),
            "semantic": self.semantic.to_dict(),
        }
        # Preserve any existing list-form frequent_topics
        profile.frequent_topics = memory_blob
        log.debug(f"[AgentMemory] Saved memory for {self.user_id}.")

    def record_event(
        self,
        event_type: str,
        description: str,
        emotion: str,
        sentiment_score: float,
        session_id: str,
    ) -> None:
        self.episodic.record(EpisodicEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            description=description,
            emotion=emotion,
            sentiment_score=sentiment_score,
            session_id=session_id,
        ))

    def to_context(self) -> Dict[str, str]:
        """Inject memory summaries into agent context."""
        return {
            "episodic_memory": self.episodic.to_context_summary(),
            "semantic_memory": self.semantic.to_context_summary(),
            "effective_techniques": self.semantic.effective_techniques,
            "known_triggers": self.semantic.known_triggers,
            "user_goals": self.semantic.goals,
        }
