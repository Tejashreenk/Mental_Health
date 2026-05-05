"""
Agent Tool Registry
────────────────────
Defines every callable tool the agents can invoke during a ReAct loop.

Each tool is a plain dataclass with:
  - name         : unique snake_case identifier
  - description  : what the tool does (used for agent reasoning)
  - parameters   : JSON-schema-style dict of accepted inputs
  - fn           : the async callable

Tools wrap existing NLP / recommendation / DB logic — agents never
call those modules directly; they always go through this registry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from nlp.pipeline import analyze as nlp_analyze, NLPAnalysis
from nlp.crisis_detector import detect_crisis
from nlp.sentiment_analyzer import analyze_sentiment_vader
from recommendation.engine import generate_recommendations, RecommendedResource
from database.models import (
    Resource, ResourceInteraction, UserMentalHealthProfile, ConversationSession, Message
)
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None

    def to_observation(self) -> str:
        if not self.success:
            return f"[{self.tool_name}] ERROR: {self.error}"
        if isinstance(self.output, str):
            return f"[{self.tool_name}] {self.output}"
        return f"[{self.tool_name}] {json.dumps(self.output, default=str, indent=2)}"


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Awaitable[ToolResult]]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

async def _tool_analyze_message(text: str, **_) -> ToolResult:
    """Run full NLP pipeline on a message."""
    try:
        result = nlp_analyze(text, use_transformers=True)
        return ToolResult(
            tool_name="analyze_message",
            success=True,
            output={
                "sentiment": result.sentiment.label,
                "sentiment_score": round(result.sentiment.score, 3),
                "emotion": result.emotion.dominant,
                "emotion_scores": result.emotion.scores,
                "intent": result.intent.label,
                "intent_confidence": result.intent.confidence,
                "crisis_detected": result.crisis.is_crisis,
                "crisis_severity": result.crisis.severity,
                "crisis_score": result.crisis.score,
                "crisis_triggers": result.crisis.triggers,
            },
        )
    except Exception as e:
        return ToolResult(tool_name="analyze_message", success=False, output=None, error=str(e))


async def _tool_assess_risk(text: str, sentiment_score: float = 0.0, **_) -> ToolResult:
    """Focused crisis/risk assessment — returns escalation instructions if needed."""
    try:
        sentiment = analyze_sentiment_vader(text)
        result = detect_crisis(text, sentiment=sentiment)
        action = "ESCALATE_IMMEDIATELY" if result.severity == "critical" else (
            "REFER_TO_PROFESSIONAL" if result.severity == "high" else (
                "MONITOR_CLOSELY" if result.severity == "moderate" else "CONTINUE_NORMAL"
            )
        )
        return ToolResult(
            tool_name="assess_risk",
            success=True,
            output={
                "risk_level": result.severity,
                "crisis_score": result.score,
                "recommended_action": action,
                "escalation_message": result.escalation_message,
                "triggers_detected": result.triggers,
            },
        )
    except Exception as e:
        return ToolResult(tool_name="assess_risk", success=False, output=None, error=str(e))


async def _tool_get_user_history(
    user_id: str,
    db: AsyncSession,
    session_limit: int = 3,
    **_,
) -> ToolResult:
    """Retrieve the user's recent session summaries and emotional trajectory."""
    try:
        sessions_result = await db.execute(
            select(ConversationSession)
            .where(ConversationSession.user_id == user_id)
            .order_by(ConversationSession.started_at.desc())
            .limit(session_limit)
        )
        sessions = sessions_result.scalars().all()

        profile_result = await db.execute(
            select(UserMentalHealthProfile).where(UserMentalHealthProfile.user_id == user_id)
        )
        profile = profile_result.scalar_one_or_none()

        history = []
        for s in sessions:
            history.append({
                "session_id": s.id,
                "started_at": str(s.started_at),
                "turn_count": s.turn_count,
                "avg_sentiment": s.session_sentiment,
                "crisis_flagged": s.crisis_flagged,
            })

        return ToolResult(
            tool_name="get_user_history",
            success=True,
            output={
                "recent_sessions": history,
                "profile": {
                    "avg_sentiment": round(profile.avg_sentiment_score, 3) if profile else None,
                    "dominant_emotion": profile.dominant_emotion if profile else "unknown",
                    "stress_level": profile.stress_level if profile else 5,
                    "risk_level": profile.risk_level if profile else "low",
                    "frequent_topics": profile.frequent_topics if profile else [],
                },
            },
        )
    except Exception as e:
        return ToolResult(tool_name="get_user_history", success=False, output=None, error=str(e))


async def _tool_search_resources(
    topics: List[str],
    emotions: List[str],
    resource_types: Optional[List[str]],
    difficulty: Optional[str],
    db: AsyncSession,
    **_,
) -> ToolResult:
    """Search the resource library by topic, emotion, type, and difficulty."""
    try:
        result = await db.execute(select(Resource))
        all_resources: List[Resource] = list(result.scalars().all())

        def matches(r: Resource) -> bool:
            topic_match = not topics or any(t in (r.topics or []) for t in topics)
            emotion_match = not emotions or any(e in (r.emotions or []) for e in emotions)
            type_match = not resource_types or r.resource_type in resource_types
            diff_match = not difficulty or r.difficulty == difficulty
            return topic_match and emotion_match and type_match and diff_match

        filtered = [r for r in all_resources if matches(r) and not r.is_crisis_resource][:10]
        return ToolResult(
            tool_name="search_resources",
            success=True,
            output=[
                {"id": r.id, "title": r.title, "type": r.resource_type,
                 "topics": r.topics, "difficulty": r.difficulty, "duration": r.duration_minutes}
                for r in filtered
            ],
        )
    except Exception as e:
        return ToolResult(tool_name="search_resources", success=False, output=None, error=str(e))


async def _tool_generate_coping_plan(
    emotion: str,
    stress_level: int,
    topics: List[str],
    **_,
) -> ToolResult:
    """Generate a structured short-term coping plan based on the user's state."""
    plans = {
        "anxiety": [
            "1. Try 4-7-8 breathing right now (5 minutes)",
            "2. Write down your top 3 worries and rate how likely each is (0–10)",
            "3. Identify one small action you can take today to address the most likely worry",
            "4. Schedule a 10-minute worry window tomorrow — contain the anxiety to that window",
        ],
        "sadness": [
            "1. Name 3 things you can physically feel right now (grounding)",
            "2. Send one message to someone you trust — it can be as simple as 'hey'",
            "3. Schedule one activity you used to enjoy, even if only for 10 minutes",
            "4. Be gentle with yourself — healing is not linear",
        ],
        "anger": [
            "1. Leave the triggering situation if safe to do so",
            "2. Box breathe: 4 in · 4 hold · 4 out · 4 hold — 4 rounds",
            "3. Write an unsent letter expressing everything you feel",
            "4. After you've calmed, identify the underlying need that wasn't met",
        ],
        "fear": [
            "1. 5-4-3-2-1 grounding: name 5 things you see, 4 you touch, 3 you hear",
            "2. Remind yourself: 'I am safe right now. This feeling will pass.'",
            "3. Slow your exhale — longer out-breath activates the calm response",
            "4. Identify: is this fear about something happening NOW or a future 'what if'?",
        ],
    }

    default_plan = [
        "1. Pause and take 5 slow breaths",
        "2. Check in with your body — where do you feel this emotion physically?",
        "3. Write 3 sentences about what you're feeling without judgment",
        "4. Identify one small self-care act you can do in the next hour",
    ]

    steps = plans.get(emotion, default_plan)

    if stress_level >= 8:
        steps.insert(0, "⚠️ Your stress level seems very high. Please be extra gentle with yourself today.")

    return ToolResult(
        tool_name="generate_coping_plan",
        success=True,
        output={"emotion": emotion, "stress_level": stress_level, "plan_steps": steps, "topics": topics},
    )


async def _tool_track_progress(
    user_id: str,
    db: AsyncSession,
    **_,
) -> ToolResult:
    """Analyse the user's emotional trajectory over recent sessions."""
    try:
        sessions_result = await db.execute(
            select(ConversationSession)
            .where(ConversationSession.user_id == user_id,
                   ConversationSession.session_sentiment != None)  # noqa: E711
            .order_by(ConversationSession.started_at.asc())
            .limit(10)
        )
        sessions = sessions_result.scalars().all()

        if not sessions:
            return ToolResult(
                tool_name="track_progress",
                success=True,
                output={"message": "Not enough session data yet to track progress.", "trend": "unknown"},
            )

        sentiments = [s.session_sentiment for s in sessions if s.session_sentiment is not None]
        if len(sentiments) < 2:
            trend = "insufficient_data"
        else:
            delta = sentiments[-1] - sentiments[0]
            trend = "improving" if delta > 0.1 else ("declining" if delta < -0.1 else "stable")

        crisis_count = sum(1 for s in sessions if s.crisis_flagged)

        return ToolResult(
            tool_name="track_progress",
            success=True,
            output={
                "session_count": len(sessions),
                "sentiment_trend": trend,
                "avg_recent_sentiment": round(sum(sentiments[-3:]) / len(sentiments[-3:]), 3) if sentiments else None,
                "crisis_episodes": crisis_count,
                "progress_note": (
                    "You've been making progress — your mood trend looks positive." if trend == "improving"
                    else "Things have been tough lately. You're not alone in this." if trend == "declining"
                    else "Your mood has been relatively stable."
                ),
            },
        )
    except Exception as e:
        return ToolResult(tool_name="track_progress", success=False, output=None, error=str(e))


async def _tool_recommend_resources(
    user_id: str,
    analysis_dict: Dict[str, Any],
    db: AsyncSession,
    user_topic_weights: Dict[str, float],
    user_modality_prefs: Dict[str, float],
    recent_topics: List[str],
    stress_level: int,
    **_,
) -> ToolResult:
    """Run the full hybrid recommendation engine and return ranked resources."""
    try:
        from nlp.pipeline import NLPAnalysis
        from nlp.sentiment_analyzer import SentimentResult
        from nlp.emotion_extractor import EmotionResult
        from nlp.intent_classifier import IntentResult
        from nlp.crisis_detector import CrisisResult

        # Reconstruct NLPAnalysis from the dict produced by analyze_message tool
        analysis = NLPAnalysis(
            sentiment=SentimentResult(
                label=analysis_dict.get("sentiment", "neutral"),
                score=analysis_dict.get("sentiment_score", 0.0),
                confidence=0.8,
                compound=analysis_dict.get("sentiment_score", 0.0),
            ),
            emotion=EmotionResult(
                dominant=analysis_dict.get("emotion", "neutral"),
                scores=analysis_dict.get("emotion_scores", {}),
            ),
            intent=IntentResult(
                label=analysis_dict.get("intent", "vent"),
                confidence=analysis_dict.get("intent_confidence", 0.8),
                all_scores={},
            ),
            crisis=CrisisResult(
                is_crisis=analysis_dict.get("crisis_detected", False),
                severity=analysis_dict.get("crisis_severity", "low"),
                score=analysis_dict.get("crisis_score", 0.0),
            ),
        )

        resources_result = await db.execute(select(Resource))
        all_resources = list(resources_result.scalars().all())
        interactions_result = await db.execute(select(ResourceInteraction))
        all_interactions = list(interactions_result.scalars().all())

        recs = generate_recommendations(
            user_id=user_id,
            analysis=analysis,
            all_resources=all_resources,
            all_interactions=all_interactions,
            user_topic_weights=user_topic_weights,
            user_modality_prefs=user_modality_prefs,
            recent_topics=recent_topics,
            stress_level=stress_level,
        )

        return ToolResult(
            tool_name="recommend_resources",
            success=True,
            output=[
                {
                    "id": r.resource_id,
                    "title": r.title,
                    "type": r.resource_type,
                    "score": r.hybrid_score,
                    "reason": r.reason,
                    "topics": r.topics,
                    "duration_minutes": r.duration_minutes,
                    "url": r.url,
                }
                for r in recs
            ],
        )
    except Exception as e:
        return ToolResult(tool_name="recommend_resources", success=False, output=None, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Tool] = {
    "analyze_message": Tool(
        name="analyze_message",
        description="Run full NLP analysis (sentiment, emotion, intent, crisis) on a user message.",
        parameters={"text": "str — the raw user message"},
        fn=_tool_analyze_message,
    ),
    "assess_risk": Tool(
        name="assess_risk",
        description="Perform a focused safety/crisis risk assessment. Use this when the message contains distress signals.",
        parameters={"text": "str — user message"},
        fn=_tool_assess_risk,
    ),
    "get_user_history": Tool(
        name="get_user_history",
        description="Retrieve the user's past session history and aggregated mental health profile.",
        parameters={"user_id": "str", "db": "AsyncSession", "session_limit": "int (default 3)"},
        fn=_tool_get_user_history,
    ),
    "search_resources": Tool(
        name="search_resources",
        description="Search the resource library for articles, exercises, techniques by topic, emotion, type, difficulty.",
        parameters={
            "topics": "List[str]", "emotions": "List[str]",
            "resource_types": "Optional[List[str]]", "difficulty": "Optional[str]",
            "db": "AsyncSession",
        },
        fn=_tool_search_resources,
    ),
    "generate_coping_plan": Tool(
        name="generate_coping_plan",
        description="Create a structured step-by-step coping plan tailored to the user's current emotion and stress level.",
        parameters={"emotion": "str", "stress_level": "int (1-10)", "topics": "List[str]"},
        fn=_tool_generate_coping_plan,
    ),
    "track_progress": Tool(
        name="track_progress",
        description="Analyse the user's emotional trajectory across past sessions to identify trends (improving/stable/declining).",
        parameters={"user_id": "str", "db": "AsyncSession"},
        fn=_tool_track_progress,
    ),
    "recommend_resources": Tool(
        name="recommend_resources",
        description="Run the hybrid recommendation engine to return ranked personalised resources.",
        parameters={
            "user_id": "str", "analysis_dict": "dict (from analyze_message output)",
            "db": "AsyncSession", "user_topic_weights": "dict", "user_modality_prefs": "dict",
            "recent_topics": "List[str]", "stress_level": "int",
        },
        fn=_tool_recommend_resources,
    ),
}


def get_tool(name: str) -> Optional[Tool]:
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[str]:
    return [
        f"  • {t.name}: {t.description}"
        for t in TOOL_REGISTRY.values()
    ]
