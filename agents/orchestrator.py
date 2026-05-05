"""
Orchestrator Agent
───────────────────
The top-level agent that:

  1. Reads the user message and quickly classifies the domain.
  2. Routes to the appropriate specialist agent:
       • crisis signals           → CrisisInterventionAgent
       • CBT / thought patterns   → CBTAgent
       • mindfulness / breathing  → MindfulnessAgent
       • grief / loss             → GriefAgent
       • check-in / progress      → ProgressAgent
       • general / mixed          → BaseAgent (generalist)
  3. Injects agent memory into the context before the specialist runs.
  4. Saves updated memory after the specialist completes.
  5. Returns the final AgentTrace with the composed reply.

This is the single entry point called by the API layer —
`api/routes.py` only needs to call `OrchestratorAgent.run()`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from agents.react_agent import BaseAgent, AgentTrace, RuleBasedReasoner, ThinkOutput
from agents.specialist_agents import (
    CrisisInterventionAgent,
    CBTAgent,
    MindfulnessAgent,
    GriefAgent,
    ProgressAgent,
)
from agents.memory import AgentMemoryManager
from agents.tools import TOOL_REGISTRY
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing rules (fast, pre-NLP heuristics for O(1) routing)
# ─────────────────────────────────────────────────────────────────────────────

_CRISIS_KEYWORDS = [
    "kill myself", "end my life", "suicide", "self-harm", "want to die",
    "hurt myself", "no reason to live", "can't go on", "cant go on",
    "better off dead",
]

_GRIEF_KEYWORDS = [
    "died", "death", "passed away", "lost my", "grief", "mourning",
    "bereavement", "funeral", "miss them", "gone forever",
]

_MINDFULNESS_KEYWORDS = [
    "meditat", "breathe", "breathing", "calm down", "relax", "grounding",
    "mindful", "present moment", "overwhelmed right now",
]

_CBT_KEYWORDS = [
    "negative thoughts", "thought", "think", "believe", "worthless",
    "nobody cares", "always fail", "catastroph", "spiral",
]

_PROGRESS_KEYWORDS = [
    "how am i doing", "making progress", "better than before",
    "check in", "checking in", "update", "last time", "since we talked",
]


def _route(text: str, nlp_intent: Optional[str] = None, nlp_emotion: Optional[str] = None) -> str:
    """
    Return the specialist agent name to route to.
    Priority: crisis > grief > mindfulness > CBT > progress > general
    """
    text_lower = text.lower()

    if any(kw in text_lower for kw in _CRISIS_KEYWORDS):
        return "crisis"

    if nlp_intent == "crisis":
        return "crisis"

    if any(kw in text_lower for kw in _GRIEF_KEYWORDS) or nlp_emotion == "grief":
        return "grief"

    if any(kw in text_lower for kw in _MINDFULNESS_KEYWORDS):
        return "mindfulness"

    if any(kw in text_lower for kw in _PROGRESS_KEYWORDS) or nlp_intent == "check_in":
        return "progress"

    if any(kw in text_lower for kw in _CBT_KEYWORDS) or nlp_intent in ("seek_advice", "vent"):
        return "cbt"

    return "general"


_AGENT_MAP: Dict[str, type] = {
    "crisis":      CrisisInterventionAgent,
    "grief":       GriefAgent,
    "mindfulness": MindfulnessAgent,
    "progress":    ProgressAgent,
    "cbt":         CBTAgent,
    "general":     BaseAgent,
}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Entry point for the agentic chat pipeline.

    Usage
    -----
    orchestrator = OrchestratorAgent()
    trace = await orchestrator.run(
        user_message="I've been feeling really anxious…",
        user_id="abc-123",
        db=db_session,
        session_id="sess-456",
        user_topic_weights={},
        user_modality_prefs={},
        recent_topics=["anxiety", "work"],
        stress_level=7,
    )
    reply = trace.final_answer
    """

    async def run(
        self,
        user_message: str,
        user_id: str,
        db: AsyncSession,
        session_id: str,
        user_topic_weights: Dict[str, float],
        user_modality_prefs: Dict[str, float],
        recent_topics: List[str],
        stress_level: int = 5,
    ) -> AgentTrace:

        # ── Load agent memory ────────────────────────────────────────────────
        memory = AgentMemoryManager(user_id=user_id, db=db)
        await memory.load()

        # ── Build shared context ─────────────────────────────────────────────
        context: Dict[str, Any] = {
            "user_id": user_id,
            "db": db,
            "session_id": session_id,
            "user_topic_weights": user_topic_weights,
            "user_modality_prefs": user_modality_prefs,
            "recent_topics": recent_topics,
            "stress_level": stress_level,
            "tools_run": [],
            # Memory injected so the reasoner can use it
            **memory.to_context(),
        }

        # ── Quick pre-routing (before NLP) for crisis safety ─────────────────
        route_key = _route(user_message)
        log.info(f"[Orchestrator] Pre-route: '{route_key}' for message: '{user_message[:60]}…'")

        # ── Instantiate specialist agent ──────────────────────────────────────
        agent_cls = _AGENT_MAP.get(route_key, BaseAgent)
        agent: BaseAgent = agent_cls()

        # ── Run the specialist's ReAct loop ───────────────────────────────────
        trace = await agent.run(user_message=user_message, context=context)

        # ── Post-run: refine route if NLP disagrees with keyword routing ──────
        analysis = context.get("analysis", {})
        nlp_route = _route(
            user_message,
            nlp_intent=analysis.get("intent"),
            nlp_emotion=analysis.get("emotion"),
        )

        if nlp_route != route_key and not trace.is_crisis:
            log.info(f"[Orchestrator] NLP re-route: '{route_key}' → '{nlp_route}'. Re-running.")
            context["tools_run"] = []  # reset tool tracking but keep observations
            agent2_cls = _AGENT_MAP.get(nlp_route, BaseAgent)
            agent2: BaseAgent = agent2_cls()
            trace = await agent2.run(user_message=user_message, context=context)

        # ── Update agent memory from this turn ───────────────────────────────
        if analysis:
            memory.record_event(
                event_type="crisis" if trace.is_crisis else "session_turn",
                description=f"User said: '{user_message[:80]}…'",
                emotion=analysis.get("emotion", "neutral"),
                sentiment_score=analysis.get("sentiment_score", 0.0),
                session_id=session_id,
            )

        recs = context.get("recommendations", [])
        for r in recs:
            title = r.get("title", "")
            if r.get("score", 0) > 0.6:
                memory.semantic.add_effective_technique(title)

        await memory.save()

        log.info(
            f"[Orchestrator] Done. Agent={agent.__class__.__name__} "
            f"Steps={trace.total_steps} Crisis={trace.is_crisis}"
        )
        return trace
