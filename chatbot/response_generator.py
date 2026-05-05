"""
Response Generator
───────────────────
Builds the final chatbot response string by combining:
  • Empathetic acknowledgement (tone-matched to detected emotion/intent)
  • Crisis escalation message (when applicable)
  • Formatted resource recommendations
  • A gentle follow-up prompt to keep the conversation going

Responses are template-driven with slot-filling — no LLM required,
but the system is architected to accept an LLM hook for richer prose.
"""
from __future__ import annotations

import random
from typing import List, Optional

from nlp.pipeline import NLPAnalysis
from nlp.crisis_detector import CrisisResult
from recommendation.engine import RecommendedResource
from utils.logger import get_logger

log = get_logger(__name__)

# ── Empathy templates keyed by (emotion, intent) ────────────────────────────

_EMPATHY_TEMPLATES = {
    ("sadness",   "vent"):        [
        "I hear you — it sounds like you're carrying a lot right now.",
        "That sounds really painful. You don't have to face this alone.",
        "I'm really sorry you're feeling this way. Your feelings are completely valid.",
    ],
    ("fear",      "vent"):        [
        "Feeling scared is hard — your worry makes complete sense given what you're describing.",
        "It sounds like anxiety is weighing heavily on you right now.",
    ],
    ("anger",     "vent"):        [
        "That frustration sounds completely understandable.",
        "It makes sense you'd feel angry about that.",
    ],
    ("sadness",   "seek_advice"): [
        "I understand things feel heavy. Let me share something that might help.",
        "I'm sorry you're struggling. Here are some ideas that others have found useful:",
    ],
    ("fear",      "seek_advice"): [
        "Anxiety can feel overwhelming, but there are things that can help.",
        "Let's work through this together. Here are some coping strategies:",
    ],
    ("neutral",   "seek_info"):   [
        "Great question. Here's what I found for you:",
        "Sure, here's some information that might be helpful:",
    ],
    ("joy",       "check_in"):    [
        "I'm really glad to hear you're feeling better!",
        "That's wonderful to hear. Keep building on those positive moments.",
    ],
    ("neutral",   "greeting"):    [
        "Hello! I'm here to support you. How are you feeling today?",
        "Hi there! I'm glad you're here. What's on your mind?",
    ],
    ("neutral",   "farewell"):    [
        "Take care of yourself. Remember, I'm always here when you need to talk.",
        "Goodbye for now. You did great reaching out today.",
    ],
}

_DEFAULT_EMPATHY = [
    "Thank you for sharing that with me.",
    "I'm here with you. It takes courage to open up.",
    "I hear you. Let's take this one step at a time.",
]

_FOLLOW_UP_PROMPTS = [
    "Would any of these feel manageable to try today?",
    "Is there a specific area you'd like to explore further?",
    "How does this feel? Would you like a different type of support?",
    "Which of these resonates most with you right now?",
    "Feel free to share more — I'm here to listen.",
]


def _pick_empathy(emotion: str, intent: str) -> str:
    key = (emotion, intent)
    templates = _EMPATHY_TEMPLATES.get(key)
    if not templates:
        # Try with just intent
        for (e, i), tmpl in _EMPATHY_TEMPLATES.items():
            if i == intent:
                templates = tmpl
                break
    if not templates:
        templates = _DEFAULT_EMPATHY
    return random.choice(templates)


def _format_resource(rec: RecommendedResource, index: int) -> str:
    lines = [f"**{index}. {rec.title}**"]
    lines.append(f"   {rec.description}")
    meta_parts = [f"Type: {rec.resource_type.replace('_', ' ').title()}"]
    if rec.duration_minutes:
        meta_parts.append(f"~{rec.duration_minutes} min")
    if rec.topics:
        meta_parts.append("Topics: " + ", ".join(rec.topics[:3]))
    lines.append("   " + " · ".join(meta_parts))
    if rec.url:
        lines.append(f"   🔗 {rec.url}")
    lines.append(f"   *Why:* {rec.reason}")
    return "\n".join(lines)


def build_response(
    analysis: NLPAnalysis,
    recommendations: List[RecommendedResource],
    include_follow_up: bool = True,
) -> str:
    """
    Construct the full chatbot response.

    Parameters
    ----------
    analysis        : NLP analysis of the user's latest message
    recommendations : ranked list from the recommendation engine
    include_follow_up: whether to append a conversational follow-up prompt
    """
    parts: List[str] = []

    # ── Crisis escalation — always first ────────────────────────────────────
    if analysis.crisis.is_crisis and analysis.crisis.escalation_message:
        parts.append(analysis.crisis.escalation_message)

    # ── Empathy opener (skip for greetings/farewells with low sentiment weight)
    if analysis.intent.label not in ("farewell",) or analysis.sentiment.score < -0.1:
        empathy = _pick_empathy(analysis.emotion.dominant, analysis.intent.label)
        parts.append(empathy)

    # ── Recommendations block ─────────────────────────────────────────────────
    if recommendations and not analysis.crisis.is_crisis:
        if analysis.intent.label in ("seek_advice", "seek_info", "vent", "check_in"):
            parts.append("\nHere are some resources that may help:\n")
            for i, rec in enumerate(recommendations, start=1):
                parts.append(_format_resource(rec, i))

    elif recommendations and analysis.crisis.is_crisis:
        parts.append("\n**Additional support resources:**\n")
        for i, rec in enumerate(recommendations, start=1):
            parts.append(_format_resource(rec, i))

    # ── Follow-up prompt ──────────────────────────────────────────────────────
    if include_follow_up and analysis.intent.label not in ("farewell", "greeting"):
        parts.append("\n" + random.choice(_FOLLOW_UP_PROMPTS))

    return "\n".join(parts)
