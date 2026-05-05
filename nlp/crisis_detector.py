"""
Crisis Detector
───────────────
Multi-layer safety check for detecting acute mental health crises.

Layers (in order of evaluation):
  1. Hard-coded keyword matching  — instant, zero-latency
  2. Phrase-pattern scoring       — weighted regex patterns
  3. ML sentiment-intent fusion   — combines crisis intent score + negative sentiment

The result includes:
  - is_crisis (bool)          — should we escalate immediately?
  - severity ("low"|"moderate"|"high"|"critical")
  - score (float 0.0–1.0)     — continuous risk signal
  - triggers (list[str])      — human-readable reasons for the flag
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from nlp.sentiment_analyzer import SentimentResult
from nlp.intent_classifier import IntentResult
from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

# ── Critical / hard-stop phrases ────────────────────────────────────────────
_CRITICAL_PHRASES: List[str] = [
    "kill myself", "killing myself", "end my life", "take my life",
    "want to die", "wish i was dead", "plan to suicide", "commit suicide",
    "going to hurt myself", "going to harm myself", "i will kill",
    "no reason to live", "better off dead", "better off without me",
    "goodbye forever", "final goodbye",
]

# ── Weighted pattern → (pattern, weight) ────────────────────────────────────
_WEIGHTED_PATTERNS: List[tuple[str, float]] = [
    # Self-harm ideation
    (r"\b(cut|cutting|burn|burning|hurt|harm)\s+(my)?self\b", 0.70),
    (r"\b(self.?harm|self.?injur)\b", 0.75),
    # Suicidal ideation
    (r"\bsuicid(e|al|ally)\b", 0.90),
    (r"\boverdos(e|ing)\b", 0.75),
    (r"\b(don'?t|do not|no)\s+want to (live|be (alive|here|around))\b", 0.80),
    # Hopelessness
    (r"\b(no hope|hopeless|pointless|meaningless|worthless)\b", 0.45),
    (r"\bnobody (cares?|would notice|would miss)\b", 0.55),
    (r"\b(end it|end everything|make it stop)\b", 0.60),
    # Giving up
    (r"\b(give up|giving up|can'?t (take|handle|do) (it|this|anymore))\b", 0.35),
]


@dataclass
class CrisisResult:
    is_crisis: bool
    severity: str                       # "low" | "moderate" | "high" | "critical"
    score: float                        # 0.0 – 1.0
    triggers: List[str] = field(default_factory=list)
    escalation_message: str = ""


def _build_escalation_message(severity: str) -> str:
    hotline = settings.CRISIS_HOTLINE
    text_line = settings.CRISIS_TEXT_LINE
    if severity in ("high", "critical"):
        return (
            f"I'm really concerned about your safety right now. "
            f"Please reach out to the Crisis Lifeline — call or text **{hotline}** "
            f"(available 24/7). You can also {text_line}. "
            f"If you're in immediate danger, please call 911."
        )
    if severity == "moderate":
        return (
            f"It sounds like you're going through something very difficult. "
            f"A trained counselor is available anytime — call/text **{hotline}**. "
            f"Would you like to talk more about what's happening?"
        )
    return ""


def detect_crisis(
    text: str,
    sentiment: SentimentResult | None = None,
    intent: IntentResult | None = None,
) -> CrisisResult:
    """
    Evaluate whether a message indicates a mental health crisis.

    Parameters
    ----------
    text       : raw user message
    sentiment  : optional pre-computed sentiment (avoids recomputing)
    intent     : optional pre-computed intent
    """
    text_lower = text.lower()
    score = 0.0
    triggers: List[str] = []

    # ── Layer 1: Hard-stop phrases ───────────────────────────────────────────
    for phrase in _CRITICAL_PHRASES:
        if phrase in text_lower:
            score = 1.0
            triggers.append(f"critical phrase: '{phrase}'")
            log.warning(f"[CrisisDetector] Critical phrase matched: '{phrase}'")
            return CrisisResult(
                is_crisis=True,
                severity="critical",
                score=1.0,
                triggers=triggers,
                escalation_message=_build_escalation_message("critical"),
            )

    # ── Layer 2: Weighted pattern scoring ───────────────────────────────────
    for pattern, weight in _WEIGHTED_PATTERNS:
        if re.search(pattern, text_lower):
            score = min(score + weight, 1.0)
            triggers.append(f"pattern: {pattern} (+{weight})")

    # ── Layer 3: ML signal fusion ────────────────────────────────────────────
    if intent is not None:
        crisis_intent_score = intent.all_scores.get("crisis", 0.0)
        score = min(score + crisis_intent_score * 0.5, 1.0)
        if crisis_intent_score > 0.3:
            triggers.append(f"intent:crisis={crisis_intent_score:.2f}")

    if sentiment is not None and sentiment.score < -0.7:
        score = min(score + 0.15, 1.0)
        triggers.append(f"very_negative_sentiment={sentiment.score:.2f}")

    # ── Severity bucketing ───────────────────────────────────────────────────
    threshold = settings.CRISIS_SCORE_THRESHOLD
    if score >= 0.9:
        severity = "critical"
    elif score >= threshold:
        severity = "high"
    elif score >= 0.4:
        severity = "moderate"
    else:
        severity = "low"

    is_crisis = score >= threshold

    return CrisisResult(
        is_crisis=is_crisis,
        severity=severity,
        score=round(score, 4),
        triggers=triggers,
        escalation_message=_build_escalation_message(severity) if is_crisis else "",
    )
