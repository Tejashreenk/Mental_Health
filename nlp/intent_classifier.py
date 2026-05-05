"""
Intent Classifier
─────────────────
Zero-shot classification using BART-MNLI.
Classifies user messages into mental-health-relevant intents.

Intents
───────
  vent          — user wants to express feelings without seeking advice
  seek_advice   — user asks for coping strategies or recommendations
  seek_info     — user wants factual information about a topic
  crisis        — signals of immediate distress or self-harm
  gratitude     — user expresses thanks or positive reflection
  greeting      — hi / hello / start of conversation
  farewell      — goodbye / end of session
  check_in      — brief mood update (how are you, feeling better, etc.)
  casual_chat   — off-topic or general conversation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

from utils.logger import get_logger

log = get_logger(__name__)

CANDIDATE_INTENTS: List[str] = [
    "vent",
    "seek_advice",
    "seek_info",
    "crisis",
    "gratitude",
    "greeting",
    "farewell",
    "check_in",
    "casual_chat",
]

# Rule-based keyword fallback
_INTENT_PATTERNS: Dict[str, List[str]] = {
    "greeting":   ["hello", "hi ", "hey ", "good morning", "good evening", "howdy"],
    "farewell":   ["bye", "goodbye", "see you", "take care", "talk later"],
    "gratitude":  ["thank", "thanks", "appreciate", "grateful", "helped me"],
    "crisis":     ["kill myself", "end my life", "suicide", "self-harm", "hurt myself",
                   "want to die", "no reason to live", "cant go on", "can't go on"],
    "seek_advice":["what should i", "how do i", "can you help", "advice", "suggest",
                   "recommend", "tips", "what can i do"],
    "seek_info":  ["what is", "explain", "tell me about", "how does", "define"],
    "vent":       ["feel", "feeling", "i am so", "i'm so", "nobody", "everything is",
                   "always", "never", "i just", "i hate"],
    "check_in":   ["feeling better", "mood today", "doing okay", "a bit better", "not great"],
    "casual_chat":["weather", "movie", "game", "funny", "lol", "haha"],
}


@dataclass
class IntentResult:
    label: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)


def _rule_based_intent(text: str) -> IntentResult:
    text_lower = text.lower()
    scores: Dict[str, float] = {i: 0.0 for i in CANDIDATE_INTENTS}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if pat in text_lower:
                scores[intent] += 1.0
    total = sum(scores.values())
    if total == 0:
        scores["vent"] = 1.0  # safe default
        total = 1.0
    scores = {k: round(v / total, 4) for k, v in scores.items()}
    label = max(scores, key=scores.get)
    return IntentResult(label=label, confidence=scores[label], all_scores=scores)


@lru_cache(maxsize=1)
def _get_zs_pipeline():
    try:
        from transformers import pipeline as hf_pipeline
        from config import get_settings
        settings = get_settings()
        log.info("Loading zero-shot intent classifier…")
        return hf_pipeline(
            "zero-shot-classification",
            model=settings.INTENT_MODEL,
            device=-1,          # CPU; set to 0 for GPU
        )
    except Exception as exc:
        log.warning(f"Zero-shot model unavailable: {exc}. Using rule-based intent.")
        return None


def classify_intent(text: str) -> IntentResult:
    """
    Returns the most probable intent for a user message.
    """
    pipeline = _get_zs_pipeline()
    if pipeline is None:
        return _rule_based_intent(text)

    try:
        # Use multi-label=False: pick the single best intent
        output = pipeline(text[:512], candidate_labels=CANDIDATE_INTENTS, multi_label=False)
        all_scores = dict(zip(output["labels"], [round(s, 4) for s in output["scores"]]))
        return IntentResult(
            label=output["labels"][0],
            confidence=round(output["scores"][0], 4),
            all_scores=all_scores,
        )
    except Exception as exc:
        log.warning(f"Intent classification failed: {exc}. Falling back to rules.")
        return _rule_based_intent(text)
