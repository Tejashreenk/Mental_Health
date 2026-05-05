"""
Emotion Extractor
─────────────────
Extracts fine-grained emotions (joy, sadness, anger, fear, surprise, disgust,
neutral) using a DistilRoBERTa model fine-tuned on GoEmotions.
Falls back to a rule-based lexicon when the model is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

from utils.logger import get_logger

log = get_logger(__name__)

EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

# Simple lexicon for rule-based fallback
_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "joy":      ["happy", "great", "wonderful", "excited", "love", "laugh", "smile", "glad", "fantastic"],
    "sadness":  ["sad", "depressed", "unhappy", "cry", "grief", "miserable", "hopeless", "lonely", "empty"],
    "anger":    ["angry", "furious", "rage", "hate", "frustrat", "annoyed", "mad", "irritated"],
    "fear":     ["scared", "afraid", "anxious", "panic", "terrified", "nervous", "worried", "dread"],
    "disgust":  ["disgusted", "gross", "sick", "revolted", "awful", "horrible"],
    "surprise": ["surprised", "shocked", "amazed", "unexpected", "unbelievable", "astonished"],
}


@dataclass
class EmotionResult:
    dominant: str               # highest-scoring emotion label
    scores: Dict[str, float] = field(default_factory=dict)  # all emotion probabilities


def _rule_based_emotion(text: str) -> EmotionResult:
    text_lower = text.lower()
    scores: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1.0
    total = sum(scores.values()) or 1.0
    scores = {e: round(v / total, 4) for e, v in scores.items()}
    if sum(scores.values()) == 0:
        scores["neutral"] = 1.0
    dominant = max(scores, key=scores.get)
    return EmotionResult(dominant=dominant, scores=scores)


@lru_cache(maxsize=1)
def _get_emotion_pipeline():
    try:
        from transformers import pipeline as hf_pipeline
        from config import get_settings
        settings = get_settings()
        log.info("Loading emotion classifier model…")
        return hf_pipeline(
            "text-classification",
            model=settings.EMOTION_MODEL,
            top_k=None,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        log.warning(f"Emotion model unavailable: {exc}. Using rule-based fallback.")
        return None


def extract_emotions(text: str) -> EmotionResult:
    """
    Classify the dominant emotion and return a probability distribution
    over all emotion classes.
    """
    pipeline = _get_emotion_pipeline()
    if pipeline is None:
        return _rule_based_emotion(text)

    try:
        raw = pipeline(text[:512])
        items = raw[0] if isinstance(raw[0], list) else raw
        scores = {item["label"].lower(): round(float(item["score"]), 4) for item in items}
        dominant = max(scores, key=scores.get)
        return EmotionResult(dominant=dominant, scores=scores)
    except Exception as exc:
        log.warning(f"Emotion inference failed: {exc}. Falling back to rule-based.")
        return _rule_based_emotion(text)
