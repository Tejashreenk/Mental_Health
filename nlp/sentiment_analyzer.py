"""
Sentiment Analyzer
──────────────────
Two-tier approach:
  1. VADER — fast, rule-based, for real-time feedback
  2. RoBERTa (HuggingFace) — deep contextual analysis, loaded lazily
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SentimentResult:
    label: str          # "positive" | "neutral" | "negative"
    score: float        # -1.0 (very negative) → +1.0 (very positive)
    confidence: float   # 0.0 – 1.0
    compound: float     # VADER compound score (-1 to 1)


_VADER = SentimentIntensityAnalyzer()


def _normalize_text(text: str) -> str:
    """Light sanitisation — strip excessive whitespace, normalise quotes."""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]  # cap length for model safety


def analyze_sentiment_vader(text: str) -> SentimentResult:
    """Fast VADER-based sentiment analysis — no GPU required."""
    text = _normalize_text(text)
    scores = _VADER.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label, score = "positive", compound
    elif compound <= -0.05:
        label, score = "negative", abs(compound)
    else:
        label, score = "neutral", 0.0

    confidence = max(scores["pos"], scores["neg"], scores["neu"])
    return SentimentResult(
        label=label,
        score=compound,
        confidence=confidence,
        compound=compound,
    )


@lru_cache(maxsize=1)
def _get_transformer_pipeline():
    """Lazily load the transformer pipeline — only once per process."""
    try:
        from transformers import pipeline as hf_pipeline
        from config import get_settings
        settings = get_settings()
        log.info("Loading sentiment transformer model…")
        return hf_pipeline(
            "sentiment-analysis",
            model=settings.SENTIMENT_MODEL,
            top_k=1,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        log.warning(f"Transformer sentiment model unavailable: {exc}. Falling back to VADER.")
        return None


def analyze_sentiment(text: str, use_transformer: bool = True) -> SentimentResult:
    """
    Full sentiment analysis.
    Falls back gracefully to VADER if the transformer is unavailable.
    """
    vader_result = analyze_sentiment_vader(text)

    if not use_transformer:
        return vader_result

    pipeline = _get_transformer_pipeline()
    if pipeline is None:
        return vader_result

    try:
        raw = pipeline(_normalize_text(text))
        top = raw[0][0] if isinstance(raw[0], list) else raw[0]
        label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral",
                     "label_0": "negative", "label_1": "neutral", "label_2": "positive"}
        label = label_map.get(top["label"].lower(), top["label"].lower())
        conf = float(top["score"])
        # Blend transformer confidence with VADER compound for a richer score
        sign = 1.0 if label == "positive" else (-1.0 if label == "negative" else 0.0)
        score = sign * conf
        return SentimentResult(
            label=label,
            score=score,
            confidence=conf,
            compound=vader_result.compound,
        )
    except Exception as exc:
        log.warning(f"Transformer inference failed: {exc}. Using VADER result.")
        return vader_result
