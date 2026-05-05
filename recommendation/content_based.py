"""
Content-Based Filtering
────────────────────────
Ranks resources by semantic similarity to the current user context
(emotion state, topics discussed, intent).

Similarity is computed as cosine similarity between:
  • The user context embedding (sentence-transformer)
  • Each resource's pre-computed embedding

Falls back to TF-IDF + keyword overlap when the embedding model is absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from database.models import Resource
from nlp.pipeline import NLPAnalysis
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ContentScore:
    resource_id: str
    score: float          # 0.0 – 1.0


def _cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


@lru_cache(maxsize=1)
def _get_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        from config import get_settings
        settings = get_settings()
        log.info("Loading sentence-transformer encoder…")
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except Exception as exc:
        log.warning(f"Sentence-transformer unavailable: {exc}. Using keyword overlap.")
        return None


def _embed(text: str) -> Optional[List[float]]:
    encoder = _get_encoder()
    if encoder is None:
        return None
    return encoder.encode(text, normalize_embeddings=True).tolist()


def _keyword_overlap(context_text: str, resource: Resource) -> float:
    """Simple Jaccard-style overlap between context words and resource topics."""
    words = set(context_text.lower().split())
    topics = set(t.lower() for t in (resource.topics or []))
    emotions = set(e.lower() for e in (resource.emotions or []))
    relevant = topics | emotions
    if not relevant:
        return 0.0
    intersection = len(words & relevant)
    return intersection / len(relevant)


def _build_context_text(analysis: NLPAnalysis, recent_topics: List[str]) -> str:
    """Compose a natural-language summary of the user's current state."""
    parts = [
        f"feeling {analysis.emotion.dominant}",
        f"mood is {analysis.sentiment.label}",
        f"intent is {analysis.intent.label}",
    ]
    if recent_topics:
        parts.append("topics: " + ", ".join(recent_topics[:5]))
    return ". ".join(parts)


def score_resources_content(
    resources: List[Resource],
    analysis: NLPAnalysis,
    recent_topics: List[str],
    user_topic_weights: Dict[str, float],
) -> List[ContentScore]:
    """
    Score every resource by content relevance to the user's current state.

    Parameters
    ----------
    resources           : candidate pool fetched from the DB
    analysis            : NLP analysis of the latest user message
    recent_topics       : topics mentioned in recent conversation turns
    user_topic_weights  : aggregated topic preferences from the user profile
    """
    context_text = _build_context_text(analysis, recent_topics)
    context_embedding = _embed(context_text)

    results: List[ContentScore] = []

    for resource in resources:
        if context_embedding is not None and resource.embedding:
            sim = _cosine(context_embedding, resource.embedding)
        else:
            sim = _keyword_overlap(context_text, resource)

        # Boost score by user's historical topic preferences
        topic_boost = 0.0
        for topic in (resource.topics or []):
            topic_boost += user_topic_weights.get(topic, 0.0)
        topic_boost = min(topic_boost * 0.1, 0.2)   # cap the boost

        # Emotion alignment bonus
        emotion_bonus = 0.1 if analysis.emotion.dominant in (resource.emotions or []) else 0.0

        final_score = min(sim + topic_boost + emotion_bonus, 1.0)
        results.append(ContentScore(resource_id=resource.id, score=round(final_score, 4)))

    return sorted(results, key=lambda x: x.score, reverse=True)
