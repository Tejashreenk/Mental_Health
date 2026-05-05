"""
Hybrid Recommendation Engine
──────────────────────────────
Combines content-based and collaborative filtering scores using a
configurable weighted blend, then applies business rules:

  • Crisis resources are surfaced immediately when crisis is detected.
  • Already-completed resources are suppressed (with a small decay).
  • Resource modality is balanced (no 5 articles in a row).
  • Difficulty is matched to the user's current stress level.

Output: ranked list of RecommendedResource objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from database.models import Resource, ResourceInteraction
from nlp.pipeline import NLPAnalysis
from recommendation.content_based import score_resources_content, ContentScore
from recommendation.collaborative import compute_cf_scores, CollaborativeScore
from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass
class RecommendedResource:
    resource_id: str
    title: str
    description: str
    resource_type: str
    url: Optional[str]
    topics: List[str]
    duration_minutes: Optional[int]
    hybrid_score: float
    content_score: float
    collaborative_score: float
    reason: str                         # human-readable explanation


def _stress_to_difficulty(stress_level: int) -> List[str]:
    """Map stress level (1–10) to acceptable resource difficulties."""
    if stress_level >= 7:
        return ["easy"]
    if stress_level >= 4:
        return ["easy", "medium"]
    return ["easy", "medium", "hard"]


def _build_reason(
    resource: Resource,
    analysis: NLPAnalysis,
    content_score: float,
    collab_score: float,
) -> str:
    parts = []
    if content_score > 0.5:
        parts.append(f"matches your current {analysis.emotion.dominant} state")
    if collab_score > 0.4:
        parts.append("helped users in similar situations")
    if analysis.intent.label == "seek_advice" and resource.resource_type == "technique":
        parts.append("directly addresses your request for coping strategies")
    if not parts:
        parts.append("relevant to what you're experiencing")
    return "; ".join(parts).capitalize() + "."


def generate_recommendations(
    user_id: str,
    analysis: NLPAnalysis,
    all_resources: List[Resource],
    all_interactions: List[ResourceInteraction],
    user_topic_weights: Dict[str, float],
    user_modality_prefs: Dict[str, float],
    recent_topics: List[str],
    stress_level: int = 5,
    completed_resource_ids: Optional[Set[str]] = None,
) -> List[RecommendedResource]:
    """
    Generate a ranked list of mental health resource recommendations.

    Parameters
    ----------
    user_id              : authenticated user's ID
    analysis             : NLP analysis of the current message
    all_resources        : full candidate pool from DB
    all_interactions     : all user interactions for collaborative filtering
    user_topic_weights   : user's aggregated topic affinity scores
    user_modality_prefs  : user's preferred resource types
    recent_topics        : topics from the last N conversation turns
    stress_level         : current stress level estimate (1–10)
    completed_resource_ids: resources to deprioritise (already consumed)
    """
    completed_resource_ids = completed_resource_ids or set()

    # ── Crisis override: return crisis resources first ───────────────────────
    if analysis.crisis.is_crisis:
        crisis_resources = [r for r in all_resources if r.is_crisis_resource]
        log.info(f"[Recommender] Crisis detected — surfacing {len(crisis_resources)} crisis resources.")
        results = []
        for r in crisis_resources[:settings.TOP_K_RECOMMENDATIONS]:
            results.append(RecommendedResource(
                resource_id=r.id,
                title=r.title,
                description=r.description,
                resource_type=r.resource_type,
                url=r.url,
                topics=r.topics or [],
                duration_minutes=r.duration_minutes,
                hybrid_score=1.0,
                content_score=1.0,
                collaborative_score=1.0,
                reason="Recommended support resources for your current situation.",
            ))
        return results

    # ── Filter by language and difficulty ────────────────────────────────────
    allowed_difficulties = _stress_to_difficulty(stress_level)
    candidates = [
        r for r in all_resources
        if not r.is_crisis_resource
        and r.difficulty in allowed_difficulties
    ]

    if not candidates:
        candidates = all_resources  # relax filter if nothing survives

    candidate_ids = [r.id for r in candidates]

    # ── Content-based scores ─────────────────────────────────────────────────
    cb_scores: Dict[str, float] = {
        s.resource_id: s.score
        for s in score_resources_content(candidates, analysis, recent_topics, user_topic_weights)
    }

    # ── Collaborative scores ─────────────────────────────────────────────────
    cf_raw = compute_cf_scores(user_id, all_interactions, candidate_ids)
    cf_scores: Dict[str, float] = {s.resource_id: s.score for s in cf_raw}

    w_cb = settings.CONTENT_WEIGHT
    w_cf = settings.COLLABORATIVE_WEIGHT

    # ── Modality diversity tracker ────────────────────────────────────────────
    modality_count: Dict[str, int] = {}
    MAX_PER_MODALITY = 2

    # ── Build hybrid scores ───────────────────────────────────────────────────
    scored: List[tuple[float, float, float, Resource]] = []
    for resource in candidates:
        cb = cb_scores.get(resource.id, 0.0)
        cf = cf_scores.get(resource.id, 0.0)
        hybrid = w_cb * cb + w_cf * cf

        # Apply completion decay
        if resource.id in completed_resource_ids:
            hybrid *= 0.4

        # Modality preference boost
        modality_pref = user_modality_prefs.get(resource.resource_type, 0.0)
        hybrid = min(hybrid + modality_pref * 0.05, 1.0)

        scored.append((hybrid, cb, cf, resource))

    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Pick top-K with modality diversity ───────────────────────────────────
    results: List[RecommendedResource] = []
    for hybrid, cb, cf, resource in scored:
        if len(results) >= settings.TOP_K_RECOMMENDATIONS:
            break
        count = modality_count.get(resource.resource_type, 0)
        if count >= MAX_PER_MODALITY:
            continue
        modality_count[resource.resource_type] = count + 1

        results.append(RecommendedResource(
            resource_id=resource.id,
            title=resource.title,
            description=resource.description,
            resource_type=resource.resource_type,
            url=resource.url,
            topics=resource.topics or [],
            duration_minutes=resource.duration_minutes,
            hybrid_score=round(hybrid, 4),
            content_score=round(cb, 4),
            collaborative_score=round(cf, 4),
            reason=_build_reason(resource, analysis, cb, cf),
        ))

    log.info(f"[Recommender] Returning {len(results)} recommendations for user {user_id}.")
    return results
