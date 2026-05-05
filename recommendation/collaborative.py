"""
Collaborative Filtering
────────────────────────
User-based collaborative filtering using implicit feedback signals.

Algorithm:
  1. Build a user × resource interaction matrix from DB interactions.
  2. Compute cosine similarity between the target user and all other users.
  3. Score resources as the weighted average rating from the top-K similar users.

Falls back to returning empty scores when there are insufficient interactions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from database.models import ResourceInteraction
from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

TOP_SIMILAR_USERS = 20   # number of neighbours to consider


@dataclass
class CollaborativeScore:
    resource_id: str
    score: float          # 0.0 – 1.0


def _interaction_to_signal(interaction: ResourceInteraction) -> float:
    """Convert a raw interaction record into a 0–1 implicit signal."""
    base = interaction.implicit_score or 0.5
    if interaction.interaction_type == "completed":
        base = max(base, 0.8)
    elif interaction.interaction_type == "saved":
        base = max(base, 0.7)
    elif interaction.interaction_type == "dismissed":
        base = min(base, 0.1)
    if interaction.helpful is True:
        base = min(base + 0.15, 1.0)
    elif interaction.helpful is False:
        base = max(base - 0.15, 0.0)
    if interaction.rating is not None:
        base = (base + (interaction.rating - 1) / 4) / 2
    return round(base, 4)


def compute_cf_scores(
    target_user_id: str,
    all_interactions: List[ResourceInteraction],
    candidate_resource_ids: List[str],
) -> List[CollaborativeScore]:
    """
    Run user-based collaborative filtering.

    Parameters
    ----------
    target_user_id         : ID of the user we are making recommendations for
    all_interactions       : all interactions loaded from DB (can be pre-filtered)
    candidate_resource_ids : subset of resource IDs to score
    """
    if len(all_interactions) < settings.MIN_INTERACTIONS_FOR_CF:
        log.debug("Insufficient interactions for collaborative filtering.")
        return [CollaborativeScore(rid, 0.0) for rid in candidate_resource_ids]

    # ── Build user and resource index maps ──────────────────────────────────
    user_ids = sorted({i.user_id for i in all_interactions})
    resource_ids = sorted({i.resource_id for i in all_interactions})
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    res_idx = {rid: i for i, rid in enumerate(resource_ids)}

    # ── Populate sparse interaction matrix ──────────────────────────────────
    rows, cols, data = [], [], []
    for interaction in all_interactions:
        u = user_idx.get(interaction.user_id)
        r = res_idx.get(interaction.resource_id)
        if u is not None and r is not None:
            rows.append(u)
            cols.append(r)
            data.append(_interaction_to_signal(interaction))

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(resource_ids)))

    if target_user_id not in user_idx:
        log.debug("Target user has no interactions yet — CF returns zeros.")
        return [CollaborativeScore(rid, 0.0) for rid in candidate_resource_ids]

    target_row = user_idx[target_user_id]
    target_vec = matrix[target_row]

    # ── User-user similarity ─────────────────────────────────────────────────
    sims = cosine_similarity(target_vec, matrix).flatten()
    sims[target_row] = 0.0   # exclude self-similarity

    top_users: List[Tuple[int, float]] = sorted(
        enumerate(sims), key=lambda x: x[1], reverse=True
    )[:TOP_SIMILAR_USERS]

    # ── Score resources via weighted neighbour ratings ───────────────────────
    scores: Dict[str, float] = {}
    weights_sum: Dict[str, float] = {}

    for u_idx, sim_score in top_users:
        if sim_score <= 0:
            continue
        for r_id in candidate_resource_ids:
            if r_id not in res_idx:
                continue
            r_idx = res_idx[r_id]
            rating = matrix[u_idx, r_idx]
            if rating > 0:
                scores[r_id] = scores.get(r_id, 0.0) + sim_score * rating
                weights_sum[r_id] = weights_sum.get(r_id, 0.0) + sim_score

    results: List[CollaborativeScore] = []
    for r_id in candidate_resource_ids:
        if weights_sum.get(r_id, 0.0) > 0:
            norm_score = min(scores[r_id] / weights_sum[r_id], 1.0)
        else:
            norm_score = 0.0
        results.append(CollaborativeScore(resource_id=r_id, score=round(norm_score, 4)))

    return sorted(results, key=lambda x: x.score, reverse=True)
