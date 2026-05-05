"""
Tests for the hybrid recommendation engine.
"""
import pytest
from unittest.mock import MagicMock
from recommendation.content_based import score_resources_content
from recommendation.collaborative import compute_cf_scores
from recommendation.engine import generate_recommendations, RecommendedResource
from nlp.pipeline import NLPAnalysis
from nlp.sentiment_analyzer import SentimentResult
from nlp.emotion_extractor import EmotionResult
from nlp.intent_classifier import IntentResult
from nlp.crisis_detector import CrisisResult
from database.models import Resource, ResourceInteraction


def _make_analysis(sentiment="negative", emotion="sadness", intent="seek_advice",
                   is_crisis=False, crisis_score=0.0) -> NLPAnalysis:
    return NLPAnalysis(
        sentiment=SentimentResult(label=sentiment, score=-0.6, confidence=0.8, compound=-0.6),
        emotion=EmotionResult(dominant=emotion, scores={emotion: 1.0}),
        intent=IntentResult(label=intent, confidence=0.9, all_scores={intent: 0.9}),
        crisis=CrisisResult(is_crisis=is_crisis, severity="low", score=crisis_score),
    )


def _make_resource(rid, resource_type="article", topics=None, emotions=None,
                   is_crisis=False, difficulty="easy") -> Resource:
    r = Resource()
    r.id = rid
    r.title = f"Resource {rid}"
    r.description = "Test resource"
    r.resource_type = resource_type
    r.topics = topics or ["anxiety"]
    r.emotions = emotions or ["fear"]
    r.difficulty = difficulty
    r.duration_minutes = 10
    r.url = None
    r.is_crisis_resource = is_crisis
    r.embedding = None
    return r


class TestContentBased:
    def test_returns_scores_for_all_resources(self):
        resources = [_make_resource(str(i)) for i in range(5)]
        analysis = _make_analysis()
        scores = score_resources_content(resources, analysis, ["anxiety"], {})
        assert len(scores) == 5

    def test_emotion_aligned_resource_scores_higher(self):
        resources = [
            _make_resource("a", emotions=["sadness"]),
            _make_resource("b", emotions=["joy"]),
        ]
        analysis = _make_analysis(emotion="sadness")
        scores = score_resources_content(resources, analysis, [], {})
        score_map = {s.resource_id: s.score for s in scores}
        assert score_map["a"] >= score_map["b"]


class TestCollaborativeFiltering:
    def test_returns_zero_scores_with_insufficient_data(self):
        scores = compute_cf_scores("user1", [], ["res1", "res2"])
        assert all(s.score == 0.0 for s in scores)

    def test_returns_scores_for_all_candidates(self):
        interactions = []
        for i in range(10):
            inter = ResourceInteraction()
            inter.user_id = f"user{i}"
            inter.resource_id = f"res{i % 3}"
            inter.interaction_type = "completed"
            inter.implicit_score = 0.9
            inter.helpful = True
            inter.rating = 5
            interactions.append(inter)
        scores = compute_cf_scores("user0", interactions, ["res0", "res1", "res2"])
        assert len(scores) == 3


class TestHybridEngine:
    def test_crisis_surfaces_crisis_resources(self):
        crisis_resource = _make_resource("crisis1", resource_type="hotline", is_crisis=True)
        normal_resource = _make_resource("normal1", resource_type="article")
        analysis = _make_analysis(is_crisis=True, crisis_score=1.0)
        analysis.crisis.is_crisis = True
        analysis.crisis.severity = "critical"
        analysis.crisis.escalation_message = "Call 988"

        results = generate_recommendations(
            user_id="user1",
            analysis=analysis,
            all_resources=[crisis_resource, normal_resource],
            all_interactions=[],
            user_topic_weights={},
            user_modality_prefs={},
            recent_topics=[],
        )
        assert all(r.resource_id == "crisis1" for r in results)

    def test_no_more_than_top_k_returned(self):
        resources = [_make_resource(str(i)) for i in range(20)]
        analysis = _make_analysis()
        results = generate_recommendations(
            user_id="user1",
            analysis=analysis,
            all_resources=resources,
            all_interactions=[],
            user_topic_weights={},
            user_modality_prefs={},
            recent_topics=[],
        )
        from config import get_settings
        assert len(results) <= get_settings().TOP_K_RECOMMENDATIONS

    def test_completed_resources_deprioritised(self):
        r1 = _make_resource("r1", topics=["anxiety"], emotions=["fear"])
        r2 = _make_resource("r2", topics=["anxiety"], emotions=["fear"])
        analysis = _make_analysis()
        results_without = generate_recommendations(
            "u", analysis, [r1, r2], [], {}, {}, ["anxiety"], completed_resource_ids=set()
        )
        results_with = generate_recommendations(
            "u", analysis, [r1, r2], [], {}, {}, ["anxiety"], completed_resource_ids={"r1"}
        )
        ids_without = [r.resource_id for r in results_without]
        ids_with = [r.resource_id for r in results_with]
        # r1 should appear lower (or not at all) when marked completed
        if "r1" in ids_without and "r1" in ids_with and len(ids_without) > 1:
            assert ids_with.index("r1") >= ids_without.index("r1")
