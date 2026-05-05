"""
Tests for the NLP pipeline — sentiment, emotion, intent, crisis detection.
"""
import pytest
from nlp.sentiment_analyzer import analyze_sentiment_vader, SentimentResult
from nlp.crisis_detector import detect_crisis
from nlp.intent_classifier import _rule_based_intent
from nlp.emotion_extractor import _rule_based_emotion


# ── Sentiment ─────────────────────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_positive_sentiment(self):
        result = analyze_sentiment_vader("I feel wonderful and so grateful today!")
        assert result.label == "positive"
        assert result.score > 0

    def test_negative_sentiment(self):
        result = analyze_sentiment_vader("Everything is hopeless and I feel terrible.")
        assert result.label == "negative"
        assert result.score < 0

    def test_neutral_sentiment(self):
        result = analyze_sentiment_vader("I went to the store today.")
        assert result.label == "neutral"


# ── Crisis Detection ──────────────────────────────────────────────────────────

class TestCrisisDetector:
    def test_critical_phrase_triggers_crisis(self):
        result = detect_crisis("I want to kill myself tonight.")
        assert result.is_crisis is True
        assert result.severity == "critical"
        assert result.score == 1.0

    def test_safe_message_no_crisis(self):
        result = detect_crisis("I had a good day at work.")
        assert result.is_crisis is False
        assert result.severity == "low"

    def test_moderate_crisis(self):
        result = detect_crisis("I feel completely hopeless and worthless.")
        assert result.severity in ("moderate", "high")

    def test_escalation_message_present_for_crisis(self):
        result = detect_crisis("I plan to end my life.")
        assert result.escalation_message != ""

    def test_no_escalation_for_safe_message(self):
        result = detect_crisis("I'm a bit stressed about work.")
        assert result.escalation_message == ""


# ── Intent Classification ─────────────────────────────────────────────────────

class TestIntentClassifier:
    def test_greeting_intent(self):
        result = _rule_based_intent("Hello! How are you?")
        assert result.label == "greeting"

    def test_advice_intent(self):
        result = _rule_based_intent("What should I do to calm down?")
        assert result.label == "seek_advice"

    def test_farewell_intent(self):
        result = _rule_based_intent("Goodbye, thanks for your help.")
        assert result.label in ("farewell", "gratitude")


# ── Emotion Extraction ────────────────────────────────────────────────────────

class TestEmotionExtractor:
    def test_sadness_detected(self):
        result = _rule_based_emotion("I feel so sad and hopeless lately.")
        assert result.dominant == "sadness"

    def test_fear_detected(self):
        result = _rule_based_emotion("I'm terrified and extremely anxious.")
        assert result.dominant == "fear"

    def test_joy_detected(self):
        result = _rule_based_emotion("I'm so happy and excited today!")
        assert result.dominant == "joy"

    def test_scores_sum_to_one(self):
        result = _rule_based_emotion("I feel angry and disgusted.")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01
