"""
NLP Pipeline — unified entry point.

Runs sentiment, emotion, intent, and crisis detection in a single call
so each module doesn't need to be imported individually.
"""
from __future__ import annotations

from dataclasses import dataclass

from nlp.sentiment_analyzer import SentimentResult, analyze_sentiment
from nlp.emotion_extractor import EmotionResult, extract_emotions
from nlp.intent_classifier import IntentResult, classify_intent
from nlp.crisis_detector import CrisisResult, detect_crisis


@dataclass
class NLPAnalysis:
    sentiment: SentimentResult
    emotion: EmotionResult
    intent: IntentResult
    crisis: CrisisResult


def analyze(text: str, use_transformers: bool = True) -> NLPAnalysis:
    """
    Full NLP analysis pipeline.

    Returns a combined NLPAnalysis object with sentiment, emotion,
    intent, and crisis results.
    """
    sentiment = analyze_sentiment(text, use_transformer=use_transformers)
    emotion = extract_emotions(text)
    intent = classify_intent(text)
    crisis = detect_crisis(text, sentiment=sentiment, intent=intent)

    return NLPAnalysis(
        sentiment=sentiment,
        emotion=emotion,
        intent=intent,
        crisis=crisis,
    )
