"""VADER lexicon engine, tuned to news and social media language."""

from __future__ import annotations

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .base import SentimentEngine, SentimentLabel, SentimentResult


class VaderEngine(SentimentEngine):
    """VADER compound-score sentiment engine."""

    name = "vader"

    def __init__(self, threshold: float = 0.05) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self._analyzer = SentimentIntensityAnalyzer()

    def _label(self, score: float) -> SentimentLabel:
        if score >= self.threshold:
            return "Positive"
        if score <= -self.threshold:
            return "Negative"
        return "Neutral"

    def score(self, text: str) -> SentimentResult:
        compound = float(self._analyzer.polarity_scores(text or "")["compound"])
        return SentimentResult(score=compound, label=self._label(compound))
