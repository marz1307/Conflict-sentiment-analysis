"""TextBlob lexical baseline engine."""

from __future__ import annotations

from textblob import TextBlob

from .base import SentimentEngine, SentimentLabel, SentimentResult


class TextBlobEngine(SentimentEngine):
    """Naive Bayes / pattern-based polarity baseline.

    TextBlob returns continuous polarity in ``[-1, 1]``. The configured
    threshold defines a symmetric neutral band around zero.
    """

    name = "textblob"

    def __init__(self, threshold: float = 0.1) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold

    def _label(self, score: float) -> SentimentLabel:
        if score > self.threshold:
            return "Positive"
        if score < -self.threshold:
            return "Negative"
        return "Neutral"

    def score(self, text: str) -> SentimentResult:
        polarity = float(TextBlob(text or "").sentiment.polarity)
        return SentimentResult(score=polarity, label=self._label(polarity))
