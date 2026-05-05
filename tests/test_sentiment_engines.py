"""Tests for sentiment engines (transformer mocked)."""

from __future__ import annotations

import pandas as pd

from conflict_sentiment.config import SentimentConfig
from conflict_sentiment.sentiment import (
    SentimentResult,
    TextBlobEngine,
    VaderEngine,
    build_engine,
)
from conflict_sentiment.sentiment.base import SentimentEngine


def test_textblob_returns_in_range() -> None:
    engine = TextBlobEngine(threshold=0.1)
    res = engine.score("This is a wonderful and uplifting story.")
    assert -1.0 <= res.score <= 1.0
    assert res.label in {"Positive", "Negative", "Neutral"}


def test_textblob_threshold_neutralises_weak_polarity() -> None:
    engine = TextBlobEngine(threshold=0.9)
    res = engine.score("This is a fine day.")
    assert res.label == "Neutral"


def test_vader_negative_label_on_violent_text() -> None:
    engine = VaderEngine(threshold=0.05)
    res = engine.score("Devastating attack kills civilians and destroys homes.")
    assert res.label == "Negative"
    assert res.score < 0


def test_vader_handles_empty_string() -> None:
    engine = VaderEngine(threshold=0.05)
    res = engine.score("")
    assert res.label == "Neutral"


def test_score_dataframe_appends_columns(synthetic_news: pd.DataFrame) -> None:
    engine = TextBlobEngine()
    df = engine.score_dataframe(
        synthetic_news.assign(text=synthetic_news["TITLE"]),
        text_column="text",
    )
    assert "textblob_sentiment_score" in df.columns
    assert "textblob_sentiment_category" in df.columns
    assert len(df) == len(synthetic_news)


def test_registry_resolves_lexical_engines() -> None:
    cfg = SentimentConfig()
    assert isinstance(build_engine("textblob", cfg), TextBlobEngine)
    assert isinstance(build_engine("vader", cfg), VaderEngine)


class _FakeTransformer(SentimentEngine):
    """Stand-in transformer that does not download HF weights."""

    name = "transformer"

    def score(self, text: str) -> SentimentResult:
        text_lower = (text or "").lower()
        if any(w in text_lower for w in ("attack", "kill", "destroy", "war")):
            return SentimentResult(score=-0.85, label="Negative", confidence=0.85)
        if any(w in text_lower for w in ("peace", "aid", "rally", "hope")):
            return SentimentResult(score=0.65, label="Positive", confidence=0.65)
        return SentimentResult(score=0.0, label="Neutral", confidence=0.7)


def test_fake_transformer_round_trip(synthetic_news: pd.DataFrame) -> None:
    engine = _FakeTransformer()
    df = engine.score_dataframe(
        synthetic_news.assign(text=synthetic_news["TITLE"] + " " + synthetic_news["DESCRIPTION"]),
        text_column="text",
    )
    assert "transformer_sentiment_score" in df.columns
    assert df["transformer_sentiment_score"].between(-1, 1).all()
    assert set(df["transformer_sentiment_category"]).issubset({"Positive", "Negative", "Neutral"})


def test_transformer_engine_lazy_load(monkeypatch) -> None:
    """The transformer must not fetch weights at construction time."""
    from conflict_sentiment.sentiment.transformer_engine import TransformerEngine

    engine = TransformerEngine(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
    assert engine._model is None  # not loaded yet


def test_transformer_engine_uses_mock(monkeypatch) -> None:
    """End-to-end test of the transformer using mocked HF objects."""
    from conflict_sentiment.sentiment import transformer_engine as te

    class _StubTokenizer:
        @classmethod
        def from_pretrained(cls, *_a, **_kw):
            return cls()

        def __call__(self, texts, **_kw):
            import torch

            return {"input_ids": torch.zeros((len(texts), 4), dtype=torch.long)}

    class _StubOutput:
        def __init__(self, logits):
            self.logits = logits

    class _StubModel:
        config = type("C", (), {"id2label": {0: "negative", 1: "neutral", 2: "positive"}})()

        @classmethod
        def from_pretrained(cls, *_a, **_kw):
            return cls()

        def to(self, *_a, **_kw):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            import torch

            n = kwargs["input_ids"].shape[0]
            logits = torch.tensor([[0.1, 0.2, 0.7]] * n)
            return _StubOutput(logits)

    monkeypatch.setattr(te, "AutoTokenizer", _StubTokenizer, raising=False)
    monkeypatch.setattr(te, "AutoModelForSequenceClassification", _StubModel, raising=False)
    # Patch the symbols inside the lazy-loaded transformers import too.
    import transformers as hf

    monkeypatch.setattr(hf, "AutoTokenizer", _StubTokenizer, raising=False)
    monkeypatch.setattr(hf, "AutoModelForSequenceClassification", _StubModel, raising=False)

    engine = te.TransformerEngine(batch_size=2)
    results = engine.score_batch(["hello", "world", "test"])
    assert len(results) == 3
    assert all(r.label == "Positive" for r in results)
    assert all(r.score > 0 for r in results)
