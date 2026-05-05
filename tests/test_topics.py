"""Tests for the LDA topic model wrapper."""

from __future__ import annotations

import pandas as pd

from conflict_sentiment.config import LDAConfig
from conflict_sentiment.topics import LDATopicModel


def _toy_corpus() -> list[list[str]]:
    base_war = ["russia", "attack", "soldier", "tank", "front", "kyiv"]
    base_diplo = ["sanction", "trade", "policy", "minister", "agreement", "summit"]
    base_aid = ["refugee", "aid", "shelter", "child", "border", "humanitarian"]
    docs: list[list[str]] = []
    for _ in range(8):
        docs.append(base_war)
    for _ in range(8):
        docs.append(base_diplo)
    for _ in range(8):
        docs.append(base_aid)
    return docs


def test_lda_fits_and_summarises() -> None:
    cfg = LDAConfig(num_topics=3, passes=5, no_below=2, no_above=0.8, top_n_words=5)
    model = LDATopicModel(cfg).fit(_toy_corpus())
    summaries = model.topic_summaries()
    assert len(summaries) == 3
    for s in summaries:
        assert len(s.words) == 5
        assert all(isinstance(w, str) for w in s.words)


def test_lda_assigns_dominant_topics() -> None:
    cfg = LDAConfig(num_topics=3, passes=5, no_below=2, no_above=0.8)
    docs = _toy_corpus()
    df = pd.DataFrame({"text": [" ".join(d) for d in docs]})
    model = LDATopicModel(cfg).fit(docs)
    out = model.assign_dominant_topics(df)
    assert "dominant_topic" in out.columns
    assert "dominant_topic_prob" in out.columns
    assert out["dominant_topic"].between(0, 2).all()
