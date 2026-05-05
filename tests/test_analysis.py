"""Tests for analysis aggregation helpers."""

from __future__ import annotations

import pandas as pd

from conflict_sentiment.analysis import (
    country_sentiment,
    publisher_sentiment,
    sentiment_over_time,
    top_n_tokens,
)


def test_publisher_sentiment_orders_ascending() -> None:
    df = pd.DataFrame(
        {
            "PUBLISHER": ["A", "A", "B", "B", "C"],
            "transformer_sentiment_score": [-0.5, -0.3, 0.2, 0.4, 0.0],
        }
    )
    series = publisher_sentiment(df)
    assert list(series.index) == ["A", "C", "B"]


def test_country_sentiment_returns_series() -> None:
    df = pd.DataFrame(
        {"COUNTRY": ["UK", "US", "UK"], "transformer_sentiment_score": [-0.1, 0.2, 0.3]}
    )
    series = country_sentiment(df)
    assert set(series.index) == {"UK", "US"}


def test_sentiment_over_time_groups_by_day() -> None:
    df = pd.DataFrame(
        {
            "PUBLISHED DATE (UTC)": ["2022-02-24", "2022-02-24", "2022-02-25"],
            "transformer_sentiment_score": [-0.5, -0.3, 0.1],
        }
    )
    out = sentiment_over_time(df)
    assert "mean_sentiment" in out.columns
    assert "article_count" in out.columns
    assert len(out) == 2


def test_top_n_tokens_basic() -> None:
    texts = ["russia ukraine war", "russia attack", "ukraine peace"]
    top = top_n_tokens(texts, n=2)
    words = [w for w, _ in top]
    assert "russia" in words
