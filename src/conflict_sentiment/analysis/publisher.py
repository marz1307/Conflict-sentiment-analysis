"""Publisher and country level sentiment aggregations."""

from __future__ import annotations

import pandas as pd


def publisher_sentiment(
    df: pd.DataFrame,
    score_column: str = "transformer_sentiment_score",
    publisher_column: str = "PUBLISHER",
    min_articles: int = 1,
) -> pd.Series:
    """Mean sentiment per publisher, sorted ascending.

    Args:
        df: Frame containing per-article sentiment scores.
        score_column: Continuous score column to aggregate.
        publisher_column: Grouping column.
        min_articles: Drop publishers with fewer than this many articles.

    Returns:
        Series indexed by publisher, sorted from most negative to most positive.
    """
    grouped = df.groupby(publisher_column)[score_column]
    counts = grouped.size()
    means = grouped.mean()
    keep = counts[counts >= min_articles].index
    return means.loc[keep].sort_values()


def country_sentiment(
    df: pd.DataFrame,
    score_column: str = "transformer_sentiment_score",
    country_column: str = "COUNTRY",
) -> pd.Series:
    """Mean sentiment per publisher country, sorted ascending."""
    return df.groupby(country_column)[score_column].mean().sort_values()
