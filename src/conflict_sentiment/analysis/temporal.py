"""Time-bucketed sentiment trends."""

from __future__ import annotations

import pandas as pd


def sentiment_over_time(
    df: pd.DataFrame,
    date_column: str = "PUBLISHED DATE (UTC)",
    score_column: str = "transformer_sentiment_score",
    freq: str = "D",
) -> pd.DataFrame:
    """Aggregate mean sentiment per time bucket.

    Args:
        df: Frame with a parseable date column.
        date_column: Column containing publication timestamps.
        score_column: Continuous score column to aggregate.
        freq: Pandas offset alias, e.g. ``"D"``, ``"W"``, ``"M"``.

    Returns:
        Frame indexed by period with ``mean_sentiment`` and ``article_count``.
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column missing: {date_column}")
    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce", utc=True)
    out = out.dropna(subset=[date_column])
    grouped = out.set_index(date_column).groupby(pd.Grouper(freq=freq))
    summary = (
        grouped[score_column]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_sentiment", "count": "article_count"})
    )
    return summary.dropna(subset=["mean_sentiment"])
