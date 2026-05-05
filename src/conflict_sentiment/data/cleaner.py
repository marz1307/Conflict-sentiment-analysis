"""Dataframe-level cleaning: text consolidation and deduplication."""

from __future__ import annotations

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TEXT_COLUMNS = ("TITLE", "DESCRIPTION")
COMBINED_TEXT_FIELD = "combined_text"


def build_combined_text(
    df: pd.DataFrame,
    columns: tuple[str, ...] = DEFAULT_TEXT_COLUMNS,
) -> pd.DataFrame:
    """Concatenate available text columns into ``combined_text``.

    Missing values are coerced to empty strings so no row is dropped solely
    because of an absent description.

    Args:
        df: Input frame.
        columns: Tuple of source columns to concatenate.

    Returns:
        Frame with a ``combined_text`` column appended.
    """
    out = df.copy()
    available = [c for c in columns if c in out.columns]
    if not available:
        raise ValueError(f"None of the requested columns are present: {columns}")
    for col in available:
        out[col] = out[col].fillna("")
    out[COMBINED_TEXT_FIELD] = out[available].agg(" ".join, axis=1).str.strip()
    return out


def clean_dataframe(
    df: pd.DataFrame,
    columns: tuple[str, ...] = DEFAULT_TEXT_COLUMNS,
) -> pd.DataFrame:
    """Run the full dataframe cleaning routine.

    Steps:
        1. Build a ``combined_text`` field from title and description.
        2. Drop rows whose combined text is empty.
        3. Deduplicate on combined text.

    Args:
        df: Input frame.
        columns: Source text columns (defaults to TITLE and DESCRIPTION).

    Returns:
        Cleaned frame with reset index.
    """
    out = build_combined_text(df, columns=columns)
    before = len(out)
    out = out[out[COMBINED_TEXT_FIELD].str.len() > 0].reset_index(drop=True)
    empty_removed = before - len(out)

    dup_count = int(out.duplicated(subset=[COMBINED_TEXT_FIELD]).sum())
    out = out.drop_duplicates(subset=[COMBINED_TEXT_FIELD]).reset_index(drop=True)

    logger.info(
        "Cleaned dataframe: %d input rows, removed %d empty, %d duplicates, %d retained",
        before,
        empty_removed,
        dup_count,
        len(out),
    )
    return out
