"""Tests for dataframe cleaning."""

from __future__ import annotations

import pandas as pd

from conflict_sentiment.data import build_combined_text, clean_dataframe


def test_build_combined_text_concatenates() -> None:
    df = pd.DataFrame({"TITLE": ["A"], "DESCRIPTION": ["B"]})
    out = build_combined_text(df)
    assert out["combined_text"].iloc[0] == "A B"


def test_clean_dataframe_drops_duplicates() -> None:
    df = pd.DataFrame(
        {
            "TITLE": ["Same", "Same", "Different"],
            "DESCRIPTION": ["text", "text", "other"],
        }
    )
    out = clean_dataframe(df)
    assert len(out) == 2


def test_clean_dataframe_removes_empty_rows() -> None:
    df = pd.DataFrame({"TITLE": ["", "real"], "DESCRIPTION": ["", "story"]})
    out = clean_dataframe(df)
    assert len(out) == 1
    assert out["combined_text"].iloc[0] == "real story"
