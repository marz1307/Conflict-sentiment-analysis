"""Tests for the data loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from conflict_sentiment.data import load_dataset


def test_load_csv_strips_columns(tmp_path: Path) -> None:
    csv = tmp_path / "sample.csv"
    pd.DataFrame([{" TITLE ": "T", " DESCRIPTION ": "D", "PUBLISHER": "P", "COUNTRY": "C"}]).to_csv(
        csv, index=False
    )
    df = load_dataset(csv)
    assert "TITLE" in df.columns
    assert "DESCRIPTION" in df.columns


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_load_missing_text_columns_raises(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    pd.DataFrame([{"FOO": "bar"}]).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_dataset(csv)


def test_load_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "x.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad)
