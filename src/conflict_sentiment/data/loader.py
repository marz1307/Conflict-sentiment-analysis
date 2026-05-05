"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

_EXPECTED_COLUMNS = {"TITLE", "DESCRIPTION", "PUBLISHER", "COUNTRY"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the conflict news dataset from disk.

    Supports ``.xls``, ``.xlsx`` and ``.csv`` based on the file suffix.

    Args:
        path: Path to the source file.

    Returns:
        DataFrame with column names stripped of leading/trailing whitespace.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is not supported or required text
            columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    df.columns = df.columns.str.strip()

    text_cols = {"TITLE", "DESCRIPTION"}
    if not text_cols.issubset(df.columns):
        missing = text_cols - set(df.columns)
        raise ValueError(f"Dataset missing required text columns: {missing}")

    logger.info("Loaded %d rows from %s", len(df), file_path)
    return df
