"""End-to-end sentiment pipeline: load, clean, score, persist."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import AppConfig, load_config
from ..data import clean_dataframe, load_dataset
from ..logging_utils import get_logger
from ..preprocessing import TextPreprocessor, ensure_nltk_resources
from ..sentiment import build_engine

logger = get_logger(__name__)

CLEAN_TEXT_COLUMN = "advanced_clean_text"


def run_sentiment_pipeline(
    config: AppConfig | None = None,
    config_path: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run load, clean, multi-engine sentiment scoring, and persist results.

    Args:
        config: Pre-loaded application config. If ``None``, loaded from
            ``config_path`` or the default location.
        config_path: Path to a YAML config (used only when ``config`` is None).
        output_csv: Optional override for the output path. When ``None``, the
            CSV is written under ``paths.processed_dir``.

    Returns:
        The annotated dataframe.
    """
    cfg = config or load_config(config_path)
    ensure_nltk_resources()

    df = load_dataset(cfg.paths.raw_data)
    df = clean_dataframe(df)

    preprocessor = TextPreprocessor()
    logger.info("Applying text preprocessing to %d rows", len(df))
    df[CLEAN_TEXT_COLUMN] = df["combined_text"].apply(preprocessor.clean)

    for engine_name in cfg.sentiment.engines:
        logger.info("Scoring with engine: %s", engine_name)
        engine = build_engine(engine_name, cfg.sentiment)
        df = engine.score_dataframe(df, text_column=CLEAN_TEXT_COLUMN)

    output = Path(output_csv) if output_csv else (cfg.paths.processed_dir / "sentiment_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Wrote sentiment results to %s", output)
    return df
