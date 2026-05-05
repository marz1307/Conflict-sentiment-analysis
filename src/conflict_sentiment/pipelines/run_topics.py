"""End-to-end LDA topic modelling pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import AppConfig, load_config
from ..data import clean_dataframe, load_dataset
from ..logging_utils import get_logger
from ..preprocessing import TextPreprocessor, ensure_nltk_resources
from ..topics import LDATopicModel

logger = get_logger(__name__)

CLEAN_TEXT_COLUMN = "advanced_clean_text"
TOKEN_COLUMN = "lda_tokens"


def run_topics_pipeline(
    config: AppConfig | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run load, clean, LDA training, and persist topic assignments.

    Args:
        config: Pre-loaded application config.
        config_path: YAML path used when ``config`` is None.
        output_dir: Override for the output directory. When ``None``, results
            are written under ``paths.processed_dir``.

    Returns:
        The dataframe with dominant-topic columns appended.
    """
    cfg = config or load_config(config_path)
    ensure_nltk_resources()

    df = load_dataset(cfg.paths.raw_data)
    df = clean_dataframe(df)

    preprocessor = TextPreprocessor()
    df[CLEAN_TEXT_COLUMN] = df["combined_text"].apply(preprocessor.clean)
    df[TOKEN_COLUMN] = df[CLEAN_TEXT_COLUMN].apply(lambda t: t.split())

    lda = LDATopicModel(cfg.lda).fit(df[TOKEN_COLUMN].tolist())
    df = lda.assign_dominant_topics(df)

    out_dir = Path(output_dir) if output_dir else cfg.paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "topic_assignments.csv", index=False)

    summaries = [
        {"topic_id": s.topic_id, "words": s.words, "weights": s.weights}
        for s in lda.topic_summaries()
    ]
    (out_dir / "topic_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    logger.info("Topic outputs written to %s", out_dir)
    return df
