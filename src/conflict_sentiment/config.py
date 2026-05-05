"""Configuration loader.

Loads YAML configuration into typed Pydantic models. All tunable parameters
(paths, sentiment thresholds, LDA hyperparameters, transformer settings) are
declared here, never hard coded into pipeline modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class PathsConfig(BaseModel):
    raw_data: Path = Path("data/raw/Russia-Ukraine Conflict.xls")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    figures_dir: Path = Path("reports/figures")
    tables_dir: Path = Path("reports/tables")


class TextBlobConfig(BaseModel):
    threshold: float = 0.1


class VaderConfig(BaseModel):
    threshold: float = 0.05


class TransformerConfig(BaseModel):
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 128
    batch_size: int = 32
    cache_dir: str | None = None
    random_state: int = 42


class SentimentConfig(BaseModel):
    engines: list[str] = Field(default_factory=lambda: ["textblob", "vader", "transformer"])
    textblob: TextBlobConfig = Field(default_factory=TextBlobConfig)
    vader: VaderConfig = Field(default_factory=VaderConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)


class LDAConfig(BaseModel):
    num_topics: int = 5
    passes: int = 10
    no_below: int = 20
    no_above: float = 0.5
    alpha: str = "auto"
    random_state: int = 42
    top_n_words: int = 15


class PlottingConfig(BaseModel):
    style: str = "whitegrid"
    palette: str = "viridis"
    figsize_default: tuple[int, int] = (12, 8)
    dpi: int = 100


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    lda: LDAConfig = Field(default_factory=LDAConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration into an :class:`AppConfig`.

    Args:
        path: Path to a YAML file. If ``None``, falls back to
            ``configs/default.yaml`` and then to in-code defaults.

    Returns:
        Validated :class:`AppConfig` instance.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file %s not found, using built-in defaults.", config_path)
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    logger.info("Loaded configuration from %s", config_path)
    return AppConfig(**raw)
