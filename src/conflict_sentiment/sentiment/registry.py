"""Engine registry and factory.

Lets the CLI and tests instantiate engines from a string name plus a typed
config block, without importing the heavy transformer module unless requested.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import SentimentConfig
from .base import SentimentEngine
from .textblob_engine import TextBlobEngine
from .vader_engine import VaderEngine


def _build_textblob(cfg: SentimentConfig) -> SentimentEngine:
    return TextBlobEngine(threshold=cfg.textblob.threshold)


def _build_vader(cfg: SentimentConfig) -> SentimentEngine:
    return VaderEngine(threshold=cfg.vader.threshold)


def _build_transformer(cfg: SentimentConfig) -> SentimentEngine:
    from .transformer_engine import TransformerEngine

    t = cfg.transformer
    return TransformerEngine(
        model_name=t.model_name,
        max_length=t.max_length,
        batch_size=t.batch_size,
        cache_dir=t.cache_dir,
        random_state=t.random_state,
    )


ENGINE_REGISTRY: dict[str, Callable[[SentimentConfig], SentimentEngine]] = {
    "textblob": _build_textblob,
    "vader": _build_vader,
    "transformer": _build_transformer,
}


def build_engine(name: str, config: SentimentConfig) -> SentimentEngine:
    """Build a sentiment engine by registry name.

    Args:
        name: Engine identifier (``"textblob"``, ``"vader"``, ``"transformer"``).
        config: Sentiment configuration block.

    Returns:
        Instantiated :class:`SentimentEngine`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in ENGINE_REGISTRY:
        raise KeyError(f"Unknown sentiment engine: {name}")
    return ENGINE_REGISTRY[name](config)
