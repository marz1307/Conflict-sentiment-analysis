"""Sentiment engines."""

from __future__ import annotations

from .base import SentimentEngine, SentimentResult
from .registry import ENGINE_REGISTRY, build_engine
from .textblob_engine import TextBlobEngine
from .transformer_engine import TransformerEngine
from .vader_engine import VaderEngine

__all__ = [
    "SentimentEngine",
    "SentimentResult",
    "TextBlobEngine",
    "VaderEngine",
    "TransformerEngine",
    "ENGINE_REGISTRY",
    "build_engine",
]
