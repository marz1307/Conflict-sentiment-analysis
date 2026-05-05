"""End-to-end pipelines."""

from __future__ import annotations

from .run_sentiment import run_sentiment_pipeline
from .run_topics import run_topics_pipeline

__all__ = ["run_sentiment_pipeline", "run_topics_pipeline"]
