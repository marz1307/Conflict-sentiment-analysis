"""Text preprocessing utilities."""

from __future__ import annotations

from .nltk_setup import ensure_nltk_resources
from .text import TextPreprocessor, clean_and_lemmatize

__all__ = ["TextPreprocessor", "clean_and_lemmatize", "ensure_nltk_resources"]
