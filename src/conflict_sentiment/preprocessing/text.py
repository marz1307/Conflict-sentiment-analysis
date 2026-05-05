"""Token-level preprocessing: URL stripping, tokenization, lemmatization."""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

URL_PATTERN = re.compile(r"http\S+")
NON_LETTER_PATTERN = re.compile(r"[^A-Za-z\s]")


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


class TextPreprocessor:
    """Reusable preprocessing pipeline.

    The pipeline mirrors the legacy notebook stages:
        1. URL removal
        2. Non-letter character stripping
        3. Lowercasing
        4. Regex tokenization (``\\w+``)
        5. English stopword filtering
        6. WordNet lemmatization
    """

    def __init__(self, extra_stopwords: set[str] | None = None) -> None:
        self._tokenizer = RegexpTokenizer(r"\w+")
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords: frozenset[str] = _english_stopwords()
        if extra_stopwords:
            self._stopwords = self._stopwords | frozenset(extra_stopwords)

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline and return whitespace-joined lemmas.

        Args:
            text: Raw input string.

        Returns:
            Cleaned, lemmatized text as a single space-joined string.
        """
        if not isinstance(text, str) or not text:
            return ""
        text = URL_PATTERN.sub(" ", text)
        text = NON_LETTER_PATTERN.sub(" ", text)
        text = text.lower()
        tokens = self._tokenizer.tokenize(text)
        tokens = [t for t in tokens if t not in self._stopwords]
        lemmas = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(lemmas)

    def tokenize(self, text: str) -> list[str]:
        """Return cleaned tokens as a list (used by topic modelling)."""
        return self.clean(text).split()


def clean_and_lemmatize(text: str) -> str:
    """Module-level convenience wrapper around :class:`TextPreprocessor`."""
    return TextPreprocessor().clean(text)
