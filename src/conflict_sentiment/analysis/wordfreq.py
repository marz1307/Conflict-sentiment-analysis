"""Token frequency utilities for exploratory analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def token_frequencies(texts: Iterable[str]) -> Counter[str]:
    """Compute a frequency counter over whitespace-tokenized texts.

    Args:
        texts: Iterable of cleaned text strings.

    Returns:
        :class:`collections.Counter` mapping token -> frequency.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        if not text:
            continue
        counter.update(text.split())
    return counter


def top_n_tokens(texts: Iterable[str], n: int = 25) -> list[tuple[str, int]]:
    """Return the ``n`` most frequent tokens in ``texts``."""
    return token_frequencies(texts).most_common(n)
