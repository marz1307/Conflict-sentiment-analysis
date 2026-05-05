"""Idempotent NLTK resource setup.

Avoids the import-time blanket downloads of the legacy notebook. Call
:func:`ensure_nltk_resources` once from CLI entrypoints.
"""

from __future__ import annotations

import nltk

from ..logging_utils import get_logger

logger = get_logger(__name__)

_REQUIRED = (
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("sentiment/vader_lexicon", "vader_lexicon"),
)


def ensure_nltk_resources(quiet: bool = True) -> None:
    """Ensure the NLTK resources used by this package are available locally.

    Idempotent: each resource is downloaded only if it cannot already be found.

    Args:
        quiet: Suppress NLTK's verbose download chatter.
    """
    for resource_path, package in _REQUIRED:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", package)
            nltk.download(package, quiet=quiet)
