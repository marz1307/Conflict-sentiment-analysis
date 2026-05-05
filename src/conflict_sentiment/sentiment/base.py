"""Abstract base class shared by every sentiment engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

SentimentLabel = Literal["Positive", "Negative", "Neutral"]


@dataclass(frozen=True)
class SentimentResult:
    """Single sentiment prediction.

    Attributes:
        score: Continuous polarity in ``[-1, 1]``. For transformer-style
            engines this is the signed confidence (positive class = +conf,
            negative = -conf, neutral = 0.0).
        label: Discrete category.
        confidence: Optional model-reported confidence (1.0 for lexicon
            engines that do not provide it explicitly).
    """

    score: float
    label: SentimentLabel
    confidence: float = 1.0


class SentimentEngine(ABC):
    """Abstract sentiment engine.

    Concrete subclasses implement :meth:`score` for a single text and may
    override :meth:`score_batch` for a more efficient batched implementation.
    """

    name: str = "abstract"

    @abstractmethod
    def score(self, text: str) -> SentimentResult:
        """Score a single text.

        Args:
            text: Input string.

        Returns:
            :class:`SentimentResult`.
        """

    def score_batch(self, texts: Iterable[str]) -> list[SentimentResult]:
        """Score an iterable of texts.

        Default implementation calls :meth:`score` per text. Override when a
        batched API offers a meaningful speedup (transformers, GPU inference).

        Args:
            texts: Iterable of input strings.

        Returns:
            List of :class:`SentimentResult`, in the same order as ``texts``.
        """
        return [self.score(t) for t in texts]

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        score_column: str | None = None,
        label_column: str | None = None,
    ) -> pd.DataFrame:
        """Score a dataframe column and append result columns.

        Args:
            df: Input frame.
            text_column: Column containing the text to score.
            score_column: Output column name for the continuous score.
                Defaults to ``f"{name}_sentiment_score"``.
            label_column: Output column name for the categorical label.
                Defaults to ``f"{name}_sentiment_category"``.

        Returns:
            New frame with score and label columns added.
        """
        score_col = score_column or f"{self.name}_sentiment_score"
        label_col = label_column or f"{self.name}_sentiment_category"
        results = self.score_batch(df[text_column].tolist())
        out = df.copy()
        out[score_col] = [r.score for r in results]
        out[label_col] = [r.label for r in results]
        return out
