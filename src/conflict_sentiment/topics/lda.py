"""Gensim LDA wrapper with reproducibility-friendly defaults."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from gensim import corpora, models

from ..config import LDAConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TopicSummary:
    """Top words for a single topic."""

    topic_id: int
    words: list[str]
    weights: list[float]


class LDATopicModel:
    """Wrapper around :class:`gensim.models.LdaModel`.

    Encapsulates dictionary construction, corpus building, training and
    document-topic assignment behind a small, typed surface.
    """

    def __init__(self, config: LDAConfig | None = None) -> None:
        self.config = config or LDAConfig()
        self.dictionary: corpora.Dictionary | None = None
        self.corpus: list[list[tuple[int, int]]] | None = None
        self.model: models.LdaModel | None = None

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> LDATopicModel:
        """Train the LDA model on a list of tokenized documents.

        Args:
            tokenized_docs: One list of tokens per document.

        Returns:
            ``self`` for chaining.
        """
        cfg = self.config
        self.dictionary = corpora.Dictionary(tokenized_docs)
        size_before = len(self.dictionary)
        self.dictionary.filter_extremes(no_below=cfg.no_below, no_above=cfg.no_above)
        size_after = len(self.dictionary)
        logger.info(
            "LDA dictionary filtered from %d to %d terms (no_below=%d, no_above=%.2f)",
            size_before,
            size_after,
            cfg.no_below,
            cfg.no_above,
        )

        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in tokenized_docs]
        self.model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=cfg.num_topics,
            random_state=cfg.random_state,
            passes=cfg.passes,
            alpha=cfg.alpha,
            per_word_topics=True,
        )
        logger.info("LDA model trained with %d topics, %d passes", cfg.num_topics, cfg.passes)
        return self

    def topic_summaries(self) -> list[TopicSummary]:
        """Return the top words and weights for each trained topic."""
        if self.model is None:
            raise RuntimeError("LDA model has not been fitted yet.")
        summaries: list[TopicSummary] = []
        for topic_id in range(self.config.num_topics):
            terms = self.model.show_topic(topicid=topic_id, topn=self.config.top_n_words)
            summaries.append(
                TopicSummary(
                    topic_id=topic_id,
                    words=[w for w, _ in terms],
                    weights=[float(p) for _, p in terms],
                )
            )
        return summaries

    def assign_dominant_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the input frame with dominant-topic columns.

        Args:
            df: Frame whose row count matches the corpus passed to :meth:`fit`.

        Returns:
            New frame with ``dominant_topic`` and ``dominant_topic_prob`` columns.
        """
        if self.model is None or self.corpus is None:
            raise RuntimeError("LDA model has not been fitted yet.")
        if len(df) != len(self.corpus):
            raise ValueError(
                f"DataFrame length {len(df)} does not match corpus length {len(self.corpus)}"
            )

        out = df.copy()
        dominant: list[int] = []
        prob: list[float] = []
        for doc_bow in self.corpus:
            topic_dist = self.model.get_document_topics(doc_bow)
            if not topic_dist:
                dominant.append(-1)
                prob.append(0.0)
                continue
            top_topic, top_prob = max(topic_dist, key=lambda x: x[1])
            dominant.append(int(top_topic))
            prob.append(float(top_prob))
        out["dominant_topic"] = dominant
        out["dominant_topic_prob"] = prob
        return out
