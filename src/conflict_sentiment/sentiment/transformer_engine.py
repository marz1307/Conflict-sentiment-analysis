"""Contextual transformer engine (CardiffNLP RoBERTa)."""

from __future__ import annotations

import os
from collections.abc import Iterable

from ..logging_utils import get_logger
from .base import SentimentEngine, SentimentLabel, SentimentResult

logger = get_logger(__name__)

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _normalize_label(raw: str) -> SentimentLabel:
    raw_lower = raw.lower()
    if "neg" in raw_lower:
        return "Negative"
    if "pos" in raw_lower:
        return "Positive"
    return "Neutral"


class TransformerEngine(SentimentEngine):
    """RoBERTa-based contextual sentiment engine.

    The transformer outputs three-class probabilities (negative / neutral /
    positive). The continuous ``score`` field on each result is the *signed*
    confidence: ``+conf`` for positive, ``-conf`` for negative, ``0.0`` for
    neutral. This mirrors the convention used in the legacy notebook and
    enables aggregate comparison with TextBlob and VADER.
    """

    name = "transformer"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        batch_size: int = 32,
        cache_dir: str | None = None,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")
        self.random_state = random_state
        self._device_override = device
        self._model = None
        self._tokenizer = None
        self._device = None
        self._id2label: dict[int, str] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        # Local imports keep torch off the import path for tests that do not
        # need the transformer.
        import numpy as np  # noqa: F401  (used by score_batch when loaded)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.random_state)

        device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading transformer %s onto %s", self.model_name, device)

        kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._id2label = dict(self._model.config.id2label)

    def score(self, text: str) -> SentimentResult:
        return self.score_batch([text])[0]

    def score_batch(self, texts: Iterable[str]) -> list[SentimentResult]:
        self._load()
        import numpy as np
        import torch

        text_list = [t if isinstance(t, str) else "" for t in texts]
        results: list[SentimentResult] = []

        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            encodings = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encodings = {k: v.to(self._device) for k, v in encodings.items()}
            with torch.no_grad():
                logits = self._model(**encodings).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            indices = np.argmax(probs, axis=1)
            confidences = probs.max(axis=1)
            for idx, conf in zip(indices, confidences, strict=True):
                raw_label = self._id2label[int(idx)]
                label = _normalize_label(raw_label)
                signed = (
                    float(conf)
                    if label == "Positive"
                    else (-float(conf) if label == "Negative" else 0.0)
                )
                results.append(SentimentResult(score=signed, label=label, confidence=float(conf)))
        return results
