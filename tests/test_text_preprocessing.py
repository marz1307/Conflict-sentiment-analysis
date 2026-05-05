"""Tests for text preprocessing."""

from __future__ import annotations

from conflict_sentiment.preprocessing import TextPreprocessor, clean_and_lemmatize


def test_clean_strips_urls_and_punctuation() -> None:
    text = "Visit https://example.com! Russia attacked Kyiv."
    cleaned = clean_and_lemmatize(text)
    assert "http" not in cleaned
    assert "!" not in cleaned
    assert "russia" in cleaned
    assert "attacked" not in cleaned  # lemmatized to attack
    assert "attack" in cleaned


def test_clean_lowercases_and_filters_stopwords() -> None:
    text = "The Soldiers were running toward the river."
    cleaned = clean_and_lemmatize(text)
    tokens = cleaned.split()
    assert "the" not in tokens
    assert "were" not in tokens
    assert "soldier" in tokens


def test_tokenize_returns_list() -> None:
    pre = TextPreprocessor()
    tokens = pre.tokenize("Tanks rolled across the border.")
    assert isinstance(tokens, list)
    assert "tank" in tokens
    assert "border" in tokens


def test_clean_handles_empty_input() -> None:
    assert clean_and_lemmatize("") == ""
    assert clean_and_lemmatize(None) == ""  # type: ignore[arg-type]
