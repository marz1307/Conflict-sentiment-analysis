"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from conflict_sentiment.preprocessing import ensure_nltk_resources

# Ensure NLTK assets are present once for the whole test session.
ensure_nltk_resources()


@pytest.fixture
def synthetic_news() -> pd.DataFrame:
    """Tiny 10-row news frame for fast unit tests."""
    rows = [
        (
            "Russia launches assault on Kyiv",
            "Heavy shelling reported in northern districts.",
            "Reuters",
            "United Kingdom",
        ),
        (
            "Ukraine receives humanitarian aid",
            "Aid convoys reach refugees on the border.",
            "BBC",
            "United Kingdom",
        ),
        (
            "EU agrees new sanctions on Moscow",
            "Energy and finance sectors targeted.",
            "Politico",
            "United States Of America",
        ),
        (
            "Peace talks scheduled for next week",
            "Diplomats announce second round of talks.",
            "Al Jazeera",
            "Qatar",
        ),
        (
            "Civilians flee combat zones",
            "Thousands displaced by ongoing fighting.",
            "The Guardian",
            "United Kingdom",
        ),
        (
            "Markets rally on ceasefire hopes",
            "Stocks gain as diplomats meet.",
            "Bloomberg",
            "United States Of America",
        ),
        (
            "Cyber attacks intensify across Europe",
            "Critical infrastructure targeted.",
            "Wired",
            "United States Of America",
        ),
        (
            "NATO reinforces eastern flank",
            "Additional troops deployed to member states.",
            "Reuters",
            "United Kingdom",
        ),
        (
            "Children evacuated to safer regions",
            "Schools moved to neighbouring countries.",
            "BBC",
            "United Kingdom",
        ),
        (
            "Energy prices surge across Europe",
            "Households brace for winter shortages.",
            "Financial Times",
            "United Kingdom",
        ),
    ]
    return pd.DataFrame(rows, columns=["TITLE", "DESCRIPTION", "PUBLISHER", "COUNTRY"])
