"""Matplotlib plotting helpers with a consistent visual style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def apply_default_style(style: str = "whitegrid") -> None:
    """Set the seaborn style used across all plots."""
    sns.set_style(style)


def _save_or_show(path: Path | None) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_top_tokens(
    tokens: list[tuple[str, int]],
    title: str = "Top Tokens",
    save_path: Path | None = None,
) -> None:
    """Bar chart of the most frequent tokens."""
    if not tokens:
        return
    words, counts = zip(*tokens, strict=True)
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(words)), counts, color="steelblue", edgecolor="black", alpha=0.8)
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_publisher_sentiment(
    series: pd.Series,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of publisher (or country) sentiment scores."""
    plt.figure(figsize=(12, max(6, len(series) * 0.4)))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in series.values]
    series.plot(kind="barh", color=colors, edgecolor="black")
    plt.axvline(x=0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Mean signed sentiment score")
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_topic_sentiment(
    topic_sentiment: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """Bar chart of mean sentiment per topic plus article counts."""
    if "mean_sentiment" not in topic_sentiment.columns:
        raise ValueError("topic_sentiment must contain a 'mean_sentiment' column")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    topics = topic_sentiment.index
    means = topic_sentiment["mean_sentiment"].values
    colors = ["#c0392b" if s < 0 else "#27ae60" for s in means]
    axes[0].bar([f"Topic {t}" for t in topics], means, color=colors, edgecolor="black", alpha=0.8)
    axes[0].axhline(y=0, color="black", linewidth=1)
    axes[0].set_title("Average sentiment by topic")
    axes[0].set_ylabel("Mean signed sentiment score")
    axes[0].grid(alpha=0.3, axis="y")
    if "article_count" in topic_sentiment.columns:
        axes[1].bar(
            [f"Topic {t}" for t in topics],
            topic_sentiment["article_count"].values,
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        axes[1].set_title("Article count per topic")
        axes[1].set_ylabel("Article count")
        axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save_or_show(save_path)
