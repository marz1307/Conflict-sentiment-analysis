"""Word cloud rendering."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from wordcloud import WordCloud


def build_wordcloud(
    texts: Iterable[str],
    width: int = 1200,
    height: int = 600,
    background_color: str = "white",
    colormap: str = "viridis",
    max_words: int = 100,
    save_path: Path | None = None,
) -> WordCloud:
    """Build (and optionally save) a wordcloud over the given texts.

    Args:
        texts: Iterable of cleaned text strings.
        width: Image width in pixels.
        height: Image height in pixels.
        background_color: Wordcloud background colour.
        colormap: Matplotlib colormap name.
        max_words: Maximum number of words to render.
        save_path: Optional output PNG path.

    Returns:
        The trained :class:`wordcloud.WordCloud` object.
    """
    combined = " ".join(t for t in texts if t)
    if not combined.strip():
        raise ValueError("Cannot build a word cloud from empty input.")
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
    ).generate(combined)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        wc.to_file(str(save_path))
    return wc
