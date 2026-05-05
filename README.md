<div align="center">

# Conflict Sentiment

**Multi-engine sentiment analysis and topic discovery over Russia and Ukraine conflict news coverage.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/ci-github%20actions-blue.svg)](.github/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Overview

This project quantifies how 8,158 English-language news articles framed the Russia and Ukraine conflict across 68 publishers and 18 countries. It runs three complementary sentiment engines (TextBlob, VADER, CardiffNLP RoBERTa) and a five-topic LDA model over the same corpus, then surfaces publisher, country, and topic level differences. The transformer's signed sentiment score is the primary cross-publisher metric; the lexicon engines provide an interpretable baseline.

## Approach

Three engines, three angles:

- **TextBlob** (lexical baseline). Naive Bayes / pattern polarity from a movie-review prior. Cheap, interpretable, but misaligned with conflict language.
- **VADER** (social and news lexicon). Handles intensifiers, negation, and the punctuation patterns common in headlines.
- **CardiffNLP RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`). Contextual transformer fine-tuned on roughly 124M tweets. Captures interactions that lexicons miss, and is the only engine producing calibrated three-class probabilities. Signed confidence (positive = +conf, negative = -conf, neutral = 0) is used for cross-publisher aggregation.

Disagreement between engines is itself signal: where lexicons and the transformer diverge, manual review is warranted before drawing inference.

## What the pipeline produces

Running the pipelines on your machine emits:

- Per-article sentiment scores from each engine (TextBlob polarity, VADER compound, RoBERTa signed score), persisted as CSV
- Aggregate sentiment distribution (mean, std, positive/neutral/negative percentages) per engine
- LDA topic assignments and topic summaries (top terms per topic, article counts, mean sentiment per topic)
- Publisher and country level mean sentiment tables
- Daily temporal sentiment trend
- Word frequency tables and word clouds

To reproduce on your machine:

```bash
make install
make download-nltk
make run-sentiment   # writes per-article + aggregate sentiment to data/processed/
make run-topics      # writes topic assignments + summaries to data/processed/
```

Outputs land in `data/processed/` (CSVs, JSON) and `reports/figures/` (charts). Both are gitignored, so each clone produces fresh artifacts. The transformer engine downloads `cardiffnlp/twitter-roberta-base-sentiment-latest` on first run and caches it under the standard Hugging Face cache directory.

## Architecture

```
                         configs/default.yaml
                                  |
                                  v
data/raw/*.xls -> loader -> cleaner -> preprocessor (URL strip, lowercase, tokenize, stopword, lemmatize)
                                                    |
                       +----------------------------+----------------------------+
                       |                            |                            |
                       v                            v                            v
                 TextBlobEngine             VaderEngine              TransformerEngine (RoBERTa)
                       \\                          |                           /
                        \\                         v                          /
                         +------------> sentiment_results.csv <-------------+
                                                  |
                              +-------------------+--------------------+
                              v                                        v
                    LDATopicModel (gensim)                     analysis (publisher,
                              |                                country, temporal,
                              v                                wordfreq)
                    topic_assignments.csv                              |
                    topic_summaries.json                               v
                                                              reports/figures
```

## Quickstart

```bash
git clone <your-fork-url> conflict-sentiment
cd conflict-sentiment
python -m venv .venv && source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
make download-nltk
make run-sentiment
make run-topics
```

The first sentiment run will download the RoBERTa weights to the local Hugging Face cache (configurable via `HF_HOME`). Subsequent runs reuse the cached model.

## Project structure

```
ukraine-russia/
  configs/                YAML configuration
  data/
    raw/                  source XLS (not committed)
    processed/            generated CSVs
  reports/
    figures/              published charts
    tables/               aggregate tables
  src/conflict_sentiment/
    data/                 loader, cleaner
    preprocessing/        text pipeline, NLTK setup
    sentiment/            base + 3 engines + registry
    topics/               gensim LDA wrapper
    analysis/             publisher, country, temporal, word freq
    visualization/        plots, wordclouds
    pipelines/            run_sentiment, run_topics
    cli.py                conflict-sentiment entrypoint
  tests/                  pytest suite (transformer mocked)
  pyproject.toml          PEP 621, hatchling
  Makefile, LICENSE, .pre-commit-config.yaml, .github/workflows/ci.yml
```

## Methodology

1. **Ingest**: `pd.read_excel` on the curated NewsData.io export.
2. **Clean**: strip whitespace from columns; build `combined_text` from TITLE plus DESCRIPTION; drop empty rows; deduplicate on combined text.
3. **Preprocess**: regex URL removal, alphabet-only filter, lowercase, regex tokenize, remove 179 English stopwords, WordNet lemmatize.
4. **Score**: each engine implements the `SentimentEngine` ABC (`score`, `score_batch`, `score_dataframe`). The registry resolves engines from YAML config.
5. **Topics**: build a Gensim dictionary, filter extremes (`no_below=20`, `no_above=0.5`), train LDA (`num_topics=5`, `passes=10`, `alpha='auto'`, `random_state=42`), assign dominant topic per document.
6. **Aggregate**: publisher and country level mean sentiment; topic / sentiment cross-tab; daily temporal trend.

## Reproducibility

| Component            | Pin                                                           |
| -------------------- | ------------------------------------------------------------- |
| Python               | >= 3.10                                                       |
| Transformer          | `cardiffnlp/twitter-roberta-base-sentiment-latest`            |
| LDA random_state     | 42                                                            |
| Transformer max_len  | 128 tokens                                                    |
| NLTK resources       | stopwords, punkt, punkt_tab, wordnet, omw-1.4, vader_lexicon  |
| Config source        | `configs/default.yaml` (override via `--config`)              |

```bash
conflict-sentiment --config configs/default.yaml sentiment
conflict-sentiment --config configs/default.yaml topics
```

## Ethical considerations

This corpus describes lethal conflict. Reducing human suffering to numerical scores is itself a methodological choice; we report it that way to enable structural comparison, not to rank events. Findings are descriptive: publisher and country aggregates reveal editorial framing tendencies, not ground-truth severity. The English-only corpus excludes Russian, Ukrainian, Arabic, and Mandarin reporting, and the transformer's training distribution skews Western, so non-Western voices may be systematically misclassified. Treat all aggregates as relative comparisons and cross-reference with independent sources before drawing policy or operational conclusions.

## Limitations and future work

- Transformer input truncated to 128 tokens; longer articles may be misjudged from their lead.
- Aggressive preprocessing strips numbers, removing casualty and economic magnitude.
- LDA topic assignments depend on random initialisation; alternative topic counts (3, 7, 10) yield different partitions.
- Temporal analysis is descriptive (rolling means and per-bucket aggregation); no formal time-series model is fit, so structural breaks and lagged effects around major events are not estimated.
- Single-language analysis: extending to Russian, Ukrainian, Arabic, and Mandarin is the priority next step.
- Future work: fine-tune a conflict-specific transformer; add LIME or SHAP explainability; correlate sentiment with on-ground escalation events.

## License

[MIT](LICENSE).

## Author

**Marvis Osazuwa** :: Analytics Engineer / Data Scientist (banking, healthcare, marketing analytics).

- Portfolio: <https://marz1307.github.io>
- LinkedIn: <https://www.linkedin.com/in/marvisosazuwa>
- Email: marvis.osazuwa@hotmail.com
