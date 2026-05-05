"""Command-line interface aggregator.

Run ``conflict-sentiment --help`` for available subcommands.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .logging_utils import configure_root_logger, get_logger
from .pipelines import run_sentiment_pipeline, run_topics_pipeline

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conflict-sentiment",
        description=(
            "Multi-engine sentiment analysis and LDA topic modelling for "
            "Russia and Ukraine conflict news coverage."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sent = sub.add_parser("sentiment", help="Run the multi-engine sentiment pipeline.")
    sent.add_argument("--output", type=Path, default=None, help="Output CSV path.")

    topics = sub.add_parser("topics", help="Run the LDA topic-modelling pipeline.")
    topics.add_argument("--output-dir", type=Path, default=None, help="Output directory.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument vector (used for testing).

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_root_logger(level=getattr(logging, args.log_level))

    if args.command == "sentiment":
        run_sentiment_pipeline(config_path=args.config, output_csv=args.output)
    elif args.command == "topics":
        run_topics_pipeline(config_path=args.config, output_dir=args.output_dir)
    else:  # pragma: no cover (argparse enforces required subcommand)
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
