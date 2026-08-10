#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rq5_reproduction.config import load_config
from rq5_reproduction.pipeline import (
    run_aggregate,
    run_all,
    run_baseline,
    run_evaluation,
    run_generation,
    run_manifest,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce the HumanEval-X behavioral-preservation experiment")
    parser.add_argument(
        "command",
        choices=["baseline", "manifest", "generate", "evaluate", "aggregate", "all"],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--config",
        default=str(script_dir / "experiment_config.json"),
        help="JSON configuration file (relative paths are resolved from the file)",
    )
    parser.add_argument("--models", nargs="+", help="Optional list of model ids to run")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs for the selected stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "baseline":
        run_baseline(config, args.overwrite)
    elif args.command == "manifest":
        run_manifest(config, args.overwrite)
    elif args.command == "generate":
        run_generation(config, args.models, args.overwrite)
    elif args.command == "evaluate":
        run_evaluation(config, args.models, args.overwrite)
    elif args.command == "aggregate":
        run_aggregate(config, args.models)
    else:
        run_all(config, args.models, args.overwrite)


if __name__ == "__main__":
    main()
