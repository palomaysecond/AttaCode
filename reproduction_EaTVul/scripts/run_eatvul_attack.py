from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import load_config
from eatvul_repro.logging_utils import setup_logging
from eatvul_repro.pipeline import run_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EaTVul FGA snippet selection and evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-run-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.max_run_samples is not None:
        cfg.raw.setdefault("attack", {})["max_run_samples"] = args.max_run_samples
    output = run_attack(cfg, args.dataset, args.target_model, mock=args.mock)
    print(output)


if __name__ == "__main__":
    main()
