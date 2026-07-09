from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import load_config
from eatvul_repro.logging_utils import setup_logging
from eatvul_repro.pipeline import train_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EaTVul BiLSTM-attention surrogate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.raw["surrogate"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg.raw["surrogate"]["patience"] = args.patience
    output = train_surrogate(
        cfg,
        args.dataset,
        args.target_model,
        mock=args.mock,
        max_train_samples=args.max_train_samples,
    )
    print(output)


if __name__ == "__main__":
    main()
