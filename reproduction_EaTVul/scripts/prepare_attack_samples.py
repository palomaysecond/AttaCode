from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import load_config
from eatvul_repro.logging_utils import setup_logging
from eatvul_repro.pipeline import prepare_attack_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare originally correct vulnerable samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--max-attack-samples", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.max_attack_samples is not None:
        cfg.raw["attack"]["max_attack_samples"] = args.max_attack_samples
    output = prepare_attack_samples(cfg, args.dataset, args.target_model, mock=args.mock)
    print(output)


if __name__ == "__main__":
    main()
