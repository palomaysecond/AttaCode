from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import load_config
from eatvul_repro.logging_utils import setup_logging
from eatvul_repro.pipeline import generate_attack_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate EaTVul adversarial snippets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--llm-model", default=None, help="Temporarily override llm.model.")
    parser.add_argument("--max-prepared-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.llm_model is not None:
        cfg.raw["llm"]["model"] = args.llm_model
    if args.max_prepared_samples is not None:
        cfg.raw.setdefault("generation", {})["max_prepared_samples"] = args.max_prepared_samples
    output = generate_attack_pool(cfg, args.dataset, args.target_model, mock_llm=args.mock_llm)
    print(output)


if __name__ == "__main__":
    main()
