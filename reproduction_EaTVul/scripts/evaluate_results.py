from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import load_config
from eatvul_repro.logging_utils import setup_logging
from eatvul_repro.pipeline import evaluate_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EaTVul ASR and AMQ.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)
    output = evaluate_all(cfg)
    print(output)


if __name__ == "__main__":
    main()

