from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eatvul_repro.config import load_config


HF_REPO_ID = "Sak1Rinn/LLM4CVD-models"

PUBLIC_CHECKPOINTS = {
    "CodeBERT": {
        "Devign": "CodeBERT/devign_0-512/checkpoint-best-f1/model.bin",
        "BigVul": "CodeBERT/bigvul_0-512/checkpoint-best-f1/model.bin",
        "DiverseVul": "CodeBERT/diversevul_0-512/checkpoint-best-f1/model.bin",
    },
    "UniXcoder": {
        "Devign": "UniXcoder/devign_0-512/checkpoint-best-f1/model.bin",
        "BigVul": "UniXcoder/bigvul_0-512/checkpoint-best-f1/model.bin",
        "DiverseVul": "UniXcoder/diversevul_0-512/checkpoint-best-f1/model.bin",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public target detector checkpoints that match the AttaCode datasets."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "eatvul_release.yaml"))
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "target_models" / "LLM4CVD"))
    parser.add_argument("--datasets", nargs="+", default=["Devign", "BigVul", "DiverseVul"])
    parser.add_argument("--target-models", nargs="+", default=["CodeBERT", "UniXcoder"])
    return parser.parse_args()


def _relative_for_yaml(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    downloaded: dict[tuple[str, str], Path] = {}
    for target_model in args.target_models:
        if target_model not in PUBLIC_CHECKPOINTS:
            print(f"[skip] No public checkpoint mapping for target model: {target_model}")
            continue
        for dataset in args.datasets:
            repo_path = PUBLIC_CHECKPOINTS[target_model].get(dataset)
            if not repo_path:
                print(f"[skip] No public checkpoint mapping for {dataset} x {target_model}")
                continue
            local_path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=repo_path,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                )
            ).resolve()
            downloaded[(dataset, target_model)] = local_path
            print(f"[ok] {dataset} x {target_model}: {local_path}")

    if downloaded:
        print("\nYAML checkpoint paths:")
        for target_model in args.target_models:
            rows = [
                (dataset, path)
                for (dataset, model_name), path in downloaded.items()
                if model_name == target_model
            ]
            if not rows:
                continue
            print(f"{target_model}:")
            print("  checkpoints:")
            for dataset, path in rows:
                print(f"    {dataset}: {_relative_for_yaml(path, cfg.root_dir)}")

    missing_codet5 = [dataset for dataset in args.datasets if "CodeT5" in args.target_models]
    if missing_codet5:
        print("\n[info] CodeT5 checkpoints were not found in this public repository.")


if __name__ == "__main__":
    main()
