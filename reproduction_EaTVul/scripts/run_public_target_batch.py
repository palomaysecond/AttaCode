from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PAIRS = [
    ("Devign", "CodeBERT"),
    ("BigVul", "CodeBERT"),
    ("DiverseVul", "CodeBERT"),
    ("Devign", "UniXcoder"),
    ("BigVul", "UniXcoder"),
    ("DiverseVul", "UniXcoder"),
]


STAGES = [
    ("prepare", "prepare_attack_samples.py"),
    ("train_surrogate", "train_surrogate.py"),
    ("generate_attack_pool", "generate_attack_pool.py"),
    ("run_attack", "run_eatvul_attack.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EaTVul public target combinations sequentially.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eatvul_release.yaml"))
    parser.add_argument("--pairs", nargs="*", default=None, help="Pairs like Devign:CodeBERT")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose main output already exists.")
    parser.add_argument("--llm-model", default=None, help="Temporarily override llm.model for generation.")
    parser.add_argument("--max-prepared-samples", type=int, default=None)
    parser.add_argument("--max-run-samples", type=int, default=None)
    return parser.parse_args()


def parse_pairs(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_PAIRS
    pairs: list[tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Pair must be Dataset:TargetModel, got {value}")
        dataset, target_model = value.split(":", 1)
        pairs.append((dataset, target_model))
    return pairs


def stage_output(dataset: str, target_model: str, stage: str) -> Path:
    base = ROOT / "outputs" / "eatvul"
    if stage == "prepare":
        return base / "prepared" / f"{dataset}_{target_model}_attack_samples.jsonl"
    if stage == "train_surrogate":
        return base / "surrogates" / dataset / target_model / "surrogate.pt"
    if stage == "generate_attack_pool":
        return base / "attack_pool" / f"{dataset}_{target_model}_pool.jsonl"
    if stage == "run_attack":
        return base / "runs" / f"{dataset}_{target_model}_eatvul_results.jsonl"
    raise ValueError(stage)


def run_stage(args: argparse.Namespace, dataset: str, target_model: str, stage: str, script: str) -> None:
    output = stage_output(dataset, target_model, stage)
    if args.resume and output.exists():
        print(f"[skip] {dataset} x {target_model} {stage}: {output}", flush=True)
        return

    logs_dir = ROOT / "outputs" / "eatvul" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset}_{target_model}_{stage}"
    stdout_path = logs_dir / f"{prefix}.out.log"
    stderr_path = logs_dir / f"{prefix}.err.log"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "--config",
        args.config,
        "--dataset",
        dataset,
        "--target-model",
        target_model,
    ]
    if stage == "generate_attack_pool":
        if args.llm_model is not None:
            cmd.extend(["--llm-model", args.llm_model])
        if args.max_prepared_samples is not None:
            cmd.extend(["--max-prepared-samples", str(args.max_prepared_samples)])
    if stage == "run_attack" and args.max_run_samples is not None:
        cmd.extend(["--max-run-samples", str(args.max_run_samples)])
    print(f"[start] {dataset} x {target_model} {stage}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
    if completed.returncode != 0:
        print(f"[fail] {dataset} x {target_model} {stage}", flush=True)
        print(f"stdout: {stdout_path}", flush=True)
        print(f"stderr: {stderr_path}", flush=True)
        raise SystemExit(completed.returncode)
    print(f"[done] {dataset} x {target_model} {stage}: {output}", flush=True)


def run_evaluate(config: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_results.py"), "--config", config]
    subprocess.run(cmd, cwd=ROOT, check=True)


def print_result(dataset: str, target_model: str) -> None:
    metrics_path = ROOT / "outputs" / "eatvul" / "runs" / f"{dataset}_{target_model}_eatvul_results.metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    print(
        "RESULT "
        f"{dataset} x {target_model}: "
        f"ASR={metrics['attack_success_rate']} "
        f"AMQ={metrics['average_model_queries']} "
        f"success={metrics['successful_attacks']}/{metrics['total_samples']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    pairs = parse_pairs(args.pairs)
    for dataset, target_model in pairs:
        print(f"=== {dataset} x {target_model} ===", flush=True)
        for stage, script in STAGES:
            run_stage(args, dataset, target_model, stage, script)
        run_evaluate(args.config)
        print_result(dataset, target_model)


if __name__ == "__main__":
    main()
