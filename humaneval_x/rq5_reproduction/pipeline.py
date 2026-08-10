from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from .config import output_path, public_config, resolve_path
from .cpp_runner import compile_and_run, compiler_version
from .dataset import HumanEvalTask, load_tasks
from .generator import GenerationContext, generate_task_stages
from .io_utils import read_jsonl, safe_identifier, sha256_file, write_json, write_jsonl
from .llm_clients import build_client
from .metrics import aggregate_results, write_summary_files
from .prompts import load_templates
from .rules import build_rule_manifest, load_importance_scores
from .syntax import CppSyntaxChecker


def _replace_guard(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")


def _task_data(config: dict[str, Any]) -> tuple[Path, list[HumanEvalTask]]:
    dataset_path = resolve_path(config, config["dataset"]["path"])
    return dataset_path, load_tasks(dataset_path)


def _model_configs(config: dict[str, Any], selected_models: list[str] | None) -> list[dict[str, Any]]:
    models = config["models"]
    if selected_models:
        wanted = set(selected_models)
        models = [model for model in models if model["id"] in wanted]
        missing = wanted.difference(model["id"] for model in models)
        if missing:
            raise ValueError(f"Unknown model ids: {sorted(missing)}")
    if not models:
        raise ValueError("No models selected")
    return models


def run_baseline(config: dict[str, Any], overwrite: bool = False) -> Path:
    dataset_path, tasks = _task_data(config)
    output_file = output_path(config, "baseline.jsonl")
    _replace_guard(output_file, overwrite)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        result = compile_and_run(task.evaluation_program, config["compiler"])
        rows.append(
            {
                "task_id": task.task_id,
                "entrypoint": task.entrypoint,
                "source_sha256": task.source_sha256,
                **result,
            }
        )
        print(f"[baseline {index}/{len(tasks)}] {task.task_id}: {result['failure_kind'] or 'pass'}")
    write_jsonl(output_file, rows)
    metadata = {
        "dataset_path": str(config["dataset"]["path"]),
        "dataset_sha256": sha256_file(dataset_path),
        "task_count": len(tasks),
        "baseline_pass_count": sum(bool(row["passed"]) for row in rows),
        "compiler_version": compiler_version(str(config["compiler"]["executable"])),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config": public_config(config),
    }
    write_json(output_path(config, "metadata.json"), metadata)
    return output_file


def run_manifest(config: dict[str, Any], overwrite: bool = False) -> Path:
    _, tasks = _task_data(config)
    baseline_file = output_path(config, "baseline.jsonl")
    if not baseline_file.exists():
        raise FileNotFoundError("Run the baseline stage before building the rule manifest")
    baseline_rows = list(read_jsonl(baseline_file))
    included_ids = {row["task_id"] for row in baseline_rows if row.get("passed")}
    strategy = str(config["rule_selection"].get("strategy", "deterministic"))
    importance_value = config["rule_selection"].get("importance_manifest")
    if strategy not in {"deterministic", "importance"}:
        raise ValueError("rule_selection.strategy must be 'deterministic' or 'importance'")
    if strategy == "importance" and not importance_value:
        raise ValueError("importance strategy requires rule_selection.importance_manifest")
    importance_path = resolve_path(config, importance_value) if strategy == "importance" else None
    importance_scores = load_importance_scores(importance_path)
    rows = build_rule_manifest(
        tasks=tasks,
        included_task_ids=included_ids,
        stages=int(config["rule_selection"]["stages"]),
        seed=int(config["rule_selection"].get("seed", 42)),
        importance_scores=importance_scores,
    )
    output_file = output_path(config, "rule_manifest.jsonl")
    _replace_guard(output_file, overwrite)
    insufficient = sum(len(row["rules"]) < int(row["stages_requested"]) for row in rows)
    if insufficient:
        task_ids = [row["task_id"] for row in rows if len(row["rules"]) < int(row["stages_requested"])]
        raise RuntimeError(f"Could not construct all requested stages for tasks: {task_ids}")
    write_jsonl(output_file, rows)
    print(f"[manifest] wrote complete three-stage schedules for {len(rows)} tasks")
    return output_file


def run_generation(
    config: dict[str, Any],
    selected_models: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    _, tasks = _task_data(config)
    baseline_rows = list(read_jsonl(output_path(config, "baseline.jsonl")))
    included_ids = {row["task_id"] for row in baseline_rows if row.get("passed")}
    manifest_rows = list(read_jsonl(output_path(config, "rule_manifest.jsonl")))
    manifest_map = {row["task_id"]: row for row in manifest_rows}
    missing_manifest = included_ids.difference(manifest_map)
    if missing_manifest:
        raise RuntimeError(f"Rule manifest is missing baseline tasks: {sorted(missing_manifest)}")
    template_paths = [resolve_path(config, value) for value in config["generation"]["prompt_templates"]]
    context = GenerationContext(
        templates=load_templates(template_paths),
        checker=CppSyntaxChecker(),
        generation_config=config["generation"],
    )
    outputs: list[Path] = []
    for model in _model_configs(config, selected_models):
        model_id = model["id"]
        output_file = output_path(config, "generations", f"{safe_identifier(model_id)}.jsonl")
        _replace_guard(output_file, overwrite)
        client = build_client(model, config["generation"])
        records: list[dict[str, Any]] = []
        selected_tasks = [task for task in tasks if task.task_id in included_ids and task.task_id in manifest_map]
        for index, task in enumerate(selected_tasks, start=1):
            task_records = generate_task_stages(task, model_id, client, manifest_map[task.task_id], context)
            records.extend(task_records)
            statuses = ",".join(record["generation_status"] for record in task_records)
            print(f"[generate {model_id} {index}/{len(selected_tasks)}] {task.task_id}: {statuses}")
        write_jsonl(output_file, records)
        outputs.append(output_file)
    return outputs


def run_evaluation(
    config: dict[str, Any],
    selected_models: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    _, tasks = _task_data(config)
    task_map = {task.task_id: task for task in tasks}
    models = _model_configs(config, selected_models)
    output_file = output_path(config, "evaluation.jsonl")
    _replace_guard(output_file, overwrite)
    rows: list[dict[str, Any]] = []
    for model in models:
        model_id = model["id"]
        generation_file = output_path(config, "generations", f"{safe_identifier(model_id)}.jsonl")
        if not generation_file.exists():
            raise FileNotFoundError(f"Generation output is missing for model {model_id}")
        generation_rows = list(read_jsonl(generation_file))
        for index, generation in enumerate(generation_rows, start=1):
            task = task_map.get(generation["task_id"])
            if task is None:
                continue
            status = generation.get("generation_status")
            candidate = generation.get("output_code", "")
            if status in {"generation_failed", "no_applicable_rule", "blocked_by_previous_stage"}:
                result = {
                    "compiled": False,
                    "passed": False,
                    "failure_kind": status,
                    "compile_returncode": None,
                    "compile_stdout": "",
                    "compile_stderr": "",
                    "compile_seconds": None,
                    "run_returncode": None,
                    "run_stdout": "",
                    "run_stderr": "",
                    "run_seconds": None,
                }
            else:
                program = candidate.rstrip() + "\n\n" + task.test.lstrip()
                result = compile_and_run(program, config["compiler"])
            row = {
                "task_id": task.task_id,
                "model_id": model_id,
                "stage": int(generation["stage"]),
                "generation_status": status,
                "rule_name": (generation.get("rule") or {}).get("rule_name"),
                "changed": bool(generation.get("changed")),
                "syntax_ok": generation.get("syntax_ok"),
                "output_sha256": generation.get("output_sha256"),
                **result,
            }
            rows.append(row)
            print(
                f"[evaluate {model_id} {index}/{len(generation_rows)}] "
                f"{task.task_id} S={row['stage']}: {result['failure_kind'] or 'pass'}"
            )
    write_jsonl(output_file, rows)
    return output_file


def run_aggregate(
    config: dict[str, Any], selected_models: list[str] | None = None
) -> dict[str, Any]:
    baseline_rows = list(read_jsonl(output_path(config, "baseline.jsonl")))
    evaluation_rows = list(read_jsonl(output_path(config, "evaluation.jsonl")))
    models = _model_configs(config, selected_models)
    summary = aggregate_results(
        baseline_rows=baseline_rows,
        evaluation_rows=evaluation_rows,
        models=models,
        stages=int(config["rule_selection"]["stages"]),
    )
    write_summary_files(resolve_path(config, config["output_dir"]), summary)
    for group in summary["groups"]:
        print(
            f"[summary] {group['display_name']} S={group['stage']} "
            f"CSR={group['compiled']}/{group['N']} "
            f"TPR={group['passed']}/{group['compiled']} "
            f"BPR={group['passed']}/{group['N']}"
        )
    return summary


def run_all(
    config: dict[str, Any],
    selected_models: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_baseline(config, overwrite)
    run_manifest(config, overwrite)
    run_generation(config, selected_models, overwrite)
    run_evaluation(config, selected_models, overwrite)
    return run_aggregate(config, selected_models)
