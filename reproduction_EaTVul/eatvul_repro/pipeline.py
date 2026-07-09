"""End-to-end pipeline stages for the EaTVul reproduction."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .data import AttackSample, CodeSample, load_attack_samples, load_samples, save_attack_samples
from .fga import CandidateSnippet, FGASelector
from .insertion import find_insertion_sites
from .llm_client import LLMProviderError, LLMRequest, build_llm_client
from .logging_utils import read_jsonl, set_seed, write_json, write_jsonl
from .metrics import compute_metrics
from .prompting import EaTVulPromptBuilder, extract_code_snippet
from .target_model import TargetModelAdapter, build_target_adapter
from .validation import SnippetValidator


def prepared_path(cfg: ExperimentConfig, dataset: str, target_model: str) -> Path:
    return cfg.output_dir / "prepared" / f"{dataset}_{target_model}_attack_samples.jsonl"


def surrogate_dir(cfg: ExperimentConfig, dataset: str, target_model: str) -> Path:
    return cfg.output_dir / "surrogates" / dataset / target_model


def attack_pool_path(cfg: ExperimentConfig, dataset: str, target_model: str) -> Path:
    return cfg.output_dir / "attack_pool" / f"{dataset}_{target_model}_pool.jsonl"


def run_path(cfg: ExperimentConfig, dataset: str, target_model: str) -> Path:
    return cfg.output_dir / "runs" / f"{dataset}_{target_model}_eatvul_results.jsonl"


def prepare_attack_samples(
    cfg: ExperimentConfig,
    dataset: str,
    target_model: str,
    target: TargetModelAdapter | None = None,
    mock: bool = False,
) -> Path:
    set_seed(cfg.seed)
    attack_cfg = cfg.section("attack")
    samples = load_samples(cfg.dataset_path(dataset, "test"))
    target = target or build_target_adapter(cfg, dataset, target_model, mock=mock)

    attack_label = int(attack_cfg.get("attack_label", 1))
    max_samples = attack_cfg.get("max_attack_samples")
    filtered = [sample for sample in samples if sample.label == attack_label]
    target_cfg = cfg.target_model_cfg(target_model)
    prepare_batch_size = int(target_cfg.get("prepare_batch_size", max(1, int(target_cfg.get("eval_batch_size", 4))) * 8))

    attack_samples: list[AttackSample] = []
    for start in range(0, len(filtered), prepare_batch_size):
        batch = filtered[start : start + prepare_batch_size]
        predictions = target.predict_many([sample.func for sample in batch])
        for sample, prediction in zip(batch, predictions):
            if bool(attack_cfg.get("only_correct_originals", True)) and prediction.label != sample.label:
                continue
            attack_samples.append(
                AttackSample(
                    idx=sample.idx,
                    label=sample.label,
                    func=sample.func,
                    original_pred=prediction.label,
                    original_prob_vulnerable=prediction.prob_vulnerable,
                    meta=sample.meta,
                )
            )
            if max_samples is not None and len(attack_samples) >= int(max_samples):
                break
        if max_samples is not None and len(attack_samples) >= int(max_samples):
            break

    output = prepared_path(cfg, dataset, target_model)
    save_attack_samples(output, attack_samples)
    write_json(
        output.with_suffix(".meta.json"),
        {
            "dataset": dataset,
            "target_model": target_model,
            "total_test_samples": len(samples),
            "vulnerable_candidates": len(filtered),
            "prepared_attack_samples": len(attack_samples),
            "preparation_queries": target.query_count,
            "prepare_batch_size": prepare_batch_size,
        },
    )
    return output


def train_surrogate(
    cfg: ExperimentConfig,
    dataset: str,
    target_model: str,
    target: TargetModelAdapter | None = None,
    mock: bool = False,
    max_train_samples: int | None = None,
) -> Path:
    from .surrogate import SurrogateTrainer

    set_seed(cfg.seed)
    train_samples = load_samples(cfg.dataset_path(dataset, "train"))
    valid_samples = load_samples(cfg.dataset_path(dataset, "valid"))
    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
        valid_samples = valid_samples[: max(2, max_train_samples // 5)]

    target = target or build_target_adapter(cfg, dataset, target_model, mock=mock)
    soft_targets = _collect_soft_targets(target, train_samples + valid_samples)
    artifacts = SurrogateTrainer(cfg.section("surrogate")).train(
        train_samples=train_samples,
        valid_samples=valid_samples,
        soft_targets=soft_targets,
        output_dir=surrogate_dir(cfg, dataset, target_model),
    )
    write_json(
        artifacts.metrics_path.with_name("distillation_meta.json"),
        {
            "dataset": dataset,
            "target_model": target_model,
            "distillation_queries": target.query_count,
            "train_samples": len(train_samples),
            "valid_samples": len(valid_samples),
        },
    )
    return artifacts.model_path.parent


def generate_attack_pool(
    cfg: ExperimentConfig,
    dataset: str,
    target_model: str,
    mock_llm: bool = False,
) -> Path:
    from .feature_selection import build_feature_bundle, select_non_vulnerable_support_vectors
    from .surrogate import load_surrogate

    set_seed(cfg.seed)
    prepared = load_attack_samples(prepared_path(cfg, dataset, target_model))
    train_samples = load_samples(cfg.dataset_path(dataset, "train"))
    model, vocab = load_surrogate(surrogate_dir(cfg, dataset, target_model), cfg.section("surrogate"))
    feature_cfg = cfg.section("feature_selection")
    generation_cfg = cfg.section("generation")
    max_prepared_samples = generation_cfg.get("max_prepared_samples")
    if max_prepared_samples is not None:
        prepared = prepared[: int(max_prepared_samples)]

    support_ids = select_non_vulnerable_support_vectors(
        train_samples,
        kernel=str(feature_cfg.get("svm_kernel", "rbf")),
        max_support_vectors=int(feature_cfg.get("max_support_vectors", 200)),
    )
    llm = build_llm_client(cfg.section("llm"), root_dir=cfg.root_dir, mock=mock_llm)
    validator = SnippetValidator(cfg.section("validation"))
    prompt_builder = EaTVulPromptBuilder(max_snippet_lines=int(generation_cfg.get("max_snippet_lines", 8)))
    pool_records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    llm_calls = 0

    for sample in prepared:
        code_sample = CodeSample(idx=sample.idx, label=sample.label, func=sample.func, meta=sample.meta)
        bundle = build_feature_bundle(
            code_sample,
            support_vector_indices=support_ids,
            surrogate_model=model,
            vocab=vocab,
            top_k=int(feature_cfg.get("attention_top_k", 10)),
        )
        sites = find_insertion_sites(sample.func, context_window_lines=int(feature_cfg.get("context_window_lines", 5)))
        for site_idx, site in enumerate(sites[: int(generation_cfg.get("snippets_per_prompt", 1))]):
            payload = prompt_builder.build(bundle.key_tokens, site)
            for attempt in range(int(generation_cfg.get("max_regeneration_attempts", 2))):
                llm_calls += 1
                try:
                    response = llm.generate(
                        LLMRequest(
                            prompt=payload.prompt,
                            sample_idx=sample.idx,
                            stage=f"generate_attempt_{attempt}",
                            metadata={"site": site.description, "key_tokens": bundle.key_tokens},
                        )
                    )
                except LLMProviderError as exc:
                    rejection_counts[f"llm_error: {str(exc)[:120]}"] += 1
                    continue
                snippet = extract_code_snippet(response)
                validation = validator.validate(sample.func, snippet, site)
                if validation.valid:
                    snippet_id = f"{sample.idx}_{site_idx}_{attempt}"
                    pool_records.append(
                        {
                            "sample_idx": sample.idx,
                            "snippet_id": snippet_id,
                            "snippet": snippet,
                            "site_line_index": site.line_index,
                            "site_description": site.description,
                            "key_tokens": bundle.key_tokens,
                            "validation_reason": validation.reason,
                        }
                    )
                    break
                rejection_counts[validation.reason] += 1

    output = attack_pool_path(cfg, dataset, target_model)
    write_jsonl(output, pool_records)
    write_json(
        output.with_suffix(".meta.json"),
        {
            "dataset": dataset,
            "target_model": target_model,
            "prepared_samples": len(prepared),
            "valid_snippets": len(pool_records),
            "llm_calls": llm_calls,
            "rejected_snippets": sum(rejection_counts.values()),
            "rejection_reasons": dict(rejection_counts),
            "support_vectors": len(support_ids),
        },
    )
    return output


def run_attack(
    cfg: ExperimentConfig,
    dataset: str,
    target_model: str,
    target: TargetModelAdapter | None = None,
    mock: bool = False,
) -> Path:
    set_seed(cfg.seed)
    prepared = load_attack_samples(prepared_path(cfg, dataset, target_model))
    attack_cfg = cfg.section("attack")
    max_run_samples = attack_cfg.get("max_run_samples")
    if max_run_samples is not None:
        prepared = prepared[: int(max_run_samples)]
    pool_records = read_jsonl(attack_pool_path(cfg, dataset, target_model))
    target = target or build_target_adapter(cfg, dataset, target_model, mock=mock)
    rng = random.Random(cfg.seed)
    selector = FGASelector(cfg.section("fga"), rng=rng)
    results = []

    global_candidates = _candidate_snippets(pool_records)
    for sample in prepared:
        sample_candidates = _candidate_snippets(
            [record for record in pool_records if int(record["sample_idx"]) == sample.idx]
        )
        candidates = sample_candidates or global_candidates
        sites = find_insertion_sites(
            sample.func,
            context_window_lines=int(cfg.section("feature_selection").get("context_window_lines", 5)),
        )
        result = selector.attack(sample, candidates, sites, target)
        results.append(result.to_record())

    output = run_path(cfg, dataset, target_model)
    write_jsonl(output, results)
    write_json(output.with_suffix(".metrics.json"), compute_metrics(results).to_record())
    return output


def evaluate_all(cfg: ExperimentConfig) -> Path:
    summaries: list[dict[str, Any]] = []
    for dataset, target_model in cfg.pairs():
        result_path = run_path(cfg, dataset, target_model)
        if not result_path.exists():
            continue
        records = read_jsonl(result_path)
        metrics = compute_metrics(records).to_record()
        summaries.append({"dataset": dataset, "target_model": target_model, **metrics})

    output = cfg.output_dir / "summary" / "eatvul_summary.json"
    write_json(output, summaries)
    csv_path = output.with_suffix(".csv")
    _write_summary_csv(csv_path, summaries)
    return output


def _collect_soft_targets(target: TargetModelAdapter, samples: list[CodeSample], batch_size: int = 32) -> dict[int, float]:
    soft_targets: dict[int, float] = {}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        predictions = target.predict_many([sample.func for sample in batch])
        for sample, prediction in zip(batch, predictions):
            soft_targets[sample.idx] = prediction.prob_vulnerable
    return soft_targets


def _candidate_snippets(records: list[dict[str, Any]]) -> list[CandidateSnippet]:
    return [
        CandidateSnippet(
            snippet_id=str(record["snippet_id"]),
            snippet=str(record["snippet"]),
            metadata={key: value for key, value in record.items() if key not in {"snippet_id", "snippet"}},
        )
        for record in records
    ]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "target_model",
        "total_samples",
        "successful_attacks",
        "attack_success_rate",
        "average_model_queries",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(column, "")) for column in columns) + "\n")
