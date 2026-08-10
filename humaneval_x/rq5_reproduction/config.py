from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "dataset": {"path": "humaneval_cpp.jsonl"},
    "output_dir": "outputs",
    "compiler": {
        "executable": "g++",
        "standard": "gnu++17",
        "flags": ["-O0", "-pipe"],
        "compile_timeout_seconds": 30,
        "run_timeout_seconds": 10,
        "memory_limit_mb": 1024,
        "max_log_characters": 12000,
    },
    "rule_selection": {
        "strategy": "deterministic",
        "seed": 42,
        "stages": 3,
        "importance_manifest": None,
    },
    "generation": {
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": 4096,
        "seed": None,
        "max_attempts": 3,
        "prompt_templates": ["prompts/cpp_robustness_benchmark.txt"],
        "refusal_keywords": [
            "I cannot",
            "I can't",
            "I apologize",
            "cannot assist",
            "As an AI language model",
        ],
    },
    "models": [],
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if config_path.suffix.lower() != ".json":
        raise ValueError("Configuration files must use JSON")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a mapping")
    config = _merge(DEFAULTS, loaded)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    validate_config(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def output_path(config: dict[str, Any], *parts: str) -> Path:
    return resolve_path(config, config["output_dir"]).joinpath(*parts)


def validate_config(config: dict[str, Any]) -> None:
    stages = int(config["rule_selection"]["stages"])
    if stages < 1:
        raise ValueError("rule_selection.stages must be at least 1")
    if int(config["generation"]["max_attempts"]) < 1:
        raise ValueError("generation.max_attempts must be at least 1")
    if not isinstance(config.get("models"), list):
        raise ValueError("models must be a list")
    seen: set[str] = set()
    for model in config["models"]:
        model_id = model.get("id")
        if not model_id or model_id in seen:
            raise ValueError("Every model must have a unique non-empty id")
        seen.add(model_id)


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the non-secret experiment settings suitable for metadata logs."""
    return {key: value for key, value in config.items() if not key.startswith("_")}
