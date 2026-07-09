"""Configuration loading and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Thin typed wrapper around the YAML config."""

    config_path: Path
    raw: dict[str, Any]

    @property
    def root_dir(self) -> Path:
        return self.config_path.parent.parent.resolve()

    @property
    def output_dir(self) -> Path:
        value = self.raw["project"]["output_dir"]
        return self.resolve_path(value)

    @property
    def seed(self) -> int:
        return int(self.raw["project"].get("seed", 42))

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root_dir / path).resolve()

    def dataset_path(self, dataset: str, split: str) -> Path:
        return self.resolve_path(self.raw["datasets"][dataset][split])

    def checkpoint_path(self, dataset: str, target_model: str) -> Path:
        value = self.raw["target_models"][target_model]["checkpoints"][dataset]
        return self.resolve_path(value)

    def target_model_cfg(self, target_model: str) -> dict[str, Any]:
        return dict(self.raw["target_models"][target_model])

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def pairs(self) -> list[tuple[str, str]]:
        return [
            (dataset, target_model)
            for dataset in self.raw["datasets"]
            for target_model in self.raw["target_models"]
        ]

    def ensure_output_dirs(self) -> None:
        for subdir in [
            "prepared",
            "surrogates",
            "attack_pool",
            "runs",
            "summary",
            "cache/llm",
            "cache/llm_errors",
            "audit",
        ]:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    cfg = ExperimentConfig(config_path=config_path, raw=raw)
    cfg.ensure_output_dirs()
    return cfg
