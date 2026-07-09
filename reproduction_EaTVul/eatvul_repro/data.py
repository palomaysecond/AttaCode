"""Dataset abstractions for AttaCode-style JSONL splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .logging_utils import read_jsonl, write_jsonl


@dataclass(frozen=True)
class CodeSample:
    idx: int
    label: int
    func: str
    meta: dict[str, Any]

    @staticmethod
    def from_record(record: dict[str, Any]) -> "CodeSample":
        idx = int(record.get("idx", record.get("index", -1)))
        if idx < 0:
            idx = abs(hash(record.get("func", ""))) % (10**12)
        meta = {key: value for key, value in record.items() if key not in {"idx", "target", "func"}}
        return CodeSample(idx=idx, label=int(record["target"]), func=str(record["func"]), meta=meta)

    def to_record(self) -> dict[str, Any]:
        return {"idx": self.idx, "target": self.label, "func": self.func, **self.meta}


@dataclass(frozen=True)
class AttackSample:
    idx: int
    label: int
    func: str
    original_pred: int
    original_prob_vulnerable: float
    meta: dict[str, Any]

    @staticmethod
    def from_record(record: dict[str, Any]) -> "AttackSample":
        return AttackSample(
            idx=int(record["idx"]),
            label=int(record["target"]),
            func=str(record["func"]),
            original_pred=int(record["original_pred"]),
            original_prob_vulnerable=float(record["original_prob_vulnerable"]),
            meta={key: value for key, value in record.items() if key not in {
                "idx",
                "target",
                "func",
                "original_pred",
                "original_prob_vulnerable",
            }},
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "target": self.label,
            "func": self.func,
            "original_pred": self.original_pred,
            "original_prob_vulnerable": self.original_prob_vulnerable,
            **self.meta,
        }


def load_samples(path: str | Path) -> list[CodeSample]:
    return [CodeSample.from_record(record) for record in read_jsonl(path)]


def save_attack_samples(path: str | Path, samples: list[AttackSample]) -> None:
    write_jsonl(path, [sample.to_record() for sample in samples])


def load_attack_samples(path: str | Path) -> list[AttackSample]:
    return [AttackSample.from_record(record) for record in read_jsonl(path)]


def samples_to_records(samples: list[CodeSample]) -> list[dict[str, Any]]:
    return [asdict(sample) for sample in samples]

