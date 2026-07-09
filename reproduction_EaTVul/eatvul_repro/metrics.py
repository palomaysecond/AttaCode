"""ASR and AMQ summarization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackMetrics:
    total_samples: int
    successful_attacks: int
    attack_success_rate: float
    average_model_queries: float

    def to_record(self) -> dict[str, float | int]:
        return {
            "total_samples": self.total_samples,
            "successful_attacks": self.successful_attacks,
            "attack_success_rate": self.attack_success_rate,
            "average_model_queries": self.average_model_queries,
        }


def compute_metrics(records: list[dict[str, object]]) -> AttackMetrics:
    total = len(records)
    success = sum(1 for record in records if record.get("status") == "success")
    queries = [int(record.get("queries", 0)) for record in records]
    asr = success / total if total else 0.0
    amq = sum(queries) / len(queries) if queries else 0.0
    return AttackMetrics(
        total_samples=total,
        successful_attacks=success,
        attack_success_rate=round(asr, 4),
        average_model_queries=round(amq, 2),
    )

