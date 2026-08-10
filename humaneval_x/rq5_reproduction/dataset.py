from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_jsonl, sha256_text


REQUIRED_FIELDS = {
    "task_id",
    "prompt",
    "declaration",
    "canonical_solution",
    "test",
    "example_test",
}


@dataclass(frozen=True)
class HumanEvalTask:
    task_id: str
    prompt: str
    declaration: str
    canonical_solution: str
    test: str
    example_test: str

    @property
    def source_code(self) -> str:
        return self.declaration.rstrip() + "\n" + self.canonical_solution.lstrip("\n")

    @property
    def evaluation_program(self) -> str:
        return self.source_code.rstrip() + "\n\n" + self.test.lstrip()

    @property
    def entrypoint(self) -> str:
        matches = re.findall(r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", self.declaration)
        if not matches:
            raise ValueError(f"Could not determine the entry point for {self.task_id}")
        return matches[-1]

    @property
    def source_sha256(self) -> str:
        return sha256_text(self.source_code)


def load_tasks(path: Path) -> list[HumanEvalTask]:
    tasks: list[HumanEvalTask] = []
    seen_ids: set[str] = set()
    for row in read_jsonl(path):
        missing = REQUIRED_FIELDS.difference(row)
        if missing:
            raise ValueError(f"Task is missing fields {sorted(missing)}: {row.get('task_id', '<unknown>')}")
        task_id = str(row["task_id"])
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen_ids.add(task_id)
        tasks.append(
            HumanEvalTask(
                task_id=task_id,
                prompt=str(row["prompt"]),
                declaration=str(row["declaration"]),
                canonical_solution=str(row["canonical_solution"]),
                test=str(row["test"]),
                example_test=str(row["example_test"]),
            )
        )
    return tasks
