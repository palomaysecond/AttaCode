from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import write_json


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}%"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def aggregate_results(
    baseline_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    stages: int,
) -> dict[str, Any]:
    included_ids = {row["task_id"] for row in baseline_rows if row.get("passed")}
    denominator = len(included_ids)
    indexed = {
        (row["model_id"], int(row["stage"]), row["task_id"]): row
        for row in evaluation_rows
        if row.get("task_id") in included_ids
    }
    groups: list[dict[str, Any]] = []
    for model in models:
        model_id = model["id"]
        for stage in range(1, stages + 1):
            rows = [indexed.get((model_id, stage, task_id)) for task_id in sorted(included_ids)]
            present = [row for row in rows if row is not None]
            compiled = sum(bool(row.get("compiled")) for row in present)
            passed = sum(bool(row.get("passed")) for row in present)
            changed = sum(bool(row.get("changed")) for row in present)
            failures = Counter((row.get("failure_kind") or "none") for row in present)
            failures["missing_result"] += denominator - len(present)
            csr = _ratio(compiled, denominator)
            tpr = _ratio(passed, compiled)
            bpr = _ratio(passed, denominator)
            groups.append(
                {
                    "model_id": model_id,
                    "display_name": model.get("display_name", model_id),
                    "stage": stage,
                    "N": denominator,
                    "present": len(present),
                    "compiled": compiled,
                    "passed": passed,
                    "changed": changed,
                    "CSR": csr,
                    "TPR": tpr,
                    "BPR": bpr,
                    "CSR_ci95": wilson_interval(compiled, denominator),
                    "BPR_ci95": wilson_interval(passed, denominator),
                    "failure_counts": dict(sorted(failures.items())),
                }
            )

    aggregate_by_stage: list[dict[str, Any]] = []
    for stage in range(1, stages + 1):
        stage_groups = [group for group in groups if group["stage"] == stage]
        total = sum(group["N"] for group in stage_groups)
        compiled = sum(group["compiled"] for group in stage_groups)
        passed = sum(group["passed"] for group in stage_groups)
        aggregate_by_stage.append(
            {
                "stage": stage,
                "N": total,
                "compiled": compiled,
                "passed": passed,
                "CSR": _ratio(compiled, total),
                "TPR": _ratio(passed, compiled),
                "BPR": _ratio(passed, total),
            }
        )
    return {"baseline_N": denominator, "groups": groups, "aggregate_by_stage": aggregate_by_stage}


def write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", summary)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_id", "generator", "stage", "N", "compiled", "passed", "changed", "CSR", "TPR", "BPR"])
        for group in summary["groups"]:
            writer.writerow(
                [
                    group["model_id"],
                    group["display_name"],
                    group["stage"],
                    group["N"],
                    group["compiled"],
                    group["passed"],
                    group["changed"],
                    _percent(group["CSR"]),
                    _percent(group["TPR"]),
                    _percent(group["BPR"]),
                ]
            )
    (output_dir / "table_rows.tex").write_text(render_latex_rows(summary), encoding="utf-8")


def render_latex_rows(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    groups = summary["groups"]
    model_names = list(dict.fromkeys(group["display_name"] for group in groups))
    for model_index, display_name in enumerate(model_names):
        model_groups = [group for group in groups if group["display_name"] == display_name]
        escaped_name = display_name.replace("_", r"\_")
        for index, group in enumerate(model_groups):
            prefix = rf"\multirow{{{len(model_groups)}}}{{*}}{{{escaped_name}}}" if index == 0 else ""
            csr = f"{group['compiled']}/{group['N']} ({_percent(group['CSR'])})"
            tpr = f"{group['passed']}/{group['compiled']} ({_percent(group['TPR'])})" if group["compiled"] else "0/0 (--)"
            bpr = f"{group['passed']}/{group['N']} ({_percent(group['BPR'])})"
            lines.append(f"{prefix} & {group['stage']} & {csr} & {tpr} & {bpr} \\\\")
        if model_index + 1 < len(model_names):
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"
