from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from .dataset import HumanEvalTask
from .io_utils import sha256_text
from .llm_clients import LLMClient
from .prompts import render_prompt
from .syntax import CppSyntaxChecker


FENCED_CODE = re.compile(r"```\s*(?:cpp|c\+\+|cc|cxx)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_cpp_code(response: str) -> str:
    matches = [match.strip() for match in FENCED_CODE.findall(response) if match.strip()]
    if matches:
        return max(matches, key=len)
    stripped = response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[^\n]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def is_refusal(response: str, keywords: Iterable[str]) -> bool:
    lowered = response.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


@dataclass
class GenerationContext:
    templates: list[str]
    checker: CppSyntaxChecker
    generation_config: dict[str, Any]


def generate_task_stages(
    task: HumanEvalTask,
    model_id: str,
    client: LLMClient,
    manifest_row: dict[str, Any],
    context: GenerationContext,
) -> list[dict[str, Any]]:
    current_code = task.source_code
    rules_by_stage = {int(rule["stage"]): rule for rule in manifest_row.get("rules", [])}
    stages = int(manifest_row["stages_requested"])
    records: list[dict[str, Any]] = []
    blocked_by_stage: int | None = None

    for stage in range(1, stages + 1):
        input_code = current_code
        rule = rules_by_stage.get(stage)
        if blocked_by_stage is not None:
            records.append(
                _base_record(
                    task=task,
                    model_id=model_id,
                    stage=stage,
                    rule=rule,
                    input_code=input_code,
                    output_code=input_code,
                    status="blocked_by_previous_stage",
                    attempts=[],
                    syntax_backend=context.checker.backend,
                    syntax_ok=False,
                    syntax_reason=f"stage {blocked_by_stage} did not produce an accepted transformation",
                )
            )
            continue
        if rule is None:
            records.append(
                _base_record(
                    task=task,
                    model_id=model_id,
                    stage=stage,
                    rule=None,
                    input_code=input_code,
                    output_code=input_code,
                    status="no_applicable_rule",
                    attempts=[],
                    syntax_backend=context.checker.backend,
                    syntax_ok=True,
                    syntax_reason="no generation attempted",
                )
            )
            blocked_by_stage = stage
            continue

        attempts: list[dict[str, Any]] = []
        output_code = ""
        status = "generation_failed"
        final_syntax_ok = False
        final_syntax_reason = "no response"
        max_attempts = int(context.generation_config.get("max_attempts", 3))
        refusal_keywords = context.generation_config.get("refusal_keywords", [])
        for attempt_index in range(max_attempts):
            template_index = attempt_index % len(context.templates)
            prompt = render_prompt(context.templates[template_index], input_code, rule, task.entrypoint, stage)
            started = time.monotonic()
            attempt_record: dict[str, Any] = {
                "attempt": attempt_index + 1,
                "template_index": template_index,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
            }
            try:
                response = client.generate(prompt)
                attempt_record["latency_seconds"] = round(time.monotonic() - started, 6)
                attempt_record["raw_response"] = response.text
                attempt_record["response_id"] = response.response_id
                attempt_record["usage"] = response.usage
            except Exception as exc:
                attempt_record["latency_seconds"] = round(time.monotonic() - started, 6)
                attempt_record["error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(attempt_record)
                continue

            if is_refusal(response.text, refusal_keywords):
                attempt_record["validation"] = "refusal"
                attempts.append(attempt_record)
                final_syntax_reason = "refusal"
                continue

            candidate = extract_cpp_code(response.text)
            if not candidate:
                attempt_record["validation"] = "empty_output"
                attempts.append(attempt_record)
                final_syntax_reason = "empty output"
                continue
            if candidate.strip() == input_code.strip():
                attempt_record["validation"] = "unchanged_output"
                attempts.append(attempt_record)
                final_syntax_reason = "the requested transformation was not applied"
                continue

            syntax = context.checker.check(candidate)
            attempt_record["syntax_ok"] = syntax.valid
            attempt_record["syntax_reason"] = syntax.reason
            attempt_record["syntax_backend"] = syntax.backend
            if not re.search(rf"\b{re.escape(task.entrypoint)}\s*\(", candidate):
                attempt_record["validation"] = "entrypoint_missing"
                attempts.append(attempt_record)
                output_code = candidate
                final_syntax_ok = syntax.valid
                final_syntax_reason = "public entry point missing"
                continue
            if not syntax.valid:
                attempt_record["validation"] = "syntax_invalid"
                attempts.append(attempt_record)
                output_code = candidate
                final_syntax_ok = False
                final_syntax_reason = syntax.reason
                continue

            attempt_record["validation"] = "accepted"
            attempts.append(attempt_record)
            output_code = candidate
            final_syntax_ok = syntax.valid
            final_syntax_reason = syntax.reason
            status = "success"
            break

        accepted_code = output_code if status == "success" else input_code
        current_code = accepted_code
        record = _base_record(
            task=task,
            model_id=model_id,
            stage=stage,
            rule=rule,
            input_code=input_code,
            output_code=accepted_code,
            status=status,
            attempts=attempts,
            syntax_backend=context.checker.backend,
            syntax_ok=final_syntax_ok,
            syntax_reason=final_syntax_reason,
        )
        record["carried_forward"] = status != "success"
        if status != "success":
            record["last_rejected_code"] = output_code or None
            record["last_rejected_sha256"] = sha256_text(output_code) if output_code else None
            blocked_by_stage = stage
        records.append(record)
    return records


def _base_record(
    task: HumanEvalTask,
    model_id: str,
    stage: int,
    rule: dict[str, Any] | None,
    input_code: str,
    output_code: str,
    status: str,
    attempts: list[dict[str, Any]],
    syntax_backend: str,
    syntax_ok: bool,
    syntax_reason: str,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "model_id": model_id,
        "stage": stage,
        "entrypoint": task.entrypoint,
        "rule": rule,
        "generation_status": status,
        "input_code": input_code,
        "output_code": output_code,
        "input_sha256": sha256_text(input_code),
        "output_sha256": sha256_text(output_code),
        "changed": input_code.strip() != output_code.strip(),
        "syntax_ok": syntax_ok,
        "syntax_reason": syntax_reason,
        "syntax_backend": syntax_backend,
        "attempts": attempts,
    }
