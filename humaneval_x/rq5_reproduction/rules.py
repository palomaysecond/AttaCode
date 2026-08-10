from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset import HumanEvalTask
from .io_utils import read_jsonl


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str
    patterns: tuple[str, ...]


RULES: tuple[RuleSpec, ...] = (
    RuleSpec("ForToWhileConversion", "Replace a for-loop with an equivalent while-loop.", (r"\bfor\s*\(",)),
    RuleSpec("WhileToForRefactoring", "Replace a while-loop with an equivalent for-loop.", (r"\bwhile\s*\(",)),
    RuleSpec("DoWhileToWhileConversion", "Replace a do-while loop with an equivalent while-loop.", (r"\bdo\b",)),
    RuleSpec("IfElseBranchSwap", "Swap if and else branches while negating the condition.", (r"\bif\s*\([^)]*\)[\s\S]{0,400}?\belse\b",)),
    RuleSpec("ElseIfToNestedIf", "Convert else-if into an equivalent nested if.", (r"\belse\s+if\s*\(",)),
    RuleSpec("NestedIfToElseIf", "Convert a single nested else-if block into else-if form.", (r"\belse\s*\{\s*if\s*\(",)),
    RuleSpec("SwitchToIfElse", "Convert a switch statement into an equivalent if-else chain.", (r"\bswitch\s*\(",)),
    RuleSpec("SplitCompoundCondition", "Split a compound short-circuit condition into nested branches.", (r"\bif\s*\([^)]*(?:&&|\|\|)[^)]*\)",)),
    RuleSpec(
        "AddRedundantStatement",
        "Insert a side-effect-free redundant statement, such as a no-op before a return.",
        (r"\b(?:int|long|float|double|bool|char|auto|size_t|string|vector\s*<[^>]+>)\s+[A-Za-z_]\w*", r"\breturn\b"),
    ),
    RuleSpec("AddUnusedParameter", "Add an unused parameter to a non-entry helper function.", ()),
    RuleSpec("WrapStatementInBlock", "Wrap a single controlled statement in braces.", (r"\b(?:if|for|while)\s*\([^)]*\)\s*(?!\{)[^;\n{}]+;",)),
    RuleSpec("UnwrapRedundantBlock", "Remove braces around a safe single-statement block.", (r"\b(?:if|for|while)\s*\([^)]*\)\s*\{\s*[^{};]+;\s*\}",)),
    RuleSpec("WrapWithConstantCondition", "Wrap a statement in an always-true condition.", (r"\b[A-Za-z_]\w*\s*(?<![=!<>])=(?!=)\s*[^;]+;",)),
    RuleSpec("ExtractSubexpression", "Extract a subexpression into a temporary variable.", (r"(?:=|\breturn)\s*[^;\n]*(?:\+|-|\*|/)[^;\n]*;",)),
    RuleSpec("ReturnViaTempVariable", "Store a return expression in a temporary before returning.", (r"\breturn\s+[^;]+;",)),
    RuleSpec("NegateWithReversedOperator", "Express a comparison with a logically reversed operator.", (r"!\s*\([^)]*(?:==|!=|<=|>=|<|>)[^)]*\)", r"\b[A-Za-z_]\w*\s*(?:<=|>=|==|!=|<|>)\s*[^;)&|]+")),
    RuleSpec("RenameVariable", "Rename a local variable consistently.", (r"\b(?:int|long|float|double|bool|char|auto|size_t|string)\s+([A-Za-z_]\w*)",)),
    RuleSpec("RenameClassAndMethod", "Rename an internal class or helper function consistently.", ()),
    RuleSpec("InlineLoopDeclaration", "Move a loop variable declaration into the for header.", (r"\b(?:int|long|size_t)\s+[A-Za-z_]\w*\s*;\s*for\s*\(",)),
    RuleSpec("ExtractLoopDeclaration", "Move a loop variable declaration outside the for header.", (r"\bfor\s*\(\s*(?:int|long|size_t|auto)\s+[A-Za-z_]\w*",)),
    RuleSpec("PromoteIntToLong", "Promote a suitable local int variable to long.", (r"\bint\s+[A-Za-z_]\w*",)),
    RuleSpec("PromoteFloatToDouble", "Promote a suitable float variable to double.", (r"\bfloat\s+[A-Za-z_]\w*",)),
    RuleSpec("ExpandCompoundAssign", "Expand a compound assignment into an explicit assignment.", (r"\b[A-Za-z_]\w*\s*(?:\+=|-=|\*=|/=|%=)\s*[^;]+;",)),
    RuleSpec("RefactorOutputAPI", "Refactor a C++ output stream call through a stream reference.", (r"\b(?:std::)?cout\s*<<",)),
    RuleSpec("ExpandUnaryOP", "Expand increment or decrement into an explicit assignment.", (r"(?:\+\+|--)[A-Za-z_]\w*|[A-Za-z_]\w*(?:\+\+|--)",)),
)

BODY_ONLY_RULES = {
    "AddRedundantStatement",
    "RenameVariable",
    "PromoteIntToLong",
    "PromoteFloatToDouble",
}


def _line_number(code: str, start: int) -> int:
    return code.count("\n", 0, start) + 1


def _snippet(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _generic_function_matches(code: str, entrypoint: str) -> list[re.Match[str]]:
    pattern = re.compile(
        r"(?:^|\n)\s*(?:template\s*<[^>]+>\s*)?"
        r"[A-Za-z_:][\w:<>,\s*&]*\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
        re.MULTILINE,
    )
    blocked = {entrypoint, "if", "for", "while", "switch", "catch", "main"}
    return [match for match in pattern.finditer(code) if match.group(1) not in blocked]


def find_occurrences(rule: RuleSpec, code: str, entrypoint: str) -> list[dict[str, Any]]:
    matches: list[re.Match[str]] = []
    if rule.name == "AddUnusedParameter":
        matches = _generic_function_matches(code, entrypoint)
    elif rule.name == "RenameClassAndMethod":
        class_matches = list(re.finditer(r"\b(?:class|struct)\s+([A-Za-z_]\w*)", code))
        matches = class_matches + _generic_function_matches(code, entrypoint)
    else:
        for pattern in rule.patterns:
            matches.extend(re.finditer(pattern, code, re.MULTILINE))

    entrypoint_definition = re.search(rf"\b{re.escape(entrypoint)}\s*\([^;{{}}]*\)\s*\{{", code)
    entrypoint_body_start = entrypoint_definition.end() if entrypoint_definition else 0
    occurrences: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    for match in sorted(matches, key=lambda item: (item.start(), item.end())):
        if rule.name in BODY_ONLY_RULES and match.start() < entrypoint_body_start:
            continue
        span = (match.start(), match.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        occurrences.append(
            {
                "start_byte": match.start(),
                "end_byte": match.end(),
                "line": _line_number(code, match.start()),
                "snippet": _snippet(match.group(0)),
            }
        )
    return occurrences


def _stable_rank(seed: int, task_id: str, rule_name: str, start_byte: int) -> float:
    payload = f"{seed}:{task_id}:{rule_name}:{start_byte}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) / float(2**256)


def load_importance_scores(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        rows: Iterable[dict[str, Any]] = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        rows = value if isinstance(value, list) else value.get("tasks", [])

    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        raw_id = row.get("task_id", row.get("index"))
        task_id = str(raw_id)
        if task_id.isdigit():
            task_id = f"CPP/{task_id}"
        candidates = row.get("rules", row.get("patterns", []))
        task_scores: dict[str, float] = {}
        for candidate in candidates:
            name = candidate.get("rule_name", candidate.get("rule", candidate.get("transformation_name")))
            if name:
                cleaned = str(name).replace("Assess", "").replace("_new", "")
                task_scores[cleaned] = float(candidate.get("score", 0.0))
        scores[task_id] = task_scores
    return scores


def build_rule_manifest(
    tasks: Iterable[HumanEvalTask],
    included_task_ids: set[str],
    stages: int,
    seed: int,
    importance_scores: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    importance_scores = importance_scores or {}
    manifest: list[dict[str, Any]] = []
    for task in tasks:
        if task.task_id not in included_task_ids:
            continue
        candidates: list[dict[str, Any]] = []
        task_scores = importance_scores.get(task.task_id, {})
        for rule in RULES:
            occurrences = find_occurrences(rule, task.source_code, task.entrypoint)
            if not occurrences:
                continue
            occurrence = min(
                occurrences,
                key=lambda item: _stable_rank(seed, task.task_id, rule.name, int(item["start_byte"])),
            )
            if rule.name in task_scores:
                score = task_scores[rule.name]
                score_source = "importance_manifest"
            else:
                score = _stable_rank(seed, task.task_id, rule.name, int(occurrence["start_byte"]))
                score_source = "deterministic"
            candidates.append(
                {
                    "rule_name": rule.name,
                    "description": rule.description,
                    "location": occurrence,
                    "score": score,
                    "score_source": score_source,
                }
            )
        candidates.sort(
            key=lambda item: (
                0 if item["score_source"] == "importance_manifest" else 1,
                -float(item["score"]),
                item["rule_name"],
            )
        )
        selected = []
        for stage, candidate in enumerate(candidates[:stages], start=1):
            selected.append({"stage": stage, **candidate})
        manifest.append(
            {
                "task_id": task.task_id,
                "entrypoint": task.entrypoint,
                "stages_requested": stages,
                "available_rule_count": len(candidates),
                "rules": selected,
            }
        )
    return manifest
