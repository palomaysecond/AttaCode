from __future__ import annotations

from pathlib import Path
from typing import Any


def load_templates(paths: list[Path]) -> list[str]:
    templates = [path.read_text(encoding="utf-8") for path in paths]
    if not templates:
        raise ValueError("At least one prompt template is required")
    required = {"{source_code}", "{rule_name}", "{rule_description}", "{entrypoint}", "{stage}"}
    for path, template in zip(paths, templates):
        missing = [field for field in required if field not in template]
        if missing:
            raise ValueError(f"Prompt template {path} is missing required fields: {missing}")
    return templates


def render_prompt(template: str, source_code: str, rule: dict[str, Any], entrypoint: str, stage: int) -> str:
    location = rule.get("location") or {}
    if not location:
        raise ValueError(f"Rule {rule.get('rule_name')} does not contain a target location")
    location_text = f"line {location['line']}, near `{location['snippet']}`"
    return template.format(
        source_code=source_code.rstrip(),
        rule_name=rule["rule_name"],
        rule_description=rule["description"],
        location=location_text,
        entrypoint=entrypoint,
        stage=stage,
    )
