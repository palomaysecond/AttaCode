"""EaTVul fragmented-context prompt construction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .insertion import InsertionSite


@dataclass(frozen=True)
class PromptPayload:
    prompt: str
    key_tokens: list[str]
    site_description: str


class EaTVulPromptBuilder:
    def __init__(self, max_snippet_lines: int = 8) -> None:
        self.max_snippet_lines = max_snippet_lines

    def build(self, key_tokens: list[str], site: InsertionSite) -> PromptPayload:
        token_text = ", ".join(key_tokens) if key_tokens else "local C/C++ syntax"
        prompt = (
            "With the partial preceding codes provided as:\n"
            f"{site.preceding_context}\n\n"
            "Please generate a short vulnerability-free C/C++ code snippet. "
            "The snippet should be semantically isolated from the original function, "
            "declare and use its own local variables, and avoid changing return values, "
            "control flow, or existing variables. "
            "Do not use any identifier, variable, field, function, macro, typedef, or custom type "
            "that appears in the preceding or following code. "
            "Use only standard scalar types such as int, unsigned int, long, or size_t. "
            "Every local variable must be declared inside the snippet and must use the prefix eatvul_. "
            f"The snippet should be no more than {self.max_snippet_lines} lines. "
            f"Try to reflect these important code features or feature categories: {token_text}.\n\n"
            "With the partial following codes provided as:\n"
            f"{site.succeeding_context}\n\n"
            "Return only raw C/C++ statements that can be inserted inside the function body. "
            "Do not include markdown fences, explanations, comments, URLs, or prose. "
            "The returned snippet must declare and use at least one eatvul_-prefixed local variable."
        )
        return PromptPayload(prompt=prompt, key_tokens=key_tokens, site_description=site.description)


def extract_code_snippet(response: str) -> str:
    text = response.strip()
    fenced = re.search(r"```(?:c|cpp|c\+\+)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
