"""Raw-source insertion-site discovery for function-level C/C++ snippets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InsertionSite:
    line_index: int
    description: str
    preceding_context: str
    succeeding_context: str


def find_insertion_sites(code: str, context_window_lines: int = 5) -> list[InsertionSite]:
    lines = code.splitlines()
    if len(lines) < 3:
        return []

    first_body = _find_first_body_line(lines)
    last_body = _find_last_body_line(lines)
    if first_body is None or last_body is None or first_body >= last_body:
        return []

    candidate_lines: list[int] = []
    for idx in range(first_body + 1, last_body):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            continue
        if stripped.endswith((";", "{", "}")) or stripped.startswith(("if ", "for ", "while ", "switch ")):
            candidate_lines.append(idx + 1)

    if not candidate_lines:
        candidate_lines = [first_body + 1]

    unique_lines = sorted(set(line for line in candidate_lines if first_body < line <= last_body))
    return [
        InsertionSite(
            line_index=line,
            description=f"before line {line + 1}",
            preceding_context="\n".join(lines[max(0, line - context_window_lines) : line]),
            succeeding_context="\n".join(lines[line : min(len(lines), line + context_window_lines)]),
        )
        for line in unique_lines
    ]


def insert_snippet(code: str, snippet: str, site: InsertionSite) -> str:
    lines = code.splitlines()
    indent = _infer_indent(lines, site.line_index)
    snippet_lines = [indent + line if line.strip() else line for line in snippet.strip().splitlines()]
    new_lines = lines[: site.line_index] + snippet_lines + lines[site.line_index :]
    return "\n".join(new_lines)


def _find_first_body_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if "{" in line:
            return index
    return None


def _find_last_body_line(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if "}" in lines[index]:
            return index
    return None


def _infer_indent(lines: list[str], line_index: int) -> str:
    for idx in range(line_index, min(len(lines), line_index + 5)):
        line = lines[idx]
        stripped = line.lstrip()
        if stripped:
            return line[: len(line) - len(stripped)]
    return "    "

