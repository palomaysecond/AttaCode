from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntaxResult:
    valid: bool
    reason: str
    backend: str


class CppSyntaxChecker:
    def __init__(self) -> None:
        try:
            from tree_sitter_languages import get_parser
        except ImportError as exc:
            raise RuntimeError(
                "Tree-sitter is required for generation. Install the dependencies in requirements.txt."
            ) from exc
        try:
            self._parser = get_parser("cpp")
        except Exception as exc:
            raise RuntimeError("Could not initialize the Tree-sitter C++ parser") from exc
        self._backend = "tree-sitter-languages"

    @property
    def backend(self) -> str:
        return self._backend

    def check(self, code: str) -> SyntaxResult:
        if not code.strip():
            return SyntaxResult(False, "empty code", self._backend)
        tree = self._parser.parse(code.encode("utf-8"))
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "ERROR" or getattr(node, "is_missing", False):
                return SyntaxResult(
                    False,
                    f"tree-sitter reported {node.type} at {node.start_point}",
                    self._backend,
                )
            stack.extend(node.children)
        return SyntaxResult(True, "ok", self._backend)
