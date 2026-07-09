"""Syntax and lightweight semantic validation for generated snippets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .insertion import InsertionSite, insert_snippet

C_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "bool",
    "true",
    "false",
    "NULL",
}

IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
PROVIDER_MESSAGE_PATTERN = re.compile(
    r"free resources|not been recharged|topup|quota|rate limit|content filter|"
    r"as an ai|i cannot|i can't|sorry|here is|generated code|explanation",
    re.IGNORECASE,
)
CODE_LIKE_PATTERN = re.compile(
    r"("
    r"\b(?:int|char|short|long|float|double|size_t|bool|unsigned|uint\d+_t)\s+\*?\s*[A-Za-z_]"
    r"|[A-Za-z_][A-Za-z0-9_]*\s*(?:=|\+=|-=|\*=|/=|%=)"
    r"|\b(?:for|if|while)\s*\("
    r"|\(void\)\s*[A-Za-z_]"
    r"|;"
    r")"
)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
    modified_code: str | None = None


class SnippetValidator:
    def __init__(self, cfg: dict[str, object]) -> None:
        self.reject_control_flow_jumps = bool(cfg.get("reject_control_flow_jumps", True))
        self.reject_original_identifier_use = bool(cfg.get("reject_original_identifier_use", True))
        self.reject_provider_messages = bool(cfg.get("reject_provider_messages", True))
        self.reject_urls = bool(cfg.get("reject_urls", True))
        self.require_code_like_snippet = bool(cfg.get("require_code_like_snippet", True))
        self.require_eatvul_prefix = bool(cfg.get("require_eatvul_prefix", True))

    def validate(self, original_code: str, snippet: str, site: InsertionSite) -> ValidationResult:
        snippet = snippet.strip()
        if not snippet:
            return ValidationResult(False, "empty snippet")
        if self.reject_provider_messages and self._looks_like_provider_message(snippet):
            return ValidationResult(False, "provider or natural-language message")
        if self.reject_urls and URL_PATTERN.search(snippet):
            return ValidationResult(False, "snippet contains URL")
        if self.require_code_like_snippet and not self._looks_like_code(snippet):
            return ValidationResult(False, "snippet is not code-like")
        if self.require_eatvul_prefix and "eatvul_" not in snippet:
            return ValidationResult(False, "snippet does not declare eatvul-prefixed local variables")
        if self.reject_control_flow_jumps and self._has_control_flow_jump(snippet):
            return ValidationResult(False, "snippet changes control flow")
        if self.reject_original_identifier_use and self._uses_original_identifiers(original_code, snippet):
            return ValidationResult(False, "snippet reuses original identifiers")

        modified_code = insert_snippet(original_code, snippet, site)
        if not self._balanced_delimiters(modified_code):
            return ValidationResult(False, "unbalanced delimiters after insertion")
        return ValidationResult(True, "ok", modified_code=modified_code)

    @staticmethod
    def _has_control_flow_jump(snippet: str) -> bool:
        return bool(re.search(r"\b(return|goto|break|continue)\b", snippet))

    @staticmethod
    def _looks_like_provider_message(snippet: str) -> bool:
        return bool(PROVIDER_MESSAGE_PATTERN.search(snippet))

    @staticmethod
    def _looks_like_code(snippet: str) -> bool:
        stripped = _strip_strings_and_comments_text(snippet)
        if not CODE_LIKE_PATTERN.search(stripped):
            return False
        alpha_words = re.findall(r"\b[A-Za-z]{3,}\b", stripped)
        semicolon_count = stripped.count(";")
        brace_count = stripped.count("{") + stripped.count("}")
        if len(alpha_words) > 20 and semicolon_count + brace_count < 2:
            return False
        return True

    @staticmethod
    def _uses_original_identifiers(original_code: str, snippet: str) -> bool:
        original = extract_identifiers(original_code)
        generated = extract_identifiers(snippet)
        generated_defs = _declared_identifiers(snippet)
        risky = generated.intersection(original) - generated_defs
        allowed_common = {"size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t"}
        return bool(risky - allowed_common)

    @staticmethod
    def _balanced_delimiters(code: str) -> bool:
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack: list[str] = []
        for char in _strip_strings_and_comments(code):
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if not stack or stack.pop() != char:
                    return False
        return not stack


def extract_identifiers(code: str) -> set[str]:
    return {token for token in IDENTIFIER_PATTERN.findall(code) if token not in C_KEYWORDS}


def _declared_identifiers(snippet: str) -> set[str]:
    declared: set[str] = set()
    declaration_pattern = re.compile(
        r"\b(?:int|char|short|long|float|double|size_t|bool|uint\d+_t)\s+\*?\s*([A-Za-z_][A-Za-z0-9_]*)"
    )
    for match in declaration_pattern.finditer(snippet):
        declared.add(match.group(1))
    for match in re.finditer(r"\bfor\s*\(\s*(?:int|size_t)\s+([A-Za-z_][A-Za-z0-9_]*)", snippet):
        declared.add(match.group(1))
    return declared


def _strip_strings_and_comments(code: str) -> Iterable[str]:
    return _strip_strings_and_comments_text(code)


def _strip_strings_and_comments_text(code: str) -> str:
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code)
    return code
