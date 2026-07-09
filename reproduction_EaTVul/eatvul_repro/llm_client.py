"""AIHubMix LLM client with prompt-response caching."""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging_utils import read_json, write_json


PROVIDER_ERROR_PATTERNS = [
    r"sorry,\s*to prevent abuse",
    r"free resources",
    r"accounts? that have not been recharged",
    r"topup",
    r"insufficient (balance|quota|credit)",
    r"exceed(?:ed|s)? .*quota",
    r"rate limit",
    r"too many requests",
    r"content filter",
    r"blocked by .*policy",
    r"model .* unavailable",
    r"invalid key",
    r"invalid api key",
]


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    sample_idx: int | str
    stage: str
    metadata: dict[str, Any]


class BaseLLMClient:
    def generate(self, request: LLMRequest) -> str:
        raise NotImplementedError


class LLMProviderError(RuntimeError):
    """Raised when the provider returns an error-like text response."""


class MockLLMClient(BaseLLMClient):
    """Deterministic local generator for lightweight debugging."""

    def generate(self, request: LLMRequest) -> str:
        return (
            "int eatvul_noise = 0;\n"
            "for (int eatvul_i = 0; eatvul_i < 3; eatvul_i++) {\n"
            "    eatvul_noise += eatvul_i;\n"
            "}\n"
            "(void)eatvul_noise;"
        )


class AIHubMixClient(BaseLLMClient):
    def __init__(self, llm_cfg: dict[str, Any], root_dir: Path) -> None:
        self.model = str(llm_cfg["model"])
        self.base_url = str(llm_cfg["base_url"])
        self.temperature = float(llm_cfg.get("temperature", 0.7))
        self.max_tokens = int(llm_cfg.get("max_tokens", 1000))
        self.n = int(llm_cfg.get("n", 1))
        self.timeout_seconds = float(llm_cfg.get("timeout_seconds", 60))
        self.max_retries = int(llm_cfg.get("max_retries", 3))
        self.retry_initial_delay_seconds = float(llm_cfg.get("retry_initial_delay_seconds", 2))
        self.api_key_env = str(llm_cfg.get("api_key_env", "AIHUBMIX_API_KEY"))
        cache_dir = Path(str(llm_cfg.get("cache_dir", "outputs/eatvul/cache/llm")))
        self.cache_dir = cache_dir if cache_dir.is_absolute() else root_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        error_cache_dir = Path(str(llm_cfg.get("error_cache_dir", "outputs/eatvul/cache/llm_errors")))
        self.error_cache_dir = error_cache_dir if error_cache_dir.is_absolute() else root_dir / error_cache_dir
        self.error_cache_dir.mkdir(parents=True, exist_ok=True)
        extra_patterns = [str(pattern) for pattern in llm_cfg.get("provider_error_patterns", [])]
        self.provider_error_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in PROVIDER_ERROR_PATTERNS + extra_patterns]

    def generate(self, request: LLMRequest) -> str:
        cache_path = self._cache_path(request)
        if cache_path.exists():
            cached = read_json(cache_path)
            cached_response = str(cached["response"])
            if self._looks_like_provider_error(cached_response):
                self._write_error_cache(request, cached_response, "cached provider error response")
                raise LLMProviderError(f"Cached provider error response detected: {cache_path}")
            return cached_response

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        response = self._request_with_retries(client, request)
        text = response.choices[0].message.content or ""
        if self._looks_like_provider_error(text):
            self._write_error_cache(request, text, "provider error response")
            raise LLMProviderError("AIHubMix returned an error-like text response")
        write_json(
            cache_path,
            {
                "provider": "aihubmix",
                "model": self.model,
                "base_url": self.base_url,
                "sample_idx": request.sample_idx,
                "stage": request.stage,
                "metadata": request.metadata,
                "prompt": request.prompt,
                "response": text,
            },
        )
        return text

    def _request_with_retries(self, client: Any, request: LLMRequest) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": request.prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n,
                )
            except Exception as exc:
                last_error = exc
                message = self._sanitize_error_text(str(exc))
                if not self._is_retryable_exception(message) or attempt >= self.max_retries:
                    self._write_error_cache(request, message, f"exception after attempt {attempt + 1}")
                    raise LLMProviderError(message) from None
                time.sleep(self.retry_initial_delay_seconds * (2**attempt))
        if last_error is not None:
            raise LLMProviderError(self._sanitize_error_text(str(last_error))) from None
        raise RuntimeError("LLM request failed without an exception")

    def _looks_like_provider_error(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.provider_error_patterns)

    @staticmethod
    def _is_retryable_exception(message: str) -> bool:
        lowered = message.lower()
        non_retryable = ["invalid api key", "incorrect api key", "insufficient balance", "not been recharged"]
        if any(fragment in lowered for fragment in non_retryable):
            return False
        retryable = ["timeout", "timed out", "429", "rate limit", "too many requests", "500", "502", "503", "504"]
        return any(fragment in lowered for fragment in retryable)

    @staticmethod
    def _sanitize_error_text(text: str) -> str:
        sanitized = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
        sanitized = re.sub(r"invalid key:\s*[^'\"\s}]+", "invalid key: [REDACTED_API_KEY]", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED_API_KEY]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def _write_error_cache(self, request: LLMRequest, response: str, reason: str) -> None:
        error_path = self.error_cache_dir / self._cache_path(request).name
        write_json(
            error_path,
            {
                "provider": "aihubmix",
                "model": self.model,
                "base_url": self.base_url,
                "sample_idx": request.sample_idx,
                "stage": request.stage,
                "metadata": request.metadata,
                "prompt": request.prompt,
                "response": self._sanitize_error_text(response),
                "reason": reason,
            },
        )

    def _cache_path(self, request: LLMRequest) -> Path:
        key_data = "|".join([self.model, request.stage, str(request.sample_idx), request.prompt])
        digest = hashlib.sha256(key_data.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"


def build_llm_client(llm_cfg: dict[str, Any], root_dir: Path, mock: bool = False) -> BaseLLMClient:
    if mock:
        return MockLLMClient()
    provider = str(llm_cfg.get("provider", "aihubmix")).lower()
    if provider != "aihubmix":
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return AIHubMixClient(llm_cfg, root_dir=root_dir)
