from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None


class LLMClient:
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, model_config: dict[str, Any], generation_config: dict[str, Any]) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the dependencies in requirements.txt before using API models") from exc

        api_key_env = model_config.get("api_key_env")
        api_key = os.getenv(api_key_env, "") if api_key_env else ""
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

        base_url = model_config.get("base_url")
        base_url_env = model_config.get("base_url_env")
        if base_url_env:
            base_url = os.getenv(base_url_env) or base_url

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = str(model_config["model"])
        self.system_prompt = model_config.get("system_prompt")
        self.generation_config = generation_config

    def generate(self, prompt: str) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": str(self.system_prompt)})
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.generation_config.get("temperature", 0.3)),
            "top_p": float(self.generation_config.get("top_p", 1.0)),
            "max_tokens": int(self.generation_config.get("max_tokens", 4096)),
        }
        seed = self.generation_config.get("seed")
        if seed is not None:
            request["seed"] = int(seed)
        response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""
        usage: dict[str, Any] = {}
        if getattr(response, "usage", None) is not None:
            usage_object = response.usage
            usage = {
                "prompt_tokens": getattr(usage_object, "prompt_tokens", None),
                "completion_tokens": getattr(usage_object, "completion_tokens", None),
                "total_tokens": getattr(usage_object, "total_tokens", None),
            }
        return LLMResponse(text=text.strip(), usage=usage, response_id=getattr(response, "id", None))


def build_client(model_config: dict[str, Any], generation_config: dict[str, Any]) -> LLMClient:
    provider = str(model_config.get("provider", "openai_compatible")).lower()
    if provider == "openai_compatible":
        return OpenAICompatibleClient(model_config, generation_config)
    raise ValueError(f"Unsupported model provider: {provider}")
