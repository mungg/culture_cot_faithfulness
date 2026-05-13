"""
Model-agnostic client wrapper for the experiment pipeline.

Usage:
    from _models import get_client
    client = get_client("gemini/gemini-2.5-flash")   # via Vertex AI, user=yekyung
    client = get_client("gemini/gemini-2.0-flash")
    client = get_client("groq/llama-3.3-70b-versatile")
    client = get_client("groq/qwen-2.5-32b")
    client = get_client("openai/gpt-4o-mini")

Each client exposes:
    client.generate_structured(prompt, schema, *, temperature, max_tokens,
                                want_thinking=False, thinking_budget=8000)
        → {"text": str, "json": dict|None, "thinking": str}

The pipeline only depends on this interface, so swapping models requires
no other changes in 01_baseline.py / 02_hints.py.
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

USER_LABEL = "yekyung"
DEFAULT_PROJECT = "gen-lang-client-0966014990"
DEFAULT_LOCATION = "us-central1"


class ModelClient(ABC):
    name: str  # provider/model id

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        want_thinking: bool = False,
        thinking_budget: int = 8000,
    ) -> dict[str, Any]:
        """Returns {'text': raw text, 'json': parsed dict or None, 'thinking': str}."""


# ─── Gemini via Vertex AI ────────────────────────────────────────────────────
class GeminiVertexClient(ModelClient):
    def __init__(self, model: str, project: str = DEFAULT_PROJECT, location: str = DEFAULT_LOCATION):
        from google import genai
        self.name = f"gemini/{model}"
        self.model = model
        self._client = genai.Client(vertexai=True, project=project, location=location)
        # Whether this model variant supports thinking (2.5+, not 2.0)
        self.supports_thinking = "2.5" in model or "3" in model
        # Cache the types module
        from google.genai import types as _types
        self._types = _types

    def generate_structured(self, prompt, schema, *, temperature=0.7, max_tokens=2048,
                            want_thinking=False, thinking_budget=8000):
        types = self._types
        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
            labels={"user": USER_LABEL},
        )
        if want_thinking and self.supports_thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget, include_thoughts=True,
            )
        cfg = types.GenerateContentConfig(**cfg_kwargs)

        for attempt in range(3):
            try:
                r = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                thinking, text = "", ""
                for part in r.candidates[0].content.parts:
                    if getattr(part, "thought", None):
                        thinking += part.text or ""
                    else:
                        text += part.text or ""
                parsed = None
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    pass
                return {"text": text, "json": parsed, "thinking": thinking}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "json": None, "thinking": ""}


# ─── Groq (Llama-3.3, Qwen, Mixtral, …) ──────────────────────────────────────
class GroqClient(ModelClient):
    def __init__(self, model: str):
        from groq import Groq
        self.name = f"groq/{model}"
        self.model = model
        self._client = Groq()  # uses GROQ_API_KEY
        self.supports_thinking = False

    def generate_structured(self, prompt, schema, *, temperature=0.7, max_tokens=2048,
                            want_thinking=False, thinking_budget=None):
        # Groq supports JSON mode via response_format
        for attempt in range(3):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": f"You must respond with valid JSON matching this schema: {json.dumps(schema)}"},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = r.choices[0].message.content or ""
                parsed = None
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    pass
                return {"text": text, "json": parsed, "thinking": ""}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "json": None, "thinking": ""}


# ─── OpenAI (GPT-4o, GPT-4o-mini, …) ─────────────────────────────────────────
class OpenAIClient(ModelClient):
    def __init__(self, model: str):
        from openai import OpenAI
        self.name = f"openai/{model}"
        self.model = model
        self._client = OpenAI()  # uses OPENAI_API_KEY
        self.supports_thinking = False

    def generate_structured(self, prompt, schema, *, temperature=0.7, max_tokens=2048,
                            want_thinking=False, thinking_budget=None):
        for attempt in range(3):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": f"Respond with valid JSON matching this schema: {json.dumps(schema)}"},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = r.choices[0].message.content or ""
                parsed = None
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    pass
                return {"text": text, "json": parsed, "thinking": ""}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "json": None, "thinking": ""}


# ─── Factory ─────────────────────────────────────────────────────────────────
def get_client(model_spec: str) -> ModelClient:
    """
    model_spec format: "<provider>/<model_name>"

      gemini/gemini-2.5-flash         # via Vertex AI, with user=yekyung
      gemini/gemini-2.0-flash
      gemini/gemini-1.5-pro
      groq/llama-3.3-70b-versatile
      groq/qwen-2.5-32b
      groq/qwen-3-32b
      openai/gpt-4o-mini
    """
    if "/" not in model_spec:
        raise ValueError(f"model_spec must be 'provider/model', got: {model_spec}")
    provider, model = model_spec.split("/", 1)
    if provider == "gemini":
        return GeminiVertexClient(model=model)
    if provider == "groq":
        return GroqClient(model=model)
    if provider == "openai":
        return OpenAIClient(model=model)
    raise ValueError(f"Unknown provider: {provider}. Add it to _models.py.")
