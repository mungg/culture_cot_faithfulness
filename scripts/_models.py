"""
Model-agnostic client wrapper for the experiment pipeline.

Usage:
    from _models import get_client
    client = get_client("gemini/gemini-2.5-flash")   # via Vertex AI
    client = get_client("gemini/gemini-2.0-flash")
    client = get_client("groq/llama-3.3-70b-versatile")
    client = get_client("groq/qwen-2.5-32b")
    client = get_client("openai/gpt-4o-mini")

Vertex/Gemini knobs are read from env vars (with sensible defaults):
    VERTEX_PROJECT       (required for non-author Vertex projects)
    VERTEX_LOCATION      (default us-central1)
    VERTEX_USER_LABEL    (billing label, default 'anon')

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
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import Any

# Vertex AI knobs — overridable via environment variables so collaborators
# can use their own project without editing source.
USER_LABEL = os.environ.get("VERTEX_USER_LABEL", "anon")
DEFAULT_PROJECT = os.environ.get("VERTEX_PROJECT", "gen-lang-client-0966014990")
DEFAULT_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


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

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        want_thinking: bool = False,
        thinking_budget: int = 8000,
    ) -> dict[str, Any]:
        """Free-form text generation (no JSON constraint).
        Returns {'text': str, 'thinking': str}. Used for safety/refusal-type tasks."""


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

    def generate_text(self, prompt, *, temperature=0.7, max_tokens=2048,
                      want_thinking=False, thinking_budget=8000):
        types = self._types
        cfg_kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
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
                return {"text": text, "thinking": thinking}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "thinking": ""}


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

    def generate_text(self, prompt, *, temperature=0.7, max_tokens=2048,
                      want_thinking=False, thinking_budget=None):
        for attempt in range(3):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return {"text": r.choices[0].message.content or "", "thinking": ""}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "thinking": ""}


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

    def generate_text(self, prompt, *, temperature=0.7, max_tokens=2048,
                      want_thinking=False, thinking_budget=None):
        for attempt in range(3):
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return {"text": r.choices[0].message.content or "", "thinking": ""}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "thinking": ""}


# ─── Local HuggingFace / Transformers ────────────────────────────────────────
class HFLocalClient(ModelClient):
    def __init__(self, model: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import import_utils as hf_import_utils

        self.name = f"hf/{model}"
        self.model = model
        self.supports_thinking = False
        self._torch = torch

        # These runs are text-only. Some cluster envs have optional vision /
        # flash-attn packages installed but ABI-mismatched, so we disable those
        # backends before importing model classes.
        hf_import_utils._torchvision_available = False
        hf_import_utils._torchvision_version = "0.0"
        hf_import_utils.is_flash_attn_2_available = lambda: False
        hf_import_utils.is_flash_attn_3_available = lambda: False
        sys.modules.pop("torchvision", None)
        sys.modules.pop("flash_attn", None)

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        dtype_name = os.environ.get("HF_DTYPE", "bfloat16")
        torch_dtype = getattr(torch, dtype_name, None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported HF_DTYPE={dtype_name!r}")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": os.environ.get("HF_DEVICE_MAP", "auto"),
        }
        attn_impl = os.environ.get("HF_ATTN_IMPL")
        model_kwargs["attn_implementation"] = attn_impl or "eager"
        self._model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"

    def _generate(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        prompt = self._render_messages(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        model_device = self._model.get_input_embeddings().weight.device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with self._torch.no_grad():
            output = self._model.generate(**inputs, **generate_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_structured(self, prompt, schema, *, temperature=0.7, max_tokens=2048,
                            want_thinking=False, thinking_budget=None):
        del want_thinking, thinking_budget
        schema_text = json.dumps(schema, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "Return valid JSON only. Do not wrap it in markdown. "
                    f"Your response must match this schema: {schema_text}"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for attempt in range(3):
            try:
                text = self._generate(messages, temperature=temperature, max_tokens=max_tokens)
                return {"text": text, "json": _extract_first_json_object(text), "thinking": ""}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "json": None, "thinking": ""}

    def generate_text(self, prompt, *, temperature=0.7, max_tokens=2048,
                      want_thinking=False, thinking_budget=None):
        del want_thinking, thinking_budget
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                text = self._generate(messages, temperature=temperature, max_tokens=max_tokens)
                return {"text": text, "thinking": ""}
            except Exception as e:
                print(f"    [{self.name}] err ({attempt+1}): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return {"text": "", "thinking": ""}


# ─── Factory ─────────────────────────────────────────────────────────────────
def get_client(model_spec: str) -> ModelClient:
    """
    model_spec format: "<provider>/<model_name>"

      gemini/gemini-2.5-flash         # via Vertex AI (set VERTEX_PROJECT)
      gemini/gemini-2.0-flash
      gemini/gemini-1.5-pro
      groq/llama-3.3-70b-versatile
      groq/qwen-2.5-32b
      groq/qwen-3-32b
      openai/gpt-4o-mini
      hf/CohereForAI/aya-expanse-8b
      hf/Qwen/Qwen3-8B
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
    if provider == "hf":
        return HFLocalClient(model=model)
    raise ValueError(f"Unknown provider: {provider}. Add it to _models.py.")
