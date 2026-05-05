"""CoordPy runtime configuration.

A small, validated config object that collects runtime settings from:
  1. explicit kwargs
  2. environment variables (``COORDPY_*``)
  3. defaults

Configuration is read from env *when the ``CoordPyConfig`` is
instantiated*, not at import time. Profiles remain pure declarations;
``CoordPyConfig`` captures runtime knobs that were previously
hard-coded (e.g. cluster endpoints) and must be operator-configurable.

Env vars
--------
  COORDPY_MODEL            — default model tag (e.g. ``qwen2.5-coder:14b``)
  COORDPY_BACKEND          — short stable backend selector
  COORDPY_LLM_BACKEND      — legacy/fallback backend selector
  COORDPY_API_BASE_URL     — short stable OpenAI-compatible base URL
  COORDPY_LLM_BASE_URL     — legacy/fallback OpenAI-compatible base URL
  COORDPY_API_KEY          — short stable provider API key
  COORDPY_LLM_API_KEY      — legacy/fallback provider API key
  COORDPY_OLLAMA_URL       — default Ollama endpoint
  COORDPY_SANDBOX          — ``in_process`` | ``subprocess`` | ``docker``
  COORDPY_OUT_DIR          — default output directory
  COORDPY_JSONL            — default input JSONL path

OpenAI-compatible fallbacks
---------------------------
  OPENAI_BASE_URL          — fallback for ``COORDPY_LLM_BASE_URL``
  OPENAI_API_KEY           — fallback for ``COORDPY_LLM_API_KEY``

No env var is required — every field has a conservative default or is
explicitly ``None`` and then validated by the caller.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any


_VALID_SANDBOXES = ("in_process", "subprocess", "docker")


@dataclasses.dataclass(frozen=True)
class CoordPyConfig:
    """Runtime configuration for a CoordPy invocation.

    Frozen so downstream code cannot mutate config mid-run. Use
    ``dataclasses.replace(cfg, ...)`` to derive a variant.
    """

    model: str | None = None
    llm_backend: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    ollama_url: str | None = None
    sandbox: str = "subprocess"
    out_dir: str | None = None
    jsonl: str | None = None

    def __post_init__(self) -> None:
        if self.sandbox not in _VALID_SANDBOXES:
            raise ValueError(
                f"invalid sandbox {self.sandbox!r}; "
                f"must be one of {_VALID_SANDBOXES}")

    @classmethod
    def from_env(cls, **overrides: Any) -> "CoordPyConfig":
        """Build a config from ``COORDPY_*`` env vars, with kwargs
        overriding env. Kwargs set to ``None`` are treated as "use env".
        """
        def pick(name: str, default: Any) -> Any:
            v = overrides.get(name)
            if v is not None:
                return v
            env_name = "COORDPY_" + name.upper()
            return os.environ.get(env_name, default)

        def pick_api_key() -> str | None:
            v = overrides.get("llm_api_key")
            if v is not None:
                return v
            return (
                os.environ.get("COORDPY_API_KEY")
                or os.environ.get("COORDPY_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

        def pick_backend() -> str | None:
            v = overrides.get("llm_backend")
            if v is not None:
                return v
            return (
                os.environ.get("COORDPY_BACKEND")
                or os.environ.get("COORDPY_LLM_BACKEND")
            )

        def pick_base_url() -> str | None:
            v = overrides.get("llm_base_url")
            if v is not None:
                return v
            return (
                os.environ.get("COORDPY_API_BASE_URL")
                or os.environ.get("COORDPY_LLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )

        return cls(
            model=pick("model", None),
            llm_backend=pick_backend(),
            llm_base_url=pick_base_url(),
            llm_api_key=pick_api_key(),
            ollama_url=pick("ollama_url", None),
            sandbox=pick("sandbox", "subprocess"),
            out_dir=pick("out_dir", None),
            jsonl=pick("jsonl", None),
        )

    def resolved_backend_name(self) -> str | None:
        if self.llm_backend:
            return self.llm_backend
        if self.llm_base_url or self.llm_api_key:
            return "openai_compatible"
        if self.ollama_url:
            return "ollama"
        return None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


__all__ = ["CoordPyConfig"]
