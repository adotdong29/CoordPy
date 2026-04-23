"""Wevra runtime configuration.

A small, validated config object that collects runtime settings from:
  1. explicit kwargs
  2. environment variables (``WEVRA_*``)
  3. defaults

Configuration is read from env *when the ``WevraConfig`` is
instantiated*, not at import time. Profiles remain pure declarations;
``WevraConfig`` captures runtime knobs that were previously
hard-coded (e.g. cluster endpoints) and must be operator-configurable.

Env vars
--------
  WEVRA_MODEL            — default model tag (e.g. ``qwen2.5-coder:14b``)
  WEVRA_OLLAMA_URL       — default LLM endpoint
  WEVRA_SANDBOX          — ``in_process`` | ``subprocess`` | ``docker``
  WEVRA_OUT_DIR          — default output directory
  WEVRA_JSONL            — default input JSONL path

No env var is required — every field has a conservative default or is
explicitly ``None`` and then validated by the caller.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any


_VALID_SANDBOXES = ("in_process", "subprocess", "docker")


@dataclasses.dataclass(frozen=True)
class WevraConfig:
    """Runtime configuration for a Wevra invocation.

    Frozen so downstream code cannot mutate config mid-run. Use
    ``dataclasses.replace(cfg, ...)`` to derive a variant.
    """

    model: str | None = None
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
    def from_env(cls, **overrides: Any) -> "WevraConfig":
        """Build a config from ``WEVRA_*`` env vars, with kwargs
        overriding env. Kwargs set to ``None`` are treated as "use env".
        """
        def pick(name: str, default: Any) -> Any:
            v = overrides.get(name)
            if v is not None:
                return v
            env_name = "WEVRA_" + name.upper()
            return os.environ.get(env_name, default)

        return cls(
            model=pick("model", None),
            ollama_url=pick("ollama_url", None),
            sandbox=pick("sandbox", "subprocess"),
            out_dir=pick("out_dir", None),
            jsonl=pick("jsonl", None),
        )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


__all__ = ["WevraConfig"]
