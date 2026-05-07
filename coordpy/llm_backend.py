"""LLM backend abstraction (SDK v3.6).

Lifts the runtime's LLM-call surface above the Ollama-only HTTP path
that ``coordpy._internal.core.llm_client.LLMClient`` historically represented.
Adds a small duck-typed Protocol matching what the runtime's inner
loop already expects (``generate``) and three concrete backend names:

  * ``OllamaBackend``         — wraps the existing Ollama client
                                 byte-for-byte unchanged.
  * ``OpenAICompatibleBackend`` — talks an OpenAI-compatible
                                  ``/v1/chat/completions`` endpoint.
  * ``MLXDistributedBackend`` — backwards-compatible alias for the
                                 same OpenAI-compatible transport,
                                 historically named for
                                 ``mlx_lm.server`` under
                                 ``mx.distributed`` / ``mpirun``.

Strict additivity. The new surface is layered *under* the existing
inner-loop call site (``_real_cells`` already accepts a duck-typed
``llm_client=`` substitute). When no backend is supplied, the
runtime instantiates ``LLMClient`` exactly as before; the W3-34
spine equivalence is preserved by construction (no new spine kinds,
no new payload shapes — the PROMPT / LLM_RESPONSE capsules still
record the prompt / response bytes' SHA-256 + length + bounded
snippet, regardless of which backend produced them).

Honest scope of this module
---------------------------

This module specifies and implements *the integration boundary*
between CoordPy and a distributed-inference path. It does NOT bring
up MLX distributed itself — that requires:

  * An out-of-process ``mpirun --hostfile <hosts>`` spanning the
    Apple Silicon machines.
  * A converted MLX model on each host.
  * ``mlx_lm.server`` started on the head node (rank 0) under the
    distributed launcher; ``mx.distributed.init`` then negotiates
    the tensor / pipeline split with the worker rank.

Operator instructions live in
``docs/MLX_DISTRIBUTED_RUNBOOK.md``. The role of this module is to
define the CoordPy-side adapter so a CoordPy run can target either
backend through the same ``SweepSpec`` / runtime call site.

Why this is the right abstraction (one paragraph)
--------------------------------------------------

The MLX distributed sharding strategy (tensor parallel + pipeline
parallel via ``mx.distributed`` collectives over MPI) is a property
of *how* the model is loaded on the Apple Silicon side; once a
single ``mlx_lm.server`` is bound to ``host:port``, the HTTP wire
shape is OpenAI-compatible regardless of whether the model is
single-host or sharded across N hosts. The CoordPy adapter is
therefore neutral on the sharding strategy. This is the smallest
honest integration boundary: one HTTP client, one Protocol, one
adapter class. We deliberately do NOT take on the cluster
bringup, the MPI configuration, or the model conversion — those
are out-of-band operator concerns and belong in the runbook, not
in the SDK.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """The minimum surface the CoordPy inner loop expects from an LLM
    backend. Matches the duck-type contract of
    ``coordpy._internal.core.llm_client.LLMClient.generate``.

    A backend MUST expose:

      * ``model``    : str — the canonical model tag recorded in
                              the PROMPT / LLM_RESPONSE capsules'
                              ``model_tag`` field.
      * ``base_url`` : str | None — provenance only; not used at
                                     call time by the runtime.
      * ``generate(prompt, max_tokens, temperature) -> str`` — the
        synchronous text-generation entrypoint.

    The Protocol is ``runtime_checkable`` so call sites can assert
    backend conformance with a single ``isinstance(b, LLMBackend)``.
    """

    model: str
    base_url: "str | None"

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str: ...


@dataclasses.dataclass
class OllamaBackend:
    """``LLMBackend`` adapter over
    ``coordpy._internal.core.llm_client.LLMClient``.

    Default behaviour is byte-for-byte identical to the existing
    runtime's implicit Ollama path: the runtime's old code path was
    ``LLMClient(model=spec.model, base_url=spec.endpoint,
    timeout=spec.llm_timeout)`` and this backend constructs the
    same object behind the scenes.
    """

    model: str
    base_url: "str | None" = None
    timeout: float = 300.0
    think: "bool | None" = None
    _client: "Any" = dataclasses.field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        from coordpy._internal.core.llm_client import LLMClient
        self._client = LLMClient(
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
            think=self.think,
        )

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        try:
            return self._client.generate(
                prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            # Wrap urllib's bare "Connection refused" / "Name or
            # service not known" with a clear message that names
            # the URL and the env var, so users don't have to
            # guess where to look.
            msg = str(e).lower()
            if any(k in msg for k in (
                "connection refused",
                "name or service not known",
                "temporary failure in name resolution",
                "no route to host",
                "network is unreachable",
            )):
                url = self.base_url or "http://127.0.0.1:11434"
                raise ConnectionError(
                    f"OllamaBackend could not reach {url} "
                    f"({type(e).__name__}: {e}). Check that an "
                    f"Ollama server is running there, or set "
                    f"COORDPY_OLLAMA_URL / pass base_url=..."
                ) from e
            raise


@dataclasses.dataclass
class OpenAICompatibleBackend:
    """``LLMBackend`` adapter for an OpenAI-compatible HTTP endpoint.

    Wire shape: ``POST <base_url>/v1/chat/completions`` with the
    standard OpenAI request body. ``mlx_lm.server`` (Apple's
    reference HTTP server for MLX models) implements this surface.

    The HTTP API is identical regardless of whether the underlying
    model is single-host or sharded across N hosts via
    ``mx.distributed`` (tensor parallel + pipeline parallel over
    MPI). The CoordPy adapter is therefore *neutral* on the sharding
    strategy: the integration boundary is one HTTP client.

    Idempotent on content: two calls with byte-identical prompts
    produce byte-identical responses provided the upstream server is
    deterministic (``temperature=0`` + fixed seed). The CoordPy
    capsule layer (PROMPT / LLM_RESPONSE) records SHA-256 of the
    bytes, so cached / deduplicated calls collapse to one capsule
    pair on the DAG (Capsule Contract C1).

    Honest scope: this class is the *client*. Bringing up the
    sharded model is out of band (see
    ``docs/MLX_DISTRIBUTED_RUNBOOK.md``). On hosts where the MLX
    server has not been started, calls will raise
    ``urllib.error.URLError`` like any other HTTP unavailability;
    the runtime handles that with the same error path as a
    misconfigured Ollama endpoint.
    This backend is provider-neutral as long as the provider exposes an
    OpenAI-compatible chat-completions endpoint. By default,
    ``from_env()`` targets the public OpenAI API; point
    ``COORDPY_API_BASE_URL`` or ``OPENAI_BASE_URL`` at any compatible
    endpoint to reuse the same surface with other providers.
    """

    model: str
    base_url: str
    timeout: float = 600.0
    api_key: "str | None" = None
    n_calls: int = 0
    total_wall_s: float = 0.0
    last_response_payload: "dict | None" = dataclasses.field(
        default=None, repr=False)

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Wrap the most common misconfig modes with a clear,
            # actionable message that names the URL and the env
            # var to set, so users don't have to read urllib's
            # bare ``HTTP Error 401`` and guess.
            if e.code in (401, 403):
                raise PermissionError(
                    f"OpenAICompatibleBackend got HTTP {e.code} "
                    f"from {url}. Check that COORDPY_API_KEY (or "
                    f"the ``api_key=`` argument) is set to a "
                    f"valid token for this provider."
                ) from e
            if e.code == 404:
                raise ValueError(
                    f"OpenAICompatibleBackend got HTTP 404 from "
                    f"{url}. The base_url should be the root that "
                    f"hosts ``/v1/chat/completions`` — e.g. "
                    f"``https://api.openai.com/v1`` (without the "
                    f"``/chat/completions`` suffix)."
                ) from e
            raise
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"OpenAICompatibleBackend could not reach {url} "
                f"({type(e).__name__}: {e}). Check that the host "
                f"is reachable and COORDPY_API_BASE_URL / "
                f"base_url= is correct."
            ) from e
        self.n_calls += 1
        self.total_wall_s += time.time() - t0
        self.last_response_payload = payload
        choices = payload.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> "OpenAICompatibleBackend":
        """Build an OpenAI-compatible backend from common env vars.

        Resolution order:

        1. explicit kwargs
        2. ``COORDPY_MODEL`` / ``COORDPY_API_BASE_URL`` /
           ``COORDPY_API_KEY``
        3. legacy ``COORDPY_LLM_BASE_URL`` / ``COORDPY_LLM_API_KEY``
        4. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
        5. OpenAI default base URL
        """
        resolved_model = model or os.environ.get("COORDPY_MODEL")
        if not resolved_model:
            raise ValueError(
                "model is required; pass model=... or set COORDPY_MODEL")
        resolved_base = (
            base_url
            or os.environ.get("COORDPY_API_BASE_URL")
            or os.environ.get("COORDPY_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com"
        )
        resolved_key = (
            api_key
            or os.environ.get("COORDPY_API_KEY")
            or os.environ.get("COORDPY_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        return cls(
            model=resolved_model,
            base_url=resolved_base,
            api_key=resolved_key,
            timeout=timeout,
        )


@dataclasses.dataclass
class MLXDistributedBackend(OpenAICompatibleBackend):
    """Backward-compatible alias for the MLX/OpenAI-compatible path.

    Historically the OpenAI-compatible backend entered the SDK as the
    MLX distributed integration boundary. It remains exported under
    this name for compatibility, but the general product-facing name is
    :class:`OpenAICompatibleBackend`.
    """


def make_backend(name: str, **kwargs: Any) -> LLMBackend:
    """Factory dispatch by string name.

    Valid names:

      * ``"ollama"``           — :class:`OllamaBackend`
      * ``"openai"``           — :class:`OpenAICompatibleBackend`
      * ``"openai_compatible"`` — :class:`OpenAICompatibleBackend`
      * ``"mlx_distributed"``  — :class:`MLXDistributedBackend`
    """
    if name == "ollama":
        return OllamaBackend(**kwargs)
    if name in ("openai", "openai_compatible", "provider"):
        return OpenAICompatibleBackend(**kwargs)
    if name == "mlx_distributed":
        return MLXDistributedBackend(**kwargs)
    raise ValueError(
        f"unknown LLM backend {name!r}; "
        f"valid: 'ollama', 'openai', 'openai_compatible', "
        f"'mlx_distributed'")


def backend_from_env(
    name: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    think: bool | None = None,
) -> LLMBackend:
    """Build a backend from explicit args plus ``COORDPY_*`` env vars.

    This is the easiest path for end users who want to supply a model
    provider API key without learning the lower-level backend classes.
    """
    backend_name = (
        name
        or os.environ.get("COORDPY_BACKEND")
        or os.environ.get("COORDPY_LLM_BACKEND")
    )
    if backend_name is None:
        if (
            os.environ.get("COORDPY_API_BASE_URL")
            or os.environ.get("COORDPY_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("COORDPY_API_KEY")
            or os.environ.get("COORDPY_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ):
            backend_name = "openai_compatible"
        else:
            backend_name = "ollama"
    if backend_name == "ollama":
        resolved_model = model or os.environ.get("COORDPY_MODEL") or "qwen2.5:0.5b"
        resolved_base = (
            base_url
            or os.environ.get("COORDPY_OLLAMA_URL")
        )
        return OllamaBackend(
            model=resolved_model,
            base_url=resolved_base,
            timeout=300.0 if timeout is None else float(timeout),
            think=think,
        )
    if backend_name in ("openai", "openai_compatible", "provider"):
        return OpenAICompatibleBackend.from_env(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=600.0 if timeout is None else float(timeout),
        )
    if backend_name == "mlx_distributed":
        resolved_model = model or os.environ.get("COORDPY_MODEL")
        resolved_base = (
            base_url
            or os.environ.get("COORDPY_API_BASE_URL")
            or os.environ.get("COORDPY_LLM_BASE_URL")
        )
        if not resolved_model or not resolved_base:
            raise ValueError(
                "mlx_distributed backend requires model and base_url; "
                "pass them explicitly or set COORDPY_MODEL and "
                "COORDPY_LLM_BASE_URL")
        resolved_key = (
            api_key
            or os.environ.get("COORDPY_API_KEY")
            or os.environ.get("COORDPY_LLM_API_KEY")
        )
        return MLXDistributedBackend(
            model=resolved_model,
            base_url=resolved_base,
            api_key=resolved_key,
            timeout=600.0 if timeout is None else float(timeout),
        )
    raise ValueError(
        f"unknown backend {backend_name!r}; valid: 'ollama', 'openai', "
        f"'openai_compatible', 'provider', 'mlx_distributed'")


def backend_from_config(
    config: Any | None,
    *,
    model: str,
    endpoint: str | None = None,
    timeout: float = 600.0,
) -> LLMBackend | None:
    """Build a backend from a ``CoordPyConfig``-shaped object.

    Resolution order:
      1. explicit backend on the config
      2. inferred OpenAI-compatible backend when a base URL or API key exists
      3. inferred Ollama backend when only an Ollama URL exists
      4. ``None`` to preserve the legacy runtime path
    """
    if config is None:
        return None
    backend_name = None
    resolver = getattr(config, "resolved_backend_name", None)
    if callable(resolver):
        backend_name = resolver()
    else:
        backend_name = getattr(config, "llm_backend", None)
    if backend_name is None:
        return None
    if backend_name == "ollama":
        return OllamaBackend(
            model=model,
            base_url=(getattr(config, "ollama_url", None) or endpoint),
            timeout=timeout,
        )
    if backend_name in {"openai", "openai_compatible", "provider"}:
        base_url = (
            getattr(config, "llm_base_url", None)
            or "https://api.openai.com"
        )
        if not base_url:
            raise ValueError(
                "OpenAI-compatible backend requires llm_base_url/base_url")
        return make_backend(
            backend_name,
            model=model,
            base_url=base_url,
            api_key=getattr(config, "llm_api_key", None),
            timeout=timeout,
        )
    if backend_name == "mlx_distributed":
        base_url = (
            getattr(config, "llm_base_url", None)
            or getattr(config, "ollama_url", None)
            or endpoint
        )
        if not base_url:
            raise ValueError(
                "MLX backend requires llm_base_url/base_url")
        return make_backend(
            backend_name,
            model=model,
            base_url=base_url,
            api_key=getattr(config, "llm_api_key", None),
            timeout=timeout,
        )
    raise ValueError(
        f"unsupported backend resolved from config: {backend_name!r}")


__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "MLXDistributedBackend",
    "make_backend",
    "backend_from_env",
    "backend_from_config",
]
