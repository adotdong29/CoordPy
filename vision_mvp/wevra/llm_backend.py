"""LLM backend abstraction (SDK v3.6).

Lifts the runtime's LLM-call surface above the Ollama-only HTTP path
that ``vision_mvp.core.llm_client.LLMClient`` historically represented.
Adds a small duck-typed Protocol matching what the runtime's inner
loop already expects (``generate``) and two concrete backends:

  * ``OllamaBackend``         — wraps the existing Ollama client
                                 byte-for-byte unchanged.
  * ``MLXDistributedBackend`` — talks an OpenAI-compatible
                                 ``/v1/chat/completions`` endpoint.
                                 Designed for an ``mlx_lm.server``
                                 launched under ``mx.distributed`` /
                                 ``mpirun`` across multiple Apple
                                 Silicon hosts so a *single* sharded
                                 model spans the cluster.

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
between Wevra and a distributed-inference path. It does NOT bring
up MLX distributed itself — that requires:

  * An out-of-process ``mpirun --hostfile <hosts>`` spanning the
    Apple Silicon machines.
  * A converted MLX model on each host.
  * ``mlx_lm.server`` started on the head node (rank 0) under the
    distributed launcher; ``mx.distributed.init`` then negotiates
    the tensor / pipeline split with the worker rank.

Operator instructions live in
``docs/MLX_DISTRIBUTED_RUNBOOK.md``. The role of this module is to
define the Wevra-side adapter so a Wevra run can target either
backend through the same ``SweepSpec`` / runtime call site.

Why this is the right abstraction (one paragraph)
--------------------------------------------------

The MLX distributed sharding strategy (tensor parallel + pipeline
parallel via ``mx.distributed`` collectives over MPI) is a property
of *how* the model is loaded on the Apple Silicon side; once a
single ``mlx_lm.server`` is bound to ``host:port``, the HTTP wire
shape is OpenAI-compatible regardless of whether the model is
single-host or sharded across N hosts. The Wevra adapter is
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
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """The minimum surface the Wevra inner loop expects from an LLM
    backend. Matches the duck-type contract of
    ``vision_mvp.core.llm_client.LLMClient.generate``.

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
    ``vision_mvp.core.llm_client.LLMClient``.

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
        from vision_mvp.core.llm_client import LLMClient
        self._client = LLMClient(
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
            think=self.think,
        )

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        return self._client.generate(
            prompt, max_tokens=max_tokens, temperature=temperature)


@dataclasses.dataclass
class MLXDistributedBackend:
    """``LLMBackend`` adapter for an OpenAI-compatible HTTP endpoint.

    Wire shape: ``POST <base_url>/v1/chat/completions`` with the
    standard OpenAI request body. ``mlx_lm.server`` (Apple's
    reference HTTP server for MLX models) implements this surface.

    The HTTP API is identical regardless of whether the underlying
    model is single-host or sharded across N hosts via
    ``mx.distributed`` (tensor parallel + pipeline parallel over
    MPI). The Wevra adapter is therefore *neutral* on the sharding
    strategy: the integration boundary is one HTTP client.

    Idempotent on content: two calls with byte-identical prompts
    produce byte-identical responses provided the upstream server is
    deterministic (``temperature=0`` + fixed seed). The Wevra
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
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        self.n_calls += 1
        self.total_wall_s += time.time() - t0
        self.last_response_payload = payload
        choices = payload.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()


def make_backend(name: str, **kwargs: Any) -> LLMBackend:
    """Factory dispatch by string name.

    Valid names:

      * ``"ollama"``           — :class:`OllamaBackend`
      * ``"mlx_distributed"``  — :class:`MLXDistributedBackend`
    """
    if name == "ollama":
        return OllamaBackend(**kwargs)
    if name == "mlx_distributed":
        return MLXDistributedBackend(**kwargs)
    raise ValueError(
        f"unknown LLM backend {name!r}; "
        f"valid: 'ollama', 'mlx_distributed'")


__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "MLXDistributedBackend",
    "make_backend",
]
