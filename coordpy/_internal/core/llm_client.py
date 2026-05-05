"""Thin Ollama HTTP client — generate + embed.

No dependencies beyond the stdlib. Ollama's REST API uses POST /api/generate
(completions) and /api/embed (embeddings). We talk to localhost:11434.

Simple token counting: we count chars/4 as a proxy for input tokens, and
take the eval_count field from responses for output tokens. Not perfect
but good enough for scaling comparisons across protocols.
"""

from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field


BASE = "http://localhost:11434"


def _post(path: str, payload: dict, timeout: float = 300.0,
           base_url: str | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    url = (base_url or BASE).rstrip("/") + path
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class LLMStats:
    prompt_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    n_generate_calls: int = 0
    n_embed_calls: int = 0
    total_wall: float = 0.0

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.embed_tokens


@dataclass
class LLMClient:
    model: str = "qwen2.5:0.5b"
    stats: LLMStats = field(default_factory=LLMStats)
    timeout: float = 300.0     # seconds — long enough for big-context 7B calls
    base_url: str | None = None  # None → localhost; Phase 42 overrides
                                  # this with the ASPEN cluster node URL.
    think: bool | None = None    # None → omit the top-level ``think`` field
                                  # (preserves Phase 42 byte-for-byte semantics).
                                  # False → opt out of thinking mode on Qwen3
                                  # reasoning models (qwen3.5:35b etc.) so the
                                  # ``response`` field is non-empty even at
                                  # bounded ``max_tokens``; True → opt in.

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.2) -> str:
        t0 = time.time()
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = bool(self.think)
        resp = _post("/api/generate", payload,
                     timeout=self.timeout, base_url=self.base_url)
        self.stats.n_generate_calls += 1
        self.stats.total_wall += time.time() - t0
        self.stats.prompt_tokens += int(resp.get("prompt_eval_count", 0) or 0)
        self.stats.output_tokens += int(resp.get("eval_count", 0) or 0)
        return resp.get("response", "").strip()

    def embed(self, text: str) -> list[float]:
        t0 = time.time()
        resp = _post("/api/embed", {"model": self.model, "input": text},
                     timeout=self.timeout, base_url=self.base_url)
        self.stats.n_embed_calls += 1
        self.stats.total_wall += time.time() - t0
        # rough char-proxy (ollama doesn't always return embed token counts)
        self.stats.embed_tokens += max(1, len(text) // 4)
        emb = resp.get("embeddings") or resp.get("embedding")
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
            return emb[0]
        return emb or []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama supports batched input for /api/embed
        t0 = time.time()
        resp = _post("/api/embed", {"model": self.model, "input": texts},
                     timeout=self.timeout, base_url=self.base_url)
        self.stats.n_embed_calls += 1
        self.stats.total_wall += time.time() - t0
        self.stats.embed_tokens += max(1, sum(len(t) for t in texts) // 4)
        embs = resp.get("embeddings") or []
        return embs
