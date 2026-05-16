"""W68 H3 — Hosted Cache-Aware Planner (Plane A).

Plans the per-turn prompts in a multi-turn run so that prefix-cache
hits on hosted providers (OpenRouter, OpenAI, etc.) are maximised.
Operates at the **content-addressed prefix-CID** layer:

* For each turn the planner concatenates a stable shared prefix
  (team instructions + capsule view header + per-task base) and a
  small variable suffix (the new role-specific block).
* The planner computes the prefix CID across turns and predicts
  the hosted-cache hit rate as ``(n_reused_prefix_tokens /
  n_total_prefix_tokens)``.
* Cross-plane bridge: the planner can ALSO report the
  prefix CIDs to the **real** V13 substrate via the
  ``substrate_prefix_reuse_counter`` axis (Plane B). The bridge is
  one-way: the planner tells the substrate ``"this prefix was
  re-used"``; the substrate counts hits and exposes the counter.

Honest scope (W68)
------------------

* Hosted-cache hit-rate estimates rely on the **declared** caching
  policy of the provider. ``W68-L-HOSTED-PREFIX-CACHE-DECLARED-CAP``.
* The cross-plane bridge does NOT prove that the hosted provider
  actually reused the prefix in its KV cache; it only proves that
  the same prefix CID was sent. ``W68-L-HOSTED-PREFIX-CID-NOT-KV-CAP``.
* The savings estimate is per-token at the visible-text surface;
  it is NOT a real-substrate KV reuse measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from .tiny_substrate_v3 import _sha256_hex


W68_HOSTED_CACHE_AWARE_PLANNER_SCHEMA_VERSION: str = (
    "coordpy.hosted_cache_aware_planner.v1")


def compute_prefix_cid(
        text: str, *, n_chars: int | None = None,
) -> str:
    """SHA256-derived prefix CID for a text snippet."""
    s = (text if n_chars is None
         else text[:int(n_chars)])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class HostedPlannedTurn:
    turn: int
    shared_prefix_text: str
    role_block_text: str
    shared_prefix_cid: str
    full_prompt_cid: str
    shared_prefix_tokens_est: int
    role_block_tokens_est: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": int(self.turn),
            "shared_prefix_cid": str(self.shared_prefix_cid),
            "full_prompt_cid": str(self.full_prompt_cid),
            "shared_prefix_tokens_est": int(
                self.shared_prefix_tokens_est),
            "role_block_tokens_est": int(
                self.role_block_tokens_est),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_planned_turn",
            "turn": self.to_dict()})


@dataclasses.dataclass
class HostedCacheAwarePlanner:
    chars_per_token: float = 4.0
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W68_HOSTED_CACHE_AWARE_PLANNER_SCHEMA_VERSION,
            "kind": "hosted_cache_aware_planner",
            "chars_per_token": float(round(
                self.chars_per_token, 12)),
        })

    def plan(
            self, *,
            shared_prefix_text: str,
            role_blocks: Sequence[str],
    ) -> tuple[
            tuple[HostedPlannedTurn, ...], dict[str, Any]]:
        prefix_cid = compute_prefix_cid(str(shared_prefix_text))
        cpt = float(self.chars_per_token)
        prefix_tokens = int(max(
            1, int(len(shared_prefix_text) / max(1.0, cpt))))
        planned: list[HostedPlannedTurn] = []
        for i, rb in enumerate(role_blocks):
            full_text = str(shared_prefix_text) + "\n" + str(rb)
            full_cid = compute_prefix_cid(full_text)
            role_tokens = int(max(
                1, int(len(rb) / max(1.0, cpt))))
            planned.append(HostedPlannedTurn(
                turn=int(i),
                shared_prefix_text=str(shared_prefix_text),
                role_block_text=str(rb),
                shared_prefix_cid=str(prefix_cid),
                full_prompt_cid=str(full_cid),
                shared_prefix_tokens_est=int(prefix_tokens),
                role_block_tokens_est=int(role_tokens),
            ))
        n_turns = int(len(planned))
        # Hosted-cache hit rate estimate: prefix reused on turns >= 1.
        if n_turns <= 1:
            est_hit_rate = 0.0
        else:
            reuse_turns = n_turns - 1
            reused_prefix_tokens = (
                int(reuse_turns * prefix_tokens))
            total_prefix_tokens = (
                int(n_turns * prefix_tokens))
            est_hit_rate = (
                float(reused_prefix_tokens)
                / float(max(1, total_prefix_tokens)))
        # Naive vs cache-aware cost ratio: cache-aware saves the
        # prefix on every reuse.
        naive_input_tokens = int(n_turns * prefix_tokens
                                 + sum(p.role_block_tokens_est
                                       for p in planned))
        cache_aware_input_tokens = int(
            prefix_tokens
            + sum(p.role_block_tokens_est for p in planned))
        savings_ratio = (
            float(naive_input_tokens
                  - cache_aware_input_tokens)
            / float(max(1, naive_input_tokens))
            if naive_input_tokens > 0 else 0.0)
        report = {
            "schema":
                W68_HOSTED_CACHE_AWARE_PLANNER_SCHEMA_VERSION,
            "kind": "cache_aware_plan_report",
            "n_turns": int(n_turns),
            "shared_prefix_cid": str(prefix_cid),
            "shared_prefix_tokens_est": int(prefix_tokens),
            "estimated_hosted_cache_hit_rate": float(round(
                est_hit_rate, 12)),
            "naive_input_tokens": int(naive_input_tokens),
            "cache_aware_input_tokens": int(
                cache_aware_input_tokens),
            "input_token_savings_ratio": float(round(
                savings_ratio, 12)),
        }
        self.audit.append({
            "kind": "plan",
            "n_turns": int(n_turns),
            "shared_prefix_cid": str(prefix_cid),
            "savings_ratio": float(round(savings_ratio, 12)),
        })
        return tuple(planned), report


def hosted_cache_aware_savings_vs_recompute(
        *, n_turns: int, prefix_tokens_per_turn: int,
        role_tokens_per_turn: int,
        hosted_cache_hit_rate: float = 1.0,
) -> dict[str, Any]:
    """Estimate token savings from cache-aware planning vs full
    recompute on every turn."""
    n = int(max(0, n_turns))
    p = int(max(0, prefix_tokens_per_turn))
    r = int(max(0, role_tokens_per_turn))
    naive = int(n * (p + r))
    if n <= 1:
        cache = naive
    else:
        cache = int(
            p   # first turn: full prefix
            + (n - 1) * int(
                (1.0 - float(hosted_cache_hit_rate)) * float(p))
            + n * r)
    saving = int(naive - cache)
    ratio = (
        float(saving) / float(naive)
        if naive > 0 else 0.0)
    return {
        "n_turns": int(n),
        "naive_input_tokens": int(naive),
        "cache_aware_input_tokens": int(cache),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
        "hosted_cache_hit_rate": float(round(
            hosted_cache_hit_rate, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedCacheAwarePlannerWitness:
    schema: str
    planner_cid: str
    n_plans: int
    last_savings_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "planner_cid": str(self.planner_cid),
            "n_plans": int(self.n_plans),
            "last_savings_ratio": float(round(
                self.last_savings_ratio, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_cache_aware_planner_witness",
            "witness": self.to_dict()})


def emit_hosted_cache_aware_planner_witness(
        planner: HostedCacheAwarePlanner,
) -> HostedCacheAwarePlannerWitness:
    last = 0.0
    for e in planner.audit:
        if "savings_ratio" in e:
            last = float(e["savings_ratio"])
    return HostedCacheAwarePlannerWitness(
        schema=W68_HOSTED_CACHE_AWARE_PLANNER_SCHEMA_VERSION,
        planner_cid=str(planner.cid()),
        n_plans=int(len(planner.audit)),
        last_savings_ratio=float(last),
    )


__all__ = [
    "W68_HOSTED_CACHE_AWARE_PLANNER_SCHEMA_VERSION",
    "compute_prefix_cid",
    "HostedPlannedTurn",
    "HostedCacheAwarePlanner",
    "hosted_cache_aware_savings_vs_recompute",
    "HostedCacheAwarePlannerWitness",
    "emit_hosted_cache_aware_planner_witness",
]
