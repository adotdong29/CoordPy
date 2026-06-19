"""W56 M9 — Deep Substrate Hybrid Stack.

Replaces the *top* ``L_out`` layers of the W55 deep proxy stack V6
with the *real* tiny substrate attention block. Reads / writes
the real per-layer KV cache. Provides the first load-bearing
"real attention block in the loop" path in the Context Zero
programme.

Conceptually:

  bottom  --- W55 deep proxy stack V6 layers (capsule-layer) ---
              |
              | (forward residual stream)
              v
  middle  --- KV bridge: project residual to per-layer K/V slots
              and write them into the tiny substrate's cache
              |
              v
  top     --- Tiny substrate attention block (real) — reads the
              injected K/V plus a follow-up "query" embedding,
              produces real attention weights + a real residual
              update
              |
              v
  out     --- Final aggregation: substrate top hidden state is
              the new outer output

Honest scope (do-not-overstate)
-------------------------------

* This still does NOT touch third-party hosted models. The
  "real attention" runs in ``coordpy.tiny_substrate`` only.
* The W55-L-OVERDEPTH-V6-CAP carries forward: more depth does
  not guarantee strict gain.
* The substrate runs in NumPy on CPU; per-step cost is
  ``O(n_tokens · d_model · n_heads)`` — fine for benchmarks,
  not for production.

What this buys
--------------

* The W56 loop has a path where *some* of the forward pass is
  computed by a real attention block over a real KV cache. The
  H13 / H14 benchmark cells measure whether ablating the
  substrate block changes the output relative to a capsule-only
  baseline — i.e., whether the substrate path is load-bearing
  rather than decorative.
* It demonstrates that the latent operating system can compose
  with a real substrate object without breaking
  content-addressing or replay-determinism.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.deep_substrate_hybrid requires numpy") from exc

from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    DeepProxyStackV6ForwardWitness,
)
from .kv_bridge import (
    KVBridgeProjection,
    inject_carrier_into_kv_cache,
)
from .tiny_substrate import (
    TinyKVCache,
    TinySubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate,
)


W56_DEEP_SUBSTRATE_HYBRID_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid.v1")
W56_DEFAULT_HYBRID_QUERY_TOKEN: int = 257  # BOS as a stable query
W56_DEFAULT_HYBRID_ABSTAIN_THRESHOLD: float = 0.10
W56_DEFAULT_HYBRID_BASE_ABSTAIN: float = 0.10
W56_DEFAULT_HYBRID_GAMMA: float = 0.05


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class DeepSubstrateHybrid:
    """The hybrid stack: W55 V6 below + tiny substrate above.

    Both halves are deterministic; the hybrid forward is replay-
    deterministic given the V6 inputs + the substrate carrier +
    the substrate params + the bridge projection seed.
    """

    deep_v6: DeepProxyStackV6
    substrate: TinySubstrateParams
    bridge: KVBridgeProjection
    base_abstain_threshold: float = (
        W56_DEFAULT_HYBRID_BASE_ABSTAIN)
    gamma: float = W56_DEFAULT_HYBRID_GAMMA

    @classmethod
    def init(
            cls, *,
            deep_v6: DeepProxyStackV6,
            substrate: TinySubstrateParams,
            n_inject_tokens: int = 2,
            carrier_dim: int | None = None,
            bridge_seed: int = 56091234,
            base_abstain_threshold: float = (
                W56_DEFAULT_HYBRID_BASE_ABSTAIN),
            gamma: float = W56_DEFAULT_HYBRID_GAMMA,
    ) -> "DeepSubstrateHybrid":
        cd = (
            int(carrier_dim)
            if carrier_dim is not None
            else int(deep_v6.inner_v5.inner_v4.factor_dim))
        bridge = KVBridgeProjection.init(
            n_layers=int(substrate.config.n_layers),
            n_inject_tokens=int(n_inject_tokens),
            carrier_dim=int(cd),
            d_model=int(substrate.config.d_model),
            seed=int(bridge_seed))
        return cls(
            deep_v6=deep_v6,
            substrate=substrate,
            bridge=bridge,
            base_abstain_threshold=float(base_abstain_threshold),
            gamma=float(gamma),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W56_DEEP_SUBSTRATE_HYBRID_SCHEMA_VERSION,
            "deep_v6_cid": str(self.deep_v6.cid()),
            "substrate_cid": str(self.substrate.cid()),
            "bridge_cid": str(self.bridge.cid()),
            "base_abstain_threshold": float(
                round(self.base_abstain_threshold, 12)),
            "gamma": float(round(self.gamma, 12)),
        })

    def compute_adaptive_threshold(
            self, query_input: Sequence[float],
    ) -> float:
        l2 = math.sqrt(sum(
            float(v) * float(v) for v in query_input))
        return float(
            self.base_abstain_threshold
            * (1.0 + self.gamma
               * math.log(1.0 + max(0.0, l2))))


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridForwardWitness:
    """Hybrid forward witness.

    Binds: V6 forward witness CID, substrate forward CID,
    bridge injection CID, KV cache CID before/after,
    measured perturbation magnitude vs ablated baseline, and
    a corruption_confidence scalar.
    """

    schema: str
    deep_v6_witness_cid: str
    substrate_forward_cid: str
    bridge_injection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    ablation_perturbation_l2: float
    adaptive_threshold: float
    abstain: bool
    corruption_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "deep_v6_witness_cid": str(self.deep_v6_witness_cid),
            "substrate_forward_cid": str(
                self.substrate_forward_cid),
            "bridge_injection_cid": str(
                self.bridge_injection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(self.post_inject_kv_cid),
            "ablation_perturbation_l2": float(round(
                self.ablation_perturbation_l2, 12)),
            "adaptive_threshold": float(round(
                self.adaptive_threshold, 12)),
            "abstain": bool(self.abstain),
            "corruption_confidence": float(round(
                self.corruption_confidence, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_forward(
        *,
        hybrid: DeepSubstrateHybrid,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
        trust_scalar: float = 1.0,
        uncertainty_scale: float = 1.0,
        substrate_query_token_ids: Sequence[int] | None = None,
        substrate_kv_cache: TinyKVCache | None = None,
) -> tuple["_np.ndarray",
            DeepSubstrateHybridForwardWitness,
            TinyKVCache]:
    """One hybrid forward pass.

    1. Run the W55 V6 deep stack forward to obtain the residual
       output.
    2. Inject the V6 residual into the tiny substrate's KV cache
       via the bridge.
    3. Run the tiny substrate forward over
       ``substrate_query_token_ids`` (default: BOS) using the
       injected cache.
    4. Compute the ablation perturbation: difference between the
       injected-substrate output and an un-injected substrate
       forward on the same query.

    Returns:
      * ``substrate_top_hidden`` — the substrate's final hidden
        state (``d_model``-vector at the last query position);
      * a content-addressed forward witness;
      * the post-forward KV cache (for downstream reuse).
    """
    # Step 1 — run the V6 deep stack.
    out, v6_witness = (
        # DeepProxyStackV6 exposes forward_value via
        # `forward_value` style helper; we use the convenience
        # `emit_deep_proxy_stack_v6_forward_witness` to keep the
        # witness consistent.
        _v6_forward_capture(
            stack=hybrid.deep_v6,
            query_input=list(query_input),
            slot_keys=[list(k) for k in slot_keys],
            slot_values=[list(v) for v in slot_values],
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index),
            trust_scalar=float(trust_scalar),
            uncertainty_scale=float(uncertainty_scale)))
    # Step 2 — bridge the V6 residual into the substrate KV
    # cache.
    base_cache = (
        substrate_kv_cache.clone()
        if substrate_kv_cache is not None
        else TinyKVCache.empty(int(
            hybrid.substrate.config.n_layers)))
    pre_inject_cid = base_cache.cid()
    injected_cache, injection_record = (
        inject_carrier_into_kv_cache(
            carrier=list(out),
            projection=hybrid.bridge,
            kv_cache=base_cache,
            position="prepend"))
    post_inject_cid = injected_cache.cid()
    # Step 3 — run the substrate forward.
    if substrate_query_token_ids is None:
        substrate_query_token_ids = [
            W56_DEFAULT_HYBRID_QUERY_TOKEN]
    inj_trace = forward_tiny_substrate(
        hybrid.substrate,
        list(substrate_query_token_ids),
        kv_cache=injected_cache,
        return_attention=False)
    # Step 4 — ablation baseline: un-injected substrate forward.
    abl_trace = forward_tiny_substrate(
        hybrid.substrate,
        list(substrate_query_token_ids),
        kv_cache=base_cache,
        return_attention=False)
    # Measure substrate-vs-ablation perturbation.
    inj_hidden = inj_trace.hidden_states[-1]
    abl_hidden = abl_trace.hidden_states[-1]
    perturb_l2 = float(_np.linalg.norm(
        inj_hidden[-1] - abl_hidden[-1]))
    # Adaptive abstain (carries forward W55-T-DEEP-V6-ADAPTIVE-
    # ABSTAIN-MONOTONICITY into the hybrid).
    thr = hybrid.compute_adaptive_threshold(query_input)
    abstain = bool(perturb_l2 < thr)
    # Corruption confidence: 1 - normalised perturbation in
    # [0, 1].
    corruption_confidence = float(
        1.0 / (1.0 + max(0.0, perturb_l2)))
    witness = DeepSubstrateHybridForwardWitness(
        schema=W56_DEEP_SUBSTRATE_HYBRID_SCHEMA_VERSION,
        deep_v6_witness_cid=str(v6_witness.cid()),
        substrate_forward_cid=str(inj_trace.cid()),
        bridge_injection_cid=str(injection_record.cid()),
        pre_inject_kv_cid=str(pre_inject_cid),
        post_inject_kv_cid=str(post_inject_cid),
        ablation_perturbation_l2=float(perturb_l2),
        adaptive_threshold=float(thr),
        abstain=abstain,
        corruption_confidence=float(corruption_confidence),
    )
    return inj_hidden[-1], witness, inj_trace.kv_cache


def _v6_forward_capture(
        *, stack: DeepProxyStackV6,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int,
        branch_index: int,
        cycle_index: int,
        trust_scalar: float,
        uncertainty_scale: float,
) -> tuple[list[float], DeepProxyStackV6ForwardWitness]:
    """Run a W55 V6 deep stack forward and capture a witness.

    Uses the existing ``emit_deep_proxy_stack_v6_forward_witness``
    helper but returns both the residual output (last layer's
    head value) and the witness.
    """
    # The W55 module exposes a forward_value method on the stack.
    # We import the helper inline to avoid a circular import in
    # the module header.
    from .deep_proxy_stack_v6 import (
        emit_deep_proxy_stack_v6_forward_witness,
    )
    witness, head_value = (
        emit_deep_proxy_stack_v6_forward_witness(
            stack=stack,
            query_input=list(query_input),
            slot_keys=[list(k) for k in slot_keys],
            slot_values=[list(v) for v in slot_values],
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index),
            trust_scalar=float(trust_scalar),
            uncertainty_scale=float(uncertainty_scale)))
    return list(head_value), witness


__all__ = [
    "W56_DEEP_SUBSTRATE_HYBRID_SCHEMA_VERSION",
    "W56_DEFAULT_HYBRID_QUERY_TOKEN",
    "W56_DEFAULT_HYBRID_BASE_ABSTAIN",
    "W56_DEFAULT_HYBRID_GAMMA",
    "DeepSubstrateHybrid",
    "DeepSubstrateHybridForwardWitness",
    "deep_substrate_hybrid_forward",
]
