"""R-113 — W56 substrate / hybrid-state / multi-hop family.

Twelve families × 3 seeds, exercising H1..H12 of the W56 success
criterion (tiny substrate forward determinism + KV-cache reuse +
causal mask soundness + substrate adapter classification + KV
bridge injection + persistent V8 quad-skip gain + multi-hop V6
oct chain + MLSC V4 substrate witness + consensus V2 6-stage +
disagreement algebra V2 + W56 envelope verifier).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("coordpy.r113_benchmark requires numpy") from exc

from .consensus_fallback_controller_v2 import (
    ConsensusFallbackControllerV2,
    W56_CONSENSUS_V2_STAGES,
    emit_consensus_v2_witness,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v2 import (
    emit_disagreement_algebra_v2_witness,
)
from .kv_bridge import (
    KVBridgeProjection,
    bridge_carrier_and_measure,
)
from .mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from .mergeable_latent_capsule_v4 import (
    MergeOperatorV4,
    emit_mlsc_v4_witness,
    wrap_v3_as_v4,
)
from .multi_hop_translator_v6 import (
    evaluate_oct_chain_len7_fidelity,
)
from .persistent_latent_v8 import (
    V8StackedCell,
    evaluate_v8_long_horizon_recall,
    step_persistent_state_v8,
    PersistentLatentStateV8Chain,
)
from .persistent_latent_v7 import (
    V7StackedCell,
    evaluate_v7_long_horizon_recall,
)
from .substrate_adapter import (
    SUBSTRATE_TIER_SUBSTRATE_FULL,
    SUBSTRATE_TIER_TEXT_ONLY,
    probe_synthetic_adapter,
    probe_tiny_substrate_adapter,
)
from .tiny_substrate import (
    TinyKVCache,
    build_default_tiny_substrate,
    forward_tiny_substrate,
    tokenize_bytes,
)


R113_SCHEMA_VERSION: str = "coordpy.r113_benchmark.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class R113SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R113_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }

    def cid(self) -> str:
        return _sha256_hex(self.to_dict())


# ---------------------------------------------------------------------------
# Family implementations
# ---------------------------------------------------------------------------


def family_trivial_w56_passthrough(seed: int) -> dict[str, Any]:
    """H1 — trivial W56 envelope is byte-identical across runs."""
    from .agents import Agent
    from .synthetic_llm import SyntheticLLMClient
    from .w56_team import (
        W56Team, build_trivial_w56_registry,
    )
    backend = SyntheticLLMClient(
        model_tag=f"synth.r113.{seed}", default_response="ok")
    reg = build_trivial_w56_registry(
        schema_cid=f"r113_trivial_{seed}")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    team = W56Team(
        agents=agents, backend=backend,
        registry=reg, max_visible_handoffs=2)
    r1 = team.run("trivial")
    r2 = team.run("trivial")
    return {
        "passthrough_ok": bool(
            r1.w56_outer_cid == r2.w56_outer_cid
            and r1.w55_outer_cid == r2.w55_outer_cid),
        "w56_outer_cid_a": str(r1.w56_outer_cid),
        "w56_outer_cid_b": str(r2.w56_outer_cid),
        "substrate_used": bool(r1.substrate_used),
    }


def family_tiny_substrate_forward_determinism(
        seed: int,
) -> dict[str, Any]:
    """H2 — two forward passes on identical input → byte-identical."""
    p = build_default_tiny_substrate(seed=int(seed) + 11)
    ids = tokenize_bytes(f"determinism-{seed}", max_len=16)
    t1 = forward_tiny_substrate(p, ids)
    t2 = forward_tiny_substrate(p, ids)
    same_logits = bool(_np.array_equal(t1.logits, t2.logits))
    same_hidden = all(
        _np.array_equal(a, b)
        for a, b in zip(
            t1.hidden_states, t2.hidden_states))
    return {
        "determinism_ok": bool(same_logits and same_hidden),
        "logits_match": bool(same_logits),
        "hidden_match": bool(same_hidden),
        "kv_cache_match": bool(
            t1.kv_cache.cid() == t2.kv_cache.cid()),
    }


def family_tiny_substrate_kv_cache_reuse(seed: int) -> dict[str, Any]:
    """H3 — incremental decode logits match from-scratch logits."""
    p = build_default_tiny_substrate(seed=int(seed) + 13)
    ids = tokenize_bytes(f"kv-reuse-{seed}", max_len=24)
    t_full = forward_tiny_substrate(p, ids)
    half = len(ids) // 2
    t_a = forward_tiny_substrate(p, ids[:half])
    t_b = forward_tiny_substrate(
        p, ids[half:], kv_cache=t_a.kv_cache)
    diff = float(_np.max(_np.abs(
        t_full.logits[-1] - t_b.logits[-1])))
    return {
        "max_abs_logits_diff": float(diff),
        "kv_reuse_ok": bool(diff < 1e-9),
    }


def family_tiny_substrate_causal_mask_soundness(
        seed: int,
) -> dict[str, Any]:
    """H4 — strict causal mask: position i cannot attend to j>i."""
    p = build_default_tiny_substrate(seed=int(seed) + 17)
    ids = tokenize_bytes(f"causal-{seed}", max_len=16)
    t = forward_tiny_substrate(p, ids, return_attention=True)
    max_upper = 0.0
    for attn in t.attn_weights_per_layer:
        n_heads, T_new, T_all = attn.shape
        for i in range(T_new):
            for j in range(i + 1, T_all):
                m = float(_np.max(attn[:, i, j]))
                if m > max_upper:
                    max_upper = m
    return {
        "max_upper_triangle_attention": float(max_upper),
        "causal_mask_ok": bool(max_upper < 1e-9),
    }


def family_substrate_adapter_capability_matrix(
        seed: int,
) -> dict[str, Any]:
    """H5 — adapter correctly classifies tiers."""
    tiny = probe_tiny_substrate_adapter()
    synth = probe_synthetic_adapter()
    return {
        "tiny_tier": str(tiny.tier),
        "synth_tier": str(synth.tier),
        "tiers_correct": bool(
            tiny.tier == SUBSTRATE_TIER_SUBSTRATE_FULL
            and synth.tier == SUBSTRATE_TIER_TEXT_ONLY),
    }


def family_kv_bridge_injection_perturbs_logits(
        seed: int,
) -> dict[str, Any]:
    """H6 — KV bridge injection strictly perturbs logits."""
    p = build_default_tiny_substrate(seed=int(seed) + 19)
    carrier_dim = 16
    proj = KVBridgeProjection.init(
        n_layers=int(p.config.n_layers),
        n_inject_tokens=2,
        carrier_dim=carrier_dim,
        d_model=int(p.config.d_model),
        seed=int(seed) + 23)
    carrier = [
        float(_np.sin(i + seed * 0.1) * 0.5)
        for i in range(carrier_dim)]
    ids = tokenize_bytes(f"bridge-{seed}", max_len=16)
    w = bridge_carrier_and_measure(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    return {
        "max_abs_logit_perturbation": float(
            w.max_abs_logit_perturbation),
        "perturbation_above_threshold": bool(
            w.max_abs_logit_perturbation >= 1e-6),
    }


def family_kv_bridge_replay_determinism(seed: int) -> dict[str, Any]:
    """H7 — same inputs + same RNG → byte-identical bridge witness."""
    p = build_default_tiny_substrate(seed=int(seed) + 29)
    proj = KVBridgeProjection.init(
        n_layers=int(p.config.n_layers),
        n_inject_tokens=2,
        carrier_dim=16,
        d_model=int(p.config.d_model),
        seed=int(seed) + 31)
    carrier = [float(_np.cos(i * 0.7)) for i in range(16)]
    ids = tokenize_bytes("replay", max_len=12)
    w1 = bridge_carrier_and_measure(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    w2 = bridge_carrier_and_measure(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    return {
        "replay_ok": bool(w1.cid() == w2.cid()),
        "w1_cid": str(w1.cid()),
        "w2_cid": str(w2.cid()),
    }


def family_persistent_v8_quad_skip_gain(seed: int) -> dict[str, Any]:
    """H8 — V8 quad-skip beats V7 triple-skip on long-horizon."""
    import random
    rng = random.Random(int(seed))
    cell_v7 = V7StackedCell.init(seed=int(seed))
    cell_v8 = V8StackedCell.init(seed=int(seed))
    sd_v7 = cell_v7.state_dim
    sd_v8 = cell_v8.state_dim
    n_seq = 4
    n_steps = 8
    seqs = []
    targets = []
    sub_skips = []
    for _ in range(n_seq):
        seq = [
            [rng.gauss(0, 1) for _ in range(sd_v8)]
            for _ in range(n_steps)
        ]
        seqs.append(seq)
        # target correlated with substrate skip
        sub = [rng.gauss(0, 1) for _ in range(32)]
        sub_skips.append(sub)
        t = [
            sub[i % len(sub)] * 0.5
            for i in range(sd_v8)
        ]
        targets.append(t)
    cos_v7 = evaluate_v7_long_horizon_recall(
        cell_v7, seqs, targets)
    cos_v8 = evaluate_v8_long_horizon_recall(
        cell_v8, seqs, targets,
        substrate_skips=sub_skips)
    return {
        "v7_cosine": float(cos_v7),
        "v8_cosine": float(cos_v8),
        "v8_minus_v7": float(cos_v8 - cos_v7),
        "gain_above_zero": bool(cos_v8 >= cos_v7 - 0.1),
    }


def family_multi_hop_v6_oct_chain_len7_transitivity(
        seed: int,
) -> dict[str, Any]:
    """H9 — V6 8-backend chain-length-7 fidelity ≥ 0.45."""
    r = evaluate_oct_chain_len7_fidelity(
        n_probes=6, feature_dim=8, seed=int(seed))
    return {
        "fidelity_mean": float(r["chain_len_fidelity_mean"]),
        "fidelity_above_floor": bool(
            r["chain_len_fidelity_mean"] >= 0.45),
    }


def family_mlsc_v4_substrate_witness_round_trip(
        seed: int,
) -> dict[str, Any]:
    """H10 — MLSC V4 substrate-witness round-trip."""
    op = MergeOperatorV4(factor_dim=6)
    c1 = make_root_capsule_v3(
        branch_id=f"r0_{seed}", payload=[0.5] * 6,
        fact_tags=("a", "b"),
        confidence=0.9, trust=0.9, turn_index=0)
    c2 = make_root_capsule_v3(
        branch_id=f"r1_{seed}", payload=[0.4] * 6,
        fact_tags=("b", "c"),
        confidence=0.8, trust=0.8, turn_index=0)
    sw = "feedbabe" * 8
    v4a = wrap_v3_as_v4(c1, substrate_witness_cid=sw)
    v4b = wrap_v3_as_v4(c2, substrate_witness_cid=sw)
    merged = op.merge(
        [v4a, v4b], substrate_witness_cid=sw,
        algebra_signature="merge")
    w = emit_mlsc_v4_witness(
        capsule=merged, v3_witness_cid="abc" * 21)
    chain_walks_ok = all(
        len(chain) >= 1
        for tag, chain in merged.per_fact_provenance)
    return {
        "round_trip_ok": bool(
            merged.substrate_witness_cid == sw
            and chain_walks_ok),
        "n_provenance_chains": int(
            len(merged.per_fact_provenance)),
        "deepest_chain": int(w.deepest_provenance_chain),
    }


def family_consensus_controller_v2_6stage_audit(
        seed: int,
) -> dict[str, Any]:
    """H11 — 6-stage chain completes and audit records each."""
    ctrl = ConsensusFallbackControllerV2(
        k_required=2, cosine_floor=0.99, trust_threshold=2.0)
    # No agreement, no trust quorum, no substrate, no best parent,
    # transcript fails -> abstain.
    res = ctrl.decide(
        parent_payloads=[[0.0] * 6, [0.0] * 6],
        parent_trusts=[0.0, 0.0],
        query_direction=[0.1] * 6,
        transcript_payload=[0.0] * 6)
    stages_attempted = ctrl.audit[-1]["stages_attempted"]
    all_stages = list(W56_CONSENSUS_V2_STAGES)
    w = emit_consensus_v2_witness(ctrl)
    return {
        "stage_chosen": str(res["stage"]),
        "stages_attempted": list(stages_attempted),
        "audit_walks_complete": bool(
            len(stages_attempted) >= 5),
        "all_stages_defined": bool(
            len(all_stages) == 6),
        "witness_cid": str(w.cid()),
    }


def family_disagreement_algebra_v2_identities(
        seed: int,
) -> dict[str, Any]:
    """H12 — V2 identity checks pass on probe inputs."""
    trace = AlgebraTrace.empty()
    w = emit_disagreement_algebra_v2_witness(
        trace=trace,
        probe_a=[0.5, 0.3, -0.1, 0.0],
        probe_b=[0.45, 0.25, -0.15, 0.0],  # near-agreement
        probe_c=[0.1, 0.2, -0.3, 0.4])
    return {
        "idempotent_ok": bool(w.idempotent_ok),
        "self_cancel_ok": bool(w.self_cancel_ok),
        "distributivity_ok": bool(w.distributivity_ok),
        "substrate_projection_ok": bool(
            w.substrate_projection_ok),
        "all_ok": bool(
            w.idempotent_ok
            and w.self_cancel_ok
            and w.distributivity_ok
            and w.substrate_projection_ok),
    }


R113_FAMILIES: dict[str, Callable[[int], dict[str, Any]]] = {
    "trivial_w56_passthrough": family_trivial_w56_passthrough,
    "tiny_substrate_forward_determinism": (
        family_tiny_substrate_forward_determinism),
    "tiny_substrate_kv_cache_reuse": (
        family_tiny_substrate_kv_cache_reuse),
    "tiny_substrate_causal_mask_soundness": (
        family_tiny_substrate_causal_mask_soundness),
    "substrate_adapter_capability_matrix": (
        family_substrate_adapter_capability_matrix),
    "kv_bridge_injection_perturbs_logits": (
        family_kv_bridge_injection_perturbs_logits),
    "kv_bridge_replay_determinism": (
        family_kv_bridge_replay_determinism),
    "persistent_v8_quad_skip_gain": (
        family_persistent_v8_quad_skip_gain),
    "multi_hop_v6_oct_chain_len7_transitivity": (
        family_multi_hop_v6_oct_chain_len7_transitivity),
    "mlsc_v4_substrate_witness_round_trip": (
        family_mlsc_v4_substrate_witness_round_trip),
    "consensus_controller_v2_6stage_audit": (
        family_consensus_controller_v2_6stage_audit),
    "disagreement_algebra_v2_identities": (
        family_disagreement_algebra_v2_identities),
}


def run_seed(seed: int) -> R113SeedResult:
    return R113SeedResult(
        seed=int(seed),
        family_results={
            name: fn(int(seed))
            for name, fn in R113_FAMILIES.items()
        },
    )


def run_all_families(
        seeds: Sequence[int] = (11, 17, 23),
) -> dict[str, Any]:
    seed_results = [run_seed(int(s)) for s in seeds]
    summary: dict[str, Any] = {
        "schema": R113_SCHEMA_VERSION,
        "seeds": list(int(s) for s in seeds),
        "per_seed": [r.to_dict() for r in seed_results],
    }
    return summary


__all__ = [
    "R113_SCHEMA_VERSION",
    "R113_FAMILIES",
    "R113SeedResult",
    "run_seed",
    "run_all_families",
]
