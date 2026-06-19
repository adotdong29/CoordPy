"""R-121 — W58 corruption / disagreement / consensus / fallback
family.

Families:

* H92  CRC V6 64-bucket KV fingerprint detect rate ≥ 0.95
* H92b CRC V6 prefix-state corruption detect rate ≥ 0.95
* H92c CRC V6 adversarial 7-bit burst detect rate ≥ 0.95
* H93  TVS V7 pick rates sum to 1
* H93b TVS V7 cache_reuse_replay arm dominates when cache fid is strict highest
* H94  Consensus V4 8-stage chain
* H94b Consensus V4 cache_reuse_replay stage fires when other stages fail
* H103 Uncertainty V6 cache-aware composite differs from V5
* H103b Uncertainty V6 pessimistic ≤ weighted ≤ optimistic
* H104 Disagreement Algebra V4 cache-reuse equivalence identity OK
* H105 Attention-steering V2 KL budget enforced
* H106 Multi-hop V8 3-axis trust composite filters down-trust paths
"""

from __future__ import annotations

import dataclasses
import json
import random
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r121_benchmark requires numpy") from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
    steer_attention_and_measure_v2,
)
from .consensus_fallback_controller_v4 import (
    ConsensusFallbackControllerV4,
    W58_CONSENSUS_V4_STAGES,
    W58_CONSENSUS_V4_STAGE_CACHE_REUSE,
)
from .corruption_robust_carrier_v6 import (
    CorruptionRobustCarrierV6,
    emit_corruption_robustness_v6_witness,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v4 import (
    emit_disagreement_algebra_v4_witness,
)
from .multi_hop_translator_v8 import (
    DecBackendChainPathV8,
    substrate_hidden_attention_trust_arbitration,
)
from .tiny_substrate_v3 import (
    build_default_tiny_substrate_v3, tokenize_bytes_v3)
from .transcript_vs_shared_arbiter_v7 import (
    W58_TVS_V7_ARMS, eight_arm_compare,
    emit_tvs_arbiter_v7_witness,
)
from .uncertainty_layer_v6 import (
    compose_uncertainty_report_v6,
    emit_uncertainty_v6_witness,
)


R121_SCHEMA_VERSION: str = "coordpy.r121_benchmark.v1"


@dataclasses.dataclass(frozen=True)
class R121SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R121_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


def family_h92_crc_v6_kv64_detect(seed: int) -> dict[str, Any]:
    crc6 = CorruptionRobustCarrierV6()
    w = emit_corruption_robustness_v6_witness(
        crc_v6=crc6, n_probes=32, seed=int(seed) + 21000)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h92_crc_v6_kv64_detect",
        "passed": bool(
            float(w.kv64_corruption_detect_rate) >= 0.95),
        "rate": float(w.kv64_corruption_detect_rate),
    }


def family_h92b_crc_v6_prefix_state_detect(
        seed: int) -> dict[str, Any]:
    crc6 = CorruptionRobustCarrierV6()
    w = emit_corruption_robustness_v6_witness(
        crc_v6=crc6, n_probes=16, seed=int(seed) + 21100)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h92b_crc_v6_prefix_state_detect",
        "passed": bool(
            float(w.prefix_state_corruption_detect_rate)
            >= 0.95),
        "rate": float(w.prefix_state_corruption_detect_rate),
    }


def family_h92c_crc_v6_adversarial_burst_detect(
        seed: int) -> dict[str, Any]:
    crc6 = CorruptionRobustCarrierV6()
    w = emit_corruption_robustness_v6_witness(
        crc_v6=crc6, n_probes=32, seed=int(seed) + 21200)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h92c_crc_v6_adversarial_burst_detect",
        "passed": bool(
            float(w.adversarial_7bit_burst_detect_rate)
            >= 0.95),
        "rate": float(w.adversarial_7bit_burst_detect_rate),
    }


def family_h93_tvs_v7_pick_rates_sum(
        seed: int) -> dict[str, Any]:
    cmp = eight_arm_compare(
        per_turn_confidences=[0.7, 0.3, 0.8, 0.1],
        per_turn_trust_scores=[0.6, 0.4, 0.5, 0.5],
        per_turn_merge_retentions=[0.5, 0.6, 0.5, 0.4],
        per_turn_tw_retentions=[0.5, 0.5, 0.5, 0.5],
        per_turn_substrate_fidelities=[0.4, 0.7, 0.3, 0.2],
        per_turn_hidden_fidelities=[0.3, 0.4, 0.9, 0.2],
        per_turn_cache_fidelities=[0.9, 0.4, 0.3, 0.1])
    s = float(sum(cmp.pick_rates.values()))
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h93_tvs_v7_pick_rates_sum",
        "passed": bool(abs(s - 1.0) < 1e-9 or s == 0.0),
        "sum": float(s),
        "n_arms": int(len(W58_TVS_V7_ARMS)),
    }


def family_h93b_tvs_v7_cache_reuse_dominates(
        seed: int) -> dict[str, Any]:
    """When cache_fidelity is strictly highest, cache_reuse_replay
    is picked at rate 1.0."""
    cmp = eight_arm_compare(
        per_turn_confidences=[0.5, 0.5, 0.5],
        per_turn_trust_scores=[0.5, 0.5, 0.5],
        per_turn_merge_retentions=[0.5, 0.5, 0.5],
        per_turn_tw_retentions=[0.5, 0.5, 0.5],
        per_turn_substrate_fidelities=[0.4, 0.3, 0.4],
        per_turn_hidden_fidelities=[0.3, 0.4, 0.3],
        per_turn_cache_fidelities=[0.9, 0.9, 0.9])
    rate = float(cmp.pick_rates.get("cache_reuse_replay", 0.0))
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h93b_tvs_v7_cache_reuse_dominates",
        "passed": bool(rate == 1.0),
        "cache_reuse_rate": float(rate),
    }


def family_h94_consensus_v4_8_stage(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h94_consensus_v4_8_stage",
        "passed": bool(len(W58_CONSENSUS_V4_STAGES) == 8),
        "n_stages": int(len(W58_CONSENSUS_V4_STAGES)),
    }


def family_h94b_consensus_v4_cache_reuse_fires(
        seed: int) -> dict[str, Any]:
    """When K-of-N + trust + substrate + logit-lens all fail,
    and a cache_reuse_oracle is wired, V4 picks the cache_reuse
    stage."""
    ctrl = ConsensusFallbackControllerV4(
        k_required=2, cosine_floor=0.99,
        trust_threshold=10.0)
    rng = random.Random(int(seed) + 21300)
    p1 = [rng.gauss(0.0, 1.0) for _ in range(6)]
    p2 = [rng.gauss(0.0, 1.0) for _ in range(6)]
    q = [rng.gauss(0.0, 1.0) for _ in range(6)]
    ctrl.substrate_oracle = None
    ctrl.logit_lens_oracle = None
    ctrl.cache_reuse_oracle = (
        lambda payloads, qd, fps: 1)
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=[(1, 2), (3, 4)],
        query_direction=q,
        transcript_payload=[0.0] * 6)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h94b_consensus_v4_cache_reuse_fires",
        "passed": bool(
            res["decision_stage"]
            == W58_CONSENSUS_V4_STAGE_CACHE_REUSE),
        "decision_stage": str(res["decision_stage"]),
    }


def family_h103_uncertainty_v6_cache_aware(
        seed: int) -> dict[str, Any]:
    """A composite with differing cache_reuse_fidelities is
    flagged ``cache_aware=True``; one with all-1.0 is False."""
    c_aware = compose_uncertainty_report_v6(
        component_confidences={"a": 0.8, "b": 0.6},
        trust_weights={"a": 0.9, "b": 0.8},
        substrate_fidelities={"a": 0.8, "b": 0.7},
        hidden_state_fidelities={"a": 0.8, "b": 0.7},
        cache_reuse_fidelities={"a": 0.9, "b": 0.3})
    c_flat = compose_uncertainty_report_v6(
        component_confidences={"a": 0.8, "b": 0.6},
        trust_weights={"a": 0.9, "b": 0.8},
        substrate_fidelities={"a": 0.8, "b": 0.7},
        hidden_state_fidelities={"a": 0.8, "b": 0.7},
        cache_reuse_fidelities={"a": 1.0, "b": 1.0})
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h103_uncertainty_v6_cache_aware",
        "passed": bool(
            c_aware.cache_aware and not c_flat.cache_aware),
        "aware": bool(c_aware.cache_aware),
        "flat": bool(c_flat.cache_aware),
    }


def family_h103b_uncertainty_v6_bracket(
        seed: int) -> dict[str, Any]:
    """For any non-empty composite, pessimistic ≤ weighted ≤
    optimistic."""
    c = compose_uncertainty_report_v6(
        component_confidences={"a": 0.7, "b": 0.5, "c": 0.3},
        trust_weights={"a": 0.9, "b": 0.7, "c": 0.5},
        substrate_fidelities={"a": 0.8, "b": 0.6, "c": 0.4},
        hidden_state_fidelities={"a": 0.7, "b": 0.5, "c": 0.4},
        cache_reuse_fidelities={"a": 0.9, "b": 0.6, "c": 0.3},
        adversarial_radius=0.1)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h103b_uncertainty_v6_bracket",
        "passed": bool(
            c.pessimistic_composite
            <= c.weighted_composite + 1e-9
            and c.weighted_composite
            <= c.optimistic_composite + 1e-9),
        "pessimistic": float(c.pessimistic_composite),
        "weighted": float(c.weighted_composite),
        "optimistic": float(c.optimistic_composite),
    }


def family_h104_disagreement_algebra_v4_identity(
        seed: int) -> dict[str, Any]:
    """V4 identity with a perfect cache-reuse oracle (matches=True)
    holds."""
    rng = random.Random(int(seed) + 21400)
    trace = AlgebraTrace.empty()
    pa = [rng.gauss(0.0, 1.0) for _ in range(6)]
    pb = [rng.gauss(0.0, 1.0) for _ in range(6)]
    pc = [rng.gauss(0.0, 1.0) for _ in range(6)]

    def cache_reuse_oracle():
        return (0.0, True)

    w = emit_disagreement_algebra_v4_witness(
        trace=trace,
        probe_a=pa, probe_b=pb, probe_c=pc,
        cache_reuse_oracle=cache_reuse_oracle)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h104_disagreement_algebra_v4_identity",
        "passed": bool(w.cache_reuse_equiv_ok),
        "ok": bool(w.cache_reuse_equiv_ok),
    }


def family_h105_attn_v2_kl_budget_enforced(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v3(seed=int(seed) + 21500)
    ids = tokenize_bytes_v3("attn-budget", max_len=10)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=16,
        seed=int(seed) + 21501)
    rng = _np.random.default_rng(int(seed) + 21502)
    carrier = list(rng.standard_normal(16).tolist())
    w = steer_attention_and_measure_v2(
        params=p, carrier=carrier, projection=proj,
        token_ids=ids, kl_budget=1.0)
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h105_attn_v2_kl_budget_enforced",
        "passed": bool(w.kl_budget_enforced),
        "max_kl": float(max(w.mean_kl_per_layer)),
        "budget": float(w.kl_budget),
        "final_clip": float(w.final_clip),
    }


def family_h106_multi_hop_v8_filters_down_trust(
        seed: int) -> dict[str, Any]:
    """A path with attention_trust=0 (zero composite) is filtered
    out under trust_floor=0.1."""
    rng = random.Random(int(seed) + 21600)
    paths = []
    for i in range(4):
        paths.append(DecBackendChainPathV8(
            chain=("A", "B"),
            payload=tuple(
                rng.uniform(-1, 1) for _ in range(6)),
            confidence=0.7,
            substrate_trust=0.8,
            hidden_trust=0.8,
            attention_trust=(0.0 if i == 0 else 0.6),
        ))
    out, info = (
        substrate_hidden_attention_trust_arbitration(
            paths=paths, trust_floor=0.1))
    # First path should be excluded.
    return {
        "schema": R121_SCHEMA_VERSION,
        "name": "h106_multi_hop_v8_filters_down_trust",
        "passed": bool(
            info["kind"] != "abstain"
            and info.get("n_survivors", 0) == 3),
        "info": dict(info),
    }


R121_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("h92_crc_v6_kv64_detect",
     family_h92_crc_v6_kv64_detect),
    ("h92b_crc_v6_prefix_state_detect",
     family_h92b_crc_v6_prefix_state_detect),
    ("h92c_crc_v6_adversarial_burst_detect",
     family_h92c_crc_v6_adversarial_burst_detect),
    ("h93_tvs_v7_pick_rates_sum",
     family_h93_tvs_v7_pick_rates_sum),
    ("h93b_tvs_v7_cache_reuse_dominates",
     family_h93b_tvs_v7_cache_reuse_dominates),
    ("h94_consensus_v4_8_stage",
     family_h94_consensus_v4_8_stage),
    ("h94b_consensus_v4_cache_reuse_fires",
     family_h94b_consensus_v4_cache_reuse_fires),
    ("h103_uncertainty_v6_cache_aware",
     family_h103_uncertainty_v6_cache_aware),
    ("h103b_uncertainty_v6_bracket",
     family_h103b_uncertainty_v6_bracket),
    ("h104_disagreement_algebra_v4_identity",
     family_h104_disagreement_algebra_v4_identity),
    ("h105_attn_v2_kl_budget_enforced",
     family_h105_attn_v2_kl_budget_enforced),
    ("h106_multi_hop_v8_filters_down_trust",
     family_h106_multi_hop_v8_filters_down_trust),
)


def run_r121(*, seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    rows: list[R121SeedResult] = []
    for s in seeds:
        results: dict[str, dict[str, Any]] = {}
        for name, fn in R121_FAMILIES:
            results[name] = fn(int(s))
        rows.append(R121SeedResult(
            seed=int(s), family_results=results))
    summary = {
        "schema": R121_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "seeds": [r.to_dict() for r in rows],
    }
    pass_counts: dict[str, int] = {}
    for r in rows:
        for k, v in r.family_results.items():
            if bool(v.get("passed", False)):
                pass_counts[k] = pass_counts.get(k, 0) + 1
    summary["pass_counts"] = pass_counts
    summary["all_passed"] = bool(all(
        pass_counts.get(name, 0) == len(seeds)
        for name, _ in R121_FAMILIES))
    return summary


__all__ = [
    "R121_SCHEMA_VERSION",
    "R121_FAMILIES",
    "R121SeedResult",
    "run_r121",
]
