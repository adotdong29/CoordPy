"""R-124 — W59 corruption / disagreement / consensus / fallback
family.

H120..H128 cell families. R-124 is the W59 *hostile-channel /
disagreement* family:

* H120  CRC V7 128-bucket KV fingerprint detect rate ≥ floor
* H120b CRC V7 cache-retrieval top-K agreement under non-target
        corruption ≥ floor
* H120c CRC V7 adversarial 9-bit burst detect rate ≥ floor
* H121  consensus V5 9-stage chain enumerated
* H121b consensus V5 retrieval_replay stage fires when only the
        retrieval oracle resolves the tie
* H121c consensus V5 abstains when all paths below floor
* H122  uncertainty V7 pessimistic ≤ weighted ≤ optimistic
* H122b uncertainty V7 retrieval_aware True when retrieval
        fidelities are distinct
* H123  disagreement algebra V5 retrieval equivalence identity
* H123b disagreement algebra V5 still validates V4 cache identity
* H124  MLSC V7 retrieval witness chain inheritance through merge
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r124_benchmark requires numpy") from exc

from .consensus_fallback_controller_v5 import (
    ConsensusFallbackControllerV5,
    W59_CONSENSUS_V5_STAGES,
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
    W59_CONSENSUS_V5_STAGE_RETRIEVAL,
)
from .corruption_robust_carrier_v7 import (
    CorruptionRobustCarrierV7,
    emit_corruption_robustness_v7_witness,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v5 import (
    emit_disagreement_algebra_v5_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import (
    MergeOperatorV7,
    emit_mlsc_v7_witness,
    wrap_v6_as_v7,
)
from .uncertainty_layer_v7 import (
    compose_uncertainty_report_v7,
    emit_uncertainty_v7_witness,
)


R124_SCHEMA_VERSION: str = "coordpy.r124_benchmark.v1"


def family_h120_crc_v7_kv128_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV7()
    w = emit_corruption_robustness_v7_witness(
        crc_v7=crc, n_probes=16, seed=int(seed) + 24000)
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h120_crc_v7_kv128_detect",
        "passed": bool(
            float(w.kv128_corruption_detect_rate) >= 0.99),
        "kv128_detect_rate": float(
            w.kv128_corruption_detect_rate),
    }


def family_h120b_crc_v7_retrieval_topk_agreement(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV7()
    w = emit_corruption_robustness_v7_witness(
        crc_v7=crc, n_probes=24, seed=int(seed) + 24100)
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h120b_crc_v7_retrieval_topk_agreement",
        "passed": bool(
            float(w.cache_retrieval_topk_agreement_rate)
            >= 0.7),
        "agreement_rate": float(
            w.cache_retrieval_topk_agreement_rate),
        "jaccard_mean": float(
            w.cache_retrieval_jaccard_mean),
    }


def family_h120c_crc_v7_adversarial_9bit_burst(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV7()
    w = emit_corruption_robustness_v7_witness(
        crc_v7=crc, n_probes=16, seed=int(seed) + 24200)
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h120c_crc_v7_adversarial_9bit_burst",
        "passed": bool(
            float(w.adversarial_9bit_burst_detect_rate)
            >= 0.95),
        "detect_rate": float(
            w.adversarial_9bit_burst_detect_rate),
    }


def family_h121_consensus_v5_stages_enumerated(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h121_consensus_v5_stages_enumerated",
        "passed": bool(len(W59_CONSENSUS_V5_STAGES) == 9),
        "n_stages": int(len(W59_CONSENSUS_V5_STAGES)),
        "stages": list(W59_CONSENSUS_V5_STAGES),
    }


def family_h121b_consensus_v5_retrieval_replay_fires(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV5(
        k_required=3, cosine_floor=0.95,
        trust_threshold=0.99)

    def retrieval_oracle(payloads, q, scores):
        return int(_np.argmax(scores))
    ctrl.retrieval_replay_oracle = retrieval_oracle
    # Two parents that AGREE only on retrieval scores; their
    # payloads point away from the query.
    rng = _np.random.default_rng(int(seed) + 24300)
    q = list(rng.standard_normal(6).tolist())
    p1 = list(rng.standard_normal(6).tolist())
    p2 = list(rng.standard_normal(6).tolist())
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=[(1, 2), (3, 4)],
        parent_retrieval_scores=[0.4, 0.9],
        query_direction=q,
        transcript_payload=[])
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h121b_consensus_v5_retrieval_replay_fires",
        "passed": bool(
            res["decision_stage"]
            == W59_CONSENSUS_V5_STAGE_RETRIEVAL),
        "decision_stage": str(res["decision_stage"]),
        "selected_index": int(res["selected_index"]),
    }


def family_h121c_consensus_v5_abstains(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV5(
        k_required=3, cosine_floor=0.99,
        trust_threshold=0.99)
    rng = _np.random.default_rng(int(seed) + 24400)
    q = list(rng.standard_normal(6).tolist())
    p1 = list(rng.standard_normal(6).tolist())
    p2 = list(rng.standard_normal(6).tolist())
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=None,
        parent_retrieval_scores=None,
        query_direction=q,
        transcript_payload=[])
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h121c_consensus_v5_abstains",
        "passed": bool(
            res["decision_stage"]
            == W59_CONSENSUS_V5_STAGE_ABSTAIN),
        "decision_stage": str(res["decision_stage"]),
    }


def family_h122_uncertainty_v7_brackets(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 24500)
    keys = ["a", "b", "c"]
    component_confidences = {k: float(rng.uniform(0.3, 0.9))
                               for k in keys}
    trust_weights = {k: float(rng.uniform(0.3, 0.9))
                       for k in keys}
    substrate_fidelities = {k: float(rng.uniform(0.3, 0.9))
                              for k in keys}
    hidden_state_fidelities = {k: float(rng.uniform(0.3, 0.9))
                                 for k in keys}
    cache_reuse_fidelities = {k: float(rng.uniform(0.3, 0.9))
                                for k in keys}
    retrieval_fidelities = {k: float(rng.uniform(0.3, 0.9))
                              for k in keys}
    comp = compose_uncertainty_report_v7(
        component_confidences=component_confidences,
        trust_weights=trust_weights,
        substrate_fidelities=substrate_fidelities,
        hidden_state_fidelities=hidden_state_fidelities,
        cache_reuse_fidelities=cache_reuse_fidelities,
        retrieval_fidelities=retrieval_fidelities)
    w = emit_uncertainty_v7_witness(composite=comp)
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h122_uncertainty_v7_brackets",
        "passed": bool(
            w.pessimistic_le_weighted
            and w.weighted_le_optimistic),
        "pessimistic": float(comp.pessimistic_composite),
        "weighted": float(comp.weighted_composite),
        "optimistic": float(comp.optimistic_composite),
        "retrieval_aware": bool(w.retrieval_aware),
    }


def family_h122b_uncertainty_v7_retrieval_aware(
        seed: int) -> dict[str, Any]:
    """If retrieval_fidelities are not all 1.0, retrieval_aware
    must be True (the sixth axis becomes informative)."""
    keys = ["a", "b"]
    comp = compose_uncertainty_report_v7(
        component_confidences={k: 0.5 for k in keys},
        trust_weights={k: 0.5 for k in keys},
        substrate_fidelities={k: 0.5 for k in keys},
        hidden_state_fidelities={k: 0.5 for k in keys},
        cache_reuse_fidelities={k: 0.5 for k in keys},
        retrieval_fidelities={"a": 0.3, "b": 0.9})
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h122b_uncertainty_v7_retrieval_aware",
        "passed": bool(comp.retrieval_aware),
        "retrieval_aware": bool(comp.retrieval_aware),
    }


def family_h123_disagreement_algebra_v5_retrieval(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace.empty()
    rng = _np.random.default_rng(int(seed) + 24600)
    pa = list(rng.standard_normal(6).tolist())
    pb = list(rng.standard_normal(6).tolist())
    pc = list(rng.standard_normal(6).tolist())

    def retrieval_oracle():
        return (True, 0.0)
    w = emit_disagreement_algebra_v5_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        retrieval_replay_oracle=retrieval_oracle)
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h123_disagreement_algebra_v5_retrieval",
        "passed": bool(w.retrieval_equiv_ok),
        "retrieval_equiv_ok": bool(w.retrieval_equiv_ok),
        "merge_idempotent_ok": bool(w.merge_idempotent_ok),
        "diff_self_cancel_ok": bool(w.diff_self_cancel_ok),
        "intersect_distrib_ok": bool(w.intersect_distrib_ok),
    }


def family_h123b_disagreement_algebra_v5_falsifier(
        seed: int) -> dict[str, Any]:
    """When the retrieval oracle reports a non-equivalence, the
    identity must fail. This is the falsifier path."""
    trace = AlgebraTrace.empty()
    pa = [0.1] * 6
    pb = [0.2] * 6
    pc = [0.3] * 6

    def bad_retrieval_oracle():
        # Argmax flipped — identity should fail.
        return (False, 10.0)
    w = emit_disagreement_algebra_v5_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        retrieval_replay_oracle=bad_retrieval_oracle)
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h123b_disagreement_algebra_v5_falsifier",
        "passed": bool(not w.retrieval_equiv_ok),
        "retrieval_equiv_ok": bool(w.retrieval_equiv_ok),
    }


def family_h124_mlsc_v7_retrieval_chain_inheritance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV7(factor_dim=6)
    # Two parents with distinct retrieval witness chains.
    v3a = make_root_capsule_v3(
        branch_id="a", payload=tuple([0.1] * 6),
        fact_tags=("a",), confidence=0.9, trust=0.9,
        turn_index=int(seed))
    v3b = make_root_capsule_v3(
        branch_id="b", payload=tuple([0.2] * 6),
        fact_tags=("b",), confidence=0.85, trust=0.85,
        turn_index=int(seed))
    v4a = wrap_v3_as_v4(v3a)
    v4b = wrap_v3_as_v4(v3b)
    v5a = wrap_v4_as_v5(v4a, attention_witness_cid="aa")
    v5b = wrap_v4_as_v5(v4b, attention_witness_cid="ab")
    v6a = wrap_v5_as_v6(
        v5a, attention_witness_chain=("a1",),
        cache_reuse_witness_cid="ca")
    v6b = wrap_v5_as_v6(
        v5b, attention_witness_chain=("b1",),
        cache_reuse_witness_cid="cb")
    v7a = wrap_v6_as_v7(
        v6a, retrieval_witness_chain=("ra1", "ra2"),
        controller_witness_cid="ctrl_a")
    v7b = wrap_v6_as_v7(
        v6b, retrieval_witness_chain=("rb1",),
        controller_witness_cid="ctrl_b")
    merged = op.merge(
        [v7a, v7b],
        retrieval_witness_chain=("merge_r",),
        controller_witness_cid="ctrl_m",
        cache_reuse_witness_cid="cache_m",
        attention_witness_chain=("attn_m",))
    chain = merged.retrieval_witness_chain
    return {
        "schema": R124_SCHEMA_VERSION,
        "name": "h124_mlsc_v7_retrieval_chain_inheritance",
        "passed": bool(
            "ra1" in chain
            and "ra2" in chain
            and "rb1" in chain
            and "merge_r" in chain),
        "chain": list(chain),
        "controller_witness_cid": str(
            merged.controller_witness_cid),
    }


def run_r124(seeds: Sequence[int] = (192, 292, 392)
              ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = [
        family_h120_crc_v7_kv128_detect,
        family_h120b_crc_v7_retrieval_topk_agreement,
        family_h120c_crc_v7_adversarial_9bit_burst,
        family_h121_consensus_v5_stages_enumerated,
        family_h121b_consensus_v5_retrieval_replay_fires,
        family_h121c_consensus_v5_abstains,
        family_h122_uncertainty_v7_brackets,
        family_h122b_uncertainty_v7_retrieval_aware,
        family_h123_disagreement_algebra_v5_retrieval,
        family_h123b_disagreement_algebra_v5_falsifier,
        family_h124_mlsc_v7_retrieval_chain_inheritance,
    ]
    for s in seeds:
        per_family: dict[str, dict[str, Any]] = {}
        for fam in families:
            res = fam(int(s))
            per_family[res["name"]] = res
        out.append({"seed": int(s),
                     "family_results": per_family})
    return out


__all__ = [
    "R124_SCHEMA_VERSION",
    "family_h120_crc_v7_kv128_detect",
    "family_h120b_crc_v7_retrieval_topk_agreement",
    "family_h120c_crc_v7_adversarial_9bit_burst",
    "family_h121_consensus_v5_stages_enumerated",
    "family_h121b_consensus_v5_retrieval_replay_fires",
    "family_h121c_consensus_v5_abstains",
    "family_h122_uncertainty_v7_brackets",
    "family_h122b_uncertainty_v7_retrieval_aware",
    "family_h123_disagreement_algebra_v5_retrieval",
    "family_h123b_disagreement_algebra_v5_falsifier",
    "family_h124_mlsc_v7_retrieval_chain_inheritance",
    "run_r124",
]
