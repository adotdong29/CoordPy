"""R-127 — W60 corruption / disagreement / consensus / fallback
family.

H139..H148 cell families.

* H139  CRC V8 256-bucket KV fingerprint detect rate ≥ floor
* H139b CRC V8 cache-retrieval post-replay top-K agreement is at
        least the pre-replay agreement (post-replay is the
        intervention; non-decrease is the H-bar)
* H139c CRC V8 adversarial 11-bit burst detect rate ≥ floor
* H140  consensus V6 10-stage chain enumerated
* H140b consensus V6 replay_controller_choice stage fires when
        only the replay oracle resolves the tie
* H140c consensus V6 abstains when all paths below floor
* H141  uncertainty V8 pessimistic ≤ weighted ≤ optimistic
* H141b uncertainty V8 replay_aware True when replay fidelities
        are distinct
* H142  disagreement algebra V6 replay equivalence identity
* H142b disagreement algebra V6 replay falsifier
* H143  MLSC V8 replay witness chain inheritance through merge
* H143b MLSC V8 substrate witness chain inheritance through merge
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r127_benchmark requires numpy") from exc

from .consensus_fallback_controller_v6 import (
    ConsensusFallbackControllerV6,
    W60_CONSENSUS_V6_STAGES,
    W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER,
)
from .consensus_fallback_controller_v5 import (
    W59_CONSENSUS_V5_STAGE_ABSTAIN,
)
from .corruption_robust_carrier_v8 import (
    CorruptionRobustCarrierV8,
    emit_corruption_robustness_v8_witness,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v6 import (
    emit_disagreement_algebra_v6_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import (
    MergeOperatorV8, wrap_v7_as_v8,
)
from .uncertainty_layer_v8 import (
    compose_uncertainty_report_v8,
    emit_uncertainty_v8_witness,
)


R127_SCHEMA_VERSION: str = "coordpy.r127_benchmark.v1"


def family_h139_crc_v8_kv256_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV8()
    w = emit_corruption_robustness_v8_witness(
        crc_v8=crc, n_probes=16, seed=int(seed) + 27000)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h139_crc_v8_kv256_detect",
        "passed": bool(
            float(w.kv256_corruption_detect_rate) >= 0.99),
        "kv256_detect_rate": float(
            w.kv256_corruption_detect_rate),
    }


def family_h139b_crc_v8_post_replay_agreement(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV8()
    w = emit_corruption_robustness_v8_witness(
        crc_v8=crc, n_probes=24, seed=int(seed) + 27100)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h139b_crc_v8_post_replay_agreement",
        "passed": bool(
            float(
                w.cache_retrieval_post_replay_topk_agreement_rate)
            >= float(
                w.cache_retrieval_topk_agreement_rate)),
        "pre_replay_agreement": float(
            w.cache_retrieval_topk_agreement_rate),
        "post_replay_agreement": float(
            w.cache_retrieval_post_replay_topk_agreement_rate),
    }


def family_h139c_crc_v8_adversarial_11bit_burst(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV8()
    w = emit_corruption_robustness_v8_witness(
        crc_v8=crc, n_probes=16, seed=int(seed) + 27200)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h139c_crc_v8_adversarial_11bit_burst",
        "passed": bool(
            float(w.adversarial_11bit_burst_detect_rate)
            >= 0.95),
        "detect_rate": float(
            w.adversarial_11bit_burst_detect_rate),
    }


def family_h140_consensus_v6_stages_enumerated(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h140_consensus_v6_stages_enumerated",
        "passed": bool(len(W60_CONSENSUS_V6_STAGES) == 10),
        "n_stages": int(len(W60_CONSENSUS_V6_STAGES)),
        "stages": list(W60_CONSENSUS_V6_STAGES),
    }


def family_h140b_consensus_v6_replay_controller_fires(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV6.init(
        k_required=3, cosine_floor=0.95,
        trust_threshold=0.99)

    def replay_oracle(payloads, q, decisions):
        # Pick whichever index has decision "choose_reuse".
        for i, d in enumerate(decisions):
            if str(d) == "choose_reuse":
                return i
        return 0
    ctrl.replay_controller_oracle = replay_oracle
    rng = _np.random.default_rng(int(seed) + 27300)
    q = list(rng.standard_normal(6).tolist())
    p1 = list(rng.standard_normal(6).tolist())
    p2 = list(rng.standard_normal(6).tolist())
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=None,
        parent_retrieval_scores=None,
        parent_replay_decisions=[
            "choose_recompute", "choose_reuse"],
        query_direction=q, transcript_payload=[])
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h140b_consensus_v6_replay_controller_fires",
        "passed": bool(
            res["decision_stage"]
            == W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER),
        "decision_stage": str(res["decision_stage"]),
        "selected_index": int(res["selected_index"]),
    }


def family_h140c_consensus_v6_abstains(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV6.init(
        k_required=3, cosine_floor=0.99,
        trust_threshold=0.99)
    rng = _np.random.default_rng(int(seed) + 27400)
    q = list(rng.standard_normal(6).tolist())
    p1 = list(rng.standard_normal(6).tolist())
    p2 = list(rng.standard_normal(6).tolist())
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=None,
        parent_retrieval_scores=None,
        parent_replay_decisions=None,
        query_direction=q, transcript_payload=[])
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h140c_consensus_v6_abstains",
        "passed": bool(
            res["decision_stage"]
            == W59_CONSENSUS_V5_STAGE_ABSTAIN),
        "decision_stage": str(res["decision_stage"]),
    }


def family_h141_uncertainty_v8_brackets(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 27500)
    keys = ["a", "b", "c"]
    cc = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    tw = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    sf = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    hf = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    cf = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    rf = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    rp = {k: float(rng.uniform(0.3, 0.9)) for k in keys}
    comp = compose_uncertainty_report_v8(
        component_confidences=cc, trust_weights=tw,
        substrate_fidelities=sf,
        hidden_state_fidelities=hf,
        cache_reuse_fidelities=cf,
        retrieval_fidelities=rf,
        replay_fidelities=rp)
    w = emit_uncertainty_v8_witness(composite=comp)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h141_uncertainty_v8_brackets",
        "passed": bool(
            w.pessimistic_le_weighted
            and w.weighted_le_optimistic),
        "pessimistic": float(comp.pessimistic_composite),
        "weighted": float(comp.weighted_composite),
        "optimistic": float(comp.optimistic_composite),
        "replay_aware": bool(w.replay_aware),
    }


def family_h141b_uncertainty_v8_replay_aware(
        seed: int) -> dict[str, Any]:
    """If replay_fidelities are not all 1.0, replay_aware must be
    True."""
    keys = ["a", "b"]
    comp = compose_uncertainty_report_v8(
        component_confidences={k: 0.5 for k in keys},
        trust_weights={k: 0.5 for k in keys},
        substrate_fidelities={k: 0.5 for k in keys},
        hidden_state_fidelities={k: 0.5 for k in keys},
        cache_reuse_fidelities={k: 0.5 for k in keys},
        retrieval_fidelities={k: 0.5 for k in keys},
        replay_fidelities={"a": 0.3, "b": 0.9})
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h141b_uncertainty_v8_replay_aware",
        "passed": bool(comp.replay_aware),
        "replay_aware": bool(comp.replay_aware),
    }


def family_h142_disagreement_algebra_v6_replay(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace.empty()
    rng = _np.random.default_rng(int(seed) + 27600)
    pa = list(rng.standard_normal(6).tolist())
    pb = list(rng.standard_normal(6).tolist())
    pc = list(rng.standard_normal(6).tolist())

    def replay_oracle():
        return (True, 0.0)
    w = emit_disagreement_algebra_v6_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        replay_controller_oracle=replay_oracle)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h142_disagreement_algebra_v6_replay",
        "passed": bool(w.replay_controller_equiv_ok),
        "replay_controller_equiv_ok": bool(
            w.replay_controller_equiv_ok),
        "merge_idempotent_ok": bool(w.merge_idempotent_ok),
    }


def family_h142b_disagreement_algebra_v6_falsifier(
        seed: int) -> dict[str, Any]:
    trace = AlgebraTrace.empty()
    pa = [0.1] * 6; pb = [0.2] * 6; pc = [0.3] * 6

    def bad_replay_oracle():
        return (False, 10.0)
    w = emit_disagreement_algebra_v6_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        replay_controller_oracle=bad_replay_oracle)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h142b_disagreement_algebra_v6_falsifier",
        "passed": bool(not w.replay_controller_equiv_ok),
        "replay_controller_equiv_ok": bool(
            w.replay_controller_equiv_ok),
    }


def family_h143_mlsc_v8_replay_chain_inheritance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV8(factor_dim=6)
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
    v8a = wrap_v7_as_v8(
        v7a, replay_witness_chain=("rep_a1", "rep_a2"),
        substrate_witness_chain=("sub_a1",),
        provenance_trust_table={"backend_a": 0.9})
    v8b = wrap_v7_as_v8(
        v7b, replay_witness_chain=("rep_b1",),
        substrate_witness_chain=("sub_b1",),
        provenance_trust_table={"backend_b": 0.85})
    merged = op.merge(
        [v8a, v8b],
        replay_witness_chain=("merge_rep",),
        substrate_witness_chain=("merge_sub",),
        provenance_trust_table={"backend_c": 0.8},
        retrieval_witness_chain=("merge_r",),
        controller_witness_cid="ctrl_m",
        cache_reuse_witness_cid="cache_m",
        attention_witness_chain=("attn_m",))
    chain = merged.replay_witness_chain
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h143_mlsc_v8_replay_chain_inheritance",
        "passed": bool(
            "rep_a1" in chain
            and "rep_a2" in chain
            and "rep_b1" in chain
            and "merge_rep" in chain),
        "chain": list(chain),
    }


def family_h143b_mlsc_v8_substrate_chain_inheritance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV8(factor_dim=6)
    v3a = make_root_capsule_v3(
        branch_id="a", payload=tuple([0.1] * 6),
        fact_tags=("a",), confidence=0.9, trust=0.9,
        turn_index=int(seed))
    v4a = wrap_v3_as_v4(v3a)
    v5a = wrap_v4_as_v5(v4a, attention_witness_cid="x")
    v6a = wrap_v5_as_v6(
        v5a, attention_witness_chain=("a1",),
        cache_reuse_witness_cid="ca")
    v7a = wrap_v6_as_v7(
        v6a, retrieval_witness_chain=("ra1",),
        controller_witness_cid="ctrl_a")
    v8a = wrap_v7_as_v8(
        v7a,
        substrate_witness_chain=("sub_in_1", "sub_in_2"),
        replay_witness_chain=("rep_a1",),
        provenance_trust_table={"a": 0.9})
    merged = op.merge(
        [v8a],
        substrate_witness_chain=("sub_merge",),
        replay_witness_chain=("rep_merge",))
    chain = merged.substrate_witness_chain
    table = dict(merged.provenance_trust_table)
    return {
        "schema": R127_SCHEMA_VERSION,
        "name": "h143b_mlsc_v8_substrate_chain_inheritance",
        "passed": bool(
            "sub_in_1" in chain
            and "sub_in_2" in chain
            and "sub_merge" in chain
            and table.get("a", 0.0) >= 0.9),
        "chain": list(chain),
        "trust_table": {k: float(v) for k, v in table.items()},
    }


def run_r127(seeds: Sequence[int] = (195, 295, 395)
              ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = [
        family_h139_crc_v8_kv256_detect,
        family_h139b_crc_v8_post_replay_agreement,
        family_h139c_crc_v8_adversarial_11bit_burst,
        family_h140_consensus_v6_stages_enumerated,
        family_h140b_consensus_v6_replay_controller_fires,
        family_h140c_consensus_v6_abstains,
        family_h141_uncertainty_v8_brackets,
        family_h141b_uncertainty_v8_replay_aware,
        family_h142_disagreement_algebra_v6_replay,
        family_h142b_disagreement_algebra_v6_falsifier,
        family_h143_mlsc_v8_replay_chain_inheritance,
        family_h143b_mlsc_v8_substrate_chain_inheritance,
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
    "R127_SCHEMA_VERSION",
    "family_h139_crc_v8_kv256_detect",
    "family_h139b_crc_v8_post_replay_agreement",
    "family_h139c_crc_v8_adversarial_11bit_burst",
    "family_h140_consensus_v6_stages_enumerated",
    "family_h140b_consensus_v6_replay_controller_fires",
    "family_h140c_consensus_v6_abstains",
    "family_h141_uncertainty_v8_brackets",
    "family_h141b_uncertainty_v8_replay_aware",
    "family_h142_disagreement_algebra_v6_replay",
    "family_h142b_disagreement_algebra_v6_falsifier",
    "family_h143_mlsc_v8_replay_chain_inheritance",
    "family_h143b_mlsc_v8_substrate_chain_inheritance",
    "run_r127",
]
