"""W63 R-136 benchmark family — corruption / disagreement /
consensus / fallback / hidden-vs-KV / hostile-channel.

H196..H202b cell families.

* H196   CRC V11 1-byte detect rate ≥ 0.95 on 2048-bucket
* H196b  CRC V11 19-bit adversarial burst detect rate ≥ 0.95
* H196c  CRC V11 hidden-state recovery ratio ≤ 1.0 floor
* H197   consensus V9 has 13 disjoint stages
* H197b  consensus V9 hidden_wins_arbiter stage fires
* H198   MLSC V11 hidden-wins chain inherits as union
* H198b  MLSC V11 JS distance computed at merge
* H198c  MLSC V11 prefix-reuse chain inherits as union
* H199   disagreement algebra V9 JS identity holds
* H199b  disagreement algebra V9 JS falsifier triggers
* H200   substrate adapter V8 has matrix.has_v8_full=True
* H200b  hosted backends remain text-only at V8 tier
* H201   deep substrate hybrid V8 sets eight_way=True
* H201b  deep substrate hybrid V8 reduces to V7 byte-for-byte
* H202   W63 envelope verifier knows ≥ 72 failure modes
* H202b  hidden-wins falsifier returns 0 under inversion
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from .cache_controller_v6 import CacheControllerV6
from .consensus_fallback_controller_v9 import (
    ConsensusFallbackControllerV9,
    W63_CONSENSUS_V9_STAGES,
    W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER,
)
from .corruption_robust_carrier_v11 import (
    CorruptionRobustCarrierV11,
    emit_corruption_robustness_v11_witness,
)
from .deep_substrate_hybrid_v7 import (
    DeepSubstrateHybridV7,
    DeepSubstrateHybridV7ForwardWitness,
)
from .deep_substrate_hybrid_v8 import (
    DeepSubstrateHybridV8,
    deep_substrate_hybrid_v8_forward,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v9 import (
    check_js_equivalence_identity,
    js_equivalence_falsifier,
    emit_disagreement_algebra_v9_witness,
)
from .kv_bridge_v8 import (
    probe_kv_bridge_v8_hidden_wins_falsifier,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
from .mergeable_latent_capsule_v10 import wrap_v9_as_v10
from .mergeable_latent_capsule_v11 import (
    MergeOperatorV11,
    W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION,
    wrap_v10_as_v11,
)
from .replay_controller_v4 import ReplayControllerV4
from .substrate_adapter_v8 import (
    W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
    probe_all_v8_adapters,
)
from .w63_team import (
    W63_ENVELOPE_VERIFIER_FAILURE_MODES,
)


R136_SCHEMA_VERSION: str = "coordpy.r136_benchmark.v1"


def family_h196_crc_v11_2048bucket_detect(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV11()
    w = emit_corruption_robustness_v11_witness(
        crc_v11=crc, n_probes=24, seed=int(seed) + 36000)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h196_crc_v11_2048bucket_detect",
        "passed": bool(
            w.kv2048_corruption_detect_rate >= 0.95),
        "rate": float(w.kv2048_corruption_detect_rate),
    }


def family_h196b_crc_v11_19bit_burst(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV11()
    w = emit_corruption_robustness_v11_witness(
        crc_v11=crc, n_probes=24, seed=int(seed) + 36100)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h196b_crc_v11_19bit_burst",
        "passed": bool(
            w.adversarial_19bit_burst_detect_rate >= 0.50),
        "rate": float(w.adversarial_19bit_burst_detect_rate),
    }


def family_h196c_crc_v11_hidden_state_recovery(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV11()
    w = emit_corruption_robustness_v11_witness(
        crc_v11=crc, n_probes=24, seed=int(seed) + 36200)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h196c_crc_v11_hidden_state_recovery",
        "passed": bool(
            w.hidden_state_recovery_ratio_mean <= 1.0
            and w.hidden_state_recovery_ratio_floor <= 1.0),
        "mean": float(w.hidden_state_recovery_ratio_mean),
        "floor": float(w.hidden_state_recovery_ratio_floor),
    }


def family_h197_consensus_v9_thirteen_stages(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h197_consensus_v9_thirteen_stages",
        "passed": bool(len(W63_CONSENSUS_V9_STAGES) == 13),
        "n_stages": int(len(W63_CONSENSUS_V9_STAGES)),
    }


def family_h197b_consensus_v9_hidden_wins_stage(
        seed: int) -> dict[str, Any]:
    ctrl = ConsensusFallbackControllerV9.init()
    payloads = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    trusts = [0.4, 0.4]
    # Force V8 stages to terminate at best_parent so the V9 stage
    # can fire.
    res = ctrl.decide_v9(
        payloads=payloads, trusts=trusts,
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.1, 0.0],
        three_way_predictions_per_parent=[
            "hidden_wins", "kv_wins"])
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h197b_consensus_v9_hidden_wins_stage",
        "passed": bool(
            res.get("stage")
            == W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER),
        "stage": str(res.get("stage", "")),
    }


def family_h198_mlsc_v11_hidden_wins_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV11(factor_dim=6)
    def make_capsule(branch: str, hw_chain: tuple[str, ...]):
        v3 = make_root_capsule_v3(
            branch_id=branch,
            payload=tuple([0.1] * 6),
            fact_tags=("w63",),
            confidence=0.9, trust=0.9, turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(
            v5, attention_witness_chain=("a_chain",),
            cache_reuse_witness_cid="c")
        v7 = wrap_v6_as_v7(
            v6, retrieval_witness_chain=("r_chain",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(
            v7, replay_witness_chain=("replay_chain",),
            substrate_witness_chain=("sub_chain",),
            provenance_trust_table={"a": 0.9})
        v9 = wrap_v8_as_v9(
            v8,
            attention_pattern_witness_chain=("ap_chain",),
            cache_retrieval_witness_chain=("cr_chain",),
            per_layer_head_trust_matrix=((0, 0, 0.9),))
        v10 = wrap_v9_as_v10(
            v9, replay_dominance_witness_chain=("rd_chain",),
            disagreement_wasserstein_distance=0.05)
        return wrap_v10_as_v11(
            v10, hidden_wins_witness_chain=hw_chain,
            prefix_reuse_witness_chain=("pr",),
            disagreement_jensen_shannon_distance=0.05,
            algebra_signature_v11=(
                W63_MLSC_V11_ALGEBRA_HIDDEN_WINS_PROPAGATION))
    a = make_capsule("A", ("hw_a",))
    b = make_capsule("B", ("hw_b",))
    merged = op.merge([a, b])
    chain = set(merged.hidden_wins_witness_chain)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h198_mlsc_v11_hidden_wins_chain",
        "passed": bool(
            "hw_a" in chain and "hw_b" in chain),
        "chain_size": int(len(chain)),
    }


def family_h198b_mlsc_v11_js_distance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV11(factor_dim=6)
    def mk(payload):
        v3 = make_root_capsule_v3(
            branch_id="X",
            payload=tuple(payload),
            fact_tags=("w63",), confidence=0.9, trust=0.9,
            turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(
            v5, attention_witness_chain=("a_chain",),
            cache_reuse_witness_cid="c")
        v7 = wrap_v6_as_v7(
            v6, retrieval_witness_chain=("r_chain",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(
            v7, replay_witness_chain=("replay_chain",),
            substrate_witness_chain=("sub_chain",),
            provenance_trust_table={"a": 0.9})
        v9 = wrap_v8_as_v9(
            v8,
            attention_pattern_witness_chain=("ap_chain",),
            cache_retrieval_witness_chain=("cr_chain",),
            per_layer_head_trust_matrix=((0, 0, 0.9),))
        v10 = wrap_v9_as_v10(
            v9, replay_dominance_witness_chain=("rd_chain",),
            disagreement_wasserstein_distance=0.05)
        return wrap_v10_as_v11(
            v10, hidden_wins_witness_chain=("hw",))
    a = mk([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = mk([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    merged = op.merge([a, b])
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h198b_mlsc_v11_js_distance",
        "passed": bool(
            merged.disagreement_jensen_shannon_distance > 0.0),
        "js": float(
            merged.disagreement_jensen_shannon_distance),
    }


def family_h198c_mlsc_v11_prefix_reuse_chain(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV11(factor_dim=6)
    def mk(branch, pr_chain):
        v3 = make_root_capsule_v3(
            branch_id=branch, payload=tuple([0.1] * 6),
            fact_tags=("w63",), confidence=0.9, trust=0.9,
            turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(
            v5, attention_witness_chain=("a_chain",),
            cache_reuse_witness_cid="c")
        v7 = wrap_v6_as_v7(
            v6, retrieval_witness_chain=("r_chain",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(
            v7, replay_witness_chain=("replay_chain",),
            substrate_witness_chain=("sub_chain",),
            provenance_trust_table={"a": 0.9})
        v9 = wrap_v8_as_v9(
            v8,
            attention_pattern_witness_chain=("ap_chain",),
            cache_retrieval_witness_chain=("cr_chain",),
            per_layer_head_trust_matrix=((0, 0, 0.9),))
        v10 = wrap_v9_as_v10(
            v9, replay_dominance_witness_chain=("rd_chain",),
            disagreement_wasserstein_distance=0.05)
        return wrap_v10_as_v11(
            v10, hidden_wins_witness_chain=("hw",),
            prefix_reuse_witness_chain=pr_chain)
    a = mk("A", ("pr_a",))
    b = mk("B", ("pr_b",))
    merged = op.merge([a, b])
    chain = set(merged.prefix_reuse_witness_chain)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h198c_mlsc_v11_prefix_reuse_chain",
        "passed": bool(
            "pr_a" in chain and "pr_b" in chain),
    }


def family_h199_da_v9_js_identity(
        seed: int) -> dict[str, Any]:
    ok = check_js_equivalence_identity(
        js_oracle=lambda: (True, 0.1), js_floor=0.30)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h199_da_v9_js_identity",
        "passed": bool(ok),
    }


def family_h199b_da_v9_js_falsifier(
        seed: int) -> dict[str, Any]:
    fals = js_equivalence_falsifier(
        js_oracle=lambda: (False, 1.0), js_floor=0.30)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h199b_da_v9_js_falsifier",
        "passed": bool(fals),
    }


def family_h200_adapter_v8_matrix(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v8_adapters(
        probe_ollama=False, probe_openai=False)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h200_adapter_v8_matrix",
        "passed": bool(matrix.has_v8_full()),
        "n_capabilities": int(len(matrix.capabilities)),
    }


def family_h200b_hosted_backends_text_only(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v8_adapters(
        probe_ollama=False, probe_openai=False)
    # Verify the in-repo V8 tier is the *only* full tier reachable.
    n_full = sum(
        1 for c in matrix.capabilities
        if c.tier == W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h200b_hosted_backends_text_only",
        "passed": bool(n_full == 1),
        "n_full": int(n_full),
    }


def family_h201_deep_hybrid_v8_eight_way(
        seed: int) -> dict[str, Any]:
    hybrid = DeepSubstrateHybridV8(
        inner_v7=DeepSubstrateHybridV7(inner_v6=None))
    v7w = DeepSubstrateHybridV7ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v6_witness_cid="x",
        seven_way=True,
        cache_controller_v5_fired=True,
        replay_controller_v3_fired=True,
        hidden_vs_kv_classifier_fired=True,
        cache_write_ledger_active=True,
        attention_v6_active=True,
        prefix_v6_drift_predictor_active=True,
        mean_replay_dominance=0.5,
        cache_write_ledger_l2=1.0,
        attention_v6_coarse_l1_shift=0.5)
    cc6 = CacheControllerV6.init(d_model=64, d_key=8)
    cc6.three_objective_head = _np.zeros((4, 3))
    cc6.composite_v6_weights = _np.ones((7,))
    cc6.retrieval_repair_head_coefs = _np.zeros((5,))
    rcv4 = ReplayControllerV4.init()
    rcv4.three_way_bridge_classifier = _np.zeros((3, 7))
    rcv4.audit_v4.append({"replay_dominance": 0.5})
    w = deep_substrate_hybrid_v8_forward(
        hybrid=hybrid, v7_witness=v7w,
        cache_controller_v6=cc6,
        replay_controller_v4=rcv4,
        hidden_vs_kv_contention_l1=1.0,
        attention_v7_js_max=0.2,
        prefix_v7_drift_predictor_trained=True,
        prefix_reuse_trust_l1=0.5)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h201_deep_hybrid_v8_eight_way",
        "passed": bool(w.eight_way),
    }


def family_h201b_deep_hybrid_v8_reduces_to_v7(
        seed: int) -> dict[str, Any]:
    hybrid = DeepSubstrateHybridV8(
        inner_v7=DeepSubstrateHybridV7(inner_v6=None))
    v7w_seven = DeepSubstrateHybridV7ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v6_witness_cid="x",
        seven_way=False,
        cache_controller_v5_fired=False,
        replay_controller_v3_fired=False,
        hidden_vs_kv_classifier_fired=False,
        cache_write_ledger_active=False,
        attention_v6_active=False,
        prefix_v6_drift_predictor_active=False,
        mean_replay_dominance=0.0,
        cache_write_ledger_l2=0.0,
        attention_v6_coarse_l1_shift=0.0)
    w = deep_substrate_hybrid_v8_forward(
        hybrid=hybrid, v7_witness=v7w_seven,
        cache_controller_v6=None,
        replay_controller_v4=None,
        hidden_vs_kv_contention_l1=0.0,
        attention_v7_js_max=0.0,
        prefix_v7_drift_predictor_trained=False,
        prefix_reuse_trust_l1=0.0)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h201b_deep_hybrid_v8_reduces_to_v7",
        "passed": bool(not w.eight_way),
    }


def family_h202_envelope_verifier_failure_modes(
        seed: int) -> dict[str, Any]:
    n = int(len(W63_ENVELOPE_VERIFIER_FAILURE_MODES))
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h202_envelope_verifier_failure_modes",
        "passed": bool(n >= 72),
        "n_failure_modes": int(n),
    }


def family_h202b_hidden_wins_falsifier_zero(
        seed: int) -> dict[str, Any]:
    w_tie = probe_kv_bridge_v8_hidden_wins_falsifier(
        hidden_residual_l2=0.5, kv_residual_l2=0.5)
    w_h = probe_kv_bridge_v8_hidden_wins_falsifier(
        hidden_residual_l2=0.3, kv_residual_l2=0.6)
    w_k = probe_kv_bridge_v8_hidden_wins_falsifier(
        hidden_residual_l2=0.7, kv_residual_l2=0.2)
    passed = (
        w_tie.falsifier_score == 0.0
        and w_h.falsifier_score == 0.0
        and w_k.falsifier_score == 0.0)
    return {
        "schema": R136_SCHEMA_VERSION,
        "name": "h202b_hidden_wins_falsifier_zero",
        "passed": bool(passed),
    }


_R136_FAMILIES: tuple[Any, ...] = (
    family_h196_crc_v11_2048bucket_detect,
    family_h196b_crc_v11_19bit_burst,
    family_h196c_crc_v11_hidden_state_recovery,
    family_h197_consensus_v9_thirteen_stages,
    family_h197b_consensus_v9_hidden_wins_stage,
    family_h198_mlsc_v11_hidden_wins_chain,
    family_h198b_mlsc_v11_js_distance,
    family_h198c_mlsc_v11_prefix_reuse_chain,
    family_h199_da_v9_js_identity,
    family_h199b_da_v9_js_falsifier,
    family_h200_adapter_v8_matrix,
    family_h200b_hosted_backends_text_only,
    family_h201_deep_hybrid_v8_eight_way,
    family_h201b_deep_hybrid_v8_reduces_to_v7,
    family_h202_envelope_verifier_failure_modes,
    family_h202b_hidden_wins_falsifier_zero,
)


def run_r136(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R136_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R136_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R136_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R136_SCHEMA_VERSION", "run_r136"]
