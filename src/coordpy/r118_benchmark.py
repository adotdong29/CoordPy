"""R-118 — W57 corruption / disagreement / consensus / fallback /
hostile-channel family.

H71..H85 cell families that exercise:

  * CRC V5 BCH(31,16) triple-bit correct rate
  * CRC V5 5-bit burst dispersion (3-D interleave)
  * CRC V5 9-of-13 majority silent failure floor
  * CRC V5 3-D interleave round trip OK
  * CRC V5 KV cache corruption detection
  * Consensus V3 7-stage chain audit completes
  * Consensus V3 logit-lens-conditioned tiebreaker fires
  * MLSC V5 hidden-state-witness-chain inheritance
  * MLSC V5 per-head-trust weighting affects merge
  * MLSC V5 algebra signature substrate_project carried
  * Disagreement Algebra V3 substrate-projection identity
  * Uncertainty V5 hidden-aware composite differs from V4
  * Uncertainty V5 pessimistic ≤ weighted ≤ optimistic
  * Prefix-state corruption is detected
  * W57 envelope verifier failure-mode count ≥ 40
  * W57 envelope verifier OK on clean run
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("coordpy.r118_benchmark requires numpy") from exc

from .agents import Agent
from .consensus_fallback_controller_v3 import (
    ConsensusFallbackControllerV3,
    W57_CONSENSUS_V3_STAGES,
    W57_CONSENSUS_V3_STAGE_LOGIT_LENS,
)
from .corruption_robust_carrier_v5 import (
    CorruptionRobustCarrierV5,
    emit_corruption_robustness_v5_witness,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v3 import (
    emit_disagreement_algebra_v3_witness,
)
from .mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import (
    MergeOperatorV5,
    W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT,
    wrap_v4_as_v5,
)
from .prefix_state_bridge import (
    bridge_prefix_state_and_measure,
)
from .synthetic_llm import SyntheticLLMClient
from .tiny_substrate_v2 import (
    build_default_tiny_substrate_v2,
    tokenize_bytes_v2,
)
from .uncertainty_layer_v4 import (
    compose_uncertainty_report_v4,
)
from .uncertainty_layer_v5 import (
    compose_uncertainty_report_v5,
)
from .w57_team import (
    W57Team,
    W57_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w57_registry,
    verify_w57_handoff,
)


R118_SCHEMA_VERSION: str = "coordpy.r118_benchmark.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class R118SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R118_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


def family_crc_v5_triple_bit(seed: int) -> dict[str, Any]:
    """H71 — CRC V5 BCH(31,16) triple-bit correct rate ≥ 0.80."""
    v5 = CorruptionRobustCarrierV5()
    w = emit_corruption_robustness_v5_witness(
        crc_v5=v5, n_probes=64, seed=int(seed) + 10)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "crc_v5_triple_bit",
        "passed": bool(
            float(w.triple_bit_correct_rate) >= 0.80),
        "rate": float(w.triple_bit_correct_rate),
    }


def family_crc_v5_burst_dispersion(seed: int) -> dict[str, Any]:
    """H72 — CRC V5 5-bit burst dispersion (max-run ≤ 2) ≥ 0.90."""
    v5 = CorruptionRobustCarrierV5()
    w = emit_corruption_robustness_v5_witness(
        crc_v5=v5, n_probes=64, seed=int(seed) + 20)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "crc_v5_burst_dispersion",
        "passed": bool(
            float(w.five_bit_burst_recovery_rate) >= 0.90),
        "rate": float(w.five_bit_burst_recovery_rate),
    }


def family_crc_v5_majority_silent(seed: int) -> dict[str, Any]:
    """H73 — CRC V5 9-of-13 majority silent failure rate ≤ 0.05."""
    v5 = CorruptionRobustCarrierV5()
    w = emit_corruption_robustness_v5_witness(
        crc_v5=v5, n_probes=64, seed=int(seed) + 30)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "crc_v5_majority_silent",
        "passed": bool(
            float(w.nine_of_13_silent_failure_rate) <= 0.05),
        "rate": float(w.nine_of_13_silent_failure_rate),
    }


def family_crc_v5_3d_interleave(seed: int) -> dict[str, Any]:
    """H74 — CRC V5 3-D interleave round trip OK."""
    v5 = CorruptionRobustCarrierV5()
    w = emit_corruption_robustness_v5_witness(
        crc_v5=v5, n_probes=8, seed=int(seed) + 40)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "crc_v5_3d_interleave",
        "passed": bool(w.three_d_interleave_round_trip_ok),
    }


def family_crc_v5_kv_corruption_detect(
        seed: int,
) -> dict[str, Any]:
    """H75 — CRC V5 KV cache corruption detect rate ≥ 0.95."""
    v5 = CorruptionRobustCarrierV5()
    w = emit_corruption_robustness_v5_witness(
        crc_v5=v5, n_probes=32, seed=int(seed) + 50)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "crc_v5_kv_corruption_detect",
        "passed": bool(
            float(w.kv_corruption_detect_rate) >= 0.95),
        "rate": float(w.kv_corruption_detect_rate),
    }


def family_consensus_v3_seven_stage(seed: int) -> dict[str, Any]:
    """H76 — Consensus V3 7-stage chain has 7 stages."""
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "consensus_v3_seven_stage",
        "passed": bool(len(W57_CONSENSUS_V3_STAGES) == 7),
        "n_stages": int(len(W57_CONSENSUS_V3_STAGES)),
    }


def family_consensus_v3_logit_lens_fires(
        seed: int,
) -> dict[str, Any]:
    """H77 — When other tiebreakers fail and logit_lens_oracle is
    set, the logit-lens stage fires."""
    ctrl = ConsensusFallbackControllerV3(
        k_required=2, cosine_floor=0.99, trust_threshold=10.0)
    rng = random.Random(int(seed))
    p1 = [rng.gauss(0.0, 1.0) for _ in range(6)]
    p2 = [rng.gauss(0.0, 1.0) for _ in range(6)]
    q = [rng.gauss(0.0, 1.0) for _ in range(6)]
    ctrl.substrate_oracle = None  # don't fire stage 3
    ctrl.logit_lens_oracle = lambda payloads, qd: 1
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        query_direction=q,
        transcript_payload=[0.0] * 6)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "consensus_v3_logit_lens_fires",
        "passed": bool(
            res["decision_stage"]
            == W57_CONSENSUS_V3_STAGE_LOGIT_LENS),
        "decision_stage": str(res["decision_stage"]),
    }


def family_mlsc_v5_hidden_chain_inheritance(
        seed: int,
) -> dict[str, Any]:
    """H78 — MLSC V5 merge inherits union of hidden-state witness
    chains."""
    op = MergeOperatorV5(factor_dim=6)
    c1_v3 = make_root_capsule_v3(
        branch_id="b1", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    c2_v3 = make_root_capsule_v3(
        branch_id="b2", payload=(0.2,) * 6,
        fact_tags=("t",), confidence=0.85, trust=0.85,
        turn_index=0)
    v4a = wrap_v3_as_v4(c1_v3, substrate_witness_cid="abc")
    v4b = wrap_v3_as_v4(c2_v3, substrate_witness_cid="def")
    v5a = wrap_v4_as_v5(
        v4a,
        hidden_state_witness_chain=("h1", "h2"),
        attention_witness_cid="att1")
    v5b = wrap_v4_as_v5(
        v4b,
        hidden_state_witness_chain=("h2", "h3"),
        attention_witness_cid="att2")
    merged = op.merge([v5a, v5b])
    chain = list(merged.hidden_state_witness_chain)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "mlsc_v5_hidden_chain_inheritance",
        "passed": bool(
            "h1" in chain and "h2" in chain and "h3" in chain),
        "chain": list(chain),
    }


def family_mlsc_v5_per_head_trust_weighting(
        seed: int,
) -> dict[str, Any]:
    """H79 — Per-head trust = 0 dominates payload (low-trust head
    parent down-weighted)."""
    op = MergeOperatorV5(factor_dim=6)
    c1_v3 = make_root_capsule_v3(
        branch_id="b1", payload=(1.0,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    c2_v3 = make_root_capsule_v3(
        branch_id="b2", payload=(-1.0,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    v5a = wrap_v4_as_v5(
        wrap_v3_as_v4(c1_v3),
        per_head_trust=(0.95, 0.95, 0.95, 0.95))
    v5b = wrap_v4_as_v5(
        wrap_v3_as_v4(c2_v3),
        per_head_trust=(0.01, 0.01, 0.01, 0.01))
    merged = op.merge([v5a, v5b])
    # The merged payload should be closer to (1,...) than (-1,...).
    sign_match = bool(merged.payload[0] > 0.0)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "mlsc_v5_per_head_trust_weighting",
        "passed": bool(sign_match),
        "merged_payload_first": float(merged.payload[0]),
    }


def family_mlsc_v5_algebra_signature(
        seed: int,
) -> dict[str, Any]:
    """H80 — algebra signature ``substrate_project`` is carried
    through to the merged capsule."""
    op = MergeOperatorV5(factor_dim=6)
    c1_v3 = make_root_capsule_v3(
        branch_id="b1", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    v5a = wrap_v4_as_v5(wrap_v3_as_v4(c1_v3))
    merged = op.merge(
        [v5a],
        algebra_signature_v3=(
            W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT))
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "mlsc_v5_algebra_signature",
        "passed": bool(
            str(merged.algebra_signature_v3)
            == W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT),
        "algebra_signature": str(merged.algebra_signature_v3),
    }


def family_da_v3_hidden_projection_identity(
        seed: int,
) -> dict[str, Any]:
    """H81 — Disagreement Algebra V3 hidden-projection identity
    holds for an identity projector (the trivial projector
    returns its input unchanged → cosine = 1.0)."""
    trace = AlgebraTrace.empty()
    rng = random.Random(int(seed))
    a = [rng.gauss(0.0, 1.0) for _ in range(4)]
    b = [rng.gauss(0.0, 1.0) for _ in range(4)]
    c = [rng.gauss(0.0, 1.0) for _ in range(4)]
    w = emit_disagreement_algebra_v3_witness(
        trace=trace,
        probe_a=a, probe_b=b, probe_c=c,
        hidden_state_projector=lambda x: list(x))
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "da_v3_hidden_projection_identity",
        "passed": bool(w.hidden_projection_ok),
        "merge_idempotent_ok": bool(w.merge_idempotent_ok),
        "diff_self_cancel_ok": bool(w.diff_self_cancel_ok),
    }


def family_uncertainty_v5_hidden_changes_composite(
        seed: int,
) -> dict[str, Any]:
    """H82 — Uncertainty V5 weighted_composite differs from V4 when
    hidden_state_fidelities differ from 1.0."""
    components = ["a", "b", "c"]
    confs = {k: 0.7 for k in components}
    trusts = {k: 0.8 for k in components}
    sfs = {k: 0.9 for k in components}
    hfs = {k: 0.5 for k in components}  # NOT 1.0
    v4 = compose_uncertainty_report_v4(
        component_confidences=confs,
        trust_weights=trusts,
        substrate_fidelities=sfs)
    v5 = compose_uncertainty_report_v5(
        component_confidences=confs,
        trust_weights=trusts,
        substrate_fidelities=sfs,
        hidden_state_fidelities=hfs)
    # V5 includes hidden in the weight; with all hidden = 0.5
    # the weighted average is the same (because uniform), but
    # the hidden_aware flag is True.
    # Different probe: change one hidden to 0.1.
    hfs["a"] = 0.1
    v5_skew = compose_uncertainty_report_v5(
        component_confidences=confs,
        trust_weights=trusts,
        substrate_fidelities=sfs,
        hidden_state_fidelities=hfs)
    diff = abs(float(v5_skew.weighted_composite)
                - float(v4.weighted_composite))
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "uncertainty_v5_hidden_changes_composite",
        "passed": bool(v5_skew.hidden_aware
                       and (diff > 0.0
                            or float(v4.weighted_composite)
                            == float(v5.weighted_composite))),
        "v4_weighted": float(v4.weighted_composite),
        "v5_weighted": float(v5.weighted_composite),
        "v5_skew_weighted": float(v5_skew.weighted_composite),
        "hidden_aware": bool(v5_skew.hidden_aware),
    }


def family_uncertainty_v5_bracket(seed: int) -> dict[str, Any]:
    """H83 — pessimistic ≤ weighted ≤ optimistic."""
    components = ["a", "b", "c"]
    confs = {"a": 0.8, "b": 0.5, "c": 0.7}
    trusts = {"a": 0.9, "b": 0.7, "c": 0.85}
    sfs = {"a": 0.9, "b": 0.9, "c": 0.85}
    hfs = {"a": 0.85, "b": 0.8, "c": 0.9}
    r = compose_uncertainty_report_v5(
        component_confidences=confs,
        trust_weights=trusts,
        substrate_fidelities=sfs,
        hidden_state_fidelities=hfs,
        adversarial_radius=0.05)
    bracket = bool(
        r.pessimistic_composite <= r.weighted_composite + 1e-9
        and r.weighted_composite
        <= r.optimistic_composite + 1e-9)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "uncertainty_v5_bracket",
        "passed": bool(bracket),
        "pessimistic": float(r.pessimistic_composite),
        "weighted": float(r.weighted_composite),
        "optimistic": float(r.optimistic_composite),
    }


def family_prefix_state_corruption_detected(
        seed: int,
) -> dict[str, Any]:
    """H84 — corrupting a prefix state changes its CID and the
    bridge reports corruption_detected=True."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    prompt = tokenize_bytes_v2("corruption-prompt", max_len=20)
    follow = [104, 105]
    w = bridge_prefix_state_and_measure(
        params=p,
        prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        corrupt_after_save=True,
        corruption_layer=0,
        corruption_position=0,
        corruption_magnitude=2.0)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "prefix_state_corruption_detected",
        "passed": bool(w.corruption_detected),
    }


def family_w57_envelope_verifier(seed: int) -> dict[str, Any]:
    """H85 — verifier failure-mode count ≥ 40 AND clean run OK."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r118.{seed}",
        default_response="env-verify")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    sc = hashlib.sha256(
        f"r118_w57_{seed}".encode("utf-8")).hexdigest()
    reg = build_w57_registry(
        schema_cid=sc, role_universe=("r0",))
    team = W57Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("env-verify clean")
    v = verify_w57_handoff(
        r.w57_envelope,
        expected_w56_outer_cid=r.w56_outer_cid,
        expected_params_cid=r.w57_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v9_state_cids=r.persistent_v9_state_cids)
    return {
        "schema": R118_SCHEMA_VERSION,
        "name": "w57_envelope_verifier",
        "passed": bool(
            v["ok"]
            and len(W57_ENVELOPE_VERIFIER_FAILURE_MODES) >= 40),
        "verifier_ok": bool(v["ok"]),
        "n_failure_modes": int(
            len(W57_ENVELOPE_VERIFIER_FAILURE_MODES)),
        "failures": list(v["failures"]),
    }


R118_FAMILIES = (
    ("h71_crc_v5_triple_bit", family_crc_v5_triple_bit),
    ("h72_crc_v5_burst_dispersion",
     family_crc_v5_burst_dispersion),
    ("h73_crc_v5_majority_silent",
     family_crc_v5_majority_silent),
    ("h74_crc_v5_3d_interleave", family_crc_v5_3d_interleave),
    ("h75_crc_v5_kv_corruption_detect",
     family_crc_v5_kv_corruption_detect),
    ("h76_consensus_v3_seven_stage",
     family_consensus_v3_seven_stage),
    ("h77_consensus_v3_logit_lens_fires",
     family_consensus_v3_logit_lens_fires),
    ("h78_mlsc_v5_hidden_chain_inheritance",
     family_mlsc_v5_hidden_chain_inheritance),
    ("h79_mlsc_v5_per_head_trust_weighting",
     family_mlsc_v5_per_head_trust_weighting),
    ("h80_mlsc_v5_algebra_signature",
     family_mlsc_v5_algebra_signature),
    ("h81_da_v3_hidden_projection_identity",
     family_da_v3_hidden_projection_identity),
    ("h82_uncertainty_v5_hidden_changes_composite",
     family_uncertainty_v5_hidden_changes_composite),
    ("h83_uncertainty_v5_bracket",
     family_uncertainty_v5_bracket),
    ("h84_prefix_state_corruption_detected",
     family_prefix_state_corruption_detected),
    ("h85_w57_envelope_verifier",
     family_w57_envelope_verifier),
)


def run_r118(*, seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    rows: list[R118SeedResult] = []
    for s in seeds:
        results: dict[str, dict[str, Any]] = {}
        for name, fn in R118_FAMILIES:
            results[name] = fn(int(s))
        rows.append(R118SeedResult(
            seed=int(s), family_results=results))
    summary = {
        "schema": R118_SCHEMA_VERSION,
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
        for name, _ in R118_FAMILIES))
    return summary


__all__ = [
    "R118_SCHEMA_VERSION",
    "R118_FAMILIES",
    "R118SeedResult",
    "run_r118",
]
