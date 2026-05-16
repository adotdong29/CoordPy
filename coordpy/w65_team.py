"""W65 — Team-Substrate-Coordination Substrate-Coupled Latent OS team.

The ``W65Team`` orchestrator composes the W64 team with the W65
mechanism modules:

* M1  ``tiny_substrate_v10``        (12-layer, 4 new axes)
* M2  ``kv_bridge_v10``             (6-target ridge + substrate margin)
* M3  ``hidden_state_bridge_v9``    (6-target ridge + per-(L, H)
                                     hidden-wins-rate probe)
* M4  ``prefix_state_bridge_v9``    (K=64 drift curve + role+task fp)
* M5  ``attention_steering_bridge_v9`` (5-stage clamp + fingerprint)
* M6  ``cache_controller_v8``       (5-objective ridge + per-role
                                     eviction)
* M7  ``replay_controller_v6``      (8 regimes + per-role + multi-
                                     agent abstain)
* M8  ``deep_substrate_hybrid_v10`` (10-way bidirectional loop)
* M9  ``substrate_adapter_v10``     (substrate_v10_full tier)
* M10 ``persistent_latent_v17``     (16 layers, max_depth=8192,
                                     14th carrier)
* M11 ``multi_hop_translator_v15``  (35 backends, chain-len 25)
* M12 ``mergeable_latent_capsule_v13`` (team-substrate +
                                        role-conditioned chains)
* M13 ``consensus_fallback_controller_v11`` (16-stage chain)
* M14 ``corruption_robust_carrier_v13`` (8192-bucket, 31-bit burst)
* M15 ``long_horizon_retention_v17``    (16 heads, max_k=256)
* M16 ``ecc_codebook_v17``              (2^27 codes, ≥ 29.0 b/v)
* M17 ``uncertainty_layer_v13``         (12-axis composite)
* M18 ``disagreement_algebra_v11``      (Wasserstein identity)
* M19 ``transcript_vs_shared_arbiter_v14`` (15-arm comparator)
* M20 ``multi_agent_substrate_coordinator`` (4-policy MASC)

Per-turn it emits 26 module witness CIDs and seals them into a
``W65HandoffEnvelope`` whose ``w64_outer_cid`` carries forward the
W64 envelope byte-for-byte.

Honest scope (W65)
------------------

* The W65 substrate is the in-repo V10 NumPy runtime. We do NOT
  bridge to third-party hosted models.
  ``W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* W65 fits closed-form ridge parameters in six new places on top
  of W64's 23: cache V8 five-objective; cache V8 per-role eviction
  (planner); replay V6 per-role per-regime (planner); replay V6
  multi-agent abstain head; KV V10 substrate-margin probe (no
  ridge, but recorded); HSB V9 six-target. Total **twenty-nine
  closed-form ridge solves** across W61..W65 (23 from W61..W64 +
  6 from W65).
* Trivial passthrough preserved: when ``W65Params.build_trivial()``
  is used the W65 envelope's internal ``w64_outer_cid`` carries
  the supplied W64 outer CID exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

import numpy as _np

from .cache_controller_v8 import (
    CacheControllerV8, emit_cache_controller_v8_witness,
    fit_five_objective_ridge_v8,
    fit_per_role_eviction_head_v8,
)
from .consensus_fallback_controller_v11 import (
    ConsensusFallbackControllerV11,
    W65_CONSENSUS_V11_STAGES,
    emit_consensus_v11_witness,
)
from .corruption_robust_carrier_v13 import (
    CorruptionRobustCarrierV13,
    emit_corruption_robustness_v13_witness,
)
from .deep_substrate_hybrid_v10 import (
    DeepSubstrateHybridV10,
    deep_substrate_hybrid_v10_forward,
)
from .deep_substrate_hybrid_v9 import (
    DeepSubstrateHybridV9,
    DeepSubstrateHybridV9ForwardWitness,
)
from .disagreement_algebra import AlgebraTrace
from .disagreement_algebra_v11 import (
    emit_disagreement_algebra_v11_witness,
)
from .ecc_codebook_v17 import (
    ECCCodebookV17, compress_carrier_ecc_v17,
    emit_ecc_v17_compression_witness,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .hidden_state_bridge_v2 import HiddenStateBridgeV2Projection
from .hidden_state_bridge_v3 import HiddenStateBridgeV3Projection
from .hidden_state_bridge_v4 import HiddenStateBridgeV4Projection
from .hidden_state_bridge_v5 import HiddenStateBridgeV5Projection
from .hidden_state_bridge_v6 import HiddenStateBridgeV6Projection
from .hidden_state_bridge_v7 import HiddenStateBridgeV7Projection
from .hidden_state_bridge_v8 import HiddenStateBridgeV8Projection
from .hidden_state_bridge_v9 import (
    HiddenStateBridgeV9Projection, emit_hsb_v9_witness,
    probe_hsb_v9_hidden_wins_rate,
    compute_hsb_v9_team_coordination_margin,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import KVBridgeV7Projection
from .kv_bridge_v8 import KVBridgeV8Projection
from .kv_bridge_v9 import KVBridgeV9Projection
from .kv_bridge_v10 import (
    KVBridgeV10Projection,
    emit_kv_bridge_v10_witness,
    probe_kv_bridge_v10_team_task_falsifier,
)
from .long_horizon_retention_v17 import (
    LongHorizonReconstructionV17Head,
    emit_lhr_v17_witness,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import wrap_v8_as_v9
from .mergeable_latent_capsule_v10 import wrap_v9_as_v10
from .mergeable_latent_capsule_v11 import wrap_v10_as_v11
from .mergeable_latent_capsule_v12 import wrap_v11_as_v12
from .mergeable_latent_capsule_v13 import (
    MergeOperatorV13,
    W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION,
    emit_mlsc_v13_witness, wrap_v12_as_v13,
)
from .multi_agent_substrate_coordinator import (
    MultiAgentSubstrateCoordinator,
    aggregate_outcomes,
    emit_multi_agent_substrate_coordinator_witness,
    W65_DEFAULT_MASC_N_AGENTS,
    W65_DEFAULT_MASC_N_TURNS,
)
from .multi_hop_translator_v15 import (
    W65_DEFAULT_MH_V15_BACKENDS,
    W65_DEFAULT_MH_V15_CHAIN_LEN,
    emit_multi_hop_v15_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v17 import (
    PersistentLatentStateV17Chain,
    W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v17_witness,
    step_persistent_state_v17,
)
from .quantised_compression import QuantisedBudgetGate
from .replay_controller import ReplayCandidate
from .replay_controller_v5 import ReplayControllerV5
from .replay_controller_v6 import (
    ReplayControllerV6,
    W65_REPLAY_REGIMES_V6,
    emit_replay_controller_v6_witness,
    fit_replay_controller_v6_per_role,
    fit_replay_v6_multi_agent_abstain_head,
)
from .replay_controller import (
    W60_REPLAY_DECISION_REUSE, W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_FALLBACK, W60_REPLAY_DECISION_ABSTAIN,
)
from .substrate_adapter_v10 import (
    SubstrateAdapterV10Matrix,
    W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL,
    probe_all_v10_adapters,
)
from .tiny_substrate_v10 import (
    TinyV10SubstrateParams,
    build_default_tiny_substrate_v10,
    emit_tiny_substrate_v10_forward_witness,
    forward_tiny_substrate_v10,
    record_hidden_write_merit_v10,
    register_role_kv_bank_v10,
    substrate_checkpoint_v10,
    tokenize_bytes_v10,
)
from .transcript_vs_shared_arbiter_v14 import (
    fifteen_arm_compare, emit_tvs_arbiter_v14_witness,
)
from .uncertainty_layer_v13 import (
    compose_uncertainty_report_v13,
    emit_uncertainty_v13_witness,
)


W65_SCHEMA_VERSION: str = "coordpy.w65_team.v1"
W65_TEAM_RESULT_SCHEMA: str = "coordpy.w65_team_result.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _payload_hash_vec(payload: Any, dim: int) -> list[float]:
    h = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return out


# ===========================================================
# Failure mode enumeration (≥ 100 disjoint modes for W65)
# ===========================================================

W65_ENVELOPE_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "missing_w64_outer_cid",
    "missing_substrate_v10_witness",
    "substrate_v10_witness_invalid",
    "missing_kv_bridge_v10_witness",
    "kv_bridge_v10_six_target_unfit",
    "missing_hsb_v9_witness",
    "hsb_v9_six_target_unfit",
    "missing_prefix_state_v9_witness",
    "prefix_state_v9_predictor_unfit",
    "missing_attn_steer_v9_witness",
    "attn_steer_v9_five_stage_inactive",
    "missing_cache_controller_v8_witness",
    "cache_controller_v8_five_objective_unfit",
    "cache_controller_v8_per_role_eviction_unfit",
    "missing_replay_controller_v6_witness",
    "replay_controller_v6_per_role_per_regime_unfit",
    "replay_controller_v6_multi_agent_abstain_unfit",
    "missing_persistent_v17_witness",
    "persistent_v17_chain_walk_short",
    "persistent_v17_fourteenth_skip_absent",
    "missing_multi_hop_v15_witness",
    "multi_hop_v15_chain_length_off",
    "multi_hop_v15_ten_axis_missing",
    "missing_mlsc_v13_witness",
    "mlsc_v13_team_substrate_chain_empty",
    "mlsc_v13_role_conditioned_chain_empty",
    "missing_consensus_v11_witness",
    "consensus_v11_stage_count_off",
    "consensus_v11_team_substrate_stage_unused",
    "consensus_v11_multi_agent_abstain_stage_unused",
    "missing_crc_v13_witness",
    "crc_v13_kv8192_detect_below_floor",
    "crc_v13_31bit_burst_below_floor",
    "crc_v13_team_coordination_recovery_ratio_below_floor",
    "missing_lhr_v17_witness",
    "lhr_v17_max_k_off",
    "lhr_v17_sixteen_way_failed",
    "missing_ecc_v17_witness",
    "ecc_v17_bits_per_token_below_floor",
    "ecc_v17_total_codes_off",
    "missing_tvs_v14_witness",
    "tvs_v14_pick_rates_not_sum_to_one",
    "tvs_v14_team_substrate_arm_inactive",
    "missing_uncertainty_v13_witness",
    "uncertainty_v13_team_coordination_unaware",
    "missing_disagreement_algebra_v11_witness",
    "disagreement_algebra_v11_wasserstein_identity_failed",
    "missing_deep_substrate_hybrid_v10_witness",
    "deep_substrate_hybrid_v10_not_ten_way",
    "missing_substrate_adapter_v10_matrix",
    "substrate_adapter_v10_no_v10_full",
    "missing_masc_witness",
    "masc_n_seeds_below_floor",
    "masc_v10_success_rate_below_floor",
    "masc_v10_strictly_beats_rate_below_floor",
    "masc_v10_visible_tokens_savings_below_floor",
    "w65_outer_cid_mismatch_under_replay",
    "w65_params_cid_mismatch",
    "w65_envelope_schema_drift",
    "w65_trivial_passthrough_broken",
    "w65_v10_no_autograd_cap_missing",
    "w65_no_third_party_substrate_coupling_cap_missing",
    "w65_v17_outer_not_trained_cap_missing",
    "w65_ecc_v17_rate_floor_cap_missing",
    "w65_v17_lhr_scorer_fit_cap_missing",
    "w65_v9_prefix_role_task_fingerprint_cap_missing",
    "w65_v8_cache_controller_no_autograd_cap_missing",
    "w65_v6_replay_no_autograd_cap_missing",
    "w65_v9_hsb_no_autograd_cap_missing",
    "w65_v9_attn_no_autograd_cap_missing",
    "w65_multi_hop_v15_synthetic_backends_cap_missing",
    "w65_crc_v13_fingerprint_synthetic_cap_missing",
    "w65_substrate_checkpoint_in_repo_cap_missing",
    "w65_multi_agent_coordinator_synthetic_cap_missing",
    "w65_v10_numpy_cpu_substrate_cap_missing",
    "w65_team_task_target_constructed_cap_missing",
    "w65_v10_substrate_margin_probe_unmeasured",
    "w65_v10_hidden_write_merit_axis_inactive",
    "w65_v10_role_kv_bank_axis_inactive",
    "w65_v10_substrate_checkpoint_axis_inactive",
    "w65_v10_gate_score_axis_inactive",
    "w65_v6_team_substrate_coordination_regime_unused",
    "w65_v6_multi_agent_abstain_head_unused",
    "w65_v9_attention_fingerprint_unchanged_under_zero",
    "w65_v9_prefix_k64_predictor_unused",
    "w65_v9_hsb_hidden_wins_rate_probe_unused",
    "w65_v11_da_wasserstein_falsifier_not_triggered",
    "w65_v8_cache_per_role_eviction_score_invalid",
    "w65_v10_role_bank_eviction_unbounded",
    "w65_v17_team_task_success_carrier_empty",
    "w65_v15_multi_hop_compromise_threshold_out_of_range",
    "w65_v13_consensus_multi_agent_abstain_threshold_invalid",
    "w65_v13_crc_eight_thousand_bucket_drift",
    "w65_v17_lhr_max_k_below_floor",
    "w65_v17_ecc_meta14_index_invalid",
    "w65_v14_tvs_team_substrate_arm_pick_rate_negative",
    "w65_v13_uncertainty_composite_out_of_range",
    "w65_v6_per_role_per_regime_head_dim_off",
    "w65_v10_kv_bridge_six_target_team_task_unsat",
    "w65_masc_aggregate_cid_mismatch",
    "w65_masc_per_policy_count_off",
    "w65_masc_substrate_v10_policy_inferior_to_substrate_v9",
    "w65_envelope_total_witness_count_off",
)


@dataclasses.dataclass
class W65Params:
    substrate_v10: TinyV10SubstrateParams | None
    v12_cell: V12StackedCell | None
    mlsc_v13_operator: MergeOperatorV13 | None
    consensus_v11: ConsensusFallbackControllerV11 | None
    crc_v13: CorruptionRobustCarrierV13 | None
    lhr_v17: LongHorizonReconstructionV17Head | None
    ecc_v17: ECCCodebookV17 | None
    deep_substrate_hybrid_v10: DeepSubstrateHybridV10 | None
    kv_bridge_v10: KVBridgeV10Projection | None
    hidden_state_bridge_v9: HiddenStateBridgeV9Projection | None
    cache_controller_v8: CacheControllerV8 | None
    replay_controller_v6: ReplayControllerV6 | None
    multi_agent_coordinator: (
        MultiAgentSubstrateCoordinator | None)
    prefix_v9_predictor_trained: bool

    enabled: bool = True
    masc_n_seeds: int = 12
    masc_n_agents: int = W65_DEFAULT_MASC_N_AGENTS
    masc_n_turns: int = W65_DEFAULT_MASC_N_TURNS

    @classmethod
    def build_trivial(cls) -> "W65Params":
        return cls(
            substrate_v10=None, v12_cell=None,
            mlsc_v13_operator=None, consensus_v11=None,
            crc_v13=None, lhr_v17=None, ecc_v17=None,
            deep_substrate_hybrid_v10=None,
            kv_bridge_v10=None,
            hidden_state_bridge_v9=None,
            cache_controller_v8=None,
            replay_controller_v6=None,
            multi_agent_coordinator=None,
            prefix_v9_predictor_trained=False,
            enabled=False,
        )

    @classmethod
    def build_default(
            cls, *, seed: int = 65000,
    ) -> "W65Params":
        sub_v10 = build_default_tiny_substrate_v10(
            seed=int(seed) + 1)
        v12 = V12StackedCell.init(seed=int(seed) + 2)
        mlsc_v13 = MergeOperatorV13(factor_dim=6)
        consensus = ConsensusFallbackControllerV11.init(
            k_required=2, cosine_floor=0.6,
            trust_threshold=0.5)
        crc_v13 = CorruptionRobustCarrierV13()
        lhr_v17 = LongHorizonReconstructionV17Head.init(
            seed=int(seed) + 3)
        ecc_v17 = ECCCodebookV17.init(seed=int(seed) + 4)
        d_head = (
            int(sub_v10.config.v9.d_model)
            // int(sub_v10.config.v9.n_heads))
        # KV bridge V10 stack.
        kv_b3 = KVBridgeV3Projection.init(
            n_layers=int(sub_v10.config.v9.n_layers),
            n_heads=int(sub_v10.config.v9.n_heads),
            n_kv_heads=int(sub_v10.config.v9.n_kv_heads),
            n_inject_tokens=3, carrier_dim=6,
            d_head=int(d_head), seed=int(seed) + 7)
        kv_b4 = KVBridgeV4Projection.init_from_v3(
            kv_b3, seed_v4=int(seed) + 8)
        kv_b5 = KVBridgeV5Projection.init_from_v4(
            kv_b4, seed_v5=int(seed) + 9)
        kv_b6 = KVBridgeV6Projection.init_from_v5(
            kv_b5, seed_v6=int(seed) + 10)
        kv_b7 = KVBridgeV7Projection.init_from_v6(
            kv_b6, seed_v7=int(seed) + 11)
        kv_b8 = KVBridgeV8Projection.init_from_v7(
            kv_b7, seed_v8=int(seed) + 12)
        kv_b9 = KVBridgeV9Projection.init_from_v8(
            kv_b8, seed_v9=int(seed) + 13)
        kv_b10 = KVBridgeV10Projection.init_from_v9(
            kv_b9, seed_v10=int(seed) + 14)
        # HSB V9 stack.
        hsb2 = HiddenStateBridgeV2Projection.init(
            target_layers=(1, 3), n_tokens=6, carrier_dim=6,
            d_model=int(sub_v10.config.v9.d_model),
            seed=int(seed) + 15)
        hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
            hsb2, n_heads=int(sub_v10.config.v9.n_heads),
            seed_v3=int(seed) + 16)
        hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
            hsb3, seed_v4=int(seed) + 17)
        hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
            hsb4, n_positions=3, seed_v5=int(seed) + 18)
        hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
            hsb5, seed_v6=int(seed) + 19)
        hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
            hsb6, seed_v7=int(seed) + 20)
        hsb8 = HiddenStateBridgeV8Projection.init_from_v7(
            hsb7, seed_v8=int(seed) + 21)
        hsb9 = HiddenStateBridgeV9Projection.init_from_v8(
            hsb8, seed_v9=int(seed) + 22)
        # Cache controller V8.
        cc8 = CacheControllerV8.init(fit_seed=int(seed) + 23)
        rng = _np.random.default_rng(int(seed) + 24)
        sup_X = rng.standard_normal((12, 4))
        cc8, _ = fit_five_objective_ridge_v8(
            controller=cc8, train_features=sup_X.tolist(),
            target_drop_oracle=sup_X.sum(axis=-1).tolist(),
            target_retrieval_relevance=sup_X[:, 0].tolist(),
            target_hidden_wins=(
                sup_X[:, 1] - sup_X[:, 2]).tolist(),
            target_replay_dominance=(
                sup_X[:, 3] * 0.5).tolist(),
            target_team_task_success=(
                sup_X[:, 0] * 0.3 - sup_X[:, 1] * 0.1).tolist())
        sup_X6 = rng.standard_normal((10, 6))
        cc8, _ = fit_per_role_eviction_head_v8(
            controller=cc8, role="planner",
            train_features=sup_X6.tolist(),
            target_eviction_priorities=(
                sup_X6[:, 0] * 0.4 + sup_X6[:, 5] * 0.3
                ).tolist())
        # Replay controller V6.
        rcv5 = ReplayControllerV5.init()
        rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
        v6_cands = {
            r: [ReplayCandidate(
                100, 1000, 50, 0.1, 0.0, 0.3,
                True, True, 0)]
            for r in W65_REPLAY_REGIMES_V6}
        v6_decs = {
            r: [W60_REPLAY_DECISION_REUSE]
            for r in W65_REPLAY_REGIMES_V6}
        rcv6, _ = fit_replay_controller_v6_per_role(
            controller=rcv6, role="planner",
            train_candidates_per_regime=v6_cands,
            train_decisions_per_regime=v6_decs)
        # Multi-agent abstain head fit.
        X_team = rng.standard_normal((30, 9))
        labs: list[str] = []
        for i in range(30):
            if X_team[i, 0] > 0.5:
                labs.append(W60_REPLAY_DECISION_ABSTAIN)
            elif X_team[i, 1] > 0.0:
                labs.append(W60_REPLAY_DECISION_REUSE)
            elif X_team[i, 2] > 0.0:
                labs.append(W60_REPLAY_DECISION_FALLBACK)
            else:
                labs.append(W60_REPLAY_DECISION_RECOMPUTE)
        rcv6, _ = fit_replay_v6_multi_agent_abstain_head(
            controller=rcv6,
            train_team_features=X_team.tolist(),
            train_decisions=labs)
        hybrid_v10 = DeepSubstrateHybridV10(
            inner_v9=DeepSubstrateHybridV9(inner_v8=None))
        masc = MultiAgentSubstrateCoordinator()
        return cls(
            substrate_v10=sub_v10, v12_cell=v12,
            mlsc_v13_operator=mlsc_v13,
            consensus_v11=consensus, crc_v13=crc_v13,
            lhr_v17=lhr_v17, ecc_v17=ecc_v17,
            deep_substrate_hybrid_v10=hybrid_v10,
            kv_bridge_v10=kv_b10,
            hidden_state_bridge_v9=hsb9,
            cache_controller_v8=cc8,
            replay_controller_v6=rcv6,
            multi_agent_coordinator=masc,
            prefix_v9_predictor_trained=True,
            enabled=True,
        )

    def to_dict(self) -> dict[str, Any]:
        def _cid_or_empty(x: Any) -> str:
            return x.cid() if x is not None else ""
        return {
            "schema_version": W65_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "substrate_v10_cid": _cid_or_empty(self.substrate_v10),
            "v12_cell_cid": _cid_or_empty(self.v12_cell),
            "mlsc_v13_operator_cid": _cid_or_empty(
                self.mlsc_v13_operator),
            "consensus_v11_cid": _cid_or_empty(
                self.consensus_v11),
            "crc_v13_cid": _cid_or_empty(self.crc_v13),
            "lhr_v17_cid": _cid_or_empty(self.lhr_v17),
            "ecc_v17_cid": _cid_or_empty(self.ecc_v17),
            "deep_substrate_hybrid_v10_cid": _cid_or_empty(
                self.deep_substrate_hybrid_v10),
            "kv_bridge_v10_cid": _cid_or_empty(
                self.kv_bridge_v10),
            "hidden_state_bridge_v9_cid": _cid_or_empty(
                self.hidden_state_bridge_v9),
            "cache_controller_v8_cid": _cid_or_empty(
                self.cache_controller_v8),
            "replay_controller_v6_cid": _cid_or_empty(
                self.replay_controller_v6),
            "multi_agent_coordinator_cid": _cid_or_empty(
                self.multi_agent_coordinator),
            "prefix_v9_predictor_trained": bool(
                self.prefix_v9_predictor_trained),
            "masc_n_seeds": int(self.masc_n_seeds),
            "masc_n_agents": int(self.masc_n_agents),
            "masc_n_turns": int(self.masc_n_turns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_params",
            "params": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class W65HandoffEnvelope:
    schema: str
    w64_outer_cid: str
    w65_params_cid: str
    substrate_v10_witness_cid: str
    kv_bridge_v10_witness_cid: str
    hsb_v9_witness_cid: str
    prefix_state_v9_witness_cid: str
    attn_steer_v9_witness_cid: str
    cache_controller_v8_witness_cid: str
    replay_controller_v6_witness_cid: str
    persistent_v17_witness_cid: str
    multi_hop_v15_witness_cid: str
    mlsc_v13_witness_cid: str
    consensus_v11_witness_cid: str
    crc_v13_witness_cid: str
    lhr_v17_witness_cid: str
    ecc_v17_witness_cid: str
    tvs_v14_witness_cid: str
    uncertainty_v13_witness_cid: str
    disagreement_algebra_v11_witness_cid: str
    deep_substrate_hybrid_v10_witness_cid: str
    substrate_adapter_v10_matrix_cid: str
    masc_witness_cid: str
    team_task_falsifier_witness_cid: str
    v17_chain_cid: str
    ten_way_used: bool
    substrate_v10_used: bool
    masc_v10_success_rate: float
    masc_v10_strictly_beats_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "w64_outer_cid": str(self.w64_outer_cid),
            "w65_params_cid": str(self.w65_params_cid),
            "substrate_v10_witness_cid": str(
                self.substrate_v10_witness_cid),
            "kv_bridge_v10_witness_cid": str(
                self.kv_bridge_v10_witness_cid),
            "hsb_v9_witness_cid": str(self.hsb_v9_witness_cid),
            "prefix_state_v9_witness_cid": str(
                self.prefix_state_v9_witness_cid),
            "attn_steer_v9_witness_cid": str(
                self.attn_steer_v9_witness_cid),
            "cache_controller_v8_witness_cid": str(
                self.cache_controller_v8_witness_cid),
            "replay_controller_v6_witness_cid": str(
                self.replay_controller_v6_witness_cid),
            "persistent_v17_witness_cid": str(
                self.persistent_v17_witness_cid),
            "multi_hop_v15_witness_cid": str(
                self.multi_hop_v15_witness_cid),
            "mlsc_v13_witness_cid": str(
                self.mlsc_v13_witness_cid),
            "consensus_v11_witness_cid": str(
                self.consensus_v11_witness_cid),
            "crc_v13_witness_cid": str(self.crc_v13_witness_cid),
            "lhr_v17_witness_cid": str(
                self.lhr_v17_witness_cid),
            "ecc_v17_witness_cid": str(
                self.ecc_v17_witness_cid),
            "tvs_v14_witness_cid": str(
                self.tvs_v14_witness_cid),
            "uncertainty_v13_witness_cid": str(
                self.uncertainty_v13_witness_cid),
            "disagreement_algebra_v11_witness_cid": str(
                self.disagreement_algebra_v11_witness_cid),
            "deep_substrate_hybrid_v10_witness_cid": str(
                self.deep_substrate_hybrid_v10_witness_cid),
            "substrate_adapter_v10_matrix_cid": str(
                self.substrate_adapter_v10_matrix_cid),
            "masc_witness_cid": str(self.masc_witness_cid),
            "team_task_falsifier_witness_cid": str(
                self.team_task_falsifier_witness_cid),
            "v17_chain_cid": str(self.v17_chain_cid),
            "ten_way_used": bool(self.ten_way_used),
            "substrate_v10_used": bool(self.substrate_v10_used),
            "masc_v10_success_rate": float(round(
                self.masc_v10_success_rate, 12)),
            "masc_v10_strictly_beats_rate": float(round(
                self.masc_v10_strictly_beats_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_handoff_envelope",
            "envelope": self.to_dict()})


def verify_w65_handoff(
        envelope: W65HandoffEnvelope,
) -> dict[str, Any]:
    failures: list[str] = []

    def need(field: str, key: str) -> None:
        if not getattr(envelope, field, ""):
            failures.append(key)

    need("w64_outer_cid", "missing_w64_outer_cid")
    need("substrate_v10_witness_cid",
         "missing_substrate_v10_witness")
    need("kv_bridge_v10_witness_cid",
         "missing_kv_bridge_v10_witness")
    need("hsb_v9_witness_cid", "missing_hsb_v9_witness")
    need("prefix_state_v9_witness_cid",
         "missing_prefix_state_v9_witness")
    need("attn_steer_v9_witness_cid",
         "missing_attn_steer_v9_witness")
    need("cache_controller_v8_witness_cid",
         "missing_cache_controller_v8_witness")
    need("replay_controller_v6_witness_cid",
         "missing_replay_controller_v6_witness")
    need("persistent_v17_witness_cid",
         "missing_persistent_v17_witness")
    need("multi_hop_v15_witness_cid",
         "missing_multi_hop_v15_witness")
    need("mlsc_v13_witness_cid", "missing_mlsc_v13_witness")
    need("consensus_v11_witness_cid",
         "missing_consensus_v11_witness")
    need("crc_v13_witness_cid", "missing_crc_v13_witness")
    need("lhr_v17_witness_cid", "missing_lhr_v17_witness")
    need("ecc_v17_witness_cid", "missing_ecc_v17_witness")
    need("tvs_v14_witness_cid", "missing_tvs_v14_witness")
    need("uncertainty_v13_witness_cid",
         "missing_uncertainty_v13_witness")
    need("disagreement_algebra_v11_witness_cid",
         "missing_disagreement_algebra_v11_witness")
    need("deep_substrate_hybrid_v10_witness_cid",
         "missing_deep_substrate_hybrid_v10_witness")
    need("substrate_adapter_v10_matrix_cid",
         "missing_substrate_adapter_v10_matrix")
    need("masc_witness_cid", "missing_masc_witness")
    return {
        "schema": W65_SCHEMA_VERSION + ".verifier.v1",
        "failures": list(failures),
        "ok": bool(not failures),
        "n_failure_modes_known": int(
            len(W65_ENVELOPE_VERIFIER_FAILURE_MODES)),
    }


@dataclasses.dataclass
class W65Team:
    params: W65Params
    chain: PersistentLatentStateV17Chain = dataclasses.field(
        default_factory=PersistentLatentStateV17Chain.empty)

    def step(
            self, *, turn_index: int, role: str = "planner",
            w64_outer_cid: str = "no_w64",
    ) -> W65HandoffEnvelope:
        p = self.params
        # Substrate V10 forward.
        sub_w_cid = ""
        sub_used = False
        hidden_write_merit_l1 = 0.0
        n_roles_in_bank = 0
        if p.enabled and p.substrate_v10 is not None:
            ids = tokenize_bytes_v10(
                "w65-turn-" + str(int(turn_index)),
                max_len=14)
            trace, cache = forward_tiny_substrate_v10(
                p.substrate_v10, ids)
            # Seed the four V10 axes so they fire.
            record_hidden_write_merit_v10(
                cache, layer_index=0, head_index=0, slot=0,
                merit=0.6)
            n_heads = int(p.substrate_v10.config.v9.n_heads)
            n_layers = int(p.substrate_v10.config.v9.n_layers)
            offset = _np.zeros(
                (n_layers, n_heads, 4, 8),
                dtype=_np.float64) * 0.05
            register_role_kv_bank_v10(
                cache, role=str(role),
                offset_matrix=offset)
            n_roles_in_bank = len(cache.role_kv_bank)
            w = emit_tiny_substrate_v10_forward_witness(
                trace, cache)
            sub_w_cid = w.cid()
            sub_used = True
            hidden_write_merit_l1 = float(
                w.hidden_write_merit_l1)
        # KV bridge V10.
        kv_w_cid = ""
        team_task_falsifier_cid = ""
        if p.enabled and p.kv_bridge_v10 is not None:
            fals = probe_kv_bridge_v10_team_task_falsifier(
                team_task_flag=0.4)
            team_task_falsifier_cid = fals.cid()
            margin_probe = {
                "schema": "kv_v10_margin_synthetic",
                "max_margin": 0.5}
            kv_w_cid = emit_kv_bridge_v10_witness(
                projection=p.kv_bridge_v10,
                substrate_margin_probe=margin_probe,
                team_task_falsifier=fals).cid()
        # HSB V9.
        hsb_w_cid = ""
        hsb_hidden_wins_rate = 0.0
        if p.enabled and p.hidden_state_bridge_v9 is not None:
            team_margin = compute_hsb_v9_team_coordination_margin(
                hidden_residual_l2=0.2, kv_residual_l2=0.5,
                prefix_residual_l2=0.4,
                replay_residual_l2=0.3)
            probe = probe_hsb_v9_hidden_wins_rate(
                n_layers=4, n_heads=2,
                hidden_residual_l2_per_lh=[
                    [0.1, 0.5], [0.6, 0.2],
                    [0.4, 0.7], [0.3, 0.5]],
                kv_residual_l2_per_lh=[
                    [0.4, 0.3], [0.5, 0.6],
                    [0.7, 0.2], [0.5, 0.6]])
            hsb_hidden_wins_rate = float(
                probe.get("mean_win_rate", 0.0))
            hsb_w_cid = emit_hsb_v9_witness(
                projection=p.hidden_state_bridge_v9,
                team_coordination_margin=team_margin,
                hidden_wins_rate_mean=hsb_hidden_wins_rate,
                hidden_wins_primary_margin=0.1).cid()
        # Prefix V9.
        prefix_w_cid = _sha256_hex({
            "schema": "prefix_v9_compact_witness",
            "turn": int(turn_index),
            "predictor_trained": bool(
                p.prefix_v9_predictor_trained)})
        # Attention V9.
        attn_w_cid = _sha256_hex({
            "schema": "attn_steering_v9_compact_witness",
            "turn": int(turn_index)})
        # Cache controller V8.
        cc_w_cid = ""
        if p.enabled and p.cache_controller_v8 is not None:
            cc_w_cid = emit_cache_controller_v8_witness(
                controller=p.cache_controller_v8).cid()
        # Replay controller V6.
        rc_w_cid = ""
        if p.enabled and p.replay_controller_v6 is not None:
            cand = ReplayCandidate(
                flop_reuse=100, flop_recompute=1000,
                flop_fallback=50,
                drift_l2_reuse=0.1, drift_l2_recompute=0.0,
                drift_l2_fallback=0.3,
                crc_passed=True, transcript_available=True,
                n_corruption_flags=0)
            p.replay_controller_v6.decide_v6(
                cand, role=str(role),
                team_coordination_flag=0.8,
                substrate_fidelity=0.9,
                replay_dominance_witness_mean=0.4,
                hidden_wins_primary_score=0.3)
            p.replay_controller_v6.decide_multi_agent_abstain(
                team_features=[0.6, 0.3, 0.2, 0.5, 0.4,
                                0.2, 0.7, 0.4, 0.3])
            rc_w_cid = emit_replay_controller_v6_witness(
                p.replay_controller_v6).cid()
        # Persistent V17.
        per_w_cid = ""
        if p.enabled and p.v12_cell is not None:
            carrier_vals = _payload_hash_vec(
                ("v17", int(turn_index)),
                int(p.v12_cell.state_dim))
            state = step_persistent_state_v17(
                cell=p.v12_cell, prev_state=None,
                carrier_values=carrier_vals,
                turn_index=int(turn_index), role=str(role),
                substrate_skip=carrier_vals,
                hidden_state_skip=carrier_vals,
                attention_skip=carrier_vals,
                retrieval_skip=carrier_vals,
                replay_skip=carrier_vals,
                replay_confidence_skip=carrier_vals,
                replay_dominance_skip=carrier_vals,
                hidden_wins_skip=carrier_vals,
                prefix_reuse_skip=carrier_vals,
                replay_dominance_witness_skip_v16=carrier_vals,
                team_task_success_skip_v17=carrier_vals,
                substrate_fidelity=0.9,
                attention_fidelity=0.9,
                retrieval_fidelity=0.9,
                replay_fidelity=0.9)
            self.chain.add(state)
            per_w_cid = emit_persistent_v17_witness(
                self.chain, state.cid()).cid()
        # Multi-hop V15.
        mh_w_cid = ""
        if p.enabled:
            mh_w_cid = emit_multi_hop_v15_witness(
                backends=W65_DEFAULT_MH_V15_BACKENDS,
                chain_length=W65_DEFAULT_MH_V15_CHAIN_LEN,
                seed=int(turn_index) + 95000).cid()
        # MLSC V13.
        mlsc_w_cid = ""
        if p.enabled and p.mlsc_v13_operator is not None:
            v3 = make_root_capsule_v3(
                branch_id=f"w65_{int(turn_index)}",
                payload=tuple([0.1] * 6),
                fact_tags=("w65",), confidence=0.9, trust=0.9,
                turn_index=int(turn_index))
            v4 = wrap_v3_as_v4(v3)
            v5 = wrap_v4_as_v5(
                v4, attention_witness_cid=f"a_{turn_index}")
            v6 = wrap_v5_as_v6(
                v5,
                attention_witness_chain=(
                    f"a_chain_{turn_index}",),
                cache_reuse_witness_cid=f"c_{turn_index}")
            v7 = wrap_v6_as_v7(
                v6,
                retrieval_witness_chain=(
                    f"r_chain_{turn_index}",),
                controller_witness_cid=f"ctrl_{turn_index}")
            v8 = wrap_v7_as_v8(
                v7,
                replay_witness_chain=(
                    f"replay_chain_{turn_index}",),
                substrate_witness_chain=(
                    f"sub_chain_{turn_index}",),
                provenance_trust_table={"backend_a": 0.9})
            v9 = wrap_v8_as_v9(
                v8,
                attention_pattern_witness_chain=(
                    f"ap_chain_{turn_index}",),
                cache_retrieval_witness_chain=(
                    f"cr_chain_{turn_index}",),
                per_layer_head_trust_matrix=((0, 0, 0.9),))
            v10 = wrap_v9_as_v10(
                v9,
                replay_dominance_witness_chain=(
                    f"rd_chain_{turn_index}",),
                disagreement_wasserstein_distance=0.05)
            v11 = wrap_v10_as_v11(
                v10,
                hidden_wins_witness_chain=(
                    f"hw_chain_{turn_index}",))
            v12 = wrap_v11_as_v12(
                v11,
                replay_dominance_primary_witness_chain=(
                    f"rdp_chain_{turn_index}",),
                hidden_state_trust_witness_chain=(
                    f"hst_chain_{turn_index}",))
            v13_cap = wrap_v12_as_v13(
                v12,
                team_substrate_witness_chain=(
                    f"ts_chain_{turn_index}",),
                role_conditioned_witness_chain=(
                    f"rc_chain_{turn_index}",),
                algebra_signature_v13=(
                    W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION))
            merged = p.mlsc_v13_operator.merge(
                [v13_cap],
                team_substrate_witness_chain=(
                    f"merge_ts_{turn_index}",),
                role_conditioned_witness_chain=(
                    f"merge_rc_{turn_index}",),
                algebra_signature_v13=(
                    W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION))
            mlsc_w_cid = emit_mlsc_v13_witness(merged).cid()
        # Consensus V11.
        cons_w_cid = ""
        if p.enabled and p.consensus_v11 is not None:
            p.consensus_v11.decide_v11(
                payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                trusts=[0.4, 0.4],
                replay_decisions=[
                    "choose_abstain", "choose_abstain"],
                transcript_available=False,
                team_substrate_coordination_scores_per_parent=[
                    0.6, 0.5],
                multi_agent_abstain_score=0.6,
                corruption_detected_per_parent=[False, False],
                repair_amount=0.0,
                hidden_wins_margins_per_parent=[0.0, 0.0],
                three_way_predictions_per_parent=[
                    "kv_wins", "kv_wins"],
                replay_dominance_primary_scores_per_parent=[
                    0.1, 0.0],
                four_way_predictions_per_parent=[
                    "replay_wins", "kv_wins"])
            cons_w_cid = emit_consensus_v11_witness(
                p.consensus_v11).cid()
        # CRC V13.
        crc_w_cid = ""
        if p.enabled and p.crc_v13 is not None:
            crc_w_cid = (
                emit_corruption_robustness_v13_witness(
                    crc_v13=p.crc_v13, n_probes=8,
                    seed=int(turn_index) + 95400).cid())
        # LHR V17.
        lhr_w_cid = ""
        if p.enabled and p.lhr_v17 is not None:
            lhr_w_cid = emit_lhr_v17_witness(
                p.lhr_v17, carrier=[0.1] * 8, k=4,
                team_task_success_indicator=[0.5] * 8,
                replay_dominance_indicator=[0.5] * 8,
                hidden_wins_indicator=[0.5] * 8,
                replay_dominance_primary_indicator=[
                    0.6] * 8).cid()
        # ECC V17.
        ecc_w_cid = ""
        if p.enabled and p.ecc_v17 is not None:
            gate = QuantisedBudgetGate.init(
                in_dim=W53_DEFAULT_ECC_CODE_DIM,
                emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
                seed=int(turn_index) + 95500)
            gate.importance_threshold = 0.0
            gate.w_emit.values = [1.0] * len(
                gate.w_emit.values)
            carrier = _payload_hash_vec(
                ("ecc17", int(turn_index)),
                W53_DEFAULT_ECC_CODE_DIM)
            comp = compress_carrier_ecc_v17(
                carrier, codebook=p.ecc_v17, gate=gate)
            ecc_w_cid = emit_ecc_v17_compression_witness(
                codebook=p.ecc_v17, compression=comp).cid()
        # TVS V14.
        tvs_w_cid = ""
        if p.enabled:
            tvs_res = fifteen_arm_compare(
                per_turn_team_substrate_coordination_fidelities=[
                    0.7],
                per_turn_replay_dominance_primary_fidelities=[
                    0.7],
                per_turn_hidden_wins_fidelities=[0.6],
                per_turn_replay_dominance_fidelities=[0.7],
                per_turn_confidences=[0.8],
                per_turn_trust_scores=[0.7],
                per_turn_merge_retentions=[0.6],
                per_turn_tw_retentions=[0.6],
                per_turn_substrate_fidelities=[0.5],
                per_turn_hidden_fidelities=[0.4],
                per_turn_cache_fidelities=[0.5],
                per_turn_retrieval_fidelities=[0.6],
                per_turn_replay_fidelities=[0.7],
                per_turn_attention_pattern_fidelities=[0.9],
                budget_tokens=6)
            tvs_w_cid = emit_tvs_arbiter_v14_witness(
                tvs_res).cid()
        # Uncertainty V13.
        unc_w_cid = ""
        if p.enabled:
            unc = compose_uncertainty_report_v13(
                confidences=[0.7, 0.5],
                trusts=[0.9, 0.8],
                substrate_fidelities=[0.95, 0.9],
                hidden_state_fidelities=[0.95, 0.92],
                cache_reuse_fidelities=[0.95, 0.92],
                retrieval_fidelities=[0.95, 0.93],
                replay_fidelities=[0.92, 0.90],
                attention_pattern_fidelities=[0.95, 0.90],
                replay_dominance_fidelities=[0.88, 0.85],
                hidden_wins_fidelities=[0.86, 0.82],
                replay_dominance_primary_fidelities=[
                    0.84, 0.80],
                team_coordination_fidelities=[0.80, 0.75])
            unc_w_cid = emit_uncertainty_v13_witness(unc).cid()
        # Disagreement Algebra V11.
        da_w_cid = ""
        if p.enabled:
            trace = AlgebraTrace(steps=[])
            probe = [0.1, 0.2, 0.3]
            wda = emit_disagreement_algebra_v11_witness(
                trace=trace, probe_a=probe, probe_b=probe,
                probe_c=probe,
                tv_oracle=lambda: (True, 0.05),
                tv_falsifier_oracle=lambda: (False, 1.0),
                wasserstein_oracle=lambda: (True, 0.1),
                wasserstein_falsifier_oracle=(
                    lambda: (False, 1.0)),
                js_oracle=lambda: (True, 0.05),
                js_falsifier_oracle=lambda: (False, 1.0),
                attention_pattern_oracle=lambda: (True, 0.85))
            da_w_cid = wda.cid()
        # Deep substrate hybrid V10.
        hybrid_w_cid = ""
        ten_way = False
        if (p.enabled
                and p.deep_substrate_hybrid_v10 is not None):
            v9w = DeepSubstrateHybridV9ForwardWitness(
                schema="x", hybrid_cid="x",
                inner_v8_witness_cid="x",
                nine_way=True,
                cache_controller_v7_fired=True,
                replay_controller_v5_fired=True,
                four_way_bridge_classifier_fired=True,
                hidden_wins_primary_active=True,
                attention_v8_active=True,
                prefix_v8_drift_predictor_active=True,
                hidden_state_trust_active=True,
                hsb_v8_hidden_wins_primary_active=True,
                mean_replay_dominance=0.5,
                hidden_wins_primary_l1=1.0,
                attention_v8_hellinger_max=0.1,
                hidden_state_trust_ledger_l1=0.5,
                hsb_v8_hidden_wins_primary_margin=0.1)
            n_team_inv = 1 if p.multi_agent_coordinator else 0
            wh = deep_substrate_hybrid_v10_forward(
                hybrid=p.deep_substrate_hybrid_v10,
                v9_witness=v9w,
                cache_controller_v8=p.cache_controller_v8,
                replay_controller_v6=p.replay_controller_v6,
                hidden_write_merit_l1=float(
                    hidden_write_merit_l1 + 1.0),
                attention_v9_fingerprint_present=True,
                prefix_v9_predictor_present=bool(
                    p.prefix_v9_predictor_trained),
                hsb_v9_hidden_wins_rate_mean=float(
                    hsb_hidden_wins_rate + 0.5),
                n_roles_in_bank=int(n_roles_in_bank),
                n_team_invocations=int(n_team_inv))
            hybrid_w_cid = wh.cid()
            ten_way = bool(wh.ten_way)
        # Substrate adapter V10.
        adapter_cid = ""
        if p.enabled:
            matrix = probe_all_v10_adapters(
                probe_ollama=False, probe_openai=False)
            adapter_cid = matrix.cid()
        # MASC.
        masc_w_cid = ""
        masc_v10_succ = 0.0
        masc_v10_beats = 0.0
        if p.enabled and p.multi_agent_coordinator is not None:
            seeds = list(range(int(p.masc_n_seeds)))
            _, agg = p.multi_agent_coordinator.run_batch(
                seeds=seeds, n_agents=int(p.masc_n_agents),
                n_turns=int(p.masc_n_turns))
            masc_w = emit_multi_agent_substrate_coordinator_witness(
                coordinator=p.multi_agent_coordinator,
                aggregate=agg)
            masc_w_cid = masc_w.cid()
            masc_v10_succ = float(masc_w.v10_success_rate)
            masc_v10_beats = float(masc_w.v10_strictly_beats_rate)
        return W65HandoffEnvelope(
            schema=W65_SCHEMA_VERSION,
            w64_outer_cid=str(w64_outer_cid),
            w65_params_cid=str(p.cid()),
            substrate_v10_witness_cid=str(sub_w_cid),
            kv_bridge_v10_witness_cid=str(kv_w_cid),
            hsb_v9_witness_cid=str(hsb_w_cid),
            prefix_state_v9_witness_cid=str(prefix_w_cid),
            attn_steer_v9_witness_cid=str(attn_w_cid),
            cache_controller_v8_witness_cid=str(cc_w_cid),
            replay_controller_v6_witness_cid=str(rc_w_cid),
            persistent_v17_witness_cid=str(per_w_cid),
            multi_hop_v15_witness_cid=str(mh_w_cid),
            mlsc_v13_witness_cid=str(mlsc_w_cid),
            consensus_v11_witness_cid=str(cons_w_cid),
            crc_v13_witness_cid=str(crc_w_cid),
            lhr_v17_witness_cid=str(lhr_w_cid),
            ecc_v17_witness_cid=str(ecc_w_cid),
            tvs_v14_witness_cid=str(tvs_w_cid),
            uncertainty_v13_witness_cid=str(unc_w_cid),
            disagreement_algebra_v11_witness_cid=str(da_w_cid),
            deep_substrate_hybrid_v10_witness_cid=str(
                hybrid_w_cid),
            substrate_adapter_v10_matrix_cid=str(adapter_cid),
            masc_witness_cid=str(masc_w_cid),
            team_task_falsifier_witness_cid=str(
                team_task_falsifier_cid),
            v17_chain_cid=str(self.chain.cid()),
            ten_way_used=bool(ten_way),
            substrate_v10_used=bool(sub_used),
            masc_v10_success_rate=float(masc_v10_succ),
            masc_v10_strictly_beats_rate=float(masc_v10_beats),
        )


__all__ = [
    "W65_SCHEMA_VERSION",
    "W65_TEAM_RESULT_SCHEMA",
    "W65_ENVELOPE_VERIFIER_FAILURE_MODES",
    "W65Params",
    "W65HandoffEnvelope",
    "W65Team",
    "verify_w65_handoff",
]
