"""CoordPy — the first finished product from the Context Zero programme.

**CoordPy is a context-capsule runtime.** Every piece of context that
crosses a role boundary, a layer boundary, or a run boundary is a
typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-stamped **capsule** — never a raw prompt string. ``RunSpec``
in, one reproducible, provenance-stamped report out, where the report
is the root of a capsule graph you can audit, replay, and trust.

Positioning (honest)
--------------------
* Context Zero is the *research programme* — 45+ phases, dozens of
  research shards, and the EXTENDED_MATH / PROOFS corpus.
* CoordPy is the *first shipped product surface* from that programme.
  Its load-bearing idea is the **Context Capsule**: the contract
  every inter-role / inter-layer / inter-run artifact satisfies.
* The individual primitives under a capsule (content addressing,
  hash-chained logs, typed claim kinds, frozen lifecycle, capability-
  style typed references) are inherited from Git / Merkle DAGs /
  IPFS / actor systems / session types. What CoordPy claims as new is
  the *unification* — one contract, implemented end-to-end in a
  runnable SDK — and the product-level decision that "context is not
  a prompt; context is an object."

Public surface (stable, contract-tested)
----------------------------------------
* Capsule primitives (SDK v3)
    * ``ContextCapsule``, ``CapsuleKind``, ``CapsuleLifecycle``
    * ``CapsuleBudget``, ``CapsuleLedger``, ``CapsuleView``
    * ``CapsuleAdmissionError``, ``CapsuleLifecycleError``
    * ``CAPSULE_VIEW_SCHEMA``, ``render_view``
    * ``build_report_ledger`` — fold a finished report into a capsule DAG
* Run model (preserved from v2)
    * ``RunSpec``, ``run``
    * ``SweepSpec``, ``run_sweep``, ``HeavyRunNotAcknowledged``
    * ``CoordPyConfig``
    * ``profiles``, ``report``, ``ci_gate``, ``import_data``,
      ``extensions``
* Provenance
    * ``PROVENANCE_SCHEMA``, ``build_manifest``
* Version / schema constants
    * ``__version__``, ``SDK_VERSION``
    * ``PRODUCT_REPORT_SCHEMA``, ``PRODUCT_REPORT_SCHEMA_V1``
    * ``CI_VERDICT_SCHEMA``, ``IMPORT_AUDIT_SCHEMA``
    * ``CAPSULE_VIEW_SCHEMA``

Everything not re-exported here is considered **internal** or
**experimental** and may change without notice. See
``docs/context_zero_master_plan.md`` for the stability matrix.
"""

from __future__ import annotations

from coordpy._version import __version__

from .config import CoordPyConfig
from .provenance import PROVENANCE_SCHEMA, build_manifest
from .run import RunSpec, run

# Public alias for the stable report shape returned by ``run``.
RunReport = dict[str, object]

# Re-export the product modules under stable names. These are part of
# the public SDK contract.
from coordpy._internal.product import profiles, report, ci_gate, import_data

# Make ``from coordpy.profiles import X`` work as a stable submodule
# path (in addition to ``coordpy.profiles.X`` attribute access).
import sys as _sys
for _alias_name, _mod in (
    ("profiles", profiles),
    ("report", report),
    ("ci_gate", ci_gate),
    ("import_data", import_data),
):
    _sys.modules.setdefault(__name__ + "." + _alias_name, _mod)
del _sys, _alias_name, _mod

# Extension system. Kept as a first-class re-export so
# ``from coordpy import extensions`` is the SDK path.
from . import extensions

# Unified runtime (Slice 2).
from .runtime import SweepSpec, run_sweep, HeavyRunNotAcknowledged

# SDK v3.6 — LLM backend abstraction. Strictly additive: when no
# backend is supplied, the runtime instantiates ``LLMClient``
# byte-for-byte unchanged. When a backend is supplied (e.g. an
# ``OpenAICompatibleBackend`` pointed at any OpenAI-compatible
# provider, or an ``MLXDistributedBackend`` whose ``base_url``
# points at an ``mlx_lm.server`` launched under ``mpirun`` across
# two Apple Silicon hosts), the inner-loop calls dispatch through
# the backend without any other change to the spine. The PROMPT /
# LLM_RESPONSE capsules' shape (SHA-256 + length + snippet) is
# preserved regardless of backend.
from .llm_backend import (
    LLMBackend, OllamaBackend, OpenAICompatibleBackend,
    MLXDistributedBackend, make_backend, backend_from_env,
    backend_from_config,
)

# Capsule runtime (Slice 3 — SDK v3). The load-bearing abstraction
# around which the rest of the SDK is centred.
from .capsule import (
    ContextCapsule, CapsuleKind, CapsuleLifecycle, CapsuleBudget,
    CapsuleLedger, CapsuleView, CAPSULE_VIEW_SCHEMA, render_view,
    verify_chain_from_view_dict,
    CapsuleAdmissionError, CapsuleLifecycleError,
    build_report_ledger,
    # Adapters — make the SDK-v3 "everything is a capsule" claim
    # operational for external callers who want to lift their own
    # substrate objects into the capsule surface.
    capsule_from_handle, capsule_from_handoff,
    capsule_from_provenance, capsule_from_sweep_cell,
    capsule_from_sweep_spec, capsule_from_profile,
    capsule_from_readiness, capsule_from_artifact,
    capsule_from_report,
    # Phase-47 cohort subsumption — additive on top of SDK v3.
    capsule_from_cohort, capsule_from_adaptive_sub_table,
    # SDK v3.2 — intra-cell + detached-witness adapters.
    capsule_from_patch_proposal, capsule_from_test_verdict,
    capsule_from_meta_manifest,
    # SDK v3.3 — sub-intra-cell parser-axis adapter.
    capsule_from_parse_outcome, PARSE_OUTCOME_ORACLE,
    # SDK v3.4 — sub-sub-intra-cell prompt + llm response adapters.
    capsule_from_prompt, capsule_from_llm_response,
    PROMPT_TEXT_CAP, LLM_RESPONSE_TEXT_CAP,
)

# Capsule-native runtime (SDK v3.1). The first execution-first
# capsule layer: capsules drive runtime, not just describe it.
from .capsule_runtime import (
    CapsuleNativeRunContext, ContentAddressMismatch,
    seal_and_write_artifact,
    CONSTRUCTION_IN_FLIGHT, CONSTRUCTION_POST_HOC,
    # SDK v3.2 — strong on-disk verification helpers.
    verify_artifacts_on_disk, verify_meta_manifest_on_disk,
)

# SDK v3.3 — runtime-checkable lifecycle audit. Mechanically
# verifies the lifecycle correspondence (W3-32 / W3-32-extended /
# W3-39) on a finished run. Returns OK/BAD/EMPTY plus a list of
# violation counterexamples.
from .lifecycle_audit import (
    CapsuleLifecycleAudit, LifecycleAuditReport,
    audit_capsule_lifecycle, audit_capsule_lifecycle_from_view,
)

# Layered API — three-tier ergonomic surfaces over the same substrate
# (end-user / developer / researcher). Purely additive.
from .api_layers import (
    CoordPySimpleAPI, CoordPyBuilderAPI, CoordPyAdvancedAPI, BuilderSpec,
)

# Stable lightweight agent-team surface. This is the product-facing
# "create a few agents and run them" path; it stays above the
# research-grade team ladder while still sealing a capsule trail.
from .agents import (
    Agent, AgentTurn, TeamResult, AgentTeam, agent, create_team,
)

# Capsule admission policies (Phase 46 research milestone). The
# SDK contract is unchanged — `CapsuleLedger.admit_and_seal` is
# byte-for-byte the same — but external callers can now opt into
# policy-driven admission by wrapping a ledger in a
# `BudgetedAdmissionLedger`. This is *additive*; no Phase-N
# substrate test is affected.
from .capsule_policy import (
    AdmissionPolicy, FIFOPolicy, KindPriorityPolicy,
    SmallestFirstPolicy, LearnedAdmissionPolicy,
    BudgetedAdmissionLedger, train_admission_policy,
    featurise_capsule, feature_index,
)

# Phase-47 bundle-aware admission policies — additive extension
# for Conjecture P46-C1 (bundle-aware admission closes the noise
# ceiling).
from .capsule_policy_bundle import (
    BundleAwarePolicy, CorroboratedAdmissionPolicy,
    PluralityBundlePolicy, BundleLearnedPolicy,
    train_bundle_policy, BundleStats,
    featurise_capsule_with_bundle, bundle_feature_index,
)

# Phase-48 bundle-aware DECODING — additive extension for
# Conjecture P47-C1 (bundle-aware decoding breaks the 0.200
# structural ceiling that admission alone cannot).
from .capsule_decoder import (
    BundleDecoder, PriorityDecoder, PluralityDecoder,
    SourceCorroboratedPriorityDecoder, LearnedBundleDecoder,
    train_learned_bundle_decoder, BUNDLE_DECODER_FEATURES,
    evaluate_decoder, DecoderResult,
)

# Phase-49 stronger bundle-aware decoder + symmetric transfer —
# additive extension for Conjecture W3-C7 (paradigm-shift bar).
# DeepSetBundleDecoder crosses the 0.400 Gate-1 threshold
# (Claim W3-23); MultitaskBundleDecoder achieves 0.350 / 0.350
# on (incident, security) under the shared-head reading
# (Claim W3-22).
from .capsule_decoder_v2 import (
    BUNDLE_DECODER_FEATURES_V2, INTERACTION_FEATURES,
    DEEPSET_PHI_FEATURES,
    LearnedBundleDecoderV2, train_learned_bundle_decoder_v2,
    InteractionBundleDecoder, train_interaction_bundle_decoder,
    MLPBundleDecoder, train_mlp_bundle_decoder,
    DeepSetBundleDecoder, train_deep_set_bundle_decoder,
    MultitaskBundleDecoder, train_multitask_bundle_decoder,
)

# SDK v3.5 — capsule-native multi-agent team coordination *research
# slice*. Strictly additive on v3.4: the run-boundary product
# runtime contract is unchanged. The new surface lives in
# ``vision_mvp.coordpy.team_coord`` and ``team_policy`` and emits
# three new closed-vocabulary capsule kinds (TEAM_HANDOFF,
# ROLE_VIEW, TEAM_DECISION). The team-level lifecycle audit
# (``audit_team_lifecycle``) mechanically verifies invariants
# T-1..T-7 (Theorem W4-1). See
# ``docs/RESULTS_COORDPY_TEAM_COORD.md`` and
# ``docs/CAPSULE_TEAM_FORMALISM.md``.
from .team_coord import (
    RoleBudget, DEFAULT_ROLE_BUDGETS,
    capsule_team_handoff, capsule_role_view, capsule_team_decision,
    AdmissionPolicy as TeamAdmissionPolicy,
    AdmissionDecision as TeamAdmissionDecision,
    FifoAdmissionPolicy as TeamFifoAdmissionPolicy,
    ClaimPriorityAdmissionPolicy as TeamClaimPriorityAdmissionPolicy,
    CoverageGuidedAdmissionPolicy as TeamCoverageGuidedAdmissionPolicy,
    CohortCoherenceAdmissionPolicy as TeamCohortCoherenceAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy
        as TeamCrossRoleCorroborationAdmissionPolicy,
    MultiServiceCorroborationAdmissionPolicy
        as TeamMultiServiceCorroborationAdmissionPolicy,
    TeamCoordinator, audit_team_lifecycle,
    TeamLifecycleAuditReport, T_INVARIANTS,
    # SDK v3.11 — bundle-aware team decoder (W10 family).
    BundleAwareTeamDecoder, decode_admitted_role_view,
    CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE,
    # SDK v3.12 — multi-round bundle-aware team decoder (W11 family).
    MultiRoundBundleDecoder, collect_admitted_handoffs,
    # SDK v3.13 — real-LLM-robust multi-round bundle decoder (W12 family).
    RobustMultiRoundBundleDecoder, CLAIM_KIND_SYNONYMS,
    normalize_claim_kind, normalize_payload, normalize_handoff,
    # SDK v3.14 — layered open-world normaliser + decoder (W13 family).
    HeuristicAbstractionRule, LayeredClaimNormalizer,
    LayeredRobustMultiRoundBundleDecoder,
    LAYERED_NORMALIZER_ABSTAIN,
    # SDK v3.15 — structured producer protocol (W14 family).
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
    ALL_PRODUCER_PROMPT_MODES,
    RoleExtractionSchema, ProducerPromptResult,
    StructuredProducerProtocol,
    INCIDENT_TRIAGE_OBSERVATION_KINDS,
    incident_triage_role_schemas,
    # SDK v3.18 — magnitude-hinted producer protocol (W17 family).
    PRODUCER_PROMPT_MAGNITUDE_HINTED,
    OperationalThreshold,
    INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS,
    incident_triage_magnitude_thresholds,
    # SDK v3.16 — attention-aware capsule context packing (W15 family).
    W15_DEFAULT_TIER_WEIGHT, W15_DEFAULT_CCK_WEIGHT,
    W15_DEFAULT_CORROBORATION_WEIGHT, W15_DEFAULT_MAGNITUDE_WEIGHT,
    W15_DEFAULT_ROUND_WEIGHT,
    W15PackedHandoff, W15PackResult,
    FifoContextPacker, CapsuleContextPacker,
    AttentionAwareBundleDecoder,
    # SDK v3.19 — bundle-relational compatibility disambiguator (W18 family).
    W18CompatibilityResult,
    RelationalCompatibilityDisambiguator,
    # SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator
    # (W19 family).
    BundleContradictionDisambiguator, W19TrustResult,
    W19_ALL_BRANCHES, W19_BRANCH_PRIMARY_TRUSTED, W19_BRANCH_INVERSION,
    W19_BRANCH_CONFOUND_RESOLVED, W19_BRANCH_ABSTAINED_NO_SIGNAL,
    W19_BRANCH_ABSTAINED_SYMMETRIC, W19_BRANCH_DISABLED,
    # SDK v3.21 — outside-witness acquisition disambiguator (W20 family).
    OutsideWitnessOracle, OutsideQuery, OutsideVerdict,
    ServiceGraphOracle, CompromisedServiceGraphOracle,
    AbstainingOracle, LLMAdjudicatorOracle,
    build_incident_triage_service_graph,
    OutsideWitnessAcquisitionDisambiguator, W20OutsideResult,
    W20_ALL_BRANCHES, W20_BRANCH_OUTSIDE_RESOLVED,
    W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC,
    W20_BRANCH_OUTSIDE_ABSTAINED, W20_BRANCH_NO_TRIGGER,
    W20_BRANCH_DISABLED, W20_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.22 — trust-weighted multi-oracle adjudicator (W21 family).
    OracleRegistration, ChangeHistoryOracle, OnCallNotesOracle,
    SingletonAsymmetricOracle, DisagreeingHonestOracle,
    W21OracleProbe, W21MultiOracleResult,
    TrustWeightedMultiOracleDisambiguator,
    W21_ALL_BRANCHES, W21_BRANCH_QUORUM_RESOLVED,
    W21_BRANCH_NO_QUORUM, W21_BRANCH_SYMMETRIC_QUORUM,
    W21_BRANCH_NO_ORACLES, W21_BRANCH_NO_TRIGGER,
    W21_BRANCH_DISABLED, W21_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.23 — capsule + audited latent-state-sharing hybrid (W22 family).
    SchemaCapsule, build_incident_triage_schema_capsule,
    LatentDigestEnvelope, LatentVerificationOutcome,
    verify_latent_digest, SharedReadCache, CachingOracleAdapter,
    EnvelopeTamperer, W22LatentResult, LatentDigestDisambiguator,
    W22_ALL_BRANCHES, W22_BRANCH_LATENT_RESOLVED,
    W22_BRANCH_LATENT_REJECTED, W22_BRANCH_NO_TRIGGER,
    W22_BRANCH_NO_SCHEMA, W22_BRANCH_DISABLED,
    W22_BRANCH_ABSTAIN_PASSTHROUGH,
    W22_DEFAULT_TRIGGER_BRANCHES,
    W22_LATENT_ENVELOPE_SCHEMA_VERSION,
    # SDK v3.26 — shared-fanout dense-control + cross-agent state reuse (W25).
    FanoutEnvelope, SharedFanoutRegistry, SharedFanoutDisambiguator,
    verify_fanout, W25FanoutResult,
    W25_FANOUT_SCHEMA_VERSION,
    W25_ALL_BRANCHES, W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    W25_BRANCH_FANOUT_CONSUMER_REJECTED,
    W25_BRANCH_NO_TRIGGER, W25_BRANCH_DISABLED,
    W25_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.27 — chain-persisted dense-control fanout +
    # per-consumer projections (W26 family).
    ChainAnchorEnvelope, ChainAdvanceEnvelope,
    ChainPersistedFanoutRegistry,
    ChainPersistedFanoutDisambiguator,
    ProjectionSlot, W26ChainResult,
    verify_chain_anchor, verify_chain_advance,
    verify_projection_subscription,
    W26_CHAIN_ANCHOR_SCHEMA_VERSION,
    W26_CHAIN_ADVANCE_SCHEMA_VERSION,
    W26_ALL_BRANCHES,
    W26_BRANCH_CHAIN_ANCHORED, W26_BRANCH_CHAIN_ADVANCED,
    W26_BRANCH_CHAIN_REJECTED, W26_BRANCH_CHAIN_RE_ANCHORED,
    W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
    W26_BRANCH_CHAIN_PROJECTION_REJECTED,
    W26_BRANCH_NO_TRIGGER, W26_BRANCH_DISABLED,
    W26_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.28 — multi-chain salience-keyed dense-control fanout +
    # per-signature scoping (W27 family).
    SalienceSignatureEnvelope, ChainPivotEnvelope,
    MultiChainPersistedFanoutRegistry,
    MultiChainPersistedFanoutDisambiguator,
    MultiChainPersistedFanoutOrchestrator,
    SharedMultiChainPool,
    W27MultiChainResult, W27OrchestratorResult,
    compute_input_signature_cid,
    verify_salience_signature, verify_chain_pivot,
    W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
    W27_CHAIN_PIVOT_SCHEMA_VERSION,
    W27_ALL_BRANCHES,
    W27_BRANCH_PIVOTED, W27_BRANCH_ANCHORED_NEW,
    W27_BRANCH_POOL_EXHAUSTED, W27_BRANCH_PIVOT_REJECTED,
    W27_BRANCH_FALLBACK_W26, W27_BRANCH_NO_TRIGGER,
    W27_BRANCH_DISABLED,
    W27_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.29 — ensemble-verified cross-model multi-chain pivot
    # ratification (W28 family). EXPERIMENTAL — see __experimental__.
    ProbeVote, EnsembleProbe, EnsembleProbeRegistration,
    DeterministicSignatureProbe, OracleConsultationProbe,
    LLMSignatureProbe,
    EnsemblePivotRatificationEnvelope,
    EnsembleRatificationRegistry,
    EnsembleVerifiedMultiChainOrchestrator,
    W28EnsembleResult,
    verify_ensemble_pivot_ratification,
    build_default_ensemble_registry,
    build_two_probe_oracle_ensemble_registry,
    build_cross_host_llm_ensemble_registry,
    W28_RATIFICATION_SCHEMA_VERSION,
    W28_ALL_BRANCHES,
    W28_BRANCH_RATIFIED, W28_BRANCH_RATIFIED_PASSTHROUGH,
    W28_BRANCH_QUORUM_BELOW_THRESHOLD, W28_BRANCH_PROBE_REJECTED,
    W28_BRANCH_NO_RATIFY_NEEDED, W28_BRANCH_FALLBACK_W27,
    W28_BRANCH_NO_TRIGGER, W28_BRANCH_DISABLED,
    W28_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.30 — geometry-partitioned product-manifold dense control +
    # audited subspace-basis payload + factoradic routing index +
    # causal-validity gate + cross-host variance witness (W29 family).
    # EXPERIMENTAL — see __experimental__.
    SubspaceBasis, verify_subspace_basis,
    compute_structural_subspace_basis,
    encode_permutation_to_factoradic,
    decode_factoradic_to_permutation,
    CrossHostVarianceWitness,
    GeometryPartitionedRatificationEnvelope,
    PartitionRegistration,
    GeometryPartitionRegistry,
    W29PartitionResult,
    GeometryPartitionedOrchestrator,
    classify_partition_id_for_cell,
    verify_geometry_partition_ratification,
    build_trivial_partition_registry,
    build_three_partition_registry,
    W29_PARTITION_SCHEMA_VERSION,
    W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
    W29_REGISTERED_PARTITION_IDS, W29_PARTITION_LABEL,
    W29_DEFAULT_ORTHOGONALITY_TOL,
    W29_BRANCH_PARTITION_RESOLVED,
    W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH,
    W29_BRANCH_PARTITION_REJECTED,
    W29_BRANCH_PARTITION_BELOW_THRESHOLD,
    W29_BRANCH_CROSS_HOST_VARIANCE_WITNESSED,
    W29_BRANCH_NO_PARTITION_NEEDED,
    W29_BRANCH_FALLBACK_W28,
    W29_BRANCH_NO_TRIGGER, W29_BRANCH_DISABLED,
    W29_ALL_BRANCHES,
    W29_DEFAULT_TRIGGER_BRANCHES,
    # SDK v3.31 — calibrated geometry-aware dense control + multi-stride
    # basis history + per-partition calibration prior + cross-host
    # disagreement-routing + ancestor-chain causal binding (W30 family).
    # EXPERIMENTAL — see __experimental__.
    BasisHistory, AncestorChain, PartitionCalibrationVector,
    CalibratedGeometryRatificationEnvelope,
    CalibratedGeometryRegistry,
    W30CalibratedResult,
    CalibratedGeometryOrchestrator,
    verify_calibrated_geometry_ratification,
    update_partition_calibration_running_mean,
    build_trivial_calibrated_registry,
    build_calibrated_registry,
    W30_CALIBRATED_SCHEMA_VERSION,
    W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD,
    W30_BRANCH_CALIBRATED_RESOLVED,
    W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH,
    W30_BRANCH_CALIBRATED_REJECTED,
    W30_BRANCH_DISAGREEMENT_ROUTED,
    W30_BRANCH_CALIBRATION_REROUTED,
    W30_BRANCH_NO_CALIBRATION_NEEDED,
    W30_BRANCH_FALLBACK_W29,
    W30_BRANCH_NO_TRIGGER,
    W30_BRANCH_DISABLED,
    W30_ALL_BRANCHES,
    # SDK v3.32 — online self-calibrated geometry-aware dense control +
    # sealed prior trajectory + adaptive threshold + W31 manifest CID
    # (W31 family).  EXPERIMENTAL — see __experimental__.
    PriorTrajectoryEntry,
    OnlineCalibratedRatificationEnvelope,
    OnlineCalibratedRegistry,
    W31OnlineResult,
    OnlineCalibratedOrchestrator,
    verify_online_calibrated_ratification,
    derive_per_cell_agreement_signal,
    compute_adaptive_threshold,
    build_trivial_online_registry,
    build_online_calibrated_registry,
    W31_ONLINE_SCHEMA_VERSION,
    W31_DEFAULT_THRESHOLD_MIN,
    W31_DEFAULT_THRESHOLD_MAX,
    W31_DEFAULT_TRAJECTORY_WINDOW,
    W31_BRANCH_ONLINE_RESOLVED,
    W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH,
    W31_BRANCH_ONLINE_REJECTED,
    W31_BRANCH_ONLINE_DISABLED,
    W31_BRANCH_ONLINE_NO_TRIGGER,
    W31_ALL_BRANCHES,
    # SDK v3.33 — long-window convergent online geometry-aware
    # dense control + EWMA prior accumulator + Page CUSUM
    # change-point detector + gold-correlated disagreement-routing
    # + W32 manifest-v2 CID (W32 family).  EXPERIMENTAL — see
    # __experimental__.
    GoldCorrelationMap, build_gold_correlation_map,
    ConvergenceStateEntry,
    LongWindowConvergentRatificationEnvelope,
    LongWindowConvergentRegistry,
    W32LongWindowResult,
    LongWindowConvergentOrchestrator,
    verify_long_window_convergent_ratification,
    update_ewma_prior, update_cusum_two_sided, detect_change_point,
    build_trivial_long_window_registry,
    build_long_window_convergent_registry,
    W32_LONG_WINDOW_SCHEMA_VERSION,
    W32_DEFAULT_EWMA_ALPHA,
    W32_DEFAULT_CUSUM_THRESHOLD,
    W32_DEFAULT_CUSUM_K,
    W32_DEFAULT_CUSUM_MAX,
    W32_DEFAULT_LONG_WINDOW,
    W32_DEFAULT_GOLD_CORRELATION_MIN,
    W32_BRANCH_LONG_WINDOW_RESOLVED,
    W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH,
    W32_BRANCH_LONG_WINDOW_REJECTED,
    W32_BRANCH_LONG_WINDOW_DISABLED,
    W32_BRANCH_LONG_WINDOW_NO_TRIGGER,
    W32_BRANCH_GOLD_CORRELATED_REROUTED,
    W32_BRANCH_CHANGE_POINT_RESET,
    W32_ALL_BRANCHES,
    # SDK v3.34 — Trust-EWMA-tracked multi-oracle adjudication (W33).
    # EXPERIMENTAL — see __experimental__.
    TrustTrajectoryEntry,
    TrustEWMARatificationEnvelope,
    TrustEWMARegistry,
    W33TrustEWMAResult,
    TrustEWMATrackedMultiOracleOrchestrator,
    verify_trust_ewma_ratification,
    derive_per_oracle_agreement_signal,
    build_trivial_trust_ewma_registry,
    build_trust_ewma_registry,
    W33_TRUST_EWMA_SCHEMA_VERSION,
    W33_DEFAULT_TRUST_THRESHOLD,
    W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
    W33_DEFAULT_EWMA_ALPHA,
    W33_BRANCH_TRUST_EWMA_RESOLVED,
    W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH,
    W33_BRANCH_TRUST_EWMA_REJECTED,
    W33_BRANCH_TRUST_EWMA_DISABLED,
    W33_BRANCH_TRUST_EWMA_NO_TRIGGER,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE,
    W33_ALL_BRANCHES,
    # SDK v3.35 — Live-aware multi-anchor adjudication + native-latent
    # audited response-feature proxy + W34 manifest-v4 CID (W34).
    # EXPERIMENTAL — see __experimental__.
    LiveOracleAttestation,
    LiveAwareMultiAnchorRatificationEnvelope,
    LiveAwareMultiAnchorRegistry,
    HostRegistration,
    W34LiveAwareResult,
    LiveAwareMultiAnchorOrchestrator,
    verify_live_aware_multi_anchor_ratification,
    derive_multi_anchor_consensus_reference,
    compute_response_feature_signature,
    apply_host_decay,
    build_trivial_live_aware_registry,
    build_live_aware_registry,
    W34_LIVE_AWARE_SCHEMA_VERSION,
    W34_DEFAULT_ANCHOR_QUORUM_MIN,
    W34_DEFAULT_HOST_DECAY_FACTOR,
    W34_DEFAULT_LIVE_ATTESTATION_TIMEOUT_MS_BUCKET,
    W34_BRANCH_LIVE_AWARE_RESOLVED,
    W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH,
    W34_BRANCH_LIVE_AWARE_REJECTED,
    W34_BRANCH_LIVE_AWARE_DISABLED,
    W34_BRANCH_LIVE_AWARE_NO_TRIGGER,
    W34_BRANCH_MULTI_ANCHOR_CONSENSUS,
    W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS,
    W34_BRANCH_HOST_DECAY_FIRED,
    W34_ALL_BRANCHES,
    # SDK v3.36 — Trust-subspace dense-control proxy + basis-history
    # projection + manifest-v5 CID (W35). EXPERIMENTAL — see
    # __experimental__.
    TrustSubspaceBasisEntry,
    TrustSubspaceDenseRatificationEnvelope,
    TrustSubspaceDenseRegistry,
    W35TrustSubspaceResult,
    TrustSubspaceDenseControlOrchestrator,
    verify_trust_subspace_dense_ratification,
    select_trust_subspace_projection,
    build_trivial_trust_subspace_registry,
    build_trust_subspace_dense_registry,
    W35_TRUST_SUBSPACE_SCHEMA_VERSION,
    W35_DEFAULT_BASIS_EWMA_ALPHA,
    W35_DEFAULT_PROJECTION_THRESHOLD,
    W35_DEFAULT_PROJECTION_MARGIN_MIN,
    W35_DEFAULT_BASIS_HISTORY_WINDOW,
    W35_DEFAULT_MIN_BASIS_OBSERVATIONS,
    W35_BRANCH_TRUST_SUBSPACE_RESOLVED,
    W35_BRANCH_TRIVIAL_TRUST_SUBSPACE_PASSTHROUGH,
    W35_BRANCH_TRUST_SUBSPACE_REJECTED,
    W35_BRANCH_TRUST_SUBSPACE_DISABLED,
    W35_BRANCH_TRUST_SUBSPACE_NO_TRIGGER,
    W35_BRANCH_BASIS_HISTORY_REROUTED,
    W35_BRANCH_BASIS_HISTORY_UNSAFE,
    W35_BRANCH_BASIS_HISTORY_ABSTAINED,
    W35_ALL_BRANCHES,
    # SDK v3.37 — host-diverse trust-subspace dense-control guard +
    # manifest-v6 CID (W36). EXPERIMENTAL — see __experimental__.
    HostDiverseBasisEntry,
    HostDiverseRatificationEnvelope,
    HostDiverseRegistry,
    W36HostDiverseResult,
    HostDiverseTrustSubspaceOrchestrator,
    verify_host_diverse_ratification,
    select_host_diverse_projection,
    build_trivial_host_diverse_registry,
    build_host_diverse_registry,
    W36_HOST_DIVERSE_SCHEMA_VERSION,
    W36_DEFAULT_MIN_DISTINCT_HOSTS,
    W36_DEFAULT_HOST_DIVERSITY_THRESHOLD,
    W36_DEFAULT_HOST_DIVERSITY_MARGIN_MIN,
    W36_BRANCH_HOST_DIVERSE_RESOLVED,
    W36_BRANCH_TRIVIAL_HOST_DIVERSE_PASSTHROUGH,
    W36_BRANCH_HOST_DIVERSE_REJECTED,
    W36_BRANCH_HOST_DIVERSE_DISABLED,
    W36_BRANCH_HOST_DIVERSE_NO_TRIGGER,
    W36_BRANCH_HOST_DIVERSE_REROUTED,
    W36_BRANCH_HOST_DIVERSE_UNSAFE,
    W36_BRANCH_HOST_DIVERSE_ABSTAINED,
    W36_ALL_BRANCHES,
    # SDK v3.38 — anchor-cross-host basis-trajectory ratification +
    # manifest-v7 CID (W37). EXPERIMENTAL — see __experimental__.
    CrossHostBasisTrajectoryEntry,
    CrossHostBasisTrajectoryRatificationEnvelope,
    CrossHostBasisTrajectoryRegistry,
    W37CrossHostTrajectoryResult,
    CrossHostBasisTrajectoryOrchestrator,
    verify_cross_host_trajectory_ratification,
    select_cross_host_trajectory_projection,
    build_trivial_cross_host_trajectory_registry,
    build_cross_host_trajectory_registry,
    W37_CROSS_HOST_TRAJECTORY_SCHEMA_VERSION,
    W37_DEFAULT_TRAJECTORY_EWMA_ALPHA,
    W37_DEFAULT_TRAJECTORY_THRESHOLD,
    W37_DEFAULT_TRAJECTORY_MARGIN_MIN,
    W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS,
    W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS,
    W37_DEFAULT_TRAJECTORY_HISTORY_WINDOW,
    W37_BRANCH_TRAJECTORY_RESOLVED,
    W37_BRANCH_TRIVIAL_TRAJECTORY_PASSTHROUGH,
    W37_BRANCH_TRAJECTORY_REJECTED,
    W37_BRANCH_TRAJECTORY_DISABLED,
    W37_BRANCH_TRAJECTORY_NO_TRIGGER,
    W37_BRANCH_TRAJECTORY_REROUTED,
    W37_BRANCH_TRAJECTORY_UNSAFE,
    W37_BRANCH_TRAJECTORY_ABSTAINED,
    W37_BRANCH_TRAJECTORY_NO_HISTORY,
    W37_BRANCH_TRAJECTORY_DISAGREEMENT,
    W37_BRANCH_TRAJECTORY_POISONED,
    W37_ALL_BRANCHES,
    # SDK v3.39 — disjoint cross-source consensus-reference trajectory-
    # divergence adjudication + manifest-v8 CID (W38). EXPERIMENTAL —
    # see __experimental__.
    ConsensusReferenceProbe,
    DisjointConsensusReferenceRatificationEnvelope,
    DisjointConsensusReferenceRegistry,
    W38DisjointConsensusReferenceResult,
    DisjointConsensusReferenceOrchestrator,
    DisjointTopologyError,
    verify_disjoint_consensus_reference_ratification,
    select_disjoint_consensus_divergence,
    build_trivial_disjoint_consensus_registry,
    build_disjoint_consensus_registry,
    W38_DISJOINT_CONSENSUS_SCHEMA_VERSION,
    W38_DEFAULT_CONSENSUS_STRENGTH_MIN,
    W38_DEFAULT_DIVERGENCE_MARGIN_MIN,
    W38_BRANCH_CONSENSUS_RESOLVED,
    W38_BRANCH_TRIVIAL_CONSENSUS_PASSTHROUGH,
    W38_BRANCH_CONSENSUS_REJECTED,
    W38_BRANCH_CONSENSUS_DISABLED,
    W38_BRANCH_CONSENSUS_NO_TRIGGER,
    W38_BRANCH_CONSENSUS_RATIFIED,
    W38_BRANCH_CONSENSUS_NO_REFERENCE,
    W38_BRANCH_CONSENSUS_DIVERGENCE_ABSTAINED,
    W38_BRANCH_CONSENSUS_REFERENCE_WEAK,
    W38_ALL_BRANCHES,
    # SDK v3.40 — multi-host disjoint quorum consensus-reference
    # ratification + manifest-v9 CID + mutually-disjoint physical-host
    # topology (W39). EXPERIMENTAL — see __experimental__.
    MultiHostDisjointQuorumProbe,
    MultiHostDisjointQuorumRatificationEnvelope,
    MultiHostDisjointQuorumRegistry,
    W39MultiHostDisjointQuorumResult,
    MultiHostDisjointQuorumOrchestrator,
    MutuallyDisjointTopologyError,
    verify_multi_host_disjoint_quorum_ratification,
    select_multi_host_disjoint_quorum_decision,
    build_trivial_multi_host_disjoint_quorum_registry,
    build_multi_host_disjoint_quorum_registry,
    W39_MULTI_HOST_DISJOINT_QUORUM_SCHEMA_VERSION,
    W39_DEFAULT_QUORUM_MIN,
    W39_DEFAULT_MIN_QUORUM_PROBES,
    W39_DEFAULT_QUORUM_STRENGTH_MIN,
    W39_DEFAULT_QUORUM_DIVERGENCE_MARGIN_MIN,
    W39_BRANCH_QUORUM_RESOLVED,
    W39_BRANCH_TRIVIAL_QUORUM_PASSTHROUGH,
    W39_BRANCH_QUORUM_REJECTED,
    W39_BRANCH_QUORUM_DISABLED,
    W39_BRANCH_QUORUM_NO_TRIGGER,
    W39_BRANCH_QUORUM_RATIFIED,
    W39_BRANCH_QUORUM_DIVERGENCE_ABSTAINED,
    W39_BRANCH_QUORUM_NO_REFERENCES,
    W39_BRANCH_QUORUM_INSUFFICIENT,
    W39_BRANCH_QUORUM_SPLIT,
    W39_BRANCH_QUORUM_REFERENCE_WEAK,
    W39_ALL_BRANCHES,
    # SDK v3.41 — cross-host response-signature heterogeneity
    # ratification + manifest-v10 CID + cross-host response-text
    # Jaccard divergence guard (W40). EXPERIMENTAL — see
    # __experimental__.
    ResponseSignatureProbe,
    MultiHostResponseHeterogeneityProbe,
    CrossHostResponseHeterogeneityRatificationEnvelope,
    CrossHostResponseHeterogeneityRegistry,
    W40CrossHostResponseHeterogeneityResult,
    CrossHostResponseHeterogeneityOrchestrator,
    verify_cross_host_response_heterogeneity_ratification,
    select_cross_host_response_heterogeneity_decision,
    build_trivial_cross_host_response_heterogeneity_registry,
    build_cross_host_response_heterogeneity_registry,
    W40_RESPONSE_HETEROGENEITY_SCHEMA_VERSION,
    W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN,
    W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES,
    W40_BRANCH_RESPONSE_SIGNATURE_RESOLVED,
    W40_BRANCH_TRIVIAL_RESPONSE_SIGNATURE_PASSTHROUGH,
    W40_BRANCH_RESPONSE_SIGNATURE_REJECTED,
    W40_BRANCH_RESPONSE_SIGNATURE_DISABLED,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER,
    W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE,
    W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED,
    W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES,
    W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT,
    W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE,
    W40_ALL_BRANCHES,
)
# W41 family - integrated multi-agent context synthesis (SDK v3.42).
# Strict superset of W40: composes the strongest old-line explicit-
# capsule trust adjudication chain (W21..W40) and the strongest
# cross-role / multi-round bundle decoder family (W7..W11) into a
# single auditable end-to-end path with one manifest-v11 envelope
# binding both axes plus a content-addressed cross-axis witness.
from .integrated_synthesis import (
    IntegratedSynthesisRatificationEnvelope,
    IntegratedSynthesisRegistry,
    W41IntegratedSynthesisResult,
    IntegratedSynthesisOrchestrator,
    verify_integrated_synthesis_ratification,
    select_integrated_synthesis_decision,
    classify_producer_axis_branch,
    classify_trust_axis_branch,
    build_integrated_synthesis_registry,
    build_trivial_integrated_synthesis_registry,
    W41_INTEGRATED_SYNTHESIS_SCHEMA_VERSION,
    W41_PRODUCER_AXIS_FIRED,
    W41_PRODUCER_AXIS_NO_TRIGGER,
    W41_TRUST_AXIS_RATIFIED,
    W41_TRUST_AXIS_ABSTAINED,
    W41_TRUST_AXIS_NO_TRIGGER,
    W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH,
    W41_BRANCH_INTEGRATED_DISABLED,
    W41_BRANCH_INTEGRATED_REJECTED,
    W41_BRANCH_INTEGRATED_PRODUCER_ONLY,
    W41_BRANCH_INTEGRATED_TRUST_ONLY,
    W41_BRANCH_INTEGRATED_BOTH_AXES,
    W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED,
    W41_BRANCH_INTEGRATED_NEITHER_AXIS,
    W41_ALL_BRANCHES,
    W41_PRODUCER_BRANCHES,
    W41_TRUST_BRANCHES,
)
# SDK v3.43 (W42) — cross-role-invariant synthesis +
# manifest-v12 CID + role-handoff-signature axis +
# composite-collusion bounding (third orthogonal evidence axis
# on top of W41 producer x trust integration).
from .role_invariant_synthesis import (
    RoleInvariantSynthesisRatificationEnvelope,
    RoleInvariancePolicyEntry,
    RoleInvariancePolicyRegistry,
    RoleInvariantSynthesisRegistry,
    W42RoleInvariantResult,
    RoleInvariantSynthesisOrchestrator,
    verify_role_invariant_synthesis_ratification,
    select_role_invariance_decision,
    compute_role_handoff_signature_cid,
    build_role_invariant_registry,
    build_trivial_role_invariant_registry,
    W42_ROLE_INVARIANT_SCHEMA_VERSION,
    W42_BRANCH_TRIVIAL_INVARIANCE_PASSTHROUGH,
    W42_BRANCH_INVARIANCE_DISABLED,
    W42_BRANCH_INVARIANCE_REJECTED,
    W42_BRANCH_INVARIANCE_NO_TRIGGER,
    W42_BRANCH_INVARIANCE_RATIFIED,
    W42_BRANCH_INVARIANCE_DIVERGED_ABSTAINED,
    W42_BRANCH_INVARIANCE_NO_POLICY,
    W42_ALL_BRANCHES,
)
from .team_policy import (
    LearnedTeamAdmissionPolicy,
    TrainSample as TeamTrainSample,
    TrainStats as TeamTrainStats,
    train_team_admission_policy,
    featurise_team_handoff,
    KNOWN_SOURCE_ROLES, KNOWN_CLAIM_KINDS,
    N_FEATURES as TEAM_FEATURE_DIM,
    FEATURE_NAMES as TEAM_FEATURE_NAMES,
)


SDK_VERSION = "coordpy.sdk.v3.43"
PRODUCT_REPORT_SCHEMA = "phase45.product_report.v2"
# Legacy schema — still emitted by mock-only runs that don't touch
# the unified runtime path. Consumers should accept both.
PRODUCT_REPORT_SCHEMA_V1 = "phase45.product_report.v1"
CI_VERDICT_SCHEMA = "phase46.ci_verdict.v1"
IMPORT_AUDIT_SCHEMA = "phase46.import_audit.v1"

# SDK v3.30 (W29) — explicit stable-vs-experimental boundary. The
# symbols enumerated below are part of the *research-grade* dense-
# control / multi-agent-coordination surface and may evolve between
# minor versions. The stable runtime contract (RunSpec → run report,
# capsule primitives, lifecycle audit) is NOT in this tuple. External
# callers depending on these symbols should pin a specific SDK version
# and watch the CHANGELOG for breaking changes.
__experimental__: tuple[str, ...] = (
    # W22 family — capsule + audited latent-state-sharing hybrid.
    "SchemaCapsule", "LatentDigestEnvelope", "LatentDigestDisambiguator",
    "verify_latent_digest", "SharedReadCache", "CachingOracleAdapter",
    "EnvelopeTamperer",
    # W25 family — shared-fanout dense-control.
    "FanoutEnvelope", "SharedFanoutRegistry", "SharedFanoutDisambiguator",
    "verify_fanout",
    # W26 family — chain-persisted dense-control fanout.
    "ChainAnchorEnvelope", "ChainAdvanceEnvelope",
    "ChainPersistedFanoutRegistry", "ChainPersistedFanoutDisambiguator",
    "ProjectionSlot",
    "verify_chain_anchor", "verify_chain_advance",
    "verify_projection_subscription",
    # W27 family — multi-chain salience-keyed dense-control fanout.
    "SalienceSignatureEnvelope", "ChainPivotEnvelope",
    "MultiChainPersistedFanoutRegistry",
    "MultiChainPersistedFanoutDisambiguator",
    "MultiChainPersistedFanoutOrchestrator",
    "SharedMultiChainPool",
    "verify_salience_signature", "verify_chain_pivot",
    "compute_input_signature_cid",
    # W28 family — ensemble-verified cross-model pivot ratification.
    "ProbeVote", "EnsembleProbe", "EnsembleProbeRegistration",
    "DeterministicSignatureProbe", "OracleConsultationProbe",
    "LLMSignatureProbe",
    "EnsemblePivotRatificationEnvelope",
    "EnsembleRatificationRegistry",
    "EnsembleVerifiedMultiChainOrchestrator",
    "verify_ensemble_pivot_ratification",
    "build_default_ensemble_registry",
    "build_two_probe_oracle_ensemble_registry",
    "build_cross_host_llm_ensemble_registry",
    # W29 family — geometry-partitioned product-manifold dense control +
    # audited subspace-basis payload + factoradic routing index +
    # causal-validity gate + cross-host variance witness.
    "SubspaceBasis", "verify_subspace_basis",
    "compute_structural_subspace_basis",
    "encode_permutation_to_factoradic",
    "decode_factoradic_to_permutation",
    "CrossHostVarianceWitness",
    "GeometryPartitionedRatificationEnvelope",
    "PartitionRegistration",
    "GeometryPartitionRegistry",
    "GeometryPartitionedOrchestrator",
    "classify_partition_id_for_cell",
    "verify_geometry_partition_ratification",
    "build_trivial_partition_registry",
    "build_three_partition_registry",
    # W30 family — calibrated geometry-aware dense control + multi-stride
    # basis history + per-partition calibration prior + cross-host
    # disagreement-routing + ancestor-chain causal binding.
    "BasisHistory", "AncestorChain", "PartitionCalibrationVector",
    "CalibratedGeometryRatificationEnvelope",
    "CalibratedGeometryRegistry",
    "W30CalibratedResult",
    "CalibratedGeometryOrchestrator",
    "verify_calibrated_geometry_ratification",
    "update_partition_calibration_running_mean",
    "build_trivial_calibrated_registry",
    "build_calibrated_registry",
    # W31 family — online self-calibrated geometry-aware dense control +
    # sealed prior trajectory + adaptive threshold + W31 manifest CID.
    "PriorTrajectoryEntry",
    "OnlineCalibratedRatificationEnvelope",
    "OnlineCalibratedRegistry",
    "W31OnlineResult",
    "OnlineCalibratedOrchestrator",
    "verify_online_calibrated_ratification",
    "derive_per_cell_agreement_signal",
    "compute_adaptive_threshold",
    "build_trivial_online_registry",
    "build_online_calibrated_registry",
    # W32 family — long-window convergent online geometry-aware
    # dense control + EWMA + Page CUSUM + gold-correlated routing
    # + manifest-v2 CID.
    "GoldCorrelationMap",
    "build_gold_correlation_map",
    "ConvergenceStateEntry",
    "LongWindowConvergentRatificationEnvelope",
    "LongWindowConvergentRegistry",
    "W32LongWindowResult",
    "LongWindowConvergentOrchestrator",
    "verify_long_window_convergent_ratification",
    "update_ewma_prior",
    "update_cusum_two_sided",
    "detect_change_point",
    "build_trivial_long_window_registry",
    "build_long_window_convergent_registry",
    # W33 family — trust-EWMA-tracked multi-oracle adjudication.
    "TrustTrajectoryEntry",
    "TrustEWMARatificationEnvelope",
    "TrustEWMARegistry",
    "W33TrustEWMAResult",
    "TrustEWMATrackedMultiOracleOrchestrator",
    "verify_trust_ewma_ratification",
    "derive_per_oracle_agreement_signal",
    "build_trivial_trust_ewma_registry",
    "build_trust_ewma_registry",
    # W34 family — live-aware multi-anchor adjudication + native-latent
    # audited response-feature proxy + W34 manifest-v4 CID.
    "LiveOracleAttestation",
    "LiveAwareMultiAnchorRatificationEnvelope",
    "LiveAwareMultiAnchorRegistry",
    "HostRegistration",
    "W34LiveAwareResult",
    "LiveAwareMultiAnchorOrchestrator",
    "verify_live_aware_multi_anchor_ratification",
    "derive_multi_anchor_consensus_reference",
    "compute_response_feature_signature",
    "apply_host_decay",
    "build_trivial_live_aware_registry",
    "build_live_aware_registry",
    # W35 family — trust-subspace dense-control proxy + basis-history
    # projection + manifest-v5 CID.
    "TrustSubspaceBasisEntry",
    "TrustSubspaceDenseRatificationEnvelope",
    "TrustSubspaceDenseRegistry",
    "W35TrustSubspaceResult",
    "TrustSubspaceDenseControlOrchestrator",
    "verify_trust_subspace_dense_ratification",
    "select_trust_subspace_projection",
    "build_trivial_trust_subspace_registry",
    "build_trust_subspace_dense_registry",
    # W36 family — host-diverse trust-subspace dense-control guard +
    # manifest-v6 CID.
    "HostDiverseBasisEntry",
    "HostDiverseRatificationEnvelope",
    "HostDiverseRegistry",
    "W36HostDiverseResult",
    "HostDiverseTrustSubspaceOrchestrator",
    "verify_host_diverse_ratification",
    "select_host_diverse_projection",
    "build_trivial_host_diverse_registry",
    "build_host_diverse_registry",
    # W37 family — anchor-cross-host basis-trajectory ratification +
    # manifest-v7 CID.
    "CrossHostBasisTrajectoryEntry",
    "CrossHostBasisTrajectoryRatificationEnvelope",
    "CrossHostBasisTrajectoryRegistry",
    "W37CrossHostTrajectoryResult",
    "CrossHostBasisTrajectoryOrchestrator",
    "verify_cross_host_trajectory_ratification",
    "select_cross_host_trajectory_projection",
    "build_trivial_cross_host_trajectory_registry",
    "build_cross_host_trajectory_registry",
    # W38 family — disjoint cross-source consensus-reference trajectory-
    # divergence adjudication + manifest-v8 CID.
    "ConsensusReferenceProbe",
    "DisjointConsensusReferenceRatificationEnvelope",
    "DisjointConsensusReferenceRegistry",
    "W38DisjointConsensusReferenceResult",
    "DisjointConsensusReferenceOrchestrator",
    "DisjointTopologyError",
    "verify_disjoint_consensus_reference_ratification",
    "select_disjoint_consensus_divergence",
    "build_trivial_disjoint_consensus_registry",
    "build_disjoint_consensus_registry",
    # W39 family — multi-host disjoint quorum consensus-reference
    # ratification + manifest-v9 CID + mutually-disjoint physical-host
    # topology.
    "MultiHostDisjointQuorumProbe",
    "MultiHostDisjointQuorumRatificationEnvelope",
    "MultiHostDisjointQuorumRegistry",
    "W39MultiHostDisjointQuorumResult",
    "MultiHostDisjointQuorumOrchestrator",
    "MutuallyDisjointTopologyError",
    "verify_multi_host_disjoint_quorum_ratification",
    "select_multi_host_disjoint_quorum_decision",
    "build_trivial_multi_host_disjoint_quorum_registry",
    "build_multi_host_disjoint_quorum_registry",
    # W40 family — cross-host response-signature heterogeneity
    # ratification + manifest-v10 CID + cross-host response-text
    # Jaccard divergence guard.
    "ResponseSignatureProbe",
    "MultiHostResponseHeterogeneityProbe",
    "CrossHostResponseHeterogeneityRatificationEnvelope",
    "CrossHostResponseHeterogeneityRegistry",
    "W40CrossHostResponseHeterogeneityResult",
    "CrossHostResponseHeterogeneityOrchestrator",
    "verify_cross_host_response_heterogeneity_ratification",
    "select_cross_host_response_heterogeneity_decision",
    "build_trivial_cross_host_response_heterogeneity_registry",
    "build_cross_host_response_heterogeneity_registry",
    # W41 family — integrated multi-agent context synthesis +
    # manifest-v11 CID + cross-axis witness CID +
    # producer-axis x trust-axis decision selector.
    "IntegratedSynthesisRatificationEnvelope",
    "IntegratedSynthesisRegistry",
    "W41IntegratedSynthesisResult",
    "IntegratedSynthesisOrchestrator",
    "verify_integrated_synthesis_ratification",
    "select_integrated_synthesis_decision",
    "classify_producer_axis_branch",
    "classify_trust_axis_branch",
    "build_integrated_synthesis_registry",
    "build_trivial_integrated_synthesis_registry",
    # W42 family — cross-role-invariant synthesis +
    # manifest-v12 CID + role-handoff-signature axis +
    # composite-collusion bounding.
    "RoleInvariantSynthesisRatificationEnvelope",
    "RoleInvariancePolicyEntry",
    "RoleInvariancePolicyRegistry",
    "RoleInvariantSynthesisRegistry",
    "W42RoleInvariantResult",
    "RoleInvariantSynthesisOrchestrator",
    "verify_role_invariant_synthesis_ratification",
    "select_role_invariance_decision",
    "compute_role_handoff_signature_cid",
    "build_role_invariant_registry",
    "build_trivial_role_invariant_registry",
)

__all__ = [
    # Execution
    "RunSpec", "run", "RunReport",
    # Unified runtime (Slice 2)
    "SweepSpec", "run_sweep", "HeavyRunNotAcknowledged",
    # SDK v3.6 — LLM backend abstraction (additive integration
    # boundary for two-Mac MLX-distributed inference).
    "LLMBackend", "OllamaBackend", "OpenAICompatibleBackend",
    "MLXDistributedBackend", "make_backend", "backend_from_env",
    "backend_from_config",
    # Capsule runtime (Slice 3 — SDK v3)
    "ContextCapsule", "CapsuleKind", "CapsuleLifecycle",
    "CapsuleBudget", "CapsuleLedger", "CapsuleView",
    "CAPSULE_VIEW_SCHEMA", "render_view",
    "verify_chain_from_view_dict",
    "CapsuleAdmissionError", "CapsuleLifecycleError",
    "build_report_ledger",
    "capsule_from_handle", "capsule_from_handoff",
    "capsule_from_provenance", "capsule_from_sweep_cell",
    "capsule_from_sweep_spec", "capsule_from_profile",
    "capsule_from_readiness", "capsule_from_artifact",
    "capsule_from_report",
    # Phase-47 cohort subsumption (additive).
    "capsule_from_cohort", "capsule_from_adaptive_sub_table",
    # SDK v3.2 — intra-cell + detached-witness adapters.
    "capsule_from_patch_proposal", "capsule_from_test_verdict",
    "capsule_from_meta_manifest",
    # SDK v3.3 — sub-intra-cell parser-axis adapter.
    "capsule_from_parse_outcome", "PARSE_OUTCOME_ORACLE",
    # SDK v3.4 — sub-sub-intra-cell PROMPT/LLM_RESPONSE adapters.
    "capsule_from_prompt", "capsule_from_llm_response",
    "PROMPT_TEXT_CAP", "LLM_RESPONSE_TEXT_CAP",
    # Capsule-native runtime (SDK v3.1) — capsules drive execution.
    "CapsuleNativeRunContext", "ContentAddressMismatch",
    "seal_and_write_artifact",
    "CONSTRUCTION_IN_FLIGHT", "CONSTRUCTION_POST_HOC",
    # SDK v3.2 — strong on-disk verification.
    "verify_artifacts_on_disk", "verify_meta_manifest_on_disk",
    # SDK v3.3 — lifecycle audit + deterministic mode opt-in
    # (``RunSpec.deterministic``).
    "CapsuleLifecycleAudit", "LifecycleAuditReport",
    "audit_capsule_lifecycle", "audit_capsule_lifecycle_from_view",
    # Capsule admission policies (Phase 46 research milestone)
    "AdmissionPolicy", "FIFOPolicy", "KindPriorityPolicy",
    "SmallestFirstPolicy", "LearnedAdmissionPolicy",
    "BudgetedAdmissionLedger", "train_admission_policy",
    "featurise_capsule", "feature_index",
    # Bundle-aware admission (Phase 47 research milestone)
    "BundleAwarePolicy", "CorroboratedAdmissionPolicy",
    "PluralityBundlePolicy", "BundleLearnedPolicy",
    "train_bundle_policy", "BundleStats",
    "featurise_capsule_with_bundle", "bundle_feature_index",
    # Bundle-aware decoding (Phase 48 research milestone)
    "BundleDecoder", "PriorityDecoder", "PluralityDecoder",
    "SourceCorroboratedPriorityDecoder", "LearnedBundleDecoder",
    "train_learned_bundle_decoder", "BUNDLE_DECODER_FEATURES",
    "evaluate_decoder", "DecoderResult",
    # Stronger decoder + symmetric transfer (Phase 49)
    "BUNDLE_DECODER_FEATURES_V2", "INTERACTION_FEATURES",
    "DEEPSET_PHI_FEATURES",
    "LearnedBundleDecoderV2", "train_learned_bundle_decoder_v2",
    "InteractionBundleDecoder", "train_interaction_bundle_decoder",
    "MLPBundleDecoder", "train_mlp_bundle_decoder",
    "DeepSetBundleDecoder", "train_deep_set_bundle_decoder",
    "MultitaskBundleDecoder", "train_multitask_bundle_decoder",
    # SDK v3.5 — capsule-native multi-agent team coordination
    # (research slice; not part of the run-boundary product
    # runtime contract).
    "RoleBudget", "DEFAULT_ROLE_BUDGETS",
    "capsule_team_handoff", "capsule_role_view",
    "capsule_team_decision",
    "TeamAdmissionPolicy", "TeamAdmissionDecision",
    "TeamFifoAdmissionPolicy", "TeamClaimPriorityAdmissionPolicy",
    "TeamCoverageGuidedAdmissionPolicy",
    "TeamCohortCoherenceAdmissionPolicy",
    "TeamCrossRoleCorroborationAdmissionPolicy",
    "TeamMultiServiceCorroborationAdmissionPolicy",
    "TeamCoordinator", "audit_team_lifecycle",
    "TeamLifecycleAuditReport", "T_INVARIANTS",
    # SDK v3.11 — bundle-aware team decoder (W10 family).
    "BundleAwareTeamDecoder", "decode_admitted_role_view",
    "CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE",
    # SDK v3.12 — multi-round bundle-aware team decoder (W11 family).
    "MultiRoundBundleDecoder", "collect_admitted_handoffs",
    # SDK v3.13 — real-LLM-robust multi-round bundle decoder (W12 family).
    "RobustMultiRoundBundleDecoder", "CLAIM_KIND_SYNONYMS",
    "normalize_claim_kind", "normalize_payload", "normalize_handoff",
    # SDK v3.14 — layered open-world normaliser + decoder (W13 family).
    "HeuristicAbstractionRule", "LayeredClaimNormalizer",
    "LayeredRobustMultiRoundBundleDecoder",
    "LAYERED_NORMALIZER_ABSTAIN",
    # SDK v3.15 — structured producer protocol (W14 family).
    "PRODUCER_PROMPT_NAIVE", "PRODUCER_PROMPT_STRUCTURED",
    "ALL_PRODUCER_PROMPT_MODES",
    "PRODUCER_PROMPT_MAGNITUDE_HINTED",
    "OperationalThreshold",
    "INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS",
    "incident_triage_magnitude_thresholds",
    "RoleExtractionSchema", "ProducerPromptResult",
    "StructuredProducerProtocol",
    "INCIDENT_TRIAGE_OBSERVATION_KINDS",
    "incident_triage_role_schemas",
    # SDK v3.16 — attention-aware capsule context packing (W15 family).
    "W15_DEFAULT_TIER_WEIGHT", "W15_DEFAULT_CCK_WEIGHT",
    "W15_DEFAULT_CORROBORATION_WEIGHT", "W15_DEFAULT_MAGNITUDE_WEIGHT",
    "W15_DEFAULT_ROUND_WEIGHT",
    "W15PackedHandoff", "W15PackResult",
    "FifoContextPacker", "CapsuleContextPacker",
    "AttentionAwareBundleDecoder",
    # SDK v3.19 — W18 family.
    "W18CompatibilityResult", "RelationalCompatibilityDisambiguator",
    # SDK v3.20 — W19 family.
    "BundleContradictionDisambiguator", "W19TrustResult",
    "W19_ALL_BRANCHES", "W19_BRANCH_PRIMARY_TRUSTED", "W19_BRANCH_INVERSION",
    "W19_BRANCH_CONFOUND_RESOLVED", "W19_BRANCH_ABSTAINED_NO_SIGNAL",
    "W19_BRANCH_ABSTAINED_SYMMETRIC", "W19_BRANCH_DISABLED",
    # SDK v3.21 — W20 family (outside-witness acquisition).
    "OutsideWitnessOracle", "OutsideQuery", "OutsideVerdict",
    "ServiceGraphOracle", "CompromisedServiceGraphOracle",
    "AbstainingOracle", "LLMAdjudicatorOracle",
    "build_incident_triage_service_graph",
    "OutsideWitnessAcquisitionDisambiguator", "W20OutsideResult",
    "W20_ALL_BRANCHES", "W20_BRANCH_OUTSIDE_RESOLVED",
    "W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC",
    "W20_BRANCH_OUTSIDE_ABSTAINED", "W20_BRANCH_NO_TRIGGER",
    "W20_BRANCH_DISABLED", "W20_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.22 — W21 family (trust-weighted multi-oracle adjudicator).
    "OracleRegistration", "ChangeHistoryOracle", "OnCallNotesOracle",
    "SingletonAsymmetricOracle", "DisagreeingHonestOracle",
    "W21OracleProbe", "W21MultiOracleResult",
    "TrustWeightedMultiOracleDisambiguator",
    "W21_ALL_BRANCHES", "W21_BRANCH_QUORUM_RESOLVED",
    "W21_BRANCH_NO_QUORUM", "W21_BRANCH_SYMMETRIC_QUORUM",
    "W21_BRANCH_NO_ORACLES", "W21_BRANCH_NO_TRIGGER",
    "W21_BRANCH_DISABLED", "W21_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.23 — W22 family (capsule + audited latent-state-sharing hybrid).
    "SchemaCapsule", "build_incident_triage_schema_capsule",
    "LatentDigestEnvelope", "LatentVerificationOutcome",
    "verify_latent_digest", "SharedReadCache", "CachingOracleAdapter",
    "EnvelopeTamperer", "W22LatentResult", "LatentDigestDisambiguator",
    "W22_ALL_BRANCHES", "W22_BRANCH_LATENT_RESOLVED",
    "W22_BRANCH_LATENT_REJECTED", "W22_BRANCH_NO_TRIGGER",
    "W22_BRANCH_NO_SCHEMA", "W22_BRANCH_DISABLED",
    "W22_BRANCH_ABSTAIN_PASSTHROUGH",
    "W22_DEFAULT_TRIGGER_BRANCHES",
    "W22_LATENT_ENVELOPE_SCHEMA_VERSION",
    # SDK v3.26 — W25 family (shared-fanout dense-control).
    "FanoutEnvelope", "SharedFanoutRegistry", "SharedFanoutDisambiguator",
    "verify_fanout", "W25FanoutResult",
    "W25_FANOUT_SCHEMA_VERSION",
    "W25_ALL_BRANCHES", "W25_BRANCH_FANOUT_PRODUCER_EMITTED",
    "W25_BRANCH_FANOUT_CONSUMER_RESOLVED",
    "W25_BRANCH_FANOUT_CONSUMER_REJECTED",
    "W25_BRANCH_NO_TRIGGER", "W25_BRANCH_DISABLED",
    "W25_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.27 — W26 family (chain-persisted dense-control fanout +
    # per-consumer projections).
    "ChainAnchorEnvelope", "ChainAdvanceEnvelope",
    "ChainPersistedFanoutRegistry",
    "ChainPersistedFanoutDisambiguator",
    "ProjectionSlot", "W26ChainResult",
    "verify_chain_anchor", "verify_chain_advance",
    "verify_projection_subscription",
    "W26_CHAIN_ANCHOR_SCHEMA_VERSION",
    "W26_CHAIN_ADVANCE_SCHEMA_VERSION",
    "W26_ALL_BRANCHES",
    "W26_BRANCH_CHAIN_ANCHORED", "W26_BRANCH_CHAIN_ADVANCED",
    "W26_BRANCH_CHAIN_REJECTED", "W26_BRANCH_CHAIN_RE_ANCHORED",
    "W26_BRANCH_CHAIN_PROJECTION_RESOLVED",
    "W26_BRANCH_CHAIN_PROJECTION_REJECTED",
    "W26_BRANCH_NO_TRIGGER", "W26_BRANCH_DISABLED",
    "W26_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.28 — W27 family.
    "SalienceSignatureEnvelope", "ChainPivotEnvelope",
    "MultiChainPersistedFanoutRegistry",
    "MultiChainPersistedFanoutDisambiguator",
    "MultiChainPersistedFanoutOrchestrator",
    "SharedMultiChainPool",
    "W27MultiChainResult", "W27OrchestratorResult",
    "compute_input_signature_cid",
    "verify_salience_signature", "verify_chain_pivot",
    "W27_SALIENCE_SIGNATURE_SCHEMA_VERSION",
    "W27_CHAIN_PIVOT_SCHEMA_VERSION",
    "W27_ALL_BRANCHES",
    "W27_BRANCH_PIVOTED", "W27_BRANCH_ANCHORED_NEW",
    "W27_BRANCH_POOL_EXHAUSTED", "W27_BRANCH_PIVOT_REJECTED",
    "W27_BRANCH_FALLBACK_W26", "W27_BRANCH_NO_TRIGGER",
    "W27_BRANCH_DISABLED",
    "W27_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.29 — W28 family (ensemble-verified cross-model multi-chain
    # pivot ratification). EXPERIMENTAL — see __experimental__ tuple.
    "ProbeVote", "EnsembleProbe", "EnsembleProbeRegistration",
    "DeterministicSignatureProbe", "OracleConsultationProbe",
    "LLMSignatureProbe",
    "EnsemblePivotRatificationEnvelope",
    "EnsembleRatificationRegistry",
    "EnsembleVerifiedMultiChainOrchestrator",
    "W28EnsembleResult",
    "verify_ensemble_pivot_ratification",
    "build_default_ensemble_registry",
    "build_two_probe_oracle_ensemble_registry",
    "build_cross_host_llm_ensemble_registry",
    "W28_RATIFICATION_SCHEMA_VERSION",
    "W28_ALL_BRANCHES",
    "W28_BRANCH_RATIFIED", "W28_BRANCH_RATIFIED_PASSTHROUGH",
    "W28_BRANCH_QUORUM_BELOW_THRESHOLD", "W28_BRANCH_PROBE_REJECTED",
    "W28_BRANCH_NO_RATIFY_NEEDED", "W28_BRANCH_FALLBACK_W27",
    "W28_BRANCH_NO_TRIGGER", "W28_BRANCH_DISABLED",
    "W28_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.30 — W29 family (geometry-partitioned product-manifold
    # dense control + audited subspace-basis payload + factoradic
    # routing + causal-validity gate + cross-host variance witness).
    # EXPERIMENTAL — see __experimental__ tuple.
    "SubspaceBasis", "verify_subspace_basis",
    "compute_structural_subspace_basis",
    "encode_permutation_to_factoradic",
    "decode_factoradic_to_permutation",
    "CrossHostVarianceWitness",
    "GeometryPartitionedRatificationEnvelope",
    "PartitionRegistration", "GeometryPartitionRegistry",
    "W29PartitionResult", "GeometryPartitionedOrchestrator",
    "classify_partition_id_for_cell",
    "verify_geometry_partition_ratification",
    "build_trivial_partition_registry",
    "build_three_partition_registry",
    "W29_PARTITION_SCHEMA_VERSION",
    "W29_PARTITION_LINEAR", "W29_PARTITION_HIERARCHICAL",
    "W29_PARTITION_CYCLIC",
    "W29_REGISTERED_PARTITION_IDS", "W29_PARTITION_LABEL",
    "W29_DEFAULT_ORTHOGONALITY_TOL",
    "W29_BRANCH_PARTITION_RESOLVED",
    "W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH",
    "W29_BRANCH_PARTITION_REJECTED",
    "W29_BRANCH_PARTITION_BELOW_THRESHOLD",
    "W29_BRANCH_CROSS_HOST_VARIANCE_WITNESSED",
    "W29_BRANCH_NO_PARTITION_NEEDED",
    "W29_BRANCH_FALLBACK_W28",
    "W29_BRANCH_NO_TRIGGER", "W29_BRANCH_DISABLED",
    "W29_ALL_BRANCHES",
    "W29_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.31 — W30 family (calibrated geometry-aware dense control +
    # multi-stride basis history + per-partition calibration prior +
    # cross-host disagreement-routing + ancestor-chain causal binding).
    # EXPERIMENTAL — see __experimental__ tuple.
    "BasisHistory", "AncestorChain", "PartitionCalibrationVector",
    "CalibratedGeometryRatificationEnvelope",
    "CalibratedGeometryRegistry",
    "W30CalibratedResult",
    "CalibratedGeometryOrchestrator",
    "verify_calibrated_geometry_ratification",
    "update_partition_calibration_running_mean",
    "build_trivial_calibrated_registry",
    "build_calibrated_registry",
    "W30_CALIBRATED_SCHEMA_VERSION",
    "W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD",
    "W30_BRANCH_CALIBRATED_RESOLVED",
    "W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH",
    "W30_BRANCH_CALIBRATED_REJECTED",
    "W30_BRANCH_DISAGREEMENT_ROUTED",
    "W30_BRANCH_CALIBRATION_REROUTED",
    "W30_BRANCH_NO_CALIBRATION_NEEDED",
    "W30_BRANCH_FALLBACK_W29",
    "W30_BRANCH_NO_TRIGGER", "W30_BRANCH_DISABLED",
    "W30_ALL_BRANCHES",
    # SDK v3.32 — W31 family (online self-calibrated geometry-aware
    # dense control + sealed prior trajectory + adaptive threshold +
    # W31 manifest CID).  EXPERIMENTAL — see __experimental__ tuple.
    "PriorTrajectoryEntry",
    "OnlineCalibratedRatificationEnvelope",
    "OnlineCalibratedRegistry",
    "W31OnlineResult",
    "OnlineCalibratedOrchestrator",
    "verify_online_calibrated_ratification",
    "derive_per_cell_agreement_signal",
    "compute_adaptive_threshold",
    "build_trivial_online_registry",
    "build_online_calibrated_registry",
    "W31_ONLINE_SCHEMA_VERSION",
    "W31_DEFAULT_THRESHOLD_MIN",
    "W31_DEFAULT_THRESHOLD_MAX",
    "W31_DEFAULT_TRAJECTORY_WINDOW",
    "W31_BRANCH_ONLINE_RESOLVED",
    "W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH",
    "W31_BRANCH_ONLINE_REJECTED",
    "W31_BRANCH_ONLINE_DISABLED",
    "W31_BRANCH_ONLINE_NO_TRIGGER",
    "W31_ALL_BRANCHES",
    # SDK v3.33 — W32 family (long-window convergent online geometry-
    # aware dense control + EWMA + Page CUSUM + gold-correlated
    # routing + manifest-v2 CID).  EXPERIMENTAL — see
    # __experimental__ tuple.
    "GoldCorrelationMap", "build_gold_correlation_map",
    "ConvergenceStateEntry",
    "LongWindowConvergentRatificationEnvelope",
    "LongWindowConvergentRegistry",
    "W32LongWindowResult",
    "LongWindowConvergentOrchestrator",
    "verify_long_window_convergent_ratification",
    "update_ewma_prior", "update_cusum_two_sided", "detect_change_point",
    "build_trivial_long_window_registry",
    "build_long_window_convergent_registry",
    "W32_LONG_WINDOW_SCHEMA_VERSION",
    "W32_DEFAULT_EWMA_ALPHA",
    "W32_DEFAULT_CUSUM_THRESHOLD",
    "W32_DEFAULT_CUSUM_K",
    "W32_DEFAULT_CUSUM_MAX",
    "W32_DEFAULT_LONG_WINDOW",
    "W32_DEFAULT_GOLD_CORRELATION_MIN",
    "W32_BRANCH_LONG_WINDOW_RESOLVED",
    "W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH",
    "W32_BRANCH_LONG_WINDOW_REJECTED",
    "W32_BRANCH_LONG_WINDOW_DISABLED",
    "W32_BRANCH_LONG_WINDOW_NO_TRIGGER",
    "W32_BRANCH_GOLD_CORRELATED_REROUTED",
    "W32_BRANCH_CHANGE_POINT_RESET",
    "W32_ALL_BRANCHES",
    # SDK v3.34 — W33 family (trust-EWMA-tracked multi-oracle
    # adjudication + manifest-v3 CID).  EXPERIMENTAL — see
    # __experimental__ tuple.
    "TrustTrajectoryEntry",
    "TrustEWMARatificationEnvelope",
    "TrustEWMARegistry",
    "W33TrustEWMAResult",
    "TrustEWMATrackedMultiOracleOrchestrator",
    "verify_trust_ewma_ratification",
    "derive_per_oracle_agreement_signal",
    "build_trivial_trust_ewma_registry",
    "build_trust_ewma_registry",
    "W33_TRUST_EWMA_SCHEMA_VERSION",
    "W33_DEFAULT_TRUST_THRESHOLD",
    "W33_DEFAULT_TRUST_TRAJECTORY_WINDOW",
    "W33_DEFAULT_EWMA_ALPHA",
    "W33_BRANCH_TRUST_EWMA_RESOLVED",
    "W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH",
    "W33_BRANCH_TRUST_EWMA_REJECTED",
    "W33_BRANCH_TRUST_EWMA_DISABLED",
    "W33_BRANCH_TRUST_EWMA_NO_TRIGGER",
    "W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN",
    "W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE",
    "W33_ALL_BRANCHES",
    # SDK v3.35 — W34 family (live-aware multi-anchor adjudication +
    # native-latent audited response-feature proxy + manifest-v4 CID).
    # EXPERIMENTAL — see __experimental__ tuple.
    "LiveOracleAttestation",
    "LiveAwareMultiAnchorRatificationEnvelope",
    "LiveAwareMultiAnchorRegistry",
    "HostRegistration",
    "W34LiveAwareResult",
    "LiveAwareMultiAnchorOrchestrator",
    "verify_live_aware_multi_anchor_ratification",
    "derive_multi_anchor_consensus_reference",
    "compute_response_feature_signature",
    "apply_host_decay",
    "build_trivial_live_aware_registry",
    "build_live_aware_registry",
    "W34_LIVE_AWARE_SCHEMA_VERSION",
    "W34_DEFAULT_ANCHOR_QUORUM_MIN",
    "W34_DEFAULT_HOST_DECAY_FACTOR",
    "W34_DEFAULT_LIVE_ATTESTATION_TIMEOUT_MS_BUCKET",
    "W34_BRANCH_LIVE_AWARE_RESOLVED",
    "W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH",
    "W34_BRANCH_LIVE_AWARE_REJECTED",
    "W34_BRANCH_LIVE_AWARE_DISABLED",
    "W34_BRANCH_LIVE_AWARE_NO_TRIGGER",
    "W34_BRANCH_MULTI_ANCHOR_CONSENSUS",
    "W34_BRANCH_MULTI_ANCHOR_NO_CONSENSUS",
    "W34_BRANCH_HOST_DECAY_FIRED",
    "W34_ALL_BRANCHES",
    # SDK v3.36 — W35 family (trust-subspace dense-control proxy +
    # basis-history projection + manifest-v5 CID).  EXPERIMENTAL —
    # see __experimental__ tuple.
    "TrustSubspaceBasisEntry",
    "TrustSubspaceDenseRatificationEnvelope",
    "TrustSubspaceDenseRegistry",
    "W35TrustSubspaceResult",
    "TrustSubspaceDenseControlOrchestrator",
    "verify_trust_subspace_dense_ratification",
    "select_trust_subspace_projection",
    "build_trivial_trust_subspace_registry",
    "build_trust_subspace_dense_registry",
    "W35_TRUST_SUBSPACE_SCHEMA_VERSION",
    "W35_DEFAULT_BASIS_EWMA_ALPHA",
    "W35_DEFAULT_PROJECTION_THRESHOLD",
    "W35_DEFAULT_PROJECTION_MARGIN_MIN",
    "W35_DEFAULT_BASIS_HISTORY_WINDOW",
    "W35_DEFAULT_MIN_BASIS_OBSERVATIONS",
    "W35_BRANCH_TRUST_SUBSPACE_RESOLVED",
    "W35_BRANCH_TRIVIAL_TRUST_SUBSPACE_PASSTHROUGH",
    "W35_BRANCH_TRUST_SUBSPACE_REJECTED",
    "W35_BRANCH_TRUST_SUBSPACE_DISABLED",
    "W35_BRANCH_TRUST_SUBSPACE_NO_TRIGGER",
    "W35_BRANCH_BASIS_HISTORY_REROUTED",
    "W35_BRANCH_BASIS_HISTORY_UNSAFE",
    "W35_BRANCH_BASIS_HISTORY_ABSTAINED",
    "W35_ALL_BRANCHES",
    # SDK v3.37 — W36 family (host-diverse trust-subspace guard +
    # manifest-v6 CID). EXPERIMENTAL — see __experimental__ tuple.
    "HostDiverseBasisEntry",
    "HostDiverseRatificationEnvelope",
    "HostDiverseRegistry",
    "W36HostDiverseResult",
    "HostDiverseTrustSubspaceOrchestrator",
    "verify_host_diverse_ratification",
    "select_host_diverse_projection",
    "build_trivial_host_diverse_registry",
    "build_host_diverse_registry",
    "W36_HOST_DIVERSE_SCHEMA_VERSION",
    "W36_DEFAULT_MIN_DISTINCT_HOSTS",
    "W36_DEFAULT_HOST_DIVERSITY_THRESHOLD",
    "W36_DEFAULT_HOST_DIVERSITY_MARGIN_MIN",
    "W36_BRANCH_HOST_DIVERSE_RESOLVED",
    "W36_BRANCH_TRIVIAL_HOST_DIVERSE_PASSTHROUGH",
    "W36_BRANCH_HOST_DIVERSE_REJECTED",
    "W36_BRANCH_HOST_DIVERSE_DISABLED",
    "W36_BRANCH_HOST_DIVERSE_NO_TRIGGER",
    "W36_BRANCH_HOST_DIVERSE_REROUTED",
    "W36_BRANCH_HOST_DIVERSE_UNSAFE",
    "W36_BRANCH_HOST_DIVERSE_ABSTAINED",
    "W36_ALL_BRANCHES",
    # SDK v3.38 — W37 family (anchor-cross-host basis-trajectory
    # ratification + manifest-v7 CID).  EXPERIMENTAL — see
    # __experimental__ tuple.
    "CrossHostBasisTrajectoryEntry",
    "CrossHostBasisTrajectoryRatificationEnvelope",
    "CrossHostBasisTrajectoryRegistry",
    "W37CrossHostTrajectoryResult",
    "CrossHostBasisTrajectoryOrchestrator",
    "verify_cross_host_trajectory_ratification",
    "select_cross_host_trajectory_projection",
    "build_trivial_cross_host_trajectory_registry",
    "build_cross_host_trajectory_registry",
    "W37_CROSS_HOST_TRAJECTORY_SCHEMA_VERSION",
    "W37_DEFAULT_TRAJECTORY_EWMA_ALPHA",
    "W37_DEFAULT_TRAJECTORY_THRESHOLD",
    "W37_DEFAULT_TRAJECTORY_MARGIN_MIN",
    "W37_DEFAULT_MIN_ANCHORED_OBSERVATIONS",
    "W37_DEFAULT_MIN_TRAJECTORY_ANCHORED_HOSTS",
    "W37_DEFAULT_TRAJECTORY_HISTORY_WINDOW",
    "W37_BRANCH_TRAJECTORY_RESOLVED",
    "W37_BRANCH_TRIVIAL_TRAJECTORY_PASSTHROUGH",
    "W37_BRANCH_TRAJECTORY_REJECTED",
    "W37_BRANCH_TRAJECTORY_DISABLED",
    "W37_BRANCH_TRAJECTORY_NO_TRIGGER",
    "W37_BRANCH_TRAJECTORY_REROUTED",
    "W37_BRANCH_TRAJECTORY_UNSAFE",
    "W37_BRANCH_TRAJECTORY_ABSTAINED",
    "W37_BRANCH_TRAJECTORY_NO_HISTORY",
    "W37_BRANCH_TRAJECTORY_DISAGREEMENT",
    "W37_BRANCH_TRAJECTORY_POISONED",
    "W37_ALL_BRANCHES",
    # W38 family — disjoint cross-source consensus-reference trajectory-
    # divergence adjudication + manifest-v8 CID.
    "ConsensusReferenceProbe",
    "DisjointConsensusReferenceRatificationEnvelope",
    "DisjointConsensusReferenceRegistry",
    "W38DisjointConsensusReferenceResult",
    "DisjointConsensusReferenceOrchestrator",
    "DisjointTopologyError",
    "verify_disjoint_consensus_reference_ratification",
    "select_disjoint_consensus_divergence",
    "build_trivial_disjoint_consensus_registry",
    "build_disjoint_consensus_registry",
    "W38_DISJOINT_CONSENSUS_SCHEMA_VERSION",
    "W38_DEFAULT_CONSENSUS_STRENGTH_MIN",
    "W38_DEFAULT_DIVERGENCE_MARGIN_MIN",
    "W38_BRANCH_CONSENSUS_RESOLVED",
    "W38_BRANCH_TRIVIAL_CONSENSUS_PASSTHROUGH",
    "W38_BRANCH_CONSENSUS_REJECTED",
    "W38_BRANCH_CONSENSUS_DISABLED",
    "W38_BRANCH_CONSENSUS_NO_TRIGGER",
    "W38_BRANCH_CONSENSUS_RATIFIED",
    "W38_BRANCH_CONSENSUS_NO_REFERENCE",
    "W38_BRANCH_CONSENSUS_DIVERGENCE_ABSTAINED",
    "W38_BRANCH_CONSENSUS_REFERENCE_WEAK",
    "W38_ALL_BRANCHES",
    # W39 family — multi-host disjoint quorum consensus-reference
    # ratification + manifest-v9 CID + mutually-disjoint physical-host
    # topology.
    "MultiHostDisjointQuorumProbe",
    "MultiHostDisjointQuorumRatificationEnvelope",
    "MultiHostDisjointQuorumRegistry",
    "W39MultiHostDisjointQuorumResult",
    "MultiHostDisjointQuorumOrchestrator",
    "MutuallyDisjointTopologyError",
    "verify_multi_host_disjoint_quorum_ratification",
    "select_multi_host_disjoint_quorum_decision",
    "build_trivial_multi_host_disjoint_quorum_registry",
    "build_multi_host_disjoint_quorum_registry",
    "W39_MULTI_HOST_DISJOINT_QUORUM_SCHEMA_VERSION",
    "W39_DEFAULT_QUORUM_MIN",
    "W39_DEFAULT_MIN_QUORUM_PROBES",
    "W39_DEFAULT_QUORUM_STRENGTH_MIN",
    "W39_DEFAULT_QUORUM_DIVERGENCE_MARGIN_MIN",
    "W39_BRANCH_QUORUM_RESOLVED",
    "W39_BRANCH_TRIVIAL_QUORUM_PASSTHROUGH",
    "W39_BRANCH_QUORUM_REJECTED",
    "W39_BRANCH_QUORUM_DISABLED",
    "W39_BRANCH_QUORUM_NO_TRIGGER",
    "W39_BRANCH_QUORUM_RATIFIED",
    "W39_BRANCH_QUORUM_DIVERGENCE_ABSTAINED",
    "W39_BRANCH_QUORUM_NO_REFERENCES",
    "W39_BRANCH_QUORUM_INSUFFICIENT",
    "W39_BRANCH_QUORUM_SPLIT",
    "W39_BRANCH_QUORUM_REFERENCE_WEAK",
    "W39_ALL_BRANCHES",
    # W40 family — cross-host response-signature heterogeneity
    # ratification + manifest-v10 CID + cross-host response-text
    # Jaccard divergence guard.
    "ResponseSignatureProbe",
    "MultiHostResponseHeterogeneityProbe",
    "CrossHostResponseHeterogeneityRatificationEnvelope",
    "CrossHostResponseHeterogeneityRegistry",
    "W40CrossHostResponseHeterogeneityResult",
    "CrossHostResponseHeterogeneityOrchestrator",
    "verify_cross_host_response_heterogeneity_ratification",
    "select_cross_host_response_heterogeneity_decision",
    "build_trivial_cross_host_response_heterogeneity_registry",
    "build_cross_host_response_heterogeneity_registry",
    "W40_RESPONSE_HETEROGENEITY_SCHEMA_VERSION",
    "W40_DEFAULT_RESPONSE_TEXT_DIVERSITY_MIN",
    "W40_DEFAULT_MIN_RESPONSE_SIGNATURE_PROBES",
    "W40_BRANCH_RESPONSE_SIGNATURE_RESOLVED",
    "W40_BRANCH_TRIVIAL_RESPONSE_SIGNATURE_PASSTHROUGH",
    "W40_BRANCH_RESPONSE_SIGNATURE_REJECTED",
    "W40_BRANCH_RESPONSE_SIGNATURE_DISABLED",
    "W40_BRANCH_RESPONSE_SIGNATURE_NO_TRIGGER",
    "W40_BRANCH_RESPONSE_SIGNATURE_DIVERSE",
    "W40_BRANCH_RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED",
    "W40_BRANCH_RESPONSE_SIGNATURE_NO_REFERENCES",
    "W40_BRANCH_RESPONSE_SIGNATURE_INSUFFICIENT",
    "W40_BRANCH_RESPONSE_SIGNATURE_INCOMPLETE",
    "W40_ALL_BRANCHES",
    # W41 family - integrated multi-agent context synthesis +
    # manifest-v11 CID + cross-axis witness + producer/trust
    # decision selector.
    "IntegratedSynthesisRatificationEnvelope",
    "IntegratedSynthesisRegistry",
    "W41IntegratedSynthesisResult",
    "IntegratedSynthesisOrchestrator",
    "verify_integrated_synthesis_ratification",
    "select_integrated_synthesis_decision",
    "classify_producer_axis_branch",
    "classify_trust_axis_branch",
    "build_integrated_synthesis_registry",
    "build_trivial_integrated_synthesis_registry",
    "W41_INTEGRATED_SYNTHESIS_SCHEMA_VERSION",
    "W41_PRODUCER_AXIS_FIRED",
    "W41_PRODUCER_AXIS_NO_TRIGGER",
    "W41_TRUST_AXIS_RATIFIED",
    "W41_TRUST_AXIS_ABSTAINED",
    "W41_TRUST_AXIS_NO_TRIGGER",
    "W41_BRANCH_TRIVIAL_INTEGRATED_PASSTHROUGH",
    "W41_BRANCH_INTEGRATED_DISABLED",
    "W41_BRANCH_INTEGRATED_REJECTED",
    "W41_BRANCH_INTEGRATED_PRODUCER_ONLY",
    "W41_BRANCH_INTEGRATED_TRUST_ONLY",
    "W41_BRANCH_INTEGRATED_BOTH_AXES",
    "W41_BRANCH_INTEGRATED_AXES_DIVERGED_ABSTAINED",
    "W41_BRANCH_INTEGRATED_NEITHER_AXIS",
    "LearnedTeamAdmissionPolicy", "TeamTrainSample", "TeamTrainStats",
    "train_team_admission_policy", "featurise_team_handoff",
    "KNOWN_SOURCE_ROLES", "KNOWN_CLAIM_KINDS",
    "TEAM_FEATURE_DIM", "TEAM_FEATURE_NAMES",
    # Layered API (end-user / developer / researcher ergonomics)
    "CoordPySimpleAPI", "CoordPyBuilderAPI", "CoordPyAdvancedAPI", "BuilderSpec",
    # Stable lightweight agent/team surface
    "Agent", "AgentTurn", "TeamResult", "AgentTeam", "agent",
    "create_team",
    # Config
    "CoordPyConfig",
    # Provenance
    "PROVENANCE_SCHEMA", "build_manifest",
    # Re-exported submodules
    "profiles", "report", "ci_gate", "import_data", "extensions",
    # Version / schema constants
    "__version__", "SDK_VERSION",
    "PRODUCT_REPORT_SCHEMA", "PRODUCT_REPORT_SCHEMA_V1",
    "CI_VERDICT_SCHEMA", "IMPORT_AUDIT_SCHEMA",
]
