"""Wevra — the first finished product from the Context Zero programme.

**Wevra is a context-capsule runtime.** Every piece of context that
crosses a role boundary, a layer boundary, or a run boundary is a
typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-stamped **capsule** — never a raw prompt string. ``RunSpec``
in, one reproducible, provenance-stamped report out, where the report
is the root of a capsule graph you can audit, replay, and trust.

Positioning (honest)
--------------------
* Context Zero is the *research programme* — 45+ phases, dozens of
  research shards, and the EXTENDED_MATH / PROOFS corpus.
* Wevra is the *first shipped product surface* from that programme.
  Its load-bearing idea is the **Context Capsule**: the contract
  every inter-role / inter-layer / inter-run artifact satisfies.
* The individual primitives under a capsule (content addressing,
  hash-chained logs, typed claim kinds, frozen lifecycle, capability-
  style typed references) are inherited from Git / Merkle DAGs /
  IPFS / actor systems / session types. What Wevra claims as new is
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
    * ``WevraConfig``
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

from vision_mvp import __version__

from .config import WevraConfig
from .provenance import PROVENANCE_SCHEMA, build_manifest
from .run import RunSpec, run

# Re-export the product modules under stable names. Downstream code
# should import from ``vision_mvp.wevra`` rather than
# ``vision_mvp.product`` — the ``.product`` path remains available
# during a deprecation window but is not part of the SDK contract.
from vision_mvp.product import profiles, report, ci_gate, import_data

# Extension system (Slice 2). Kept as a first-class re-export so
# ``from vision_mvp.wevra import extensions`` is the SDK path.
from vision_mvp.wevra import extensions

# Unified runtime (Slice 2).
from .runtime import SweepSpec, run_sweep, HeavyRunNotAcknowledged

# SDK v3.6 — LLM backend abstraction. Strictly additive: when no
# backend is supplied, the runtime instantiates ``LLMClient``
# byte-for-byte unchanged. When a backend is supplied (e.g. an
# ``MLXDistributedBackend`` whose ``base_url`` points at an
# ``mlx_lm.server`` launched under ``mpirun`` across two Apple
# Silicon hosts), the inner-loop calls dispatch through the
# backend without any other change to the spine. The PROMPT /
# LLM_RESPONSE capsules' shape (SHA-256 + length + snippet) is
# preserved regardless of backend.
from .llm_backend import (
    LLMBackend, OllamaBackend, MLXDistributedBackend, make_backend,
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
    WevraSimpleAPI, WevraBuilderAPI, WevraAdvancedAPI, BuilderSpec,
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
# ``vision_mvp.wevra.team_coord`` and ``team_policy`` and emits
# three new closed-vocabulary capsule kinds (TEAM_HANDOFF,
# ROLE_VIEW, TEAM_DECISION). The team-level lifecycle audit
# (``audit_team_lifecycle``) mechanically verifies invariants
# T-1..T-7 (Theorem W4-1). See
# ``docs/RESULTS_WEVRA_TEAM_COORD.md`` and
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


SDK_VERSION = "wevra.sdk.v3.26"
PRODUCT_REPORT_SCHEMA = "phase45.product_report.v2"
# Legacy schema — still emitted by mock-only runs that don't touch
# the unified runtime path. Consumers should accept both.
PRODUCT_REPORT_SCHEMA_V1 = "phase45.product_report.v1"
CI_VERDICT_SCHEMA = "phase46.ci_verdict.v1"
IMPORT_AUDIT_SCHEMA = "phase46.import_audit.v1"

__all__ = [
    # Execution
    "RunSpec", "run",
    # Unified runtime (Slice 2)
    "SweepSpec", "run_sweep", "HeavyRunNotAcknowledged",
    # SDK v3.6 — LLM backend abstraction (additive integration
    # boundary for two-Mac MLX-distributed inference).
    "LLMBackend", "OllamaBackend", "MLXDistributedBackend",
    "make_backend",
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
    "LearnedTeamAdmissionPolicy", "TeamTrainSample", "TeamTrainStats",
    "train_team_admission_policy", "featurise_team_handoff",
    "KNOWN_SOURCE_ROLES", "KNOWN_CLAIM_KINDS",
    "TEAM_FEATURE_DIM", "TEAM_FEATURE_NAMES",
    # Layered API (end-user / developer / researcher ergonomics)
    "WevraSimpleAPI", "WevraBuilderAPI", "WevraAdvancedAPI", "BuilderSpec",
    # Config
    "WevraConfig",
    # Provenance
    "PROVENANCE_SCHEMA", "build_manifest",
    # Re-exported submodules
    "profiles", "report", "ci_gate", "import_data", "extensions",
    # Version / schema constants
    "__version__", "SDK_VERSION",
    "PRODUCT_REPORT_SCHEMA", "PRODUCT_REPORT_SCHEMA_V1",
    "CI_VERDICT_SCHEMA", "IMPORT_AUDIT_SCHEMA",
]
