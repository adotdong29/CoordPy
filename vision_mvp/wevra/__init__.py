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

# Capsule runtime (Slice 3 — SDK v3). The load-bearing abstraction
# around which the rest of the SDK is centred.
from .capsule import (
    ContextCapsule, CapsuleKind, CapsuleLifecycle, CapsuleBudget,
    CapsuleLedger, CapsuleView, CAPSULE_VIEW_SCHEMA, render_view,
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

SDK_VERSION = "wevra.sdk.v3"
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
    # Capsule runtime (Slice 3 — SDK v3)
    "ContextCapsule", "CapsuleKind", "CapsuleLifecycle",
    "CapsuleBudget", "CapsuleLedger", "CapsuleView",
    "CAPSULE_VIEW_SCHEMA", "render_view",
    "CapsuleAdmissionError", "CapsuleLifecycleError",
    "build_report_ledger",
    "capsule_from_handle", "capsule_from_handoff",
    "capsule_from_provenance", "capsule_from_sweep_cell",
    "capsule_from_sweep_spec", "capsule_from_profile",
    "capsule_from_readiness", "capsule_from_artifact",
    "capsule_from_report",
    # Phase-47 cohort subsumption (additive).
    "capsule_from_cohort", "capsule_from_adaptive_sub_table",
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
