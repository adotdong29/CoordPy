"""Phase 71 — bounded-window session compaction + intra-cell resample-quorum
+ real cross-process producer/decoder wire (SDK v3.25, W24 family anchor).

The follow-up to SDK v3.24 (W23) on the named research direction the
W23 milestone explicitly left open: (a) the per-cell verbose digest
cost still grows with cumulative state on long sessions, and (b)
W23-C-MITIGATION-LIVE-VARIANCE — the intra-cell variance that the
PER_CELL_NONCE mitigation cannot touch.

The W24 family ships three composable mechanisms:

  * **W24-1 Bounded-window session compaction**
    (:class:`MultiCellSessionCompactor`). Replaces the W23 verbose
    digest+delta pair with a fixed-size :class:`SessionCompactEnvelope`
    on cells beyond ``compact_window``; visible-token cost stays
    O(window) regardless of session length.

  * **W24-2 Intra-cell resample-quorum**
    (:class:`ResampleQuorumCachingOracleAdapter`). Resamples a wrapped
    oracle M times within one cell and returns the majority. Mitigates
    intra-cell probabilistic drift the PER_CELL_NONCE policy cannot
    touch.

  * **W24-3 Real cross-process producer/decoder wire**
    (:class:`CrossProcessProducerDecoderWire`). Spawns a Python
    subprocess that round-trips JSON envelopes via stdin/stdout pipes.
    Real OS-level wire — strictly stronger proxy for the cross-host
    claim than the W23 within-process round-trip.

Phase-71 sub-banks
==================

* **R-71-LONG-SESSION** (W24-1 efficiency anchor; default config).
  16-cell session through one persistent compactor; cells 0..2 are
  ``W24_BRANCH_BELOW_WINDOW`` (W24 reduces to W23 on those); cell
  3..15 fire ``W24_BRANCH_COMPACT_RESOLVED``. **Strict success bar**:
  ``mean_n_w24_visible_tokens_to_decider`` strictly below the W23
  baseline AND ``correctness_ratified_rate = 1.000`` AND
  ``compact_verifies_ok_rate = 1.000`` on cells beyond the window.

* **R-71-LONG-SESSION-SUPER-TOKEN** (W24-1 + W23 super-token
  composition).  Same 16-cell session BUT the inner W23 fires
  ``use_super_token = True``.  W24 *still* compacts on cells beyond
  the window — the W24-1 win composes multiplicatively with the W23
  super-token win on the visible-token axis.

* **R-71-INTRA-CELL-FLIP** (W24-2 mitigation anchor).  An
  :class:`IntraCellFlippingOracle` is added to the registry; on its
  FIRST consult per cell it returns a decoy-asymmetric reply; on
  subsequent consults within the same cell it returns gold-asymmetric.
  Two regimes:
    - ``capsule_w23_quorum_keyed`` (PER_CELL_NONCE only): on every
      cell the *first* sample is decoy → quorum may form on decoy.
    - ``capsule_w24_resample_quorum`` (M=3 resample): cells with
      sample #1 = decoy + #2,3 = gold form quorum on gold.
  **Strict success bar**:
  ``correctness_ratified_rate(capsule_w24_resample_quorum) >
   correctness_ratified_rate(capsule_w23_quorum_keyed)`` (intra-cell
  mitigation is measurable). Names W24-2.

* **R-71-CROSS-PROCESS** (W24-3 real-wire anchor).  Same as
  R-71-LONG-SESSION but the W24 layer carries a real
  :class:`CrossProcessProducerDecoderWire` through which every
  compact envelope is round-tripped via subprocess pipes. **Strict
  requirement**: ``cross_process_round_trip_bytes_total > 0`` AND
  ``cross_process_failures = 0`` AND
  ``correctness_ratified_rate = 1.000``.

* **R-71-NO-COMPACT** (W24-Λ-no-compact named falsifier).  Short
  session with chain reset every cell — no cells exceed the
  ``compact_window`` threshold; W24 fires ``W24_BRANCH_BELOW_WINDOW``
  on every cell and reduces to W23 byte-for-byte. Names the
  structural limit when there is no multi-cell state to compact.

* **R-71-COMPACT-TAMPERED** (W24-3 trust-boundary falsifier).  The
  controller's ``verifier_window_cids_override`` is set to a phantom
  tuple after the genesis-plus-window cells.  Every post-genesis
  compact envelope fires ``window_cids_mismatch`` →
  ``W24_BRANCH_COMPACT_REJECTED`` → fall through to W23 byte-for-byte.

Theorem family W24 (minted by this milestone)
=============================================

* **W24-1** *(proved-conditional + proved-empirical)*. On
  R-71-LONG-SESSION, the :class:`MultiCellSessionCompactor` strictly
  reduces ``mean_n_w24_visible_tokens_to_decider`` over the W23
  baseline on cells beyond ``compact_window`` AND records
  ``compact_verifies_ok_rate = 1.000`` AND preserves
  ``accuracy_full`` byte-for-byte. Stable across 5/5 seeds.
* **W24-2** *(proved-empirical, mitigation)*. On
  R-71-INTRA-CELL-FLIP, the :class:`ResampleQuorumCachingOracleAdapter`
  achieves strictly higher ``correctness_ratified_rate`` than the
  W23 PER_CELL_NONCE baseline. Empirically discharges
  W23-C-MITIGATION-LIVE-VARIANCE on the synthetic intra-cell pattern.
* **W24-3** *(proved-empirical n=8 + proved by inspection)*.
  Trust-boundary soundness. On R-71-COMPACT-TAMPERED, every tampered
  window is rejected by the controller; the W24 layer fires
  :data:`W24_BRANCH_COMPACT_REJECTED` and the W23 answer is preserved
  byte-for-byte. The verification function is short and the failure
  modes are enumerated; soundness holds by inspection.
* **W24-Λ-no-compact** *(proved-empirical, named falsifier)*. On
  R-71-NO-COMPACT, ``mean_n_w24_visible_tokens_to_decider == W23
  baseline`` by construction (W24 reduces to W23). Names the
  structural limit when the chain length stays below the window.
* **W24-Λ-real** *(proved-conditional + empirical-research,
  partially discharged)*. The cross-process wire is the strongest
  cross-process honesty this repo can validate on Mac-1 alone. Mac
  2 unreachable for 18 milestones; no true two-host execution
  validated in SDK v3.25.

CLI
---

::

    # R-71-LONG-SESSION (W24-1 anchor):
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank long_session --decoder-budget -1 \\
        --K-auditor 12 --n-eval 16 --out -

    # R-71-INTRA-CELL-FLIP (W24-2 mitigation anchor):
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank intra_cell_flip --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-71-CROSS-PROCESS (W24-3 real-wire anchor):
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank cross_process --decoder-budget -1 \\
        --K-auditor 12 --n-eval 16 --out -

    # R-71-COMPACT-TAMPERED (W24-3 trust falsifier):
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank compact_tampered --decoder-budget -1 \\
        --K-auditor 12 --n-eval 16 --out -

    # R-71-NO-COMPACT (W24-Λ-no-compact named falsifier):
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank no_compact --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --cross-regime-synthetic --K-auditor 12 --n-eval 16 --out -

    # Seed-stability sweep on the headline regime:
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank long_session --seed-sweep \\
        --K-auditor 12 --n-eval 16 --out -

    # Live LLM probe (Mac-1 Ollama):
    python3 -m vision_mvp.experiments.phase71_session_compaction \\
        --bank intra_cell_flip --live-llm-adjudicator \\
        --adjudicator-model mixtral:8x7b \\
        --K-auditor 12 --n-eval 4 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ROLE_AUDITOR,
    grade_answer,
)
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, AttentionAwareBundleDecoder,
    BundleContradictionDisambiguator,
    CachingOracleAdapter,
    ChangeHistoryOracle,
    CompromisedServiceGraphOracle,
    CrossCellDeltaDisambiguator,
    CrossHostProducerDecoderProxy,
    CrossProcessProducerDecoderWire,
    FifoAdmissionPolicy,
    IntraCellFlippingOracle,
    LatentDigestDisambiguator,
    MultiCellSessionCompactor,
    OracleRegistration,
    OutsideQuery, OutsideVerdict,
    QuorumKeyedSharedReadCache,
    QuorumKeyedCachingOracleAdapter,
    RelationalCompatibilityDisambiguator,
    ResampleQuorumCachingOracleAdapter,
    RoleBudget,
    SchemaCapsule, ServiceGraphOracle, SharedReadCache,
    SessionCompactEnvelope,
    SuperTokenRegistry,
    TeamCoordinator, audit_team_lifecycle,
    TrustWeightedMultiOracleDisambiguator,
    build_incident_triage_schema_capsule,
    collect_admitted_handoffs, _DecodedHandoff,
    verify_session_compact,
    CACHE_FRESHNESS_BYTE_IDENTICAL,
    CACHE_FRESHNESS_PER_CELL_NONCE,
    W22_BRANCH_LATENT_RESOLVED,
    W23_BRANCH_DELTA_RESOLVED, W23_BRANCH_GENESIS,
    W24_BRANCH_COMPACT_RESOLVED, W24_BRANCH_COMPACT_REJECTED,
    W24_BRANCH_BELOW_WINDOW, W24_BRANCH_NO_TRIGGER,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    MultiRoundScenario, _as_incident_scenario,
)
from vision_mvp.experiments.phase66_deceptive_ambiguity import (
    _build_round_candidates_p66,
)
from vision_mvp.experiments.phase67_outside_information import (
    _bench_property_p67, _P67_EXPECTED_SHAPE,
)
from vision_mvp.experiments.phase68_multi_oracle_adjudication import (
    build_phase68_bank,
)


# =============================================================================
# Phase-71 banks — same R-69-CACHE-FANOUT bundle ecology as R-70; the
# *session topology* (long vs reset-per-cell), the *intra-cell oracle
# pattern*, the *real-wire toggle*, and the *verifier window override*
# are the bank-specific dimensions.
# =============================================================================

_VALID_BANKS_P71 = (
    "long_session",                   # W24-1 efficiency anchor.
    "long_session_super_token",       # W24-1 + W23 super-token.
    "intra_cell_flip",                 # W24-2 mitigation anchor.
    "cross_process",                   # W24-3 real-wire anchor.
    "no_compact",                      # W24-Λ-no-compact (single-cell sessions).
    "compact_tampered",                # W24-3 trust-boundary falsifier.
)


_R71_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("capsule_w22_hybrid",         "w22_hybrid"),
    ("capsule_w23_delta",          "w23_delta"),
    ("capsule_w23_quorum_keyed",   "w23_quorum_keyed"),
    ("capsule_w24_compact",        "w24_compact"),
    ("capsule_w24_resample_quorum","w24_resample_quorum"),
)


def _make_factory_p71(name: str, priorities, budgets):
    def fac(round_idx: int = 1, cands=None,
             ) -> dict[str, AdmissionPolicy]:
        return {r: FifoAdmissionPolicy() for r in budgets}
    return fac


def _round_hint_from_ledger(ledger: CapsuleLedger,
                              role_view_cids: Sequence[str]
                              ) -> list[int]:
    seen: set[str] = set()
    hint: list[int] = []
    for cid in role_view_cids:
        if not cid or cid not in ledger:
            continue
        cap = ledger.get(cid)
        r_idx = (cap.payload.get("round", 0)
                   if isinstance(cap.payload, dict) else 0)
        for p in cap.parents:
            if p in seen or p not in ledger:
                continue
            if ledger.get(p).kind != CapsuleKind.TEAM_HANDOFF:
                continue
            seen.add(p)
            hint.append(int(r_idx) or 1)
    return hint


def _build_per_round(union, round_index_hint):
    if round_index_hint is None:
        return [list(union)]
    max_round = max(round_index_hint) if round_index_hint else 1
    per_round = [[] for _ in range(max(1, int(max_round)))]
    for h, ridx in zip(union, round_index_hint):
        slot = max(0, int(ridx) - 1)
        while slot >= len(per_round):
            per_round.append([])
        per_round[slot].append(h)
    return per_round


def _build_w22_stack(*, T_decoder, oracle_registrations,
                       quorum_min, min_trust_sum,
                       schema, cache):
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    w22 = LatentDigestDisambiguator(
        inner=w21, schema=schema, cache=cache)
    return w22, inner_w15


def _decode_with_w22_only(union, T_decoder, round_index_hint,
                            oracle_registrations, *, schema, cache,
                            quorum_min=2, min_trust_sum=0.0):
    w22, inner_w15 = _build_w22_stack(
        T_decoder=T_decoder, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum,
        schema=schema, cache=cache)
    per_round = _build_per_round(union, round_index_hint)
    ans = w22.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
    if "multi_oracle" in ans:
        stats["multi_oracle"] = ans["multi_oracle"]
    if "latent_hybrid" in ans:
        stats["latent_hybrid"] = ans["latent_hybrid"]
    return ans, stats


def _decode_with_w23(union, T_decoder, round_index_hint,
                       oracle_registrations, *, schema, cache,
                       quorum_min=2, min_trust_sum=0.0,
                       w23: CrossCellDeltaDisambiguator):
    w22, inner_w15 = _build_w22_stack(
        T_decoder=T_decoder, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum,
        schema=schema, cache=cache)
    w23.inner = w22
    w23.schema = schema
    per_round = _build_per_round(union, round_index_hint)
    ans = w23.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
    for k in ("multi_oracle", "latent_hybrid", "session_delta_hybrid",
                "session_delta_envelope", "session_digest_envelope",
                "super_token_reference"):
        if k in ans:
            stats[k] = ans[k]
    return ans, stats


def _decode_with_w24(union, T_decoder, round_index_hint,
                       oracle_registrations, *, schema, cache,
                       quorum_min=2, min_trust_sum=0.0,
                       w24: MultiCellSessionCompactor):
    """Wraps W23 inside W24 (compactor)."""
    w22, inner_w15 = _build_w22_stack(
        T_decoder=T_decoder, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum,
        schema=schema, cache=cache)
    w24.inner.inner = w22
    w24.inner.schema = schema
    w24.schema = schema
    per_round = _build_per_round(union, round_index_hint)
    ans = w24.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
    for k in ("multi_oracle", "latent_hybrid", "session_delta_hybrid",
                "session_delta_envelope", "session_digest_envelope",
                "super_token_reference",
                "session_compact_hybrid", "session_compact_envelope"):
        if k in ans:
            stats[k] = ans[k]
    return ans, stats


def _build_oracles_for_bank_p71(*, bank: str,
                                   shared_cache: SharedReadCache,
                                   quorum_keyed_cache: QuorumKeyedSharedReadCache,
                                   intra_cell_oracle: IntraCellFlippingOracle | None,
                                   resample_count: int,
                                   with_compromised: bool = True,
                                   with_honest_oracles: bool = True,
                                   llm_adjudicator: Any | None = None,
                                   ):
    """Build (w22_regs, w23_regs, w24_regs).  The W24 path uses
    :class:`ResampleQuorumCachingOracleAdapter` on the intra-cell
    oracle, so M consults per cell hit the inner oracle directly.
    Deterministic oracles in the W24 path are the same caching
    adapters as W23 (no resampling needed).

    On the ``intra_cell_flip`` bank, ``with_honest_oracles = False``
    so that the flipping oracle's vote is decisive in the W21
    multi-oracle quorum (otherwise service_graph + change_history
    would form gold quorum independently and the intra-cell drift
    would be invisible at the W21 layer)."""
    inner_compromised = CompromisedServiceGraphOracle(
        oracle_id="compromised_registry")
    inner_service_graph = ServiceGraphOracle(oracle_id="service_graph")
    inner_change_history = ChangeHistoryOracle(oracle_id="change_history")

    # W22 registrations (BYTE_IDENTICAL via SharedReadCache).
    w22_regs: list[OracleRegistration] = []
    if with_compromised:
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=inner_compromised, cache=shared_cache,
                oracle_id="compromised_registry"),
            trust_prior=0.8, role_label="compromised_registry"))
    if with_honest_oracles:
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=inner_service_graph, cache=shared_cache,
                oracle_id="service_graph"),
            trust_prior=1.0, role_label="service_graph"))
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=inner_change_history, cache=shared_cache,
                oracle_id="change_history"),
            trust_prior=1.0, role_label="change_history"))
    if intra_cell_oracle is not None:
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=intra_cell_oracle, cache=shared_cache,
                oracle_id="intra_cell_flipping"),
            trust_prior=0.7, role_label="intra_cell_flipping"))
    if llm_adjudicator is not None:
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=llm_adjudicator, cache=shared_cache,
                oracle_id=getattr(llm_adjudicator, "oracle_id",
                                     "llm_adjudicator")),
            trust_prior=0.7, role_label="llm_adjudicator"))

    # W23 registrations (QuorumKeyedSharedReadCache; PER_CELL_NONCE on
    # probabilistic oracles, BYTE_IDENTICAL on deterministic ones).
    quorum_keyed_cache.set_policy(
        "compromised_registry", CACHE_FRESHNESS_BYTE_IDENTICAL)
    quorum_keyed_cache.set_policy(
        "service_graph", CACHE_FRESHNESS_BYTE_IDENTICAL)
    quorum_keyed_cache.set_policy(
        "change_history", CACHE_FRESHNESS_BYTE_IDENTICAL)
    if intra_cell_oracle is not None:
        quorum_keyed_cache.set_policy(
            "intra_cell_flipping", CACHE_FRESHNESS_PER_CELL_NONCE)
    if llm_adjudicator is not None:
        quorum_keyed_cache.set_policy(
            getattr(llm_adjudicator, "oracle_id", "llm_adjudicator"),
            CACHE_FRESHNESS_PER_CELL_NONCE)

    w23_regs: list[OracleRegistration] = []
    if with_compromised:
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=inner_compromised, cache=quorum_keyed_cache,
                oracle_id="compromised_registry"),
            trust_prior=0.8, role_label="compromised_registry"))
    if with_honest_oracles:
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=inner_service_graph, cache=quorum_keyed_cache,
                oracle_id="service_graph"),
            trust_prior=1.0, role_label="service_graph"))
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=inner_change_history, cache=quorum_keyed_cache,
                oracle_id="change_history"),
            trust_prior=1.0, role_label="change_history"))
    if intra_cell_oracle is not None:
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=intra_cell_oracle, cache=quorum_keyed_cache,
                oracle_id="intra_cell_flipping"),
            trust_prior=0.7, role_label="intra_cell_flipping"))
    if llm_adjudicator is not None:
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=llm_adjudicator, cache=quorum_keyed_cache,
                oracle_id=getattr(llm_adjudicator, "oracle_id",
                                     "llm_adjudicator")),
            trust_prior=0.7, role_label=getattr(
                llm_adjudicator, "oracle_id", "llm_adjudicator")))

    # W24 registrations: deterministic oracles use the QuorumKeyed
    # caching adapter (same as W23 — they don't need resampling); the
    # intra-cell oracle uses the W24 :class:`ResampleQuorumCachingOracleAdapter`
    # with M = ``resample_count``. The cache is the same instance so
    # cross-cell wire savings on deterministic oracles are preserved.
    w24_regs: list[OracleRegistration] = []
    if with_compromised:
        w24_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=inner_compromised, cache=quorum_keyed_cache,
                oracle_id="compromised_registry"),
            trust_prior=0.8, role_label="compromised_registry"))
    if with_honest_oracles:
        w24_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=inner_service_graph, cache=quorum_keyed_cache,
                oracle_id="service_graph"),
            trust_prior=1.0, role_label="service_graph"))
        w24_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=inner_change_history, cache=quorum_keyed_cache,
                oracle_id="change_history"),
            trust_prior=1.0, role_label="change_history"))
    if intra_cell_oracle is not None:
        w24_regs.append(OracleRegistration(
            oracle=ResampleQuorumCachingOracleAdapter(
                inner=intra_cell_oracle, cache=quorum_keyed_cache,
                oracle_id="intra_cell_flipping",
                sample_count=int(resample_count),
                majority_threshold=(int(resample_count) + 1) // 2),
            trust_prior=0.7, role_label="intra_cell_flipping"))
    if llm_adjudicator is not None:
        w24_regs.append(OracleRegistration(
            oracle=ResampleQuorumCachingOracleAdapter(
                inner=llm_adjudicator, cache=quorum_keyed_cache,
                oracle_id=getattr(llm_adjudicator, "oracle_id",
                                     "llm_adjudicator"),
                sample_count=int(resample_count),
                majority_threshold=(int(resample_count) + 1) // 2),
            trust_prior=0.7, role_label=getattr(
                llm_adjudicator, "oracle_id", "llm_adjudicator")))

    return tuple(w22_regs), tuple(w23_regs), tuple(w24_regs)


def _set_cell_nonce_on_regs(regs: tuple[OracleRegistration, ...],
                              cell_nonce: str) -> None:
    for reg in regs:
        oracle = reg.oracle
        if hasattr(oracle, "cell_nonce"):
            oracle.cell_nonce = str(cell_nonce)


def _run_capsule_strategy_p71(
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands, round2_cands,
        T_decoder: int | None = None,
        oracle_registrations_w22: tuple[OracleRegistration, ...] = (),
        oracle_registrations_w23: tuple[OracleRegistration, ...] = (),
        oracle_registrations_w24: tuple[OracleRegistration, ...] = (),
        quorum_min: int = 2,
        min_trust_sum: float = 0.0,
        schema: SchemaCapsule,
        shared_cache: SharedReadCache,
        quorum_keyed_cache: QuorumKeyedSharedReadCache,
        cell_index: int = 0,
        cell_nonce: str = "",
        w23_persistent: CrossCellDeltaDisambiguator | None = None,
        w24_persistent: MultiCellSessionCompactor | None = None,
        w24_resample_persistent: MultiCellSessionCompactor | None = None,
        bank: str = "long_session",
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase71_session_compaction",
    )
    coord.advance_round(1)
    for (src, to, kind, payload, _evs) in round1_cands:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv1 = coord.role_view_cid(ROLE_AUDITOR)

    coord.advance_round(1)
    coord.policy_per_role = policy_per_role_factory(round_idx=2,
                                                       cands=round2_cands)
    for (src, to, kind, payload, _evs) in round2_cands:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv2 = coord.role_view_cid(ROLE_AUDITOR)

    pack_stats: dict[str, Any] = {}
    union = collect_admitted_handoffs(ledger, [rv1, rv2])
    round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))

    if decoder_mode == "w22_hybrid":
        answer, pack_stats = _decode_with_w22_only(
            union, T_decoder=T_decoder, round_index_hint=round_hint,
            oracle_registrations=oracle_registrations_w22,
            schema=schema, cache=shared_cache,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    elif decoder_mode == "w23_delta":
        if w23_persistent is None:
            raise ValueError("w23_persistent required for w23_delta")
        _set_cell_nonce_on_regs(oracle_registrations_w23, cell_nonce)
        w23_persistent.use_super_token = False
        answer, pack_stats = _decode_with_w23(
            union, T_decoder=T_decoder, round_index_hint=round_hint,
            oracle_registrations=oracle_registrations_w22,
            schema=schema, cache=shared_cache,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum,
            w23=w23_persistent)
    elif decoder_mode == "w23_quorum_keyed":
        if w23_persistent is None:
            raise ValueError("w23_persistent required for w23_quorum_keyed")
        _set_cell_nonce_on_regs(oracle_registrations_w23, cell_nonce)
        w23_persistent.use_super_token = False
        answer, pack_stats = _decode_with_w23(
            union, T_decoder=T_decoder, round_index_hint=round_hint,
            oracle_registrations=oracle_registrations_w23,
            schema=schema, cache=quorum_keyed_cache,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum,
            w23=w23_persistent)
    elif decoder_mode == "w24_compact":
        if w24_persistent is None:
            raise ValueError("w24_persistent required for w24_compact")
        _set_cell_nonce_on_regs(oracle_registrations_w23, cell_nonce)
        # On long_session_super_token bank, the inner W23 fires
        # super-token; W24 still compacts on cells beyond the window.
        w24_persistent.inner.use_super_token = (
            bank == "long_session_super_token")
        answer, pack_stats = _decode_with_w24(
            union, T_decoder=T_decoder, round_index_hint=round_hint,
            oracle_registrations=oracle_registrations_w22,
            schema=schema, cache=shared_cache,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum,
            w24=w24_persistent)
    elif decoder_mode == "w24_resample_quorum":
        if w24_resample_persistent is None:
            raise ValueError(
                "w24_resample_persistent required for w24_resample_quorum")
        _set_cell_nonce_on_regs(oracle_registrations_w24, cell_nonce)
        w24_resample_persistent.inner.use_super_token = False
        answer, pack_stats = _decode_with_w24(
            union, T_decoder=T_decoder, round_index_hint=round_hint,
            oracle_registrations=oracle_registrations_w24,
            schema=schema, cache=quorum_keyed_cache,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum,
            w24=w24_resample_persistent)
    else:
        raise ValueError(f"unknown decoder_mode {decoder_mode!r}")

    decision_payload = {
        k: v for k, v in answer.items()
        if k not in (
            "multi_oracle", "latent_hybrid", "latent_envelope",
            "session_delta_hybrid", "session_delta_envelope",
            "session_digest_envelope", "super_token_reference",
            "session_compact_hybrid", "session_compact_envelope",
        )
    }
    coord.seal_team_decision(
        team_role=ROLE_AUDITOR, decision=decision_payload,
        extra_role_view_cids=[rv1] if rv1 and rv1 != rv2 else ())
    audit = audit_team_lifecycle(ledger)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))

    rv2_cap = ledger.get(rv2) if rv2 else None
    n_admitted_r2 = (rv2_cap.payload.get("n_admitted")
                       if rv2_cap is not None
                       and isinstance(rv2_cap.payload, dict) else 0)
    n_tokens_r2 = (rv2_cap.payload.get("n_tokens_admitted", 0)
                     if rv2_cap is not None
                     and isinstance(rv2_cap.payload, dict) else 0)
    admitted_kinds: set[tuple[str, str]] = set()
    for cid in (rv1, rv2):
        if not cid or cid not in ledger:
            continue
        cap = ledger.get(cid)
        for p in cap.parents:
            if p in ledger:
                handoff = ledger.get(p)
                if handoff.kind != CapsuleKind.TEAM_HANDOFF:
                    continue
                payload = (handoff.payload
                            if isinstance(handoff.payload, dict) else {})
                admitted_kinds.add((str(payload.get("source_role", "")),
                                     str(payload.get("claim_kind", ""))))
    required = {(role, kind)
                 for (role, kind, _p, _evs) in incident_sc.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    result = StrategyResult(
        strategy=strategy_name,
        scenario_id=sc.scenario_id,
        answer=answer,
        grading=grading,
        failure_kind=failure_kind,
        n_admitted_auditor=int(n_admitted_r2 or 0),
        n_dropped_auditor_budget=0,
        n_dropped_auditor_capacity=0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=coord.stats()["n_team_handoff"],
        n_role_view=coord.stats()["n_role_view"],
        n_team_decision=coord.stats()["n_team_decision"],
        audit_ok=audit.is_ok(),
        n_tokens_admitted=int(n_tokens_r2 or 0),
    )
    return result, pack_stats


def _maybe_install_compact_tamper(
        w24: MultiCellSessionCompactor) -> None:
    """For R-71-COMPACT-TAMPERED: install a phantom window override
    so the controller's :func:`verify_session_compact` fires
    ``window_cids_mismatch`` on every post-genesis cell."""
    # We need a window of length compact_window-1; install all-zero
    # CIDs of the right shape so the size check passes but the cids
    # check fails. Setting after the chain has at least 1 cell is
    # safe.
    K = max(1, int(w24.compact_window))
    w24.verifier_window_cids_override = tuple(["0" * 64] * (K - 1))


def build_phase71_bank(*, n_replicates: int = 4, seed: int = 11
                          ) -> list[MultiRoundScenario]:
    """Build the Phase-71 bank — same R-69-CACHE-FANOUT bundle shape as
    Phase 70. The session length, oracle pattern, and verifier overrides
    are bank-specific."""
    return build_phase68_bank(n_replicates=n_replicates, seed=seed)


def run_phase71(*,
                  bank: str = "long_session",
                  n_eval: int | None = None,
                  K_auditor: int = 12,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 4,
                  T_decoder: int | None = None,
                  quorum_min: int = 2,
                  min_trust_sum: float = 0.0,
                  llm_adjudicator: Any | None = None,
                  use_cross_host_proxy: bool = True,
                  use_cross_process_wire: bool = False,
                  super_token_hex_prefix_len: int = 16,
                  compact_window: int = 4,
                  resample_count: int = 3,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 71 over one of the named banks."""
    if bank not in _VALID_BANKS_P71:
        raise ValueError(
            f"unknown bank {bank!r}; valid: {_VALID_BANKS_P71}")
    bank_obj = build_phase71_bank(
        n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank_obj = bank_obj[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    schema = build_incident_triage_schema_capsule()

    shared_cache = SharedReadCache()
    quorum_keyed_cache = QuorumKeyedSharedReadCache()

    intra_cell_oracle: IntraCellFlippingOracle | None = None
    # On the intra_cell_flip bank we give EACH strategy its own
    # IntraCellFlippingOracle + QuorumKeyedSharedReadCache instance so
    # the strategies don't pollute each other's cache state — that way
    # the W23-PER_CELL_NONCE and W24-RESAMPLE comparisons are clean.
    intra_cell_oracle_per_strategy: dict[str, IntraCellFlippingOracle] = {}
    quorum_keyed_cache_per_strategy: dict[str, QuorumKeyedSharedReadCache] = {}
    shared_cache_per_strategy: dict[str, SharedReadCache] = {}

    if bank == "intra_cell_flip":
        # Pattern: every cell's consult #1 is BAD (decoy); consult
        # #2..M are GOOD (gold). With per-strategy oracles AND
        # per-strategy caches, each strategy's behaviour is
        # independent of the others.
        for strategy_name, _dmode in _R71_STRATEGIES:
            intra_cell_oracle_per_strategy[strategy_name] = (
                IntraCellFlippingOracle(
                    bad_consult_indices=frozenset({1})))
            quorum_keyed_cache_per_strategy[strategy_name] = (
                QuorumKeyedSharedReadCache())
            shared_cache_per_strategy[strategy_name] = SharedReadCache()
        # Pick the first strategy's oracle as the public reference
        # (used by oracle-build and by external callers that need
        # one).
        intra_cell_oracle = (
            intra_cell_oracle_per_strategy["capsule_w22_hybrid"])

    # On the intra_cell_flip bank, isolate the flipping oracle's
    # vote in the W21 multi-oracle quorum (otherwise honest oracles
    # mask the intra-cell drift).
    if bank == "intra_cell_flip":
        with_honest_oracles_bank = False
        with_compromised_bank = False
        quorum_min_bank = 1
    else:
        with_honest_oracles_bank = True
        with_compromised_bank = True
        quorum_min_bank = int(quorum_min)

    w22_regs, w23_regs, w24_regs = _build_oracles_for_bank_p71(
        bank=bank, shared_cache=shared_cache,
        quorum_keyed_cache=quorum_keyed_cache,
        intra_cell_oracle=intra_cell_oracle,
        resample_count=resample_count,
        with_compromised=with_compromised_bank,
        with_honest_oracles=with_honest_oracles_bank,
        llm_adjudicator=llm_adjudicator)

    cross_host_proxy = (
        CrossHostProducerDecoderProxy() if use_cross_host_proxy else None)
    cross_process_wire = None
    if use_cross_process_wire or bank == "cross_process":
        cross_process_wire = CrossProcessProducerDecoderWire()
        cross_process_wire.start()

    # Persistent W23 (delta only) instance for the W23 baseline.
    w23_delta_persistent = CrossCellDeltaDisambiguator(
        inner=LatentDigestDisambiguator(),
        schema=schema,
        super_token_registry=SuperTokenRegistry(),
        quorum_keyed_cache=None,
        cross_host_proxy=cross_host_proxy,
        super_token_hex_prefix_len=super_token_hex_prefix_len,
        use_super_token=False)

    # Persistent W24 compactor instance.
    w24_inner_w23 = CrossCellDeltaDisambiguator(
        inner=LatentDigestDisambiguator(),
        schema=schema,
        super_token_registry=SuperTokenRegistry(),
        quorum_keyed_cache=None,
        cross_host_proxy=cross_host_proxy,
        super_token_hex_prefix_len=super_token_hex_prefix_len,
        use_super_token=False)
    w24_compact_persistent = MultiCellSessionCompactor(
        inner=w24_inner_w23,
        schema=schema,
        compact_window=int(compact_window),
        cross_process_wire=cross_process_wire)

    # Persistent W24 resample-quorum instance (for W24-2 anchor).
    w24_resample_inner_w23 = CrossCellDeltaDisambiguator(
        inner=LatentDigestDisambiguator(),
        schema=schema,
        super_token_registry=SuperTokenRegistry(),
        quorum_keyed_cache=quorum_keyed_cache,
        cross_host_proxy=cross_host_proxy,
        super_token_hex_prefix_len=super_token_hex_prefix_len,
        use_super_token=False)
    w24_resample_persistent = MultiCellSessionCompactor(
        inner=w24_resample_inner_w23,
        schema=schema,
        compact_window=int(compact_window),
        cross_process_wire=None)

    def _post_cell_hook(strategy_name: str, cell_index: int) -> None:
        if (bank == "compact_tampered"
                and cell_index >= int(compact_window) - 1
                and strategy_name == "capsule_w24_compact"):
            _maybe_install_compact_tamper(w24_compact_persistent)

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R71_STRATEGIES
    }

    for cell_index, sc in enumerate(bank_obj):
        round1_cands = _build_round_candidates_p66(sc.round1_emissions)
        round2_cands = _build_round_candidates_p66(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p67(
            sc, round1_cands, round2_cands)
        cell_nonce = f"cell_{cell_index}_seed_{bank_seed}"

        # On the no_compact bank, RESET every persistent W24 chain
        # after every cell so the chain stays at length 1 and the
        # compactor never fires (W24-Λ-no-compact).
        if bank == "no_compact":
            w23_delta_persistent.reset_session()
            w24_compact_persistent.reset_session()
            w24_resample_persistent.reset_session()

        # On intra_cell_flip, reset every strategy's private oracle's
        # consult counter at the start of the cell so the *first
        # sample of each cell* is the bad one (intra-cell pattern).
        if bank == "intra_cell_flip":
            for o in intra_cell_oracle_per_strategy.values():
                o.n_consults = 0

        for (sname, dmode) in _R71_STRATEGIES:
            fac = _make_factory_p71(sname, priorities, budgets)
            # On intra_cell_flip, build fresh per-strategy
            # oracle/cache registrations so each strategy is
            # independent (no cache pollution between strategies).
            if bank == "intra_cell_flip":
                strat_w22_regs, strat_w23_regs, strat_w24_regs = (
                    _build_oracles_for_bank_p71(
                        bank=bank,
                        shared_cache=shared_cache_per_strategy[sname],
                        quorum_keyed_cache=
                            quorum_keyed_cache_per_strategy[sname],
                        intra_cell_oracle=
                            intra_cell_oracle_per_strategy[sname],
                        resample_count=resample_count,
                        with_compromised=with_compromised_bank,
                        with_honest_oracles=with_honest_oracles_bank,
                        llm_adjudicator=llm_adjudicator))
                strat_shared_cache = shared_cache_per_strategy[sname]
                strat_quorum_keyed_cache = (
                    quorum_keyed_cache_per_strategy[sname])
            else:
                strat_w22_regs = w22_regs
                strat_w23_regs = w23_regs
                strat_w24_regs = w24_regs
                strat_shared_cache = shared_cache
                strat_quorum_keyed_cache = quorum_keyed_cache
            r, ps = _run_capsule_strategy_p71(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands, round2_cands=round2_cands,
                T_decoder=T_decoder,
                oracle_registrations_w22=strat_w22_regs,
                oracle_registrations_w23=strat_w23_regs,
                oracle_registrations_w24=strat_w24_regs,
                quorum_min=quorum_min_bank, min_trust_sum=min_trust_sum,
                schema=schema,
                shared_cache=strat_shared_cache,
                quorum_keyed_cache=strat_quorum_keyed_cache,
                cell_index=cell_index, cell_nonce=cell_nonce,
                w23_persistent=w23_delta_persistent,
                w24_persistent=w24_compact_persistent,
                w24_resample_persistent=w24_resample_persistent,
                bank=bank)
            results.append(r)
            if ps:
                pack_stats_per_strategy[sname].append({
                    "scenario_id": sc.scenario_id,
                    "cell_index": cell_index,
                    **ps,
                })
            _post_cell_hook(sname, cell_index)

    # Cleanup cross-process wire if started.
    cross_process_stats: dict[str, Any] = {}
    if cross_process_wire is not None:
        cross_process_stats = cross_process_wire.stats()
        cross_process_wire.stop()

    strategy_names = tuple(s[0] for s in _R71_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w24_compact_minus_w23_delta": _gap(
            "capsule_w24_compact", "capsule_w23_delta"),
        "w24_resample_minus_w23_quorum": _gap(
            "capsule_w24_resample_quorum", "capsule_w23_quorum_keyed"),
        "w24_compact_minus_w22": _gap(
            "capsule_w24_compact", "capsule_w22_hybrid"),
    }

    audit_ok_grid: dict[str, bool] = {}
    for s in strategy_names:
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    bench_summary = {
        "n_scenarios": len(bench_property_per_scenario),
        "scenarios_with_symmetric_corroboration": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("symmetric_corroboration_holds")),
        "scenarios_with_expected_shape": sum(
            1 for v in bench_property_per_scenario.values()
            if tuple(v.get("shape", ())) == _P67_EXPECTED_SHAPE),
        "K_auditor": K_auditor,
        "T_decoder": T_decoder,
        "bank": bank,
        "compact_window": int(compact_window),
        "resample_count": int(resample_count),
        "quorum_min": int(quorum_min),
        "min_trust_sum": float(min_trust_sum),
        "schema_cid": schema.cid,
        "schema_n_canonical_tokens": schema.n_canonical_tokens,
        "n_oracle_registrations_w22": len(w22_regs),
        "n_oracle_registrations_w23": len(w23_regs),
        "n_oracle_registrations_w24": len(w24_regs),
        "use_cross_host_proxy": bool(use_cross_host_proxy),
        "use_cross_process_wire": bool(cross_process_wire is not None),
        "super_token_hex_prefix_len": int(super_token_hex_prefix_len),
        "intra_cell_oracle_present": intra_cell_oracle is not None,
    }

    def _agg_w24(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)

        def get_sc(r):
            return r.get("session_compact_hybrid") or {}

        def get_sd(r):
            return r.get("session_delta_hybrid") or {}

        def get_lh(r):
            return r.get("latent_hybrid") or {}

        s_w24_visible = sum(
            int(get_sc(r).get("n_w24_visible_tokens_to_decider", 0))
            for r in rows)
        s_w23_visible = sum(
            int(get_sc(r).get("n_w23_visible_tokens_to_decider", 0))
            or int(get_sd(r).get("n_w23_visible_tokens_to_decider", 0))
            or int(get_lh(r).get("n_visible_tokens_to_decider", 0))
            for r in rows)
        s_savings = sum(
            int(get_sc(r).get("n_w23_minus_w24_savings", 0))
            for r in rows)
        s_compact_n = sum(
            int(get_sc(r).get("n_compact_tokens", 0)) for r in rows)
        n_compact_resolved = sum(
            1 for r in rows
            if get_sc(r).get("decoder_branch") == W24_BRANCH_COMPACT_RESOLVED)
        n_compact_rejected = sum(
            1 for r in rows
            if get_sc(r).get("decoder_branch") == W24_BRANCH_COMPACT_REJECTED)
        n_below_window = sum(
            1 for r in rows
            if get_sc(r).get("decoder_branch") == W24_BRANCH_BELOW_WINDOW)
        n_no_trigger = sum(
            1 for r in rows
            if get_sc(r).get("decoder_branch") == W24_BRANCH_NO_TRIGGER)
        n_compact_ok = sum(
            1 for r in rows if get_sc(r).get("compact_verification_ok"))
        return {
            "n_cells": n,
            "n_w24_compact_resolved_cells": int(n_compact_resolved),
            "n_w24_compact_rejected_cells": int(n_compact_rejected),
            "n_w24_below_window_cells": int(n_below_window),
            "n_w24_no_trigger_cells": int(n_no_trigger),
            "n_w24_visible_tokens_to_decider_sum": int(s_w24_visible),
            "n_w24_visible_tokens_to_decider_avg": (
                round(s_w24_visible / n, 4) if n else 0.0),
            "n_w23_visible_tokens_to_decider_sum": int(s_w23_visible),
            "n_w23_visible_tokens_to_decider_avg": (
                round(s_w23_visible / n, 4) if n else 0.0),
            "n_w23_minus_w24_savings_sum": int(s_savings),
            "n_w23_minus_w24_savings_avg": (
                round(s_savings / n, 4) if n else 0.0),
            "compact_n_tokens_avg": (
                round(s_compact_n / n, 4) if n else 0.0),
            "compact_verifies_ok_rate": (
                round(n_compact_ok / n, 4) if n else 0.0),
        }

    def _agg_w23_or_w22(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)

        def get_sd(r):
            return r.get("session_delta_hybrid") or {}

        def get_lh(r):
            return r.get("latent_hybrid") or {}

        s_w23_visible = sum(
            int(get_sd(r).get("n_w23_visible_tokens_to_decider", 0))
            or int(get_lh(r).get("n_visible_tokens_to_decider", 0))
            for r in rows)
        s_w22_visible = sum(
            int(get_sd(r).get("n_w22_visible_tokens_to_decider", 0))
            or int(get_lh(r).get("n_visible_tokens_to_decider", 0))
            for r in rows)
        return {
            "n_cells": n,
            "n_w23_visible_tokens_to_decider_avg": (
                round(s_w23_visible / n, 4) if n else 0.0),
            "n_w22_visible_tokens_to_decider_avg": (
                round(s_w22_visible / n, 4) if n else 0.0),
        }

    pack_stats_summary = {
        s: (_agg_w24(pack_stats_per_strategy.get(s, []))
             if s in ("capsule_w24_compact", "capsule_w24_resample_quorum")
             else _agg_w23_or_w22(pack_stats_per_strategy.get(s, [])))
        for s in strategy_names
    }

    eff_w22 = pack_stats_summary.get("capsule_w22_hybrid", {})
    eff_w23 = pack_stats_summary.get("capsule_w23_delta", {})
    eff_w23_quorum = pack_stats_summary.get(
        "capsule_w23_quorum_keyed", {})
    eff_w24 = pack_stats_summary.get("capsule_w24_compact", {})
    eff_w24_resample = pack_stats_summary.get(
        "capsule_w24_resample_quorum", {})

    eff_compare = {
        "w22_visible_tokens_per_cell":
            float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0)),
        "w23_visible_tokens_per_cell":
            float(eff_w23.get("n_w23_visible_tokens_to_decider_avg", 0.0)),
        "w23_quorum_visible_tokens_per_cell":
            float(eff_w23_quorum.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
        "w24_compact_visible_tokens_per_cell":
            float(eff_w24.get(
                "n_w24_visible_tokens_to_decider_avg", 0.0)),
        "w24_resample_visible_tokens_per_cell":
            float(eff_w24_resample.get(
                "n_w24_visible_tokens_to_decider_avg", 0.0)),
        "w24_compact_savings_per_cell": round(
            float(eff_w24.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0))
            - float(eff_w24.get(
                "n_w24_visible_tokens_to_decider_avg", 0.0)),
            4),
        "w24_compact_savings_pct": (round(
            (float(eff_w24.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0))
             - float(eff_w24.get(
                 "n_w24_visible_tokens_to_decider_avg", 0.0)))
            / max(1.0, float(eff_w24.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)))
            * 100, 4)),
        "compact_verifies_ok_rate":
            float(eff_w24.get("compact_verifies_ok_rate", 0.0)),
        "compact_n_tokens_avg":
            float(eff_w24.get("compact_n_tokens_avg", 0.0)),
        "n_w24_compact_resolved_cells":
            int(eff_w24.get("n_w24_compact_resolved_cells", 0)),
        "n_w24_compact_rejected_cells":
            int(eff_w24.get("n_w24_compact_rejected_cells", 0)),
        "n_w24_below_window_cells":
            int(eff_w24.get("n_w24_below_window_cells", 0)),
        "cross_process_round_trip_bytes_total":
            int(cross_process_stats.get("n_bytes_serialised", 0)),
        "cross_process_round_trips":
            int(cross_process_stats.get("n_round_trips", 0)),
        "cross_process_failures":
            int(cross_process_stats.get("n_failures", 0)),
    }

    correctness_ratified_per_strategy: dict[str, list[bool]] = {}
    for s_name in (
            "capsule_w23_delta", "capsule_w23_quorum_keyed",
            "capsule_w24_compact", "capsule_w24_resample_quorum"):
        cells: list[bool] = []
        for sc in bank_obj:
            w22_r = next((r for r in results
                            if r.strategy == "capsule_w22_hybrid"
                            and r.scenario_id == sc.scenario_id), None)
            other_r = next((r for r in results
                            if r.strategy == s_name
                            and r.scenario_id == sc.scenario_id), None)
            if w22_r is None or other_r is None:
                continue
            sv22 = tuple(sorted(map(str,
                                       w22_r.answer.get("services", ()))))
            sv = tuple(sorted(map(str,
                                       other_r.answer.get("services", ()))))
            cells.append(sv22 == sv)
        correctness_ratified_per_strategy[s_name] = cells

    correctness_ratified_rates = {
        s_name: round(sum(c) / max(1, len(c)), 4)
        for s_name, c in correctness_ratified_per_strategy.items()
    }

    # W24-2 mitigation advantage: w24_resample_quorum vs w23_quorum_keyed.
    intra_cell_mitigation_advantage = round(
        float(pooled["capsule_w24_resample_quorum"]["accuracy_full"])
        - float(pooled["capsule_w23_quorum_keyed"]["accuracy_full"]),
        4)
    # accuracy of w24_compact vs w23_delta (should be 1.000 == on every regime
    # except compact_tampered where w24 falls through to w23).
    compaction_correctness_preserved = round(
        float(pooled["capsule_w24_compact"]["accuracy_full"])
        - float(pooled["capsule_w23_delta"]["accuracy_full"]),
        4)

    if verbose:
        print(f"[phase71] bank={bank} schema_cid={schema.cid[:16]} "
              f"T_decoder={T_decoder} n_eval={len(bank_obj)} "
              f"K_auditor={K_auditor} bank_seed={bank_seed} "
              f"compact_window={compact_window} "
              f"resample_count={resample_count}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase71]   {s:36s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase71] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)
        print(f"[phase71] visible_tokens_per_cell w22="
              f"{eff_compare['w22_visible_tokens_per_cell']:.2f} "
              f"w23={eff_compare['w23_visible_tokens_per_cell']:.2f} "
              f"w24_compact={eff_compare['w24_compact_visible_tokens_per_cell']:.2f}",
              file=sys.stderr, flush=True)
        print(f"[phase71] w24_compact_savings_per_cell="
              f"{eff_compare['w24_compact_savings_per_cell']:.2f} "
              f"w24_compact_savings_pct="
              f"{eff_compare['w24_compact_savings_pct']:+.2f}%",
              file=sys.stderr, flush=True)
        print(f"[phase71] compact_verifies_ok_rate="
              f"{eff_compare['compact_verifies_ok_rate']:.3f} "
              f"intra_cell_mitigation_advantage="
              f"{intra_cell_mitigation_advantage:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase71] cross_process_round_trips="
              f"{eff_compare['cross_process_round_trips']} "
              f"bytes_total="
              f"{eff_compare['cross_process_round_trip_bytes_total']} "
              f"failures="
              f"{eff_compare['cross_process_failures']}",
              file=sys.stderr, flush=True)
        print(f"[phase71] correctness_ratified_rates="
              f"{correctness_ratified_rates}",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase71.session_compaction.v1",
        "config": {
            "bank": bank, "n_eval": len(bank_obj),
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "K_producer": K_producer, "T_producer": T_producer,
            "inbox_capacity": inbox_capacity, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates, "T_decoder": T_decoder,
            "quorum_min": int(quorum_min),
            "min_trust_sum": float(min_trust_sum),
            "use_cross_host_proxy": bool(use_cross_host_proxy),
            "use_cross_process_wire":
                bool(cross_process_wire is not None),
            "super_token_hex_prefix_len": int(super_token_hex_prefix_len),
            "compact_window": int(compact_window),
            "resample_count": int(resample_count),
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "pack_stats_summary": pack_stats_summary,
        "eff_compare": eff_compare,
        "correctness_ratified_rates": correctness_ratified_rates,
        "intra_cell_mitigation_advantage_w24_minus_w23":
            intra_cell_mitigation_advantage,
        "compaction_correctness_preserved":
            compaction_correctness_preserved,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
        "cross_process_stats": cross_process_stats,
    }


def run_phase71_seed_stability_sweep(*,
                                          bank: str = "long_session",
                                          T_decoder: int | None = None,
                                          n_eval: int = 16,
                                          K_auditor: int = 12,
                                          quorum_min: int = 2,
                                          compact_window: int = 4,
                                          resample_count: int = 3,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase71.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "quorum_min": int(quorum_min),
                    "compact_window": int(compact_window),
                    "resample_count": int(resample_count),
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase71(bank=bank, T_decoder=T_decoder,
                            n_eval=n_eval, K_auditor=K_auditor,
                            bank_seed=seed, quorum_min=quorum_min,
                            compact_window=int(compact_window),
                            resample_count=int(resample_count),
                            bank_replicates=max(4, n_eval // 4 + 1))
        out["per_seed"][str(seed)] = {
            "headline_gap": rep["headline_gap"],
            "pooled": {
                k: v["accuracy_full"]
                for k, v in rep["pooled"].items()
            },
            "eff_compare": rep["eff_compare"],
            "correctness_ratified_rates":
                rep["correctness_ratified_rates"],
            "audit_ok_grid": rep["audit_ok_grid"],
        }
    compact_savings = [
        out["per_seed"][str(s)]["eff_compare"][
            "w24_compact_savings_per_cell"] for s in seeds
    ]
    out["min_w24_compact_savings_per_cell"] = (
        min(compact_savings) if compact_savings else 0.0)
    out["mean_w24_compact_savings_per_cell"] = (
        round(sum(compact_savings) / len(compact_savings), 4)
        if compact_savings else 0.0)
    compact_correct = [
        out["per_seed"][str(s)]["correctness_ratified_rates"].get(
            "capsule_w24_compact", 0.0) for s in seeds
    ]
    out["min_w24_compact_correctness_ratified_rate"] = (
        min(compact_correct) if compact_correct else 0.0)
    return out


def run_cross_regime_synthetic_p71(*,
                                      n_eval: int = 16,
                                      bank_seed: int = 11,
                                      K_auditor: int = 12,
                                      T_auditor: int = 256,
                                      T_decoder_tight: int = 24,
                                      quorum_min: int = 2,
                                      compact_window: int = 4,
                                      resample_count: int = 3,
                                      ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase71.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
            "quorum_min": int(quorum_min),
            "compact_window": int(compact_window),
            "resample_count": int(resample_count),
        },
        "regimes": {},
    }
    cells = [
        ("R-71-LONG-SESSION-LOOSE",         "long_session",            None,            16),
        ("R-71-LONG-SESSION-TIGHT",         "long_session",            T_decoder_tight, 16),
        ("R-71-LONG-SESSION-SUPER-TOKEN",  "long_session_super_token", None,            16),
        ("R-71-INTRA-CELL-FLIP",            "intra_cell_flip",         None,            8),
        ("R-71-CROSS-PROCESS",              "cross_process",            None,            16),
        ("R-71-NO-COMPACT",                 "no_compact",               None,            8),
        ("R-71-COMPACT-TAMPERED",           "compact_tampered",         None,            16),
    ]
    for (regime_name, bank, T_dec, regime_n_eval) in cells:
        nev = min(int(n_eval), regime_n_eval)
        rep = run_phase71(
            bank=bank, n_eval=nev, K_auditor=K_auditor,
            T_auditor=T_auditor, T_decoder=T_dec,
            bank_seed=bank_seed, quorum_min=quorum_min,
            bank_replicates=max(4, nev // 4 + 1),
            compact_window=int(compact_window),
            resample_count=int(resample_count))
        out["regimes"][regime_name] = {
            "bank": bank,
            "T_decoder": T_dec,
            "headline_gap": rep["headline_gap"],
            "pooled_accuracy_full": {
                k: v["accuracy_full"]
                for k, v in rep["pooled"].items()
            },
            "eff_compare": rep["eff_compare"],
            "correctness_ratified_rates":
                rep["correctness_ratified_rates"],
            "intra_cell_mitigation_advantage_w24_minus_w23":
                rep["intra_cell_mitigation_advantage_w24_minus_w23"],
            "compaction_correctness_preserved":
                rep["compaction_correctness_preserved"],
            "audit_ok_grid": rep["audit_ok_grid"],
        }
    return out


def _make_llm_adjudicator(model: str, *, base_url: str | None = None,
                           timeout: float = 120.0):
    """Build an :class:`LLMAdjudicatorOracle` against a live Ollama
    backend (Mac-1 default)."""
    from vision_mvp.wevra.team_coord import LLMAdjudicatorOracle
    from vision_mvp.wevra.llm_backend import OllamaBackend
    backend_url = base_url or os.environ.get(
        "WEVRA_OLLAMA_URL", "http://127.0.0.1:11434")
    backend = OllamaBackend(model=model, base_url=backend_url,
                              timeout=timeout)
    return LLMAdjudicatorOracle(
        oracle_id=f"ollama_{model}", backend=backend,
        max_response_tokens=24)


def _cli() -> None:
    ap = argparse.ArgumentParser(prog="phase71_session_compaction")
    ap.add_argument("--bank", default="long_session",
                     choices=_VALID_BANKS_P71)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--K-auditor", type=int, default=12)
    ap.add_argument("--T-auditor", type=int, default=256)
    ap.add_argument("--K-producer", type=int, default=6)
    ap.add_argument("--T-producer", type=int, default=96)
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument("--bank-replicates", type=int, default=4)
    ap.add_argument("--decoder-budget", type=int, default=-1)
    ap.add_argument("--compact-window", type=int, default=4)
    ap.add_argument("--resample-count", type=int, default=3)
    ap.add_argument("--quorum-min", type=int, default=2)
    ap.add_argument("--min-trust-sum", type=float, default=0.0)
    ap.add_argument("--out", default="-")
    ap.add_argument("--seed-sweep", action="store_true")
    ap.add_argument("--cross-regime-synthetic", action="store_true")
    ap.add_argument("--live-llm-adjudicator", action="store_true")
    ap.add_argument("--adjudicator-model", default="mixtral:8x7b")
    ap.add_argument("--adjudicator-base-url", default=None)
    ap.add_argument("--no-cross-host-proxy", action="store_true")
    ap.add_argument("--cross-process-wire", action="store_true",
                     help="enable real subprocess wire (forced on for "
                          "the cross_process bank)")
    ap.add_argument("--super-token-hex-prefix-len", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    T_decoder = None if args.decoder_budget < 0 else int(args.decoder_budget)
    llm_adj = None
    if args.live_llm_adjudicator:
        llm_adj = _make_llm_adjudicator(
            args.adjudicator_model,
            base_url=args.adjudicator_base_url)

    if args.cross_regime_synthetic:
        out = run_cross_regime_synthetic_p71(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            T_decoder_tight=24, quorum_min=args.quorum_min,
            compact_window=args.compact_window,
            resample_count=args.resample_count)
    elif args.seed_sweep:
        out = run_phase71_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            quorum_min=args.quorum_min,
            compact_window=args.compact_window,
            resample_count=args.resample_count)
    else:
        out = run_phase71(
            bank=args.bank, n_eval=args.n_eval,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            K_producer=args.K_producer, T_producer=args.T_producer,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            T_decoder=T_decoder, quorum_min=args.quorum_min,
            min_trust_sum=args.min_trust_sum,
            llm_adjudicator=llm_adj,
            use_cross_host_proxy=not args.no_cross_host_proxy,
            use_cross_process_wire=args.cross_process_wire,
            compact_window=args.compact_window,
            resample_count=args.resample_count,
            super_token_hex_prefix_len=args.super_token_hex_prefix_len,
            verbose=args.verbose)

    if args.out == "-":
        print(json.dumps(out, indent=2, default=str))
    else:
        with open(args.out, "w") as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f"[phase71] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
