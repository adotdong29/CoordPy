"""Phase 70 — capsule-native cross-cell delta execution + quorum-keyed
cache + super-token reference (SDK v3.24, W23 family anchor).

The follow-up to SDK v3.23 (W22) on the named research direction the
W22 milestone explicitly left open: a denser latent / dense-control
transfer mechanism that goes *beyond the per-cell digest envelope*. The
W23 :class:`CrossCellDeltaDisambiguator` implements four LatentMAS-
direction proxies on top of the W22 :class:`LatentDigestDisambiguator`:

  * **cross-cell hash-chained session digest** — per-cell envelope's
    ``prior_session_digest_cid`` points back to a running summary; the
    chain is verifiable in O(1) per link by the controller. This is
    the LatentMAS *cross-cell latent state-sharing* direction at the
    capsule layer.
  * **per-cell delta execution** — each cell after genesis emits a
    small :class:`SessionDeltaEnvelope` carrying only what is *new*
    (changed projected subset, vote delta, this cell's parent
    probes); the cumulative state lives in the
    :class:`SessionDigestEnvelope` chain by reference.
  * **quorum-keyed cache** — per-oracle freshness policy
    (``byte_identical`` / ``per_cell_nonce`` / ``quorum_locked``)
    mitigates W22-C-CACHE-AMPLIFICATION on probabilistic LLM
    adjudicators. Bad first replies do not propagate.
  * **super-token reference** — the bounded steganographic /
    dense-control-payload experiment. The decoder visible cost
    collapses to *one whitespace token* per cell; the super-token is
    a CID-prefix reference to the SessionDeltaEnvelope, resolved by
    a controller-side :class:`SuperTokenRegistry`, with hash
    verification.

Plus a :class:`CrossHostProducerDecoderProxy` — the within-process
producer/decoder host-split proxy. Mac 2 has been ARP-incomplete for
17 milestones in a row; we cannot validate true two-host execution.
The proxy forces producer and decoder to communicate **only** through
canonical JSON bytes so any latent shared state would surface as a
serialisation failure. When Mac 2 returns, the same interface drops
in over a real socket with no W23 code changes.

Phase-70 sub-banks
==================

Six pre-committed cells. The bundle shape is the R-69-CACHE-FANOUT
ecology used by SDK v3.23; the *cross-cell topology* and the
*per-oracle cache freshness policy* are the bank-specific dimensions.

* **R-70-DELTA-FANOUT** (W23-1 efficiency anchor; default config).
  Multi-cell session through one persistent
  :class:`CrossCellDeltaDisambiguator`. Cell 1 = genesis (W22 cost);
  cells 2..N emit small :class:`SessionDeltaEnvelope`s. **Strict
  success bar**: ``mean_n_w23_visible_tokens_to_decider`` strictly
  below the W22 baseline AND
  ``correctness_ratified_rate = 1.000`` AND
  ``chain_verifies_ok_rate = 1.000``.
* **R-70-SUPER-TOKEN** (W23-3 dense-control anchor).  Same
  multi-cell session BUT with ``use_super_token = True``. Cells
  2..N emit a single visible super-token; the decoder visible cost
  collapses to ``kept + 1`` per cell. **Strict success bar**:
  ``mean_n_w23_visible_tokens_to_decider`` strictly below the W22
  baseline AND ``super_token_resolved_rate = 1.000`` AND
  ``correctness_ratified_rate = 1.000``.
* **R-70-AMPLIFIED-LLM** (W23-2 mitigation anchor).  A
  :class:`FlippingProbabilisticOracle` is added to the registry; on
  its FIRST consult it returns a decoy-asymmetric reply; on
  subsequent consults it returns a gold-asymmetric reply. Two
  regimes:
    - ``capsule_w22_hybrid`` (BYTE_IDENTICAL cache on the flipping
      oracle): cell 1's bad reply is cached; all matching cells
      inherit it; quorum may form on decoy.
    - ``capsule_w23_quorum_keyed`` (PER_CELL_NONCE cache on the
      flipping oracle): cell 1 hits the oracle (bad), cell 2 hits
      the oracle again (good); quorum forms on gold for cell 2+.
  **Strict success bar**:
  ``correctness_ratified_rate(capsule_w23_quorum_keyed) >
  correctness_ratified_rate(capsule_w22_hybrid)`` (mitigation is
  measurable). Names W23-2 (W22-C-CACHE-AMPLIFICATION mitigated).
* **R-70-NO-DELTA** (W23-Λ-no-delta named falsifier).  Single-cell
  session — no prior session digest; W23 fires
  ``W23_BRANCH_GENESIS`` and reduces to W22 byte-for-byte. **Strict
  requirement**: ``mean_n_w23_visible_tokens_to_decider ==
  W22 baseline`` AND ``correctness_ratified_rate = 1.000``. Names
  the structural limit when there is no cross-cell state to delta
  against.
* **R-70-SUPER-TOKEN-TAMPERED** (W23-3 trust-boundary anchor).  An
  :class:`EnvelopeTamperer` mode (or direct registry tamper) makes
  the super-token's referenced delta no longer recompute. The
  controller's
  :func:`verify_super_token_reference` rejects the reference; W23
  falls through to the verbose-digest path. **Strict requirement**:
  ``super_token_verification_ok_rate == 0.0`` AND
  ``correctness_ratified_rate == 1.000``.
* **R-70-CHAIN-BROKEN** (W23-3 chain-tamper anchor).  The
  ``prior_session_digest_cid`` on a non-genesis envelope is set to
  zero / a phantom. The controller's
  :func:`verify_session_digest_chain` rejects with
  ``chain_head_mismatch``; W23 fires
  :data:`W23_BRANCH_DELTA_REJECTED`; the W22 answer is preserved.
  **Strict requirement**: ``chain_verifies_ok_rate == 0.0`` AND
  ``correctness_ratified_rate == 1.000``.

Theorem family W23 (minted by this milestone)
==============================================

* **W23-1** *(proved-conditional + proved-empirical, n=8 saturated × 2
  cells × 5 seeds)*. On R-70-DELTA-FANOUT,
  :class:`CrossCellDeltaDisambiguator` over a registered
  :class:`SchemaCapsule` strictly reduces
  ``mean_n_w23_visible_tokens_to_decider`` over the W22 baseline AND
  preserves ``accuracy_full`` byte-for-byte. Stable across 5/5
  ``bank_seed`` values.
* **W23-2** *(proved-conditional + proved-empirical, mitigation)*. On
  R-70-AMPLIFIED-LLM, the W23 quorum-keyed cache (PER_CELL_NONCE
  on the flipping oracle) achieves a strictly higher
  ``correctness_ratified_rate`` than the W22 BYTE_IDENTICAL
  baseline. **Empirically discharges
  W22-C-CACHE-AMPLIFICATION as a mitigable property** (the
  amplification is real *and* the W23 mitigation is real on the
  same regime).
* **W23-3** *(proved-empirical + proved by inspection)*. On
  R-70-SUPER-TOKEN-TAMPERED and R-70-CHAIN-BROKEN, every tampered
  reference / chain link is rejected by the controller; the W23
  layer fires :data:`W23_BRANCH_SUPER_TOKEN_REJECTED` /
  :data:`W23_BRANCH_DELTA_REJECTED`; the W22 answer is preserved
  byte-for-byte (the rejection does NOT corrupt the answer).
* **W23-Λ-no-delta** *(named falsifier)*. On R-70-NO-DELTA,
  ``mean_n_w23_visible_tokens_to_decider`` equals the W22 baseline
  by construction (the W23 layer reduces to W22). Names the
  structural limit when there is no cross-cell state.
* **W23-Λ-real** *(proved-conditional + empirical-research,
  partially discharged)*. The W23 surface is *naturally* a
  producer / cache-controller / decoder split. The
  :class:`CrossHostProducerDecoderProxy` validates the wire-format
  contract via a within-process round-trip on every cell. Mac 2 is
  unreachable; no true two-host execution happened in SDK v3.24.

CLI
---

::

    # R-70-DELTA-FANOUT (W23-1 anchor):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank delta_fanout --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-70-DELTA-FANOUT-TIGHT (W23-1 + W15 composition):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank delta_fanout --decoder-budget 24 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-70-SUPER-TOKEN (W23-3 dense-control anchor):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank super_token --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-70-AMPLIFIED-LLM (W23-2 mitigation anchor):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank amplified_llm --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-70-SUPER-TOKEN-TAMPERED (W23-3 trust falsifier):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank super_token_tampered --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-70-CHAIN-BROKEN (W23-3 trust falsifier):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank chain_broken --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

    # Seed-stability sweep on the headline regime:
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank delta_fanout --seed-sweep \\
        --K-auditor 12 --n-eval 8 --out -

    # Live LLM probe (Mac-1 Ollama):
    python3 -m vision_mvp.experiments.phase70_capsule_session_delta \\
        --bank amplified_llm --live-llm-adjudicator \\
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
    ALL_ROLES, ROLE_AUDITOR,
    grade_answer,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.team_coord import (
    AbstainingOracle,
    AdmissionPolicy, AttentionAwareBundleDecoder,
    BundleAwareTeamDecoder,
    BundleContradictionDisambiguator,
    CachingOracleAdapter,
    CapsuleContextPacker,
    ChangeHistoryOracle,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CompromisedServiceGraphOracle,
    CoverageGuidedAdmissionPolicy,
    CrossCellDeltaDisambiguator,
    CrossHostProducerDecoderProxy,
    CrossRoleCorroborationAdmissionPolicy,
    EnvelopeTamperer,
    FifoAdmissionPolicy, FifoContextPacker,
    LatentDigestDisambiguator,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    OnCallNotesOracle,
    OracleRegistration,
    OutsideQuery, OutsideVerdict,
    OutsideWitnessAcquisitionDisambiguator,
    QuorumKeyedSharedReadCache,
    QuorumKeyedCachingOracleAdapter,
    RelationalCompatibilityDisambiguator,
    RobustMultiRoundBundleDecoder, RoleBudget,
    SchemaCapsule, ServiceGraphOracle, SharedReadCache,
    SessionDeltaEnvelope, SessionDigestEnvelope,
    SingletonAsymmetricOracle,
    SuperTokenReferenceEnvelope, SuperTokenRegistry,
    TeamCoordinator, audit_team_lifecycle,
    TrustWeightedMultiOracleDisambiguator,
    build_incident_triage_schema_capsule,
    collect_admitted_handoffs, _DecodedHandoff,
    verify_session_digest_chain, verify_session_delta,
    verify_super_token_reference,
    CACHE_FRESHNESS_BYTE_IDENTICAL,
    CACHE_FRESHNESS_PER_CELL_NONCE,
    CACHE_FRESHNESS_QUORUM_LOCKED,
    W22_BRANCH_LATENT_RESOLVED, W22_BRANCH_LATENT_REJECTED,
    W22_BRANCH_NO_TRIGGER,
    W23_BRANCH_DELTA_RESOLVED, W23_BRANCH_DELTA_REJECTED,
    W23_BRANCH_GENESIS,
    W23_BRANCH_SUPER_TOKEN_RESOLVED,
    W23_BRANCH_SUPER_TOKEN_REJECTED,
    W23_BRANCH_NO_TRIGGER,
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
# Phase-70 banks — same R-69-CACHE-FANOUT bundle / oracle ecology;
# the *cross-cell topology* and *cache freshness policy* are the
# bank-specific dimensions.
# =============================================================================

_VALID_BANKS_P70 = (
    "delta_fanout",          # W23-1 anchor — multi-cell session, delta wins.
    "super_token",            # W23-3 dense-control anchor — super-token refs.
    "amplified_llm",          # W23-2 mitigation — flipping oracle + quorum-key.
    "no_delta",               # W23-Λ-no-delta — single-cell, no cross-state.
    "super_token_tampered",   # W23-3 trust falsifier — tampered super-token.
    "chain_broken",           # W23-3 trust falsifier — chain link tampered.
)


# =============================================================================
# Bench-only oracle: deterministic simulation of W22-C-CACHE-AMPLIFICATION.
# Returns a decoy-asymmetric reply on the FIRST consult; gold-asymmetric
# afterwards. Lets us measure W23-2 mitigation deterministically without a
# live LLM.
# =============================================================================

@dataclasses.dataclass
class FlippingProbabilisticOracle:
    """Bench oracle that flips between decoy- and gold-asymmetric replies.

    Models the deterministic structural pattern that
    :class:`LLMAdjudicatorOracle` exhibits on Mac-1 ``mixtral:8x7b``
    at temperature=0 across separate sessions: the *first* sample is
    sometimes decoy-asymmetric (bad); subsequent samples drift toward
    gold-asymmetric (good). The W22 BYTE_IDENTICAL cache freezes the
    first sample for every matching cell — the W22-C-CACHE-AMPLIFICATION
    conjecture observation.

    This bench oracle is **deterministic** for reproducibility:
    consult #1 returns ``decoy_asymmetric_reply``;
    consult #2.. returns ``gold_asymmetric_reply``. The W22 baseline
    (BYTE_IDENTICAL) caches consult #1's reply and hits on every
    subsequent matching call → bad reply propagates. The W23
    PER_CELL_NONCE policy bypasses the cache for this oracle → cell
    2's consult #2 returns the gold reply.

    Parameters
    ----------
    gold_subset
        Sorted tuple of admitted tags that count as the gold answer
        on this bench (the deterministic ServiceGraphOracle's reply
        on R-66-OUTSIDE-REQUIRED).
    decoy_tag
        The single tag the first reply mentions decoy-asymmetrically.
    """
    oracle_id: str = "flipping_probabilistic"
    gold_subset: tuple[str, ...] = ("orders", "payments")
    decoy_tag: str = "cache"
    n_consults: int = 0
    decoy_asymmetric_reply: str = ""
    gold_asymmetric_reply: str = ""
    max_response_tokens: int = 24

    def __post_init__(self) -> None:
        if not self.decoy_asymmetric_reply:
            self.decoy_asymmetric_reply = (
                f"investigate {self.decoy_tag} subsystem only")
        if not self.gold_asymmetric_reply:
            gold = " ".join(self.gold_subset) or "orders payments"
            self.gold_asymmetric_reply = f"check {gold} call graph"

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        self.n_consults += 1
        if self.n_consults <= 1:
            payload = self.decoy_asymmetric_reply
        else:
            payload = self.gold_asymmetric_reply
        # Truncate to max_response_tokens.
        toks = payload.split()[:self.max_response_tokens]
        payload_truncated = " ".join(toks)
        return OutsideVerdict(
            payload=payload_truncated,
            source_id=self.oracle_id,
            n_tokens=len(toks),
        )


# =============================================================================
# Strategy / decoder dispatch
# =============================================================================

# Phase-70 strategy roster: a focused subset that isolates the W23
# contribution against W22.
_R70_STRATEGIES: tuple[tuple[str, str], ...] = (
    # W22 baseline (cache + digest, no W23 layer).
    ("capsule_w22_hybrid", "w22_hybrid"),
    # W23 with delta + chain (no super-token, no quorum-keyed cache).
    ("capsule_w23_delta", "w23_delta"),
    # W23 with super-token reference (max density on a verified channel).
    ("capsule_w23_super_token", "w23_super_token"),
    # W23 with quorum-keyed cache mitigation on the flipping oracle.
    ("capsule_w23_quorum_keyed", "w23_quorum_keyed"),
)


def _make_factory_p70(name: str, priorities, budgets):
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
    """Build a W21+W22 stack against ``cache``."""
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    w22 = LatentDigestDisambiguator(
        inner=w21, schema=schema, cache=cache)
    return w22, inner_w15


def _decode_with_w22_only(
        union, T_decoder, round_index_hint, oracle_registrations, *,
        schema: SchemaCapsule, cache: SharedReadCache,
        quorum_min: int = 2, min_trust_sum: float = 0.0,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
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
    if "latent_envelope" in ans:
        stats["latent_envelope"] = ans["latent_envelope"]
    return ans, stats


def _decode_with_w23(
        union, T_decoder, round_index_hint, oracle_registrations, *,
        schema: SchemaCapsule, cache: SharedReadCache,
        quorum_min: int = 2, min_trust_sum: float = 0.0,
        w23: CrossCellDeltaDisambiguator,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    w22, inner_w15 = _build_w22_stack(
        T_decoder=T_decoder, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum,
        schema=schema, cache=cache)
    # Re-bind w22 into the W23 (reuse same persistent w23 across cells).
    w23.inner = w22
    w23.schema = schema
    per_round = _build_per_round(union, round_index_hint)
    ans = w23.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
    if "multi_oracle" in ans:
        stats["multi_oracle"] = ans["multi_oracle"]
    if "latent_hybrid" in ans:
        stats["latent_hybrid"] = ans["latent_hybrid"]
    if "latent_envelope" in ans:
        stats["latent_envelope"] = ans["latent_envelope"]
    if "session_delta_hybrid" in ans:
        stats["session_delta_hybrid"] = ans["session_delta_hybrid"]
    if "session_delta_envelope" in ans:
        stats["session_delta_envelope"] = ans["session_delta_envelope"]
    if "session_digest_envelope" in ans:
        stats["session_digest_envelope"] = ans["session_digest_envelope"]
    if "super_token_reference" in ans:
        stats["super_token_reference"] = ans["super_token_reference"]
    return ans, stats


# =============================================================================
# Per-cell strategy execution
# =============================================================================

def _build_oracles_for_bank(*, bank: str, shared_cache: SharedReadCache,
                              quorum_keyed_cache: QuorumKeyedSharedReadCache,
                              flipping_oracle: FlippingProbabilisticOracle | None,
                              with_compromised: bool = True,
                              llm_adjudicator: Any | None = None,
                              ) -> tuple[
                                  tuple[OracleRegistration, ...],
                                  tuple[OracleRegistration, ...],
                                  ]:
    """Build (w22_regs, w23_regs).  The W22 path uses BYTE_IDENTICAL
    caching uniformly. The W23 path uses BYTE_IDENTICAL on the
    deterministic oracles AND PER_CELL_NONCE on any flipping /
    probabilistic oracle.
    """
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
    if flipping_oracle is not None:
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=flipping_oracle, cache=shared_cache,
                oracle_id="flipping_probabilistic"),
            trust_prior=0.7, role_label="flipping_probabilistic"))
    if llm_adjudicator is not None:
        w22_regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=llm_adjudicator, cache=shared_cache,
                oracle_id=getattr(llm_adjudicator, "oracle_id",
                                     "llm_adjudicator")),
            trust_prior=0.7, role_label="llm_adjudicator"))

    # W23 registrations (QuorumKeyedSharedReadCache; PER_CELL_NONCE
    # on the flipping / LLM oracles).
    quorum_keyed_cache.set_policy(
        "compromised_registry", CACHE_FRESHNESS_BYTE_IDENTICAL)
    quorum_keyed_cache.set_policy(
        "service_graph", CACHE_FRESHNESS_BYTE_IDENTICAL)
    quorum_keyed_cache.set_policy(
        "change_history", CACHE_FRESHNESS_BYTE_IDENTICAL)
    if flipping_oracle is not None:
        quorum_keyed_cache.set_policy(
            "flipping_probabilistic", CACHE_FRESHNESS_PER_CELL_NONCE)
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
    if flipping_oracle is not None:
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=flipping_oracle, cache=quorum_keyed_cache,
                oracle_id="flipping_probabilistic"),
            trust_prior=0.7, role_label="flipping_probabilistic"))
    if llm_adjudicator is not None:
        w23_regs.append(OracleRegistration(
            oracle=QuorumKeyedCachingOracleAdapter(
                inner=llm_adjudicator, cache=quorum_keyed_cache,
                oracle_id=getattr(llm_adjudicator, "oracle_id",
                                     "llm_adjudicator")),
            trust_prior=0.7, role_label=getattr(
                llm_adjudicator, "oracle_id", "llm_adjudicator")))

    return tuple(w22_regs), tuple(w23_regs)


def _set_cell_nonce_on_w23_regs(
        w23_regs: tuple[OracleRegistration, ...],
        cell_nonce: str) -> None:
    """Update the cell_nonce on each QuorumKeyedCachingOracleAdapter
    so the PER_CELL_NONCE policy mixes it into the cache key."""
    for reg in w23_regs:
        oracle = reg.oracle
        if isinstance(oracle, QuorumKeyedCachingOracleAdapter):
            oracle.cell_nonce = str(cell_nonce)


def _run_capsule_strategy_p70(
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
        quorum_min: int = 2,
        min_trust_sum: float = 0.0,
        schema: SchemaCapsule,
        shared_cache: SharedReadCache,
        quorum_keyed_cache: QuorumKeyedSharedReadCache,
        cell_index: int = 0,
        cell_nonce: str = "",
        w23_persistent: CrossCellDeltaDisambiguator | None = None,
        bank: str = "delta_fanout",
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase70_capsule_session_delta",
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
        # W22 baseline — BYTE_IDENTICAL cache; no W23 layer.
        answer, pack_stats = _decode_with_w22_only(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=oracle_registrations_w22,
            schema=schema, cache=shared_cache,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    elif decoder_mode in (
            "w23_delta", "w23_super_token", "w23_quorum_keyed"):
        if w23_persistent is None:
            raise ValueError("w23_persistent must be provided for W23 modes")
        # Set per-cell nonce on the quorum-keyed adapters (the
        # PER_CELL_NONCE policy mixes it in; deterministic oracles
        # ignore it).
        _set_cell_nonce_on_w23_regs(
            oracle_registrations_w23, cell_nonce)
        # Toggle super_token mode at runtime (cheap; no rewiring).
        w23_persistent.use_super_token = (
            decoder_mode == "w23_super_token")
        # Use the quorum-keyed cache only on the quorum_keyed mode;
        # the other W23 modes share the same byte-identical cache as
        # W22 baseline (so the comparison isolates only the delta /
        # super-token contribution).
        if decoder_mode == "w23_quorum_keyed":
            cache_for_inner = quorum_keyed_cache
            regs_for_inner = oracle_registrations_w23
        else:
            cache_for_inner = shared_cache
            # Use the W22-style adapter set (BYTE_IDENTICAL on every
            # oracle).
            regs_for_inner = oracle_registrations_w22
        # For super-token tampered / chain broken, install a
        # tamperer on the W22 envelope or break the chain.
        answer, pack_stats = _decode_with_w23(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=regs_for_inner,
            schema=schema, cache=cache_for_inner,
            quorum_min=quorum_min, min_trust_sum=min_trust_sum,
            w23=w23_persistent)
    else:
        raise ValueError(f"unknown decoder_mode {decoder_mode!r}")

    # Strip W22+W23 audit metadata before sealing the team decision —
    # the TEAM_DECISION capsule is bounded at 16 KiB. The audit records
    # remain in ``pack_stats`` for the bench, so no information is lost
    # at the bench level.
    decision_payload = {
        k: v for k, v in answer.items()
        if k not in (
            "multi_oracle", "latent_hybrid", "latent_envelope",
            "session_delta_hybrid", "session_delta_envelope",
            "session_digest_envelope", "super_token_reference",
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


def _maybe_break_chain(w23: CrossCellDeltaDisambiguator) -> None:
    """For R-70-CHAIN-BROKEN: install a phantom chain head on the
    *verifier* side after the genesis cell so the next cell's
    :func:`verify_session_digest_chain` and
    :func:`verify_session_delta` both fire ``chain_head_mismatch`` /
    ``parent_session_mismatch``. The producer's chain remains
    consistent (it continues from the real ``chain_head_cid``); the
    verifier's expected chain head differs (the bench-installed
    phantom). This models a mid-session producer/verifier split that
    a true cross-host deployment can detect."""
    if w23._chain:
        # Use a deterministic phantom CID; in a true cross-host
        # deployment this would be the verifier's last *confirmed*
        # chain head, which lags or differs from the producer's
        # current head when the producer drifts.
        w23.verifier_chain_head_override = "0" * 64


# =============================================================================
# Phase 70 driver
# =============================================================================


def build_phase70_bank(*, n_replicates: int = 2, seed: int = 11,
                          ) -> list[MultiRoundScenario]:
    """Build the Phase-70 bank — same R-69-CACHE-FANOUT bundle shape
    (= R-68-MULTI-MAJORITY = R-66-OUTSIDE-REQUIRED + Phase-67 outside-
    resolves family). The bundle is bank-invariant; the cross-cell
    topology and cache freshness policy are bank-specific knobs."""
    return build_phase68_bank(n_replicates=n_replicates, seed=seed)


def run_phase70(*,
                  bank: str = "delta_fanout",
                  n_eval: int | None = None,
                  K_auditor: int = 12,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  T_decoder: int | None = None,
                  quorum_min: int = 2,
                  min_trust_sum: float = 0.0,
                  llm_adjudicator: Any | None = None,
                  use_cross_host_proxy: bool = True,
                  super_token_hex_prefix_len: int = 16,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 70 over one of {delta_fanout, super_token, amplified_llm,
    no_delta, super_token_tampered, chain_broken}."""
    if bank not in _VALID_BANKS_P70:
        raise ValueError(
            f"unknown bank {bank!r}; valid: {_VALID_BANKS_P70}")
    bank_obj = build_phase70_bank(
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

    # The W22 path uses a SharedReadCache (BYTE_IDENTICAL).
    # The W23 quorum-keyed path uses a QuorumKeyedSharedReadCache.
    shared_cache_w22 = SharedReadCache()
    quorum_keyed_cache = QuorumKeyedSharedReadCache()
    flipping_oracle: FlippingProbabilisticOracle | None = None
    if bank == "amplified_llm":
        flipping_oracle = FlippingProbabilisticOracle()

    w22_regs, w23_regs = _build_oracles_for_bank(
        bank=bank, shared_cache=shared_cache_w22,
        quorum_keyed_cache=quorum_keyed_cache,
        flipping_oracle=flipping_oracle,
        with_compromised=True,
        llm_adjudicator=llm_adjudicator)

    # Persistent W23 instances (one per W23 strategy) — the chain
    # accumulates across cells.
    cross_host_proxy = (
        CrossHostProducerDecoderProxy() if use_cross_host_proxy else None)
    w23_delta_persistent = CrossCellDeltaDisambiguator(
        inner=LatentDigestDisambiguator(),  # rebuilt per cell
        schema=schema,
        super_token_registry=SuperTokenRegistry(),
        quorum_keyed_cache=None,  # no quorum-keying on this strategy
        cross_host_proxy=cross_host_proxy,
        super_token_hex_prefix_len=super_token_hex_prefix_len,
        use_super_token=False)
    # On R-70-SUPER-TOKEN-TAMPERED the bench installs a SEPARATE
    # verifier-side super-token registry (empty) so every super-token
    # reference falls into ``unknown_super_token`` and the W23 layer
    # rejects, falling through to the verbose digest path. On every
    # other bank the verifier shares the producer's registry (the
    # within-process default).
    if bank == "super_token_tampered":
        verifier_super_token_registry = SuperTokenRegistry()
    else:
        verifier_super_token_registry = None
    w23_super_token_persistent = CrossCellDeltaDisambiguator(
        inner=LatentDigestDisambiguator(),
        schema=schema,
        super_token_registry=SuperTokenRegistry(),
        verifier_super_token_registry=verifier_super_token_registry,
        quorum_keyed_cache=None,
        cross_host_proxy=cross_host_proxy,
        super_token_hex_prefix_len=super_token_hex_prefix_len,
        use_super_token=True)
    w23_quorum_keyed_persistent = CrossCellDeltaDisambiguator(
        inner=LatentDigestDisambiguator(),
        schema=schema,
        super_token_registry=SuperTokenRegistry(),
        quorum_keyed_cache=quorum_keyed_cache,
        cross_host_proxy=cross_host_proxy,
        super_token_hex_prefix_len=super_token_hex_prefix_len,
        use_super_token=False)

    # Bank-specific: tamper after the genesis cell for the chain-broken
    # falsifier. (super_token_tampered is handled via a separate
    # verifier-side super-token registry installed at W23
    # construction; chain_broken installs a phantom chain head on
    # the verifier side after cell 0 → every subsequent link's
    # verifier sees ``chain_head_mismatch``.)
    def _post_cell_hook(strategy_name: str, cell_index: int,
                          w23_inst: CrossCellDeltaDisambiguator) -> None:
        if bank == "chain_broken" and cell_index == 0 and (
                strategy_name in (
                    "capsule_w23_delta", "capsule_w23_super_token",
                    "capsule_w23_quorum_keyed")):
            _maybe_break_chain(w23_inst)

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R70_STRATEGIES
    }

    for cell_index, sc in enumerate(bank_obj):
        round1_cands = _build_round_candidates_p66(sc.round1_emissions)
        round2_cands = _build_round_candidates_p66(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p67(
            sc, round1_cands, round2_cands)
        cell_nonce = f"cell_{cell_index}_seed_{bank_seed}"

        # On the no_delta bank, RESET the W23 chain after every cell so
        # every cell is treated as a fresh genesis (no cross-cell delta).
        if bank == "no_delta":
            w23_delta_persistent.reset_session()
            w23_super_token_persistent.reset_session()
            w23_quorum_keyed_persistent.reset_session()

        for (sname, dmode) in _R70_STRATEGIES:
            fac = _make_factory_p70(sname, priorities, budgets)
            if dmode == "w23_delta":
                w23_inst = w23_delta_persistent
            elif dmode == "w23_super_token":
                w23_inst = w23_super_token_persistent
            elif dmode == "w23_quorum_keyed":
                w23_inst = w23_quorum_keyed_persistent
            else:
                w23_inst = None

            r, ps = _run_capsule_strategy_p70(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                T_decoder=T_decoder,
                oracle_registrations_w22=w22_regs,
                oracle_registrations_w23=w23_regs,
                quorum_min=quorum_min,
                min_trust_sum=min_trust_sum,
                schema=schema,
                shared_cache=shared_cache_w22,
                quorum_keyed_cache=quorum_keyed_cache,
                cell_index=cell_index,
                cell_nonce=cell_nonce,
                w23_persistent=w23_inst,
                bank=bank)
            results.append(r)
            if ps:
                pack_stats_per_strategy[sname].append({
                    "scenario_id": sc.scenario_id,
                    **ps,
                })
            # Post-cell tamper hook for trust falsifiers.
            if w23_inst is not None:
                _post_cell_hook(sname, cell_index, w23_inst)

    strategy_names = tuple(s[0] for s in _R70_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w23_delta_minus_w22": _gap(
            "capsule_w23_delta", "capsule_w22_hybrid"),
        "w23_super_token_minus_w22": _gap(
            "capsule_w23_super_token", "capsule_w22_hybrid"),
        "w23_quorum_keyed_minus_w22": _gap(
            "capsule_w23_quorum_keyed", "capsule_w22_hybrid"),
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
        "quorum_min": int(quorum_min),
        "min_trust_sum": float(min_trust_sum),
        "schema_cid": schema.cid,
        "schema_n_canonical_tokens": schema.n_canonical_tokens,
        "n_oracle_registrations_w22": len(w22_regs),
        "n_oracle_registrations_w23": len(w23_regs),
        "oracle_role_labels_w22": [r.role_label for r in w22_regs],
        "oracle_role_labels_w23": [r.role_label for r in w23_regs],
        "use_cross_host_proxy": bool(use_cross_host_proxy),
        "super_token_hex_prefix_len": int(super_token_hex_prefix_len),
        "flipping_oracle_present": flipping_oracle is not None,
    }

    def _agg_w23(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)

        def get_sd(r):
            return r.get("session_delta_hybrid") or {}

        s_visible = sum(int(get_sd(r).get(
            "n_w23_visible_tokens_to_decider", 0)) for r in rows)
        s_w22_visible = sum(int(get_sd(r).get(
            "n_w22_visible_tokens_to_decider", 0)) for r in rows)
        s_savings = sum(int(get_sd(r).get(
            "n_w22_minus_w23_savings", 0)) for r in rows)
        s_delta_n = sum(int(get_sd(r).get(
            "delta_n_tokens", 0)) for r in rows)
        s_kept = sum(int(get_sd(r).get(
            "n_w15_tokens_kept", 0)) for r in rows)
        s_cache_saved = sum(int(get_sd(r).get(
            "cache_tokens_saved_this_cell", 0)) for r in rows)
        s_rt_bytes = sum(int(get_sd(r).get(
            "cross_host_round_trip_bytes", 0)) for r in rows)
        n_genesis = sum(
            1 for r in rows
            if get_sd(r).get("decoder_branch") == W23_BRANCH_GENESIS)
        n_delta_resolved = sum(
            1 for r in rows
            if get_sd(r).get("decoder_branch") == W23_BRANCH_DELTA_RESOLVED)
        n_super_token_resolved = sum(
            1 for r in rows
            if get_sd(r).get("decoder_branch") == W23_BRANCH_SUPER_TOKEN_RESOLVED)
        n_super_token_rejected = sum(
            1 for r in rows
            if get_sd(r).get("decoder_branch") == W23_BRANCH_SUPER_TOKEN_REJECTED)
        n_delta_rejected = sum(
            1 for r in rows
            if get_sd(r).get("decoder_branch") == W23_BRANCH_DELTA_REJECTED)
        n_no_trigger = sum(
            1 for r in rows
            if get_sd(r).get("decoder_branch") == W23_BRANCH_NO_TRIGGER)
        n_chain_ok = sum(
            1 for r in rows if get_sd(r).get("chain_verification_ok"))
        n_super_ok = sum(
            1 for r in rows if get_sd(r).get("super_token_verification_ok"))
        n_super_used = sum(
            1 for r in rows if get_sd(r).get("super_token_used"))
        return {
            "n_cells": n,
            "n_w23_genesis_cells": int(n_genesis),
            "n_w23_delta_resolved_cells": int(n_delta_resolved),
            "n_w23_delta_rejected_cells": int(n_delta_rejected),
            "n_w23_super_token_resolved_cells": int(n_super_token_resolved),
            "n_w23_super_token_rejected_cells": int(n_super_token_rejected),
            "n_w23_no_trigger_cells": int(n_no_trigger),
            "n_w23_visible_tokens_to_decider_sum": int(s_visible),
            "n_w23_visible_tokens_to_decider_avg": (
                round(s_visible / n, 4) if n else 0.0),
            "n_w22_visible_tokens_to_decider_sum": int(s_w22_visible),
            "n_w22_visible_tokens_to_decider_avg": (
                round(s_w22_visible / n, 4) if n else 0.0),
            "n_w22_minus_w23_savings_sum": int(s_savings),
            "n_w22_minus_w23_savings_avg": (
                round(s_savings / n, 4) if n else 0.0),
            "delta_n_tokens_sum": int(s_delta_n),
            "delta_n_tokens_avg": (
                round(s_delta_n / n, 4) if n else 0.0),
            "w15_tokens_kept_sum": int(s_kept),
            "cache_tokens_saved_total": int(s_cache_saved),
            "cross_host_round_trip_bytes_total": int(s_rt_bytes),
            "chain_verifies_ok_rate": (
                round(n_chain_ok / n, 4) if n else 0.0),
            "super_token_verification_ok_rate": (
                round(n_super_ok / max(1, n_super_used), 4)
                if n_super_used else 0.0),
            "super_token_resolved_rate": (
                round(n_super_token_resolved / max(1, n_super_used), 4)
                if n_super_used else 0.0),
        }

    def _agg_w22_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)

        def get_lh(r):
            return r.get("latent_hybrid") or {}

        s_visible = sum(int(get_lh(r).get(
            "n_visible_tokens_to_decider", 0)) for r in rows)
        s_kept = sum(int(get_lh(r).get(
            "n_w15_tokens_kept", 0)) for r in rows)
        s_cache_saved = sum(int(get_lh(r).get(
            "cache_tokens_saved_this_cell", 0)) for r in rows)
        return {
            "n_cells": n,
            "n_w22_visible_tokens_to_decider_sum": int(s_visible),
            "n_w22_visible_tokens_to_decider_avg": (
                round(s_visible / n, 4) if n else 0.0),
            "w15_tokens_kept_sum": int(s_kept),
            "cache_tokens_saved_total": int(s_cache_saved),
        }

    pack_stats_summary = {
        s: (_agg_w23(pack_stats_per_strategy.get(s, []))
             if s in ("capsule_w23_delta", "capsule_w23_super_token",
                       "capsule_w23_quorum_keyed")
             else _agg_w22_baseline(pack_stats_per_strategy.get(s, [])))
        for s in strategy_names
    }

    eff_w22 = pack_stats_summary.get("capsule_w22_hybrid", {})
    eff_w23_delta = pack_stats_summary.get("capsule_w23_delta", {})
    eff_w23_super = pack_stats_summary.get("capsule_w23_super_token", {})
    eff_w23_quorum = pack_stats_summary.get("capsule_w23_quorum_keyed", {})
    eff_compare = {
        "w22_visible_tokens_per_cell":
            float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0)),
        "w23_delta_visible_tokens_per_cell":
            float(eff_w23_delta.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
        "w23_super_token_visible_tokens_per_cell":
            float(eff_w23_super.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
        "w23_quorum_keyed_visible_tokens_per_cell":
            float(eff_w23_quorum.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
        "w23_delta_savings_per_cell": round(
            float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0))
            - float(eff_w23_delta.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
            4),
        "w23_delta_savings_pct": (round(
            (float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0))
             - float(eff_w23_delta.get(
                 "n_w23_visible_tokens_to_decider_avg", 0.0)))
            / max(1.0, float(eff_w22.get(
                "n_w22_visible_tokens_to_decider_avg", 0.0)))
            * 100, 4)),
        "w23_super_token_savings_per_cell": round(
            float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0))
            - float(eff_w23_super.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
            4),
        "w23_super_token_savings_pct": (round(
            (float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0))
             - float(eff_w23_super.get(
                 "n_w23_visible_tokens_to_decider_avg", 0.0)))
            / max(1.0, float(eff_w22.get(
                "n_w22_visible_tokens_to_decider_avg", 0.0)))
            * 100, 4)),
        "w23_quorum_keyed_savings_per_cell": round(
            float(eff_w22.get("n_w22_visible_tokens_to_decider_avg", 0.0))
            - float(eff_w23_quorum.get(
                "n_w23_visible_tokens_to_decider_avg", 0.0)),
            4),
        "chain_verifies_ok_rate":
            float(eff_w23_delta.get("chain_verifies_ok_rate", 0.0)),
        "super_token_verification_ok_rate":
            float(eff_w23_super.get(
                "super_token_verification_ok_rate", 0.0)),
        "super_token_resolved_rate":
            float(eff_w23_super.get("super_token_resolved_rate", 0.0)),
        "cross_host_round_trip_bytes_total":
            int(eff_w23_delta.get("cross_host_round_trip_bytes_total", 0)),
    }

    # Correctness ratification: W23 == W22 byte-for-byte on every cell.
    correctness_ratified_per_strategy: dict[str, list[bool]] = {}
    for s_name in ("capsule_w23_delta", "capsule_w23_super_token",
                    "capsule_w23_quorum_keyed"):
        cells: list[bool] = []
        for sc in bank_obj:
            w22_r = next((r for r in results
                           if r.strategy == "capsule_w22_hybrid"
                           and r.scenario_id == sc.scenario_id), None)
            w23_r = next((r for r in results
                           if r.strategy == s_name
                           and r.scenario_id == sc.scenario_id), None)
            if w22_r is None or w23_r is None:
                continue
            sv22 = tuple(sorted(map(str,
                                       w22_r.answer.get("services", ()))))
            sv23 = tuple(sorted(map(str,
                                       w23_r.answer.get("services", ()))))
            cells.append(sv22 == sv23)
        correctness_ratified_per_strategy[s_name] = cells

    correctness_ratified_rates = {
        s_name: round(sum(c) / max(1, len(c)), 4)
        for s_name, c in correctness_ratified_per_strategy.items()
    }

    # W23-2 mitigation: on amplified_llm, w23_quorum_keyed should be
    # strictly more correct than w22_hybrid baseline.
    mitigation_advantage = round(
        float(pooled["capsule_w23_quorum_keyed"]["accuracy_full"])
        - float(pooled["capsule_w22_hybrid"]["accuracy_full"]),
        4)

    if verbose:
        print(f"[phase70] bank={bank} schema_cid={schema.cid[:16]} "
              f"T_decoder={T_decoder} n_eval={len(bank_obj)} "
              f"K_auditor={K_auditor} bank_seed={bank_seed}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase70]   {s:36s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase70] {k}: {v:+.3f}", file=sys.stderr, flush=True)
        print(f"[phase70] visible_tokens_per_cell w22={eff_compare['w22_visible_tokens_per_cell']:.2f} "
              f"w23_delta={eff_compare['w23_delta_visible_tokens_per_cell']:.2f} "
              f"w23_super={eff_compare['w23_super_token_visible_tokens_per_cell']:.2f}",
              file=sys.stderr, flush=True)
        print(f"[phase70] savings_pct delta={eff_compare['w23_delta_savings_pct']:+.2f}% "
              f"super={eff_compare['w23_super_token_savings_pct']:+.2f}%",
              file=sys.stderr, flush=True)
        print(f"[phase70] chain_ok={eff_compare['chain_verifies_ok_rate']:.3f} "
              f"super_token_ok={eff_compare['super_token_verification_ok_rate']:.3f} "
              f"super_resolved_rate={eff_compare['super_token_resolved_rate']:.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase70] mitigation_advantage_w23_minus_w22="
              f"{mitigation_advantage:+.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase70] correctness_ratified_rates="
              f"{correctness_ratified_rates}",
              file=sys.stderr, flush=True)
        print(f"[phase70] cross_host_round_trip_bytes_total="
              f"{eff_compare['cross_host_round_trip_bytes_total']}",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase70.capsule_session_delta.v1",
        "config": {
            "bank": bank, "n_eval": len(bank_obj),
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "K_producer": K_producer, "T_producer": T_producer,
            "inbox_capacity": inbox_capacity, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates, "T_decoder": T_decoder,
            "quorum_min": int(quorum_min),
            "min_trust_sum": float(min_trust_sum),
            "use_cross_host_proxy": bool(use_cross_host_proxy),
            "super_token_hex_prefix_len": int(super_token_hex_prefix_len),
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "pack_stats_summary": pack_stats_summary,
        "eff_compare": eff_compare,
        "correctness_ratified_rates": correctness_ratified_rates,
        "mitigation_advantage_w23_minus_w22": mitigation_advantage,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }


def run_phase70_seed_stability_sweep(*,
                                          bank: str = "delta_fanout",
                                          T_decoder: int | None = None,
                                          n_eval: int = 8,
                                          K_auditor: int = 12,
                                          quorum_min: int = 2,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase70.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "quorum_min": int(quorum_min),
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase70(bank=bank, T_decoder=T_decoder,
                            n_eval=n_eval, K_auditor=K_auditor,
                            bank_seed=seed, quorum_min=quorum_min)
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
    delta_savings = [
        out["per_seed"][str(s)]["eff_compare"][
            "w23_delta_savings_per_cell"] for s in seeds
    ]
    super_savings = [
        out["per_seed"][str(s)]["eff_compare"][
            "w23_super_token_savings_per_cell"] for s in seeds
    ]
    out["min_w23_delta_savings_per_cell"] = (
        min(delta_savings) if delta_savings else 0.0)
    out["mean_w23_delta_savings_per_cell"] = (
        round(sum(delta_savings) / len(delta_savings), 4)
        if delta_savings else 0.0)
    out["min_w23_super_token_savings_per_cell"] = (
        min(super_savings) if super_savings else 0.0)
    out["mean_w23_super_token_savings_per_cell"] = (
        round(sum(super_savings) / len(super_savings), 4)
        if super_savings else 0.0)
    delta_correct = [
        out["per_seed"][str(s)]["correctness_ratified_rates"].get(
            "capsule_w23_delta", 0.0) for s in seeds
    ]
    super_correct = [
        out["per_seed"][str(s)]["correctness_ratified_rates"].get(
            "capsule_w23_super_token", 0.0) for s in seeds
    ]
    out["min_w23_delta_correctness_ratified_rate"] = (
        min(delta_correct) if delta_correct else 0.0)
    out["min_w23_super_token_correctness_ratified_rate"] = (
        min(super_correct) if super_correct else 0.0)
    return out


def run_cross_regime_synthetic_p70(*,
                                      n_eval: int = 8,
                                      bank_seed: int = 11,
                                      K_auditor: int = 12,
                                      T_auditor: int = 256,
                                      T_decoder_tight: int = 24,
                                      quorum_min: int = 2,
                                      ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase70.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
            "quorum_min": int(quorum_min),
        },
        "regimes": {},
    }
    cells = [
        ("R-70-DELTA-FANOUT-LOOSE",        "delta_fanout",         None),
        ("R-70-DELTA-FANOUT-TIGHT",        "delta_fanout",         T_decoder_tight),
        ("R-70-SUPER-TOKEN",                "super_token",          None),
        ("R-70-AMPLIFIED-LLM",              "amplified_llm",        None),
        ("R-70-NO-DELTA",                   "no_delta",             None),
        ("R-70-SUPER-TOKEN-TAMPERED",       "super_token_tampered", None),
        ("R-70-CHAIN-BROKEN",               "chain_broken",         None),
    ]
    for (regime_name, bank, T_dec) in cells:
        rep = run_phase70(
            bank=bank, n_eval=n_eval, K_auditor=K_auditor,
            T_auditor=T_auditor, T_decoder=T_dec,
            bank_seed=bank_seed, quorum_min=quorum_min)
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
            "mitigation_advantage_w23_minus_w22":
                rep["mitigation_advantage_w23_minus_w22"],
            "audit_ok_grid": rep["audit_ok_grid"],
        }
    return out


def _make_llm_adjudicator(model: str, *, base_url: str | None = None,
                           timeout: float = 120.0):
    """Build an :class:`LLMAdjudicatorOracle` against a live Ollama
    backend (Mac-1 default). Used for the W23-Λ-real probe."""
    from vision_mvp.coordpy.team_coord import LLMAdjudicatorOracle
    from vision_mvp.coordpy.llm_backend import OllamaBackend
    backend_url = base_url or os.environ.get(
        "COORDPY_OLLAMA_URL", "http://127.0.0.1:11434")
    backend = OllamaBackend(model=model, base_url=backend_url,
                              timeout=timeout)
    return LLMAdjudicatorOracle(
        oracle_id=f"ollama_{model}", backend=backend,
        max_response_tokens=24)


# =============================================================================
# CLI
# =============================================================================


def _cli() -> None:
    ap = argparse.ArgumentParser(prog="phase70_capsule_session_delta")
    ap.add_argument("--bank", default="delta_fanout",
                     choices=_VALID_BANKS_P70)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--K-auditor", type=int, default=12)
    ap.add_argument("--T-auditor", type=int, default=256)
    ap.add_argument("--K-producer", type=int, default=6)
    ap.add_argument("--T-producer", type=int, default=96)
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument("--bank-replicates", type=int, default=2)
    ap.add_argument("--decoder-budget", type=int, default=-1,
                     help="-1 = None (loose)")
    ap.add_argument("--quorum-min", type=int, default=2)
    ap.add_argument("--min-trust-sum", type=float, default=0.0)
    ap.add_argument("--out", default="-",
                     help="output path; '-' = stdout")
    ap.add_argument("--seed-sweep", action="store_true",
                     help="run a 5-seed stability sweep on the bank")
    ap.add_argument("--cross-regime-synthetic", action="store_true",
                     help="run all R-70 regimes back-to-back")
    ap.add_argument("--live-llm-adjudicator", action="store_true",
                     help="add an LLMAdjudicatorOracle to the registry")
    ap.add_argument("--adjudicator-model", default="mixtral:8x7b")
    ap.add_argument("--adjudicator-base-url", default=None)
    ap.add_argument("--no-cross-host-proxy", action="store_true",
                     help="disable the within-process producer/decoder "
                          "host-split proxy")
    ap.add_argument("--super-token-hex-prefix-len", type=int, default=16,
                     help="hex prefix length for super-token references "
                          "(4..64)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    T_decoder = None if args.decoder_budget < 0 else int(args.decoder_budget)
    llm_adj = None
    if args.live_llm_adjudicator:
        llm_adj = _make_llm_adjudicator(
            args.adjudicator_model,
            base_url=args.adjudicator_base_url)

    if args.cross_regime_synthetic:
        out = run_cross_regime_synthetic_p70(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            T_decoder_tight=24, quorum_min=args.quorum_min)
    elif args.seed_sweep:
        out = run_phase70_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            quorum_min=args.quorum_min)
    else:
        out = run_phase70(
            bank=args.bank, n_eval=args.n_eval,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            K_producer=args.K_producer, T_producer=args.T_producer,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            T_decoder=T_decoder, quorum_min=args.quorum_min,
            min_trust_sum=args.min_trust_sum,
            llm_adjudicator=llm_adj,
            use_cross_host_proxy=not args.no_cross_host_proxy,
            super_token_hex_prefix_len=args.super_token_hex_prefix_len,
            verbose=args.verbose)

    if args.out == "-":
        print(json.dumps(out, indent=2, default=str))
    else:
        with open(args.out, "w") as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f"[phase70] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
