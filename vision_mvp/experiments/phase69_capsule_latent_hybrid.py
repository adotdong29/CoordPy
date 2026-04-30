"""Phase 69 — capsule + audited latent-state-sharing hybrid (SDK v3.23,
W22 family anchor).

The follow-up to SDK v3.22 (W21) on the named research direction the
W21 milestone explicitly left open: combining the capsule-native
trust-weighted multi-oracle adjudicator with the LatentMAS direction
(collective KV pooling / latent hidden-state transfer / super-token
side channels) in a way that yields *measurable* efficiency gains
without weakening correctness or audit.

The W22 :class:`LatentDigestDisambiguator` implements the smallest
honest hybrid this repo can verify end-to-end:

  * **schema-passing** — closed-vocabulary type schema is content-
    addressed (:class:`SchemaCapsule`) and shared across roles via
    CID. The bundle carries the CID once per session, not the full
    schema text per cell.
  * **delta execution** — instead of replaying the verbose W21
    audit (per-oracle probe records, per-tag counts, raw oracle
    payload bytes) into the final decoder, the W22 layer emits a
    single :class:`LatentDigestEnvelope` per cell — the per-tag
    vote summary + projected subset + provenance — and calls
    :func:`verify_latent_digest` to gate it.
  * **shared-read cache** — :class:`SharedReadCache` is a CID-keyed
    write-once-read-many proxy for the LatentMAS shared-KV-read
    idea. :class:`CachingOracleAdapter` wraps any
    :class:`OutsideWitnessOracle` and routes every
    ``consult(query)`` through the cache; identical OutsideQueries
    across cells collapse to one wire-side oracle call.
  * **controller-side verification** — :func:`verify_latent_digest`
    enforces (a) hash chain (envelope ``digest_cid`` recomputes
    from canonical bytes), (b) schema-CID match, (c) parent-CID
    seal. On any failure the W22 layer rejects the digest and
    falls through to the W21 answer; the trust boundary is
    explicit and the explicit-capsule path is never bypassed.

Phase-69 sub-banks
==================

Five pre-committed cells.  The bundle shape is the R-66-OUTSIDE-
REQUIRED + R-68-MULTI-MAJORITY oracle ecology used by SDK v3.22; the
*per-bank* knobs vary the cache-and-trust topology.

* **R-69-CACHE-FANOUT** (W22-1 efficiency anchor; default config).
  Every cell of the bank shares a single :class:`SharedReadCache`
  (the bench driver creates one cache per bank execution and plumbs
  it through to every cell's :class:`CachingOracleAdapter`).
  Identical OutsideQueries (same admitted_tags, same
  elected_root_cause, same oracle_id) across cells collapse to one
  wire-side oracle call. The W22 layer emits one envelope per cell
  in place of the verbose W21 audit. **Strict success bar**:
  ``cache_tokens_saved_total > 0`` AND
  ``mean_n_visible_tokens_to_decider`` strictly below the W21
  baseline AND no regression on ``accuracy_full`` (= W21 baseline).
* **R-69-NO-CACHE** (W22-Λ-no-cache falsifier).  Each cell uses a
  fresh :class:`SharedReadCache` (the cache is reset per cell). No
  cross-cell wire-side savings; ``cache_tokens_saved_total == 0``.
  The W22 envelope still compresses the per-cell audit (digest vs
  verbose), so ``mean_n_visible_tokens_to_decider`` may still be
  below the W21 baseline due to the digest, but the *cache*
  contribution alone is null. Names the structural limit when there
  are no repeated reads.
* **R-69-POISONED-DIGEST** (W22-3 trust-boundary anchor).  An
  :class:`EnvelopeTamperer` (mode ``flip_projected_subset``) flips
  the envelope's ``projected_subset`` after the W21 vote but before
  the controller verifies. The recomputed ``digest_cid`` no longer
  matches the envelope's stored ``digest_cid``;
  :func:`verify_latent_digest` rejects with
  ``reason="hash_mismatch"``; W22 fires
  :data:`W22_BRANCH_LATENT_REJECTED`; the explicit-capsule path
  (W21's answer) stays sound. **Strict requirement**:
  ``verification_ok_rate == 0.0`` AND
  ``accuracy_full == W21 baseline`` (the rejection does NOT
  corrupt the answer).
* **R-69-SCHEMA-DRIFT** (W22-3 trust-boundary anchor).  The
  envelope is signed against the *production* schema but the
  controller registers a *different* schema (e.g.
  ``version="v9"``). :func:`verify_latent_digest` rejects with
  ``reason="schema_cid_mismatch"``; W22 falls through to W21.
  Same strict requirement as POISONED-DIGEST.
* **R-69-NO-TRIGGER** (W22 backward-compat anchor).  A cell where
  the inner W21 abstains (e.g. ``W21_BRANCH_NO_QUORUM`` because the
  registered oracles disagree completely). W22 fires
  :data:`W22_BRANCH_NO_TRIGGER` and reduces to W21 byte-for-byte;
  no envelope is emitted; ``digest_n_tokens == 0``. Backward-compat
  test that W22 doesn't fire on non-trigger branches.

Theorem family W22 (minted by this milestone)
==============================================

* **W22-1** (proved-conditional + proved-empirical, n=8 saturated × 2
  cells × 5 seeds). On R-69-CACHE-FANOUT, pairing the W21
  :class:`TrustWeightedMultiOracleDisambiguator` with the W22
  :class:`LatentDigestDisambiguator` over a shared
  :class:`SharedReadCache` (and the same R-68-MULTI-MAJORITY oracle
  registry) strictly improves ``mean_n_visible_tokens_to_decider``
  over the W21 baseline by an empirically-measured margin AND
  records ``cache_tokens_saved_total > 0`` AND ties W21 byte-for-
  byte on ``accuracy_full``.
* **W22-2** (proved-empirical, byte-for-byte). On every cell of
  R-69-CACHE-FANOUT, the W22 ``answer["services"]`` equals the W21
  ``answer["services"]`` byte-for-byte. The latent digest is a
  *summary* of the W21 vote outcome, not a re-projection. The W22
  layer's correctness is therefore exactly W21's correctness on
  this bench.
* **W22-3** (proved-empirical + proved by inspection). On
  R-69-POISONED-DIGEST and R-69-SCHEMA-DRIFT, every tampered
  envelope is rejected by :func:`verify_latent_digest`; the W22
  layer fires :data:`W22_BRANCH_LATENT_REJECTED` on every cell;
  the W22 answer field equals the W21 answer field byte-for-byte
  (the rejection does NOT corrupt the answer; it merely flags
  the digest as untrusted). The verification function is short
  and the failure modes are enumerated; soundness holds by
  inspection.
* **W22-Λ-no-cache** (proved-empirical, named falsifier). On
  R-69-NO-CACHE, ``cache_tokens_saved_total == 0`` by
  construction; the wire-side savings claim of W22-1 does NOT
  hold. The digest-only contribution to
  ``mean_n_visible_tokens_to_decider`` is preserved but the
  cache-tokens-saved contribution is null. Names the structural
  limit when there are no repeated reads to amortise.
* **W22-Λ-real** (proved-conditional + empirical-research). Live-
  LLM transfer of the cache+digest stack on Mac-1 Ollama is
  conditional on the LLM emitting deterministic replies for
  identical OutsideQueries. At ``temperature=0`` modern Ollama
  models are byte-stable on identical prompts; the cache hit
  path returns the recorded reply bytes verbatim and verification
  passes. The bench reports the live-LLM cache hit rate as a
  diagnostic.

Phase-69 also closes a partial discharge of the SDK v3.22
W21-C-CALIBRATED-TRUST conjecture *in the wire-cost direction*: the
W22 cache discharges the cost concern of consulting all N oracles
on every cell. The W21-C-CALIBRATED-TRUST conjecture in the
*correctness* direction (low trust priors on uncalibrated oracles)
remains open and is orthogonal to W22.

CLI
---

::

    # R-69-CACHE-FANOUT (W22-1 anchor, synthetic):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank cache_fanout --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-69-CACHE-FANOUT-TIGHT (W22-1 + W15 composition):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank cache_fanout --decoder-budget 24 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-69-NO-CACHE (W22-Λ-no-cache falsifier):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank no_cache --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-69-POISONED-DIGEST (W22-3 trust falsifier):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank poisoned_digest --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-69-SCHEMA-DRIFT (W22-3 trust falsifier):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank schema_drift --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-69-NO-TRIGGER (W22 backward-compat):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank no_trigger --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

    # Seed-stability sweep on the headline regime:
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank cache_fanout --seed-sweep \\
        --K-auditor 12 --n-eval 8 --out -

    # Live LLM probe (Mac-1 Ollama; cache+digest with one LLM
    # adjudicator alongside two deterministic oracles):
    python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \\
        --bank cache_fanout --live-llm-adjudicator \\
        --adjudicator-model mixtral:8x7b \\
        --K-auditor 12 --n-eval 4 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR,
    grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
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
    CrossRoleCorroborationAdmissionPolicy,
    EnvelopeTamperer,
    FifoAdmissionPolicy, FifoContextPacker,
    LatentDigestDisambiguator,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    OnCallNotesOracle,
    OracleRegistration,
    OutsideWitnessAcquisitionDisambiguator,
    RelationalCompatibilityDisambiguator,
    RobustMultiRoundBundleDecoder, RoleBudget,
    SchemaCapsule, ServiceGraphOracle, SharedReadCache,
    SingletonAsymmetricOracle,
    TeamCoordinator, audit_team_lifecycle,
    TrustWeightedMultiOracleDisambiguator,
    build_incident_triage_schema_capsule,
    collect_admitted_handoffs, _DecodedHandoff,
    W22_BRANCH_LATENT_RESOLVED, W22_BRANCH_LATENT_REJECTED,
    W22_BRANCH_NO_TRIGGER,
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
    build_phase67_bank,
    _bench_property_p67, _P67_EXPECTED_SHAPE,
)
from vision_mvp.experiments.phase68_multi_oracle_adjudication import (
    build_phase68_bank,
)


# =============================================================================
# Phase-69 banks — same R-68-MULTI-MAJORITY bundle / oracle ecology;
# the *cache-and-trust topology* is the bank-specific dimension.
# =============================================================================

_VALID_BANKS_P69 = (
    "cache_fanout",     # W22-1 anchor — shared cache across cells.
    "no_cache",         # W22-Λ-no-cache falsifier — fresh cache per cell.
    "poisoned_digest",  # W22-3 — envelope tamper, expect rejection.
    "schema_drift",     # W22-3 — schema mismatch, expect rejection.
    "no_trigger",       # W22 backward-compat — W21 abstains, W22 reduces.
)


def _w22_oracle_registrations(
        cache: SharedReadCache, *,
        with_compromised: bool = True,
        llm_adjudicator: Any | None = None,
        ) -> tuple[OracleRegistration, ...]:
    """Default W22 registry: same R-68-MULTI-MAJORITY shape as Phase 68
    BUT every oracle is wrapped in a :class:`CachingOracleAdapter`
    routing through ``cache``.

    The ordering matches Phase 68: compromised first (so the W20
    baseline picks it and fails), then the two honest deterministic
    oracles. W22 consults all three; the deterministic two form
    quorum=2 on gold; W22 emits one envelope summarising the vote.
    """
    inner_compromised = CompromisedServiceGraphOracle(
        oracle_id="compromised_registry")
    inner_service_graph = ServiceGraphOracle(oracle_id="service_graph")
    inner_change_history = ChangeHistoryOracle(oracle_id="change_history")
    regs: list[OracleRegistration] = []
    if with_compromised:
        regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=inner_compromised, cache=cache,
                oracle_id="compromised_registry"),
            trust_prior=0.8, role_label="compromised_registry"))
    regs.append(OracleRegistration(
        oracle=CachingOracleAdapter(
            inner=inner_service_graph, cache=cache,
            oracle_id="service_graph"),
        trust_prior=1.0, role_label="service_graph"))
    regs.append(OracleRegistration(
        oracle=CachingOracleAdapter(
            inner=inner_change_history, cache=cache,
            oracle_id="change_history"),
        trust_prior=1.0, role_label="change_history"))
    if llm_adjudicator is not None:
        regs.append(OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=llm_adjudicator, cache=cache,
                oracle_id=getattr(llm_adjudicator, "oracle_id",
                                     "llm_adjudicator")),
            trust_prior=0.7, role_label="llm_adjudicator"))
    return tuple(regs)


def _no_trigger_oracle_registrations(
        cache: SharedReadCache,
        ) -> tuple[OracleRegistration, ...]:
    """R-69-NO-TRIGGER registry — three :class:`SingletonAsymmetricOracle`s
    each pointing at a different admitted tag. No tag receives ≥
    ``quorum_min = 2`` votes; the inner W21 fires
    :data:`W21_BRANCH_NO_QUORUM`; the W22 layer correctly fires
    :data:`W22_BRANCH_NO_TRIGGER` and reduces to W21 byte-for-byte
    (the answer field is the W21 abstention; the digest is empty)."""
    return (
        OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=SingletonAsymmetricOracle(
                    target="first", oracle_id="singleton_first"),
                cache=cache, oracle_id="singleton_first"),
            trust_prior=1.0, role_label="singleton_first"),
        OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=SingletonAsymmetricOracle(
                    target="middle", oracle_id="singleton_middle"),
                cache=cache, oracle_id="singleton_middle"),
            trust_prior=1.0, role_label="singleton_middle"),
        OracleRegistration(
            oracle=CachingOracleAdapter(
                inner=SingletonAsymmetricOracle(
                    target="last", oracle_id="singleton_last"),
                cache=cache, oracle_id="singleton_last"),
            trust_prior=1.0, role_label="singleton_last"),
    )


def build_phase69_bank(*, n_replicates: int = 2, seed: int = 11,
                          ) -> list[MultiRoundScenario]:
    """Build the Phase-69 bank — same R-68-MULTI-MAJORITY bundle shape
    (which is the R-66-OUTSIDE-REQUIRED + Phase-67 outside-resolves
    family). The bundle is bank-invariant; the cache-and-trust
    topology is the bank-specific knob (set at strategy-dispatch
    time, not in the scenario)."""
    return build_phase68_bank(n_replicates=n_replicates, seed=seed)


# =============================================================================
# Strategy / decoder dispatch
# =============================================================================

# Phase-69 strategy roster: a focused subset that isolates the W22
# contribution. The full SDK ladder is run on the cross-regime
# synthetic summary; the per-bank runs use this lean roster so the
# token-efficiency comparison is apples-to-apples.
_R69_STRATEGIES: tuple[tuple[str, str], ...] = (
    # W21 baseline (no cache, no digest) — the apples-to-apples
    # comparison anchor for W22-1 and W22-2.
    ("capsule_multi_oracle", "multi_oracle"),
    # W21 baseline + cache only (no digest) — isolates the cache
    # contribution alone (efficiency without the digest).
    ("capsule_multi_oracle_cached", "multi_oracle_cached"),
    # W22 hybrid — cache + digest + verification (the load-bearing
    # SDK v3.23 result).
    ("capsule_w22_hybrid", "w22_hybrid"),
    # W22 hybrid with digest only (no cache) — isolates the digest
    # contribution alone.
    ("capsule_w22_hybrid_no_cache", "w22_hybrid_no_cache"),
)


def _make_factory_p69(name: str, priorities, budgets):
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


def _decode_with_w21_no_cache(
        union, T_decoder, round_index_hint, oracle_registrations, *,
        quorum_min: int = 2, min_trust_sum: float = 0.0):
    """W21 baseline — no cache, no digest. The apples-to-apples
    comparison anchor for W22 efficiency claims."""
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    per_round = _build_per_round(union, round_index_hint)
    ans = w21.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
        stats["n_handoffs_decoder_input"] = int(pack.n_kept)
        stats["n_handoffs_admitted"] = int(pack.n_input)
    if "multi_oracle" in ans:
        stats["multi_oracle"] = ans["multi_oracle"]
    return ans, stats


def _decode_with_w22(
        union, T_decoder, round_index_hint, oracle_registrations, *,
        quorum_min: int = 2, min_trust_sum: float = 0.0,
        schema: SchemaCapsule, cache: SharedReadCache | None,
        tamperer: EnvelopeTamperer | None = None,
        verifier_schema_override: SchemaCapsule | None = None,
        require_verification: bool = True,
        session_state: dict[str, Any] | None = None,
        ):
    """W22 hybrid — wrap the W21 stack with the LatentDigestDisambiguator.

    Parameters
    ----------
    schema
        The SchemaCapsule the *envelope* is signed against (the
        producer's view).
    cache
        SharedReadCache shared across cells (or None to disable
        shared-read accounting at the W22 audit; the cache is still
        owned by the caller and may have entries).
    tamperer
        Optional EnvelopeTamperer for falsifier tests
        (R-69-POISONED-DIGEST).
    verifier_schema_override
        If provided, the controller verifies the envelope against
        this schema (instead of ``schema``) — used by
        R-69-SCHEMA-DRIFT to install a different schema CID at the
        verifier.
    """
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    # The producer-side ``schema`` is the production schema the
    # envelope is signed against. The ``verifier_schema`` is the
    # controller's view; on schema drift these differ and the
    # W22 layer's verifier fires ``schema_cid_mismatch``.
    w22 = LatentDigestDisambiguator(
        inner=w21,
        schema=schema,  # producer signs against this
        verifier_schema=verifier_schema_override,  # controller view
        cache=cache,
        tamperer=tamperer,
        require_verification=require_verification,
    )
    # Thread session-level "schema already shared" flag from the bench
    # driver — this models a session that spans many cells, where the
    # SchemaCapsule is sent *once* and referenced by CID elsewhere.
    if session_state is not None and session_state.get(
            "schema_already_shared"):
        w22._schema_already_shared = True
    per_round = _build_per_round(union, round_index_hint)
    ans = w22.decode_rounds(per_round)
    # Mark the schema as shared after the first envelope is signed,
    # so subsequent cells claim the schema-shared bonus.
    if session_state is not None:
        if w22._schema_already_shared:
            session_state["schema_already_shared"] = True
        else:
            # First cell that produced an envelope — mark for next.
            if w22.last_envelope is not None:
                session_state["schema_already_shared"] = True
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
        stats["n_handoffs_decoder_input"] = int(pack.n_kept)
        stats["n_handoffs_admitted"] = int(pack.n_input)
    if "multi_oracle" in ans:
        stats["multi_oracle"] = ans["multi_oracle"]
    if "latent_hybrid" in ans:
        stats["latent_hybrid"] = ans["latent_hybrid"]
    if "latent_envelope" in ans:
        stats["latent_envelope"] = ans["latent_envelope"]
    return ans, stats


def _run_capsule_strategy_p69(
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands, round2_cands,
        T_decoder: int | None = None,
        oracle_registrations: tuple[OracleRegistration, ...] = (),
        quorum_min: int = 2,
        min_trust_sum: float = 0.0,
        schema: SchemaCapsule,
        shared_cache: SharedReadCache | None = None,
        per_cell_cache: bool = False,
        tamperer: EnvelopeTamperer | None = None,
        verifier_schema_override: SchemaCapsule | None = None,
        session_state: dict[str, Any] | None = None,
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase69_capsule_latent_hybrid",
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

    if decoder_mode == "multi_oracle":
        # W21 baseline — no cache wrapping, the registrations should
        # be the un-wrapped variants. We rebuild registrations from
        # the inner of any CachingOracleAdapter.
        bare_regs = tuple(
            OracleRegistration(
                oracle=(reg.oracle.inner
                          if isinstance(reg.oracle, CachingOracleAdapter)
                          else reg.oracle),
                trust_prior=reg.trust_prior,
                role_label=reg.role_label)
            for reg in oracle_registrations)
        answer, pack_stats = _decode_with_w21_no_cache(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=bare_regs,
            quorum_min=quorum_min,
            min_trust_sum=min_trust_sum)
    elif decoder_mode == "multi_oracle_cached":
        # W21 with shared-cache wrapping — isolates the cache
        # contribution alone. No digest emission.
        regs = oracle_registrations
        answer, pack_stats = _decode_with_w21_no_cache(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=regs,
            quorum_min=quorum_min,
            min_trust_sum=min_trust_sum)
    elif decoder_mode == "w22_hybrid":
        # The load-bearing W22 strategy — cache + digest +
        # verification.
        cache = (
            SharedReadCache() if per_cell_cache else shared_cache)
        regs = oracle_registrations
        if per_cell_cache:
            # Re-wrap registrations with a fresh cache for this cell.
            regs = tuple(
                OracleRegistration(
                    oracle=CachingOracleAdapter(
                        inner=(reg.oracle.inner
                                  if isinstance(reg.oracle, CachingOracleAdapter)
                                  else reg.oracle),
                        cache=cache,
                        oracle_id=reg.role_label),
                    trust_prior=reg.trust_prior,
                    role_label=reg.role_label)
                for reg in oracle_registrations)
        answer, pack_stats = _decode_with_w22(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=regs,
            quorum_min=quorum_min,
            min_trust_sum=min_trust_sum,
            schema=schema, cache=cache,
            tamperer=tamperer,
            verifier_schema_override=verifier_schema_override,
            session_state=session_state)
    elif decoder_mode == "w22_hybrid_no_cache":
        # W22 with digest only — fresh per-cell cache so wire-side
        # savings are zero. Isolates the digest contribution.
        local_cache = SharedReadCache()
        regs = tuple(
            OracleRegistration(
                oracle=CachingOracleAdapter(
                    inner=(reg.oracle.inner
                              if isinstance(reg.oracle, CachingOracleAdapter)
                              else reg.oracle),
                    cache=local_cache,
                    oracle_id=reg.role_label),
                trust_prior=reg.trust_prior,
                role_label=reg.role_label)
            for reg in oracle_registrations)
        answer, pack_stats = _decode_with_w22(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=regs,
            quorum_min=quorum_min,
            min_trust_sum=min_trust_sum,
            schema=schema, cache=local_cache,
            tamperer=tamperer,
            verifier_schema_override=verifier_schema_override)
    else:
        raise ValueError(f"unknown decoder_mode {decoder_mode!r}")

    coord.seal_team_decision(
        team_role=ROLE_AUDITOR, decision=answer,
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


# =============================================================================
# Phase 69 driver
# =============================================================================


def run_phase69(*,
                  bank: str = "cache_fanout",
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
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 69 over one of {cache_fanout, no_cache, poisoned_digest,
    schema_drift, no_trigger}."""
    if bank not in _VALID_BANKS_P69:
        raise ValueError(
            f"unknown bank {bank!r}; valid: {_VALID_BANKS_P69}")
    bank_obj = build_phase69_bank(
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
    # The verifier's schema can differ from the producer's schema —
    # used by R-69-SCHEMA-DRIFT to install a *different* schema at
    # the controller and trigger schema_cid_mismatch.
    verifier_schema_override: SchemaCapsule | None = None
    if bank == "schema_drift":
        verifier_schema_override = dataclasses.replace(
            schema, version="v9_drifted")

    # The bank-specific tamperer.
    tamperer: EnvelopeTamperer | None = None
    if bank == "poisoned_digest":
        tamperer = EnvelopeTamperer(
            mode="flip_projected_subset",
            admitted_tags=tuple(schema.closed_vocab_services))

    # The bank-specific oracle registry: cache_fanout / no_cache /
    # poisoned_digest / schema_drift use the R-68-MULTI-MAJORITY
    # oracle ecology (compromised first, then service_graph and
    # change_history); no_trigger uses three singleton oracles so
    # the inner W21 abstains.
    shared_cache: SharedReadCache | None = SharedReadCache()
    per_cell_cache = (bank == "no_cache")
    if bank == "no_trigger":
        oracle_regs = _no_trigger_oracle_registrations(shared_cache)
    else:
        oracle_regs = _w22_oracle_registrations(
            shared_cache, with_compromised=True,
            llm_adjudicator=llm_adjudicator)

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R69_STRATEGIES
    }

    # The W22 layer is *stateful across cells*: ``_schema_already_shared``
    # toggles to True after the first envelope is signed, so cell 2..N
    # claim the schema-shared bonus. We pass a session_state dict that
    # the per-cell dispatch threads through to the LatentDigestDisambig
    # uator (so the bench driver can keep one logical W22 session
    # across the bank's cells).
    session_state: dict[str, Any] = {
        "schema_already_shared": False,
        "n_cells_processed": 0,
    }

    for sc in bank_obj:
        round1_cands = _build_round_candidates_p66(sc.round1_emissions)
        round2_cands = _build_round_candidates_p66(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p67(
            sc, round1_cands, round2_cands)
        for (sname, dmode) in _R69_STRATEGIES:
            fac = _make_factory_p69(sname, priorities, budgets)
            r, ps = _run_capsule_strategy_p69(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                T_decoder=T_decoder,
                oracle_registrations=oracle_regs,
                quorum_min=quorum_min,
                min_trust_sum=min_trust_sum,
                schema=schema,
                shared_cache=shared_cache,
                per_cell_cache=per_cell_cache,
                tamperer=tamperer,
                verifier_schema_override=verifier_schema_override,
                session_state=session_state)
            results.append(r)
            if ps:
                pack_stats_per_strategy[sname].append({
                    "scenario_id": sc.scenario_id,
                    **ps,
                })

    strategy_names = tuple(s[0] for s in _R69_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w22_minus_w21": _gap(
            "capsule_w22_hybrid", "capsule_multi_oracle"),
        "w22_cached_minus_uncached": _gap(
            "capsule_w22_hybrid", "capsule_w22_hybrid_no_cache"),
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
        "expected_shape": list(_P67_EXPECTED_SHAPE),
        "K_auditor": K_auditor,
        "T_decoder": T_decoder,
        "bank": bank,
        "quorum_min": int(quorum_min),
        "min_trust_sum": float(min_trust_sum),
        "schema_cid": schema.cid,
        "schema_n_canonical_tokens": schema.n_canonical_tokens,
        "verifier_schema_cid": (
            verifier_schema_override.cid
            if verifier_schema_override is not None else schema.cid),
        "tamperer_mode": (tamperer.mode if tamperer is not None else None),
        "shared_cache_used": (shared_cache is not None and not per_cell_cache),
        "n_oracle_registrations": len(oracle_regs),
        "oracle_role_labels": [r.role_label for r in oracle_regs],
    }

    def _agg_w22(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)

        def get_lh(r):
            return r.get("latent_hybrid") or {}

        def get_mo(r):
            return r.get("multi_oracle") or {}

        s_visible = sum(int(get_lh(r).get(
            "n_visible_tokens_to_decider", 0)) for r in rows)
        s_digest_n = sum(int(get_lh(r).get(
            "digest_n_tokens", 0)) for r in rows)
        s_kept = sum(int(get_lh(r).get(
            "n_w15_tokens_kept", 0)) or int(r.get("tokens_kept", 0))
                      for r in rows)
        s_w21_outside = sum(int(get_lh(r).get(
            "n_w21_outside_tokens_total", 0)) for r in rows)
        s_w21_audit = sum(int(get_lh(r).get(
            "n_w21_verbose_audit_tokens", 0)) for r in rows)
        s_cache_saved = sum(int(get_lh(r).get(
            "cache_tokens_saved_this_cell", 0)) for r in rows)
        s_schema_saved = sum(int(get_lh(r).get(
            "schema_shared_tokens_saved_this_cell", 0)) for r in rows)
        s_cache_hits = sum(int(get_lh(r).get(
            "n_cache_hits_this_cell", 0)) for r in rows)
        s_cache_misses = sum(int(get_lh(r).get(
            "n_cache_misses_this_cell", 0)) for r in rows)
        n_triggered = sum(
            1 for r in rows if get_lh(r).get("triggered"))
        n_verified_ok = sum(
            1 for r in rows if get_lh(r).get("verification_ok"))
        n_resolved = sum(
            1 for r in rows
            if get_lh(r).get("decoder_branch") == W22_BRANCH_LATENT_RESOLVED)
        n_rejected = sum(
            1 for r in rows
            if get_lh(r).get("decoder_branch") == W22_BRANCH_LATENT_REJECTED)
        n_no_trigger = sum(
            1 for r in rows
            if get_lh(r).get("decoder_branch") == W22_BRANCH_NO_TRIGGER)
        rejection_reasons: dict[str, int] = {}
        for r in rows:
            if get_lh(r).get("decoder_branch") == W22_BRANCH_LATENT_REJECTED:
                reason = str(get_lh(r).get("verification_reason", "unknown"))
                rejection_reasons[reason] = (
                    rejection_reasons.get(reason, 0) + 1)
        denom_audit = max(1, s_w21_audit)
        compression = (s_digest_n / denom_audit) if denom_audit else 0.0
        return {
            "n_cells": n,
            "n_w22_triggered_cells": int(n_triggered),
            "n_w22_resolved_cells": int(n_resolved),
            "n_w22_rejected_cells": int(n_rejected),
            "n_w22_no_trigger_cells": int(n_no_trigger),
            "verification_ok_rate": round(n_verified_ok / n, 4) if n else 0.0,
            "rejection_reasons": rejection_reasons,
            "n_visible_tokens_to_decider_sum": int(s_visible),
            "n_visible_tokens_to_decider_avg": (
                round(s_visible / n, 4) if n else 0.0),
            "digest_n_tokens_sum": int(s_digest_n),
            "digest_n_tokens_avg": (
                round(s_digest_n / n, 4) if n else 0.0),
            "w15_tokens_kept_sum": int(s_kept),
            "w21_outside_tokens_total_sum": int(s_w21_outside),
            "w21_verbose_audit_tokens_sum": int(s_w21_audit),
            "digest_compression_ratio_overall":
                round(float(compression), 6),
            "cache_tokens_saved_total": int(s_cache_saved),
            "cache_tokens_saved_avg": (
                round(s_cache_saved / n, 4) if n else 0.0),
            "schema_shared_tokens_saved_total": int(s_schema_saved),
            "n_cache_hits_total": int(s_cache_hits),
            "n_cache_misses_total": int(s_cache_misses),
            "cache_hit_rate": (
                round(s_cache_hits / max(1, s_cache_hits + s_cache_misses), 4)
                if (s_cache_hits + s_cache_misses) > 0 else 0.0),
        }

    def _agg_w21(rows: list[dict[str, Any]]) -> dict[str, Any]:
        # Apples-to-apples baseline accounting for the W21-only
        # strategy: the visible context in the no-W22 path is the
        # W15-kept bundle + the verbose W21 audit + the W21 outside
        # tokens summed.
        if not rows:
            return {}
        n = len(rows)
        s_kept = sum(int(r.get("tokens_kept", 0)) for r in rows)
        s_w21_outside = sum(int(r.get("multi_oracle", {}).get(
            "n_outside_tokens_total", 0)) for r in rows)
        # Verbose audit cost — what the W21 audit dict is when
        # whitespace-tokenised at the same convention.
        from vision_mvp.wevra.team_coord import _whitespace_token_count
        s_w21_audit = 0
        for r in rows:
            mo = r.get("multi_oracle") or {}
            if mo:
                s_w21_audit += _whitespace_token_count(json.dumps(
                    mo, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True))
        s_visible = s_kept + s_w21_outside + s_w21_audit
        return {
            "n_cells": n,
            "n_visible_tokens_to_decider_sum": int(s_visible),
            "n_visible_tokens_to_decider_avg": (
                round(s_visible / n, 4) if n else 0.0),
            "w15_tokens_kept_sum": int(s_kept),
            "w21_outside_tokens_total_sum": int(s_w21_outside),
            "w21_verbose_audit_tokens_sum": int(s_w21_audit),
        }

    pack_stats_summary = {
        s: (_agg_w22(pack_stats_per_strategy.get(s, []))
             if s in ("capsule_w22_hybrid",
                       "capsule_w22_hybrid_no_cache")
             else _agg_w21(pack_stats_per_strategy.get(s, [])))
        for s in strategy_names
    }

    # Efficiency comparison: W22 hybrid vs W21 baseline.
    eff_w21 = pack_stats_summary.get("capsule_multi_oracle", {})
    eff_w22 = pack_stats_summary.get("capsule_w22_hybrid", {})
    eff_w22_no_cache = pack_stats_summary.get(
        "capsule_w22_hybrid_no_cache", {})
    eff_compare = {
        "w21_visible_tokens_per_cell":
            float(eff_w21.get("n_visible_tokens_to_decider_avg", 0.0)),
        "w22_visible_tokens_per_cell":
            float(eff_w22.get("n_visible_tokens_to_decider_avg", 0.0)),
        "w22_no_cache_visible_tokens_per_cell":
            float(eff_w22_no_cache.get("n_visible_tokens_to_decider_avg", 0.0)),
        "visible_tokens_savings_per_cell": round(
            float(eff_w21.get("n_visible_tokens_to_decider_avg", 0.0))
            - float(eff_w22.get("n_visible_tokens_to_decider_avg", 0.0)),
            4),
        "visible_tokens_savings_pct": (round(
            (float(eff_w21.get("n_visible_tokens_to_decider_avg", 0.0))
             - float(eff_w22.get("n_visible_tokens_to_decider_avg", 0.0)))
            / max(1.0,
                   float(eff_w21.get("n_visible_tokens_to_decider_avg", 0.0)))
            * 100, 4)),
        "cache_tokens_saved_total":
            int(eff_w22.get("cache_tokens_saved_total", 0)),
        "cache_hit_rate":
            float(eff_w22.get("cache_hit_rate", 0.0)),
        "digest_compression_ratio":
            float(eff_w22.get("digest_compression_ratio_overall", 0.0)),
        "verification_ok_rate":
            float(eff_w22.get("verification_ok_rate", 0.0)),
    }

    # Correctness ratification check (W22-2): on every cell, the W22
    # answer's services field must equal the W21 answer's services
    # field. We surface this as a cell-level boolean array.
    correctness_ratified_cells: list[bool] = []
    for sc in bank_obj:
        w21_r = next((r for r in results
                       if r.strategy == "capsule_multi_oracle"
                       and r.scenario_id == sc.scenario_id), None)
        w22_r = next((r for r in results
                       if r.strategy == "capsule_w22_hybrid"
                       and r.scenario_id == sc.scenario_id), None)
        if w21_r is None or w22_r is None:
            continue
        sv21 = tuple(sorted(map(str, w21_r.answer.get("services", ()))))
        sv22 = tuple(sorted(map(str, w22_r.answer.get("services", ()))))
        correctness_ratified_cells.append(sv21 == sv22)
    correctness_ratified_rate = (
        round(sum(correctness_ratified_cells)
              / max(1, len(correctness_ratified_cells)), 4))

    if verbose:
        print(f"[phase69] bank={bank} schema_cid={schema.cid[:16]} "
              f"verifier_schema_cid={(verifier_schema_override.cid[:16] if verifier_schema_override else schema.cid[:16])} "
              f"tamperer={tamperer.mode if tamperer else 'none'} "
              f"per_cell_cache={per_cell_cache} "
              f"T_decoder={T_decoder} n_eval={len(bank_obj)} "
              f"K_auditor={K_auditor} bank_seed={bank_seed}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase69]   {s:36s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase69] {k}: {v:+.3f}", file=sys.stderr, flush=True)
        print(f"[phase69] visible_tokens_per_cell w21={eff_compare['w21_visible_tokens_per_cell']:.2f} "
              f"w22={eff_compare['w22_visible_tokens_per_cell']:.2f} "
              f"savings_pct={eff_compare['visible_tokens_savings_pct']:+.2f}%",
              file=sys.stderr, flush=True)
        print(f"[phase69] cache_tokens_saved_total={eff_compare['cache_tokens_saved_total']} "
              f"hit_rate={eff_compare['cache_hit_rate']:.3f} "
              f"verification_ok_rate={eff_compare['verification_ok_rate']:.3f}",
              file=sys.stderr, flush=True)
        print(f"[phase69] digest_compression_ratio={eff_compare['digest_compression_ratio']:.4f} "
              f"correctness_ratified_rate={correctness_ratified_rate:.3f}",
              file=sys.stderr, flush=True)

    return {
        "schema": "phase69.capsule_latent_hybrid.v1",
        "config": {
            "bank": bank, "n_eval": len(bank_obj),
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "K_producer": K_producer, "T_producer": T_producer,
            "inbox_capacity": inbox_capacity, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates, "T_decoder": T_decoder,
            "quorum_min": int(quorum_min),
            "min_trust_sum": float(min_trust_sum),
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "pack_stats_summary": pack_stats_summary,
        "eff_compare": eff_compare,
        "correctness_ratified_rate": correctness_ratified_rate,
        "correctness_ratified_cells_count": int(
            sum(correctness_ratified_cells)),
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }


def run_phase69_seed_stability_sweep(*,
                                          bank: str = "cache_fanout",
                                          T_decoder: int | None = None,
                                          n_eval: int = 8,
                                          K_auditor: int = 12,
                                          quorum_min: int = 2,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase69.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "quorum_min": int(quorum_min),
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase69(bank=bank, T_decoder=T_decoder,
                            n_eval=n_eval, K_auditor=K_auditor,
                            bank_seed=seed, quorum_min=quorum_min)
        out["per_seed"][str(seed)] = {
            "headline_gap": rep["headline_gap"],
            "pooled": {
                k: v["accuracy_full"]
                for k, v in rep["pooled"].items()
            },
            "eff_compare": rep["eff_compare"],
            "correctness_ratified_rate": rep["correctness_ratified_rate"],
            "audit_ok_grid": rep["audit_ok_grid"],
        }
    savings = [
        out["per_seed"][str(s)]["eff_compare"][
            "visible_tokens_savings_per_cell"]
        for s in seeds
    ]
    out["min_visible_tokens_savings_per_cell"] = (
        min(savings) if savings else 0.0)
    out["max_visible_tokens_savings_per_cell"] = (
        max(savings) if savings else 0.0)
    out["mean_visible_tokens_savings_per_cell"] = (
        round(sum(savings) / len(savings), 4) if savings else 0.0)
    cache_saved = [
        out["per_seed"][str(s)]["eff_compare"]["cache_tokens_saved_total"]
        for s in seeds
    ]
    out["min_cache_tokens_saved_total"] = (
        min(cache_saved) if cache_saved else 0.0)
    out["mean_cache_tokens_saved_total"] = (
        round(sum(cache_saved) / len(cache_saved), 4)
        if cache_saved else 0.0)
    correct = [
        out["per_seed"][str(s)]["correctness_ratified_rate"] for s in seeds
    ]
    out["min_correctness_ratified_rate"] = (
        min(correct) if correct else 0.0)
    out["mean_correctness_ratified_rate"] = (
        round(sum(correct) / len(correct), 4) if correct else 0.0)
    return out


def run_cross_regime_synthetic_p69(*,
                                      n_eval: int = 8,
                                      bank_seed: int = 11,
                                      K_auditor: int = 12,
                                      T_auditor: int = 256,
                                      T_decoder_tight: int = 24,
                                      quorum_min: int = 2,
                                      ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase69.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
            "quorum_min": int(quorum_min),
        },
        "regimes": {},
    }
    cells = [
        ("R-69-CACHE-FANOUT-LOOSE",   "cache_fanout",     None),
        ("R-69-CACHE-FANOUT-TIGHT",   "cache_fanout",     T_decoder_tight),
        ("R-69-NO-CACHE",              "no_cache",         None),
        ("R-69-POISONED-DIGEST",       "poisoned_digest",  None),
        ("R-69-SCHEMA-DRIFT",          "schema_drift",     None),
        ("R-69-NO-TRIGGER",            "no_trigger",       None),
    ]
    for (regime_name, bank, T_dec) in cells:
        rep = run_phase69(
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
            "correctness_ratified_rate": rep["correctness_ratified_rate"],
            "audit_ok_grid": rep["audit_ok_grid"],
        }
    return out


def _make_llm_adjudicator(model: str, *, base_url: str | None = None,
                           timeout: float = 120.0):
    """Build an :class:`LLMAdjudicatorOracle` against a live Ollama
    backend (Mac-1 default). Used for the W22-Λ-real probe."""
    from vision_mvp.wevra.team_coord import LLMAdjudicatorOracle
    from vision_mvp.wevra.llm_backend import OllamaBackend
    backend_url = base_url or os.environ.get(
        "WEVRA_OLLAMA_URL", "http://127.0.0.1:11434")
    backend = OllamaBackend(model=model, base_url=backend_url,
                              timeout=timeout)
    return LLMAdjudicatorOracle(
        oracle_id=f"ollama_{model}", backend=backend,
        max_response_tokens=24)


# =============================================================================
# CLI
# =============================================================================


def _cli() -> None:
    ap = argparse.ArgumentParser(prog="phase69_capsule_latent_hybrid")
    ap.add_argument("--bank", default="cache_fanout",
                     choices=_VALID_BANKS_P69)
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
                     help="run all R-69 regimes back-to-back")
    ap.add_argument("--live-llm-adjudicator", action="store_true",
                     help="add an LLMAdjudicatorOracle to the registry")
    ap.add_argument("--adjudicator-model", default="mixtral:8x7b")
    ap.add_argument("--adjudicator-base-url", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    T_decoder = None if args.decoder_budget < 0 else int(args.decoder_budget)
    llm_adj = None
    if args.live_llm_adjudicator:
        llm_adj = _make_llm_adjudicator(
            args.adjudicator_model,
            base_url=args.adjudicator_base_url)

    if args.cross_regime_synthetic:
        out = run_cross_regime_synthetic_p69(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            T_decoder_tight=24, quorum_min=args.quorum_min)
    elif args.seed_sweep:
        out = run_phase69_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            quorum_min=args.quorum_min)
    else:
        out = run_phase69(
            bank=args.bank, n_eval=args.n_eval,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            K_producer=args.K_producer, T_producer=args.T_producer,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            T_decoder=T_decoder, quorum_min=args.quorum_min,
            min_trust_sum=args.min_trust_sum,
            llm_adjudicator=llm_adj,
            verbose=args.verbose)

    if args.out == "-":
        print(json.dumps(out, indent=2, default=str))
    else:
        with open(args.out, "w") as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f"[phase69] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
