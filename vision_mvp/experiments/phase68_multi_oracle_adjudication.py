"""Phase 68 — trust-weighted multi-oracle adjudication under partial
oracle compromise (SDK v3.22, W21 family anchor).

The follow-up to SDK v3.21 (W20) on the named research frontier the W20
milestone explicitly left conjectural: the **multi-source escape**
(W20-C-MULTI-ORACLE) from the W20-Λ-compromised wall — *one* registered
oracle is adversarial AND its reply is the only outside signal the
runtime sees, so W20 trusts it and FAILS.

The W20-Λ-compromised anchor (Phase-67 R-67-OUTSIDE-COMPROMISED) proves
that *any* single-oracle method that trusts its registered oracle
projects to whatever the oracle says — wins when the oracle is honest,
fails when it is compromised. SDK v3.21 named the next research move
as **multi-oracle adjudication** — a scorer that consults *every*
registered outside source and projects only when a quorum of
independently-replying oracles agrees on a non-empty proper asymmetric
subset.

W21's :class:`TrustWeightedMultiOracleDisambiguator` is the smallest
move in that direction. Phase-68 is the named harder regime where the
W21 method's required ingredient — at least ``quorum_min`` independently
honest oracles among the registered set — is consistently present
across the *positive-anchor* sub-bank (R-68-MULTI-MAJORITY) AND where
three named falsifier regimes (R-68-MULTI-NO-QUORUM,
R-68-MULTI-ALL-COMPROMISED, R-68-MULTI-PARTIAL) make the W21
conditionality sharp.

Recap of why the W20-Λ-compromised wall stands without W21
==========================================================

On R-67-OUTSIDE-COMPROMISED the bench property is:

  * The R-66-OUTSIDE-REQUIRED bundle shape (deceptive primary +
    symmetric secondary witness) — every closed-form bundle-only
    scorer ties FIFO at 0.000 by abstention.
  * Exactly one outside oracle is registered, and it returns a
    *decoy-asymmetric* reply for every cell (a corrupted service
    registry / compromised LLM adjudicator).

Under this regime:

  * The W19 inner abstains via :data:`W19_BRANCH_ABSTAINED_SYMMETRIC`.
  * W20 fires :data:`W20_BRANCH_OUTSIDE_RESOLVED` (the oracle's reply
    is asymmetric across admitted tags — the projection rule fires)
    and trusts the oracle's decoy-only reply; FAILS at 0.000.
  * **No single-oracle method can detect the compromise** because
    it has no second source to cross-check against. The escape is
    multi-source aggregation.

Phase-68 sub-banks
==================

Five pre-committed cells plus a cross-regime synthetic summary.
Every sub-bank uses the same R-66-OUTSIDE-REQUIRED bundle shape; the
*only* variable is the **registered oracle set**.

* **R-68-SINGLE-CLEAN** (W21-3-B reduces-to-W20 anchor, *trade-off*
  baseline; one honest :class:`ServiceGraphOracle`,
  ``T_decoder=None``). Single-oracle regime: W20 wins (the
  ServiceGraphOracle reply is asymmetric on gold). W21 with default
  ``quorum_min = 2`` abstains via :data:`W21_BRANCH_NO_QUORUM`
  (only one vote per gold tag, below the quorum threshold). W21
  with ``quorum_min = 1`` (override) ties W20 byte-for-byte. The
  trade-off makes the quorum knob explicit.
* **R-68-MULTI-MAJORITY** (W21-1 anchor; three oracles —
  :class:`CompromisedServiceGraphOracle` registered FIRST,
  :class:`ServiceGraphOracle`, :class:`ChangeHistoryOracle` —
  ``T_decoder=None``). The W20 baseline (single-oracle interface,
  picks the first registered = compromised) FAILS at 0.000. The
  W21 method consults all three; gold tags receive 2 votes (from
  ServiceGraph + ChangeHistory), decoy receives 1; quorum_min = 2
  fires :data:`W21_BRANCH_QUORUM_RESOLVED` and projects the answer
  to the gold pair. **+1.000 strict gain over W20** (which trusts
  the compromised oracle on the same regime).
* **R-68-MULTI-MAJORITY-TIGHT** (W21-1 + W15 composition anchor;
  same three oracles, ``T_decoder=24``). Same R-68-MULTI-MAJORITY
  shape under decoder-side budget pressure. The W21 layer reads
  only ``query.admitted_tags``, NOT the full bundle; the inner
  W15 ``tokens_kept`` is byte-for-byte unchanged. The W21
  ``n_outside_tokens_total`` records the strict additional cost
  of N oracle queries (≤ N × ``max_response_tokens``).
* **R-68-MULTI-NO-QUORUM** (W21-Λ-no-quorum falsifier; three
  :class:`SingletonAsymmetricOracle` instances each pointing at a
  different admitted tag). Each tag receives exactly one vote; no
  tag reaches ``quorum_min = 2``; W21 abstains via
  :data:`W21_BRANCH_NO_QUORUM`; ties FIFO at 0.000. Names the
  structural limit when registered oracles disagree completely on
  what the answer is.
* **R-68-MULTI-ALL-COMPROMISED** (W21-Λ-all-compromised falsifier;
  three :class:`CompromisedServiceGraphOracle` instances all
  emitting decoy-asymmetric). Decoy receives 3 votes; quorum
  forms on decoy; W21 projects to {decoy} and FAILS at 0.000.
  Names the structural limit when *all* N registered oracles are
  jointly compromised — the W21 escape is *partial* by design,
  bounded above by the integrity of the registered oracle set.
* **R-68-MULTI-PARTIAL** (W21-Λ-partial falsifier under default
  ``quorum_min = 2``; conditional success under ``quorum_min = 1``;
  two partial-honest :class:`OnCallNotesOracle` (one emits the
  first gold tag, the other emits the second gold tag) plus one
  :class:`AbstainingOracle`). Each gold tag gets one vote; no tag
  reaches ``quorum_min = 2``; W21 abstains. With override
  ``quorum_min = 1``, W21 projects to the gold pair (each tag has
  a single supporting oracle) — but the bench *pre-commits* the
  default ``quorum_min = 2`` so this is a falsifier on the
  default config. The :data:`W21_BRANCH_NO_QUORUM` audit fires.

Theorem family W21 (minted by this milestone)
==============================================

* **W21-1** (proved-conditional + proved-empirical, n=8 saturated × 5
  seeds × 2 cells). Pairing the W19
  :class:`BundleContradictionDisambiguator` with the W21
  :class:`TrustWeightedMultiOracleDisambiguator` over a registered
  oracle set with at least ``quorum_min`` honest oracles strictly
  improves ``accuracy_full`` over the strongest non-W21 capsule
  baseline (incl. W20 with the first-registered compromised oracle)
  by ``+1.000`` on R-68-MULTI-MAJORITY-LOOSE AND on
  R-68-MULTI-MAJORITY-TIGHT (``T_decoder = 24``), stable across 5/5
  alternate ``bank_seed`` values. The first capsule-native multi-
  agent-coordination method that crosses the W20-Λ-compromised wall
  on a regime where the wall actually applies.
* **W21-2** (proved by inspection + mechanically-checked).
  Determinism + closed-form correctness. The W21 scorer is byte-
  stable: ``decode_rounds`` returns byte-for-byte identical answers
  given byte-identical inputs and a deterministic registered oracle
  set; the per-tag votes depend only on each oracle's reply tokens
  and the admitted tag set; the projection rule is positive-set
  with quorum + trust-sum thresholds.
* **W21-3-A** (proved-empirical full programme regression). On
  R-54..R-67 default banks (the W19 non-trigger paths), the W21
  method ties the W19 method byte-for-byte on the answer field
  via :data:`W21_BRANCH_NO_TRIGGER`. With ``enabled = False`` OR
  no oracles registered, W21 reduces to W19 byte-for-byte.
* **W21-3-B** (proved-empirical, R-67-OUTSIDE-RESOLVES anchor).
  On R-67-OUTSIDE-RESOLVES with a single registered honest
  oracle AND ``quorum_min = 1``, W21 ties W20 byte-for-byte on
  the answer field. Establishes that the W21 layer is a strict
  generalisation of W20 — the special case ``quorum_min = 1, |reg|
  = 1`` recovers W20's projection rule exactly.
* **W21-Λ-no-quorum** (proved-empirical, n=8 saturated). On
  R-68-MULTI-NO-QUORUM, the registered singleton oracles each emit
  a different admitted tag; no tag reaches ``quorum_min = 2``; W21
  abstains via :data:`W21_BRANCH_NO_QUORUM`; ties FIFO at 0.000.
  Names the structural limit when registered oracles disagree
  completely.
* **W21-Λ-all-compromised** (proved-empirical, n=8 saturated). On
  R-68-MULTI-ALL-COMPROMISED, every registered oracle returns a
  decoy-asymmetric reply; quorum forms on decoy; W21 projects to
  decoy and FAILS at 0.000. Names the structural limit when ALL
  registered oracles are jointly compromised — the fix is NOT a
  richer aggregator; it is *oracle integrity* at the registered-
  set level. The conditional W21-C-CALIBRATED-TRUST (low trust
  priors on uncalibrated oracles) is conjectural.
* **W21-Λ-partial** (proved-empirical, n=8 saturated, default
  ``quorum_min = 2``). On R-68-MULTI-PARTIAL, two partial-honest
  oracles each emit a different element of the gold pair; no tag
  reaches ``quorum_min = 2``; W21 abstains via
  :data:`W21_BRANCH_NO_QUORUM`; ties FIFO at 0.000. Names the
  trade-off between quorum strictness and partial-evidence
  recovery — the conditional W21-C-PARTIAL-RECOVERY (with
  ``quorum_min = 1``, W21 recovers the gold pair on the same
  regime) is conjectural-empirical.
* **W21-Λ-real** (proved-conditional + empirical-research). Real-LLM
  transfer of the W21 escape via :class:`LLMAdjudicatorOracle` as
  one of the registered oracles is conditional on the LLM emitting
  a reply whose token set finds asymmetric service mentions
  through the same closure W19 / W18 / W13 / W12 share. The
  natural extension W21-C-LIVE-WITH-REGISTRY pairs an LLM
  adjudicator with a deterministic ServiceGraphOracle as its
  trusted "registry" — the deterministic oracle anchors the
  closure and the LLM adjudicator votes on top.
* **W21-C-CALIBRATED-TRUST** (conjectural). With trust priors
  calibrated against held-out historical agreement, a low prior on
  an uncalibrated oracle excludes its vote from quorum aggregation
  via the ``min_trust_sum`` floor — escapes W21-Λ-all-compromised
  on regimes where ≥ ``quorum_min`` calibrated oracles remain
  honest.
* **W21-C-LIVE-WITH-REGISTRY** (conjectural). An LLM adjudicator
  paired with a deterministic ServiceGraphOracle as a registered
  trusted "registry" extends W21 to the W20-Λ-real wall —
  partially discharges W20-C-LIVE-WITH-REGISTRY.

Phase-68 also closes a *partial* discharge of W20-Λ-compromised on
the *bundle-resolvable-by-quorum* direction: the wall remains a real
structural limit *for any single-oracle method*, but the
multi-oracle axis crosses it when ≥ ``quorum_min`` honest oracles
are registered. The W20-Λ-compromised wall is therefore named
*partially discharged* by W21-1 (positive direction) AND remains
real on the *no-quorum* and *all-compromised* directions.

CLI
---

::

    # R-68-MULTI-MAJORITY (W21-1 anchor, synthetic):
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_majority --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-68-MULTI-MAJORITY-TIGHT (W21-1 + W15 composition):
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_majority --decoder-budget 24 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-68-MULTI-NO-QUORUM (W21-Λ-no-quorum falsifier):
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_no_quorum --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-68-MULTI-ALL-COMPROMISED (W21-Λ-all-compromised falsifier):
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_all_compromised --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-68-MULTI-PARTIAL (W21-Λ-partial falsifier):
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_partial --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

    # Seed-stability sweep on the headline regime:
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_majority --seed-sweep \\
        --K-auditor 12 --n-eval 8 --out -

    # Live LLM adjudicator probe (Mac-1 Ollama; mixed registry +
    # LLM oracles for the W21-C-LIVE-WITH-REGISTRY conjecture):
    python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \\
        --bank multi_majority --live-mixed-registry \\
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
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
    AbstainingOracle,
    AdmissionPolicy, AttentionAwareBundleDecoder,
    BundleAwareTeamDecoder,
    BundleContradictionDisambiguator,
    CapsuleContextPacker,
    ChangeHistoryOracle,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CompromisedServiceGraphOracle,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    DisagreeingHonestOracle,
    FifoAdmissionPolicy, FifoContextPacker,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    OnCallNotesOracle,
    OracleRegistration,
    OutsideWitnessAcquisitionDisambiguator,
    RelationalCompatibilityDisambiguator,
    RobustMultiRoundBundleDecoder, RoleBudget,
    ServiceGraphOracle,
    SingletonAsymmetricOracle,
    TeamCoordinator, audit_team_lifecycle,
    TrustWeightedMultiOracleDisambiguator,
    collect_admitted_handoffs, _DecodedHandoff,
    normalize_payload,
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


# =============================================================================
# Phase-68 banks — same R-66-OUTSIDE-REQUIRED bundle shape; the
# *registered oracle set* is the bank-specific dimension.
# =============================================================================

_VALID_BANKS = (
    "single_clean",
    "multi_majority",
    "multi_no_quorum",
    "multi_all_compromised",
    "multi_partial",
)


def _bank_to_oracle_registrations(
        bank: str,
        *, llm_adjudicator: Any | None = None,
        live_mixed_registry: bool = False,
        ) -> tuple[OracleRegistration, ...]:
    """Map a Phase-68 bank label to the registered oracle set.

    The bank chooses the oracle "ecology"; the W21 method consults
    *every* registered oracle once per cell. The W20 baseline (single-
    oracle interface) always picks the *first* registered oracle —
    the bank's ordering is intentional and pre-committed.
    """
    if bank == "single_clean":
        return (
            OracleRegistration(
                oracle=ServiceGraphOracle(),
                trust_prior=1.0,
                role_label="service_registry"),
        )
    if bank == "multi_majority":
        # Order matters: the W20 baseline picks the FIRST registered
        # oracle. We register the compromised one first, so W20 fails
        # by construction. W21 consults all three and out-votes the
        # compromised one.
        regs: list[OracleRegistration] = [
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="compromised_registry"),
                trust_prior=0.8,
                role_label="compromised_registry"),
            OracleRegistration(
                oracle=ServiceGraphOracle(
                    oracle_id="service_graph"),
                trust_prior=1.0,
                role_label="service_graph"),
            OracleRegistration(
                oracle=ChangeHistoryOracle(
                    oracle_id="change_history"),
                trust_prior=1.0,
                role_label="change_history"),
        ]
        if live_mixed_registry and llm_adjudicator is not None:
            # W21-C-LIVE-WITH-REGISTRY conjecture probe:
            # add the live LLM adjudicator as a fourth registered
            # oracle (alongside the deterministic registry +
            # change-log "trusted registry" anchors).
            regs.append(OracleRegistration(
                oracle=llm_adjudicator,
                trust_prior=0.7,
                role_label="llm_adjudicator"))
        return tuple(regs)
    if bank == "multi_no_quorum":
        # Three singleton oracles each pointing at a different
        # admitted tag — no tag receives ≥ quorum_min = 2 votes.
        return (
            OracleRegistration(
                oracle=SingletonAsymmetricOracle(
                    target="first", oracle_id="singleton_first"),
                trust_prior=1.0,
                role_label="singleton_first"),
            OracleRegistration(
                oracle=SingletonAsymmetricOracle(
                    target="middle", oracle_id="singleton_middle"),
                trust_prior=1.0,
                role_label="singleton_middle"),
            OracleRegistration(
                oracle=SingletonAsymmetricOracle(
                    target="last", oracle_id="singleton_last"),
                trust_prior=1.0,
                role_label="singleton_last"),
        )
    if bank == "multi_all_compromised":
        # Three differently-labelled compromised oracles all emitting
        # decoy-asymmetric — quorum forms on decoy.
        return (
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="compromised_registry"),
                trust_prior=0.8,
                role_label="compromised_registry"),
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="compromised_change_log"),
                trust_prior=0.7,
                role_label="compromised_change_log"),
            OracleRegistration(
                oracle=CompromisedServiceGraphOracle(
                    oracle_id="compromised_oncall"),
                trust_prior=0.6,
                role_label="compromised_oncall"),
        )
    if bank == "multi_partial":
        # Two partial-honest oracles each emitting a different element
        # of the gold pair (no overlap) + one abstainer. Each gold
        # tag receives exactly one vote; no tag reaches quorum_min = 2.
        return (
            OracleRegistration(
                oracle=OnCallNotesOracle(
                    emit_partial_only=True, partial_index=0,
                    oracle_id="oncall_partial_a"),
                trust_prior=1.0,
                role_label="oncall_partial_a"),
            OracleRegistration(
                oracle=OnCallNotesOracle(
                    emit_partial_only=True, partial_index=1,
                    oracle_id="oncall_partial_b"),
                trust_prior=1.0,
                role_label="oncall_partial_b"),
            OracleRegistration(
                oracle=AbstainingOracle(oracle_id="abstainer"),
                trust_prior=1.0,
                role_label="abstainer"),
        )
    raise ValueError(f"unknown bank {bank!r}")


def _bank_to_w20_oracle(bank: str,
                          regs: tuple[OracleRegistration, ...],
                          ) -> Any:
    """The W20 baseline picks the FIRST registered oracle by
    convention. This is the intentional weakness of single-oracle
    interfaces — the runtime can't tell which oracle is trustworthy.
    """
    if not regs:
        return AbstainingOracle(oracle_id="no_registered")
    return regs[0].oracle


# Phase 68 bank reuses Phase 67's R-66-OUTSIDE-REQUIRED bundle shape
# from build_phase67_bank with bank="outside_resolves" — the bundle
# is bank-invariant; only the registered oracle set varies.
def build_phase68_bank(*, n_replicates: int = 2, seed: int = 11,
                          ) -> list[MultiRoundScenario]:
    """Build the Phase-68 bank — uses the R-66-OUTSIDE-REQUIRED bundle
    shape (same as Phase-67's ``outside_resolves`` family). The
    bundle is bank-invariant across all Phase-68 sub-banks; the
    registered oracle set is the bank-specific knob (registered at
    strategy-dispatch time, not in the scenario)."""
    return build_phase67_bank(
        bank="outside_resolves", n_replicates=n_replicates, seed=seed)


# =============================================================================
# Strategy / decoder dispatch
# =============================================================================

# Phase-68 strategy roster: same SDK strategies as Phase-67 plus the
# new W21 multi-oracle adjudicator.
_R68_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("capsule_fifo", "per_round"),
    ("capsule_priority", "per_round"),
    ("capsule_coverage", "per_round"),
    ("capsule_cohort_buffered", "per_round"),
    ("capsule_corroboration", "per_round"),
    ("capsule_multi_service", "per_round"),
    ("capsule_multi_round", "multi_round_bundle"),
    ("capsule_robust_multi_round", "robust_multi_round"),
    ("capsule_layered_multi_round", "layered_multi_round"),
    ("capsule_layered_fifo_packed", "fifo_packed_layered"),
    ("capsule_attention_aware", "attention_aware"),
    ("capsule_relational_compat", "relational_compat"),
    ("capsule_bundle_contradiction", "bundle_contradiction"),
    # SDK v3.21 — W20 anchor: single-oracle outside-witness acquisition.
    ("capsule_outside_witness", "outside_witness"),
    # SDK v3.22 — W21 anchor: trust-weighted multi-oracle adjudicator.
    ("capsule_multi_oracle", "multi_oracle"),
)


def _make_factory(name: str, priorities, budgets):
    def fac(round_idx: int = 1, cands=None,
             ) -> dict[str, AdmissionPolicy]:
        if name == "capsule_fifo":
            return {r: FifoAdmissionPolicy() for r in budgets}
        if name == "capsule_priority":
            return {r: ClaimPriorityAdmissionPolicy(
                priorities=priorities, threshold=0.65) for r in budgets}
        if name == "capsule_coverage":
            return {r: CoverageGuidedAdmissionPolicy() for r in budgets}
        cands = cands or []
        cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
        if name == "capsule_cohort_buffered":
            policy = (CohortCoherenceAdmissionPolicy
                      .from_candidate_payloads([c[3] for c in cands_aud]))
            return {r: policy for r in budgets}
        if name == "capsule_corroboration":
            policy = (CrossRoleCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud]))
            return {r: policy for r in budgets}
        if name == "capsule_multi_service":
            policy = (MultiServiceCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud],
                          top_k=4, min_corroborated_roles=2))
            return {r: policy for r in budgets}
        if name in ("capsule_multi_round",
                     "capsule_robust_multi_round",
                     "capsule_layered_multi_round",
                     "capsule_layered_fifo_packed",
                     "capsule_attention_aware",
                     "capsule_relational_compat",
                     "capsule_bundle_contradiction",
                     "capsule_outside_witness",
                     "capsule_multi_oracle"):
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
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


def _decode_with_packer(union, packer, T_decoder, round_index_hint):
    first = MultiRoundBundleDecoder().decode_rounds([union])
    elected = str(first.get("root_cause", "unknown"))
    pack = packer.pack(union, elected_root_cause=elected,
                          T_decoder=T_decoder,
                          round_index_hint=round_index_hint)
    kept = [k.handoff for k in pack.kept]
    ans = LayeredRobustMultiRoundBundleDecoder().decode_rounds([kept])
    stats = pack.as_dict()
    stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
    stats["n_handoffs_decoder_input"] = int(pack.n_kept)
    stats["n_handoffs_admitted"] = int(pack.n_input)
    return ans, stats


def _decode_with_w18(union, T_decoder, round_index_hint):
    inner = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner)
    if round_index_hint is None:
        per_round = [list(union)]
    else:
        max_round = max(round_index_hint) if round_index_hint else 1
        per_round = [[] for _ in range(max(1, int(max_round)))]
        for h, ridx in zip(union, round_index_hint):
            slot = max(0, int(ridx) - 1)
            while slot >= len(per_round):
                per_round.append([])
            per_round[slot].append(h)
    ans = w18.decode_rounds(per_round)
    pack = inner.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
        stats["n_handoffs_decoder_input"] = int(pack.n_kept)
        stats["n_handoffs_admitted"] = int(pack.n_input)
    return ans, stats


def _decode_with_w19(union, T_decoder, round_index_hint):
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    if round_index_hint is None:
        per_round = [list(union)]
    else:
        max_round = max(round_index_hint) if round_index_hint else 1
        per_round = [[] for _ in range(max(1, int(max_round)))]
        for h, ridx in zip(union, round_index_hint):
            slot = max(0, int(ridx) - 1)
            while slot >= len(per_round):
                per_round.append([])
            per_round[slot].append(h)
    ans = w19.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
        stats["n_handoffs_decoder_input"] = int(pack.n_kept)
        stats["n_handoffs_admitted"] = int(pack.n_input)
    return ans, stats


def _decode_with_w20(union, T_decoder, round_index_hint, oracle):
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w20 = OutsideWitnessAcquisitionDisambiguator(inner=w19, oracle=oracle)
    if round_index_hint is None:
        per_round = [list(union)]
    else:
        max_round = max(round_index_hint) if round_index_hint else 1
        per_round = [[] for _ in range(max(1, int(max_round)))]
        for h, ridx in zip(union, round_index_hint):
            slot = max(0, int(ridx) - 1)
            while slot >= len(per_round):
                per_round.append([])
            per_round[slot].append(h)
    ans = w20.decode_rounds(per_round)
    pack = inner_w15.last_pack_result
    stats: dict[str, Any] = {}
    if pack is not None:
        stats = pack.as_dict()
        stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
        stats["n_handoffs_decoder_input"] = int(pack.n_kept)
        stats["n_handoffs_admitted"] = int(pack.n_input)
    if "outside" in ans:
        stats["outside"] = ans["outside"]
    return ans, stats


def _decode_with_w21(union, T_decoder, round_index_hint,
                       oracle_registrations, *,
                       quorum_min: int = 2,
                       min_trust_sum: float = 0.0):
    inner_w15 = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner_w15)
    w19 = BundleContradictionDisambiguator(inner=w18)
    w21 = TrustWeightedMultiOracleDisambiguator(
        inner=w19, oracle_registrations=oracle_registrations,
        quorum_min=quorum_min, min_trust_sum=min_trust_sum)
    if round_index_hint is None:
        per_round = [list(union)]
    else:
        max_round = max(round_index_hint) if round_index_hint else 1
        per_round = [[] for _ in range(max(1, int(max_round)))]
        for h, ridx in zip(union, round_index_hint):
            slot = max(0, int(ridx) - 1)
            while slot >= len(per_round):
                per_round.append([])
            per_round[slot].append(h)
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


def _run_capsule_strategy(
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands, round2_cands,
        T_decoder: int | None = None,
        oracle: Any | None = None,
        oracle_registrations: tuple[OracleRegistration, ...] = (),
        quorum_min: int = 2,
        min_trust_sum: float = 0.0,
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase68_multi_oracle_adjudication",
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

    if decoder_mode == "attention_aware":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_packer(
            union, CapsuleContextPacker(), T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "fifo_packed_layered":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_packer(
            union, FifoContextPacker(), T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "relational_compat":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_w18(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "bundle_contradiction":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_w19(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "outside_witness":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_w20(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint, oracle=oracle)
    elif decoder_mode == "multi_oracle":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_w21(
            union, T_decoder=T_decoder,
            round_index_hint=round_hint,
            oracle_registrations=oracle_registrations,
            quorum_min=quorum_min,
            min_trust_sum=min_trust_sum)
    elif decoder_mode == "layered_multi_round":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        decoder = LayeredRobustMultiRoundBundleDecoder()
        answer = decoder.decode_rounds([union])
    elif decoder_mode == "robust_multi_round":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        answer = RobustMultiRoundBundleDecoder().decode_rounds([union])
    elif decoder_mode == "multi_round_bundle":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        answer = MultiRoundBundleDecoder().decode_rounds([union])
    elif decoder_mode == "single_round_bundle":
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])
        decoder = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=2)
        answer = decoder.decode(r2_handoffs)
    else:
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])

        @dataclasses.dataclass(frozen=True)
        class _Shim:
            source_role: str
            claim_kind: str
            payload: str
            n_tokens: int = 1

        shimmed = [_Shim(h.source_role, h.claim_kind, h.payload)
                    for h in r2_handoffs]
        answer = _phase31_decoder_from_handoffs(shimmed)

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


def _run_substrate_strategy(
        sc: MultiRoundScenario,
        round1_cands, round2_cands, inbox_capacity) -> StrategyResult:
    incident_sc = _as_incident_scenario(sc)
    from vision_mvp.core.role_handoff import HandoffRouter, RoleInbox
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))
    for round_idx, cands in ((1, round1_cands), (2, round2_cands)):
        for (src, _to, kind, payload, evids) in cands:
            router.emit(
                source_role=src, source_agent_id=ALL_ROLES.index(src),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=round_idx)
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    held = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    answer = _phase31_decoder_from_handoffs(held)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))
    admitted_kinds = {(h.source_role, h.claim_kind) for h in held}
    required = {(role, kind)
                 for (role, kind, _p, _evs) in incident_sc.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    return StrategyResult(
        strategy="substrate", scenario_id=sc.scenario_id,
        answer=answer, grading=grading, failure_kind=failure_kind,
        n_admitted_auditor=len(held),
        n_dropped_auditor_budget=auditor_inbox.n_overflow if auditor_inbox else 0,
        n_dropped_auditor_capacity=auditor_inbox.n_dedup if auditor_inbox else 0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=0, n_role_view=0, n_team_decision=0,
        audit_ok=False,
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


# =============================================================================
# Phase 68 driver
# =============================================================================


def run_phase68(*,
                  bank: str = "multi_majority",
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
                  live_mixed_registry: bool = False,
                  oracle_registrations_override:
                      tuple[OracleRegistration, ...] | None = None,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 68 over one of {single_clean, multi_majority,
    multi_no_quorum, multi_all_compromised, multi_partial}.

    The W21 method consults the registered oracle set; the W20
    baseline picks the FIRST registered oracle (the bank's
    pre-committed ordering)."""
    if bank not in _VALID_BANKS:
        raise ValueError(
            f"unknown bank {bank!r}; valid: {_VALID_BANKS}")
    bank_obj = build_phase68_bank(
        n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank_obj = bank_obj[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    if oracle_registrations_override is not None:
        regs = oracle_registrations_override
    else:
        regs = _bank_to_oracle_registrations(
            bank, llm_adjudicator=llm_adjudicator,
            live_mixed_registry=live_mixed_registry)
    w20_oracle = _bank_to_w20_oracle(bank, regs)

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R68_STRATEGIES
    }
    w20_branch_counts: dict[str, int] = {}
    w21_branch_counts: dict[str, int] = {}

    for sc in bank_obj:
        round1_cands = _build_round_candidates_p66(sc.round1_emissions)
        round2_cands = _build_round_candidates_p66(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p67(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R68_STRATEGIES:
            fac = _make_factory(sname, priorities, budgets)
            r, ps = _run_capsule_strategy(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                T_decoder=T_decoder,
                oracle=(w20_oracle if dmode == "outside_witness" else None),
                oracle_registrations=(regs if dmode == "multi_oracle" else ()),
                quorum_min=quorum_min,
                min_trust_sum=min_trust_sum)
            results.append(r)
            if ps:
                pack_stats_per_strategy[sname].append({
                    "scenario_id": sc.scenario_id,
                    **ps,
                })
                if sname == "capsule_outside_witness":
                    out = ps.get("outside") or {}
                    branch = str(out.get("decoder_branch", "unknown"))
                    w20_branch_counts[branch] = (
                        w20_branch_counts.get(branch, 0) + 1)
                if sname == "capsule_multi_oracle":
                    mo = ps.get("multi_oracle") or {}
                    branch = str(mo.get("decoder_branch", "unknown"))
                    w21_branch_counts[branch] = (
                        w21_branch_counts.get(branch, 0) + 1)

    strategy_names = ("substrate",) + tuple(s[0] for s in _R68_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w21_minus_w20": _gap(
            "capsule_multi_oracle", "capsule_outside_witness"),
        "w21_minus_w19": _gap(
            "capsule_multi_oracle", "capsule_bundle_contradiction"),
        "w21_minus_w18": _gap(
            "capsule_multi_oracle", "capsule_relational_compat"),
        "w21_minus_attention_aware": _gap(
            "capsule_multi_oracle", "capsule_attention_aware"),
        "w21_minus_fifo": _gap(
            "capsule_multi_oracle", "capsule_fifo"),
        "w21_minus_substrate": _gap(
            "capsule_multi_oracle", "substrate"),
        "max_non_w21_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_multi_oracle"),
    }

    audit_ok_grid: dict[str, bool] = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
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
        "n_oracle_registrations": len(regs),
        "oracle_role_labels": [r.role_label for r in regs],
        "oracle_ids": [getattr(r.oracle, "oracle_id", "?") for r in regs],
        "trust_priors": [r.trust_prior for r in regs],
        "quorum_min": int(quorum_min),
        "min_trust_sum": float(min_trust_sum),
        "w20_chosen_oracle_id": getattr(w20_oracle, "oracle_id", "?"),
    }

    def _agg_packstats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        s_in = sum(r.get("tokens_input", 0) for r in rows)
        s_kept = sum(r.get("tokens_kept", 0) for r in rows)
        s_drop = sum(r.get("n_dropped_budget", 0) for r in rows)
        s_h_in = sum(r.get("n_handoffs_admitted", 0) for r in rows)
        s_h_kept = sum(r.get("n_handoffs_decoder_input", 0) for r in rows)
        n_w20_triggered = sum(
            1 for r in rows
            if r.get("outside", {}).get("triggered"))
        s_outside_tokens = sum(
            int(r.get("outside", {}).get("n_outside_tokens", 0))
            for r in rows)
        n_w21_triggered = sum(
            1 for r in rows
            if r.get("multi_oracle", {}).get("triggered"))
        s_w21_outside_tokens_total = sum(
            int(r.get("multi_oracle", {}).get(
                "n_outside_tokens_total", 0))
            for r in rows)
        s_w21_outside_queries = sum(
            int(r.get("multi_oracle", {}).get("n_outside_queries", 0))
            for r in rows)
        return {
            "n_cells": n,
            "tokens_input_sum": int(s_in),
            "tokens_kept_sum": int(s_kept),
            "n_dropped_budget_sum": int(s_drop),
            "tokens_kept_over_input": (
                round(s_kept / s_in, 4) if s_in > 0 else 0.0),
            "handoffs_admitted_sum": int(s_h_in),
            "handoffs_decoder_input_sum": int(s_h_kept),
            "fraction_handoffs_kept": (
                round(s_h_kept / s_h_in, 4) if s_h_in > 0 else 0.0),
            "n_w20_triggered_cells": int(n_w20_triggered),
            "outside_tokens_sum": int(s_outside_tokens),
            "outside_tokens_per_cell_avg": (
                round(s_outside_tokens / n, 4) if n > 0 else 0.0),
            "n_w21_triggered_cells": int(n_w21_triggered),
            "n_w21_outside_queries_sum": int(s_w21_outside_queries),
            "w21_outside_queries_per_cell_avg": (
                round(s_w21_outside_queries / n, 4) if n > 0 else 0.0),
            "w21_outside_tokens_total_sum":
                int(s_w21_outside_tokens_total),
            "w21_outside_tokens_total_per_cell_avg": (
                round(s_w21_outside_tokens_total / n, 4)
                if n > 0 else 0.0),
        }

    pack_stats_summary = {
        s: _agg_packstats(pack_stats_per_strategy.get(s, []))
        for s in ("capsule_layered_fifo_packed",
                   "capsule_attention_aware",
                   "capsule_relational_compat",
                   "capsule_bundle_contradiction",
                   "capsule_outside_witness",
                   "capsule_multi_oracle")
    }

    if verbose:
        print(f"[phase68] bank={bank} n_oracles={len(regs)} "
              f"quorum_min={quorum_min} T_decoder={T_decoder} "
              f"n_eval={len(bank_obj)} K_auditor={K_auditor} "
              f"bank_seed={bank_seed}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase68]   {s:34s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase68] {k}: {v:+.3f}", file=sys.stderr, flush=True)
        if w20_branch_counts:
            print(f"[phase68] w20 branch counts: {w20_branch_counts}",
                  file=sys.stderr, flush=True)
        if w21_branch_counts:
            print(f"[phase68] w21 branch counts: {w21_branch_counts}",
                  file=sys.stderr, flush=True)

    return {
        "schema": "phase68.multi_oracle.v1",
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
        "w20_branch_counts": w20_branch_counts,
        "w21_branch_counts": w21_branch_counts,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }


def run_phase68_seed_stability_sweep(*,
                                          bank: str = "multi_majority",
                                          T_decoder: int | None = None,
                                          n_eval: int = 8,
                                          K_auditor: int = 12,
                                          quorum_min: int = 2,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase68.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "quorum_min": int(quorum_min),
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase68(bank=bank, T_decoder=T_decoder,
                            n_eval=n_eval, K_auditor=K_auditor,
                            bank_seed=seed, quorum_min=quorum_min)
        out["per_seed"][str(seed)] = {
            "headline_gap": rep["headline_gap"],
            "pooled": {
                k: v["accuracy_full"]
                for k, v in rep["pooled"].items()
            },
            "scenarios_with_symmetric_corroboration":
                rep["bench_summary"]["scenarios_with_symmetric_corroboration"],
            "scenarios_with_expected_shape":
                rep["bench_summary"]["scenarios_with_expected_shape"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "w20_branch_counts": rep["w20_branch_counts"],
            "w21_branch_counts": rep["w21_branch_counts"],
        }
    gaps_w21_w20 = [
        out["per_seed"][str(s)]["headline_gap"]["w21_minus_w20"]
        for s in seeds
    ]
    gaps_w21_aa = [
        out["per_seed"][str(s)]["headline_gap"]["w21_minus_attention_aware"]
        for s in seeds
    ]
    out["min_w21_minus_w20"] = min(gaps_w21_w20) if gaps_w21_w20 else 0.0
    out["max_w21_minus_w20"] = max(gaps_w21_w20) if gaps_w21_w20 else 0.0
    out["mean_w21_minus_w20"] = (
        round(sum(gaps_w21_w20) / len(gaps_w21_w20), 4)
        if gaps_w21_w20 else 0.0)
    out["min_w21_minus_attention_aware"] = (
        min(gaps_w21_aa) if gaps_w21_aa else 0.0)
    out["max_w21_minus_attention_aware"] = (
        max(gaps_w21_aa) if gaps_w21_aa else 0.0)
    out["mean_w21_minus_attention_aware"] = (
        round(sum(gaps_w21_aa) / len(gaps_w21_aa), 4)
        if gaps_w21_aa else 0.0)
    return out


def run_cross_regime_synthetic(*,
                                  n_eval: int = 8,
                                  bank_seed: int = 11,
                                  K_auditor: int = 12,
                                  T_auditor: int = 256,
                                  T_decoder_tight: int = 24,
                                  quorum_min: int = 2,
                                  ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase68.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
            "quorum_min": int(quorum_min),
        },
    }
    out["r68_single_clean"] = run_phase68(
        bank="single_clean", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=quorum_min)
    out["r68_multi_majority_loose"] = run_phase68(
        bank="multi_majority", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=quorum_min)
    out["r68_multi_majority_tight"] = run_phase68(
        bank="multi_majority", T_decoder=T_decoder_tight,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=quorum_min)
    out["r68_multi_no_quorum"] = run_phase68(
        bank="multi_no_quorum", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=quorum_min)
    out["r68_multi_all_compromised"] = run_phase68(
        bank="multi_all_compromised", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=quorum_min)
    out["r68_multi_partial"] = run_phase68(
        bank="multi_partial", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=quorum_min)
    # Conditional check: R-68-MULTI-PARTIAL with quorum_min=1 recovers.
    out["r68_multi_partial_q1"] = run_phase68(
        bank="multi_partial", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        quorum_min=1)

    def _acc(cell: str, strategy: str) -> float:
        return float(out[cell]["pooled"][strategy]["accuracy_full"])

    out["headline_summary"] = {
        "r68_single_clean_w20": _acc("r68_single_clean",
                                       "capsule_outside_witness"),
        "r68_single_clean_w21": _acc("r68_single_clean",
                                       "capsule_multi_oracle"),
        "r68_multi_majority_loose_w20": _acc(
            "r68_multi_majority_loose", "capsule_outside_witness"),
        "r68_multi_majority_loose_w21": _acc(
            "r68_multi_majority_loose", "capsule_multi_oracle"),
        "r68_multi_majority_tight_w20": _acc(
            "r68_multi_majority_tight", "capsule_outside_witness"),
        "r68_multi_majority_tight_w21": _acc(
            "r68_multi_majority_tight", "capsule_multi_oracle"),
        "r68_multi_no_quorum_w21": _acc("r68_multi_no_quorum",
                                          "capsule_multi_oracle"),
        "r68_multi_all_compromised_w21": _acc(
            "r68_multi_all_compromised", "capsule_multi_oracle"),
        "r68_multi_partial_w21": _acc("r68_multi_partial",
                                        "capsule_multi_oracle"),
        "r68_multi_partial_q1_w21": _acc("r68_multi_partial_q1",
                                            "capsule_multi_oracle"),
        "w21_minus_w20_multi_majority_loose":
            round(_acc("r68_multi_majority_loose",
                        "capsule_multi_oracle")
                   - _acc("r68_multi_majority_loose",
                           "capsule_outside_witness"), 4),
        "w21_minus_w20_multi_majority_tight":
            round(_acc("r68_multi_majority_tight",
                        "capsule_multi_oracle")
                   - _acc("r68_multi_majority_tight",
                           "capsule_outside_witness"), 4),
    }
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 68 — trust-weighted multi-oracle "
                     "adjudication under partial oracle compromise "
                     "(SDK v3.22 / W21 family).")
    p.add_argument("--bank", type=str, default="multi_majority",
                    choices=_VALID_BANKS)
    p.add_argument("--K-auditor", type=int, default=12)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--decoder-budget", type=int, default=-1,
                    help="Strict T_decoder budget. -1 = None.")
    p.add_argument("--quorum-min", type=int, default=2)
    p.add_argument("--min-trust-sum", type=float, default=0.0)
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--cross-regime-synthetic", action="store_true")
    p.add_argument("--live-mixed-registry", action="store_true",
                    help="Add a live LLMAdjudicatorOracle as a "
                         "fourth registered oracle alongside the "
                         "deterministic registry + change-log.")
    p.add_argument("--adjudicator-model", type=str,
                    default="mixtral:8x7b",
                    help="Ollama model tag for the live adjudicator.")
    p.add_argument("--adjudicator-endpoint", type=str,
                    default="http://127.0.0.1:11434")
    p.add_argument("--adjudicator-timeout", type=float, default=600.0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def _maybe_build_live_adjudicator(args) -> Any | None:
    if not args.live_mixed_registry:
        return None
    try:
        from vision_mvp.wevra.team_coord import LLMAdjudicatorOracle
        from vision_mvp.wevra.llm_backend import OllamaBackend
        backend = OllamaBackend(
            model=args.adjudicator_model,
            base_url=args.adjudicator_endpoint,
            timeout=args.adjudicator_timeout)
        return LLMAdjudicatorOracle(
            backend=backend,
            oracle_id=f"ollama_{args.adjudicator_model}")
    except Exception as exc:  # noqa: BLE001
        print(f"[phase68] live-adjudicator unavailable: {exc!r}",
              file=sys.stderr, flush=True)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    T_dec = None if args.decoder_budget < 0 else int(args.decoder_budget)
    live_adjudicator = _maybe_build_live_adjudicator(args)
    if args.cross_regime_synthetic:
        report = run_cross_regime_synthetic(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            quorum_min=args.quorum_min)
    elif args.seed_sweep:
        report = run_phase68_seed_stability_sweep(
            bank=args.bank, T_decoder=T_dec,
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            quorum_min=args.quorum_min)
    else:
        report = run_phase68(
            bank=args.bank, n_eval=args.n_eval,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates, T_decoder=T_dec,
            quorum_min=args.quorum_min,
            min_trust_sum=args.min_trust_sum,
            llm_adjudicator=live_adjudicator,
            live_mixed_registry=args.live_mixed_registry,
            verbose=not args.quiet)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out == "-":
        print(text)
    elif args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        if not args.quiet:
            print(f"[phase68] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
