"""Phase 65 — relational-compatibility disambiguation under symmetric
corroboration (SDK v3.19, W18 family anchor).

The follow-up to SDK v3.18 (W17) on the named research frontier the W17
milestone explicitly left conjectural: **the symmetric-corroboration
wall** (W17-Λ-symmetric).

The W17-Λ-symmetric anchor (Phase-64-SYM) proves that *every* closed-form
salience packer + admission policy in the SDK ties FIFO at 0.000 by
construction when the bipartite ``(role × tag, kind, magnitude)``
multiset is identical for gold and decoy. SDK v3.18 named the next
research move: a *richer disambiguator* that consumes information *beyond*
the bipartite multiset — specifically, the round-2 specific-tier
disambiguator's *payload text* itself.

W18's :class:`RelationalCompatibilityDisambiguator` is the smallest
move in that direction:

  1. Run the W15 attention-aware decoder over the admitted bundle to
     elect a tentative root_cause AND get the W15 pack stats.
  2. Identify the round-2 specific-tier disambiguator(s) in the bundle.
  3. Tokenise the disambiguator's payload (lower-cased, split on
     non-identifier chars, compound identifiers preserved).
  4. Score each admitted service tag (from the union of admitted
     handoffs) by mention-and-relational-compound match.
  5. Project the answer set: keep tags with positive score iff at
     least one but not all admitted tags have positive score; else
     abstain (fall through to W15 byte-for-byte).

Phase-65 is the named harder regime where the W18 method's required
ingredient — a *relational mention* in the round-2 disambiguator —
is consistently present across every scenario family (R-65-COMPAT)
AND where three named falsifier regimes (R-65-NO-COMPAT,
R-65-CONFOUND, R-65-DECEIVE) make the W18 conditionality sharp.

Recap of why the W17-Λ-symmetric wall stands without W18
========================================================

On R-64-SYM (Phase-64-SYM), the bench property is:

  * Both gold A/B AND the decoy are mentioned by ≥ 2 distinct routed
    producer roles in round-1 via generic-noise kinds (LATENCY_SPIKE
    / ERROR_RATE_SPIKE / FW_BLOCK_SURGE) with comparable magnitudes.
  * Round-2 carries one specific-tier disambiguator (DEADLOCK_SUSPECTED
    / POOL_EXHAUSTION / DISK_FILL_CRITICAL / SLOW_QUERY_OBSERVED) with
    NO ``service=`` token.

Under this regime:

  * The W11 contradiction-aware drop fires symmetrically on every
    service tag (gold AND decoy are noise-corroborated by ≥ 2 distinct
    roles), dropping every tag.
  * The W15 hypothesis-preserving pack keeps every tag's
    representatives — but the W11 drop downstream still fires
    symmetrically.
  * The W14H magnitude-hint is silent on the symmetric ambiguity.
  * Result: every capsule strategy ties FIFO at ``accuracy_full =
    0.000`` (W17-Λ-symmetric).

On R-64-SYM only the *deadlock* scenarios carry a relational mention
in round-2 (``relation=orders_payments_join``); the pool / disk /
slow_query disambiguators do not. So W18 partially recovers on R-64-SYM
(2/8 cells); R-65 makes the relational-mention convention *consistent*
across every scenario family so the W18-1 strict gain is uniform.

Phase-65 sub-banks
==================

Five pre-committed cells plus a cross-regime synthetic summary:

* **R-65-COMPAT-LOOSE** (synthetic identity producer, ``T_decoder=None``).
  All four scenario families carry a *gold-only* relational mention in
  round-2:
    - deadlock: ``relation={A}_{B}_join wait_chain=2``
    - pool: ``pool_chain={A}_{B} exhaustion=upstream_{A}_to_{B}``
    - disk: ``mount=/storage/{A}/{B}/var/log used=99%``
    - slow_query: ``query_path=svc_{A}_then_svc_{B} mean_ms=4100``
  The W18 anchor: W18 strictly beats every W11..W17 method by ``+1.000``;
  every other capsule strategy ties FIFO at 0.000 by W17-Λ-symmetric.
* **R-65-COMPAT-TIGHT** (synthetic identity producer, ``T_decoder=24``).
  Same shape under decoder-side budget pressure. The W18 disambiguator
  runs *after* the W15 pack; the W18 candidate set is the union of
  admitted service tags in the *kept* bundle. Token-budget honesty
  preserved byte-for-byte.
* **R-65-NO-COMPAT** (W18-Λ-no-compat falsifier; synthetic,
  ``T_decoder=None``). Round-2 disambiguator's payload contains only
  the kind token + numeric magnitudes — *no* service-tag mention.
  W18 abstains; ties FIFO at 0.000.
* **R-65-CONFOUND** (W18-Λ-confound falsifier; synthetic,
  ``T_decoder=None``). Round-2 disambiguator mentions BOTH gold AND
  decoy service tags. W18's compatibility score is non-zero on every
  admitted tag (or zero on every admitted tag); W18 abstains and
  falls through to the W15 answer (which itself ties FIFO by
  W17-Λ-symmetric).
* **R-65-DECEIVE** (W18-Λ-deceive falsifier; synthetic,
  ``T_decoder=None``). Round-2 disambiguator mentions DECOY service
  tags but NOT gold. W18 trusts its evidence and FAILS at 0.000 (the
  named structural limit when round-2 evidence is adversarially-
  relational; no closed-form bundle-relational scorer can escape this
  regime without an outside-information axis — W18-C-OUTSIDE,
  conjectural).

Theorem family W18 (minted by this milestone)
==============================================

* **W18-Λ-sym (extended)** (proved-empirical + structural sketch).
  W17-Λ-symmetric extends to R-65-COMPAT verbatim *for every method
  pre-W18*. Every closed-form salience scorer ties FIFO at 0.000 on
  R-65-COMPAT by construction — the bench property is a strict
  refinement of R-64-SYM.
* **W18-1** (proved-conditional + proved-empirical n=8 saturated × 5
  seeds). Pairing the W15 ``AttentionAwareBundleDecoder`` with the
  W18 :class:`RelationalCompatibilityDisambiguator` strictly improves
  ``accuracy_full`` over the strongest non-W18 capsule baseline by
  ``+1.000`` on R-65-COMPAT-LOOSE AND on R-65-COMPAT-TIGHT, stable
  across 5/5 alternate ``bank_seed`` values. The first capsule-native
  multi-agent-coordination method that crosses the symmetric-
  corroboration wall on a regime where the wall actually applies.
* **W18-2** (proved by inspection + mechanically-checked). The W18
  scorer is deterministic and byte-stable: ``decode_rounds`` returns
  byte-for-byte identical answers given byte-identical inputs; the
  per-tag scores depend only on the tokenised disambiguator payload
  and the admitted service-tag set.
* **W18-3** (proved-empirical full programme regression). On
  R-54..R-64 default banks the W18 method ties the W15 method
  byte-for-byte on the answer field — either via abstention (no
  relational mention in round-2) or via strict-asymmetric projection
  that lands on the same gold subset (e.g. R-58 deadlock).
* **W18-Λ-no-compat** (proved-empirical n=8 saturated). On
  R-65-NO-COMPAT, every cell's round-2 disambiguator carries no
  service-tag mention; W18 abstains and ties FIFO at 0.000. Names
  the structural limit when round-2 evidence carries no relational
  compatibility cue.
* **W18-Λ-confound** (proved-empirical n=8 saturated). On
  R-65-CONFOUND, every cell's round-2 disambiguator mentions both
  gold and decoy; W18's compatibility score is uniform and W18
  abstains; W15's symmetric output ties FIFO at 0.000.
* **W18-Λ-deceive** (proved-empirical n=8 saturated). On
  R-65-DECEIVE, every cell's round-2 disambiguator mentions decoy
  but not gold; W18 picks decoy and FAILS at 0.000. Names the
  structural limit when round-2 evidence is adversarially-relational.
* **W18-C-LEARNED** (conjectural). A small distilled compatibility
  scorer (e.g. character-bigram model + relational-compound
  embedding) outperforms the closed-form mention-matching rule on
  a held-out cross-bench when the LLM emits free-form relational
  mentions outside the synthetic bench's exact-match closure.
* **W18-C-OUTSIDE** (conjectural). A scorer with access to an
  outside-information axis (e.g. service-graph topology, prior
  reliability scores) can detect the W18-Λ-deceive regime by
  cross-referencing the round-2 mention against an outside source
  and rejecting the projection when the cross-reference fails.
* **W18-Λ-real** (proved-conditional + empirical-research,
  conjectural-empirical-on-Mac-1). Real-LLM transfer of the
  relational-mention convention: the LLM must emit the same
  relational-compound forms the synthetic bench uses; if the LLM
  emits a free-form relational mention, the W18 exact-match layer
  misses by construction.

CLI
---

::

    # R-65-COMPAT-LOOSE (W18-1 anchor, synthetic):
    python3 -m vision_mvp.experiments.phase65_relational_disambiguation \\
        --bank compat --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-65-COMPAT-TIGHT (synthetic with budget):
    python3 -m vision_mvp.experiments.phase65_relational_disambiguation \\
        --bank compat --decoder-budget 24 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-65-NO-COMPAT falsifier:
    python3 -m vision_mvp.experiments.phase65_relational_disambiguation \\
        --bank no_compat --decoder-budget -1 --K-auditor 12 --n-eval 8 --out -

    # R-65-CONFOUND falsifier:
    python3 -m vision_mvp.experiments.phase65_relational_disambiguation \\
        --bank confound --decoder-budget -1 --K-auditor 12 --n-eval 8 --out -

    # R-65-DECEIVE falsifier:
    python3 -m vision_mvp.experiments.phase65_relational_disambiguation \\
        --bank deceive --decoder-budget -1 --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase65_relational_disambiguation \\
        --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.team_coord import (
    AdmissionPolicy, AttentionAwareBundleDecoder,
    BundleAwareTeamDecoder,
    CapsuleContextPacker,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, FifoContextPacker,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    RelationalCompatibilityDisambiguator,
    RobustMultiRoundBundleDecoder, RoleBudget,
    TeamCoordinator, audit_team_lifecycle,
    collect_admitted_handoffs, _DecodedHandoff,
    normalize_payload,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    MultiRoundScenario, _as_incident_scenario,
    _build_round_candidates,
)


# =============================================================================
# Phase-65 scenario family
# =============================================================================
#
# Symmetric-corroboration shape (same as Phase-64-SYM) PLUS a relational-
# compatibility convention in the round-2 disambiguator's payload that
# differs across the four sub-banks (COMPAT / NO-COMPAT / CONFOUND /
# DECEIVE). The four scenario families (deadlock / pool / disk /
# slow_query) all carry the same relational-mention shape so the W18
# strict-gain claim is uniform across the bank.


_REMEDIATION = {
    "deadlock":             "enforce_lock_ordering_in_orders",
    "pool_exhaustion":      "raise_pool_cap_or_fix_upstream",
    "slow_query_cascade":   "index_or_split_slow_query",
    "disk_fill":            "rotate_logs_and_clear_backup",
}


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


def _round1_symmetric(A: str, B: str, decoy: str
                          ) -> dict[str, tuple[tuple[str, str], ...]]:
    """Build the round-1 emissions shape that mirrors R-64-SYM:
    monitor + network each emit four generic-noise mentions (two for
    gold A/B, two for the decoy) with comparable magnitudes — so the
    bipartite ``(role × tag, kind, magnitude)`` multiset is identical
    for gold and decoy after subscription-routing."""
    return {
        ROLE_MONITOR: (
            _emit("LATENCY_SPIKE", f"p95_ms=2200 service={A}"),
            _emit("ERROR_RATE_SPIKE", f"error_rate=0.20 service={B}"),
            _emit("LATENCY_SPIKE", f"p95_ms=2100 service={decoy}"),
            _emit("ERROR_RATE_SPIKE", f"error_rate=0.18 service={decoy}"),
        ),
        ROLE_NETWORK: (
            _emit("FW_BLOCK_SURGE", f"rule=deny count=11 service={A}"),
            _emit("FW_BLOCK_SURGE", f"rule=deny count=10 service={B}"),
            _emit("FW_BLOCK_SURGE", f"rule=deny count=9 service={decoy}"),
            _emit("FW_BLOCK_SURGE", f"rule=deny count=8 service={decoy}"),
        ),
        ROLE_DB_ADMIN: (),
        ROLE_SYSADMIN: (),
    }


# Round-2 disambiguator-payload templates per sub-bank. The first
# placeholder is the gold-A tag, the second is gold-B, the third is the
# decoy. Each template produces the bytes that the W18 scorer reads.
#
# The templates are pre-committed and form the W18 closed-vocabulary
# relational-mention contract for Phase-65. Every variant the synthetic
# bench emits is *inside* the W18 closure (the exact-match + compound-
# split layer); free-form natural-language relational mentions fall
# *outside* the closure (the W18-C-LEARNED extension axis).

_R2_PAYLOAD_COMPAT = {
    "deadlock":           "deadlock relation={A}_{B}_join wait_chain=2",
    "pool_exhaustion":    ("pool active=200/200 waiters=145 cluster=primary "
                            "pool_chain={A}_{B} exhaustion=upstream_{A}_to_{B}"),
    "disk_fill":          ("/var/log used=99% fs=/ host=primary "
                            "mount=/storage/{A}/{B}/var/log"),
    "slow_query_cascade": ("q#9 mean_ms=4100 cluster=primary "
                            "query_path=svc_{A}_then_svc_{B}"),
}

_R2_PAYLOAD_NO_COMPAT = {
    "deadlock":           "deadlock wait_chain=2 detected_at=t=120",
    "pool_exhaustion":    "pool active=200/200 waiters=145 cluster=primary",
    "disk_fill":          "/var/log used=99% fs=/ host=primary",
    "slow_query_cascade": "q#9 mean_ms=4100 cluster=primary",
}

_R2_PAYLOAD_CONFOUND = {
    "deadlock":           "deadlock relation={A}_{B}_{decoy}_join wait_chain=2",
    "pool_exhaustion":    ("pool active=200/200 waiters=145 cluster=primary "
                            "pool_chain={A}_{B}_{decoy}"),
    "disk_fill":          ("/var/log used=99% fs=/ host=primary "
                            "mount=/storage/{A}/{B}/{decoy}/var/log"),
    "slow_query_cascade": ("q#9 mean_ms=4100 cluster=primary "
                            "query_path=svc_{A}_then_svc_{B}_via_svc_{decoy}"),
}

_R2_PAYLOAD_DECEIVE = {
    "deadlock":           "deadlock relation={decoy}_{decoy}_join wait_chain=2",
    "pool_exhaustion":    ("pool active=200/200 waiters=145 cluster=primary "
                            "pool_chain={decoy}_{decoy}"),
    # Note: avoid the literal ``/storage/`` path component here — when
    # the gold A tag for the disk_fill family is ``storage``, that
    # path component would accidentally re-introduce a gold mention
    # and the bench would no longer be a clean DECEIVE regime
    # (the W18 strict-asymmetric branch would land on
    # ``{storage, decoy}`` instead of ``{decoy}`` on disk-fill cells).
    "disk_fill":          ("/var/log used=99% fs=/ host=primary "
                            "mount=/srv/{decoy}/{decoy}/var/log"),
    "slow_query_cascade": ("q#9 mean_ms=4100 cluster=primary "
                            "query_path=svc_{decoy}_then_svc_{decoy}"),
}


def _r2_disambiguator_kind(root_cause: str) -> str:
    return {
        "deadlock":           "DEADLOCK_SUSPECTED",
        "pool_exhaustion":    "POOL_EXHAUSTION",
        "disk_fill":          "DISK_FILL_CRITICAL",
        "slow_query_cascade": "SLOW_QUERY_OBSERVED",
    }[root_cause]


def _r2_disambiguator_role(root_cause: str) -> str:
    return {
        "deadlock":           ROLE_DB_ADMIN,
        "pool_exhaustion":    ROLE_DB_ADMIN,
        "disk_fill":          ROLE_SYSADMIN,
        "slow_query_cascade": ROLE_DB_ADMIN,
    }[root_cause]


def _build_p65_scenario(*, root_cause: str, A: str, B: str, decoy: str,
                            bank: str) -> MultiRoundScenario:
    if bank == "compat":
        templates = _R2_PAYLOAD_COMPAT
    elif bank == "no_compat":
        templates = _R2_PAYLOAD_NO_COMPAT
    elif bank == "confound":
        templates = _R2_PAYLOAD_CONFOUND
    elif bank == "deceive":
        templates = _R2_PAYLOAD_DECEIVE
    else:
        raise ValueError(f"unknown bank {bank!r}")
    template = templates[root_cause]
    payload = template.format(A=A, B=B, decoy=decoy)
    kind = _r2_disambiguator_kind(root_cause)
    role = _r2_disambiguator_role(root_cause)
    sid = f"p65{bank}_{root_cause}_{A}_{B}__sym_{decoy}"
    description = (
        f"Phase-65-{bank}: symmetric-corroboration. Gold {A}/{B} AND "
        f"decoy {decoy} all corroborated by 2 distinct routed roles "
        f"(monitor + network). Round-2 {kind} payload: {payload!r}.")
    round2 = {ROLE_MONITOR: (), ROLE_NETWORK: (),
              ROLE_DB_ADMIN: (), ROLE_SYSADMIN: ()}
    round2[role] = (_emit(kind, payload),)
    return MultiRoundScenario(
        scenario_id=sid,
        description=description,
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause=root_cause,
        gold_remediation=_REMEDIATION[root_cause],
        round1_emissions=_round1_symmetric(A, B, decoy),
        round2_emissions=round2,
    )


_P65_FAMILIES = (
    ("deadlock",           "orders",  "payments"),
    ("pool_exhaustion",    "api",     "db"),
    ("disk_fill",          "storage", "logs_pipeline"),
    ("slow_query_cascade", "web",     "db_query"),
)
_P65_DECOYS = (
    "search_index", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "cache", "scratch_pool",
)


def build_phase65_bank(*, bank: str, n_replicates: int = 2,
                          seed: int = 11
                          ) -> list[MultiRoundScenario]:
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for (root_cause, A, B) in _P65_FAMILIES:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = _P65_DECOYS[(i + r) % len(_P65_DECOYS)]
            sc = _build_p65_scenario(
                root_cause=root_cause, A=A, B=B, decoy=chosen, bank=bank)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


# =============================================================================
# Bench-property witnesses
# =============================================================================


def _bench_property_p65(sc: MultiRoundScenario,
                          round1_cands, round2_cands,
                          ) -> dict[str, Any]:
    """Mechanically-verified Phase-65 bench property witnesses.

    Symmetric round-1 corroboration AND named relational-mention
    convention in round-2.
    """
    def _service_role_set(cands, service):
        roles: set[str] = set()
        for (src, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            np = normalize_payload(payload)
            if f"service={service}" in np:
                roles.add(src)
        return roles

    A, B = sc.gold_services_pair
    decoy = sc.decoy_storm_service
    A_roles = _service_role_set(round1_cands, A)
    B_roles = _service_role_set(round1_cands, B)
    decoy_roles = _service_role_set(round1_cands, decoy)
    sym_corr = (len(A_roles) >= 2 and len(B_roles) >= 2
                  and len(decoy_roles) >= 2)

    # Round-2 disambiguator payload — what W18 would read.
    r2_payloads: list[str] = []
    for (src, to, _k, payload, _e) in round2_cands:
        if to == ROLE_AUDITOR:
            r2_payloads.append(payload)
    r2_text = " ".join(r2_payloads).lower()

    A_in_r2 = (A.lower() in r2_text)
    B_in_r2 = (B.lower() in r2_text)
    decoy_in_r2 = (decoy.lower() in r2_text)

    if A_in_r2 and B_in_r2 and not decoy_in_r2:
        r2_class = "compat"        # gold-only mentions
    elif (not A_in_r2) and (not B_in_r2) and (not decoy_in_r2):
        r2_class = "no_compat"     # no service mentions
    elif A_in_r2 and B_in_r2 and decoy_in_r2:
        r2_class = "confound"      # both gold and decoy
    elif decoy_in_r2 and not (A_in_r2 or B_in_r2):
        r2_class = "deceive"       # decoy only
    else:
        r2_class = "mixed"         # any other partial pattern

    return {
        "gold_a_role_count": len(A_roles),
        "gold_b_role_count": len(B_roles),
        "decoy_role_count": len(decoy_roles),
        "symmetric_corroboration_holds": sym_corr,
        "r2_class": r2_class,
        "r2_mentions_gold_a": A_in_r2,
        "r2_mentions_gold_b": B_in_r2,
        "r2_mentions_decoy": decoy_in_r2,
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
    }


# =============================================================================
# Strategy / decoder dispatch
# =============================================================================


_R65_STRATEGIES: tuple[tuple[str, str], ...] = (
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
    # SDK v3.19 — W18 anchor: relational-compatibility disambiguator.
    ("capsule_relational_compat", "relational_compat"),
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
                     "capsule_relational_compat"):
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
    return fac


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
    """Run the W18 disambiguator. The W18 method composes the W15
    attention-aware decoder *and* the relational-compatibility
    projection. Returns the same shape as :func:`_decode_with_packer`
    plus a ``compatibility`` block in the answer dict."""
    inner = AttentionAwareBundleDecoder(T_decoder=T_decoder)
    w18 = RelationalCompatibilityDisambiguator(inner=inner)
    # We have to feed per-round bundles; the union here is already
    # flat. Reconstruct two-round split using round_index_hint.
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
    if "compatibility" in ans:
        stats["compatibility"] = ans["compatibility"]
    return ans, stats


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


def _run_capsule_strategy(
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands, round2_cands,
        T_decoder: int | None = None,
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase65_relational_disambig",
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
# Phase 65 driver
# =============================================================================


def run_phase65(*,
                  bank: str = "compat",
                  n_eval: int | None = None,
                  K_auditor: int = 12,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  T_decoder: int | None = None,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 65 over one of {compat, no_compat, confound, deceive}."""
    bank_obj = build_phase65_bank(
        bank=bank, n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank_obj = bank_obj[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R65_STRATEGIES
    }

    for sc in bank_obj:
        round1_cands = _build_round_candidates(sc.round1_emissions)
        round2_cands = _build_round_candidates(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p65(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R65_STRATEGIES:
            fac = _make_factory(sname, priorities, budgets)
            r, ps = _run_capsule_strategy(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                T_decoder=T_decoder)
            results.append(r)
            if ps:
                pack_stats_per_strategy[sname].append({
                    "scenario_id": sc.scenario_id,
                    **ps,
                })

    strategy_names = ("substrate",) + tuple(s[0] for s in _R65_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w18_minus_attention_aware": _gap(
            "capsule_relational_compat", "capsule_attention_aware"),
        "w18_minus_layered": _gap(
            "capsule_relational_compat", "capsule_layered_multi_round"),
        "w18_minus_fifo": _gap(
            "capsule_relational_compat", "capsule_fifo"),
        "w18_minus_substrate": _gap(
            "capsule_relational_compat", "substrate"),
        "max_non_w18_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_relational_compat"),
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
        "scenarios_r2_class_compat": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("r2_class") == "compat"),
        "scenarios_r2_class_no_compat": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("r2_class") == "no_compat"),
        "scenarios_r2_class_confound": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("r2_class") == "confound"),
        "scenarios_r2_class_deceive": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("r2_class") == "deceive"),
        "K_auditor": K_auditor,
        "T_decoder": T_decoder,
        "bank": bank,
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
        n_abstain = sum(
            1 for r in rows
            if r.get("compatibility", {}).get("abstained"))
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
            "n_w18_abstain_cells": int(n_abstain),
        }

    pack_stats_summary = {
        s: _agg_packstats(pack_stats_per_strategy.get(s, []))
        for s in ("capsule_layered_fifo_packed",
                   "capsule_attention_aware",
                   "capsule_relational_compat")
    }

    if verbose:
        print(f"[phase65] bank={bank} T_decoder={T_decoder} "
              f"n_eval={len(bank_obj)} K_auditor={K_auditor} "
              f"bank_seed={bank_seed}", file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase65]   {s:32s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase65] {k}: {v:+.3f}", file=sys.stderr, flush=True)

    return {
        "schema": "phase65.relational_disambiguation.v1",
        "config": {
            "bank": bank, "n_eval": len(bank_obj),
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "K_producer": K_producer, "T_producer": T_producer,
            "inbox_capacity": inbox_capacity, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates, "T_decoder": T_decoder,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "pack_stats_summary": pack_stats_summary,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }


def run_phase65_seed_stability_sweep(*,
                                          bank: str = "compat",
                                          T_decoder: int | None = None,
                                          n_eval: int = 8,
                                          K_auditor: int = 12,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase65.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase65(bank=bank, T_decoder=T_decoder,
                          n_eval=n_eval, K_auditor=K_auditor,
                          bank_seed=seed)
        out["per_seed"][str(seed)] = {
            "headline_gap": rep["headline_gap"],
            "pooled": {
                k: v["accuracy_full"]
                for k, v in rep["pooled"].items()
            },
            "scenarios_with_symmetric_corroboration":
                rep["bench_summary"]["scenarios_with_symmetric_corroboration"],
            "audit_ok_grid": rep["audit_ok_grid"],
        }
    gaps = [out["per_seed"][str(s)]["headline_gap"]["w18_minus_attention_aware"]
             for s in seeds]
    out["min_w18_minus_attention_aware"] = min(gaps) if gaps else 0.0
    out["max_w18_minus_attention_aware"] = max(gaps) if gaps else 0.0
    out["mean_w18_minus_attention_aware"] = (
        round(sum(gaps) / len(gaps), 4) if gaps else 0.0)
    return out


def run_cross_regime_synthetic(*,
                                  n_eval: int = 8,
                                  bank_seed: int = 11,
                                  K_auditor: int = 12,
                                  T_auditor: int = 256,
                                  T_decoder_tight: int = 24,
                                  ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase65.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
        },
    }
    out["r65_compat_loose"] = run_phase65(
        bank="compat", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r65_compat_tight"] = run_phase65(
        bank="compat", T_decoder=T_decoder_tight,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r65_no_compat"] = run_phase65(
        bank="no_compat", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r65_confound"] = run_phase65(
        bank="confound", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r65_deceive"] = run_phase65(
        bank="deceive", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)

    def _acc(cell: str, strategy: str) -> float:
        return float(out[cell]["pooled"][strategy]["accuracy_full"])

    out["headline_summary"] = {
        "r65_compat_loose_w18":
            _acc("r65_compat_loose", "capsule_relational_compat"),
        "r65_compat_loose_attention_aware":
            _acc("r65_compat_loose", "capsule_attention_aware"),
        "r65_compat_loose_substrate":
            _acc("r65_compat_loose", "substrate"),
        "r65_compat_tight_w18":
            _acc("r65_compat_tight", "capsule_relational_compat"),
        "r65_compat_tight_attention_aware":
            _acc("r65_compat_tight", "capsule_attention_aware"),
        "r65_no_compat_w18":
            _acc("r65_no_compat", "capsule_relational_compat"),
        "r65_no_compat_attention_aware":
            _acc("r65_no_compat", "capsule_attention_aware"),
        "r65_confound_w18":
            _acc("r65_confound", "capsule_relational_compat"),
        "r65_deceive_w18":
            _acc("r65_deceive", "capsule_relational_compat"),
        "w18_minus_attention_compat_loose":
            round(_acc("r65_compat_loose", "capsule_relational_compat")
                   - _acc("r65_compat_loose", "capsule_attention_aware"), 4),
        "w18_minus_attention_compat_tight":
            round(_acc("r65_compat_tight", "capsule_relational_compat")
                   - _acc("r65_compat_tight", "capsule_attention_aware"), 4),
    }
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 65 — relational-compatibility "
                     "disambiguation under symmetric corroboration "
                     "(SDK v3.19 / W18 family).")
    p.add_argument("--bank", type=str, default="compat",
                    choices=("compat", "no_compat", "confound", "deceive"))
    p.add_argument("--K-auditor", type=int, default=12)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--decoder-budget", type=int, default=-1,
                    help="Strict T_decoder budget. -1 = None.")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--cross-regime-synthetic", action="store_true")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    T_dec = None if args.decoder_budget < 0 else int(args.decoder_budget)
    if args.cross_regime_synthetic:
        report = run_cross_regime_synthetic(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor)
    elif args.seed_sweep:
        report = run_phase65_seed_stability_sweep(
            bank=args.bank, T_decoder=T_dec,
            n_eval=args.n_eval, K_auditor=args.K_auditor)
    else:
        report = run_phase65(
            bank=args.bank, n_eval=args.n_eval,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates, T_decoder=T_dec,
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
            print(f"[phase65] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
