"""Phase 67 — outside-information acquisition under symmetric
bundle-internal contradiction (SDK v3.21, W20 family anchor).

The follow-up to SDK v3.20 (W19) on the named research frontier the W19
milestone explicitly left conjectural: the **outside-information
escape** (W19-C-OUTSIDE) from BOTH bundle-only walls — W19-Λ-total (no
asymmetric witness anywhere in the bundle) and W19-Λ-outside (witnesses
exist but are themselves symmetric across primary's named set and the
complement).

The W19-Λ-outside anchor (Phase-66 R-66-OUTSIDE-REQUIRED) proves that
*every* closed-form bundle-only scorer ties FIFO at 0.000 by abstention
when the asymmetric-witness count is symmetric across the candidate
set. SDK v3.20 named the next research move as **outside information**
— a scorer with access to an evidence channel that does *not* live
inside the W15-packed bundle.

W20's :class:`OutsideWitnessAcquisitionDisambiguator` is the smallest
move in that direction. When the inner W19 abstains via the symmetric
branch, W20 issues exactly *one* hypothesis-conditioned query to a
registered :class:`OutsideWitnessOracle`; the oracle's reply is parsed
through the same closed-vocabulary closure W18 / W19 use on in-bundle
witnesses. The asymmetric outside witness lets W20 project the answer
onto the strict-max-aw subset of the admitted tag set.

Phase-67 is the named harder regime where the W20 method's required
ingredient — at least one *trusted asymmetric outside witness* — is
consistently present across the *positive-anchor* sub-bank
(R-67-OUTSIDE-RESOLVES) AND where three named falsifier regimes
(R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED, R-67-JOINT-DECEPTION)
make the W20 conditionality sharp.

Recap of why the W19-Λ-outside wall stands without W20
=======================================================

On R-66-OUTSIDE-REQUIRED the bench property is:

  * Both gold A/B AND the decoy mentioned by ≥ 2 distinct routed
    producer roles in round-1 via generic-noise kinds with comparable
    magnitudes (W17-Λ-symmetric *applies*).
  * Round-2 carries one specific-tier *primary* disambiguator with a
    payload mentioning ONLY the decoy (a deceptive primary).
  * Round-2 also carries one specific-tier *secondary* witness from
    monitor whose payload mentions ALL three (gold A, gold B, decoy)
    — so the asymmetric-witness count is uniform across the admitted
    tag set.

Under this regime:

  * W17-Λ-symmetric extends verbatim for every method pre-W18.
  * W18 sees positive scores on every admitted tag from the
    concatenated disambiguator text → abstains; W15 inner ties FIFO.
  * W19's witness counter (excluding primary) sees aw uniform across
    {A, B, decoy} → confound branch finds no strict-max subset
    proper of admitted_tags → abstains via
    ``W19_BRANCH_ABSTAINED_SYMMETRIC``; ties FIFO at 0.000.
  * **No closed-form bundle-only scorer can escape this regime**
    without an additional information channel.

Phase-67 sub-banks
==================

Five pre-committed cells plus a cross-regime synthetic summary.
Every sub-bank uses the same R-66-OUTSIDE-REQUIRED bundle shape
(deceptive primary + symmetric secondary witness); the *only*
variable is the registered outside oracle.

* **R-67-OUTSIDE-REQUIRED-BASELINE** (W20-3 backward-compat anchor;
  no oracle registered, ``T_decoder=None``). Verifies that with no
  oracle (or an :class:`AbstainingOracle`) W20 reduces to W19
  byte-for-byte AND ties FIFO at 0.000. Establishes that the W20
  method is purely additive and does not regress.
* **R-67-OUTSIDE-RESOLVES-LOOSE** (W20-1 anchor; deterministic
  :class:`ServiceGraphOracle`, ``T_decoder=None``). The oracle
  returns a service-graph reply mentioning the *gold pair*
  asymmetrically (``service=A service=B``) since the gold pair has
  a true topological edge in
  :func:`build_incident_triage_service_graph` and the decoy is
  isolated. W20 fires
  :data:`W20_BRANCH_OUTSIDE_RESOLVED`; projects the answer to
  ``{A, B}``. **+1.000 strict gain over W19** (which abstains via
  :data:`W19_BRANCH_ABSTAINED_SYMMETRIC` on the same regime).
* **R-67-OUTSIDE-RESOLVES-TIGHT** (W20-1 + W15 composition anchor;
  deterministic :class:`ServiceGraphOracle`, ``T_decoder=24``).
  Same R-67-OUTSIDE-RESOLVES shape under decoder-side budget
  pressure. The W20 oracle reads only ``query.admitted_tags`` (the
  W15-packed-and-W19-admitted set), NOT the full bundle; the
  inner W15 ``tokens_kept`` is byte-for-byte unchanged. The W20
  ``n_outside_tokens`` is recorded as a *strict additional* token
  cost (≤ ``max_response_tokens``). Bounded-context honesty
  preserved.
* **R-67-OUTSIDE-NONE** (W20-Λ-none falsifier;
  :class:`AbstainingOracle`, ``T_decoder=None``). The oracle has
  no information about the symmetric-witness ambiguity; W20 falls
  through to W19's symmetric abstention; ties FIFO at 0.000.
  Names the structural limit when the registered outside source
  has no signal.
* **R-67-OUTSIDE-COMPROMISED** (W20-Λ-compromised falsifier;
  :class:`CompromisedServiceGraphOracle`, ``T_decoder=None``).
  The oracle returns a *decoy-asymmetric* reply mentioning ONLY
  the decoy. W20 trusts the oracle and projects to the decoy
  set; FAILS at 0.000. Names the structural limit when the
  outside source itself is adversarial — the fix is NOT a
  richer scorer; it is *oracle integrity*.
* **R-67-JOINT-DECEPTION** (W20-Λ-joint-deception falsifier;
  :class:`CompromisedServiceGraphOracle`, ``T_decoder=None``,
  primary AND secondary AND oracle ALL favour decoy).
  Equivalent to R-67-OUTSIDE-COMPROMISED on the answer field
  (W20 fails) but the bench exhibits the named *joint*
  consistency property (the deception is internally coherent
  across primary, secondary, AND outside source). Names the
  structural limit when ALL evidence channels are jointly
  compromised.

Theorem family W20 (minted by this milestone)
==============================================

* **W20-Λ-outside (extension)** (proved-empirical + structural
  sketch). W19-Λ-outside extends verbatim to R-67-OUTSIDE-REQUIRED-
  BASELINE (the same bundle shape, no oracle). Every capsule
  strategy in the SDK ties FIFO at 0.000 by construction; W19
  abstains via :data:`W19_BRANCH_ABSTAINED_SYMMETRIC`.
* **W20-1** (proved-conditional + proved-empirical, n=8 saturated × 5
  seeds × 2 cells). Pairing the W19
  :class:`BundleContradictionDisambiguator` with the W20
  :class:`OutsideWitnessAcquisitionDisambiguator` over a clean
  :class:`ServiceGraphOracle` strictly improves ``accuracy_full``
  over the strongest non-W20 capsule baseline by ``+1.000`` on
  R-67-OUTSIDE-RESOLVES-LOOSE AND on R-67-OUTSIDE-RESOLVES-TIGHT
  (``T_decoder = 24``), stable across 5/5 alternate ``bank_seed``
  values. The first capsule-native multi-agent-coordination
  method that crosses the W19-Λ-outside wall on a regime where
  the wall actually applies.
* **W20-2** (proved by inspection + mechanically-checked). The W20
  scorer is deterministic and byte-stable: ``decode_rounds``
  returns byte-for-byte identical answers given byte-identical
  inputs and a deterministic oracle; the per-tag outside-witness
  count depends only on the oracle's reply tokens and the
  admitted tag set.
* **W20-3** (proved-empirical full programme regression). On
  R-54..R-66 default banks, the W20 method ties the W19 method
  byte-for-byte on the answer field — either via *no-trigger*
  (W19's branch is not in the trigger set) or via *outside-
  abstained* (oracle returns ``None`` because the admitted set
  has no asymmetric topological edge). With ``enabled = False``,
  the W20 method reduces to W19 byte-for-byte.
* **W20-Λ-none** (proved-empirical, n=8 saturated). On
  R-67-OUTSIDE-NONE, the registered :class:`AbstainingOracle`
  returns ``None`` for every cell; W20 falls through to W19's
  symmetric abstention; ties FIFO at 0.000. Names the structural
  limit when the registered outside source has no signal.
* **W20-Λ-compromised** (proved-empirical, n=8 saturated). On
  R-67-OUTSIDE-COMPROMISED, the registered
  :class:`CompromisedServiceGraphOracle` returns a decoy-only
  asymmetric reply for every cell; W20 trusts the oracle and
  projects to decoy; FAILS at 0.000. Names the structural limit
  when the outside source itself is adversarial — the escape is
  *partial* by design. Bounded above by oracle integrity.
* **W20-Λ-joint-deception** (proved-empirical, n=8 saturated). On
  R-67-JOINT-DECEPTION, primary + secondary + oracle all favour
  decoy AND are internally consistent; no single-oracle
  cross-check can detect the deception. W20 ties W19 at 0.000.
  Names the structural limit when *all* evidence channels are
  jointly compromised.
* **W20-Λ-real** (proved-conditional + empirical-research,
  conjectural-empirical-on-Mac-1). Real-LLM transfer of the
  outside-witness convention via :class:`LLMAdjudicatorOracle`
  is conditional on the LLM emitting a reply whose token set
  matches the admitted-tag closure. If the LLM emits a free-form
  natural-language reply (e.g. "I think it's the order-payment
  pair"), the W20 exact-match parser misses by construction; the
  W20-Λ-real wall is the same closed-vocabulary boundary that
  bounds W19 / W18 / W13 / W12.
* **W20-C-LEARNED** (conjectural). A small distilled outside-
  source scorer (e.g. a learned trust-weighted aggregator over
  multiple oracles + character-bigram bundle features)
  outperforms the closed-form rule on a held-out cross-bench
  where the LLM emits free-form replies outside the synthetic
  bench's exact-match closure.
* **W20-C-MULTI-ORACLE** (conjectural). Consulting *multiple*
  outside oracles (e.g. service-graph + change-history + on-call
  notes) and aggregating via majority / weighted vote escapes
  W20-Λ-compromised when *some* oracles remain trustworthy. The
  W20-C-MULTI-ORACLE axis is bounded above by the same
  closed-vocabulary closure; aggregation is a research move,
  not a structural escape from oracle compromise.

Phase-67 also closes a *partial* discharge of W19-Λ-outside on the
*bundle-resolvable-by-oracle* direction: the wall remains a real
structural limit *for any bundle-only scorer*, but the
outside-information axis crosses it when an asymmetric outside
source is registered. The W19-Λ-outside wall is therefore named
*partially discharged* by W20-1 (positive direction) AND remains
real on the *no-oracle* and *compromised-oracle* directions.

CLI
---

::

    # R-67-OUTSIDE-RESOLVES-LOOSE (W20-1 anchor, synthetic):
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank outside_resolves --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-67-OUTSIDE-RESOLVES-TIGHT (W20-1 + W15 composition):
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank outside_resolves --decoder-budget 24 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-67-OUTSIDE-NONE (W20-Λ-none falsifier):
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank outside_none --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-67-OUTSIDE-COMPROMISED (W20-Λ-compromised falsifier):
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank outside_compromised --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-67-JOINT-DECEPTION (W20-Λ-joint-deception falsifier):
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank joint_deception --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

    # Seed-stability sweep on the headline regime:
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank outside_resolves --seed-sweep \\
        --K-auditor 12 --n-eval 8 --out -

    # Live LLM adjudicator probe (Mac-1 Ollama; honest negative when
    # the LLM emits free-form replies):
    python3 -m vision_mvp.experiments.phase67_outside_information \\
        --bank outside_resolves --live-adjudicator \\
        --adjudicator-model qwen2.5-coder:7b \\
        --K-auditor 12 --n-eval 4 --out -
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
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, AbstainingOracle, AttentionAwareBundleDecoder,
    BundleAwareTeamDecoder,
    BundleContradictionDisambiguator,
    CapsuleContextPacker,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CompromisedServiceGraphOracle,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, FifoContextPacker,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    OutsideWitnessAcquisitionDisambiguator,
    RelationalCompatibilityDisambiguator,
    RobustMultiRoundBundleDecoder, RoleBudget,
    ServiceGraphOracle,
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
from vision_mvp.experiments.phase66_deceptive_ambiguity import (
    _BANK_PRIMARY_TEMPLATES,
    _R2_PRIMARY_DECEIVE,
    _R2_SECONDARY_OUTSIDE_REQUIRED,
    _build_round_candidates_p66,
    _r2_primary_kind, _r2_primary_role,
    _r2_secondary_kind, _r2_secondary_role,
    _round1_symmetric, _emit, _REMEDIATION,
    _P66_FAMILIES, _P66_DECOYS,
)


# =============================================================================
# Phase-67 scenario family
# =============================================================================
#
# Every Phase-67 sub-bank reuses the R-66-OUTSIDE-REQUIRED bundle shape:
# a deceptive primary (mentions decoy only) AND a symmetric secondary
# witness (mentions ALL three). The variable across sub-banks is the
# *registered oracle*, NOT the bundle.
#
# This is intentional — the milestone story is:
#   "Bundle-only methods cannot escape this regime; the only thing
#    that varies the W20 result is the integrity / signal of the
#    outside source."


_VALID_BANKS = (
    "outside_required_baseline",
    "outside_resolves",
    "outside_none",
    "outside_compromised",
    "joint_deception",
)


def _build_p67_scenario(*, root_cause: str, A: str, B: str, decoy: str,
                            bank: str) -> MultiRoundScenario:
    """Build one Phase-67 scenario.

    All five sub-banks use the same R-66-OUTSIDE-REQUIRED primary +
    secondary shape (deceptive primary + symmetric secondary
    witness). The bundle shape is bank-invariant; the oracle is
    bank-specific (registered at strategy-dispatch time, not in the
    scenario).
    """
    if bank not in _VALID_BANKS:
        raise ValueError(f"unknown bank {bank!r}; valid: {_VALID_BANKS}")
    primary_template = _R2_PRIMARY_DECEIVE[root_cause]
    primary_payload = primary_template.format(A=A, B=B, decoy=decoy)
    primary_kind = _r2_primary_kind(root_cause)
    primary_role = _r2_primary_role(root_cause)

    secondary_template = _R2_SECONDARY_OUTSIDE_REQUIRED[root_cause]
    secondary_payload = secondary_template.format(A=A, B=B, decoy=decoy)
    secondary_kind = _r2_secondary_kind(root_cause)
    secondary_role = _r2_secondary_role(root_cause)

    sid = f"p67{bank}_{root_cause}_{A}_{B}__sym_{decoy}"
    description = (
        f"Phase-67-{bank}: R-66-OUTSIDE-REQUIRED bundle (symmetric "
        f"witness pattern). Primary {primary_kind} from {primary_role}: "
        f"{primary_payload!r}. Secondary {secondary_kind} from "
        f"{secondary_role}: {secondary_payload!r}.")

    round2: dict[str, tuple[tuple[str, str], ...]] = {
        ROLE_MONITOR: (), ROLE_NETWORK: (),
        ROLE_DB_ADMIN: (), ROLE_SYSADMIN: ()
    }
    primary_emissions = [_emit(primary_kind, primary_payload)]
    secondary_emissions = [_emit(secondary_kind, secondary_payload)]
    if primary_role == secondary_role:
        round2[primary_role] = tuple(primary_emissions + secondary_emissions)
    else:
        round2[primary_role] = tuple(primary_emissions)
        round2[secondary_role] = tuple(secondary_emissions)

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


def build_phase67_bank(*, bank: str, n_replicates: int = 2,
                          seed: int = 11
                          ) -> list[MultiRoundScenario]:
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for (root_cause, A, B) in _P66_FAMILIES:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = _P66_DECOYS[(i + r) % len(_P66_DECOYS)]
            sc = _build_p67_scenario(
                root_cause=root_cause, A=A, B=B, decoy=chosen, bank=bank)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


# =============================================================================
# Bench-property witnesses
# =============================================================================


def _bench_property_p67(sc: MultiRoundScenario,
                          round1_cands, round2_cands,
                          ) -> dict[str, Any]:
    """Mechanically-verified Phase-67 bench property witnesses.

    Phase-67 reuses the R-66-OUTSIDE-REQUIRED bundle shape for every
    sub-bank — so the bench-property invariants are: (a) round-1
    symmetric corroboration, (b) primary mentions decoy only,
    (c) secondary mentions all three. The bench-property witnesses
    are bank-invariant.
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

    primary_payloads: list[str] = []
    secondary_payloads: list[str] = []
    primary_role = _r2_primary_role(sc.gold_root_cause)
    primary_kind = _r2_primary_kind(sc.gold_root_cause)
    secondary_kind_canonical = _r2_secondary_kind(sc.gold_root_cause)
    for (src, to, kind, payload, _e) in round2_cands:
        if to != ROLE_AUDITOR:
            continue
        if src == primary_role and kind == primary_kind:
            primary_payloads.append(payload)
        elif kind == secondary_kind_canonical:
            secondary_payloads.append(payload)

    primary_text = " ".join(primary_payloads).lower()
    secondary_text = " ".join(secondary_payloads).lower()

    def _classify(text: str) -> str:
        if not text:
            return "absent"
        a_in = (A.lower() in text)
        b_in = (B.lower() in text)
        d_in = (decoy.lower() in text)
        if a_in and b_in and not d_in:
            return "gold_only"
        if (not a_in) and (not b_in) and (not d_in):
            return "no_mention"
        if a_in and b_in and d_in:
            return "all_three"
        if d_in and not (a_in or b_in):
            return "decoy_only"
        return "mixed"

    primary_class = _classify(primary_text)
    secondary_class = _classify(secondary_text)

    return {
        "gold_a_role_count": len(A_roles),
        "gold_b_role_count": len(B_roles),
        "decoy_role_count": len(decoy_roles),
        "symmetric_corroboration_holds": sym_corr,
        "primary_class": primary_class,
        "secondary_class": secondary_class,
        "shape": [primary_class, secondary_class],
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
    }


# All Phase-67 sub-banks share the same R-66-OUTSIDE-REQUIRED bundle
# shape (decoy_only primary, all_three secondary).
_P67_EXPECTED_SHAPE: tuple[str, str] = ("decoy_only", "all_three")


# =============================================================================
# Strategy / decoder dispatch
# =============================================================================


_R67_STRATEGIES: tuple[tuple[str, str], ...] = (
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
    # SDK v3.19 — W18 anchor.
    ("capsule_relational_compat", "relational_compat"),
    # SDK v3.20 — W19 anchor.
    ("capsule_bundle_contradiction", "bundle_contradiction"),
    # SDK v3.21 — W20 anchor: outside-witness acquisition (the oracle
    # is bank-specific — registered at dispatch time).
    ("capsule_outside_witness", "outside_witness"),
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
                     "capsule_outside_witness"):
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
    if "compatibility" in ans:
        stats["compatibility"] = ans["compatibility"]
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
    if "compatibility" in ans:
        stats["compatibility"] = ans["compatibility"]
    if "trust" in ans:
        stats["trust"] = ans["trust"]
    return ans, stats


def _decode_with_w20(union, T_decoder, round_index_hint, oracle):
    """Run the W20 outside-witness-acquisition disambiguator."""
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
    if "compatibility" in ans:
        stats["compatibility"] = ans["compatibility"]
    if "trust" in ans:
        stats["trust"] = ans["trust"]
    if "outside" in ans:
        stats["outside"] = ans["outside"]
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


def _bank_to_oracle(bank: str, *, llm_adjudicator: Any | None = None
                      ) -> Any:
    """Map a Phase-67 bank label to the registered outside oracle."""
    if bank == "outside_required_baseline":
        return AbstainingOracle(oracle_id="baseline_no_oracle")
    if bank == "outside_resolves":
        if llm_adjudicator is not None:
            return llm_adjudicator
        return ServiceGraphOracle()
    if bank == "outside_none":
        return AbstainingOracle()
    if bank == "outside_compromised":
        return CompromisedServiceGraphOracle()
    if bank == "joint_deception":
        # joint-deception is the same oracle as compromised — the
        # *bundle* exhibits the named joint-consistency property.
        return CompromisedServiceGraphOracle(
            oracle_id="joint_deception_oracle")
    raise ValueError(f"unknown bank {bank!r}")


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
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase67_outside_information",
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
# Phase 67 driver
# =============================================================================


def run_phase67(*,
                  bank: str = "outside_resolves",
                  n_eval: int | None = None,
                  K_auditor: int = 12,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 2,
                  T_decoder: int | None = None,
                  oracle_override: Any | None = None,
                  llm_adjudicator: Any | None = None,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 67 over one of {outside_required_baseline,
    outside_resolves, outside_none, outside_compromised,
    joint_deception}. The ``oracle_override`` argument lets a
    caller register a custom oracle (e.g. a live LLM adjudicator)
    overriding the bank-default."""
    if bank not in _VALID_BANKS:
        raise ValueError(
            f"unknown bank {bank!r}; valid: {_VALID_BANKS}")
    bank_obj = build_phase67_bank(
        bank=bank, n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank_obj = bank_obj[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    if oracle_override is not None:
        oracle = oracle_override
    else:
        oracle = _bank_to_oracle(bank, llm_adjudicator=llm_adjudicator)

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R67_STRATEGIES
    }
    w20_branch_counts: dict[str, int] = {}

    for sc in bank_obj:
        round1_cands = _build_round_candidates_p66(sc.round1_emissions)
        round2_cands = _build_round_candidates_p66(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p67(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R67_STRATEGIES:
            fac = _make_factory(sname, priorities, budgets)
            r, ps = _run_capsule_strategy(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                T_decoder=T_decoder,
                oracle=oracle if dmode == "outside_witness" else None)
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

    strategy_names = ("substrate",) + tuple(s[0] for s in _R67_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w20_minus_w19": _gap(
            "capsule_outside_witness", "capsule_bundle_contradiction"),
        "w20_minus_w18": _gap(
            "capsule_outside_witness", "capsule_relational_compat"),
        "w20_minus_attention_aware": _gap(
            "capsule_outside_witness", "capsule_attention_aware"),
        "w20_minus_layered": _gap(
            "capsule_outside_witness", "capsule_layered_multi_round"),
        "w20_minus_fifo": _gap(
            "capsule_outside_witness", "capsule_fifo"),
        "w20_minus_substrate": _gap(
            "capsule_outside_witness", "substrate"),
        "max_non_w20_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_outside_witness"),
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
        "oracle_id": getattr(oracle, "oracle_id", "no_oracle"),
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
        n_w18_abstain = sum(
            1 for r in rows
            if r.get("compatibility", {}).get("abstained"))
        n_w19_abstain = sum(
            1 for r in rows
            if r.get("trust", {}).get("abstained"))
        n_w20_triggered = sum(
            1 for r in rows
            if r.get("outside", {}).get("triggered"))
        n_w20_resolved = sum(
            1 for r in rows
            if r.get("outside", {}).get("decoder_branch")
                == "outside_resolved")
        n_w20_abstain = sum(
            1 for r in rows
            if r.get("outside", {}).get("abstained")
                and r.get("outside", {}).get("triggered"))
        s_outside_tokens = sum(
            int(r.get("outside", {}).get("n_outside_tokens", 0))
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
            "n_w18_abstain_cells": int(n_w18_abstain),
            "n_w19_abstain_cells": int(n_w19_abstain),
            "n_w20_triggered_cells": int(n_w20_triggered),
            "n_w20_resolved_cells": int(n_w20_resolved),
            "n_w20_abstain_cells": int(n_w20_abstain),
            "outside_tokens_sum": int(s_outside_tokens),
            "outside_tokens_per_cell_avg": (
                round(s_outside_tokens / n, 4) if n > 0 else 0.0),
        }

    pack_stats_summary = {
        s: _agg_packstats(pack_stats_per_strategy.get(s, []))
        for s in ("capsule_layered_fifo_packed",
                   "capsule_attention_aware",
                   "capsule_relational_compat",
                   "capsule_bundle_contradiction",
                   "capsule_outside_witness")
    }

    if verbose:
        print(f"[phase67] bank={bank} oracle={bench_summary['oracle_id']} "
              f"T_decoder={T_decoder} n_eval={len(bank_obj)} "
              f"K_auditor={K_auditor} bank_seed={bank_seed}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase67]   {s:34s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase67] {k}: {v:+.3f}", file=sys.stderr, flush=True)
        if w20_branch_counts:
            print(f"[phase67] w20 branch counts: {w20_branch_counts}",
                  file=sys.stderr, flush=True)

    return {
        "schema": "phase67.outside_information.v1",
        "config": {
            "bank": bank, "n_eval": len(bank_obj),
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "K_producer": K_producer, "T_producer": T_producer,
            "inbox_capacity": inbox_capacity, "bank_seed": bank_seed,
            "bank_replicates": bank_replicates, "T_decoder": T_decoder,
            "oracle_id": getattr(oracle, "oracle_id", "no_oracle"),
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "pack_stats_summary": pack_stats_summary,
        "w20_branch_counts": w20_branch_counts,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }


def run_phase67_seed_stability_sweep(*,
                                          bank: str = "outside_resolves",
                                          T_decoder: int | None = None,
                                          n_eval: int = 8,
                                          K_auditor: int = 12,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase67.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase67(bank=bank, T_decoder=T_decoder,
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
            "scenarios_with_expected_shape":
                rep["bench_summary"]["scenarios_with_expected_shape"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "w20_branch_counts": rep["w20_branch_counts"],
        }
    gaps_w20_w19 = [
        out["per_seed"][str(s)]["headline_gap"]["w20_minus_w19"]
        for s in seeds
    ]
    gaps_w20_aa = [
        out["per_seed"][str(s)]["headline_gap"]["w20_minus_attention_aware"]
        for s in seeds
    ]
    out["min_w20_minus_w19"] = min(gaps_w20_w19) if gaps_w20_w19 else 0.0
    out["max_w20_minus_w19"] = max(gaps_w20_w19) if gaps_w20_w19 else 0.0
    out["mean_w20_minus_w19"] = (
        round(sum(gaps_w20_w19) / len(gaps_w20_w19), 4)
        if gaps_w20_w19 else 0.0)
    out["min_w20_minus_attention_aware"] = (
        min(gaps_w20_aa) if gaps_w20_aa else 0.0)
    out["max_w20_minus_attention_aware"] = (
        max(gaps_w20_aa) if gaps_w20_aa else 0.0)
    out["mean_w20_minus_attention_aware"] = (
        round(sum(gaps_w20_aa) / len(gaps_w20_aa), 4)
        if gaps_w20_aa else 0.0)
    return out


def run_cross_regime_synthetic(*,
                                  n_eval: int = 8,
                                  bank_seed: int = 11,
                                  K_auditor: int = 12,
                                  T_auditor: int = 256,
                                  T_decoder_tight: int = 24,
                                  ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase67.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
        },
    }
    out["r67_outside_required_baseline"] = run_phase67(
        bank="outside_required_baseline", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r67_outside_resolves_loose"] = run_phase67(
        bank="outside_resolves", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r67_outside_resolves_tight"] = run_phase67(
        bank="outside_resolves", T_decoder=T_decoder_tight,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r67_outside_none"] = run_phase67(
        bank="outside_none", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r67_outside_compromised"] = run_phase67(
        bank="outside_compromised", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r67_joint_deception"] = run_phase67(
        bank="joint_deception", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)

    def _acc(cell: str, strategy: str) -> float:
        return float(out[cell]["pooled"][strategy]["accuracy_full"])

    out["headline_summary"] = {
        "r67_outside_required_baseline_w20": _acc(
            "r67_outside_required_baseline", "capsule_outside_witness"),
        "r67_outside_required_baseline_w19": _acc(
            "r67_outside_required_baseline", "capsule_bundle_contradiction"),
        "r67_outside_resolves_loose_w20": _acc(
            "r67_outside_resolves_loose", "capsule_outside_witness"),
        "r67_outside_resolves_loose_w19": _acc(
            "r67_outside_resolves_loose", "capsule_bundle_contradiction"),
        "r67_outside_resolves_tight_w20": _acc(
            "r67_outside_resolves_tight", "capsule_outside_witness"),
        "r67_outside_resolves_tight_w19": _acc(
            "r67_outside_resolves_tight", "capsule_bundle_contradiction"),
        "r67_outside_none_w20": _acc(
            "r67_outside_none", "capsule_outside_witness"),
        "r67_outside_compromised_w20": _acc(
            "r67_outside_compromised", "capsule_outside_witness"),
        "r67_joint_deception_w20": _acc(
            "r67_joint_deception", "capsule_outside_witness"),
        "w20_minus_w19_outside_resolves_loose":
            round(_acc("r67_outside_resolves_loose",
                        "capsule_outside_witness")
                   - _acc("r67_outside_resolves_loose",
                           "capsule_bundle_contradiction"), 4),
        "w20_minus_w19_outside_resolves_tight":
            round(_acc("r67_outside_resolves_tight",
                        "capsule_outside_witness")
                   - _acc("r67_outside_resolves_tight",
                           "capsule_bundle_contradiction"), 4),
    }
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 67 — outside-witness acquisition under "
                     "symmetric bundle-internal contradiction "
                     "(SDK v3.21 / W20 family).")
    p.add_argument("--bank", type=str, default="outside_resolves",
                    choices=_VALID_BANKS)
    p.add_argument("--K-auditor", type=int, default=12)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--decoder-budget", type=int, default=-1,
                    help="Strict T_decoder budget. -1 = None.")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--cross-regime-synthetic", action="store_true")
    p.add_argument("--live-adjudicator", action="store_true",
                    help="Use a live LLMAdjudicatorOracle (Mac-1 Ollama). "
                         "Falls back to ServiceGraphOracle if unreachable.")
    p.add_argument("--adjudicator-model", type=str,
                    default="qwen2.5-coder:7b",
                    help="Ollama model tag for the live adjudicator.")
    p.add_argument("--adjudicator-endpoint", type=str,
                    default="http://127.0.0.1:11434")
    p.add_argument("--adjudicator-timeout", type=float, default=300.0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def _maybe_build_live_adjudicator(args) -> Any | None:
    if not args.live_adjudicator:
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
        print(f"[phase67] live-adjudicator unavailable: {exc!r}",
              file=sys.stderr, flush=True)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    T_dec = None if args.decoder_budget < 0 else int(args.decoder_budget)
    live_adjudicator = _maybe_build_live_adjudicator(args)
    if args.cross_regime_synthetic:
        report = run_cross_regime_synthetic(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor)
    elif args.seed_sweep:
        report = run_phase67_seed_stability_sweep(
            bank=args.bank, T_decoder=T_dec,
            n_eval=args.n_eval, K_auditor=args.K_auditor)
    else:
        report = run_phase67(
            bank=args.bank, n_eval=args.n_eval,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates, T_decoder=T_dec,
            llm_adjudicator=live_adjudicator,
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
            print(f"[phase67] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
