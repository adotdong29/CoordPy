"""Phase 62 — attention-aware capsule context packing under decoder-side
budget pressure.

SDK v3.16, W15 family anchor. **First decoder-side context-packing move**
in the CoordPy programme — the direct response to a structural gap that
SDK v3.15 left unaddressed: even when the bench property holds and the
cross-round decoder is in scope, a tight *decoder-side* token budget
``T_decoder`` can drop the round-2 specific-tier disambiguator under
naive FIFO packing. The W14 producer-protocol layer ensures the right
ingredients are *emitted*; the W15 attention-aware packer ensures the
right ingredients are *fed to the decoder* under bounded context.

Recap of the W15 structural move
=================================

Phase-58/59/60/61 admit handoffs under the per-round role budget
``K_role / T_role``. The cross-round decoder then consumes the
*concatenation* of admitted bundles. Under a strict downstream
``T_decoder`` (e.g. an LLM context window for an audit agent that
re-decodes the bundle), the prior decoders effectively rely on a FIFO
pack: handoffs are emitted in arrival order; truncation drops the
*tail* — which is where the round-2 disambiguator lives. This is
**W15-Λ-budget** (proved-empirical + structural sketch): under
``T_decoder`` strictly below the admitted union's token sum,
FIFO-packed cross-round decoders cannot retain the round-2 specific-
tier claim AND enough round-1 evidence simultaneously.

Phase-62 attacks both root causes
==================================

1. **Multi-hypothesis comparable-magnitude events.** Each scenario has
   2 gold services + 2 decoy services, every service mentioned by ≥ 2
   distinct producer roles in round 1 with operationally comparable
   magnitudes. The disambiguation is recoverable *only* from the
   round-2 specific causal claim AND cross-role corroboration of
   round-1 noise.

2. **Decoder-side token budget.** A new strict budget ``T_decoder`` is
   enforced over the cross-round bundle the decoder consumes. Default
   is loose (W15-3 backward-compat anchor); the
   ``--decoder-budget-tight`` mode sets ``T_decoder`` strictly below
   the admitted union's token sum so the budget bites.

3. **Salience-aware packing (W15-1 anchor).** The
   :class:`AttentionAwareBundleDecoder` runs a first-pass priority
   decode to elect a tentative ``root_cause``, scores every handoff by
   a closed-form salience function (tier + CCK + corroboration +
   magnitude + round index), and packs greedy by salience under
   ``T_decoder``. The kept bundle preserves the round-2 disambiguator
   AND the highest-salience round-1 mentions for each service.

Three pre-committed sub-banks
==============================

* **R-62-default** (synthetic, no decoder budget; W15-3 backward-
  compat anchor). Same multi-hypothesis events; ``T_decoder=None``.
  Salience reordering is a no-op on the answer field; the W15
  decoder ties the W13 layered decoder at ``accuracy_full = 1.000``.
  *Sanity anchor* — confirms the multi-hypothesis redesign does not
  break the synthetic property.

* **R-62-tightbudget** (synthetic, tight ``T_decoder``; W15-1
  anchor). Same events; ``T_decoder`` set strictly below the
  admitted union's token sum. Under FIFO packing the round-2
  disambiguator falls past the budget and the priority decoder
  elects a generic root_cause. The W15 salience packer keeps the
  disambiguator first; ``accuracy_full`` recovers. **The first
  synthetic bench in the programme that gates a decoder-side
  packing intervention as a load-bearing layer.**

* **R-62-saturation** (named falsifier; W15-Λ-budget anchor). Same
  events; ``T_decoder`` is large (≥ admitted union token sum). With
  no budget pressure, salience reordering is a no-op on the answer
  field — the W15 decoder ties FIFO-packed and unpacked W13 alike.
  Verifies the W15-1 win is *conditional* on budget pressure.

Theorem cross-reference (W15 family)
=====================================

* **W15-Λ-budget** (proved-empirical + structural sketch): on
  R-62-tightbudget under FIFO packing, every cross-round decoder ties
  FIFO at 0.000 — the round-2 disambiguator is past the budget by
  construction. Extends the W14-Λ-prompt extraction-floor argument
  to the decoder-side axis: a never-decoded handoff has no influence
  on the answer.
* **W15-1** (proved-conditional + proved-empirical): pairing the
  ``CapsuleContextPacker`` (salience-scored) with the W13 layered
  decoder achieves ``accuracy_full = 1.000`` on R-62-tightbudget
  default, +1.000 over FIFO-packed-layered, stable across 5/5
  alternate ``bank_seed`` values. The W15-1 win is *conditional* on
  (a) the bench property holding, (b) ``T_decoder`` below the union
  token sum, (c) round-2 carrying a specific-tier disambiguating
  claim with no service= token.
* **W15-2** (proved + mechanically-checked): salience score is a
  closed-form deterministic function of the handoff's bytes plus the
  elected root_cause; the pack is byte-deterministic given a
  canonical tie-break order; the FIFO packer's tie-break is the
  arrival order; the saturation falsifier verifies the no-op
  property.
* **W15-3** (proved-empirical full programme-wide regression): on
  R-54..R-61 default banks, the W15 decoder with ``T_decoder=None``
  ties the W13 layered decoder byte-for-byte on the answer field
  (the ``pack_stats`` block is purely additive).
* **W15-Λ-degenerate** (proved-empirical): on regimes without
  multi-hypothesis ambiguity (R-54..R-57; |gold_services| ≤ 2; W13
  trust-admission fallback already produces the gold-shaped admitted
  set), the W15 decoder reduces to W13. The W15-1 win is *not*
  observable on those regimes — by construction, not by failure.

CLI
---

::

    # R-62-default (synthetic, no budget; W15-3 backward-compat anchor):
    python3 -m vision_mvp.experiments.phase62_attention_aware_packing \\
        --K-auditor 12 --n-eval 8 --out -

    # R-62-tightbudget (W15-1 anchor):
    python3 -m vision_mvp.experiments.phase62_attention_aware_packing \\
        --decoder-budget 24 --K-auditor 12 --n-eval 8 --out -

    # R-62-saturation (W15-Λ-budget falsifier):
    python3 -m vision_mvp.experiments.phase62_attention_aware_packing \\
        --decoder-budget 9999 --K-auditor 12 --n-eval 8 --out -

    # 5-seed stability sweep on the tight-budget regime:
    python3 -m vision_mvp.experiments.phase62_attention_aware_packing \\
        --seed-sweep --decoder-budget 24 --n-eval 8 --out -

    # Cross-regime (default / tight / saturation):
    python3 -m vision_mvp.experiments.phase62_attention_aware_packing \\
        --cross-regime --n-eval 8 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    IncidentScenario,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.team_coord import (
    AdmissionPolicy, BundleAwareTeamDecoder,
    AttentionAwareBundleDecoder, CapsuleContextPacker,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, FifoContextPacker,
    INCIDENT_TRIAGE_OBSERVATION_KINDS,
    LayeredClaimNormalizer,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
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
)


# =============================================================================
# Phase-62 multi-hypothesis scenario family
# =============================================================================
#
# Each scenario carries 2 gold services + 2 decoy services; every service
# is mentioned by ≥ 2 distinct producer roles in round 1 (monitor +
# network) with comparable magnitudes. Round 2 carries a single
# specific-tier disambiguating claim with no ``service=`` token from a
# diagnostic role (db_admin or sysadmin).
#
# The redesign drives the admitted-union token sum up so a strict
# ``T_decoder`` bites; under FIFO packing the round-2 disambiguator —
# which arrives last in the natural round-by-round order — is the first
# casualty of truncation. The W15 packer's salience score puts the
# disambiguator FIRST, defeating that failure mode by construction.


_REMEDIATION = {
    "deadlock":             "enforce_lock_ordering_in_orders",
    "pool_exhaustion":      "raise_pool_cap_or_fix_upstream",
    "slow_query_cascade":   "index_or_split_slow_query",
    "disk_fill":            "rotate_logs_and_clear_backup",
}


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


def _build_p62_deadlock(
        decoy_a: str = "search_index",
        decoy_b: str = "metrics",
        ) -> MultiRoundScenario:
    """Phase-62 deadlock scenario.

    Asymmetric corroboration shape (W11-drop compatible):
    * Gold services A/B: 1 distinct role each (monitor only) in round 1.
    * Decoy services D_a/D_b: ≥ 2 distinct roles each (monitor + network)
      in round 1.

    Multi-hypothesis: 2 gold + 2 decoys → 4 service hypotheses survive
    naive admission. The W15 hypothesis-preserving packer keeps at
    least one round-1 mention per gold service AND drops the round-1
    decoy mentions when budget bites. After the W11 contradiction-aware
    drop, decoys (multi-role-noise-only) are removed from the answer
    and golds (single-role) survive. Without hypothesis preservation,
    salience-greedy packing picks decoy mentions over gold mentions
    (decoy corr=2 > gold corr=1) and drops golds entirely.
    """
    A, B = "orders", "payments"
    return MultiRoundScenario(
        scenario_id=(
            f"p62_deadlock_orders_payments__multi_{decoy_a}_{decoy_b}"),
        description=(
            f"Phase-62: multi-hypothesis. Gold {A}/{B} (single-role "
            f"corroboration via monitor) + decoys {decoy_a}/{decoy_b} "
            f"(cross-role corroboration via monitor + network). Round-2: "
            f"db_admin emits DEADLOCK_SUSPECTED (no service= token)."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_a,  # primary decoy for legacy fields
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4500 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2100 service={decoy_a}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={decoy_a}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=1900 service={decoy_b}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.16 service={decoy_b}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=14 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy_b}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={decoy_b}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("DEADLOCK_SUSPECTED",
                       f"deadlock relation={A}_{B}_join wait_chain=2"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_p62_pool(decoy_a: str = "archival",
                      decoy_b: str = "metrics",
                      ) -> MultiRoundScenario:
    A, B = "api", "db"
    return MultiRoundScenario(
        scenario_id=f"p62_pool_api_db__multi_{decoy_a}_{decoy_b}",
        description=(
            f"Phase-62: multi-hypothesis. Gold {A}/{B} (single-role "
            f"corroboration) + decoys {decoy_a}/{decoy_b} (cross-role "
            f"corroborated). Round-2: db_admin emits POOL_EXHAUSTION."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_a,
        gold_root_cause="pool_exhaustion",
        gold_remediation=_REMEDIATION["pool_exhaustion"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4200 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.42 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2400 service={decoy_a}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.20 service={decoy_a}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2200 service={decoy_b}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.17 service={decoy_b}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=14 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={decoy_b}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy_b}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("POOL_EXHAUSTION",
                       "pool active=200/200 waiters=145 cluster=primary"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_p62_disk(decoy_a: str = "search_index",
                      decoy_b: str = "telemetry",
                      ) -> MultiRoundScenario:
    A, B = "storage", "logs_pipeline"
    return MultiRoundScenario(
        scenario_id=f"p62_disk_storage_logs__multi_{decoy_a}_{decoy_b}",
        description=(
            f"Phase-62: multi-hypothesis. Gold {A}/{B} + decoys "
            f"{decoy_a}/{decoy_b}. Round-2: sysadmin emits "
            f"DISK_FILL_CRITICAL."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_a,
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={A}"),
                _emit("LATENCY_SPIKE", f"p95_ms=4500 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=3100 service={decoy_a}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.24 service={decoy_a}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2800 service={decoy_b}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.19 service={decoy_b}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={decoy_b}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy_b}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_SYSADMIN: (
                _emit("DISK_FILL_CRITICAL",
                       "/var/log used=99% fs=/ host=primary"),
            ),
            ROLE_DB_ADMIN: (),
        },
    )


def _build_p62_slow_query(decoy_a: str = "metrics",
                            decoy_b: str = "audit_jobs",
                            ) -> MultiRoundScenario:
    A, B = "web", "db"
    return MultiRoundScenario(
        scenario_id=f"p62_slow_query_web_db__multi_{decoy_a}_{decoy_b}",
        description=(
            f"Phase-62: multi-hypothesis. Gold {A}/{B} + decoys "
            f"{decoy_a}/{decoy_b}. Round-2: db_admin emits "
            f"SLOW_QUERY_OBSERVED."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy_a,
        gold_root_cause="slow_query_cascade",
        gold_remediation=_REMEDIATION["slow_query_cascade"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4100 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.32 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2900 service={decoy_a}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.16 service={decoy_a}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2600 service={decoy_b}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.14 service={decoy_b}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={decoy_a}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy_b}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={decoy_b}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       "q#9 mean_ms=4100 cluster=primary"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


_P62_BASE_BUILDERS = (
    _build_p62_deadlock, _build_p62_pool, _build_p62_disk,
    _build_p62_slow_query,
)
_P62_KNOWN_DECOYS = (
    "search_index", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "cache", "scratch_pool",
)


def build_phase62_bank(*, n_replicates: int = 2,
                          seed: int = 11
                          ) -> list[MultiRoundScenario]:
    """Phase-62 default bank — multi-hypothesis comparable-magnitude
    scenarios.

    Mechanically-checkable closure invariants:

    * 2 gold services + 2 distinct decoy services per scenario.
    * Every service (gold + decoy) has ≥ 2 distinct producer roles
      mentioning it in round 1 (cross-role corroboration property).
    * Round-1 emissions are exclusively generic-tier; round-2 carries
      exactly one specific-tier disambiguating claim with no
      ``service=`` token; the decoy services are not mentioned in
      round 2.
    """
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for builder in _P62_BASE_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            j = rng.randrange(0, 1 << 16)
            d_a = _P62_KNOWN_DECOYS[(i + r) % len(_P62_KNOWN_DECOYS)]
            # Pick a different decoy for the second slot.
            d_b = _P62_KNOWN_DECOYS[(j + r + 3) % len(_P62_KNOWN_DECOYS)]
            if d_b == d_a:
                d_b = _P62_KNOWN_DECOYS[
                    (j + r + 4) % len(_P62_KNOWN_DECOYS)]
            sc = builder(decoy_a=d_a, decoy_b=d_b)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


# =============================================================================
# Bench-property helpers — five named properties for R-62
# =============================================================================


_GENERIC_NOISE_AFTER_NORMALIZATION = frozenset(
    INCIDENT_TRIAGE_OBSERVATION_KINDS)


def _decoy_pair_for(sc: MultiRoundScenario) -> tuple[str, str]:
    """Extract both decoy service names from the scenario's round-1
    emissions (the bank constructor stores only the *primary* decoy on
    ``decoy_storm_service``; the second appears in payloads only)."""
    seen: list[str] = []
    gold_a, gold_b = sc.gold_services_pair
    for emissions in (sc.round1_emissions, sc.round2_emissions):
        for role, evs in emissions.items():
            for (kind, payload) in evs:
                m = re.search(r"\bservice=([\w-]+)", payload)
                if m:
                    tag = m.group(1)
                    if tag in (gold_a, gold_b):
                        continue
                    if tag not in seen:
                        seen.append(tag)
    if len(seen) >= 2:
        return (seen[0], seen[1])
    if len(seen) == 1:
        return (seen[0], seen[0])
    return ("", "")


def _bench_property(sc: MultiRoundScenario,
                     round1_cands, round2_cands,
                     ) -> dict[str, Any]:
    """Verify the five R-62 closure properties post-normalisation."""
    norm = LayeredClaimNormalizer()

    def _norm_kinds(cands):
        out = set()
        for c in cands:
            if c[1] != ROLE_AUDITOR:
                continue
            out.add(norm.normalize(c[2]))
        return out

    def _service_to_roles(cands) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for (src, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            np = normalize_payload(payload)
            m = re.search(r"\bservice=([\w-]+)", np)
            if m:
                out.setdefault(m.group(1), set()).add(src)
        return out

    r1_kinds = _norm_kinds(round1_cands)
    r2_kinds = _norm_kinds(round2_cands)
    decoy_a, decoy_b = _decoy_pair_for(sc)
    gold_a, gold_b = sc.gold_services_pair
    decoys = tuple(d for d in (decoy_a, decoy_b) if d)
    golds = (gold_a, gold_b)

    r1_svc = _service_to_roles(round1_cands)
    r2_svc = _service_to_roles(round2_cands)

    round1_only_generic = bool(r1_kinds) and r1_kinds.issubset(
        _GENERIC_NOISE_AFTER_NORMALIZATION)
    round2_only_specific = bool(r2_kinds) and not (
        r2_kinds & _GENERIC_NOISE_AFTER_NORMALIZATION)
    decoy_only_in_round1 = all(d not in r2_svc for d in decoys)
    # Asymmetric corroboration: every decoy has ≥ 2 distinct roles
    # mentioning it (the W11-noise-decoy-drop trigger condition); every
    # gold has exactly 1 distinct role (so W11 does NOT drop golds).
    # This is the load-bearing structural separator for R-62 — without
    # it, multi-hypothesis with symmetric corroboration causes W11 to
    # drop everything, which is W15-Λ-symmetric (out-of-scope for the
    # current milestone).
    all_decoys_corr = all(
        len(r1_svc.get(d, set())) >= 2 for d in decoys if d)
    all_golds_single_role = all(
        len(r1_svc.get(g, set())) == 1 for g in golds if g)
    multi_hypothesis = (len(set(decoys)) >= 2 and len(set(golds)) >= 2)

    return {
        "round1_only_generic_noise": bool(round1_only_generic),
        "round2_only_specific": bool(round2_only_specific),
        "decoy_only_in_round1": bool(decoy_only_in_round1),
        "all_decoys_round1_corroborated": bool(all_decoys_corr),
        "all_golds_single_role": bool(all_golds_single_role),
        "multi_hypothesis": bool(multi_hypothesis),
        "r62_property_holds": bool(
            round1_only_generic and round2_only_specific
            and decoy_only_in_round1 and all_decoys_corr
            and all_golds_single_role and multi_hypothesis),
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
        "decoys": list(decoys),
        "golds": list(golds),
    }


# =============================================================================
# Identity (no-filter) extractor
# =============================================================================


@dataclasses.dataclass
class IdentityExtractor:
    """Pass every emission through to the candidate stream verbatim.
    The R-62 producer side is *not* under test here — Phase-61 already
    closed that axis. Phase-62 evaluates downstream packing only, so
    the identity extractor is the right baseline."""

    seed: int = 11

    def extract_round(self,
                       sc: MultiRoundScenario,
                       round_idx: int,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        emissions = (sc.round1_emissions if round_idx == 1
                       else sc.round2_emissions)
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            role_em = emissions.get(role, ())
            for (kind, payload) in role_em:
                consumers = subs.consumers(role, kind)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
        return out


# =============================================================================
# Phase-62 driver
# =============================================================================


def _make_factory(name: str, priorities, budgets):
    """Per-strategy admission policy factory. Same shape as Phase-58/61
    drivers; all R-62 capsule strategies use FIFO admission (the
    bench gates the *decoder* axis, not the admission axis)."""

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
        if name in ("capsule_multi_service",):
            policy = (MultiServiceCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud],
                          top_k=4, min_corroborated_roles=2))
            return {r: policy for r in budgets}
        if name in ("capsule_multi_round",
                     "capsule_robust_multi_round",
                     "capsule_layered_multi_round",
                     "capsule_layered_fifo_packed",
                     "capsule_attention_aware"):
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
    return fac


def _decode_with_packer(
        union: list[_DecodedHandoff],
        packer,
        T_decoder: int | None,
        round_index_hint: list[int],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the W13 layered decoder on a packed bundle (FIFO or
    salience). Returns ``(answer, pack_stats)`` so the driver can
    record packing diagnostics regardless of the packer used.
    """
    # First-pass elect for the salience packer. FIFO ignores it.
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
    """Execute one (scenario × strategy) cell. Returns the standard
    ``StrategyResult`` plus a per-cell ``pack_stats`` block exposing
    the W15 packing diagnostics. ``pack_stats`` is empty for
    decoder_modes that do not pack."""
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase62_attention_aware_packing",
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
        # W15 — salience-aware packing.
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        # Build round-index hint by looking up each handoff's round in
        # the ledger.
        round_hint: list[int] = []
        for cid in (rv1, rv2):
            if not cid or cid not in ledger:
                continue
            cap = ledger.get(cid)
            r_idx = (cap.payload.get("round", 0)
                       if isinstance(cap.payload, dict) else 0)
            for p in cap.parents:
                if p in ledger and ledger.get(p).kind == CapsuleKind.TEAM_HANDOFF:
                    round_hint.append(int(r_idx) or 1)
        # `collect_admitted_handoffs` deduplicates — round_hint should
        # be aligned with the deduplicated union. Re-derive cleanly:
        round_hint = []
        seen: set[str] = set()
        for cid in (rv1, rv2):
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
                round_hint.append(int(r_idx) or 1)
        answer, pack_stats = _decode_with_packer(
            union, CapsuleContextPacker(),
            T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "fifo_packed_layered":
        # W13 + FIFO pack — the load-bearing baseline for R-62.
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = []
        seen = set()
        for cid in (rv1, rv2):
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
                round_hint.append(int(r_idx) or 1)
        answer, pack_stats = _decode_with_packer(
            union, FifoContextPacker(),
            T_decoder=T_decoder,
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
                source_role=src,
                source_agent_id=ALL_ROLES.index(src),
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


# Strategy/decoder-mode pairs evaluated on every scenario.
_R62_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("capsule_fifo", "per_round"),
    ("capsule_priority", "per_round"),
    ("capsule_coverage", "per_round"),
    ("capsule_cohort_buffered", "per_round"),
    ("capsule_corroboration", "per_round"),
    ("capsule_multi_service", "per_round"),
    ("capsule_multi_round", "multi_round_bundle"),
    ("capsule_robust_multi_round", "robust_multi_round"),
    ("capsule_layered_multi_round", "layered_multi_round"),
    # W15 baseline: W13 + FIFO pack under T_decoder.
    ("capsule_layered_fifo_packed", "fifo_packed_layered"),
    # W15 method: W13 + salience pack under T_decoder.
    ("capsule_attention_aware", "attention_aware"),
)


def run_phase62(*,
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
    """Drive Phase 62 end-to-end.

    Parameters
    ----------
    T_decoder
        Strict decoder-side token budget enforced over the cross-round
        bundle. ``None`` disables (W15-3 backward-compat sanity
        anchor); a finite value enables salience-aware packing for the
        ``capsule_attention_aware`` strategy and FIFO packing for
        ``capsule_layered_fifo_packed``.

    Pre-committed default config (R-62-default sanity anchor):
    ``K_auditor=12``, ``T_auditor=256``, ``T_decoder=None`` (no
    budget pressure), ``n_eval=8`` (4 builders × 2 replicates),
    ``bank_seed=11``.
    """
    bank = build_phase62_bank(n_replicates=bank_replicates,
                                seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    extractor_obj = IdentityExtractor(seed=bank_seed)

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R62_STRATEGIES
    }
    for sc in bank:
        round1_cands = extractor_obj.extract_round(sc, 1)
        round2_cands = extractor_obj.extract_round(sc, 2)
        bench_property_per_scenario[sc.scenario_id] = _bench_property(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R62_STRATEGIES:
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

    strategy_names = ("substrate",) + tuple(s[0] for s in _R62_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        # The W15-1 strict-gain claim:
        "attention_minus_fifo_packed": gap(
            "capsule_attention_aware", "capsule_layered_fifo_packed"),
        # vs unpacked layered (sanity for W15-3 / W15-Λ-budget):
        "attention_minus_layered": gap(
            "capsule_attention_aware", "capsule_layered_multi_round"),
        # vs substrate FIFO:
        "attention_minus_fifo": gap(
            "capsule_attention_aware", "capsule_fifo"),
        # baseline gap (the failure mode being measured):
        "fifo_packed_minus_layered": gap(
            "capsule_layered_fifo_packed",
            "capsule_layered_multi_round"),
        "max_non_attention_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_attention_aware"),
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
        "scenarios_with_property": sum(
            1 for v in bench_property_per_scenario.values()
            if v["r62_property_holds"]),
        "scenarios_with_multi_hypothesis": sum(
            1 for v in bench_property_per_scenario.values()
            if v["multi_hypothesis"]),
        "scenarios_with_all_decoys_corr": sum(
            1 for v in bench_property_per_scenario.values()
            if v["all_decoys_round1_corroborated"]),
        "scenarios_with_all_golds_single_role": sum(
            1 for v in bench_property_per_scenario.values()
            if v["all_golds_single_role"]),
        "K_auditor": K_auditor,
        "T_decoder": T_decoder,
    }

    # Pack-statistics aggregation.
    def _agg_packstats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        s_in = sum(r.get("tokens_input", 0) for r in rows)
        s_kept = sum(r.get("tokens_kept", 0) for r in rows)
        s_drop = sum(r.get("n_dropped_budget", 0) for r in rows)
        s_h_in = sum(r.get("n_handoffs_admitted", 0) for r in rows)
        s_h_kept = sum(r.get("n_handoffs_decoder_input", 0) for r in rows)
        # Pos of first causal claim, averaged when available (-1 = absent).
        pos = [r.get("position_of_first_causal_claim", -1) for r in rows]
        n_have = sum(1 for p in pos if p >= 0)
        avg_pos = (sum(p for p in pos if p >= 0) / max(1, n_have)
                    if n_have > 0 else -1.0)
        # Hypothesis count averages.
        hin = [r.get("hypothesis_count_input", 0) for r in rows]
        hkept = [r.get("hypothesis_count_kept", 0) for r in rows]
        return {
            "n_cells": n,
            "tokens_input_sum": int(s_in),
            "tokens_kept_sum": int(s_kept),
            "n_dropped_budget_sum": int(s_drop),
            "tokens_kept_over_input": (round(s_kept / s_in, 4)
                                          if s_in > 0 else 0.0),
            "handoffs_admitted_sum": int(s_h_in),
            "handoffs_decoder_input_sum": int(s_h_kept),
            "fraction_handoffs_kept": (round(s_h_kept / s_h_in, 4)
                                          if s_h_in > 0 else 0.0),
            "position_of_first_causal_claim_avg": round(
                float(avg_pos), 4),
            "n_with_causal_claim_kept": int(n_have),
            "hypothesis_count_input_avg": round(
                sum(hin) / n, 4) if n else 0.0,
            "hypothesis_count_kept_avg": round(
                sum(hkept) / n, 4) if n else 0.0,
        }

    pack_stats_summary = {
        s: _agg_packstats(pack_stats_per_strategy.get(s, []))
        for s in ("capsule_layered_fifo_packed",
                   "capsule_attention_aware")
    }

    if verbose:
        print(f"[phase62] T_decoder={T_decoder}, n_eval={len(bank)}, "
              f"K_auditor={K_auditor}",
              file=sys.stderr, flush=True)
        print(f"[phase62] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase62]   {s:32s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase62] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)

    out: dict[str, Any] = {
        "schema": "phase62.attention_aware_packing.v1",
        "config": {
            "n_eval": len(bank), "K_auditor": K_auditor,
            "T_auditor": T_auditor, "K_producer": K_producer,
            "T_producer": T_producer, "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "T_decoder": T_decoder,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "pack_stats_summary": pack_stats_summary,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }
    return out


def run_phase62_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 8, K_auditor: int = 12, T_auditor: int = 256,
        T_decoder: int | None = 24,
        ) -> dict[str, Any]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase62(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=2,
            T_decoder=T_decoder, verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
            "pack_stats_summary": rep["pack_stats_summary"],
        }
    return {
        "schema": "phase62.attention_aware_packing_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor, "T_auditor": T_auditor,
        "T_decoder": T_decoder, "n_eval": n_eval,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 8, bank_seed: int = 11,
                                K_auditor: int = 12,
                                T_auditor: int = 256,
                                T_decoder_tight: int = 24,
                                T_decoder_saturation: int = 9999,
                                ) -> dict[str, Any]:
    """Single regime table comparing R-62's three sub-banks.

    Reports the headline accuracy_full per (T_decoder regime) so the
    W15-Λ-budget / W15-1 / W15-3 separation is read off one table.
    """
    out: dict[str, Any] = {
        "schema": "phase62.cross_regime.v1",
        "config": {"n_eval": n_eval, "bank_seed": bank_seed,
                    "K_auditor": K_auditor, "T_auditor": T_auditor,
                    "T_decoder_tight": T_decoder_tight,
                    "T_decoder_saturation": T_decoder_saturation},
    }
    out["r62_default"] = run_phase62(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=None, verbose=False)
    out["r62_tightbudget"] = run_phase62(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=T_decoder_tight, verbose=False)
    out["r62_saturation"] = run_phase62(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=T_decoder_saturation, verbose=False)
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 62 — attention-aware capsule context packing "
                     "(SDK v3.16 / W15 family).")
    p.add_argument("--K-auditor", type=int, default=12)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--decoder-budget", type=int, default=-1,
                    help="Strict T_decoder budget. -1 = None (no "
                          "budget pressure, W15-3 sanity anchor).")
    p.add_argument("--cross-regime", action="store_true")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    T_dec = None if args.decoder_budget < 0 else int(args.decoder_budget)
    if args.cross_regime:
        report = run_cross_regime_summary(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor)
    elif args.seed_sweep:
        report = run_phase62_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            T_decoder=(T_dec if T_dec is not None else 24))
    else:
        report = run_phase62(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            T_decoder=T_dec,
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
            print(f"[phase62] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
