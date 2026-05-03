"""Phase 66 — bundle-contradiction-aware trust-weighted disambiguation
under deceptive / confounded round-2 evidence (SDK v3.20, W19 family
anchor).

The follow-up to SDK v3.19 (W18) on the named research frontier the
W18 milestone explicitly left conjectural / open: the **deceptive-
ambiguity wall** (W18-Λ-deceive) and the **confound wall**
(W18-Λ-confound).

The W18-Λ-deceive anchor (Phase-65 R-65-DECEIVE) proves that *any*
closed-form bundle-relational scorer that **trusts** the round-2
disambiguator's payload fails when the payload mentions decoy service
tags but NOT gold (an adversarial relational signal). The W18-Λ-confound
anchor (Phase-65 R-65-CONFOUND) proves that the same scorer abstains
when the round-2 payload mentions BOTH gold AND decoy tags
(symmetric relational signal), falling through to the W15 answer which
itself ties FIFO at 0.000 by W17-Λ-symmetric.

SDK v3.19 named the next research move: a *richer disambiguator* that
consumes information *beyond* the round-2 disambiguator's payload
itself — specifically, the *consistency between* the primary
disambiguator and *independent asymmetric witnesses* elsewhere in
the bundle. W19's :class:`BundleContradictionDisambiguator` is the
smallest move in that direction.

Phase-66 is the named harder regime where the W19 method's required
ingredient — at least one *independent asymmetric witness* in the
admitted union (a specific-tier handoff OTHER than the canonical
primary disambiguator) — is consistently present across the
*positive-anchor* and the two *resolvable-deception* sub-banks
(R-66-CORROBORATED, R-66-DECEIVE-NAIVE, R-66-CONFOUND-RESOLVABLE)
AND where two named falsifier regimes (R-66-DECEIVE-TOTAL,
R-66-OUTSIDE-REQUIRED) make the W19 conditionality sharp.

Recap of why the W18-Λ-deceive wall stands without W19
========================================================

On R-65-DECEIVE the bench property is:

  * Both gold A/B AND the decoy mentioned by ≥ 2 distinct routed
    producer roles in round-1 via generic-noise kinds with comparable
    magnitudes (W17-Λ-symmetric *applies*).
  * Round-2 carries one specific-tier disambiguator with a payload
    text that names DECOY service tags (e.g. ``relation=cache_cache_join``)
    but NOT gold.

Under this regime:

  * W17-Λ-symmetric extends verbatim to R-65-DECEIVE for every
    closed-form salience packer (W11..W17).
  * W18 trusts the round-2 evidence: its strict-asymmetric branch
    fires (decoy has positive score; gold has zero score) and it
    projects the answer to the decoy-only set; ``services_correct``
    fails by construction at 0.000.
  * **No closed-form bundle-relational scorer that** *trusts* **its
    evidence can escape this regime** without an additional
    information channel.

R-65-CONFOUND is the symmetric companion: round-2 names BOTH gold
AND decoy. W18's scorer is non-zero on every admitted tag; W18
abstains; the W15 inner answer is empty by W11 contradiction-aware
drop firing symmetrically; ties FIFO at 0.000.

Phase-66 sub-banks
==================

Five pre-committed cells plus a cross-regime synthetic summary:

* **R-66-CORROBORATED** (positive sanity anchor; synthetic identity
  producer, ``T_decoder=None``). Same R-65-COMPAT round-1 symmetric-
  corroboration shape PLUS:
    - Primary round-2 disambiguator (db_admin / sysadmin) emits the
      same gold-only relational compound as R-65-COMPAT.
    - Secondary round-2 witness (monitor) emits a specific-tier
      synonym kind (``DEADLOCK_DETECTED``, ``POOL_EXHAUSTED``,
      ``DISK_FILL_DETECTED``, ``SLOW_QUERY_DETECTED``) with a
      gold-only relational compound payload.
  Both W18 (via primary alone) AND W19 (via primary + secondary
  consistency) recover gold; W18 = W19 = 1.000. R-66-3 backward-
  compat anchor.
* **R-66-DECEIVE-NAIVE** (W19-1-deceive anchor; synthetic identity
  producer, ``T_decoder=None``). Same R-65-DECEIVE round-2 primary
  payload (mentions decoy via ``relation={decoy}_{decoy}_*``) PLUS:
    - Secondary round-2 witness (monitor) emits a specific-tier
      synonym kind with a gold-only relational compound.
  W18's full-disambiguator scorer sees positive scores on every
  admitted tag (primary contributes decoy hits; secondary contributes
  gold hits); W18 abstains; W15 inner is empty; ties FIFO at 0.000.
  W19's witness counter excludes the primary; aw(gold) > aw(decoy)
  via the secondary; W19 fires the confound-resolved branch and
  projects to gold. **+1.000 strict separation.**
* **R-66-CONFOUND-RESOLVABLE** (W19-1-confound anchor; synthetic
  identity producer, ``T_decoder=None``). Same R-65-CONFOUND round-2
  primary payload (mentions all three) PLUS:
    - Secondary round-2 witness (monitor) emits a specific-tier
      synonym kind with a gold-only relational compound.
  W18 abstains (full-set hit); W15 inner ties FIFO. W19's witness
  counter sees aw(gold) > aw(decoy) and fires the confound-resolved
  branch; projects to gold. **+1.000 strict separation.**
* **R-66-DECEIVE-NAIVE-TIGHT** (W19-1 + W15 composition anchor;
  synthetic identity producer, ``T_decoder=24``). Same R-66-DECEIVE-
  NAIVE shape under decoder-side budget pressure. The W19 scorer
  reads only the W15-packed bundle; the W15 packer must keep BOTH
  the primary disambiguator AND the secondary witness within
  ``T_decoder``. Token-budget honesty preserved byte-for-byte
  relative to W15.
* **R-66-DECEIVE-TOTAL** (W19-Λ-total falsifier; synthetic,
  ``T_decoder=None``). Same R-65-DECEIVE primary payload (mentions
  decoy) and *no* secondary witness. The bundle carries no
  independent asymmetric witness; W19 cannot detect deception.
  W19 trusts primary (W19_BRANCH_PRIMARY_TRUSTED) and FAILS at
  0.000 — same as W18 on R-65-DECEIVE. Names the structural limit
  when the bundle is exhausted of asymmetric signal.
* **R-66-OUTSIDE-REQUIRED** (W19-Λ-outside falsifier; synthetic,
  ``T_decoder=None``). Same R-65-DECEIVE primary AND a *symmetric*
  secondary witness that mentions BOTH gold AND decoy (so aw is
  uniform across all three admitted tags). W19 abstains
  (W19_BRANCH_ABSTAINED_SYMMETRIC); W15 inner is empty; ties FIFO
  at 0.000. Names the structural limit when bundle-internal
  contradiction is itself symmetric — the natural escape requires
  outside information (W19-C-OUTSIDE, conjectural).

Theorem family W19 (minted by this milestone)
==============================================

* **W19-Λ-deceive (extension)** (proved-empirical + structural
  sketch). W18-Λ-deceive extends to R-66-DECEIVE-NAIVE for every
  *trusting* closed-form bundle-relational scorer. W18's behaviour
  on R-66-DECEIVE-NAIVE is *abstention* (full-set hit) rather than
  the R-65-DECEIVE strict-asymmetric pick of decoy — both yield
  ``accuracy_full = 0.000`` on the answer field; the W19 strict
  gain is on this *combined* "deception that the bundle's secondary
  witnesses contradict" regime.
* **W19-1** (proved-conditional + proved-empirical n=8 saturated × 5
  seeds × 3 deceptive regimes). Pairing the W18
  ``RelationalCompatibilityDisambiguator`` with the W19
  :class:`BundleContradictionDisambiguator` strictly improves
  ``accuracy_full`` over the strongest non-W19 capsule baseline by
  ``+1.000`` on R-66-DECEIVE-NAIVE AND on R-66-CONFOUND-RESOLVABLE
  (both at ``T_decoder = None``) AND on R-66-DECEIVE-NAIVE-TIGHT
  (``T_decoder = 24``), stable across 5/5 alternate ``bank_seed``
  values. The first capsule-native multi-agent-coordination method
  that crosses the deceptive-ambiguity wall on regimes where the
  bundle carries an independent asymmetric witness for gold.
* **W19-2** (proved by inspection + mechanically-checked). The W19
  witness counter is deterministic and byte-stable: ``decode_rounds``
  returns byte-for-byte identical answers given byte-identical
  inputs; the per-tag witness counts depend only on the
  ``(source_role, claim_kind, payload_sha)`` triples in the union
  and the canonical primary's index.
* **W19-3** (proved-empirical full programme regression). On
  R-54..R-65 default banks the W19 method ties the W18 method
  byte-for-byte on the answer field — either via no-witness-no-
  inversion (W19 reduces to W18 when aw is uniformly zero) or via
  primary-trusted (W19 trusts W18's strict-asymmetric pick when
  witnesses agree with primary).
* **W19-Λ-total** (proved-empirical n=8 saturated). On
  R-66-DECEIVE-TOTAL, every cell's bundle carries no independent
  asymmetric witness; W19 reduces to W18 and FAILS at 0.000. Names
  the structural limit when the bundle is exhausted of asymmetric
  signal — escape requires outside information.
* **W19-Λ-outside** (proved-empirical n=8 saturated). On
  R-66-OUTSIDE-REQUIRED, every cell's bundle carries witnesses but
  the witness count is *symmetric* between primary's named set and
  the complement; W19 abstains and ties FIFO at 0.000. Names the
  structural limit when bundle-internal contradiction is itself
  symmetric — escape requires outside information.
* **W19-C-LEARNED** (conjectural). A small distilled trust scorer
  (e.g. logistic regression over per-tag witness-count features
  plus character-bigram bundle features) outperforms the closed-
  form rule on a held-out cross-bench where the LLM emits free-form
  asymmetric witnesses outside the synthetic bench's exact-match
  closure.
* **W19-C-OUTSIDE** (conjectural). A scorer with access to an
  outside-information axis (e.g. service-graph topology, prior
  reliability scores, cross-incident historical evidence) can
  detect the W19-Λ-outside regime by cross-referencing the
  symmetric witness pattern against an outside source. The natural
  escape from the bundle-only closed-form scope.
* **W19-Λ-real** (proved-conditional + empirical-research,
  conjectural-empirical-on-Mac-1). Real-LLM transfer of the
  asymmetric-witness convention: the LLM must emit the secondary
  witness in the same closed-vocabulary form the synthetic bench
  uses (synonym specific-tier kinds + relational-compound payloads).
  If the LLM emits free-form natural-language witnesses, the W19
  exact-match layer misses by construction.
* **W19-C-CROSS-BENCH** (conjectural). The W19 method generalises
  to non-incident-triage benchmark families when the family admits
  a closed-vocabulary specific-tier set AND the round-2 disambiguator
  has at least one independent specific-tier witness in the union.

CLI
---

::

    # R-66-CORROBORATED (W19-3 + W18 ratification anchor):
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
        --bank corroborated --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-66-DECEIVE-NAIVE (W19-1 deceive anchor):
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
        --bank deceive_naive --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-66-CONFOUND-RESOLVABLE (W19-1 confound anchor):
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
        --bank confound_resolvable --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-66-DECEIVE-NAIVE-TIGHT (W19-1 with budget):
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
        --bank deceive_naive --decoder-budget 24 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-66-DECEIVE-TOTAL (W19-Λ-total falsifier):
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
        --bank deceive_total --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # R-66-OUTSIDE-REQUIRED (W19-Λ-outside falsifier):
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
        --bank outside_required --decoder-budget -1 \\
        --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase66_deceptive_ambiguity \\
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
    BundleContradictionDisambiguator,
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


# Phase-66 secondary-witness routing extension. The canonical incident-
# triage subscription table only routes specific-tier kinds from the
# *canonical* producer roles (db_admin → DEADLOCK_SUSPECTED, sysadmin →
# DISK_FILL_CRITICAL, etc.). The W19 secondary witness is, by design, a
# specific-tier emission from a *different* producer role (typically
# monitor), so it has no subscription consumer in the canonical table.
#
# This helper extends the canonical ``_build_round_candidates`` output
# with a small *secondary-witness* routing table that admits the
# pre-committed Phase-66 secondary kinds → auditor. The extension is
# closed-vocabulary, deterministic, and has no effect outside Phase-66
# (the canonical kinds it allows are ones Phase-66 emits explicitly,
# and the synonym-resolution layer in W12 / W13 normalises them to
# their canonical specific-tier kinds before W18 / W19 score them).

_P66_SECONDARY_ROUTES: tuple[tuple[str, str], ...] = (
    # (source_role, claim_kind) — every (role, kind) the Phase-66
    # secondary witness emits gets a one-line route to auditor.
    (ROLE_MONITOR, "DEADLOCK_DETECTED"),
    (ROLE_MONITOR, "POOL_EXHAUSTED"),
    (ROLE_MONITOR, "DISK_FILL_DETECTED"),
    (ROLE_MONITOR, "SLOW_QUERY_DETECTED"),
)


def _build_round_candidates_p66(
        emissions: dict[str, tuple[tuple[str, str], ...]],
        ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
    """Phase-66 round-candidates builder.

    Same shape as
    ``vision_mvp.experiments.phase58_multi_round_decoder._build_round_candidates``,
    but additionally routes the pre-committed Phase-66 secondary
    witness kinds (``_P66_SECONDARY_ROUTES``) to the auditor when
    they appear in the emission stream. The canonical subscription
    table is left byte-for-byte unchanged; the augmentation is
    Phase-66-local.
    """
    out = list(_build_round_candidates(emissions))
    seen: set[tuple[str, str, str, str]] = set()
    for (s, t, k, p, _e) in out:
        seen.add((s, t, k, p))
    for (src_role, kind) in _P66_SECONDARY_ROUTES:
        for (k_em, payload) in emissions.get(src_role, ()):
            if k_em != kind:
                continue
            key = (src_role, ROLE_AUDITOR, kind, payload)
            if key in seen:
                continue
            seen.add(key)
            out.append((src_role, ROLE_AUDITOR, kind, payload, (0,)))
    return out


# =============================================================================
# Phase-66 scenario family
# =============================================================================
#
# Symmetric-corroboration shape (same as Phase-64-SYM / Phase-65) PLUS a
# *primary* round-2 specific-tier disambiguator with a relational compound
# payload AND a *secondary* round-2 specific-tier witness (under a synonym
# kind) from a different producer role. The five sub-banks
# (CORROBORATED / DECEIVE_NAIVE / CONFOUND_RESOLVABLE / DECEIVE_TOTAL /
# OUTSIDE_REQUIRED) vary the primary's relational-compound payload AND
# the presence/shape of the secondary witness.


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
    """Build the round-1 emissions shape that mirrors R-64-SYM /
    R-65: monitor + network each emit four generic-noise mentions
    (two for gold A/B, two for the decoy) with comparable magnitudes
    — so the bipartite ``(role × tag, kind, magnitude)`` multiset is
    identical for gold and decoy after subscription-routing."""
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


# =============================================================================
# Round-2 PRIMARY disambiguator templates (per scenario family ×
# sub-bank). The primary is the *canonical* W18 disambiguator —
# highest-priority specific-tier handoff in canonical sort order.
# =============================================================================

# CORROBORATED: primary names gold only (same as R-65-COMPAT).
_R2_PRIMARY_CORROBORATED = {
    "deadlock":           "deadlock relation={A}_{B}_join wait_chain=2",
    "pool_exhaustion":    ("pool active=200/200 waiters=145 cluster=primary "
                            "pool_chain={A}_{B} exhaustion=upstream_{A}_to_{B}"),
    "disk_fill":          ("/var/log used=99% fs=/ host=primary "
                            "mount=/storage/{A}/{B}/var/log"),
    "slow_query_cascade": ("q#9 mean_ms=4100 cluster=primary "
                            "query_path=svc_{A}_then_svc_{B}"),
}

# DECEIVE_NAIVE: primary names decoy only (same as R-65-DECEIVE).
_R2_PRIMARY_DECEIVE = {
    "deadlock":           "deadlock relation={decoy}_{decoy}_join wait_chain=2",
    "pool_exhaustion":    ("pool active=200/200 waiters=145 cluster=primary "
                            "pool_chain={decoy}_{decoy}"),
    # Avoid the literal ``/storage/`` path component (same reason as
    # R-65-DECEIVE): when the gold A tag for disk_fill is ``storage``,
    # ``/storage/...`` would re-introduce a gold mention and break the
    # clean DECEIVE shape.
    "disk_fill":          ("/var/log used=99% fs=/ host=primary "
                            "mount=/srv/{decoy}/{decoy}/var/log"),
    "slow_query_cascade": ("q#9 mean_ms=4100 cluster=primary "
                            "query_path=svc_{decoy}_then_svc_{decoy}"),
}

# CONFOUND_RESOLVABLE: primary names all three (same as R-65-CONFOUND).
_R2_PRIMARY_CONFOUND = {
    "deadlock":           "deadlock relation={A}_{B}_{decoy}_join wait_chain=2",
    "pool_exhaustion":    ("pool active=200/200 waiters=145 cluster=primary "
                            "pool_chain={A}_{B}_{decoy}"),
    "disk_fill":          ("/var/log used=99% fs=/ host=primary "
                            "mount=/storage/{A}/{B}/{decoy}/var/log"),
    "slow_query_cascade": ("q#9 mean_ms=4100 cluster=primary "
                            "query_path=svc_{A}_then_svc_{B}_via_svc_{decoy}"),
}


# =============================================================================
# Round-2 SECONDARY witness templates. The secondary is a
# *specific-tier synonym kind* (DEADLOCK_DETECTED / POOL_EXHAUSTED /
# DISK_FILL_DETECTED / SLOW_QUERY_DETECTED — all in CLAIM_KIND_SYNONYMS
# and resolving to the same canonical specific-tier kind as the primary)
# emitted by the *monitor* role (different from the primary's role —
# db_admin or sysadmin). This makes the secondary an *independent*
# specific-tier witness whose payload mentions service tags
# asymmetrically across the candidate set.
# =============================================================================

# CORROBORATED: secondary also names gold only.
_R2_SECONDARY_CORROBORATED = {
    "deadlock":           "deadlock relation={A}_{B}_join detected_at=t=120",
    "pool_exhaustion":    ("pool exhaustion observed pool_chain={A}_{B} "
                            "queue_depth=145"),
    "disk_fill":          ("/var/log saturated mount=/storage/{A}/{B}/log "
                            "remaining_bytes=128"),
    "slow_query_cascade": ("q#9 traced query_path=svc_{A}_then_svc_{B} "
                            "p99_ms=4500"),
}

# DECEIVE_NAIVE: secondary names gold only (CONTRADICTS the primary's
# decoy-only mention — this is the load-bearing W19 witness).
_R2_SECONDARY_DECEIVE_NAIVE = _R2_SECONDARY_CORROBORATED  # gold only

# CONFOUND_RESOLVABLE: secondary names gold only (RESOLVES the
# primary's confound — the W19 witness disambiguates by counting).
_R2_SECONDARY_CONFOUND_RESOLVABLE = _R2_SECONDARY_CORROBORATED  # gold only

# DECEIVE_TOTAL: NO secondary (the W19-Λ-total falsifier).
_R2_SECONDARY_DECEIVE_TOTAL: dict[str, str] | None = None

# OUTSIDE_REQUIRED: secondary names ALL three (gold AND decoy; symmetric
# witness pattern — the W19-Λ-outside falsifier).
_R2_SECONDARY_OUTSIDE_REQUIRED = {
    "deadlock":           ("deadlock relation={A}_{B}_{decoy}_join "
                            "detected_at=t=120"),
    "pool_exhaustion":    ("pool exhaustion observed "
                            "pool_chain={A}_{B}_{decoy} queue_depth=145"),
    "disk_fill":          ("/var/log saturated "
                            "mount=/storage/{A}/{B}/{decoy}/log "
                            "remaining_bytes=128"),
    "slow_query_cascade": ("q#9 traced "
                            "query_path=svc_{A}_then_svc_{B}_via_svc_{decoy} "
                            "p99_ms=4500"),
}


def _r2_primary_kind(root_cause: str) -> str:
    return {
        "deadlock":           "DEADLOCK_SUSPECTED",
        "pool_exhaustion":    "POOL_EXHAUSTION",
        "disk_fill":          "DISK_FILL_CRITICAL",
        "slow_query_cascade": "SLOW_QUERY_OBSERVED",
    }[root_cause]


def _r2_primary_role(root_cause: str) -> str:
    # Primary uses db_admin (or sysadmin for disk) — same as
    # R-58 / R-65.
    return {
        "deadlock":           ROLE_DB_ADMIN,
        "pool_exhaustion":    ROLE_DB_ADMIN,
        "disk_fill":          ROLE_SYSADMIN,
        "slow_query_cascade": ROLE_DB_ADMIN,
    }[root_cause]


def _r2_secondary_kind(root_cause: str) -> str:
    # Secondary uses synonym specific-tier kinds in
    # CLAIM_KIND_SYNONYMS — they resolve to the same canonical
    # specific-tier kind as the primary's, so the priority decoder's
    # elected root_cause is unchanged.
    return {
        "deadlock":           "DEADLOCK_DETECTED",
        "pool_exhaustion":    "POOL_EXHAUSTED",
        "disk_fill":          "DISK_FILL_DETECTED",
        "slow_query_cascade": "SLOW_QUERY_DETECTED",
    }[root_cause]


def _r2_secondary_role(root_cause: str) -> str:
    # Secondary always uses monitor — distinct from the primary's
    # role (db_admin / sysadmin). This makes the secondary an
    # *independent* asymmetric witness in the bipartite (role,
    # claim_kind) bundle.
    return ROLE_MONITOR


_BANK_PRIMARY_TEMPLATES = {
    "corroborated":         _R2_PRIMARY_CORROBORATED,
    "deceive_naive":        _R2_PRIMARY_DECEIVE,
    "confound_resolvable":  _R2_PRIMARY_CONFOUND,
    "deceive_total":        _R2_PRIMARY_DECEIVE,
    "outside_required":     _R2_PRIMARY_DECEIVE,
}

_BANK_SECONDARY_TEMPLATES: dict[str, dict[str, str] | None] = {
    "corroborated":         _R2_SECONDARY_CORROBORATED,
    "deceive_naive":        _R2_SECONDARY_DECEIVE_NAIVE,
    "confound_resolvable":  _R2_SECONDARY_CONFOUND_RESOLVABLE,
    "deceive_total":        _R2_SECONDARY_DECEIVE_TOTAL,
    "outside_required":     _R2_SECONDARY_OUTSIDE_REQUIRED,
}


def _build_p66_scenario(*, root_cause: str, A: str, B: str, decoy: str,
                            bank: str) -> MultiRoundScenario:
    if bank not in _BANK_PRIMARY_TEMPLATES:
        raise ValueError(f"unknown bank {bank!r}")
    primary_template = _BANK_PRIMARY_TEMPLATES[bank][root_cause]
    primary_payload = primary_template.format(A=A, B=B, decoy=decoy)
    primary_kind = _r2_primary_kind(root_cause)
    primary_role = _r2_primary_role(root_cause)

    secondary_templates = _BANK_SECONDARY_TEMPLATES[bank]
    secondary_payload: str | None = None
    secondary_kind: str | None = None
    secondary_role: str | None = None
    if secondary_templates is not None:
        secondary_payload = secondary_templates[root_cause].format(
            A=A, B=B, decoy=decoy)
        secondary_kind = _r2_secondary_kind(root_cause)
        secondary_role = _r2_secondary_role(root_cause)

    sid = f"p66{bank}_{root_cause}_{A}_{B}__sym_{decoy}"
    description = (
        f"Phase-66-{bank}: symmetric-corroboration round-1. "
        f"Primary {primary_kind} from {primary_role}: {primary_payload!r}. "
        + (f"Secondary {secondary_kind} from {secondary_role}: "
           f"{secondary_payload!r}." if secondary_payload else
           "No secondary witness (W19-Λ-total)."))

    round2: dict[str, tuple[tuple[str, str], ...]] = {
        ROLE_MONITOR: (), ROLE_NETWORK: (),
        ROLE_DB_ADMIN: (), ROLE_SYSADMIN: ()
    }
    primary_emissions: list[tuple[str, str]] = [
        _emit(primary_kind, primary_payload)]
    secondary_emissions: list[tuple[str, str]] = []
    if secondary_payload is not None and secondary_kind is not None:
        secondary_emissions.append(_emit(secondary_kind, secondary_payload))

    if primary_role == secondary_role:
        round2[primary_role] = tuple(primary_emissions + secondary_emissions)
    else:
        round2[primary_role] = tuple(primary_emissions)
        if secondary_role is not None:
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


_P66_FAMILIES = (
    ("deadlock",           "orders",  "payments"),
    ("pool_exhaustion",    "api",     "db"),
    ("disk_fill",          "storage", "logs_pipeline"),
    ("slow_query_cascade", "web",     "db_query"),
)
_P66_DECOYS = (
    "search_index", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "cache", "scratch_pool",
)


def build_phase66_bank(*, bank: str, n_replicates: int = 2,
                          seed: int = 11
                          ) -> list[MultiRoundScenario]:
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for (root_cause, A, B) in _P66_FAMILIES:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = _P66_DECOYS[(i + r) % len(_P66_DECOYS)]
            sc = _build_p66_scenario(
                root_cause=root_cause, A=A, B=B, decoy=chosen, bank=bank)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


# =============================================================================
# Bench-property witnesses
# =============================================================================


def _bench_property_p66(sc: MultiRoundScenario,
                          round1_cands, round2_cands,
                          ) -> dict[str, Any]:
    """Mechanically-verified Phase-66 bench property witnesses.

    Symmetric round-1 corroboration AND named primary-vs-secondary
    relational-mention pattern in round-2.
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

    # Round-2: classify every routed handoff as primary vs secondary.
    # Primary = highest-priority specific-tier kind; Secondary =
    # same canonical kind from a different producer role (synonym).
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

    # Compute the named bank-shape witness:
    #   corroborated         => (gold_only, gold_only)
    #   deceive_naive        => (decoy_only, gold_only)
    #   confound_resolvable  => (all_three, gold_only)
    #   deceive_total        => (decoy_only, absent)
    #   outside_required     => (decoy_only, all_three)
    shape = (primary_class, secondary_class)

    return {
        "gold_a_role_count": len(A_roles),
        "gold_b_role_count": len(B_roles),
        "decoy_role_count": len(decoy_roles),
        "symmetric_corroboration_holds": sym_corr,
        "primary_class": primary_class,
        "secondary_class": secondary_class,
        "shape": list(shape),
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
    }


_BANK_EXPECTED_SHAPE: dict[str, tuple[str, str]] = {
    "corroborated":         ("gold_only", "gold_only"),
    "deceive_naive":        ("decoy_only", "gold_only"),
    "confound_resolvable":  ("all_three", "gold_only"),
    "deceive_total":        ("decoy_only", "absent"),
    "outside_required":     ("decoy_only", "all_three"),
}


# =============================================================================
# Strategy / decoder dispatch
# =============================================================================


_R66_STRATEGIES: tuple[tuple[str, str], ...] = (
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
    # SDK v3.20 — W19 anchor: bundle-contradiction-aware trust-weighted
    # disambiguator.
    ("capsule_bundle_contradiction", "bundle_contradiction"),
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
                     "capsule_bundle_contradiction"):
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
    """Run the W18 disambiguator. Composes the W15 attention-aware
    decoder + the W18 relational-compatibility projection. Returns
    the same shape as :func:`_decode_with_packer` plus a
    ``compatibility`` block in the answer dict."""
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
    """Run the W19 bundle-contradiction-aware disambiguator.
    Composes the W15 attention-aware decoder + the W18 relational-
    compatibility projection + the W19 trust-weighted refinement.
    Returns the same shape as :func:`_decode_with_packer` plus a
    ``compatibility`` block AND a ``trust`` block in the answer dict.
    """
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
        team_tag="phase66_deceptive_ambiguity",
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
# Phase 66 driver
# =============================================================================


_VALID_BANKS = ("corroborated", "deceive_naive", "confound_resolvable",
                  "deceive_total", "outside_required")


def run_phase66(*,
                  bank: str = "deceive_naive",
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
    """Drive Phase 66 over one of {corroborated, deceive_naive,
    confound_resolvable, deceive_total, outside_required}."""
    if bank not in _VALID_BANKS:
        raise ValueError(
            f"unknown bank {bank!r}; valid: {_VALID_BANKS}")
    bank_obj = build_phase66_bank(
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
        s[0]: [] for s in _R66_STRATEGIES
    }
    branch_counts: dict[str, int] = {}

    for sc in bank_obj:
        round1_cands = _build_round_candidates_p66(sc.round1_emissions)
        round2_cands = _build_round_candidates_p66(sc.round2_emissions)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_p66(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R66_STRATEGIES:
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
                if sname == "capsule_bundle_contradiction":
                    trust = ps.get("trust") or {}
                    branch = str(trust.get("decoder_branch", "unknown"))
                    branch_counts[branch] = (
                        branch_counts.get(branch, 0) + 1)

    strategy_names = ("substrate",) + tuple(s[0] for s in _R66_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def _gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "w19_minus_w18": _gap(
            "capsule_bundle_contradiction", "capsule_relational_compat"),
        "w19_minus_attention_aware": _gap(
            "capsule_bundle_contradiction", "capsule_attention_aware"),
        "w19_minus_layered": _gap(
            "capsule_bundle_contradiction", "capsule_layered_multi_round"),
        "w19_minus_fifo": _gap(
            "capsule_bundle_contradiction", "capsule_fifo"),
        "w19_minus_substrate": _gap(
            "capsule_bundle_contradiction", "substrate"),
        "max_non_w19_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_bundle_contradiction"),
    }

    audit_ok_grid: dict[str, bool] = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    expected_shape = _BANK_EXPECTED_SHAPE.get(bank)
    bench_summary = {
        "n_scenarios": len(bench_property_per_scenario),
        "scenarios_with_symmetric_corroboration": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("symmetric_corroboration_holds")),
        "scenarios_with_expected_shape": sum(
            1 for v in bench_property_per_scenario.values()
            if expected_shape is not None
            and tuple(v.get("shape", ())) == expected_shape),
        "expected_shape": (list(expected_shape)
                            if expected_shape is not None else None),
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
        n_w18_abstain = sum(
            1 for r in rows
            if r.get("compatibility", {}).get("abstained"))
        n_w19_abstain = sum(
            1 for r in rows
            if r.get("trust", {}).get("abstained"))
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
        }

    pack_stats_summary = {
        s: _agg_packstats(pack_stats_per_strategy.get(s, []))
        for s in ("capsule_layered_fifo_packed",
                   "capsule_attention_aware",
                   "capsule_relational_compat",
                   "capsule_bundle_contradiction")
    }

    if verbose:
        print(f"[phase66] bank={bank} T_decoder={T_decoder} "
              f"n_eval={len(bank_obj)} K_auditor={K_auditor} "
              f"bank_seed={bank_seed}", file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase66]   {s:34s} acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase66] {k}: {v:+.3f}", file=sys.stderr, flush=True)
        if branch_counts:
            print(f"[phase66] w19 branch counts: {branch_counts}",
                  file=sys.stderr, flush=True)

    return {
        "schema": "phase66.deceptive_ambiguity.v1",
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
        "w19_branch_counts": branch_counts,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }


def run_phase66_seed_stability_sweep(*,
                                          bank: str = "deceive_naive",
                                          T_decoder: int | None = None,
                                          n_eval: int = 8,
                                          K_auditor: int = 12,
                                          seeds: Sequence[int] = (
                                              11, 17, 23, 29, 31),
                                          ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase66.seed_stability.v1",
        "config": {"bank": bank, "T_decoder": T_decoder,
                    "n_eval": n_eval, "K_auditor": K_auditor,
                    "seeds": list(seeds)},
        "per_seed": {},
    }
    for seed in seeds:
        rep = run_phase66(bank=bank, T_decoder=T_decoder,
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
            "w19_branch_counts": rep["w19_branch_counts"],
        }
    gaps_w19_w18 = [
        out["per_seed"][str(s)]["headline_gap"]["w19_minus_w18"]
        for s in seeds
    ]
    gaps_w19_aa = [
        out["per_seed"][str(s)]["headline_gap"]["w19_minus_attention_aware"]
        for s in seeds
    ]
    out["min_w19_minus_w18"] = min(gaps_w19_w18) if gaps_w19_w18 else 0.0
    out["max_w19_minus_w18"] = max(gaps_w19_w18) if gaps_w19_w18 else 0.0
    out["mean_w19_minus_w18"] = (
        round(sum(gaps_w19_w18) / len(gaps_w19_w18), 4)
        if gaps_w19_w18 else 0.0)
    out["min_w19_minus_attention_aware"] = (
        min(gaps_w19_aa) if gaps_w19_aa else 0.0)
    out["max_w19_minus_attention_aware"] = (
        max(gaps_w19_aa) if gaps_w19_aa else 0.0)
    out["mean_w19_minus_attention_aware"] = (
        round(sum(gaps_w19_aa) / len(gaps_w19_aa), 4)
        if gaps_w19_aa else 0.0)
    return out


def run_cross_regime_synthetic(*,
                                  n_eval: int = 8,
                                  bank_seed: int = 11,
                                  K_auditor: int = 12,
                                  T_auditor: int = 256,
                                  T_decoder_tight: int = 24,
                                  ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "phase66.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
        },
    }
    out["r66_corroborated"] = run_phase66(
        bank="corroborated", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r66_deceive_naive_loose"] = run_phase66(
        bank="deceive_naive", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r66_deceive_naive_tight"] = run_phase66(
        bank="deceive_naive", T_decoder=T_decoder_tight,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r66_confound_resolvable"] = run_phase66(
        bank="confound_resolvable", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r66_deceive_total"] = run_phase66(
        bank="deceive_total", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)
    out["r66_outside_required"] = run_phase66(
        bank="outside_required", T_decoder=None,
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor)

    def _acc(cell: str, strategy: str) -> float:
        return float(out[cell]["pooled"][strategy]["accuracy_full"])

    out["headline_summary"] = {
        "r66_corroborated_w19": _acc(
            "r66_corroborated", "capsule_bundle_contradiction"),
        "r66_corroborated_w18": _acc(
            "r66_corroborated", "capsule_relational_compat"),
        "r66_deceive_naive_loose_w19": _acc(
            "r66_deceive_naive_loose", "capsule_bundle_contradiction"),
        "r66_deceive_naive_loose_w18": _acc(
            "r66_deceive_naive_loose", "capsule_relational_compat"),
        "r66_deceive_naive_tight_w19": _acc(
            "r66_deceive_naive_tight", "capsule_bundle_contradiction"),
        "r66_deceive_naive_tight_w18": _acc(
            "r66_deceive_naive_tight", "capsule_relational_compat"),
        "r66_confound_resolvable_w19": _acc(
            "r66_confound_resolvable", "capsule_bundle_contradiction"),
        "r66_confound_resolvable_w18": _acc(
            "r66_confound_resolvable", "capsule_relational_compat"),
        "r66_deceive_total_w19": _acc(
            "r66_deceive_total", "capsule_bundle_contradiction"),
        "r66_outside_required_w19": _acc(
            "r66_outside_required", "capsule_bundle_contradiction"),
        "w19_minus_w18_deceive_naive_loose":
            round(_acc("r66_deceive_naive_loose",
                        "capsule_bundle_contradiction")
                   - _acc("r66_deceive_naive_loose",
                           "capsule_relational_compat"), 4),
        "w19_minus_w18_deceive_naive_tight":
            round(_acc("r66_deceive_naive_tight",
                        "capsule_bundle_contradiction")
                   - _acc("r66_deceive_naive_tight",
                           "capsule_relational_compat"), 4),
        "w19_minus_w18_confound_resolvable":
            round(_acc("r66_confound_resolvable",
                        "capsule_bundle_contradiction")
                   - _acc("r66_confound_resolvable",
                           "capsule_relational_compat"), 4),
    }
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 66 — bundle-contradiction-aware "
                     "trust-weighted disambiguation under deceptive / "
                     "confounded round-2 evidence (SDK v3.20 / W19 family).")
    p.add_argument("--bank", type=str, default="deceive_naive",
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
        report = run_phase66_seed_stability_sweep(
            bank=args.bank, T_decoder=T_dec,
            n_eval=args.n_eval, K_auditor=args.K_auditor)
    else:
        report = run_phase66(
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
            print(f"[phase66] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
