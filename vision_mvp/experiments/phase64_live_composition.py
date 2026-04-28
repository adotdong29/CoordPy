"""Phase 64 — fresh live-Ollama end-to-end composition + magnitude-
hinted producer protocol + symmetric-corroboration limit theorem.

SDK v3.18, W17 family anchor. The follow-up to SDK v3.17 (W16) on the
three honest gaps the prior milestone left explicit:

* the strongest end-to-end real-LLM result was on **recorded** bytes
  (W16-Λ-real-replay), not a fresh live probe (W16-C-LIVE-OLLAMA was
  conjectural pending Mac-1 reachable);
* the **1/8 model-side judgment miss** persisted on the W14 anchor
  (the ``slow_query_archival`` scenario where 14B judged decoy
  magnitudes "not severe enough" *relative* to the larger gold
  spike);
* the **symmetric-corroboration wall** (W15-C-SYMMETRIC /
  W16-C-SYMMETRIC) was named but not built or proved.

Phase-64 attacks all three on one driver.

Recap of why W17 needs its own bench
=====================================

The Phase-61 + Phase-63-replay capture revealed that the LLM-side
ambiguity-erasure failure on R-61-OLLAMA-A's slow_query scenario is
*not* a magnitude-threshold problem (the decoy magnitudes are above
every operational threshold) — it is a **relative-magnitude**
problem: the model judges each event against the *other* events in
the same prompt and skips events that look "small" by comparison,
even when they qualify by their own absolute thresholds.

The W14 structured prompt does NOT close this gap because it does
not give the LLM a concrete *named* lower bound. The W17
magnitude-hinted prompt does — and forbids relative-magnitude
skipping explicitly.

Phase-64 sub-banks
==================

Five sub-banks plus a named falsifier and a symmetric-corroboration
sub-bank:

* **R-64-baseline** (synthetic identity producer, magnitude-hinted
  prompt, ``T_decoder=None``). Sanity anchor — every cross-round
  capsule decoder hits 1.000; the magnitude-hinted prompt's
  rendering is byte-stable.
* **R-64-W14H-only** (synthetic mag-filter producer, magnitude-
  hinted prompt, ``T_decoder=None``). Synthetic counterpart of the
  W17-1 anchor — under magnitude-hint the synthetic mag-filter
  producer's per-event mandate survives. W17-3 backward-compat:
  both magnitude-hinted *and* legacy-structured render produce the
  same downstream answer on the synthetic side.
* **R-64-LIVE-STRUCT** (real Ollama 14B, *structured* prompt,
  ``T_decoder=14``). Reproduces the W14 anchor's 1/8 miss on a
  *fresh* live probe — the W17-Λ-no-hint structural anchor
  (closes the comparison set so the magnitude-hint extension is
  the load-bearing improvement, not a re-run).
* **R-64-LIVE-MAGHINT** (real Ollama 14B, *magnitude-hinted*
  prompt, ``T_decoder=14``). **The W17-1 anchor.** Strong
  success when bench property holds in ≥ 7/8 AND composed
  accuracy_full strictly improves over both substrate FIFO AND
  the FIFO-packed-W14H-only baseline by ≥ 0.50 on the live
  stream.
* **R-64-LIVE-XMODEL** (real Ollama qwen3.5:35b, *magnitude-
  hinted* prompt, ``T_decoder=14``). Cross-model probe
  (W17-C-XMODEL).
* **R-64-LIVE-NAIVE** (real Ollama 14B, *naive* prompt,
  ``T_decoder=14``). Named falsifier — live counterpart of
  W14-Λ-prompt + W15-Λ-budget joint failure.
* **R-64-SYM** (synthetic, deterministic identity extractor on a
  *symmetric-corroboration* bank where decoy and gold are both
  mentioned by ≥ 2 distinct producer roles via the *same* CCK
  tier as the gold; ``T_decoder ∈ {None, 24}``). **The
  W17-Λ-symmetric anchor.** Even with the W14 magnitude-hint AND
  W15 packer in scope, the bench property's *asymmetric*
  corroboration ingredient is removed; every capsule strategy
  ties FIFO at 0.000 by construction.

Theorem family W17 (minted by this milestone)
==============================================

* **W17-1** (proved-conditional + proved-empirical / empirical-
  research). Pairing the W17 :class:`StructuredProducerProtocol`
  in mode = ``magnitude_hinted`` with the W14 producer + W15
  decoder pipeline strictly improves real-LLM accuracy_full over
  the legacy structured prompt on the same fresh probe AND ties
  FIFO at 0.000 on R-64-SYM (so the strict gain is *conditional*
  on the asymmetric-corroboration bench property).
* **W17-Λ-no-hint** (proved-empirical on the live axis). Under
  the legacy structured prompt + tight ``T_decoder = 14``, the
  fresh live probe reproduces the Phase-61 7/8 + W16-Λ-real-replay
  0.500 envelope; the magnitude-hint extension, not a prompt
  re-run, is the load-bearing improvement.
* **W17-Λ-symmetric** (proved-empirical + structural sketch).
  **The first explicit symmetric-corroboration limit theorem in
  the programme.** On R-64-SYM (decoy = gold by every
  cross-role-corroboration metric the SDK tracks) every capsule
  strategy ties FIFO at 0.000 — the W11 contradiction-aware drop
  fires symmetrically; the W15 hypothesis-preserving pack
  preserves both; the W14H magnitude-hint is silent on the
  symmetric ambiguity. The argument is that ``services_correct``
  set-equality is an asymmetric oracle: when the two answer-set
  hypotheses are observationally indistinguishable from the
  bipartite ``(role × tag, kind, magnitude)`` multiset, no
  service-blind admission policy AND no closed-form salience
  packer can prefer one over the other.
* **W17-2** (proved by inspection + mechanically-checked).
  Magnitude-hinted prompt rendering is byte-for-byte deterministic
  given ``(schema, events, round_idx, OperationalThreshold table)``;
  with an empty threshold table, the prompt reduces to the W14
  structured prompt with the anti-relative-magnitude clause
  appended (W17-3 backward-compat reduction).
* **W17-3** (proved-empirical full programme regression). On
  R-54..R-63 default banks, swapping ``mode='structured'`` for
  ``mode='magnitude_hinted'`` with the
  :data:`incident_triage_magnitude_thresholds` table produces the
  same downstream answer (the synthetic side is unchanged because
  the magnitude-filter extractor's behaviour is governed by its
  own threshold parameters, not by the prompt thresholds).
* **W17-C-XMODEL** (empirical-research / conjectural). The W17-1
  win at qwen2.5:14b-32k transfers to qwen3.5:35b-MoE on the same
  fresh probe under the magnitude-hinted prompt.
* **W17-C-LEARNED-HINT** (conjectural). A learned threshold
  variant (per-kind data-driven thresholds rather than the
  pre-committed defaults) outperforms the closed-form
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` across 5/5
  random splits.

CLI
---

::

    # R-64-baseline (synthetic, sanity anchor):
    python3 -m vision_mvp.experiments.phase64_live_composition \\
        --extractor identity --prompt-mode magnitude_hinted \\
        --K-auditor 8 --n-eval 8 --out -

    # R-64-LIVE-MAGHINT (W17-1 anchor; fresh live qwen2.5:14b-32k):
    python3 -m vision_mvp.experiments.phase64_live_composition \\
        --extractor ollama --prompt-mode magnitude_hinted \\
        --decoder-budget 14 --K-auditor 8 --n-eval 8 \\
        --model qwen2.5:14b-32k --endpoint http://192.168.12.191:11434 \\
        --out docs/data/phase64_live_maghint_qwen2_5_14b_n8.json

    # R-64-LIVE-XMODEL (cross-model probe; fresh live qwen3.5:35b):
    python3 -m vision_mvp.experiments.phase64_live_composition \\
        --extractor ollama --prompt-mode magnitude_hinted \\
        --decoder-budget 14 --K-auditor 8 --n-eval 8 \\
        --model qwen3.5:35b --endpoint http://192.168.12.191:11434 \\
        --out docs/data/phase64_live_maghint_qwen3_5_35b_n8.json

    # R-64-SYM (W17-Λ-symmetric wall; synthetic):
    python3 -m vision_mvp.experiments.phase64_live_composition \\
        --extractor identity --bank symmetric \\
        --decoder-budget 24 --K-auditor 12 --n-eval 8 --out -

    # Cross-regime synthetic summary:
    python3 -m vision_mvp.experiments.phase64_live_composition \\
        --cross-regime-synthetic --K-auditor 8 --n-eval 8 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.llm_backend import LLMBackend, OllamaBackend
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, AttentionAwareBundleDecoder,
    BundleAwareTeamDecoder,
    CapsuleContextPacker,
    CLAIM_KIND_SYNONYMS,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, FifoContextPacker,
    INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    OperationalThreshold,
    PRODUCER_PROMPT_MAGNITUDE_HINTED,
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
    ProducerPromptResult,
    RobustMultiRoundBundleDecoder, RoleBudget,
    RoleExtractionSchema,
    StructuredProducerProtocol,
    TeamCoordinator, audit_team_lifecycle,
    collect_admitted_handoffs, _DecodedHandoff,
    incident_triage_magnitude_thresholds,
    incident_triage_role_schemas,
    normalize_payload,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    MultiRoundScenario, _as_incident_scenario,
)
from vision_mvp.experiments.phase61_producer_ambiguity_preservation import (
    CapturingOllamaExtractor, IdentityExtractor,
    MagnitudeFilteringExtractor, _bench_property,
    build_phase61_bank, _parse_structured_response,
)
from vision_mvp.experiments.phase59_real_llm_multi_round import (
    _parse_ollama_response,
)


# =============================================================================
# Symmetric-corroboration bank — W17-Λ-symmetric anchor
# =============================================================================
#
# In R-58 / R-61 the asymmetry is the load-bearing structural feature:
# decoy = corroborated by ≥ 2 distinct roles, gold = single role. The
# W11 contradiction-aware drop fires on the multi-role-noise-only
# decoy and preserves the single-role gold; the answer set ends up
# correct.
#
# In R-64-SYM we *remove* this asymmetry. Both gold and decoy are
# mentioned by ≥ 2 distinct producer roles in round 1 via generic-noise
# kinds with comparable magnitudes. The round-2 specific-tier
# disambiguator still names the gold root_cause but does NOT carry a
# ``service=`` token (R-58 invariant), so the answer must be derived
# from cross-role + cross-round structure alone.
#
# Under this regime:
#   * The W11 contradiction-aware drop fires on BOTH gold and decoy
#     (both are noise-corroborated by ≥ 2 roles) — drops everything,
#     including the gold answer.
#   * The W15 hypothesis-preserving pack keeps both gold and decoy
#     representatives (each gets one slot) — preserves both, but the
#     downstream priority decoder still cannot distinguish them.
#   * The W14H magnitude-hint instructs the LLM to emit every
#     qualifying event — but the resulting symmetric stream is
#     observationally indistinguishable.
#
# Therefore every capsule strategy ties FIFO at 0.000 by construction.
# This is **W17-Λ-symmetric** — the named structural limit at the
# symmetric-corroboration regime, the wall the programme has been
# pointing toward since SDK v3.16.


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


_SYM_REMEDIATION = {
    "deadlock":             "enforce_lock_ordering_in_orders",
    "pool_exhaustion":      "raise_pool_cap_or_fix_upstream",
    "slow_query_cascade":   "index_or_split_slow_query",
    "disk_fill":            "rotate_logs_and_clear_backup",
}


def _build_p64sym_deadlock(
        decoy: str = "search_index") -> MultiRoundScenario:
    """Phase-64-SYM deadlock scenario.

    Symmetric-corroboration shape:
    * Gold services A/B: each mentioned by 2 distinct roles
      (``monitor`` via ``LATENCY_SPIKE`` / ``ERROR_RATE_SPIKE`` AND
      ``network`` via ``FW_BLOCK_SURGE``) — the same two roles that
      route generic-noise kinds to the auditor under
      ``incident_triage.build_role_subscriptions()``.
    * Decoy service: also mentioned by the same 2 distinct roles
      (monitor + network) via generic-noise kinds with comparable
      magnitudes.

    Under :func:`build_role_subscriptions`, only ``monitor`` and
    ``network`` route the generic-noise kinds (``LATENCY_SPIKE``,
    ``ERROR_RATE_SPIKE``, ``FW_BLOCK_SURGE``) to the auditor in
    round 1 (db_admin and sysadmin's emissions of these kinds are
    silently filtered by the subscription table). So a *truly*
    symmetric bench can use only those two roles for round-1
    corroboration. The round-2 disambiguator (``DEADLOCK_SUSPECTED``)
    names the gold root_cause without ``service=`` token; the
    cross-role corroboration count for gold and decoy is identical
    under any service-blind admission policy AND under the W11
    contradiction-aware drop (both gold and decoy are noise-only,
    multi-role-corroborated).
    """
    A, B = "orders", "payments"
    return MultiRoundScenario(
        scenario_id=(
            f"p64sym_deadlock_orders_payments__sym_{decoy}"),
        description=(
            f"Phase-64-SYM: symmetric-corroboration. Gold {A}/{B} "
            f"AND decoy {decoy} all mentioned by 2 distinct routed "
            f"roles (monitor + network) via generic-noise kinds "
            f"with comparable magnitudes. W17-Λ-symmetric anchor."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="deadlock",
        gold_remediation=_SYM_REMEDIATION["deadlock"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=2200 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.20 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2100 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={A}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={B}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy}"),
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


def _build_p64sym_pool(decoy: str = "archival") -> MultiRoundScenario:
    A, B = "api", "db"
    return MultiRoundScenario(
        scenario_id=f"p64sym_pool_api_db__sym_{decoy}",
        description=(
            f"Phase-64-SYM: symmetric-corroboration. Gold {A}/{B} AND "
            f"decoy {decoy} all corroborated by 2 distinct routed "
            f"roles (monitor + network). W17-Λ-symmetric anchor."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="pool_exhaustion",
        gold_remediation=_SYM_REMEDIATION["pool_exhaustion"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=2300 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.19 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2200 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.16 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={A}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={B}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={decoy}"),
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


def _build_p64sym_disk(decoy: str = "search_index"
                          ) -> MultiRoundScenario:
    A, B = "storage", "logs_pipeline"
    return MultiRoundScenario(
        scenario_id=f"p64sym_disk_storage_logs__sym_{decoy}",
        description=(
            f"Phase-64-SYM: symmetric-corroboration. Gold {A}/{B} AND "
            f"decoy {decoy} all corroborated by 2 distinct routed "
            f"roles (monitor + network). W17-Λ-symmetric anchor."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="disk_fill",
        gold_remediation=_SYM_REMEDIATION["disk_fill"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.22 service={A}"),
                _emit("LATENCY_SPIKE", f"p95_ms=2400 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2300 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={A}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={B}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=7 service={decoy}"),
            ),
            ROLE_SYSADMIN: (),
            ROLE_DB_ADMIN: (),
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


def _build_p64sym_slow_query(decoy: str = "metrics"
                                ) -> MultiRoundScenario:
    A, B = "web", "db"
    return MultiRoundScenario(
        scenario_id=f"p64sym_slow_query_web_db__sym_{decoy}",
        description=(
            f"Phase-64-SYM: symmetric-corroboration. Gold {A}/{B} AND "
            f"decoy {decoy} all corroborated by 2 distinct routed "
            f"roles (monitor + network). W17-Λ-symmetric anchor."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="slow_query_cascade",
        gold_remediation=_SYM_REMEDIATION["slow_query_cascade"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=2300 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={B}"),
                _emit("LATENCY_SPIKE",
                       f"p95_ms=2200 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.15 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={A}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={B}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy}"),
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


_P64SYM_BUILDERS = (
    _build_p64sym_deadlock, _build_p64sym_pool, _build_p64sym_disk,
    _build_p64sym_slow_query,
)
_P64SYM_DECOYS = (
    "search_index", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "cache", "scratch_pool",
)


def build_phase64_sym_bank(*, n_replicates: int = 2,
                              seed: int = 11
                              ) -> list[MultiRoundScenario]:
    """Phase-64-SYM bank — symmetric-corroboration anchor.

    Mechanically-checkable invariant (W17-Λ-symmetric closure):
    * For every scenario, every gold service AND the decoy service
      are mentioned by ≥ 2 distinct producer roles in round 1 via
      generic-noise kinds with magnitudes inside the operational
      qualifying band. Verified by
      ``Phase64SymBankShapeTests::test_every_scenario_is_symmetric``.
    """
    import random
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for builder in _P64SYM_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = _P64SYM_DECOYS[(i + r) % len(_P64SYM_DECOYS)]
            sc = builder(decoy=chosen)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


def _phase64_sym_property(sc: MultiRoundScenario,
                            round1_cands, round2_cands
                            ) -> dict[str, Any]:
    """Verify R-64-SYM symmetric-corroboration property
    post-normalisation. Symmetry means: every gold service AND the
    decoy service are mentioned by ≥ 2 distinct producer roles in
    round 1, with comparable magnitudes."""
    def _service_role_set(cands, service):
        roles: set[str] = set()
        for (src, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            np = normalize_payload(payload)
            if f"service={service}" in np:
                roles.add(src)
        return roles
    gold_a, gold_b = sc.gold_services_pair
    decoy = sc.decoy_storm_service
    gold_a_roles = _service_role_set(round1_cands, gold_a)
    gold_b_roles = _service_role_set(round1_cands, gold_b)
    decoy_roles = _service_role_set(round1_cands, decoy)
    both_golds_corr = (
        len(gold_a_roles) >= 2 and len(gold_b_roles) >= 2)
    decoy_corr = len(decoy_roles) >= 2
    sym = both_golds_corr and decoy_corr
    return {
        "gold_a_role_count": len(gold_a_roles),
        "gold_b_role_count": len(gold_b_roles),
        "decoy_role_count": len(decoy_roles),
        "both_golds_cross_role_corroborated": both_golds_corr,
        "decoy_cross_role_corroborated": decoy_corr,
        "symmetric_corroboration_holds": sym,
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
    }


# =============================================================================
# Strategy / decoder dispatch — same pattern as Phase-63
# =============================================================================


_R64_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("capsule_fifo", "per_round"),
    ("capsule_priority", "per_round"),
    ("capsule_coverage", "per_round"),
    ("capsule_cohort_buffered", "per_round"),
    ("capsule_corroboration", "per_round"),
    ("capsule_multi_service", "per_round"),
    ("capsule_multi_round", "multi_round_bundle"),
    ("capsule_robust_multi_round", "robust_multi_round"),
    ("capsule_layered_multi_round", "layered_multi_round"),
    # W14-only-budgeted baseline (FIFO pack on the structured /
    # magnitude-hinted stream).
    ("capsule_layered_fifo_packed", "fifo_packed_layered"),
    # W14H + W15 composed.
    ("capsule_attention_aware", "attention_aware"),
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
                     "capsule_attention_aware"):
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


def _round_hint_from_ledger(ledger,
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
        team_tag="phase64_live_composition",
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
            union, CapsuleContextPacker(),
            T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "fifo_packed_layered":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
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


# =============================================================================
# Phase 64 driver
# =============================================================================


def _build_extractor(extractor_kind: str, *,
                       prompt_mode: str,
                       seed: int,
                       endpoint: str,
                       model: str,
                       timeout_s: float,
                       threshold_p95_ms: float,
                       threshold_error_rate: float,
                       threshold_fw_count: int,
                       think: "bool | None" = None,
                       ):
    if extractor_kind == "identity":
        return IdentityExtractor(seed=seed)
    if extractor_kind == "magnitude_filter":
        return MagnitudeFilteringExtractor(
            seed=seed,
            threshold_p95_ms=threshold_p95_ms,
            threshold_error_rate=threshold_error_rate,
            threshold_fw_count=threshold_fw_count,
            prompt_mode=prompt_mode)
    if extractor_kind == "ollama":
        backend = OllamaBackend(
            model=model, base_url=endpoint, timeout=timeout_s,
            think=think)
        return CapturingOllamaExtractor(
            backend=backend,
            fallback_cfg=dict(
                seed=seed,
                threshold_p95_ms=threshold_p95_ms,
                threshold_error_rate=threshold_error_rate,
                threshold_fw_count=threshold_fw_count),
            prompt_mode=prompt_mode)
    raise ValueError(
        f"unknown extractor {extractor_kind!r}; "
        f"valid: identity / magnitude_filter / ollama")


def _build_bank(bank_kind: str, *,
                  n_replicates: int, bank_seed: int
                  ) -> list[MultiRoundScenario]:
    if bank_kind == "phase61":
        return build_phase61_bank(
            n_replicates=n_replicates, seed=bank_seed)
    if bank_kind == "symmetric":
        return build_phase64_sym_bank(
            n_replicates=n_replicates, seed=bank_seed)
    raise ValueError(
        f"unknown bank {bank_kind!r}; valid: phase61 / symmetric")


def run_phase64(*,
                 n_eval: int | None = None,
                 K_auditor: int = 8,
                 T_auditor: int = 256,
                 K_producer: int = 6,
                 T_producer: int = 96,
                 inbox_capacity: int | None = None,
                 bank_seed: int = 11,
                 bank_replicates: int = 2,
                 bank: str = "phase61",
                 T_decoder: int | None = None,
                 extractor: str = "identity",
                 prompt_mode: str = PRODUCER_PROMPT_MAGNITUDE_HINTED,
                 magnitude_hinted_schema: bool = True,
                 threshold_p95_ms: float = 1000.0,
                 threshold_error_rate: float = 0.10,
                 threshold_fw_count: int = 5,
                 endpoint: str = "http://192.168.12.191:11434",
                 model: str = "qwen2.5:14b-32k",
                 timeout_s: float = 300.0,
                 think: "bool | None" = None,
                 verbose: bool = False,
                 ) -> dict[str, Any]:
    """Drive Phase 64 end-to-end (one cell of the W17 cross-regime).

    Pre-committed default config (R-64-baseline sanity anchor):
    ``extractor='identity'``, ``prompt_mode='magnitude_hinted'``,
    ``magnitude_hinted_schema=True``, ``T_decoder=None``,
    ``K_auditor=8``, ``n_eval=8``, ``bank_seed=11``, ``bank='phase61'``.

    The W17-1 anchor is reached with ``extractor='ollama'``,
    ``prompt_mode='magnitude_hinted'``, ``T_decoder=14``, fresh live
    Mac-1 endpoint reachable.
    """
    bank_obj = _build_bank(
        bank, n_replicates=bank_replicates, bank_seed=bank_seed)
    if n_eval is not None:
        bank_obj = bank_obj[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    extractor_obj = _build_extractor(
        extractor, prompt_mode=prompt_mode, seed=bank_seed,
        endpoint=endpoint, model=model, timeout_s=timeout_s,
        threshold_p95_ms=threshold_p95_ms,
        threshold_error_rate=threshold_error_rate,
        threshold_fw_count=threshold_fw_count,
        think=think)
    schemas = incident_triage_role_schemas(
        magnitude_hinted=bool(magnitude_hinted_schema))
    protocol = StructuredProducerProtocol(mode=prompt_mode)

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    sym_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R64_STRATEGIES
    }
    prompt_records: list[dict[str, Any]] = []
    for sc in bank_obj:
        recorded: list[ProducerPromptResult] = []
        round1_cands = extractor_obj.extract_round(
            sc, 1, protocol=protocol, schemas=schemas,
            record_prompts=recorded)
        round2_cands = extractor_obj.extract_round(
            sc, 2, protocol=protocol, schemas=schemas,
            record_prompts=recorded)
        bench_property_per_scenario[sc.scenario_id] = _bench_property(
            sc, round1_cands, round2_cands)
        if bank == "symmetric":
            sym_property_per_scenario[sc.scenario_id] = (
                _phase64_sym_property(sc, round1_cands, round2_cands))
        for pr in recorded:
            prompt_records.append({
                "scenario_id": sc.scenario_id,
                "role": pr.role,
                "round_idx": pr.round_idx,
                "mode": pr.mode,
                "n_kinds_in_scope": len(pr.kinds_in_scope),
                "prompt_sha256": hashlib.sha256(
                    pr.text.encode("utf-8")).hexdigest()[:16],
            })
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R64_STRATEGIES:
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

    strategy_names = ("substrate",) + tuple(s[0] for s in _R64_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "composed_minus_fifo_packed": gap(
            "capsule_attention_aware", "capsule_layered_fifo_packed"),
        "composed_minus_fifo": gap(
            "capsule_attention_aware", "capsule_fifo"),
        "composed_minus_layered": gap(
            "capsule_attention_aware", "capsule_layered_multi_round"),
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
            if v.get("delayed_causal_evidence_property_holds")),
        "scenarios_with_decoy_corroboration": sum(
            1 for v in bench_property_per_scenario.values()
            if v.get("round1_decoy_corroborated")),
        "K_auditor": K_auditor,
        "T_decoder": T_decoder,
        "bank": bank,
    }
    if bank == "symmetric":
        bench_summary["scenarios_with_symmetric_corroboration"] = sum(
            1 for v in sym_property_per_scenario.values()
            if v.get("symmetric_corroboration_holds"))

    def _agg_packstats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        s_in = sum(r.get("tokens_input", 0) for r in rows)
        s_kept = sum(r.get("tokens_kept", 0) for r in rows)
        s_drop = sum(r.get("n_dropped_budget", 0) for r in rows)
        s_h_in = sum(r.get("n_handoffs_admitted", 0) for r in rows)
        s_h_kept = sum(r.get("n_handoffs_decoder_input", 0) for r in rows)
        pos = [r.get("position_of_first_causal_claim", -1) for r in rows]
        n_have = sum(1 for p in pos if p >= 0)
        avg_pos = (sum(p for p in pos if p >= 0) / max(1, n_have)
                    if n_have > 0 else -1.0)
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

    extractor_stats: dict[str, Any] = {
        "extractor": extractor,
        "prompt_mode": prompt_mode,
        "magnitude_hinted_schema": bool(magnitude_hinted_schema),
        "T_decoder": T_decoder,
        "threshold_p95_ms": threshold_p95_ms,
        "threshold_error_rate": threshold_error_rate,
        "threshold_fw_count": threshold_fw_count,
    }
    if isinstance(extractor_obj, MagnitudeFilteringExtractor):
        extractor_stats["n_emitted"] = extractor_obj.n_emitted
        extractor_stats["n_filtered_by_threshold"] = (
            extractor_obj.n_filtered_by_threshold)
        extractor_stats["n_compressed_by_prompt"] = (
            extractor_obj.n_compressed_by_prompt)
        extractor_stats["n_emitted_under_protocol"] = (
            extractor_obj.n_emitted_under_protocol)
    if isinstance(extractor_obj, CapturingOllamaExtractor):
        extractor_stats["n_real_calls"] = extractor_obj.n_real_calls
        extractor_stats["n_failed_calls"] = extractor_obj.n_failed_calls
        extractor_stats["n_synthetic_fallbacks"] = (
            extractor_obj.n_synthetic_fallbacks)
        extractor_stats["total_wall_s"] = round(
            extractor_obj.total_wall_s, 3)
        extractor_stats["endpoint"] = endpoint
        extractor_stats["model"] = model

    if verbose:
        print(f"[phase64] extractor={extractor}, "
              f"prompt_mode={prompt_mode}, T_decoder={T_decoder}, "
              f"n_eval={len(bank_obj)}, K_auditor={K_auditor}, "
              f"bank={bank}",
              file=sys.stderr, flush=True)
        print(f"[phase64] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank_obj)}",
              file=sys.stderr, flush=True)
        print(f"[phase64] decoy_corroboration in "
              f"{bench_summary['scenarios_with_decoy_corroboration']}/{len(bank_obj)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase64]   {s:32s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase64] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)

    out: dict[str, Any] = {
        "schema": "phase64.live_composition.v1",
        "config": {
            "n_eval": len(bank_obj), "K_auditor": K_auditor,
            "T_auditor": T_auditor, "K_producer": K_producer,
            "T_producer": T_producer, "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "bank": bank, "T_decoder": T_decoder,
            "extractor": extractor, "prompt_mode": prompt_mode,
            "magnitude_hinted_schema": bool(magnitude_hinted_schema),
            "threshold_p95_ms": threshold_p95_ms,
            "threshold_error_rate": threshold_error_rate,
            "threshold_fw_count": threshold_fw_count,
            "endpoint": endpoint, "model": model,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "extractor_stats": extractor_stats,
        "pack_stats_summary": pack_stats_summary,
        "prompt_records": prompt_records,
        "scenarios_evaluated": [sc.scenario_id for sc in bank_obj],
        "n_results": len(results),
    }
    if bank == "symmetric":
        out["sym_property_per_scenario"] = sym_property_per_scenario
    if isinstance(extractor_obj, CapturingOllamaExtractor):
        raw_per_sc: dict[str, list[dict[str, Any]]] = {}
        for entry in extractor_obj.raw_responses.values():
            raw_per_sc.setdefault(entry["scenario_id"], []).append(entry)
        out["raw_responses_per_scenario"] = raw_per_sc
    return out


def run_cross_regime_synthetic(*,
                                  n_eval: int = 8,
                                  bank_seed: int = 11,
                                  K_auditor: int = 8,
                                  T_auditor: int = 256,
                                  T_decoder_tight: int = 14,
                                  ) -> dict[str, Any]:
    """Three pre-committed synthetic sub-banks for the W17 wall +
    backward-compat surface. Live cells are NOT run here — invoke
    ``run_phase64(extractor='ollama', ...)`` separately for those."""
    out: dict[str, Any] = {
        "schema": "phase64.cross_regime_synthetic.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
        },
    }
    # 1. R-64-baseline (synthetic identity, magnitude-hinted; sanity).
    out["r64_baseline"] = run_phase64(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=None, bank="phase61",
        extractor="identity",
        prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
        magnitude_hinted_schema=True,
        verbose=False)
    # 2. R-64-W14H-only (synthetic mag-filter, magnitude-hinted prompt).
    out["r64_w14h_only"] = run_phase64(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=None, bank="phase61",
        extractor="magnitude_filter",
        prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
        magnitude_hinted_schema=True,
        verbose=False)
    # 3. R-64-SYM (symmetric-corroboration; W17-Λ-symmetric anchor).
    out["r64_sym_loose"] = run_phase64(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=12, T_auditor=T_auditor,
        T_decoder=None, bank="symmetric",
        extractor="identity",
        prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
        magnitude_hinted_schema=True,
        verbose=False)
    out["r64_sym_tight"] = run_phase64(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=12, T_auditor=T_auditor,
        T_decoder=24, bank="symmetric",
        extractor="identity",
        prompt_mode=PRODUCER_PROMPT_MAGNITUDE_HINTED,
        magnitude_hinted_schema=True,
        verbose=False)
    # Headline summary across the four cells.
    def _acc(cell_key: str, strategy: str) -> float:
        return float(
            out[cell_key]["pooled"][strategy]["accuracy_full"])

    out["headline_summary"] = {
        "r64_baseline_attention_aware":
            _acc("r64_baseline", "capsule_attention_aware"),
        "r64_w14h_only_attention_aware":
            _acc("r64_w14h_only", "capsule_attention_aware"),
        "r64_sym_loose_max_capsule": max(
            _acc("r64_sym_loose", s)
            for s in (
                "capsule_attention_aware",
                "capsule_layered_multi_round",
                "capsule_layered_fifo_packed",
                "capsule_robust_multi_round",
                "capsule_multi_round",
                "capsule_corroboration",
                "capsule_multi_service",
            )
        ),
        "r64_sym_tight_max_capsule": max(
            _acc("r64_sym_tight", s)
            for s in (
                "capsule_attention_aware",
                "capsule_layered_multi_round",
                "capsule_layered_fifo_packed",
                "capsule_robust_multi_round",
                "capsule_multi_round",
                "capsule_corroboration",
                "capsule_multi_service",
            )
        ),
    }
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 64 — fresh live-Ollama composition + "
                     "magnitude-hinted protocol + symmetric-corroboration "
                     "wall (SDK v3.18 / W17 family).")
    p.add_argument("--K-auditor", type=int, default=8)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--bank", type=str, default="phase61",
                    choices=("phase61", "symmetric"))
    p.add_argument("--decoder-budget", type=int, default=-1,
                    help="Strict T_decoder budget. -1 = None.")
    p.add_argument("--extractor", type=str, default="identity",
                    choices=("identity", "magnitude_filter", "ollama"))
    p.add_argument(
        "--prompt-mode", type=str,
        default=PRODUCER_PROMPT_MAGNITUDE_HINTED,
        choices=(PRODUCER_PROMPT_NAIVE,
                 PRODUCER_PROMPT_STRUCTURED,
                 PRODUCER_PROMPT_MAGNITUDE_HINTED))
    p.add_argument(
        "--no-magnitude-hinted-schema", dest="magnitude_hinted_schema",
        action="store_false",
        help="Disable the W17 magnitude-hint table on the schema "
              "(structurally equivalent to legacy structured prompt).")
    p.set_defaults(magnitude_hinted_schema=True)
    p.add_argument("--threshold-p95-ms", type=float, default=1000.0)
    p.add_argument("--threshold-error-rate", type=float, default=0.10)
    p.add_argument("--threshold-fw-count", type=int, default=5)
    p.add_argument("--endpoint", type=str,
                    default="http://192.168.12.191:11434")
    p.add_argument("--model", type=str, default="qwen2.5:14b-32k")
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--think-false", dest="think", action="store_const",
                    const=False, default=None,
                    help="Pass think=false to the Ollama backend "
                          "(required for qwen3.5:35b-style reasoning "
                          "models that otherwise emit empty content "
                          "when their thinking-mode tokens are hidden).")
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
    else:
        report = run_phase64(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            bank=args.bank,
            T_decoder=T_dec,
            extractor=args.extractor, prompt_mode=args.prompt_mode,
            magnitude_hinted_schema=args.magnitude_hinted_schema,
            threshold_p95_ms=args.threshold_p95_ms,
            threshold_error_rate=args.threshold_error_rate,
            threshold_fw_count=args.threshold_fw_count,
            endpoint=args.endpoint, model=args.model,
            timeout_s=args.timeout_s,
            think=args.think,
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
            print(f"[phase64] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
