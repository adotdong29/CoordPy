"""Phase 38 Part A — two-layer ensemble composition on the contested bank.

Phase 37 closed two reply-axis gaps — biased-primary and
malformed-syntactic — via ``core/reply_ensemble``. Theorem P37-4
formalised the residual: a reply-axis ensemble placed strictly
below a noise wrapper cannot defend against that noise.
Conjecture C37-2 named the composition that *would*: stack an
extractor-axis ensemble and a reply-axis ensemble so each
defends at a different boundary.

This driver measures that composition on the Phase-35 contested
bank under a *conjunction* noise cell — a layer-1 adversary
drops the gold claim at the extractor boundary AND a layer-2
adversary emits biased INDEPENDENT_ROOT on every reply. Neither
single-layer ensemble alone closes this cell by construction;
the question is whether the two-layer ensemble does.

Configurations (5):
  - ``baseline``        — Phase-31 regex extractor + Phase-37
    single LLMThreadReplier. Null defense.
  - ``extractor_only``  — Phase-38 UnionClaimExtractor (regex
    primary + narrative secondary) + Phase-37 single replier.
    Defends layer-1.
  - ``reply_only``      — Phase-31 regex + Phase-37 dual_agree
    EnsembleReplier. Defends layer-2.
  - ``two_layer``       — Phase-38 UnionClaimExtractor + Phase-
    37 dual_agree EnsembleReplier. Defends both layers.
  - ``two_layer_path_union`` — Phase-38 UnionClaimExtractor +
    Phase-38 PathUnion combiner over (primary replier,
    secondary replier). Tests PathUnion mode's behaviour above
    a noise wrapper.

Noise cells (5):
  - ``clean``                 — no noise.
  - ``ext_drop_gold``         — layer-1 adversary drops the
    gold IR claim from its producer role, budget=1.
  - ``rep_biased_primary``    — layer-2 adversary: primary
    replier always emits INDEPENDENT_ROOT.
  - ``conjunction``           — both ext_drop_gold AND
    rep_biased_primary active.
  - ``adv_drop_root``         — legacy Phase-37 adversarial
    reply wrapper (noise on the replier's OUTPUT). Reference
    cell to reproduce Theorem P37-4's powerless-single-layer
    finding.

Reported metrics (per cell × config):
  * dyn accuracy_full, contested_accuracy_full;
  * adaptive_sub accuracy_full, contested_accuracy_full;
  * per-layer stats: ext union counters, replier ensemble
    counters, adversary counters;
  * handoff-recall proxy: ``n_handoffs_delivered`` mean;
  * reply accuracy proxy: fraction of correct final causality
    classes emitted on the candidates.

Reproducible commands:

    python3 -m vision_mvp.experiments.phase38_two_layer_ensemble \\
        --seeds 35 36 --distractor-counts 6 \\
        --out vision_mvp/results_phase38_two_layer_ensemble.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.extractor_adversary import (
    DropGoldClaimExtractor, NarrativeSecondaryExtractor,
    UnionClaimExtractor, build_union_extractor,
)
from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig, LLMThreadReplier,
    causality_extractor_from_replier,
)
from vision_mvp.core.reply_ensemble import (
    EnsembleReplier, MODE_DUAL_AGREE,
    causality_extractor_from_ensemble,
)
from vision_mvp.core.reply_noise import (
    AdversarialReplyConfig, ReplyCorruptionReport,
    ReplyNoiseConfig, ADVERSARIAL_REPLY_MODE_DROP_ROOT,
    adversarial_reply_extractor,
    noisy_causality_extractor,
)
from vision_mvp.core.two_layer_ensemble import (
    PATH_MODE_UNION_ROOT, PathUnionCausalityExtractor,
    TwoLayerDefense,
)
from vision_mvp.experiments.phase36_llm_replies import (
    ScenarioAwareMockReplier,
)
from vision_mvp.experiments.phase37_reply_ensemble import (
    BiasedIRMockStub,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_ADAPTIVE_SUB, STRATEGY_DYNAMIC,
    STRATEGY_STATIC_HANDOFF, MockContestedAuditor,
    build_contested_bank, infer_causality_hypothesis,
    run_contested_loop,
)
from vision_mvp.tasks.incident_triage import (
    extract_claims_for_role,
)


# =============================================================================
# Builder helpers
# =============================================================================


def _clean_replier() -> LLMThreadReplier:
    stub = ScenarioAwareMockReplier(malformed_prob=0.0,
                                      out_of_vocab_prob=0.0)
    return LLMThreadReplier(
        llm_call=stub,
        config=LLMReplyConfig(witness_token_cap=12),
        cache={})


def _biased_replier() -> LLMThreadReplier:
    return LLMThreadReplier(
        llm_call=BiasedIRMockStub(),
        config=LLMReplyConfig(witness_token_cap=12),
        cache={})


def _scenario_gold_drop_target(scenario_id: str
                                ) -> tuple[str, str]:
    """Return the (role, claim_kind) to drop for the adversarial
    layer-1 attack. For every contested scenario this is the
    producer of the gold-root-cause claim.
    """
    # Mirror the Phase-35 scenario bank's gold assignments.
    m = {
        "contested_deadlock_vs_shadow_cron":
            ("db_admin", "DEADLOCK_SUSPECTED"),
        "contested_tls_vs_disk_shadow":
            ("network", "TLS_EXPIRED"),
        "contested_dns_vs_pool_symptom":
            ("network", "DNS_MISROUTE"),
        "contested_cron_vs_oom_shadow":
            ("sysadmin", "OOM_KILL"),
        "contested_dns_vs_tls_shadow":
            ("network", "DNS_MISROUTE"),
        "concordant_disk_fill":
            ("sysadmin", "DISK_FILL_CRITICAL"),
    }
    if scenario_id in m:
        return m[scenario_id]
    return ("sysadmin", "DISK_FILL_CRITICAL")


# -----------------------------------------------------------------------------
# Config construction
# -----------------------------------------------------------------------------


@dataclass
class ConfigSpec:
    name: str
    has_ext_ensemble: bool
    has_reply_ensemble: bool
    path_union_mode: str | None = None  # None | PATH_MODE_*


CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec("baseline", False, False),
    ConfigSpec("extractor_only", True, False),
    ConfigSpec("reply_only", False, True),
    ConfigSpec("two_layer", True, True),
    ConfigSpec("two_layer_path_union", True, False,
                path_union_mode=PATH_MODE_UNION_ROOT),
)


# Noise cells
CELLS: tuple[tuple[str, dict], ...] = (
    ("clean", {}),
    ("ext_drop_gold", {"_drop_gold": True}),
    ("rep_biased_primary", {"_biased_primary": True}),
    ("conjunction", {"_drop_gold": True,
                       "_biased_primary": True}),
    ("adv_drop_root", {
        "adversarial": ADVERSARIAL_REPLY_MODE_DROP_ROOT,
        "budget": 1}),
)


# -----------------------------------------------------------------------------
# Per-scenario configuration: builds the (claim_extractor,
# causality_extractor) pair for one (config × cell) slot. Must be
# per-scenario so the adversary's budget is reset per scenario and
# so the gold-claim target matches the scenario.
# -----------------------------------------------------------------------------


def _build_claim_extractor(cfg: ConfigSpec, cell: dict,
                             scenario_id: str,
                             ) -> tuple[Callable | None, dict]:
    drop = cell.get("_drop_gold", False)
    if not cfg.has_ext_ensemble and not drop:
        return None, {}
    if not cfg.has_ext_ensemble and drop:
        role, kind = _scenario_gold_drop_target(scenario_id)
        adv = DropGoldClaimExtractor(
            target_role=role, target_kind=kind,
            base=extract_claims_for_role, budget=1)
        return adv, {"adversary": adv.stats,
                      "ext_union": None}
    # cfg.has_ext_ensemble True
    if drop:
        role, kind = _scenario_gold_drop_target(scenario_id)
        primary = DropGoldClaimExtractor(
            target_role=role, target_kind=kind,
            base=extract_claims_for_role, budget=1)
    else:
        primary = extract_claims_for_role
    secondary = NarrativeSecondaryExtractor()
    union = UnionClaimExtractor(primary=primary,
                                 secondary=secondary)
    stats = {
        "adversary": getattr(primary, "stats",
                              lambda: {})
            if hasattr(primary, "stats") else (lambda: {}),
        "ext_union": union,
        "ext_secondary": secondary,
    }
    return union, stats


def _build_causality_extractor(cfg: ConfigSpec, cell: dict,
                                 ) -> tuple[Callable | None, dict]:
    """Construct the causality extractor for the given config and cell.

    Returns ``(extractor, diag)`` where ``diag`` collects the
    per-call replier / ensemble stats objects for later reporting.
    """
    biased = cell.get("_biased_primary", False)
    has_reply_ens = cfg.has_reply_ensemble
    has_path_union = cfg.path_union_mode is not None

    # 1. Path-union mode — tested without reply-axis ensemble; the
    #    two paths are each a single replier. Implementation:
    #      primary  = noise-wrapped replier extractor
    #      secondary = clean replier extractor
    #    combined by PathUnionCausalityExtractor.
    if has_path_union:
        primary_rep = (_biased_replier() if biased
                        else _clean_replier())
        p_ext = causality_extractor_from_replier(primary_rep)
        # Apply Phase-37-style noise wrapper to the primary's output.
        adv_spec = {k: v for k, v in cell.items()
                     if not k.startswith("_")}
        if adv_spec.get("adversarial"):
            adv = AdversarialReplyConfig(
                target_mode=adv_spec["adversarial"],
                budget=adv_spec.get("budget", 1))
            p_ext = adversarial_reply_extractor(p_ext, adv)
        elif adv_spec:
            cfg_n = ReplyNoiseConfig(
                drop_prob=adv_spec.get("drop_prob", 0.0),
                mislabel_prob=adv_spec.get(
                    "mislabel_prob", 0.0),
                seed=adv_spec.get("seed", 38))
            p_ext = noisy_causality_extractor(p_ext, cfg_n)
        secondary_rep = _clean_replier()
        s_ext = causality_extractor_from_replier(secondary_rep)
        combiner = PathUnionCausalityExtractor(
            primary=p_ext, secondary=s_ext,
            mode=cfg.path_union_mode)
        return combiner, {
            "primary_replier_stats": primary_rep.stats,
            "secondary_replier_stats": secondary_rep.stats,
            "path_union": combiner,
        }

    # 2. No path-union. Either single replier or reply ensemble.
    primary_rep = (_biased_replier() if biased
                    else _clean_replier())
    diag: dict = {"primary_replier_stats": primary_rep.stats}
    if has_reply_ens:
        # Phase-37 EnsembleReplier (dual_agree) with secondary =
        # clean scenario-aware replier. Above-the-noise combiner
        # lives *within* EnsembleReplier's _dual_agree; below-the-
        # noise wrapper still defeats it per P37-4 — intentionally
        # kept to show this cleanly on the ``adv_drop_root`` cell.
        secondary_rep = _clean_replier()
        ensemble = EnsembleReplier(
            mode=MODE_DUAL_AGREE, primary=primary_rep,
            secondary=secondary_rep)
        ext = causality_extractor_from_ensemble(ensemble)
        diag["reply_ensemble"] = ensemble
        diag["secondary_replier_stats"] = secondary_rep.stats
    else:
        ext = causality_extractor_from_replier(primary_rep)

    # Apply any reply-axis noise wrapper (``adv_drop_root`` /
    # ``mislabel`` / ``drop``).
    adv_spec = {k: v for k, v in cell.items()
                 if not k.startswith("_")}
    rep_rep = None
    if adv_spec.get("adversarial"):
        rep_rep = ReplyCorruptionReport()
        adv = AdversarialReplyConfig(
            target_mode=adv_spec["adversarial"],
            budget=adv_spec.get("budget", 1))
        ext = adversarial_reply_extractor(ext, adv, report=rep_rep)
    elif adv_spec:
        rep_rep = ReplyCorruptionReport()
        cfg_n = ReplyNoiseConfig(
            drop_prob=adv_spec.get("drop_prob", 0.0),
            mislabel_prob=adv_spec.get("mislabel_prob", 0.0),
            seed=adv_spec.get("seed", 38))
        ext = noisy_causality_extractor(ext, cfg_n,
                                          report=rep_rep)
    if rep_rep is not None:
        diag["reply_corruption"] = rep_rep
    return ext, diag


# =============================================================================
# Per-scenario, per-config runner
# =============================================================================


def _run_one_config_cell(bank,
                           cfg: ConfigSpec, cell_name: str,
                           cell: dict, seed: int, strategies,
                           auditor,
                           max_events_in_prompt: int = 200,
                           inbox_capacity: int = 32):
    """Run the full contested loop for this (cfg, cell).

    For each (scenario, strategy) we construct a *fresh*
    ``claim_extractor`` and a fresh causality extractor. The
    reason: ``run_contested_loop`` calls
    ``run_contested_handoff_protocol`` twice per scenario under
    dynamic strategies (once for the non-dynamic pre-snapshot,
    once inside the dynamic branch); if we share a stateful
    adversary across those calls its budget is consumed by the
    first call and the dynamic branch sees no damage. Running per
    (scenario, strategy) with a fresh adversary keeps the
    Phase-37 budget-1 adversarial semantics intact.
    """
    per_scenario_pooled: dict[str, dict] = {}
    per_scenario_diag: list[dict] = []
    for scenario in bank:
        # Per-strategy, fresh adversary + fresh causality extractor.
        per_strat_meas: dict[str, list] = {}
        collected_diag: dict = {}
        for strat in strategies:
            claim_ext, claim_diag = _build_claim_extractor(
                cfg, cell, scenario.scenario_id)
            causal_ext, causal_diag = _build_causality_extractor(
                cfg, cell)
            rep = run_contested_loop(
                [scenario], auditor, strategies=(strat,),
                seed=seed,
                max_events_in_prompt=max_events_in_prompt,
                inbox_capacity=inbox_capacity,
                causality_extractor=causal_ext,
                claim_extractor=claim_ext,
            )
            for m in rep.measurements:
                per_strat_meas.setdefault(m.strategy, []).append(m)
            # Collect diag from the first strategy — the adversary
            # budget semantics are per-strategy so we only record
            # from one run. (The primary value of diagnostics here
            # is the ensemble / path-union stats which are also
            # per-run; we pick dynamic's if available.)
            if strat == (strategies[0] if strategies
                          else None) or not collected_diag:
                collected_diag = {
                    "claim": claim_diag, "causal": causal_diag}
        for strat, ms in per_strat_meas.items():
            acc = per_scenario_pooled.setdefault(strat, {
                "n": 0, "n_correct": 0,
                "n_contested": 0, "n_contested_correct": 0,
                "n_handoffs_delivered_total": 0,
            })
            for m in ms:
                acc["n"] += 1
                if m.grading["full_correct"]:
                    acc["n_correct"] += 1
                if m.contested:
                    acc["n_contested"] += 1
                    if m.grading["full_correct"]:
                        acc["n_contested_correct"] += 1
                acc["n_handoffs_delivered_total"] += \
                    m.n_handoffs_delivered
        diag_row = {
            "scenario_id": scenario.scenario_id,
            "config": cfg.name, "cell": cell_name,
        }
        claim_diag = collected_diag.get("claim", {})
        causal_diag = collected_diag.get("causal", {})
        if claim_diag.get("ext_union") is not None:
            diag_row["ext_union_stats"] = claim_diag[
                "ext_union"].stats()
        if claim_diag.get("ext_secondary") is not None:
            diag_row["ext_narrative_stats"] = claim_diag[
                "ext_secondary"].stats()
        if "adversary" in claim_diag and callable(
                claim_diag["adversary"]):
            try:
                diag_row["ext_adversary_stats"] = \
                    claim_diag["adversary"]()
            except TypeError:
                pass
        if "reply_ensemble" in causal_diag:
            diag_row["reply_ensemble_stats"] = \
                causal_diag["reply_ensemble"].stats.as_dict()
        if "path_union" in causal_diag:
            diag_row["path_union_stats"] = \
                causal_diag["path_union"].stats.as_dict()
        if "reply_corruption" in causal_diag:
            diag_row["reply_corruption_stats"] = \
                causal_diag["reply_corruption"].as_dict()
        per_scenario_diag.append(diag_row)
    # Normalise into accuracy means.
    pooled_means: dict[str, dict] = {}
    for strat, acc in per_scenario_pooled.items():
        n = max(1, acc["n"])
        ncn = max(1, acc["n_contested"])
        pooled_means[strat] = {
            "n": acc["n"],
            "accuracy_full": round(acc["n_correct"] / n, 4),
            "n_contested": acc["n_contested"],
            "contested_accuracy_full": round(
                acc["n_contested_correct"] / ncn, 4),
            "n_handoffs_delivered_total": acc[
                "n_handoffs_delivered_total"],
        }
    return pooled_means, per_scenario_diag


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int,
                      default=[35, 36])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--configs", nargs="+",
                      default=[c.name for c in CONFIGS])
    ap.add_argument("--cells", nargs="+",
                      default=[n for (n, _) in CELLS])
    ap.add_argument("--strategies", nargs="+",
                      default=[STRATEGY_DYNAMIC,
                                STRATEGY_ADAPTIVE_SUB,
                                STRATEGY_STATIC_HANDOFF])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor = MockContestedAuditor()
    cfg_by_name = {c.name: c for c in CONFIGS}
    cell_by_name = dict(CELLS)

    t0 = time.time()
    rows: list[dict] = []

    print(f"\n[phase38-A] configs={args.configs} "
          f"cells={args.cells}", flush=True)

    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            for cell_name in args.cells:
                cell = cell_by_name[cell_name]
                for cfg_name in args.configs:
                    cfg = cfg_by_name[cfg_name]
                    pooled, diag = _run_one_config_cell(
                        bank, cfg, cell_name, cell, seed,
                        args.strategies, auditor)
                    row = {
                        "distractors_per_role": k,
                        "seed": seed,
                        "config": cfg_name,
                        "cell": cell_name,
                        "pooled": pooled,
                        "diag": diag,
                    }
                    rows.append(row)
                    summary = " | ".join(
                        f"{s}: acc={p['accuracy_full']:.3f} "
                        f"contest={p['contested_accuracy_full']:.3f}"
                        for s, p in sorted(pooled.items()))
                    print(f"[phase38-A] k={k} seed={seed} "
                          f"cell={cell_name:22} "
                          f"cfg={cfg_name:24}  {summary}",
                          flush=True)

    wall = time.time() - t0

    # Pooled over seeds and distractor counts, per (cell, config).
    pool: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        k = (r["cell"], r["config"])
        pool.setdefault(k, []).append(r)
    pooled_out: list[dict] = []
    print()
    print("=" * 82)
    print("PHASE 38 PART A — POOLED (cell, config) over seeds / k")
    print("=" * 82)
    for (cell, cfg_name), rs in sorted(pool.items()):
        strat_means: dict[str, dict] = {}
        for strat in args.strategies:
            accs = []
            conts = []
            for r in rs:
                p = r["pooled"].get(strat)
                if not p:
                    continue
                accs.append(p["accuracy_full"])
                conts.append(p["contested_accuracy_full"])
            if accs:
                strat_means[strat] = {
                    "n_points": len(accs),
                    "accuracy_full_mean": round(
                        sum(accs) / len(accs), 4),
                    "contested_accuracy_full_mean": round(
                        sum(conts) / max(1, len(conts)), 4),
                }
        print(f"  cell={cell:22} config={cfg_name:24}")
        for strat, p in strat_means.items():
            print(f"    {strat:>14}  acc={p['accuracy_full_mean']:.3f}  "
                  f"contest={p['contested_accuracy_full_mean']:.3f}  "
                  f"(n={p['n_points']})")
        pooled_out.append({
            "cell": cell, "config": cfg_name,
            "per_strategy_means": strat_means,
        })

    payload = {
        "config": vars(args),
        "cells": list(cell_by_name.keys()),
        "configs": [c.name for c in CONFIGS],
        "rows": rows,
        "pooled": pooled_out,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    print(f"\n[phase38-A] wall = {wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
