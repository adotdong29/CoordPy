"""Phase 47 — bundle-aware capsule admission learning.

Phase 46 (``phase46_capsule_learning``) shipped the per-capsule
admission learning result: a logistic-regression policy on
capsule headers beat every non-learned heuristic on admit-
precision at tight budgets. The downstream decoder on the
noise-poisoned Phase-31 bundle remained stuck at 0.225 full
accuracy — the honest next-step P46-C1 ("bundle-aware admission
closes the noise ceiling").

This Phase 47 driver tests P46-C1 directly. It evaluates three
bundle-aware policies
(``CorroboratedAdmissionPolicy`` / ``PluralityBundlePolicy`` /
``BundleLearnedPolicy``) against the strongest Phase-46 baseline
(``LearnedAdmissionPolicy``) and the fixed heuristics
(``FIFOPolicy``, ``KindPriorityPolicy``) on the same
Phase-31 incident-triage benchmark under noisy extractors.

The headline metric is **decoder accuracy on the full admit
set** (not on the causal-slice oracle): the fraction of test
instances where the Phase-31 priority decoder's inferred
root_cause equals the gold. A win here — any lift over 0.225 —
is the empirical anchor of P46-C1.

Runs in ~15–20 s on stdlib Python. No external ML dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from typing import Any, Sequence

# Enforce deterministic dataset collection. ``noisy_extractor`` uses
# Python's built-in ``hash()`` for seed derivation, which is salted
# per-process unless PYTHONHASHSEED is fixed. This experiment's test
# numbers are not meaningful without reproducible dataset generation.
if os.environ.get("PYTHONHASHSEED") != "0":
    # Re-exec under PYTHONHASHSEED=0 if not already set. This is
    # the simplest way to guarantee deterministic dataset collection
    # without requiring every CLI caller to remember the flag.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv],
                os.environ)

from vision_mvp.tasks.incident_triage import (
    build_scenario_bank, handoff_is_relevant, run_handoff_protocol,
    _decoder_from_handoffs, ROLE_AUDITOR,
    extract_claims_for_role,
)
from vision_mvp.core.extractor_noise import (
    NoiseConfig, noisy_extractor, incident_triage_known_kinds,
)
from vision_mvp.wevra.capsule import (
    capsule_from_handoff,
)
from vision_mvp.wevra.capsule_policy import (
    BudgetedAdmissionLedger, FIFOPolicy, KindPriorityPolicy,
    LearnedAdmissionPolicy, SmallestFirstPolicy,
    train_admission_policy, ADMIT,
)
from vision_mvp.wevra.capsule_policy_bundle import (
    BundleStats, BundleLearnedPolicy, CorroboratedAdmissionPolicy,
    PluralityBundlePolicy, train_bundle_policy,
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
)


def _decoder_aware_labels(instance):
    """Stricter labels than ``handoff_is_relevant``: label = 1 iff
    the capsule's implied root_cause equals the instance's gold.

    A capsule whose implied root_cause does NOT equal the gold is
    treated as label=0 regardless of whether it is causal by the
    Phase-31 oracle — because admitting it can only *shift* the
    priority decoder away from the gold (or leave it unchanged).
    This is a training objective directly aligned with decoder
    accuracy, at the cost of losing signal on causal claims that
    are correlated with the gold but don't themselves imply it
    (e.g. a LATENCY_SPIKE in a memory_leak scenario).
    """
    gold_rc = instance["gold_root_cause"]
    labels = []
    for c in instance["offered_capsules"]:
        md = c.metadata_dict()
        kind = md.get("claim_kind")
        implied = DEFAULT_CLAIM_TO_ROOT_CAUSE.get(
            kind, "") if isinstance(kind, str) else ""
        labels.append(1 if implied == gold_rc else 0)
    return labels


# ---------------------------------------------------------------------------
# Dataset assembly — same shape as Phase 46; each record keeps its offered
# set so bundle features are well-defined.
# ---------------------------------------------------------------------------


def _gold_root_cause(scenario) -> str:
    from vision_mvp.tasks.incident_triage import (
        _decoder_from_handoffs as _dec)
    from vision_mvp.core.role_handoff import TypedHandoff
    ts = []
    for (role, kind, payload, evids) in scenario.causal_chain:
        ts.append(TypedHandoff(
            handoff_id=0, source_role=role, source_agent_id=0,
            to_role=ROLE_AUDITOR, claim_kind=kind, payload=payload,
            source_event_ids=tuple(evids), round=1,
            payload_cid="0" * 64, prev_chain_hash="",
            chain_hash="", emitted_at=0.0,
        ))
    return _dec(ts)["root_cause"]


def collect_dataset(seeds: Sequence[int] = (31, 32, 33, 34, 35,
                                              36, 37, 38, 39, 40),
                    distractor_grid: Sequence[int] = (
                        6, 20, 60, 120),
                    spurious_prob: float = 0.30,
                    mislabel_prob: float = 0.10,
                    ) -> tuple[list[dict[str, Any]],
                                list[dict[str, Any]]]:
    """Build per-capsule records + per-instance records.

    Returns:
      * ``records`` — one dict per capsule with instance_key,
        is_causal, capsule, handoff, gold_root_cause.
      * ``instances`` — one dict per (scenario, seed, k) instance
        with the full offered capsule list. This is what bundle-
        aware policies consume.
    """
    records: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    known_kinds = incident_triage_known_kinds()
    for seed in seeds:
        for k in distractor_grid:
            scenarios = build_scenario_bank(
                seed=seed, distractors_per_role=k)
            for scenario in scenarios:
                noise = NoiseConfig(
                    spurious_prob=spurious_prob,
                    mislabel_prob=mislabel_prob,
                    seed=seed * 17 + k)
                noisy = noisy_extractor(
                    extract_claims_for_role, known_kinds, noise)
                router = run_handoff_protocol(
                    scenario, extractor=noisy)
                inbox = router.inboxes.get(ROLE_AUDITOR)
                if inbox is None:
                    continue
                handoffs = list(inbox.peek())
                gold_root_cause = _gold_root_cause(scenario)
                offered_capsules = []
                instance_key = (
                    f"{scenario.scenario_id}/k{k}/s{seed}")
                for h in handoffs:
                    cap = capsule_from_handoff(h)
                    cap = dataclasses.replace(cap, n_tokens=h.n_tokens)
                    offered_capsules.append(cap)
                    records.append({
                        "scenario_id": scenario.scenario_id,
                        "k": k, "seed": seed,
                        "instance_key": instance_key,
                        "capsule": cap, "handoff": h,
                        "is_causal": int(
                            handoff_is_relevant(h, scenario)),
                        "gold_root_cause": gold_root_cause,
                    })
                instances.append({
                    "instance_key": instance_key,
                    "scenario_id": scenario.scenario_id,
                    "k": k, "seed": seed,
                    "offered_capsules": offered_capsules,
                    "offered_handoffs": handoffs,
                    "is_causal": [
                        int(handoff_is_relevant(h, scenario))
                        for h in handoffs],
                    "gold_root_cause": gold_root_cause,
                })
    return records, instances


# ---------------------------------------------------------------------------
# Evaluation — per-instance decoder accuracy under a policy + budget.
# ---------------------------------------------------------------------------


def _eval_instance(instance, policy, budget):
    capsules = instance["offered_capsules"]
    handoffs = instance["offered_handoffs"]
    is_causal = instance["is_causal"]
    gold_rc = instance["gold_root_cause"]
    bal = BudgetedAdmissionLedger(
        budget_tokens=budget, policy=policy)
    bal.offer_all_batched(capsules)
    admitted_cids = {d.capsule_cid for d in bal.decisions
                      if d.decision == ADMIT}
    admitted_h = [h for h, c in zip(handoffs, capsules)
                   if c.cid in admitted_cids]
    admitted_causal_h = [h for h, c, l in
                          zip(handoffs, capsules, is_causal)
                          if c.cid in admitted_cids and l]
    decoded = _decoder_from_handoffs(admitted_h)
    decoded_clean = _decoder_from_handoffs(admitted_causal_h)
    n_admit = len(admitted_h)
    n_admit_causal = len(admitted_causal_h)
    n_offered_causal = sum(is_causal)
    return {
        "decoded_root_cause": decoded["root_cause"],
        "decoded_clean_root_cause": decoded_clean["root_cause"],
        "gold_root_cause": gold_rc,
        "decoder_correct": int(decoded["root_cause"] == gold_rc),
        "decoder_correct_clean": int(
            decoded_clean["root_cause"] == gold_rc),
        "n_admitted": n_admit,
        "n_admitted_causal": n_admit_causal,
        "n_offered": len(capsules),
        "n_offered_causal": n_offered_causal,
        "budget_used": bal.budget_used,
        "budget_utilization": (
            bal.budget_used / budget if budget > 0 else 0.0),
    }


def evaluate(instances, policy, budget):
    rows = [_eval_instance(inst, policy, budget) for inst in instances]
    if not rows:
        return None
    n = len(rows)
    n_correct = sum(r["decoder_correct"] for r in rows)
    n_correct_clean = sum(r["decoder_correct_clean"] for r in rows)
    n_admit = sum(r["n_admitted"] for r in rows)
    n_admit_causal = sum(r["n_admitted_causal"] for r in rows)
    n_offer = sum(r["n_offered"] for r in rows)
    n_offer_causal = sum(r["n_offered_causal"] for r in rows)
    mean_util = sum(r["budget_utilization"] for r in rows) / n
    return {
        "policy": policy.name,
        "budget": budget,
        "n_instances": n,
        "decoder_accuracy": n_correct / n,
        "decoder_accuracy_clean": n_correct_clean / n,
        "policy_precision": (
            n_admit_causal / n_admit if n_admit else 0.0),
        "policy_recall": (
            n_admit_causal / n_offer_causal
            if n_offer_causal else 0.0),
        "mean_n_admitted": n_admit / n,
        "mean_n_offered": n_offer / n,
        "mean_n_offered_causal": n_offer_causal / n,
        "mean_budget_utilization": mean_util,
    }


def split_instances_by_seed(instances, train_frac=0.8, seed=0):
    seeds = sorted({inst["seed"] for inst in instances})
    rng = random.Random(seed)
    rng.shuffle(seeds)
    n_train = max(1, int(len(seeds) * train_frac))
    train_seeds = set(seeds[:n_train])
    test_seeds = set(seeds[n_train:])
    train = [inst for inst in instances
              if inst["seed"] in train_seeds]
    test = [inst for inst in instances
             if inst["seed"] in test_seeds]
    return train, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_phase47(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        budget_grid: Sequence[int] = (
            16, 32, 48, 64, 96, 128, 256),
        seeds: Sequence[int] = (31, 32, 33, 34, 35,
                                36, 37, 38, 39, 40),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
        min_sources_grid: Sequence[int] = (2,),
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[phase47] collecting dataset (seeds={list(seeds)}, "
          f"k ∈ {list(distractor_grid)})…")
    records, instances = collect_dataset(
        seeds=seeds, distractor_grid=distractor_grid,
        spurious_prob=spurious_prob,
        mislabel_prob=mislabel_prob)
    n_total = len(records)
    n_causal = sum(r["is_causal"] for r in records)
    print(f"[phase47]   {n_total} capsules, {n_causal} causal "
          f"({100 * n_causal / max(1, n_total):.1f} %); "
          f"{len(instances)} instances")
    train_inst, test_inst = split_instances_by_seed(
        instances, seed=seeds[0])
    print(f"[phase47]   train instances: {len(train_inst)}, "
          f"test instances: {len(test_inst)}")

    # Train per-capsule learned (Phase 46 baseline).
    train_cap_examples = [
        (r["capsule"], r["is_causal"])
        for r in records
        if any(inst["instance_key"] == r["instance_key"]
               for inst in train_inst)]
    print(f"[phase47] training LearnedAdmissionPolicy (Phase 46) on "
          f"{len(train_cap_examples)} per-capsule examples…")
    learned = train_admission_policy(
        train_cap_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    learned.name = "learned (p46)"

    # Train bundle-aware learned policy with the per-capsule
    # "is_causal" gold label (this is the P46-style objective
    # extended with bundle features — training signal is
    # classifier-style, not decoder-style).
    print(f"[phase47] training BundleLearnedPolicy (causal labels)"
          f" on {len(train_inst)} train instances…")
    train_bundle_examples = [
        (inst["offered_capsules"], inst["offered_capsules"],
         inst["is_causal"])
        for inst in train_inst]
    bundle_learned = train_bundle_policy(
        train_bundle_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    bundle_learned.name = "bundle_learned(causal)"

    # Also train a bundle-aware policy with decoder-aware labels
    # (implied_root_cause == gold). This is the stricter
    # objective that directly targets decoder accuracy.
    print(f"[phase47] training BundleLearnedPolicy (decoder-aware "
          f"labels) on {len(train_inst)} train instances…")
    train_bundle_dec_examples = [
        (inst["offered_capsules"], inst["offered_capsules"],
         _decoder_aware_labels(inst))
        for inst in train_inst]
    bundle_learned_dec = train_bundle_policy(
        train_bundle_dec_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    bundle_learned_dec.name = "bundle_learned(dec)"

    # Top weights (for the writeup).
    top_weights = sorted(
        bundle_learned.weights.items(),
        key=lambda kv: -abs(kv[1]))[:15]
    print("[phase47]   top |bundle weights|:")
    for k, w in top_weights:
        print(f"      {k:32s}  {w:+.4f}")

    # Policy roster.
    policies: list[Any] = [
        FIFOPolicy(),
        KindPriorityPolicy(cutoff=8),
        learned,
    ]
    # One CorroboratedAdmissionPolicy per min_sources.
    for ms in min_sources_grid:
        cp = CorroboratedAdmissionPolicy(
            min_sources=ms, inner_policy=learned)
        cp.name = f"corroborated(ms={ms},inner=learned)"
        policies.append(cp)
    # PluralityBundlePolicy.
    pbp = PluralityBundlePolicy()
    policies.append(pbp)
    # Learned bundle policies.
    policies.append(bundle_learned)
    policies.append(bundle_learned_dec)

    # Sweep budgets × policies.
    print(f"[phase47] sweeping budgets {list(budget_grid)} × "
          f"{len(policies)} policies…")
    test_results: list[dict] = []
    train_results: list[dict] = []
    for B in budget_grid:
        for pol in policies:
            row_t = evaluate(test_inst, pol, B)
            if row_t is not None:
                row_t["split"] = "test"
                test_results.append(row_t)
            row_tr = evaluate(train_inst, pol, B)
            if row_tr is not None:
                row_tr["split"] = "train"
                train_results.append(row_tr)

    out = {
        "schema": "wevra.phase47.bundle_learning.v1",
        "n_records": n_total, "n_causal_records": n_causal,
        "n_train_instances": len(train_inst),
        "n_test_instances": len(test_inst),
        "policies": [p.name for p in policies],
        "budgets": list(budget_grid),
        "distractor_grid": list(distractor_grid),
        "spurious_prob": spurious_prob,
        "mislabel_prob": mislabel_prob,
        "test_results": test_results,
        "train_results": train_results,
        "top_weights": [{"feature": k, "weight": w}
                          for k, w in top_weights],
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, "results_phase47_bundle_learning.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase47] wrote {out_path} ({out['wall_seconds']} s)")
    _print_summary(test_results, "TEST")
    _print_summary(train_results, "TRAIN")
    return out


def _print_summary(rows, split_name):
    if not rows:
        print(f"\n[phase47] === {split_name}: (no rows)")
        return
    policies = list({r["policy"]: None for r in rows}.keys())
    budgets = sorted({r["budget"] for r in rows})
    for metric_key, metric_label in (
            ("decoder_accuracy",
             "decoder accuracy (FULL admit set — P46-C1 anchor)"),
            ("decoder_accuracy_clean",
             "decoder accuracy on causal slice of admit"),
            ("policy_precision", "admit-precision"),
            ("policy_recall", "admit-recall"),
            ("mean_budget_utilization", "budget utilization"),
    ):
        print(f"\n[phase47] === {split_name}: {metric_label} ===")
        hdr = "  budget | " + " | ".join(
            f"{p:>34s}" for p in policies)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for B in budgets:
            cells = []
            for p in policies:
                row = next((r for r in rows
                             if r["policy"] == p
                             and r["budget"] == B), None)
                cells.append(
                    f"{row[metric_key]:6.3f}" if row else "     ?")
            print(f"  {B:>6d} | " + " | ".join(
                f"{c:>34s}" for c in cells))


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 47: bundle-aware capsule admission "
                     "learning (P46-C1 anchor)")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seeds", nargs="+", type=int,
                    default=[31, 32, 33, 34, 35,
                             36, 37, 38, 39, 40])
    p.add_argument("--budgets", nargs="+", type=int,
                    default=[16, 32, 48, 64, 96, 128, 256])
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    args = p.parse_args()
    run_phase47(
        out_dir=args.out_dir, seeds=tuple(args.seeds),
        budget_grid=tuple(args.budgets),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob)


if __name__ == "__main__":
    _cli()
