"""Phase 46 (Capsule research milestone) — capsule admission policy
learning under bounded budget.

This experiment is the empirical anchor for Conjecture W3-C4
("admission policy is learnable") in
``docs/CAPSULE_FORMALISM.md``. It uses the Phase-31 incident-triage
benchmark as a real, on-disk task family with an oracle for handoff
relevance (``handoff_is_relevant``) — that oracle is the *gold
label* the learned admission policy is trained to recover from
header-level capsule features alone.

Setup
=====

1. **Dataset.** For each scenario in the Phase-31 bank with
   distractor density ``k`` swept across ``{6, 20, 60, 120}``, we
   run the typed-handoff substrate (``run_handoff_protocol``),
   collect every emitted ``TypedHandoff`` (causal *and*
   distractor), and lift each to a ``HANDOFF`` capsule via
   ``capsule_from_handoff``. The Phase-31 oracle
   ``handoff_is_relevant`` provides the gold (1 = causal,
   0 = distractor).

2. **Train / test split.** Scenarios are partitioned by
   ``scenario_id`` into 80% train / 20% test. The split is
   deterministic across seeds.

3. **Policies compared.**
     * ``FIFOPolicy`` — accept until budget exhausted (the SDK-v3
       default).
     * ``KindPriorityPolicy`` — admit by hand-coded priority over
       claim_kind (the strongest non-learned baseline; bakes in
       the same domain knowledge the substrate decoder uses).
     * ``SmallestFirstPolicy`` — admit smallest first (greedy
       budget; uninformed about kind).
     * ``LearnedAdmissionPolicy`` — logistic regression over the
       closed feature vocabulary in
       ``vision_mvp/wevra/capsule_policy.py``; trained on the
       train split using ``train_admission_policy``.

4. **Metric.** For a budget ``B`` (in tokens), each policy
   produces an admitted set. The *downstream metric* is the
   accuracy of the Phase-31 ``_decoder_from_handoffs`` on the
   admitted-set bundle: did the auditor still recover the gold
   root-cause label?

5. **Sweep.** Budgets ``B ∈ {16, 32, 64, 96, 128, 256}`` tokens.
   At each budget × policy cell we report:
     * ``policy_precision``  — admit-set precision against the
       relevance gold (sanity).
     * ``policy_recall``     — admit-set recall against the
       relevance gold (sanity).
     * ``decoder_accuracy``  — fraction of test scenarios where
       the decoder's root_cause matches the gold.
     * ``mean_n_admitted``   — mean admitted-set size at this cell.

A policy that wins on ``decoder_accuracy`` at *tighter* budgets
than the heuristic strictly dominates the heuristic on the
substrate's downstream goal. That is the falsifier criterion for
Conjecture W3-C4.

Output
======

Writes ``results_phase46_capsule_learning.json`` to the working
directory by default. Run with::

    python -m vision_mvp.experiments.phase46_capsule_learning \\
        --out-dir /tmp/wevra_phase46

Reproducible — uses only stdlib + numpy-free Python (``random``,
``math``, ``json``).
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

from vision_mvp.tasks.incident_triage import (
    build_scenario_bank, handoff_is_relevant, run_handoff_protocol,
    _decoder_from_handoffs, ROLE_AUDITOR,
    extract_claims_for_role,
)
from vision_mvp.core.extractor_noise import (
    NoiseConfig, noisy_extractor, incident_triage_known_kinds,
)
from vision_mvp.wevra.capsule import (
    capsule_from_handoff, ContextCapsule,
)
from vision_mvp.wevra.capsule_policy import (
    BudgetedAdmissionLedger, FIFOPolicy, KindPriorityPolicy,
    LearnedAdmissionPolicy, SmallestFirstPolicy,
    train_admission_policy, ADMIT,
)


# =============================================================================
# Dataset assembly — capsules + gold labels
# =============================================================================


def _gold_root_cause(scenario) -> str:
    """Return the canonical root_cause label the decoder would
    produce given the gold causal_chain. Mirrors the priority list
    in ``_decoder_from_handoffs`` but operates on the gold pairs
    directly so the test does not depend on which distractors
    happened to extract."""
    from vision_mvp.tasks.incident_triage import (
        _decoder_from_handoffs as _dec)
    # Synthesise minimal handoffs that carry only the gold claims.
    # The decoder reads ``claim_kind`` and the payload's
    # service=… tokens; the gold tokens live on the causal events.
    # Easiest: build minimal handoffs from causal_chain triples.
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


def collect_dataset(seeds: Sequence[int] = (31, 32, 33, 34, 35),
                    distractor_grid: Sequence[int] = (6, 20, 60, 120),
                    spurious_prob: float = 0.30,
                    mislabel_prob: float = 0.10,
                    ) -> list[dict[str, Any]]:
    """For each (seed × scenario × k) build the handoff bundle
    under a noisy extractor and emit one record per delivered
    handoff with its gold relevance label and its lifted capsule.

    The noisy extractor (Phase-32 ``noisy_extractor`` wrapper)
    injects ``spurious_prob``-fraction-of-distractor-events worth
    of structurally-valid but causally-irrelevant claims —
    creating the real causal/non-causal mixture the learned
    admission policy must discriminate. Without this layer, the
    Phase-31 regex extractor produces a 100%-causal stream and
    admission has nothing to learn.
    """
    records: list[dict[str, Any]] = []
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
                for h in handoffs:
                    cap = capsule_from_handoff(h)
                    cap = dataclasses.replace(cap, n_tokens=h.n_tokens)
                    records.append({
                        "scenario_id": scenario.scenario_id,
                        "k": k,
                        "seed": seed,
                        "instance_key": (
                            f"{scenario.scenario_id}/k{k}/s{seed}"),
                        "capsule": cap,
                        "handoff": h,
                        "is_causal": int(
                            handoff_is_relevant(h, scenario)),
                        "gold_root_cause": gold_root_cause,
                    })
    return records


# =============================================================================
# Policy evaluation under a budget
# =============================================================================


def _decoder_acc_under_admit(scenario_records, policy, budget):
    """For one scenario's handoff bundle, run the policy under the
    given token budget, and check whether the decoder's
    root_cause matches the gold.

    Also reports an "oracle-clean" decoder accuracy: the decoder's
    root_cause when restricted to the *causal* slice of the
    admitted set. This separates two distinct failure modes:

      * **admission failure** — the policy admitted the wrong
        capsules (signaled by low admit-precision + low oracle-
        clean accuracy).
      * **bundle poisoning** — the admit set contains spurious
        high-priority claims that swamp the decoder's priority
        rule (signaled by HIGH oracle-clean accuracy but LOW
        full-bundle accuracy at high budgets).
    """
    capsules = [r["capsule"] for r in scenario_records]
    handoffs = [r["handoff"] for r in scenario_records]
    is_causal = [r["is_causal"] for r in scenario_records]
    gold_rc = scenario_records[0]["gold_root_cause"]
    bal = BudgetedAdmissionLedger(budget_tokens=budget, policy=policy)
    bal.offer_all_batched(capsules)
    admitted_cids = {d.capsule_cid for d in bal.decisions
                      if d.decision == ADMIT}
    admitted_handoffs = [h for h, c in zip(handoffs, capsules)
                          if c.cid in admitted_cids]
    admitted_causal_handoffs = [h for h, c, l in
                                  zip(handoffs, capsules, is_causal)
                                  if c.cid in admitted_cids and l]
    decoded = _decoder_from_handoffs(admitted_handoffs)
    decoded_clean = _decoder_from_handoffs(admitted_causal_handoffs)
    n_admit = len(admitted_handoffs)
    n_admit_causal = len(admitted_causal_handoffs)
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
        "n_offered_causal": sum(is_causal),
        "policy": policy.name,
        "budget": budget,
    }


def evaluate_policy_at_budget(records, instance_keys, policy, budget):
    """Aggregate per-instance decoder accuracy across a set of
    instances (a (scenario, seed, k) triple is one instance)."""
    by_instance: dict[str, list[dict]] = {}
    for r in records:
        by_instance.setdefault(r["instance_key"], []).append(r)
    rows = []
    for sid in instance_keys:
        srs = by_instance.get(sid, [])
        if not srs:
            continue
        rows.append(_decoder_acc_under_admit(srs, policy, budget))
    if not rows:
        return None
    n = len(rows)
    n_correct = sum(r["decoder_correct"] for r in rows)
    n_correct_clean = sum(r["decoder_correct_clean"] for r in rows)
    n_admit = sum(r["n_admitted"] for r in rows)
    n_admit_causal = sum(r["n_admitted_causal"] for r in rows)
    n_offer = sum(r["n_offered"] for r in rows)
    n_offer_causal = sum(r["n_offered_causal"] for r in rows)
    return {
        "policy": policy.name,
        "budget": budget,
        "n_scenarios": n,
        "decoder_accuracy": n_correct / n,
        "decoder_accuracy_clean": n_correct_clean / n,
        "policy_precision": (n_admit_causal / n_admit) if n_admit else 0.0,
        "policy_recall": (
            n_admit_causal / n_offer_causal if n_offer_causal else 0.0),
        "mean_n_admitted": n_admit / n,
        "mean_n_offered": n_offer / n,
        "mean_n_offered_causal": n_offer_causal / n,
    }


# =============================================================================
# Train / test split — by scenario_id
# =============================================================================


def split_instances(records, train_frac: float = 0.8, seed: int = 0,
                    ) -> tuple[list[str], list[str]]:
    """Hold-out by *seed* — every scenario_id appears in both train
    and test, but at *different* seeds. This tests whether the
    learned policy generalises across distractor draws of the same
    scenario family, not whether it memorises one scenario."""
    instance_keys = sorted({r["instance_key"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(seeds)
    n_train_seeds = max(1, int(len(seeds) * train_frac))
    train_seeds = set(seeds[:n_train_seeds])
    test_seeds = set(seeds[n_train_seeds:])
    train_keys = [k for k in instance_keys
                   if any(k.endswith(f"/s{s}") for s in train_seeds)]
    test_keys = [k for k in instance_keys
                  if any(k.endswith(f"/s{s}") for s in test_seeds)]
    return train_keys, test_keys


# =============================================================================
# Main entry
# =============================================================================


def run_phase46(out_dir: str = ".",
                distractor_grid: Sequence[int] = (6, 20, 60, 120),
                budget_grid: Sequence[int] = (16, 32, 48, 64, 96, 128, 256),
                seeds: Sequence[int] = (31, 32, 33, 34, 35, 36, 37, 38, 39, 40),
                spurious_prob: float = 0.30,
                mislabel_prob: float = 0.10,
                ) -> dict[str, Any]:
    t0 = time.time()
    print(f"[phase46] collecting dataset "
          f"(seeds={list(seeds)}, k ∈ {list(distractor_grid)})…")
    records = collect_dataset(
        seeds=seeds, distractor_grid=distractor_grid,
        spurious_prob=spurious_prob, mislabel_prob=mislabel_prob)
    n_total = len(records)
    n_causal = sum(r["is_causal"] for r in records)
    print(f"[phase46]   {n_total} capsules, {n_causal} causal "
          f"({100*n_causal/max(1,n_total):.1f} %)")
    train_ids, test_ids = split_instances(records, seed=seeds[0])
    print(f"[phase46]   train instances: {len(train_ids)}, "
          f"test instances: {len(test_ids)}")
    train_examples = [(r["capsule"], r["is_causal"]) for r in records
                       if r["instance_key"] in train_ids]

    print(f"[phase46] training LearnedAdmissionPolicy on "
          f"{len(train_examples)} examples…")
    learned = train_admission_policy(
        train_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    # Top weights for the writeup.
    top_weights = sorted(
        learned.weights.items(), key=lambda kv: -abs(kv[1]))[:12]
    print("[phase46]   top |weights|:")
    for k, w in top_weights:
        print(f"      {k:24s}  {w:+.4f}")

    policies = [
        FIFOPolicy(),
        SmallestFirstPolicy(),
        KindPriorityPolicy(cutoff=4),
        KindPriorityPolicy(cutoff=8),
        learned,
    ]
    # Rename the second KindPriority for clarity in the table.
    policies[2].name = "kind_priority(top4)"
    policies[3].name = "kind_priority(top8)"

    print(f"[phase46] sweeping budgets {list(budget_grid)} × "
          f"{len(policies)} policies on {len(test_ids)} test scenarios "
          f"(also reporting train-set accuracy for diagnosis)…")
    test_results: list[dict] = []
    train_results: list[dict] = []
    for B in budget_grid:
        for pol in policies:
            row_test = evaluate_policy_at_budget(
                records, test_ids, pol, B)
            if row_test is not None:
                row_test["split"] = "test"
                test_results.append(row_test)
            row_tr = evaluate_policy_at_budget(
                records, train_ids, pol, B)
            if row_tr is not None:
                row_tr["split"] = "train"
                train_results.append(row_tr)

    out = {
        "schema": "wevra.phase46.capsule_learning.v1",
        "n_records": n_total,
        "n_causal_records": n_causal,
        "train_scenarios": sorted(train_ids),
        "test_scenarios": sorted(test_ids),
        "policies": [p.name for p in policies],
        "budgets": list(budget_grid),
        "distractor_grid": list(distractor_grid),
        "test_results": test_results,
        "train_results": train_results,
        "learned_policy": learned.to_dict(),
        "top_weights": [
            {"feature": k, "weight": w} for k, w in top_weights],
        "wall_seconds": round(time.time() - t0, 3),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, "results_phase46_capsule_learning.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase46] wrote {out_path} ({out['wall_seconds']} s)")
    _print_summary_table(test_results, "TEST")
    _print_summary_table(train_results, "TRAIN")
    return out


def _print_summary_table(rows, split_name: str) -> None:
    if not rows:
        print(f"\n[phase46] === {split_name}: (no rows)")
        return
    policies = sorted({r["policy"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})

    def _table(metric_key, metric_label):
        print(f"\n[phase46] === {split_name}: {metric_label} ===")
        header = "  budget | " + " | ".join(f"{p:>22s}" for p in policies)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for B in budgets:
            cells = []
            for p in policies:
                row = next((r for r in rows
                             if r["policy"] == p and r["budget"] == B),
                            None)
                cells.append(
                    f"{row[metric_key]:6.3f}" if row else "     ?")
            print(f"  {B:>6d} | " + " | ".join(
                f"{c:>22s}" for c in cells))

    _table("decoder_accuracy_clean",
            "decoder accuracy on causal slice of admit (oracle-clean)")
    _table("decoder_accuracy",
            "decoder accuracy on full admit set (bundle-poisoned by noise)")
    _table("policy_precision",
            "admit-precision (admitted ∩ causal) / admitted")
    _table("policy_recall",
            "admit-recall    (admitted ∩ causal) / offered_causal")


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 46: capsule admission policy learning")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seeds", nargs="+", type=int,
                    default=[31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
    p.add_argument("--budgets", nargs="+", type=int,
                    default=[16, 32, 48, 64, 96, 128, 256])
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    args = p.parse_args()
    run_phase46(
        out_dir=args.out_dir,
        seeds=tuple(args.seeds),
        budget_grid=tuple(args.budgets),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob)


if __name__ == "__main__":
    _cli()
