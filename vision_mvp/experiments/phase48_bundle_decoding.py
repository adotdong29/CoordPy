"""Phase 48 — bundle-aware capsule DECODING.

Phase 46 and Phase 47 ran the per-capsule and bundle-aware
**admission** experiments and cleanly characterised the ceiling:
on the Phase-31 incident-triage bench under
``noisy_extractor(spurious_prob=0.30)`` the priority-decoder's
full-bundle accuracy is bounded by the **structural** ceiling
0.200 (Theorem P47-1), independent of admission policy. P46-C1
in its strong form — "bundle-aware admission lifts accuracy
past 0.30" — is **falsified** on this bench (Phase 47 § 2.4).

The Phase-47 note named the next research front:

> "The next paradigm-shift candidate on the capsule ML axis is
> bundle-aware DECODING (Conjecture P47-C1), not more admission
> work."

This driver is the first direct attack on P47-C1.  The
admitted set is fixed (we use a strong admission baseline for
every decoder under test); only the decoder changes.  Four
decoders:

  * ``priority``                — first-match over a priority
    order.  Baseline (the status-quo Phase-31 decoder).
  * ``plurality``               — argmax over implied-``root_cause``
    vote counts, ties broken by priority order.
  * ``src_corroborated_priority`` — priority with a
    ``min_sources = 2`` veto (lone-high-priority claims do not
    count toward the first match).
  * ``learned_bundle_decoder``  — multinomial logistic
    regression over class-agnostic bundle-shape features
    (votes, sources, priority-shape).  Trained on the
    ``(admitted_bundle, gold_root_cause)`` pairs from the train
    split's instances; weights are a single interpretable
    vector (10 floats).

Headline metric: **test-set decoder accuracy on the FULL admit
set** as a function of (admission policy, decoder, budget).

Runs deterministically in ~8–12 s with no external ML deps.

Theoretical anchor:
  * `docs/CAPSULE_FORMALISM.md` § 4.C, Theorems W3-17 / W3-18.
  * `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE3.md` § 2.
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

# Enforce deterministic dataset collection — ``noisy_extractor``
# uses Python ``hash()`` which is salted per-process unless
# PYTHONHASHSEED is set.
if os.environ.get("PYTHONHASHSEED") != "0":
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
    ContextCapsule, capsule_from_handoff,
)
from vision_mvp.wevra.capsule_policy import (
    BudgetedAdmissionLedger, FIFOPolicy, KindPriorityPolicy,
    LearnedAdmissionPolicy, train_admission_policy, ADMIT,
)
from vision_mvp.wevra.capsule_policy_bundle import (
    BundleLearnedPolicy, CorroboratedAdmissionPolicy,
    PluralityBundlePolicy, train_bundle_policy,
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    DEFAULT_HIGH_PRIORITY_CUTOFF,
)
from vision_mvp.wevra.capsule_decoder import (
    PriorityDecoder, PluralityDecoder,
    SourceCorroboratedPriorityDecoder,
    LearnedBundleDecoder, train_learned_bundle_decoder,
    BUNDLE_DECODER_FEATURES, evaluate_decoder,
)


# ---------------------------------------------------------------------------
# Dataset collection (mirrors phase47_bundle_learning for apples-to-apples).
# ---------------------------------------------------------------------------


def _gold_root_cause(scenario) -> str:
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
    return _decoder_from_handoffs(ts)["root_cause"]


def collect_dataset(seeds: Sequence[int] = (
                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40),
                    distractor_grid: Sequence[int] = (
                        6, 20, 60, 120),
                    spurious_prob: float = 0.30,
                    mislabel_prob: float = 0.10,
                    ) -> tuple[list[dict[str, Any]],
                                list[dict[str, Any]]]:
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
                gold_rc = _gold_root_cause(scenario)
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
                        "gold_root_cause": gold_rc,
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
                    "gold_root_cause": gold_rc,
                })
    return records, instances


def split_instances_by_seed(instances, train_frac=0.8, seed=0):
    seeds = sorted({inst["seed"] for inst in instances})
    rng = random.Random(seed)
    rng.shuffle(seeds)
    n_train = max(1, int(len(seeds) * train_frac))
    train_seeds = set(seeds[:n_train])
    train = [inst for inst in instances
              if inst["seed"] in train_seeds]
    test = [inst for inst in instances
             if inst["seed"] not in train_seeds]
    return train, test


# ---------------------------------------------------------------------------
# Admission: return the admitted subset under a policy + budget.
# ---------------------------------------------------------------------------


def admit(instance, policy, budget) -> list[ContextCapsule]:
    capsules = instance["offered_capsules"]
    bal = BudgetedAdmissionLedger(
        budget_tokens=budget, policy=policy)
    bal.offer_all_batched(capsules)
    admitted = {d.capsule_cid for d in bal.decisions
                 if d.decision == ADMIT}
    return [c for c in capsules if c.cid in admitted]


# ---------------------------------------------------------------------------
# Evaluation sweep.
# ---------------------------------------------------------------------------


def evaluate_cell(instances, admission_policy, decoder, budget):
    """Admit each instance under ``admission_policy @ budget``,
    then run ``decoder`` on the admitted bundle; return accuracy
    and per-scenario breakdown."""
    n = 0
    n_correct = 0
    per_scen_correct: dict[str, int] = {}
    per_scen_total: dict[str, int] = {}
    admit_sizes: list[int] = []
    admit_causal_counts: list[int] = []
    for inst in instances:
        admitted = admit(inst, admission_policy, budget)
        gold = inst["gold_root_cause"]
        out = decoder.decode(admitted)
        sid = inst["scenario_id"]
        per_scen_total[sid] = per_scen_total.get(sid, 0) + 1
        correct = int(out == gold)
        if correct:
            per_scen_correct[sid] = per_scen_correct.get(sid, 0) + 1
        n += 1
        n_correct += correct
        admit_sizes.append(len(admitted))
        causal_ids = {c.cid for c, lab
                       in zip(inst["offered_capsules"],
                              inst["is_causal"]) if lab}
        admit_causal_counts.append(
            sum(1 for c in admitted if c.cid in causal_ids))
    per_scenario = {
        sid: (per_scen_correct.get(sid, 0) / per_scen_total[sid])
        for sid in per_scen_total}
    return {
        "admission_policy": admission_policy.name,
        "decoder": decoder.name,
        "budget": budget,
        "n_instances": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "per_scenario_accuracy": per_scenario,
        "mean_n_admitted": (sum(admit_sizes) / n) if n else 0.0,
        "mean_n_admitted_causal": (
            sum(admit_causal_counts) / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# LearnedBundleDecoder training helper — build (admitted, gold_rc) pairs
# at one admission cell.
# ---------------------------------------------------------------------------


def training_bundles(instances, admission_policy, budget,
                     ) -> list[tuple[list[ContextCapsule], str]]:
    out: list[tuple[list[ContextCapsule], str]] = []
    for inst in instances:
        admitted = admit(inst, admission_policy, budget)
        out.append((admitted, inst["gold_root_cause"]))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_phase48(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        budget_grid: Sequence[int] = (16, 32, 48, 64, 96, 128, 256),
        seeds: Sequence[int] = (31, 32, 33, 34, 35,
                                36, 37, 38, 39, 40),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[phase48] collecting Phase-31 noisy dataset "
          f"(seeds={list(seeds)}, k ∈ {list(distractor_grid)})…")
    records, instances = collect_dataset(
        seeds=seeds, distractor_grid=distractor_grid,
        spurious_prob=spurious_prob,
        mislabel_prob=mislabel_prob)
    n_total = len(records)
    n_causal = sum(r["is_causal"] for r in records)
    print(f"[phase48]   {n_total} capsules, {n_causal} causal "
          f"({100 * n_causal / max(1, n_total):.1f} %); "
          f"{len(instances)} instances")
    train_inst, test_inst = split_instances_by_seed(
        instances, seed=seeds[0])
    print(f"[phase48]   train={len(train_inst)} test={len(test_inst)}")

    # Training admission policies (FIFO + learned + bundle-learned).
    print("[phase48] training Phase-46 LearnedAdmissionPolicy…")
    train_cap_examples = [
        (r["capsule"], r["is_causal"]) for r in records
        if any(inst["instance_key"] == r["instance_key"]
               for inst in train_inst)]
    learned_admit = train_admission_policy(
        train_cap_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    learned_admit.name = "learned(p46)"

    print("[phase48] training Phase-47 BundleLearnedPolicy (causal labels)…")
    train_bundle = [(inst["offered_capsules"],
                       inst["offered_capsules"],
                       inst["is_causal"])
                      for inst in train_inst]
    bundle_admit = train_bundle_policy(
        train_bundle, n_epochs=300, lr=0.5, l2=1e-3, seed=seeds[0])
    bundle_admit.name = "bundle_learned_admit"

    # Admission policy roster — we evaluate each decoder against
    # the strongest Phase-47 admission baseline by default (FIFO
    # gives admission-side null; learned_admit is the best per-
    # capsule; bundle_admit is the Phase-47 bundle-aware).
    admission_policies: list[Any] = [
        FIFOPolicy(),
        learned_admit,
        bundle_admit,
    ]

    # Train the LearnedBundleDecoder.  We use the widest budget
    # that saturates admission-recall so the training bundles
    # look most like the full offered set — the decoder should
    # learn shape robustly regardless of admit policy; to keep
    # the decoder "header-level only" we train on FIFO admission
    # at B = 256 so the decoder sees the full offered distribution.
    print("[phase48] training LearnedBundleDecoder on train split "
          "(FIFO @ B=256)…")
    fifo = FIFOPolicy()
    rc_alphabet = tuple(sorted({
        inst["gold_root_cause"] for inst in instances}))
    print(f"[phase48]   rc_alphabet = {rc_alphabet}")
    train_pairs = training_bundles(train_inst, fifo, 256)
    decoder_trained = train_learned_bundle_decoder(
        train_pairs,
        rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs=300, lr=0.5, l2=1e-3, seed=seeds[0])
    decoder_trained.name = "learned_bundle_decoder"
    print("[phase48]   trained bundle-decoder weights:")
    for k, w in sorted(decoder_trained.weights.items(),
                        key=lambda kv: -abs(kv[1])):
        print(f"      {k:28s}  {w:+.4f}")

    # Decoder roster.
    decoders: list[Any] = [
        PriorityDecoder(),
        PluralityDecoder(),
        SourceCorroboratedPriorityDecoder(min_sources=2),
        decoder_trained,
    ]

    # Sweep the grid: (admission × decoder × budget) on train + test.
    print(f"[phase48] sweeping {len(admission_policies)} admission × "
          f"{len(decoders)} decoders × {len(budget_grid)} budgets…")
    test_results: list[dict[str, Any]] = []
    train_results: list[dict[str, Any]] = []
    for B in budget_grid:
        for ap in admission_policies:
            for dec in decoders:
                row_t = evaluate_cell(test_inst, ap, dec, B)
                row_t["split"] = "test"
                test_results.append(row_t)
                row_tr = evaluate_cell(train_inst, ap, dec, B)
                row_tr["split"] = "train"
                train_results.append(row_tr)

    # Oracle-causal-slice decoder accuracy — drops spurious capsules
    # before decoding to isolate "decoder mechanism" from "admission
    # failure".  Baseline upper bound for each decoder.
    print("[phase48] evaluating decoders on causal-slice (oracle) …")
    causal_results = []
    for dec in decoders:
        n = 0
        n_correct = 0
        per_scen_correct: dict[str, int] = {}
        per_scen_total: dict[str, int] = {}
        for inst in test_inst:
            causal = [c for c, lab in zip(
                inst["offered_capsules"], inst["is_causal"]) if lab]
            out = dec.decode(causal)
            gold = inst["gold_root_cause"]
            sid = inst["scenario_id"]
            per_scen_total[sid] = per_scen_total.get(sid, 0) + 1
            correct = int(out == gold)
            if correct:
                per_scen_correct[sid] = per_scen_correct.get(
                    sid, 0) + 1
            n += 1
            n_correct += correct
        causal_results.append({
            "decoder": dec.name,
            "n_instances": n,
            "accuracy_causal_slice": (
                n_correct / n) if n else 0.0,
            "per_scenario_accuracy": {
                sid: (per_scen_correct.get(sid, 0) /
                       per_scen_total[sid])
                for sid in per_scen_total},
        })

    out = {
        "schema": "wevra.phase48.bundle_decoding.v1",
        "n_records": n_total,
        "n_causal_records": n_causal,
        "n_train_instances": len(train_inst),
        "n_test_instances": len(test_inst),
        "admission_policies": [p.name for p in admission_policies],
        "decoders": [d.name for d in decoders],
        "budgets": list(budget_grid),
        "distractor_grid": list(distractor_grid),
        "spurious_prob": spurious_prob,
        "mislabel_prob": mislabel_prob,
        "rc_alphabet": list(rc_alphabet),
        "decoder_features": list(BUNDLE_DECODER_FEATURES),
        "decoder_weights": dict(decoder_trained.weights),
        "structural_ceiling": round(1.0 / len(rc_alphabet), 4),
        "test_results": test_results,
        "train_results": train_results,
        "causal_slice_results": causal_results,
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_phase48_bundle_decoding.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase48] wrote {path} ({out['wall_seconds']} s)")
    _print_summary(out)
    return out


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_summary(out):
    ceiling = out["structural_ceiling"]
    print(f"\n[phase48] structural priority-ceiling = "
          f"{ceiling:.3f} (= 1 / |rc_alphabet|)")
    # Per-decoder test accuracy (aggregated over admission policies
    # and budgets) + best cell.
    results = out["test_results"]
    decoders = out["decoders"]
    admit_policies = out["admission_policies"]
    budgets = out["budgets"]

    print("\n[phase48] === TEST decoder accuracy "
          "(admission × decoder @ best-in-family budget) ===")
    # Best (budget, admission) cell per decoder.
    print(f"  {'decoder':<28s}  {'best acc':>10s}  "
          f"{'@ budget':>10s}  {'@ admission':>22s}")
    for dec in decoders:
        best = None
        for r in results:
            if r["decoder"] != dec:
                continue
            if best is None or r["accuracy"] > best["accuracy"]:
                best = r
        if best is not None:
            print(f"  {dec:<28s}  {best['accuracy']:>10.3f}  "
                  f"{best['budget']:>10d}  "
                  f"{best['admission_policy']:>22s}")

    # Per-decoder causal-slice (oracle) accuracy — the decoder-
    # mechanism upper bound.
    print("\n[phase48] === TEST decoder accuracy on CAUSAL SLICE "
          "(admission-poisoning removed) ===")
    for row in out["causal_slice_results"]:
        print(f"  {row['decoder']:<28s}  "
              f"{row['accuracy_causal_slice']:>10.3f}   "
              f"per-scen = {row['per_scenario_accuracy']}")

    # FIFO × all decoders at every budget — the "admission doesn't
    # help" reference row.
    print("\n[phase48] === TEST FIFO admission × decoder (no "
          "admission-side work) ===")
    hdr = f"  {'budget':>7s} | "
    for dec in decoders:
        hdr += f"{dec:>28s} | "
    print(hdr)
    for B in budgets:
        row = f"  {B:>7d} | "
        for dec in decoders:
            cell = next((r for r in results
                          if r["decoder"] == dec
                          and r["admission_policy"] == "fifo"
                          and r["budget"] == B), None)
            row += (f"{cell['accuracy']:>28.3f} | "
                     if cell else f"{'?':>28s} | ")
        print(row)

    # Strongest-admission × decoder at every budget.
    # Use "bundle_learned_admit" which is the Phase-47 bundle-aware
    # admission policy (the strongest from that programme).
    for admit_name in ("learned(p46)", "bundle_learned_admit"):
        print(f"\n[phase48] === TEST {admit_name} admission × decoder ===")
        hdr = f"  {'budget':>7s} | "
        for dec in decoders:
            hdr += f"{dec:>28s} | "
        print(hdr)
        for B in budgets:
            row = f"  {B:>7d} | "
            for dec in decoders:
                cell = next((r for r in results
                              if r["decoder"] == dec
                              and r["admission_policy"] == admit_name
                              and r["budget"] == B), None)
                row += (f"{cell['accuracy']:>28.3f} | "
                         if cell else f"{'?':>28s} | ")
            print(row)

    # Ceiling-break check.
    any_break = False
    for r in results:
        if r["accuracy"] > ceiling + 1e-9:
            any_break = True
            break
    if any_break:
        print(f"\n[phase48] CEILING BROKEN: at least one (admission,"
              f" decoder, budget) cell beats the structural priority "
              f"ceiling {ceiling:.3f}.")
    else:
        print(f"\n[phase48] CEILING INTACT: no cell beats "
              f"{ceiling:.3f}.")


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 48: bundle-aware capsule DECODING "
                     "(P47-C1 anchor)")
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
    run_phase48(
        out_dir=args.out_dir, seeds=tuple(args.seeds),
        budget_grid=tuple(args.budgets),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob)


if __name__ == "__main__":
    _cli()
