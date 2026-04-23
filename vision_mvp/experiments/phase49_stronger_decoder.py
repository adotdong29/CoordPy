"""Phase 49 — stronger bundle-aware decoder.

Phase 48 (``phase48_bundle_decoding``) shipped the first
decoder that breaks the 0.200 Phase-31 structural ceiling
(Claim W3-19 at +15..+17.5 pp).  The explicit paradigm-shift
threshold (Conjecture W3-C7) asks for ≥ 0.400 test accuracy
with approximately-symmetric cross-domain transfer.  Phase 48
hit 0.375 — short of the 0.400 bar.

This driver is the first direct attack on the ≥ 0.400 half of
W3-C7.  Under the same data-generating process as Phase 48
(Phase-31 noisy bench, ``spurious_prob=0.30, mislabel_prob=0.10``,
by-seed 80/20 train/test) we compare:

  * Phase-48 baselines: ``priority``, ``learned_bundle_decoder``
    (V1, 10 features).
  * Phase-49 V2 linear: ``learned_bundle_decoder_v2`` (20 features;
    V1 + 10 domain-invariant relative features).
  * Phase-49 interaction: ``interaction_bundle_decoder``
    (V2 + all pairwise crosses; 191 features, linear).
  * Phase-49 MLP: ``mlp_bundle_decoder`` (V2 + 1-hidden-layer
    MLP; ~170 parameters).
  * Phase-49 DeepSet: ``deep_set_bundle_decoder`` (per-capsule
    φ summed + V2 + 1-hidden-layer MLP; ~290 parameters).

Headline metric: test-set decoder accuracy on the FULL admit
set as a function of (admission policy, decoder, budget) — the
same cell definition as Phase 48 for apples-to-apples.

Secondary metrics:
  * Best cell per decoder across the full grid.
  * Oracle-clean causal-slice accuracy (admission-noise
    removed) — the decoder-mechanism upper bound.
  * Seed-robustness: run across a canonical seed-split +
    report the by-seed std-dev of the best cell.
  * Ceiling-break summary: does any (admission, decoder,
    budget) cell cross 0.400?

Runs deterministically in ~60–90 s.

Theoretical anchor:
  * ``docs/CAPSULE_FORMALISM.md`` § 4.D Theorems W3-20 / W3-21
    and Conjecture W3-C7.
  * ``docs/RESULTS_CAPSULE_RESEARCH_MILESTONE4.md`` § 1.
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

# Deterministic dataset collection.
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
    BudgetedAdmissionLedger, FIFOPolicy,
    LearnedAdmissionPolicy, train_admission_policy, ADMIT,
)
from vision_mvp.wevra.capsule_policy_bundle import (
    BundleLearnedPolicy, train_bundle_policy,
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    DEFAULT_HIGH_PRIORITY_CUTOFF,
)
from vision_mvp.wevra.capsule_decoder import (
    PriorityDecoder, LearnedBundleDecoder,
    train_learned_bundle_decoder, evaluate_decoder,
)
from vision_mvp.wevra.capsule_decoder_v2 import (
    BUNDLE_DECODER_FEATURES_V2,
    LearnedBundleDecoderV2, train_learned_bundle_decoder_v2,
    InteractionBundleDecoder, train_interaction_bundle_decoder,
    MLPBundleDecoder, train_mlp_bundle_decoder,
    DeepSetBundleDecoder, train_deep_set_bundle_decoder,
)


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


def collect_dataset(seeds: Sequence[int],
                    distractor_grid: Sequence[int],
                    spurious_prob: float,
                    mislabel_prob: float,
                    ) -> tuple[list[dict], list[dict]]:
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


def admit(instance, policy, budget) -> list[ContextCapsule]:
    capsules = instance["offered_capsules"]
    bal = BudgetedAdmissionLedger(
        budget_tokens=budget, policy=policy)
    bal.offer_all_batched(capsules)
    admitted = {d.capsule_cid for d in bal.decisions
                 if d.decision == ADMIT}
    return [c for c in capsules if c.cid in admitted]


def evaluate_cell(instances, admission_policy, decoder, budget):
    n = 0
    n_correct = 0
    per_scen: dict[str, dict[str, int]] = {}
    for inst in instances:
        admitted = admit(inst, admission_policy, budget)
        gold = inst["gold_root_cause"]
        out = decoder.decode(admitted)
        sid = inst["scenario_id"]
        if sid not in per_scen:
            per_scen[sid] = {"correct": 0, "total": 0}
        per_scen[sid]["total"] += 1
        correct = int(out == gold)
        if correct:
            per_scen[sid]["correct"] += 1
        n += 1
        n_correct += correct
    per_scenario = {
        sid: (v["correct"] / v["total"]) if v["total"] else 0.0
        for sid, v in per_scen.items()}
    return {
        "admission_policy": admission_policy.name,
        "decoder": decoder.name,
        "budget": budget,
        "n_instances": n,
        "accuracy": (n_correct / n) if n else 0.0,
        "per_scenario_accuracy": per_scenario,
    }


def training_bundles(instances, admission_policy, budget):
    out = []
    for inst in instances:
        admitted = admit(inst, admission_policy, budget)
        out.append((admitted, inst["gold_root_cause"]))
    return out


def run_phase49(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        budget_grid: Sequence[int] = (16, 32, 48, 64, 96, 128, 256),
        seeds: Sequence[int] = (31, 32, 33, 34, 35,
                                36, 37, 38, 39, 40),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[phase49] collecting Phase-31 noisy dataset "
          f"(seeds={list(seeds)}, k ∈ {list(distractor_grid)})…")
    records, instances = collect_dataset(
        seeds=seeds, distractor_grid=distractor_grid,
        spurious_prob=spurious_prob,
        mislabel_prob=mislabel_prob)
    n_total = len(records)
    n_causal = sum(r["is_causal"] for r in records)
    print(f"[phase49]   {n_total} capsules, {n_causal} causal "
          f"({100 * n_causal / max(1, n_total):.1f} %); "
          f"{len(instances)} instances")
    train_inst, test_inst = split_instances_by_seed(
        instances, seed=seeds[0])
    print(f"[phase49]   train={len(train_inst)} test={len(test_inst)}")

    # Train admission policies (Phase-46 + Phase-47) for
    # comparability with the Phase-48 results table.
    print("[phase49] training Phase-46 LearnedAdmissionPolicy…")
    train_cap_examples = [
        (r["capsule"], r["is_causal"]) for r in records
        if any(inst["instance_key"] == r["instance_key"]
               for inst in train_inst)]
    learned_admit = train_admission_policy(
        train_cap_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    learned_admit.name = "learned(p46)"

    print("[phase49] training Phase-47 BundleLearnedPolicy…")
    train_bundle = [(inst["offered_capsules"],
                       inst["offered_capsules"],
                       inst["is_causal"])
                      for inst in train_inst]
    bundle_admit = train_bundle_policy(
        train_bundle, n_epochs=300, lr=0.5, l2=1e-3, seed=seeds[0])
    bundle_admit.name = "bundle_learned_admit"

    admission_policies: list[Any] = [
        FIFOPolicy(),
        learned_admit,
        bundle_admit,
    ]

    rc_alphabet = tuple(sorted({
        inst["gold_root_cause"] for inst in instances}))
    print(f"[phase49]   rc_alphabet = {rc_alphabet}")

    # Phase-49 training-data augmentation.  Phase-48 trained the
    # decoder on FIFO @ B=256 bundles only.  This produces a
    # decoder whose training distribution differs from its
    # deployment distribution (the bundle the decoder sees at
    # eval time is admission-filtered).  Phase 49 augments the
    # training set with bundles produced under MULTIPLE admission
    # cells — a distribution-matching that lifts the decoder by
    # ≈ +2 pp on the bundle_learned_admit @ B=48 cell.  The V1
    # baseline is still trained on the Phase-48 FIFO@256 setup
    # for apples-to-apples comparability with W3-19.
    fifo = FIFOPolicy()
    print("[phase49] training V1 baseline (FIFO @ B=256)…")
    train_pairs_v1 = training_bundles(train_inst, fifo, 256)
    print("[phase49] building augmented training set "
          "(union of admission cells)…")
    train_pairs_aug: list[tuple[list[ContextCapsule], str]] = []
    aug_cells = [
        (fifo, 256), (fifo, 96), (fifo, 48),
        (learned_admit, 96), (learned_admit, 48),
        (bundle_admit, 96), (bundle_admit, 48),
    ]
    for inst in train_inst:
        gold = inst["gold_root_cause"]
        for (pol, B) in aug_cells:
            bundle = admit(inst, pol, B)
            train_pairs_aug.append((bundle, gold))
    print(f"[phase49]   aug train pairs: {len(train_pairs_aug)} "
          f"(={len(train_inst)} instances × {len(aug_cells)} cells)")
    train_pairs = train_pairs_v1  # used for V1 baseline below

    # --- Phase-48 baseline (V1, FIFO@256 only — reproduces W3-19) ---
    dec_v1 = train_learned_bundle_decoder(
        train_pairs_v1,
        rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs=300, lr=0.5, l2=1e-3, seed=seeds[0])
    dec_v1.name = "learned_bundle_decoder"

    # --- Phase-49 V2 linear (augmented training) ---
    dec_v2 = train_learned_bundle_decoder_v2(
        train_pairs_aug,
        rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs=500, lr=0.5, l2=1e-3, seed=seeds[0])
    dec_v2.name = "learned_bundle_decoder_v2"

    # --- Phase-49 interaction (augmented training) ---
    dec_int = train_interaction_bundle_decoder(
        train_pairs_aug,
        rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        n_epochs=500, lr=0.3, l2=1e-2, seed=seeds[0])
    dec_int.name = "interaction_bundle_decoder"

    # --- Phase-49 MLP (augmented training) ---
    dec_mlp = train_mlp_bundle_decoder(
        train_pairs_aug,
        rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        hidden_size=12, n_epochs=600, lr=0.1, l2=1e-3,
        seed=seeds[0])
    dec_mlp.name = "mlp_bundle_decoder"

    # --- Phase-49 DeepSet (augmented training) ---
    dec_ds = train_deep_set_bundle_decoder(
        train_pairs_aug,
        rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        hidden_size=10, n_epochs=600, lr=0.1, l2=1e-3,
        seed=seeds[0])
    dec_ds.name = "deep_set_bundle_decoder"

    print("[phase49]   trained V1 decoder weights "
          "(top 5 by |w|):")
    for k, w in sorted(dec_v1.weights.items(),
                         key=lambda kv: -abs(kv[1]))[:5]:
        print(f"      {k:28s}  {w:+.4f}")
    print("[phase49]   trained V2 decoder weights "
          "(top 8 by |w|):")
    for k, w in sorted(dec_v2.weights.items(),
                         key=lambda kv: -abs(kv[1]))[:8]:
        print(f"      {k:28s}  {w:+.4f}")

    decoders: list[Any] = [
        PriorityDecoder(),
        dec_v1,
        dec_v2,
        dec_int,
        dec_mlp,
        dec_ds,
    ]

    # Sweep (admission × decoder × budget).
    print(f"[phase49] sweeping {len(admission_policies)} admission × "
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

    # Oracle-clean causal-slice.
    print("[phase49] evaluating decoders on causal-slice (oracle)…")
    causal_results = []
    for dec in decoders:
        n = 0
        n_correct = 0
        per_scen: dict[str, dict[str, int]] = {}
        for inst in test_inst:
            causal = [c for c, lab in zip(
                inst["offered_capsules"], inst["is_causal"]) if lab]
            out = dec.decode(causal)
            gold = inst["gold_root_cause"]
            sid = inst["scenario_id"]
            if sid not in per_scen:
                per_scen[sid] = {"correct": 0, "total": 0}
            per_scen[sid]["total"] += 1
            correct = int(out == gold)
            if correct:
                per_scen[sid]["correct"] += 1
            n += 1
            n_correct += correct
        causal_results.append({
            "decoder": dec.name,
            "n_instances": n,
            "accuracy_causal_slice": (
                n_correct / n) if n else 0.0,
            "per_scenario_accuracy": {
                sid: (v["correct"] / v["total"])
                for sid, v in per_scen.items()},
        })

    # Serialise non-numeric weights.
    def _weights_dump(dec):
        if hasattr(dec, "to_dict"):
            try:
                return dec.to_dict()
            except Exception:
                return {"name": getattr(dec, "name", "?")}
        return {"name": getattr(dec, "name", "?")}

    out = {
        "schema": "wevra.phase49.stronger_decoder.v1",
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
        "v2_feature_list": list(BUNDLE_DECODER_FEATURES_V2),
        "augmented_training_cells": [
            (p.name, B) for (p, B) in aug_cells],
        "n_train_pairs_v1": len(train_pairs_v1),
        "n_train_pairs_aug": len(train_pairs_aug),
        "decoder_weights_v1": dict(dec_v1.weights),
        "decoder_weights_v2": dict(dec_v2.weights),
        "decoder_dump_interaction": _weights_dump(dec_int),
        "decoder_dump_mlp": _weights_dump(dec_mlp),
        "decoder_dump_deepset": _weights_dump(dec_ds),
        "structural_ceiling": round(1.0 / len(rc_alphabet), 4),
        "paradigm_shift_threshold": 0.400,
        "test_results": test_results,
        "train_results": train_results,
        "causal_slice_results": causal_results,
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_phase49_stronger_decoder.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase49] wrote {path} ({out['wall_seconds']} s)")
    _print_summary(out)
    return out


def _print_summary(out):
    ceiling = out["structural_ceiling"]
    threshold = out["paradigm_shift_threshold"]
    print(f"\n[phase49] structural priority-ceiling = "
          f"{ceiling:.3f}; paradigm-shift threshold "
          f"(W3-C7) = {threshold:.3f}")
    results = out["test_results"]
    decoders = out["decoders"]
    budgets = out["budgets"]

    print("\n[phase49] === TEST best cell per decoder ===")
    print(f"  {'decoder':<30s}  {'best acc':>10s}  "
          f"{'@ budget':>10s}  {'@ admission':>22s}")
    for dec in decoders:
        best = None
        for r in results:
            if r["decoder"] != dec:
                continue
            if best is None or r["accuracy"] > best["accuracy"]:
                best = r
        if best is not None:
            tag = (" ≥0.400" if best["accuracy"] >= threshold
                    else ("  >0.200" if best["accuracy"] > ceiling
                          else "  below"))
            print(f"  {dec:<30s}  {best['accuracy']:>10.3f}"
                  f"{tag}  "
                  f"{best['budget']:>10d}  "
                  f"{best['admission_policy']:>22s}")

    print("\n[phase49] === TEST decoder accuracy on CAUSAL SLICE "
          "(admission-poisoning removed) ===")
    for row in out["causal_slice_results"]:
        print(f"  {row['decoder']:<30s}  "
              f"{row['accuracy_causal_slice']:>10.3f}")

    # bundle_learned_admit × decoder at every budget (best admission
    # from Phase 47).
    print("\n[phase49] === TEST bundle_learned_admit × decoder ===")
    hdr = f"  {'budget':>7s} | "
    for dec in decoders:
        hdr += f"{dec:>30s} | "
    print(hdr)
    for B in budgets:
        row = f"  {B:>7d} | "
        for dec in decoders:
            cell = next((r for r in results
                          if r["decoder"] == dec
                          and r["admission_policy"] == "bundle_learned_admit"
                          and r["budget"] == B), None)
            row += (f"{cell['accuracy']:>30.3f} | "
                     if cell else f"{'?':>30s} | ")
        print(row)

    any_crosses = any(r["accuracy"] >= threshold
                       for r in results)
    any_break = any(r["accuracy"] > ceiling + 1e-9
                     for r in results)
    if any_crosses:
        print(f"\n[phase49] PARADIGM-SHIFT GATE-1: at least one "
              f"cell crosses {threshold:.3f} (≥ 2× ceiling).")
    elif any_break:
        print(f"\n[phase49] CEILING BROKEN (> {ceiling:.3f}) but "
              f"GATE-1 NOT MET (no cell ≥ {threshold:.3f}).")
    else:
        print(f"\n[phase49] CEILING INTACT (no cell > "
              f"{ceiling:.3f}).")


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 49: stronger bundle-aware decoder")
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
    run_phase49(
        out_dir=args.out_dir, seeds=tuple(args.seeds),
        budget_grid=tuple(args.budgets),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob)


if __name__ == "__main__":
    _cli()
