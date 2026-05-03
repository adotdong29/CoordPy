"""Phase 51 — cohort-relational decoder frontier.

Phase 50 closed the strict reading of Conjecture W3-C7 with
two proved limitation theorems: W3-24 (post-search winner's-
curse bias) and W3-29 (Bayes-divergence zero-shot risk lower
bound for the linear class-agnostic family).  The Phase-50
sign-stable DeepSet achieves gap = 0.000 zero-shot on
(incident, security) but at level 0.237 — direction-invariant
but materially below the within-domain optimum.  The W3-C9
refined bar accepts the gap reading of Gate 2; the remaining
honest open question is the **level** of direction-invariant
zero-shot transfer.

Phase 51 attacks this with the smallest serious **relational**
decoder — a hypothesis class structurally outside the magnitude-
monoid linear family (Theorem W3-30 proves strict containment
over DeepSet).  The driver:

  1. Trains the Phase-51 ``CohortRelationalDecoder`` on the
     Phase-31 noisy incident-triage bench and evaluates
     decoder accuracy at the Phase-50 pre-committed cell
     (``bundle_learned_admit @ B = 64``) at
     $n_{\\rm test} = 80$.  Reports Gate-1 status under the
     W3-C9 defensible bar (point-estimate at $n=80$) and the
     strict reading (point-estimate + 95 % Wilson CI lower
     bound).

  2. Runs the zero-shot transfer study on (incident,
     security) — trained on one domain, evaluated on the
     other with the SAME weight vector — and reports:
       * within-domain accuracies (for context);
       * cross-domain accuracies (i → s, s → i);
       * direction-invariance **gap** (W3-C9 bar);
       * per-direction **penalty** (aspirational bar);
       * **level** := min(cross_i_to_s, cross_s_to_i), the
         Phase-51 specific metric.
     Compares against the Phase-50 baseline level of 0.237
     (sign-stable DeepSet from ``phase50_zero_shot_transfer``).

  3. Outputs an honest JSON record of the result.  The
     programme's next position is then:
       * If level > 0.237 with gap ≤ 0.05 — **W3-31 is met
         empirically**; the relational axis delivers level-lift
         under direction-invariance; Conjecture W3-C10's
         strict-level-ceiling reading is partially supported
         at the new level.
       * If level ≤ 0.237 with gap ≤ 0.05 — **W3-31 is
         falsified empirically**; the relational axis does
         NOT close the level gap on (incident, security);
         Conjecture W3-C10 is supported strongly.
       * If gap > 0.05 — **direction-invariance is not
         achieved**; the decoder is in a different regime than
         sign-stable DeepSet.

Theoretical anchors:
  * ``docs/CAPSULE_FORMALISM.md`` § 4.F, Theorem W3-30, Claim
    W3-31, Conjecture W3-C10.
  * ``docs/RESULTS_CAPSULE_RESEARCH_MILESTONE6.md`` (to be
    written).

Wall time: ≈ 30–60 s on a 2024 M-class MacBook.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from typing import Any, Callable, Sequence

try:
    import numpy as np
except ImportError as ex:
    raise ImportError("phase51 requires numpy") from ex

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv],
                os.environ)

from vision_mvp.coordpy.capsule import (
    ContextCapsule, capsule_from_handoff,
)
from vision_mvp.coordpy.capsule_policy import (
    BudgetedAdmissionLedger, FIFOPolicy,
    LearnedAdmissionPolicy, train_admission_policy, ADMIT,
)
from vision_mvp.coordpy.capsule_policy_bundle import (
    BundleLearnedPolicy, train_bundle_policy,
    DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    DEFAULT_HIGH_PRIORITY_CUTOFF,
)
from vision_mvp.coordpy.capsule_decoder import (
    PriorityDecoder, LearnedBundleDecoder,
    train_learned_bundle_decoder, evaluate_decoder,
)
from vision_mvp.coordpy.capsule_decoder_v2 import (
    DeepSetBundleDecoder, train_deep_set_bundle_decoder,
)
from vision_mvp.coordpy.capsule_decoder_relational import (
    CohortRelationalDecoder, train_cohort_relational_decoder,
    COHORT_RELATIONAL_FEATURES,
)
from vision_mvp.experiments.phase50_gate1_ci import (
    wilson_ci, clopper_pearson_ci,
)
from vision_mvp.experiments.phase50_zero_shot_transfer import (
    DomainSpec, _incident_spec, _security_spec,
    collect_domain_instances, split_by_seed, eval_decoder,
    train_sign_stable_deepset,
    SIGN_STABLE_PHI_IDX,
)


# =============================================================================
# Part A — Gate-1 sweep on Phase-31 pre-committed cell
# =============================================================================


def _gold_root_cause(scenario) -> str:
    from vision_mvp.core.role_handoff import TypedHandoff
    from vision_mvp.tasks.incident_triage import (
        _decoder_from_handoffs, ROLE_AUDITOR,
    )
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


def collect_incident_dataset(seeds, distractor_grid,
                              spurious_prob, mislabel_prob):
    from vision_mvp.tasks.incident_triage import (
        build_scenario_bank, handoff_is_relevant,
        run_handoff_protocol, ROLE_AUDITOR,
        extract_claims_for_role,
    )
    from vision_mvp.core.extractor_noise import (
        NoiseConfig, noisy_extractor, incident_triage_known_kinds,
    )
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
                    cap = dataclasses.replace(
                        cap, n_tokens=h.n_tokens)
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
    for inst in instances:
        admitted = admit(inst, admission_policy, budget)
        gold = inst["gold_root_cause"]
        out = decoder.decode(admitted)
        n += 1
        if out == gold:
            n_correct += 1
    return {
        "admission_policy": admission_policy.name,
        "decoder": decoder.name,
        "budget": budget,
        "n_instances": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
    }


def training_bundles(instances, admission_policy, budget):
    out = []
    for inst in instances:
        admitted = admit(inst, admission_policy, budget)
        out.append((admitted, inst["gold_root_cause"]))
    return out


def run_gate1_on_incident(
        seeds: Sequence[int] = tuple(range(31, 51)),
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
) -> dict[str, Any]:
    """Gate-1 check at the Phase-50 pre-committed cell
    (bundle_learned_admit @ B=64) at n_test = 80 (20 seeds,
    4 train / 1 test).  Matches Phase-49's sampling exactly for
    apples-to-apples with W3-23."""
    t0 = time.time()
    print(f"[phase51-G1] collecting incident dataset "
          f"(n_seeds={len(seeds)})…")
    records, instances = collect_incident_dataset(
        seeds, distractor_grid, spurious_prob, mislabel_prob)
    print(f"  {len(records)} capsules; {len(instances)} instances")
    train_inst, test_inst = split_instances_by_seed(
        instances, seed=seeds[0])
    print(f"  train={len(train_inst)} test={len(test_inst)}")

    # Train admission policies (same as Phase 50).
    train_cap_examples = [
        (r["capsule"], r["is_causal"]) for r in records
        if any(inst["instance_key"] == r["instance_key"]
               for inst in train_inst)]
    learned_admit = train_admission_policy(
        train_cap_examples, n_epochs=300, lr=0.5, l2=1e-3,
        seed=seeds[0])
    learned_admit.name = "learned(p46)"
    train_bundle = [(inst["offered_capsules"],
                       inst["offered_capsules"],
                       inst["is_causal"])
                      for inst in train_inst]
    bundle_admit = train_bundle_policy(
        train_bundle, n_epochs=300, lr=0.5, l2=1e-3, seed=seeds[0])
    bundle_admit.name = "bundle_learned_admit"

    rc_alphabet = tuple(sorted({
        inst["gold_root_cause"] for inst in instances}))
    print(f"  rc_alphabet = {rc_alphabet}")

    # Augmented training (Phase-49/Phase-50 recipe).
    fifo = FIFOPolicy()
    aug_cells = [
        (fifo, 256), (fifo, 96), (fifo, 48),
        (learned_admit, 96), (learned_admit, 48),
        (bundle_admit, 96), (bundle_admit, 48),
    ]
    train_pairs_aug: list[tuple[list[ContextCapsule], str]] = []
    for inst in train_inst:
        gold = inst["gold_root_cause"]
        for (pol, B) in aug_cells:
            bundle = admit(inst, pol, B)
            train_pairs_aug.append((bundle, gold))
    print(f"  aug training pairs: {len(train_pairs_aug)}")

    # Train the Phase-51 cohort-relational decoder.
    print("[phase51-G1] training CohortRelationalDecoder…")
    dec_rel = train_cohort_relational_decoder(
        train_pairs_aug, rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        hidden_size=10, n_epochs=500, lr=0.1, l2=1e-3,
        seed=seeds[0])

    # Also retrain the Phase-49 DeepSet for direct apples-to-
    # apples comparison at the same train/test seeds and
    # augmented pairs.
    print("[phase51-G1] training DeepSet (reproducer)…")
    dec_ds = train_deep_set_bundle_decoder(
        train_pairs_aug, rc_alphabet=rc_alphabet,
        claim_to_root_cause=DEFAULT_CLAIM_TO_ROOT_CAUSE,
        priority_order=DEFAULT_PRIORITY_ORDER,
        high_priority_cutoff=DEFAULT_HIGH_PRIORITY_CUTOFF,
        hidden_size=10, n_epochs=600, lr=0.1, l2=1e-3,
        seed=seeds[0])

    # Evaluate at the pre-committed cell
    # (bundle_learned_admit @ B=64).
    print("[phase51-G1] evaluating at pre-committed cell "
          "(bundle_learned_admit @ B=64)…")
    cell_rel = evaluate_cell(test_inst, bundle_admit, dec_rel, 64)
    cell_rel["decoder"] = "cohort_relational_decoder"
    cell_ds = evaluate_cell(test_inst, bundle_admit, dec_ds, 64)
    cell_ds["decoder"] = "deep_set_bundle_decoder"
    cell_pri = evaluate_cell(
        test_inst, bundle_admit, PriorityDecoder(), 64)
    cell_pri["decoder"] = "priority"

    # Sweep across additional budget cells so we can also report
    # "best-cell" alongside the pre-committed cell.
    best_rel: dict[str, Any] | None = None
    best_ds: dict[str, Any] | None = None
    sweep: list[dict[str, Any]] = []
    for admission_policy in [FIFOPolicy(), learned_admit, bundle_admit]:
        for B in [48, 64, 96]:
            r_rel = evaluate_cell(test_inst, admission_policy, dec_rel, B)
            r_rel["decoder"] = "cohort_relational_decoder"
            r_ds = evaluate_cell(test_inst, admission_policy, dec_ds, B)
            r_ds["decoder"] = "deep_set_bundle_decoder"
            sweep.append(r_rel)
            sweep.append(r_ds)
            if best_rel is None or r_rel["accuracy"] > best_rel["accuracy"]:
                best_rel = r_rel
            if best_ds is None or r_ds["accuracy"] > best_ds["accuracy"]:
                best_ds = r_ds

    def _ci_row(r):
        n, k = r["n_instances"], r["n_correct"]
        wi = wilson_ci(k, n); cp = clopper_pearson_ci(k, n)
        return {**r, "wilson_ci_lo": wi[0], "wilson_ci_hi": wi[1],
                 "clopper_pearson_ci_lo": cp[0],
                 "clopper_pearson_ci_hi": cp[1]}

    threshold = 0.400
    out = {
        "schema": "coordpy.phase51.gate1_incident.v1",
        "n_seeds": len(seeds),
        "n_train_instances": len(train_inst),
        "n_test_instances": len(test_inst),
        "threshold": threshold,
        "apriori_cell": ["bundle_learned_admit", 64],
        "apriori_cells": [
            _ci_row(cell_pri), _ci_row(cell_ds), _ci_row(cell_rel),
        ],
        "best_cell_per_decoder": [
            _ci_row(best_ds), _ci_row(best_rel),
        ],
        "sweep_rows": sweep,
        "wall_seconds": round(time.time() - t0, 3),
        "decoder_shape": {
            "cohort_relational_decoder": {
                "input_dim": int(len(COHORT_RELATIONAL_FEATURES)),
                "hidden_size": int(dec_rel.hidden_size),
                "n_params": int(dec_rel.W1.size + dec_rel.b1.size
                                 + dec_rel.w2.size + 1),
            },
        },
    }
    return out, dec_rel, dec_ds, bundle_admit, learned_admit


# =============================================================================
# Part B — Zero-shot transfer study (incident, security)
# =============================================================================


def collect_domain_training_bundles(
        spec: DomainSpec,
        seeds: Sequence[int],
        distractor_grid: Sequence[int],
        spurious_prob: float,
        mislabel_prob: float,
        admission_policies: list[tuple[Any, int]] | None = None,
        ) -> tuple[list[dict], list[tuple], dict[str, Any]]:
    """Collect (bundle, gold_label) pairs for one domain across
    multiple admission cells for augmented training."""
    insts = collect_domain_instances(
        spec, seeds=seeds, distractor_grid=distractor_grid,
        spurious_prob=spurious_prob,
        mislabel_prob=mislabel_prob)
    if admission_policies is None:
        admission_policies = [(FIFOPolicy(), 256)]
    pairs: list[tuple] = []
    for inst in insts:
        caps = inst["offered_capsules"]
        for (pol, B) in admission_policies:
            bal = BudgetedAdmissionLedger(
                budget_tokens=B, policy=pol)
            bal.offer_all_batched(caps)
            admitted_cids = {d.capsule_cid for d in bal.decisions
                              if d.decision == ADMIT}
            admitted = [c for c in caps if c.cid in admitted_cids]
            pairs.append((admitted, inst["gold_label"]))
    stats = {
        "n_instances": len(insts),
        "n_pairs": len(pairs),
        "domain": spec.name,
    }
    return insts, pairs, stats


def make_cross_cohort_relational(src, tgt_spec: DomainSpec
                                  ) -> CohortRelationalDecoder:
    """Copy weights, swap alphabet for zero-shot cross-domain
    deployment."""
    return CohortRelationalDecoder(
        W1=src.W1.copy(), b1=src.b1.copy(),
        w2=src.w2.copy(), b2=src.b2,
        hidden_size=src.hidden_size,
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff,
    )


def run_zero_shot_transfer(
        train_seeds: Sequence[int] = tuple(range(31, 47)),
        test_seeds: Sequence[int] = tuple(range(47, 51)),
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
        ) -> dict[str, Any]:
    """Phase 51 zero-shot transfer on (incident, security).

    Trains ``CohortRelationalDecoder`` on one domain, deploys on
    the other with the SAME weight vector (alphabet + claim
    mapping swapped out).  Reports direction-invariance gap,
    per-direction penalty, and the Phase-51 "level" metric.

    Also retrains the Phase-50 sign-stable DeepSet on the same
    training bundles for apples-to-apples comparison.
    """
    t0 = time.time()
    specs = [_incident_spec(), _security_spec()]
    all_seeds = tuple(sorted(set(train_seeds) | set(test_seeds)))

    # Collect raw instances.
    insts_by_domain: dict[str, list[dict]] = {}
    for spec in specs:
        print(f"[phase51-ZS] collecting {spec.name}…")
        insts = collect_domain_instances(
            spec, seeds=all_seeds,
            distractor_grid=distractor_grid,
            spurious_prob=spurious_prob,
            mislabel_prob=mislabel_prob)
        insts_by_domain[spec.name] = insts
        print(f"  {spec.name}: {len(insts)} instances; "
              f"alphabet {spec.label_alphabet}")

    # Train admission policies per domain (used in augmented
    # bundle generation).  Match Phase 50's zero-shot pipeline.
    from vision_mvp.core.role_handoff import TypedHandoff
    trained_admit: dict[str, dict[str, Any]] = {}
    for spec in specs:
        insts = insts_by_domain[spec.name]
        train_pool = [i for i in insts if i["seed"] in set(train_seeds)]
        # Build "is_causal" labels by replaying the handoffs through
        # the domain's scenario bank — we don't have them on the
        # instance dict, so use FIFO as fallback (cheap, no admission
        # policy used for zero-shot transfer evaluation).
        trained_admit[spec.name] = {
            "fifo": FIFOPolicy(),
        }

    # Build per-domain augmented training bundles (FIFO @ 256,
    # 96, 48 — simpler than the Phase-31 7-cell aug because we
    # don't have is_causal labels for non-incident domains; this
    # is honestly the same set used in Phase 50 ZS study).
    per_domain_pairs: dict[str, list[tuple]] = {}
    per_domain_test_insts: dict[str, list[dict]] = {}
    for spec in specs:
        insts = insts_by_domain[spec.name]
        train_pool = [i for i in insts if i["seed"] in set(train_seeds)]
        test_pool = [i for i in insts if i["seed"] in set(test_seeds)]
        per_domain_test_insts[spec.name] = test_pool
        # Augment across three FIFO budget cells.
        pairs: list[tuple] = []
        for inst in train_pool:
            caps = inst["offered_capsules"]
            for B in [256, 96, 48]:
                bal = BudgetedAdmissionLedger(
                    budget_tokens=B, policy=FIFOPolicy())
                bal.offer_all_batched(caps)
                admitted_cids = {d.capsule_cid for d in bal.decisions
                                  if d.decision == ADMIT}
                admitted = [c for c in caps if c.cid in admitted_cids]
                pairs.append((admitted, inst["gold_label"]))
        per_domain_pairs[spec.name] = pairs
        print(f"  {spec.name}: train_pairs={len(pairs)} "
              f"test_insts={len(test_pool)}")

    # Train a CohortRelationalDecoder per domain, plus the
    # Phase-50 sign-stable DeepSet for baseline comparison.
    per_domain_rel: dict[str, CohortRelationalDecoder] = {}
    per_domain_ssds: dict[str, Any] = {}
    per_domain_ds: dict[str, DeepSetBundleDecoder] = {}
    for spec in specs:
        print(f"[phase51-ZS] training CohortRelationalDecoder on "
              f"{spec.name}…")
        dec = train_cohort_relational_decoder(
            per_domain_pairs[spec.name],
            rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            hidden_size=10, n_epochs=500, lr=0.1, l2=1e-3,
            seed=train_seeds[0])
        per_domain_rel[spec.name] = dec

        print(f"[phase51-ZS] training SignStableDeepSet on "
              f"{spec.name}…")
        ssds = train_sign_stable_deepset(
            per_domain_pairs[spec.name],
            rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            hidden_size=8, n_epochs=500, lr=0.1, l2=1e-3,
            seed=train_seeds[0])
        per_domain_ssds[spec.name] = ssds

        print(f"[phase51-ZS] training DeepSet (full) on "
              f"{spec.name}…")
        ds = train_deep_set_bundle_decoder(
            per_domain_pairs[spec.name],
            rc_alphabet=spec.label_alphabet,
            claim_to_root_cause=spec.claim_to_label,
            priority_order=spec.priority_order,
            hidden_size=10, n_epochs=500, lr=0.1, l2=1e-3,
            seed=train_seeds[0])
        per_domain_ds[spec.name] = ds

    # Evaluate within-domain and cross-domain.  For cross-domain,
    # copy the weights and swap alphabet/claim-map/priority via
    # the ``make_cross_*`` helpers.
    from vision_mvp.experiments.phase50_zero_shot_transfer import (
        make_cross_deepset, make_cross_stable_deepset,
    )

    spec_by_name = {s.name: s for s in specs}

    def _within(spec, decoder):
        test = per_domain_test_insts[spec.name]
        pairs = [(i["offered_capsules"], i["gold_label"]) for i in test]
        n, n_correct = len(pairs), 0
        for (bundle, gold) in pairs:
            if decoder.decode(bundle) == gold:
                n_correct += 1
        return n_correct / n if n else 0.0, n, n_correct

    def _cross(src_spec, tgt_spec, maker, decoder):
        cross_dec = maker(decoder, tgt_spec)
        test = per_domain_test_insts[tgt_spec.name]
        pairs = [(i["offered_capsules"], i["gold_label"]) for i in test]
        n, n_correct = len(pairs), 0
        for (bundle, gold) in pairs:
            if cross_dec.decode(bundle) == gold:
                n_correct += 1
        return n_correct / n if n else 0.0, n, n_correct

    family_results: dict[str, dict[str, Any]] = {}
    for family_name, (decoders, maker) in {
            "cohort_relational": (per_domain_rel,
                                    make_cross_cohort_relational),
            "sign_stable_deepset": (per_domain_ssds,
                                     make_cross_stable_deepset),
            "deepset_full": (per_domain_ds, make_cross_deepset),
    }.items():
        row: dict[str, Any] = {}
        for spec in specs:
            w_acc, w_n, w_k = _within(spec, decoders[spec.name])
            row[f"within_{spec.name}"] = {
                "accuracy": w_acc, "n": w_n, "k": w_k}
        inc, sec = spec_by_name["incident"], spec_by_name["security"]
        i_to_s_acc, i_to_s_n, i_to_s_k = _cross(
            inc, sec, maker, decoders["incident"])
        s_to_i_acc, s_to_i_n, s_to_i_k = _cross(
            sec, inc, maker, decoders["security"])
        row["incident_to_security"] = {
            "accuracy": i_to_s_acc, "n": i_to_s_n, "k": i_to_s_k}
        row["security_to_incident"] = {
            "accuracy": s_to_i_acc, "n": s_to_i_n, "k": s_to_i_k}
        row["gap"] = abs(i_to_s_acc - s_to_i_acc)
        row["penalty_i_to_s"] = (
            row[f"within_security"]["accuracy"] - i_to_s_acc)
        row["penalty_s_to_i"] = (
            row[f"within_incident"]["accuracy"] - s_to_i_acc)
        row["max_penalty"] = max(
            row["penalty_i_to_s"], row["penalty_s_to_i"])
        row["level"] = min(i_to_s_acc, s_to_i_acc)
        family_results[family_name] = row

    # Phase-51 verdicts.
    sign_stable_level = family_results["sign_stable_deepset"]["level"]
    relational_level = family_results["cohort_relational"]["level"]
    relational_gap = family_results["cohort_relational"]["gap"]
    gap_bar = 0.05
    level_bar = sign_stable_level

    verdict_w3_31 = {
        "gap_bar": gap_bar,
        "level_bar": level_bar,
        "relational_gap": relational_gap,
        "relational_level": relational_level,
        "relational_sign_stable_deepset_level": sign_stable_level,
        "w3_31_gap_met":
            bool(relational_gap <= gap_bar),
        "w3_31_level_met":
            bool(relational_level > sign_stable_level),
        "w3_31_strict_met":
            bool(relational_gap <= gap_bar
                  and relational_level > sign_stable_level),
    }

    out = {
        "schema": "coordpy.phase51.zero_shot.v1",
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "distractor_grid": list(distractor_grid),
        "spurious_prob": spurious_prob,
        "mislabel_prob": mislabel_prob,
        "family_results": family_results,
        "verdict_w3_31": verdict_w3_31,
        "wall_seconds": round(time.time() - t0, 3),
    }
    return out


# =============================================================================
# Entry
# =============================================================================


def _print_gate1_summary(out: dict[str, Any]) -> None:
    print(f"\n[phase51-G1] === Gate-1 pre-committed cell "
          f"(bundle_learned_admit @ B=64) ===")
    print(f"  n_test = {out['n_test_instances']}")
    print(f"  threshold = {out['threshold']:.3f}")
    for r in out["apriori_cells"]:
        wi = (r["wilson_ci_lo"], r["wilson_ci_hi"])
        print(f"  {r['decoder']:<32s}  acc={r['accuracy']:.3f}  "
              f"{r['n_correct']:>3d}/{r['n_instances']:<3d}  "
              f"wilson=[{wi[0]:.3f},{wi[1]:.3f}]")
    print(f"\n[phase51-G1] === Best cell per Phase-51 decoder ===")
    for r in out["best_cell_per_decoder"]:
        wi = (r["wilson_ci_lo"], r["wilson_ci_hi"])
        print(f"  {r['decoder']:<32s}  acc={r['accuracy']:.3f}  "
              f"{r['n_correct']:>3d}/{r['n_instances']:<3d}  "
              f"wilson=[{wi[0]:.3f},{wi[1]:.3f}]  "
              f"@ {r['admission_policy']} B={r['budget']}")


def _print_zero_shot_summary(out: dict[str, Any]) -> None:
    print(f"\n[phase51-ZS] === Zero-shot transfer on "
          f"(incident, security) ===")
    hdr = (f"  {'family':<22s}  {'within_i':>9s}  {'within_s':>9s}  "
           f"{'i→s':>6s}  {'s→i':>6s}  {'gap':>6s}  "
           f"{'max_pen':>8s}  {'level':>6s}")
    print(hdr)
    for name, row in out["family_results"].items():
        print(f"  {name:<22s}  "
              f"{row['within_incident']['accuracy']:>9.3f}  "
              f"{row['within_security']['accuracy']:>9.3f}  "
              f"{row['incident_to_security']['accuracy']:>6.3f}  "
              f"{row['security_to_incident']['accuracy']:>6.3f}  "
              f"{row['gap']:>6.3f}  "
              f"{row['max_penalty']:>+8.3f}  "
              f"{row['level']:>6.3f}")
    v = out["verdict_w3_31"]
    print(f"\n[phase51-ZS] W3-31 verdict:")
    print(f"  gap bar           : {v['gap_bar']:.3f} — "
          f"relational gap {v['relational_gap']:.3f} → "
          f"{'MET' if v['w3_31_gap_met'] else 'not met'}")
    print(f"  level bar (SSDs)  : {v['level_bar']:.3f} — "
          f"relational level {v['relational_level']:.3f} → "
          f"{'MET (strict >)' if v['w3_31_level_met'] else 'not met'}")
    print(f"  W3-31 strict      : "
          f"{'MET' if v['w3_31_strict_met'] else 'not met'}")


def run_phase51(out_dir: str = ".",
                 *,
                 gate1_seeds: Sequence[int] = tuple(range(31, 51)),
                 zs_train_seeds: Sequence[int] = tuple(range(31, 47)),
                 zs_test_seeds: Sequence[int] = tuple(range(47, 51)),
                 distractor_grid: Sequence[int] = (6, 20, 60, 120),
                 spurious_prob: float = 0.30,
                 mislabel_prob: float = 0.10,
                 ) -> dict[str, Any]:
    print(f"[phase51] running Gate-1 sweep on Phase-31 bench…")
    g1_out, dec_rel, dec_ds, bundle_admit, learned_admit = \
        run_gate1_on_incident(
            seeds=gate1_seeds,
            distractor_grid=distractor_grid,
            spurious_prob=spurious_prob,
            mislabel_prob=mislabel_prob)
    _print_gate1_summary(g1_out)

    print(f"\n[phase51] running zero-shot transfer study…")
    zs_out = run_zero_shot_transfer(
        train_seeds=zs_train_seeds, test_seeds=zs_test_seeds,
        distractor_grid=distractor_grid,
        spurious_prob=spurious_prob,
        mislabel_prob=mislabel_prob)
    _print_zero_shot_summary(zs_out)

    full = {
        "schema": "coordpy.phase51.v1",
        "gate1": g1_out,
        "zero_shot": zs_out,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_phase51_relational_decoder.json")
    with open(path, "w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"\n[phase51] wrote {path}")
    return full


def _cli():
    p = argparse.ArgumentParser(
        description="Phase 51: relational-decoder frontier")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--gate1-seeds", nargs="+", type=int,
                    default=list(range(31, 51)))
    p.add_argument("--zs-train-seeds", nargs="+", type=int,
                    default=list(range(31, 47)))
    p.add_argument("--zs-test-seeds", nargs="+", type=int,
                    default=list(range(47, 51)))
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    args = p.parse_args()
    run_phase51(out_dir=args.out_dir,
                 gate1_seeds=tuple(args.gate1_seeds),
                 zs_train_seeds=tuple(args.zs_train_seeds),
                 zs_test_seeds=tuple(args.zs_test_seeds),
                 distractor_grid=tuple(args.distractor_grid),
                 spurious_prob=args.spurious_prob,
                 mislabel_prob=args.mislabel_prob)


if __name__ == "__main__":
    _cli()
