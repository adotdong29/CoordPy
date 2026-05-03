"""Phase 48 — bundle-aware decoder cross-domain transfer.

Companion to ``phase48_bundle_decoding.py``.  The Phase-47
admission-side transfer study (``phase47_cross_domain``)
measured whether a learned *admission* policy trained on one
non-code domain transfers to another.  The result was
asymmetric and task-family-indexed:

    incident → security: +21 pp above base rate    (strong)
    security → incident:  -3 pp                    (negative)
    compliance ↔ X:      ≤ +6 pp                   (trivial)

This driver runs the **decoder-side** analogue.  The
``LearnedBundleDecoder`` (Phase 48) stores 10 class-agnostic
shape weights; its featuriser takes a (bundle,
claim_to_canonical_label, priority_order) triple.  Cross-domain
transfer asks: do the weights learned on one domain's
(bundle, gold_label) pairs lift decoder accuracy on a
different domain, compared to the baseline PriorityDecoder on
that domain?

Setup
-----

Two operational-detection domains (the Phase-47 "transfer
works" pair):

  * D1 = incident_triage       — 5 scenarios × 5 gold root-
    causes (disk_fill / tls_expiry / dns_misroute / memory_leak /
    deadlock).
  * D2 = security_escalation   — 5 scenarios × 5 gold
    classifications (data_exfil / ransomware / supply_chain /
    insider_threat / reconnaissance).

Compliance is intentionally omitted — its decoder shape is a
priority-threshold-over-verdict, not an argmax-over-classes,
so decoder-side transfer requires a different adapter.  This
omission is the honest signal that "decoder-side transfer" is
a task-family question, not a universal one.

For each domain we:

  1. Collect noisy ``(offered_capsules, gold_label)`` instances
     using the domain's scenario bank + Phase-32 noisy extractor.
  2. Split by seed (train / test).
  3. Train a ``LearnedBundleDecoder`` on the training split.

Transfer cells
--------------

For each ordered pair ``(A, B)`` of domains:

  * ``acc_within`` — decoder-trained-on-A tested on A's test split.
  * ``acc_cross``  — decoder-trained-on-A tested on B's test split
    (using domain B's claim-to-label map + priority order at decode
    time).

We also report the baseline ``PriorityDecoder(B)`` accuracy on
B's test split so the cross-cell's lift above that baseline is
explicit.

Runs deterministically in ~30-45 s with no external ML deps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from typing import Any, Callable, Sequence

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv],
                os.environ)

from vision_mvp.coordpy.capsule import (
    ContextCapsule, capsule_from_handoff,
)
from vision_mvp.coordpy.capsule_decoder import (
    BUNDLE_DECODER_FEATURES, LearnedBundleDecoder,
    PluralityDecoder, PriorityDecoder,
    SourceCorroboratedPriorityDecoder,
    evaluate_decoder, train_learned_bundle_decoder,
)


# =============================================================================
# Domain adapters
# =============================================================================


@dataclasses.dataclass
class DomainSpec:
    """Everything this driver needs to lift one domain into
    (bundle, gold_label) instances + train a decoder on its
    shape."""
    name: str
    build_scenario_bank: Callable
    run_handoff_protocol: Callable
    extract_claims_for_role: Callable
    handoff_is_relevant: Callable
    noisy_known_kinds: Callable
    auditor_role: str
    claim_to_label: dict[str, str]
    priority_order: tuple[str, ...]
    gold_label_fn: Callable  # scenario → str
    label_alphabet: tuple[str, ...]


def _incident_spec() -> DomainSpec:
    from vision_mvp.tasks.incident_triage import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_AUDITOR,
    )
    from vision_mvp.core.extractor_noise import (
        incident_triage_known_kinds,
    )
    from vision_mvp.coordpy.capsule_policy_bundle import (
        DEFAULT_CLAIM_TO_ROOT_CAUSE, DEFAULT_PRIORITY_ORDER,
    )
    return DomainSpec(
        name="incident",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        extract_claims_for_role=extract_claims_for_role,
        handoff_is_relevant=handoff_is_relevant,
        noisy_known_kinds=incident_triage_known_kinds,
        auditor_role=ROLE_AUDITOR,
        claim_to_label=dict(DEFAULT_CLAIM_TO_ROOT_CAUSE),
        priority_order=DEFAULT_PRIORITY_ORDER,
        gold_label_fn=lambda s: s.gold_root_cause,
        label_alphabet=(
            "deadlock", "disk_fill", "dns_misroute",
            "memory_leak", "tls_expiry",
        ),
    )


# Security domain — derive claim_to_label from scenarios.  For
# each claim_kind seen in any scenario's causal chain, assign
# the gold_classification of the *first* scenario containing it.
# This gives a well-defined map even when a claim is shared
# across scenarios (priority: the scenario where it's "more
# load-bearing" is usually listed first).
def _security_spec() -> DomainSpec:
    from vision_mvp.tasks.security_escalation import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_CISO,
    )
    from vision_mvp.core.extractor_noise import (
        security_escalation_known_kinds,
    )
    # Build a claim-to-classification map by scanning one
    # instance of the scenario bank (seed is irrelevant for
    # causal_chain content).
    scenarios = build_scenario_bank(seed=31, distractors_per_role=0)
    claim_to_label: dict[str, str] = {}
    for s in scenarios:
        for (_role, kind, _payload, _evs) in s.causal_chain:
            if kind not in claim_to_label:
                claim_to_label[kind] = s.gold_classification
    # Priority order — for a causal chain, earlier claims are
    # more load-bearing.  Collect in order of first appearance
    # across scenarios.
    order: list[str] = []
    for s in scenarios:
        for (_role, kind, _payload, _evs) in s.causal_chain:
            if kind not in order:
                order.append(kind)
    return DomainSpec(
        name="security",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        extract_claims_for_role=extract_claims_for_role,
        handoff_is_relevant=handoff_is_relevant,
        noisy_known_kinds=security_escalation_known_kinds,
        auditor_role=ROLE_CISO,
        claim_to_label=claim_to_label,
        priority_order=tuple(order),
        gold_label_fn=lambda s: s.gold_classification,
        label_alphabet=tuple(sorted({
            s.gold_classification for s in scenarios})),
    )


ALL_DOMAINS: tuple[DomainSpec, ...] = (
    _incident_spec(), _security_spec())


# =============================================================================
# Dataset collection per domain
# =============================================================================


def collect_domain_instances(
        spec: DomainSpec,
        seeds: Sequence[int],
        distractor_grid: Sequence[int],
        spurious_prob: float, mislabel_prob: float,
        ) -> list[dict[str, Any]]:
    from vision_mvp.core.extractor_noise import (
        NoiseConfig, noisy_extractor,
    )
    known_kinds = spec.noisy_known_kinds()
    out: list[dict[str, Any]] = []
    for seed in seeds:
        for k in distractor_grid:
            scenarios = spec.build_scenario_bank(
                seed=seed, distractors_per_role=k)
            for scenario in scenarios:
                noise = NoiseConfig(
                    spurious_prob=spurious_prob,
                    mislabel_prob=mislabel_prob,
                    seed=seed * 17 + k
                         + hash(spec.name) % 1000)
                noisy = noisy_extractor(
                    spec.extract_claims_for_role,
                    known_kinds, noise)
                router = spec.run_handoff_protocol(
                    scenario, extractor=noisy)
                inbox = router.inboxes.get(spec.auditor_role)
                if inbox is None:
                    continue
                handoffs = list(inbox.peek())
                offered_capsules = []
                for h in handoffs:
                    cap = capsule_from_handoff(h)
                    cap = dataclasses.replace(
                        cap, n_tokens=h.n_tokens)
                    offered_capsules.append(cap)
                out.append({
                    "domain": spec.name,
                    "scenario_id": scenario.scenario_id,
                    "k": k, "seed": seed,
                    "offered_capsules": offered_capsules,
                    "gold_label": spec.gold_label_fn(scenario),
                })
    return out


def split_by_seed(instances, train_seeds, test_seeds):
    tr = [i for i in instances if i["seed"] in set(train_seeds)]
    te = [i for i in instances if i["seed"] in set(test_seeds)]
    return tr, te


# =============================================================================
# Evaluation helpers
# =============================================================================


def eval_decoder(instances, decoder):
    """Return decoder accuracy and per-scenario accuracy on a
    list of instances."""
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    scen = [i["scenario_id"] for i in instances]
    r = evaluate_decoder(pairs, decoder, scenario_keys=scen)
    return r


def train_on_instances(instances, spec: DomainSpec,
                        seed: int, n_epochs: int = 300,
                        lr: float = 0.5, l2: float = 1e-3):
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    return train_learned_bundle_decoder(
        pairs,
        rc_alphabet=spec.label_alphabet,
        claim_to_root_cause=spec.claim_to_label,
        priority_order=spec.priority_order,
        n_epochs=n_epochs, lr=lr, l2=l2, seed=seed,
    )


def make_cross_domain_decoder(
        src_decoder: LearnedBundleDecoder,
        tgt_spec: DomainSpec,
        ) -> LearnedBundleDecoder:
    """Copy learned weights from a source-domain decoder into a
    target-domain decoder (re-targeted with the target's
    alphabet + maps).  This is the operational form of
    "weight-vector transfer": same 10 floats, different
    claim_to_label + rc_alphabet.
    """
    return LearnedBundleDecoder(
        weights=dict(src_decoder.weights),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src_decoder.high_priority_cutoff,
    )


# =============================================================================
# Main
# =============================================================================


def run_phase48_transfer(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        train_seeds: Sequence[int] = (31, 32, 33, 34, 35,
                                        36, 37, 38),
        test_seeds: Sequence[int] = (39, 40),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
        n_epochs: int = 300,
) -> dict[str, Any]:
    t0 = time.time()
    all_seeds = tuple(sorted(set(train_seeds) | set(test_seeds)))

    # Collect per-domain.
    instances_by_domain: dict[str, list[dict]] = {}
    base_rates: dict[str, float] = {}
    for spec in ALL_DOMAINS:
        print(f"[phase48-transfer] collecting {spec.name}…")
        insts = collect_domain_instances(
            spec, seeds=all_seeds,
            distractor_grid=distractor_grid,
            spurious_prob=spurious_prob,
            mislabel_prob=mislabel_prob)
        print(f"  {spec.name}: {len(insts)} instances; "
              f"alphabet {spec.label_alphabet}")
        # Base rate — the maximum class frequency (best-constant
        # classifier).
        counts: dict[str, int] = {}
        for inst in insts:
            g = inst["gold_label"]
            counts[g] = counts.get(g, 0) + 1
        base = max(counts.values()) / len(insts) if insts else 0.0
        base_rates[spec.name] = base
        instances_by_domain[spec.name] = insts
        print(f"    gold distribution: {counts} "
              f"(best-constant {base:.3f})")

    # Split.
    splits: dict[str, tuple[list, list]] = {}
    for spec in ALL_DOMAINS:
        tr, te = split_by_seed(
            instances_by_domain[spec.name],
            train_seeds, test_seeds)
        splits[spec.name] = (tr, te)
        print(f"[phase48-transfer] {spec.name}: "
              f"train={len(tr)} test={len(te)}")

    # Train per-domain learned decoders.
    learned_decoders: dict[str, LearnedBundleDecoder] = {}
    for spec in ALL_DOMAINS:
        tr, _te = splits[spec.name]
        print(f"[phase48-transfer] training learned_bundle_decoder on "
              f"{spec.name} ({len(tr)} instances)…")
        dec = train_on_instances(
            tr, spec, seed=train_seeds[0], n_epochs=n_epochs)
        dec.name = f"learned_decoder_{spec.name}"
        learned_decoders[spec.name] = dec
        print(f"    weights for {spec.name}:")
        for k, w in sorted(dec.weights.items(),
                            key=lambda kv: -abs(kv[1])):
            print(f"      {k:28s}  {w:+.4f}")

    # Transfer matrix: for each (train_spec, test_spec) pair,
    # evaluate the train_spec's decoder on test_spec's test set.
    cells: list[dict[str, Any]] = []
    for train_spec in ALL_DOMAINS:
        for test_spec in ALL_DOMAINS:
            src_dec = learned_decoders[train_spec.name]
            if train_spec.name == test_spec.name:
                dec = src_dec
            else:
                dec = make_cross_domain_decoder(
                    src_dec, test_spec)
                dec.name = (f"learned_decoder_{train_spec.name}"
                            f"→{test_spec.name}")
            tr, te = splits[test_spec.name]
            result = eval_decoder(te, dec)
            # Baselines on the test_spec's test set.
            pri = PriorityDecoder(
                priority_order=test_spec.priority_order,
                claim_to_root_cause=dict(test_spec.claim_to_label))
            pri_result = eval_decoder(te, pri)
            plu = PluralityDecoder(
                priority_order=test_spec.priority_order,
                claim_to_root_cause=dict(test_spec.claim_to_label))
            plu_result = eval_decoder(te, plu)
            base = base_rates[test_spec.name]
            cells.append({
                "train_domain": train_spec.name,
                "test_domain": test_spec.name,
                "within_domain": (
                    train_spec.name == test_spec.name),
                "decoder_accuracy": result.accuracy,
                "priority_accuracy": pri_result.accuracy,
                "plurality_accuracy": plu_result.accuracy,
                "base_rate": base,
                "lift_above_base": result.accuracy - base,
                "lift_above_priority": (
                    result.accuracy - pri_result.accuracy),
                "n_test_instances": result.n_instances,
                "per_scenario_accuracy": (
                    result.per_scenario_accuracy),
            })

    # Feature-level comparison of per-domain weights.
    feature_comparison: list[dict[str, Any]] = []
    for f in BUNDLE_DECODER_FEATURES:
        row: dict[str, Any] = {"feature": f}
        for spec in ALL_DOMAINS:
            row[spec.name] = round(
                learned_decoders[spec.name].weights.get(f, 0.0), 4)
        ws = [row[spec.name] for spec in ALL_DOMAINS]
        row["sign_agreement"] = int(
            all(w >= 0 for w in ws) or all(w <= 0 for w in ws))
        row["min_abs"] = round(min(abs(w) for w in ws), 4)
        row["max_abs"] = round(max(abs(w) for w in ws), 4)
        feature_comparison.append(row)

    out = {
        "schema": "coordpy.phase48.decoder_transfer.v1",
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "distractor_grid": list(distractor_grid),
        "spurious_prob": spurious_prob,
        "mislabel_prob": mislabel_prob,
        "domains": [spec.name for spec in ALL_DOMAINS],
        "label_alphabets": {
            spec.name: list(spec.label_alphabet)
            for spec in ALL_DOMAINS},
        "base_rates": {k: round(v, 4)
                         for k, v in base_rates.items()},
        "transfer_cells": cells,
        "feature_comparison": feature_comparison,
        "per_domain_weights": {
            d: p.to_dict() for d, p in learned_decoders.items()},
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_phase48_decoder_transfer.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase48-transfer] wrote {path} "
          f"({out['wall_seconds']} s)")
    _print_summary(out)
    return out


def _print_summary(out):
    domains = out["domains"]
    base_rates = out["base_rates"]

    print(f"\n[phase48-transfer] === TRANSFER MATRIX "
          f"(decoder accuracy) ===")
    print(f"  {'train\\test':<20s}", end="")
    for d in domains:
        print(f"  {d:>14s}", end="")
    print()
    for td in domains:
        print(f"  {td:<20s}", end="")
        for ed in domains:
            cell = next((c for c in out["transfer_cells"]
                          if c["train_domain"] == td
                          and c["test_domain"] == ed), None)
            print(f"  {cell['decoder_accuracy']:>14.3f}", end="")
        print()

    print(f"\n[phase48-transfer] === BASELINE PriorityDecoder "
          f"accuracy on each test set ===")
    for d in domains:
        # All rows for test_domain=d have same priority/base_rate.
        cell = next((c for c in out["transfer_cells"]
                      if c["test_domain"] == d), None)
        print(f"  {d:<20s}  priority = {cell['priority_accuracy']:.3f}"
              f"  plurality = {cell['plurality_accuracy']:.3f}"
              f"  base_rate = {base_rates[d]:.3f}")

    print(f"\n[phase48-transfer] === CROSS-DOMAIN LIFTS ===")
    for c in out["transfer_cells"]:
        if c["within_domain"]:
            tag = "[within]"
        else:
            tag = "[cross] "
        print(f"  {tag} {c['train_domain']:<12s} → "
              f"{c['test_domain']:<12s}  "
              f"dec={c['decoder_accuracy']:.3f}  "
              f"pri={c['priority_accuracy']:.3f}  "
              f"base={c['base_rate']:.3f}  "
              f"lift_over_base={c['lift_above_base']:+.3f}  "
              f"lift_over_priority={c['lift_above_priority']:+.3f}")

    print(f"\n[phase48-transfer] === FEATURE SIGN AGREEMENT "
          f"across {len(domains)} domains ===")
    for r in out["feature_comparison"]:
        agree = "agree" if r["sign_agreement"] else "DISAGREE"
        row = f"  {r['feature']:<28s}"
        for d in domains:
            row += f"  {d}={r[d]:+.3f}"
        row += f"  [{agree}]"
        print(row)


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 48: decoder-side cross-domain transfer")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--train-seeds", nargs="+", type=int,
                    default=[31, 32, 33, 34, 35, 36, 37, 38])
    p.add_argument("--test-seeds", nargs="+", type=int,
                    default=[39, 40])
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    p.add_argument("--n-epochs", type=int, default=300)
    args = p.parse_args()
    run_phase48_transfer(
        out_dir=args.out_dir,
        train_seeds=tuple(args.train_seeds),
        test_seeds=tuple(args.test_seeds),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob,
        n_epochs=args.n_epochs)


if __name__ == "__main__":
    _cli()
