"""Phase 47 — cross-domain capsule admission transfer.

Tests Conjecture P46-C2 in ``docs/RESULTS_CAPSULE_LEARNING.md``
§ 6: does a learned capsule admission policy trained on one
non-code domain (Phase-31 incident triage) transfer to another
(Phase-32 compliance review or Phase-33 security escalation)?

Experimental design
-------------------

Three task families — all built on the same typed-handoff
substrate (``vision_mvp.core.role_handoff``), with domain-specific
role casts, claim-kind catalogues, and scenario banks:

    D1 = incident_triage   (Phase 31; 5 scenarios, 4 roles
                             + auditor, 11 claim kinds)
    D2 = compliance_review (Phase 32; 5 scenarios, 4 roles
                             + compliance_officer, 13 kinds)
    D3 = security_escalation (Phase 33; 5 scenarios, 4 roles
                             + CISO, 15 kinds)

For each domain we:

1. Generate capsule records under the Phase-32 ``noisy_extractor``
   wrapper with matching (spurious_prob, mislabel_prob).
2. Split by seed — 80 % train, 20 % test.
3. Train a per-domain ``LearnedAdmissionPolicy`` with the
   closed feature vocabulary in
   ``vision_mvp/wevra/capsule_policy.py``.

Transfer cells
--------------

For each (train-domain, test-domain) pair:

  * **Within-domain** (diag): precision + recall on the
    training domain's held-out test set.
  * **Cross-domain** (off-diag): precision + recall on a
    *different* domain's entire capsule set (no train/test
    needed; the policy has never seen this domain's seeds).

We also evaluate a **pooled** policy trained on the union of
two domains' train splits and tested on the held-out third.

Feature-level analysis
----------------------

For each pair we dump the weight vector so a reader can inspect
which features transferred (survived in sign and magnitude) and
which are domain-specific.

Reading
-------

The transfer story is a **three-by-three matrix of admit-precision
gaps** + a **three-row pooled-held-out** table + a feature-level
attribution. If the cross-domain cells are within ~3 pp of the
within-domain diagonal, P46-C2 is supported. If the gap is
20 pp+, transfer is failing and header-level admission is
domain-specific. Empirical outcome (with all three domains
tested, ~0.6 k capsules per cross cell) is reported in
``docs/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md`` § 4.

Determinism
-----------

``noisy_extractor`` uses Python's hash() for key derivation, which
is salted per-process unless ``PYTHONHASHSEED`` is fixed. This
driver re-execs under ``PYTHONHASHSEED=0`` to guarantee
reproducible dataset collection across runs.
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


from vision_mvp.wevra.capsule import capsule_from_handoff
from vision_mvp.wevra.capsule_policy import (
    BudgetedAdmissionLedger, FIFOPolicy, KindPriorityPolicy,
    LearnedAdmissionPolicy, train_admission_policy,
    ADMIT,
)


# =============================================================================
# Domain adapters
# =============================================================================


@dataclasses.dataclass
class DomainSpec:
    """Everything this driver needs to lift one domain to capsules."""
    name: str
    build_scenario_bank: Callable
    run_handoff_protocol: Callable
    handoff_is_relevant: Callable
    extract_claims_for_role: Callable
    noisy_known_kinds: Callable
    auditor_role: str


def _incident_spec() -> DomainSpec:
    from vision_mvp.tasks.incident_triage import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_AUDITOR,
    )
    from vision_mvp.core.extractor_noise import (
        incident_triage_known_kinds,
    )
    return DomainSpec(
        name="incident",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        handoff_is_relevant=handoff_is_relevant,
        extract_claims_for_role=extract_claims_for_role,
        noisy_known_kinds=incident_triage_known_kinds,
        auditor_role=ROLE_AUDITOR,
    )


def _compliance_spec() -> DomainSpec:
    from vision_mvp.tasks.compliance_review import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_COMPLIANCE,
    )
    from vision_mvp.core.extractor_noise import (
        compliance_review_known_kinds,
    )
    return DomainSpec(
        name="compliance",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        handoff_is_relevant=handoff_is_relevant,
        extract_claims_for_role=extract_claims_for_role,
        noisy_known_kinds=compliance_review_known_kinds,
        auditor_role=ROLE_COMPLIANCE,
    )


def _security_spec() -> DomainSpec:
    from vision_mvp.tasks.security_escalation import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_CISO,
    )
    from vision_mvp.core.extractor_noise import (
        security_escalation_known_kinds,
    )
    return DomainSpec(
        name="security",
        build_scenario_bank=build_scenario_bank,
        run_handoff_protocol=run_handoff_protocol,
        handoff_is_relevant=handoff_is_relevant,
        extract_claims_for_role=extract_claims_for_role,
        noisy_known_kinds=security_escalation_known_kinds,
        auditor_role=ROLE_CISO,
    )


ALL_DOMAINS: tuple[DomainSpec, ...] = (
    _incident_spec(), _compliance_spec(), _security_spec())


# =============================================================================
# Dataset collection per domain
# =============================================================================


def collect_domain_dataset(
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
                    seed=seed * 17 + k + hash(spec.name) % 1000)
                noisy = noisy_extractor(
                    spec.extract_claims_for_role,
                    known_kinds, noise)
                router = spec.run_handoff_protocol(
                    scenario, extractor=noisy)
                inbox = router.inboxes.get(spec.auditor_role)
                if inbox is None:
                    continue
                handoffs = list(inbox.peek())
                instance_key = (
                    f"{spec.name}/{scenario.scenario_id}/k{k}/s{seed}")
                for h in handoffs:
                    cap = capsule_from_handoff(h)
                    cap = dataclasses.replace(
                        cap, n_tokens=h.n_tokens)
                    out.append({
                        "domain": spec.name,
                        "scenario_id": scenario.scenario_id,
                        "k": k, "seed": seed,
                        "instance_key": instance_key,
                        "capsule": cap, "handoff": h,
                        "is_causal": int(
                            spec.handoff_is_relevant(h, scenario)),
                    })
    return out


# =============================================================================
# Evaluation utilities
# =============================================================================


def evaluate_on(records, policy, budget):
    """Aggregate admit precision/recall over a record set at one
    budget cell.

    We group by instance_key so each scenario instance gets its
    own BudgetedAdmissionLedger (matching the Phase-46 evaluation
    shape).
    """
    by_inst: dict[str, list[dict]] = {}
    for r in records:
        by_inst.setdefault(r["instance_key"], []).append(r)
    n_admit = 0
    n_admit_causal = 0
    n_offer = 0
    n_offer_causal = 0
    for inst_key, rs in by_inst.items():
        capsules = [r["capsule"] for r in rs]
        is_causal = [r["is_causal"] for r in rs]
        bal = BudgetedAdmissionLedger(
            budget_tokens=budget, policy=policy)
        bal.offer_all_batched(capsules)
        admitted = {d.capsule_cid for d in bal.decisions
                     if d.decision == ADMIT}
        n_admit += sum(1 for c in capsules if c.cid in admitted)
        n_admit_causal += sum(
            1 for c, l in zip(capsules, is_causal)
            if c.cid in admitted and l)
        n_offer += len(capsules)
        n_offer_causal += sum(is_causal)
    return {
        "n_offer": n_offer,
        "n_offer_causal": n_offer_causal,
        "n_admit": n_admit,
        "n_admit_causal": n_admit_causal,
        "admit_precision": (
            n_admit_causal / n_admit if n_admit else 0.0),
        "admit_recall": (
            n_admit_causal / n_offer_causal
            if n_offer_causal else 0.0),
    }


def _split_by_seed(records, train_seeds, test_seeds):
    tr = [r for r in records if r["seed"] in train_seeds]
    te = [r for r in records if r["seed"] in test_seeds]
    return tr, te


# =============================================================================
# Main
# =============================================================================


def run_phase47_transfer(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        budget_grid: Sequence[int] = (16, 32, 64, 128),
        train_seeds: Sequence[int] = (31, 32, 33, 34, 35, 36, 37, 38),
        test_seeds: Sequence[int] = (39, 40),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
        n_epochs: int = 300,
        lr: float = 0.5,
        l2: float = 1e-3,
) -> dict[str, Any]:
    t0 = time.time()
    all_seeds = tuple(sorted(set(train_seeds) | set(test_seeds)))
    records_by_domain: dict[str, list[dict]] = {}
    for spec in ALL_DOMAINS:
        print(f"[phase47-transfer] collecting {spec.name}…")
        rs = collect_domain_dataset(
            spec, seeds=all_seeds,
            distractor_grid=distractor_grid,
            spurious_prob=spurious_prob,
            mislabel_prob=mislabel_prob)
        n_causal = sum(r["is_causal"] for r in rs)
        records_by_domain[spec.name] = rs
        print(f"  {spec.name}: {len(rs)} capsules "
              f"({100 * n_causal / max(1, len(rs)):.1f} % causal)")

    # Train per-domain policies.
    per_domain_policies: dict[str, LearnedAdmissionPolicy] = {}
    per_domain_train: dict[str, list[dict]] = {}
    per_domain_test: dict[str, list[dict]] = {}
    for spec in ALL_DOMAINS:
        rs = records_by_domain[spec.name]
        tr, te = _split_by_seed(rs, set(train_seeds), set(test_seeds))
        per_domain_train[spec.name] = tr
        per_domain_test[spec.name] = te
        print(f"[phase47-transfer] training learned-{spec.name} on "
              f"{len(tr)} train caps…")
        pol = train_admission_policy(
            [(r["capsule"], r["is_causal"]) for r in tr],
            n_epochs=n_epochs, lr=lr, l2=l2, seed=train_seeds[0])
        pol.name = f"learned_{spec.name}"
        per_domain_policies[spec.name] = pol

    # Transfer matrix: for each (train_domain, test_domain) pair,
    # evaluate admit precision/recall per budget.
    pairs = []
    for train_spec in ALL_DOMAINS:
        for test_spec in ALL_DOMAINS:
            pol = per_domain_policies[train_spec.name]
            rs = per_domain_test[test_spec.name]
            for B in budget_grid:
                metric = evaluate_on(rs, pol, B)
                pairs.append({
                    "train_domain": train_spec.name,
                    "test_domain": test_spec.name,
                    "policy": pol.name,
                    "budget": B,
                    "within_domain": (
                        train_spec.name == test_spec.name),
                    **metric,
                })

    # Pooled-train + held-out-domain experiment: for each held-out
    # domain D_out, train on the union of the OTHER two domains'
    # train splits and evaluate on D_out's test split.
    pooled_policies: dict[str, LearnedAdmissionPolicy] = {}
    for held_out in ALL_DOMAINS:
        pooled_examples = []
        for spec in ALL_DOMAINS:
            if spec.name == held_out.name:
                continue
            for r in per_domain_train[spec.name]:
                pooled_examples.append(
                    (r["capsule"], r["is_causal"]))
        print(f"[phase47-transfer] training pooled "
              f"(held out {held_out.name}) on "
              f"{len(pooled_examples)} caps…")
        pol = train_admission_policy(
            pooled_examples, n_epochs=n_epochs, lr=lr, l2=l2,
            seed=train_seeds[0])
        pol.name = f"pooled_holdout_{held_out.name}"
        pooled_policies[held_out.name] = pol

    pooled_rows = []
    for held_out in ALL_DOMAINS:
        pol = pooled_policies[held_out.name]
        rs = per_domain_test[held_out.name]
        for B in budget_grid:
            metric = evaluate_on(rs, pol, B)
            pooled_rows.append({
                "held_out_domain": held_out.name,
                "policy": pol.name,
                "budget": B, **metric,
            })

    # Feature-level attribution — weights that survived across
    # per-domain policies vs ones that diverge.
    feature_axes: set[str] = set()
    for pol in per_domain_policies.values():
        feature_axes.update(pol.weights.keys())
    attribution: list[dict] = []
    for f in sorted(feature_axes):
        row = {"feature": f}
        for spec in ALL_DOMAINS:
            pol = per_domain_policies[spec.name]
            row[spec.name] = round(pol.weights.get(f, 0.0), 4)
        # Transferability score: smallest-abs weight across
        # domains (features survive when they matter everywhere).
        weights = [row[spec.name] for spec in ALL_DOMAINS]
        row["min_abs"] = round(min(abs(w) for w in weights), 4)
        row["max_abs"] = round(max(abs(w) for w in weights), 4)
        row["sign_agreement"] = int(
            all(w >= 0 for w in weights)
            or all(w <= 0 for w in weights))
        attribution.append(row)

    out = {
        "schema": "wevra.phase47.cross_domain_transfer.v1",
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "distractor_grid": list(distractor_grid),
        "budgets": list(budget_grid),
        "spurious_prob": spurious_prob,
        "mislabel_prob": mislabel_prob,
        "records_per_domain": {
            d: len(rs) for d, rs in records_by_domain.items()},
        "transfer_cells": pairs,
        "pooled_holdout_rows": pooled_rows,
        "feature_attribution": attribution,
        "per_domain_weights": {
            d: p.to_dict() for d, p in per_domain_policies.items()},
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, "results_phase47_cross_domain.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase47-transfer] wrote {out_path} "
          f"({out['wall_seconds']} s)")
    _print_summary(pairs, pooled_rows, attribution)
    return out


def _print_summary(pairs, pooled_rows, attribution):
    # Transfer precision matrix at budget=32 (most selective
    # cell where learned policies have the biggest gap over
    # heuristics).
    budgets = sorted({p["budget"] for p in pairs})
    domains = sorted({p["train_domain"] for p in pairs})
    for B in (32, 64):
        print(f"\n[phase47-transfer] === admit_precision @ B={B} ===")
        hdr = f"  {'train\\test':<18s}"
        for d in domains:
            hdr += f"  {d:>12s}"
        print(hdr)
        for td in domains:
            row = f"  {td:<18s}"
            for ed in domains:
                cell = next(
                    (p for p in pairs
                     if p["train_domain"] == td
                     and p["test_domain"] == ed
                     and p["budget"] == B), None)
                row += (
                    f"  {cell['admit_precision']:>12.3f}"
                    if cell else f"  {'?':>12s}")
            print(row)

        print(f"\n[phase47-transfer] === admit_recall @ B={B} ===")
        hdr = f"  {'train\\test':<18s}"
        for d in domains:
            hdr += f"  {d:>12s}"
        print(hdr)
        for td in domains:
            row = f"  {td:<18s}"
            for ed in domains:
                cell = next(
                    (p for p in pairs
                     if p["train_domain"] == td
                     and p["test_domain"] == ed
                     and p["budget"] == B), None)
                row += (
                    f"  {cell['admit_recall']:>12.3f}"
                    if cell else f"  {'?':>12s}")
            print(row)

    print(f"\n[phase47-transfer] === POOLED held-out "
          f"admit_precision @ B=32 ===")
    print(f"  {'held-out':<18s}  {'precision':>12s}  "
          f"{'recall':>12s}")
    for row in pooled_rows:
        if row["budget"] != 32:
            continue
        print(f"  {row['held_out_domain']:<18s}  "
              f"{row['admit_precision']:>12.3f}  "
              f"{row['admit_recall']:>12.3f}")

    # Feature attribution — surviving vs domain-specific.
    print(f"\n[phase47-transfer] === FEATURE ATTRIBUTION "
          f"(sign-agreement across domains, top 15) ===")
    top = sorted(
        attribution,
        key=lambda r: -r["min_abs"])[:15]
    for r in top:
        agree = "✓" if r["sign_agreement"] else "✗"
        print(f"  {r['feature']:<28s}  "
              f"inc={r.get('incident', 0):+.3f}  "
              f"cmp={r.get('compliance', 0):+.3f}  "
              f"sec={r.get('security', 0):+.3f}  "
              f"[sign:{agree}]")


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 47: cross-domain capsule admission "
                     "transfer (P46-C2 anchor)")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--train-seeds", nargs="+", type=int,
                    default=[31, 32, 33, 34, 35, 36, 37, 38])
    p.add_argument("--test-seeds", nargs="+", type=int,
                    default=[39, 40])
    p.add_argument("--budgets", nargs="+", type=int,
                    default=[16, 32, 64, 128])
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    args = p.parse_args()
    run_phase47_transfer(
        out_dir=args.out_dir,
        train_seeds=tuple(args.train_seeds),
        test_seeds=tuple(args.test_seeds),
        budget_grid=tuple(args.budgets),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob)


if __name__ == "__main__":
    _cli()
