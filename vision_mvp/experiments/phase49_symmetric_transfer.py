"""Phase 49 — symmetric decoder-side cross-domain transfer.

Phase 48's decoder-side transfer study (``phase48_decoder_transfer``)
reported a sharply asymmetric result on two operational-detection
domains (incident, security):

    incident → security: 0.300  (full transfer, matches within)
    security → incident: 0.125  (negative, below priority 0.212)

with a feature sign-flip on ``lone_top_priority_flag``
(+1.13 on incident, −1.19 on security) as the structural
signature.  Conjecture W3-C6 states the asymmetry is
task-family-indexed.  Conjecture W3-C7 names symmetric
transfer as the second gate of the paradigm-shift bar
(within-domain minus 5 pp).

This driver attacks the asymmetric-transfer half of W3-C7 on
four routes:

  1. **V2 linear transfer** — ``LearnedBundleDecoderV2`` uses
     the richer V2 feature vocabulary with domain-invariant
     relative features (rank, delta-vs-best-other, ratio).
     Hypothesis: a linear decoder over V2 features has more
     symmetric transfer because the domain-invariant relative
     features do NOT sign-flip across task families.

  2. **DeepSet transfer** — ``DeepSetBundleDecoder`` with
     per-capsule φ + sum + MLP.  Hypothesis: the deeper
     hypothesis class can internalise the domain sign-flip
     by learning conditional non-linear combinations.

  3. **Multitask** — ``MultitaskBundleDecoder`` pooled-trained
     on both domains with domain-specific heads.  At test
     time each domain applies its own head.  Measures:
       * within-domain with multitask vs within-domain alone
         (does pooled training help?)
       * cross-weight transfer (drop the domain head, use only
         shared weights) — is the shared head symmetric?

  4. **Feature-level sign-agreement** — re-measure the
     sign-agreement rate across domains for each feature class
     (V1, V2-delta, V2-rank).  Hypothesis W3-C8: V2 features
     disagree less often than V1 features across domains.

Setup mirrors Phase 48's decoder-transfer driver for apples-
to-apples comparison: two domains (incident, security), Phase-32
noisy extractor, 20 seeds, by-seed 16/4 split, spurious=0.30,
mislabel=0.10.

Theoretical anchor:
  * ``docs/CAPSULE_FORMALISM.md`` § 4.D, Theorems W3-21 /
    W3-22, Conjecture W3-C8.
  * ``docs/RESULTS_CAPSULE_RESEARCH_MILESTONE4.md`` § 2.
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
    LearnedBundleDecoder, PluralityDecoder, PriorityDecoder,
    evaluate_decoder, train_learned_bundle_decoder,
)
from vision_mvp.coordpy.capsule_decoder_v2 import (
    BUNDLE_DECODER_FEATURES_V2,
    LearnedBundleDecoderV2, train_learned_bundle_decoder_v2,
    DeepSetBundleDecoder, train_deep_set_bundle_decoder,
    MultitaskBundleDecoder, train_multitask_bundle_decoder,
)


@dataclasses.dataclass
class DomainSpec:
    name: str
    build_scenario_bank: Callable
    run_handoff_protocol: Callable
    extract_claims_for_role: Callable
    handoff_is_relevant: Callable
    noisy_known_kinds: Callable
    auditor_role: str
    claim_to_label: dict[str, str]
    priority_order: tuple[str, ...]
    gold_label_fn: Callable
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


def _security_spec() -> DomainSpec:
    from vision_mvp.tasks.security_escalation import (
        build_scenario_bank, run_handoff_protocol,
        handoff_is_relevant, extract_claims_for_role,
        ROLE_CISO,
    )
    from vision_mvp.core.extractor_noise import (
        security_escalation_known_kinds,
    )
    scenarios = build_scenario_bank(seed=31, distractors_per_role=0)
    claim_to_label: dict[str, str] = {}
    for s in scenarios:
        for (_role, kind, _payload, _evs) in s.causal_chain:
            if kind not in claim_to_label:
                claim_to_label[kind] = s.gold_classification
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


def collect_domain_instances(
        spec: DomainSpec, seeds: Sequence[int],
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


def eval_decoder(instances, decoder):
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    scen = [i["scenario_id"] for i in instances]
    r = evaluate_decoder(pairs, decoder, scenario_keys=scen)
    return r


def train_on_domain_v1(instances, spec: DomainSpec,
                        seed: int, n_epochs: int = 300):
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    return train_learned_bundle_decoder(
        pairs, rc_alphabet=spec.label_alphabet,
        claim_to_root_cause=spec.claim_to_label,
        priority_order=spec.priority_order,
        n_epochs=n_epochs, lr=0.5, l2=1e-3, seed=seed)


def train_on_domain_v2(instances, spec: DomainSpec,
                        seed: int, n_epochs: int = 500):
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    return train_learned_bundle_decoder_v2(
        pairs, rc_alphabet=spec.label_alphabet,
        claim_to_root_cause=spec.claim_to_label,
        priority_order=spec.priority_order,
        n_epochs=n_epochs, lr=0.5, l2=1e-3, seed=seed)


def train_on_domain_deepset(instances, spec: DomainSpec,
                              seed: int, n_epochs: int = 500,
                              hidden_size: int = 10):
    pairs = [(i["offered_capsules"], i["gold_label"])
              for i in instances]
    return train_deep_set_bundle_decoder(
        pairs, rc_alphabet=spec.label_alphabet,
        claim_to_root_cause=spec.claim_to_label,
        priority_order=spec.priority_order,
        hidden_size=hidden_size,
        n_epochs=n_epochs, lr=0.1, l2=1e-3, seed=seed)


def make_cross_v1(src, tgt_spec: DomainSpec):
    return LearnedBundleDecoder(
        weights=dict(src.weights),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_v2(src, tgt_spec: DomainSpec):
    return LearnedBundleDecoderV2(
        weights=dict(src.weights),
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def make_cross_deepset(src, tgt_spec: DomainSpec):
    return DeepSetBundleDecoder(
        W1=src.W1.copy(), b1=src.b1.copy(),
        w2=src.w2.copy(), b2=src.b2,
        hidden_size=src.hidden_size,
        rc_alphabet=tgt_spec.label_alphabet,
        claim_to_root_cause=dict(tgt_spec.claim_to_label),
        priority_order=tuple(tgt_spec.priority_order),
        high_priority_cutoff=src.high_priority_cutoff)


def run_phase49_transfer(
        out_dir: str = ".",
        distractor_grid: Sequence[int] = (6, 20, 60, 120),
        train_seeds: Sequence[int] = (31, 32, 33, 34, 35,
                                        36, 37, 38),
        test_seeds: Sequence[int] = (39, 40),
        spurious_prob: float = 0.30,
        mislabel_prob: float = 0.10,
        n_epochs: int = 500,
) -> dict[str, Any]:
    t0 = time.time()
    all_seeds = tuple(sorted(set(train_seeds) | set(test_seeds)))

    instances_by_domain: dict[str, list[dict]] = {}
    base_rates: dict[str, float] = {}
    for spec in ALL_DOMAINS:
        print(f"[phase49-transfer] collecting {spec.name}…")
        insts = collect_domain_instances(
            spec, seeds=all_seeds,
            distractor_grid=distractor_grid,
            spurious_prob=spurious_prob,
            mislabel_prob=mislabel_prob)
        print(f"  {spec.name}: {len(insts)} instances; "
              f"alphabet {spec.label_alphabet}")
        counts: dict[str, int] = {}
        for inst in insts:
            g = inst["gold_label"]
            counts[g] = counts.get(g, 0) + 1
        base = max(counts.values()) / len(insts) if insts else 0.0
        base_rates[spec.name] = base
        instances_by_domain[spec.name] = insts

    splits: dict[str, tuple[list, list]] = {}
    for spec in ALL_DOMAINS:
        tr, te = split_by_seed(
            instances_by_domain[spec.name],
            train_seeds, test_seeds)
        splits[spec.name] = (tr, te)
        print(f"[phase49-transfer] {spec.name}: "
              f"train={len(tr)} test={len(te)}")

    # ------------------------------------------------------------
    # Per-domain V1 + V2 + DeepSet decoders.
    # ------------------------------------------------------------
    dec_v1: dict[str, Any] = {}
    dec_v2: dict[str, Any] = {}
    dec_ds: dict[str, Any] = {}
    for spec in ALL_DOMAINS:
        tr, _te = splits[spec.name]
        print(f"[phase49-transfer] training V1 decoder on "
              f"{spec.name} ({len(tr)} insts)…")
        d1 = train_on_domain_v1(tr, spec, seed=train_seeds[0])
        d1.name = f"v1_{spec.name}"
        dec_v1[spec.name] = d1
        print(f"[phase49-transfer] training V2 decoder on "
              f"{spec.name}…")
        d2 = train_on_domain_v2(tr, spec, seed=train_seeds[0])
        d2.name = f"v2_{spec.name}"
        dec_v2[spec.name] = d2
        print(f"[phase49-transfer] training DeepSet decoder on "
              f"{spec.name}…")
        dds = train_on_domain_deepset(
            tr, spec, seed=train_seeds[0], hidden_size=10)
        dds.name = f"deepset_{spec.name}"
        dec_ds[spec.name] = dds

    # ------------------------------------------------------------
    # Multitask decoder: pooled training across both domains.
    # ------------------------------------------------------------
    print(f"[phase49-transfer] training MultitaskBundleDecoder "
          f"(pooled training)…")
    pooled_train: list[tuple[list[ContextCapsule], str, str]] = []
    for spec in ALL_DOMAINS:
        tr, _te = splits[spec.name]
        for inst in tr:
            pooled_train.append(
                (inst["offered_capsules"],
                 inst["gold_label"],
                 spec.name))
    domain_specs = {
        spec.name: {
            "rc_alphabet": list(spec.label_alphabet),
            "claim_to_root_cause": dict(spec.claim_to_label),
            "priority_order": list(spec.priority_order),
        }
        for spec in ALL_DOMAINS
    }
    mt = train_multitask_bundle_decoder(
        pooled_train, domain_specs=domain_specs,
        n_epochs=n_epochs, lr=0.3, l2_shared=1e-3,
        l2_domain=5e-3, seed=train_seeds[0])

    # ------------------------------------------------------------
    # Transfer matrices: (train, test, decoder-family).
    # ------------------------------------------------------------
    cells: list[dict[str, Any]] = []
    for train_spec in ALL_DOMAINS:
        for test_spec in ALL_DOMAINS:
            _, te = splits[test_spec.name]
            within = (train_spec.name == test_spec.name)
            # V1 (Phase-48 baseline):
            v1 = (dec_v1[train_spec.name] if within
                   else make_cross_v1(dec_v1[train_spec.name],
                                        test_spec))
            v1r = eval_decoder(te, v1)
            # V2:
            v2 = (dec_v2[train_spec.name] if within
                   else make_cross_v2(dec_v2[train_spec.name],
                                        test_spec))
            v2r = eval_decoder(te, v2)
            # DeepSet:
            ds = (dec_ds[train_spec.name] if within
                   else make_cross_deepset(dec_ds[train_spec.name],
                                             test_spec))
            dsr = eval_decoder(te, ds)
            # Priority baseline on test.
            pri = PriorityDecoder(
                priority_order=test_spec.priority_order,
                claim_to_root_cause=dict(test_spec.claim_to_label))
            prir = eval_decoder(te, pri)
            cells.append({
                "train_domain": train_spec.name,
                "test_domain": test_spec.name,
                "within_domain": within,
                "v1_accuracy": v1r.accuracy,
                "v2_accuracy": v2r.accuracy,
                "deepset_accuracy": dsr.accuracy,
                "priority_accuracy": prir.accuracy,
                "base_rate": base_rates[test_spec.name],
                "n_test_instances": v1r.n_instances,
            })

    # ------------------------------------------------------------
    # Multitask rows: per-domain head + shared-head-only.
    # ------------------------------------------------------------
    multitask_cells: list[dict[str, Any]] = []
    for test_spec in ALL_DOMAINS:
        _, te = splits[test_spec.name]
        pri = PriorityDecoder(
            priority_order=test_spec.priority_order,
            claim_to_root_cause=dict(test_spec.claim_to_label))
        prir = eval_decoder(te, pri)
        # Per-domain head.
        mt_dom = mt.set_domain(
            test_spec.name,
            test_spec.label_alphabet,
            test_spec.claim_to_label,
            test_spec.priority_order)
        mt_dom.name = f"multitask_{test_spec.name}"
        mt_r = eval_decoder(te, mt_dom)
        # Shared head only — zero-out the domain head.
        mt_shared = dataclasses.replace(
            mt,
            w_shared=dict(mt.w_shared),
            w_domain={d: {k: 0.0 for k in w}
                        for d, w in mt.w_domain.items()},
        ).set_domain(
            test_spec.name,
            test_spec.label_alphabet,
            test_spec.claim_to_label,
            test_spec.priority_order)
        mt_shared.name = f"multitask_shared_{test_spec.name}"
        mt_sr = eval_decoder(te, mt_shared)
        multitask_cells.append({
            "test_domain": test_spec.name,
            "priority_accuracy": prir.accuracy,
            "base_rate": base_rates[test_spec.name],
            "multitask_per_domain_acc": mt_r.accuracy,
            "multitask_shared_acc": mt_sr.accuracy,
            "n_test_instances": mt_r.n_instances,
        })

    # ------------------------------------------------------------
    # Feature sign-agreement on V2 weights.
    # ------------------------------------------------------------
    feature_comparison: list[dict[str, Any]] = []
    for f in BUNDLE_DECODER_FEATURES_V2:
        row: dict[str, Any] = {"feature": f}
        for spec in ALL_DOMAINS:
            row[spec.name] = round(
                dec_v2[spec.name].weights.get(f, 0.0), 4)
        ws = [row[spec.name] for spec in ALL_DOMAINS]
        row["sign_agreement"] = int(
            all(w >= 0 for w in ws) or all(w <= 0 for w in ws))
        row["max_abs"] = round(max(abs(w) for w in ws), 4)
        feature_comparison.append(row)

    # V1-vs-V2 sign-agreement rate across features.
    v1_sign_agreement_rate = None  # compute from the
                                    # Phase-48 weights cached
                                    # in the per-domain V1 objects.
    v1_signs: dict[str, dict[str, float]] = {}
    for spec in ALL_DOMAINS:
        v1_signs[spec.name] = dict(dec_v1[spec.name].weights)
    v1_agree = 0
    v1_total = 0
    for f in v1_signs["incident"].keys():
        ws = [v1_signs[spec.name].get(f, 0.0)
              for spec in ALL_DOMAINS]
        v1_agree += int(all(w >= 0 for w in ws)
                         or all(w <= 0 for w in ws))
        v1_total += 1
    v1_sign_agreement_rate = (
        v1_agree / v1_total) if v1_total else 0.0
    v2_sign_agreement_rate = (
        sum(r["sign_agreement"] for r in feature_comparison)
        / max(1, len(feature_comparison)))

    # ------------------------------------------------------------
    # Summary metrics.
    # ------------------------------------------------------------
    def _cell(train, test, family):
        for c in cells:
            if (c["train_domain"] == train
                and c["test_domain"] == test):
                return c[f"{family}_accuracy"]
        return None

    summary = {}
    for fam in ("v1", "v2", "deepset"):
        within_inc = _cell("incident", "incident", fam)
        within_sec = _cell("security", "security", fam)
        cross_i2s = _cell("incident", "security", fam)
        cross_s2i = _cell("security", "incident", fam)
        symmetry_gap = abs(cross_i2s - cross_s2i)
        transfer_penalty_i2s = within_sec - cross_i2s
        transfer_penalty_s2i = within_inc - cross_s2i
        symmetric_transfer_bar_5pp_met = (
            transfer_penalty_i2s <= 0.05
            and transfer_penalty_s2i <= 0.05)
        summary[fam] = {
            "within_incident": within_inc,
            "within_security": within_sec,
            "incident_to_security": cross_i2s,
            "security_to_incident": cross_s2i,
            "symmetry_gap": symmetry_gap,
            "transfer_penalty_i2s": transfer_penalty_i2s,
            "transfer_penalty_s2i": transfer_penalty_s2i,
            "symmetric_transfer_bar_5pp_met":
                symmetric_transfer_bar_5pp_met,
        }

    out = {
        "schema": "coordpy.phase49.symmetric_transfer.v1",
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
        "multitask_cells": multitask_cells,
        "feature_comparison_v2": feature_comparison,
        "v1_sign_agreement_rate": v1_sign_agreement_rate,
        "v2_sign_agreement_rate": v2_sign_agreement_rate,
        "per_domain_v1_weights": {
            spec.name: dict(dec_v1[spec.name].weights)
            for spec in ALL_DOMAINS},
        "per_domain_v2_weights": {
            spec.name: dict(dec_v2[spec.name].weights)
            for spec in ALL_DOMAINS},
        "multitask_shared_weights": dict(mt.w_shared),
        "multitask_domain_weights": {
            d: dict(w) for d, w in mt.w_domain.items()},
        "summary": summary,
        "wall_seconds": round(time.time() - t0, 3),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir,
                         "results_phase49_symmetric_transfer.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[phase49-transfer] wrote {path} "
          f"({out['wall_seconds']} s)")
    _print_summary(out)
    return out


def _print_summary(out):
    domains = out["domains"]
    print(f"\n[phase49-transfer] === TRANSFER MATRICES "
          f"(decoder accuracy by family) ===")
    for fam in ("v1", "v2", "deepset"):
        print(f"\n  --- {fam} ---")
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
                print(f"  {cell[f'{fam}_accuracy']:>14.3f}", end="")
            print()

    print(f"\n[phase49-transfer] === MULTITASK "
          f"(pooled training, per-domain head) ===")
    for row in out["multitask_cells"]:
        print(f"  {row['test_domain']:<15s}  "
              f"per-domain-head = {row['multitask_per_domain_acc']:.3f}  "
              f"shared-head-only = {row['multitask_shared_acc']:.3f}  "
              f"priority = {row['priority_accuracy']:.3f}")

    print(f"\n[phase49-transfer] === SYMMETRY SUMMARY ===")
    for fam in ("v1", "v2", "deepset"):
        s = out["summary"][fam]
        label = ("SYMMETRIC" if s["symmetric_transfer_bar_5pp_met"]
                  else "ASYMMETRIC")
        print(f"  {fam:<10s}  {label:<12s}  "
              f"i→s={s['incident_to_security']:.3f}  "
              f"s→i={s['security_to_incident']:.3f}  "
              f"gap={s['symmetry_gap']:.3f}  "
              f"penalty_i→s={s['transfer_penalty_i2s']:+.3f}  "
              f"penalty_s→i={s['transfer_penalty_s2i']:+.3f}")

    print(f"\n[phase49-transfer] === FEATURE SIGN AGREEMENT "
          f"(V2 weights) ===")
    print(f"  V1 feature sign-agreement rate: "
          f"{out['v1_sign_agreement_rate']:.3f}")
    print(f"  V2 feature sign-agreement rate: "
          f"{out['v2_sign_agreement_rate']:.3f}")
    for r in out["feature_comparison_v2"]:
        agree = "agree" if r["sign_agreement"] else "DISAGREE"
        row = f"  {r['feature']:<32s}"
        for d in domains:
            row += f"  {d}={r[d]:+.3f}"
        row += f"  [{agree}]"
        print(row)


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 49: symmetric decoder-side "
                     "cross-domain transfer")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--train-seeds", nargs="+", type=int,
                    default=[31, 32, 33, 34, 35, 36, 37, 38])
    p.add_argument("--test-seeds", nargs="+", type=int,
                    default=[39, 40])
    p.add_argument("--distractor-grid", nargs="+", type=int,
                    default=[6, 20, 60, 120])
    p.add_argument("--spurious-prob", type=float, default=0.30)
    p.add_argument("--mislabel-prob", type=float, default=0.10)
    p.add_argument("--n-epochs", type=int, default=500)
    args = p.parse_args()
    run_phase49_transfer(
        out_dir=args.out_dir,
        train_seeds=tuple(args.train_seeds),
        test_seeds=tuple(args.test_seeds),
        distractor_grid=tuple(args.distractor_grid),
        spurious_prob=args.spurious_prob,
        mislabel_prob=args.mislabel_prob,
        n_epochs=args.n_epochs)


if __name__ == "__main__":
    _cli()
