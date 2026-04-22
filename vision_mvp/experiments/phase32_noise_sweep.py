"""Phase 32 Part B — controlled extractor-noise sweep.

Falsifies or confirms Conjecture C31-7 → Theorem P32-2 by running the
Phase-31 incident-triage substrate AND the Phase-32 compliance-review
substrate under explicit, parameterised extractor noise.

Noise axes (``core/extractor_noise.NoiseConfig``):
  * drop_prob         — per-causal-claim silently dropped (recall).
  * spurious_prob     — per-benign-event spurious claim (precision).
  * mislabel_prob     — per-emission relabel-kind (type confusion).
  * payload_corrupt_prob — per-emission payload token drop.

The sweep reports, per (domain, drop, spurious, mislabel), the
substrate's full accuracy, handoff recall / precision, and failure
attribution distribution. It is deterministic per ``--seed``.

Reproducible command:

    python3 -m vision_mvp.experiments.phase32_noise_sweep --mock \\
        --domain both \\
        --drop-probs 0.0 0.1 0.25 0.5 \\
        --spurious-probs 0.0 0.05 0.1 \\
        --mislabel-probs 0.0 0.25 \\
        --seeds 31 32 \\
        --out vision_mvp/results_phase32_noise_sweep.json

Scope: this is *mock* by default (the upper-bound ceiling), because
the claim the theorem makes is on the substrate's *delivery* under
noise — the LLM's downstream transcription is a Phase-31 axis the
sweep is not trying to re-measure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.extractor_noise import (
    NoiseConfig, noisy_extractor,
    incident_triage_known_kinds, compliance_review_known_kinds,
)


def _run_incident(k: int, seed: int, noise: NoiseConfig) -> dict:
    from vision_mvp.tasks.incident_triage import (
        MockIncidentAuditor, build_scenario_bank as _bank,
        extract_claims_for_role, run_incident_loop,
        STRATEGY_SUBSTRATE,
    )
    ex = noisy_extractor(extract_claims_for_role,
                          incident_triage_known_kinds(), noise)
    bank = _bank(seed=seed, distractors_per_role=k)
    aud = MockIncidentAuditor()
    rep = run_incident_loop(bank, aud, strategies=(STRATEGY_SUBSTRATE,),
                             seed=seed, extractor=ex)
    p = rep.pooled()[STRATEGY_SUBSTRATE]
    return p


def _run_compliance(k: int, seed: int, noise: NoiseConfig) -> dict:
    from vision_mvp.tasks.compliance_review import (
        MockComplianceAuditor, build_scenario_bank as _bank,
        extract_claims_for_role, run_compliance_loop,
        STRATEGY_SUBSTRATE,
    )
    ex = noisy_extractor(extract_claims_for_role,
                          compliance_review_known_kinds(), noise)
    bank = _bank(seed=seed, distractors_per_role=k)
    aud = MockComplianceAuditor()
    rep = run_compliance_loop(bank, aud, strategies=(STRATEGY_SUBSTRATE,),
                                seed=seed, extractor=ex)
    p = rep.pooled()[STRATEGY_SUBSTRATE]
    return p


def _row(domain: str, noise: NoiseConfig, k: int, seed: int, p: dict,
         ) -> dict:
    return {
        "domain": domain,
        "drop_prob": noise.drop_prob,
        "spurious_prob": noise.spurious_prob,
        "mislabel_prob": noise.mislabel_prob,
        "payload_corrupt_prob": noise.payload_corrupt_prob,
        "seed": seed,
        "k": k,
        "accuracy_full": p["accuracy_full"],
        "mean_prompt_tokens": p["mean_prompt_tokens"],
        "mean_handoff_recall": p["mean_handoff_recall"],
        "mean_handoff_precision": p.get("mean_handoff_precision"),
        "truncated_count": p["truncated_count"],
        "failure_hist": p["failure_hist"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", default=True,
                      help="Deterministic mock auditor — the sweep "
                           "ceiling claim.")
    ap.add_argument("--domain", default="both",
                      choices=["incident", "compliance", "both"])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[20])
    ap.add_argument("--drop-probs", nargs="+", type=float,
                      default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--spurious-probs", nargs="+", type=float,
                      default=[0.0, 0.05, 0.1])
    ap.add_argument("--mislabel-probs", nargs="+", type=float,
                      default=[0.0, 0.25])
    ap.add_argument("--payload-corrupt-probs", nargs="+", type=float,
                      default=[0.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31, 32])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []

    domains: list[str] = []
    if args.domain in ("incident", "both"):
        domains.append("incident")
    if args.domain in ("compliance", "both"):
        domains.append("compliance")

    grid = list(itertools.product(
        args.drop_probs, args.spurious_probs, args.mislabel_probs,
        args.payload_corrupt_probs, args.seeds, args.distractor_counts))

    print(f"[phase32/B] running {len(grid)} noise points × "
          f"{len(domains)} domains = {len(grid) * len(domains)} runs",
          flush=True)

    for (drop, sp, mis, corr, seed, k) in grid:
        noise = NoiseConfig(
            drop_prob=drop, spurious_prob=sp,
            mislabel_prob=mis, payload_corrupt_prob=corr,
            seed=seed)
        for domain in domains:
            runner = _run_incident if domain == "incident" \
                else _run_compliance
            p = runner(k, seed, noise)
            rows.append(_row(domain, noise, k, seed, p))

    wall = time.time() - t0
    print(f"[phase32/B] wall = {wall:.1f}s", flush=True)

    # Pooled per (domain, drop, spurious, mislabel) across seeds / k.
    pooled: dict[str, dict] = {}
    for r in rows:
        key = (r["domain"], r["drop_prob"], r["spurious_prob"],
                r["mislabel_prob"], r["payload_corrupt_prob"])
        slot = pooled.setdefault(
            f"{key[0]}_drop{key[1]}_sp{key[2]}_mis{key[3]}_"
            f"corr{key[4]}",
            {"domain": r["domain"], "drop_prob": r["drop_prob"],
             "spurious_prob": r["spurious_prob"],
             "mislabel_prob": r["mislabel_prob"],
             "payload_corrupt_prob": r["payload_corrupt_prob"],
             "n_runs": 0, "accuracy_mean": 0.0,
             "recall_mean": 0.0, "precision_mean": 0.0,
             "tokens_mean": 0.0, "failure_hist": {}})
        slot["n_runs"] += 1
        slot["accuracy_mean"] += r["accuracy_full"]
        slot["recall_mean"] += r["mean_handoff_recall"]
        slot["precision_mean"] += (r["mean_handoff_precision"] or 0.0)
        slot["tokens_mean"] += r["mean_prompt_tokens"]
        for kk, vv in r["failure_hist"].items():
            slot["failure_hist"][kk] = \
                slot["failure_hist"].get(kk, 0) + vv
    for key, slot in pooled.items():
        n = max(1, slot["n_runs"])
        slot["accuracy_mean"] = round(slot["accuracy_mean"] / n, 4)
        slot["recall_mean"] = round(slot["recall_mean"] / n, 4)
        slot["precision_mean"] = round(slot["precision_mean"] / n, 4)
        slot["tokens_mean"] = round(slot["tokens_mean"] / n, 2)

    print()
    print("=" * 100)
    print("PHASE 32 / PART B POOLED — extractor-noise sweep")
    print("=" * 100)
    header = ("  domain     drop    sp      mis     acc     recall  "
              "prec    tokens  failure_hist")
    print(header)
    for key in sorted(pooled.keys()):
        s = pooled[key]
        print(f"  {s['domain']:>10}  {s['drop_prob']:>4.2f}  "
              f"{s['spurious_prob']:>4.2f}   "
              f"{s['mislabel_prob']:>4.2f}   "
              f"{s['accuracy_mean']:>4.2f}   {s['recall_mean']:>4.2f}   "
              f"{s['precision_mean']:>4.2f}   "
              f"{s['tokens_mean']:>6.0f}  {s['failure_hist']}")

    payload = {
        "config": {
            "domain": args.domain,
            "distractor_counts": args.distractor_counts,
            "drop_probs": args.drop_probs,
            "spurious_probs": args.spurious_probs,
            "mislabel_probs": args.mislabel_probs,
            "payload_corrupt_probs": args.payload_corrupt_probs,
            "seeds": args.seeds,
        },
        "rows": rows,
        "pooled": pooled,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
