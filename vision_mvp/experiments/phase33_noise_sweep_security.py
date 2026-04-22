"""Phase 33 — controlled extractor-noise sweep on the security-
escalation domain.

Extends the Phase-32 ``phase32_noise_sweep`` machinery to a third
domain (Phase 33 Part C). Reuses the same ``core/extractor_noise``
wrapper and the same NoiseConfig grid; the purpose is to confirm
the Theorem-P32-2 two-regime graceful-degradation bound on a
*third* decoder shape (max-ordinal severity + claim-set
classification), not just on Phase 31's priority-order and Phase
32's monotone-verdict + strict-flags decoders.

Reproducible command:

    python3 -m vision_mvp.experiments.phase33_noise_sweep_security \\
        --drop-probs 0.0 0.1 0.25 0.5 \\
        --spurious-probs 0.0 0.05 0.1 \\
        --mislabel-probs 0.0 0.25 \\
        --seeds 33 34 \\
        --out vision_mvp/results_phase33_noise_sweep_security.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.extractor_noise import (
    NoiseConfig, noisy_extractor, security_escalation_known_kinds,
)


def _run_security(k: int, seed: int, noise: NoiseConfig) -> dict:
    from vision_mvp.tasks.security_escalation import (
        MockSecurityAuditor, build_scenario_bank as _bank,
        extract_claims_for_role, run_security_loop,
        STRATEGY_SUBSTRATE,
    )
    ex = noisy_extractor(extract_claims_for_role,
                          security_escalation_known_kinds(), noise)
    bank = _bank(seed=seed, distractors_per_role=k)
    aud = MockSecurityAuditor()
    rep = run_security_loop(bank, aud,
                              strategies=(STRATEGY_SUBSTRATE,),
                              seed=seed, extractor=ex)
    return rep.pooled()[STRATEGY_SUBSTRATE]


def main() -> int:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--seeds", nargs="+", type=int, default=[33, 34])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rows: list[dict] = []

    grid = list(itertools.product(
        args.drop_probs, args.spurious_probs, args.mislabel_probs,
        args.payload_corrupt_probs, args.seeds,
        args.distractor_counts))

    print(f"[phase33/sec-noise] running {len(grid)} noise points",
          flush=True)
    for (drop, sp, mis, corr, seed, k) in grid:
        noise = NoiseConfig(drop_prob=drop, spurious_prob=sp,
                             mislabel_prob=mis,
                             payload_corrupt_prob=corr, seed=seed)
        p = _run_security(k, seed, noise)
        rows.append({
            "domain": "security",
            "drop_prob": drop, "spurious_prob": sp,
            "mislabel_prob": mis, "payload_corrupt_prob": corr,
            "seed": seed, "k": k,
            "accuracy_full": p["accuracy_full"],
            "mean_prompt_tokens": p["mean_prompt_tokens"],
            "mean_handoff_recall": p["mean_handoff_recall"],
            "mean_handoff_precision": p.get(
                "mean_handoff_precision"),
            "truncated_count": p["truncated_count"],
            "failure_hist": p["failure_hist"],
        })

    wall = time.time() - t0

    # Pool per (drop, sp, mis, corr).
    pooled: dict[str, dict] = {}
    for r in rows:
        key = (r["drop_prob"], r["spurious_prob"],
                r["mislabel_prob"], r["payload_corrupt_prob"])
        slot_key = (f"security_drop{key[0]}_sp{key[1]}_"
                     f"mis{key[2]}_corr{key[3]}")
        slot = pooled.setdefault(slot_key, {
            "domain": "security",
            "drop_prob": key[0], "spurious_prob": key[1],
            "mislabel_prob": key[2], "payload_corrupt_prob": key[3],
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
    print("=" * 90)
    print("PHASE 33 SECURITY NOISE POOLED")
    print("=" * 90)
    print("  drop   sp    mis   acc   recall  prec   tokens  fhist")
    for key in sorted(pooled.keys()):
        s = pooled[key]
        print(f"  {s['drop_prob']:>4.2f}  {s['spurious_prob']:>4.2f}  "
              f"{s['mislabel_prob']:>4.2f}  "
              f"{s['accuracy_mean']:>4.2f}  {s['recall_mean']:>4.2f}  "
              f"{s['precision_mean']:>4.2f}  "
              f"{s['tokens_mean']:>6.0f}  {s['failure_hist']}")

    payload = {
        "config": {
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
