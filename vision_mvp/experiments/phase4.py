"""Phase 4 experiments — massive-N scaling and holographic vs hierarchical.

1. Massive-N sweep: run hierarchical at N ∈ {5 000, 20 000, 50 000, 100 000}.
   Confirm O(log N) writes/ctx hold at 5 orders of magnitude.
2. Hierarchical vs holographic head-to-head.
3. Final consolidated scaling table.
"""

from __future__ import annotations
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from vision_mvp.tasks.drifting_consensus import DriftingConsensus
from vision_mvp.protocols.hierarchical import run_hierarchical
from vision_mvp.protocols.holographic import run_holographic


def make_task(n: int, d: int = 64, n_steps: int = 60,
              noise: float = 1.0, drift: float = 0.05, seed: int = 0):
    rank = max(2, math.ceil(math.log2(max(n, 2))))
    t = DriftingConsensus(n_agents=n, dim=d, intrinsic_rank=rank,
                          n_steps=n_steps, noise=noise, drift_sigma=drift,
                          seed=seed)
    t.generate()
    return t


def exp1_massive_n():
    """Push hierarchical to ever-larger N."""
    rows = []
    for n in [5_000, 20_000, 50_000, 100_000]:
        # Shorter horizon at massive N to keep wall time reasonable
        n_steps = 40 if n >= 50_000 else 60
        task = make_task(n=n, n_steps=n_steps, drift=0.05, seed=1)
        t0 = time.time()
        r = run_hierarchical(task, surprise_tau=0.5, decay=0.85, pca_lr=0.1,
                             pred_lr=0.005, pred_hidden=8, seed=1)
        wall = time.time() - t0

        # Steady state (second half of run)
        half = n_steps // 2
        est = r.estimates_over_time[half:]
        truth = task.trajectory[half:]
        truth_norm = np.linalg.norm(truth, axis=1) + 1e-8
        err = np.linalg.norm(est - truth[:, None, :], axis=2).mean(axis=1) / truth_norm

        rows.append({
            "N": n,
            "n_steps": n_steps,
            "workspace_size": r.workspace_size,
            "peak_agent_context": int(r.bus_summary["peak_agent_context"]),
            "steady_writes_per_round": float(np.mean(r.writes_per_round[half:])),
            "steady_error": float(err.mean()),
            "final_alignment": r.subspace_alignment[-1],
            "wall_sec": round(wall, 2),
            "total_tokens": int(r.bus_summary["total_tokens"]),
        })
    return rows


def exp2_hierarchical_vs_holographic():
    """Same task, compare hierarchical (O(log N) workspace) vs holographic
    (O(N^{2/3}) boundary)."""
    rows = []
    for n in [1_000, 10_000, 50_000]:
        n_steps = 60 if n < 50_000 else 40
        task = make_task(n=n, n_steps=n_steps, drift=0.05, seed=2)

        # Hierarchical
        t0 = time.time()
        rh = run_hierarchical(task, surprise_tau=0.5, decay=0.85, pca_lr=0.1,
                              pred_hidden=8, pred_lr=0.005, seed=2)
        wh = time.time() - t0

        # Holographic
        t0 = time.time()
        ro = run_holographic(task, surprise_tau=0.5, decay=0.85, pca_lr=0.1,
                             pred_lr=0.005, seed=2)
        wo = time.time() - t0

        half = n_steps // 2
        for (name, r, wall) in (("hierarchical", rh, wh), ("holographic", ro, wo)):
            est = r.estimates_over_time[half:]
            truth = task.trajectory[half:]
            tn = np.linalg.norm(truth, axis=1) + 1e-8
            err = np.linalg.norm(est - truth[:, None, :], axis=2).mean(axis=1) / tn
            row = {
                "protocol": name, "N": n, "n_steps": n_steps,
                "peak_ctx": int(r.bus_summary["peak_agent_context"]),
                "steady_writes": float(np.mean(r.writes_per_round[half:])),
                "steady_err": float(err.mean()),
                "total_tokens": int(r.bus_summary["total_tokens"]),
                "wall_sec": round(wall, 2),
            }
            if name == "holographic":
                row["boundary_size"] = r.boundary_size
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["massive", "holo", "all"], default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bundle = {}

    if args.exp in ("massive", "all"):
        print("=" * 78)
        print("EXPERIMENT 1 — Massive-N scaling (hierarchical)")
        print("=" * 78)
        rows = exp1_massive_n()
        print(f"{'N':>8} | {'steps':>5} | {'wrksp':>5} | {'peak_ctx':>8} | "
              f"{'writes/rnd':>10} | {'steady_err':>10} | {'align':>5} | "
              f"{'wall_s':>6} | {'tokens':>13}")
        for r in rows:
            print(f"{r['N']:>8} | {r['n_steps']:>5} | {r['workspace_size']:>5} | "
                  f"{r['peak_agent_context']:>8} | "
                  f"{r['steady_writes_per_round']:>10.2f} | "
                  f"{r['steady_error']:>10.4f} | "
                  f"{r['final_alignment']:>5.2f} | "
                  f"{r['wall_sec']:>6.1f} | {r['total_tokens']:>13}")
        bundle["massive"] = rows

    if args.exp in ("holo", "all"):
        print()
        print("=" * 78)
        print("EXPERIMENT 2 — Hierarchical vs Holographic")
        print("=" * 78)
        rows = exp2_hierarchical_vs_holographic()
        print(f"{'protocol':>13} | {'N':>6} | {'peak':>4} | {'writes':>6} | "
              f"{'err':>6} | {'tokens':>12} | {'wall':>5} | {'boundary':>8}")
        for r in rows:
            print(f"{r['protocol']:>13} | {r['N']:>6} | {r['peak_ctx']:>4} | "
                  f"{r['steady_writes']:>6.1f} | {r['steady_err']:>6.4f} | "
                  f"{r['total_tokens']:>12} | {r['wall_sec']:>5.1f} | "
                  f"{r.get('boundary_size', '-'):>8}")
        bundle["holo_vs_hier"] = rows

    if args.out:
        with open(args.out, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
