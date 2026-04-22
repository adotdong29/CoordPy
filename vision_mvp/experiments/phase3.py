"""Phase 3 experiments.

1. Long-horizon steady state (500 steps) — does everything eventually work?
2. Scale sweep with workspace — writes/round should be O(log N), not O(N).
3. Write-reduction over time — does surprise filter eventually fire?
4. Shock response with full stack.
"""

from __future__ import annotations
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from vision_mvp.tasks.drifting_consensus import DriftingConsensus
from vision_mvp.protocols.hierarchical import run_hierarchical
from vision_mvp.protocols.adaptive import run_adaptive, run_naive_drift


def make_task(n: int, d: int = 64, n_steps: int = 150,
              noise: float = 1.0, drift: float = 0.1,
              shock_at: int | None = None, seed: int = 0):
    rank = max(2, math.ceil(math.log2(max(n, 2))))
    t = DriftingConsensus(n_agents=n, dim=d, intrinsic_rank=rank,
                          n_steps=n_steps, noise=noise, drift_sigma=drift,
                          shock_at=shock_at, seed=seed)
    t.generate()
    return t


def exp1_long_horizon():
    """Long-horizon: 500 steps, N=500. Measure steady-state post warm-up."""
    task = make_task(n=500, n_steps=500, drift=0.05, seed=1)
    t0 = time.time()
    r = run_hierarchical(task, surprise_tau=0.5, pca_lr=0.12, seed=1)
    wall = time.time() - t0

    est = r.estimates_over_time
    truth = task.trajectory
    truth_norm = np.linalg.norm(truth, axis=1) + 1e-8
    per_t_err = np.linalg.norm(est - truth[:, None, :], axis=2).mean(axis=1) / truth_norm

    warm_up = per_t_err[:50].mean()
    steady = per_t_err[100:].mean()
    mean_writes = np.mean(r.writes_per_round)
    warmup_writes = np.mean(r.writes_per_round[:50])
    steady_writes = np.mean(r.writes_per_round[100:])
    return {
        "N": task.n_agents,
        "steps": task.n_steps,
        "workspace_size": r.workspace_size,
        "wall_sec": round(wall, 2),
        "warmup_err": float(warm_up),
        "steady_err": float(steady),
        "final_alignment": r.subspace_alignment[-1],
        "mean_writes": float(mean_writes),
        "warmup_writes": float(warmup_writes),
        "steady_writes": float(steady_writes),
        "total_tokens": r.bus_summary["total_tokens"],
        "peak_agent_context": r.bus_summary["peak_agent_context"],
        "oracle_err": task.evaluate_tracking(est)["oracle_tracking_error"],
    }


def exp2_scale_sweep():
    """Does workspace cap writes at O(log N)?"""
    rows = []
    for n in [50, 200, 1000, 5000]:
        task = make_task(n=n, n_steps=200, drift=0.05, seed=1)
        r = run_hierarchical(task, surprise_tau=0.5, pca_lr=0.15, seed=1)
        steady_writes = float(np.mean(r.writes_per_round[100:]))
        max_writes = max(r.writes_per_round)
        err_steady = float(np.linalg.norm(
            r.estimates_over_time[100:] - task.trajectory[100:, None, :],
            axis=2).mean() / (np.linalg.norm(task.trajectory[100:], axis=1).mean() + 1e-8))
        rows.append({
            "N": n,
            "workspace_size": r.workspace_size,
            "steady_writes_per_round": steady_writes,
            "max_writes": max_writes,
            "steady_err": err_steady,
            "total_tokens": r.bus_summary["total_tokens"],
            "peak_agent_context": r.bus_summary["peak_agent_context"],
            "final_alignment": r.subspace_alignment[-1],
        })
    return rows


def exp3_shock():
    """Shock at t=100, N=500, 300 steps.

    Slower decay + slower PCA so basis doesn't re-pivot mid-run.
    """
    task = make_task(n=500, n_steps=300, drift=0.05, shock_at=100, seed=1)
    r = run_hierarchical(task, surprise_tau=0.5, decay=0.9, pca_lr=0.06, seed=1)
    est = r.estimates_over_time
    truth = task.trajectory
    truth_norm = np.linalg.norm(truth, axis=1) + 1e-8
    err = np.linalg.norm(est - truth[:, None, :], axis=2).mean(axis=1) / truth_norm
    return {
        "per_step_error": err.tolist(),
        "writes_per_step": r.writes_per_round,
        "alignment_per_step": r.subspace_alignment,
        "workspace_size": r.workspace_size,
    }


def exp4_phase_comparison():
    """Head-to-head at the same task: naive vs Phase-2 adaptive vs Phase-3 hierarchical."""
    rows = []
    for n in [100, 500]:
        task = make_task(n=n, n_steps=200, drift=0.05, seed=2)
        # Naive
        if n <= 200:
            r = run_naive_drift(task)
            rows.append(dict(protocol="naive", N=n, **{
                "total_tokens": r.bus_summary["total_tokens"],
                "peak_agent_context": r.bus_summary["peak_agent_context"],
                "mean_tracking_error": r.task_metrics["mean_tracking_error"],
                "writes_per_round": r.writes_per_round[0],
            }))
        # Phase 2 adaptive
        r2 = run_adaptive(task, surprise_tau=0.03, decay=0.7, pca_lr=0.15, seed=2)
        rows.append(dict(protocol="phase2", N=n, **{
            "total_tokens": r2.bus_summary["total_tokens"],
            "peak_agent_context": r2.bus_summary["peak_agent_context"],
            "mean_tracking_error": r2.task_metrics["mean_tracking_error"],
            "writes_per_round": sum(r2.writes_per_round) / len(r2.writes_per_round),
        }))
        # Phase 3 hierarchical
        r3 = run_hierarchical(task, surprise_tau=0.5, pca_lr=0.15, seed=2)
        rows.append(dict(protocol="phase3", N=n, **{
            "total_tokens": r3.bus_summary["total_tokens"],
            "peak_agent_context": r3.bus_summary["peak_agent_context"],
            "mean_tracking_error": r3.task_metrics["mean_tracking_error"],
            "writes_per_round": sum(r3.writes_per_round) / len(r3.writes_per_round),
        }))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["long", "scale", "shock", "compare", "all"],
                    default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bundle = {}

    if args.exp in ("long", "all"):
        print("=" * 70)
        print("EXPERIMENT 1 — Long horizon (N=500, 500 steps)")
        print("=" * 70)
        r = exp1_long_horizon()
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        bundle["long"] = r

    if args.exp in ("scale", "all"):
        print()
        print("=" * 70)
        print("EXPERIMENT 2 — Scale sweep (does workspace keep writes O(log N)?)")
        print("=" * 70)
        rows = exp2_scale_sweep()
        print(f"{'N':>6} | {'wrksp':>5} | {'steady_writes':>13} | {'max_writes':>10} | "
              f"{'steady_err':>10} | {'align':>5} | {'peak_ctx':>8}")
        for r in rows:
            print(f"{r['N']:>6} | {r['workspace_size']:>5} | "
                  f"{r['steady_writes_per_round']:>13.2f} | {r['max_writes']:>10} | "
                  f"{r['steady_err']:>10.4f} | {r['final_alignment']:>5.2f} | "
                  f"{int(r['peak_agent_context']):>8}")
        bundle["scale"] = rows

    if args.exp in ("shock", "all"):
        print()
        print("=" * 70)
        print("EXPERIMENT 3 — Shock at t=100 (N=500)")
        print("=" * 70)
        s = exp3_shock()
        err = s["per_step_error"]
        writes = s["writes_per_step"]
        for t in range(0, len(err), 20):
            print(f"  t={t:>4}  err={err[t]:.4f}  writes={writes[t]:>4}")
        bundle["shock"] = s

    if args.exp in ("compare", "all"):
        print()
        print("=" * 70)
        print("EXPERIMENT 4 — Phase comparison")
        print("=" * 70)
        rows = exp4_phase_comparison()
        print(f"{'protocol':>10} | {'N':>5} | {'tracking_err':>12} | {'tokens':>12} | "
              f"{'peak_ctx':>8} | {'writes/rnd':>10}")
        for r in rows:
            print(f"{r['protocol']:>10} | {r['N']:>5} | "
                  f"{r['mean_tracking_error']:>12.4f} | "
                  f"{int(r['total_tokens']):>12} | "
                  f"{int(r['peak_agent_context']):>8} | "
                  f"{r['writes_per_round']:>10.1f}")
        bundle["compare"] = rows

    if args.out:
        with open(args.out, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
