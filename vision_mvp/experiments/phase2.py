"""Phase 2 experiments — learned basis, drift, shock response."""

from __future__ import annotations
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.drifting_consensus import DriftingConsensus
from vision_mvp.protocols.adaptive import run_adaptive, run_naive_drift


def run_one(n: int, d: int, noise: float, n_steps: int, drift: float,
            shock_at: int | None, seed: int, which: str) -> dict:
    rank = max(2, math.ceil(math.log2(max(n, 2))))
    task = DriftingConsensus(
        n_agents=n, dim=d, intrinsic_rank=rank, n_steps=n_steps,
        noise=noise, drift_sigma=drift, shock_at=shock_at,
        shock_magnitude=5.0, seed=seed,
    )
    task.generate()
    t0 = time.time()
    if which == "adaptive":
        r = run_adaptive(task, surprise_tau=0.03, decay=0.7, pca_lr=0.1, seed=seed)
    else:
        r = run_naive_drift(task)
    wall = time.time() - t0

    return {
        "protocol": which,
        "N": n,
        "d": d,
        "rank": rank,
        "n_steps": n_steps,
        "seed": seed,
        "drift_sigma": drift,
        "shock_at": shock_at,
        "wall_sec": round(wall, 3),
        "total_tokens": r.bus_summary["total_tokens"],
        "peak_agent_context": r.bus_summary["peak_agent_context"],
        "mean_tracking_error": r.task_metrics["mean_tracking_error"],
        "oracle_tracking_error": r.task_metrics["oracle_tracking_error"],
        "final_alignment": r.subspace_alignment[-1] if r.subspace_alignment else None,
        "mean_writes_per_round": sum(r.writes_per_round) / max(len(r.writes_per_round), 1),
    }


def experiment_drift_sweep():
    """Experiment 1: adaptive vs naive across N."""
    rows = []
    for n in [20, 100, 500, 2000]:
        for s in [1, 2, 3]:
            rows.append(run_one(n=n, d=64, noise=1.0, n_steps=80,
                                drift=0.1, shock_at=None, seed=s,
                                which="adaptive"))
            if n <= 200:
                rows.append(run_one(n=n, d=64, noise=1.0, n_steps=80,
                                    drift=0.1, shock_at=None, seed=s,
                                    which="naive"))
    return rows


def experiment_shock_response():
    """Experiment 2: inject a shock at t=30. Trace recovery."""
    rank = max(2, math.ceil(math.log2(200)))
    task = DriftingConsensus(n_agents=200, dim=64, intrinsic_rank=rank,
                             n_steps=80, noise=1.0, drift_sigma=0.05,
                             shock_at=30, shock_magnitude=5.0, seed=1)
    task.generate()
    r = run_adaptive(task, surprise_tau=0.03, decay=0.7, pca_lr=0.15, seed=1)
    # Per-step tracking error
    import numpy as np
    err = np.linalg.norm(
        r.estimates_over_time - task.trajectory[:, None, :], axis=2).mean(axis=1)
    truth_norm = np.linalg.norm(task.trajectory, axis=1) + 1e-8
    rel_err = err / truth_norm
    return {
        "n": task.n_agents,
        "shock_at": task.shock_at,
        "per_step_error": rel_err.tolist(),
        "writes_per_step": r.writes_per_round,
        "alignment_per_step": r.subspace_alignment,
    }


def experiment_basis_learning():
    """Experiment 3: how fast does the learned basis align with the true one?"""
    rank = max(2, math.ceil(math.log2(500)))
    task = DriftingConsensus(n_agents=500, dim=64, intrinsic_rank=rank,
                             n_steps=60, noise=1.0, drift_sigma=0.1,
                             shock_at=None, seed=42)
    task.generate()
    r = run_adaptive(task, surprise_tau=0.03, decay=0.7, pca_lr=0.15, seed=42)
    return {
        "n": task.n_agents,
        "alignment_per_step": r.subspace_alignment,
    }


def aggregate(rows):
    groups = {}
    for r in rows:
        k = (r["protocol"], r["N"])
        groups.setdefault(k, []).append(r)
    out = []
    for (p, n), g in groups.items():
        o = {"protocol": p, "N": n, "seeds": len(g)}
        for key in ("total_tokens", "peak_agent_context", "mean_tracking_error",
                    "oracle_tracking_error", "final_alignment",
                    "mean_writes_per_round", "wall_sec"):
            vals = [r[key] for r in g if r[key] is not None]
            if vals:
                o[key] = sum(vals) / len(vals)
        out.append(o)
    out.sort(key=lambda x: (x["N"], x["protocol"]))
    return out


def format_table(agg):
    cols = ["N", "protocol", "peak_agent_context", "total_tokens",
            "mean_tracking_error", "oracle_tracking_error",
            "final_alignment", "mean_writes_per_round"]
    widths = {c: max(len(c), 14) for c in cols}
    lines = [" | ".join(c.rjust(widths[c]) for c in cols),
             "-+-".join("-" * widths[c] for c in cols)]
    for r in agg:
        cells = []
        for c in cols:
            v = r.get(c, "-")
            if isinstance(v, float):
                cells.append(f"{v:.4f}".rjust(widths[c]))
            else:
                cells.append(str(v).rjust(widths[c]))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["sweep", "shock", "basis", "all"], default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = {}

    if args.exp in ("sweep", "all"):
        print("=" * 70)
        print("EXPERIMENT 1 — Drift sweep (80 time steps, drift σ=0.1)")
        print("=" * 70)
        rows = experiment_drift_sweep()
        agg = aggregate(rows)
        print(format_table(agg))
        bundle["sweep_rows"] = rows
        bundle["sweep_aggregate"] = agg

    if args.exp in ("shock", "all"):
        print()
        print("=" * 70)
        print("EXPERIMENT 2 — Shock response at t=30 (N=200)")
        print("=" * 70)
        shock = experiment_shock_response()
        err = shock["per_step_error"]
        writes = shock["writes_per_step"]
        align = shock["alignment_per_step"]
        print(f"{'t':>4} | {'rel_err':>10} | {'writes':>7} | {'alignment':>10}")
        for t in range(0, len(err), 5):
            print(f"{t:>4} | {err[t]:>10.4f} | {writes[t]:>7d} | {align[t]:>10.4f}")
        bundle["shock"] = shock

    if args.exp in ("basis", "all"):
        print()
        print("=" * 70)
        print("EXPERIMENT 3 — Basis learning (N=500, 60 steps)")
        print("=" * 70)
        bl = experiment_basis_learning()
        align = bl["alignment_per_step"]
        print(f"{'t':>4} | {'alignment':>10}")
        for t in range(0, len(align), 5):
            print(f"{t:>4} | {align[t]:>10.4f}")
        bundle["basis_learning"] = bl

    if args.out:
        with open(args.out, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\nWrote {args.out}")
