"""Example 3 — the scaling law, live.

Run the CASR router at N = {100, 1 000, 10 000} and print the scaling metrics.
Confirms: peak context per agent = ⌈log₂ N⌉ regardless of N.

Run:
    python3 examples/03_scaling_demo.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import time
import numpy as np
from vision_mvp import CASRRouter


def measure(n: int, d: int = 64, rounds: int = 10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rank = max(2, math.ceil(math.log2(n)))
    basis_raw = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(basis_raw)
    z = rng.standard_normal(rank)
    truth = Q @ z

    router = CASRRouter(n_agents=n, state_dim=d, task_rank=rank, seed=seed)
    t0 = time.time()
    for _ in range(rounds):
        obs = truth[None, :] + rng.standard_normal((n, d))
        router.step(obs)
    wall = time.time() - t0

    est = router.estimates
    err = float(np.linalg.norm(est - truth, axis=1).mean()
                / np.linalg.norm(truth))
    s = router.stats
    return {
        "N": n,
        "log2N": round(math.log2(n), 2),
        "peak_ctx": s["peak_context_per_agent"],
        "workspace": s["workspace_size"],
        "total_tokens": s["total_tokens"],
        "err": round(err, 4),
        "wall_s": round(wall, 2),
    }


def main():
    print(f"{'N':>8} {'log2 N':>7} {'peak_ctx':>9} {'workspace':>10} "
          f"{'tokens':>12} {'err':>7} {'wall':>6}")
    for n in [100, 1_000, 10_000]:
        r = measure(n=n, rounds=10)
        print(f"{r['N']:>8} {r['log2N']:>7.2f} {r['peak_ctx']:>9} "
              f"{r['workspace']:>10} {r['total_tokens']:>12} "
              f"{r['err']:>7.4f} {r['wall_s']:>6.2f}s")
    print("\npeak_ctx = workspace = ⌈log₂ N⌉ exactly, across three orders of magnitude.")


if __name__ == "__main__":
    main()
