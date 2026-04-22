"""Example 2 — tracking a drifting truth.

The hidden truth is a random walk. The router must track it over many rounds.
Demonstrates continual adaptation, not one-shot convergence.

Run:
    python3 examples/02_drift_tracking.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vision_mvp import CASRRouter


def main():
    rng = np.random.default_rng(7)
    N, d, rank = 500, 32, 8

    basis_raw = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(basis_raw)
    z = rng.standard_normal(rank)

    router = CASRRouter(
        n_agents=N, state_dim=d, task_rank=rank,
        observation_noise=1.0, decay=0.8, seed=0,
    )

    errors = []
    for t in range(100):
        # Truth does a random walk in the low-rank subspace
        z = z + 0.05 * rng.standard_normal(rank)
        truth = Q @ z
        obs = truth[None, :] + rng.standard_normal((N, d))
        est = router.step(obs)

        truth_norm = float(np.linalg.norm(truth) + 1e-8)
        err = float(np.linalg.norm(est - truth, axis=1).mean() / truth_norm)
        errors.append(err)

    print("Drift-tracking over 100 rounds.")
    print(f"  mean error over first 20 rounds (warmup): {np.mean(errors[:20]):.4f}")
    print(f"  mean error over last 50 rounds (steady):  {np.mean(errors[50:]):.4f}")
    print(f"  peak context per agent: {router.stats['peak_context_per_agent']}")


if __name__ == "__main__":
    main()
