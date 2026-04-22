"""Example 1 — basic consensus.

1000 agents each see a noisy observation of a hidden truth vector. They must
agree on the truth using the CASRRouter. Shows the bare-bones API.

Run:
    python3 examples/01_basic_consensus.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vision_mvp import CASRRouter


def main():
    rng = np.random.default_rng(0)
    N, d, rank = 1000, 64, 10

    # Hidden truth (low-rank)
    basis_raw = rng.standard_normal((d, rank))
    Q, _ = np.linalg.qr(basis_raw)
    z = rng.standard_normal(rank)
    truth = Q @ z

    # Build the router
    router = CASRRouter(
        n_agents=N, state_dim=d, task_rank=rank,
        observation_noise=1.0, seed=0,
    )

    # Give each round, each agent, a fresh noisy observation
    for t in range(30):
        obs = truth[None, :] + rng.standard_normal((N, d))
        router.step(obs)

    est = router.estimates
    err = float(np.linalg.norm(est - truth, axis=1).mean()
                / np.linalg.norm(truth))

    print(f"After 30 rounds, {N} agents converge on the truth.")
    print(f"  relative tracking error: {err:.4f}")
    for k, v in router.stats.items():
        print(f"  {k}: {v}")
    print("\nNote: peak_context_per_agent = manifold_dim = workspace_size = "
          f"⌈log₂ {N}⌉ = {int(np.ceil(np.log2(N)))}")


if __name__ == "__main__":
    main()
