# Results — W51 Persistent Cross-Backend Latent Coordination (PXBLC)

> Sealed results notes for the W51 milestone. Post-W50 research
> milestone, 2026-05-11. Pre-committed success criterion:
> `docs/SUCCESS_CRITERION_W51_PERSISTENT_LATENT_COORDINATION.md`.
> No version bump (``coordpy.__version__`` remains ``0.5.20``).
> No PyPI release.

## Headline outcome

W51 ships **six orthogonal capsule-native advances** layered
on top of W50. **18 of 18** pre-committed H bars pass at the
declared thresholds. Cumulative trust boundary across W22..W51 =
**367 enumerated failure modes** (343 from W22..W50 + 24 new
at W51).

## H1..H18 measured outcomes (3 seeds, mean)

### R-100 family (cross-backend / persistent / transfer)

| H | Hypothesis | Threshold | Measured | Result |
|---|-----------|-----------|----------|--------|
| H1 | Trivial W51 passthrough = W50 byte-identical | `passthrough_ok = 1.0` | 1.000 | **PASS** |
| H2 | Persistent latent state long-horizon gain | `Δrecall ≥ +0.20` | +0.945 | **PASS** |
| H3 | Triple-backend transitivity direct fidelity | `≥ 0.85` | 0.887 | **PASS** |
| H3 | Triple-backend transitivity gap | `≤ 0.10` | 0.087 | **PASS** |
| H4 | Deep stack V2 L=6 structural floor + non-regression | `acc ≥ 0.65` AND `Δ ≥ -0.05` | 0.833 / +0.014 | **PASS** |
| H5 | Branch-specialised heads gain | `Δ ≥ +0.05` | +0.056 | **PASS** |
| H6 | Branch/cycle memory head gain | `Δrecall ≥ +0.15` | +0.208 | **PASS** |
| H7 | Triple-backend Ollama realism anchor | `anchor_skipped_ok = 1.0` OR `fidelity ≥ 0.60` | 1.000 (skip path) | **PASS (skip path)** |
| H8 | W51 envelope verifier soundness | `score = 1.0` | 1.000 | **PASS** |
| H9 | W51 replay determinism | `replay_ok = 1.0` | 1.000 | **PASS** |
| H10 | Cross-backend translator compromise cap | `protect_rate ≥ 0.70` | 0.944 | **PASS** |

### R-101 family (long-horizon retention / reconstruction / compression)

| H | Hypothesis | Threshold | Measured | Result |
|---|-----------|-----------|----------|--------|
| H11 | 12-turn cosine retention | `cosine ≥ 0.60` AND `Δ ≥ +0.20` | 0.707 / +0.945 | **PASS** |
| H12 | 16-turn cosine retention stretch | `cosine ≥ 0.40` | 0.796 | **PASS** |
| H13 | Reconstruction V3 MSE at k=5 | `≤ 0.50` | 0.409 | **PASS** |
| H14 | Reconstruction V3 MSE at k=8 stretch | `≤ 0.60` | 0.462 | **PASS** |
| H15 | Hierarchical compression bits/visible-token | `≥ 12.0` | 13.000 | **PASS** |
| H16 | Compression degradation curve min bits | `≥ 4.0` | 6.167 | **PASS** |
| H17 | W51 multi-block distribution cap | `protect_rate ≥ 0.70` | 0.771 | **PASS** |
| H18 | Deep stack overdepth cap reproduces | `Δ(L=6, L=4) ≤ +0.05` | -0.050 | **PASS** |

**Final verdict**: strong success. All 18 H bars met. H7
records the deterministic skip witness when the Ollama anchor
is unreachable — this is the explicit, pre-committed
``W51-L-CROSS-TOKENIZER-TRIPLE-CAP`` carry-forward path. When
``COORDPY_W51_OLLAMA_REACHABLE=1`` is set and an Ollama daemon
is running, ``examples/w51_replay_live.py`` runs the
real-LLM triple probe and records a bounded triple-backend
transitivity score.

## Cumulative theorem registry deltas

W51 adds:

* **14 theorems** (W51-T-*) — see `docs/THEOREM_REGISTRY.md`.
* **10 limitation theorems** (W51-L-*) — see same.
* **4 carry-forward conjectures** (W51-C-*) — see same.
* **24 disjoint envelope failure modes** — disjoint from W22..W50's 343.
  **Cumulative trust boundary across W22..W51 = 367 modes.**

## Per-component verdicts (honest)

| Component | Verdict |
|-----------|---------|
| M1 Persistent shared latent state V3 (GRU + cross-role mixer) | **behaviourally useful** on H2 + H11 + H12; **structurally useful** always (chain-walkable state CID) |
| M2 Triple-backend translator (with transitivity loss) | **behaviourally useful** on H3 under synthetic primary; **structurally useful** always (transitivity witness CID) |
| M3 Deep proxy stack V2 (L=6 + branch/cycle heads + temperature) | **behaviourally useful** on H5 (branch specialisation); **structurally useful** always (per-layer + per-head CIDs). **L=6 vs L=4 alone does NOT strictly win** under pure-Python autograd — honest non-monotonicity, documented at H4 + H18. |
| M4 Hierarchical compression V3 (K1=32 coarse + K2=16 fine) | **behaviourally useful** on H15 (13 bits/token at full emit); **structurally useful** always (round-trips through coarse + fine codebook CIDs) |
| M5 Long-horizon reconstruction V3 (max_k=8, two-headed) | **behaviourally useful** on H13 + H14; **structurally useful** always (degradation curve recorded) |
| M6 Branch/cycle memory head (per-page storage + consensus) | **behaviourally useful** on H6; **structurally useful** always (per-page CIDs) |
| Triple anchor (Ollama best-effort) | **skip path falsifiable**, real-LLM bounded fidelity when reachable |
| Deep stack overdepth cap (H18) | **limitation reproduces honestly** — L=6 does not strictly improve on shallow regimes |
| Rate-floor V2 cap | **limitation reproduces honestly** — 20-bit target exceeds K1=32 + K2=16 capacity (~9 bits/pair) |
| Translator compromise cap (H10) | **limitation reproduces honestly** — forged training → translator cannot recover |
| PXBLC distribution cap (H17) | **limitation reproduces honestly** — forged training cannot recover |

## What was NOT done — honest scope

* W51 does NOT touch transformer-internal hidden state, KV
  cache bytes, attention weights, embeddings, or real
  tokenizers.
* W51 does NOT close ``W47-C-DEEP-TRANSFORMER-COUPLING``,
  ``W48-C-DEEP-TRANSFORMER-COUPLING``,
  ``W48-C-REAL-KV-COUPLED-PROXY``,
  ``W48-C-MULTI-HOST-SHARED-STATE``,
  ``W49-C-DEEP-TRANSFORMER-COUPLING``,
  ``W50-C-DEEP-TRANSFORMER-COUPLING``,
  ``W50-C-REAL-KV-COUPLED-PROXY``, or
  ``W50-C-MULTI-HOST-SHARED-STATE``.
* W51 does NOT close
  ``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER``. The conjecture
  is **sharpened** forward to
  ``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY``: the triple-
  backend translator + transitivity loss is trained and
  auditable on capsule-layer carriers, but behavioural
  transitivity across **genuinely different tokenizers**
  still requires backend-side adapters out of W51 scope.
* W51 L=6 deep stack does not show strict accuracy gain
  over L=4 under pure-Python autograd. The H4 bar is a
  structural floor + non-regression, not strict gain. The
  branch/cycle-specialised heads (H5) provide the actual
  behavioural win at M3.
* W51 reconstruction V3 MSE ceiling is ~0.40-0.50 at k=5
  under W47 pure-Python autograd. Tighter convergence
  requires NumPy/JAX/PyTorch bindings.
* W51 H18 cap reproduces: L=6 may regress slightly on
  shallow regimes due to extra parameters under the bounded
  training budget.

## Files added

```
coordpy/persistent_shared_latent.py        # M1
coordpy/cross_backend_translator.py        # M2
coordpy/deep_proxy_stack_v2.py             # M3
coordpy/hierarchical_compression.py        # M4
coordpy/long_horizon_retention.py          # M5
coordpy/branch_cycle_memory.py             # M6
coordpy/w51_team.py                        # composition
coordpy/r100_benchmark.py                  # 11 families
coordpy/r101_benchmark.py                  # 8 families
tests/test_persistent_shared_latent_w51.py
tests/test_cross_backend_translator_w51.py
tests/test_deep_proxy_stack_v2_w51.py
tests/test_hierarchical_compression_w51.py
tests/test_long_horizon_retention_w51.py
tests/test_branch_cycle_memory_w51.py
tests/test_w51_trivial_passthrough_byte_identical.py
tests/test_w51_team_envelope_chain.py
tests/test_r100_benchmark.py
tests/test_r101_benchmark.py
examples/w51_smoke_driver.py
examples/w51_replay_live.py
docs/SUCCESS_CRITERION_W51_PERSISTENT_LATENT_COORDINATION.md
docs/RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md
```

Existing docs updated: `docs/RESEARCH_STATUS.md`,
`docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
`docs/HOW_NOT_TO_OVERSTATE.md`,
`docs/context_zero_master_plan.md`,
`papers/context_as_objects.md`, `CHANGELOG.md`.

## Release / version status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**.
* **No new public symbol** in `coordpy/__init__.py`. W51
  modules ship at explicit-import paths only.
* Smoke driver `tests/test_smoke_full.py` and W50 baselines
  (R-98 / R-99 / W50 envelope chain) remain byte-identical.

## Storyline post-W51

* **W43** — executable product-manifold capsules
* **W44** — live manifold-conditioned coordination
* **W45** — learned manifold controller
* **W46** — manifold memory controller
* **W47** — autograd manifold stack (pure-Python AD + Adam)
* **W48** — shared-state transformer-proxy (pseudo-KV bank)
* **W49** — multi-block cross-bank coordination (L=2 proxy)
* **W50** — cross-backend latent coordination (5 advances)
* **W51** — persistent cross-backend latent coordination
  (6 advances: GRU persistent state + triple-backend
  translator + L=6 deep stack V2 + hierarchical compression
  K1=32 × K2=16 + long-horizon reconstruction V3 max_k=8 +
  branch/cycle-specialised memory head)
