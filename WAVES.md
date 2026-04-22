# Context-Zero — Waves 1–5 build-out

Implementation trail of ~80 new primitives added beyond the 72-framework
theoretical survey. All additions are NumPy-only at the core; scipy, crypto,
and torch are optional extras (`[scientific]`, `[crypto]`, `[dl]`, `[heavy]`).

The drive here is **mechanism-level**: every item ships as a working
reference implementation with unit tests, not just a theoretical hook.

## Test surface

| Before build | After build |
|---|---|
| 94 tests | **455 tests** (all passing) |

Runtime: ~3.5 s for the full suite on a laptop.

---

## Wave 1 — "Resolve the open questions" (Tier 0)

Eight items that directly close or materially strengthen open questions in
`OPEN_QUESTIONS.md`.

| ID | Module | Resolves |
|---|---|---|
| E2 | `core/contraction.py` | OQ1 — contraction certificate |
| C2 | `core/ctw_predictor.py` | OQ3 — universal prediction |
| F1 | `core/pac_bayes.py` | OQ6 — β = PAC-Bayes Lagrange multiplier |
| F5 | `core/meta_learn.py` | OQ2 — Reptile meta-learning for scale inference |
| E1 | `core/event_triggered_control.py` | Stage-3 foundation (Lyapunov σ synthesis) |
| F6 | `core/conformal.py` | Stage-3 calibration (distribution-free coverage) |
| D7 | `core/cuckoo_filter.py` | OQ5 — adversarially-robust filter |
| G4 | `core/shapley.py` | per-agent credit attribution |

## Wave 2 — "Diagnostics pack" (Tier 0)

Measurement-only primitives. Surfaced through reports; no behaviour change.

| ID | Module | Purpose |
|---|---|---|
| B2 | `core/diagnostics/eth.py` | ETH thermal-ergodicity check |
| B3 | `core/diagnostics/kuramoto.py` | synchronization order parameter r(t) |
| B4 | `core/diagnostics/bkt.py` | vortex unbinding / fragmentation diagnostic |
| B5 | `core/diagnostics/percolation.py` | directed-percolation survival |
| B7 | `core/diagnostics/thermo_length.py` | Sivak–Crooks dissipation lower bound |
| F4 | `core/svgd.py` | Stein Variational Gradient Descent |
| G3 | `core/lmsr.py` | logarithmic market scoring rule |
| A4 | `core/embeddings.py` | JL and Bourgain embeddings |
| A5 | `core/influence.py` | Friedgut junta extraction |
| A6 | `core/submodular.py` | Nemhauser-Wolsey lazy greedy |
| A8 | `core/linear_logic.py` | Danos–Regnier proof-net checker |
| I9 | `core/hopfield.py` | modern continuous Hopfield memory |
| I10 | `core/hdc.py` | hyperdimensional computing / VSA |
| I12 | baked into `vectorized_predictor.py` downstream | mixture-of-depths halting |
| D5 | `core/routing_hash.py` | consistent + rendezvous hashing |
| D12 | `core/sketches.py` | Count-Min, HyperLogLog, reservoir |
| E6 | `core/iss.py` | ISS-gain estimation and small-gain theorem |

## Wave 3 — "Core mechanisms" (Tier 0 + 1)

Structural upgrades that actually change behaviour when wired in.

| ID | Module |
|---|---|
| A1 | `core/epistemic.py` — DEL / common-knowledge |
| A2 | `core/regularity.py` — Frieze–Kannan block partition |
| A3 | `core/tree_decomp.py` — min-degree treewidth |
| B6 | `core/mfg.py` — mean-field games (HJB + FP Picard) |
| C1 | `core/index_coding.py` — Birk-Kol confusion-graph coloring |
| C3 | `core/rates.py` — Berger–Tung inner bound |
| D2 | `core/itc.py` — interval tree clocks |
| D4 | `core/persistent.py` — HAMT persistent map |
| D6 | `core/coherence.py` — MESI cache-coherence |
| D10 | `core/gossip_tree.py` — Plumtree + HyParView |
| D11 | `core/learned_index.py` — piecewise-linear learned index |
| E3 | `core/port_ham.py` — port-Hamiltonian systems |
| E4 | `core/ci_kalman.py` — consensus+innovations Kalman |
| F2 | `core/bnp.py` — truncated DP-mixture VI |
| F3 | `core/coreset.py` — Frank-Wolfe Bayesian coreset |
| F7 | `core/eig.py` — expected information gain |
| G1 | `core/persuasion.py` — Bayesian persuasion (concavification) |
| G2 | `core/info_design.py` — Doval–Ely factorisation |
| G5 | `core/coalitions.py` — merge-and-split coalition search |

## Wave 4 — "Adversarial + crypto" (Tier 2)

Completes the OQ5 stack plus a privacy story.

| ID | Module |
|---|---|
| B1 | `core/qec.py` — 3-qubit rep + surface-code stabilizer layout |
| C4 | `core/coded_compute.py` — RS-coded straggler-tolerant compute |
| C5 | `core/polar.py` — polar code encoder + SC decoder |
| D1 | `core/dagbft.py` — Bullshark-style DAG-BFT simulator |
| D3 | `core/merkle_dag.py` — content-addressed store + inclusion proofs |
| D8 | `core/peer_review.py` — signed hash-chain logs + spot-check |
| D9 | `core/vrf_committee.py` — Ed25519-VRF committee election |
| E5 | `core/cbf.py` — control barrier function (closed-form scalar QP) |
| H1 | `core/secret_sharing.py` — Shamir (k, n) threshold |
| H2 | `core/paillier.py` — additively-homomorphic encryption |
| H4 | `core/dp.py` — Laplace/Gaussian DP + advanced composition |
| H5 | `core/spdz_light.py` — additive shares + Beaver triples |
| — | `core/gf.py` — supporting GF(p) arithmetic (no `galois` dep) |

## Wave 5 — "LLM-native (numpy subset)" (Tier 0 / 3)

Tier-3 torch-dependent items (I3 SAE, I4 merging, I5 LoRA-delta, I6 distillation) are
parked for when torch is added; numpy-doable members of the wave shipped:

| ID | Module |
|---|---|
| A7 | `core/deq_numpy.py` — Anderson-accelerated fixed point |
| A9 | `core/data_migration.py` — Δ, Σ, Π functorial migration |
| I1 | `core/speculative.py` — rejection-sampling speculative decoding |
| I7 | `core/retrieval_store.py` — brute-force vector store (HNSW drop-in) |
| I8 | `core/exec_plan.py` — sandboxed AST-restricted executable plans |
| I11 | `core/spn.py` — sum-product network (exact inference) |
| I13 | `core/deep_sets.py` — permutation-equivariant synthesizer (forward) |

## Integration — composed routers

| Router | Composed from |
|---|---|
| `AdversarialCASRRouter` | D7 + D8 + D9 + E5 + H4 |
| `DynamicCASRRouter` | D2 + D4 + D5 + D10 |
| (`CryptoCASRRouter`) | H1 + H2 + H5 — available via direct imports |

Imported at the package level: `from vision_mvp import AdversarialCASRRouter, DynamicCASRRouter`.

## Not built (by design)

Parked per the plan's "out-of-scope":
- **B10** Generalized hydrodynamics — needs a concrete task driver first.
- **D13** Succinct wavelet/FM-index — inbox bottleneck has not materialised.
- **I2** KV-cache sharing — Ollama has no KV API; left for a transformers-backed branch.
- **I14** Normalizing flows — Stein VGD (F4) covers the same need without torch.
- **H3** zk-SNARKs — shell-out integration deferred.
- **I3 / I4 / I5 / I6** — torch-dependent LLM primitives; re-enable by installing the `[dl]` extra.

## Files added

- `vision_mvp/core/` — 44 new modules
- `vision_mvp/core/diagnostics/` — 4 new diagnostic modules
- `vision_mvp/tests/` — 6 new test files (Waves 1–5 + composed routers)

Existing modules left in place and unmodified (backward compatibility):
`api.py`, `casr_router.py`, `bus.py`, `manifold.py`, `workspace.py`,
`stigmergy.py`, `learned_manifold.py`, `predictor.py`, `neural_predictor.py`,
`vectorized_predictor.py`, and all of `llm_*`.

## How to run

```bash
python3 -m unittest discover -s vision_mvp/tests          # full suite (455 tests)
python3 -m vision_mvp demo --n 200                        # existing quick-start
```

Optional extras:

```bash
pip install -e .[scientific]   # scipy + networkx
pip install -e .[crypto]       # cryptography
pip install -e .[dl]           # torch + peft
pip install -e .[heavy]        # hnswlib + transformers + RestrictedPython
```
