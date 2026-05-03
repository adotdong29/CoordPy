# SDK v3.28 — W27 Multi-Chain Salience-Keyed Dense-Control Fanout Results

**Date**: 2026-04-30
**Milestone**: SDK v3.28
**Method**: W27 `MultiChainPersistedFanoutOrchestrator`
              (+ audited `MultiChainPersistedFanoutDisambiguator`)
**Benchmark family**: R-74 (six sub-banks + signature_period sweep)
**Mac-2 status**: ARP-incomplete (22nd consecutive milestone); all
                   results Mac-1 only.

---

## 1. What W27 Is (and Is Not)

**W26 (SDK v3.27)** amortised the producer's per-cell salience-token
cost across cells inside a single chain window via 1-token
chain-advance references.  W26 has two structural limits:

* **W26-Λ-divergent** — when the gold subset / canonical compact
  state changes, the inner W22 chain breaks; W26 falls through to
  W25 (no chain savings).
* **W26-C-DIVERGENCE-RECOVERY** (open conjecture in v3.27): a
  smarter chain-replay mechanism could recover savings on
  divergent cells via a new mechanism.

**W27 (SDK v3.28)** implements the smallest honest version of that
mechanism at the *capsule layer*: a **bounded pool of parallel
chains, keyed by salience signature** (SHA-256 over the canonical
compact state).  The signature is computed deterministically from
the producer's *input handoffs* — independent of the inner W22..W25
state — so the pool can route each cell *before* the inner
disambiguators are invoked.

W27 ships **two complementary surfaces**:

* The **`MultiChainPersistedFanoutOrchestrator`** is the load-bearing
  W27 implementation.  It wraps a team-wide
  :class:`SharedMultiChainPool` that maintains one independent
  W26 disambiguator per (signature, agent) and routes each cell
  via :func:`compute_input_signature_cid`.
* The **`MultiChainPersistedFanoutDisambiguator`** is the audited
  pivot wrapper.  It adds two content-addressed envelopes
  (:class:`SalienceSignatureEnvelope`, :class:`ChainPivotEnvelope`)
  on top of *one* W26 stack and ships
  :func:`verify_salience_signature` (4 failure modes) plus
  :func:`verify_chain_pivot` (8 failure modes) for trust-boundary
  auditing.

This is **not** a learned model, **not** a transformer KV-cache
transplant, and **not** a multi-host execution.  It is:

- A bounded-pool routing layer over a content-addressed signature.
- Per-signature-isolation of W26 chains (each chain keeps its own
  registries, projection slots, and verifier state).
- Two enumerated-failure-mode verification functions on top of W26's
  six (W26 anchor + advance + projection-subscription).

**Honest scope.** W27 changes how cells are *routed* to chains; it
does not add a new information channel.  Token savings come from
isolating divergent cells into their own chains so each chain
amortises cleanly.  Correctness gains come from enabling
**per-signature scoping** (e.g., per-signature oracle registration)
that a W26 single-stack cannot deliver.

---

## 2. Success Bar (pre-committed, now discharged)

The W27 success bar requires a regime where W26's single-stack
fails AND W27's per-signature isolation rescues both correctness
and tokens.

**W27-1 (efficiency + correctness anchor)**.  On
**R-74-XORACLE-RECOVER** (1 producer + K=3 consumers, 16 cells, 2
distinct gold-subset signatures, signature_period=4, partial
ServiceGraphOracle scoped to GOLD_A on the W26 baseline):

* W26 single-stack: ``correctness_ratified_rate = 0.500``
  (8/16 cells; 8 phase-B cells abstain because the partial oracle
  doesn't know GOLD_B).
* W27 orchestrator with per-signature oracle registration:
  ``correctness_ratified_rate = 1.000`` AND
  ``mean_total_w27_visible_tokens`` strictly less than
  ``mean_total_w26_visible_tokens`` AND ``pool_size ≤ max_active_chains``.

**W27-2 (seed stability)**.  W27-1 holds across all 5 pre-committed
seeds (11, 17, 23, 29, 31).

**W27-3 (trust soundness)**.  On **R-74-PIVOT-TAMPERED** (the
audited disambig wrapper's pivot is corrupted), every tampered
pivot is rejected by ``verify_chain_pivot``; the orchestrator path
is unaffected (correctness preserved byte-for-byte).  On
**R-74-SIGNATURE-DRIFT** (signature_cid replaced with a stale
value), the per-signature lookup falls through cleanly.

**W27-Λ-single-signature (named falsifier)**.  On **R-74-CHAIN-SHARED**
(every cell produces the same canonical state), W27 reduces to W26
byte-for-byte: ``mean_savings_w27_vs_w26_per_cell = 0.000``.

**W27-Λ-pool-exhausted (named falsifier)**.  On **R-74-POOL-EXHAUSTED**
(``max_active_chains = 2`` but 4 distinct signatures in the bench),
the controller rejects new anchors beyond the bound and W27 falls
through to W26.

**W27-3-A (backward-compat)**.  ``enabled = False`` reduces to W26
byte-for-byte.

**Release-readiness clause (W27-RR)**.  All pre-existing W21..W26
test surfaces pass; the new surface (orchestrator + pool +
audited disambig wrapper) is purely additive on top of W26.

---

## 3. Benchmark Results

### R-74-XORACLE-RECOVER (W27-1 anchor, default config)

| metric | W26 baseline | W27 orchestrator |
|--------|-------------:|-----------------:|
| K_consumers | 3 | 3 |
| n_cells | 16 | 16 |
| signature_period | 4 | 4 |
| pool_size_final | n/a | **2** |
| mean_total_visible_tokens/cell | 29.50 | **7.00** |
| **savings_pct_w27_vs_w26** | — | **−76.27%** |
| **savings_pct_w27_vs_w25** | — | −80.35% |
| **savings_pct_w27_vs_w24** | — | −88.03% |
| **correctness_ratified_rate** | 0.500 | **1.000** |
| consumer_pivoted_rate | n/a | 1.000 |
| branch_counts_producer | `{chain_anchored: 1, chain_advanced: 7, no_trigger: 8}` | `{anchored_new: 2, pivoted: 14}` |

**W27 strictly improves both efficiency AND correctness over W26.**
The W26 baseline's partial ServiceGraphOracle (which only knows
GOLD_A) abstains on phase-B cells, so the W26 chain
fires NO_TRIGGER on cells 4-7 and 12-15.  The W27 orchestrator's
per-signature oracle registration scopes a fresh oracle to each
gold pair, so both phases resolve cleanly via the matching slot's
stack.

### R-74-CHAIN-SHARED (W27-Λ-single-signature falsifier)

| metric | W26 | W27 |
|--------|----:|----:|
| pool_size_final | n/a | 1 |
| mean_total_visible_tokens/cell | 5.50 | **5.50** |
| mean_savings_w27_vs_w26 | — | **0.000** |
| correctness_ratified_rate | 1.000 | 1.000 |
| branch_counts_producer | `{chain_anchored: 1, chain_advanced: 15}` | `{anchored_new: 1, pivoted: 15}` |

W27 reduces to W26 byte-for-byte on a single-signature workload —
the pool contains exactly one chain and every cell is a pivot to
that chain.  **W27-Λ-single-signature falsifier discharged.**

### R-74-DIVERGENT-RECOVER (within-graph divergence; isolation cost)

| metric | W26 | W27 |
|--------|----:|----:|
| pool_size_final | n/a | 2 |
| mean_total_visible_tokens/cell | 5.50 | **7.00** |
| savings_pct_w27_vs_w26 | — | **−27.27%** |
| correctness_ratified_rate | 1.000 | 1.000 |
| branch_counts_producer | `{chain_anchored: 1, chain_advanced: 15}` | `{anchored_new: 2, pivoted: 14}` |

When both gold subsets are within the default oracle's scope, the
W26 single-stack handles divergence cleanly and W27 pays a small
isolation cost (one extra anchor — ~24 tokens spread over 16 cells
≈ 1.5 tokens/cell).  This is the honest W27 cost-of-isolation when
W26 doesn't fail.

### R-74-POOL-EXHAUSTED (W27-Λ-pool-exhausted, max=2 vs 4 signatures)

| metric | value |
|--------|-------|
| max_active_chains | 2 |
| signatures observed | 4 |
| pool_size_final | 2 |
| pool_exhausted_rejections | non-zero |
| correctness_ratified_rate | 1.000 (preserved via fallback) |
| mean_total_visible_tokens/cell (W27) | 8.50 |
| branch_counts_producer | mix of `anchored_new`, `pivoted`, `pool_exhausted` |

**Pool capacity bound enforced.**  Cells whose signature exceeds
the pool fall through to a single fallback W26 disambiguator (which
mirrors the original W26 behaviour byte-for-byte).  Correctness
is preserved.

### R-74-PIVOT-TAMPERED (W27-3 trust falsifier)

| metric | value |
|--------|-------|
| pivot_tampering | parent_advance_cid overwritten in registry |
| n_pool_exhausted_rejections_disambig | 0 |
| correctness_ratified_rate (W27 orchestrator) | 1.000 |

The tampering only corrupts the audited disambig wrapper's
registry; the orchestrator path is unaffected.  Tampering on the
disambig wrapper is detected by ``verify_chain_pivot`` and the
controller rejects subsequent pivots.

### R-74-SIGNATURE-DRIFT (W27-3 trust falsifier)

| metric | value |
|--------|-------|
| stale signature_cid | inserted |
| correctness_ratified_rate (W27 orchestrator) | 1.000 |

Stale signatures are not in the pool; the orchestrator
falls through to W26 cleanly.

---

## 4. Seed Stability (5/5 seeds on R-74-XORACLE-RECOVER)

| seed | savings_w27_vs_w26 | savings_w27_vs_w25 | corr_W27 | corr_W26 | pool |
|------|------------------:|------------------:|---------:|---------:|----:|
| 11   | 22.500             | 28.625             | 1.000    | 0.500    | 2 |
| 17   | 22.500             | 28.625             | 1.000    | 0.500    | 2 |
| 23   | 22.500             | 28.625             | 1.000    | 0.500    | 2 |
| 29   | 22.500             | 28.625             | 1.000    | 0.500    | 2 |
| 31   | 22.500             | 28.625             | 1.000    | 0.500    | 2 |

**W27-1 discharged on all 5 seeds.  W27-2 discharged.**  Savings
are deterministic per signature_period because the bench inputs
are byte-stable per phase.

---

## 5. Signature-Period Sweep

| period | W26 | W27 | savings_pct | corr_W26 | corr_W27 | pool |
|-------:|----:|----:|------------:|---------:|---------:|-----:|
|      1 | 5.50| 7.00| −27.27%     | 1.000    | 1.000    | 2 |
|      2 | 5.50| 7.00| −27.27%     | 1.000    | 1.000    | 2 |
|      4 | 5.50| 7.00| −27.27%     | 1.000    | 1.000    | 2 |
|      8 | 5.50| 7.00| −27.27%     | 1.000    | 1.000    | 2 |
|     16 | 5.50| 5.50|   0.00%     | 1.000    | 1.000    | 1 |

On the within-graph divergent regime, W27's isolation cost is
**constant** — one extra anchor per phase regardless of how
finely the bench alternates.  Period=16 has only one signature
(no divergence within window), and W27 = W26 byte-for-byte.  This
discharges the conjectural relationship
``W27_extra_cost = (M - 1) × C_anchor`` empirically at M ∈ {1, 2}.

---

## 6. What This Means / Honest Claims

**What is claimed**:

* On **R-74-XORACLE-RECOVER**, W27 strictly reduces total visible
  tokens by **−76.27%** over W26 AND simultaneously raises
  correctness from 0.500 to 1.000.  This is the first capsule-native
  multi-agent-coordination method to **simultaneously improve
  efficiency and correctness over W26** on a regime where W26's
  single-stack scope actually limits correctness.
* The trust boundary is sound: pivot tampering and signature drift
  are detected and rejected via the new
  ``verify_chain_pivot`` (8 failure modes) and
  ``verify_salience_signature`` (4 failure modes).
* W27 reduces to W26 byte-for-byte on single-signature workloads
  (W27-Λ-single-signature falsifier discharged).
* The pool size is bounded by ``max_active_chains``; new
  signatures beyond the bound deterministically fall through to
  a fallback W26 stack (correctness preserved).
* Discharges **W26-C-DIVERGENCE-RECOVERY** in the *per-signature
  scoping* direction: W27 recovers correctness on divergent
  benches via independent oracles per signature.  It does NOT
  discharge it in the *single-oracle global state recovery*
  direction (W26 single-stack with full oracle scope already
  handles within-graph divergence cleanly).

**What is NOT claimed**:

* No claim that W27 improves over W26 when the W26 single-stack
  doesn't fail.  On R-74-DIVERGENT-RECOVER (within-graph
  divergence) W27 actually **costs more** than W26 by one extra
  anchor per signature (a measured 27.27% extra cost).  W27's
  value is **conditional** on W26 actually failing.
* No claim about real hardware sharing, tensor parallelism, or
  LatentMAS shared KV pools — W27 is a capsule-layer pool only.
* No Mac-2 live data (Mac-2 ARP-incomplete, 22nd consecutive
  milestone).  W27 inherits the W24 ``CrossProcessProducerDecoderWire``
  proxy as the strongest cross-process honesty validated on this repo.
* No cross-LLM live evaluation in this milestone (the live
  cross-model variance reduction work named as a follow-up).

---

## 7. Regression

* **W27 unit + integration tests**: 22/22 pass.
* **W26 phase73 surface**: 63/63 pass (preserved byte-for-byte).
* **W25 phase72 surface**: 31/31 pass.
* **W24 phase71 surface + W23 phase70 surface**: 72/72 pass + 6
  subtests.
* **IS-1 + IS-2 theorem tests**: 14/14 pass.
* **W18..W22 surface** (phase69 + multi-oracle + outside +
  relational + bundle-contradiction): 203/203 pass.
* **Producer + team_coord + attention + capsules** surface:
  103/103 pass.

**Total focused regression**: **508/508 pass** in ≤ 60s combined.

---

## 8. Data Artifacts

* `docs/data/phase74_cross_regime.json` — 12-row matrix
  (6 banks × 2 T_decoder).
* `docs/data/phase74_xoracle_seed_sweep.json` — 5-seed stability
  sweep on R-74-XORACLE-RECOVER.
* `docs/data/phase74_signature_period_sweep.json` — period sweep
  on R-74-DIVERGENT-RECOVER.

---

## 9. Theoretical Statements (W27 family)

* **W27-1** (proved-empirical, n=80 saturated across 5 seeds × 2
  T_decoder).  W27 strictly improves both ``mean_total_visible_tokens``
  AND ``correctness_ratified_rate`` over W26 on R-74-XORACLE-RECOVER
  at default config (-22.5 tokens/cell, +0.500 correctness).

* **W27-2** (proved-empirical, 5/5 seeds).  The W27-1 advance is
  stable across all 5 pre-committed ``bank_seed`` values.

* **W27-3** (proved-empirical, n=16 trust falsifier saturated +
  proved-by-inspection on the verify_* failure-mode enumerations).
  The trust boundary is sound: pivot tampering and signature
  drift are rejected; correctness preserved on the W26 fall-through.

* **W27-3-A** (proved-empirical, R-74-CHAIN-SHARED).  With a
  single signature in the pool, W27 reduces to W26 byte-for-byte.
  Backward-compat anchor.

* **W27-3-B** (proved-empirical, regression).  The W27 surface
  introduces no regression on R-73 (W26), R-72 (W25), R-71 (W24),
  R-70 (W23), R-69 (W22), or any earlier W21..W26 family test.

* **W27-Λ-single-signature** (named falsifier, proved-empirical).
  Discharges as predicted: zero savings, byte-for-byte W26 on a
  single-signature workload.

* **W27-Λ-pool-exhausted** (named falsifier, proved-empirical).
  Pool capacity bound enforced; cells beyond the bound fall
  through to a fallback W26.  Correctness preserved.

* **W27-Λ-pivot-tampered** (named falsifier, proved-empirical +
  mechanically-checked via ``verify_chain_pivot``).  Pivot
  tampering rejected on every attempt.

* **W27-Λ-signature-drift** (named falsifier, proved-empirical).
  Stale signatures fall through cleanly.

* **W27-L (per-signature isolation lower bound)** (proved, by
  inspection).  Any capsule-native multi-agent coordination
  strategy that maintains M independent chains for M distinct
  salience signatures has a per-cell cost lower-bounded by
  ``M × C_anchor / N + (N - M) × (1 + K) / N`` over N cells.
  At M=2, N=16, K=3, C≈14.6: floor ≈ ``(2 × 14.6 + 14 × 4) / 16
  = 5.53``, measured 7.00 (close to floor, with envelope/
  signature overhead bounded).

* **W27-C-MULTI-SIGNATURE-SCALING** (conjectural).  At fixed
  ``max_active_chains = M`` and N cells, the W27 mean visible
  cost per cell tends to ``(1 + K)`` as N → ∞ for stable benches
  with M ≤ ``max_active_chains`` distinct signatures.
  **Falsifier**: a workload where the per-cell anchor cost grows
  with M would break this asymptote.  Measured at M ∈ {1, 2} on
  R-74; the M → ∞ asymptote is unverified.

* **W27-C-CROSS-HOST** (conjectural; Mac-2 unreachable).  Total
  cross-host wire bytes per signature: anchor envelope (~700 bytes)
  + (N-1) × advance bytes (~460 bytes/cell) per chain in the pool.
  At M chains the wire cost is M × (anchor + (N/M - 1) × advance)
  ≈ N × (advance + anchor / N), which is identical to W26 on
  per-cell wire cost.  **Falsifier**: a workload where the
  signature-routing overhead exceeds the per-cell saving would
  break the wire-cost equivalence.  Mac-2 ARP-incomplete for 22
  consecutive milestones; no real two-host measurement exists.

---

## 10. Honest Limitations

* The R-74-XORACLE-RECOVER bench is constructed: the W26 baseline
  uses a *partial* ServiceGraphOracle that only knows GOLD_A.
  In a real deployment with full oracle coverage, the W26
  single-stack would handle within-graph divergence cleanly and
  W27's correctness gain would not appear.  The honest claim is
  that **W27 enables per-signature scoping** that the W26
  single-stack architecturally cannot.
* The R-74 bench inherits the synthetic R-69-CACHE-FANOUT oracle
  ecology.  Real production deployments would have richer oracle
  registries, but the architectural mechanism (per-signature
  isolation) is independent of the bench specifics.
* No real cross-host (Mac-1 → Mac-2) execution; Mac-2
  ARP-incomplete for 22 consecutive milestones.
* The bounded chain pool size ``max_active_chains`` is the
  critical safety knob.  A workload with adversarially many
  signatures could exhaust the pool, and W27 would degrade to
  W26.  This is named as W27-Λ-pool-exhausted and benchmarked
  empirically on the pool_exhausted regime.
* No live cross-model variance reduction in this milestone.
  Live cross-LLM probes are deferred to a follow-up.
* W27 introduces extra cost (~1.5 tokens/cell at K=3) on
  workloads where W26 already handles divergence — this is
  honest and explicitly measured on R-74-DIVERGENT-RECOVER.
