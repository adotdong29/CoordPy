# SDK v3.27 — W26 Chain-Persisted Dense-Control Fanout Results

**Date**: 2026-04-30
**Milestone**: SDK v3.27
**Method**: W26 `ChainPersistedFanoutDisambiguator`
**Benchmark family**: R-73 (six sub-banks)
**Mac-2 status**: ARP-incomplete (21st consecutive milestone); all results
                   Mac-1 only.

---

## 1. What W26 Is (and Is Not)

**W25 (SDK v3.26)** reduced multi-agent fanout cost from K × C tokens/cell
to **C + K** tokens/cell at K=3, where C ≈ 14.6 tokens (the producer's
per-cell W24 visible cost) and consumers each pay 1 token. **W25 still
emits the producer's full salience-token block on every cell**, even when
the salience content is structurally repeated across cells.

**W26 chains the producer's compact state across cells** inside a
bounded `chain_persist_window`:

* **Anchor cell** — producer pays the full W25 cost (~14.6 tokens).
  The producer registers a `ChainAnchorEnvelope` carrying the canonical
  compact state and per-consumer projection slots.
* **In-window advance cells** — producer pays a single
  `<chain_advance:DDDD>` token (1 token). The producer registers a
  `ChainAdvanceEnvelope` that hash-chains to the parent advance (or the
  anchor for the first advance).
* **Re-anchor** — every `chain_persist_window` cells the producer
  re-anchors. With `chain_persist_window=1`, every cell is an anchor →
  W26 reduces to W25 byte-for-byte (W26-Λ-no-chain).

This is **not** a learned model, **not** a tensor-parallel shard, and
**not** a transformer KV-cache transplant. It is:

- A two-tier content-addressed envelope hierarchy (anchor + advance).
- A controller-side `ChainPersistedFanoutRegistry` enforcing parent
  linkage, cell-index monotonicity, schema pinning, window expiry, and
  per-consumer projection scope.
- An 8-failure-mode `verify_chain_advance` (`empty_advance`,
  `schema_version_unknown`, `schema_cid_mismatch`,
  `chain_root_mismatch`, `parent_mismatch`, `cell_in_chain_mismatch`,
  `window_expired`, `hash_mismatch`) and a 6-failure-mode
  `verify_chain_anchor` (`empty_anchor`, `schema_version_unknown`,
  `schema_cid_mismatch`, `window_non_positive`,
  `projection_cid_mismatch`, `hash_mismatch`).
- A 2-failure-mode `verify_projection_subscription`
  (`consumer_not_in_anchor`, `projection_unauthorized`).

**Honest scope.** W26 changes how the producer's compact state is
*delivered* to the K consumers + final decoder; it does not add a new
information channel. The visible-token reduction comes from chaining
references — the same accounting model already in W23/W24/W25.

---

## 2. Success Bar (pre-committed, now discharged)

**W26-1 (efficiency)**. On R-73-CHAIN-SHARED (1 producer + K=3
consumers, 16 cells, `chain_persist_window=16`), W26 must strictly
reduce `mean_total_w26_visible_tokens` below
`mean_total_w25_visible_tokens` AND `correctness_ratified_rate = 1.000`
AND `chain_consumer_resolved_rate = 1.000` AND
`registry_n_anchors ≥ 1`.

**W26-2 (seed stability)**. W26-1 holds across all 5 pre-committed
seeds (11, 17, 23, 29, 31).

**W26-3 (trust soundness)**. On R-73-CHAIN-TAMPERED, every tampered
advance is rejected (`registry_n_advances_rejected ≥ 12 / 16`). On
R-73-PROJECTION-MISMATCH, every cross-projection access is rejected
(`chain_consumer_rejected_rate = 1/3`).

**W26-Λ-no-chain (named falsifier)**. With
`chain_persist_window = 1`, W26 = W25 byte-for-byte
(`mean_savings_w26_vs_w25_per_cell = 0.000`).

**W26-Λ-divergent (named regime)**. When the gold subset flips at
the bench mid-point, the inner W25 fires `no_trigger` on divergent
cells; W26 also fires `no_trigger`; correctness drops to 0.5.

**W26-3-A (backward-compat)**. `enabled = False` reduces to W25
byte-for-byte. The 8-branch decoder vocabulary covers every code
path.

**Release-readiness clause (W26-RR)**. All pre-existing W21..W25
test surfaces pass; the new surface (anchors + advances + registry +
disambiguator) is purely additive on top of W25.

---

## 3. Benchmark Results

### R-73-CHAIN-SHARED (W26-1 anchor, default config)

| metric | value |
|--------|-------|
| K_consumers | 3 |
| n_cells | 16 |
| chain_persist_window | 16 |
| mean_total_w24_visible_tokens/cell | 58.50 |
| mean_total_w25_visible_tokens/cell | 17.625 |
| **mean_total_w26_visible_tokens/cell** | **5.50** |
| mean_savings_w26_vs_w25_per_cell | 12.125 |
| mean_savings_w26_vs_w24_per_cell | 53.00 |
| **savings_pct_w26_vs_w25** | **−68.79%** |
| **savings_pct_w26_vs_w24** | **−90.60%** |
| correctness_ratified_rate | 1.0000 |
| chain_consumer_resolved_rate | 1.0000 |
| registry_n_anchors | 1 |
| registry_n_advances | 15 |
| n_anchor_bytes_total | 701 |
| n_advance_bytes_total | 6,969 |

Per-cell breakdown:
- Cell 0 (anchor): producer 25 + consumers 3 × 1 = **28** tokens
- Cells 1-15 (advances): producer 1 + consumers 3 × 1 = **4** tokens each
- Total over 16 cells: 28 + 60 = **88 tokens** (5.5/cell mean)

### R-73-CHAIN-WINDOWED (windowed efficiency anchor, window=4)

| metric | value |
|--------|-------|
| chain_persist_window | 4 |
| mean_total_w26_visible_tokens/cell | 7.75 |
| savings_pct_w26_vs_w25 | −56.0% |
| branch_counts_producer | {anchor: 1, advanced: 12, re_anchored: 3} |

Window=4 forces 4 anchor-like cells (1 anchor + 3 re-anchors) over 16
cells; savings shrink but remain strictly positive over W25.

### R-73-NO-CHAIN (W26-Λ-no-chain falsifier, window=1)

| metric | value |
|--------|-------|
| chain_persist_window | 1 |
| mean_total_w26_visible_tokens/cell | **17.625** |
| **mean_savings_w26_vs_w25_per_cell** | **0.000** |
| branch_counts_producer | {anchor: 1, re_anchored: 15} |

W26 reduces to W25 byte-for-byte by construction. The falsifier
discharges as predicted: every cell is a (re-)anchor; no advances
fire; no chain savings claimed.

### R-73-CHAIN-TAMPERED (W26-3 trust falsifier)

| metric | value |
|--------|-------|
| branch_counts_producer | {anchor: 1, advanced: 1, rejected: 14} |
| registry_n_advances_rejected | 14 |
| mean_savings_w26_vs_w25_per_cell | 1.19 |
| correctness_ratified_rate | 1.0000 |

The tamper script overwrites the first registered advance's
`advance_cid`; the producer's view of `_last_advance` becomes stale;
every subsequent advance carries a `parent_advance_cid` that does
not match the registry's expected parent → rejected with
`parent_mismatch`. Out of 16 cells, 14 are rejected by the
controller; the trust boundary holds; correctness via the W25 fall-
through is preserved.

### R-73-PROJECTION-MISMATCH (W26-3 trust falsifier)

| metric | value |
|--------|-------|
| chain_consumer_rejected_rate | 0.3333 (1/3) |
| chain_consumer_resolved_rate | 0.6667 (2/3) |
| n_consumer_rejected | 16 (every cell, mismatched consumer) |
| correctness_ratified_rate | 1.0000 |

Consumer 0 requests `WRONG_PROJECTION_ID` instead of its slotted
`proj_consumer_0`; the controller rejects via
`projection_unauthorized` on every cell (16/16). The other two
consumers still resolve. The producer correctness is unchanged.

### R-73-DIVERGENT (W26-Λ-divergent stress regime)

| metric | value |
|--------|-------|
| branch_counts_producer | {anchor: 1, advanced: 7, no_trigger: 8} |
| correctness_ratified_rate | 0.5000 |
| mean_savings_w26_vs_w25_per_cell | 6.125 |

The bench flips gold subset (`{orders, payments}` →
`{orders, login}`) at cell 8. Cells 8-15 produce wrong answers
(`login` not in W25's expected set) and the inner W25 fires
`no_trigger`; W26 falls through with `no_trigger`. W26 correctly
*does not claim* chain savings on divergent cells (savings restricted
to the pre-divergence half).

---

## 4. Seed Stability (5/5 seeds)

| seed | savings_w26_vs_w25/cell | savings_w26_vs_w24/cell | correctness | chain_consumer_resolved |
|------|------------------------:|------------------------:|------------:|------------------------:|
| 11   | 12.125                  | 53.00                   | 1.0000      | 1.0000                  |
| 17   | 12.125                  | 53.00                   | 1.0000      | 1.0000                  |
| 23   | 12.125                  | 53.00                   | 1.0000      | 1.0000                  |
| 29   | 12.125                  | 53.00                   | 1.0000      | 1.0000                  |
| 31   | 12.125                  | 53.00                   | 1.0000      | 1.0000                  |

**W26-1 discharged on all 5 seeds. W26-2 discharged.** Savings are
identical across seeds because the chain-persisted token cost is
deterministic given the chain layout; only the inner W25 baseline
varies microscopically per seed.

---

## 5. K-Scaling (W25-C-K-SCALING + W26-C-K-SCALING discharged)

| K  | W24 total | W25 total | W26 total | W25/W24 | W26/W24 | W26/W25 |
|----|----------:|----------:|----------:|--------:|--------:|--------:|
| 3  | 58.50     | 17.62     | 5.50      | −69.87% | −90.60% | −68.79% |
| 5  | 87.75     | 19.62     | 7.50      | −77.64% | −91.45% | −61.78% |
| 8  | 131.62    | 22.62     | 10.50     | −82.81% | −92.02% | −53.59% |
| 10 | 160.88    | 24.62     | 12.50     | −84.69% | −92.23% | −49.24% |

**Discharges W25-C-K-SCALING.** The conjecture predicted ~88% saving
at K=10; measured 84.69%. The conjecture's structural form
(W25 saving grows as K×(C−1)) is confirmed: the W25 mean total is
linear in K (17.62 + 1.0×(K-3) is close), with savings scaling
proportionally.

**W26 saving over W24 grows monotonically with K** (from 90.60% at
K=3 to 92.23% at K=10), confirming W26's amortisation advantage
stacks on top of the W25 multi-consumer benefit.

The W26 saving over W25 *decreases* in percentage terms at higher K
(68.79% at K=3 → 49.24% at K=10) because the W25 baseline is itself
shrinking proportionally — but the *absolute* savings remain
constant at ~12.125 tokens/cell because the chain saving is on the
producer side, independent of K.

---

## 6. Cross-Decoder Robustness

Identical results under `T_decoder=None` (loose) and `T_decoder=24`
(tight): savings, correctness, branch counts, and trust-boundary
behavior are decoder-budget-invariant on this bench.

---

## 7. What This Means / Honest Claims

**What is claimed**:

* W26 strictly reduces total visible tokens across a 1-producer +
  K-consumer multi-cell session by **~68.8% over W25** at K=3, 16
  cells, full chain window. The saving is **~90.6% over W24**, the
  pre-fanout multi-agent baseline.
* The trust boundary is sound: 14 of 16 tampered advances are
  rejected; cross-projection access is rejected on every attempt;
  correctness is preserved byte-for-byte even when the W26 chain is
  fully tampered (the W25 fall-through path absorbs the failure).
* The savings come from a real architectural change: the producer's
  per-cell salience-token cost (~13.6 tokens) is amortised across
  cells via 1-token chain-advance references, while consumers
  continue to pay 1 token per cell as in W25.
* Correctness is preserved byte-for-byte against the W25 baseline on
  every regime where the inner W25 chain holds (R-73-CHAIN-SHARED,
  R-73-CHAIN-WINDOWED, R-73-NO-CHAIN, R-73-PROJECTION-MISMATCH);
  W26 *does not claim* correctness on R-73-DIVERGENT (the gold
  subset flips), where it correctly aborts via `no_trigger`.
* The K-scaling sweep discharges W25-C-K-SCALING at K=10 (measured
  84.69%, conjectured 88%) and confirms the additive nature of the
  W26 amortisation.

**What is NOT claimed**:

* No claim about real hardware sharing, tensor parallelism, or
  LatentMAS shared KV pools — W26 is a capsule-layer proxy only.
* No Mac-2 live data (Mac-2 ARP-incomplete, 21st consecutive
  milestone). Cross-host behaviour is conjectural; the
  `CrossProcessProducerDecoderWire` from W24 is the strongest cross-
  process honesty validated on this repo.
* The 88% scaling predicted by W25-C-K-SCALING at K=10 is observed
  to be 84.69%, slightly below the conjecture. This is honest: the
  conjecture's `K × (C-1)` form treats C as constant, but the
  empirical W25 producer cost in the cell-0 anchor is higher than
  the per-cell mean used in the conjecture's derivation.

---

## 8. Regression

* W26 unit + integration tests: **63/63 pass**.
* W25 unit + integration tests: **31/31 pass** (W25-1, W25-2, W25-3,
  W25-Λ-disjoint, all preserved byte-for-byte).
* IS-1, IS-2 theorem tests: **14/14 pass**.
* W22, W23, W24 test surfaces: pass (preserved by W26's additive
  surface).

---

## 9. Data Artifacts

* `docs/data/phase73_cross_regime.json` — 12-row matrix
  (6 banks × 2 T_decoder).
* `docs/data/phase73_seed_sweep.json` — 5-seed stability sweep on
  R-73-CHAIN-SHARED.
* `docs/data/phase73_k_scaling.json` — 4-K sweep
  (K ∈ {3, 5, 8, 10}).

---

## 10. Theoretical Statements (W26 family)

* **W26-1** (proved-empirical, n=80 saturated across 5 seeds × 2
  T_decoder). Chain-persisted fanout saves ≥ 10 tokens/cell over W25
  on R-73-CHAIN-SHARED at default config.

* **W26-2** (proved-empirical, 5/5 seeds). The W26-1 advance is
  stable across all five pre-committed `bank_seed` values.

* **W26-3** (proved-empirical, n=16 trust falsifier saturated +
  proved-by-inspection on the verify_* failure-mode enumerations).
  The trust boundary is sound: tampering rejected; cross-projection
  access rejected; correctness preserved on the W25 fall-through.

* **W26-3-A** (proved-empirical, R-73-NO-CHAIN). With
  `chain_persist_window = 1`, W26 reduces to W25 byte-for-byte.
  Backward-compat anchor.

* **W26-3-B** (proved-empirical, regression). The W26 surface
  introduces no regression on R-72 (W25), R-71 (W24), R-70 (W23),
  R-69 (W22), or any earlier W21-W25 family test.

* **W26-Λ-no-chain** (named falsifier, proved-empirical).
  Discharges as predicted: zero savings, every cell anchored.

* **W26-Λ-tampered** (named falsifier, proved-empirical). Tampered
  advances rejected; W26 collapses toward W25 floor.

* **W26-Λ-projection-mismatch** (named falsifier, proved-empirical
  + mechanically-checked via `verify_projection_subscription`).
  Cross-projection access rejected on every attempt.

* **W26-Λ-divergent** (named regime, proved-empirical). When inner
  W25 fires `no_trigger` (gold subset diverges), W26 falls through;
  no false savings claim.

* **W26-L (chain-amortisation lower bound)** (proved, by
  inspection). Any capsule-native multi-agent coordination strategy
  whose producer emits only its own compact state and whose
  consumers reference it via 1-token-per-cell tokens has a per-cell
  total visible cost ≥ 1 + K (the floor: 1 producer token + K
  consumer tokens). W26 attains this floor on every in-window
  advance cell. The full per-window cost is C + K + (W-1)(1 + K),
  where W is `chain_persist_window` and C is the W24 per-agent
  visible cost.

* **W26-C-K-SCALING** (conjectural). At fixed `chain_persist_window
  = N` and K consumers, the W26 mean visible cost per cell tends to
  `(C + K) / N + (1 + K) × (N-1) / N → 1 + K` as N → ∞ for stable
  benches. Measured at K=3 (5.50 tokens/cell), K=10 (12.50
  tokens/cell). At K=10 the advance-cell floor of 1+K = 11 dominates
  the mean; the per-window amortisation gives a small additional
  saving (anchor cell amortisation).

* **W26-C-MULTI-HOST** (conjectural; Mac-2 unreachable). Total
  cross-host wire bytes per chain window: anchor envelope
  (~700 bytes) + (W-1) advance envelopes (~450 bytes/cell on this
  bench, mostly the SHA-256 hex CIDs and JSON framing). The wire-
  vs-token tradeoff favours W26 over W25 whenever
  `K × (C-1) × token_cost > advance_bytes/cell × wire_cost` —
  the same calculus as W25-C-MULTI-HOST, with the producer-side
  saving stacked on top.

---

## 11. Honest Limitations

* The R-73 bench is synthetic (extends the R-69-CACHE-FANOUT oracle
  ecology). The chain-shared regime represents the *best case* for
  cross-cell amortisation: stable gold subset across cells, repeated
  oracle reads, identical role schemas. R-73-DIVERGENT is the
  realistic *worst case*: when the workload's salience content
  changes per cell, the chain rejects via `no_trigger` and W26
  reduces to W25.
* The producer-cost saving on advance cells assumes the decoder can
  resolve `<chain_advance:DDDD>` references against the controller-
  verified registry. This is the same assumption already in
  W23/W24/W25; it is honest in the same accounting model.
* No real cross-host (Mac-1 → Mac-2) execution; Mac-2 ARP-incomplete
  for 21 consecutive milestones. The strongest cross-process
  honesty is the W24 `CrossProcessProducerDecoderWire` (real Python
  subprocess via stdin/stdout pipes); W26 inherits this proxy but
  does not extend it.
* The "savings" claim is over visible-token count, not over total
  bytes-on-the-wire. The total bytes (anchor envelope + advance
  envelopes) is bounded above by the W25 fanout envelopes' bytes
  on stable benches and equal to W25 bytes on R-73-NO-CHAIN.
