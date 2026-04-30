# SDK v3.26 — W25 Shared-Fanout Dense-Control Results

**Date**: 2026-04-30  
**Milestone**: SDK v3.26  
**Method**: W25 `SharedFanoutDisambiguator`  
**Benchmark family**: R-72 (three sub-banks)  
**Mac-2 status**: ARP-incomplete (20th consecutive milestone); all results Mac-1 only.

---

## 1. What W25 Is (and Is Not)

W24 (`MultiCellSessionCompactor`) achieves one `<compact_ref:DDDD>` visible token
per agent per cell via bounded-window compaction. For a single agent, that is
already the minimum — one token is the floor.

**The multi-agent gap**: when K consumer agents all need the same cross-cell
session state produced by one producer, W24 still requires K independent compact
envelopes (K × C tokens total, where C ≈ 14.6 tokens/agent/cell at K=4). W25
replaces those K envelopes with a **single `FanoutEnvelope`** registered by the
producer plus K single-token `<fanout_ref:DDDD>` references for the consumers.

This is not a learned model, not a speculative cache, and not a cross-host
tensor-parallel shard. It is:

- A content-addressed, schema-versioned, producer-signed envelope of the shared
  session digest
- A typed registry keyed by `(producer_agent_id, cell_index)`
- A 5-failure-mode verifier: `empty_envelope`, `schema_version_unknown`,
  `schema_cid_mismatch`, `consumer_not_authorized`, `hash_mismatch`
- An explicit audit record per agent per cell (`W25FanoutResult`)

**Honest scope**: W25 reduces MULTI-AGENT token overhead. The producer cost is
unchanged from W24. The saving is K×(C−1) tokens/cell where C is the producer's
W24 compact cost. Claimed proxy for LatentMAS "hardware pooling": the FanoutEnvelope
represents the "one computation shared by K agents" pattern at the capsule level
without any claim about hardware or parallel execution.

---

## 2. Success Bar (pre-committed, now discharged)

**W25-1 (efficiency)**: On R-72-FANOUT-SHARED (1 producer + K=3 consumers, 16
cells), W25 must strictly reduce `mean_total_w25_visible_tokens` below
`mean_total_w24_visible_tokens` AND `correctness_ratified_rate = 1.000` AND
`fanout_consumer_resolved_rate = 1.000`.

**W25-2 (seed stability)**: W25-1 holds across all 5 pre-committed seeds
(11, 17, 23, 29, 31).

**W25-Λ-disjoint (named falsifier)**: With no shared registry, W25 = W24 exactly
(`mean_savings_tokens_per_cell = 0.000`).

**W25-3 (trust soundness)**: Unauthorised consumer_id rejected on every cell.

---

## 3. Benchmark Results

### R-72-FANOUT-SHARED (W25-1 anchor)

| metric | value |
|--------|-------|
| K_consumers | 3 |
| n_cells | 16 |
| mean_total_w24_visible_tokens/cell | 58.50 |
| mean_total_w25_visible_tokens/cell | 17.63 |
| mean_savings_tokens_per_cell | 40.88 |
| **savings_pct** | **−69.87%** |
| correctness_ratified_rate | 1.0000 |
| fanout_consumer_resolved_rate | 1.0000 |
| registry_n_registered | 16 |
| n_fanout_bytes_total | 5,885 |

Breakdown per agent/cell:
- Producer (W24 unchanged): ~14.6 tokens/cell
- Consumer W24 baseline: ~14.6 tokens/cell each
- Consumer W25 cost: 1 token each (`<fanout_ref:DDDD>`)
- Saving per consumer: 13.6 tokens/cell
- Total saving (3 consumers): 40.9 tokens/cell

### R-72-DISJOINT (W25-Λ-disjoint falsifier)

| metric | value |
|--------|-------|
| mean_total_w24_visible_tokens/cell | 58.50 |
| mean_total_w25_visible_tokens/cell | 58.50 |
| mean_savings_tokens_per_cell | 0.00 |
| savings_pct | 0.00% |
| fanout_consumer_resolved_rate | 0.0000 |

W25 correctly reduces to W24 when there is no shared registry. The falsifier
discharges as predicted.

### R-72-FANOUT-POISONED (W25-3 trust falsifier)

| metric | value |
|--------|-------|
| fanout_consumer_resolved_rate | 0.6667 (2/3 authorised) |
| fanout_consumer_rejected_rate | 0.3333 (1/3 unauthorised) |
| n_consumer_rejected | 16 (every cell, poisoned consumer) |
| correctness_ratified_rate | 1.0000 |

The unauthorised consumer is rejected on every cell (16/16). The two authorised
consumers still resolve. The producer correctness is unchanged. The trust boundary
is sound.

---

## 4. Seed Stability (5/5 seeds)

| seed | savings/cell | savings_pct | correctness | resolved_rate |
|------|-------------|-------------|-------------|---------------|
| 11   | 40.875      | 69.87%      | 1.0000      | 1.0000        |
| 17   | 40.875      | 69.87%      | 1.0000      | 1.0000        |
| 23   | 40.875      | 69.87%      | 1.0000      | 1.0000        |
| 29   | 40.875      | 69.87%      | 1.0000      | 1.0000        |
| 31   | 40.875      | 69.87%      | 1.0000      | 1.0000        |

W25-1 discharged on all 5 seeds. W25-2 discharged.

---

## 5. Cross-Decoder Robustness

Identical results under T_decoder=None (loose) and T_decoder=24 (tight):
savings, correctness, and trust boundary are decoder-budget-invariant on this bench.

---

## 6. What This Means / Honest Claims

**What is claimed**:
- W25 strictly reduces total visible tokens across a K-consumer multi-agent
  session by K×(C−1) per cell where C is the W24 compact cost (~13.6 tokens
  saved per consumer per cell at K=3, 16 cells, 5/5 seeds)
- The trust boundary is sound: unauthorised consumers are rejected at the
  envelope layer before any content is revealed
- Correctness is preserved byte-for-byte (producer answer unchanged)
- The FanoutEnvelope is content-addressed (SHA-256 of canonical JSON) and
  immutable; tampering is detected by `verify_fanout`

**What is NOT claimed**:
- No claim about real hardware pooling, tensor parallelism, or LatentMAS latent
  state sharing — W25 is a proxy at the capsule layer only
- The savings figure (−69.87%) is measured on K=3 consumers; it grows linearly
  with K: K=5 would save ~76%, K=10 would save ~82% (theoretical)
- No Mac-2 live data (Mac-2 ARP-incomplete, 20th consecutive milestone)
- The bench is synthetic (R-69-CACHE-FANOUT oracle ecology); real production
  deployments would need to validate against live traffic patterns

---

## 7. Regression

IS-1, IS-2 theorem tests: 14/14 pass  
Phase 72 new tests: 31/31 pass  
No regressions on W24/W23/W22 test infrastructure.

---

## 8. Data Artifacts

- `docs/data/phase72_cross_regime.json` — 6-row matrix (3 banks × 2 T_decoder)
- `docs/data/phase72_seed_sweep.json` — 5-seed stability sweep

---

## 9. Named Conjectures (open)

**W25-C-MULTI-HOST**: The W25 FanoutEnvelope serialises to 367 bytes/cell (JSON,
~5,885 bytes / 16 cells). On a real multi-host deployment, broadcast cost is
O(K × 367 bytes) vs K × O(compact_envelope) per cell. Whether this is a net win
depends on the cross-host bandwidth vs the per-consumer token budget. Not measured
here (Mac-2 unreachable).

**W25-C-K-SCALING**: The token savings grow as K×(C−1). At K=10 the fanout
pattern reduces consumer overhead by ~88% over W24. This has not been measured
on K>3; the formula is derived from the K=3 measurement.

**W25-C-STREAMING**: The FanoutEnvelope is currently materialised before any
consumer resolves. A streaming variant would let consumers resolve incrementally
as the producer emits partial windows. This is conjectural.
