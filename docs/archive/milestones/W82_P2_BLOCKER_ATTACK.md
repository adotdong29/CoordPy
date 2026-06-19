# W82 — P2 Blocker Attack

> Closes the six P2 child blocker meta-issues from issue #4 in
> the order recommended by the meta issue's Phase 4 → Phase 5
> sequencing: **#10 → #15 → #13 → #18 → #11 → #16**. Last
> touched: 2026-05-18.

## Why this milestone exists

W80 closed the five P0 blocker meta-issues (#5, #6, #8, #12,
#17 — runtime instrumentation contract, frontier-quality local
runtime, second backend + parity matrix, live local-model
evaluation, living capability matrix).

W81 closed the five P1 blocker meta-issues (#7, #9, #14, #19,
#20 — deployable substrate gateway, sequence-conditioned
learned consolidation, learned multi-runtime economics,
differentiable memory substrate, adversarial consensus & repair).

W82 closes the **six P2 blocker meta-issues** (#10, #11, #13,
#15, #16, #18) — the "structural scale / portability /
hardening" phase from the meta issue's recommended sequencing.

## Sequencing rationale

Per the meta issue (#4):

> Phase 4 — Structural scale and harder proofs of
> failure/success:
> 11. #10 far-horizon blackout benchmarks
> 12. #15 simultaneous compound-failure suite
> 13. #13 cross-runtime / cross-tokenizer portability
> 14. #18 integrity / rollback / branch-merge verification
> 15. #11 event-sourced global memory graph and query layer
>
> Phase 5 — Distributed substrate:
> 16. #16 distributed multi-host substrate coordination and
>     migration

W82 follows this order exactly. Each issue depends on
infrastructure delivered by the prior issue:

* **#10** depends on the W79 `long_horizon_reconstruction_
  substrate_v2` (P0) and `bounded_window_baseline_v2` (P0).
* **#15** depends on #10's scenario builder + the W81
  `adversarial_consensus_repair_v1` (P1).
* **#13** depends on the W80 `runtime_instrumentation_v1` (P0)
  and `capability_matrix_v1` (P0) concepts.
* **#18** is foundational (only depends on stdlib hashlib +
  hmac).
* **#11** depends on #18's content-addressing primitives
  (Merkle).
* **#16** depends on #18 (integrity verdicts), #13 (portability
  projector), and #11 (event graph).

## What's new

Six new modules under `coordpy/`, six new test files under
`tests/`, and three updated docs (`RESEARCH_STATUS.md`,
`THEOREM_REGISTRY.md`, `HOW_NOT_TO_OVERSTATE.md`).

| P2  | Module added                                          | Tests |
|-----|-------------------------------------------------------|-------|
| #10 | `coordpy.far_horizon_blackout_benchmark_v1`           | 16    |
| #15 | `coordpy.simultaneous_compound_failure_benchmark_v1`  | 14    |
| #13 | `coordpy.cross_runtime_state_portability_v1`          | 16    |
| #18 | `coordpy.cryptographic_state_integrity_v1`            | 23    |
| #11 | `coordpy.event_sourced_memory_graph_v1`               | 21    |
| #16 | `coordpy.distributed_substrate_coordination_v1`       | 21    |

**111 new tests, all passing on Python 3.11 with NumPy.** All
P0/P1 baselines stay green and untouched. None of the six
modules are added to the `coordpy` top-level namespace —
they're explicit-import only.

## Summary by issue

### #10 — Far-horizon blackout benchmarks
[`coordpy/far_horizon_blackout_benchmark_v1.py`]

A horizon-ladder benchmark family covering horizons of
{1000, 4000, 16_000, 64_000} plus an additional stress horizon
of 100_000+. Generates synthetic scenarios with interleaved
irrelevant traffic, multiple restart cycles, and branch /
rejoin offsets — designed to stretch far past today's default
comfort zone (~256–4096 turns).

Strategy set: four canonical strategies — `transcript_only`,
`bounded_window_k{4, 32, 64, 128}`, `rolling_summary`,
`lhr_substrate_v2`.

Reported per-cell metrics: `task_success`,
`reconstruction_fidelity`, `visible_tokens_used`,
`replay_flops`, `recompute_flops`, `carrier_walks`.

Result: the W79 `lhr_substrate_v2` strictly dominates every
baseline at every horizon on the default ladder; substrate-
side failures (carrier truncated below horizon) are reported
honestly.

### #15 — Simultaneous compound-failure benchmark
[`coordpy/simultaneous_compound_failure_benchmark_v1.py`]

Stacks five failure factors — `CONTRADICTION`, `CORRUPTION`,
`REPLACEMENT`, `RESTART`, `BLACKOUT` — into a 2^5 = 32-mask
sweep. The W82 `compound_repair` strategy composes:

1. The W79 substrate (defeats blackout + restart).
2. The W81 adversarial consensus repair line (defeats
   corruption + contradiction).
3. A trimmed-mean replacement-aware filter (defeats
   replacement).

Plus a primary-failure-factor attribution that is per-strategy-
aware: bounded-window's primary failure under all-5-active is
correctly attributed to `blackout`, not `corruption`.

Result: on the load-bearing all-5-active mask, W82
compound-repair succeeds at 100% across seeds while every
prior baseline (naive majority, bounded-window k128, substrate
v2 only) succeeds at 0%.

### #13 — Cross-runtime / cross-tokenizer state portability
[`coordpy/cross_runtime_state_portability_v1.py`]

Three explicit fidelity tiers: `EXACT_REPLAY` (same
signature, byte-identical), `APPROXIMATE_SEMANTIC` (different
signature, anchor-preserving), `NON_PORTABLE` (backend-specific
axes; explicitly dropped on egress).

Core objects: `RuntimeSignatureV1`, `PortableStateCarrierV1`,
`PortabilityProjectorV1`, `PortabilityFidelityReportV1`.

Result: same-signature round-trip is bit-identical via the
carrier's `exact_replay_payload`; cross-signature transfer
(hidden_dim 8 → 12, vocab 32 → 24) preserves anchor cosine ≥
0.95 and anchor-classifier preservation ≥ 90%. The bench
explicitly demonstrates raw-hidden-coord-0 is NOT portable
across different-hidden-dim signatures (honest-scope test).

### #18 — Cryptographic state integrity / rollback / branch-merge
[`coordpy/cryptographic_state_integrity_v1.py`]

Unified integrity subsystem:

* `StateSnapshotV1` — content-addressed, optionally
  HMAC-SHA256 signed.
* `MerkleHashTreeV1` — O(log n) inclusion proofs.
* `RollbackAnchorV1` — tagged anchors restoring under
  integrity verification.
* `BranchMergeWitnessV1` — proves merge safety.
* `IntegrityVerdict` enum — `OK`, `CORRUPT`,
  `PROVENANCE_VIOLATION`, `BAD_SIGNATURE`, `UNSIGNED`.

End-to-end bench exercises: clean chain verifies; single-byte
silent corruption detected; rollback to pre-corruption anchor
preserves integrity; clean branch-merge verified; unsafe
merge (parent mismatch) flagged as `PROVENANCE_VIOLATION`.

### #11 — Event-sourced global memory graph
[`coordpy/event_sourced_memory_graph_v1.py`]

Append-only `EventNodeV1` / `EventGraphV1` with first-class
branch + merge (events with 2+ parents). Four declarative
query kinds: `BY_EVENT_ID`, `BY_KIND`, `BY_BRANCH`,
`BY_ANCESTOR_PATH`. Each query produces a `QueryPlanV1`
with explicit step lists, a `QueryAnswerV1`, and a
`ProvenanceCertificateV1` listing every contributing event
CID.

`DerivedSummaryViewV1` is recomputed *from* the primary store
— summarisation is a derived view, not the primary carrier.

Load-bearing 100-event by-ancestor-path bench: graph wins
100% while `bounded_handoff_k32` and `trajectory_slice_k16`
baselines win 0%. The graph strictly beats every bounded
baseline on a task that requires walking through deep
ancestry.

### #16 — Distributed multi-host substrate coordination
[`coordpy/distributed_substrate_coordination_v1.py`]

Simulated multi-host substrate (in-process for V1; transport
is a function call). Three hosts each with their own event
graph and runtime signature.

* `MigrationEnvelopeV1` — content-addressed Merkle-rooted
  package of events being shipped.
* `PartitionEventV1` — simulated network partition.
* `SyncDecisionV1` — explicit consistency verdict (`EXACT`,
  `EVENTUAL`, `APPROXIMATE`, `BEST_EFFORT`).
* `heal_partition_and_sync_v1` — eventual-consistency
  convergence with pre-/post-heal Merkle roots recorded.

Bench: partition detected; heal+sync converges (every host's
Merkle root identical); cross-runtime migration leg
(hidden_dim 8 → 12) verified `OK`.

## Honest scope

All the W82 caveats listed in
`docs/HOW_NOT_TO_OVERSTATE.md`'s W82 section are tracked as
`W82-L-*` rows in `docs/THEOREM_REGISTRY.md`. The key ones:

* All six modules are explicit-import only.
* The far-horizon bench is synthetic — fidelity is the metric,
  not live LLM correctness.
* The compound-failure bench caps each factor's strength;
  under unbounded factor budgets every strategy is provably
  broken.
* Portability is anchor-preserving, not bit-identical across
  signatures (the EXACT_REPLAY tier is reserved for same
  signature only).
* The integrity layer's HMAC is in-memory keyed — it catches
  silent tampering but is not a PKI scheme.
* The event-graph is in-memory only; on-disk persistence is
  out of V1 scope.
* The distributed bench is in-process simulation; real TCP /
  RPC transport is W83+ work.

## Test plan

- [x] `tests/test_w82_far_horizon_blackout_benchmark.py` — 16
  tests (horizon ladder, content addressing, strategy set,
  metric completeness, substrate dominates, failure curves,
  carrier-truncation failure mode, 100k stress horizon).
- [x] `tests/test_w82_simultaneous_compound_failure_benchmark.py`
  — 14 tests (factor masks, 32-mask sweep, scenario
  determinism, strategy set, failure attribution, all-5-
  active win, deterministic CID, W82 fails honestly under
  extreme factor budget).
- [x] `tests/test_w82_cross_runtime_state_portability.py` —
  16 tests (signature content addressing, three tiers,
  same-sig bit-identical, cross-sig cosine, cross-sig
  classification, non-portable drop, honest-scope coord-0,
  determinism).
- [x] `tests/test_w82_cryptographic_state_integrity.py` — 23
  tests (snapshot content addressing, payload re-hash, Merkle
  inclusion log n, HMAC sign+verify, bad HMAC, unsigned
  verdict, all 5 verdicts enumerate, silent corruption,
  provenance violation, rollback round-trip, clean merge,
  unsafe merge, end-to-end bench, integrity-aware fallback
  decision maps each verdict).
- [x] `tests/test_w82_event_sourced_memory_graph.py` — 21
  tests (event content addressing, append-only, orphan
  parent rejected, branch / merge first-class, four query
  kinds, query plans, ancestor path through merge,
  provenance certificate, derived view, baseline failure,
  graph dominates baselines, carrier-fallback recovers
  evicted events).
- [x] `tests/test_w82_distributed_substrate_coordination.py`
  — 21 tests (consistency verdicts enumerate, host content
  addressing, envelope content addressing, envelope verify,
  apply idempotent, corrupt envelope detected, partition
  content addressing, heal+sync converges, cross-runtime
  envelope, replicate skip exclude, deterministic bench,
  budget-aware migration policy accepts / rejects on bytes /
  events / freshness).

Total: **111 tests, all passing on Python 3.11 with NumPy.**
W79 baseline (28/28) + P0/P1 baselines unchanged.

## Beyond definition-of-done — candidate-direction hardenings

The W82 modules also close several "candidate direction" gaps
called out in the meta issue but not strictly in any
issue's Definition of Done:

* **#11 carrier-fallback query**
  (`execute_query_with_carrier_fallback_v1`). Bridges the
  live event graph and a long-horizon-carrier mapping
  (`event_id → event_cid`): a `BY_EVENT_ID` query that
  misses the graph can be answered by the carrier with a
  `carrier_fallback` detail tag and a clean content-
  addressed provenance trail.
* **#16 budget-aware migration policy**
  (`MigrationBudgetPolicyV1`, `MigrationBudgetDecisionV1`).
  Three explicit budgets — max bytes per migration, max
  events per migration, minimum freshness age — with
  content-addressed accept/reject decisions and explicit
  rejection reasons. Migrations that exceed budget are
  auditably refused, not silently truncated.
* **#18 integrity-aware abstain/fallback decisions**
  (`integrity_aware_fallback_decision_v1`). Maps each of the
  five integrity verdicts to a recommended action
  (`COMMIT`, `ROLLBACK`, `ABSTAIN`, `ESCALATE`). Composes
  with the W81 adversarial-consensus `abstain`/`escalate`
  vocabulary.

## Files changed

Six new modules under `coordpy/`, six new test files under
`tests/`, one new doc (`docs/W82_P2_BLOCKER_ATTACK.md`),
and three doc amendments (`RESEARCH_STATUS.md`,
`THEOREM_REGISTRY.md`, `HOW_NOT_TO_OVERSTATE.md`). Nothing in
the pre-existing modules is modified — the diff is purely
additive against the W81 baseline.
