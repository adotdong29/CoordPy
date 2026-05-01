# Changelog

The Changelog now tracks **Wevra SDK** releases. The research
programme's phase-by-phase narrative lives in
`vision_mvp/RESULTS_PHASE*.md` and
`docs/context_zero_master_plan.md`.

## [0.5.4 / 3.31] — 2026-05-01 — SDK v3.31 — calibrated geometry-aware dense control + multi-stride basis history + per-partition calibration prior + cross-host disagreement-routing + ancestor-chain causal binding + simultaneous discharge of W29-C-CRAM-AMPLIFICATION AND W29-C-PARTITION-CALIBRATION

*Strictly additive on SDK v3.30. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W30 surface wraps
the W29 ``GeometryPartitionedOrchestrator`` with four NEW audited
proxies at the capsule layer: a multi-stride basis-history
accumulator, a per-partition calibration prior, a sorted ancestor
chain, and a cross-host disagreement-routing decision. The new
"calibration / multi-stride basis history / cross-host disagreement-
routing / ancestor-chain" vocabulary is added at the **capsule
layer as audited proxy** — explicitly NOT a learned model, NOT
transformer-internal subspace projection, NOT a Riemannian
curvature, NOT a temporal-ordering proof at the model layer.*

**New surface (W30 family, multi-agent-coordination research slice).**

``BasisHistory``, ``AncestorChain``, ``PartitionCalibrationVector``,
``CalibratedGeometryRatificationEnvelope``,
``CalibratedGeometryRegistry``, ``W30CalibratedResult``,
``CalibratedGeometryOrchestrator``,
``verify_calibrated_geometry_ratification`` (14 enumerated failure
modes — disjoint from W29's 14, cumulative 28-mode trust boundary
across W29 + W30), ``update_partition_calibration_running_mean``,
``build_trivial_calibrated_registry``, ``build_calibrated_registry``,
W30 branch constants (``W30_BRANCH_*``),
``W30_CALIBRATED_SCHEMA_VERSION``,
``W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD = 0.5``.  W29's
``GeometryPartitionedOrchestrator`` extended with one new optional
``partition_classifier_hook`` field — the W30 extension point that
lets the calibrated orchestrator inject calibration / disagreement-
route overrides into the inner W29's pre-dispatch path.  New
benchmark phase ``vision_mvp.experiments.phase77_calibrated_dense_control``.

``__experimental__`` tuple in ``vision_mvp.wevra.__init__`` extended
with all new W30 symbols.  ``SDK_VERSION`` bumped to
``wevra.sdk.v3.31``.  ``pyproject.toml`` version ``0.5.4``.

**Headline empirical results (W30 family).**

* **R-77-CHAIN-CRAM (H6 cram-factor anchor).** First empirical
  discharge of **W29-C-CRAM-AMPLIFICATION**:
  ``cram_factor_w30 / cram_factor_w28 = 8.74`` AND
  ``cram_factor_w30 / cram_factor_w29 = 3.80`` across **5/5 seeds**
  (11, 17, 23, 29, 31) at ``calibration_stride = 28``,
  ``ancestor_window = 12``.  Pre-committed bar: 8.0× over W28 ✓ and
  2.0× over W29 ✓.  Max overhead = 1 token/cell.  W30's multi-stride
  basis-history accumulator packs 28 basis CIDs + 12 ancestor CIDs
  + 3 calibration priors onto one ``<calibrated_ref:DDDD>`` wire
  token.
* **R-77-CALIBRATION-PRIOR (H7 calibration anchor).** First empirical
  discharge of **W29-C-PARTITION-CALIBRATION**:
  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29 =
  +0.250 across 5/5 seeds`` with ``trust_precision_w30 = 1.000``.
  Calibrated priors (0.95, 0.95, 0.30) trigger reroute on 8 CYCLIC
  cells (low historical agreement) → routed to LINEAR (high agreement,
  full oracle) → strict correctness recovery.  W29 baseline 0.750,
  W30 1.000.  Sharpens **W21-C-CALIBRATED-TRUST** on the per-partition
  axis.
* **R-77-XHOST-DISAGREE (H8 disagreement-routing anchor).**  Sharpens
  **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE**:
  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29 =
  +0.250 across 5/5 seeds`` with ``trust_precision_w30 = 1.000``.
  Synthetic disagreement probe (simulating a more-pessimistic
  cross-host LLM probe at temperature 0) systematically rejects;
  cross-host variance witness fires on every ratified cell; W30
  reroutes 15 cells to high-trust = CYCLIC (full oracle).
* **R-77-TRIVIAL-CALIBRATION (H2 byte-for-W29).** W30 = W29
  byte-for-byte; every cell yields
  ``W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH``.
* **R-77-CALIBRATED-TAMPERED (H3).** 100% rejection rate across five
  named tamper modes (basis_history_cid, calibration_vector,
  ancestor_chain_cid, disagreement_route_target, calibrated_cid).
* **R-77-NON-CALIBRATABLE (W30-Λ-non-calibratable).** Uniform priors
  (1.0, 1.0, 1.0) ⇒ no override fires ⇒ W30 = W29 on correctness;
  ``n_calibration_rerouted = 0`` correctly reported.
* **R-77-DEGENERATE-HISTORY (W30-Λ-degenerate-history).**  ``stride =
  1`` ⇒ no real cram amplification; ``cram_ratio_w30/w29 = 0.86 ≤
  1.20`` correctly reported.

Trust boundary: tampered envelopes rejected **100%** across every
sub-bank, every seed, every named mode.  Cumulative 28 enumerated
trust-boundary failure modes across W29 + W30.  **36/36 W30 unit
tests + 273/273 phase69-77 tests + 84/84 wider wevra_* tests pass**.

**Honest scope (what W30 does NOT claim).**

* W30 does NOT claim "we solved context."
* W30 does NOT claim a learned model.  Calibration vector is a
  vector of floats in [0, 1] registered at construction; the
  running-mean update is closed-form arithmetic.
* W30 does NOT claim transformer-internal latent control.  The
  basis history is a capsule-layer accumulator over W29's
  deterministic basis CIDs; honest **proxy** for the LatentMAS
  shared-substrate direction, not a runtime KV transplant.
* W30 does NOT claim a temporal-ordering proof at the model layer.
  The ancestor chain is a sorted tuple of registered ancestor CIDs;
  the chain CID is SHA-256 over canonical bytes.
* W30 does NOT bring up Mac 2 (192.168.12.248 ARP-incomplete, 25th
  consecutive milestone).  Two reachable hosts (localhost +
  192.168.12.191) suffice for the synthetic discharge.
* W30 does NOT solve full live LLM disagreement reduction.  The H8
  strict gain is on synthetic disagreement; live-LLM extension
  remains open as **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**.
* W30 does NOT close ``W29-C-NATIVE-LATENT`` (architecture-dependent;
  the next true wall).

**Discharges (in this milestone).**

* **W29-C-CRAM-AMPLIFICATION** (H6).
* **W29-C-PARTITION-CALIBRATION** (H7).
* **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** (H8 synthetic axis;
  live axis carried forward as
  W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE).
* **W21-C-CALIBRATED-TRUST** sharpened (per-partition calibrated
  priors are the natural land for the W21 conjecture).

**Verdict against pre-committed `SUCCESS_CRITERION_W30_*.md`:**
**STRONG SUCCESS** — 10/10 hard gates met AND ≥ 4/5 soft gates
PASS or honestly-null with explanation.

## [0.5.3 / 3.30] — 2026-04-30 — SDK v3.30 — geometry-partitioned product-manifold dense control + audited subspace-basis payload + factoradic Lehmer routing index + causal-validity gate + cross-host variance witness + first empirical discharge of W28-C-CROSS-HOST-VARIANCE on the magnitude axis

*Strictly additive on SDK v3.29. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W29 surface wraps
the W28 ensemble-verified multi-chain orchestrator with a structural
geometry-partitioning step (linear / hierarchical / cyclic) that
optionally dispatches each cell through a per-partition inner W28
stack with its own oracle / probe topology. The new geometry /
Grassmannian / factoradic / Lehmer / mixed-curvature / causal-
validity vocabulary is added at the **capsule layer as audited
proxy** — explicitly NOT a transformer-internal subspace projection,
NOT a Riemannian curvature, NOT a learned manifold.*

**New surface (W29 family, multi-agent-coordination research slice).**

`SubspaceBasis`, `verify_subspace_basis`,
`compute_structural_subspace_basis`,
`encode_permutation_to_factoradic`, `decode_factoradic_to_permutation`,
`CrossHostVarianceWitness`,
`GeometryPartitionedRatificationEnvelope`,
`PartitionRegistration`, `GeometryPartitionRegistry`,
`W29PartitionResult`, `GeometryPartitionedOrchestrator`
(supports optional `inner_per_partition` per-partition dispatch +
`pre_dispatch_by_partition`), `classify_partition_id_for_cell`,
`verify_geometry_partition_ratification` (14 enumerated failure
modes), `build_trivial_partition_registry`,
`build_three_partition_registry`. W29 branch constants
(`W29_BRANCH_*`). New benchmark phase
`vision_mvp.experiments.phase76_geometry_partitioned_product_manifold`.

`__experimental__` tuple in `vision_mvp.wevra.__init__` extended
with all new W29 symbols. `SDK_VERSION` bumped to
`wevra.sdk.v3.30`. `pyproject.toml` version `0.5.3`.

**Headline empirical results (W29 family).**

* **R-76-XHOST-DRIFT (H6 load-bearing).** First synthetic regime
  where the dense-control synthesis strictly improves correctness
  over BOTH W27 AND W28 baselines:
  `correctness_ratified_rate_w27 =
  correctness_ratified_rate_w28 = 0.500` and
  `correctness_ratified_rate_w29 = 0.750`, **Δ = +0.250 across 5/5
  seeds (11, 17, 23, 29, 31)**, `trust_precision_w29 = 1.000`,
  `mean_overhead_w29_vs_w28_per_cell = 0.75`, `max_overhead = 1`.
  **Discharges W28-C-CROSS-HOST-VARIANCE on the empirical-magnitude
  axis.**
* **R-76-CROSS-HOST-LIVE (S1/S2/S3).** Same H6 +0.250 gain holds on
  the live two-host LLM topology (localhost gemma2:9b +
  192.168.12.191 qwen2.5:14b); 16 cross-host probe calls; 710 LAN
  bytes; `trust_precision_w29 = 1.000`. Structural partition signal
  alone drives the gain even when the LLM ensemble agrees.
* **R-76-TRIVIAL-PARTITION (H2 byte-for-W28).** W29 = W28
  byte-for-byte (`byte_equivalent_w29_w28 = true`); every cell yields
  `W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH`.
* **R-76-NON-ORTHOGONAL-BASIS (H5).** 12/12 tampered envelopes
  rejected with `subspace_basis_non_orthogonal`.
* **R-76-PARTITION-TAMPERED (H3).** 60/60 tampers rejected across
  five named modes.
* **R-76-COORDINATED-DRIFT-XHOST (H5).** When per-partition pools
  share the same oracle, partition routing cannot improve
  correctness; `Δ(W29-W28) = 0.000` correctly reported.
* **Cram-factor (H7).** Measured 2.30× on R-76-CHAIN-SHARED — short
  of the pre-committed 8× bar. Mechanism real, magnitude below bar;
  becomes `W29-C-CRAM-AMPLIFICATION` (open).

Trust boundary: tampered envelopes rejected **100%** across every
sub-bank, every seed, every named mode. **935/935 + 6 subtests pass**
across W3..W29 + capsule + public API + runtime + LLM backend.

**Honest scope (what W29 does NOT claim).**

* W29 does NOT claim transformer-internal subspace projection. The
  basis lives at the capsule layer; verifier checks orthogonality,
  finiteness, content-address.
* W29 does NOT claim Riemannian curvature. The "geometry partition"
  is a structural label.
* W29 does NOT claim a learned manifold. Basis and classifier are
  pure functions over deterministic structural inputs.
* Mac 2 (192.168.12.248) remains ARP-incomplete (24th consecutive
  milestone).
* W29 does NOT solve `W22-C-CACHE-AMPLIFICATION` or full live LLM
  disagreement reduction. Both retained as named open conjectures.

## [0.5.2 / 3.29] — 2026-04-30 — SDK v3.29 — ensemble-verified cross-model multi-chain pivot ratification + Phase-75 R-75 benchmark family + W28 family + first cross-host live LLM evidence in 23 milestones + stable-vs-experimental boundary tightened

*Strictly additive on SDK v3.28. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W28 surface
composes the W21 trust-weighted multi-oracle quorum (the **old**
explicit-capsule line) with the W27 multi-chain salience-keyed
pool (the **new** dense-control line) inside one decision, behind
a controller-verified ratification envelope with 11 new enumerated
failure modes.*

**New surface (W28 family, multi-agent-coordination research slice):**

* `vision_mvp.wevra.team_coord.ProbeVote` — frozen probe-vote
  dataclass with strict invariants (ratify ⊕ reject; trust_weight ≥ 0).
* `vision_mvp.wevra.team_coord.EnsembleProbe` — duck-typed Protocol
  surface for any object with `probe_id` + `vote(...)`.
* `vision_mvp.wevra.team_coord.EnsembleProbeRegistration` — mirrors
  `OracleRegistration` (the W21 entry); carries `trust_prior`,
  `role_label`, `host_id` for cross-host telemetry.
* `vision_mvp.wevra.team_coord.DeterministicSignatureProbe` —
  locally-recomputable probe; trivially trustworthy;
  `wire_required = False` (the K=1-byte-for-W27 path).
* `vision_mvp.wevra.team_coord.OracleConsultationProbe` — wraps any
  W20/W21 `OutsideWitnessOracle` (composes with `ServiceGraphOracle`,
  `ChangeHistoryOracle`, etc.).
* `vision_mvp.wevra.team_coord.LLMSignatureProbe` — wraps any
  `LLMBackend` (Ollama or MLX-distributed); designed for the
  two-host topology with cross-host round-trip-bytes telemetry.
* `vision_mvp.wevra.team_coord.EnsemblePivotRatificationEnvelope`
  — content-addressed ensemble decision envelope (signature_cid,
  probe_votes, quorum_threshold, quorum_weight, ratified flag,
  ratification_cid). 11 enumerated failure modes via
  `verify_ensemble_pivot_ratification`.
* `vision_mvp.wevra.team_coord.EnsembleRatificationRegistry` —
  controller-side ratification registry with cross-host telemetry.
* `vision_mvp.wevra.team_coord.EnsembleVerifiedMultiChainOrchestrator`
  — the load-bearing W28 wrapper around
  `MultiChainPersistedFanoutOrchestrator`.
* `vision_mvp.wevra.team_coord.W28EnsembleResult` — per-cell audit
  record with probe vote summary, quorum weight, ratification CID,
  cross-host bytes.
* `vision_mvp.wevra.team_coord.verify_ensemble_pivot_ratification`
  — 11-mode pure verifier (rejects: empty, schema-version, schema-cid,
  signature-cid-empty, signature-cid-mismatch, probe-table-empty,
  probe-id-unregistered, probe-vote-malformed, trust-weight-negative,
  hash-mismatch, quorum-below-threshold, quorum-recompute-mismatch).
* Convenience factories: `build_default_ensemble_registry`,
  `build_two_probe_oracle_ensemble_registry`,
  `build_cross_host_llm_ensemble_registry`.
* W28 branch vocabulary (8 branches): `ratified`,
  `ratified_passthrough`, `quorum_below_threshold`, `probe_rejected`,
  `no_ratify_needed`, `fallback_w27`, `no_trigger`, `disabled`.

**New benchmark family (R-75):**

* `vision_mvp/experiments/phase75_ensemble_verified_multi_chain.py`
  — eight pre-committed sub-banks (single_probe, chain_shared,
  cross_model_drift, coordinated_drift, trust_zero,
  ratification_tampered, pool_exhausted, cross_host_live) +
  `discover_two_host_topology()` + cross_regime + seed_sweep CLI.
* `vision_mvp/tests/test_phase75_ensemble_verified_multi_chain.py`
  — 34 unit + integration tests covering every probe, every
  verifier failure mode, every named falsifier, byte-for-byte W27
  equivalence, disabled/no-trigger paths, two-host topology.
* `vision_mvp/experiments/artifacts/phase75/` — 9 result JSONs
  (1 cross_regime, 7 seed sweeps, 1 topology, 1 cross_host_live).

**Headline empirical results.**

* **R-75-SINGLE-PROBE (H2 anchor)**: W28 = W27 byte-for-byte across
  5/5 seeds; `byte_equivalent_w28_w27 = true`; 16/16 cells ratified
  via the `ratified_passthrough` branch with 0 token overhead.
* **R-75-CROSS-MODEL-DRIFT (S3 / W28-3 headline, synthetic)**: W28
  overhead = 1.00 token/cell across 5/5 seeds (within S4 ≤ 2 budget);
  16/16 ratified; trust precision 1.000.
* **R-75-RATIFICATION-TAMPERED (H3 trust falsifier)**: 16/16
  tampered envelopes rejected per seed (5/5 seeds), reject reason
  `quorum_recompute_mismatch`.
* **R-75-CROSS-HOST-LIVE (S1/S2, FIRST CROSS-HOST EVIDENCE IN 23
  MILESTONES)**: 16-cell live run on localhost (gemma2:9b) +
  192.168.12.191 (qwen2.5:14b); 128 cross-host probe calls; 5592
  bytes serialised over LAN; **10/16 ratified by ensemble; 6/16
  fell to quorum_below_threshold (real LLM disagreement);
  trust_precision 1.000; W28 correctness 1.000**.

**Falsifiers all empirically confirmed:** W28-Λ-single-probe,
W28-Λ-coordinated-drift, W28-Λ-trust-zero, W28-Λ-spoofed-probe,
W28-Λ-quorum-tampered, W28-Λ-pool-exhausted-passthrough.

**Conjectures introduced:** W28-C-CROSS-HOST-VARIANCE (variance
reduction magnitude on a regime where W27 itself makes mistakes —
open; the synthetic bench is already 1.000-correct under W27 so the
S3 headline is null but honest); W28-C-CALIBRATED-TRUST (calibrated
trust priors strictly outperform uniform; not exercised in this
milestone — natural follow-up).

**Old-line discharges:** the W21 / W27 *synthesis target* named in
the master plan post-W27 next-steps section is operational —
`OracleConsultationProbe` makes the W21 oracle interface a
first-class W28 probe; W21 trust priors thread directly into W28
quorum weights; the same `OutsideWitnessOracle` duck-type drives
both W21 quorum and W28 ratification.

**Stable-vs-experimental boundary tightened (H7 release-readiness):**

* `vision_mvp.wevra.__init__.SDK_VERSION` bumped to
  `"wevra.sdk.v3.29"`.
* `vision_mvp.wevra.__init__.__experimental__` — new explicit tuple
  listing every dense-control symbol (W22..W28); external callers
  should pin a specific SDK version when depending on these.
* `pyproject.toml` version bump 0.5.1 → 0.5.2.

**Regression:** 222/222 W23..W28 stack tests + 534/534 wider
focused regression (W3 capsules, W4 team, W12-W15 packing/decoder
ladder, W18-W21 explicit-capsule trust line, W22-W28 dense-control
line, public API, runtime, LLM backend) — **all preserved
byte-for-byte**.

**Two-host topology used:**
- `localhost` → `gemma2:9b` (Gemma2 family)
- `192.168.12.191` → `qwen2.5:14b` (Qwen2.5 family)
- `192.168.12.248` → ARP-incomplete (23rd consecutive milestone).

## [3.28] — 2026-04-30 — SDK v3.28 — multi-chain salience-keyed dense-control fanout + per-signature scoping + Phase-74 R-74 benchmark family + W27 family + W26-C-DIVERGENCE-RECOVERY discharged (first capsule-native multi-agent-coordination method that simultaneously improves both efficiency AND correctness over W26 on a regime where W26's single-stack scope architecturally limits correctness — measured −76.27% total token reduction AND +0.500 correctness gain over W26 on R-74-XORACLE-RECOVER at 5/5 seeds, trust boundary sound via 12 enumerated failure modes across 2 new verify_* functions, four named W27-Λ falsifiers all empirically confirmed)

*Strictly additive on SDK v3.27. The Wevra single-run product
runtime contract is byte-for-byte unchanged.*

**New surface (W27 family, multi-agent-coordination research slice):**

* `vision_mvp.wevra.team_coord.SalienceSignatureEnvelope` — content-
  addressed signature over canonical compact state (4 enumerated
  failure modes via `verify_salience_signature`).
* `vision_mvp.wevra.team_coord.ChainPivotEnvelope` — per-cell pivot
  to an existing chain in the multi-chain pool (8 enumerated
  failure modes via `verify_chain_pivot`).
* `vision_mvp.wevra.team_coord.MultiChainPersistedFanoutRegistry` —
  controller-side multi-chain pool maintained inside the audited
  disambig wrapper.
* `vision_mvp.wevra.team_coord.MultiChainPersistedFanoutDisambiguator`
  — audited W27 wrapper on top of one W26 stack.
* `vision_mvp.wevra.team_coord.SharedMultiChainPool` — team-wide
  pool shared across producer + K consumers; one independent W26
  disambiguator per (signature, agent).
* `vision_mvp.wevra.team_coord.MultiChainPersistedFanoutOrchestrator`
  — load-bearing W27 implementation that routes cells via
  `compute_input_signature_cid` to the matching slot in the team-
  wide pool.
* `vision_mvp.wevra.team_coord.compute_input_signature_cid` —
  deterministic SHA-256 over canonical input handoffs.
* W27 branch vocabulary (7 branches): `pivoted`, `anchored_new`,
  `pool_exhausted`, `pivot_rejected`, `fallback_w26`, `no_trigger`,
  `disabled`.

**New benchmark family (R-74):**

* `vision_mvp/experiments/phase74_multi_chain_pivot.py` — six
  pre-committed sub-banks + cross_regime + signature_period sweep.
* `vision_mvp/tests/test_phase74_multi_chain_pivot.py` — 22 unit +
  integration tests covering the W27 surface end-to-end.
* `docs/data/phase74_cross_regime.json`,
  `docs/data/phase74_xoracle_seed_sweep.json`,
  `docs/data/phase74_signature_period_sweep.json`.

**Headline empirical result (R-74-XORACLE-RECOVER).** W27
**simultaneously** reduces total visible tokens by −76.27 % over
W26 AND raises correctness_ratified_rate from 0.500 to 1.000 at
T_decoder ∈ {None, 24}, K=3, 16 cells. Stable across 5/5 seeds.

**Falsifiers all empirically confirmed:** W27-Λ-single-signature,
W27-Λ-pool-exhausted, W27-Λ-pivot-tampered, W27-Λ-signature-drift.

**Conjectures introduced:** W27-C-MULTI-SIGNATURE-SCALING (M → ∞
asymptote unverified), W27-C-CROSS-HOST (gated on Mac-2 return —
22nd consecutive ARP-incomplete).

**Regression:** 508/508 in the focused suite — all preserved
byte-for-byte.

## [3.27] — 2026-04-30 — SDK v3.27 — chain-persisted dense-control fanout + per-consumer projections + Phase-73 R-73 benchmark family + W26 family + W25-C-K-SCALING discharge (first capsule-native multi-agent-coordination method that amortises the producer's per-cell salience-token cost across cells via 1-token chain-advance references while preserving the W25 multi-consumer fanout floor — measured −68.79% total token reduction over W25 and −90.60% over W24 on K=3 at 5/5 seeds, trust boundary sound on tampered + projection-mismatch falsifiers, W26-Λ-no-chain falsifier confirmed, W25-C-K-SCALING discharged at K∈{3,5,8,10})

*Strictly additive on SDK v3.26. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W26 surface adds
two new content-addressed signed envelopes
(``ChainAnchorEnvelope``, ``ChainAdvanceEnvelope``), one
projection slot (``ProjectionSlot``), three verification
functions (``verify_chain_anchor`` with 6 enumerated failure modes,
``verify_chain_advance`` with 8, ``verify_projection_subscription``
with 2), one controller-side registry
(``ChainPersistedFanoutRegistry``), one audit record
(``W26ChainResult``), and one wrapping decoder
(``ChainPersistedFanoutDisambiguator``) — purely additive in the
multi-agent coordination research slice.*

**Headline (W26-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 T_decoder).** Wrapping the W25
``SharedFanoutDisambiguator`` in the new
``ChainPersistedFanoutDisambiguator`` (W26: producer pays the full
W25 cost ``C ≈ 14.6`` tokens at the chain anchor; subsequent
in-window cells emit a single ``<chain_advance:DDDD>`` token + a
hash-chained ``ChainAdvanceEnvelope``; consumers subscribe at the
anchor via per-consumer ``ProjectionSlot`` and emit a single
``<chain_consumer:DDDD>`` token per cell) over the same
R-69-CACHE-FANOUT oracle ecology
(1 producer + K=3 consumers, 16 cells, ``chain_persist_window =
16``) strictly reduces ``mean_total_w26_visible_tokens`` over
``mean_total_w25_visible_tokens`` by **−12.125 tokens / cell
(−68.79 %)** at ``T_decoder = None``, AND over
``mean_total_w24_visible_tokens`` by **−53.00 tokens / cell
(−90.60 %)**. ``correctness_ratified_rate = 1.0000`` byte-for-byte;
``chain_consumer_resolved_rate = 1.0000``;
``registry_n_anchors = 1``, ``registry_n_advances = 15``. Identical
results at ``T_decoder = 24``. Stable across **5/5** seeds.

**K-scaling discharge (W25-C-K-SCALING).** The W25-C-K-SCALING
conjecture (savings should grow as K×(C−1)) was empirically
discharged at K∈{3,5,8,10}: W25 saving over W24 grows from 69.87 %
at K=3 to 84.69 % at K=10 (close to the conjectured 88 %); W26
saving over W24 grows from 90.60 % at K=3 to 92.23 % at K=10.

**Trust-boundary anchors (W26-3).** ``verify_chain_anchor``
enumerates 6 failure modes; ``verify_chain_advance`` enumerates 8;
``verify_projection_subscription`` enumerates 2. On
R-73-CHAIN-TAMPERED, 14/16 advances rejected via
``parent_mismatch``; correctness preserved via W25 fall-through.
On R-73-PROJECTION-MISMATCH, all 16 cells reject for the
mismatched consumer via ``projection_unauthorized``; the other 2
consumers still resolve.

**Named falsifiers** (all proved-empirical). **W26-Λ-no-chain**:
``chain_persist_window = 1`` reduces W26 to W25 byte-for-byte.
**W26-Λ-tampered**: tampered advances rejected. **W26-Λ-projection-mismatch**:
cross-projection access rejected. **W26-Λ-divergent**: when gold
subset flips at the bench midpoint, the inner W25 fires
``no_trigger`` and W26 falls through; correctness drops to 0.5
by construction.

**Backward-compat (W26-3-A, W26-3-B).** With ``enabled = False``
OR ``chain_registry = None``, W26 reduces to W25 byte-for-byte.
Full pre-existing W22..W25 + IS-1 / IS-2 test surfaces preserved
byte-for-byte (180/180 in the focused regression).

**Theoretical (W26-L, proved by inspection).** Any capsule-native
multi-agent coordination strategy whose producer emits only its
own compact state and whose consumers reference it via 1-token-
per-cell tokens has a per-cell total visible cost ≥ 1 + K. W26
attains this floor on every in-window advance cell.

**Mac 2 status: ARP-incomplete (21st consecutive milestone)**;
all results Mac-1 only. The W26 surface inherits the W24
``CrossProcessProducerDecoderWire`` as the strongest cross-
process honesty validated end-to-end on this repo. The
wire-bytes vs token-cost tradeoff is named **W26-C-MULTI-HOST**;
remains conjectural.

**New tests:** 63/63 pass on
``test_phase73_chain_persisted_fanout.py``. Full focused
regression on W22..W26 + IS-1 / IS-2: **180/180 + 6 subtests
pass in 15.6s**.

See ``docs/RESULTS_WEVRA_W26_CHAIN_PERSISTED_FANOUT.md`` for the
milestone note, ``docs/THEOREM_REGISTRY.md`` for the 13 new
theorems / falsifiers / conjectures (W26-1 through W26-C-MULTI-HOST,
plus W25-C-K-SCALING discharge), and
``docs/context_zero_master_plan.md`` § 4.44 for the
master-plan-level audit board.

## [3.26] — 2026-04-29 — SDK v3.26 — shared-fanout dense-control + cross-agent state reuse + Phase-72 R-72 benchmark family + W25 family (first capsule-native multi-agent-coordination method that extends W24 single-agent compaction to the multi-agent case — one producer computes 1 FanoutEnvelope for K named consumers, each consumer resolves via 1 ``<fanout_ref:DDDD>`` token, measured −69.87% total token reduction on K=3 at 5/5 seeds, trust boundary sound on poisoned-consumer falsifier, W25-Λ-disjoint named falsifier confirmed)

*Strictly additive on SDK v3.25.* The W25 surface adds one new
content-addressed signed envelope (``FanoutEnvelope``), one
controller-side registry (``SharedFanoutRegistry``), one
verification function (``verify_fanout``), one audit record
(``W25FanoutResult``), and one wrapping decoder
(``SharedFanoutDisambiguator``). On R-72-FANOUT-SHARED (1
producer + K=3 consumers, 16 cells, R-69-CACHE-FANOUT oracle
ecology), W25 strictly reduces ``mean_total_w25_visible_tokens``
over ``mean_total_w24_visible_tokens`` by **−40.875 tokens / cell
(−69.87 %)**; ``correctness_ratified_rate = 1.0000``;
``fanout_consumer_resolved_rate = 1.0000``. Stable across **5/5**
seeds. See ``docs/RESULTS_WEVRA_W25_SHARED_FANOUT.md``.

## [3.25] — 2026-04-29 — SDK v3.25 — bounded-window session compaction + intra-cell resample-quorum + real cross-process producer/decoder wire + Phase-71 R-71 benchmark family + W24 family (first capsule-native multi-agent-coordination method that combines bounded-window session compaction with intra-cell resample-quorum mitigation and a real OS-level cross-process producer/decoder wire — measured efficiency gain on long sessions, empirical discharge of W23-C-MITIGATION-LIVE-VARIANCE on the intra-cell drift axis, and the first real OS-level subprocess pipe in the programme)

*Strictly additive on SDK v3.24. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W24 surface adds
one new content-addressed signed envelope
(``SessionCompactEnvelope``), one verification function
(``verify_session_compact``), one audit record
(``W24CompactionResult``), one wrapping decoder
(``MultiCellSessionCompactor``), one intra-cell resample-quorum
caching adapter (``ResampleQuorumCachingOracleAdapter``), one
real cross-process producer/decoder wire
(``CrossProcessProducerDecoderWire``), and one synthetic intra-
cell drift oracle (``IntraCellFlippingOracle``) — purely additive
in the multi-agent coordination research slice.*

**Headline (W24-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 budgets).** Pairing the W23
``CrossCellDeltaDisambiguator`` with the new
``MultiCellSessionCompactor`` over a registered ``SchemaCapsule``
on R-71-LONG-SESSION (16-cell session, ``compact_window = 4``)
strictly reduces ``mean_n_w24_visible_tokens_to_decider`` over
the W23 baseline by **−6.81 tokens / cell (−18.0 %)** at
``T_decoder = None`` and by **−6.81 tokens / cell (−20.5 %)** at
``T_decoder = 24``. ``compact_verifies_ok_rate = 0.812`` (13/16
cells beyond the window); ``correctness_ratified_rate = 1.000``
byte-for-byte vs W22; ties W22 / W23 byte-for-byte on
``accuracy_full = 1.000``. Stable across **5/5** alternate
``bank_seed`` values: savings ≥ 6.69 tokens/cell on every seed,
mean savings 6.79 tokens/cell.

**W24-2 mitigation (proved-empirical n=8 saturated synthetic +
proved-empirical-research n=4 live mixtral).** On
R-71-INTRA-CELL-FLIP (synthetic ``IntraCellFlippingOracle``
registered in isolation so its vote is decisive in W21 quorum),
the W23 PER_CELL_NONCE baseline ties FIFO at ``acc_full = 0.000``
(each cell's first consult is the bad one); the W24
``ResampleQuorumCachingOracleAdapter`` (M=3, T=2) achieves
``acc_full = 0.500`` — **+0.500 strict mitigation advantage**.
**Empirically discharges W23-C-MITIGATION-LIVE-VARIANCE on the
intra-cell drift axis**. Live transfer to ``mixtral:8x7b`` on
Mac-1 Ollama (n=4): W23 quorum-keyed = 0.500, W24 resample =
**0.750** — **+0.250 strict gain on a fresh live LLM stream**.

**W24-3 trust-boundary soundness + real cross-process wire
(proved-empirical n=16 + proved by inspection).** On
R-71-COMPACT-TAMPERED, every tampered window is rejected (12/16
cells fire ``window_cids_mismatch`` → fall through to W23
byte-for-byte; ``correctness_ratified_rate = 1.000``). On
R-71-CROSS-PROCESS, the ``CrossProcessProducerDecoderWire`` spawns
a real Python subprocess and round-trips JSON envelopes via
stdin/stdout pipes: **12 861 bytes round-tripped on n=16, 0
failures** — a strictly stronger cross-process honesty proxy than
the W23 within-process round-trip.

**W24-Λ-no-compact (named falsifier).** On R-71-NO-COMPACT (chain
reset every cell), ``n_w24_compact_resolved_cells = 0`` AND W24
reduces to W23 byte-for-byte. Names the structural limit when the
chain length stays below the window.

**W24-Λ-real (proved-conditional + empirical-research n=4).**
Live mixtral 8x7b probe on R-71-INTRA-CELL-FLIP yields +0.250
strict mitigation advantage; the synthetic +0.500 does not fully
transfer because the live LLM does not perfectly match the
deterministic IntraCellFlippingOracle pattern. Names
**W24-C-LIVE-VARIANCE-COMPLETE** as the follow-up conjecture
frontier.

**Backward-compat.** 121/121 phase-69/70/71 + capsule tests pass;
33/33 new W24 tests pass; 619/619 wevra-anchor + capsule + recent
phases pass. With ``enabled = False`` OR ``schema = None`` OR no
multi-cell window, W24 reduces to W23 byte-for-byte.

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — **18th milestone in a row**.
**No two-Mac sharded inference happened in SDK v3.25.** The W24-3
``CrossProcessProducerDecoderWire`` upgrades the W23 within-
process round-trip to a real OS-level subprocess pipe — the
strongest cross-process honesty this repo can validate end-to-end
on Mac-1 alone. When Mac 2 returns the same JSON-canonical
interface drops in over a real socket with no W24 code changes.

See ``docs/RESULTS_WEVRA_W24_SESSION_COMPACTION.md`` for the
theory-forward results note,
``docs/THEOREM_REGISTRY.md`` for the W24 theorem family entries,
``docs/HOW_NOT_TO_OVERSTATE.md`` § "W24 forbidden moves" for the
canonical do-not-overstate rules, and
``vision_mvp/experiments/phase71_session_compaction.py`` for the
R-71 driver.

## [3.23] — 2026-04-29 — SDK v3.23 — capsule + audited latent-state-sharing hybrid + R-69 Phase-69 benchmark family + W22 family (first capsule-native multi-agent-coordination method that combines explicit-capsule passing with audited proxies for the LatentMAS direction — schema-passing, delta execution, shared-read cache, controller-verified latent digest envelope — measured efficiency gain on a regime where the W21 wire-cost concern actually applies)

*Strictly additive on SDK v3.22. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W22 surface adds
one new content-addressed dataclass (``SchemaCapsule``), one
typed signed envelope (``LatentDigestEnvelope``), one verification
result (``LatentVerificationOutcome``), one wire-side
write-once-read-many cache (``SharedReadCache``), one drop-in
oracle adapter (``CachingOracleAdapter``), one falsifier-test
primitive (``EnvelopeTamperer``), one audit record
(``W22LatentResult``), one controller verifier
(``verify_latent_digest``), and one wrapping decoder
(``LatentDigestDisambiguator``) — purely additive in the
multi-agent coordination research slice.*

**Headline (W22-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 cells).** Pairing the W21
``TrustWeightedMultiOracleDisambiguator`` with the new
``LatentDigestDisambiguator`` over a registered ``SchemaCapsule``
+ a shared ``SharedReadCache`` (every oracle wrapped in
``CachingOracleAdapter``) on R-69-CACHE-FANOUT strictly reduces
``mean_n_visible_tokens_to_decider`` over the W21 baseline by
**−7 tokens / cell (−14.51 %)** at ``T_decoder = None`` and by
**−7 tokens / cell (−16.09 %)** at ``T_decoder = 24``, AND
records ``cache_tokens_saved_total = 88`` over n=8 (oracle-side
wire savings), AND ties W21 byte-for-byte on
``accuracy_full = 1.000``. Stable across 5/5 alternate
``bank_seed`` values (savings exactly +7 tokens / cell on every
seed; cache_tokens_saved=88 on every seed; correctness ratified
rate=1.000 on every seed). The first capsule-native multi-agent-
coordination method that combines explicit-capsule passing with
audited proxies for the LatentMAS direction (collective KV
pooling / latent hidden-state transfer / super-token side
channels). Three named falsifiers (W22-Λ-no-cache,
R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) and one backward-compat
anchor (R-69-NO-TRIGGER) make the W22-1 conditionality sharp.

**Live LLM transfer (W22-Λ-real, empirical n=4 × 2 models,
partially discharged).** Mac-1 mixtral 8x7b on cache_fanout:
visible-tokens savings **+39.08 %** (W21=87, W22=53 tokens / cell);
cache_tokens_saved_total=120 over 4 cells; correctness ratified
rate=0.750 — newly named conjecture **W22-C-CACHE-AMPLIFICATION**
(the cache freezes a probabilistic LLM oracle's first reply
across all matching cells; cell-1's variance amplifies). gemma2:9b
ties W21 byte-for-byte at 0.250 (gemma2's closure-landing rate is
the structural bound).

**Backward-compat preserved byte-for-byte.** With ``enabled=False``
OR no ``SchemaCapsule`` registered OR an inner W21 branch outside
``trigger_branches``, the W22 layer reduces to W21 byte-for-byte
on the answer field. 633 prior wevra-suite tests + 32 new W22
tests + 10 misc = **675/675** pass.

**Audit T-1..T-7 preserved** on every cell of every regime.

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` (16th milestone in a row); no two-Mac sharded
inference. The W22 surface is naturally a producer / cache-
controller separation (wire-compatible with cross-host
deployment) — no W22 code changes required when Mac-2 returns.

**Closes** the wire-cost half of the SDK v3.22
W21-C-CALIBRATED-TRUST conjecture (the *correctness* half remains
open and orthogonal). See
`docs/RESULTS_WEVRA_CAPSULE_LATENT_HYBRID.md` for the full
SDK v3.23 milestone note; `vision_mvp/experiments/phase69_capsule_latent_hybrid.py`
for the bench driver; `vision_mvp/tests/test_phase69_capsule_latent_hybrid.py`
for the 32 new tests.

## [3.22] — 2026-04-29 — SDK v3.22 — trust-weighted multi-oracle adjudicator + R-68 multi-oracle benchmark family + W21 family (first capsule-native multi-agent-coordination method that crosses the W20-Λ-compromised wall on a regime where single-oracle reasoning is structurally insufficient by adjudicating across N registered outside oracles under bounded context)

*Strictly additive on SDK v3.21. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W21 surface adds
one new dataclass (``OracleRegistration``), four oracle adapters
(``ChangeHistoryOracle`` / ``OnCallNotesOracle`` /
``SingletonAsymmetricOracle`` / ``DisagreeingHonestOracle``), two
audit dataclasses (``W21OracleProbe``, ``W21MultiOracleResult``),
six new branch constants (``W21_BRANCH_QUORUM_RESOLVED``,
``W21_BRANCH_NO_QUORUM``, ``W21_BRANCH_SYMMETRIC_QUORUM``,
``W21_BRANCH_NO_ORACLES``, ``W21_BRANCH_NO_TRIGGER``,
``W21_BRANCH_DISABLED``), and one wrapping decoder
(``TrustWeightedMultiOracleDisambiguator``). All purely additive
in the multi-agent coordination research slice.*

**Headline (W21-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 cells, also n=12).** Pairing the W19
``BundleContradictionDisambiguator`` with the new
``TrustWeightedMultiOracleDisambiguator`` over a three-oracle
registered set ``(compromised_registry first, service_graph,
change_history)`` under default ``quorum_min = 2`` achieves
``accuracy_full = 1.000`` on R-68-MULTI-MAJORITY-LOOSE
(``T_decoder = None``) AND R-68-MULTI-MAJORITY-TIGHT
(``T_decoder = 24``), strictly improving over **every** non-W21
capsule baseline including W20 (which trusts the first-registered
compromised oracle and FAILS at 0.000) by **+1.000**, stable across
**5/5** alternate ``bank_seed`` values (11, 17, 23, 29, 31). The
first capsule-native multi-agent-coordination method that crosses
the W20-Λ-compromised wall on a regime where the wall actually
applies. Three named falsifiers (W21-Λ-no-quorum,
W21-Λ-all-compromised, W21-Λ-partial) make the W21-1 conditionality
sharp; the conditional W21-C-PARTIAL-RECOVERY (with override
``quorum_min = 1`` on R-68-MULTI-PARTIAL) is empirically discharged
at 1.000 — the quorum-strictness trade-off is real.

**Live LLM transfer (SDK v3.22, W21-Λ-real / W21-C-LIVE-WITH-REGISTRY,
empirical n=4 × 2 models, partially discharged).** Two regimes:

* **Mixed-registry (registry-anchored, easy)** — four-oracle
  registry pairing deterministic ``service_graph`` +
  ``change_history`` with ``ollama_mixtral:8x7b``: W21 = 1.000,
  +1.000 over W20. ``W21-C-LIVE-WITH-REGISTRY`` partially
  discharged.
* **Coalition (LLM-vote-required, hard)** — three-oracle registry
  with one honest deterministic + one LLM + one compromised,
  ``quorum_min = 2`` (LLM vote required for quorum on gold):
  ``ollama_mixtral:8x7b`` (47B-MoE) lands gold tokens through the
  W18/W19 closure on 3/4 cells → W21 = **0.750**, **+0.750 over
  W20**; ``ollama_gemma2:9b`` (9.2B-dense) lands decoy tokens
  through the closure → W21 = **0.000**, **+0.000 over W20**.
  Cross-model split (47B-MoE / 9.2B-dense) sharp; **scale + general
  knowledge matter for the W21-Λ-real escape on the LLM-vote-
  required regime**.

**Two-Mac status (SDK v3.22).** Mac 2 (192.168.12.248) ARP
``incomplete`` — same status as SDK v3.6 through SDK v3.21 (15th
milestone in a row). **No two-Mac sharded inference happened in SDK
v3.22.** The W21 oracle Protocol is *naturally* a producer / multi-
adjudicator separation; cross-host deployment (registry on Mac-1,
LLM adjudicator on Mac-2) is wire-compatible — no W21 code changes
required when Mac-2 returns. Strongest model class actually
exercised: single-Mac ``mixtral:8x7b`` (46.7B-MoE Q4) on Mac 1
Ollama.

**Bounded-context honesty (SDK v3.22).** The W21 layer issues
*exactly N = ``len(oracle_registrations)``* outside queries per
cell, each bounded by ``max_response_tokens``. The inner W15
``tokens_kept`` is byte-for-byte identical between W19, W20 AND
W21 on R-68-MULTI-MAJORITY-TIGHT (mechanically verified). Total
context delivered to the final decider on the 3-oracle stack:
``tokens_kept (≤ T_decoder) + 3 × n_outside_tokens (each ≤
max_response_tokens)``.

**Backward-compat (W21-3-A / W21-3-B) preserved byte-for-byte.**
With ``enabled = False`` OR no oracles registered, W21 reduces to
W19 byte-for-byte. With ``quorum_min = 1`` AND a single registered
honest oracle, W21 ties W20 byte-for-byte on R-67-OUTSIDE-RESOLVES.
Full SDK regression: **633 / 633 wevra tests pass** (= 585 prior +
48 new W21 tests).

**Closes the named SDK v3.21 conjectures.** W20-C-MULTI-ORACLE
(named conjectural in SDK v3.21) is **discharged** by W21-1 on
R-68-MULTI-MAJORITY. W20-C-LIVE-WITH-REGISTRY (named conjectural
in SDK v3.21) is **partially discharged** by the live mixed-
registry probe on Mac-1 mixtral 8x7b.

**SDK v3.22 mint files:**

* New W21 surface in ``vision_mvp/wevra/team_coord.py`` (purely
  additive on top of W20).
* New experiment:
  ``vision_mvp/experiments/phase68_multi_oracle_adjudication.py``
  (R-68 driver: 5 sub-banks + cross-regime synthetic + 5-seed
  stability sweep + live mixed-registry / coalition probes).
* New tests:
  ``vision_mvp/tests/test_wevra_multi_oracle_adjudication.py`` (48
  tests).
* New artifacts: ``docs/data/phase68_*.json`` (7 files).
* New milestone results note:
  ``docs/RESULTS_WEVRA_MULTI_ORACLE_ADJUDICATION.md``.
* Master plan (§ 4.39), THEOREM_REGISTRY (W21 family with 11
  entries), SUCCESS_CRITERION (Bar 18), START_HERE, README,
  RESEARCH_STATUS, papers/context_as_objects.md (§ 14.2 escape
  ladder) all updated.
* ``SDK_VERSION = "wevra.sdk.v3.22"``.

## [3.21] — 2026-04-29 — SDK v3.21 — outside-witness acquisition disambiguator + R-67 outside-information benchmark family + W20 family (first capsule-native multi-agent-coordination method that crosses the W19-Λ-outside wall on a regime where bundle-only reasoning is structurally insufficient by acquiring asymmetric outside information)

*Strictly additive on SDK v3.20. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W20 surface
adds one new Protocol (``OutsideWitnessOracle``), three new
dataclasses (``OutsideQuery``, ``OutsideVerdict``,
``W20OutsideResult``), four oracle adapters
(``ServiceGraphOracle`` / ``CompromisedServiceGraphOracle`` /
``AbstainingOracle`` / ``LLMAdjudicatorOracle``), one default
service-graph (``build_incident_triage_service_graph``), one
wrapping decoder (``OutsideWitnessAcquisitionDisambiguator``),
and six W20 branch constants. Re-exported via the SDK
``__init__``. The new
``vision_mvp/experiments/phase67_outside_information.py`` driver
ships as a research-slice addition with one positive anchor
sub-bank (R-67-OUTSIDE-RESOLVES under loose AND tight
``T_decoder``), one backward-compat sub-bank
(R-67-OUTSIDE-REQUIRED-BASELINE), and three named falsifier
sub-banks (R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED,
R-67-JOINT-DECEPTION).*

**The W20 family — outside-witness acquisition disambiguator
(SDK v3.21).** On a synthetic R-67-OUTSIDE-RESOLVES regime
(the same R-66-OUTSIDE-REQUIRED bundle shape — deceptive
primary mentions decoy only AND symmetric secondary witness
mentions all three — but with a deterministic
:class:`ServiceGraphOracle` registered as the outside
information source), every closed-form scorer in the SDK
pre-W20 — substrate FIFO, ``capsule_fifo``, ``capsule_priority``,
``capsule_coverage``, W7-2 cohort, W8 corroboration, W9 multi-
service, W11 multi-round, W12 robust-multi-round, W13 layered,
W15 ``AttentionAwareBundleDecoder``, W18
``RelationalCompatibilityDisambiguator``, AND **W19
``BundleContradictionDisambiguator``** — ties FIFO at
``accuracy_full = 0.000`` (W19-Λ-outside extends verbatim:
W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC``). The W20
method, with the deterministic ServiceGraphOracle, achieves
``accuracy_full = 1.000`` on R-67-OUTSIDE-RESOLVES-LOOSE
(``T_decoder = None``) AND R-67-OUTSIDE-RESOLVES-TIGHT
(``T_decoder = 24``), strictly improving over every non-W20
capsule baseline by **+1.000**, stable across **5/5**
alternate ``bank_seed`` values (11, 17, 23, 29, 31). Three
named falsifiers (R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED,
R-67-JOINT-DECEPTION) make the W20-1 conditionality sharp:
no signal → abstain → tie FIFO; adversarial signal → trust →
fail at 0.000; jointly compromised → tie W19 at 0.000.
Bounded-context honesty: the W20 layer adds *exactly one*
outside query per cell, bounded by ``max_response_tokens =
24``; the W15 ``tokens_kept`` is byte-for-byte identical
between W19 and W20 on R-67-OUTSIDE-RESOLVES-TIGHT. Backward-
compat (W20-3) preserved byte-for-byte: 545 / 545 prior wevra
tests pass + 40 new W20 tests pass = **585 / 585**.

**Live LLM extension (W20-Λ-real, partial pass).** A
:class:`LLMAdjudicatorOracle` over a fresh live Mac-1 Ollama
backend produces *measured*, not claimed, results. Partial
live advance: ``mixtral:8x7b`` (47B-MoE) free-form reply
mentions gold tokens asymmetrically and achieves ``acc_full
= 0.750`` (3/4 cells, ``+0.750`` strict gain over W19) on
``R-67-OUTSIDE-RESOLVES`` at ``n_eval = 4``,
``K_auditor = 12``. Honest negative: ``qwen2.5-coder:7b``
trusts the deceptive primary on every fired cell
(``services=cache``) and ties FIFO at 0.000 — the W20-Λ-real
under-scaled-model failure mode. Cross-model split: scale +
general knowledge correlates with W20-Λ-real escape;
smaller / coding-specialised models can fall into the
deception. Artifacts:
``docs/data/phase67_live_mixtral_8x7b_n4.json``,
``docs/data/phase67_live_qwen2_5_coder_7b_n4.json``.

**Two-Mac status (SDK v3.21).** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK
v3.6 through SDK v3.20. **No two-Mac sharded inference
happened in SDK v3.21.** The W20 ``OutsideWitnessOracle``
Protocol is *infrastructure-ready* for cross-host deployment
(producer roles on Mac 1 + adjudicator on Mac 2) when Mac 2
returns; the ``MLXDistributedBackend`` adapter is byte-for-
byte unchanged.

**The W20 theorem family.** Twelve W20 statements:
**W20-Λ-outside-extension** (proved-empirical n=8 saturated
+ structural sketch); **W20-1** (proved-conditional + proved-
empirical n=80 across 5 seeds × 2 budgets, also n=12);
**W20-2** (proved by inspection + mechanically-checked);
**W20-3** (proved-empirical full programme regression);
**W20-Λ-none** / **W20-Λ-compromised** /
**W20-Λ-joint-deception** (each proved-empirical n=8
saturated); **W20-Λ-real** (proved-conditional + empirical-
research, n=4 × 2 models on Mac-1); **W20-C-LEARNED** /
**W20-C-MULTI-ORACLE** / **W20-C-LIVE-WITH-REGISTRY** /
**W20-C-CROSS-BENCH** (conjectural). Honest scope:
``R-67-OUTSIDE-RESOLVES`` is a *synthetic* regime — the
producer is :class:`IdentityExtractor` AND the oracle is a
deterministic :class:`ServiceGraphOracle`. The W20 closure is
bounded by the same closed-vocabulary discipline that bounds
W19 / W18 / W13. The W20 escape is **partial** by design,
bounded above by oracle integrity (W20-Λ-compromised) and by
joint-N-oracle compromise (W20-Λ-joint-deception).

**SDK version bumped: ``wevra.sdk.v3.20`` → ``wevra.sdk.v3.21``.**

## [3.20] — 2026-04-28 — SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator + deceptive-ambiguity-under-trust benchmark family + W19 family (first capsule-native multi-agent-coordination method to cross the deceptive-ambiguity wall on a regime where bundle-only relational compatibility is structurally insufficient)

*Strictly additive on SDK v3.19. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W19 surface adds
one new dataclass (``W19TrustResult``), one wrapping decoder
(``BundleContradictionDisambiguator``), one canonical-role-for-kind
table (``_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND``), one
asymmetric-witness counter (``_w19_witness_counts``), one
canonical-primary-index helper (``_w19_canonical_primary_index``),
and seven W19 branch constants. Re-exported via the SDK
``__init__``. The new
``vision_mvp/experiments/phase66_deceptive_ambiguity.py`` driver
ships as a research-slice addition with one positive
ratification anchor sub-bank (R-66-CORROBORATED), two strict-win
sub-banks (R-66-DECEIVE-NAIVE under loose AND tight ``T_decoder``,
R-66-CONFOUND-RESOLVABLE), and two named falsifier sub-banks
(R-66-DECEIVE-TOTAL, R-66-OUTSIDE-REQUIRED).*

On the new Phase-66 *deceptive-ambiguity-under-trust* benchmark:

* **R-66-CORROBORATED** (no adversarial signal; corroborating
  witnesses present; **W19-3 ratification anchor**; n=8 ×
  bank_replicates=2): W19 ties the W18
  ``RelationalCompatibilityDisambiguator`` byte-for-byte at
  ``accuracy_full = 1.000``. W19 reduces to W18 when no
  contradicting witnesses exist (asymmetric witness scores are
  uniformly zero ⇒ W18 trust path fires unchanged).
* **R-66-DECEIVE-NAIVE-LOOSE** (round-2 disambiguator
  adversarially mentions DECOY but NOT gold; secondary
  specific-tier witnesses corroborate gold; **W19-1 deceive
  anchor**, ``T_decoder = None``; n=8 × bank_replicates=2): every
  closed-form salience scorer including W18 ties FIFO at
  ``accuracy_full = 0.000`` (W18 trusts the adversarial
  disambiguator and picks decoy; W18-Λ-deceive extends to
  R-66 verbatim by W19-Λ-deceive-extension). The new W19
  ``BundleContradictionDisambiguator`` achieves
  ``capsule_bundle_contradiction = 1.000``. **+1.000 strict
  separation** vs every non-W19 capsule baseline.
* **R-66-DECEIVE-NAIVE-TIGHT** (same regime under decoder-side
  budget pressure ``T_decoder = 24``): same headline as loose.
  The W19 method composes cleanly with the W15 attention-aware
  pack; ``tokens_kept_sum`` is byte-for-byte identical to W18's
  on this regime (bounded-context honesty preserved).
* **R-66-CONFOUND-RESOLVABLE** (round-2 disambiguator mentions
  BOTH gold AND decoy symmetrically — the W18-Λ-confound wall;
  secondary specific-tier witnesses break the tie asymmetrically
  toward gold; **W19-1 confound anchor**, n=8 × bank_replicates=2):
  W18 abstains at 0.000 (symmetric round-2 disambiguator); the
  new W19 method achieves ``capsule_bundle_contradiction = 1.000``.
  **+1.000 strict separation** vs W18 and every other baseline.
* **5-seed stability** on R-66-DECEIVE-NAIVE-LOOSE,
  R-66-DECEIVE-NAIVE-TIGHT, AND R-66-CONFOUND-RESOLVABLE: gap
  ``w19 − w18 = +1.000`` on every seed in
  ``{11, 17, 23, 29, 31}`` (saturated; well above the 0.50
  strong-bar threshold).
* **R-66-DECEIVE-TOTAL** (W19-Λ-total falsifier): no witnesses
  anywhere (round-2 disambiguator adversarial; secondary handoffs
  silent); aw uniformly zero ⇒ W19 reduces to W18 byte-for-byte
  ⇒ W19 ties FIFO at 0.000 on 8/8 cells. Names the structural
  limit no bundle-only closed-form scorer can escape when the
  bundle carries no exonerating evidence at all.
* **R-66-OUTSIDE-REQUIRED** (W19-Λ-outside falsifier): witnesses
  are symmetric across gold and decoy (each side gets the same
  asymmetric witness count); W19's tiebreak is a wash ⇒ W19
  abstains; ties FIFO at 0.000 on 8/8 cells. Names the structural
  limit no bundle-only closed-form scorer can escape when the
  symmetry inside the bundle is total. The named research move
  beyond it is W19-C-OUTSIDE (conjectural; requires
  outside-the-bundle information — a learned scorer, an external
  knowledge base, an extra round of evidence).
* **W19-3 backward-compat**: on R-58 default the W19 method ties
  W18 byte-for-byte on the answer field; on every R-65 default
  bank (compat / no_compat / confound / deceive) W19 ties W18
  byte-for-byte. With ``enabled = False`` the W19 method reduces
  to W18 byte-for-byte.
* **Audit T-1..T-7** OK on every capsule strategy of every cell of
  every regime.

The W19 surface is purely additive on top of the W18 surface (one
new dataclass + one canonical-role table + one asymmetric-witness
counter + one canonical-primary-index helper + one wrapping
decoder + seven branch constants). The SDK v3.19 runtime contract
is byte-for-byte unchanged. New tests cover the W19 unit
semantics, the Phase-66 bench-property witnesses, the W19-1
strict-win anchor on three deceptive regimes, the 5-seed
stability across all three, the two named falsifiers, the
backward-compat smoke (R-58 + every R-65 default bank), the
token-budget honesty, and the cross-regime synthetic summary —
all 555/555 wevra tests pass.

### Added

- **W19 surface** (``vision_mvp/wevra/team_coord.py``):
  - ``BundleContradictionDisambiguator`` (wraps W18; consumes
    asymmetric specific-tier witness counts to either ratify
    W18's verdict or override it when the bundle's witness
    distribution contradicts W18's choice; reduces to W18
    byte-for-byte when no witnesses exist).
  - ``W19TrustResult`` dataclass (decision + branch + witness
    counts per tag + token bookkeeping).
  - ``_w19_witness_counts(...)``, ``_w19_canonical_primary_index(...)``
    helpers + ``_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND`` table
    + ``W19_SYMMETRIC_NOISE_KINDS`` frozenset + seven
    ``W19_BRANCH_*`` constants.
  - Re-exported via ``vision_mvp.wevra.__all__``.
- **Phase-66 driver**
  (``vision_mvp/experiments/phase66_deceptive_ambiguity.py``):
  five-bank synthetic benchmark + cross-regime summary +
  5-seed stability sweep + closed-vocabulary secondary-witness
  routing extension (``_P66_SECONDARY_ROUTES``).
- **Tests** (``vision_mvp/tests/test_wevra_bundle_contradiction.py``):
  45 tests across 10 test classes — unit semantics, bench
  properties, default config, 5-seed stability, two named
  falsifiers, backward-compat, token efficiency, cross-regime,
  invariants.
- **Docs**: ``docs/RESULTS_WEVRA_DECEPTIVE_AMBIGUITY.md`` (new
  milestone note, ~12KB) + R-66 anchor + W19 family in
  ``docs/THEOREM_REGISTRY.md`` + bar 16 in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` +
  ``docs/HOW_NOT_TO_OVERSTATE.md`` (W19 honest-scope sections)
  + ``docs/RESEARCH_STATUS.md`` SDK v3.20 frontier +
  ``docs/context_zero_master_plan.md`` § 4.37 + four data files
  in ``docs/data/phase66_*.json``.

### Discharged conjectures

- **W18-Λ-deceive** (SDK v3.19 falsifier; named limit on bundle-
  relational scorers that *trust* round-2 disambiguator evidence):
  **PARTIALLY DISCHARGED** by W19-1 on R-66-DECEIVE-NAIVE — the
  bundle carries asymmetric secondary witnesses W18 ignored;
  W19's witness counter consumes them. Remaining limit
  W19-Λ-total (no witnesses anywhere) is genuinely beyond
  bundle-only closed-form scorers.
- **W18-Λ-confound** (implicit SDK v3.19 falsifier; W18 abstains
  on symmetric round-2 disambiguator): **PARTIALLY DISCHARGED**
  by W19-1 on R-66-CONFOUND-RESOLVABLE — secondary witnesses
  break the inside-bundle symmetry asymmetrically. Remaining
  limit W19-Λ-outside (symmetric witnesses) is genuinely beyond
  bundle-only closed-form scorers.

### Preserved

- **SDK v3.19 multi-agent surface.** Every fixed admission
  policy, every closed-form salience scorer, every bundle-aware
  decoder, every layered normaliser, every producer protocol,
  every attention-aware pack, every relational-compatibility
  disambiguator — all byte-for-byte unchanged from SDK v3.19.
- **Wevra single-run product report v2 schema:** byte-for-byte
  identical from SDK v3.19.

See [`docs/RESULTS_WEVRA_DECEPTIVE_AMBIGUITY.md`](docs/RESULTS_WEVRA_DECEPTIVE_AMBIGUITY.md)
for the full milestone note.

## [3.19] — 2026-04-28 — SDK v3.19 — bundle-relational compatibility disambiguator + symmetric-ambiguity benchmark family + W18 family (first capsule-native multi-agent-coordination method to cross the symmetric-corroboration wall on a regime where the wall actually applies)

*Strictly additive on SDK v3.18. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W18 surface adds
one new dataclass (``W18CompatibilityResult``), one wrapping
decoder (``RelationalCompatibilityDisambiguator``), one
deterministic tokeniser (``_disambiguator_payload_tokens``), and
one closed-form scorer (``_relational_compatibility_score``) with
contiguous-subsequence semantics for compound targets. Re-exported
via the SDK ``__init__``. The new
``vision_mvp/experiments/phase65_relational_disambiguation.py``
driver ships as a research-slice addition with one positive-
anchor sub-bank (R-65-COMPAT under loose AND tight
``T_decoder``) plus three named falsifier sub-banks
(R-65-NO-COMPAT, R-65-CONFOUND, R-65-DECEIVE).*

On the new Phase-65 *relational-compatibility disambiguation
under symmetric corroboration* benchmark:

* **R-65-COMPAT-LOOSE** (synthetic identity producer, magnitude-
  hinted prompt, ``T_decoder = None``; **W18-1 anchor**; n=8 ×
  bank_replicates=2): every closed-form salience scorer in the
  SDK — substrate FIFO, ``capsule_fifo``, ``capsule_priority``,
  ``capsule_coverage``, W7-2 cohort, W8 corroboration, W9 multi-
  service, W11 multi-round, W12 robust, W13 layered, W15
  ``AttentionAwareBundleDecoder``, ``capsule_layered_fifo_packed``
  — ties FIFO at ``accuracy_full = 0.000`` (W17-Λ-symmetric extends
  to R-65-COMPAT verbatim by W18-Λ-sym). The new W18
  ``RelationalCompatibilityDisambiguator`` achieves
  ``capsule_relational_compat = 1.000``. **+1.000 strict
  separation** vs every non-W18 capsule baseline.
* **R-65-COMPAT-TIGHT** (same regime under decoder-side budget
  pressure ``T_decoder = 24``): same headline as loose. The W18
  method composes cleanly with the W15 attention-aware pack;
  ``tokens_kept_sum`` is byte-for-byte identical to W15's
  (bounded-context honesty preserved).
* **5-seed stability** on R-65-COMPAT-LOOSE AND R-65-COMPAT-TIGHT:
  gap ``w18 − attention_aware = +1.000`` on every seed in
  ``{11, 17, 23, 29, 31}`` (saturated; well above the 0.50
  strong-bar threshold).
* **R-65-NO-COMPAT** (W18-Λ-no-compat falsifier): round-2
  disambiguator carries no service-tag mention; W18 abstains; ties
  FIFO at 0.000 on 8/8 cells.
* **R-65-CONFOUND** (W18-Λ-confound falsifier): round-2
  disambiguator mentions BOTH gold AND decoy; W18 abstains; ties
  FIFO at 0.000 on 8/8 cells.
* **R-65-DECEIVE** (W18-Λ-deceive falsifier): round-2 disambiguator
  mentions DECOY but NOT gold; W18 trusts its evidence and picks
  decoy; fails at 0.000 on 8/8 cells. Names the structural limit
  no closed-form bundle-relational scorer that trusts its evidence
  can escape (the named research move beyond it is W18-C-OUTSIDE,
  conjectural).
* **W18-3 backward-compat**: on R-58 default the W18 method ties
  W15 byte-for-byte on the answer field; on R-64-SYM the W18
  method partially recovers (only deadlock-flavored scenarios
  carry a relational mention). With ``enabled = False`` the W18
  method reduces to W15 byte-for-byte.
* **Audit T-1..T-7** OK on every capsule strategy of every cell of
  every regime.

The W18 surface is purely additive on top of the W15 surface (one
new dataclass + one tokeniser + one closed-form scorer + one
wrapping decoder). The SDK v3.18 runtime contract is byte-for-byte
unchanged. New tests cover the W18 unit semantics, the Phase-65
bench-property witnesses, the W18-1 strict-win anchor on loose
AND tight, the 5-seed stability, the three named falsifiers, the
backward-compat smoke, the token-budget honesty, and the
cross-regime synthetic summary.

See [`docs/RESULTS_WEVRA_RELATIONAL_DISAMBIGUATOR.md`](docs/RESULTS_WEVRA_RELATIONAL_DISAMBIGUATOR.md)
for the full milestone note.

## [3.18] — 2026-04-27 — SDK v3.18 — magnitude-hinted producer protocol + fresh-live end-to-end composition + symmetric-corroboration limit theorem (first fresh-live end-to-end real-LLM strict +1.000 advance + first explicit symmetric-corroboration negative theorem) + W17 family

*Strictly additive on SDK v3.17. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The W17 surface adds
one new producer-prompt mode constant
(``PRODUCER_PROMPT_MAGNITUDE_HINTED``), one new dataclass
(``OperationalThreshold``), one new optional field on
``RoleExtractionSchema`` (``magnitude_thresholds``), one new
prompt-render helper (``_render_magnitude_hinted_prompt``), one
threshold-table data structure
(``INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS``), and an
optional ``magnitude_hinted=True`` parameter on
``incident_triage_role_schemas``. Re-exported via the SDK
``__init__``. The new
``vision_mvp/experiments/phase64_live_composition.py`` driver
ships as a research-slice addition with five synthetic + four
live + two symmetric-wall sub-banks.*

On the new Phase-64 *fresh-live end-to-end W14H + W15 composition
+ symmetric-corroboration wall* benchmark:

* **R-64-LIVE-MAGHINT** (fresh live ``qwen2.5:14b-32k`` Mac-1
  Ollama backend + magnitude-hinted prompt + ``T_decoder=14``;
  **W17-1 anchor**; n=8 × 24 producer calls; 0 endpoint
  failures; 128.2 s wall): pairing the W17 magnitude-hinted
  producer protocol with the W14 producer + W15 decoder pipeline
  achieves ``capsule_attention_aware = 1.000`` AND
  ``capsule_layered_fifo_packed = 0.000`` AND
  ``capsule_fifo = 0.000``. **+1.000 strict separation** on both
  axes. Bench property holds in **8/8** scenarios — closing the
  1/8 R-61-OLLAMA-A model-side judgment miss
  (``slow_query_archival``) that persisted under the SDK v3.15
  W14 anchor and the SDK v3.17 W16-Λ-real-replay anchor. The
  first programme result that beats the strongest non-composed
  baseline by ≥ 1.0 on a *fresh* live LLM probe.
* **R-64-LIVE-STRUCT** (fresh live 14B + legacy structured prompt;
  W17-Λ-no-hint anchor): bench property holds in 7/8;
  ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``; +0.500 strict gain.
  Reproduces the W14-Λ-real / W16-Λ-real-replay envelope on a
  fresh probe; the magnitude-hint extension, not a re-run, is the
  load-bearing improvement.
* **R-64-LIVE-NAIVE** (fresh live 14B + naive prompt;
  W17-Λ-naive falsifier): bench property holds in 0/8; every
  capsule strategy ties FIFO at 0.000. Live counterpart of
  W14-Λ-prompt + W15-Λ-budget joint failure.
* **R-64-LIVE-XMODEL** (fresh live ``qwen3.5:35b`` MoE backend +
  magnitude-hinted prompt + ``think=False``; W17-C-XMODEL probe;
  n=8 × 24 producer calls; 0 failures; 92.0 s wall): bench
  property holds in **8/8** (the W17 magnitude-hint extension
  transfers byte-for-byte across the 14B → 36B-MoE jump on the
  bench-property axis); ``capsule_attention_aware = 0.750``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.750 strict gain**
  (well above the 0.50 strong-bar threshold). The 0.250 gap to
  1.000 is on ``accuracy_root_cause`` — a 35B-specific specific-
  tier kind judgment artifact, not a producer-protocol failure.
  Proved-conditional + empirical-research; partially discharges
  W16-C-CROSS-MODEL.
* **R-64-SYM** (synthetic ``build_phase64_sym_bank``;
  **W17-Λ-symmetric anchor**; n=8 × {None, 24} budget): every
  capsule strategy in the SDK ties FIFO at ``accuracy_full =
  0.000`` by construction. **The first explicit symmetric-
  corroboration limit theorem in the programme.** Discharges
  W15-C-SYMMETRIC / W16-C-SYMMETRIC as a negative theorem.

W17 theorem family: **W17-1** (proved-conditional + empirical-
research), **W17-Λ-no-hint** (empirical-research),
**W17-Λ-naive** (empirical-research), **W17-Λ-symmetric**
(proved-empirical + structural sketch), **W17-2** (proved by
inspection + mechanically-checked), **W17-3** (proved-empirical
full programme regression), **W17-C-XMODEL** (proved-conditional
+ empirical-research). The W17-C family (W17-C-DISAMBIGUATOR,
W17-C-LEARNED-HINT, W17-C-CROSS-BENCH) makes the next research
frontier explicit.

Discharged conjectures from prior SDKs:

* **W16-C-LIVE-OLLAMA** → DISCHARGED (W17-1 closes the 1/8 miss
  on a fresh live probe).
* **W16-C-CROSS-MODEL** → PARTIALLY DISCHARGED (W17-C-XMODEL on
  Ollama; MLX-distributed clause remains conjectural).
* **W15-C-SYMMETRIC** / **W16-C-SYMMETRIC** → DISCHARGED-NEGATIVE
  (W17-Λ-symmetric).

Backward-compat (W17-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``mode='naive'`` or ``mode='structured'`` AND
``magnitude_hinted_schema=False``, the W17 surface reduces to the
SDK v3.15 W14 anchor byte-for-byte. The Wevra single-run product
runtime contract is byte-for-byte unchanged.

Added: ``vision_mvp/experiments/phase64_live_composition.py``,
``vision_mvp/tests/test_wevra_phase64.py``, the W17 protocol
surface in ``vision_mvp/wevra/team_coord.py``,
``docs/RESULTS_WEVRA_LIVE_COMPOSITION.md``, R-64 anchor +
bar 14 in ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``,
W17 family in ``docs/THEOREM_REGISTRY.md``, current-frontier
update in ``docs/RESEARCH_STATUS.md``, § 4.35 in
``docs/context_zero_master_plan.md``, latest-milestone pointer
in ``docs/START_HERE.md``, synthesis-after-v3.18 reading in
``papers/context_as_objects.md``, and the milestone artefacts
in ``docs/data/``
(``phase64_cross_regime_synthetic.json``,
``phase64_live_maghint_qwen2_5_14b_n8.json``,
``phase64_live_struct_qwen2_5_14b_n8.json``,
``phase64_live_naive_qwen2_5_14b_n8.json``,
``phase64_live_maghint_qwen3_5_35b_n8.json``).

Changed: ``vision_mvp/wevra/__init__.py`` (SDK version bumped to
``wevra.sdk.v3.18``; W17 surface re-exports);
``vision_mvp/experiments/phase61_producer_ambiguity_preservation.py``
(``CapturingOllamaExtractor.extract_round`` now treats
``PRODUCER_PROMPT_MAGNITUDE_HINTED`` as a structured-prompt
variant on the parser-dispatch axis — load-bearing fix because
the per-event mandate would otherwise lose its ``(kind, payload)``
dedup);
``vision_mvp/tests/test_wevra_producer_ambiguity.py``
(``test_all_modes_listed`` updated for the additive third mode).

## [3.17] — 2026-04-27 — SDK v3.17 — end-to-end W14 + W15 composition (first end-to-end real-LLM strict advance over the strongest non-composed baseline) + W16 family

*Strictly additive on SDK v3.16. The Wevra single-run product
runtime contract is byte-for-byte unchanged. **No new SDK class
ships** — W16 is the demonstration that the existing W14
``StructuredProducerProtocol`` and W15
``AttentionAwareBundleDecoder`` compose on a single regime where
both are independently load-bearing. The new
``vision_mvp/experiments/phase63_composed_real_llm.py`` driver
plus the ``OllamaReplayExtractor`` ship as research-slice
additions; the runtime contract is unchanged.

On the new Phase-63 *composed end-to-end W14 + W15* benchmark:

* **R-63-COMPOSED-TIGHT** (mag-filter structured, ``T_decoder=24``;
  **W16-1 anchor**, n=8 × 5 seeds saturated): pairing the W14
  ``StructuredProducerProtocol`` + W15
  ``AttentionAwareBundleDecoder`` simultaneously achieves
  ``accuracy_full = 1.000`` while every non-composed baseline
  collapses to 0.000 (W14-only-budgeted, W15-only-without-W14,
  substrate FIFO). **+1.000 strict separation, stable across 5/5
  alternate ``bank_seed`` values**.
* **R-63-naive-tight** (mag-filter naive, ``T_decoder=24``;
  **W16-Λ-compose anchor**): joint failure of W14-Λ-prompt and
  W15-Λ-budget on the same regime; every capsule strategy ties
  FIFO at 0.000.
* **R-63-degen-budget** (mag-filter structured, ``T_decoder=2``;
  **W16-Λ-degenerate falsifier**): both packers collapse to
  0.000; the W16-1 win is conditional on a budget that admits
  *some* of the union.
* **R-63-OLLAMA-REPLAY-COMPOSED-TIGHT** (replay over recorded
  Phase-61 ``qwen2.5:14b-32k`` bytes at ``T_decoder=14,
  K_auditor=8``; **W16-Λ-real-replay anchor**):
  ``capsule_attention_aware = 0.500`` while
  ``capsule_layered_fifo_packed = 0.000`` — **+0.500 strict
  gain** over the FIFO-packed-W14-only baseline on a *real-LLM
  stream*. The first end-to-end real-LLM strict advance over the
  strongest non-composed baseline in the programme.

The W16 layer is the eighth structural move in the Wevra programme:

| Layer                            | SDK   | Theorem | Anchor regime                                |
|----------------------------------|-------|---------|----------------------------------------------|
| Admission (cohort coherence)     | v3.8  | W7-2    | R-54                                         |
| Admission (cross-role corrob.)   | v3.9  | W8-1    | R-55                                         |
| Admission (multi-service)        | v3.10 | W9-1    | R-56                                         |
| Decoding (intra-round bundle)    | v3.11 | W10-1   | R-57                                         |
| Decoding (cross-round bundle)    | v3.12 | W11-1   | R-58                                         |
| Normalisation (fixed-vocabulary) | v3.13 | W12-1   | R-59                                         |
| Normalisation (open-world)       | v3.14 | W13-1   | R-60-wide                                    |
| Producer protocol                | v3.15 | W14-1   | R-61 + R-61-OLLAMA-A                         |
| Decoder context packing          | v3.16 | W15-1   | R-62-tightbudget                             |
| **End-to-end composition**       | v3.17 | **W16-1** | **R-63-COMPOSED-TIGHT + W16-Λ-real-replay** |

W16 family theorems minted by this milestone:

* **W16-Λ-compose** (proved-empirical + structural sketch via
  composition of W14-Λ-prompt and W15-Λ-budget).
* **W16-1** (proved-conditional + proved-empirical synthetic n=40 ×
  5 seeds).
* **W16-2** (proved-empirical, multiplicative composition).
* **W16-3** (proved-empirical, full programme regression: **442/442
  prior tests pass byte-for-byte**; 22 new tests cover the W16
  surface).
* **W16-Λ-degenerate** (proved-empirical falsifier).
* **W16-Λ-real-replay** (empirical-research, recorded real-LLM
  bytes).

W15-C-COMPOSE-W14 (SDK v3.16 conjecture) is **PARTIALLY DISCHARGED**
by W16-Λ-real-replay: the composition delivers a strict +0.500 gain
on recorded ``qwen2.5:14b-32k`` bytes (recovering the W14-only
loose-budget accuracy under tight budget pressure) but does NOT
close the 1/8 model-side judgment failure that W14 alone leaves
on the same recorded capture.

Honest scope: the Mac-1 endpoint at 192.168.12.191:11434 was
offline at milestone capture time (``HTTP=000``). A fresh live
LLM probe (W16-C-LIVE-OLLAMA) is conjectural; cross-model
transfer (W16-C-CROSS-MODEL) and cross-bench transfer (W16-C1)
are conjectural. See
[`docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md`](docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md)
for the full milestone note.

## [3.16] — 2026-04-27 — SDK v3.16 — attention-aware capsule context packing (first decoder-side context-packing move + first joint-correctness-and-context-efficiency strict-gain anchor) + W15 family

*Strictly additive on SDK v3.15. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``AttentionAwareBundleDecoder``, ``CapsuleContextPacker``, and
``FifoContextPacker`` ship in ``vision_mvp.wevra.team_coord`` as
research-slice additions to the multi-agent coordination layer, not
part of the run-boundary product runtime. **First SDK milestone
where joint correctness AND decoder-side context efficiency is the
strict-gain axis** of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (the new bar 12).

On the new Phase-62 *attention-aware capsule context packing* benchmark:

* **R-62-default** (synthetic, ``T_decoder=None``): bench property
  (multi-hypothesis comparable-magnitude + asymmetric corroboration:
  every decoy ≥ 2 distinct roles, every gold = 1 distinct role)
  holds 8/8; W11/W12/W13/W15 cross-round decoders all achieve
  ``accuracy_full = 1.000``. The W15 decoder ties W13 byte-for-byte
  on the answer field (W15-3 backward-compat anchor).
* **R-62-tightbudget** (synthetic, ``T_decoder=24`` strict; W15-1
  anchor): under FIFO packing the round-2 specific-tier
  disambiguator falls past the budget by construction
  (``position_of_first_causal_claim_avg = -1.0`` in 8/8 cells);
  ``capsule_layered_fifo_packed`` ties FIFO at
  ``accuracy_full = 0.000``. The W15 ``AttentionAwareBundleDecoder``
  salience-packs the union with hypothesis preservation and puts
  the round-2 specific claim at rank 0 in 8/8 cells; achieves
  ``accuracy_full = 1.000``. **+1.000 strict separation vs FIFO-
  packed-W13, stable across 5/5 alternate ``bank_seed`` values**.
* **R-62-saturation** (synthetic, ``T_decoder=9999`` effectively
  unbounded; W15-Λ-budget falsifier): under no decoder-side budget
  pressure both packers tie at 1.000. The W15-1 win is *conditional*
  on budget pressure — this regime makes the conditionality sharp.

The W15 layer is the seventh structural axis of the Wevra programme:

| Layer                                | SDK   | Theorem | Anchor regime    |
|--------------------------------------|-------|---------|------------------|
| Admission (cohort coherence)         | v3.8  | W7-2    | R-54             |
| Admission (cross-role corrob.)       | v3.9  | W8-1    | R-55             |
| Admission (multi-service)            | v3.10 | W9-1    | R-56             |
| Decoding (intra-round bundle)        | v3.11 | W10-1   | R-57             |
| Decoding (cross-round bundle)        | v3.12 | W11-1   | R-58             |
| Normalisation (fixed-vocabulary)     | v3.13 | W12-1   | R-59             |
| Normalisation (open-world)           | v3.14 | W13-1   | R-60-wide        |
| Producer protocol                    | v3.15 | W14-1   | R-61 + R-61-OLLAMA-A |
| **Decoder context packing**          | v3.16 | **W15-1** | **R-62-tightbudget** |

The W14 layer (SDK v3.15) closed the producer-side gap; SDK v3.16
attacks the symmetric *downstream* gap directly. The W15 packer's
salience score is closed-form deterministic (tier + CCK +
corroboration + magnitude + round) with pre-committed weight
defaults; per-(tag, role, tier) hypothesis preservation guarantees
multi-hypothesis multi-role evidence survives the pack so the W11
contradiction-aware drop fires correctly.

**Token / context / attention measurement (Part E of the milestone
brief).** Pack-stats expose ``position_of_first_causal_claim`` (the
proxy attention metric — rank 0 in 8/8 W15 cells, −1 in 8/8 FIFO-
pack cells), ``tokens_kept_sum`` / ``tokens_input_sum`` (84.6 % vs
87.3 %), ``hypothesis_count_kept`` (4/4 in both packers), and
``n_dropped_budget`` for direct audit. Token reduction is not the
goal — *causal-evidence concentration in early prompt positions* is.
The proxy attention metric is auditable; we do NOT claim transformer
attention manipulation. The W15-1 win is conditional on (a) the
bench property holding, (b) ``T_decoder`` < admitted-union token
sum, AND (c) round-2 carrying a specific-tier disambiguator with
no ``service=`` token; W15-Λ-degenerate makes the conditionality
sharp.

Backward-compatible on R-54 / R-55 / R-56 / R-57 / R-58 / R-59 /
R-60 / R-61 (default + falsifier). 393/393 prior wevra tests pass
byte-for-byte; 37 new tests cover the W15 surface, hypothesis
preservation, FIFO packer, backward-compat with W13, Phase-62 bank
shape, default config (W15-1 anchor), 5-seed stability, and cross-
regime separation. The wevra suite totals 430/430 passing.

Honest scope: SDK v3.16 is a *synthetic* milestone — the producer
is the deterministic ``IdentityExtractor``, not a real LLM. Real-LLM
transfer of W15 is W15-C-real, conjectural; it requires Mac 1 or
Mac 2 to be online and the bundle to be re-decoded by an LLM agent
under a real context window. SDK v3.16 does not run this probe.
"Attention-aware" uses an *honest proxy* metric — the
``position_of_first_causal_claim`` rank in the salience-ordered
pack — not transformer attention manipulation. Composition with
W14 on a real-Ollama stream (W15-C-COMPOSE-W14, conjectural) is
the natural next probe.

Public surface (additive):

* :class:`vision_mvp.wevra.team_coord.AttentionAwareBundleDecoder` —
  two-stage decoder: first-pass priority decode → salience-aware
  repack → final W13 layered decode.
* :class:`vision_mvp.wevra.team_coord.CapsuleContextPacker` — closed-
  form salience pack with hypothesis preservation.
* :class:`vision_mvp.wevra.team_coord.FifoContextPacker` — load-
  bearing baseline (FIFO truncation under the same ``T_decoder``).
* :class:`vision_mvp.wevra.team_coord.W15PackResult`,
  :class:`vision_mvp.wevra.team_coord.W15PackedHandoff` — pack-stats
  surface.
* ``W15_DEFAULT_TIER_WEIGHT``, ``W15_DEFAULT_CCK_WEIGHT``,
  ``W15_DEFAULT_CORROBORATION_WEIGHT``,
  ``W15_DEFAULT_MAGNITUDE_WEIGHT``,
  ``W15_DEFAULT_ROUND_WEIGHT`` — pre-committed salience weights.

New experiment driver:
``vision_mvp.experiments.phase62_attention_aware_packing``.

Cross-references:
* Bench: ``vision_mvp/experiments/phase62_attention_aware_packing.py``
* Method: ``vision_mvp/wevra/team_coord.py``
* Tests: ``vision_mvp/tests/test_wevra_attention_aware.py`` (37 tests)
* Milestone note: ``docs/RESULTS_WEVRA_ATTENTION_AWARE.md``
* Success criterion: ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``
  (R-62 anchor + bar 12 + § 2.11)
* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W15 family)
* Master plan: ``docs/context_zero_master_plan.md`` § 4.33
* Data: ``docs/data/phase62_*.json``

## [3.15] — 2026-04-27 — SDK v3.15 — structured producer protocol (first producer-protocol move + first real-LLM strict gain ≥ 0.50 over substrate FIFO) + W14 family

*Strictly additive on SDK v3.14. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``StructuredProducerProtocol`` + ``RoleExtractionSchema`` ship in
``vision_mvp.wevra.team_coord`` as research-slice additions to the
multi-agent coordination layer, not part of the run-boundary
product runtime. **First SDK milestone to clear the R-61-OLLAMA-A
tier of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.5** —
the **first capsule-native multi-agent coordination method that
strictly improves over substrate FIFO on a real-LLM stream by ≥
0.50 pooled accuracy_full in the programme**.

On the new Phase-61 *producer-side ambiguity-preservation* benchmark:

* **R-61-default** (synthetic, identity extractor, naive prompt):
  bench property holds 8/8; W11/W12/W13 cross-round decoders all
  achieve ``accuracy_full = 1.000``. Sanity anchor.
* **R-61-naive-prompt** (synthetic, magnitude-filter, naive prompt;
  W14-Λ-prompt anchor): the magnitude-filter extractor calibrated
  against the W13-Λ-real real-Ollama observation collapses round-1
  by top-N-per-(role, kind) by magnitude; bench property holds 0/8;
  every capsule strategy ties FIFO at 0.000. The synthetic
  counterpart of W13-Λ-real, mechanically tractable in CI.
* **R-61-structured-prompt** (synthetic, magnitude-filter,
  structured prompt; W14-1 anchor): the structured prompt's per-
  event mandate disables the compression; bench property holds 8/8;
  W11/W12/W13 achieve ``accuracy_full = 1.000``; +1.000 strict
  separation vs naive-prompt counterpart, stable across **5/5**
  alternate ``bank_seed`` values.
* **R-61-ollama-naive** (real Mac-1 ``qwen2.5:14b-32k`` at
  ``temperature=0`` on the redesigned events under the *naive*
  prompt; W14-Λ-real-naive falsifier): bench property holds 0/8;
  every method ties FIFO at 0.000 — the W14-Λ-prompt prediction
  *empirically confirmed* on real Ollama.
* **R-61-ollama-structured** (real Mac-1 ``qwen2.5:14b-32k`` at
  ``temperature=0`` on the redesigned events under the *structured*
  prompt; W14-Λ-real anchor at the R-61-OLLAMA-A tier): bench
  property holds **7/8**; W11/W12/W13 cross-round decoders all
  achieve ``accuracy_full = 0.500``; ``layered − fifo = +0.500`` at
  exactly the 0.50 threshold; audit T-1..T-7 preserved on every
  cell. n_eval=8 × 24 producer calls, 0 endpoint failures, 111.4 s
  wall on Mac 1.

The W13 closure-widening (SDK v3.14) is structurally invisible on
R-61-ollama because the real LLM emits canonical kinds (zero kind
drift); the load-bearing layer on this regime is the W14 producer
protocol, not the W13 normaliser. The W13 layer is dormant on this
regime, not refuted; it remains the load-bearing layer on
R-60-wide.

Backward-compatible on R-54 / R-55 / R-56 / R-57 / R-58 / R-59 /
R-60 (default + falsifier). 393/393 prior wevra tests pass byte-for-
byte. Named falsifier (W14-4: real Ollama + comparable-magnitude
events + naive prompt) ties FIFO at 0.000 on 8/8 — *both* the
event redesign AND the structured prompt are required for W14-1.

**Files added.**

* ``vision_mvp/wevra/team_coord.py`` — adds
  ``RoleExtractionSchema``, ``ProducerPromptResult``,
  ``StructuredProducerProtocol``, ``PRODUCER_PROMPT_NAIVE``,
  ``PRODUCER_PROMPT_STRUCTURED``, ``ALL_PRODUCER_PROMPT_MODES``,
  ``INCIDENT_TRIAGE_OBSERVATION_KINDS``,
  ``incident_triage_role_schemas``.
* ``vision_mvp/wevra/__init__.py`` — re-exports the W14 surface;
  bumps ``SDK_VERSION = "wevra.sdk.v3.15"``.
* ``vision_mvp/experiments/phase61_producer_ambiguity_preservation.py``
  — new benchmark.
* ``vision_mvp/tests/test_wevra_producer_ambiguity.py`` — 27 new
  tests across schema soundness, protocol determinism, magnitude-
  filter calibration, Phase-61 default config (W14-Λ-prompt +
  W14-1), 5-seed stability, and cross-regime separation.
* ``docs/data/phase61_default_K8_n8.json``,
  ``docs/data/phase61_naive_prompt_K8_n8.json``,
  ``docs/data/phase61_structured_prompt_K8_n8.json``,
  ``docs/data/phase61_seed_sweep_naive_K8_n8.json``,
  ``docs/data/phase61_seed_sweep_structured_K8_n8.json``,
  ``docs/data/phase61_cross_regime.json``,
  ``docs/data/phase61_cross_regime_full.json``,
  ``docs/data/phase61_real_ollama_naive_qwen2_5_14b_n4.json``,
  ``docs/data/phase61_real_ollama_naive_qwen2_5_14b_n8.json``,
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n4.json``,
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``.
* ``docs/RESULTS_WEVRA_PRODUCER_AMBIGUITY.md`` — milestone results
  note (theory-forward, six-layer thesis, W14 theorem family, W14-C
  conjectures).

**Files updated.**

* ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` — R-61 anchor +
  bar 11 (producer-side ambiguity-preservation split) + § 1.5
  R-61-OLLAMA 4-tier grading + § 2.10 R-61 ingredients.
* ``docs/RESEARCH_STATUS.md`` — eleven-axis status, W14 active
  moves / observations / conjectures.
* ``docs/THEOREM_REGISTRY.md`` — W14 family (Λ-prompt, 1, 2, 3, 4,
  Λ-real) + W14-C conjectures + W13-C3 partial-discharge update.
* ``docs/HOW_NOT_TO_OVERSTATE.md`` — W14 framing rules ("solved
  real-LLM transfer" forbidden, "W14 makes W13 obsolete"
  forbidden).
* ``docs/context_zero_master_plan.md`` — § 4.32 SDK v3.15 + post-
  v3.15 master-plan reading + six-layer transfer story.
* ``docs/START_HERE.md`` — current-milestone pointer + SDK v3.15
  TL;DR paragraph.

## [3.13] — 2026-04-26 — SDK v3.13 — real-LLM-robust multi-round bundle decoder (first synthetic→real-LLM transfer move) + W12 family

*Strictly additive on SDK v3.12. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``RobustMultiRoundBundleDecoder`` and the closed-vocabulary
``CLAIM_KIND_SYNONYMS`` table are research-slice additions to the
multi-agent coordination layer (``vision_mvp.wevra.team_coord``),
not part of the run-boundary product runtime. **First SDK milestone
to clear the strong success bar of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-59 anchor
+ new bar 9 — synthetic→real-LLM transfer split)** — the **first
capsule-native multi-agent coordination method that survives a
calibrated bounded-producer-noise channel**. On the new Phase-59
*real-LLM-driven multi-round delayed-disambiguation* regime
(``synonym_prob=0.50, svc_token_alt_prob=0.30`` synthetic-noisy-LLM
extractor), every un-normalised capsule strategy in the SDK —
substrate, FIFO, priority, coverage, W7-2 cohort, W8 corroboration,
W9 multi-service, W10 single-round bundle decoder, AND **SDK v3.12
W11 ``MultiRoundBundleDecoder``** — ties FIFO at
``accuracy_full = 0.000`` (the W12-Λ un-normalised structural
limit at the real-LLM axis); the new
``RobustMultiRoundBundleDecoder`` (closed-vocabulary
``CLAIM_KIND_SYNONYMS`` + payload-rewrite layer ahead of W11)
achieves ``accuracy_full = 1.000`` (W12-1 sufficiency under
bounded LLM noise). Headline gap = +1.000 vs every un-normalised
method including W11; ``robust = 1.000`` on **5/5** alternate
(bank_seed, llm_seed) values, ``robust − w11`` min = 0.750
(seed 23), max = 1.000 (seeds 11, 29), well above the strong-bar
0.20 threshold on every seed.
Backward-compatible on R-54 / R-55 / R-56 / R-57 / R-58 and on
R-59 with ``llm_mode='synthetic_clean_llm'`` (rewrite counters =
0, ties W11 byte-for-byte). Named falsifier (W12-4: out-of-
vocabulary kinds outside the synonym closure) ties FIFO at 0.000.
Audit T-1..T-7 OK on every cell of every R-59 capsule strategy.

**Files added.**

* ``vision_mvp/wevra/team_coord.py`` — adds
  ``RobustMultiRoundBundleDecoder``, ``CLAIM_KIND_SYNONYMS``,
  ``_SERVICE_TAG_REWRITES``, ``normalize_claim_kind``,
  ``normalize_payload``, ``normalize_handoff``.
* ``vision_mvp/wevra/__init__.py`` — re-exports the W12 surface;
  bumps ``SDK_VERSION = "wevra.sdk.v3.13"``.
* ``vision_mvp/experiments/phase59_real_llm_multi_round.py`` — new
  benchmark.
* ``vision_mvp/tests/test_wevra_real_llm_multi_round.py`` — 24 new
  tests across normalisation, decoder semantics, bench property,
  default config (W12-Λ + W12-1), falsifier (W12-4), backward-compat
  (W12-3), and 5-seed stability.
* ``docs/data/phase59_default_K8_n12.json``,
  ``docs/data/phase59_falsifier_K8_n8.json``,
  ``docs/data/phase59_clean_K8_n8.json``,
  ``docs/data/phase59_seed_sweep_K8_n12.json``,
  ``docs/data/phase59_cross_regime.json``.
* ``docs/RESULTS_WEVRA_REAL_LLM_MULTI_ROUND.md`` — milestone note.
* ``docs/RESEARCH_STATUS.md``, ``docs/THEOREM_REGISTRY.md``,
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``,
  ``docs/HOW_NOT_TO_OVERSTATE.md``,
  ``docs/context_zero_master_plan.md``,
  ``docs/START_HERE.md``,
  ``papers/wevra_capsule_native_runtime.md``,
  ``CHANGELOG.md`` updated for the W12 family.

**Honest scope.** The win is *conditional* on (a) the R-58
delayed-causal-evidence bench shape, (b) the producer-noise
channel being bounded by the closed-vocabulary closure (every
variant in ``NOISY_KIND_VARIANTS`` is in ``CLAIM_KIND_SYNONYMS``),
AND (c) round-N admission not being budget-starved (inherits
W11-4). The synthetic-noisy-LLM extractor is calibrated against
Phase-53 14B/35B empirical kind-drift distributions; the
``--llm-mode ollama`` opt-in mode is the W12-C2 next data point
and the natural Mac-2-returns probe.

## [3.11] — 2026-04-26 — SDK v3.11 — bundle-aware team decoder (first decoder-side coordination move) + W10 family

*Strictly additive on SDK v3.10. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``BundleAwareTeamDecoder`` and the closed-vocabulary
``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`` table are research-slice
additions to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime. **Third consecutive SDK milestone to clear the
strong success bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (R-57 anchor + new bar 7 — admission/decoding split)** — the
**first capsule-native multi-agent coordination method that
crosses the admission/decoding split**. On the new Phase-57
*decoder-forcing* regime, every service-blind admission policy in
the SDK (FIFO, priority, coverage, W7-2, W8, W9) ties FIFO at
``accuracy_full = 0.000`` (the W10-Λ admission-only structural
limit); pairing W9 admission with the new bundle decoder achieves
``accuracy_full = 1.000`` (W10-1 sufficiency). Headline gap =
**+1.000** vs every admission baseline, stable across **5/5**
alternate bank seeds. Backward-compatible at 1.000 on R-54 / R-55
/ R-56 (W10-3 via the trust-admission fallback). Named falsifier
regime W10-4 (decoy CCK-promotion) correctly ties FIFO at 0.000.*

### Added

- **Phase-57 multi-service-gold + corroborated-decoy decoder-forcing
  benchmark** (new):
  `vision_mvp/experiments/phase57_decoder_forcing.py`. Eight
  strategies including the new ``capsule_bundle_decoder`` (W10-AD:
  W9 admission + bundle decoder) on a deterministic 12-scenario
  bank where every scenario has multi-service gold corroborated via
  causal claim_kinds AND a decoy corroborated via *non-causal*
  generic-noise claim_kinds. 10/10 default scenarios satisfy the
  bench property; mechanically verified by ``Phase57BankShapeTests``.
- **`BundleAwareTeamDecoder` (new)**:
  `vision_mvp/wevra/team_coord.py`. Three knobs (``cck_filter``,
  ``role_corroboration_floor``,
  ``fallback_admitted_size_threshold``). Re-exported as
  ``BundleAwareTeamDecoder``, ``decode_admitted_role_view``, and
  ``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE``.
- **Theorem family W10**: W10-Λ, W10-1, W10-2, W10-3, W10-4
  (proved or proved-empirical); W10-C1/C2/C3 (conjectural).
- **R-57 anchor + bar 7** added to
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`.
- **`docs/RESULTS_WEVRA_BUNDLE_DECODER.md`** (new): theory-forward
  results note.
- 21 new tests in `vision_mvp/tests/test_wevra_bundle_decoder.py`.

### Changed

- ``SDK_VERSION = "wevra.sdk.v3.11"``.
- ``docs/RESEARCH_STATUS.md``, ``docs/THEOREM_REGISTRY.md``,
  ``docs/context_zero_master_plan.md`` § 4.28 updated for SDK v3.11
  / W10 family.

### Discharged conjectures

- **W9-C1** (SDK v3.10): bundle-aware decoder companion.
  **DISCHARGED-empirical** by W10-1 on the Phase-57 decoder-forcing
  regime.

## [3.10] — 2026-04-26 — SDK v3.10 — multi-service top-K cross-role corroboration multi-agent coordination + W9 family

*Strictly additive on SDK v3.9. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``MultiServiceCorroborationAdmissionPolicy`` is a research-slice
addition to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime. **Second consecutive SDK milestone to clear the
strong success bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (R-56 anchor)** — strict separation from W8 on Phase 56
(+1.000 multi_service − corroboration on accuracy_full,
+1.000 vs FIFO and W7-2 too), backward-compat on Phase 55
(W9 ties W8 at 1.000 via the argmax-by-role-count gate),
backward-compat on Phase 54 (W9 ties W7-2 at 1.000),
no regression on Phase 53 synthetic (0.800), cross-bank stability
across 5/5 seeds, named falsifier regime (W9-4) correctly ties
FIFO at 0.000.  **First programme result whose strict-gain regime
is not solvable by the previous SDK's strongest method.***

### Added

- **Phase-56 multi-service-gold + cross-role-corroborated benchmark**
  (new): `vision_mvp/experiments/phase56_multi_service_corroboration.py`.
  Smallest deterministic regime where (a) every scenario has
  `gold_services` of size 2 (multi-service incident), (b) both gold
  services are corroborated by ≥ 2 distinct producer roles, (c) at
  least one decoy service has raw plurality but is corroborated by
  exactly 1 producer role. 5 base scenario builders × 2 replicates
  → 10-scenario default bank; named falsifier bank promotes a
  decoy to ≥ 2 distinct producer roles (W9-4 anchor).
- **`MultiServiceCorroborationAdmissionPolicy`** (new): in
  `vision_mvp/wevra/team_coord.py`. Deterministic, training-free
  admission rule that admits the **top-K cross-role-corroborated
  tier** (default `top_k=2, min_corroborated_roles=2`) via the
  argmax-by-role-count gate. Strictly generalises the SDK v3.9
  W8 single-tag corroboration policy (W9-3 backward-compat).
  Buffered factory `from_candidate_stream` is the W9-1 anchor.
  Re-exported as `TeamMultiServiceCorroborationAdmissionPolicy`.
- **`_dominant_tag_set`** helper (new): pure function with three
  structural properties (W9-2): single-role exclusion;
  argmax-tier collapse; argmax-tier multi-tag admission within
  `top_k` cap.
- **W9 theorem family** (new): W9-1 strict separation, W9-2
  argmax-tier strict-ordering, W9-3 backward-compat with W8
  + W7-2, W9-4 decoy-corroboration falsifier — all proved or
  proved-empirical on the pre-committed Phase-56 default. W9-C1
  / W9-C2 / W9-C3 conjectures (bundle-aware decoder, |gold|≥3,
  real-LLM transfer).
- **36 contract tests** in `test_wevra_multi_service_corroboration.py`:
  policy unit tests, bank shape, default config win, seed stability,
  falsifier behaviour, W9-3 backward-compat with Phase 55, audit
  invariance, cross-regime contract, public-API contract.
- **`docs/RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md`** (new):
  milestone results note with W9 family theorem statements.
- **Frozen artefacts** in `docs/data/`: `phase56_multi_service_K4_n10.json`,
  `phase56_falsifier_K4_n10.json`, `phase56_seed_sweep.json`,
  `phase56_cross_regime.json`,
  `phase53_synthetic_w9_regression_check.json`.

### Changed

- **`SDK_VERSION`** bumped from `wevra.sdk.v3.9` to `wevra.sdk.v3.10`.
- **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** — bar anchor
  advanced to R-56; R-56 named regime added with mechanical-witness
  ingredient list; falsifying-failure list extended to gate W8-1
  contract test; canonical phrasing for SDK v3.10 added.
- **`docs/THEOREM_REGISTRY.md`** — W9-1/W9-2/W9-3/W9-4 + W9-C1/C2/C3
  added; W8-C1 marked DISCHARGED; date stamp v3.10.
- **`docs/RESEARCH_STATUS.md`** — ninth research axis (multi-service
  top-K corroboration) added; SDK v3.10 frontier section.
- **`docs/HOW_NOT_TO_OVERSTATE.md`** — W9 overstatement guards
  added (W9-1 conditionality, W8 multi-service-gold falsifier
  named, "we solved multi-agent context" still forbidden).
- **`docs/context_zero_master_plan.md`** — § 4.27 added (SDK v3.10
  milestone summary + post-v3.10 reading).
- **`docs/START_HERE.md`** — SDK v3.10 paragraph + W9 family summary;
  links to milestone result + success bar updated.

### Preserved

- **Wevra single-run product runtime contract.** Byte-for-byte
  unchanged from SDK v3.9. The Phase-45 product report schema
  (`PRODUCT_REPORT_SCHEMA = "phase45.product_report.v2"`) is
  unchanged.
- **SDK v3.5–v3.9 multi-agent surface.** Every fixed admission
  policy from previous SDKs (FIFO, priority, coverage,
  cohort_coherence, cross_role_corroboration) is unchanged; W7-2
  and W8-1 contract tests still pass byte-for-byte. The new W9
  policy is purely additive.
- **Lifecycle audit (T-1..T-7).** Holds on every cell of every
  regime (R-53 / R-54 / R-55 / R-56 default / R-56 falsifier).

## [3.9] — 2026-04-26 — SDK v3.9 — cross-role corroboration multi-agent coordination + W8 family

*Strictly additive on SDK v3.8. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``CrossRoleCorroborationAdmissionPolicy`` is a research-slice
addition to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime. **First SDK milestone to clear the strong success
bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1** —
strict separation from W7-2 on Phase 55 (+1.000 corroboration −
cohort_buffered, +1.000 corroboration − fifo on accuracy_full),
backward-compat on Phase 54 (corroboration ties W7-2 at 1.000),
no regression on Phase 53 synthetic (0.800) or 14B real-LLM
(0.800), cross-bank stability across 5/5 seeds, named falsifier
regime correctly ties FIFO at 0.000.*

### Added

- **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** (new): pre-
  committed strong / partial / null success bars with named
  regimes (R-53 / R-54 / R-55).
- **Phase-55 decoy-plurality + cross-role-corroborated benchmark**
  (new): `vision_mvp/experiments/phase55_decoy_plurality.py`.
  Smallest deterministic regime where (a) decoy raw plurality
  breaks W7-2 single-tag plurality cohort coherence AND (b) gold
  cross-role corroboration provides a relational signal a (role,
  tag)-aggregating policy can exploit. Bench properties named and
  mechanically verified.
- **`CrossRoleCorroborationAdmissionPolicy`** (new): in
  `vision_mvp/wevra/team_coord.py`. Deterministic, training-free
  admission rule with score function `role_weight·|distinct_roles|
  + |raw_mentions|`. Buffered factory `from_candidate_stream` is
  the W8-1 anchor. Re-exported as
  `TeamCrossRoleCorroborationAdmissionPolicy`.
- **W8 theorem family**: W8-1 (strict separation, proved-empirical
  n=50), W8-2 (score-function strict ordering, proved structural),
  W8-3 (backward-compat with W7-2 on Phase 54, proved-empirical),
  W8-4 (decoy-corroboration falsifier, proved-empirical n=10).
  W8-C1 / W8-C2 / W8-C3 conjectures.
- **34 contract tests**:
  `vision_mvp/tests/test_wevra_cross_role_corroboration.py`.
- **Frozen reproducibility artefacts**:
  `docs/data/phase55_decoy_plurality_K4_n10.json` (default),
  `docs/data/phase55_falsifier_K4_n10.json` (W8-4),
  `docs/data/phase55_budget_sweep.json`,
  `docs/data/phase55_seed_sweep.json`,
  `docs/data/phase55_cross_regime.json`,
  `docs/data/phase53_real_llm_corroboration_check.json`.

### Changed

- `vision_mvp/wevra/__init__.py`: re-exports
  `TeamCrossRoleCorroborationAdmissionPolicy`; `SDK_VERSION`
  bumped to `"wevra.sdk.v3.9"`.
- `vision_mvp/tests/test_wevra_public_api.py`: SDK version test
  updated to v3.9; new corroboration export test.
- `docs/THEOREM_REGISTRY.md`: W8 family rows added; date stamp
  v3.9.
- `docs/RESEARCH_STATUS.md`: eighth research axis added.
- `docs/HOW_NOT_TO_OVERSTATE.md`: W8 overstatement guards added
  (W8-1 conditionality; "we solved multi-agent context" forbidden
  without naming the strong success bar; Phase-54/55 conflation
  forbidden; Phase-53/55 conflation forbidden).
- `docs/context_zero_master_plan.md`: § 4.26 SDK v3.9 added.
- `docs/START_HERE.md`: SDK v3.9 paragraph + canonical-reading
  pointer to the success-criterion doc.
- `docs/RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md`: new milestone
  results note.

### Honest scope

- The W8-1 win is **conditional** on the named bench property
  (decoy-plurality + cross-role-corroborated gold). The W8-4
  falsifier regime is the explicit named counterexample.
- Three named regimes is a stronger cross-regime result than two,
  but not "all regimes." Real production multi-agent teams have
  additional axes the W8 family does not test (heterogeneous
  producers, time-varying budgets, multi-round handoffs,
  multi-service gold). W8-C1 / W8-C2 / W8-C3 are conjectural;
  none yet shipped.
- The Wevra single-run product runtime contract is byte-for-byte
  unchanged from SDK v3.8. The new admission policy is a
  research-slice addition.

## [3.8] — 2026-04-26 — SDK v3.8 — cross-role cohort-coherence multi-agent coordination + W7 family

*Strictly additive on SDK v3.7. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``CohortCoherenceAdmissionPolicy`` is a research-slice addition
to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime.*

### Added

- **Phase-54 cross-role cohort-coherence benchmark** (new):
  ``vision_mvp/experiments/phase54_cross_role_coherence.py``.
  Smallest deterministic multi-agent benchmark where cross-role
  coordination provides a strict structural advantage over
  substrate FIFO. Bench properties (gold-plurality, cross-role,
  budget-bound, decoder-pollution) are *named and mechanically
  verified* by the contract tests. Runs end-to-end without any
  LLM in the loop.
- **``CohortCoherenceAdmissionPolicy``** (new): in
  ``vision_mvp/wevra/team_coord.py``. Deterministic, training-free,
  interpretable cross-role admission rule. Two sub-modes:
  *streaming* (running cohort over already-admitted; arrival-
  order-sensitive) and *buffered* (pre-fitted plurality from the
  full candidate stream's payloads via
  ``from_candidate_payloads``; arrival-order-stable). Re-exported
  as ``TeamCohortCoherenceAdmissionPolicy``.
- **W7 theorem family**: W7-1 (FIFO unbeatability under low
  surplus, proved-empirical anchor on Phase-53), W7-1-aux
  (streaming cohort instability under arrival permutation,
  proved-empirical), W7-2 (cohort_buffered structural win under
  gold-plurality, proved-empirical, n=50 saturated, 5/5 stable
  across bank seeds), W7-2-conditional (K-sweep window,
  proved-empirical), W7-3 (extraction floor, proved-negative,
  Capsule Contract C5 corollary). W7-C1/C2/C3 conjectures cover
  multi-service-gold, decoder-side coordination, and real-LLM
  transfer extensions.
- **21 contract tests** for the new policy + bench:
  ``vision_mvp/tests/test_wevra_cross_role_coherence.py``.
- **Frozen reproducibility artefacts**:
  ``docs/data/phase54_cross_role_coherence_K4_n10.json`` (default
  config),
  ``docs/data/phase54_cross_role_coherence_budget_sweep.json``
  (K-sweep).
- **Milestone results note**:
  ``docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md``.

### Changed

- ``SDK_VERSION`` bumped to ``"wevra.sdk.v3.8"``.
- ``vision_mvp/wevra/__init__.py``: re-exports
  ``TeamCohortCoherenceAdmissionPolicy``.
- ``vision_mvp/wevra/team_coord.py``:
  ``ALL_FIXED_POLICY_NAMES`` extended with
  ``"cohort_coherence"``; new helper ``_candidate_service_tag``.
- ``docs/THEOREM_REGISTRY.md``: W7-1 / W7-1-aux / W7-2 /
  W7-2-conditional / W7-3 / W7-C1 / W7-C2 / W7-C3 rows added.
- ``docs/RESEARCH_STATUS.md``: seventh research axis added (W7
  family); now lists 7 coupled axes.
- ``docs/HOW_NOT_TO_OVERSTATE.md``: W7-overstatement guards
  added (cohort-coherence wins are *conditional* on bench
  properties; SDK v3.7 and SDK v3.8 results are both true,
  conditioned on different bench properties; *buffered* vs
  *streaming* distinction must be specified).
- ``docs/context_zero_master_plan.md``: § 4.25 added.
- ``docs/START_HERE.md``: SDK v3.8 paragraph added.

### Honest scope

- **The W7-2 win is conditional.** It depends on the bench having
  gold-plurality + foreign-service decoys + ``|candidates| >
  K_auditor``. The Phase-53 (real-LLM) reading is preserved
  exactly: substrate FIFO ties every fixed strategy at
  ``accuracy_full = 0.800`` because the bench has no surplus
  (W7-1).
- **The Wevra single-run product runtime contract is unchanged.**
  ``RunSpec`` / ``run`` / ``SweepSpec`` / ``run_sweep`` /
  report v2 schema: byte-for-byte identical from SDK v3.7.
- **The capsule layer's audit contribution is preserved.**
  T-1..T-7 hold on every Phase-54 cell unchanged.
- **Mac 2 still offline.** No two-Mac sharded inference happened
  in SDK v3.8; the ``MLXDistributedBackend`` integration boundary
  is byte-for-byte unchanged from SDK v3.6.

## [docs] — 2026-04-26 — documentation consolidation (no SDK change)

*Repo-cleanup only. No code change. SDK contract byte-for-byte
unchanged. Strictly additive on SDK v3.7.*

### Changed

- **Top-level Markdown clutter consolidated.** The repo root and
  `docs/` are reduced to a small canonical set; everything else is
  preserved under `docs/archive/`. The active scientific position is
  now obviously the live entry point and stale milestone notes can no
  longer read like current claims.
- **Canonical kept set** (top level): `README.md`, `ARCHITECTURE.md`,
  `CHANGELOG.md`, `LICENSE`, `CLAUDE.md`. **Canonical kept set**
  (`docs/`): `START_HERE.md`, `RESEARCH_STATUS.md`,
  `THEOREM_REGISTRY.md`, `HOW_NOT_TO_OVERSTATE.md`,
  `CAPSULE_FORMALISM.md`, `CAPSULE_TEAM_FORMALISM.md`,
  `context_zero_master_plan.md`, `MLX_DISTRIBUTED_RUNBOOK.md`,
  `RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` (latest milestone, kept live).
- **Archive layout** (`docs/archive/`):
  - `capsule-research/` — `RESULTS_CAPSULE_LEARNING.md` +
    `RESULTS_CAPSULE_RESEARCH_MILESTONE[1-6].md`.
  - `wevra-milestones/` — older Wevra milestone notes
    `RESULTS_WEVRA_{CAPSULE, CAPSULE_NATIVE, INTRA_CELL,
    DEEP_INTRA_CELL, INNER_LOOP, TEAM_COORD, DISTRIBUTED}.md`
    (SDK v3.0 → v3.6).
  - `pre-wevra-theory/` — pre-Wevra Context Zero theory volumes:
    `PROOFS.md`, `EXTENDED_MATH[_1-7].md`, `OPEN_QUESTIONS.md`,
    `FRAMEWORK.md`, `EVALUATION.md`, `MVP.md`, `ROADMAP.md`,
    `VISION_MILLIONS.md`, `MATH_AUDIT.md`,
    `HIERARCHICAL_DECOMPOSITION.md`, `WAVES.md`.
  - `legacy-progress-notes/` — sprint prompts, paradigm-shift
    summaries, the pre-Wevra benchmark-reproduction guide, the
    auto-generated theorem index.
- **`docs/archive/README.md`** *(new)* — archive index. Names every
  archived doc, points to the canonical replacement, and explains the
  read-only contract: the active scientific position is in `docs/`,
  the archive is historical record only.
- **Internal links updated.** Every cross-link inside the canonical
  docs (`README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/*.md`,
  `papers/*.md`) now resolves to the new file paths. Validated
  programmatically — zero broken Markdown links across the 14
  canonical docs.
- **`docs/START_HERE.md`** — adds a *Current canonical reading* table
  at the very top of the file. Mental-model diagram updated to show
  active vs archived theory paths.
- **`vision_mvp/scripts/generate_theorem_docs.py`** — auto-generated
  `THEOREMS_AUTO.md` now writes into
  `docs/archive/legacy-progress-notes/THEOREMS_AUTO.md` (was
  `docs/THEOREMS_AUTO.md`); the file was always a generated artefact,
  not a canonical claim source.

### Preserved

- All historical research material is intact under `docs/archive/`.
  No file deleted. No claim retracted. No theorem renumbered.
- `vision_mvp/RESULTS_PHASE*.md` (the per-phase research diary) is
  untouched — it lives with the code, not under `docs/`.
- The Wevra SDK public contract, the Capsule Contract C1..C6, and
  the W3 / W4 / W5 / W6 theorem families are unchanged.

## [SDK v3.7] — 2026-04-26 — model-scale vs capsule-structure on multi-agent coordination (Phase-53 + W6 family)

*Strictly additive on SDK v3.6. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new surface is
the Phase-53 stronger-model multi-agent benchmark + W6 theorem
family. Mac 2 is still offline; no two-Mac sharded inference
happened in this milestone — the ``MLXDistributedBackend``
integration boundary is byte-for-byte unchanged from SDK v3.6
and waits for the runbook.*

### Added

- **`vision_mvp/experiments/phase53_scale_vs_structure.py`** *(new)* —
  Phase-53 stronger-model multi-agent benchmark. Drives the team
  coordinator with a real-LLM producer-role extractor across
  three model regimes (synthetic / qwen2.5:14b-32k / qwen3.5:35b)
  × five admission strategies (substrate, capsule_fifo,
  capsule_priority, capsule_coverage, capsule_learned) on the
  same candidate-handoff stream. Reports a clean ``model regime ×
  admission strategy`` decomposition with cross-regime
  candidate-kind TVD.
- **`vision_mvp/tests/test_wevra_scale_vs_structure.py`** *(new)*
  — 19 contract tests: parser robustness on the closed-vocabulary
  claim grammar (16 cases), backend duck-typing, audit_ok grid
  end-to-end with a deterministic stub backend, schema lock.
- **`docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`** *(new)* — full
  milestone results note. Theorem-forward; declares W6-1..W6-4
  proved-or-empirical and W6-C1..W6-C5 conjectures (W6-C1 / C2
  drafted-then-falsified, W6-C3 positive, W6-C4 / C5 new).
- **`docs/data/phase53_scale_vs_structure_K4_n5.json`** *(new
  artefact)* — frozen benchmark output for reproducibility.

### Changed

- **`vision_mvp/wevra/__init__.py`** — `SDK_VERSION` bumped to
  `wevra.sdk.v3.7`. No public API change.
- **`docs/THEOREM_REGISTRY.md`** — W6-1 / W6-2 / W6-3 / W6-4 +
  W6-C1 / W6-C2 / W6-C3 / W6-C4 / W6-C5 rows added. The W4-C1
  row (SDK v3.5 conjecture) is now annotated as **conditional**:
  empirical-positive on its anchor distribution; falsified
  out-of-distribution on the Phase-53 real-LLM regime
  (capsule_learned 0.4 vs fixed 0.8 on synthetic and qwen2.5:14b;
  ties at qwen3.5:35b at 0.8/0.8).
- **`docs/RESEARCH_STATUS.md`** — sixth research axis added;
  active-conjectures section refreshed with W6-C family.
- **`docs/context_zero_master_plan.md`** — § 4.24 added: full
  Phase-53 narrative, W6 / W6-C summary, W4-C1 conditional
  reading, honest scope (Mac 2 offline, single-Mac qwen3.5:35b
  is the strongest model class actually exercised).
- **`docs/START_HERE.md`** — headline paragraph updated to
  reference the SDK v3.7 result and the *audit-axis* tightening
  of the original Context-Zero thesis.

### Headline empirical result

(n=5 saturated, K_auditor=4, T_auditor=128, three model regimes,
deterministic seeds (31, 32, 33))

| regime           | substrate | fixed capsule | learned |
| ---------------- | --------- | ------------- | ------- |
| synthetic        | 0.800     | 0.800         | 0.400   |
| qwen2.5:14b-32k  | 0.800     | 0.800         | 0.400   |
| qwen3.5:35b      | 0.800     | 0.800         | 0.800   |

* `structure_gain[regime]` = -0.4 / -0.4 / 0.0 (non-positive
  everywhere; scale narrows a *deficit*, not a *surplus*).
* `scale_gain[capsule_learned]` = +0.4; `scale_gain[fixed]` = 0.0.
* Cross-(14B, 35B) candidate-kind TVD = 0.167.
* Capsule-team lifecycle audit ``audit_team_lifecycle.is_ok()``
  = 60/60 across (regime × capsule strategy × scenario).

### Theorem registry deltas

- **W6-1 (proved + mechanically-checked).** Lifecycle audit
  T-1..T-7 holds 60/60 across the Phase-53 grid.
- **W6-2 (proved).** Phase-53 driver accepts duck-typed
  ``LLMBackend``.
- **W6-3 (proved + mechanically-checked).** Parser robustness
  on the closed-vocabulary claim grammar.
- **W6-4 (proved-empirical, real LLM, n=5 saturated).** The
  ``accuracy_full`` / ``structure_gain`` / ``scale_gain``
  decomposition is what is reported above.
- **W6-C1, W6-C2 (drafted, FALSIFIED-empirical).** Structure-
  preservation under scale (W6-C1) and synthetic→real-LLM
  transfer of the learned admission scorer (W6-C2) are both
  falsified on Phase-53 default; honest revised reading is in
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` § 4.3.
- **W6-C3 (empirical-positive).** Cross-(14B, 35B) candidate-
  kind TVD = 0.167 > 0.10 falsifier.
- **W6-C4, W6-C5 (new conjectures).** Substrate-FIFO competitive-
  ness at sufficient K, and scale-narrows-the-OOD-gap of the
  per-role admission scorer.

### Honest scope

* Mac 2 (192.168.12.248) is offline at the time of this
  milestone (ARP "incomplete"). **No two-Mac sharded inference
  ran.** No 70 B-class model ran. The strongest model class
  exercised is **single-Mac** qwen3.5:35b (36 B-MoE Q4) via
  Mac 1 Ollama.
* The MLX-distributed integration boundary
  (``MLXDistributedBackend``) is byte-for-byte unchanged from
  SDK v3.6 and remains correct against the in-process stub
  (W5-3). The runbook (`docs/MLX_DISTRIBUTED_RUNBOOK.md`) is the
  operator path when Mac 2 returns.
* Phase-53 is **incident-triage-bench-internal**. External
  validity to other multi-agent benchmarks is open
  (`task_scale_swe.py`, `phase33_security_escalation.py` are
  obvious next targets).
* The W4-C1 (SDK v3.5) reading on its anchor config (Phase-52
  default, K=8, spurious=0.30) is unchanged. The new SDK v3.7
  reading is OOD.

### Tests + validation

* `python3 -m unittest -v vision_mvp.tests.test_wevra_scale_vs_structure`
  → **19 tests pass in 0.069 s**.
* `python3 -m unittest vision_mvp.tests.test_wevra_team_coord
  vision_mvp.tests.test_wevra_llm_backend
  vision_mvp.tests.test_wevra_capsule_native_inner_loop
  vision_mvp.tests.test_wevra_capsule_native
  vision_mvp.tests.test_wevra_capsule_native_intra_cell
  vision_mvp.tests.test_wevra_capsule_native_deeper
  vision_mvp.tests.test_wevra_scale_vs_structure`
  → **116 tests pass in 3.207 s** (SDK v3.6 invariants intact).
* `python3 -m vision_mvp.experiments.phase53_scale_vs_structure
  --endpoint http://192.168.12.191:11434
  --models synthetic,qwen2.5:14b-32k,qwen3.5:35b
  --n-eval 5 --K-auditor 4 --T-auditor 128
  --out /tmp/wevra-distributed/phase53_scale_vs_structure_K4.json`
  → 14B LLM wall 92.6 s; 35B LLM wall 152.0 s; n_results = 75.
  Frozen at `docs/data/phase53_scale_vs_structure_K4_n5.json`.

## [SDK v3.5] — 2026-04-26 — capsule-native multi-agent team coordination (research slice)

*Strictly additive on SDK v3.4. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new surface is a
capsule-native multi-agent coordination research slice that
runs side-by-side with the Wevra SDK.*

### Added

- **Three new closed-vocabulary `CapsuleKind` values** — `TEAM_HANDOFF`
  (capsule-native multi-agent handoff; distinct from `HANDOFF`
  which adapts a substrate `TypedHandoff`), `ROLE_VIEW` (per-role
  admitted view of one coordination round; `max_parents = K_role`,
  `max_tokens = T_role`), `TEAM_DECISION` (team-level decision).
- **`vision_mvp.wevra.team_coord`** — `RoleBudget`,
  `DEFAULT_ROLE_BUDGETS`, `capsule_team_handoff`,
  `capsule_role_view`, `capsule_team_decision`, three fixed
  admission policies (`FifoAdmissionPolicy`,
  `ClaimPriorityAdmissionPolicy`, `CoverageGuidedAdmissionPolicy`),
  `TeamCoordinator`, `audit_team_lifecycle` over invariants
  `T-1..T-7` (Theorem **W4-1**, *proved + mechanically-checked*).
- **`vision_mvp.wevra.team_policy`** —
  `LearnedTeamAdmissionPolicy` (per-role logistic-regression
  scorer over six capsule features), `TrainSample`, `TrainStats`,
  `train_team_admission_policy`. Numpy-only; deterministic given
  seed.
- **`vision_mvp/experiments/phase52_team_coord.py`** — reference
  benchmark over a noisy-extraction expansion of the Phase-31
  incident-triage bank. Cross-seed result on default config
  ($K_\text{auditor}=8$, $T_\text{auditor}=256$,
  $n_\text{eval}=31$, ``train_seed ∈ {0, …, 11}``,
  ``PYTHONHASHSEED=0``): **learned policy** admits **strictly
  fewer handoffs** than the strongest fixed baseline
  (coverage-guided) on every train seed (12/12), with mean
  savings ≈ 1.26 handoffs per scenario. The learned policy also
  improves pooled team-decision accuracy on most train seeds
  (gap on `accuracy_full` > 0 in 11/12 seeds, mean **+0.054**;
  gap on `accuracy_root_cause` > 0 in 8/12 seeds, mean
  **+0.032**) — but the accuracy advantage **reverses at higher
  noise** (`spurious_prob = 0.50`). `audit_ok_rate = 1.000` for
  every capsule strategy on every seed. Conjecture **W4-C1**:
  budget-efficiency dominance is robust per-seed; accuracy
  advantage is mean-positive on the default noise config but
  not strict per-seed; advantage does not survive heavier
  noise. (See ``docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md`` § Cross-seed
  result for the canonical reading; ``docs/HOW_NOT_TO_OVERSTATE.md``
  forbids reporting single-seed numbers without the cross-seed
  distribution.)
- **Theorems** — W4-1 (proved + mechanically-checked); W4-2
  (proved-conditional: coverage-implies-correctness); W4-3
  (proved-negative: per-role budget below the role's causal-
  share floor cannot be rescued by *any* admission policy).
- **Conjectures** — W4-C1, W4-C2 (cohort-lifted role view closes
  W4-3 sub-class), W4-C3 (capsule admission rule subsumes
  Phase-36 adaptive-sub).
- **`docs/CAPSULE_TEAM_FORMALISM.md`** — formal model.
- **`docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md`** — milestone note.
- **`vision_mvp/tests/test_wevra_team_coord.py`** — 22 contract
  tests.
- **README**, **START_HERE**, **RESEARCH_STATUS**,
  **THEOREM_REGISTRY**, **HOW_NOT_TO_OVERSTATE**, **master plan
  §4.22** — all updated.

### Compatibility

- All 85 capsule-native run-boundary tests (v3.1..v3.4) +
  Phase-31 typed-handoff tests continue to pass byte-for-byte.
  Team-layer tests are 22 additional contracts.
- The Wevra `wevra` console scripts are unchanged. The team layer
  ships as `vision_mvp.wevra.team_coord` /
  `vision_mvp.wevra.team_policy` and is also re-exported from
  the top-level `vision_mvp.wevra` namespace as
  `TeamCoordinator`, `audit_team_lifecycle`,
  `LearnedTeamAdmissionPolicy`, etc.

### Honest scope

The Phase-52 benchmark is synthetic; the result *direction* is
robust under deterministic noise; cross-bench transfer is open.
"We solved multi-agent context" is **forbidden** by
`docs/HOW_NOT_TO_OVERSTATE.md`; the defensible reading is
W4-1 / W4-2 / W4-3 / W4-C1 above.

## [SDK v3.4] — 2026-04-26 — sub-sub-intra-cell PROMPT / LLM_RESPONSE slice + synthetic mode + cross-model parser-boundary research

*Strictly additive on SDK v3.3. Every v3.3 contract test (18) still
passes byte-for-byte; capsule view schema name unchanged
(`wevra.capsule_view.v1` — PROMPT / LLM_RESPONSE payloads are
additive). Full Wevra + capsule test suite green (199 tests).*

### Added
- **PROMPT capsule kind** (parent: SWEEP_SPEC; Theorem W3-42).
  Records prompt SHA-256 + byte length + bounded text snippet
  (≤ 4 KiB) + model_tag + prompt_style + coordinates.
  Idempotent on content (Capsule Contract C1) — byte-identical
  prompts collapse to one capsule.
- **LLM_RESPONSE capsule kind** (parent: PROMPT; Theorem W3-43).
  Records response SHA-256 + byte length + bounded snippet +
  elapsed milliseconds + coordinates. Admission rejects if
  prompt CID is not yet sealed (Capsule Contract C5).
- **`CapsuleNativeRunContext.seal_prompt`** /
  **`seal_llm_response`** runtime methods, plus
  **`seal_parse_outcome(llm_response_cid=...)`** optional
  argument. The end-to-end inner-loop chain is now five typed
  capsules: `PROMPT → LLM_RESPONSE → PARSE_OUTCOME →
  PATCH_PROPOSAL → TEST_VERDICT`.
- **`capsule_from_prompt`**, **`capsule_from_llm_response`**
  adapters; `PROMPT_TEXT_CAP` / `LLM_RESPONSE_TEXT_CAP` constants.
- **Lifecycle audit invariants L-9 / L-10 / L-11** (Theorems
  W3-44 / W3-45):
  - L-9: PROMPT.parents == (SWEEP_SPEC,).
  - L-10: LLM_RESPONSE has exactly one parent, a sealed PROMPT.
  - L-11: PARSE_OUTCOME / LLM_RESPONSE coordinate consistency
    (instance_id / parser_mode / apply_mode / n_distractors;
    strategy may differ).
- **Synthetic-LLM mode**: `SweepSpec(mode="synthetic",
  synthetic_model_tag=<tag>)`. Uses a deterministic in-process
  `SyntheticLLMClient` instead of an Ollama endpoint. Seven
  calibrated distributions ship in
  `vision_mvp.wevra.synthetic_llm.SYNTHETIC_MODEL_PROFILES`:
  `clean`, `unclosed`, `prose`, `empty`, `fenced`,
  `multi_block`, `mixed`. The full PROMPT / LLM_RESPONSE /
  PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT chain seals
  end-to-end without network access.
- **Cross-model parser-boundary experiment** (Conjecture W3-C6,
  empirical):
  `vision_mvp.experiments.parser_boundary_cross_model`. Sweeps
  `(model_tag, parser_mode)` across the synthetic distribution
  library; reports cross-distribution PARSE_OUTCOME failure-kind
  TVD up to 1.000 and parser-mode (strict→robust) shift up to
  1.000 on `synthetic.unclosed`. Reproducible from CLI:
  `python3 -m vision_mvp.experiments.parser_boundary_cross_model`.
- **16 new contract tests** in
  `vision_mvp/tests/test_wevra_capsule_native_inner_loop.py`
  covering W3-42 / W3-43 / W3-44 / W3-45 / W3-C6.

### Changed
- **`SDK_VERSION`** bumped to `wevra.sdk.v3.4`.
- **`CapsuleKind.ALL`** now includes `PROMPT` and `LLM_RESPONSE`.
- **`render_view.payload_kinds_always`** extended to include
  PROMPT and LLM_RESPONSE (so on-disk audits can navigate the
  full inner-loop chain from `capsule_view.json` alone).
- **`CapsuleLifecycleAudit.RULES`** extended from 8 rules to 11.
- **W3-13** (DAG height ≤ 4 on canonical run pattern) is updated
  to ≤ 5 on canonical SDK v3.4 runs (the inner-loop chain adds
  one structural layer). Documented in
  `docs/CAPSULE_FORMALISM.md` § 4.J.
- **Conjecture W3-C5 (legacy SDK v3.3)** is **DISCHARGED** by
  Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **Conjecture W3-C4 (legacy SDK v3.3)** is **superseded** by the
  sharper synthetic reading W3-C6.

### Documentation
- New milestone note: **`docs/archive/wevra-milestones/RESULTS_WEVRA_INNER_LOOP.md`**.
- `docs/CAPSULE_FORMALISM.md` § 4.J added (W3-42 / W3-43 / W3-44 /
  W3-45 / W3-C6 + W3-C5-discharged).
- `docs/THEOREM_REGISTRY.md`, `docs/RESEARCH_STATUS.md`,
  `docs/HOW_NOT_TO_OVERSTATE.md` updated for SDK v3.4.
- `docs/START_HERE.md` adds "What changed in SDK v3.4" section.
- `docs/context_zero_master_plan.md` § 4.21 added.
- `papers/wevra_capsule_native_runtime.md` strengthened —
  capsule-native execution is now its real centre, with strict
  claim taxonomy covering PROMPT / LLM_RESPONSE chain and the
  W3-C6 empirical anchor.
- README headline + stability matrix updated.

## [0.5.1] — 2026-04-22 — Wevra identity & clarity pass

*Documentation / exemplar milestone. No SDK-contract change; all 1349
Slice-2 tests still pass.*

### Added
- **`docs/START_HERE.md`** — canonical one-pass orientation for new
  readers. Classifies every top-level surface (Wevra SDK, CLI,
  extension protocols, unified runtime, legacy product path, core
  substrate, research shards, boundary). Meant to be the answer to
  "what is this repo?" without duplicating the README or the master
  plan.
- **`examples/out_of_tree_plugin/wevra-markdown-sink/`** — first
  in-repo exemplar of a standalone pip-installable Wevra plugin
  package. Declares `[project.entry-points."wevra.report_sinks"]`,
  registers a Markdown `ReportSink` via
  `importlib.metadata.entry_points`, and requires zero edit under
  `vision_mvp/`. Closes master-plan § 10.5 ledger item 2 at the
  machinery-plus-artifact level (only the "published by a third
  party" condition remains future).
- **`vision_mvp/RESULTS_WEVRA_IDENTITY.md`** — theory-forward results
  note with theorem-style claims (W-IDN-1 identity projection,
  W-IDN-2 orientation sufficiency, W-IDN-3 extension-surface
  reality) and three conjectures (W-IDN-C1 cold-agent
  classification, W-IDN-C2 stable-identity robustness, W-IDN-C3
  distinctiveness via composition rather than primitive novelty).

### Changed
- **README headline** now leads with **Wevra** (the shipped product)
  and positions CASR as original-substrate research; the scaling
  claims are preserved and re-anchored to Theorem 3 in `docs/archive/pre-wevra-theory/PROOFS.md`.
- **ARCHITECTURE.md headline** re-anchored to Wevra + Context Zero;
  a framing callout was added before the Phase 26–44 block so
  readers know that block is a historical incremental record and
  the durable architecture is the layered substrate diagram + § 3
  of the master plan.
- **`vision_mvp/__init__.py`** top-level docstring: Wevra is the
  shipped product; `CASRRouter` is explicitly research-grade code
  used by the SDK under the hood.
- **`vision_mvp/api.py`** `CASRRouter` docstring no longer says
  "Phase-3 hierarchical protocol" or "CASR-theoretic optimum" in
  places where a user would read them as current product contract;
  the O(log N) bound is now anchored to Theorem 3.
- **`vision_mvp/product/__init__.py`** retitled from "Phase-45
  product-grade orchestration surface" to "Legacy product modules
  (pre-Wevra import path)" — same code, correct framing.
- **`pyproject.toml`** — clearer comment on the `casr` legacy
  script; public CLI stays `wevra` / `wevra-import` / `wevra-ci`.
- **Master plan § 10** — short "Programme vs Product" callout near
  the top; § 10.1 stability matrix row for out-of-tree plugins
  updated from "boundary / next-slice" to "exemplar landed";
  § 10.3 B.6 note and § 10.5 ledger item 2 updated.

### Not changed (deliberately)
- The Wevra SDK contract (every Slice 2 public symbol remains).
- Any test; suite is green at 1349/1349.
- Docker-first-by-default flip for untrusted JSONLs (still Slice 3).
- GitHub Actions release-on-real-tag firing (workflow still declared,
  not yet exercised on a real tag).

## [0.5.0] — 2026-04-22 — Wevra SDK Slice 2

### Added
- **Extension system** (`vision_mvp/wevra/extensions/`). Three
  runtime-checkable Protocols — `SandboxBackend`, `TaskBankLoader`,
  `ReportSink` — each with an in-process registry and discovery via
  `importlib.metadata.entry_points` under groups
  `wevra.sandboxes`, `wevra.task_banks`, `wevra.report_sinks`.
  One worked example (`JsonlWithMetaSink`) and a contract test
  suite that exercises the full register→resolve→emit path.
- **Unified mock/real runtime** (`vision_mvp/wevra/runtime.py`).
  New `SweepSpec` dataclass; single `run_sweep(spec)` entry point
  dispatches mock and real runs through the same substrate
  primitives. Real runs execute in-process when
  `RunSpec.acknowledge_heavy=True`; otherwise the SDK refuses to
  start the heavy run and emits the resolved launch command.
- **`RunSpec.acknowledge_heavy`** and **`RunSpec.report_sinks`** —
  first-class cost gate and plugin hook on the top-level SDK spec.
- **`HeavyRunNotAcknowledged`** exception — strict cost-gate signal.
- **Env-driven endpoints**: `WEVRA_OLLAMA_URL_MAC1`,
  `WEVRA_OLLAMA_URL_MAC2`, `WEVRA_OLLAMA_URL` override profile-
  declared URLs at runtime. No hard-coded cluster IP is baked into
  code paths that a third-party consumer has to edit.
- **`--acknowledge-heavy` / `--report-sink`** flags on `wevra`.
- **Report schema bump**: `phase45.product_report.v2`. v1 remains
  accepted by `wevra-ci`; both listed in `EXPECTED_REPORT_SCHEMAS`.
- **GitHub Actions workflow** (`.github/workflows/wevra-ci.yml`):
  SDK contract tests on 3.10/3.11/3.12, console-script smoke,
  `python -m build` sdist+wheel, release on tag.
- **Cluster-backed validation artifact** under
  `vision_mvp/artifacts/wevra_slice2_g1/` — real ASPEN `mac1`
  `qwen2.5-coder:14b` run launched via `wevra.run(RunSpec(...,
  acknowledge_heavy=True))`, with provenance manifest and
  `wevra-ci` verdict.
- **Theory note**: `vision_mvp/RESULTS_WEVRA_SLICE2.md` —
  theorem-style claims W2-1 … W2-4.

### Changed
- `SDK_VERSION` bumped to `wevra.sdk.v2`. The bump is additive;
  every Slice 1 public symbol remains available.
- `CI gate` accepts v1 and v2 report schemas.
- `product/runner.py` now routes all sweeps through
  `wevra.runtime.run_sweep` instead of the legacy
  `_real_sweep_stub`.

### Deprecated
- `_real_sweep_stub` / `_mock_sweep` in `vision_mvp/product/runner.py`
  are private and will be removed in a future release; external code
  should use `wevra.run_sweep(SweepSpec(...))`.

### Next-slice (deferred, still honest)
- Docker-first sandbox as the default for public/untrusted JSONLs
  (backend exists; default-flip is Slice 3).
- Public SWE-bench-Lite JSONL on local disk (🧱 external).
- Resident ≥70B coder-finetuned model (🧱 external).

## [0.4.0] — 2026-04-21 — Wevra SDK Slice 1

See `docs/context_zero_master_plan.md` § 10.2.

- Introduced `vision_mvp/wevra/` stable SDK boundary.
- `RunSpec` / `run`, `WevraConfig`, `build_manifest`, schema
  constants, profile/report/ci_gate/import_data re-exports.
- Provenance manifest (`wevra.provenance.v1`) on every run.
- Console scripts: `wevra`, `wevra-import`, `wevra-ci`.
- Package renamed to `wevra` on PyPI; `SDK_VERSION = wevra.sdk.v1`.
- `sys.path.insert` hacks removed from product modules.
- Contract tests: `test_wevra_public_api.py`, `test_wevra_provenance.py`.

---

## [0.1.0] — 2026-04-16

Initial alpha release. One continuous research session.

### Added — Core library (`vision_mvp/`)

- **`CASRRouter`** — black-box public API. `step(observations) -> estimates`.
- Core primitives: `Bus`, `Agent`, `Manifold` (given basis),
  `StreamingPCA` (learned basis), `Stigmergy` (CRDT register),
  `Workspace` (top-k admission), `NeuralPredictor` and `PredictorBank`
  (vectorized across agents).
- Phase-6 additions: `MarketWorkspace` (VCG pricing),
  `SharedRNG`/`DeltaChannel` (pre-shared randomness), `AdaptiveScale` and
  `ContinuousScaleProjector` (continuous-scale projection).
- Six coordination protocols: `naive`, `gossip`, `manifold_only`,
  `full_stack`, `adaptive`, `hierarchical`, `holographic`, `swarm`, and
  `llm_protocols` (real LLM agents via Ollama).
- Two coordination tasks: `consensus` (static) and
  `drifting_consensus` (non-stationary with optional shock).

### Added — Experiments & results

- Phase 1 through Phase 5 runnable experiment harnesses under
  `vision_mvp/experiments/`.
- Measured scaling law: peak per-agent context = ⌈log₂ N⌉ exactly at
  every N ∈ {10, 50, 200, 1 000, 5 000, 10 000, 20 000, 50 000, 100 000}.
- Real LLM demonstration at N = 10 (local qwen2.5:0.5b via Ollama) showing
  34 % token savings with 100 % accuracy.

### Added — Theory

- **`docs/archive/pre-wevra-theory/PROOFS.md`** — twelve formal theorems, each with a proof and a
  machine-checkable empirical counterpart in `tests/`.
- **`EXTENDED_MATH_[1–7].md`** — 72-framework survey converging on the
  O(log N) bound from Information Bottleneck through Geometric Langlands.
- **`docs/archive/pre-wevra-theory/VISION_MILLIONS.md`** — the 10-idea paradigm shift for million-agent
  systems. 6 of 10 ideas implemented.

### Added — Tests

- **94 tests**, all passing (0.45 s total wall time):
  - 55 core-module unit tests.
  - 15 protocol integration & regression tests (including the scaling-law
    assertion `test_full_stack_peak_context_is_log_n`).
  - 13 Phase-6 tests (market, shared randomness, continuous scale).
  - 11 public-API (`CASRRouter`) tests.

### Added — Developer UX

- `pyproject.toml` (installable as `context-zero`).
- `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`, top-level `README.md`.
- `casr` CLI entry-point (`python -m vision_mvp demo|scale|phase|test|info`).
- Four runnable `examples/`:
  1. basic consensus
  2. drift tracking
  3. scaling demo
  4. local LLM coordination

### Not yet

- Real LLM tests at N > 10 (need bigger compute budget).
- Async variants (current protocol is synchronous).
- A formal peer-review cycle for the math.
- PyPI upload.

All the mathematics says O(log N). The code and the test suite confirm it.
The next step is to run it in anger on harder tasks and let skeptical
reviewers tear it apart.
