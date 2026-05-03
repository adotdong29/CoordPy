# RESULTS — CoordPy SDK v3.36 / W35
# Trust-subspace dense-control proxy + basis-history projection + manifest-v5

**Milestone**: SDK v3.36 (W35 family).
**Date**: 2026-05-02.
**Headline**: W35 is the strongest honest next-step proxy this repo
can support for the native-latent / trust-subspace direction without
entering a transformer runtime.  It wraps W34 with a controller-
verified **trust-subspace dense-control proxy**: one basis entry per
oracle, derived from W21 probe top_sets, W33 EWMA trust, W34
live-attestation/response-feature state, top-set stability, and host
health.  W35 uses that basis only when it can safely convert W34's
NO_CONSENSUS abstention into a verified reroute.

On **R-82-TRUST-SUBSPACE-SHIFT** (16 cells × 5 seeds), W34 abstains
on 6 disputed cells.  W35 safely reroutes 5/6 through the stable
`change_history` basis direction and improves correctness from
0.625 to **0.9375** at **+0.3125** over W34, with
trust precision preserved at **1.000** and one visible-token overhead
per cell.  Mean structured state transferred through the W35 dense
envelope is **13,016.5 bits per visible W35 token**.

This does **not** close W33-C-NATIVE-LATENT.  W35 is an audited
capsule-layer proxy, not a transformer-internal hidden-state
projection, not a KV-cache transplant, and not a learned trust model.
It narrows the gap by making the proxy denser, typed, verified, and
load-bearing.

---

## 1. Mechanism

W35 adds these experimental SDK symbols under the stable/experimental
boundary:

- `TrustSubspaceBasisEntry`
- `TrustSubspaceDenseRatificationEnvelope`
- `TrustSubspaceDenseRegistry`
- `W35TrustSubspaceResult`
- `TrustSubspaceDenseControlOrchestrator`
- `select_trust_subspace_projection`
- `verify_trust_subspace_dense_ratification`
- `build_trivial_trust_subspace_registry`
- `build_trust_subspace_dense_registry`

The projection selector groups basis entries by current `top_set` and
ranks each group by average projection score, not by a single
outlier.  A projection is accepted only when it passes the registered
threshold and margin.  This fixed a local blocker discovered during
the milestone: a bad majority group with one high-score member must
not outrank a stable singleton unless the group average is actually
strong.

The W35 manifest-v5 CID is computed over:

- `parent_w34_cid`
- `basis_state_cid`
- `live_attestation_cid`
- `projection_audit_cid`

The W35 outer CID additionally seals schema, schema CID, cell index,
and the manifest-v5 CID.

---

## 2. Trust Boundary

`verify_trust_subspace_dense_ratification` is a pure verifier with
14 enumerated failure modes, disjoint from the W22/W29/W30/W31/W32/
W33/W34 sets:

| # | Failure mode |
| --- | --- |
| 1 | `empty_w35_envelope` |
| 2 | `w35_schema_version_unknown` |
| 3 | `w35_schema_cid_mismatch` |
| 4 | `w34_parent_cid_mismatch` |
| 5 | `w35_projection_branch_unknown` |
| 6 | `w35_basis_entry_unregistered_oracle` |
| 7 | `w35_basis_score_out_of_range` |
| 8 | `w35_basis_stability_out_of_range` |
| 9 | `w35_basis_state_cid_mismatch` |
| 10 | `w35_projection_top_set_unregistered` |
| 11 | `w35_projection_margin_out_of_range` |
| 12 | `w35_live_attestation_cid_mismatch` |
| 13 | `w35_manifest_v5_cid_mismatch` |
| 14 | `w35_outer_cid_mismatch` |

The W35 test suite now exercises every failure mode plus clean
envelope sanity, projection-selection behavior, and Phase82
benchmark/falsifier behavior.  Cumulative enumerated trust boundary
across W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 is **98**.

---

## 3. Benchmark Family R-82

R-82 is a small benchmark family designed to make W35 matter only
when its basis-history signal is load-bearing.

| Bank | Purpose | Result |
| --- | --- | --- |
| `trivial_w35` | Backward-compatibility: W35 disabled + manifest-v5 disabled | W35 = W34 byte-for-byte across 5/5 seeds |
| `no_anchor_disagreement` | Falsifier where W34 already has consensus | W35 correctness = W34 = 1.000; no correctness lift |
| `trust_subspace_shift` | Load-bearing W35 regime: W34 abstains, one basis direction remains stable | W35 correctness 0.9375 vs W34 0.625; +0.3125 across 5/5 seeds |
| `frozen_basis` | Attenuation regime with reduced basis history | W35 still improves but only +0.2500; 4 reroutes rather than 5 |
| `all_anchor_compromised` | Hard falsifier: all basis directions move wrong together | W35 cannot recover; delta = 0.000 |

The benchmark compares W21, W33, W34, and W35 on the same Phase67-
derived outside-information regimes.  W21/W33 establish the old
explicit capsule line; W34 establishes the live-aware multi-anchor
abstention baseline; W35 tests whether dense basis history safely
turns abstention into a correct reroute.

---

## 4. Empirical Results

All numbers below are mean/min/max-identical across seeds
`(11, 17, 23, 29, 31)` unless otherwise noted.

### R-82-TRUST-SUBSPACE-SHIFT

- W21 correctness/trust precision: 0.625 / 0.625.
- W33 correctness/trust precision: 0.625 / 0.625.
- W34 correctness/trust precision: 0.625 / 1.000.
- W35 correctness/trust precision: **0.9375 / 1.000**.
- Delta W35-W34 correctness: **+0.3125**.
- W34 NO_CONSENSUS cells: 6.
- Actual W35 basis-history reroutes: 5.
- Mean W34 visible tokens: 13.0.
- Mean W35 visible tokens: 14.0.
- Mean overhead: 1.0 token/cell.
- Mean structured bits per 16-cell seed: 208,264.
- Mean structured state per visible W35 token: **13,016.5 bits**.

### Trivial and no-benefit controls

- `trivial_w35`: W35 = W34 byte-for-byte, overhead 0, correctness
  and trust precision 1.000.
- `no_anchor_disagreement`: W35 adds no correctness lift; correctness
  and trust precision 1.000; overhead 1 token/cell because nontrivial
  W35 audit state is enabled.

### Attenuation and falsifier regimes

- `frozen_basis`: W35 correctness 0.875 vs W34 0.625; delta +0.2500,
  trust precision 1.000, four reroutes.  The result shows basis
  weakening attenuates but does not eliminate the W35 gain.
- `all_anchor_compromised`: W21/W33/W34/W35 all remain at 0.625
  correctness and 0.625 trust precision; W35 reroutes 0 cells and
  delta W35-W34 = 0.000.  This is the hard falsifier:
  **W35-L-ALL-BASIS-COMPROMISED**.

---

## 5. Live / Two-Mac Evidence

Mac/live status checked on 2026-05-02:

- `localhost:11434` reachable; 8 Ollama model tags advertised.
- `192.168.12.191:11434` reachable; 5 Ollama model tags advertised.
- `192.168.12.248:11434` still timed out on `/api/tags`.

The full W34 xLLM pilot was started but exceeded the practical
turn-time budget after roughly nine minutes and was stopped.  The
replacement bounded live topology probe ran 5 gold-verifiable prompts
across local `qwen2.5:0.5b` and remote `qwen2.5:14b` using the W34
preflight/chat-template discipline:

- 10/10 probes responsive.
- Overall responsive accuracy: 0.700.
- Cross-host agreements: 2/5 prompts.
- Cross-host disagreements: 3/5 prompts.
- Gold-correlated disagreements: 3/3.
- Example disagreements: `sha` vs `sha256`, `\&` vs `>>`,
  `pseudopolynomial` vs `np`.

Artifact:
`vision_mvp/experiments/artifacts/phase82/xllm_bounded_live_topology_2026_05_02.json`.

This strengthens the live disagreement evidence but does not close
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE on the systematic magnitude
axis.  It is bounded two-host evidence, not true broader multi-host
evidence.

---

## 6. Theorem-Style Claims

### W35-1 — Verifier Boundary

**Claim**: The W35 trust-subspace dense envelope is mechanically
verifiable against 14 disjoint failure modes.
**Status**: proved by inspection + mechanically checked.
**Anchor**: `verify_trust_subspace_dense_ratification`;
`test_phase82_trust_subspace_dense_control.py`.

### W35-2 — Trivial Reduction

**Claim**: When the W35 trust-subspace layer is disabled and
manifest-v5 is disabled, W35 reduces to W34 byte-for-byte.
**Status**: empirical across 5/5 seeds.
**Anchor**: `trivial_w35_seed_sweep.json`.

### W35-3 — Conditional Basis-History Sufficiency

**Claim**: If W34 abstains because anchors disagree, and the W35
basis history contains a stable, high-margin trusted direction whose
current top_set remains consistent, W35 can convert abstention into a
correct reroute without trust-precision regression.
**Status**: proved-conditional + empirical on R-82-TRUST-SUBSPACE-
SHIFT.
**Measured**: correctness +0.3125 over W34, trust precision 1.000,
one-token overhead.

### W35-4 — Dense-State Audit Efficiency

**Claim**: W35 can transfer controller-verified structured state at
far higher density than visible context while preserving auditability.
**Status**: empirical + mechanically bounded.
**Measured**: mean 13,016.5 structured bits per visible W35 token on
R-82-TRUST-SUBSPACE-SHIFT.

### W35-L-ALL-BASIS-COMPROMISED

**Limitation**: If every registered basis direction moves together to
the same wrong answer, capsule-layer trust-subspace projection cannot
recover; it has no independent signal left.
**Status**: empirical falsifier + proof sketch by inspection.
**Measured**: W35-W34 correctness delta 0.000; trust precision remains
0.625 on `all_anchor_compromised`.

### W35-C-NATIVE-LATENT

**Conjecture remains open**: a true transformer-internal trust
subspace or hidden-state projection may separate regimes that W35
cannot, but this repo currently lacks architecture-specific access to
measure that honestly.

---

## 7. Master-Plan Consequence

Geometry-aware dense-control integration materially helped on one
specific blocker: W34's safe abstention can now be followed by an
audited dense-control reroute when basis history is stable.  Trust
and audit survived: the W35 envelope has 14 verifier modes, manifest-
v5 CID binding, and per-cell structured-state CIDs.  Bounded-context
efficiency improved in the density sense: one visible W35 token
carries roughly 13k bits of verified structured controller state.

The result does not broaden multi-host evidence beyond a two-host
bounded live fallback; Mac 2 remains unreachable.  Release readiness
improves because the stable/experimental boundary is cleaner and the
remaining blockers are sharper, not because W35 is product-stable.

The original thesis is materially stronger than W34 in a narrow,
important way: explicit capsule reasoning and geometry/dense-control
proxies now compose in one mechanism.  The deeper trust/semantics wall
remains: native-latent / model-internal trust subspace is still open.

---

## 8. Validation

- `python3 -m unittest vision_mvp.tests.test_phase82_trust_subspace_dense_control`
  — 22 tests pass.
- `python3 -m unittest vision_mvp.tests.test_phase82_trust_subspace_dense_control vision_mvp.tests.test_phase81_live_aware_multi_anchor vision_mvp.tests.test_phase80_trust_ewma_tracked vision_mvp.tests.test_coordpy_public_api`
  — 115 tests pass.
- `PYTHONHASHSEED=0 python3 -m unittest discover -s vision_mvp/tests`
  — 2480 tests pass in 244.338 s.
- Phase82 seed sweeps regenerated for all five banks under
  `vision_mvp/experiments/artifacts/phase82/`.
- `python3 -m py_compile vision_mvp/coordpy/team_coord.py vision_mvp/experiments/phase82_trust_subspace_dense_control.py vision_mvp/coordpy/__init__.py`
  passes.
- Public import check confirms `coordpy.sdk.v3.36` and W35 experimental
  symbols.
- `git diff --check` passes.
