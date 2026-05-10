# Pre-committed success criterion — W43 Product-Manifold Capsule

> Programme step: post-CoordPy 0.5.20. Mints axis 40 in the Context
> Zero programme. Strictly additive on top of the released v3.43 line.
> Honest scope: capsule-layer only. The transformer-internal
> mechanisms (mixed-curvature attention, collective KV pooling, full
> Grassmannian homotopy) remain conjectural.

## Mechanism

W43 introduces the **Product-Manifold Capsule (PMC)** — the first
capsule-native layer in CoordPy that decomposes each cell's
coordination state across a product manifold with six interlocking
channels:

* a **hyperbolic branch channel** (Poincare-disk-style encoding of
  the parent-DAG depth + branch path)
* a **spherical consensus channel** (unit-norm signature over the
  cell's claim_kinds, giving cosine-distance consensus)
* a **euclidean attribute channel** (linear vector of cell
  attributes)
* a **factoradic route channel** (Lehmer-coded permutation of role
  arrival order; ceil(log2(n!)) bits at zero visible-token cost)
* a **subspace state channel** (bounded-rank orthonormal basis
  representing the cell's admissible-interpretation subspace; an
  approximation of one point on the Grassmannian Gr(k, d))
* a **causal-clock channel** (Lamport vector clock + dependency-
  closure admissibility check)

W43 is strictly additive on top of W42: when configured trivially
(`pmc_enabled=False`, `manifest_v13_disabled=True`, all
`abstain_on_*=False`), the orchestrator reduces to the W42 result
byte-for-byte (the **W43-L-TRIVIAL-PASSTHROUGH** falsifier).

## Pre-committed hypotheses (H1..H10)

Each hypothesis is testable from the bundled R-90 benchmark family
(``coordpy.r90_benchmark``) and its accompanying pytest tests
(``tests/test_r90_benchmark.py``). Pre-committed seed set:
``(0, 1, 2, 3, 4)``.

* **H1 — Trivial passthrough is byte-for-W42.**
  Family `r90_trivial_pmc`: all three arms (W42 passthrough, W42
  active, W43 PMC active) achieve `passthrough_ok = 1.0` across
  all 5 seeds. The trivially-configured W43 orchestrator emits no
  envelope and adds zero overhead tokens.
  **Pass if**: `min == max == 1.0` for all three arms.

* **H2 — Hyperbolic round-trip is bit-perfect.**
  Family `r90_long_branch`: at depth = 12, the W43 hyperbolic
  channel encodes the first 2*dim path bits and round-trips them
  bit-for-bit.
  **Pass if**: `branch_round_trip_ok = 1.0` across 5/5 seeds.

* **H3 — Spherical channel strictly improves trust precision on
  consensus cycles.**
  Family `r90_consensus_cycle`: half the cells diverge from the
  registered expectation. The W42 baseline achieves chance-level
  precision (~0.5..0.6 across seeds); W43 ratifies on agreement
  and abstains on divergence, achieving 1.0.
  **Pass if**: `pmc.mean - w42.mean >= 0.25`.

* **H4 — Cram-frontier strictly advances per cell.**
  Family `r90_compact_state_transfer`: the W43 envelope adds
  7 * 256 + ceil(log2(n!)) structured bits per cell at 1
  visible-token overhead, vs the W42 envelope's 6 * 256 bits per
  cell at 1 visible-token overhead.
  **Pass if**: `pmc.min - w42.mean >= 256` (strictly more bits per
  visible token than W42).

* **H5 — Causal violations are perfectly detected.**
  Family `r90_causal_violation`: cells whose Lamport vector clocks
  break the partial order are routed to
  `pmc_causal_violation_abstained`.
  **Pass if**: `pmc.min == max == 1.0` AND `w42.max == 0.0`.

* **H6 — Factoradic side-channel carries log2(n!) bits at zero
  visible cost.**
  Family `r90_routing_compression`: with n_roles=8, the factoradic
  channel adds ceil(log2(8!)) = 16 bits per envelope.
  **Pass if**: `pmc.min == max == 16.0`.

* **H7 — No false abstention on the linear-flow falsifier.**
  Family `r90_linear_flow_falsifier`: a regime where the geometry
  adds nothing must NOT trigger spurious abstentions.
  **Pass if**: `pmc.min == max == 1.0`.

* **H8 — Subspace-drift channel strictly improves trust precision.**
  Family `r90_subspace_drift`: half the cells drift to an
  orthogonal subspace. W42 cannot detect it; W43 abstains via
  `pmc_subspace_drift_abstained`.
  **Pass if**: `pmc.min == max == 1.0` AND
  `pmc.mean - w42.mean >= 0.25`.

* **H9 — Verifier enumerates >=18 disjoint failure modes.**
  ``verify_product_manifold_ratification`` returns the empty-
  envelope failure as the first mode and recomputes every component
  CID; tampering with any subfield is detected through one of the
  disjoint named modes (``w43_*_cid_mismatch``,
  ``w43_*_cid_invalid``, ``w43_token_accounting_invalid``,
  ``w43_spherical_agreement_invalid``,
  ``w43_subspace_drift_invalid``,
  ``w43_decision_branch_unknown``,
  ``w43_schema_version_unknown``,
  ``w43_schema_cid_mismatch``,
  ``w42_parent_cid_mismatch``,
  ``w43_role_handoff_signature_cid_mismatch``,
  ``w43_outer_cid_mismatch``,
  ``w43_manifest_v13_cid_mismatch``,
  ``w43_manifold_witness_cid_mismatch``,
  ``w43_manifold_audit_cid_mismatch``,
  ``w43_causal_clock_cid_mismatch``,
  ``w43_subspace_basis_cid_mismatch``,
  ``w43_factoradic_route_cid_mismatch``,
  ``w43_manifold_state_cid_mismatch``).
  **Pass if**: a successful verification reports `n_checks >= 18`.

* **H10 — Stable SDK contract preserved.**
  CoordPy 0.5.20's stable smoke driver
  (`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED" with
  the W43 module on disk. The W43 surface is reachable only
  through `from coordpy.product_manifold import ...` and is
  **not** exported under `coordpy.__experimental__` at this
  milestone, so the stable v0.5.20 wheel's public surface is
  byte-for-byte unchanged.
  **Pass if**: smoke driver exits with "ALL CHECKS PASSED".

## Outcome buckets

* **Strong success** — H1..H10 all pass cleanly. The W43 line is
  declared a **research milestone** (not a release candidate); the
  W43 module ships in the source tree but is held outside the
  stable SDK contract until the cross-host live evidence is
  acquired.

* **Partial success** — H1..H7 pass; H8 or H9 partially pass (e.g.
  the verifier enumerates >=10 but <18 modes). The W43 line is
  recorded as a research artefact with a documented gap.

* **Failure** — any of H1..H7 fails, OR H10 fails (the stable
  SDK contract regresses). The W43 module is held back from the
  master tree until the failure is closed.

## Limitation theorems pre-committed at the milestone

* **W43-L-TRIVIAL-PASSTHROUGH** (code-backed) — the trivially-
  configured PMC reduces to W42 byte-for-byte.

* **W43-L-DUAL-CHANNEL-COLLUSION-CAP** (proved-conditional) — when
  the adversary controls BOTH the spherical-channel signature AND
  the registered subspace basis, W43 cannot recover.

* **W43-L-FORGED-CAUSAL-CLOCK-CAP** (proved-conditional) — when an
  adversary forges a monotone Lamport vector clock that satisfies
  the closure property but does NOT match the registered topology,
  W43 detects via `causal_clock_cid_mismatch` (when the registered
  topology hash is bound to the policy entry) but cannot infer the
  true ordering from the capsule layer alone.

* **W43-L-OBLIVIOUS-FACTORADIC-CAP** (proved-conditional) — when
  role names are silently permuted in the registered universe, the
  factoradic channel routes via the registered permutation, not
  the intended one. Cross-host topology disambiguation is out of
  scope at the capsule layer.

## Conjectures left open

* **W43-C-MIXED-CURVATURE-LATENT** — full transformer-internal
  mixed-curvature attention requires substrate access not available
  to the capsule layer.

* **W43-C-COLLECTIVE-KV-POOLING** — host-collective KV-cache
  sharing requires transformer-internal hooks not exposed at the
  capsule layer.

* **W43-C-FULL-GRASSMANNIAN-HOMOTOPY** — a true Gr(k, d) homotopy
  requires continuous representation that the bounded basis proxy
  approximates only locally.
