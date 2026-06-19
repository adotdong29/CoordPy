# Results - CoordPy SDK v3.37 / W36
# Host-diverse trust-subspace guard + manifest-v6

Date: 2026-05-02.

W36 is a narrow hardening layer on top of W35.  W35 showed that a
controller-verified trust-subspace proxy can safely reroute some W34
NO_CONSENSUS abstentions.  W36 addresses the next blocker: a dense
projection can still be unsafe if all support is co-located on one
host, unavailable as live attestation, or spoofed by a compromised
host boundary.

The result is not native latent transfer.  It is an audited capsule-
layer proxy around host-diverse dense-control support.

---

## 1. New mechanism

The W36 family adds:

- `HostDiverseBasisEntry`
- `HostDiverseRatificationEnvelope`
- `HostDiverseRegistry`
- `W36HostDiverseResult`
- `HostDiverseTrustSubspaceOrchestrator`
- `select_host_diverse_projection`
- `verify_host_diverse_ratification`
- `build_trivial_host_diverse_registry`
- `build_host_diverse_registry`

W36 wraps W35 and forms host-attested basis entries from W35 basis
state plus W34 `LiveOracleAttestation` records.  A projection is
eligible only when its support has at least `min_distinct_hosts`
distinct registered healthy hosts and passes threshold/margin checks.
Unsafe or unverifiable branches reject or abstain.

Manifest-v6 binds:

1. W35 parent CID.
2. Host-diverse basis-state CID.
3. W36 projection-audit CID.
4. Registered host-topology CID.

---

## 2. Benchmark family

Phase83 introduces four small regimes:

| Regime | Purpose |
| --- | --- |
| `trivial_w36` | Byte-for-W35 preservation when W36 and manifest-v6 are disabled |
| `host_diverse_recover` | W35 ratifies unsafe support; W36 requires distinct healthy hosts and reroutes/abstains |
| `host_spoofed_consensus` | All directions move together on one unhealthy spoofed host; W36 should not claim correctness recovery |
| `no_live_attestation` | Falsifier: requiring host diversity without live attestations causes broad abstention and correctness loss |

The benchmark output now reports substrate/FIFO, W21, W33, W34, W35,
and W36, so the old explicit capsule line and the dense-control line
are compared in one place.

---

## 3. Empirical results

All seed sweeps use seeds `11,17,23,29,31`, `n_eval=16`.

### R-83-HOST-DIVERSE-RECOVER

Across 5/5 seeds:

- substrate/FIFO correctness: 0.000.
- W21 correctness: 1.000.
- W33/W34/W35 correctness: 0.625.
- W36 correctness: **0.9375**.
- `min_delta_correctness_w36_w35 = +0.3125`.
- W35 trust precision: 0.6667.
- W36 trust precision: **1.000**.
- `min_delta_trust_precision_w36_w35 = +0.3333`.
- W36 reroutes 5 cells and abstains 1 unsafe W35 ratification per
  seed.
- W36 overhead: 1 visible token/cell.
- Structured state density: about **13,948.5-13,959.0 bits per
  visible W36 token**.

This is the main W36 blocker-removal result.  It does not dominate
the old W21 explicit oracle baseline on this synthetic regime; the
scientific claim is narrower: W36 hardens the W33-W35 trust stack at
the host boundary and fixes a W35 unsafe-ratification case.

Artifact:
`vision_mvp/experiments/artifacts/phase83/host_diverse_recover_seed_sweep.json`.

### R-83-HOST-SPOOFED-CONSENSUS

Across 5/5 seeds:

- substrate/FIFO correctness: 0.000.
- W21/W33/W34/W35/W36 correctness: 0.625.
- W35 trust precision: 0.625.
- W36 trust precision: **1.000**.
- `min_delta_correctness_w36_w35 = 0.000`.
- `min_delta_trust_precision_w36_w35 = +0.375`.
- W36 reroutes 0 cells and abstains on 6 wrong W35 ratifications per
  seed.
- W36 overhead: 1 visible token/cell.
- Structured state density: about **13,733.0-13,759.5 bits per
  visible W36 token**.

This is a trust/audit improvement, not a correctness recovery.  It is
the named spoofed-host falsifier.

Artifact:
`vision_mvp/experiments/artifacts/phase83/host_spoofed_consensus_seed_sweep.json`.

### R-83-TRIVIAL-W36

Across 5/5 seeds:

- W36 = W35 byte-for-byte.
- correctness and trust precision remain 1.000 for W21/W33/W34/W35/W36.
- overhead = 0.

Artifact:
`vision_mvp/experiments/artifacts/phase83/trivial_w36_seed_sweep.json`.

### R-83-NO-LIVE-ATTESTATION

Across 5/5 seeds:

- W21/W33/W34/W35 correctness: 1.000.
- W36 correctness: 0.000.
- `min_delta_correctness_w36_w35 = -1.000`.
- W36 trust precision: 1.000 because it abstains.
- W36 abstains on all 16 cells/seed.

This is the important safety result: forcing a host-diverse guard
without live attestations is too conservative and destroys correctness.
W36 must be used only when the host-attestation path is real.

Artifact:
`vision_mvp/experiments/artifacts/phase83/no_live_attestation_seed_sweep.json`.

---

## 4. Live / two-Mac evidence

Fresh preflight on 2026-05-02:

- `localhost:11434`: reachable; 8 model tags advertised.
- `192.168.12.191:11434`: reachable; 5 model tags advertised.
- `192.168.12.248:11434`: `/api/tags` timed out at 5 seconds.

`phase81_xllm_preflight_only.py` with Phase83 output recorded:

- 9/10 model-host preflights passed.
- 1 unreachable host: `192.168.12.248`.

Artifact:
`vision_mvp/experiments/artifacts/phase83/xllm_preflight_only_2026_05_02.json`.

Bounded W36 live topology probe:

- Hosts: local `qwen2.5:0.5b` as producer candidate and remote
  `qwen2.5:14b` on `192.168.12.191` as adjudicator candidate.
- 5 gold-verifiable prompts.
- 10/10 calls responsive.
- responsive accuracy: 0.600.
- cross-host disagreement rate: **4/5 = 0.800**.
- gold-correlated disagreement rate: **4/4 = 1.000**.

Artifact:
`vision_mvp/experiments/artifacts/phase83/xllm_host_diverse_bounded_probe_2026_05_02.json`.

This is stronger two-reachable-host evidence than W35's bounded probe,
but it is still not three-host evidence.  Mac 2 remains unavailable.

---

## 5. Trust boundary

`verify_host_diverse_ratification` enumerates 14 W36 failure modes:

1. `empty_w36_envelope`
2. `w36_schema_version_unknown`
3. `w36_schema_cid_mismatch`
4. `w35_parent_cid_mismatch`
5. `w36_projection_branch_unknown`
6. `w36_basis_entry_unregistered_oracle`
7. `w36_basis_host_unregistered`
8. `w36_basis_score_out_of_range`
9. `w36_basis_health_out_of_range`
10. `w36_basis_state_cid_mismatch`
11. `w36_projection_top_set_unregistered`
12. `w36_host_diversity_requirement_invalid`
13. `w36_manifest_v6_cid_mismatch`
14. `w36_outer_cid_mismatch`

Cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36
trust boundary: **112 enumerated failure modes**.

---

## 6. Theory-forward claims

### W36-1 - verifier boundary

**Claim**: The W36 host-diverse envelope is mechanically verifiable
against 14 disjoint failure modes.

**Status**: proved by inspection + mechanically checked.

### W36-2 - trivial reduction

**Claim**: Disabled W36 + disabled manifest-v6 reduces to W35
byte-for-byte.

**Status**: empirical across 5/5 seeds.

### W36-3 - host-diverse sufficiency

**Claim**: If W35 ratifies a dense projection whose support is unsafe
but another high-score separated projection is supported by at least
two distinct healthy registered hosts, W36 can reroute or abstain
without trust-precision regression.

**Status**: proved-conditional + empirical on R-83-HOST-DIVERSE-
RECOVER.

### W36-4 - spoofed-host limitation

**Claim**: If every dense basis direction moves to the same wrong
answer on one unhealthy/spoofed host, host-diverse dense control cannot
recover correctness at the capsule layer.  The safe W36 behavior is
abstention.

**Status**: empirical + proof sketch by inspection.

### W36-L-NO-LIVE-ATTESTATION

**Claim**: A host-diverse guard without live attestations can preserve
trust precision by abstention but can destroy correctness.

**Status**: empirical falsifier.

### W36-C-NATIVE-LATENT

**Conjecture**: True transformer-internal trust-state projection may
separate regimes where all capsule-visible host/probe/evidence signals
are either absent or corrupted.  W36 narrows the audited proxy but does
not close this.

**Status**: conjectural and architecture-dependent.

---

## 7. Release boundary

Release readiness improved because:

- W36 is strictly additive and experimental.
- Stable runtime contract is unchanged.
- SDK version and package version are bumped.
- Success bar, results note, theorem registry, README, START_HERE,
  master plan, changelog, and paper marker are updated.
- Focused regression remains green.

Release readiness is not fully closed because:

- W36 is not native latent transfer.
- Mac 2 remains unavailable.
- True three-host live evidence remains open.
- W36 can be unsafe when live attestations are absent.
- W21 still dominates W36 on one synthetic Phase83 recovery metric,
  so W36 is a trust-stack hardening result, not a universal successor
  to every older explicit-capsule baseline.
