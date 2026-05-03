# Results - CoordPy SDK v3.38 / W37
# Anchor-cross-host basis-trajectory ratification + manifest-v7

Date: 2026-05-02.

W37 is a cross-cell hardening layer on top of W36.  W36 abstains
whenever the current cell has fewer than ``min_distinct_hosts`` healthy
attested hosts -- even when the single remaining host has been
independently anchored across multiple earlier cells by other healthy
hosts.  W37 makes that historical cross-host anchoring a typed audited
proxy: a single-host current cell can be safely rerouted iff its
``(host_id, oracle_id, top_set)`` trajectory entry has been cross-host
anchored above threshold across at least
``min_anchored_observations`` historical cells with at least
``min_trajectory_anchored_hosts`` distinct anchor hosts.

The result is not native latent transfer.  It is a closed-form,
zero-parameter, audited capsule-layer proxy around cross-cell host
trajectory state.

---

## 1. New mechanism

The W37 family adds:

- `CrossHostBasisTrajectoryEntry`
- `CrossHostBasisTrajectoryRatificationEnvelope`
- `CrossHostBasisTrajectoryRegistry`
- `W37CrossHostTrajectoryResult`
- `CrossHostBasisTrajectoryOrchestrator`
- `select_cross_host_trajectory_projection`
- `verify_cross_host_trajectory_ratification`
- `build_trivial_cross_host_trajectory_registry`
- `build_cross_host_trajectory_registry`

W37 wraps W36.  At each cell, W37:

1. Reads the current W36 host-attested basis without mutating
   trajectory state.
2. Snapshots the prior trajectory state (the *select-before-update*
   discipline).
3. Calls `select_cross_host_trajectory_projection`, which scores each
   ``(host_id, oracle_id, top_set)`` triple from the current cell
   against the trajectory state and returns one of:
   ``REROUTED`` (single-host current cell with a cross-host anchored
   historical trajectory above threshold), ``DISAGREEMENT`` (two or
   more current top_sets are simultaneously trusted within margin),
   ``NO_HISTORY``, or ``UNSAFE`` (a candidate exists but fails one of
   the anchored-observation, anchored-host, or threshold gates).
4. Sealing the decision in a manifest-v7 envelope.
5. Records the current cell's observation only when the cell has
   >= 2 distinct healthy attested hosts (otherwise the EWMA stays
   pinned to its prior value -- a host is not penalised for being
   the only reachable host when the live infrastructure offers no
   co-attesters).

Manifest-v7 binds:

1. W36 parent CID.
2. Cross-host trajectory state CID.
3. Trajectory audit CID.
4. Trajectory topology CID (the union of registered hosts and
   registered anchor hosts).

---

## 2. Benchmark family

Phase 84 introduces five small regimes:

| Regime | Purpose |
| --- | --- |
| `trivial_w37` | Byte-for-W36 preservation when trajectory is disabled, single-host reroute is disabled, and manifest-v7 is disabled |
| `single_host_trajectory_recover` | Prefix half: 3 healthy hosts attest each cell; trajectory accumulates cross-host anchored entries.  Recovery half: only mac1 attests, W36 abstains, W37 reroutes on the trajectory-anchored top_set |
| `trajectory_disagreement` | Prefix half cross-host anchors a gold top_set; recovery half is single-host *and* the oracles flip to a compromised top_set.  W37 must not reroute on the historically-anchored top_set because the current basis disagrees |
| `no_trajectory_history` | Every cell is single-host (no anchoring ever forms).  W37 must reduce to W36 |
| `poisoned_trajectory` | Single-host trajectory with no cross-host anchoring at any cell.  W37 must not satisfy the ``min_trajectory_anchored_hosts`` requirement |

The benchmark output reports substrate/FIFO, W21, W33, W34, W35, W36,
and W37 in one place so the old explicit capsule line and the
trajectory-anchored dense-control line can be compared.

The phase 84 bank uses `build_phase67_bank("outside_resolves",
n_replicates=4)` interleaved by family so the prefix and recovery
halves cover the same gold top_sets.

---

## 3. Empirical results

All seed sweeps use seeds `11,17,23,29,31`, `n_eval=16`.

### R-84-SINGLE-HOST-TRAJECTORY-RECOVER

Across 5/5 seeds:

- substrate/FIFO correctness: 0.000.
- W21 correctness: 1.000.
- W33/W34/W35 correctness: 1.000.
- W36 correctness: **0.500**.
- W37 correctness: **1.000**.
- `min_delta_correctness_w37_w36 = +0.500`.
- `max_delta_correctness_w37_w36 = +0.500`.
- W36 trust precision: 1.000.
- W37 trust precision: **1.000**.
- W37 reroutes 8 cells per seed and abstains on 0 unsafe ratifications.
- W37 overhead: 1 visible token/cell.
- Structured state density: about **29,564.5 bits per visible W37
  token** on seed 11.

This is the load-bearing W37 blocker-removal result: when a cell
falls back to a single healthy host, W37 uses the historical
cross-host anchored trajectory to safely reroute where W36 had to
abstain.  The 0.500 W36 baseline reflects the prefix-half W36
ratifications only; W37 recovers all 8 recovery-half cells.

Artifact:
`vision_mvp/experiments/artifacts/phase84/single_host_trajectory_recover_seed_sweep.json`.

### R-84-TRIVIAL-W37

Across 5/5 seeds:

- W37 = W36 byte-for-byte.
- correctness and trust precision remain 1.000 for every layer.
- overhead = 0.
- `all_byte_equivalent_w37_w36 = true`.

Artifact:
`vision_mvp/experiments/artifacts/phase84/trivial_w37_seed_sweep.json`.

### R-84-NO-TRAJECTORY-HISTORY

Across 5/5 seeds:

- W36 correctness: 0.000 (every cell is single-host so W36 abstains).
- W37 correctness: 0.000 (no anchoring ever forms; W37 preserves
  W36 abstention).
- `min_delta_correctness_w37_w36 = 0.000`.
- W37 trust precision: 1.000 (abstaining preserves trust precision).

This is the named no-history falsifier: without any cross-host
anchored trajectory, W37 cannot reroute and must preserve W36
behavior.

Artifact:
`vision_mvp/experiments/artifacts/phase84/no_trajectory_history_seed_sweep.json`.

### R-84-POISONED-TRAJECTORY

Across 5/5 seeds:

- W36 correctness: 0.000 (single-host every cell ⇒ host-diversity
  fails ⇒ W36 abstains).
- W37 correctness: 0.000 (single-host trajectory ⇒ no
  cross-host anchoring ever forms ⇒ W37 cannot pass the
  ``min_trajectory_anchored_hosts`` requirement ⇒ preserves
  abstention).
- `min_delta_correctness_w37_w36 = 0.000`.
- W37 trust precision: 1.000.

This is the named poisoned-trajectory falsifier.  Even though the
single host produces a consistent (and therefore *self-anchored*)
top_set across cells, the cross-host requirement explicitly prevents
single-host evidence from ever becoming trajectory-trusted.

Artifact:
`vision_mvp/experiments/artifacts/phase84/poisoned_trajectory_seed_sweep.json`.

### R-84-TRAJECTORY-DISAGREEMENT

Across 5/5 seeds:

- W36 correctness: 0.500 (prefix half ratifies, recovery half
  abstains).
- W37 correctness: 0.500 (recovery half: current basis emits the
  compromised top_set; trajectory anchors the prefix gold top_set;
  the current basis does not match the anchored trajectory key, so
  W37 returns NO_HISTORY for the recovery cells and preserves W36
  abstention).
- `min_delta_correctness_w37_w36 = 0.000`.
- W37 trust precision: 1.000.

This is the disagreement falsifier.  W37 explicitly does not commit
the historically-anchored top_set against the current cell's
disagreeing basis.

Artifact:
`vision_mvp/experiments/artifacts/phase84/trajectory_disagreement_seed_sweep.json`.

---

## 4. Live / two-Mac evidence

Fresh preflight on 2026-05-02:

- `localhost:11434`: reachable; 8 model tags advertised.
- `192.168.12.191:11434`: reachable; 5 model tags advertised
  (qwen2.5:14b, qwen2.5:14b-32k, qwen2.5-coder:14b-32k,
  qwen2.5-coder:14b, qwen3.5:35b).
- `192.168.12.248:11434`: `/api/tags` timed out at 5 seconds
  (Mac 2 ARP-incomplete for the **30th milestone in a row**).

`phase81_xllm_preflight_only.py` recorded 9/10 model-host
preflight passes, 1 unreachable host.

Artifact:
`vision_mvp/experiments/artifacts/phase84/xllm_preflight_only_2026_05_02.json`.

Bounded W37 cross-host trajectory probe
(`phase84_xllm_trajectory_probe.py`):

- Hosts: local `gemma2:9b` and remote `qwen2.5:14b` on
  `192.168.12.191`.
- 8 gold-verifiable one-word prompts at temperature 0.
- 8/8 calls responsive on both hosts.
- 8/8 cross-host **anchored agreements** (both hosts emit the
  same lowercase first-token answer).
- 8/8 anchored agreements are **gold-correlated**.
- Anchored rate: **1.000**.
- Gold-correlated anchored rate: **1.000**.

Artifact:
`vision_mvp/experiments/artifacts/phase84/xllm_trajectory_probe_2026_05_02.json`.

This is the strongest two-reachable-host live trajectory evidence the
infrastructure currently supports: the W37 anchored-observation
discipline is empirically realisable on a real cross-architecture
gemma2/qwen2.5 host pair at temperature 0 on gold-verifiable prompts.
It does **not** close the W37-C-MULTI-HOST conjecture, which requires
three reachable hosts.

---

## 5. Trust boundary

`verify_cross_host_trajectory_ratification` enumerates 14 W37 failure
modes:

1. `empty_w37_envelope`
2. `w37_schema_version_unknown`
3. `w37_schema_cid_mismatch`
4. `w36_parent_cid_mismatch`
5. `w37_projection_branch_unknown`
6. `w37_trajectory_entry_unregistered_oracle`
7. `w37_trajectory_entry_unregistered_host`
8. `w37_trajectory_ewma_out_of_range`
9. `w37_trajectory_observation_count_invalid`
10. `w37_trajectory_state_cid_mismatch`
11. `w37_projection_top_set_unregistered`
12. `w37_trajectory_requirement_invalid`
13. `w37_trajectory_topology_cid_mismatch`
14. `w37_manifest_v7_cid_mismatch` (and
    `w37_outer_cid_mismatch` co-defined)

Cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36 +
W37 trust boundary: **126 enumerated failure modes**.

---

## 6. Theory-forward claims

### W37-1 - verifier boundary

**Claim**: The W37 cross-host trajectory envelope is mechanically
verifiable against 14 disjoint failure modes.

**Status**: proved by inspection + mechanically checked.

### W37-2 - trivial reduction

**Claim**: Disabled trajectory + disabled single-host reroute +
disabled manifest-v7 reduces to W36 byte-for-byte.

**Status**: empirical across 5/5 seeds.

### W37-3 - cross-host trajectory sufficiency

**Claim**: If the current cell has only one healthy attested host but
that host has a cross-host anchored trajectory above threshold across
at least ``min_anchored_observations`` historical cells with at least
``min_trajectory_anchored_hosts`` distinct anchor hosts, W37 can safely
reroute on the trajectory-anchored top_set without trust-precision
regression.

**Status**: proved-conditional + empirical on
R-84-SINGLE-HOST-TRAJECTORY-RECOVER (5/5 seeds, +0.500 correctness
gain over W36, trust precision = 1.000).

### W37-4 - poisoned-trajectory limitation

**Claim**: A trajectory accumulated on a single host without cross-
host anchoring can never satisfy the
``min_trajectory_anchored_hosts`` requirement.  W37 will not reroute
on such a trajectory.

**Status**: empirical falsifier on R-84-POISONED-TRAJECTORY
(W37 abstains 16/16 cells/seed across 5/5 seeds).

### W37-L-MULTI-HOST-COLLUSION-CAP

**Claim**: When two registered hosts simultaneously emit a coordinated
wrong top_set across at least ``min_anchored_observations`` cells, the
trajectory crosses the anchored thresholds and W37 can be made to
reroute on the wrong top_set.  The mechanism is the W37 analog of
W34-L-MULTI-ANCHOR-CAP and is closed only by transformer-internal
evidence outside the capsule layer.

**Status**: proved-conditional limitation theorem.

### W37-C-NATIVE-LATENT

**Conjecture**: True transformer-internal trust-state projection may
separate regimes where all capsule-visible host/probe/evidence/
trajectory signals are either absent or coordinated.  W37 narrows the
audited proxy further along the cross-cell axis but does not close
this.

**Status**: conjectural and architecture-dependent.

### W37-C-MULTI-HOST

**Conjecture**: A 3+ reachable-host topology would let the
``min_trajectory_anchored_hosts`` knob be raised to 3, defeating
two-host collusion attacks at the capsule layer.  Hardware-bounded
until Mac 2 (or a third reachable host) joins the topology.

**Status**: conjectural, infrastructure-bounded.

---

## 7. Release boundary

Release readiness improved because:

- W37 is strictly additive and experimental.
- Stable runtime contract is unchanged.
- SDK version (`coordpy.sdk.v3.38`) and package version (`0.5.11`) are
  bumped.
- Success bar, results note, theorem registry, README, START_HERE,
  master plan, changelog, and paper marker are updated.
- Focused W22..W37 regression remains green at 563/563 in the
  phase69-84 slice; the broader pytest sweep is run once during the
  milestone.

Release readiness is not fully closed because:

- W37 is not native latent transfer.
- Mac 2 remains unavailable (30th milestone).
- True three-host live evidence remains open
  (W37-C-MULTI-HOST, W37-C-NATIVE-LATENT).
- W37-L-MULTI-HOST-COLLUSION-CAP cannot be closed at the capsule
  layer.
- W21 still dominates W37 on regimes where multi-oracle quorum is
  enough by itself; W37 is a trust-stack hardening result for the
  cross-cell, single-host recovery case, not a universal successor
  to every older explicit-capsule baseline.
