# Results - CoordPy SDK v3.39 / W38
# Disjoint cross-source consensus-reference trajectory-divergence
# adjudication + manifest-v8 CID

Date: 2026-05-02.

W38 is a **collusion-bounding** layer on top of W37.  W37's deepest open
wall, ``W37-L-MULTI-HOST-COLLUSION-CAP``, is: *if two registered hosts
simultaneously emit a coordinated wrong top_set across at least
``min_anchored_observations`` cells, the trajectory crosses the anchored
thresholds and W37 can be made to reroute on the wrong top_set.*  W37
cannot break this at the capsule layer.

W38 raises the capsule-layer adversary bar from "compromise 2 of N
trajectory hosts" to "compromise 2 of N trajectory hosts **and** the
disjoint registered consensus reference".  The mechanism: at every cell
where W37 chooses to reroute on the trajectory-anchored top_set, W38
cross-checks that top_set against a controller-pre-registered
``ConsensusReferenceProbe`` whose host topology is **mechanically
disjoint** from the W37 trajectory hosts (the registry refuses to
construct otherwise via :class:`DisjointTopologyError`).  When the W37
candidate top_set diverges from the consensus reference by >=
``divergence_margin_min`` (Jaccard), W38 abstains via the
``CONSENSUS_DIVERGENCE_ABSTAINED`` branch.

W38 does **not** close W37-L-MULTI-HOST-COLLUSION-CAP and it does
**not** close native latent.  It bounds the collusion attack
conditional on the existence of an uncompromised disjoint consensus
reference.  When that reference is itself compromised in lock-step,
W38 cannot recover; this is the new
**W38-L-CONSENSUS-COLLUSION-CAP** limitation theorem (the W38 analog
of W34-L-MULTI-ANCHOR-CAP and W37-L-MULTI-HOST-COLLUSION-CAP).

The result is not native latent transfer.  It is a closed-form,
zero-parameter, audited capsule-layer proxy with a mechanically
enforced disjoint-topology precondition.

---

## 1. New mechanism

The W38 family adds:

- ``ConsensusReferenceProbe``
- ``DisjointConsensusReferenceRatificationEnvelope``
- ``DisjointConsensusReferenceRegistry``
- ``W38DisjointConsensusReferenceResult``
- ``DisjointConsensusReferenceOrchestrator``
- ``DisjointTopologyError``
- ``select_disjoint_consensus_divergence``
- ``verify_disjoint_consensus_reference_ratification``
- ``build_trivial_disjoint_consensus_registry``
- ``build_disjoint_consensus_registry``

W38 wraps W37.  At each cell, W38:

1. Reads W37's last-cell decision and projection top_set.
2. Asks the registered consensus-reference provider for a
   :class:`ConsensusReferenceProbe` for the current cell.
3. Calls :func:`select_disjoint_consensus_divergence`, which returns
   one of:

   * ``CONSENSUS_NO_REFERENCE`` -- no probe registered;
   * ``CONSENSUS_NO_TRIGGER`` -- W37 did not reroute, nothing to
     cross-check;
   * ``CONSENSUS_REFERENCE_WEAK`` -- the probe's
     ``consensus_strength`` is below the registry's
     ``consensus_strength_min``;
   * ``CONSENSUS_DIVERGENCE_ABSTAINED`` -- W37 rerouted but the
     candidate top_set diverges from the consensus reference top_set
     by >= ``divergence_margin_min``;
   * ``CONSENSUS_RATIFIED`` -- W37 rerouted and the candidate
     top_set agrees with the consensus reference.

4. Seals the decision in a manifest-v8 envelope.

Manifest-v8 binds:

1. W37 parent CID.
2. Consensus reference state CID (over the per-cell probe).
3. Divergence audit CID.
4. Consensus topology CID (the registered consensus host/oracle topology
   plus the trajectory host set, so any swap is detected).
5. Consensus probe CID (per-cell probe content).

The disjoint-topology precondition is enforced in *two* places:
- :class:`DisjointConsensusReferenceRegistry`'s ``__post_init__``
  raises :class:`DisjointTopologyError` when ``consensus_host_ids ∩
  trajectory_host_ids ≠ ∅``;
- :func:`verify_disjoint_consensus_reference_ratification` rejects an
  envelope whose ``consensus_topology_cid`` matches a registered
  topology with overlapping host sets
  (``w38_disjoint_topology_violation`` failure mode).

---

## 2. Benchmark family

Phase 85 introduces five small regimes:

| Regime | Purpose |
| --- | --- |
| ``trivial_w38`` | Byte-for-W37 preservation when consensus is disabled, divergence-abstain is disabled, and manifest-v8 is disabled |
| ``colluded_cross_host_trajectory`` | Prefix half: 3 hosts attest, W37 trajectory accumulates anchored observations.  Recovery half: only mac1 attests; W37 reroutes on the anchored top_set; the disjoint consensus reference disagrees and W38 abstains |
| ``no_collusion_consensus_agrees`` | Consensus reference agrees with W37 reroute; W38 ratifies; no regression versus W37 |
| ``consensus_also_compromised`` | Disjoint consensus reference is itself compromised (emits the same top_set as the colluding trajectory hosts); W38 cannot recover; W38-L-CONSENSUS-COLLUSION-CAP fires |
| ``no_consensus_reference`` | Consensus probe always returns ``None``; W38 returns ``CONSENSUS_NO_REFERENCE``; correctness/trust-precision unchanged from W37 |

The bench correctness is measured against a per-cell *bench gold* that
the W38 threat model treats as the consensus reference's claim (in the
colluded bank, the recovery-half bench gold is the disjoint decoy that
the trajectory hosts collude against).  The bench output reports
substrate/FIFO, W21, W33, W34, W35, W36, W37, and W38 in one place so
the old explicit capsule line, the dense-control / geometry line, the
W37 cross-cell trajectory line, and the W38 disjoint-consensus line
can be compared at a glance.

The phase 85 bank uses ``build_phase67_bank("outside_resolves",
n_replicates=4)`` interleaved by family.

---

## 3. Empirical results

All seed sweeps use seeds ``11, 17, 23, 29, 31``, ``n_eval=16``.

### R-85-COLLUDED-CROSS-HOST-TRAJECTORY (load-bearing)

Across 5/5 seeds:

- substrate/FIFO correctness: 0.0.
- W21 correctness: 0.5 (commits to colluded answer in recovery half).
- W21 trust precision: **0.5** (trust dropped by collusion).
- W33/W34/W35 trust precision: 0.5 (same as W21; no recovery).
- W36 trust precision: **1.000** (abstains on the recovery half via
  host diversity).
- W37 trust precision: **0.5** (reroutes on the colluded
  trajectory-anchored top_set in the recovery half).
- W38 trust precision: **1.000**.
- ``min_delta_trust_precision_w38_w37 = +0.500``.
- ``max_delta_trust_precision_w38_w37 = +0.500``.
- W38 reroutes 0 cells, abstains via DIVERGENCE on 8 cells/seed,
  ratifies 0 cells, no_reference 0.
- W38 overhead: 1 visible token/cell.
- Structured state density: about **9,072 bits per visible W38 token**.

This is the load-bearing W38 collusion-bounding result.  It is *not*
closure of W37-L-MULTI-HOST-COLLUSION-CAP in general; it is the
strongest honest capsule-layer bound this repo can support without
native latent transfer.  The bound is valid only when an uncompromised
disjoint consensus reference is registered.

Artifact:
``vision_mvp/experiments/artifacts/phase85/colluded_cross_host_trajectory_seed_sweep.json``.

### R-85-TRIVIAL-W38

Across 5/5 seeds:

- W38 = W37 byte-for-byte.
- correctness and trust precision identical to W37.
- overhead = 0.
- ``all_byte_equivalent_w38_w37 = true``.

Artifact:
``vision_mvp/experiments/artifacts/phase85/trivial_w38_seed_sweep.json``.

### R-85-NO-COLLUSION-CONSENSUS-AGREES

Across 5/5 seeds:

- W38 correctness: same as W37 (delta = 0).
- W38 trust precision: same as W37 (delta = 0).
- ``min_delta_correctness_w38_w37 = 0.0``.
- ``min_delta_trust_precision_w38_w37 = 0.0``.

Artifact:
``vision_mvp/experiments/artifacts/phase85/no_collusion_consensus_agrees_seed_sweep.json``.

### R-85-CONSENSUS-ALSO-COMPROMISED (W38-L-CONSENSUS-COLLUSION-CAP)

Across 5/5 seeds:

- W38 trust precision: 0.5 (cannot recover; consensus reference is
  itself compromised in lock-step with the colluding trajectory hosts).
- ``min_delta_trust_precision_w38_w37 = 0.0``.
- ``max_delta_trust_precision_w38_w37 = 0.0``.

This is the named W38-L-CONSENSUS-COLLUSION-CAP limitation theorem
firing.  Closure requires native-latent evidence outside the capsule
layer (W38-L-NATIVE-LATENT-GAP) or a 3+-host disjoint consensus
topology (W38-C-MULTI-HOST), which remains hardware-bounded until
Mac 2 (or a third reachable host) joins the lab topology.

Artifact:
``vision_mvp/experiments/artifacts/phase85/consensus_also_compromised_seed_sweep.json``.

### R-85-NO-CONSENSUS-REFERENCE

Across 5/5 seeds:

- W38 correctness: same as W37 (delta = 0).
- W38 trust precision: same as W37 (delta = 0).
- W38 returns ``CONSENSUS_NO_REFERENCE`` on every cell where the
  registered provider yields ``None``.

The W38 audit envelope still records the NO_REFERENCE branch -- the
audit trail is load-bearing even when the cross-check has nothing to
compare against.

Artifact:
``vision_mvp/experiments/artifacts/phase85/no_consensus_reference_seed_sweep.json``.

---

## 4. Live / two-Mac evidence

Fresh preflight on 2026-05-02:

- ``localhost:11434``: reachable; 8 model tags advertised.
- ``192.168.12.191:11434``: reachable; 5 model tags advertised
  (qwen2.5:14b, qwen2.5:14b-32k, qwen2.5-coder:14b-32k,
  qwen2.5-coder:14b, qwen3.5:35b).
- ``192.168.12.248:11434``: ``/api/tags`` timed out at 5 seconds;
  ``ping`` fails with "Host is down"; ARP entry incomplete.  Mac 2
  remains ARP-incomplete for the **31st milestone in a row**.

Bounded W38 cross-source consensus probe
(``phase85_xllm_consensus_probe.py``):

- Trajectory hosts: ``mac1`` (localhost ``gemma2:9b``) and
  ``mac_remote`` (192.168.12.191 ``qwen2.5:14b``).
- Disjoint consensus host: ``mac_consensus`` (192.168.12.191
  ``qwen3.5:35b`` -- a different model class on the same physical
  host; defensible weak proxy for capsule-layer disjointness, *not*
  a true 3-host disjoint topology).
- 8 gold-verifiable one-word prompts at temperature 0,
  ``num_predict=4``, stop tokens.

Per-probe results recorded in
``vision_mvp/experiments/artifacts/phase85/xllm_consensus_probe_2026_05_02.json``.

This live probe materially strengthens the capsule-layer
disjoint-consensus motivation but does **not** close W38-C-MULTI-HOST
(true 3-host disjoint topology, hardware-bounded) and does **not**
close W38-L-CONSENSUS-COLLUSION-CAP (no live colluded scenario was
attempted; the live probe is gold-correlated agreement evidence,
not an attack-recovery measurement).

---

## 5. Trust boundary

``verify_disjoint_consensus_reference_ratification`` enumerates 14
W38 failure modes:

1. ``empty_w38_envelope``
2. ``w38_schema_version_unknown``
3. ``w38_schema_cid_mismatch``
4. ``w37_parent_cid_mismatch``
5. ``w38_projection_branch_unknown``
6. ``w38_consensus_host_unregistered``
7. ``w38_consensus_oracle_unregistered``
8. ``w38_disjoint_topology_violation``
9. ``w38_consensus_strength_out_of_range``
10. ``w38_divergence_threshold_invalid``
11. ``w38_consensus_state_cid_mismatch``
12. ``w38_consensus_probe_cid_mismatch``
13. ``w38_consensus_topology_cid_mismatch``
14. ``w38_manifest_v8_cid_mismatch`` (and ``w38_outer_cid_mismatch``
    co-defined)

Cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36 +
W37 + W38 trust boundary: **140 enumerated failure modes**.

---

## 6. Theory-forward claims

### W38-1 - verifier boundary

**Claim**: The W38 disjoint-consensus envelope is mechanically
verifiable against 14 disjoint failure modes.

**Status**: proved by inspection + mechanically checked.

### W38-2 - trivial reduction

**Claim**: Disabled consensus + disabled divergence-abstain + disabled
manifest-v8 reduces to W37 byte-for-byte.

**Status**: empirical across 5/5 seeds.

### W38-3 - disjoint-consensus collusion bound (load-bearing)

**Claim**: If the current cell would reroute a W37 trajectory-anchored
top_set produced by two colluding hosts, AND a registered
``ConsensusReferenceProbe`` exists whose host topology is
mechanically disjoint from the W37 trajectory hosts, AND that probe's
``consensus_strength`` is at least ``consensus_strength_min``, AND its
top_set diverges from the W37 candidate top_set by at least
``divergence_margin_min``, THEN W38 abstains via DIVERGENCE_ABSTAINED
without trust-precision regression.

**Status**: proved-conditional + empirical on
R-85-COLLUDED-CROSS-HOST-TRAJECTORY (5/5 seeds, +0.500 trust
precision gain over W37, trust precision = 1.000).

### W38-4 - disjoint-topology mechanical enforcement

**Claim**: A ``DisjointConsensusReferenceRegistry`` whose
``consensus_host_ids ∩ trajectory_host_ids ≠ ∅`` cannot be
constructed; the verifier additionally rejects envelopes whose
registered topology matches an overlapping host set
(``w38_disjoint_topology_violation``).

**Status**: proved by inspection + mechanically tested.

### W38-L-CONSENSUS-COLLUSION-CAP

**Claim**: When the disjoint consensus reference is itself compromised
in lock-step with the colluding trajectory hosts (i.e. emits the same
wrong top_set), W38 cannot recover at the capsule layer.  This is the
W38 analog of W34-L-MULTI-ANCHOR-CAP and W37-L-MULTI-HOST-COLLUSION-CAP
and is closed only by transformer-internal evidence outside the capsule
layer or by a 3+-host disjoint consensus topology.

**Status**: proved-conditional limitation theorem + empirical on
R-85-CONSENSUS-ALSO-COMPROMISED (5/5 seeds, delta = 0).

### W38-L-DISJOINT-CONSENSUS-REQUIRED

**Claim**: If no consensus reference probe is registered for the
current cell, W38 cannot detect divergence and falls through to W37's
decision (NO_REFERENCE branch).  W38 is insufficient for any regime
where a disjoint consensus reference cannot be sourced.

**Status**: proved by inspection + empirical on
R-85-NO-CONSENSUS-REFERENCE.

### W38-C-NATIVE-LATENT

**Conjecture**: True transformer-internal trust-state projection may
separate regimes where all capsule-visible host/probe/evidence/
trajectory/consensus signals are either absent or coordinated.  W38
narrows the audited proxy further along the cross-source axis but
does not close this.

**Status**: conjectural and architecture-dependent.

### W38-C-MULTI-HOST

**Conjecture**: A 3+ reachable-host topology with at least one host
genuinely disjoint from the trajectory hosts (not just a different
model on the same physical host) would let the consensus reference be
sourced from a true off-cluster oracle, defeating the
``W38-L-CONSENSUS-COLLUSION-CAP`` collusion attack at the capsule
layer.  Hardware-bounded until Mac 2 (or a third reachable host)
joins the topology.

**Status**: conjectural, infrastructure-bounded.

---

## 7. Release boundary

Release readiness improved because:

- W38 is strictly additive and experimental.
- Stable runtime contract is unchanged.
- SDK version (``coordpy.sdk.v3.39``) and package version (``0.5.12``)
  are bumped.
- The repo's ``vision_mvp.__version__`` and ``pyproject.toml``
  ``project.version`` are now both ``0.5.12`` (the lingering 0.5.9
  vs 0.5.11 misalignment from earlier milestones is closed).
- Success bar, results note, theorem registry, README, START_HERE,
  master plan, changelog, and paper marker are updated.
- Focused W22..W38 regression at **594/594 phase69-85** is green;
  the broad ``pytest vision_mvp/tests`` sweep is run during the
  milestone with the count recorded.
- The cumulative enumerated trust boundary is now **140 failure
  modes**.
- The stable-vs-experimental boundary is final: every W22..W38
  symbol is exported under ``__experimental__`` and the stable
  ``RunSpec → run report`` runtime contract is byte-for-byte
  unchanged.

Release readiness is not fully closed because:

- W38 is not native latent transfer.
- Mac 2 remains unavailable (31st milestone).
- True three-host live evidence remains open
  (W38-C-MULTI-HOST, W38-C-NATIVE-LATENT).
- W38-L-CONSENSUS-COLLUSION-CAP cannot be closed at the capsule
  layer.
- W21 still dominates W38 on regimes where multi-oracle quorum is
  enough by itself; W38 is a trust-stack hardening result for the
  cross-cell, single-host recovery case under capsule-layer
  collusion threat, not a universal successor to every older
  explicit-capsule baseline.
