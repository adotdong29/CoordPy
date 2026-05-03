# Results - Wevra SDK v3.40 / W39
# Multi-host disjoint quorum consensus-reference ratification +
# manifest-v9 CID + mutually-disjoint physical-host topology

Date: 2026-05-02.

W39 is a **multi-host quorum-bounding** layer on top of W38.  W38's
deepest open wall, ``W38-L-CONSENSUS-COLLUSION-CAP``, is: *if the
single registered disjoint consensus reference is itself compromised
in lock-step with the colluding trajectory hosts, W38 cannot recover
at the capsule layer.*  W38 cannot break this with a single
disjoint consensus probe.

W39 raises the capsule-layer adversary bar from "compromise 2 of N
trajectory hosts AND the single disjoint registered consensus
reference" to "compromise 2 of N trajectory hosts AND ``quorum_min``
of K mutually-disjoint registered consensus references, each on a
physically distinct host pool".  The mechanism: at every cell where
W37/W38 chooses to reroute, W39 cross-checks the W37/W38 candidate
top_set against K controller-pre-registered
:class:`MultiHostDisjointQuorumProbe` member probes whose host pools
are (a) mechanically disjoint from the W37 trajectory hosts (W38's
precondition) AND (b) **mutually disjoint from one another** (the
new W39 precondition).  When at least ``quorum_min`` of the K
member probes diverge from the W37/W38 candidate top_set, W39
abstains via the ``QUORUM_DIVERGENCE_ABSTAINED`` branch.

W39 does **not** close ``W38-L-CONSENSUS-COLLUSION-CAP`` in general
and it does **not** close native-latent transfer.  When all K
disjoint probes are themselves compromised in lock-step with the
colluding trajectory hosts, W39 cannot recover; this is the new
**W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP** limitation theorem
(the W39 analog of ``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``, and
``W38-L-CONSENSUS-COLLUSION-CAP``).

The result is not native latent transfer.  It is a closed-form,
zero-parameter, audited capsule-layer proxy with two mechanically
enforced disjointness preconditions: **trajectory disjointness** and
**mutual disjointness**.

---

## 1. New mechanism

The W39 family adds:

- ``MultiHostDisjointQuorumProbe``
- ``MultiHostDisjointQuorumRatificationEnvelope``
- ``MultiHostDisjointQuorumRegistry``
- ``W39MultiHostDisjointQuorumResult``
- ``MultiHostDisjointQuorumOrchestrator``
- ``MutuallyDisjointTopologyError`` (NEW; W39-specific)
- ``select_multi_host_disjoint_quorum_decision``
- ``verify_multi_host_disjoint_quorum_ratification``
- ``build_trivial_multi_host_disjoint_quorum_registry``
- ``build_multi_host_disjoint_quorum_registry``

W39 wraps W38.  At each cell, W39:

1. Reads W38's last-cell decision and projection top_set.
2. Calls the registered quorum-probe provider for the current cell,
   producing a :class:`MultiHostDisjointQuorumProbe` with K member
   probes, each tagged with its (host_id, oracle_id, top_set,
   consensus_strength) and each registered against a controller-pre-
   registered host pool that is mutually disjoint from every other
   member probe's pool.
3. Calls :func:`select_multi_host_disjoint_quorum_decision`, which
   tallies (n_agree, n_disagree, n_weak) over the K member probes
   and returns one of:

   * ``QUORUM_NO_REFERENCES`` -- no probe registered;
   * ``QUORUM_NO_TRIGGER`` -- W37/W38 did not reroute;
   * ``QUORUM_INSUFFICIENT`` -- fewer than ``min_quorum_probes``
     member probes are registered for the current cell;
   * ``QUORUM_REFERENCE_WEAK`` -- *all* member probes have
     ``consensus_strength`` below ``consensus_strength_min``;
   * ``QUORUM_DIVERGENCE_ABSTAINED`` -- at least ``quorum_min`` of
     the strong member probes diverge from the W38 candidate top_set;
   * ``QUORUM_RATIFIED`` -- at least ``quorum_min`` of the strong
     member probes agree with the W38 candidate top_set AND fewer
     than ``quorum_min`` disagree;
   * ``QUORUM_SPLIT`` -- W37/W38 rerouted but neither side reaches
     ``quorum_min``; W39 falls through to W38's decision.

4. Seals the decision in a manifest-v9 envelope.

Manifest-v9 binds:

1. W38 parent CID.
2. Quorum state CID (over the entire member probe list).
3. Quorum decision CID (over n_agree, n_disagree, n_weak, n_total,
   decision_top_set, per_probe_divergence_scores, per_probe_branches).
4. Quorum audit CID (over the named-branch decision tuple and
   thresholds).
5. Quorum topology CID (each registered host pool's host/oracle
   IDs PLUS the trajectory host set).
6. Mutual-disjointness CID (an explicit per-pair-intersection
   witness so the verifier can defense-in-depth check that the
   declared topology is mutually disjoint, even if the outer
   topology CID happens to match an unrelated registered topology
   of equal canonical form).

The disjoint-topology preconditions are enforced in **two** places:

- :class:`MultiHostDisjointQuorumRegistry`'s ``__post_init__``
  raises :class:`DisjointTopologyError` when any ``pool_host_ids ∩
  trajectory_host_ids ≠ ∅`` (inherited from W38) AND raises the
  new :class:`MutuallyDisjointTopologyError` when any
  ``pool_i_host_ids ∩ pool_j_host_ids ≠ ∅`` for ``i != j``.
- :func:`verify_multi_host_disjoint_quorum_ratification` rejects
  envelopes whose ``registered_quorum_pool_host_ids`` claim an
  overlapping trajectory pool
  (``w39_quorum_disjoint_topology_violation``) OR an overlapping
  pair of quorum pools
  (``w39_quorum_mutual_disjointness_violation``).

---

## 2. Benchmark family

Phase 86 introduces five small regimes:

| Regime | Purpose |
| --- | --- |
| ``trivial_w39`` | Byte-for-W38 preservation when quorum is disabled, divergence-abstain is disabled, and manifest-v9 is disabled |
| ``multi_host_colluded_consensus`` | Prefix half: trajectory hosts + W38 single consensus + W39 quorum members all agree.  Recovery half: the trajectory hosts are colluding AND the W38 single consensus is COMPROMISED IN LOCK-STEP (the W38-L-CONSENSUS-COLLUSION-CAP regime), but the W39 multi-host disjoint quorum points to the truthful gold; W39 abstains |
| ``no_regression_quorum_agrees`` | Quorum members all agree with the W37/W38 reroute; W39 ratifies; no regression |
| ``full_quorum_collusion`` | Every disjoint quorum probe is compromised in lock-step (the new W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP fires); W39 cannot recover |
| ``insufficient_quorum`` | Only 1 member probe is provided in the recovery half (below ``min_quorum_probes=2``); W39 reduces to W38 via QUORUM_INSUFFICIENT |

The bench output reports substrate/FIFO, W21, W36, W37, W38, and W39
in one place so the old explicit-capsule line, the dense-control /
geometry line, the W37 cross-cell trajectory line, the W38 disjoint-
consensus line, and the new W39 multi-host disjoint quorum line can
be compared at a glance.

---

## 3. Empirical results

All seed sweeps use seeds ``11, 17, 23, 29, 31``, ``n_eval=16``.

### R-86-MULTI-HOST-COLLUDED-CONSENSUS (load-bearing)

Across 5/5 seeds:

- substrate/FIFO correctness: 0.0.
- W21 trust precision: 0.5.
- W36 trust precision: 0.5.
- W37 trust precision: 0.5 (reroutes on the colluded trajectory-
  anchored top_set in the recovery half).
- W38 trust precision: 0.5 (the new R-86 threat model has the W38
  *single* consensus also compromised in lock-step => W38-L-CONSENSUS-
  COLLUSION-CAP fires).
- W39 trust precision: **1.000**.
- ``min_delta_trust_precision_w39_w38 = +0.500``.
- ``max_delta_trust_precision_w39_w38 = +0.500``.
- W39 abstains via QUORUM_DIVERGENCE on 8 cells/seed, ratifies 0
  cells, no_references 0, insufficient 0, split 0.
- W39 overhead: 1 visible token/cell.
- Structured state density: about **24.4k bits per visible W39
  token** (the densest milestone-level capsule-layer audited proxy
  the programme has measured to date; W38 carried about 9.07k
  bits/token on R-85-COLLUDED-CROSS-HOST-TRAJECTORY).

This is the load-bearing W39 multi-host quorum-bounding result.
It is *not* closure of ``W38-L-CONSENSUS-COLLUSION-CAP`` in general;
it is the strongest honest capsule-layer bound this repo can support
without native latent transfer or a stable K+1-physical-host
topology.  The bound is valid only when at least ``quorum_min`` of
the K registered probes remain uncompromised.

Artifact:
``vision_mvp/experiments/artifacts/phase86/multi_host_colluded_consensus_seed_sweep.json``.

### R-86-TRIVIAL-W39

Across 5/5 seeds:

- W39 = W38 byte-for-byte.
- correctness and trust precision identical to W38.
- overhead = 0.
- ``all_byte_equivalent_w39_w38 = true``.

Artifact:
``vision_mvp/experiments/artifacts/phase86/trivial_w39_seed_sweep.json``.

### R-86-NO-REGRESSION-QUORUM-AGREES

Across 5/5 seeds:

- W39 correctness: same as W38 (delta = 0).
- W39 trust precision: same as W38 (delta = 0).
- W39 ratifies via QUORUM_RATIFIED on 8 cells/seed in the recovery
  half.
- ``min_delta_correctness_w39_w38 = 0.0``.
- ``min_delta_trust_precision_w39_w38 = 0.0``.

Artifact:
``vision_mvp/experiments/artifacts/phase86/no_regression_quorum_agrees_seed_sweep.json``.

### R-86-FULL-QUORUM-COLLUSION (W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP)

Across 5/5 seeds:

- W39 trust precision: 0.5 (cannot recover; every disjoint quorum
  probe is itself compromised in lock-step with the colluding
  trajectory hosts).
- ``min_delta_trust_precision_w39_w38 = 0.0``.
- ``max_delta_trust_precision_w39_w38 = 0.0``.

This is the named ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``
limitation theorem firing.  Closure requires native-latent evidence
outside the capsule layer (``W39-L-NATIVE-LATENT-GAP``) or a
K+1-host disjoint consensus topology with at least one new genuinely
uncompromised host pool.

Artifact:
``vision_mvp/experiments/artifacts/phase86/full_quorum_collusion_seed_sweep.json``.

### R-86-INSUFFICIENT-QUORUM

Across 5/5 seeds:

- W39 correctness: same as W38 (delta = 0).
- W39 trust precision: same as W38 (delta = 0).
- W39 returns ``QUORUM_INSUFFICIENT`` on every recovery cell where
  fewer than ``min_quorum_probes=2`` member probes are present.

Artifact:
``vision_mvp/experiments/artifacts/phase86/insufficient_quorum_seed_sweep.json``.

---

## 4. Live / multi-Mac evidence

### 4.1 Lab topology resolution (W38-C-MULTI-HOST partial discharge)

The historical Mac-2 endpoint (``192.168.12.248``) has been
ARP-incomplete for the **31st milestone in a row**.  This milestone
identified ``192.168.12.101`` as the fresh candidate for the third
physical host role and discharged the stale repo pin on ``.248``
in the W39 live xllm probe.

On first contact in this session, ``.101`` was preflight-reachable:
``/api/tags`` returned a ``qwen3.5:35b`` (36.0B MoE Q4_K_M) +
``qwen2.5:14b-32k`` inventory in milliseconds, ``ping`` reported
~3.5 ms RTT.  This is a strict improvement vs ``.248`` which has
never been reachable in any milestone.  ``W38-C-MULTI-HOST`` is
**partially discharged** at the topology layer: a third physical
host candidate is reachable for preflight on cold contact.

Artifact:
``vision_mvp/experiments/artifacts/phase86/host_topology_resolution_2026_05_02.json``.

### 4.2 W39-INFRA-1: .101 inference path bounded under one-word probe budget

The ``.101`` host's Ollama inference path subsequently degraded
under the capsule-layer one-word probe budget.  Empirical
observations recorded during the milestone:

- ``/api/chat`` with ``qwen3.5:35b`` at ``temperature=0`` +
  ``num_predict=4``: timed out at 300 seconds.
- ``/api/chat`` with ``qwen2.5:14b-32k`` at ``temperature=0`` +
  ``num_predict=4``: timed out at 60 seconds AND at 240 seconds.
- ``/api/generate`` with ``num_predict=1``: timed out at 60 seconds.
- ``/api/show``: timed out at 30 seconds.
- ``/api/ps``: timed out at 10 seconds.
- ``/api/tags`` (after the inference attempts): returned an empty
  reply from server at 5 seconds (TCP connect succeeds but Ollama
  HTTP listener immediately closes the connection).
- ``ping``: subsequently went to 100% packet loss.
- After a 30-attempt 10-second-interval poll over 5 minutes:
  every endpoint still returns "Empty reply from server"; no
  recovery observed.
- SSH access to ``.101`` was attempted via ``qdong``, ``root``,
  ``admin``, ``macm1``, ``mac1``, ``ollama``: all rejected
  (publickey/password authentication failed); no SSH credentials
  available to restart the Ollama service remotely.

This is recorded as ``W39-INFRA-1``.  ``W38-C-MULTI-HOST`` is
therefore **partially discharged at the topology layer (a third
physical host candidate is preflight-reachable on cold contact)
but remains open at the live-inference layer (the third host's
inference path is bounded under the one-word probe budget after
first model-load contact).**

### 4.3 W39-INFRA-1 fallback: cross-physical-host K=2 quorum on two reachable Macs

The W39 live xllm probe was made robust to the W39-INFRA-1 finding
via an explicit fallback: when ``mac_off_cluster_a`` (``.101``) is
unreachable, the probe swaps to ``localhost`` running
``llama3.1:8b`` (a model class genuinely different from the
trajectory's ``gemma2:9b``), so the live K=2 quorum becomes:

- ``mac_off_cluster_a``: ``localhost`` running ``llama3.1:8b``;
- ``mac_quorum_b``: ``192.168.12.191`` running
  ``qwen2.5-coder:14b-32k``.

These are two **physically distinct hosts**, each running a
different model class from the trajectory pair (``gemma2:9b`` and
``qwen2.5:14b``) and from the W38 single consensus reference
(``qwen2.5-coder:14b``).

Bounded W39 live probe (``phase86_xllm_quorum_probe.py``,
2026-05-02) on 8 gold-verifiable one-word prompts at temperature 0
+ ``num_predict=4`` + stop tokens:

- 8/8 responsive on all 5 hosts (mac1 + mac_remote +
  mac_consensus + mac_off_cluster_a + mac_quorum_b) -- **first
  measured 5-host live W39 disjoint-quorum probe in the
  programme**.
- 7/8 trajectory_pair_agrees (1 disagreement is the well-known
  ``num_predict=4`` truncation pattern: gold ``h2o`` ->
  ``mac_remote`` answered ``h``).
- 7/8 consensus_agrees_with_trajectory.
- 7/8 quorum_a_agrees_with_trajectory.
- 7/8 quorum_b_agrees_with_trajectory.
- **8/8 quorum_a gold-correlated**.
- **8/8 quorum_b gold-correlated**.
- 8/8 K=2 quorum members simultaneously responsive.
- 7/8 K=2 quorum members both agree with the trajectory pair
  (W39 ratifies in 7 cells).
- 0/8 K=2 quorum members both disagree with the trajectory pair
  (no live abstention triggered, expected since the live oracles
  are not actually colluding).

A new live finding worth highlighting: on the ``h2o`` probe, the
trajectory pair disagreed (``mac1=h2o`` vs ``mac_remote=h`` due to
``num_predict=4`` truncation), but **both quorum members got
``h2o`` correct**.  The W39 multi-host disjoint quorum can therefore
catch live trajectory truncation errors at the capsule layer when
the quorum members are not subject to the same truncation regime;
this is empirical evidence of a recovery axis at the live layer
(``W39-C-LIVE-TRUNCATION-RECOVERY`` is a new conjecture; we do not
register it as a theorem because we did not run an attack-recovery
experiment, but the per-probe data supports the conjecture
honestly).

Per-probe results recorded in
``vision_mvp/experiments/artifacts/phase86/xllm_quorum_probe_2026_05_02.json``.

This live probe materially strengthens the capsule-layer multi-host
disjoint-quorum motivation along the live-evidence axis but does
**not** close ``W39-C-MULTI-HOST`` (a stable K+1-host live
inference topology, hardware-bounded) and does **not** close
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` (no live colluded
scenario was attempted; the live probe is gold-correlated agreement
evidence, not an attack-recovery measurement).

---

## 5. Trust boundary

``verify_multi_host_disjoint_quorum_ratification`` enumerates 14
W39 failure modes:

1. ``empty_w39_envelope``
2. ``w39_schema_version_unknown``
3. ``w39_schema_cid_mismatch``
4. ``w38_parent_cid_mismatch``
5. ``w39_projection_branch_unknown``
6. ``w39_quorum_probe_unregistered_host``
7. ``w39_quorum_probe_unregistered_oracle``
8. ``w39_quorum_disjoint_topology_violation``
9. ``w39_quorum_mutual_disjointness_violation`` (NEW; W39-specific)
10. ``w39_quorum_thresholds_invalid``
11. ``w39_quorum_state_cid_mismatch``
12. ``w39_quorum_decision_cid_mismatch``
13. ``w39_quorum_topology_cid_mismatch``
14. ``w39_manifest_v9_cid_mismatch`` (and ``w39_outer_cid_mismatch``
    co-defined)

Cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36 +
W37 + W38 + W39 trust boundary: **154 enumerated failure modes**.

---

## 6. Theory-forward claims

### W39-1 - verifier boundary

**Claim**: The W39 multi-host disjoint quorum envelope is
mechanically verifiable against 14 disjoint failure modes.

**Status**: proved by inspection + mechanically checked.

### W39-2 - trivial reduction

**Claim**: Disabled quorum + disabled divergence-abstain + disabled
manifest-v9 reduces to W38 byte-for-byte.

**Status**: empirical across 5/5 seeds.

### W39-3 - multi-host disjoint quorum collusion bound (load-bearing)

**Claim**: If the current cell would reroute a W37 trajectory-
anchored top_set produced by colluding hosts AND the registered W38
single disjoint consensus reference is itself compromised in
lock-step (the ``W38-L-CONSENSUS-COLLUSION-CAP`` regime), AND a
registered ``MultiHostDisjointQuorumProbe`` exists with K member
probes whose host pools are mechanically disjoint from the W37
trajectory hosts AND mutually disjoint from one another, AND fewer
than ``quorum_min`` of those member probes are themselves
compromised in lock-step (i.e. at least
``K - quorum_min + 1`` member probes remain uncompromised), THEN
W39 abstains via ``QUORUM_DIVERGENCE_ABSTAINED`` without
trust-precision regression.

**Status**: proved-conditional + empirical on
R-86-MULTI-HOST-COLLUDED-CONSENSUS (5/5 seeds, +0.500 trust
precision gain over W38, trust precision = 1.000, overhead = 1
token/cell).

### W39-4 - mutual-disjointness mechanical enforcement

**Claim**: A ``MultiHostDisjointQuorumRegistry`` whose any two
registered quorum host pools have non-empty intersection cannot be
constructed (raises :class:`MutuallyDisjointTopologyError` in
``__post_init__``); the verifier additionally rejects an envelope
whose declared quorum topology has overlapping pools
(``w39_quorum_mutual_disjointness_violation``).

**Status**: proved by inspection + mechanically tested.

### W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP

**Claim**: When all K registered disjoint quorum probes are
themselves compromised in lock-step with the colluding trajectory
hosts (i.e. all emit the same wrong top_set), W39 cannot recover
at the capsule layer.  This is the W39 analog of
``W34-L-MULTI-ANCHOR-CAP``, ``W37-L-MULTI-HOST-COLLUSION-CAP``,
and ``W38-L-CONSENSUS-COLLUSION-CAP`` and is closed only by
transformer-internal evidence outside the capsule layer or by a
K+1-host disjoint consensus topology with at least one
uncompromised pool.

**Status**: proved-conditional limitation theorem + empirical on
R-86-FULL-QUORUM-COLLUSION (5/5 seeds, delta = 0).

### W39-L-INSUFFICIENT-QUORUM

**Claim**: If fewer than ``min_quorum_probes`` member probes are
registered for the current cell, W39 cannot detect quorum
divergence and falls through to W38's decision (the
``QUORUM_INSUFFICIENT`` branch).  W39 is insufficient for any
regime where a K=``min_quorum_probes`` mutually-disjoint quorum
cannot be sourced.

**Status**: proved by inspection + empirical on
R-86-INSUFFICIENT-QUORUM.

### W39-C-NATIVE-LATENT

**Conjecture**: True transformer-internal trust-state projection
may separate regimes where all capsule-visible host/probe/evidence/
trajectory/consensus/quorum signals are either absent or
coordinated.  W39 narrows the audited proxy further along the
multi-host quorum axis but does not close this.

**Status**: conjectural and architecture-dependent.

### W39-C-MULTI-HOST

**Conjecture**: A K+1-host topology with at least one new genuinely
uncompromised host pool would let the W39 quorum size be raised
beyond ``quorum_min``, defeating the
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` collusion attack at
the capsule layer.  Currently partially discharged at the topology
layer via ``192.168.12.101`` (preflight-reachable on cold contact)
but still open at the live-inference layer (``W39-INFRA-1``).

**Status**: conjectural; partially discharged at the topology
layer; still open at the live-inference layer; hardware-bounded.

### W39-C-LIVE-TRUNCATION-RECOVERY (NEW)

**Conjecture**: At the live inference layer, the W39 multi-host
disjoint quorum can recover from trajectory-pair-only truncation
errors when the quorum members are not subject to the same
``num_predict``/stop-token regime as the trajectory pair.  This is
a recovery axis distinct from the collusion bound: it does not
require any host to be adversarial; it only requires the quorum
members to use a different (longer) generation budget than the
trajectory pair.

**Status**: empirical-suggestive on the 2026-05-02 live xllm
quorum probe (``h2o`` truncation case: trajectory pair disagreed,
both quorum members got ``h2o`` correct), conjectural without an
attack-recovery experiment; sharper validation requires a dedicated
live truncation-recovery bench.

---

## 7. Release boundary

Release readiness improved because:

- W39 is strictly additive and experimental.
- Stable runtime contract is unchanged.
- SDK version (``wevra.sdk.v3.40``) and package version (``0.5.13``)
  are bumped.
- The repo's ``vision_mvp.__version__`` and ``pyproject.toml``
  ``project.version`` are now both ``0.5.13`` (alignment is
  maintained).
- Success bar, results note, theorem registry, README, START_HERE,
  master plan, changelog, and paper marker are updated.
- Focused W22..W39 regression at **625/625 phase69-86** is green;
  broad spot checks at 569/569 (364 phase11-39 + 205 phase40-51 +
  phase6, excluding the pre-existing ``test_phase50_ci_and_zero_shot``
  collection-time hang carried forward from W38).
- The cumulative enumerated trust boundary is now **154 failure
  modes** (W22..W39).
- The stable-vs-experimental boundary is final: every W22..W39
  symbol is exported under ``__experimental__`` and the stable
  ``RunSpec → run report`` runtime contract is byte-for-byte
  unchanged.
- The historical Mac-2 stale repo pin (``192.168.12.248``) has been
  discharged in favour of ``192.168.12.101`` as the reachable third
  physical host candidate; the W39 live xllm probe accepts
  ``WEVRA_OLLAMA_URL_MAC2`` env-var override and falls through a
  candidate list including ``.101`` before declaring "Mac 2
  unreachable"; the W39-INFRA-1 fallback path keeps the live K=2
  quorum probe useful even when ``.101``'s inference path is
  bounded.
- **First measured 5-host live W39 disjoint-quorum probe in the
  programme**: 8/8 responsive on all 5 hosts at temperature 0 +
  ``num_predict=4``, 7/8 trajectory_pair_agrees, 8/8 quorum_a
  gold-correlated, 8/8 quorum_b gold-correlated, 8/8 K=2 quorum
  size simultaneously responsive.

Release readiness is not fully closed because:

- W39 is not native latent transfer.
- ``.101`` Ollama inference path is bounded under one-word probe
  budget after first model-load contact (W39-INFRA-1).
- True K+1-host live inference evidence remains open
  (``W39-C-MULTI-HOST``, ``W39-C-NATIVE-LATENT``).
- ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` cannot be closed at
  the capsule layer.
- W21 still dominates W39 on regimes where multi-oracle quorum is
  enough by itself; W39 is a trust-stack hardening result for the
  cross-cell, single-host recovery case under capsule-layer
  collusion threat with the W38 single consensus *also*
  compromised, not a universal successor to every older
  explicit-capsule baseline.
