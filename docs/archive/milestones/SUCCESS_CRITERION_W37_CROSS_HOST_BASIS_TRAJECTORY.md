# Success criterion - CoordPy SDK v3.38 / W37
# Anchor-cross-host basis-trajectory ratification + manifest-v7 CID

**Pre-committed before final W37 verdict/release decision**:
2026-05-02.
**Target**: SDK v3.38 / W37 family.
**Position relative to W36**: W36 hardened W35's dense-control proxy
with a host-diversity guard.  Each W36 ratification snapshots one cell.
W36 cannot reason about the *trajectory* of a basis direction across
cells, and it abstains whenever the current cell has fewer than
``min_distinct_hosts`` healthy attested hosts -- even if the single
remaining host has been independently anchored by other healthy hosts
in earlier cells.

W37 must therefore be judged as a blocker-removal milestone, not as
another local benchmark win.  The target blocker is the cross-cell,
cross-host audited proxy frontier.  The strongest honest next step
this repo can support without transformer-internal access is to
maintain a **cross-host basis-trajectory state** that records, per
``(host_id, oracle_id)``, an EWMA over anchored top-set observations,
and to allow safe reroute on a single healthy host *only* when its
trajectory has been cross-host anchored historically.

---

## 1. Mechanism bar

W37 must implement a real method beyond W36:

1. Wrap W36 with an **anchor-cross-host basis-trajectory** layer.
2. Maintain per ``(host_id, oracle_id)`` a closed-form EWMA over
   anchored top-set observations.  An observation is *anchored* iff,
   at the cell where it was recorded, at least one other registered
   healthy host attested the same top_set.
3. Expose a single new safety knob ``allow_single_host_trajectory_reroute``
   that, when enabled, lets W37 convert a W36 host-diversity abstention
   into a reroute *iff* the supporting host has a cross-host anchored
   trajectory above threshold across at least
   ``min_anchored_observations`` historical cells *and* at least
   ``min_trajectory_anchored_hosts`` distinct historical anchor hosts.
4. When the trajectory is empty, disagrees across hosts, or fails the
   anchored-observation requirement, W37 must preserve W36 behavior
   (abstain or reject).
5. Seal the new state in a content-addressed **manifest-v7** envelope
   over four CIDs (parent_w36_cid, cross_host_trajectory_state_cid,
   trajectory_audit_cid, trajectory_topology_cid).
6. Keep the native-latent gap explicit: no transformer hidden-state,
   KV-cache, attention-weight, or embedding-table access is added.

The permitted headline is **anchor-cross-host basis-trajectory
audited proxy**, not native latent transfer.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W37 mechanism beyond W36 | New orchestrator, registry, envelope, verifier, trajectory state, and selector are implemented; W37 can make a routing decision W36 cannot make |
| **H2** | Trust boundary | `verify_cross_host_trajectory_ratification` enumerates 14 disjoint W37 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W37 old-regime preservation | With trajectory disabled, single-host reroute disabled, and manifest-v7 disabled, W37 reduces to W36 byte-for-byte across 5 seeds |
| **H4** | Load-bearing single-host trajectory recovery | On R-84-SINGLE-HOST-TRAJECTORY-RECOVER, W37 improves correctness over W36 by at least +0.20, restores trust precision to >= 0.95, and adds <= 1 visible token/cell |
| **H5** | Trajectory disagreement falsifier | On R-84-TRAJECTORY-DISAGREEMENT, W37 must not claim correctness gain over W36 |
| **H6** | No-trajectory-history falsifier | On R-84-NO-TRAJECTORY-HISTORY, W37 reduces to W36 byte-for-byte |
| **H7** | Poisoned trajectory falsifier | On R-84-POISONED-TRAJECTORY, single-host trajectory built on a single poisoned host must not produce a reroute (anchored-observation requirement must block) |
| **H8** | Old explicit capsule line preserved | Focused W33, W34, W35, W36 regression slices stay green; W37 composes W21/W33/W34/W35/W36 rather than bypassing them |
| **H9** | Dense-control/geometry line strengthened | W37 transfers controller-verified structured state at density >= 10,000 bits per visible W37 token on the load-bearing regime |
| **H10** | Live/two-host evidence | Re-check usable hosts; if Mac 2 is unreachable, record exact fallback and run the strongest bounded live trajectory probe practical |
| **H11** | Broad regression confidence | Full `pytest vision_mvp/tests -q` runs to completion at least once during the milestone with the result counted (or honestly carried forward with named exclusions); focused W22..W37 regression is green |
| **H12** | Release-readiness clause | Versioning, changelog, success bar, results note, theorem registry, README/START_HERE/master plan/paper markers updated only if H1..H11 pass and the stable runtime remains unchanged |

**Hard-gate aggregate**:

- **Strong success** = 11-12 gates pass, with no trust/audit weakening.
- **Partial success** = 9-10 gates pass, with exact blockers carried
  forward.
- **Failure** = <= 8 gates pass, any verifier weakening, or any
  unbounded native-latent/live claim.

---

## 3. Soft gates

| Gate | Description | Target |
| --- | --- | --- |
| **S1** | Stronger live disagreement evidence | Bounded live probe observes cross-host disagreement with gold-correlated winner, or records honestly-null |
| **S2** | Mac 2 | `192.168.12.248:11434/api/tags` succeeds, or timeout evidence is recorded for the 30th milestone |
| **S3** | Stable-vs-experimental boundary | W37 remains under `__experimental__`; stable runtime contract unchanged |
| **S4** | Theory | Add one conditional sufficiency claim, one limitation/falsifier claim, and one native-latent gap claim |
| **S5** | Paper/master-plan synthesis | The old explicit capsule line and dense-control/geometry line read as a single stack with a host-trust + trajectory-anchored boundary |
| **S6** | Full broad regression actually counted | `pytest vision_mvp/tests -q` completes successfully and the count is recorded in the milestone report |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W37-L-TRIVIAL-PASSTHROUGH**: trajectory disabled +
  ``allow_single_host_trajectory_reroute=False`` + manifest-v7 disabled
  reduces W37 to W36 byte-for-byte.
- **W37-L-NO-TRAJECTORY-HISTORY**: when no anchored observations exist
  before the current cell, W37 cannot reroute on a single healthy host
  and must preserve W36 abstention.
- **W37-L-TRAJECTORY-DISAGREEMENT**: when historical trajectories from
  distinct hosts disagree, W37 must not commit to either; preserves W36
  abstention.
- **W37-L-POISONED-TRAJECTORY**: a trajectory accumulated on a single
  poisoned host (without cross-host anchoring) must not satisfy the
  ``min_trajectory_anchored_hosts`` requirement and must therefore
  not produce a reroute.
- **W37-L-MULTI-HOST-COLLUSION-CAP**: if two registered hosts
  simultaneously emit a coordinated wrong top_set across enough cells
  to cross the anchored-observation threshold, W37 cannot recover at
  the capsule layer; this is the W37 analog of the W34-L-MULTI-ANCHOR-
  CAP limitation theorem and is closed only by transformer-internal
  evidence, which W37 does not access.
- **W37-L-NATIVE-LATENT-GAP**: if a regime requires transformer-
  internal evidence not visible through trajectory observations,
  EWMA, response signatures, host attestations, anchor consensus, and
  registries, W37 is insufficient.

---

## 5. Claim boundary

W37 may claim:

- an audited anchor-cross-host basis-trajectory ratification proxy;
- measured W37-over-W36 correctness gain on a regime where W36
  abstains because of insufficient current-cell host diversity but a
  cross-host anchored trajectory is available;
- preserved W36 behavior on the trivial path;
- explicit safety on the trajectory-disagreement, no-history,
  poisoned-trajectory, and multi-host-collusion falsifiers;
- broader regression confidence if the full `vision_mvp/tests` suite
  is run end-to-end during the milestone.

W37 may not claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- true three-host evidence while Mac 2 is unavailable;
- closure of W37-L-MULTI-HOST-COLLUSION-CAP at the capsule layer;
- solved context for multi-agent teams;
- release readiness independent of blocker removal.
