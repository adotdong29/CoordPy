# Success criterion - CoordPy SDK v3.39 / W38
# Disjoint cross-source consensus-reference trajectory-divergence
# adjudication + manifest-v8 CID

**Pre-committed before final W38 verdict / release decision**:
2026-05-02.
**Target**: SDK v3.39 / W38 family.
**Position relative to W37**: W37 maintains a per-(host, oracle,
top_set) EWMA over anchored historical observations and converts a
W36 single-host abstention into a safe reroute when the supporting
host has a cross-host anchored trajectory above threshold across at
least ``min_anchored_observations`` historical cells with at least
``min_trajectory_anchored_hosts`` distinct anchor hosts.

W37's deepest open wall is **W37-L-MULTI-HOST-COLLUSION-CAP**: when
two registered hosts simultaneously emit a coordinated wrong top_set
across at least ``min_anchored_observations`` cells, the trajectory
crosses the anchored thresholds and W37 can be made to reroute on
the wrong top_set.  W37 cannot break this at the capsule layer.

W38 must therefore be judged as a **collusion-bounding** milestone,
not as another local benchmark win.  The target is to bound (not
close) W37-L-MULTI-HOST-COLLUSION-CAP at the capsule layer by
adding a **disjoint cross-source consensus reference**: a
controller-pre-registered probe whose host topology is disjoint from
W37's trajectory hosts and whose top_set is independently sealed
into the W38 envelope.  The W38 mechanism turns a W37 reroute into
an abstention whenever the W37 candidate top_set and the disjoint
consensus reference top_set diverge.  When the disjoint consensus
reference itself is compromised, W38 cannot recover; this is the
new **W38-L-CONSENSUS-COLLUSION-CAP** limitation theorem (the W38
analog of W34-L-MULTI-ANCHOR-CAP and W37-L-MULTI-HOST-COLLUSION-CAP).

The honest reading: the W38 mechanism raises the capsule-layer
adversary bar from "compromise 2 of N trajectory hosts" to
"compromise 2 of N trajectory hosts **and** the disjoint registered
consensus reference".  It does not break native-latent or close
W37-C-MULTI-HOST.

---

## 1. Mechanism bar

W38 must implement a real method beyond W37:

1. Wrap W37 with a **disjoint consensus reference** layer.
2. Maintain a controller-side ``ConsensusReferenceProbe`` per cell
   sourced from a registered topology that is mechanically disjoint
   from the W37 trajectory hosts (the ``DisjointConsensusReference
   Registry`` enforces ``consensus_host_ids ∩ trajectory_host_ids
   = ∅`` at registration time).
3. Expose a single new safety knob
   ``allow_consensus_reference_divergence_abstain`` that, when
   enabled, lets W38 convert a W37 ``REROUTED`` decision into a
   ``DIVERGENCE_ABSTAINED`` decision iff the W37 candidate top_set
   does not match the registered consensus reference top_set within
   ``divergence_margin_min``.
4. When the consensus reference is empty, weak (below
   ``consensus_strength_min``), or topologically equal to the W37
   trajectory host set, W38 must preserve W37 behavior byte-for-byte.
5. Seal the new state in a content-addressed **manifest-v8** envelope
   over five CIDs (parent_w37_cid, consensus_reference_state_cid,
   divergence_audit_cid, consensus_topology_cid, consensus_probe_cid).
6. Keep the native-latent gap explicit: no transformer hidden-state,
   KV-cache, attention-weight, or embedding-table access is added.
   The disjoint consensus reference is an audited capsule-layer
   probe over a registered disjoint host topology, NOT a runtime
   ground-truth oracle.

The permitted headline is **disjoint cross-source consensus-
reference trajectory-divergence audited proxy**, not native latent
transfer.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W38 mechanism beyond W37 | New orchestrator, registry, envelope, verifier, consensus-reference state, divergence selector, and disjoint-topology enforcement implemented; W38 makes a routing decision W37 cannot make |
| **H2** | Trust boundary | ``verify_disjoint_consensus_reference_ratification`` enumerates 14 disjoint W38 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W38 old-regime preservation | With consensus reference disabled and manifest-v8 disabled, W38 reduces to W37 byte-for-byte across 5 seeds |
| **H4** | Load-bearing colluded-trajectory mitigation | On R-85-COLLUDED-CROSS-HOST-TRAJECTORY, W38 improves trust precision over W37 by at least +0.20 and restores trust precision to >= 0.95 with no correctness regression below W37, adding <= 1 visible token/cell |
| **H5** | No-collusion-consensus-agrees falsifier | On R-85-NO-COLLUSION-CONSENSUS-AGREES, W38 must not weaken W37 (delta_correctness >= 0, delta_trust_precision >= 0) |
| **H6** | Consensus-also-compromised falsifier | On R-85-CONSENSUS-ALSO-COMPROMISED, W38 must not claim recovery; the W38-L-CONSENSUS-COLLUSION-CAP limitation theorem fires (W38 = W37 within 1 visible token) |
| **H7** | No-consensus-reference falsifier | On R-85-NO-CONSENSUS-REFERENCE, W38 correctness/trust precision are unchanged from W37 (delta = 0); the W38 audit envelope still records the NO_REFERENCE branch (the audit trail is load-bearing even when the cross-check has nothing to compare against) |
| **H8** | Old explicit capsule line preserved | Focused W33, W34, W35, W36, W37 regression slices stay green; W38 composes W21/W33/W34/W35/W36/W37 rather than bypassing them |
| **H9** | Disjoint-topology mechanically enforced | Building a W38 registry where ``consensus_host_ids ∩ trajectory_host_ids ≠ ∅`` raises ``DisjointTopologyError`` at construction time; the verifier additionally rejects an envelope whose ``consensus_topology_cid`` claims an overlapping host set |
| **H10** | Live/two-host evidence | Re-check usable hosts; if Mac 2 is unreachable, record exact fallback and run the strongest bounded live consensus-reference cross-architecture probe practical |
| **H11** | Broad regression confidence | Full ``pytest vision_mvp/tests -q`` runs to completion at least once during the milestone with the result counted (or honestly carried forward with named exclusions); focused W22..W38 regression is green |
| **H12** | Release-readiness clause | Versioning, changelog, success bar, results note, theorem registry, README/START_HERE/master plan/paper markers updated only if H1..H11 pass and the stable runtime remains unchanged; vision_mvp ``__version__`` aligned with pyproject |

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
| **S1** | Stronger live cross-source evidence | Bounded live probe observes either cross-architecture consensus-reference agreement or honest divergence, with gold-correlated labelling |
| **S2** | Mac 2 | ``192.168.12.248:11434/api/tags`` succeeds, or timeout evidence is recorded for the 31st milestone in a row |
| **S3** | Stable-vs-experimental boundary | W38 remains under ``__experimental__``; stable runtime contract unchanged |
| **S4** | Theory | Add one conditional sufficiency claim, one limitation/falsifier claim, and one native-latent gap claim |
| **S5** | Paper/master-plan synthesis | Old explicit-capsule line, dense-control/geometry line, and W38 disjoint-consensus line read as a single arc with an explicit release boundary |
| **S6** | Full broad regression actually counted | ``pytest vision_mvp/tests -q`` completes successfully and the count is recorded in the milestone report |
| **S7** | Release-candidate gate | If H1..H12 + S3 pass and Mac 2 is bounded, declare a release-candidate (``RC1``) status for the SDK v3.39 line, with a final "stable vs experimental" cut-list pinned in README |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W38-L-TRIVIAL-PASSTHROUGH**: consensus reference disabled +
  ``allow_consensus_reference_divergence_abstain=False`` + manifest-v8
  disabled reduces W38 to W37 byte-for-byte.
- **W38-L-NO-CONSENSUS-REFERENCE**: when no registered consensus
  reference probe exists, W38 cannot detect divergence and must
  preserve W37 behavior byte-for-byte (or abstain conservatively
  under a separate strict-knob).
- **W38-L-NO-COLLUSION-CONSENSUS-AGREES**: when the W37 reroute
  agrees with the consensus reference (no collusion), W38 must not
  weaken W37; correctness and trust precision are preserved.
- **W38-L-CONSENSUS-COLLUSION-CAP**: when the disjoint consensus
  reference is itself compromised in lock-step with the colluding
  trajectory hosts, W38 cannot recover at the capsule layer; this is
  the W38 analog of W34-L-MULTI-ANCHOR-CAP and W37-L-MULTI-HOST-
  COLLUSION-CAP and is closed only by transformer-internal evidence,
  which W38 does not access.
- **W38-L-DISJOINT-TOPOLOGY-VIOLATION**: a registered consensus
  reference whose host topology overlaps the trajectory host set is
  rejected at registration time (and any envelope claiming such a
  topology is rejected by the verifier).
- **W38-L-NATIVE-LATENT-GAP**: if a regime requires transformer-
  internal evidence not visible through trajectory observations,
  EWMA, response signatures, host attestations, anchor consensus,
  registries, and disjoint consensus references, W38 is insufficient.

---

## 5. Claim boundary

W38 may claim:

- an audited disjoint cross-source consensus-reference trajectory-
  divergence ratification proxy;
- measured W38-over-W37 trust-precision gain on a regime where W37
  reroutes on a colluded trajectory but a disjoint consensus
  reference detects the divergence;
- preserved W37 behavior on the trivial path, the no-consensus-
  reference falsifier, and the no-collusion-consensus-agrees
  regime;
- explicit safety on the consensus-collusion-cap falsifier and the
  disjoint-topology-violation falsifier;
- broader regression confidence if the full ``vision_mvp/tests``
  suite is run end-to-end during the milestone;
- a release-candidate declaration **only** when H1..H12 + S3 pass
  and Mac 2 is bounded honestly.

W38 may not claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- closure of W37-L-MULTI-HOST-COLLUSION-CAP at the capsule layer
  (W38 only **bounds** it by a disjoint-consensus precondition);
- closure of W38-L-CONSENSUS-COLLUSION-CAP at the capsule layer;
- true three-host live evidence while Mac 2 is unavailable;
- solved context for multi-agent teams;
- release readiness independent of the explicit RC1 gate.
