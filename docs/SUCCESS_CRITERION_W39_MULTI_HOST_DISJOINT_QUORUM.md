# Success criterion - Wevra SDK v3.40 / W39
# Multi-host disjoint quorum consensus-reference ratification +
# manifest-v9 CID + mutually-disjoint physical-host topology

**Pre-committed before final W39 verdict / release decision**:
2026-05-02.
**Target**: SDK v3.40 / W39 family.
**Position relative to W38**: W38 wraps W37 with a single
controller-pre-registered ``ConsensusReferenceProbe`` whose host
topology is mechanically disjoint from the W37 trajectory hosts.
W38 raises the capsule-layer adversary bar from "compromise 2 of N
trajectory hosts" to "compromise 2 of N trajectory hosts AND the
disjoint registered consensus reference".

W38's deepest open wall is **W38-L-CONSENSUS-COLLUSION-CAP**: when
the disjoint consensus reference is itself compromised in lock-step
with the colluding trajectory hosts, W38 cannot recover at the
capsule layer.  W38's named open conjecture **W38-C-MULTI-HOST**
states that a 3+-host topology with at least one host genuinely
disjoint from the trajectory hosts (not just a different model on
the same physical host) would let the consensus reference be
sourced from a true off-cluster oracle, defeating the
``W38-L-CONSENSUS-COLLUSION-CAP`` collusion attack at the capsule
layer.

W39 must therefore be judged as a **multi-host quorum-bounding**
milestone, not as another local benchmark win.  The target is to
*bound (not close)* W38-L-CONSENSUS-COLLUSION-CAP at the capsule
layer by adding a **multi-host disjoint quorum** consensus reference:
K controller-pre-registered probes, each sourced from a physically
distinct host pool, each mechanically disjoint from the W37
trajectory hosts AND mutually disjoint from every other probe's
host pool.  W39 ratifies the W37 reroute only when at least
``quorum_min`` of the registered disjoint probes agree with the W37
candidate top_set, and abstains via the ``QUORUM_DIVERGENCE_ABSTAINED``
branch as soon as ``quorum_min`` of them disagree.

When all K disjoint probes are themselves compromised in lock-step
with the colluding trajectory hosts, W39 cannot recover; this is the
new **W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP** limitation theorem
(the W39 analog of ``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``, and
``W38-L-CONSENSUS-COLLUSION-CAP``).

The honest reading: the W39 mechanism raises the capsule-layer
adversary bar from "compromise 2 of N trajectory hosts AND the
single disjoint registered consensus reference" to "compromise 2 of
N trajectory hosts AND ``quorum_min`` of the K mutually-disjoint
registered consensus references, each on a physically distinct
host pool".  It does not break native-latent transfer and does not
close the architecture-dependent ``W38-C-NATIVE-LATENT`` /
``W39-C-NATIVE-LATENT`` line.  The new conjecture
``W39-C-NATIVE-LATENT`` carries forward; it is closed only by
transformer-internal evidence outside the capsule layer.

A second axis of release readiness: the lab topology stale-pin on
``192.168.12.248`` (the historical "Mac 2") has been independently
diagnosed as stale and replaced by ``192.168.12.101`` as the
reachable third physical host.  W39 must encode this honestly: the
runtime, experiments, and live xLLM probe paths must accept an
explicit override and fall through a candidate list including
``.101`` before declaring "Mac 2 unreachable".

---

## 1. Mechanism bar

W39 must implement a real method beyond W38:

1. Wrap W38 with a **multi-host disjoint quorum** layer.
2. Maintain a controller-side ``MultiHostDisjointQuorumProbe`` per
   cell sourced from K registered host pools that are
   mechanically disjoint from the W37 trajectory hosts AND
   mutually disjoint from one another (the
   ``MultiHostDisjointQuorumRegistry`` enforces both at registration
   time and any envelope claiming a violating topology is rejected
   by the verifier).
3. Expose a single new safety knob
   ``allow_disjoint_quorum_divergence_abstain`` that, when enabled,
   lets W39 convert a W37/W38 ``REROUTED``/``RATIFIED`` decision into a
   ``QUORUM_DIVERGENCE_ABSTAINED`` decision iff at least
   ``quorum_min`` of the registered disjoint probes diverge from the
   W37 candidate top_set within ``divergence_margin_min``.
4. When fewer than ``min_quorum_probes`` probes are present for the
   current cell, W39 must reduce to W38 byte-for-byte (the
   ``QUORUM_INSUFFICIENT`` falsifier branch).
5. When at least ``quorum_min`` of the registered disjoint probes
   AGREE with the W37 candidate top_set AND fewer than
   ``quorum_min`` of them disagree, W39 must ratify the W37 reroute
   (the ``QUORUM_RATIFIED`` branch).
6. When the disjoint quorum probes split (no side reaches
   ``quorum_min``), W39 must fall through to W38's decision (the
   ``QUORUM_SPLIT`` branch).
7. Seal the new state in a content-addressed **manifest-v9** envelope
   over six CIDs (parent_w38_cid, quorum_state_cid, quorum_audit_cid,
   quorum_topology_cid, quorum_decision_cid, mutual_disjointness_cid).
8. Keep the native-latent gap explicit: no transformer hidden-state,
   KV-cache, attention-weight, or embedding-table access is added.
   The disjoint quorum is K audited capsule-layer probes, NOT a
   runtime ground-truth oracle and NOT a transformer-internal
   trust-state projection.
9. Mechanically enforce **mutual** disjointness: a registry whose
   any two registered quorum host pools have non-empty intersection
   raises ``MutuallyDisjointTopologyError`` at construction time;
   the verifier rejects an envelope claiming an overlapping pool
   pair (``w39_quorum_mutual_disjointness_violation``).
10. Resolve the lab topology stale-pin: ``192.168.12.101`` is
    accepted as a reachable disjoint third physical-host candidate
    in the live xLLM quorum probe; the milestone records the
    honest infra observation about ``.101``'s inference-path
    behavior under one-word probe budget.

The permitted headline is **multi-host disjoint quorum
consensus-reference audited proxy**, not native latent transfer
and not closure of ``W38-L-CONSENSUS-COLLUSION-CAP`` in general.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W39 mechanism beyond W38 | New orchestrator, registry, envelope, verifier, multi-host disjoint quorum probe, quorum-decision selector, mechanical mutual-disjointness enforcement, and manifest-v9 CID implemented; W39 makes a routing decision W38 cannot make |
| **H2** | Trust boundary | ``verify_multi_host_disjoint_quorum_ratification`` enumerates 14 disjoint W39 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W39 old-regime preservation | With multi-host quorum disabled and manifest-v9 disabled, W39 reduces to W38 byte-for-byte across 5 seeds |
| **H4** | Load-bearing multi-host-colluded-consensus mitigation | On R-86-MULTI-HOST-COLLUDED-CONSENSUS, W39 improves trust precision over W38 by at least +0.20 and restores trust precision to >= 0.95 with no correctness regression below W38, adding <= 1 visible token/cell |
| **H5** | No-regression-quorum-agrees falsifier | On R-86-NO-REGRESSION-QUORUM-AGREES, W39 must not weaken W38 (delta_correctness >= 0, delta_trust_precision >= 0) |
| **H6** | Full-quorum-collusion limitation theorem | On R-86-FULL-QUORUM-COLLUSION, W39 must not claim recovery; the W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP limitation theorem fires (W39 = W38 within 1 visible token); delta_trust_precision = 0 across 5/5 seeds |
| **H7** | Insufficient-quorum falsifier | On R-86-INSUFFICIENT-QUORUM, when fewer than ``quorum_min`` registered disjoint probes are present, W39 reduces to W38 byte-for-byte (delta = 0); the W39 audit envelope still records the QUORUM_INSUFFICIENT branch |
| **H8** | Old explicit capsule line preserved | Focused W33, W34, W35, W36, W37, W38 regression slices stay green; W39 composes W21/W33/W34/W35/W36/W37/W38 rather than bypassing them |
| **H9** | Mutual-disjointness mechanically enforced | Building a W39 registry whose any two registered quorum host pools have non-empty intersection raises ``MutuallyDisjointTopologyError`` at construction time; the verifier additionally rejects an envelope whose registered topology claims an overlapping pool pair (``w39_quorum_mutual_disjointness_violation``) |
| **H10** | Live/three-host evidence | Re-resolve the lab topology: confirm ``192.168.12.101`` is accepted as the reachable third physical host (Mac 2 candidate); record ``192.168.12.248`` honestly as historical/stale; run the strongest bounded live multi-host disjoint quorum cross-architecture probe practical |
| **H11** | Broad regression confidence | Full ``pytest vision_mvp/tests -q`` runs to completion at least once during the milestone with the result counted (or honestly carried forward with named exclusions); focused W22..W39 regression is green |
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
| **S1** | Stronger live multi-host evidence | Bounded live xLLM quorum probe observes either cross-architecture quorum agreement or honest divergence at a 3-physical-host topology, with gold-correlated labelling, with ``192.168.12.101`` accepted as the reachable third physical host |
| **S2** | Mac 2 stale-pin discharged | The historical ``192.168.12.248`` "Mac 2" address is honestly downgraded to historical/stale; the new ``192.168.12.101`` candidate is recorded with the diagnosed inference-path infra observation (W39-INFRA-1) if applicable |
| **S3** | Stable-vs-experimental boundary | W39 remains under ``__experimental__``; stable runtime contract unchanged |
| **S4** | Theory | Add one conditional sufficiency claim, one limitation/falsifier claim, and one native-latent gap claim |
| **S5** | Paper/master-plan synthesis | Old explicit-capsule line, dense-control/geometry line, W38 disjoint-consensus line, and W39 multi-host quorum line read as a single arc with an explicit release boundary |
| **S6** | Full broad regression actually counted | ``pytest vision_mvp/tests -q`` completes successfully and the count is recorded in the milestone report |
| **S7** | Release-candidate gate | If H1..H12 + S3 pass and the new lab topology is recorded honestly, declare a release-candidate (``RC1``) status for the SDK v3.40 line, with a final "stable vs experimental" cut-list pinned in README |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W39-L-TRIVIAL-PASSTHROUGH**: multi-host quorum disabled +
  ``allow_disjoint_quorum_divergence_abstain=False`` + manifest-v9
  disabled reduces W39 to W38 byte-for-byte.
- **W39-L-INSUFFICIENT-QUORUM**: when fewer than
  ``min_quorum_probes`` registered disjoint probes are present
  for the current cell, W39 cannot detect quorum divergence and
  must preserve W38 behavior byte-for-byte.
- **W39-L-NO-REGRESSION-QUORUM-AGREES**: when ``quorum_min`` of the
  registered disjoint probes AGREE with the W37 candidate top_set
  AND fewer than ``quorum_min`` of them disagree, W39 must ratify
  without correctness or trust-precision regression vs W38.
- **W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP**: when all K
  registered disjoint probes are themselves compromised in
  lock-step with the colluding trajectory hosts (i.e. all emit the
  same wrong top_set), W39 cannot recover at the capsule layer;
  this is the W39 analog of ``W34-L-MULTI-ANCHOR-CAP``,
  ``W37-L-MULTI-HOST-COLLUSION-CAP``, and
  ``W38-L-CONSENSUS-COLLUSION-CAP`` and is closed only by
  transformer-internal evidence which W39 does not access OR by a
  K+1-host disjoint topology.
- **W39-L-MUTUAL-DISJOINTNESS-VIOLATION**: a registered W39 quorum
  whose any two registered host pools have non-empty intersection
  is rejected at registration time (and any envelope claiming such
  a topology is rejected by the verifier).
- **W39-L-NATIVE-LATENT-GAP**: if a regime requires
  transformer-internal evidence not visible through trajectory
  observations, EWMA, response signatures, host attestations,
  anchor consensus, single-disjoint-consensus references, OR
  multi-host disjoint quorum references, W39 is insufficient.

---

## 5. Claim boundary

W39 may claim:

- an audited multi-host disjoint quorum consensus-reference
  ratification proxy with mutually-disjoint physical-host topology
  enforcement;
- measured W39-over-W38 trust-precision gain on a regime where
  W38's single disjoint consensus reference is itself compromised
  but a quorum of disjoint physical hosts catches the divergence;
- preserved W38 behavior on the trivial path, the
  insufficient-quorum falsifier, and the
  no-regression-quorum-agrees regime;
- explicit safety on the full-quorum-collusion limitation theorem
  and the mutual-disjointness-violation falsifier;
- broader regression confidence if the full ``vision_mvp/tests``
  suite is run end-to-end during the milestone;
- resolution of the historical ``192.168.12.248`` Mac 2 stale-pin
  in favor of the reachable ``192.168.12.101`` third physical host
  candidate, with honest residual infra observation if the
  inference path is bounded;
- a release-candidate declaration **only** when H1..H12 + S3 pass
  and the lab topology is recorded honestly.

W39 may not claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- closure of ``W37-L-MULTI-HOST-COLLUSION-CAP`` at the capsule layer
  (only further-bounded);
- closure of ``W38-L-CONSENSUS-COLLUSION-CAP`` at the capsule layer
  (only further-bounded by raising the adversary bar from a single
  disjoint reference to a K-of-N mutually-disjoint quorum);
- closure of the new ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``
  limitation theorem at the capsule layer;
- a true K+1-host live disjoint topology while only one
  off-cluster physical host is reachable for live inference;
- solved context for multi-agent teams;
- release readiness independent of the explicit RC1 gate.
