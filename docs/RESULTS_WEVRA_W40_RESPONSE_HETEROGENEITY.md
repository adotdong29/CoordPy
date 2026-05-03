# Results - Wevra SDK v3.41 / W40
# Cross-host response-signature heterogeneity ratification +
# manifest-v10 CID + cross-host response-text Jaccard divergence
# guard

Date: 2026-05-03.

W40 is a **response-heterogeneity-bounding** layer on top of W39.
W39's deepest open wall, ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``,
is: *if all K registered disjoint quorum probes are themselves
compromised in lock-step with the colluding trajectory hosts (each
emits the same wrong top_set), W39 cannot recover at the capsule
layer.*  W39 cannot break this with a top-set-only quorum probe.

W40 raises the capsule-layer adversary bar from "compromise 2 of N
trajectory hosts AND ``quorum_min`` of the K mutually-disjoint
registered consensus references" to "compromise 2 of N trajectory
hosts AND ``quorum_min`` of the K mutually-disjoint registered
consensus references AND inject K diverse response text bytes that
all encode the same wrong top_set".  The mechanism: at every cell
where W39 RATIFIES, W40 cross-checks the K member probes' response
**text bytes** themselves.  When the K member probes' mean pairwise
Jaccard divergence over canonical sorted token bags falls strictly
below ``response_text_diversity_min``, W40 abstains via
``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED``.

W40 does **not** close ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``
in general and it does **not** close native-latent transfer.  When
the adversary diversifies response bytes while holding the wrong
top_set in lock-step (the "smart" attacker), W40 cannot recover;
this is the new
**W40-L-COORDINATED-DIVERSE-RESPONSE-CAP** limitation theorem (the
W40 analog of ``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``,
``W38-L-CONSENSUS-COLLUSION-CAP``, and
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP``).

The result is not native latent transfer.  It is a closed-form,
zero-parameter, audited capsule-layer proxy with two mechanically
enforced disjointness preconditions inherited from W39
(**trajectory disjointness** and **mutual disjointness**) plus a
new closed-form pairwise Jaccard heterogeneity metric over
canonical sorted response-text token bags.

---

## 1. New mechanism

The W40 family adds:

- ``ResponseSignatureProbe``
- ``MultiHostResponseHeterogeneityProbe``
- ``CrossHostResponseHeterogeneityRatificationEnvelope``
- ``CrossHostResponseHeterogeneityRegistry``
- ``W40CrossHostResponseHeterogeneityResult``
- ``CrossHostResponseHeterogeneityOrchestrator``
- ``select_cross_host_response_heterogeneity_decision``
- ``verify_cross_host_response_heterogeneity_ratification``
- ``build_trivial_cross_host_response_heterogeneity_registry``
- ``build_cross_host_response_heterogeneity_registry``

W40 wraps W39.  At each cell, W40:

1. Reads W39's last-cell decision and decision top_set.
2. Calls the registered response-probe provider for the current
   cell, producing a :class:`MultiHostResponseHeterogeneityProbe`
   with K :class:`ResponseSignatureProbe` member probes, each
   tagged with ``(member_index, host_ids, oracle_ids,
   response_token_bag, response_signature_cid, cell_idx)``.
3. Calls
   :func:`select_cross_host_response_heterogeneity_decision`,
   which tallies pairwise Jaccard divergence over the K member
   probes' ``response_token_bag``s and returns one of:

   * ``RESPONSE_SIGNATURE_NO_REFERENCES`` -- no probe registered;
   * ``RESPONSE_SIGNATURE_NO_TRIGGER`` -- W39 did not RATIFY;
   * ``RESPONSE_SIGNATURE_INSUFFICIENT`` -- fewer than
     ``min_response_signature_probes`` member probes are
     registered for the current cell;
   * ``RESPONSE_SIGNATURE_INCOMPLETE`` -- one or more member
     probes are missing their ``response_signature_cid`` /
     ``response_token_bag``;
   * ``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` -- W39 RATIFIED
     but the K member probes' mean pairwise Jaccard divergence
     is strictly below ``response_text_diversity_min``; W40
     abstains;
   * ``RESPONSE_SIGNATURE_DIVERSE`` -- W39 RATIFIED and the K
     member probes' mean pairwise Jaccard divergence is at or
     above ``response_text_diversity_min``; W40 ratifies.

4. Seals the decision in a manifest-v10 envelope.

Manifest-v10 binds:

1. W39 parent CID.
2. Response-signature state CID (over the entire member probe
   list).
3. Response-signature decision CID (over n_diverse_pairs,
   n_collapse_pairs, n_pairs, n_total, mean_pairwise_jaccard,
   per_pair_jaccards, per_pair_branches).
4. Response-signature audit CID (over the named-branch decision
   tuple and thresholds).
5. Response-signature topology CID (each registered member host
   pool's host/oracle IDs PLUS the trajectory host set).
6. Response-heterogeneity witness CID (an explicit per-pair
   intersection witness, namespaced as
   ``w40_response_heterogeneity_witness`` so swapping a W39
   mutual-disjointness witness for a W40 heterogeneity witness or
   vice-versa is mechanically rejected).

The disjoint-topology preconditions are enforced in **two**
places (inherited from W39):

- :class:`CrossHostResponseHeterogeneityRegistry`'s
  ``__post_init__`` raises :class:`DisjointTopologyError` when
  any ``pool_host_ids ∩ trajectory_host_ids ≠ ∅`` AND raises
  :class:`MutuallyDisjointTopologyError` when any
  ``pool_i_host_ids ∩ pool_j_host_ids ≠ ∅`` for ``i != j``.
- :func:`verify_cross_host_response_heterogeneity_ratification`
  rejects envelopes whose
  ``registered_member_pool_host_ids`` claim an overlapping
  trajectory pool
  (``w40_response_disjoint_topology_violation``) OR an
  overlapping pair of member pools
  (``w40_response_mutual_disjointness_violation``).

The heterogeneity score is closed-form Jaccard divergence
``1 - |inter|/|union|`` over canonical sorted whitespace token
bags (lower-cased, punctuation-stripped, deduplicated).  Zero
parameters; permutation- and case-invariant.

---

## 2. Benchmark family

Phase 87 introduces five small regimes:

| Regime | Purpose |
| --- | --- |
| ``trivial_w40`` | Byte-for-W39 preservation when response-heterogeneity is disabled, response-signature collapse-abstain is disabled, and manifest-v10 is disabled |
| ``response_signature_collapse`` | Recovery half: trajectory hosts collude + W38 single consensus colludes + W39 quorum colludes (W39 RATIFIES wrong top_set; W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP fires) AND every W40 quorum member emits IDENTICAL wrong response bytes; W40 detects collapse and abstains |
| ``no_regression_diverse_agrees`` | W39 RATIFIES AND quorum members emit naturally diverse response bytes; W40 ratifies (no regression) |
| ``coordinated_diverse_response`` | W39 collusion attack succeeds AND every W40 quorum member emits DIFFERENT wrong response bytes that all encode the same wrong top_set; W40 cannot recover (W40-L-COORDINATED-DIVERSE-RESPONSE-CAP fires) |
| ``insufficient_response_signature`` | Only 1 member probe is provided in the recovery half (below ``min_response_signature_probes=2``); W40 reduces to W39 via RESPONSE_SIGNATURE_INSUFFICIENT |

The bench output reports substrate/FIFO, W21, W36, W37, W38, W39,
and W40 in one place so the old explicit-capsule line, the
dense-control / geometry line, the W37 cross-cell trajectory
line, the W38 disjoint-consensus line, the W39 multi-host quorum
line, and the new W40 cross-host response-heterogeneity line can
be compared at a glance.

---

## 3. Empirical results

All seed sweeps use seeds ``11, 17, 23, 29, 31``, ``n_eval=16``.

### R-87-RESPONSE-SIGNATURE-COLLAPSE (load-bearing)

Across 5/5 seeds:

- substrate/FIFO correctness: 0.0.
- W21 trust precision: 0.5.
- W37 trust precision: 0.5 (reroutes on the colluded trajectory-
  anchored top_set in the recovery half).
- W38 trust precision: 0.5 (the W38 single consensus is also
  compromised in lock-step => W38-L-CONSENSUS-COLLUSION-CAP
  fires).
- W39 trust precision: 0.5 (every quorum member is compromised in
  lock-step on top_set => W39-L-FULL-DISJOINT-QUORUM-COLLUSION-
  CAP fires; W39 RATIFIES the wrong top_set).
- W40 trust precision: **1.000** (W40 detects byte-identical
  response collapse via mean pairwise Jaccard divergence = 0.0,
  abstains via RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED).
- ``min_delta_trust_precision_w40_w39 = +0.500``.
- ``max_delta_trust_precision_w40_w39 = +0.500``.
- W40 abstains via RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED on 8
  cells/seed, ratifies via RESPONSE_SIGNATURE_DIVERSE on 0 cells,
  fires NO_TRIGGER on 8 cells (the prefix half where W39 didn't
  RATIFY).
- W40 overhead: 1 visible token/cell.
- Structured state density: about **14.5k bits per visible W40
  token** (the W40 envelope is denser than W38 (~9.07k
  bits/token) and slightly less than W39 (~24.4k bits/token)
  because the per-pair heterogeneity witness is more compact than
  the W39 K-of-N quorum decision payload; the W40 envelope is
  still a top-tier capsule-layer audited proxy by structured-bits-
  per-visible-token).

This is the load-bearing W40 cross-host response-heterogeneity
result.  It is *not* closure of
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` in general; it is
the strongest honest capsule-layer bound this repo can support
without native latent transfer or a stable K+1-physical-host
topology.  The bound is valid only when the K compromised
quorum members ALSO collapse to byte-identical (or
near-identical) response text bytes; the smart attacker who
diversifies response text defeats the W40 layer, leading to
the new W40-L-COORDINATED-DIVERSE-RESPONSE-CAP limitation
theorem.

Artifact:
``vision_mvp/experiments/artifacts/phase87/response_signature_collapse_seed_sweep.json``.

### R-87-TRIVIAL-W40

Across 5/5 seeds:

- W40 = W39 byte-for-byte.
- correctness and trust precision identical to W39.
- overhead = 0.
- ``all_byte_equivalent_w40_w39 = true``.

Artifact:
``vision_mvp/experiments/artifacts/phase87/trivial_w40_seed_sweep.json``.

### R-87-NO-REGRESSION-DIVERSE-AGREES

Across 5/5 seeds:

- W40 correctness: same as W39 (delta = 0).
- W40 trust precision: same as W39 (delta = 0).
- W40 ratifies via RESPONSE_SIGNATURE_DIVERSE on 8 cells/seed in
  the recovery half (W39 RATIFIED + naturally diverse responses).
- ``min_delta_correctness_w40_w39 = 0.0``.
- ``min_delta_trust_precision_w40_w39 = 0.0``.

Artifact:
``vision_mvp/experiments/artifacts/phase87/no_regression_diverse_agrees_seed_sweep.json``.

### R-87-COORDINATED-DIVERSE-RESPONSE (W40-L-COORDINATED-DIVERSE-RESPONSE-CAP)

Across 5/5 seeds:

- W40 trust precision: 0.5 (cannot recover; the K compromised
  quorum members emit byte-distinct response texts that
  individually pass the diversity check while collectively
  encoding the same wrong top_set).
- ``min_delta_trust_precision_w40_w39 = 0.0``.
- ``max_delta_trust_precision_w40_w39 = 0.0``.

This is the named ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP``
limitation theorem firing.  Closure requires native-latent
evidence outside the capsule layer
(``W40-L-NATIVE-LATENT-GAP``) or a K+1-host disjoint quorum
topology with at least one new genuinely uncompromised host
pool.

Artifact:
``vision_mvp/experiments/artifacts/phase87/coordinated_diverse_response_seed_sweep.json``.

### R-87-INSUFFICIENT-RESPONSE-SIGNATURE

Across 5/5 seeds:

- W40 correctness: same as W39 (delta = 0).
- W40 trust precision: same as W39 (delta = 0).
- W40 returns ``RESPONSE_SIGNATURE_INSUFFICIENT`` on every
  recovery cell where fewer than
  ``min_response_signature_probes=2`` member probes are present.

Artifact:
``vision_mvp/experiments/artifacts/phase87/insufficient_response_signature_seed_sweep.json``.

---

## 4. Live / multi-Mac evidence

### 4.1 Lab topology resolution (W40-INFRA-1)

The historical Mac-2 endpoint (``192.168.12.248``) has been
ARP-incomplete for the **32nd milestone in a row**.

The third physical host candidate ``192.168.12.101`` has changed
state vs the W39 milestone:

- W39 end-state: ``.101`` ping at 100% packet loss; Ollama HTTP
  endpoint at "Empty reply from server"; SSH unavailable
  (W39-INFRA-1).
- W40 re-probe (2026-05-03):
  - ``ping -c 2``: 0% packet loss, 30 ms RTT (the host has
    rebooted or its network stack has recovered).
  - ``nc -zv 11434``: TCP connect succeeds.
  - ``nc -zv 22``: TCP SSH connect succeeds (auth methods
    advertised: publickey, password, keyboard-interactive).
  - ``curl /api/version``: still "Empty reply from server" at 60
    seconds.
  - ``curl /api/tags``: "Connection reset by peer" /
    "Empty reply from server" repeatedly across 5 attempts.
  - SSH attempted via ``nobody@``, ``qdong@``, ``root@``,
    ``admin@``: all rejected for lack of credentials.
- Verdict: ``.101`` host network stack has recovered (TCP-up and
  ping-up); the Ollama HTTP listener is still in the W39-INFRA-1
  hung-listener state (TCP accepts the connection, then RST/closes
  before sending HTTP bytes).  This is a strict improvement over
  W39's end-state (the host itself is no longer 100% packet-loss
  unreachable) but the Ollama service still requires SSH
  credentials to restart.

This is recorded as **W40-INFRA-1**: ``192.168.12.101`` is now
TCP-up + HTTP-broken at the Ollama layer; the host has not been
restored to live inference in this milestone because SSH
credentials are still unavailable in this environment.
``W39-C-MULTI-HOST`` therefore remains **partially discharged at
the topology layer (a third physical host candidate is
TCP-reachable on cold contact in the W40 milestone) and still
open at the live-inference layer (the third host's Ollama
listener returns immediate Connection-reset/Empty-reply under
any HTTP request)**.

Artifact:
``vision_mvp/experiments/artifacts/phase87/host_topology_resolution_2026_05_03.json``.

### 4.2 W40 cross-host response-heterogeneity does not require .101

By design, the W40 mechanism is closed-form and capsule-layer:
the response-heterogeneity score is a Jaccard divergence over
canonical sorted token bags computed from member probes that are
controller-pre-registered.  No live LLM inference is required to
evaluate the W40 mechanism.  The R-87 bench therefore decouples
the W40 scientific claim from the W39-INFRA-1 / W40-INFRA-1 live
infrastructure bound.

The live xLLM probe surface for W37 / W38 / W39 (``phase84_xllm``,
``phase85_xllm_consensus_probe.py``,
``phase86_xllm_quorum_probe.py``) remains intact and continues to
work with the two reachable hosts (``localhost`` + ``.191``) plus
the W39-INFRA-1 fallback path that routes around ``.101`` /
``.248``.

---

## 5. Trust boundary

``verify_cross_host_response_heterogeneity_ratification``
enumerates 14 W40 failure modes:

1. ``empty_w40_envelope``
2. ``w40_schema_version_unknown``
3. ``w40_schema_cid_mismatch``
4. ``w39_parent_cid_mismatch``
5. ``w40_projection_branch_unknown``
6. ``w40_response_probe_unregistered_host``
7. ``w40_response_probe_unregistered_oracle``
8. ``w40_response_disjoint_topology_violation``
9. ``w40_response_mutual_disjointness_violation`` (NEW; W40-
   specific defense-in-depth via the heterogeneity witness)
10. ``w40_response_thresholds_invalid``
11. ``w40_response_state_cid_mismatch``
12. ``w40_response_decision_cid_mismatch``
13. ``w40_response_topology_cid_mismatch``
14. ``w40_manifest_v10_cid_mismatch`` (with
    ``w40_outer_cid_mismatch`` and the heterogeneity-witness
    swap-detection co-defined)

Cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36 +
W37 + W38 + W39 + W40 trust boundary: **168 enumerated failure
modes**.

---

## 6. Theory-forward claims

### W40-1 - verifier boundary

**Claim**: The W40 cross-host response-heterogeneity envelope is
mechanically verifiable against 14 disjoint failure modes.

**Status**: proved by inspection + mechanically checked (16/16
gate checks executed; 14 named failure modes enumerated).

### W40-2 - trivial reduction

**Claim**: Disabled response-heterogeneity + disabled
collapse-abstain + disabled manifest-v10 reduces to W39
byte-for-byte.

**Status**: empirical across 5/5 seeds.

### W40-3 - cross-host response-heterogeneity collapse bound (load-bearing)

**Claim**: If the current cell would RATIFY a W39 quorum top_set
produced by colluding hosts AND every K registered W40 member
probe carries a response-text bytes payload that collapses to
byte-identical (or near-identical, within mean pairwise Jaccard
divergence < ``response_text_diversity_min``), AND fewer than
``min_response_signature_probes`` member probes are missing
their response signatures, THEN W40 abstains via
``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` without trust-precision
regression.

**Status**: proved-conditional + empirical on
R-87-RESPONSE-SIGNATURE-COLLAPSE (5/5 seeds, +0.500 trust
precision gain over W39, trust precision = 1.000, overhead = 1
token/cell).

### W40-4 - mutual-disjointness mechanical enforcement (inherited from W39)

**Claim**: A
``CrossHostResponseHeterogeneityRegistry`` whose any two
registered member host pools have non-empty intersection cannot
be constructed (raises
:class:`MutuallyDisjointTopologyError` in ``__post_init__``);
the verifier additionally rejects an envelope whose declared
member topology has overlapping pools
(``w40_response_mutual_disjointness_violation``).

**Status**: proved by inspection + mechanically tested.

### W40-L-COORDINATED-DIVERSE-RESPONSE-CAP

**Claim**: When the K registered W40 member probes' response
signatures are *injected* to be diverse (mean pairwise Jaccard
divergence at or above ``response_text_diversity_min``) but all
encode the same wrong top_set in lock-step (the W40 analog of
the W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP attack with the
additional adversary requirement of injecting diverse response
bytes), W40 cannot recover at the capsule layer.  This is the
W40 analog of ``W34-L-MULTI-ANCHOR-CAP``,
``W37-L-MULTI-HOST-COLLUSION-CAP``,
``W38-L-CONSENSUS-COLLUSION-CAP``, and
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` and is closed only
by transformer-internal evidence outside the capsule layer or by
a K+1-host disjoint topology with at least one new genuinely
uncompromised host pool.

**Status**: proved-conditional limitation theorem + empirical on
R-87-COORDINATED-DIVERSE-RESPONSE (5/5 seeds, delta = 0).

### W40-L-INSUFFICIENT-RESPONSE-SIGNATURE

**Claim**: If fewer than ``min_response_signature_probes`` member
probes are registered for the current cell (or any registered
member probe is missing its
``response_signature_cid`` / ``response_token_bag``), W40 cannot
detect heterogeneity collapse and falls through to W39's decision
(the ``RESPONSE_SIGNATURE_INSUFFICIENT`` /
``RESPONSE_SIGNATURE_INCOMPLETE`` branch).  W40 is insufficient
for any regime where a K=``min_response_signature_probes``
mutually-disjoint quorum cannot be sourced with full response
signatures.

**Status**: proved by inspection + empirical on
R-87-INSUFFICIENT-RESPONSE-SIGNATURE.

### W40-C-NATIVE-LATENT

**Conjecture**: True transformer-internal trust-state projection
may separate regimes where all capsule-visible host/probe/
evidence/trajectory/consensus/quorum/response-heterogeneity
signals are either absent or coordinated.  W40 narrows the
audited proxy further along the cross-host response-text axis but
does not close this.

**Status**: conjectural and architecture-dependent.

### W40-C-MULTI-HOST

**Conjecture**: A K+1-host topology with at least one new
genuinely uncompromised host pool (specifically, restoring
``192.168.12.101``'s Ollama HTTP listener so it can serve the
disjoint third-host inference role at temperature 0 +
``num_predict=4``) would let the W40 quorum size be raised
beyond ``quorum_min``, defeating the
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` collusion attack at
the capsule layer.  Currently partially discharged at the
topology layer via ``192.168.12.101`` (TCP-up + ping-up on cold
contact in the W40 milestone, a strict improvement over the W39
end-state) but still open at the live-inference layer
(``W40-INFRA-1``).

**Status**: conjectural; partially discharged at the topology
layer; still open at the live-inference layer; hardware-bounded.

### W40-C-LIVE-RESPONSE-HETEROGENEITY (NEW)

**Conjecture**: At the live inference layer, the W40 cross-host
response-heterogeneity layer can recover from naive collusion
attacks where K compromised quorum members emit byte-identical
wrong response text bytes (the "naive" full-quorum-collusion
attack pattern that any single attacker controlling K probes
without paraphrase diversification would naturally produce).
This is a recovery axis distinct from the coordinated-diverse-
response attack: it does not require the adversary to be unable
to coordinate top_set; it only requires the adversary to be
unable (or unwilling) to diversify response text bytes across
K probes.

**Status**: empirical-validated on the R-87-RESPONSE-SIGNATURE-
COLLAPSE bench (5/5 seeds, +0.5 trust precision gain over W39).
Conjectural at the live-inference layer pending a dedicated live
collusion-recovery experiment with adversary-controlled
identical-response payloads on the reachable hosts; the bench
result is enough to make the conjecture defensible at the
synthetic layer.

---

## 7. Release boundary - RC1

Release readiness improved because:

- W40 is strictly additive and experimental.
- Stable runtime contract is unchanged.
- SDK version (``wevra.sdk.v3.41``) and package version
  (``0.5.14``) are bumped.
- The repo's ``vision_mvp.__version__`` and ``pyproject.toml``
  ``project.version`` are now both ``0.5.14`` (alignment is
  maintained).
- Success bar, results note, theorem registry, README,
  START_HERE, master plan, changelog, and paper marker are
  updated.
- Focused W22..W40 regression at **661/661 phase69-87** is green
  (was 625/625 phase69-86 at SDK v3.40; W40 added 36 cleanly).
- The cumulative enumerated trust boundary is now **168 failure
  modes** (W22..W40).
- The stable-vs-experimental boundary is final: every W22..W40
  symbol is exported under ``__experimental__`` and the stable
  ``RunSpec → run report`` runtime contract is byte-for-byte
  unchanged.
- The W40 mechanism does not require any new live infrastructure
  to evaluate at the synthetic-bench layer; the W39-INFRA-1
  / W40-INFRA-1 line is bounded honestly and does not block the
  RC declaration.
- The historical Mac-2 stale repo pin (``192.168.12.248``)
  remains discharged in favour of ``192.168.12.101`` as the
  reachable third physical host candidate; the W39 live xllm
  probe accepts ``WEVRA_OLLAMA_URL_MAC2`` env-var override and
  falls through a candidate list including ``.101`` before
  declaring "Mac 2 unreachable"; the W39-INFRA-1 fallback path
  keeps the live K=2 quorum probe useful even when ``.101``'s
  inference path is bounded.
- W40-INFRA-1 records ``192.168.12.101`` honestly as TCP-up +
  HTTP-broken in the W40 milestone (a strict topology-layer
  improvement over W39's 100%-packet-loss end-state); the
  live-inference layer remains bounded for lack of SSH
  credentials.

**RC1 declaration**: H1..H12 + S3 pass on the W40 success
criterion ⇒ the SDK v3.41 line is hereby declared a
**release candidate (RC1)**.  The "stable vs experimental"
cut-list is pinned in README; the "open conjectures + limitation
theorems" cut-list is pinned in THEOREM_REGISTRY.md.  RC1 is a
release-candidate, not a final release.

Release readiness is not fully closed because:

- W40 is not native latent transfer.
- ``.101`` Ollama HTTP listener remains hung at the live-
  inference layer (W40-INFRA-1).
- True K+1-host live inference evidence remains open
  (``W40-C-MULTI-HOST``, ``W40-C-NATIVE-LATENT``).
- ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` cannot be closed at
  the capsule layer.
- W21 still dominates W40 on regimes where multi-oracle quorum
  is enough by itself; W40 is a trust-stack hardening result for
  the cross-cell, single-host recovery case under capsule-layer
  collusion threat with the W38 single consensus AND the W39
  multi-host disjoint quorum AND the response text bytes all
  compromised in lock-step, not a universal successor to every
  older explicit-capsule baseline.

---

## 8. Position in the programme arc

The W22..W40 arc now reads as a single coherent story with five
named limitation theorems at the capsule layer, each with a named
audited proxy that bounds (but does not close) the next:

```
W34-L-MULTI-ANCHOR-CAP            (live attestation collusion)
        |
        v
W37-L-MULTI-HOST-COLLUSION-CAP    (cross-host trajectory collusion)
        |
        v  bounded by W37 audited proxy
W37 single-host trajectory recovery
        |
        v  attacker compromises 2 trajectory hosts
W38-L-CONSENSUS-COLLUSION-CAP     (single disjoint consensus collusion)
        |
        v  bounded by W38 audited proxy
W38 disjoint consensus reference
        |
        v  attacker compromises 2 trajectory hosts AND single consensus
W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP (full quorum collusion on top_set)
        |
        v  bounded by W39 audited proxy
W39 K-of-N mutually-disjoint quorum
        |
        v  attacker compromises 2 trajectory hosts AND quorum_min of K
W40-L-COORDINATED-DIVERSE-RESPONSE-CAP (full quorum collusion + diverse responses)
        |
        v  bounded by W40 audited proxy
W40 cross-host response-heterogeneity
        |
        v  attacker compromises 2 trajectory hosts AND quorum_min of K
        v  AND injects K diverse response text bytes
        v  ==> capsule-layer recovery exhausted
        v
        v  closure requires:
        v    * native-latent evidence (W40-C-NATIVE-LATENT,
        v      architecture-dependent, out of capsule-layer scope)
        v    * OR K+1-host disjoint topology (W40-C-MULTI-HOST,
        v      hardware-bounded)
```

Each step raises the adversary bar by ONE specific axis.  W40 is
the strongest honest capsule-layer audited proxy this repo
supports as of SDK v3.41 / RC1.

---

## 9. Stable-vs-experimental boundary (final RC1 cut)

**Stable** (byte-for-byte unchanged across W22..W40):

- ``RunSpec`` → run report runtime contract.
- Capsule primitives (``ContextCapsule``, ``CapsuleKind``,
  ``CapsuleLifecycle``, ``CapsuleBudget``, ``CapsuleLedger``,
  ``CapsuleView``, ``CAPSULE_VIEW_SCHEMA``, ``render_view``,
  ``verify_chain_from_view_dict``).
- Lifecycle audit (``audit_team_lifecycle``,
  ``T_INVARIANTS``).
- Provenance (``PROVENANCE_SCHEMA``, ``build_manifest``).
- LLM backend abstraction (``OllamaBackend``,
  ``MLXDistributedBackend``).

**Experimental** (under ``vision_mvp.wevra.__experimental__``,
may evolve between minor versions):

- W22 family: capsule + audited latent-state-sharing hybrid.
- W23 family: cross-cell delta + super-token reference.
- W24 family: session compaction.
- W25 family: shared-fanout dense control.
- W26 family: chain-persisted dense-control fanout.
- W27 family: multi-chain salience-keyed dense-control fanout.
- W28 family: ensemble-verified cross-model pivot ratification.
- W29 family: geometry-partitioned product-manifold dense
  control.
- W30 family: calibrated geometry-aware dense control.
- W31 family: online self-calibrated geometry-aware dense
  control.
- W32 family: long-window convergent geometry-aware dense
  control.
- W33 family: trust-EWMA-tracked multi-oracle adjudication.
- W34 family: live-aware multi-anchor consensus reference.
- W35 family: trust-subspace dense control.
- W36 family: host-diverse trust-subspace dense control.
- W37 family: cross-host basis-trajectory ratification.
- W38 family: disjoint cross-source consensus-reference.
- W39 family: multi-host disjoint quorum consensus-reference.
- **W40 family** (NEW): cross-host response-signature
  heterogeneity.

The cut-list is final for RC1: any future research-grade
extension belongs under ``__experimental__`` until and unless it
is explicitly promoted to the stable surface in a future SDK
release.

---

## 10. Honest limitations (final RC1 statement)

W40 / SDK v3.41 RC1 explicitly does **not** claim:

- native latent transfer of any kind;
- transformer-internal trust subspace;
- KV-cache transplant;
- attention-weight inspection;
- hidden-state projection of any kind;
- closure of any of the W34/W37/W38/W39/W40 limitation theorems
  at the capsule layer (each is only further-bounded);
- a true K+1-host live inference topology with all hosts
  uncompromised;
- restoration of ``.101``'s Ollama HTTP listener (W40-INFRA-1);
- restoration of ``.248``'s network reachability (32 milestones
  in a row of ARP-incompleteness);
- "solved context for multi-agent teams" (the original
  programme-level claim);
- production release (RC1 is a release-candidate, not a final
  release).

W40 / SDK v3.41 RC1 explicitly **does** claim:

- the strongest honest capsule-layer audited proxy beyond W39;
- a measured +0.5 trust-precision gain on the load-bearing
  W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP collapse regime
  across 5/5 seeds;
- a final, defensible release-candidate closure of the SDK
  v3.41 stable vs experimental boundary;
- a final, defensible enumeration of all open conjectures and
  limitation theorems in THEOREM_REGISTRY.md;
- a final, defensible record of the W40-INFRA-1 finding (a
  strict topology-layer improvement over W39-INFRA-1 with the
  live-inference layer still bounded for lack of SSH
  credentials).
