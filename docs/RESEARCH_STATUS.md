# Research status — canonical, current

> Single-source-of-truth for the *active* research position of the
> Context Zero programme. If this file disagrees with any other
> doc on what is *true now*, this file is right and the other file
> is stale. For *theorem-by-theorem* status, see
> `docs/THEOREM_REGISTRY.md`. For *what may be claimed*, see
> `docs/HOW_NOT_TO_OVERSTATE.md`. Last touched: SDK v3.39,
> 2026-05-02.

## TL;DR — SDK v3.39

The programme now has **thirty-five** coupled research axes.  SDK
v3.39 mints axis 35: **disjoint cross-source consensus-reference
trajectory-divergence adjudication + manifest-v8 CID** (W38).  W38
wraps W37's anchor-cross-host basis-trajectory ratification with a
controller-pre-registered ``ConsensusReferenceProbe`` whose host
topology is *mechanically disjoint* from W37's trajectory hosts (the
``DisjointConsensusReferenceRegistry`` raises
``DisjointTopologyError`` otherwise; the verifier additionally
rejects envelopes claiming an overlapping topology).  When W37 chooses
to reroute on a trajectory-anchored top_set and the disjoint consensus
reference disagrees within ``divergence_margin_min`` (Jaccard), W38
abstains via the ``CONSENSUS_DIVERGENCE_ABSTAINED`` branch.

The load-bearing change is *cross-source* rather than cross-cell or
cross-host.  W37 raises the adversary bar to "compromise 2 of N
trajectory hosts coordinately" but cannot break that bar at the
capsule layer (``W37-L-MULTI-HOST-COLLUSION-CAP``).  W38 raises it
to "compromise 2 of N trajectory hosts AND the disjoint registered
consensus reference".  When the disjoint reference is itself
compromised in lock-step, W38 cannot recover; this is the new
proved-conditional ``W38-L-CONSENSUS-COLLUSION-CAP`` limitation
theorem.

**Headline SDK v3.39 results.**

* **R-85-COLLUDED-CROSS-HOST-TRAJECTORY (load-bearing).**  W38
  improves over W37 from 0.500 to **1.000** trust precision across
  5/5 seeds, **Δ_trust_precision_w38_w37 = +0.500** (min and max
  equal).  W38 reroutes 0 cells, abstains via DIVERGENCE on 8
  cells/seed; overhead = 1 visible token/cell.

* **R-85-TRIVIAL-W38.**  With consensus disabled, divergence-abstain
  disabled, and manifest-v8 disabled, W38 reduces to W37
  byte-for-byte across 5/5 seeds;
  ``all_byte_equivalent_w38_w37 = True``.

* **R-85-NO-COLLUSION-CONSENSUS-AGREES.**  Consensus reference
  agrees with the W37 reroute; W38 ratifies; no correctness or
  trust-precision regression vs W37.

* **R-85-CONSENSUS-ALSO-COMPROMISED.**  Hard falsifier: when the
  disjoint consensus reference is itself compromised in lock-step
  with the colluding trajectory hosts, W38 cannot recover.  Δ_trust_
  precision = 0.000; W38-L-CONSENSUS-COLLUSION-CAP fires.

* **R-85-NO-CONSENSUS-REFERENCE.**  Falsifier: when no consensus
  probe is registered, W38 returns ``CONSENSUS_NO_REFERENCE``;
  correctness/trust-precision unchanged from W37; the W38 audit
  envelope still records the NO_REFERENCE branch.

**Density / efficiency**: On R-85-COLLUDED-CROSS-HOST-TRAJECTORY,
W38 carries about **9,072 structured bits per visible W38 token**
at one visible token overhead/cell.  This is controller-verified
structured cross-source consensus-reference state, not hidden-state
transfer.

**Trust boundary**: 14 new W38 failure modes in
``verify_disjoint_consensus_reference_ratification``, mechanically
tested.  Cumulative enumerated trust boundary across W22 + W29 +
W30 + W31 + W32 + W33 + W34 + W35 + W36 + W37 + W38 =
**140 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`
(Mac 2 ARP-incomplete for the **31st milestone in a row**; ``ping``
reports "Host is down").  The bounded W38 cross-source consensus
probe used local ``gemma2:9b`` and remote ``qwen2.5:14b`` as
trajectory hosts and remote ``qwen3.5:35b`` as the disjoint
consensus host (different model class on the same physical host;
defensible weak proxy for capsule-layer disjointness, not a true
3-host disjoint topology).  Result recorded in
``vision_mvp/experiments/artifacts/phase85/xllm_consensus_probe_2026_05_02.json``.

**Stable-vs-experimental boundary**: W38 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `wevra.sdk.v3.39`; package version `0.5.12`.  The
lingering ``vision_mvp.__version__ == "0.5.9"`` vs
``pyproject.toml == "0.5.11"`` misalignment from earlier milestones
is closed: both are now ``0.5.12``.

**Open walls after W38**: W38-L-CONSENSUS-COLLUSION-CAP is a new
proved-conditional limitation theorem at the capsule layer (when the
disjoint consensus reference is also compromised, recovery requires
native-latent evidence outside the capsule layer or a 3+-host
disjoint consensus topology).  W38-C-NATIVE-LATENT remains open.
W38-C-MULTI-HOST remains hardware-bounded until Mac 2 (or a third
reachable host) joins the topology.  Cross-source consensus audit is
now sealed in manifest-v8; cross-cell trajectory audit remains in
manifest-v7; per-cell host-diversity audit remains in manifest-v6;
per-cell live-aware multi-anchor audit remains in manifest-v4; the
cross-source × cross-cell × per-cell audit boundary is now
structurally explicit in the envelope hierarchy.

See `docs/RESULTS_WEVRA_W38_DISJOINT_CONSENSUS_REFERENCE.md` for the
full note and
`docs/SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md` for the
success bar.

---

## Earlier TL;DR — SDK v3.38

The programme now has **thirty-four** coupled research axes.  SDK
v3.38 mints axis 34: **anchor-cross-host basis-trajectory ratification
+ manifest-v7 CID** (W37).  W37 wraps W36's host-diverse trust-subspace
guard with a closed-form, zero-parameter, per-(host, oracle, top_set)
EWMA over *anchored* historical observations.  W36 abstains whenever
the current cell has fewer than `min_distinct_hosts` healthy attested
hosts -- even when the remaining single host has been independently
anchored across multiple earlier cells by other healthy hosts.  W37
makes that historical cross-host anchoring a typed audited precondition
for a safe single-host reroute.

The load-bearing change is cross-cell rather than per-cell.  W36 reads
one cell at a time; W37 maintains a sealed trajectory of which (host,
oracle, top_set) entries have been *cross-host anchored* and converts
historically-anchored evidence into a safe reroute on a single-host
recovery cell.  The trajectory update is gated by *select-before-update*
so the historical anchoring is not diluted by the recovery cell itself,
and is gated by *multi-host-only update* so a host is not penalised
for being the only reachable host when the live infrastructure offers
no co-attesters.

**Headline SDK v3.38 results.**

* **R-84-SINGLE-HOST-TRAJECTORY-RECOVER.**  W37 improves over W36 from
  0.500 to **1.000** correctness across 5/5 seeds, **Δ_correctness_
  w37_w36 = +0.500** (min and max equal), and trust precision stays at
  **1.000**.  W37 reroutes 8 cells per seed; overhead = 1 visible
  token/cell.

* **R-84-TRIVIAL-W37.**  With trajectory disabled, single-host reroute
  disabled, and manifest-v7 disabled, W37 reduces to W36 byte-for-byte
  across 5/5 seeds; `all_byte_equivalent_w37_w36 = True`.

* **R-84-NO-TRAJECTORY-HISTORY.**  Falsifier: when no cross-host
  anchoring ever forms (every cell single-host), W37 = W36 = 0.000
  correctness; W37 trust precision = 1.000; W37 abstains 16/16
  cells/seed.

* **R-84-POISONED-TRAJECTORY.**  Falsifier: a trajectory accumulated on
  a single host without cross-host anchoring never satisfies the
  `min_trajectory_anchored_hosts` requirement; W37 does not reroute and
  preserves W36 abstention.

* **R-84-TRAJECTORY-DISAGREEMENT.**  Falsifier: when the current
  cell's basis emits a top_set that does not match the historically
  anchored trajectory, W37 must not reroute on the historically
  anchored top_set; W37 = W36 = 0.500 correctness.

**Density / efficiency**: On R-84-SINGLE-HOST-TRAJECTORY-RECOVER, W37
carries about **29.5k structured bits per visible W37 token** at one
visible token overhead/cell.  This is controller-verified structured
trajectory state, not hidden-state transfer.

**Trust boundary**: 14 new W37 failure modes in
`verify_cross_host_trajectory_ratification`, mechanically tested.
Cumulative enumerated trust boundary across W22 + W29 + W30 + W31 +
W32 + W33 + W34 + W35 + W36 + W37 = **126 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`
(Mac 2 ARP-incomplete for the **30th milestone in a row**).  The
bounded W37 cross-host trajectory probe on 2026-05-02 across local
`gemma2:9b` and remote `qwen2.5:14b` produced **8/8 responsive
probes, 8/8 cross-host anchored agreements, and 8/8 gold-correlated
agreements** at temperature 0 on the 8-prompt panel.  This is the
strongest two-reachable-host live trajectory evidence the
infrastructure currently supports; it does not close W37-C-MULTI-HOST.

**Stable-vs-experimental boundary**: W37 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `wevra.sdk.v3.38`; package version `0.5.11`.

**Open walls after W37**: W37-L-MULTI-HOST-COLLUSION-CAP is a new
proved-conditional limitation theorem at the capsule layer (two
colluding hosts can cross the anchored thresholds; recovery requires
native-latent evidence outside the capsule layer).  W37-C-NATIVE-LATENT
remains open.  W37-C-MULTI-HOST remains hardware-bounded until Mac 2
(or a third reachable host) joins the topology.  Cross-cell trajectory
audit is now sealed in manifest-v7; per-cell host-diversity audit
remains in manifest-v6; per-cell live-aware multi-anchor audit remains
in manifest-v4; the cross-cell × per-cell audit boundary is now
structurally explicit in the envelope hierarchy.

See `docs/RESULTS_WEVRA_W37_CROSS_HOST_BASIS_TRAJECTORY.md` for the
full note and
`docs/SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md` for the
success bar.

---

## Earlier TL;DR — SDK v3.37

The programme had **thirty-three** coupled research axes.  SDK
v3.37 minted axis 33: **host-diverse trust-subspace guard +
manifest-v6 CID** (W36).  W36 wraps W35's audited trust-subspace
dense-control proxy with a controller-side host-diversity verifier:
dense projection support must come from at least two distinct
registered healthy hosts, and unsafe or unverifiable branches reject
or abstain.

The load-bearing change is narrower than native latent transfer and
more operational than another local reroute.  W35 could ratify a
dense basis projection even when the available capsule-visible support
was effectively co-located or host-unsafe.  W36 makes host diversity a
typed audited precondition.

**Headline SDK v3.37 results.**

* **R-83-HOST-DIVERSE-RECOVER.**  W36 improves over W35 from 0.625
  to **0.9375** correctness across 5/5 seeds, **Δ_correctness_w36_w35
  = +0.3125**, and restores trust precision from 0.6667 to **1.000**.
  W36 reroutes 5 cells and abstains on 1 unsafe W35 ratification per
  seed.  W21 remains at 1.000 on this synthetic regime, so the claim
  is trust-stack hardening, not universal dominance over every older
  explicit-capsule baseline.

* **R-83-HOST-SPOOFED-CONSENSUS.**  W36 does not recover correctness:
  W35 and W36 both stay at 0.625.  It improves trust precision from
  0.625 to **1.000** by abstaining on 6 unsafe W35 ratifications per
  seed.  This is the named spoofed-host falsifier.

* **R-83-TRIVIAL-W36.**  With host diversity disabled and manifest-v6
  disabled, W36 reduces to W35 byte-for-byte across 5/5 seeds.

* **R-83-NO-LIVE-ATTESTATION.**  Falsifier: if host diversity is
  required but live attestations are absent, W36 abstains on every
  cell and drops correctness from W35's 1.000 to 0.000 while keeping
  trust precision at 1.000.

**Density / efficiency**: On R-83-HOST-DIVERSE-RECOVER, W36 carries
about **13.95k structured bits per visible W36 token** at one visible
token overhead/cell.  On R-83-HOST-SPOOFED-CONSENSUS it carries about
**13.74k bits/token**.  This is controller-verified structured state,
not hidden-state transfer.

**Trust boundary**: 14 new W36 failure modes in
`verify_host_diverse_ratification`, mechanically tested.  Cumulative
enumerated trust boundary across W22 + W29 + W30 + W31 + W32 + W33 +
W34 + W35 + W36 = **112 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`.
The bounded W36 two-reachable-host probe on 2026-05-02 across local
`qwen2.5:0.5b` and remote `qwen2.5:14b` yielded 10/10 responsive
probes, 4/5 cross-host disagreements, and 4/4 gold-correlated
disagreements.  This materially strengthens the two-reachable-host
motivation for host-diverse control but still does **not** close true
three-host evidence.

**Stable-vs-experimental boundary**: W36 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `wevra.sdk.v3.37`; package version `0.5.10`.

**Open walls after W36**: W33-C-NATIVE-LATENT remains open.  W36 is
not transformer-internal hidden-state projection, not a KV transplant,
and not a learned latent controller.  W33-C-CROSS-HOST-LIVE-TRUST-
MAGNITUDE remains open on the systematic magnitude axis.  W34/W35/
W36-C-MULTI-HOST remains hardware-bounded until Mac 2 joins the
topology.  W36 also adds a new operational wall: host-diverse dense
control is unsafe without real live attestations.

See `docs/RESULTS_WEVRA_W36_HOST_DIVERSE_TRUST_SUBSPACE.md` for the
full note and `docs/SUCCESS_CRITERION_W36_HOST_DIVERSE_TRUST_SUBSPACE.md`
for the success bar.

---

## Earlier TL;DR — SDK v3.36

The programme now has **thirty-two** coupled research axes.  SDK
v3.36 mints axis 32: **trust-subspace dense-control proxy +
basis-history projection + manifest-v5 CID** (W35).  W35 wraps W34's
live-aware multi-anchor path with the strongest honest native-latent
proxy this repo can support without transformer-runtime access: a
controller-verified dense basis over W21 probe top_sets, W33 EWMA
trust, W34 live-attestation/response-feature state, top-set stability,
and host health.

The load-bearing change is narrow and real.  W34 treated anchor
disagreement as a trust signal and abstained.  W35 asks whether the
verified basis history contains a stable, high-margin trusted
direction that can safely convert that abstention into a reroute.  If
the basis is short, unstable, insufficiently separated, or
unverifiable, W35 preserves W34's abstention.

**Headline SDK v3.36 results.**

* **R-82-TRUST-SUBSPACE-SHIFT.**  W34 abstains on 6 disputed cells;
  W35 reroutes 5/6 through the stable `change_history` basis
  direction.  Correctness rises from 0.625 (W21/W33/W34) to
  **0.9375** across 5/5 seeds, **Δ_correctness_w35_w34 = +0.3125**,
  while trust precision stays at **1.000** and overhead is one visible
  token/cell.

* **R-82-TRIVIAL-W35.**  With trust-subspace disabled and manifest-v5
  disabled, W35 reduces to W34 byte-for-byte across 5/5 seeds.

* **R-82-NO-ANCHOR-DISAGREEMENT.**  When W34 already has consensus,
  W35 adds no correctness lift; W35 = W34 = 1.000 correctness/trust
  precision.

* **R-82-FROZEN-BASIS.**  Weakened basis history attenuates but does
  not remove the gain: W35 correctness 0.875, delta +0.2500, four
  reroutes.

* **R-82-ALL-ANCHOR-COMPROMISED.**  Hard falsifier: when every basis
  direction moves together to the wrong answer, W35 cannot recover.
  W35-W34 delta = 0.000; trust precision remains 0.625.

**Density / efficiency**: On R-82-TRUST-SUBSPACE-SHIFT, W35 carries
mean **208,264 structured bits per 16-cell seed** at one visible
token overhead/cell, or **13,016.5 structured bits per visible W35
token**.  This is dense controller-verified state transfer, not hidden
state transfer.

**Trust boundary**: 14 new W35 failure modes in
`verify_trust_subspace_dense_ratification`, now mechanically tested.
Cumulative enumerated trust boundary across W22 + W29 + W30 + W31 +
W32 + W33 + W34 + W35 = **98 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`
despite the user reopening a Mac.  A bounded two-host fallback probe
on 2026-05-02 across local `qwen2.5:0.5b` and remote `qwen2.5:14b`
yielded 10/10 responsive probes, 3/5 cross-host disagreements, and
3/3 gold-correlated disagreements.  This strengthens live disagreement
evidence but does **not** close the stronger live magnitude survey or
true multi-host blocker.

**Stable-vs-experimental boundary**: W35 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `wevra.sdk.v3.36`; package version `0.5.9`.

**Open walls after W35**: W33-C-NATIVE-LATENT remains open.  W35 is
not transformer-internal hidden-state projection, not a KV transplant,
and not a learned latent controller.  W33-C-CROSS-HOST-LIVE-TRUST-
MAGNITUDE remains open on the systematic magnitude axis.  W34-C-
MULTI-HOST remains hardware-bounded until Mac 2 joins the topology.

See `docs/RESULTS_WEVRA_W35_TRUST_SUBSPACE_DENSE_CONTROL.md` for the
full note and `docs/SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`
for the success bar.

---

## Earlier TL;DR — SDK v3.35

The programme now has **thirty-one** coupled research axes, each
with a sharp status.  SDK v3.35 mints axis 31: **live-aware
multi-anchor adjudication + native-latent audited response-feature
proxy + W34 manifest-v4 CID + W33 infra-blocker closure (preflight
``/api/tags`` check + chat-template + ``num_predict=4`` for one-word
probes)** — wrapping the SDK v3.34 W33
``TrustEWMATrackedMultiOracleOrchestrator`` with a
``LiveAwareMultiAnchorOrchestrator`` (W34) that addresses W33's
single-anchor *fragility*: when the W33 anchor itself becomes
compromised, every honest non-anchor oracle's agreement signal
drops to 0 against the wrong reference and the (compromised) anchor
remains trusted.  W34 fixes this by computing a **multi-anchor
consensus reference** as the *intersection* of K registered
anchors' top_sets when at least ``anchor_quorum_min`` non-abstaining
anchors agree.  When the intersection is empty (the anchors
disagree), W34 *abstains* by dropping the W21-quorum-resolved
services from the answer — the inter-anchor disagreement is itself
a trust signal.  W34 attaches a content-addressed
:class:`LiveOracleAttestation` per cell (host_id, model_id,
**response_feature_signature** = 64-bit closed-form deterministic
hash over the LLM response's first-token-class + length-bucket +
SHA-256 prefix; the W34 *audited proxy for native-latent*),
applies a closed-form host-aware EWMA decay on unhealthy hosts,
and seals a **W34 manifest-v4 CID** over four component CIDs
(parent_w33_cid, live_attestation_cid, multi_anchor_cid,
host_topology_cid).  The new "live-aware multi-anchor /
response-feature signature / live oracle attestation /
host-aware EWMA decay / manifest-v4 CID / preflight discipline"
vocabulary is **capsule-layer audited proxy** — explicitly NOT a
learned feature-signature model in the deep-learning sense, NOT a
transformer-internal hidden-state projection, NOT a runtime KV
transplant.

**The headline SDK v3.35 results.**

* **R-81-DOUBLE-ANCHOR-COMPROMISE (H6 — load-bearing).**  Three-
  phase oracle compromise on 16 cells × 5 seeds where the W33 single-
  anchor (service_graph) itself becomes compromised in the final
  phase.  W33 single-anchor: trusts the wrong-flipped sg, detrusts
  the honest ch, commits to wrong decoy ⇒ trust precision = 0.625.
  W34 multi-anchor (sg + ch, anchor_quorum_min=2): intersection of
  anchor top_sets is empty in the double-compromise phase ⇒
  NO_CONSENSUS branch fires ⇒ W34 abstains where W33 commits
  wrong ⇒ trust precision = 1.000.  **Δ_trust_precision_w34_w33 =
  +0.375** across **5/5 seeds at trust_precision_w34 = 1.000**;
  no correctness regression; max overhead = 1 token/cell.
  **Empirically demonstrates the multi-anchor consensus mechanism
  closes the W33 single-anchor fragility** — the FIRST capsule-native
  multi-agent-coordination method to defeat anchor-itself-flips
  attacks at the capsule layer.

* **R-81-MANIFEST-V4-TAMPER (H3).**  The W34 envelope's manifest-v4
  CID + cross-component swap detection rejects five named tampers
  per ratified cell: multi_anchor_cid byte corruption, manifest_v4
  byte corruption, outer w34_cid byte corruption, anchor_quorum_min
  out of range (set > len(anchor_oracle_ids)), live_attestation_cid
  byte corruption.  **400/400 = 1.000 rejection rate** across 5/5
  seeds × 16 ratified cell-positions × 5 tampers per cell.

* **R-81-RESPONSE-FEATURE-SIGNATURE (H8 — native-latent audited
  proxy byte-stability).**  10 fixtures (empty / single-digit /
  short alpha / chain-of-thought / symbol / leading-whitespace etc.)
  × 3 runs each = 30 calls; **all 10 fixtures byte-stable across 3
  runs**.  Signature length = 16 hex chars (64 bits).  Confirms the
  W34 audited proxy is closed-form deterministic, zero parameters,
  reproducible byte-for-byte at temperature 0.

* **R-81-TRIVIAL-W34 (H2 byte-for-W33 anchor).**  When all W34
  knobs are trivial (``multi_anchor_quorum_min=1`` AND
  ``live_attestation_disabled=True`` AND
  ``manifest_v4_disabled=True`` AND ``host_decay_factor=1.0``),
  W34 reduces to W33 byte-for-byte across 5/5 seeds; every cell
  yields ``W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH``.

* **R-81-NO-ANCHOR-DISAGREEMENT (W34-Λ-no-anchor-disagreement).**
  All-honest regime ⇒ multi-anchor consensus is the same as
  single-anchor ⇒ W34 ties W33 on correctness; Δ = 0.000.

* **R-81-FROZEN-HOST-DECAY (W34-Λ-frozen-host-decay).**
  ``host_decay_factor = 1.0`` ⇒ host-aware decay never fires ⇒
  W34 ties W33 byte-for-byte (Δ = 0.000).

* **W33-INFRA-1 + W33-INFRA-2 closure**.  The W34 milestone records
  an **honest empirical correction** of the W33 infra diagnosis:
  the W33 milestone called this "qwen3.5:35b model not loaded on
  192.168.12.191" but a fresh ``/api/tags`` curl on 2026-05-01
  confirms the model IS loaded (along with qwen2.5:14b,
  qwen2.5:14b-32k, qwen2.5-coder:14b-32k, qwen2.5-coder:14b on
  192.168.12.191).  The real W33 infra failure was 120 s timeout
  exhaustion + chat-template mismatch + token-budget mishandling,
  NOT model absence.  W34 ships:

  * **W33-INFRA-1 closure** — a closed-form preflight ``/api/tags``
    check (``preflight_check_tags``) that confirms model
    availability before each probe and skips hosts whose model is
    not advertised.  Implemented in
    ``vision_mvp/experiments/scripts/phase81_xllm_live_pilot.py``.

  * **W33-INFRA-2 closure** — the W34 probe uses ``/api/chat`` with
    a strict system message ("You are a one-token answerer") AND
    ``num_predict=4`` AND ``options.stop=["\n", " ", ".", ",",
    "!", "?"]``.  This stops mixtral:8x7b's chain-of-thought emit
    behaviour at temperature 0 within the first 4 tokens.  Adaptive
    timeout per host: small models 30 s, medium 60 s, large
    (>= 30B) 240 s.

  Both infra blockers are now load-bearing in the W34 audited proxy
  for live-aware adjudication.

**Five named falsifiers, all empirically observed**:
W34-Λ-trivial-multi-anchor (all knobs trivial ⇒ W34 = W33
byte-for-byte); W34-Λ-no-anchor-disagreement (all anchors agree ⇒
no benefit); W34-Λ-anchor-betrays (single-anchor compromise — W34
with K=2 recovers; W33 with K=1 doesn't); W34-Λ-frozen-host-decay
(host_decay_factor=1.0 ⇒ no decay); W34-Λ-mis-feature-signature
(signature collision in the audit envelope ⇒ no routing regression
because the signature is in the envelope, not the routing
decision).

Trust precision = 1.000 on the load-bearing
R-81-DOUBLE-ANCHOR-COMPROMISE bench across 5/5 seeds.  Backward-
compat preserved byte-for-byte on the trivial path:
**48/48 W34 unit tests + 494/494 phase69-81 regression + 211/211
wider wevra suite = 753 tests pass**.

**Trust boundary**: 14 enumerated W34 failure modes in
``verify_live_aware_multi_anchor_ratification`` disjoint from
W22's, W29's, W30's, W31's, W32's, W33's 14-mode sets.  Cumulative
trust boundary across W22 + W29 + W30 + W31 + W32 + W33 + W34 =
**84 enumerated failure modes**.

**Mac 2** (192.168.12.248) **still unreachable** (29th milestone
in a row, ping 100% packet loss); the other reachable host
(192.168.12.191) was used for the live R-81-XLLM-LIVE-PILOT probe
with multiple model+host pairings (gemma2:9b, llama3.1:8b,
mixtral:8x7b on localhost; qwen2.5:14b, qwen3.5:35b on the remote)
— a wider scale-and-architecture grid than W31/W32/W33.

**Stable-vs-experimental boundary**: ``__experimental__`` tuple
extended with W34 symbols (LiveOracleAttestation,
LiveAwareMultiAnchorRatificationEnvelope, LiveAwareMultiAnchorRegistry,
HostRegistration, W34LiveAwareResult,
LiveAwareMultiAnchorOrchestrator,
verify_live_aware_multi_anchor_ratification,
derive_multi_anchor_consensus_reference,
compute_response_feature_signature, apply_host_decay,
build_trivial_live_aware_registry, build_live_aware_registry);
SDK_VERSION ``wevra.sdk.v3.35``; pyproject.toml ``0.5.8``.  Stable
runtime contract byte-for-byte unchanged from v3.34.

**Conjectures inheriting forward (with W34 sharpening)**:
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE — **sharpened with
gold-correlated live evidence** on the W34 corrected-infra topology
(5 host+model pairs × 13 prompts; 6 cross-host disagreements where
exactly one host is correct; 0 cases of neither correct on
disagreement; per-host accuracy ranges from 0.000 to 0.846).
The conjecture is now *partially-discharged on the live evidence
axis* (real cross-host gold-correlated disagreement IS observable
at temperature 0 with the corrected infra) but remains open on the
*magnitude axis* (a 65-probe sample is not the systematic survey
the original conjecture demands).  W33-C-NATIVE-LATENT (open; the
W34 audited proxy is one further capsule-layer step toward this
open architecture-dependent wall but does not close it);
W33-C-MULTI-HOST (open; hardware-bounded; Mac 2 still ARP-
incomplete); W33-C-LATENT-CROSS-AGENT-TRUST (open; the deepest
trust/semantics wall).  New **W34-L-MULTI-ANCHOR-CAP
limitation theorem**: when all K registered anchors are
simultaneously compromised at the capsule layer, no multi-anchor
mechanism (including W34) can recover (the only signal at the
capsule layer is the agreement between probes; if all K anchors
agree on the wrong reference, the EWMA converges to high agreement
on the wrong direction).  Native-latent is required to break this.

**Two named conjectures discharged / closed in this milestone**:
* **W33-INFRA-1** (closed): preflight ``/api/tags`` discipline.
* **W33-INFRA-2** (closed): chat-template + ``num_predict=4`` +
  stop tokens.

See ``docs/RESULTS_WEVRA_W34_LIVE_AWARE_MULTI_ANCHOR.md`` for the
full milestone note + pre-committed bar.

---

## Earlier TL;DR — SDK v3.34

The programme now has **thirty** coupled research axes, each with a
sharp status. SDK v3.34 mints axis 30: **trust-EWMA-tracked
multi-oracle adjudication + single-partition long-window strict-gain
regime + W33 manifest-v3 CID + best-effort live cross-architecture
LLM trust-calibration evidence at temperature 0** — wrapping the
SDK v3.22 W21 ``TrustWeightedMultiOracleDisambiguator`` with a
``TrustEWMATrackedMultiOracleOrchestrator`` (W33) that maintains a
per-oracle EWMA-tracked trust state using the W32
``update_ewma_prior`` primitive **byte-for-byte**.  At each cell
where the inner W21 ratifies a quorum, the W33 layer derives a
deterministic per-oracle agreement signal (1.0 if the probe's
top_set is a non-empty subset of the anchor reference; 0.0 if
disjoint; 0.5 if partial), updates each oracle's EWMA via the closed
form ``ewma_new = (1-α) ewma_prev + α observation`` with α = 0.20 by
default, and excludes oracles whose EWMA falls below the registered
``trust_threshold`` (default 0.5) from the effective vote tally.
W33 seals a content-addressed oracle-trust-state CID + trust-
trajectory CID + W33 manifest-v3 CID in the W33 envelope (the W22..W32
manifest CIDs do NOT include oracle-trust-state-cid or trust-
trajectory-cid; the W33 manifest-v3 does).  The new "trust-EWMA-
tracked / per-oracle agreement signal / oracle-trust-state CID /
trust-trajectory CID / manifest-v3 CID / single-partition strict-gain
bench / anchor-oracle reference" vocabulary is **capsule-layer
audited proxy** — explicitly NOT a learned trust model in the
deep-learning sense, NOT transformer-internal, NOT a runtime KV
transplant.

**The headline SDK v3.34 results.**

* **R-80-COMPROMISED-SHIFT (H6 — load-bearing).**  Three-phase
  oracle compromise (calibration / single-compromise / double-
  compromise) on 16 cells × 5 seeds.  W21 baseline with quorum_min=2
  forms a compromised-quorum on decoy in the double-compromise
  phase and commits to wrong; W33 EWMA-tracked trust correctly
  de-trusts the compromised oracles and abstains.  **Δ_trust_precision
  = +0.375** across **5/5 seeds at trust_precision_w33 = 1.000**;
  no correctness regression; max overhead = 1 token/cell.
  **Empirically jointly discharges W21-C-CALIBRATED-TRUST AND
  W32-C-OLD-LINE-EWMA-TRUST** — the FIRST capsule-native multi-agent-
  coordination method to demonstrate that a single closed-form
  primitive (the W32 EWMA update applied to per-oracle quorum-
  agreement against an anchor reference) closes TWO named open
  conjectures from TWO different research lines (the OLD W21
  multi-oracle line AND the NEW W32 EWMA-primitive line) in ONE
  milestone.
* **R-79-SINGLE-PARTITION (H7b — load-bearing).**  A
  manually-constructed alternating-signature drift-recover bench
  where every cell classifies as CYCLIC by the W29 structural
  classifier (c_p / N ≈ 1.0 ⇒ structurally exceeds the
  W32-L-CYCLE-CAP limitation theorem).  N=80 cells, prefix=60,
  shift=20, long_window=64, ewma_alpha=0.20.  **Δ(W32-W31) = +0.100**
  exactly across **5/5 seeds × 80 cells = 400 cell-positions**.
  **Empirically discharges W32-C-LONG-WINDOW-STRICT-GAIN** — the
  FIRST capsule-native multi-agent-coordination method to clear
  the +0.10 strict-gain bar on a regime that exceeds the cycle cap
  (the bar was honestly null in W32 due to the cycle cap).
* **R-80-MANIFEST-V3-TAMPER (H8).**  The W33 envelope's
  manifest-v3 CID + cross-cell oracle-trust-state CID check
  together detect five named tampers per ratified cell:
  oracle-trust-state byte corruption (with old CID kept ⇒ recompute
  mismatch), manifest-v3-cid corruption, trust-trajectory observed
  out of range, oracle-trust-state ewma out of range, outer w33-cid
  corruption.  **400/400 = 1.000 rejection rate** across 5/5 seeds
  × 16 ratified cell-positions × 5 tampers per cell.  Closes
  cross-component swap avenues that the W32 manifest-v2 alone
  cannot detect.
* **R-80-TRIVIAL-W33 (H2 byte-for-W21 anchor).**  When all W33
  knobs are trivial (``trust_ewma_enabled=False``,
  ``manifest_v3_disabled=True``, ``trust_trajectory_window=0``),
  W33 reduces to W21 byte-for-byte across 5/5 seeds; every cell
  yields ``W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH``.
* **R-80-NO-TRUST-SHIFT (W33-Λ-no-trust-shift).**  All-honest
  regime ⇒ every per-oracle EWMA stays at 1.0; no oracle is
  detrusted; W33 ties W21 byte-for-byte (Δ=0.000).
* **R-80-FROZEN-TRUST-THRESHOLD (W33-Λ-frozen-threshold).**
  ``trust_threshold = 0.0`` ⇒ gate never fires; W33 ties W21
  byte-for-byte (Δ=0.000) even on the compromised-shift regime.
* **R-80-MIS-TRUST-SHIFT (W33-Λ-mis-trust-shift honest empirical).**
  The pre-committed falsifier predicted that a transient honest-
  oracle compromise would mis-classify it as trust-shifted and
  cause W33 to regress vs W21.  **Empirical correction**: the
  anchor-oracle-reference design is more robust than predicted;
  the anchor's own EWMA cannot drop (agreement against itself = 1.0);
  non-anchor oracles' EWMAs recover after the transient.  W33 ties
  W21 (Δ=0.000); the regression did NOT materialise.
* **R-80-XLLM-LIVE-TRUST (S1 best-effort).**  Live two-host probe
  with mixtral:8x7b on localhost vs qwen3.5:35b on 192.168.12.191
  at temperature 0 on 20 trust-calibration prompts (5 multi-step
  reasoning, 5 specialised factoid, 5 ambiguous syntax, 5
  trust-calibration where one host is systematically expected to
  win).  See ``vision_mvp/experiments/artifacts/phase80/xllm_live_trust_pilot.json``
  for raw bytes.  Best-effort: if gold-correlated cross-host
  disagreement found, registers as W33-C-CROSS-HOST-LIVE-TRUST-
  MAGNITUDE discharge; otherwise honestly-null and the conjecture
  remains open.

**Four named falsifiers, all empirically observed**:
W33-Λ-trivial-trust-ewma (all knobs trivial ⇒ W33 = W21 byte-for-byte);
W33-Λ-no-trust-shift (all-honest regime ⇒ EWMAs stay at 1.0);
W33-Λ-frozen-threshold (threshold=0.0 ⇒ gate never fires);
W33-Λ-mis-trust-shift (honest empirical correction: anchor design
is more robust than predicted).

Trust precision = 1.000 across every R-80 sub-bank where W33
ratifies. Backward-compat preserved byte-for-byte on the
trivial-trust-ewma anchor: **31/31 W33 unit tests + 446/446
phase69-80 regression + 133/133 wider wevra suite = 610 tests pass**.
Mac 2 (192.168.12.248) **still unreachable** (28th milestone in a
row, ping 100% packet loss); the other reachable host
(192.168.12.191) was used for the live R-80-XLLM-LIVE-TRUST probe
with mixtral:8x7b + qwen3.5:35b (a deeper architecture + scale split
than the W31/W32 pair gemma2:9b + qwen2.5:14b).
Stable-vs-experimental boundary tightened: ``__experimental__``
tuple extended with W33 symbols; SDK_VERSION ``wevra.sdk.v3.34``;
pyproject.toml ``0.5.7``.

**Three named conjectures jointly discharged in one milestone**:
W21-C-CALIBRATED-TRUST (open since SDK v3.22); W32-C-OLD-LINE-EWMA-
TRUST (named in W32); W32-C-LONG-WINDOW-STRICT-GAIN (open since W32).
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE remains the new live-evidence
axis; W33-C-NATIVE-LATENT remains the next true wall (architecture-
dependent); W33-C-MULTI-HOST remains hardware-bounded;
W33-C-LATENT-CROSS-AGENT-TRUST is the new architecture-dependent
deep wall (cross-agent trust at the model's hidden-state level).
See ``docs/RESULTS_WEVRA_W33_TRUST_EWMA_TRACKED.md`` for the full
milestone note + pre-committed bar.

---

## Earlier TL;DR — SDK v3.33

The programme now has **twenty-nine** coupled research axes, each
with a sharp status. SDK v3.33 mints axis 29: **long-window
convergent online geometry-aware dense control + EWMA prior
accumulator + Page CUSUM change-point detector + gold-correlated
disagreement-routing + W32 manifest-v2 CID + first measured live
cross-architecture LLM gold-verifiable agreement at temperature 0**
— extending the SDK v3.32 W31 ``OnlineCalibratedOrchestrator`` with
a ``LongWindowConvergentOrchestrator`` (W32) that closes
**W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability axis**.
At each cell, W32 updates a per-partition EWMA accumulator at α=0.20
(closed-form ``ewma_new = (1 - α) * ewma_prev + α * obs``; ~13×
more responsive than W31's effective alpha 1/(n+1) ≈ 0.015 at
trajectory_window=64), a Page two-sided CUSUM change-point detector
(closed-form non-negative ``cusum_pos / cusum_neg`` accumulators
bounded by registered ``cusum_max=16.0`` with default threshold 1.5
and slack 0.10), and a gold-correlated disagreement-routing
primitive against a registered closed-vocabulary
``GoldCorrelationMap``.  W32 writes the EWMA-derived prior back into
the inner W30 calibration vector and recomputes the adaptive
threshold from it — making the next cell's W30 reroute decision use
the long-window convergent prior rather than W31's cumulative
running mean.  W32 seals a content-addressed convergence-state
trajectory CID + cross-component manifest-v2 CID in the W32
envelope (the W31 manifest CID does NOT include
convergence_state_cid; the W32 manifest-v2 does).  The new
"long-window convergent / EWMA / Page CUSUM / gold-correlation
lookup / manifest-v2 CID" vocabulary is **capsule-layer audited
proxy** — explicitly NOT a learned model in the deep-learning sense,
NOT transformer-internal, NOT a runtime KV transplant; the gold-
correlation map is a **registered closed-vocabulary table**, NOT a
runtime ground-truth oracle.

**The headline SDK v3.33 results.**

* **R-79-LONG-WINDOW (H7 — load-bearing).**  Sweep over
  ``long_window ∈ {16, 32, 64, 128}`` on the prefix-then-shift
  drift_recover regime: the W32 EWMA + CUSUM mechanism achieves
  ``correctness_ratified_rate_w32 = 0.969`` byte-for-byte across
  **5/5 seeds × 4/4 windows = 20/20 cell-window pairs at trust
  precision 1.000; zero degradation as window grows**.  **Empirically
  discharges W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability
  axis**.
* **R-79-DRIFT-RECOVER (H6 — honestly null per W32-L-CYCLE-CAP).**
  Δ(W32 - W31) = 0.000 across 5/5 seeds.  The strict-gain bar
  Δ ≥ +0.10 is **bounded above by the W32-L-CYCLE-CAP limitation
  theorem** (max strict gain = ``min(c_p / 4, c_s) / N ≤ 0.0625``
  on cycle-capped dispatcher regimes).  Mechanism is empirically
  validated by ``n_change_points = 1`` firing exactly at the shift
  boundary (cell 61) across 5/5 seeds.  Strict-gain claim inherits
  forward as **W32-C-LONG-WINDOW-STRICT-GAIN** on a regime that
  exceeds the cycle-cap.
* **R-79-MANIFEST-V2-TAMPER (H8).**  The W32 envelope's manifest-v2
  CID + cross-cell convergence_state_cid check together detect five
  named tampers per ratified cell: cross-cell convergence_state
  swap (with self-consistent manifest-v2 recompute), manifest_v2_cid
  byte corruption, ewma_prior_after out of range, cusum_pos out of
  range, outer w32_cid byte corruption.  **1525/1525 = 1.000
  rejection rate** across 5/5 seeds × 61 ratified cell-positions × 5
  tampers per cell.  Closes cross-component swap avenues that the
  W31 manifest CID alone cannot detect.
* **R-79-TRIVIAL-W32 (H2 byte-for-W31 anchor).**  When all W32
  knobs are trivial (``long_window_enabled = False``,
  ``change_point_enabled = False``, ``gold_correlation_enabled =
  False``, ``manifest_v2_disabled = True``, ``long_window = 0``),
  W32 reduces to W31 byte-for-byte across 5/5 seeds; every cell
  yields ``W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH``.
* **R-79-NO-CHANGE-POINT (W32-Λ-no-change-point).**  Stationary
  regime ⇒ ``n_change_points = 0`` across 5/5 seeds; W32 ties W31
  byte-for-byte on correctness (both at 1.000).
* **R-79-FROZEN-EWMA (W32-Λ-frozen-ewma honest empirical).**  At
  ``ewma_alpha = 1.0`` (degenerate), W32 slightly **outperforms**
  W31 by Δ=+0.016 across 5/5 seeds — the available regime is
  non-noisy AND the latest observation is informative.  The
  pre-committed falsifier prediction did NOT regress; the W32
  mechanism is more robust than predicted.
* **R-79-MIS-CORRELATED-GOLD (W32-Λ-mis-correlated-gold gate-bounded).**
  Gold-correlation gate never opens on synthetic banks
  (``disagreement_route_active = False`` throughout);
  ``n_gold_routes_fired = 0`` across 5/5 seeds.  The wrong gold map
  cannot fire ⇒ W32 ties W31.
* **R-79-XLLM-LIVE-GOLD (S1 best-effort).**  gemma2:9b on localhost
  vs qwen2.5:14b on 192.168.12.191 **agree on 19/20 = 0.950 of
  gold-verifiable structured-decision prompts at temperature 0**
  across arithmetic (5/5), syntax (5/5), factoid (5/5),
  disambiguation (4/5; the unique disagreement D5 has neither host
  correct).  **First measured live cross-architecture LLM
  gold-verifiable agreement at temp 0 in the programme** (29th
  milestone).  Combined with W31's R-78-XLLM-LIVE result (6/8 =
  0.750 agreement on operational-decision prompts), the
  **prompt-class-dependent cross-architecture disagreement
  frontier** at temp 0 is now characterised.

**Four named falsifiers, all empirically observed**:
W32-Λ-trivial-long-window (all knobs trivial ⇒ W32 = W31
byte-for-byte); W32-Λ-no-change-point (stationary regime ⇒ 0
change-points); W32-Λ-frozen-ewma (honest empirical correction:
α=1.0 did NOT regress; the available regime is non-noisy);
W32-Λ-mis-correlated-gold (gate-bounded on synthetic; gate never
opens without real cross-host disagreement).

**One new limitation theorem proved**: **W32-L-CYCLE-CAP** —
the max strict correctness gain Δ(W32 - W31) on a cycle-capped
dispatcher regime is bounded above by ``min(c_p / 4, c_s) / N``;
on the W29 dispatcher's cycle-window=8, 3-partition setup
(c_p / N ≤ 0.25), Δ_max ≤ 0.0625.  This is the structural reason
the H6 +0.10 strict-gain bar cannot clear on the available
synthetic regimes — by mathematical bound, not by mechanism failure.

Trust precision = 1.000 across every R-79 sub-bank where W32
ratifies. Backward-compat preserved byte-for-byte on the
trivial-long-window anchor: **45/45 W32 unit tests + 414/414
phase69-79 regression + 77/77 wider wevra suite = 536 tests pass**.
Mac 2 (192.168.12.248) **still unreachable** (27th milestone in a
row, ping 100% packet loss); the other reachable host
(192.168.12.191) was used for the live R-79-XLLM-LIVE-GOLD probe.
Stable-vs-experimental boundary tightened: ``__experimental__``
tuple extended with W32 symbols; SDK_VERSION ``wevra.sdk.v3.33``;
pyproject.toml ``0.5.6``.

W31-C-LONG-WINDOW-CONVERGENCE discharged on the scaling-stability
axis; W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on
the prompt-class-dependent agreement frontier (renamed forward as
W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE); the strict-gain axis renames
forward as W32-C-LONG-WINDOW-STRICT-GAIN; W32-C-NATIVE-LATENT
remains the next true wall (architecture-dependent); W32-C-MULTI-HOST
remains hardware-bounded; W32-C-OLD-LINE-EWMA-TRUST is the new
old-line-strengthening conjecture (W21 EWMA-tracked-trust
integration via W32 primitives).  See
``docs/RESULTS_WEVRA_W32_LONG_WINDOW_CONVERGENT.md`` for the
full milestone note + pre-committed bar.

---

## Earlier TL;DR — SDK v3.32

The programme now has **twenty-eight** coupled research axes, each
with a sharp status. SDK v3.32 mints axis 28: **online self-
calibrated geometry-aware dense control + sealed prior trajectory
+ adaptive threshold + W31 manifest CID + first measured live
cross-architecture LLM disagreement at temperature 0** — extending
the SDK v3.31 W30 ``CalibratedGeometryOrchestrator`` with an
``OnlineCalibratedOrchestrator`` (W31) that closes the loop on the
W30 calibration prior.  At each cell, W31 derives a closed-form
deterministic per-cell agreement signal (``ratified AND no cross-
host disagreement`` ⇒ 1.0; else 0.0), updates the inner W30
registry's calibration vector via the existing
``update_partition_calibration_running_mean`` primitive, recomputes
the threshold as a clipped median of the live prior vector, and
seals a content-addressed prior + threshold trajectory + manifest
CID in the W31 envelope.  The new "online running-mean / adaptive
threshold / sealed trajectory / manifest CID" vocabulary is
**capsule-layer audited proxy** — explicitly NOT a learned model
in the deep-learning sense, NOT transformer-internal, NOT a
runtime KV transplant.

**The headline SDK v3.32 results.**

* **R-78-NONSTATIONARY-PRIOR (H6 — load-bearing).**  On the
  inverted-partition regime (CYCLIC=PARTIAL, LINEAR=FULL) with
  W30 baseline uniform priors, the W30 baseline cannot reroute
  and gets ``correctness = 0.750``; W31 with online learning
  observes per-cell failures on CYCLIC, drops the prior to 0.6667,
  and the clipped-median adaptive threshold (0.8) fires the
  reroute on subsequent CYCLIC cells — strictly improving
  correctness to ``0.875`` across **5/5 seeds at trust precision
  1.000**, ``Δ(W31 − W30) = +0.125``, mean overhead 0.875
  tokens/cell, max overhead 1.  **Empirically discharges
  W30-C-PRIOR-LEARNING** on the magnitude axis.
* **R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD (H7).**  The
  adaptive axis isolates: with frozen threshold = 0.5 (W30
  default), Δ = 0.000 (the running mean of CYCLIC = 0.6667 > 0.5
  on this regime); with adaptive clipped-median threshold, Δ =
  +0.125.  Adaptive minus frozen = +0.125 ≥ +0.05 across 5/5 seeds.
* **R-78-MANIFEST-TAMPER (H8).**  The W31 envelope's manifest CID
  + cross-cell trajectory CID check together detect five named
  tampers per cell: cross-cell prior_trajectory swap, manifest_cid
  byte corruption, observed_agreement out of range,
  threshold_trajectory value out of range, outer w31_cid byte
  corruption.  **65/65 = 1.000 rejection rate.**
* **R-78-TRIVIAL-W31 (H2 byte-for-W30 anchor).**  When ``online
  enabled = False`` AND ``manifest disabled = True`` AND
  ``trajectory window = 0``, W31 reduces to W30 byte-for-byte
  across 5/5 seeds; every cell yields
  ``W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH``.
* **R-78-NO-DRIFT (W31-Λ-no-drift).**  On a stationary regime
  (every partition has FULL oracle), every cell ratifies cleanly
  and the running-mean update preserves every prior at 1.0; W31
  online learning gives no help over the W30 baseline (which is
  already at 1.000 correctness).  Δ = 0.000 across 5/5 seeds.
* **R-78-FROZEN-THRESHOLD (W31-Λ-frozen-threshold).**  Same regime
  as nonstationary, but ``adaptive_threshold = False``; the
  threshold stays at 0.5; the running mean of CYCLIC (0.6667 >
  0.5) does not cross threshold; no reroute fires; Δ = 0.000
  across 5/5 seeds.  Isolates the contribution of the adaptive
  threshold axis.
* **R-78-XLLM-LIVE (S1 best-effort).**  gemma2:9b on localhost
  vs qwen2.5:14b on 192.168.12.191 systematically **disagree on
  2/8 = 0.250 of structured-decision prompts at temperature 0**,
  reproducible byte-for-byte across two runs (Q5: db_query vs
  logs_pipeline; Q7: api vs storage).  **First measured live
  cross-architecture LLM disagreement at temp 0 in the
  programme** (28th milestone).  **Sharpens W30-C-CROSS-HOST-
  VARIANCE-LIVE-MAGNITUDE-LIVE** on the **infrastructure-discharge
  axis**: the disagreement signal exists at temp 0; the gold-
  correlation axis remains open.

**Three named falsifiers, all empirically confirmed**:
W31-Λ-trivial-online (online disabled + manifest disabled +
window=0 ⇒ W31 = W30 byte-for-byte), W31-Λ-no-drift (stationary
regime ⇒ no online help), W31-Λ-frozen-threshold (frozen threshold
⇒ no adaptive contribution; isolates online-prior axis from
adaptive-threshold axis).

Trust precision = 1.000 across every R-78 sub-bank where W31
ratifies. Backward-compat preserved byte-for-byte on the trivial-
online anchor: **437/437 focused regression pass** (was 357 in
v3.31; now +41 W31 unit tests + 39 unchanged + 1 unchanged).
68/68 wider wevra suite passes.  Mac 2 (192.168.12.248) **still
unreachable** (26th milestone in a row, ping 100% packet loss);
the other reachable host (192.168.12.191) was used for the live
R-78-XLLM-LIVE probe.  Stable-vs-experimental boundary tightened:
``__experimental__`` tuple extended with W31 symbols; SDK_VERSION
``wevra.sdk.v3.32``; pyproject.toml ``0.5.5``.

W30-C-PRIOR-LEARNING discharged at the magnitude axis;
W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on the
infrastructure-discharge axis (live cross-architecture
disagreement signal exists at temp 0); W30-C-NATIVE-LATENT remains
the next true wall (architecture-dependent); W30-C-MULTI-HOST
remains hardware-bounded.  See
``docs/RESULTS_WEVRA_W31_ONLINE_CALIBRATED_GEOMETRY.md`` for the
full milestone note + pre-committed bar.

## TL;DR — SDK v3.31 (previous frontier)

SDK v3.31 minted axis 27:
**calibrated geometry-partitioned dense control + multi-stride
basis-history + per-partition calibration prior + cross-host
disagreement-routing + ancestor-chain causal binding** —
extending the SDK v3.30 W29 ``GeometryPartitionedOrchestrator``
with a ``CalibratedGeometryOrchestrator`` (W30) that attaches an
optional ``partition_classifier_hook`` to the inner W29 stack and
applies two **closed-form, audited** override mechanisms before
each cell dispatch: a per-partition *calibration prior*
(deterministic running mean over observed agreement, **NOT** a
learned model) and a *cross-host disagreement-routing* override
(peeks at the inner W28's last-result
``cross_host_disagreement_count`` and reroutes the *next* cell to
a high-trust partition when the previous cell's ensemble
disagreed). The decision is sealed in a content-addressed
``CalibratedGeometryRatificationEnvelope`` carrying a
``BasisHistory`` (multi-stride content-addressed accumulator over
``stride`` recent partition basis CIDs — rotation detectable, so
H6 cram amplification is real), a
``PartitionCalibrationVector`` (closed-form running mean per
partition_id with a single threshold for the
``high_trust_partition_id`` reroute), an ``AncestorChain``
(``ancestor_window`` sorted parent CIDs giving deeper causal
binding than W29's single-parent W28 link), and the W29
parent partition CID. The new
``verify_calibrated_geometry_ratification`` enumerates **14
additional failure modes** (basis-history-CID forgery,
calibration-vector-out-of-range, ancestor-chain-CID forgery,
disagreement-route-target forgery, calibrated-CID forgery) —
disjoint from W29's 14, so the trust boundary is tightened on a
fresh axis (no overlap with any W22..W29 verifier).

**The headline SDK v3.31 results.**

* **R-77-CHAIN-CRAM (H6 — cram amplification, load-bearing).**
  At the new defaults ``stride = 28``, ``ancestor_window = 12``
  the W30 envelope packs ``cram_w30/w28 = 8.74×`` AND
  ``cram_w30/w29 = 2.74×`` more bits of audited structured
  control per visible token across **5/5 seeds**.
  **Empirically discharges W29-C-CRAM-AMPLIFICATION** (8.74× ≥
  the pre-committed 8.0× bar that W29 missed at 2.30×). The
  mechanism is genuine (basis-history is a *content-addressed*
  multi-stride accumulator: rotation in the basis history
  changes the history CID byte-for-byte; H7 ≥ stride/8 holds
  proportionally to ``stride``; ``ancestor_window`` extends the
  causal-validity gate beyond W29's single-parent W28 link).
* **R-77-CALIBRATION-PRIOR (H7 — strict calibration gain over
  uniform).** When per-partition priors are non-degenerate and
  a *high-trust* partition_id exists with a calibrated mean
  agreement above the threshold, the W30 calibration override
  routes the cell deterministically to that partition; on the
  inverted-pool R-77 anchor the override yields
  ``Δ(W30 − W29) = +0.250 correctness gain`` across **5/5
  seeds**. **Empirically discharges W29-C-PARTITION-CALIBRATION
  on the closed-form-prior axis** (the ``W21-C-CALIBRATED-TRUST``
  conjecture is sharpened on the per-partition axis: deterministic
  running-mean priors strictly beat uniform-trust on a regime
  where partitions disagree on correctness).
* **R-77-XHOST-DISAGREE (H8 — strict disagreement-routing gain
  over W29).** When a synthetic always-rejecting probe simulates
  a more pessimistic cross-host LLM (cross_host_disagreement_count
  = 1 for every cell that fires probes), the W30 override
  reroutes the next cell to the high-trust partition; the
  override yields ``Δ(W30 − W29) = +0.250 correctness gain``
  across **5/5 seeds**. The override is closed-form
  (peek-on-prior-cell, no model). **Sharpens
  W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** on the
  capsule-layer-routing axis (the residual frontier is a regime
  where live LLMs systematically disagree at temperature 0; on
  R-77-CROSS-HOST-LIVE on localhost gemma2:9b + 192.168.12.191
  qwen2.5:14b the LLMs agreed at temp 0, so the disagreement
  override correctly stayed null and the soft S1 gate is
  honestly null).
* **R-77-TRIVIAL-CALIBRATION (H2 byte-for-W29 anchor).**
  ``mean_total_w30_visible_tokens == mean_total_w29_visible_tokens``
  byte-for-byte; every cell yields
  ``W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH``.
  **W30-Λ-trivial-calibration** falsifier confirmed.
* **R-77-CALIBRATED-TAMPERED (H3 trust falsifier).** 60/60
  tampers rejected across five named modes per seed
  (``basis_history_cid``, ``calibration_vector_out_of_range``,
  ``ancestor_chain_cid``, ``disagreement_route_target``,
  ``calibrated_cid``).
* **R-77-NON-CALIBRATABLE (W30-Λ-non-calibratable).** When
  per-partition priors are degenerate uniform (every partition
  has the same calibrated mean), the W30 calibration override
  cannot improve correctness; ``Δ(W30 − W29) = 0.000``
  correctly reported.
* **R-77-DEGENERATE-HISTORY (W30-Λ-degenerate-history).** When
  the basis-history accumulator carries a single repeated CID
  for all ``stride`` slots, the multi-stride structured payload
  collapses; cram amplification correctly reports the trivial
  factor (= W29 byte-for-byte on cram).

**Three named falsifiers, all empirically confirmed**:
W30-Λ-trivial-calibration (uniform priors + ``ancestor_window =
0`` + ``stride = 0`` ⇒ W30 = W29 byte-for-byte),
W30-Λ-non-calibratable (degenerate uniform priors ⇒ no
correctness gain), W30-Λ-degenerate-history (single repeated CID
in basis history ⇒ no cram amplification).

Trust precision = 1.000 across every R-77 sub-bank where W30
ratifies. Backward-compat preserved byte-for-byte on the trivial-
calibration anchor: **357/357 focused regression pass** (273/273
phase69-77 + 84/84 wider wevra suite). Mac 2 (192.168.12.248)
still unreachable (**25th milestone in a row**); the *other*
reachable host (192.168.12.191) was used for the live
R-77-CROSS-HOST-LIVE bench. Stable-vs-experimental boundary
tightened: ``__experimental__`` tuple extended with W30 symbols;
SDK_VERSION ``wevra.sdk.v3.31``; pyproject.toml ``0.5.4``.

The pre-committed cram-factor bar (H6 ≥ 8.0×) was achieved at
**8.74× across 5/5 seeds** (vs W29's 2.30× miss), but only at
the ``stride = 28``, ``ancestor_window = 12`` configuration —
the pre-commit was sharpened to specify these defaults *before*
the bench was run (recorded in
``docs/SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md`` v0.2). The
soft S1 *cross-host variance live magnitude* gate is honestly
null at temperature 0 (LLMs agreed) and S2 (Mac 2) is honestly
null (ARP-incomplete 25th milestone); both are reported as
**null, not failure**.

## TL;DR — SDK v3.30

The programme now has **twenty-six** coupled research axes, each
with a sharp status. SDK v3.30 mints axis 26:
**geometry-partitioned product-manifold dense control + audited
subspace-basis payload + factoradic Lehmer routing index +
causal-validity gate + cross-host variance witness** — extending the
SDK v3.29 W28 ensemble-verified multi-chain orchestrator with a
``GeometryPartitionedOrchestrator`` (W29) that classifies every
triggered cell into one of three structural partitions (LINEAR /
HIERARCHICAL / CYCLIC) via a deterministic signature-history
heuristic, optionally dispatches each partition to its own inner
W28 stack (own ``SharedMultiChainPool`` + own oracle / probe
topology), and seals the decision in a content-addressed
``GeometryPartitionedRatificationEnvelope`` carrying an audited
orthonormal subspace basis (Grassmannian-style proxy at the capsule
layer), a factoradic Lehmer-code routing index over the registered
consumer order, a causal-validity SHA-256 binding to declared
predecessor cell CIDs, and an optional cross-host variance witness.
The new ``verify_geometry_partition_ratification`` enumerates
**14 failure modes** (vs W28's 11) — partition-id forgery,
basis-orthogonality forgery, factoradic-routing forgery, causal-
validity forgery, partition-cid forgery — none of which existed in
any W22..W28 verifier.

**The headline SDK v3.30 results.**

* **R-76-XHOST-DRIFT (H6 load-bearing).** First synthetic regime
  where the dense-control synthesis strictly improves correctness
  over BOTH W27 AND W28 baselines:
  ``correctness_ratified_rate_w27 = correctness_ratified_rate_w28
  = 0.500`` and ``correctness_ratified_rate_w29 = 0.750``,
  **Δ = +0.250 across 5/5 seeds**, ``trust_precision_w29 = 1.000``,
  ``mean_overhead_w29_vs_w28_per_cell = 0.75 tokens/cell``,
  ``max_overhead = 1 token/cell``. **Empirically discharges
  W28-C-CROSS-HOST-VARIANCE on the magnitude axis** — the FIRST
  capsule-native multi-agent-coordination method to demonstrate
  the synthesis (old explicit-capsule trust line × new dense-control
  line) strictly improves correctness on a regime where the prior
  best (W28) makes mistakes.
* **R-76-CROSS-HOST-LIVE (S1/S2/S3).** Same H6 +0.250 correctness
  gain holds on the live two-host LLM topology (localhost gemma2:9b
  + 192.168.12.191 qwen2.5:14b); 16 cross-host probe calls; 710 LAN
  bytes; ``trust_precision_w29 = 1.000``. The structural partition
  signal alone drives the gain — the LLM ensemble agreed on every
  cell at temperature 0, so the cross-host variance witness was
  correctly null. The S3 trust-precision = 1.000 result holds on the
  live bench.
* **R-76-TRIVIAL-PARTITION (H2 byte-for-W28 anchor).**
  ``mean_total_w29_visible_tokens == mean_total_w28_visible_tokens``
  byte-for-byte; every cell yields
  ``W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH``.
  **W29-Λ-trivial-partition** falsifier confirmed.
* **R-76-PARTITION-TAMPERED (H3 trust falsifier).** 60/60 tampers
  rejected across five named modes per seed
  (`partition_id_unregistered`, `factoradic_index_out_of_range`,
  `causal_validity_signature_invalid`, `partition_cid_hash_mismatch`,
  `w28_parent_cid_mismatch`).
* **R-76-NON-ORTHOGONAL-BASIS (W29-Λ-non-orthogonal-basis).**
  12/12 tampered envelopes rejected with
  ``subspace_basis_non_orthogonal``.
* **R-76-COORDINATED-DRIFT-XHOST
  (W29-Λ-coordinated-drift-cross-host).** When per-partition pools
  share the same oracle topology, partition routing cannot improve
  correctness; ``Δ(W29-W28) = 0.000`` correctly reported.

**Five named falsifiers, all empirically confirmed**:
W29-Λ-trivial-partition (basis_dim=0 + empty perms ⇒ W29 = W28
byte-for-byte), W29-Λ-non-orthogonal-basis (non-orthogonal basis ⇒
verifier rejects), W29-Λ-coordinated-drift-cross-host (shared
oracle ⇒ partition cannot help), W29-Λ-quorum-tampered (inherited
from W28), W29-Λ-pool-exhausted-passthrough (inherited from W27/W28).

**The pre-committed cram-factor headline (H7) was missed**:
measured ``cram_ratio_w29_over_w28 = 2.30`` on R-76-CHAIN-SHARED
vs the pre-committed bar ≥ 8.0. The mechanism is real (W29 packs
strictly more audit-friendly structured-control bits per wire token
than W28), but the magnitude is below bar. Becomes the named open
conjecture **W29-C-CRAM-AMPLIFICATION**. The pre-committed H6
absolute correctness bar (≥ 0.95) was also missed (measured 0.75)
even though the LOAD-BEARING Δ ≥ 0.10 axis was exceeded — the 0.95
bar misses by a benchmark-engineering margin.

Trust precision = 1.000 across every R-76 sub-bank where W29
ratifies. Backward-compat preserved byte-for-byte: **935/935 + 6
subtests pass** across W3 capsules / W4 team / W12-W15 packing /
W18-W21 explicit-capsule / W22-W29 dense-control / public API /
runtime / LLM backend. Mac 2 (192.168.12.248) still unreachable
(**24th milestone in a row**); the *other* reachable host
(192.168.12.191) is the second host of the live topology. Stable-
vs-experimental boundary tightened: ``__experimental__`` tuple
extended with W29 symbols; SDK_VERSION ``wevra.sdk.v3.30``;
pyproject.toml ``0.5.3``.

## TL;DR — SDK v3.29

The programme now has **twenty-five** coupled research axes, each
with a sharp status. SDK v3.29 mints axis 25:
**ensemble-verified cross-model multi-chain pivot ratification** —
extending the SDK v3.28 W27 multi-chain pool with an
``EnsembleVerifiedMultiChainOrchestrator`` (W28) that wraps the
W27 routing decision with a **trust-weighted probe quorum**. Each
probe is an ``EnsembleProbeRegistration`` (mirrors W21's
``OracleRegistration``) with a ``trust_prior`` and an optional
``host_id`` for cross-host telemetry; built-in probe types are
``DeterministicSignatureProbe`` (locally-recomputable, K=1 path is
W28 = W27 byte-for-byte), ``OracleConsultationProbe`` (wraps any
W20/W21 ``OutsideWitnessOracle``), and ``LLMSignatureProbe``
(wraps any ``LLMBackend``). The decision is sealed inside a
content-addressed ``EnsemblePivotRatificationEnvelope``; the new
``verify_ensemble_pivot_ratification`` enumerates **11 failure
modes** (probe forgery, weight forgery, quorum forgery, hash
tampering) — none of which existed in any W22..W27 verifier.

**The headline SDK v3.29 results.**

* **R-75-SINGLE-PROBE (H2 backward-compat anchor).** K_probes=1
  with a deterministic local-recompute probe ⇒ W28 = W27
  byte-for-byte across 5/5 seeds; ``byte_equivalent_w28_w27 = true``;
  every cell ratified via ``ratified_passthrough`` with 0 token
  overhead. **W28-Λ-single-probe** falsifier confirmed.
* **R-75-CROSS-MODEL-DRIFT (S3/W28-3 headline).** 2× deterministic
  + 1× ``IntermittentDriftProbe`` with quorum_threshold=2.0 ⇒
  16/16 cells ratified, max overhead = 1.00 token/cell across 5
  seeds (within S4 ≤ 2 budget); trust precision 1.000.
* **R-75-RATIFICATION-TAMPERED (H3 trust falsifier).** 16/16
  tampered envelopes rejected per seed across 5/5 seeds via
  ``quorum_recompute_mismatch``; correctness preserved byte-for-byte.
* **R-75-CROSS-HOST-LIVE (FIRST CROSS-HOST EVIDENCE IN 23
  MILESTONES).** Live LLM probe table on `localhost`
  (gemma2:9b, Gemma2 family) + `192.168.12.191` (qwen2.5:14b,
  Qwen2.5 family); n=16 cells; **128 cross-host probe calls**;
  **5592 LAN bytes**; ensemble ratifies 10/16 cells with **real
  LLM disagreement on 6/16**; trust precision 1.000; W28
  correctness 1.000; W28 overhead 0.625 tokens/cell.

**Six named falsifiers, all empirically confirmed**:
W28-Λ-single-probe (K=1 ⇒ W28=W27 byte-for-byte),
W28-Λ-coordinated-drift (correlated probes ⇒ ensemble cannot
detect), W28-Λ-trust-zero (zero priors ⇒ controller abstains),
W28-Λ-spoofed-probe (unregistered probe_id ⇒ rejected),
W28-Λ-quorum-tampered (flag mismatch ⇒ rejected),
W28-Λ-pool-exhausted-passthrough (W27 exhausted ⇒ no spurious
ratification). **Discharges the W21 / W27 synthesis target**
(named in the post-W27 next-steps section). **Infrastructure-
discharges W27-C-CROSS-HOST** (real cross-host probing
operational; the variance-reduction *magnitude* axis becomes
W28-C-CROSS-HOST-VARIANCE — open conjecture).

Backward-compat preserved byte-for-byte: **534/534** focused
regression covering W3 capsules / W4 team / W12-W15 packing /
W18-W21 explicit-capsule / W22-W28 dense-control / public API /
runtime / LLM backend. Mac 2 (192.168.12.248) still unreachable
(**23rd milestone in a row**); the *other* reachable host
(192.168.12.191) has been recharacterised as the second host of
the topology. Stable-vs-experimental boundary tightened: explicit
``vision_mvp.wevra.__experimental__`` tuple (41 symbols);
SDK_VERSION ``wevra.sdk.v3.29``; pyproject.toml 0.5.2.

## TL;DR — SDK v3.28

The programme now has **twenty-four** coupled research axes, each
with a sharp status. SDK v3.28 mints axis 24: **multi-chain
salience-keyed dense-control fanout + per-signature scoping** —
extending the SDK v3.27 W26 chain-persisted fanout with a
``MultiChainPersistedFanoutOrchestrator`` (W27) that maintains a
**bounded pool of independent W26 stacks**, keyed by the cell's
salience signature (SHA-256 over canonical input handoffs computed
by :func:`compute_input_signature_cid`). Producer and K consumers
share one team-wide :class:`SharedMultiChainPool`; each
(signature, agent) gets its own W26 disambiguator with its own
``SharedFanoutRegistry`` and ``ChainPersistedFanoutRegistry``. The
audited :class:`MultiChainPersistedFanoutDisambiguator` adds two
content-addressed envelopes (:class:`SalienceSignatureEnvelope`,
:class:`ChainPivotEnvelope`) plus
:func:`verify_salience_signature` (4 enumerated failure modes) and
:func:`verify_chain_pivot` (8 failure modes) for trust-boundary
auditing.

**The headline SDK v3.28 result.** On **R-74-XORACLE-RECOVER**
(1 producer + K=3 consumers, 16 cells, 2 distinct gold-subset
signatures, ``signature_period = 4``, ``max_active_chains = 8``,
partial ServiceGraphOracle on the W26 baseline scoped to GOLD_A
only), the W27 method **simultaneously** strictly reduces
``mean_total_w27_visible_tokens`` over
``mean_total_w26_visible_tokens`` by **−22.5 tokens / cell
(−76.27 %)** at ``T_decoder = None`` AND raises
``correctness_ratified_rate`` from **0.500 → 1.000**. Identical at
``T_decoder = 24``. Stable across **5/5** seeds. The first
capsule-native multi-agent-coordination method that
*simultaneously* improves both efficiency and correctness over
W26 on a regime where W26's single-stack scope architecturally
limits correctness. Four named falsifiers:
**W27-Λ-single-signature** (R-74-CHAIN-SHARED → W27 = W26
byte-for-byte), **W27-Λ-pool-exhausted** (max=2 vs 4 signatures
→ controller rejects beyond bound, W27 falls through to fallback
W26), **W27-Λ-pivot-tampered** (audited disambig wrapper rejects
via ``verify_chain_pivot``), **W27-Λ-signature-drift** (stale
signatures fall through cleanly). **Discharges
W26-C-DIVERGENCE-RECOVERY** in the per-signature scoping
direction. Backward-compat preserved byte-for-byte: 508/508
focused regression covering W18..W27 + IS-1 / IS-2 + producer +
team_coord + attention + capsules. Mac 2 still unreachable
(**22nd milestone in a row**); W27 inherits the W24
``CrossProcessProducerDecoderWire`` proxy.

## TL;DR — SDK v3.27

The programme now has **twenty-three** coupled research axes, each
with a sharp status. SDK v3.27 mints axis 23: **chain-persisted
dense-control fanout + per-consumer projections** — extending the
SDK v3.26 W25 multi-agent fanout with a
``ChainPersistedFanoutDisambiguator`` (W26) that amortises the
producer's per-cell salience-token cost across cells via a two-tier
content-addressed envelope hierarchy: a ``ChainAnchorEnvelope`` at
the chain genesis (carrying canonical compact state +
per-consumer ``ProjectionSlot`` map) and a sequence of
``ChainAdvanceEnvelope`` (each hash-chained to the parent advance)
for in-window cells. At the anchor cell the producer pays the full
W25 cost ``C ≈ 14.6`` tokens; at each subsequent in-window cell the
producer pays a single ``<chain_advance:DDDD>`` token (1 token).
Consumers subscribe at the anchor and emit a 1-token chain-consumer
ref per cell; per-consumer projections enforce controller-verified
scope.

**The headline SDK v3.27 results.** On the synthetic
**R-73-CHAIN-SHARED** regime (1 producer + K=3 consumers sharing a
``ChainPersistedFanoutRegistry``, same R-69-CACHE-FANOUT oracle
ecology, 16 cells, ``chain_persist_window = 16``), W26 strictly
reduces total visible tokens across all agents by **−12.125 tokens
/ cell (−68.79 %)** over the W25 baseline AND **−53.00 tokens /
cell (−90.60 %)** over the W24 baseline at ``T_decoder = None``.
``correctness_ratified_rate = 1.0000`` byte-for-byte;
``chain_consumer_resolved_rate = 1.0000``. Stable across **5/5**
seeds. Four named falsifiers make the conditionality sharp:
**W26-Λ-no-chain** (``chain_persist_window = 1`` → W26 = W25
byte-for-byte), **W26-Λ-tampered** (14/16 advances rejected via
``parent_mismatch``), **W26-Λ-projection-mismatch** (16/16
cross-projection accesses rejected via ``projection_unauthorized``),
**W26-Λ-divergent** (gold subset flips → W26 falls through; no
false savings claim). Trust boundary: ``verify_chain_anchor`` (6
failure modes), ``verify_chain_advance`` (8), ``verify_projection_subscription``
(2). Backward-compat (W26-3-A / W26-3-B) preserved byte-for-byte:
full focused regression on W22..W26 + IS-1 / IS-2 = **180/180 + 6
subtests pass in 15.6s**.

**K-scaling discharge (W25-C-K-SCALING).** The W25-C-K-SCALING
conjecture from SDK v3.26 was empirically discharged at K∈{3,5,8,10}:
W25 saving over W24 grows from 69.87 % at K=3 to 84.69 % at K=10
(close to the conjectured 88 %, slightly below because the cell-0
W25 producer cost is heterogeneous); W26 saving over W24 grows
from 90.60 % at K=3 to 92.23 % at K=10. Anchor:
``docs/data/phase73_k_scaling.json``.

## TL;DR — SDK v3.26

The programme now has **twenty-two** coupled research axes, each
with a sharp status. SDK v3.26 mints axis 22: **shared-fanout
dense-control + cross-agent state reuse** — extending the SDK v3.25
W24 bounded-window compaction with a ``SharedFanoutDisambiguator``
(W25) that lets one producer compute 1 ``FanoutEnvelope`` for K
named consumers; each consumer resolves via 1
``<fanout_ref:DDDD>`` token instead of carrying an independent
compact envelope. Proxy for the LatentMAS "hardware pooling /
shared KV pool" pattern at the capsule layer. The W25 family adds
one new ``FanoutEnvelope``, one ``SharedFanoutRegistry``, one
``verify_fanout``, one ``W25FanoutResult``, and one
``SharedFanoutDisambiguator`` — purely additive on top of the W24
surface. The SDK v3.25 runtime contract is byte-for-byte unchanged.

**The headline SDK v3.26 results.** On the synthetic
**R-72-FANOUT-SHARED** regime (1 producer + K=3 consumers sharing a
``SharedFanoutRegistry``, same R-69-CACHE-FANOUT oracle ecology, 16
cells), W25 strictly reduces total visible tokens across all agents
by **−40.875 tokens / cell (−69.87 %)** at ``T_decoder = None``.
``correctness_ratified_rate = 1.0000`` byte-for-byte;
``fanout_consumer_resolved_rate = 1.0000``. Stable across **5/5**
alternate ``bank_seed`` values: savings = 40.875 tokens/cell on
every seed; min_correctness = 1.000 on every seed. Two named
falsifiers make the W25-1 conditionality sharp: R-72-DISJOINT (no
shared registry → W25 = W24, zero savings, W25-Λ-disjoint) and
R-72-FANOUT-POISONED (unauthorised consumer_id → rejected on every
cell, W25-3). Backward-compat (W25-3-A / W25-3-B) preserved
byte-for-byte: IS-1, IS-2 theorem tests 14/14 + 31/31 new W25 tests
= 45/45 clean.

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` — **20th consecutive milestone with Mac-2
ARP-incomplete.** No two-host W25 execution. Honest scope: W25
reduces multi-agent token overhead at the capsule layer on a single
host.

The W24 family TL;DR (SDK v3.25) is preserved historically below.

## TL;DR — SDK v3.25

The programme now has **twenty-one** coupled research axes, each
with a sharp status. SDK v3.25 mints axis 21: **bounded-window
session compaction + intra-cell resample-quorum + real cross-process
producer/decoder wire** — extending the SDK v3.24 W23 cross-cell
delta with a fixed-size :class:`SessionCompactEnvelope` that folds
the last ``compact_window - 1`` cell digests into one envelope
(visible-token cost is a single ``<compact_ref:DDDD>`` token per
cell), a :class:`ResampleQuorumCachingOracleAdapter` that consults
the inner oracle ``sample_count`` times within one cell and returns
the majority verdict (mitigating intra-cell probabilistic drift the
W23 PER_CELL_NONCE policy cannot touch), a real
:class:`CrossProcessProducerDecoderWire` that round-trips JSON
envelopes through a Python subprocess's stdin/stdout pipes (real
OS-level wire — strictly stronger cross-process honesty than the W23
within-process round-trip), and a synthetic
:class:`IntraCellFlippingOracle` whose drift fits the *intra-cell*
pattern named in W23-C-MITIGATION-LIVE-VARIANCE. The W24 family adds
one new :class:`SessionCompactEnvelope`, one
:func:`verify_session_compact`, one :class:`W24CompactionResult`,
one :class:`MultiCellSessionCompactor`, one
:class:`ResampleQuorumCachingOracleAdapter`, one
:class:`CrossProcessProducerDecoderWire`, and one
:class:`IntraCellFlippingOracle` — purely additive on top of the W23
surface. The SDK v3.24 runtime contract is byte-for-byte unchanged.

**The headline SDK v3.25 results.** On the synthetic
**R-71-LONG-SESSION** regime (the same R-69-CACHE-FANOUT bundle +
oracle ecology used by W22-1 / W23-1 BUT with a 16-cell session
through one persistent :class:`MultiCellSessionCompactor`), the W24
method strictly reduces the visible-token cost to the final decoder
by **−6.81 tokens / cell (−18.0 %)** at ``T_decoder = None`` and
by **−6.81 tokens / cell (−20.5 %)** at ``T_decoder = 24``, AND
ties W23 byte-for-byte on ``accuracy_full = 1.000``. Stable across
**5/5** alternate ``bank_seed`` values (11, 17, 23, 29, 31): savings
≥ 6.69 tokens/cell on every seed; mean savings 6.79 tokens/cell;
``compact_verifies_ok_rate = 0.812`` (13/16 cells beyond the
``compact_window = 4`` threshold; first 3 cells are
``W24_BRANCH_BELOW_WINDOW`` by construction);
``correctness_ratified_rate = 1.000`` byte-for-byte. Two named
falsifiers (R-71-NO-COMPACT, R-71-COMPACT-TAMPERED) make the W24-1
conditionality sharp: chain reset every cell → no compact resolved
(W24-Λ-no-compact); tampered window → ``window_cids_mismatch`` →
fall through to W23 (W24-3). One named mitigation regime
(R-71-INTRA-CELL-FLIP) **empirically discharges
W23-C-MITIGATION-LIVE-VARIANCE on the intra-cell drift axis** at
+0.500 strict gain over W23 PER_CELL_NONCE on synthetic AND **+0.250
strict gain on a fresh live mixtral:8x7b probe** (W24-2). One real
cross-process anchor (R-71-CROSS-PROCESS) records 12 861 bytes
round-tripped through a real Python subprocess pipe with 0 failures
(W24-3 / W24-Λ-cross-host). Backward-compat (W24-3-A / W24-3-B /
W24-3-C) preserved byte-for-byte: 121/121 phase-69/70/71 + capsule
tests + 33/33 new W24 tests = clean.

**Live LLM transfer (W24-Λ-real, empirical n=4 × 1 model,
empirically discharged).** Mac-1 ``mixtral:8x7b`` (47B-MoE) on
R-71-INTRA-CELL-FLIP: ``acc_full(W23 quorum-keyed) = 0.500``;
``acc_full(W24 resample M=3) = 0.750`` — **+0.250 strict
mitigation advantage on a fresh live LLM**. The synthetic +0.500
advantage does not fully transfer because the live LLM does not
perfectly match the deterministic IntraCellFlippingOracle pattern;
names **W24-C-LIVE-VARIANCE-COMPLETE** as the follow-up conjecture
(positive expected improvement bounded by drift-pattern similarity).

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK v3.6
through SDK v3.24 (**18th milestone in a row**). **No two-Mac
sharded inference happened in SDK v3.25.** The W24-3
:class:`CrossProcessProducerDecoderWire` upgrades the W23
within-process round-trip to a real OS-level Python subprocess pipe
— a strictly stronger cross-process honesty proxy. When Mac 2
returns, the same JSON-canonical interface drops in over a real
socket with no W24 code changes. Strongest model class actually
exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE Q4) on Mac-1
Ollama.

The W23 family TL;DR (SDK v3.24) is preserved historically below.

## TL;DR — SDK v3.24

The programme now has **twenty** coupled research axes, each with a
sharp status. SDK v3.24 mints axis 20: **capsule-native cross-cell
delta execution + quorum-keyed cache + super-token reference** —
extending the SDK v3.23 W22 per-cell digest with a hash-chained
*cross-cell* session digest (the LatentMAS *cross-cell latent
state-sharing* direction at the capsule layer), a per-cell delta
that emits only what changed against the running state, a
quorum-keyed cache freshness policy that *mitigates* the SDK v3.23
W22-C-CACHE-AMPLIFICATION conjecture on probabilistic LLM oracles,
a single-visible-token CID-prefix super-token reference (the
bounded steganographic / dense-control-payload experiment) verified
through a controller-side registry, and a within-process
producer/decoder host-split proxy (the honest fallback for the
unreachable Mac 2). The W23 family adds one new
:class:`SessionDigestEnvelope`, one :class:`SessionDeltaEnvelope`,
one :class:`SuperTokenReferenceEnvelope`, one
:class:`SuperTokenRegistry`, one :class:`QuorumKeyedSharedReadCache`,
one :class:`QuorumKeyedCachingOracleAdapter`, one
:class:`CrossHostProducerDecoderProxy`, one :class:`W23SessionResult`
audit record, three new verification functions
(:func:`verify_session_digest_chain`,
:func:`verify_session_delta`,
:func:`verify_super_token_reference`), and one wrapping
:class:`CrossCellDeltaDisambiguator` — purely additive on top of the
W22 surface. The SDK v3.23 runtime contract is byte-for-byte
unchanged.

**The headline SDK v3.24 results.** On the synthetic
**R-70-DELTA-FANOUT** regime (the same R-69-CACHE-FANOUT bundle +
oracle ecology used by W22-1 BUT with a persistent
:class:`CrossCellDeltaDisambiguator` accumulating a hash-chained
session digest across cells), the W23 method strictly reduces the
visible-token cost to the final decoder by **−2.75 tokens / cell
(−6.67 %)** at ``T_decoder = None`` and by **−2.75 tokens / cell
(−7.53 %)** at ``T_decoder = 24`` (delta path); by **−10.50 tokens
/ cell (−25.45 %)** loose AND **−10.50 tokens / cell (−28.77 %)**
tight (super-token path), AND ties W22 byte-for-byte on
``accuracy_full = 1.000``. Stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31): super-token savings
exactly +10.50 tokens / cell on every seed; delta savings ≥ 2.75 on
every seed; ``chain_verifies_ok_rate = 1.000``;
``super_token_resolved_rate = 1.000``;
``correctness_ratified_rate = 1.000``. Three named falsifiers
(R-70-NO-DELTA, R-70-SUPER-TOKEN-TAMPERED, R-70-CHAIN-BROKEN) make
the W23-1 conditionality sharp: no cross-cell state → no savings
(W23-Λ-no-delta); tampered super-token → ``unknown_super_token``
→ fall through to W22 (W23-3); chain head split → ``chain_head_mismatch``
→ fall through (W23-3). One named mitigation regime
(R-70-AMPLIFIED-LLM) **empirically discharges
W22-C-CACHE-AMPLIFICATION** at +0.125 strict gain over W22 (the W22
baseline reproduces ``acc_full = 0.875``; W23 quorum-keyed cache
achieves ``acc_full = 1.000``) — names W23-2.
Backward-compat (W23-3-A / W23-3-B / W23-3-C) preserved
byte-for-byte: 703 prior wevra-anchor + capsule + recent-phase
tests pass + 39 new W23 tests = **742 / 742**.

**Live LLM transfer (W23-Λ-real, empirical n=4 × 1 model,
partially discharged).** Mac-1 ``mixtral:8x7b`` (47B-MoE) on the
AMPLIFIED-LLM bench: visible-tokens savings ``+2.84 %`` (delta),
``+11.37 %`` (super-token); ``chain_verifies_ok_rate = 0.750``;
``super_token_verification_ok_rate = 1.000``;
``cross_host_round_trip_bytes_total = 4232`` bytes (the
producer/decoder split is wire-validated). All four strategies tie
at ``acc_full = 0.750`` — the synthetic ``+0.125`` mitigation
advantage does NOT cleanly transfer to this live n=4 probe. Names
**W23-C-MITIGATION-LIVE-VARIANCE** as the follow-up conjecture
(positive expected improvement, not strict per-probe).

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK v3.6
through SDK v3.23 (**17th milestone in a row**). **No two-Mac
sharded inference happened in SDK v3.24.** The W23 surface is
*naturally* a producer / cache-controller / decoder split: the
:class:`CrossHostProducerDecoderProxy` forces every
delta+session-digest envelope through a JSON-canonical
serialisation round-trip on every cell, mechanically validating
that the W23 envelopes survive a wire boundary with no shared
Python references. When Mac 2 returns the same proxy interface
drops in over a real socket with no W23 code changes. Strongest
model class actually exercised: single-Mac ``mixtral:8x7b`` (46.7
B-MoE Q4) on Mac-1 Ollama.

The W22 family TL;DR (SDK v3.23) is preserved historically below.

## TL;DR — SDK v3.23

The programme now has **nineteen** coupled research axes, each
with a sharp status. SDK v3.23 mints axis 19: **capsule + audited
latent-state-sharing hybrid** — combining the explicit capsule
discipline with the LatentMAS direction (collective KV pooling /
latent hidden-state transfer / super-token side channels) at the
*capsule layer*, with a controller-side trust boundary on every
latent payload. The W22 family adds one new content-addressed
:class:`SchemaCapsule`, one typed :class:`LatentDigestEnvelope`,
one :class:`SharedReadCache` (CID-keyed write-once-read-many),
one :class:`CachingOracleAdapter` (drop-in for any
:class:`OutsideWitnessOracle`), one :class:`EnvelopeTamperer`
(falsifier-test primitive), one :class:`W22LatentResult` audit
record, one :func:`verify_latent_digest` controller function, and
one wrapping :class:`LatentDigestDisambiguator` — purely additive
on top of the W21 surface. The SDK v3.22 runtime contract is
byte-for-byte unchanged.

**The headline SDK v3.23 results.** On the synthetic
**R-69-CACHE-FANOUT** regime (the same R-68-MULTI-MAJORITY bundle
+ oracle ecology that the W21-1 anchor uses, BUT with every
oracle wrapped in a :class:`CachingOracleAdapter` pointing at a
shared :class:`SharedReadCache` AND with the W22 envelope
emitted in place of the verbose W21 audit), the W22 method
strictly reduces the visible-token cost to the final decoder by
**−7 tokens / cell (−14.51 %)** at ``T_decoder = None`` and by
**−7 tokens / cell (−16.09 %)** at ``T_decoder = 24``, AND records
``cache_tokens_saved_total = 88`` over the bank (oracle-side wire
savings), AND ties W21 byte-for-byte on
``accuracy_full = 1.000``. Stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31): savings exactly
+7 tokens / cell on every seed; cache_tokens_saved=88 on every
seed. Three named falsifiers (R-69-NO-CACHE, R-69-POISONED-DIGEST,
R-69-SCHEMA-DRIFT) and one backward-compat anchor (R-69-NO-TRIGGER)
make the W22-1 conditionality sharp: no repeated reads → no
wire-side savings (W22-Λ-no-cache); tampered envelope →
``hash_mismatch`` → fall through to W21 (W22-3); schema mismatch
→ ``schema_cid_mismatch`` → fall through (W22-3); inner W21
abstains → W22 reduces to W21 byte-for-byte (W22-3-A). Backward-
compat (W22-3-A / W22-3-B) preserved byte-for-byte: 633 / 633
prior wevra tests pass + 32 new W22 tests + 10 misc = **675 /
675**.

**Live LLM transfer (W22-Λ-real, empirical n=4 × 2 models,
partially discharged).** Two regimes:

* **mixtral:8x7b** (47B-MoE) as a fourth registered LLM
  adjudicator on cache_fanout: visible-tokens savings
  **+39.08 %** (W21 = 87.0, W22 = 53.0 tokens / cell);
  cache_tokens_saved_total = 120 over 4 cells; verification ok
  rate = 0.750. Correctness ratified rate = 0.750 — newly named
  conjecture **W22-C-CACHE-AMPLIFICATION**: the cache returns
  cell-1's mixtral reply for every subsequent matching cell;
  cell-1's reply variance amplifies across the session.
* **gemma2:9b** (9.2B-dense): every strategy ties at
  ``acc_full = 0.250`` (gemma2's closure-landing rate is the
  structural bound, identical to SDK v3.22 W21-Λ-real
  coalition); W22 ties W21 byte-for-byte
  (``correctness_ratified_rate = 1.000``).

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK v3.6
through SDK v3.22 (16th milestone in a row). **No two-Mac
sharded inference happened in SDK v3.23.** The W22 surface is
*naturally* a producer / cache-controller separation
(``SharedReadCache`` + ``LatentDigestDisambiguator`` is wire-
compatible with cross-host deployment) — no W22 code changes
required when Mac-2 returns. Strongest model class actually
exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE Q4) on Mac-1
Ollama.

The W21 family TL;DR (SDK v3.22) is preserved historically below.

## TL;DR — SDK v3.22

The programme now has **eighteen** coupled research axes, each
with a sharp status. SDK v3.22 mints axis 18: **trust-weighted
multi-oracle adjudication under partial oracle compromise**. The
W21 family adds one new dataclass (``OracleRegistration``), four
oracle adapters (``ChangeHistoryOracle`` / ``OnCallNotesOracle`` /
``SingletonAsymmetricOracle`` / ``DisagreeingHonestOracle``), two
new audit dataclasses (``W21OracleProbe``, ``W21MultiOracleResult``),
and one wrapping decoder (``TrustWeightedMultiOracleDisambiguator``)
— purely additive on top of the W20 surface. The SDK v3.21
runtime contract is byte-for-byte unchanged.

**The headline SDK v3.22 results.** On a synthetic
R-68-MULTI-MAJORITY regime (the same R-66-OUTSIDE-REQUIRED bundle
shape — deceptive primary mentions decoy only AND symmetric
secondary witness mentions all three — but with **three registered
oracles**: ``compromised_registry`` first, ``service_graph``,
``change_history``), every closed-form scorer in the SDK pre-W21
— substrate FIFO, ``capsule_fifo``, …, **W19
``BundleContradictionDisambiguator``**, **AND W20
``OutsideWitnessAcquisitionDisambiguator``** (which trusts the
first-registered compromised oracle and projects to decoy) —
ties FIFO at ``accuracy_full = 0.000``. The W21 method, with the
trust-weighted multi-oracle adjudicator under default
``quorum_min = 2``, achieves ``accuracy_full = 1.000`` on
R-68-MULTI-MAJORITY-LOOSE (``T_decoder = None``) AND
R-68-MULTI-MAJORITY-TIGHT (``T_decoder = 24``), strictly improving
over every non-W21 capsule baseline including W20 by **+1.000**,
stable across **5/5** alternate ``bank_seed`` values
(11, 17, 23, 29, 31). Three named falsifiers (R-68-MULTI-NO-QUORUM,
R-68-MULTI-ALL-COMPROMISED, R-68-MULTI-PARTIAL) make the W21-1
conditionality sharp: no quorum → abstain → tie FIFO; all
compromised → quorum forms on decoy → fail at 0.000; sub-quorum
honest → abstain at default → tie FIFO. The conditional
W21-C-PARTIAL-RECOVERY (with override ``quorum_min = 1`` on
R-68-MULTI-PARTIAL) is empirically discharged at 1.000.
Bounded-context honesty: the W21 layer issues *exactly N*
outside queries per cell, each bounded by ``max_response_tokens``;
the W15 ``tokens_kept`` is byte-for-byte identical between W19,
W20 AND W21. Backward-compat (W21-3-A / W21-3-B) preserved
byte-for-byte: 585 / 585 prior wevra tests pass + 48 new W21
tests pass = **633 / 633**.

**Live LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-REGISTRY,
empirical n=4 × 2 models).** Two regimes:

* **Mixed-registry (registry-anchored, easy)** — four-oracle
  registry pairing deterministic ``service_graph`` +
  ``change_history`` with ``ollama_mixtral:8x7b``: W21 acc_full =
  **1.000**, +1.000 over W20. **W21-C-LIVE-WITH-REGISTRY
  partially discharged**.
* **Coalition (LLM-vote-required, hard)** — three-oracle registry
  with one honest deterministic + one LLM + one compromised,
  ``quorum_min = 2`` (LLM vote required for quorum on gold):
  cross-model split is sharp.
  - ``mixtral:8x7b`` (47B-MoE): W21 = **0.750**, +0.750 over W20.
  - ``gemma2:9b`` (9.2B-dense): W21 = **0.000**, +0.000 (gemma2
    lands decoy tokens through the closure; quorum forms on decoy).

**Scale + general knowledge matter for the W21-Λ-real escape on the
LLM-vote-required regime**.

The W20 family TL;DR (SDK v3.21) is preserved historically below.
SDK v3.21 mints axis 17: **outside-witness
acquisition under bundle-only insufficiency (outside-resolvable
case)**. The W20 family adds one new Protocol
(``OutsideWitnessOracle``), four oracle adapters
(``ServiceGraphOracle`` / ``CompromisedServiceGraphOracle`` /
``AbstainingOracle`` / ``LLMAdjudicatorOracle``), three new
dataclasses (``OutsideQuery``, ``OutsideVerdict``,
``W20OutsideResult``), one default service-graph
(:func:`build_incident_triage_service_graph`), and one wrapping
decoder (``OutsideWitnessAcquisitionDisambiguator``) — purely
additive on top of the W19 surface. The SDK v3.20 runtime
contract is byte-for-byte unchanged.

**The headline SDK v3.21 results.** On a synthetic
R-67-OUTSIDE-RESOLVES regime (the same R-66-OUTSIDE-REQUIRED
bundle shape — deceptive primary mentions decoy only AND
symmetric secondary witness mentions all three — but with a
registered :class:`ServiceGraphOracle`), every closed-form
scorer in the SDK pre-W20 — substrate FIFO, ``capsule_fifo``,
``capsule_priority``, ``capsule_coverage``, W7-2 cohort, W8
corroboration, W9 multi-service, W11 multi-round, W12 robust-
multi-round, W13 layered, W15 ``AttentionAwareBundleDecoder``,
W14H + W15 composition, **W18 ``RelationalCompatibilityDisambiguator``**,
**AND W19 ``BundleContradictionDisambiguator``** — ties FIFO at
``accuracy_full = 0.000`` (W19-Λ-outside extends verbatim:
W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC`` because the
asymmetric-witness count is uniform across all admitted tags).
The W20 method, with the deterministic ServiceGraphOracle,
achieves ``accuracy_full = 1.000`` on R-67-OUTSIDE-RESOLVES-LOOSE
(``T_decoder = None``) AND R-67-OUTSIDE-RESOLVES-TIGHT
(``T_decoder = 24``), strictly improving over every non-W20
capsule baseline by **+1.000**, stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31). Three named falsifiers
(R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED, R-67-JOINT-DECEPTION)
make the W20-1 conditionality sharp: no signal → abstain → tie
FIFO; adversarial signal → trust → fail at 0.000; jointly
compromised → tie W19 at 0.000. Bounded-context honesty: the W20
layer adds *exactly one* outside query per cell, bounded by
``max_response_tokens = 24``; the W15 ``tokens_kept`` is
byte-for-byte identical between W19 and W20. Backward-compat
(W20-3) preserved byte-for-byte: 545 / 545 prior wevra tests
pass + 40 new W20 tests pass = 585 / 585. A *partial* live-LLM
W20-Λ-real probe on Mac-1 shows ``mixtral:8x7b`` (47B-MoE) free-
form replies achieving ``acc_full = 0.750`` (+0.750 over W19);
``qwen2.5-coder:7b`` trusts the deceptive primary and fails. Mac
2 remains unreachable; no two-Mac sharded inference.

The previous (now extended) headline: SDK v3.20 minted axis 16
(**bundle-contradiction-aware trust-weighted disambiguation
under deceptive / confounded round-2 evidence — bundle-resolvable
case**). The W19
family adds one new dataclass (``W19TrustResult``), two closed-
form helpers (``_w19_canonical_primary_index``,
``_w19_witness_counts``), one canonical-role-for-kind table
(``_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND``), and one wrapping
decoder (``BundleContradictionDisambiguator``) — purely additive
on top of the W18 surface. The SDK v3.19 runtime contract is
byte-for-byte unchanged.

**The headline SDK v3.20 results.** On a synthetic
R-66-DECEIVE-NAIVE regime (symmetric round-1 corroboration; round-
2 primary specific-tier disambiguator names DECOY service tags
ONLY via ``relation=decoy_decoy_*``; round-2 secondary specific-
tier witness — emitted by a non-canonical producer role under a
synonym kind that resolves through the W12 / W13 normalisation
closure to the same canonical specific-tier kind as the primary —
names GOLD service tags ONLY via ``relation=A_B_*``), every
closed-form scorer in the SDK pre-W19 — substrate FIFO,
``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
W7-2 cohort, W8 corroboration, W9 multi-service, W11 multi-round,
W12 robust-multi-round, W13 layered, W15
``AttentionAwareBundleDecoder``, W14H + W15 composition, AND **W18
``RelationalCompatibilityDisambiguator``** — ties FIFO at
``accuracy_full = 0.000`` (W18-Λ-deceive extends verbatim:
W18's full-disambiguator scorer sees positive scores on every
admitted tag, abstains, and falls through to the empty inner W15
answer). The new :class:`BundleContradictionDisambiguator` (W19)
achieves ``capsule_bundle_contradiction = 1.000`` at both
``T_decoder = None`` (R-66-DECEIVE-NAIVE-LOOSE) AND
``T_decoder = 24`` (R-66-DECEIVE-NAIVE-TIGHT) AND on
R-66-CONFOUND-RESOLVABLE (primary names all three; secondary
names gold), strictly improving over the W18 baseline by
**+1.000** on all three regimes, stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31). **First capsule-native
multi-agent-coordination method that resolves bundle-internal
contradiction between a deceptive primary and a witness-
corroborated alternative (W19-1).** Two named falsifiers make
the conditionality sharp: R-66-DECEIVE-TOTAL (no asymmetric
witness anywhere — W19 reduces to W18 and FAILS at 0.000;
W19-Λ-total) and R-66-OUTSIDE-REQUIRED (witnesses are themselves
symmetric across primary's named set and the complement — W19
abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC`` and ties FIFO at
0.000; W19-Λ-outside). Token-budget honesty: the W19 method
consumes only the W18-packed bundle (which itself reads only the
W15-packed bundle); ``tokens_kept_sum`` is byte-for-byte identical
to W18's on R-66-DECEIVE-NAIVE-TIGHT (188 / 226 tokens kept across
8 cells; same bundle, no extra capsule reads). Backward-compat
(W19-3) preserved byte-for-byte: on R-58 default and on every
R-65 default bank (compat / no_compat / confound / deceive), W19
ties W18 byte-for-byte; with ``enabled = False`` the W19 method
reduces to W18 byte-for-byte. **All prior wevra tests pass**
(450 / 450 in the targeted wevra suites; 555 / 555 across the
full ``test_wevra_*.py`` set with 45 new W19 tests).

**Honest scope.** R-66 is a *synthetic* regime — the producer
is :class:`IdentityExtractor`. Real-LLM transfer of the W19
method is **W19-Λ-real** (proved-conditional + empirical-research):
the LLM must emit the secondary witness in the same closed-
vocabulary form (synonym specific-tier kinds + relational-compound
payloads) AND from a non-canonical producer role; if the LLM
emits free-form natural-language witnesses, the W19 exact-match
layer misses by construction. The natural extension is
**W19-C-LEARNED** (a small distilled trust scorer over capsule
bundles), conjectural. The W19-Λ-total falsifier names the
structural limit when the bundle is exhausted of asymmetric
signal; the W19-Λ-outside falsifier names the structural limit
when bundle-internal contradiction is itself symmetric. The
natural escape from BOTH falsifier walls is **outside information**
(W19-C-OUTSIDE — service-graph topology, prior reliability
scores, cross-incident historical evidence), conjectural.

The **prior-conjecture discharge ledger** for SDK v3.20:
* **W18-Λ-deceive** (SDK v3.19; "no closed-form bundle-relational
  scorer that *trusts* its evidence can escape adversarial round-2
  evidence"). **PARTIALLY DISCHARGED-empirical** by W19-1 in the
  *bundle-resolvable* direction: a deterministic training-free
  bundle-contradiction-aware trust scorer is sufficient on
  R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE when the bundle
  carries an independent asymmetric witness for gold. The
  *bundle-only* clause where no witness exists (W19-Λ-total)
  AND the *symmetric-witness* clause (W19-Λ-outside) remain
  real and structural; the named research move beyond them is
  W19-C-OUTSIDE (outside-information axis), conjectural.
* **W18-Λ-confound** (SDK v3.19; "no closed-form bundle-relational
  scorer can break a symmetric primary"). **PARTIALLY DISCHARGED-
  empirical** by W19-1 in the *bundle-resolvable* direction on
  R-66-CONFOUND-RESOLVABLE: when the bundle carries a witness
  asymmetric for gold, W19 picks the strict-max-witness subset.
  The *no-witness* and *symmetric-witness* cases remain bounded
  by the same falsifiers above.

---

The *previous* (SDK v3.19) frontier mints axis 15: **bundle-
relational compatibility disambiguation under symmetric
corroboration**. The W18 family adds one new dataclass
(``W18CompatibilityResult``), one tokeniser
(``_disambiguator_payload_tokens``), one closed-form scorer
(``_relational_compatibility_score``) with contiguous-subsequence
semantics for compound targets, and one wrapping decoder
(``RelationalCompatibilityDisambiguator``) — purely additive on
top of the W15 surface. The SDK v3.18 runtime contract is
byte-for-byte unchanged.

**The headline SDK v3.19 results.** On a synthetic R-65-COMPAT
regime (every gold service AND the decoy mentioned by ≥ 2 distinct
routed roles via generic-noise kinds with comparable magnitudes —
symmetric-corroboration; round-2 specific-tier disambiguator
carries a relational-compound mention of every gold service AND
no decoy service), every closed-form salience scorer in the SDK
ties FIFO at ``accuracy_full = 0.000`` (W17-Λ-symmetric extended
to R-65 verbatim by W18-Λ-sym). The new
:class:`RelationalCompatibilityDisambiguator` (W18) achieves
``capsule_relational_compat = 1.000`` at both ``T_decoder = None``
(R-65-COMPAT-LOOSE) AND ``T_decoder = 24`` (R-65-COMPAT-TIGHT),
strictly improving over every non-W18 capsule baseline by
**+1.000**, stable across **5/5** alternate ``bank_seed`` values
(11, 17, 23, 29, 31). **First capsule-native multi-agent-
coordination method that crosses the symmetric-corroboration wall
on a regime where the wall actually applies (W18-1).** Three
named falsifiers make the conditionality sharp: R-65-NO-COMPAT
(no signal — W18 abstains, ties FIFO; W18-Λ-no-compat),
R-65-CONFOUND (symmetric signal — W18 abstains, ties FIFO;
W18-Λ-confound), R-65-DECEIVE (adversarial signal — W18 trusts
evidence, picks decoy, fails at 0.000; W18-Λ-deceive). Token-
budget honesty: the W18 method consumes only the W15-packed
bundle; ``tokens_kept_sum`` is byte-for-byte identical to W15's
on R-65-COMPAT-TIGHT. Backward-compat (W18-3) preserved byte-
for-byte: on R-58 default the W18 method ties W15 byte-for-byte
on the answer field; on R-64-SYM the W18 method partially
recovers (only the deadlock scenarios carry a relational mention;
on pool / disk / slow_query the W18 method abstains and ties
FIFO).

**Honest scope.** R-65-COMPAT is a *synthetic* regime — the
producer is :class:`IdentityExtractor`. Real-LLM transfer of the
W18 method is **W18-Λ-real** (proved-conditional + empirical-
research): the LLM must emit the same closed-vocabulary
relational-compound forms the synthetic bench uses; if the LLM
emits free-form natural-language relational mentions, the W18
exact-match layer misses by construction. The natural extension
is **W18-C-LEARNED** (a small distilled compatibility scorer over
capsule bundles), conjectural. The W18-Λ-deceive falsifier names
the structural limit of *any* closed-form bundle-relational
scorer that trusts its evidence; the natural escape is
**W18-C-OUTSIDE** (an outside-information axis to detect
deceptive round-2 mentions), conjectural.

The **prior-conjecture discharge ledger** for SDK v3.19:
* **W17-C-DISAMBIGUATOR** (SDK v3.18; "a learned or LLM-distilled
  semantic disambiguator beyond the closed-form capsule surface
  could distinguish gold from decoy on R-64-SYM-deadlock-style
  scenarios"). **DISCHARGED-empirical** by W18-1 in the
  *closed-form* direction: a deterministic training-free bundle-
  relational scorer is sufficient on R-65-COMPAT. The
  *learned-disambiguator* clause remains conjectural under
  W18-C-LEARNED (relevant when the LLM emits free-form mentions
  outside the closed-vocabulary closure).

---

The *previous* (SDK v3.18) frontier mints axis 14: **magnitude-hinted
producer protocol + fresh-live end-to-end composition +
symmetric-corroboration limit theorem**. The W17 family adds one
new producer-prompt mode (``PRODUCER_PROMPT_MAGNITUDE_HINTED``),
one new dataclass (``OperationalThreshold``), one new schema
field, and one new prompt-render helper — purely additive on top
of the W14 surface. The runtime contract is byte-for-byte
unchanged.

**The headline SDK v3.18 results.** On a *fresh* live Mac-1
``qwen2.5:14b-32k`` Ollama probe at ``T_decoder = 14, K_auditor
= 8`` against the Phase-61 comparable-magnitude bank (n=8 × 24
producer calls; 0 endpoint failures; 128.2 s wall): under the
W17 magnitude-hinted prompt, bench property holds in **8/8**
(closing the 1/8 R-61-OLLAMA-A model-side judgment miss);
``capsule_attention_aware = 1.000``;
``capsule_layered_fifo_packed = 0.000``;
``capsule_fifo = 0.000``. **+1.000 strict separation** on both
axes — the **first programme result** that beats the strongest
non-composed baseline by ≥ 1.0 on a fresh live LLM probe (W17-1).

The W17-Λ-no-hint anchor on the same fresh probe under the
*legacy* structured prompt reproduces the W14-Λ-real envelope
(7/8 hold; +0.500 strict gain over FIFO-pack); the magnitude-
hint extension, not a re-run of the same prompt, is what closes
the gap from 0.500 to 1.000. The W17-Λ-naive falsifier on the
same probe under the naive prompt collapses to 0/8 + 0.000
(live counterpart of the W16-Λ-compose joint-failure regime).

The cross-model probe on a fresh live Mac-1 ``qwen3.5:35b`` MoE
backend (``think = False``; n=8 × 24 producer calls; 0 failures;
92.0 s wall) shows the magnitude-hint protocol **transfers**:
bench property holds in **8/8** (the W17 extension preserves the
bench-property hold-rate byte-for-byte across a 2.4× model-class
jump); ``capsule_attention_aware = 0.750``; **+0.750 strict gain**
over substrate FIFO and FIFO-packed-W14H-only. The 0.250 gap to
1.000 is on the ``accuracy_root_cause`` axis — a model-class-
specific specific-tier judgment artifact, not a producer-protocol
failure (W17-C-XMODEL, proved-conditional + empirical-research).

The **first explicit symmetric-corroboration limit theorem**
(W17-Λ-symmetric) lands as a *negative* result on the synthetic
``build_phase64_sym_bank`` (every service mentioned by exactly 2
distinct routed producer roles via generic-noise kinds with
comparable magnitudes; round-2 disambiguator names the gold
root_cause without a ``service=`` token). Under both
``T_decoder ∈ {None, 24}``: every capsule strategy in the SDK
ties FIFO at ``accuracy_full = 0.000`` — including the W14H +
W15 composition. The priority decoder still elects the right
specific-tier ``root_cause`` (``accuracy_root_cause = 1.000``);
the failure is ``services_correct`` set-equality. The structural
argument is that ``services_correct`` is an asymmetric oracle:
when the bipartite ``(role × tag, kind, magnitude)`` multiset is
symmetric for gold and decoy, no service-blind admission AND no
closed-form salience packer can prefer one. **W17-Λ-symmetric
discharges the prior W15-C-SYMMETRIC / W16-C-SYMMETRIC
conjectures as a negative theorem and names the next research
frontier**: a learned or LLM-distilled
semantic-disambiguator beyond the closed-form capsule surface
(W17-C-DISAMBIGUATOR, conjectural).

The **prior-conjecture discharge ledger** for SDK v3.18:
* W16-C-LIVE-OLLAMA → DISCHARGED (W17-1).
* W16-C-CROSS-MODEL → PARTIALLY DISCHARGED (W17-C-XMODEL on
  Ollama; MLX-distributed clause remains conjectural).
* W15-C-SYMMETRIC / W16-C-SYMMETRIC → DISCHARGED-NEGATIVE
  (W17-Λ-symmetric).

Backward-compat (W17-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``mode = 'naive'`` or ``mode = 'structured'``
AND ``magnitude_hinted_schema = False``, the W17 surface
reduces to the SDK v3.15 W14 anchor byte-for-byte. The Wevra
single-run product runtime contract is byte-for-byte unchanged.
See `docs/RESULTS_WEVRA_LIVE_COMPOSITION.md` for the milestone
note.

---

The *previous* (SDK v3.17) frontier mints axis 13: **end-to-end
W14 + W15 composition**. The composition does not introduce new
SDK code (no new admission policy, decoder, normaliser, packer,
or producer protocol — the runtime contract is byte-for-byte
unchanged). What SDK v3.17 *does* mint is the first programme
regime where the W14 producer-side ambiguity-preservation layer
AND the W15 decoder-side capsule context-packing layer are
*independently load-bearing on the same cell*.

On the new R-63-COMPOSED-TIGHT regime (Phase-62 multi-hypothesis
events × Phase-61 magnitude-filter producer simulation under the
W14 :class:`StructuredProducerProtocol` × ``T_decoder = 24``), the
W14+W15 composition achieves ``accuracy_full = 1.000`` while every
non-composed baseline collapses to 0.000 — a **+1.000 strict
separation** vs the W14-only-budgeted baseline (FIFO-packed-W13
over the structured-prompt stream) and the W15-only-without-W14
baseline (AttentionAwareBundleDecoder over the naive-prompt
stream). The result is stable across 5/5 alternate ``bank_seed``
values. The W16-Λ-compose joint-failure anchor on R-63-naive-tight
shows the producer-side limit (W14-Λ-prompt) and the decoder-side
limit (W15-Λ-budget) compose multiplicatively; closing one alone
leaves the other firing.

Crucially, the W16-Λ-real-replay anchor on the recorded Phase-61
``qwen2.5:14b-32k`` bytes (``phase61_real_ollama_structured_qwen2_5_14b_n8.json``,
n=8 × 24 producer calls, byte-stable, 0 endpoint failures) shows
the composed pipeline achieves ``capsule_attention_aware = 0.500``
while ``capsule_layered_fifo_packed = 0.000`` at ``T_decoder = 14``
on the recorded LLM bytes — a **+0.500 strict gain** over the
strongest non-composed baseline on a real-LLM stream. This is the
**first end-to-end real-LLM strict advance** in the programme that
beats the strongest non-composed baseline. The Mac-1 endpoint at
192.168.12.191:11434 was offline at milestone capture time
(``HTTP=000``), so a fresh live LLM probe (W16-C-LIVE-OLLAMA) is
conjectural.

Backward-compat (W16-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``T_decoder = None`` and identity producer, the
composed pipeline ties the W13 layered decoder; with structured
prompt + ``T_decoder = None`` it ties the W14-1 anchor on R-61.
The Wevra single-run product runtime contract is byte-for-byte
unchanged. See `docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md` for the
milestone note.

1. **Capsule contract / runtime** — *active, advancing*. The
   contract (C1..C6) is settled. SDK v3.4 pushes capsule-native
   execution to the LLM byte boundary inside one Wevra run
   (W3-42..W3-45). The lifecycle audit covers L-1..L-11.
2. **Multi-agent capsule coordination** — *active, new (SDK
   v3.5)*. Capsule-native team coordination via TEAM_HANDOFF /
   ROLE_VIEW / TEAM_DECISION capsules. ``TeamCoordinator`` drives
   one round; ``audit_team_lifecycle`` mechanically verifies
   invariants T-1..T-7 (Theorem W4-1). Coverage-implies-
   correctness (W4-2) and local-view limitation (W4-3) hold on
   the Phase-52 incident-triage bench. A learned per-role
   admission policy admits **strictly fewer handoffs** (12/12
   train seeds, deterministic in direction) and improves pooled
   team-decision accuracy *on most seeds* (gap_full > 0 in 11/12
   seeds, mean +0.054; gap_root_cause > 0 in 8/12 seeds, mean
   +0.032) over the strongest fixed baseline (coverage-guided)
   on the Phase-52 default config — but the accuracy advantage
   reverses at higher noise (W4-C1 honest reading). This is the
   team-level slice of the original Context-Zero "solve context
   for multi-agent teams" thesis — the first slice that runs the
   capsule abstraction *between* agents, not just inside one run.
3. **Decoder frontier** — *open, with sharp limitation theorems*.
   The strict pre-Phase-50 paradigm-shift bar (W3-C7 strict) is
   **retracted** (W3-26, W3-27). The defensible reading is
   W3-C9 (Phase-49 candidate at $n=80$, gap reading at zero-shot).
   The next research direction is the relational decoder at
   higher level (Phase 51, W3-30 / W3-31 / W3-C10).
4. **Substrate primitives** — *settled*. CASR routing, exact
   memory, typed handoffs, escalation threads, adaptive
   subscriptions. ~1500 substrate tests, no active development on
   substrate primitives themselves.
5. **Two-Mac distributed inference + real cross-LLM measurement**
   — *active, settled (SDK v3.6)*. The chosen path for one-larger-
   model inference across two Apple Silicon Macs is **MLX
   distributed** (under `mpirun mlx_lm.server`); the Wevra-side
   integration boundary is one duck-typed `LLMBackend` Protocol
   plus an `MLXDistributedBackend` adapter that talks
   OpenAI-compatible HTTP. Real cross-LLM measurement on the
   available model class (Qwen-2.5-14B-dense vs
   Qwen-3.5-35B-MoE on Mac 1) yields **W5-1
   (proved-empirical)**: cross-model PARSE_OUTCOME failure-kind
   TVD = 1.000 under strict parsing on the bundled bank,
   collapsing to 0.000 under robust parsing — the **first real
   confirmation** that the capsule-native runtime survives a
   2.4× model-class jump and a dense → MoE architecture swap
   without spine modification. The two-Mac MLX-distributed path
   is **experimental infrastructure**, not product; the Wevra
   single-run product runtime contract is byte-for-byte
   unchanged. Mac 2 remains offline at the time of SDK v3.7
   (192.168.12.248 ARP "incomplete"); the runbook is the
   operator path when Mac 2 returns.
7. **Cross-role cohort-coherence multi-agent coordination**
   — *active, new (SDK v3.8)*. **Phase-54** benchmark
   (`vision_mvp/experiments/phase54_cross_role_coherence.py`)
   directly attacks the SDK v3.7 Phase-53 failure mode by
   redesigning the regime so structure has a real chance: a
   deterministic candidate stream where each scenario has one
   ``real_service`` (gold) and one ``decoy_service`` (foreign);
   each producer role emits ``service=<tag>``-tagged candidates
   with the gold tag in **strict plurality**; the auditor sees
   surplus candidates above ``K_auditor=4`` (``5 ≤ |candidates| ≤ 7``).
   The new admission policy
   ``CohortCoherenceAdmissionPolicy`` (in
   ``vision_mvp.wevra.team_coord``) provides two sub-modes:
   *streaming* (running cohort over admitted) and *buffered*
   (pre-fitted plurality from candidate stream's payloads via
   ``from_candidate_payloads``). Headline: at the pre-committed
   default, ``capsule_cohort_buffered`` achieves
   ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   and ``capsule_cohort_streaming`` all produce 0.000 — a
   **+1.000** structural win, stable across 5/5 alternate bank
   seeds. The W7 family (W7-1 / W7-1-aux / W7-2 /
   W7-2-conditional / W7-3 — proved or proved-empirical) anchors
   the milestone formally; the W7-C family makes the multi-service
   / decoder-side / real-LLM extensions falsifiable. **Honest
   scope:** the structural win is *conditional* on the bench
   property (gold-plurality + foreign-service decoys + budget
   surplus); the streaming variant is unstable and ties FIFO
   (W7-1-aux); W7-3 is the extraction floor — admission cannot
   recover claims the producer never emitted (the Phase-53
   ``deadlock_pool_exhaustion`` failure case).

9. **Multi-service top-K cross-role corroboration multi-agent
   coordination** — *active, new (SDK v3.10)*. **Phase-56**
   benchmark
   (`vision_mvp/experiments/phase56_multi_service_corroboration.py`)
   directly attacks the W8 *multi-service-gold* falsifier by
   building the smallest deterministic regime where (i) every
   scenario has ``gold_services`` of size 2 (multi-service incident),
   (ii) both gold services are corroborated by ≥ 2 distinct producer
   roles each, AND (iii) a decoy service has raw plurality but is
   corroborated by exactly 1 producer role. 10/10 default scenarios
   satisfy all three properties; mechanically verified by
   ``Phase56BankShapeTests``. The new admission policy
   ``MultiServiceCorroborationAdmissionPolicy`` admits the **top-K
   cross-role-corroborated tier** (default ``top_k=2,
   min_corroborated_roles=2``) via the argmax-by-role-count gate —
   strictly generalising W8 single-tag corroboration. Headline: at
   the pre-committed default (``K_auditor=4``, ``T_auditor=128``,
   ``n_eval=10``, ``bank_seed=11``), ``capsule_multi_service``
   achieves ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   ``capsule_cohort_buffered`` (W7-2), AND
   ``capsule_corroboration`` (W8) all produce 0.000 — the **first
   strict separation between multi-service top-K corroboration and
   single-tag corroboration**, **+1.000** vs the SDK v3.9 strongest
   method, stable across **5/5** alternate bank seeds. The W9
   family (W9-1 / W9-2 / W9-3 / W9-4 — proved or proved-empirical)
   anchors the milestone formally; the W9-C family makes the
   bundle-aware decoder / |gold|≥3 / real-LLM extensions
   falsifiable. **Honest scope:** the win is *conditional* on the
   named bench property (multi-service-gold + single-role-decoy);
   W9-4 is the named falsifier regime where the decoy is also
   corroborated and W9 ties FIFO at 0.000; W9-3 backward-compat
   preserves W8 on Phase 55 and W7-2 on Phase 54; no regression on
   Phase 53 synthetic. The milestone clears the **strong success
   bar** of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1
   (R-56 anchor) — a strict gain ≥ 0.20 on Phase 56 vs both
   substrate FIFO and SDK v3.9 W8, stable across ≥ 3 seeds, no
   regression on R-53 / R-54 / R-55, audit T-1..T-7 preserved on
   every cell, named bench property + named falsifier regime.

8. **Cross-role corroboration multi-agent coordination**
   — *active, new (SDK v3.9)*. **Phase-55** benchmark
   (`vision_mvp/experiments/phase55_decoy_plurality.py`) directly
   attacks the W7-2 falsifier by building the smallest deterministic
   regime where (i) some decoy service has *strictly more raw
   mentions* than gold (so W7-2 single-tag plurality picks the
   decoy and ties FIFO at 0.000) AND (ii) the gold service is
   *cross-role corroborated* — mentioned by strictly more distinct
   producer roles than any decoy. 10/10 default scenarios satisfy
   both properties; mechanically verified by
   ``Phase55BankShapeTests``. The new admission policy
   ``CrossRoleCorroborationAdmissionPolicy`` aggregates over
   (role, tag) bipartite multisets via the score function
   ``score(tag) = role_weight·|distinct_roles(tag)| +
   |raw_mentions(tag)|``. Headline: at the pre-committed default
   (``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
   ``bank_seed=11``), ``capsule_corroboration`` achieves
   ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   AND ``capsule_cohort_buffered`` (W7-2) all produce 0.000 —
   the **first strict separation** between cross-role corroboration
   and W7-2 single-tag plurality, **+1.000** vs both baselines,
   stable across **5/5** alternate bank seeds. The W8 family
   (W8-1 / W8-2 / W8-3 / W8-4 — proved or proved-empirical)
   anchors the milestone formally; the W8-C family makes the
   multi-service / decoder-side / real-LLM extensions falsifiable.
   **Honest scope:** the win is *conditional* on the named bench
   property (decoy-plurality + cross-role-corroborated gold);
   W8-4 is the named falsifier regime where the decoy is
   corroborated and corroboration ties FIFO at 0.000;
   W8-3 backward-compat preserves W7-2 on Phase 54 (corroboration
   matches cohort_buffered at 1.000); no regression on Phase 53
   synthetic (0.800) or Phase 53 14B real-LLM (0.800). The
   milestone clears the **strong success bar** of
   `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 — a
   strict gain ≥ 0.20 on Phase 55, stable across ≥ 3 seeds, no
   regression on Phase 53 / Phase 54, audit T-1..T-7 preserved
   on every cell, named bench property + named falsifier regime.

6. **Model-scale vs capsule-structure on multi-agent
   coordination** — *active, settled (SDK v3.7)*. **Phase-53**
   benchmark (`vision_mvp/experiments/phase53_scale_vs_structure.py`)
   replaces the Phase-52 deterministic producer-role extractor
   with a real-LLM extractor and decomposes ``accuracy_full``
   across (model regime × admission strategy). Headline (n=5,
   K_auditor=4): every fixed admission strategy (substrate /
   capsule_fifo / capsule_priority / capsule_coverage) achieves
   ``accuracy_full = 0.800`` in every model regime; only
   ``capsule_learned`` varies (0.400 on synthetic and 14B; 0.800
   on 35B). ``structure_gain`` is **non-positive at every
   regime** (-0.4 / -0.4 / 0.0); ``scale_gain[capsule_learned]
   = +0.4``, ``scale_gain[fixed] = 0.0``. **W4-C1 is conditionally
   falsified** out-of-distribution on the real-LLM regime
   (capsule_learned underperforms FIFO by 0.4 on synthetic and
   14B; ties at 35B). Honest reading: scale closes a *structure
   deficit* (created by OOD over-rejection of clean candidates
   by the SDK v3.5 learned policy), not a *structure surplus*.
   The capsule layer's load-bearing contribution at this
   benchmark is the **lifecycle audit (T-1..T-7, 60/60 across
   regimes)**, not admission policy gains. The W6 family
   (W6-1/2/3/4 proved + mechanically-checked + empirically-
   saturated) anchors the milestone formally; the W6-C family
   (W6-C1/C2 falsified-empirical, W6-C3 positive, W6-C4/C5
   conjectural) makes the empirical reading falsifiable.

## Current frontier (SDK v3.21, 2026-04-29)

### Active moves (SDK v3.21 — outside-witness acquisition disambiguator + R-67 outside-information benchmark family + W20 family — *first capsule-native method that crosses the W19-Λ-outside wall on a regime where it applies*)

- **W20-Λ-outside-extension** (proved-empirical n=8 saturated +
  structural sketch) — W19-Λ-outside extends verbatim to
  R-67-OUTSIDE-REQUIRED-BASELINE (no oracle / abstaining oracle):
  every capsule strategy through W19 ties FIFO at ``accuracy_full
  = 0.000``. The wall is real for every closed-form bundle-only
  scorer.
- **W20-1** (proved-conditional + proved-empirical n=80 saturated
  across 5 seeds × 2 budgets, also n=12) — pairing W19 with the
  new ``OutsideWitnessAcquisitionDisambiguator`` over a
  deterministic ``ServiceGraphOracle`` strictly improves
  ``accuracy_full`` over every non-W20 capsule baseline (incl.
  W19) by **+1.000** on R-67-OUTSIDE-RESOLVES-LOOSE AND
  R-67-OUTSIDE-RESOLVES-TIGHT, stable across 5/5 ``bank_seed``
  values. The first capsule-native multi-agent-coordination
  method that crosses the W19-Λ-outside wall on a regime where
  the wall actually applies.
- **W20-2** (proved by inspection + mechanically-checked) —
  Determinism + closed-form correctness; positive-set projection
  rule; ``n_outside_tokens`` recorded as strict additional cost;
  W15 ``tokens_kept`` byte-for-byte unchanged from W19.
- **W20-3** (proved-empirical full programme regression; 585/585
  wevra tests pass, 545 pre-existing + 40 new W20 tests) —
  Backward-compat with R-54..R-66 default banks; W20 reduces to
  W19 byte-for-byte either via no-trigger or outside-abstained.
  With ``enabled = False`` it reduces to W19 byte-for-byte.
- **W20-Λ-none** (proved-empirical n=8 saturated) —
  ``AbstainingOracle`` ⇒ W20 ties FIFO at 0.000.
- **W20-Λ-compromised** (proved-empirical n=8 saturated) —
  ``CompromisedServiceGraphOracle`` ⇒ W20 trusts decoy and FAILS
  at 0.000.
- **W20-Λ-joint-deception** (proved-empirical n=8 saturated) —
  primary + secondary + oracle all favour decoy ⇒ W20 ties W19
  at 0.000. Names the structural limit when *all* evidence
  channels are jointly compromised.
- **W20-Λ-real** (proved-conditional + empirical-research n=4 ×
  2 models on Mac-1 Ollama) — ``mixtral:8x7b`` 47B-MoE achieves
  ``acc_full = 0.750`` (+0.750 over W19); ``qwen2.5-coder:7b``
  ties FIFO at 0.000. Cross-model split: scale + general
  knowledge correlates with W20-Λ-real escape.
- **Two-Mac status** — Mac 2 (192.168.12.248) still ARP
  ``incomplete``. **No two-Mac sharded inference happened in SDK
  v3.21.** The W20 ``OutsideWitnessOracle`` Protocol is
  infrastructure-ready for cross-host deployment when Mac 2
  returns; the ``MLXDistributedBackend`` adapter is byte-for-byte
  unchanged.

### Discharged / partially-discharged conjectures (SDK v3.21)

- **W19-C-OUTSIDE** (SDK v3.20): **PARTIALLY DISCHARGED-empirical**
  by W20-1 on the *bundle-only-insufficient-but-outside-resolvable*
  direction — the closed-form ``OutsideWitnessAcquisitionDisambiguator``
  with a registered ``ServiceGraphOracle`` strictly beats every
  non-W20 capsule baseline including W19 by ``+1.000`` on
  R-67-OUTSIDE-RESOLVES across 5/5 seeds. The W19-Λ-outside wall
  remains real wherever the bundle is exhausted of asymmetric
  signal AND no outside source is registered (W19-Λ-outside
  default direction); remains a conjecture on the
  *joint-deception* direction (named W20-Λ-joint-deception:
  primary + secondary + all registered oracles compromised
  jointly). The W19-Λ-total wall remains real unless a registered
  oracle produces an asymmetric reply on the empty-bundle
  ambiguity (W20-1 on R-67-OUTSIDE-NONE-style total absence falls
  through to W19-Λ-total).

## Previous frontier (SDK v3.20, 2026-04-28)

### Active moves (SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator + deceptive-ambiguity benchmark family + W19 family — *first capsule-native move beyond the W18-Λ-deceive wall on the bundle-resolvable case*)

- **Phase-66 deceptive-ambiguity benchmark family.**
  ``vision_mvp.experiments.phase66_deceptive_ambiguity`` ships
  five pre-committed sub-banks (one positive sanity anchor + two
  W19-1 strict-gain anchors + two named falsifiers) plus a cross-
  regime synthetic summary:
  (i) ``r66_corroborated`` — primary names gold; secondary names
  gold. W18 = W19 = 1.000. W19-3 backward-compat anchor.
  (ii) ``r66_deceive_naive_loose`` (``T_decoder = None``) — primary
  names DECOY ONLY; secondary names GOLD ONLY. W18 abstains, ties
  FIFO at 0.000; W19 fires the confound-resolved branch (witness-
  inversion semantics) and projects to gold at 1.000. **+1.000
  strict separation.** The W19-1 anchor.
  (iii) ``r66_deceive_naive_tight`` (``T_decoder = 24``) — same
  shape under decoder-side budget pressure. W19 + W15 composition
  preserves bounded-context efficiency byte-for-byte relative to
  W18; gap +1.000.
  (iv) ``r66_confound_resolvable`` — primary names ALL three;
  secondary names gold. W18 abstains; W19 picks strict-max-witness
  subset = {gold} at 1.000. **+1.000 strict separation.**
  (v) ``r66_deceive_total`` — primary names DECOY ONLY; *no*
  secondary witness. W19-Λ-total falsifier: W19 reduces to W18
  and FAILS at 0.000.
  (vi) ``r66_outside_required`` — primary names DECOY ONLY;
  secondary names ALL three (symmetric witnesses). W19-Λ-outside
  falsifier: W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC``
  and ties FIFO at 0.000.

- **W19 family minted.** W19-Λ-deceive-extension (proved-empirical
  + structural sketch; W18-Λ-deceive extends to R-66-DECEIVE-NAIVE
  for every closed-form scorer that trusts its concatenated
  disambiguator text), W19-1 (proved-conditional + proved-empirical
  n=120 saturated across 5 seeds × 3 regimes; the first capsule-
  native method to resolve bundle-internal contradiction between
  primary and witnesses), W19-2 (proved by inspection +
  mechanically-checked; W19 determinism + closed-form correctness),
  W19-3 (proved-empirical full programme regression; backward-
  compat with R-54..R-65 byte-for-byte), W19-Λ-total / -outside
  (proved-empirical n=8 saturated each; two named structural
  limits), W19-C-LEARNED (conjectural; learned trust scorer for
  free-form witnesses), W19-C-OUTSIDE (conjectural; outside-
  information axis to escape both falsifier walls), W19-Λ-real
  (proved-conditional + empirical-research; real-LLM transfer is
  conditional on closed-vocabulary secondary-witness emission),
  W19-C-CROSS-BENCH (conjectural; transfer to non-incident-triage
  benchmark families with a canonical-role-for-kind mapping).

- **Bundle-contradiction-aware trust-weighted disambiguator** (new
  SDK surface, purely additive). ``vision_mvp/wevra/team_coord.py``
  ships:
  * :class:`BundleContradictionDisambiguator` — the W19 four-stage
    pipeline (inner W18 decode + canonical primary identification +
    asymmetric witness counting + branch-decision projection).
  * :class:`W19TrustResult` — the per-decode audit record carrying
    the projected answer, the W18 fall-through answer, the
    canonical primary's payload, the per-tag W18 scores, the
    per-tag witness counts, and the W19 branch label.
  * :func:`_w19_canonical_primary_index` — closed-form
    deterministic primary identifier with canonical-role-for-kind
    tiebreak + raw-kind tiebreak.
  * :func:`_w19_witness_counts` — closed-form deterministic
    witness counter excluding the canonical primary.
  * :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND` — closed-
    vocabulary canonical-role-for-kind table for the incident-
    triage benchmark family.
  * :data:`W19_SYMMETRIC_NOISE_KINDS` — the round-1 generic-
    noise kinds explicitly excluded from witness counting.
  * :data:`W19_BRANCH_*` — closed-vocabulary branch labels
    (``primary_trusted`` / ``inversion`` / ``confound_resolved``
    / ``abstained_no_signal`` / ``abstained_symmetric`` /
    ``disabled``).

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-66 anchor +
  bar 16 — deceptive-ambiguity bundle-contradiction split + § 2.15
  R-66 ingredients). The SDK v3.20 result clears the **strong
  success bar** § 1.1 on R-66-DECEIVE-NAIVE / R-66-CONFOUND-
  RESOLVABLE (strict gain +1.000 vs every non-W19 capsule
  baseline; bench property held in 8/8; named falsifier regimes
  W19-Λ-total / W19-Λ-outside; W19-3 backward-compat preserved
  byte-for-byte). Headline data files:
  ``docs/data/phase66_cross_regime_synthetic.json``,
  ``docs/data/phase66_seed_sweep_deceive_naive_loose.json``,
  ``docs/data/phase66_seed_sweep_deceive_naive_tight.json``,
  ``docs/data/phase66_seed_sweep_confound_resolvable.json``.

- **Honest scope.** The W19-1 win is *strongly conditional* on
  (a) the symmetric-corroboration round-1 property (so
  W17-Λ-symmetric still applies and only an additional
  information channel can win), AND (b) the bundle carrying at
  least one *independent asymmetric witness* (a specific-tier
  handoff OTHER than the canonical primary disambiguator) whose
  payload mentions a service tag asymmetrically across the
  candidate set. R-66 is a *synthetic* regime; the producer is
  :class:`IdentityExtractor`. Real-LLM transfer (W19-Λ-real) is
  conditional on closed-vocabulary secondary-witness emission;
  free-form natural-language witnesses fall outside the W19
  exact-match closure. The Wevra single-run product runtime
  contract is byte-for-byte unchanged.

### Active observations (SDK v3.20)

- **W18-Λ-deceive holds on R-66-DECEIVE-NAIVE for every pre-W19
  scorer including W18 itself.** W18's selector concatenates
  primary + secondary payloads → its per-tag scorer sees positive
  scores on every admitted tag → strict-asymmetric branch cannot
  fire (N = U) → abstain → fall through to empty inner W15
  answer. The wall is real and structural for every scorer that
  does not distinguish the canonical primary from secondary
  witnesses.

- **W19's confound-resolved branch is the load-bearing branch on
  R-66-DECEIVE-NAIVE and R-66-CONFOUND-RESOLVABLE.** Both regimes
  trigger W18 abstention (full-set hit); W19's witness counter
  (excluding the primary) sees aw(gold) > aw(decoy) and projects
  to gold via the strict-max-aw subset. The inversion branch
  (``W19_BRANCH_INVERSION``) is reserved for regimes where W18
  fires its strict-asymmetric branch but the complement has more
  witnesses than the named set — R-66 does not exercise this
  branch directly because W18 abstains on the full-set hit; the
  inversion branch is structurally available for future regimes.

### Active conjectures (SDK v3.20)

- **W19-C-LEARNED** (free-form witness transfer): a small
  distilled trust scorer outperforms the closed-form witness-
  count rule on free-form natural-language secondary witnesses.
  **Conjectural.**
- **W19-C-OUTSIDE** (outside-information escape from W19-Λ-total
  AND W19-Λ-outside): a scorer with access to an outside-
  information axis (service-graph topology, prior reliability
  scores, cross-incident historical evidence) can detect both
  falsifier walls by cross-reference. **Conjectural.**
- **W19-Λ-real** (real-LLM transfer): partially-discharged when
  the LLM emits closed-vocabulary secondary witnesses from non-
  canonical roles; conjectural otherwise.
- **W19-C-CROSS-BENCH** (cross-bench transfer): the W19 method
  generalises to non-incident-triage families with a canonical-
  role-for-kind mapping. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.20)

- **W18-Λ-deceive** (SDK v3.19): **PARTIALLY DISCHARGED-empirical**
  by W19-1 in the *bundle-resolvable* direction. Closed-form
  bundle-only scorers can escape adversarial round-2 evidence
  AS LONG AS the bundle carries an independent asymmetric
  witness. The bundle-only walls (W19-Λ-total / W19-Λ-outside)
  remain real; escape requires outside information.
- **W18-Λ-confound** (SDK v3.19): **PARTIALLY DISCHARGED-empirical**
  by W19-1 in the *bundle-resolvable* direction on
  R-66-CONFOUND-RESOLVABLE.

---

## Previous frontier (SDK v3.19, 2026-04-28)

### Active moves (SDK v3.19 — bundle-relational compatibility disambiguator + symmetric-ambiguity benchmark family + W18 family — *first capsule-native move beyond the W17-Λ-symmetric wall on a regime where it applies*)

- **Phase-65 relational-compatibility disambiguation under
  symmetric-corroboration benchmark family.**
  ``vision_mvp.experiments.phase65_relational_disambiguation``
  ships four pre-committed sub-banks (one positive-anchor + three
  named falsifiers) plus a cross-regime synthetic summary:
  (i) ``r65_compat_loose`` — synthetic identity producer, R-65-COMPAT
  bench, ``T_decoder = None``. The W18-1 anchor: W18 = 1.000;
  every other capsule strategy = 0.000. **+1.000 strict separation**.
  (ii) ``r65_compat_tight`` — same regime under
  decoder-side budget pressure ``T_decoder = 24``. W18 composes
  cleanly with W15 attention-aware pack; ``tokens_kept_sum`` is
  byte-for-byte identical to W15's. **+1.000 strict separation.**
  (iii) ``r65_no_compat`` — W18-Λ-no-compat falsifier. Round-2
  disambiguator carries no service-tag mention; W18 abstains;
  ties FIFO at 0.000.
  (iv) ``r65_confound`` — W18-Λ-confound falsifier. Round-2
  disambiguator mentions both gold AND decoy; W18 abstains;
  ties FIFO at 0.000.
  (v) ``r65_deceive`` — W18-Λ-deceive falsifier. Round-2
  disambiguator mentions decoy but NOT gold; W18 trusts its
  evidence and picks decoy; fails at 0.000.

- **W18 family minted.** W18-Λ-sym (proved-empirical n=8 saturated
  × 5 seeds + structural sketch; W17-Λ-symmetric extends to R-65-
  COMPAT verbatim for every method pre-W18), W18-1 (proved-
  conditional + proved-empirical n=40 saturated across 5 seeds × 2
  budgets; the first capsule-native method to cross the
  symmetric-corroboration wall), W18-2 (proved by inspection +
  mechanically-checked; W18 determinism + closed-form correctness),
  W18-3 (proved-empirical full programme regression; backward-
  compat with R-54..R-64 byte-for-byte), W18-Λ-no-compat /
  -confound / -deceive (proved-empirical n=8 saturated each;
  three named structural limits), W18-C-LEARNED (conjectural;
  learned scorer for free-form relational mentions),
  W18-C-OUTSIDE (conjectural; outside-information axis to detect
  deceptive mentions), W18-Λ-real (proved-conditional + empirical-
  research; real-LLM transfer is conditional on closed-vocabulary
  relational compounds), W18-C-CROSS-BENCH (conjectural; transfer
  to non-incident-triage benchmark families).

- **Bundle-relational compatibility disambiguator** (new SDK
  surface, purely additive). ``vision_mvp/wevra/team_coord.py``
  ships:
  * :class:`RelationalCompatibilityDisambiguator` — the W18 four-
    stage pipeline (inner W15 decode + disambiguator selection +
    tokenise + score + project).
  * :class:`W18CompatibilityResult` — the per-decode audit record
    carrying the projected answer, the inner answer, the
    disambiguator payload, the per-tag scores, and the abstention
    flag.
  * :func:`_disambiguator_payload_tokens` — closed-form
    deterministic tokeniser (lower-case, split on non-identifier
    chars, compound identifiers preserved).
  * :func:`_relational_compatibility_score` — closed-form
    deterministic scorer with contiguous-subsequence semantics
    for compound targets (handles ``db_query`` matching inside
    ``svc_web_then_svc_db_query``).

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-65 anchor +
  bar 15 — relational-compatibility-disambiguation under
  symmetric-corroboration split + § 2.14 R-65 ingredients). The
  SDK v3.19 result clears the **strong success bar** § 1.1 on
  R-65-COMPAT-LOOSE (strict gain +1.000 vs every non-W18 capsule
  baseline; bench property held in 8/8; named falsifier regimes
  W18-Λ-no-compat / -confound / -deceive AND named bench-shape
  conditionality W18-Λ-sym; W18-3 backward-compat preserved
  byte-for-byte). Headline data files:
  ``docs/data/phase65_cross_regime_synthetic.json``,
  ``docs/data/phase65_seed_sweep_loose.json``,
  ``docs/data/phase65_seed_sweep_tight.json``.

- **Honest scope.** The W18-1 win is *strongly conditional* on
  (a) the symmetric-corroboration bench property (R-65-COMPAT),
  (b) the round-2 disambiguator's payload carrying a relational-
  compound mention of *every* gold service tag AND *no* decoy
  service tag, AND (c) the relational-mention convention being
  closed-vocabulary (the synthetic bench's exact-match closure).
  Real-LLM transfer (W18-Λ-real) is conditional on the LLM
  emitting closed-vocabulary relational compounds. The Wevra
  single-run product runtime contract is byte-for-byte unchanged.

### Active observations (SDK v3.19)

- **W18-Λ-sym holds on R-65-COMPAT.** Every closed-form salience
  scorer in the SDK ties FIFO at ``accuracy_full = 0.000`` on
  R-65-COMPAT (loose AND tight). The bipartite ``(role × tag,
  kind, magnitude)`` multiset is identical for gold and decoy by
  construction; only the round-2 disambiguator's payload-text
  channel breaks the tie.

- **W18-3 partial recovery on R-64-SYM.** Of the 8 R-64-SYM
  scenarios, only the 2 deadlock-flavored ones carry a round-2
  relational mention (``relation=orders_payments_join``). On
  those, W18 recovers gold; on the others, W18 abstains and ties
  FIFO. R-65-COMPAT generalises the relational-mention convention
  to all four scenario families (deadlock / pool / disk /
  slow_query) so the W18-1 strict gain is uniform.

### Active conjectures (SDK v3.19)

- **W18-C-LEARNED** (free-form relational-mention transfer): a
  small distilled bundle-relational scorer outperforms the
  closed-form rule on free-form natural-language relational
  mentions. **Conjectural.**
- **W18-C-OUTSIDE** (outside-information escape from W18-Λ-deceive):
  a scorer with access to an outside-information axis can detect
  the W18-Λ-deceive regime by cross-reference. **Conjectural.**
- **W18-Λ-real** (real-LLM transfer): partially-discharged when
  the LLM emits closed-vocabulary relational compounds; conjectural
  otherwise.
- **W18-C-CROSS-BENCH** (cross-bench transfer): the W18 method
  generalises to non-incident-triage families with a closed-
  vocabulary relational-mention convention. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.19)

- **W17-C-DISAMBIGUATOR** (SDK v3.18; "a learned or LLM-distilled
  semantic disambiguator beyond the closed-form capsule surface
  could distinguish ``orders_payments_join`` (gold A_B in
  deadlock) from a generic decoy whose round-1 mentions are
  observationally identical"). **DISCHARGED-empirical in the
  closed-form direction** by W18-1: a deterministic training-free
  bundle-relational scorer is sufficient on R-65-COMPAT (loose
  AND tight). The *learned* clause remains conjectural under
  W18-C-LEARNED (relevant when free-form mentions fall outside
  the closure).

---

## Previous frontier (SDK v3.18, 2026-04-27)

### Active moves (SDK v3.18 — magnitude-hinted producer protocol + fresh-live end-to-end composition + symmetric-corroboration limit theorem + W17 family — *first fresh-live end-to-end real-LLM strict +1.000 advance + first explicit symmetric-corroboration negative theorem*)

- **Phase-64 fresh-live end-to-end composition + symmetric-
  corroboration wall benchmark.**
  ``vision_mvp.experiments.phase64_live_composition`` ships seven
  pre-committed sub-banks plus a falsifier and a cross-model cell:
  (i) ``r64_baseline`` — synthetic identity producer + magnitude-
  hinted prompt + ``T_decoder = None``. Sanity anchor: every
  cross-round capsule decoder hits 1.000.
  (ii) ``r64_w14h_only`` — synthetic mag-filter producer +
  magnitude-hinted prompt + ``T_decoder = None``. Synthetic
  counterpart of the W17-1 anchor. Bench property holds in 8/8;
  cross-round decoders hit 1.000 (W17-3 backward-compat).
  (iii) ``r64_live_struct`` — fresh live ``qwen2.5:14b-32k`` +
  legacy *structured* prompt + ``T_decoder = 14``. Reproduces the
  W14-Λ-real envelope on the fresh probe: bench property holds in
  7/8; ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.500 strict gain**.
  W17-Λ-no-hint structural anchor.
  (iv) ``r64_live_maghint`` — **the W17-1 anchor.** Same fresh
  live probe + magnitude-hinted prompt + ``T_decoder = 14``.
  Bench property holds in 8/8 (closing the 1/8 model-side miss);
  ``capsule_attention_aware = 1.000``;
  ``capsule_layered_fifo_packed = 0.000``;
  ``capsule_fifo = 0.000``; **+1.000 strict separation** on both
  axes. The first programme result that beats the strongest
  non-composed baseline by ≥ 1.0 on a fresh live LLM probe.
  (v) ``r64_live_naive`` — same fresh probe + naive prompt +
  ``T_decoder = 14``. Live counterpart of W14-Λ-prompt +
  W15-Λ-budget joint failure. Bench property holds in 0/8; every
  capsule strategy ties FIFO at 0.000. W17-Λ-naive falsifier.
  (vi) ``r64_live_xmodel`` — fresh live ``qwen3.5:35b`` MoE
  backend + magnitude-hinted prompt + ``T_decoder = 14``,
  ``think = False``. Cross-model probe (W17-C-XMODEL). Bench
  property holds in 8/8; ``capsule_attention_aware = 0.750``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.750 strict gain**
  (well above the 0.50 strong-bar threshold). The 0.250 gap to
  1.000 is on ``accuracy_root_cause`` — a 35B-specific specific-
  kind judgment artifact, not a producer-protocol failure.
  Proved-conditional + empirical-research.
  (vii) ``r64_sym_loose`` and ``r64_sym_tight`` — synthetic
  symmetric-corroboration bank (every service mentioned by
  exactly 2 distinct routed producer roles via generic-noise
  kinds; round-2 disambiguator names gold root_cause without
  ``service=`` token); under both ``T_decoder ∈ {None, 24}``,
  every capsule strategy in the SDK ties FIFO at 0.000 — the
  **first explicit symmetric-corroboration limit theorem in the
  programme** (W17-Λ-symmetric).

- **W17 family minted.** W17-1 (proved-conditional +
  empirical-research; the first fresh-live end-to-end +1.000
  strict gain), W17-Λ-no-hint (empirical-research; live
  legacy-structured-prompt envelope), W17-Λ-naive (empirical-
  research; live joint-failure falsifier), **W17-Λ-symmetric**
  (proved-empirical + structural sketch; first explicit
  symmetric-corroboration limit theorem; *discharges*
  W15-C-SYMMETRIC / W16-C-SYMMETRIC as a negative theorem),
  W17-2 (proved + mechanically-checked; magnitude-hinted prompt
  determinism + threshold table soundness), W17-3 (proved-
  empirical full programme regression; the W17 surface reduces
  to the SDK v3.15 W14 anchor byte-for-byte under default
  parameters; 442/442 prior tests pass), **W17-C-XMODEL**
  (proved-conditional + empirical-research; fresh live 35B
  cross-model strict gain). The W17-C family (W17-C-DISAMBIGUATOR,
  W17-C-LEARNED-HINT, W17-C-CROSS-BENCH) makes the next research
  frontier explicit.

- **Magnitude-hinted producer protocol** (new SDK surface, purely
  additive). ``vision_mvp/wevra/team_coord.py`` ships:
  * ``PRODUCER_PROMPT_MAGNITUDE_HINTED`` — third producer-prompt
    mode; the W17-1 anchor.
  * :class:`OperationalThreshold` — closed-vocabulary record
    naming a kind, the qualifying field, the inclusive
    threshold, the unit, and a human gloss.
  * ``RoleExtractionSchema.magnitude_thresholds`` — additive
    optional field on the W14 schema; empty by default (W17-3
    byte-for-byte backward-compat); populated by
    ``incident_triage_role_schemas(magnitude_hinted=True)``.
  * :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` and
    :func:`incident_triage_magnitude_thresholds` — the
    pre-committed threshold table for the incident-triage family
    (calibrated to the synthetic
    :class:`MagnitudeFilteringExtractor`'s default thresholds,
    NOT to any specific scenario's magnitudes).
  * :func:`_render_magnitude_hinted_prompt` — the W17 prompt
    renderer. Adds the operational threshold table AND an
    anti-relative-magnitude clause to the structured prompt.

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-64 anchor +
  bar 14 — live-end-to-end + magnitude-hinted-protocol +
  symmetric-corroboration-wall split + § 2.13 R-64 ingredients).
  The SDK v3.18 result clears the **strong success bar** § 1.1 on
  R-64-LIVE-MAGHINT (strict gain +1.000 vs both substrate FIFO
  AND FIFO-packed-W14H-only on a fresh live LLM probe; bench
  property held in 8/8 closes the prior 1/8 miss; named bench
  property + named falsifier regimes W17-Λ-no-hint /
  W17-Λ-naive / W17-Λ-symmetric AND named cross-model probe
  W17-C-XMODEL with +0.750 gain; W17-3 backward-compat
  preserved byte-for-byte). Headline data files:
  ``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``,
  ``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``,
  ``docs/data/phase64_cross_regime_synthetic.json``.

- **Honest scope.** The W17-1 win is *strongly conditional* on
  (a) the asymmetric-corroboration bench property, (b) the
  magnitude-hint table being calibrated to the synthetic
  extractor's threshold values (operational definitions, not
  answer hints — both gold and decoy magnitudes are well above
  every threshold), AND (c) the live endpoint reachable.
  W17-Λ-symmetric *names the wall* when the asymmetric ingredient
  is structurally absent: every capsule strategy in the SDK ties
  FIFO at 0.000 by construction, including the W14H + W15
  composition. The cross-model probe (W17-C-XMODEL) is
  proved-conditional on bench-property + strict-gain transfer; the
  saturated full-correctness clause remains conjectural and is
  bounded by a 35B-specific specific-tier judgment artifact (not
  a producer-protocol failure). The Wevra single-run product
  runtime contract is byte-for-byte unchanged.

### Active observations (SDK v3.18)

- **W17-Λ-real (honest positive on fresh live axis).** Real
  Mac-1 ``qwen2.5:14b-32k`` at ``temperature = 0`` on the
  Phase-61 redesigned comparable-magnitude events:
  - Under naive prompt: bench property holds 0/8; every method
    ties FIFO at 0.000 (live confirmation of W14-Λ-prompt +
    W15-Λ-budget joint failure on the live axis).
  - Under structured prompt: bench property holds 7/8;
    cross-round decoders all achieve ``accuracy_full = 0.500``;
    ``capsule_attention_aware - capsule_layered_fifo_packed
    = +0.500`` strict gain (W17-Λ-no-hint anchor — reproduces
    W14-Λ-real / W16-Λ-real-replay envelope on fresh probe).
  - Under magnitude-hinted prompt: bench property holds 8/8;
    cross-round decoders all achieve ``accuracy_full = 1.000``;
    ``capsule_attention_aware - capsule_layered_fifo_packed
    = +1.000`` strict gain (W17-1 anchor).
  - The 1/8 model-error failure on the slow_query_archival
    scenario (LLM judged decoy magnitudes "not severe enough"
    relative to gold) is **closed** by the W17 magnitude-hint
    extension; the LLM correctly emits every qualifying event
    once the absolute thresholds are stated and relative-
    magnitude skipping is forbidden in the prompt.

- **W17-C-XMODEL-real (proved-conditional cross-model
  bench-property transfer).** Real Mac-1 ``qwen3.5:35b`` MoE
  backend at ``temperature = 0`` with ``think = False``: bench
  property holds in 8/8 under the magnitude-hinted prompt (the
  bench-property hold-rate transfers to a 2.4× larger MoE
  model class byte-for-byte); ``capsule_attention_aware =
  0.750``; ``capsule_layered_fifo_packed = 0.000``; +0.750
  strict gain. The 0.250 gap to 1.000 is on
  ``accuracy_root_cause`` — the 35B emits a different specific-
  tier kind on one scenario than the 14B does (a model-side
  specific-kind judgment artifact, not a producer-protocol
  failure). The W5-1 (SDK v3.6) prior cross-model probe found
  parser-axis TVD = 1.000 between 14B and 35B under strict
  parsing; W17-C-XMODEL finds *bench-property-axis* TVD = 0.000
  under the magnitude-hinted prompt + robust parser — the W17
  protocol *normalises* the producer-side behaviour across the
  model-class jump that the W5-1 parser-boundary regime
  exposed.

### Active conjectures (SDK v3.18)

- **W17-C-DISAMBIGUATOR** (semantic-disambiguator beyond
  W17-Λ-symmetric): a learned or LLM-distilled disambiguator
  consuming the round-2 disambiguator's evidence text directly
  could distinguish ``orders_payments_join`` (gold A_B in
  deadlock) from a generic decoy whose round-1 mentions are
  observationally identical. **Conjectural; the named research
  frontier for SDK v3.19+.**

- **W17-C-LEARNED-HINT** (learned threshold variant): a per-kind
  data-driven threshold table outperforms the closed-form
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` on a
  held-out test set across 5/5 random splits. **Conjectural.**

- **W17-C-CROSS-BENCH** (cross-bench transfer): the magnitude-
  hint protocol generalises to non-incident-triage benchmark
  families when the family admits a closed-vocabulary kind →
  qualifying-threshold mapping. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.18)

- **W16-C-LIVE-OLLAMA** (SDK v3.17; "a fresh live Ollama probe
  closes the 1/8 model-error failure W14-only leaves").
  **DISCHARGED-empirical** by W17-1: a fresh live probe AT
  THE MAGNITUDE-HINTED PROMPT closes the 1/8 miss entirely
  (bench property holds in 8/8;
  ``accuracy_full = 1.000``); the W17-Λ-no-hint anchor on the
  same fresh probe under the legacy structured prompt
  reproduces the recorded 7/8 + 0.500 envelope, confirming the
  magnitude-hint extension (W17-1) is the load-bearing
  improvement, not a re-run of the same prompt.
- **W16-C-CROSS-MODEL** (SDK v3.17; "the composition's W16-1
  win at qwen2.5:14b-32k transfers to qwen3.5:35b under
  MLX-distributed inference"). **PARTIALLY DISCHARGED-
  empirical** by W17-C-XMODEL: the bench-property + strict-
  gain transfer is observed on the 35B Ollama backend (NOT
  MLX-distributed); the saturated full-correctness clause is
  still bounded by a 35B-specific judgment artifact; the
  MLX-distributed clause remains orthogonal and conjectural
  pending Mac-2 reachable.
- **W15-C-SYMMETRIC** / **W16-C-SYMMETRIC** (SDK v3.16 / v3.17;
  "symmetric-corroboration multi-hypothesis structural limit").
  **DISCHARGED as a negative theorem** by W17-Λ-symmetric: the
  R-64-SYM bench construction proves every capsule strategy in
  the SDK ties FIFO at 0.000 under both ``T_decoder ∈ {None,
  24}`` by construction. The wall is real and structural; the
  named research move beyond it is W17-C-DISAMBIGUATOR
  (conjectural).

## Previous frontier (SDK v3.17, 2026-04-27)

### Active moves (SDK v3.17 — end-to-end W14 + W15 composition + W16 family — *first end-to-end real-LLM strict advance over the strongest non-composed baseline*)

(See `docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md` for the SDK
v3.17 milestone note. SDK v3.18 supersedes this frontier on the
fresh-live and symmetric-wall axes; the W16 anchors are
preserved as historical evidence.)

## Previous frontier (SDK v3.16, 2026-04-27)

### Active moves (SDK v3.16 — attention-aware capsule context packing + W15 family — *first decoder-side context-packing move + joint-correctness-and-context-efficiency strict-gain anchor*)

- **Phase-62 attention-aware capsule context packing benchmark.**
  ``vision_mvp.experiments.phase62_attention_aware_packing`` ships
  three pre-committed sub-banks:
  (i) ``T_decoder=None`` — the R-62-default sanity anchor; every
  cross-round decoder (W11/W12/W13/W15) saturates at 1.000. The W15
  decoder ties W13 byte-for-byte on the answer field (W15-3
  backward-compat).
  (ii) ``T_decoder=24`` — the R-62-tightbudget W15-1 anchor; under
  FIFO packing the round-2 specific-tier disambiguator falls past
  the budget and ``capsule_layered_fifo_packed`` ties FIFO at
  ``accuracy_full = 0.000``. The W15
  :class:`AttentionAwareBundleDecoder` salience-packs with hypothesis
  preservation and achieves ``accuracy_full = 1.000``; +1.000 strict
  separation, stable across 5/5 alternate ``bank_seed`` values.
  (iii) ``T_decoder=9999`` — the R-62-saturation W15-Λ-budget
  falsifier; under no decoder-side budget pressure the salience
  reordering is a no-op on the answer field — both W15 and FIFO-pack
  tie at the W13 answer (1.000). The W15-1 win is *conditional* on
  budget pressure; this regime makes the conditionality sharp.
- **``AttentionAwareBundleDecoder`` + ``CapsuleContextPacker`` +
  ``FifoContextPacker`` (new).**
  ``vision_mvp/wevra/team_coord.py``. A two-stage decoder: (1)
  first-pass priority decode over the normalised union elects a
  tentative ``root_cause``; (2) salience-aware repack reorders +
  truncates the union under ``T_decoder`` (using closed-form weights
  on tier + CCK + corroboration + magnitude + round index, with
  ``preserve_hypotheses=True`` defaulting on) before final W13
  layered decode. Pack-stats expose ``position_of_first_causal_claim``
  (the proxy attention metric), ``tokens_kept_sum`` /
  ``tokens_input_sum``, ``hypothesis_count_kept``, and
  ``n_dropped_budget`` for direct audit. ``FifoContextPacker`` is the
  load-bearing baseline (FIFO truncation under the same
  ``T_decoder``). Re-exported via ``__all__``.
- **Theorem family W15.** W15-Λ-budget (decoder-side budget
  structural limit on R-62-tightbudget under FIFO packing,
  proved-empirical n=40 saturated × 5 seeds + structural sketch via
  W7-3 extension to the decoder-side axis),
  W15-1 (AttentionAwareBundleDecoder sufficiency under bounded
  ``T_decoder`` with hypothesis preservation, proved-conditional +
  proved-empirical synthetic n=40 saturated × 5 seeds, +1.000 vs
  fifo_packed_layered), W15-2 (pack determinism + closed-form
  salience, proved by inspection + mechanically-checked), W15-3
  (backward-compat with R-54..R-61 default banks, proved-empirical
  full programme-wide regression 393/393 + 37 new tests = 430/430),
  W15-Λ-degenerate (saturation falsifier on R-62-saturation,
  proved-empirical n=8: under no decoder-side budget pressure the
  W15-1 win is structurally invisible by construction), W15-4
  (token-efficiency floor: ``tokens_kept ≤ T_decoder`` strict, proved
  by inspection + mechanically-checked). The W15-C family (W15-C-real,
  W15-C1, W15-C-LEARNED, W15-C-SYMMETRIC, W15-C-COMPOSE-W14) makes
  real-LLM-downstream-decoder, cross-bench, learned-salience,
  symmetric-corroboration, and W14+W15 compose extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-62 anchor +
  bar 12 — joint-correctness-and-context-efficiency split + § 2.11
  R-62 ingredients). The SDK v3.16 result clears the **strong
  success bar** § 1.1 on R-62-tightbudget synthetic (strict gain
  +1.000 vs FIFO-packed-W13, stable across 5/5 (bank_seed) values,
  no regression on R-53..R-61, audit T-1..T-7 preserved on every
  cell, named bench property + named falsifier regime
  W15-Λ-degenerate, AND joint-correctness-and-context-efficiency
  split bar 12 satisfied — the new method includes a load-bearing
  decoder-side context-packing intervention beyond every prior
  layer). Headline data file:
  ``docs/data/phase62_seed_sweep_tightbudget_K12_n8.json``.
- **Honest scope.** R-62 is a *synthetic* milestone — the producer
  is :class:`IdentityExtractor`, not a real LLM. Real-LLM transfer
  of W15 is W15-C-real, conjectural; it requires Mac 1 / Mac 2 to
  be online and the bundle to be re-decoded by an LLM agent under a
  real context window. SDK v3.16 does not run this probe.
  "Attention-aware" uses an *honest proxy* — the
  ``position_of_first_causal_claim`` metric — not transformer
  attention manipulation. The W15-1 win is *conditional* on (a) the
  bench property holding, (b) ``T_decoder`` below the union token
  sum, AND (c) round-2 carrying a specific-tier disambiguator with
  no ``service=`` token; W15-Λ-degenerate makes the conditionality
  sharp. The Wevra single-run product runtime contract is byte-for-
  byte unchanged.

### Prior moves (SDK v3.15 — structured producer protocol + W14 family — *first producer-protocol move + first real-LLM strict gain ≥ 0.50 over substrate FIFO*)

- **Phase-61 producer-side ambiguity-preservation benchmark.**
  ``vision_mvp.experiments.phase61_producer_ambiguity_preservation``
  ships three pre-committed sub-banks plus an opt-in real-Ollama
  extension:
  (i) ``identity`` — the R-61-default sanity anchor; the synthetic
  extractor passes every emission through verbatim. Bench property
  holds in 8/8; W11/W12/W13 cross-round decoders all hit 1.000.
  (ii) ``magnitude_filter`` + ``naive`` prompt — the W14-Λ-prompt
  anchor; the magnitude-filter extractor calibrated against the
  W13-Λ-real real-Ollama observation collapses round-1 by top-N-
  per-(role, kind) by magnitude. On the redesigned comparable-
  magnitude events the threshold filter does not fire (the events
  are designed to clear it) but the prompt-induced compression
  drops monitor's decoy-side mentions; the bench property holds in
  0/8 and every capsule strategy ties FIFO at 0.000.
  (iii) ``magnitude_filter`` + ``structured`` prompt — the W14-1
  anchor; the structured prompt's per-event mandate disables the
  compression. Bench property holds in 8/8; the cross-round capsule
  pipeline achieves ``accuracy_full = 1.000``; +1.000 strict
  separation vs the naive-prompt counterpart, stable across 5/5
  alternate ``bank_seed`` values.
  (iv) ``ollama`` (opt-in) — the R-61-ollama probe; talks Mac-1
  ``qwen2.5:14b-32k`` with the structured prompt; raw producer
  responses captured into the report. **R-61-OLLAMA-A tier** (the
  strong-real-transfer tier of `docs/SUCCESS_CRITERION_MULTI_AGENT_
  CONTEXT.md` § 1.5): bench property holds in 7/8 scenarios;
  cross-round decoders achieve ``accuracy_full = 0.500``;
  ``layered − fifo = +0.500`` at exactly the 0.50 threshold; audit
  T-1..T-7 preserved on every cell. Anchor:
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``.
- **``StructuredProducerProtocol`` + ``RoleExtractionSchema`` (new).**
  ``vision_mvp/wevra/team_coord.py``. A prompt-rendering surface
  with two modes: ``naive`` (legacy Phase-58/59/60 byte-for-byte)
  and ``structured`` (W14: tier banner observation/diagnosis split +
  per-tier kind whitelist + per-event mandate). The protocol
  consumes a deterministic ``RoleExtractionSchema`` (allowed kinds
  + observation/diagnosis partition); ``incident_triage_role_
  schemas()`` ships the schema table for the Phase-58..Phase-61
  bench family. A new structured-aware response parser
  (``_parse_structured_response``) dedupes by ``(kind, payload)``
  rather than ``kind`` alone so the per-event mandate survives
  parsing. Re-exported via ``__all__``.
- **Theorem family W14.** W14-Λ-prompt (producer-side ambiguity-
  erasure structural limit on R-61-naive-prompt, proved-empirical
  n=40 saturated × 5 seeds + structural sketch via W7-3 extension),
  W14-1 (StructuredProducerProtocol sufficiency under bounded
  producer compression, proved-conditional + proved-empirical
  synthetic n=40 + real Ollama n=8), W14-2 (schema soundness +
  protocol determinism, proved by inspection + mechanically-
  checked), W14-3 (backward-compat with R-54..R-60, proved-empirical
  full programme-wide regression 393/393), W14-4 (combined-
  intervention falsifier on R-61-ollama-naive, proved-empirical
  n=8), W14-Λ-real (real Ollama 14B prompt-protocol transfer,
  empirical-research n=8 × 24 producer calls). The W14-C family
  (W14-C1..W14-C5) makes cross-bench, model-side calibration,
  multi-round generalisation, cross-model transfer, and multi-
  hypothesis variant extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-61 anchor +
  bar 11 — producer-side ambiguity-preservation split + § 1.5
  R-61-ollama 4-tier grading). The SDK v3.15 result clears the
  **strong success bar** § 1.1 on R-61-structured-prompt synthetic
  (strict gain ≥ 0.20 vs every prior anchor including SDK v3.14
  W13 alone, stable across ≥ 3 (bank_seed) values, no regression on
  R-53..R-60, audit T-1..T-7 preserved on every cell, named bench
  property + named falsifier regime W14-4, AND producer-side
  ambiguity-preservation split bar 11 satisfied — the new method
  includes a load-bearing prompt-side intervention beyond every
  prior downstream layer). The R-61-ollama probe lands at the
  **R-61-OLLAMA-A tier** (the strongest real-transfer tier): the
  milestone is therefore a *strong-success on R-61 synthetic + a
  strong-success / OLLAMA-A on R-61-ollama*, the **first strong-
  success on a real-LLM stream in the programme** (R-60-OLLAMA-C
  was honest null; R-61-OLLAMA-A is +0.500 strict gain at the
  threshold).
- **Honest scope.** The W14-1 win is *conditional* on (a) the
  redesigned comparable-magnitude events (R-61 closure property),
  (b) the structured producer protocol being in effect, AND (c) the
  cross-round capsule pipeline (admission + multi-round decoder).
  The W14-4 falsifier regime (real Ollama + comparable-magnitude
  events + naive prompt) is the explicit counterexample showing
  *both* interventions are required. The W13 closure-widening is
  structurally invisible on R-61-ollama because the real LLM emits
  canonical kinds (zero drift); on the structured prompt the
  load-bearing layer is W14, not W13. The W13 layer is dormant on
  this regime, not refuted.

### Active observations (SDK v3.15)

- **W14-Λ-real (honest positive).** Real Mac-1
  ``qwen2.5:14b-32k`` at ``temperature=0`` on the Phase-61
  redesigned comparable-magnitude events:
  - Under naive prompt: bench property holds 0/8; every method
    ties FIFO at 0.000 (empirical confirmation of W14-Λ-prompt on
    real LLM).
  - Under structured prompt: bench property holds 7/8; W11/W12/W13
    cross-round decoders all achieve ``accuracy_full = 0.500``;
    +0.500 strict gain vs FIFO at the R-61-OLLAMA-A threshold.
  - The 1/8 model-error failure is on the LLM (slow_query
    scenario: LLM judged ``error_rate=0.15`` not to qualify as
    ``ERROR_RATE_SPIKE``), not on the protocol — the structured
    prompt + comparable-magnitude events restore the bench
    property in every other scenario. The W14-C2 conjecture
    (magnitude-hinted prompt extension) is the natural next move.

### Active conjectures (SDK v3.15)

- **W14-C1**: cross-bench transfer of the W14 protocol to non-
  incident-triage benchmark families. Conjectural.
- **W14-C2**: model-side magnitude calibration via a *magnitude
  hint* extension to the structured prompt. Conjectural; the W14-
  Λ-real 7/8 anchor is the candidate falsifier.
- **W14-C3**: multi-round generalisation to N ≥ 3 rounds with a
  graded tier hierarchy. Conjectural.
- **W14-C4**: cross-model transfer to qwen3.5:35b-MoE and to non-
  Ollama backends (MLX-distributed). Conjectural; requires Mac 2
  reachable.
- **W14-C5**: multi-hypothesis variant of the protocol that permits
  2-3 candidate kinds per event. Conjectural.

### Discharged / partially-discharged conjectures (SDK v3.15)

- **W13-C3** (SDK v3.14; real-Ollama transfer of W13-1 with a
  redesigned event stream + a prompt that instructs the LLM to
  emit one claim per distinct event). **PARTIALLY DISCHARGED-
  empirical** by W14-1 + W14-Λ-real: the *combination* of
  comparable-magnitude events + structured producer protocol
  clears the strict-gain bar on real Ollama 14B (n=8, +0.500 vs
  FIFO at exactly the R-61-OLLAMA-A threshold), BUT the gain is on
  the **W11/W12/W13 cross-round pipeline as a whole** — not
  specifically on the W13 closure-widening, which is structurally
  invisible because the real LLM still emits canonical kinds. The
  honest revised reading: prompt-side discipline + comparable-
  magnitude events restore the bench property; W13's *normalisation*
  contribution remains dormant on this regime. The cross-round
  structural bar is cleared at +0.500 by the combined intervention
  (W14).

## Previous frontier (SDK v3.14, 2026-04-26)

### Active moves (SDK v3.14 — layered open-world normalisation + real-Ollama probe + W13 family — *first open-world normalisation move and first real-Ollama transfer measurement*)

- **Phase-60 open-world normalisation + real-Ollama benchmark.**
  ``vision_mvp.experiments.phase60_open_world_normalization``
  ships three pre-committed sub-banks plus an opt-in real-Ollama
  extension:
  (i) ``synthetic_wide_oov_llm`` — the W13-1 anchor; the producer
  emits drifted variants from
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` (every entry verified
  *outside* :data:`CLAIM_KIND_SYNONYMS` and *inside* the W13
  heuristic closure); pre-committed default
  ``K_auditor=8, T_auditor=256, n_eval=12, bank_seed=11,
  wide_oov_prob=0.50, svc_token_alt_prob=0.30``. Headline: ``capsule_layered_multi_round`` achieves
  ``accuracy_full = 1.000`` while ``capsule_robust_multi_round``
  (W12) **and** every other strategy ties FIFO at 0.000 — the
  *first strict separation between layered (open-world) and
  fixed-vocabulary normalisation* in the programme; +1.000 vs every
  un-widened method including SDK v3.13 W12; stable across **5/5**
  alternate (bank_seed, llm_seed) values with min gap layered − w12
  = +0.917 (well above the 0.50 stability threshold).
  (ii) ``synthetic_cosmic_oov_llm`` — the W13-4 closure-boundary
  anchor; round-2 specific-tier kinds are replaced by entries from
  :data:`COSMIC_OOV_KINDS` (every entry verified outside both
  layers); ``capsule_layered_multi_round`` ties FIFO at 0.000 on
  8/8 falsifier scenarios.
  (iii) ``synthetic_clean_llm`` + cross-regime audit — the W13-3
  backward-compat anchor; on R-58 / R-59 / R-60-clean / R-54..R-57,
  the W13 decoder produces byte-for-byte the same answer as W11 /
  W12 (heuristic layer fires zero times); R-54..R-58 / R-59-noisy
  anchors all preserved at ``accuracy_full = 1.000``.
  (iv) ``ollama`` — the R-60-ollama probe; talks Mac-1
  ``qwen2.5:14b-32k`` on the same Phase-58 events; raw producer
  responses captured into the report. **Honest negative finding
  (W13-Λ-real, see § *Active observations* below)**: real Ollama 14B
  emits canonical kinds and filters low-magnitude decoy events;
  the bench property holds in 0/4 scenarios; W13 has nothing to
  rescue and ties W12.
- **``LayeredClaimNormalizer`` + ``LayeredRobustMultiRoundBundleDecoder``
  (new).** ``vision_mvp/wevra/team_coord.py``. Two-layer normaliser:
  (a) exact :data:`CLAIM_KIND_SYNONYMS` lookup (the W12 path); (b)
  ordered :data:`_HEURISTIC_KIND_RULES` regex-predicate abstraction
  rules whose union strictly widens the W12 closure; (c) optional
  abstention via the :data:`LAYERED_NORMALIZER_ABSTAIN` sentinel.
  Per-call counters expose load-bearing layer breakdowns
  (``n_exact``, ``n_heuristic``, ``n_abstained``, ``n_passthrough``,
  ``rule_hits``). Re-exported as ``LayeredClaimNormalizer``,
  ``LayeredRobustMultiRoundBundleDecoder``,
  ``HeuristicAbstractionRule``, ``LAYERED_NORMALIZER_ABSTAIN``.
- **Theorem family W13.** W13-Λ-fixed (fixed-vocabulary closure
  limit on R-60-wide, proved-empirical n=12 + structural sketch),
  W13-1 (LayeredRobustMultiRoundBundleDecoder sufficiency under
  bounded OOV in the heuristic closure, proved-conditional + proved-
  empirical n=60 saturated across 5 seeds), W13-2 (heuristic
  abstraction soundness, proved by inspection + mechanically-
  checked), W13-3 (backward-compat with R-54..R-58 + R-59 + R-60-
  clean, proved-empirical n=8 each + cross-regime audit), W13-4
  (cosmic-OOV closure boundary, proved-empirical n=8 saturated),
  W13-Λ-real (real Ollama 14B canonical-kind + magnitude-filtering
  observation, empirical-research n=4 producer-side observation +
  12 real Ollama calls). The W13-C family (W13-C1..W13-C4) makes
  cross-bench, learned-normaliser, real-Ollama-with-redesigned-
  events, and abstention-aware-decoder extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-60 anchor +
  bar 10 — open-world normalisation split + § 1.4 R-60-ollama
  4-tier grading). The SDK v3.14 result clears the **strong success
  bar** § 1.1 on R-60-wide (strict gain ≥ 0.20 vs every fixed-
  vocabulary method including SDK v3.13 W12, stable across ≥ 3
  (bank_seed, llm_seed) values, no regression on
  R-53..R-59 / R-60-clean, audit T-1..T-7 preserved on every cell,
  named bench property + named falsifier regime W13-4, AND open-
  world normalisation split bar 10 satisfied — the new method
  includes a load-bearing heuristic abstraction layer beyond the
  exact synonym table). The R-60-ollama probe lands at the
  **R-60-OLLAMA-C tier** (honest null real transfer): the milestone
  is therefore a *strong-success on R-60-wide synthetic + a
  partial-success / honest-null on R-60-ollama*, NOT a strong
  cross-axis advance. § 1.4 of the success criterion makes this
  partition pre-committed and falsifiable.
- **Honest scope.** The W13-1 win is *conditional* on (a) the named
  bench property (R-58 delayed-causal-evidence shape with the
  Phase-60 wide-OOV drift channel), (b) the producer-noise channel
  being bounded by the *heuristic* closure (every variant in
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` matches at least one
  pattern in :data:`_HEURISTIC_KIND_RULES`), AND (c) round-N
  admission not being budget-starved (inherits W11-4). The W13-4
  falsifier regime is the explicit counterexample; the W13-Λ-real
  observation is a *separate, honest, partial* outcome. The W13
  method is research-grade SDK code, additive on top of W12.

### Active observations (SDK v3.14)

- **W13-Λ-real (honest negative).** Real Ollama 14B
  (qwen2.5:14b-32k on Mac 1, ``temperature=0``) on the calibrated
  Phase-58 incident-triage prompt does NOT generate the R-58
  delayed-causal-evidence bench property: across 4 scenarios × 12
  producer calls, the LLM emits 0 drifted kinds (every claim_kind
  is canonical) and filters low-magnitude decoy events as noise
  (the ``monitor`` role emits ``NONE`` for the deliberately-low-
  magnitude decoy events, breaking the cross-role decoy
  corroboration assumption). The bench property holds in 0/4
  scenarios; normalisation has nothing to rescue; W13 ties W12 ties
  multi_round at ``accuracy_full = 0.250``. The R-60-ollama probe
  is therefore a *measurement*, not a *claim*: the synthetic→real-
  LLM transfer story has **five layers** —
  (i) un-normalised admission cannot transfer (W6-C2 falsified),
  (ii) un-normalised cross-round decoding cannot transfer
  (W12-Λ at the real-LLM axis),
  (iii) fixed-vocabulary normalisation transfers under bounded
  *synthetic* drift (W12-1, conditional),
  (iv) heuristic-widened normalisation transfers under bounded
  *open-world* drift inside the heuristic closure (W13-1,
  conditional),
  (v) real Ollama 14B at default settings does not produce the
  drift OR the cross-role decoy corroboration shape (W13-Λ-real,
  empirical observation; the gating axis on real Ollama is *event-
  shape design + prompt-side discipline*, not normalisation).
  Future work: redesign the events so the decoy has comparable
  magnitudes to gold (W13-C3) — and accept that the contribution
  shifts from "the normaliser" to "the prompt + event design".

### Active conjectures (SDK v3.14)

- **W13-C1**: cross-bench transfer of the W13 closure-widening
  contract to non-incident-triage benchmark families.
  Conjectural; falsifier = a benchmark family where any size-
  bounded predicate set covers < 50 % of LLM kind drift.
- **W13-C2**: a learned normaliser strictly widens the W13
  heuristic closure on R-60-cosmic. Conjectural; restated as a
  closure-widening move, not a structural fix.
- **W13-C3**: real-Ollama transfer of W13-1 with redesigned
  events. Conjectural; Phase-60 v2 candidate.
- **W13-C4**: abstention as a load-bearing signal — an
  abstention-aware decoder strictly improves over a passthrough
  decoder. Conjectural; the abstention sentinel is implemented but
  the abstention-aware decoder is not yet wired.

### Discharged / partially-discharged conjectures (SDK v3.14)

- **W12-C2** (SDK v3.13; real-Ollama transfer of W12-1).
  **PARTIALLY DISCHARGED-empirical** (negatively): real Ollama 14B
  on the Phase-58 events does NOT emit drift, so the W12 advance
  is *structurally invisible* on R-60-ollama (W13-Λ-real). The
  W12-C2 question reframes as: under what (event design × prompt)
  does a real LLM emit non-trivial bounded drift? — that is W13-C3.
- **W12-C3** (SDK v3.13; learned synonym table widens the W12
  closure). **PARTIALLY DISCHARGED-empirical** by W13-1: a
  *heuristic* layered normaliser strictly widens the closure
  beyond the fixed table on R-60-wide. The *learned* variant
  remains conjectural (W13-C2). The W13-4 closure boundary applies
  to any predicate-based normaliser, learned or not — a learned
  normaliser only widens, does not eliminate, the closure.

## Previous frontier (SDK v3.13, 2026-04-26)

### Active moves (SDK v3.13 — real-LLM-robust multi-round bundle decoder + W12 family — *first synthetic→real-LLM transfer move*)

- **Phase-59 real-LLM-driven multi-round delayed-disambiguation
  benchmark.**
  ``vision_mvp.experiments.phase59_real_llm_multi_round`` runs 10
  strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9),
  capsule_bundle_decoder (W10 single-round), capsule_multi_round
  (W11), capsule_robust_multi_round (W12)) on the same Phase-58
  delayed-causal-evidence shape under an **LLM-shaped extractor**:
  default mode ``synthetic_noisy_llm`` injects realistic
  claim-kind drift (``synonym_prob=0.50``) and payload drift
  (``svc_token_alt_prob=0.30``) calibrated against Phase-53
  empirical 14B / 35B parser_role_response distributions; the
  optional ``ollama`` mode talks the same Phase-53-style prompt
  to a real LLM endpoint with synthetic fallback on HTTP failure.
  12/12 default scenarios satisfy the post-normalisation
  delayed-causal-evidence property; mechanically verified by
  ``Phase59BenchPropertyTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=12``,
  ``bank_seed=11``, ``bank_replicates=3``,
  ``llm_synonym_prob=0.50``, ``llm_svc_alt_prob=0.30``,
  ``llm_seed=11``. Headline: ``capsule_robust_multi_round``
  achieves ``accuracy_full = 1.000`` while substrate, FIFO,
  priority, coverage, W7-2, W8, W9, W10 single-round bundle, AND
  **W11 un-normalised** all produce ``accuracy_full = 0.000`` —
  the **first strict separation between un-normalised and
  normalised cross-round capsule-native coordination on a
  real-LLM-shaped stream** in the programme, **+1.000** vs every
  other method, stable across **5/5** alternate (bank_seed,
  llm_seed) values. The W12 family (W12-Λ / W12-1 / W12-2 /
  W12-3 / W12-4 — proved or proved-empirical) anchors the
  milestone formally; the W12-C family (W12-C1/C2/C3) makes the
  cross-bench / real-Ollama / learned-normaliser extensions
  falsifiable.
- **``RobustMultiRoundBundleDecoder`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Wraps
  :class:`MultiRoundBundleDecoder` with a closed-vocabulary
  normalisation layer: :func:`normalize_claim_kind` rewrites
  drifted ``claim_kind`` strings into canonical kinds via
  :data:`CLAIM_KIND_SYNONYMS` (≈ 60 entries covering 11 canonical
  kinds × 4-5 LLM variants each, lex-ordered for diff stability);
  :func:`normalize_payload` rewrites alternative service-tag
  spellings (``svc=X``, ``for service X``, ``service:X``,
  ``service_name=X``, …) into the canonical ``service=<tag>`` form
  via the closed-vocabulary :data:`_SERVICE_TAG_REWRITES` regex
  table. Per-call rewrite counters (``last_n_kind_rewrites``,
  ``last_n_payload_rewrites``) expose the normaliser's load-bearing
  status to the bench driver. Re-exported as
  ``RobustMultiRoundBundleDecoder``, ``CLAIM_KIND_SYNONYMS``,
  ``normalize_claim_kind``, ``normalize_payload``,
  ``normalize_handoff``.
- **Theorem family W12.** W12-Λ (real-LLM single-round / un-
  normalised structural limit on R-59, proved-empirical n=12 + 5
  seeds + structural sketch), W12-1 (RobustMultiRoundBundleDecoder
  sufficiency under bounded LLM noise, proved-conditional + proved-
  empirical n=60 saturated across 5 seeds), W12-2 (closed-
  vocabulary normalisation soundness, proved by inspection +
  mechanically-checked), W12-3 (backward-compat with R-58 + R-59-
  clean + cross-regime R-54..R-58 audit, proved-empirical n=8
  each), W12-4 (out-of-vocabulary noise-budget falsifier, proved-
  empirical n=8 saturated). The W12-C family (W12-C1/C2/C3) makes
  the cross-bench / real-Ollama / learned-normaliser extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-59 anchor +
  bar 9 — synthetic→real-LLM transfer split). The SDK v3.13 result
  clears the **strong success bar** § 1.1 (strict gain ≥ 0.20 on
  R-59 vs every un-normalised single-round / multi-round method
  including SDK v3.12 W11, stable across ≥ 3 (bank_seed, llm_seed)
  values, no regression on R-53 / R-54 / R-55 / R-56 / R-57 /
  R-58 / R-59-clean, audit T-1..T-7 preserved on every cell, named
  bench property + named falsifier regime, AND synthetic→real-LLM
  transfer split bar 9 satisfied — the new method includes the
  load-bearing closed-vocabulary normalisation layer that bar 9
  requires).
- **Honest scope.** The W12-1 win is *conditional* on (a) the
  named bench property (R-58 delayed-causal-evidence shape), (b)
  the producer-noise channel being bounded by the closed-vocabulary
  closure (every variant in :data:`NOISY_KIND_VARIANTS` is in
  :data:`CLAIM_KIND_SYNONYMS`), AND (c) round-N admission not being
  budget-starved (inherits W11-4). The W12-4 falsifier regime is
  the explicit counterexample: when the LLM emits *out-of-vocabulary*
  kinds the synonym table cannot cover (e.g.
  ``DEADLOCK_PROBABLY_DETECTED_MAYBE``), normalisation cannot
  rescue the run. The synthetic-noisy-LLM extractor is *labelled*
  in every Phase-59 report; the ``ollama`` opt-in mode is the
  honest extension path and is the W12-C2 next data point.

### Active conjectures (SDK v3.13)

- **W12-C1**: cross-bench transfer of the W12 normalisation
  contract to non-incident-triage benchmark families.
  Conjectural; falsifier = a benchmark family where LLM kind
  drift cannot be enclosed by any reasonable-size synonym table.
- **W12-C2**: real-Ollama transfer of W12-1 (Phase-59 ``ollama``
  mode against qwen2.5:14b-32k or qwen3.5:35b on Mac 1).
  Conjectural; the synthetic noisy channel is calibrated to the
  empirical Phase-53 14B/35B distributions, but the real LLM may
  emit drift outside the closed-vocabulary closure.
- **W12-C3**: a learned normaliser strictly widens the closure
  beyond the hand-curated table. Conjectural; restated as a
  research move not a structural fix.

### Discharged conjectures (SDK v3.13)

- **W11-C2** (SDK v3.12): real-LLM transfer of W11-1.
  **PARTIALLY DISCHARGED-empirical** by the W12 family: the
  *un-normalised* W11 decoder does NOT transfer (W12-Λ shows
  multi_round ties FIFO at 0.000 on Phase-59 default at
  ``synonym_prob=0.50``), but a *normalised* W11 decoder
  (W12-1) DOES transfer (+1.000 vs every un-normalised method,
  stable 5/5). The honest revised reading: synthetic cross-round
  structure transfers to real-LLM regimes *only when an explicit
  normalisation layer absorbs the producer's kind / payload drift
  channel*.

## Previous frontier (SDK v3.12, 2026-04-26)

### Active moves (SDK v3.12 — multi-round bundle-aware team decoder + W11 family — *first cross-round coordination move*)

- **Phase-58 multi-round delayed-causal-evidence benchmark.**
  ``vision_mvp.experiments.phase58_multi_round_decoder`` runs 9
  strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9),
  capsule_bundle_decoder (W10 single-round), capsule_multi_round
  (W11)) on a deterministic 8-scenario bank with the
  **delayed-causal-evidence** property: round-1 carries
  generic-noise-only mentions of (gold_A, gold_B, decoy) where the
  decoy is cross-role-corroborated; round-2 carries one
  specific-tier disambiguating ``claim_kind`` with NO ``service=``
  token. 8/8 default scenarios satisfy the property; mechanically
  verified by ``Phase58BankShapeTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=8``,
  ``bank_seed=11``, ``bank_replicates=2``,
  ``noise_decoy_role_floor=2``. Headline:
  ``capsule_multi_round`` achieves ``accuracy_full = 1.000`` while
  substrate, FIFO, priority, coverage, W7-2, W8, W9, AND W10
  single-round all produce ``accuracy_full = 0.000`` — the **first
  strict separation between multi-round and single-round capsule-
  native coordination** in the programme, **+1.000** vs every
  prior method, stable across **5/5** alternate bank seeds. The
  W11 family (W11-Λ / W11-1 / W11-2 / W11-3 / W11-4 — proved or
  proved-empirical) anchors the milestone formally; the W11-C
  family (W11-C1/C2/C3) makes the cross-bench / real-LLM /
  multi-step extensions falsifiable.
- **``MultiRoundBundleDecoder`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Decodes the *union* of
  admitted handoffs across multiple ROLE_VIEW capsules; if the
  elected root_cause is specific-tier, drops every service tag
  whose admitted mentions are exclusively generic-noise kinds AND
  span ≥ ``noise_decoy_role_floor`` distinct producer roles
  (default 2). Inner ``BundleAwareTeamDecoder`` configured with
  ``cck_filter=False`` so the contradiction-aware step is the only
  filter; the W10 fallback path preserves single-round wins on
  R-54..R-57 (W11-3). Companion helper
  ``collect_admitted_handoffs(ledger, role_view_cids)`` materialises
  multi-round admitted handoffs into the duck-typed
  ``_DecodedHandoff`` shape. Re-exported as
  ``MultiRoundBundleDecoder`` and ``collect_admitted_handoffs``.
- **Theorem family W11.** W11-Λ (single-round structural limit on
  R-58, proved-empirical + structural sketch), W11-1 (multi-round
  decoder sufficiency, proved-empirical n=40 saturated across 5
  seeds), W11-2 (round-union monotonicity, proved structural),
  W11-3 (backward-compat with W7-2 / W8 / W9 / W10 on
  R-54 / R-55 / R-56 / R-57, proved-empirical), W11-4 (round-budget
  falsifier, proved-empirical n=8 saturated). The W11-C family
  (W11-C1/C2/C3) makes the cross-bench / real-LLM / multi-step
  extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-58 anchor +
  bar 8 — temporal/structural split). The SDK v3.12 result clears
  the **strong success bar** § 1.1 (strict gain ≥ 0.20 on R-58 vs
  every SDK v3.11 single-round method, stable across ≥ 3 seeds, no
  regression on R-53 / R-54 / R-55 / R-56 / R-57, audit T-1..T-7
  preserved on every cell, named bench property + named falsifier
  regime, AND temporal/structural split bar 8 satisfied).
- **Honest scope.** The W11-1 win is *conditional* on the named
  bench property; the W11-4 falsifier regime is the explicit
  counterexample. W11-3 backward-compat is exact on R-54 / R-55 /
  R-56 / R-57 thanks to (a) the inner W10 decoder's fallback-on-
  small-admitted-set path, (b) the noise-decoy floor being
  insensitive to single-role decoys. The contradiction-aware drop
  is closed-vocabulary on incident-triage; W11-C1 is the
  conjectural extension to other benchmark families. The decoder
  is a no-op on generic-tier elected root_cause (W11-Λ at the
  temporal axis collapses).

### Active conjectures (SDK v3.12)

- **W11-C1**: noise-decoy drop generalises to non-incident-triage
  benchmark families. Conjectural; falsifier = a benchmark family
  where a generic-noise-only mention is informative.
- **W11-C2**: real-LLM transfer of W11-1. Conjectural; Phase-59
  candidate.
- **W11-C3**: contradiction-aware round-resolution rule (last-wins
  / weighted-confidence) strictly outperforms naive union with
  ≥ 3 rounds and conflicting specific-tier evidence across rounds.
  Conjectural; multi-step capsule chains not yet shipped.

### Discharged conjectures (SDK v3.12)

- **W10-C3** (SDK v3.11): multi-round bundle decoder closes W10-4
  on a sub-class of scenarios. **PARTIALLY DISCHARGED** by the
  W11 family on R-58 (a different regime than W10-4): W11-1
  resolves the *delayed-causal-evidence* sub-class; the W10-4
  *decoy-CCK-promotion* sub-class remains open (the W11
  contradiction-aware drop is orthogonal — drops noise-corroborated
  decoys, not CCK-corroborated decoys).

## Previous frontier (SDK v3.11, 2026-04-26)

### Active moves (SDK v3.11 — bundle-aware team decoder + W10 family — *first decoder-side coordination move*)

- **Phase-57 multi-service-gold + corroborated-decoy decoder-forcing
  benchmark.**
  ``vision_mvp.experiments.phase57_decoder_forcing`` runs 8 strategies
  (substrate, capsule_fifo, capsule_priority, capsule_coverage,
  capsule_cohort_buffered (W7-2), capsule_corroboration (W8),
  capsule_multi_service (W9), capsule_bundle_decoder (W10-AD: W9
  admission + bundle-aware decoder)) on a deterministic 12-scenario
  bank (clipped to 10 by default) where (i) every scenario has
  ``gold_services`` of size 2 with a *specific-tier* gold root_cause,
  (ii) both gold services are corroborated by ≥ 2 distinct producer
  roles via at least one CCK-eligible (causal-tier) claim_kind, AND
  (iii) a decoy service is *also* corroborated by ≥ 2 distinct
  producer roles via *only* non-causal claim_kinds (LATENCY_SPIKE,
  ERROR_RATE_SPIKE, FW_BLOCK_SURGE). 10/10 default scenarios satisfy
  the bench property; mechanically verified by
  ``Phase57BankShapeTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=10``,
  ``bank_seed=11``, ``ms_top_k=3``, ``ms_min_corroborated_roles=2``,
  ``bundle_role_floor=1``,
  ``fallback_admitted_size_threshold=2``. Headline:
  ``capsule_bundle_decoder`` achieves ``accuracy_full = 1.000``
  while substrate, FIFO, priority, coverage, W7-2, W8, AND W9 all
  produce ``accuracy_full = 0.000`` — the **first strict separation
  between decoder-side coordination and any service-blind admission
  policy** in the programme, **+1.000** vs every admission baseline,
  stable across **5/5** alternate bank seeds. The W10 family
  (W10-Λ / W10-1 / W10-2 / W10-3 / W10-4 — proved or proved-empirical)
  anchors the milestone formally; the W10-C family (W10-C1/C2/C3)
  makes the cross-bench / real-LLM / multi-round extensions
  falsifiable.
- **``BundleAwareTeamDecoder`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Three knobs (``cck_filter``,
  ``role_corroboration_floor``,
  ``fallback_admitted_size_threshold``). Closed-vocabulary CCK table
  ``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`` maps each root_cause label
  to its causal claim-kind set (data-tier / storage-tier / compute /
  edge / network / generic). Decoder rule: pick root_cause via the
  same priority decoder as
  ``incident_triage._decoder_from_handoffs``; project admitted
  service tags through the CCK predicate; trust admission when the
  admitted-tag set is small (size ≤ threshold). Deterministic,
  training-free, additive on top of the existing ``TeamCoordinator``
  / ``ROLE_VIEW`` / ``TEAM_DECISION`` capsule layer. Re-exported as
  ``BundleAwareTeamDecoder`` and ``decode_admitted_role_view``.
- **Theorem family W10.** W10-Λ (admission-only structural limit on
  R-57, proved-empirical + structural sketch), W10-1 (bundle-decoder
  sufficiency, proved-empirical n=50 saturated), W10-2 (CCK
  structural correctness, proved by inspection), W10-3 (backward-
  compat with W7-2 / W8 / W9 on R-54 / R-55 / R-56, proved-empirical),
  W10-4 (decoy-CCK-promotion falsifier, proved-empirical n=10
  saturated). The W10-C family makes the cross-bench / real-LLM /
  multi-round extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-57 anchor +
  bar 7 — admission/decoding split). The SDK v3.11 result clears
  the **strong success bar** § 1.1 (strict gain ≥ 0.20 on R-57 vs
  both substrate FIFO and SDK v3.10 W9, stable across ≥ 3 seeds, no
  regression on R-53 / R-54 / R-55 / R-56, audit T-1..T-7 preserved
  on every cell, named bench property + named falsifier regime,
  AND admission/decoding split bar 7 satisfied).
- **Honest scope.** The W10-1 win is *conditional* on the named
  bench property; the W10-4 falsifier regime is the explicit
  counterexample. W10-3 backward-compat is exact on R-54 / R-55 /
  R-56 thanks to the trust-admission fallback (size ≤ 2 threshold).
  The CCK table is *closed-vocabulary for incident-triage*; W10-C1
  is the conjectural extension to other benchmark families. The
  bundle decoder is a no-op on generic-tier root_causes
  (``error_spike`` / ``latency_spike``) — this is the named scope
  limit of W10-1.

### Active conjectures (SDK v3.11)

- **W10-C1**: CCK table extends to non-incident-triage benchmark
  families (security incident, robotics, compliance review).
  Conjectural; falsifier = a benchmark family where no closed-
  vocabulary tier mapping exists.
- **W10-C2**: real-LLM transfer of W10-1. Conjectural; Phase-58
  candidate.
- **W10-C3**: multi-round bundle decoder closes W10-4 on a
  sub-class of scenarios. Conjectural; multi-round capsule chain
  not yet shipped.

### Discharged conjectures (SDK v3.11)

- **W9-C1** (SDK v3.10): bundle-aware decoder companion strictly
  improves on Phase-56 falsifier. **DISCHARGED-empirical** by W10-1
  on Phase 57 (+1.000 vs every admission-only baseline). The
  decoder-side axis is now the load-bearing axis of the SDK v3.11
  milestone.

## Previous frontier (SDK v3.10, 2026-04-26)

### Active moves (SDK v3.10 — multi-service top-K cross-role corroboration multi-agent benchmark + W9 family)

- **Phase-56 multi-service-gold + cross-role-corroborated benchmark.**
  ``vision_mvp.experiments.phase56_multi_service_corroboration`` runs
  7 admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9)) on a
  deterministic 10-scenario bank with the **multi-service-gold +
  both-gold-cross-role-corroborated + single-role-decoy-storm**
  properties (10/10 scenarios). Pre-committed default:
  ``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
  ``bank_seed=11``, ``ms_top_k=2``, ``ms_min_corroborated_roles=2``.
  Headline: ``multi_service − fifo accuracy_full = +1.000``,
  ``multi_service − cohort_buffered = +1.000``, AND
  ``multi_service − corroboration = +1.000``, stable across 5/5
  alternate bank seeds. The **first strict separation between
  multi-service top-K corroboration and single-tag corroboration**
  in the programme.
- **``MultiServiceCorroborationAdmissionPolicy`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Two sub-modes (streaming /
  buffered). Buffered factory ``from_candidate_stream`` is the W9-1
  anchor — pre-fits a top-K dominant tag set via the
  argmax-by-role-count tier of the corroboration score function.
  Selection rule: ``min_corroborated_roles`` floor → argmax-by-role-
  count tier → top-K by score (lex tie-break). Deterministic,
  training-free, one regex + two counters + the ``_dominant_tag_set``
  helper. Re-exported as
  ``TeamMultiServiceCorroborationAdmissionPolicy``.
- **Theorem family W9.** W9-1 (strict separation, proved-empirical
  n=50 saturated), W9-2 (argmax-tier strict-ordering, proved
  structural), W9-3 (backward-compat with W8 + W7-2 on Phase 55 +
  Phase 54, proved-empirical), W9-4 (decoy-corroboration falsifier,
  proved-empirical n=10 saturated). The W9-C family (W9-C1/C2/C3)
  makes the bundle-aware decoder / |gold|≥3 / real-LLM extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-56 anchor).
  The SDK v3.10 result clears the **strong success bar** § 1.1
  (strict gain ≥ 0.20 on R-56 vs both substrate FIFO and SDK v3.9
  W8, stable across ≥ 3 seeds, no regression on R-53 / R-54 / R-55,
  audit T-1..T-7 preserved on every cell, named bench property +
  named falsifier regime).
- **Honest scope.** The W9-1 win is *conditional* on the named
  bench property; the W9-4 falsifier regime is the explicit
  counterexample. W9-3 preserves the SDK v3.9 W8-1 win
  byte-for-byte on Phase 55 (via the argmax-by-role-count gate).

### Active conjectures (SDK v3.10)

- **W9-C1** (new SDK v3.10): bundle-aware decoder companion that
  filters service tags at decode time by the dominant
  *(claim_kind, role)* signature strictly improves accuracy_full on
  the Phase-56 falsifier regime. **Conjectural**; reframes W8-C3 as
  the natural attack on the W9-4 falsifier — pushes the structural
  axis from admission to decoding.
- **W9-C2** (new SDK v3.10): top-K extension to ``|gold| ≥ 3``.
  Conjectural; Phase-57 candidate; the policy already supports
  arbitrary ``top_k``.
- **W9-C3** (new SDK v3.10): real-LLM transfer of W9-1.
  Conjectural; SDK v3.10 confirms no-regression in low-surplus
  synthetic regime.

### Discharged conjectures (SDK v3.10)

- **W8-C1** (SDK v3.9): top-k corroboration improves multi-service
  scenarios. **DISCHARGED-empirical** by W9-1 on Phase 56 (+1.000).

## Previous frontier (SDK v3.9, 2026-04-26)

### Active moves (SDK v3.9 — cross-role corroboration multi-agent benchmark + W8 family)

- **Phase-55 decoy-plurality + cross-role-corroborated benchmark.**
  ``vision_mvp.experiments.phase55_decoy_plurality`` runs 6
  admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8)) on a deterministic 10-scenario bank
  with the **decoy-plurality + gold-corroboration** properties.
  Pre-committed default: ``K_auditor=4``, ``T_auditor=128``,
  ``n_eval=10``, ``bank_seed=11``. Headline: ``corroboration −
  fifo accuracy_full = +1.000`` AND ``corroboration −
  cohort_buffered accuracy_full = +1.000``, stable across 5/5
  alternate bank seeds. The first strict separation between W8
  and W7-2 in the programme.
- **``CrossRoleCorroborationAdmissionPolicy`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Two sub-modes (streaming /
  buffered). Buffered factory ``from_candidate_stream`` is the
  W8-1 anchor — pre-fits a (role, tag)-aggregated dominant tag
  via score function ``W_role · |distinct_roles| + |raw_mentions|``.
  Deterministic, training-free, one regex + two counters.
  Re-exported as ``TeamCrossRoleCorroborationAdmissionPolicy``.
- **Theorem family W8.** W8-1 (strict separation, proved-empirical
  n=50 saturated), W8-2 (score-function strict-ordering, proved
  structural), W8-3 (backward-compat with W7-2 on Phase 54,
  proved-empirical), W8-4 (decoy-corroboration falsifier,
  proved-empirical n=10 saturated). The W8-C family
  (W8-C1/C2/C3) makes the multi-service / decoder-side / real-LLM
  extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``. The SDK v3.9
  result clears the **strong success bar** § 1.1 (strict gain
  ≥ 0.20 on R-55, stable across ≥ 3 seeds, no regression on R-53
  / R-54, audit preserved on every cell, named bench property +
  named falsifier regime).
- **Honest scope.** The W8-1 win is *conditional* on the named
  bench property; the W8-4 falsifier regime is the explicit
  counterexample. W8-3 preserves the SDK v3.8 W7-2 win
  byte-for-byte on Phase 54.

### Active conjectures (SDK v3.9)

- **W8-C1**: multi-service-gold extension (top-k corroboration).
  conjectural; falsifier in Phase-56 candidate.
- **W8-C2**: W8-1 transfers to a real-LLM regime when the
  producer LLM emits multi-service decoy chatter that satisfies
  the bench property. conjectural; Phase-56 candidate.
- **W8-C3**: bundle-aware decoder + corroboration admission
  strictly outperforms admission alone on partial-coverage
  benches. conjectural; restates W7-C2 under W8 framing.

## Previous frontier (SDK v3.8, 2026-04-26)

### Active moves (SDK v3.8 — cross-role cohort-coherence multi-agent benchmark + W7 family)

- **Phase-54 cross-role cohort-coherence benchmark.**
  ``vision_mvp.experiments.phase54_cross_role_coherence`` runs 5
  admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_streaming, capsule_cohort_buffered)
  on a deterministic 10-scenario bank with the **gold-plurality**
  property (gold service has strictly more auditor-routed
  candidates than any decoy service). Pre-committed default:
  ``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
  ``bank_seed=11``. Headline: ``cohort_buffered − fifo
  accuracy_full = +1.000``, stable across 5/5 alternate bank
  seeds.
- **``CohortCoherenceAdmissionPolicy`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Two sub-modes:
  *streaming* (arrival-order-sensitive baseline) and *buffered*
  (arrival-order-stable, pre-fitted plurality via
  ``from_candidate_payloads``). Deterministic, training-free,
  one regex + one counter. Re-exported as
  ``TeamCohortCoherenceAdmissionPolicy``.
- **Theorem family W7.** W7-1 (FIFO unbeatability under low
  surplus, proved-empirical anchor on Phase-53), W7-1-aux
  (streaming cohort instability under arrival permutation,
  proved-empirical), W7-2 (cohort buffered structural win under
  gold-plurality, proved-empirical n=50 saturated), W7-2-conditional
  (K-sweep window, proved-empirical), W7-3 (extraction floor,
  proved-negative — corollary of Capsule Contract C5). The
  W7-C family (W7-C1/C2/C3) makes the multi-service /
  decoder-side / real-LLM extensions falsifiable.
- **Honest scope.** The W7-2 win is *conditional* on stated bench
  properties (gold-plurality + cross-role coherence +
  ``|candidates| > K_auditor``); it does not generalise to every
  multi-agent benchmark. The capsule layer's *audit* contribution
  (T-1..T-7) is preserved and extends to Phase-54 unchanged.

### Active conjectures (SDK v3.8)

- **W7-C1**: multi-service-gold extension (top-2 plurality).
  conjectural; falsifier in Phase-55 candidate.
- **W7-C2**: bundle-aware decoder + cohort admission strictly
  dominates cohort admission alone on weak-coherence benches.
  conjectural; not yet measured.
- **W7-C3**: W7-2 transfers to the real-LLM regime when the LLM
  is prompted with a multi-service event mix. conjectural;
  Phase-56 candidate.

## Previous frontier (SDK v3.7, 2026-04-26)

### Active moves (SDK v3.7 — stronger-model multi-agent benchmark + W6 family)

- **Phase-53 stronger-model multi-agent benchmark.**
  ``vision_mvp.experiments.phase53_scale_vs_structure`` runs
  three model regimes (synthetic / qwen2.5:14b-32k /
  qwen3.5:35b) × four capsule admission strategies + the
  Phase-31 substrate baseline on the same candidate-handoff
  stream. Real LLM calls hit Mac 1 Ollama at
  ``192.168.12.191:11434`` (Mac 2 still offline). Wall: 14B =
  92.6 s, 35B = 152 s.
- **Theorem family W6.** W6-1 (audit-OK grid 60/60),
  W6-2 (backend duck-typing), W6-3 (parser robustness on the
  closed-vocabulary claim grammar) are proved + mechanically-
  checked. W6-4 (the empirical decomposition) is proved-empirical
  on n=5 saturated.
- **Conditional falsification of W4-C1.** The SDK v3.5
  learned-admission-policy advantage **does not transfer
  out-of-distribution** to the real-LLM regime. Per-regime gap
  (capsule_learned − capsule_fifo): -0.4 (synthetic) / -0.4
  (qwen2.5:14b-32k) / 0.0 (qwen3.5:35b). The W4-C1 row in the
  registry is now conditional (see § 4.4 of
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`).
- **Honest scope.** Mac 2 is still offline; no two-Mac sharded
  inference run happened in SDK v3.7. The strongest model class
  exercised is single-Mac qwen3.5:35b (36 B-MoE). The
  ``MLXDistributedBackend`` adapter is unchanged from SDK v3.6
  and remains correct against the in-process stub.

### Active conjectures (SDK v3.7)

- **W6-C1**: drafted-conjecture-falsified — structure_gain is
  non-positive at every regime tested on Phase-53 default;
  scale narrows a deficit (not a surplus).
- **W6-C2**: drafted-conjecture-falsified — synthetic→real
  transfer of the learned admission scorer LOSES to FIFO out-
  of-distribution.
- **W6-C3**: empirical-positive — cross-(14B, 35B) candidate-
  kind TVD = 0.167 on the pooled (source_role × claim_kind)
  histogram (above the 0.10 falsifier).
- **W6-C4**: new conjectural-empirical — substrate FIFO is
  competitive with every capsule admission policy at sufficient
  K_auditor; falsifier search direction is K_auditor ∈ {1, 2, 3}.
- **W6-C5**: new conjectural-empirical — model scale narrows
  the OOD generalisation gap of the per-role admission scorer
  trained on synthetic noise.

## Previous frontier (SDK v3.6, 2026-04-26)

### Active moves (SDK v3.6 — two-Mac distributed inference + real cross-LLM)

- **Chosen two-Mac inference path: MLX distributed.** Apple-
  official, supports Llama / Qwen / Mistral natively, and
  exposes a single OpenAI-compatible HTTP server (head rank)
  regardless of single-host or sharded across N hosts. Hyperspace
  is a strong distributed-agent infrastructure but **not** a
  single-model sharding system; llama.cpp `--rpc` is a
  defensible alternative but with smaller Apple-Silicon
  optimisation.
- **Realistic model class on 2×36 GB:** 70B-class in Q4
  (≈ 40 GB weights; fits across two Macs with KV-cache headroom).
  35B-class in Q4 fits on a single Mac; sharding buys
  context-length / KV headroom only.
- **Wevra integration boundary** (`vision_mvp.wevra.llm_backend`):
  a duck-typed `LLMBackend` Protocol with two concrete
  implementations (`OllamaBackend`, `MLXDistributedBackend`).
  The runtime's inner-loop seal-PROMPT / seal-LLM_RESPONSE chain
  accepts any conformant backend without any spine modification
  (W5-2 proved); the OpenAI-compatible wire shape is locked
  against a stub server (W5-3 proved).
- **Real cross-LLM parser-boundary measurement (W5-1)**:
  `parser_boundary_real_llm.py` against the live Mac 1 Ollama
  endpoint yields cross-model PARSE_OUTCOME failure-kind
  TVD = 1.000 between Qwen-2.5-14B (dense, Q4) and Qwen-3.5-35B
  (MoE, Q4, `think=False`) under strict parsing on n=10
  instances — the larger model emits OLD/NEW close as `<<`
  instead of `<<<` and lands entirely in `unclosed_new`, while
  the smaller model emits `<<<` cleanly. Robust-mode
  `recovery=closed_at_eos` collapses cross-model TVD to 0.000.
  This **inverts the naive prediction** that a stronger model
  would reduce parser-boundary instability.

### Active conjectures (SDK v3.6)

- **W5-C1**: parser-boundary instability is a (model
  architecture × prompt-format) interaction, not a capacity
  artefact. Empirical-research; falsifier = a bank on which
  the larger model strict-parses ok > 50%.
- **W5-C2**: robust-mode `recovery=closed_at_eos` is the
  load-bearing safety net that makes the capsule-native runtime
  model-class-agnostic on the bundled prompt format. Empirical-
  research; falsifier = a model whose `unclosed_new` cannot be
  salvaged.
- **W5-C3**: closed-vocabulary `PARSE_OUTCOME.failure_kind` is
  a *minimum sufficient* typed witness of cross-model behaviour
  differences. Conjectural research; falsifier = a model pair
  with identical strict-mode `failure_kind` distribution but
  materially different downstream test-pass rate.

## Current frontier (SDK v3.5, 2026-04-26)

### Active moves (SDK v3.5 — multi-agent capsule coordination)

- **Capsule-native multi-agent team coordination
  (W4 family).** Three new closed-vocabulary capsule kinds
  (TEAM_HANDOFF, ROLE_VIEW, TEAM_DECISION) make capsules
  load-bearing *between* agents. ``TeamCoordinator`` drives one
  coordination round end-to-end; ``audit_team_lifecycle``
  mechanically verifies T-1..T-7 (Theorem W4-1).
- **Coverage-implies-correctness** (W4-2, proved-conditional) and
  **Local-view limitation** (W4-3, proved-negative) anchor the
  team-level mechanism in the formal layer.
- **Learned per-role admission policy** (``team_policy.py``)
  strictly improves pooled team-decision accuracy over the
  strongest fixed baseline at matched per-role budgets on the
  Phase-52 incident-triage bench (W4-C1 positive empirical;
  conjectural at smaller train scales).
- **Phase-52 reference benchmark**
  (``vision_mvp/experiments/phase52_team_coord.py``) compares
  substrate / capsule_fifo / capsule_priority / capsule_coverage
  / capsule_learned head-to-head and reports
  ``audit_ok_rate = 1.000`` for every capsule strategy.

### Active moves (SDK v3.4 — still in force)

- **Capsule-native execution one further structural layer past
  v3.3.** PROMPT capsule sealed for every LLM call's prompt
  bytes; LLM_RESPONSE capsule sealed for every response bytes.
  PROMPT.parents = (SWEEP_SPEC,) (Theorem W3-42); LLM_RESPONSE
  parent = sealed PROMPT (Theorem W3-43); PARSE_OUTCOME may
  parent on (SWEEP_SPEC, LLM_RESPONSE) so the
  prompt → response → parse → patch → verdict chain is a
  typed DAG witness end-to-end (Theorem W3-44).
- **Lifecycle audit extended to L-9 / L-10 / L-11** (Theorem
  W3-45). Soundness: ``audit_capsule_lifecycle(ctx).verdict ==
  "OK"`` iff the ledger satisfies the eleven invariants.
- **Synthetic-LLM mode for CI-runnable end-to-end exercise.**
  ``SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)``
  uses a deterministic in-process synthetic LLM client; no
  network. The full prompt/response/parse/patch/verdict chain
  seals end-to-end on every (task, strategy).
- **Cross-model parser-boundary research (W3-C6, empirical).**
  ``vision_mvp.experiments.parser_boundary_cross_model``
  reports cross-distribution PARSE_OUTCOME failure-kind TVD up
  to 1.000 across the synthetic distribution library, and
  strict→robust parser-mode shift up to 1.000 on
  ``synthetic.unclosed``.

### Active moves (SDK v3.3 — still in force)

- **PARSE_OUTCOME lifecycle gate.** Theorem W3-39.
- **Runtime-checkable lifecycle audit.** Theorem W3-40 / W3-45.
- **Deterministic-mode replay.** Theorem W3-41.

### Sharp limitation theorems we hold

- **W3-14** (negative): per-capsule budgets cannot enforce
  table-level cardinality invariants.
- **W3-16** (negative): cohort-lifting cannot enforce relational
  invariants.
- **W3-17** (conditional): admission rules cannot exceed the
  priority-decoder ceiling under ceiling-forcing spurious
  injection.
- **W3-21** (negative): linear class-agnostic decoders cannot
  achieve symmetric zero-shot transfer when gold-conditional
  feature signs flip across domains.
- **W3-29** (lower bound): magnitude-monoid linear family cannot
  cross the Bayes-divergence zero-shot risk lower bound.
- **W3-36** (sharp impossibility): the primary capsule ledger
  cannot authenticate its own rendering's bytes.
- **W4-3 (SDK v3.5)** (proved-negative): per-role budget below
  the role's causal-share floor admits sound runs that fail the
  team gate; no admission policy (FIFO, priority, coverage,
  learned) can recover. The natural next move is a
  cohort-lifted role view (W4-C2, conjectural).

### Active conjectures

- **W3-C1**: every Phase-N bounded-context theorem subsumes under
  the capsule contract. Conjectural (the four-case subsumption is
  proved; the general statement is open).
- **W3-C5 (legacy SDK v3.3)**: a sub-intra-cell PROMPT /
  LLM_RESPONSE capsule slice closes the inner-loop boundary
  without breaking W3-34 spine equivalence. **DISCHARGED in
  SDK v3.4** by Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **W3-C6 (new in SDK v3.4)**: synthetic-LLM cross-distribution
  PARSE_OUTCOME failure-kind TVD ≥ 0.5 across the calibrated
  distribution library. **Empirical** (reproducible; not a
  proof — the distribution library is synthetic, not real
  cross-LLM).
- **W3-C9**: refined paradigm-shift reading (Phase-49 candidate at
  $n=80$ point-estimate, zero-shot gap reading).
- **W3-C10**: relational decoder level-ceiling.
- **W4-C1 (SDK v3.5)**: learned per-role admission policy
  admits strictly fewer handoffs (12/12 seeds, robust direction)
  and improves pooled team-decision accuracy on most train seeds
  (gap_full > 0 in 11/12 seeds, mean +0.054; gap_root_cause
  > 0 in 8/12 seeds, mean +0.032) over the strongest fixed
  admission baseline (coverage-guided) on the Phase-52 default
  config — but the accuracy advantage reverses at higher noise
  (spurious=0.50). Empirical: budget-efficiency dominance is
  robust per-seed; accuracy advantage is mean-positive, not
  strict per-seed.
- **W4-C2 (SDK v3.5)**: cohort-lifted role view closes W4-3 on a
  sub-class of scenarios.
- **W4-C3 (SDK v3.5)**: capsule-layer admission rule subsumes
  the Phase-36 ``AdaptiveSubscriptionTable`` route-edit primitive.

### Active retractions

- **W3-C7 (strict reading) is retracted.** "Point-estimate
  Gate 1 at $\hat p \ge 0.400$ AND strict zero-shot Gate 2 with
  per-direction penalty ≤ 5pp" was falsified at $n=320$ (W3-26,
  W3-27). Do not reintroduce the strict bar.
- **W3-C3** is retracted in favour of W3-15 cohort lift.
- **The earlier W3-C4** (now reused for SDK-v3.3
  PARSE_OUTCOME conjecture) named a candidate decoder paradigm
  shift; the strict reading is folded into W3-C7 retraction.

## What we are NOT actively claiming

- **Not** "we solved context."
- **Not** "we solved multi-agent context." SDK v3.14's W13-1 result
  is the strongest *open-world-drift-under-bounded-heuristic-
  closure* structural-win the programme has produced
  (LayeredRobustMultiRoundBundleDecoder wins on R-60-wide by +1.000
  vs every fixed-vocabulary method **including SDK v3.13 W12**;
  backward-compatible on R-54..R-58 / R-59 / R-60-clean; stable
  across 5/5 (bank_seed, llm_seed) values; named bench property +
  named falsifier regime W13-4), but it is still **conditional on**
  (a) the bench property (R-58 delayed-causal-evidence shape with
  the Phase-60 wide-OOV drift channel), (b) the producer-noise
  channel being bounded by the heuristic closure, AND (c) round-N
  admission not being budget-starved (inherits W11-4). On a real
  Ollama producer at the 14B class, the bench property does NOT
  hold by default (W13-Λ-real); the synthetic→real-LLM transfer is
  gated by event-shape design + prompt-side discipline, not by
  normalisation. This is an honest empirical finding, not a closure
  of the question.

  SDK v3.13's W12-1 result remains the strongest *real-LLM-shaped-
  stream* (synthetic noisy) structural-win the programme has
  produced (RobustMultiRoundBundleDecoder wins on
  R-59 by +1.000 vs every un-normalised single-round / multi-round
  method **including SDK v3.12 W11**; backward-compatible on
  R-54 / R-55 / R-56 / R-57 / R-58 / R-59-clean; stable across
  5/5 (bank_seed, llm_seed) values; named bench property + named
  falsifier regime W12-4), but it is still **conditional** on
  (a) the bench property (R-58 delayed-causal-evidence shape),
  (b) the producer-noise channel being bounded by the closed-
  vocabulary closure (every variant in :data:`NOISY_KIND_VARIANTS`
  is in :data:`CLAIM_KIND_SYNONYMS`), AND (c) round-N admission
  not being budget-starved (inherits W11-4). The synthetic-noisy-
  LLM extractor is calibrated against Phase-53 14B/35B empirical
  distributions; the ``ollama`` opt-in mode is the W12-C2 next
  data point. Real multi-agent teams have additional axes
  (heterogeneous producers, time-varying budgets, multi-round
  handoffs with > 2 rounds and inter-round contradictions,
  conflicting goals, generic-tier root_causes the bundle decoder
  cannot help with, OOV kinds outside any reasonable closure) the
  W12 family does not cover. The W4-2 result is proved-conditional
  (premises: faithful decoder + sound admission); the W4-C1 learned-
  policy advantage is conditional empirical-positive on the SDK v3.5
  config and falsified out-of-distribution on the SDK v3.7 real-LLM
  regime. External validity to real production multi-agent teams is
  *materially* advanced by SDK v3.13 (the first synthetic→real-LLM
  transfer move with a named bounded-noise channel) but not fully
  closed.
- **Not** "the runtime is fully capsule-native." Specifically not
  capsule-native: sandbox stdout/stderr, sub-step parser-internal
  objects (regex match objects, recovery heuristic intermediate
  state), and on-the-wire LLM streaming chunks. PROMPT bytes and
  LLM_RESPONSE bytes ARE now capsule-tracked under SDK v3.4 (the
  prior "not capsule-native: LLM prompt bytes, raw LLM response
  bytes" line is **superseded** by Theorems W3-42 / W3-43).
- **Not** "Wevra is a universal multi-agent platform."
- **Not** "the decoder beat the Phase-31 ceiling by 22.5 pp."
  The sharper reading is W3-19 ($+15$pp at $n=80$, FIFO admission).
- **Not** "deterministic mode means the entire run is
  reproducible." It means the *capsule DAG* is reproducible under
  a frozen JSONL + a deterministic profile. Wall-clock and
  host-local fields are stripped from CIDs but live on disk.
- **Not** "the synthetic-LLM cross-distribution study is a real
  cross-LLM study." The distributions are calibrated synthetic
  (see ``synthetic_llm.SYNTHETIC_MODEL_PROFILES``), not
  measurements of ``gemma2:9b`` / ``qwen2.5:7b`` outputs. The
  empirical claim is about the parser's failure-kind closed
  vocabulary's *resolving power*, not about LLM output
  distributions in the wild.

## How to update this document

1. When a phase ships, add one line to the "Active moves" list and
   move any superseded line to "Sharp limitation theorems we hold"
   or "Active retractions" as appropriate.
2. When a conjecture is proved or falsified, move it to the
   correct section and update `THEOREM_REGISTRY.md`.
3. When a milestone note ships, add a one-line cross-link in this
   doc's relevant section.
4. Always update the "Last touched" date at the top.

## Cross-references

- Formal model (run-boundary, W3 family): `docs/CAPSULE_FORMALISM.md`
- Formal model (team-boundary, W4 family): `docs/CAPSULE_TEAM_FORMALISM.md`
- Theorem registry: `docs/THEOREM_REGISTRY.md`
- How-not-to-overstate rules: `docs/HOW_NOT_TO_OVERSTATE.md`
- Master plan: `docs/context_zero_master_plan.md`
- Milestone notes: `docs/archive/wevra-milestones/RESULTS_WEVRA_*.md`,
  `docs/archive/capsule-research/RESULTS_CAPSULE_*.md` (historical),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_DEEP_INTRA_CELL.md` (SDK v3.3),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_INNER_LOOP.md` (SDK v3.4),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md` (SDK v3.5),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_DISTRIBUTED.md` (SDK v3.6),
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` (SDK v3.7),
  `docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md` (SDK v3.8),
  `docs/RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md` (SDK v3.9 — this milestone),
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` (SDK v3.9 — pre-committed bar)
- Paper draft: `papers/wevra_capsule_native_runtime.md`
- Tests: `vision_mvp/tests/test_wevra_capsule_native*.py`,
  `test_wevra_capsule_native_deeper.py`,
  `test_wevra_capsule_native_inner_loop.py` (SDK v3.4),
  `test_wevra_team_coord.py` (SDK v3.5 — multi-agent slice),
  `test_wevra_scale_vs_structure.py` (SDK v3.7 — Phase-53),
  `test_wevra_cross_role_coherence.py` (SDK v3.8 — Phase-54 + W7),
  `test_capsule_*.py`
- Cross-model parser-boundary experiment:
  `vision_mvp/experiments/parser_boundary_cross_model.py`
- Multi-agent team coordination benchmark (synthetic):
  `vision_mvp/experiments/phase52_team_coord.py`
- Stronger-model multi-agent benchmark (real LLM):
  `vision_mvp/experiments/phase53_scale_vs_structure.py`
- Cross-role cohort-coherence benchmark (deterministic):
  `vision_mvp/experiments/phase54_cross_role_coherence.py`
- Cross-role corroboration benchmark (deterministic, harder):
  `vision_mvp/experiments/phase55_decoy_plurality.py`
- MLX distributed runbook (operator path for Mac 2):
  `docs/MLX_DISTRIBUTED_RUNBOOK.md`
