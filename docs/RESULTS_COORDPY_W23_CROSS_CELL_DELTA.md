# SDK v3.24 — capsule-native cross-cell delta execution + quorum-keyed cache + super-token reference (W23 family)

> Theory-forward results note for the post-W22 milestone. This is the
> first capsule-native multi-agent-coordination method that *combines*
> explicit-capsule passing with audited proxies for the LatentMAS
> *cross-cell latent state-sharing*, *delta execution*,
> *super-token side channel*, and *quorum-keyed cache* directions —
> measured on regimes where (a) consecutive cells share enough state
> for delta execution to compress visible context, (b) probabilistic
> LLM oracles reproduce W22-C-CACHE-AMPLIFICATION (and the W23
> mitigation discharges it), and (c) a controller-side super-token
> registry verifies a single-visible-token CID reference against an
> auditable trust boundary. The W22-1 strict-efficiency result is
> preserved byte-for-byte; W23 contributes on the **cross-cell
> efficiency**, **mitigation**, **dense-control**, and
> **producer-decoder split** axes. Date stamp: 2026-04-29.

## TL;DR

* **W23-1 (cross-cell efficiency, proved-empirical, n=8 saturated × 5
  seeds × 2 cells).** On R-70-DELTA-FANOUT, pairing the W22
  ``LatentDigestDisambiguator`` with the W23
  ``CrossCellDeltaDisambiguator`` over a registered
  ``SchemaCapsule`` strictly reduces
  ``mean_n_w23_visible_tokens_to_decider`` over the W22 baseline.
  At ``T_decoder = None`` (loose): W22 = 41.25, W23 delta = 38.50
  (−2.75 tokens/cell, **−6.67 %**); W23 super-token = 30.75 (−10.50
  tokens/cell, **−25.45 %**). At ``T_decoder = 24`` (tight): W22 =
  36.50, W23 delta = 33.75 (−2.75 tokens/cell, **−7.53 %**); W23
  super-token = 26.00 (−10.50 tokens/cell, **−28.77 %**).
  ``correctness_ratified_rate = 1.000`` byte-for-byte vs W22 on every
  cell of every W23 strategy. ``chain_verifies_ok_rate = 1.000``
  on every cell. Stable across **5/5** ``bank_seed`` values
  (11, 17, 23, 29, 31): super-token savings exactly **+10.50
  tokens/cell on every seed**; delta savings ≥ 2.75 tokens/cell on
  every seed. Code anchor:
  ``vision_mvp.experiments.phase70_capsule_session_delta.run_phase70``.
  Artifacts: ``docs/data/phase70_delta_fanout_K12_n8.json``,
  ``docs/data/phase70_delta_fanout_tight_K12_n8.json``,
  ``docs/data/phase70_super_token_K12_n8.json``,
  ``docs/data/phase70_seed_sweep_loose_K12_n8.json``,
  ``docs/data/phase70_cross_regime_synthetic.json``.
* **W23-2 (mitigation of W22-C-CACHE-AMPLIFICATION,
  proved-empirical).** On R-70-AMPLIFIED-LLM (a
  ``FlippingProbabilisticOracle`` that returns a decoy-asymmetric
  reply on consult #1 and gold-asymmetric replies thereafter — a
  deterministic structural simulation of the
  ``mixtral:8x7b`` first-sample drift named in SDK v3.23
  W22-C-CACHE-AMPLIFICATION), the W22 baseline reproduces the
  amplification: ``accuracy_full = 0.875`` (the cache freezes the
  bad first reply across all matching cells). The W23
  ``QuorumKeyedSharedReadCache`` with the
  ``CACHE_FRESHNESS_PER_CELL_NONCE`` policy on the flipping oracle
  achieves ``accuracy_full = 1.000``: cell 1 hits the oracle (bad
  reply); cell 2 misses the cache (per-cell nonce mixes in), hits
  the oracle again (good reply); quorum forms on gold; the
  amplification is mitigated. **+0.125 strict mitigation advantage
  over W22**. The W23 delta and super-token strategies (without
  quorum-keyed cache) tie the W22 baseline at 0.875 — isolating the
  quorum-keyed cache as the load-bearing W23-2 component, not the
  delta or super-token.
* **W23-3 (trust-boundary soundness, proved-empirical + proved by
  inspection).** Two failure modes are mechanically tested by
  enumerated falsifiers; on every cell the controller rejects and
  the W23 layer falls through to W22 byte-for-byte:
  * **R-70-SUPER-TOKEN-TAMPERED** (verifier registry split: producer
    registers in one ``SuperTokenRegistry``, controller verifies
    against an empty/separate registry):
    ``super_token_verification_ok_rate = 0.000`` on every cell;
    rejection reason = ``unknown_super_token``;
    ``correctness_ratified_rate = 1.000`` (W23 super-token answer =
    W22 answer byte-for-byte; ``w23_super_token_savings_per_cell =
    0.0`` — fall-through to verbose digest).
  * **R-70-CHAIN-BROKEN** (verifier chain-head override: after the
    genesis cell, the controller's expected
    ``prior_chain_head_cid`` is set to a phantom; the producer
    continues with the real chain head):
    ``chain_verifies_ok_rate = 0.125`` (only the genesis cell
    verifies; cells 1..7 fire ``chain_head_mismatch``);
    ``correctness_ratified_rate = 1.000`` (W22 answer preserved on
    rejection); ``w23_delta_savings_per_cell = 0.0`` —
    fall-through to verbose digest.
  The verification functions
  (``verify_session_digest_chain``, ``verify_session_delta``,
  ``verify_super_token_reference``) are short (≈ 30–40 lines each),
  the failure modes are enumerated (``"ok"`` /
  ``"hash_mismatch"`` / ``"schema_cid_mismatch"`` /
  ``"schema_version_unknown"`` / ``"empty_envelope"`` /
  ``"chain_head_mismatch"`` / ``"parent_session_mismatch"`` /
  ``"unknown_super_token"`` / ``"delta_cid_mismatch"``); soundness
  holds by inspection.
* **W23-Λ-no-delta (proved-empirical, named falsifier).** On
  R-70-NO-DELTA (chain reset every cell → every cell is a
  W23_BRANCH_GENESIS):
  ``w23_delta_savings_per_cell = 0.0``;
  ``w23_super_token_savings_per_cell = 0.0`` by construction (the
  W23 layer reduces to W22 byte-for-byte). Names the structural
  limit when there is no cross-cell state to delta against.
* **W23-Λ-real (proved-conditional + empirical-research, partially
  discharged).** Live W23 transfer on Mac-1 Ollama
  (``mixtral:8x7b``, 47B-MoE) on the AMPLIFIED-LLM regime + a live
  LLM adjudicator: visible-tokens W22 = 52.75/cell, W23 delta =
  51.25/cell, W23 super-token = 46.75/cell (savings +2.84 % delta,
  **+11.37 % super-token**). All four strategies tie at
  ``acc_full = 0.750`` (no mitigation advantage on this particular
  4-cell live probe; the live LLM does not reproduce the
  deterministic flipping pattern at this n).
  ``correctness_ratified_rates`` show ``capsule_w23_quorum_keyed =
  0.5`` (the quorum-keyed mitigation **changes** the outcome on half
  the cells but does not strictly improve overall accuracy on this
  live regime). ``chain_verifies_ok_rate = 0.750``;
  ``super_token_verification_ok_rate = 1.000``;
  ``cross_host_round_trip_bytes_total = 4232`` (the producer/decoder
  split is wire-validated). Newly named conjecture
  **W23-C-MITIGATION-LIVE-VARIANCE**: the synthetic mitigation
  advantage (deterministic flipping → +0.125 W23 ≫ W22) does not
  cleanly transfer to live mixtral on n=4 because the live LLM's
  drift pattern is not pure first-sample-decoy; the mitigation is
  *measurable in expectation* but not *strict per-probe*. Artifact:
  ``docs/data/phase70_live_mixtral_8x7b_n4.json``.
* **Two-Mac infrastructure (W23-Λ-cross-host).** Mac 2
  (192.168.12.248) ARP ``incomplete`` at milestone capture — same
  status as SDK v3.6 through SDK v3.23 (**17th milestone in a
  row**). **No two-Mac sharded inference happened in SDK v3.24.**
  The W23 surface is *naturally* a producer / cache-controller /
  decoder split: the
  :class:`CrossHostProducerDecoderProxy` forces every
  delta+session-digest envelope through a JSON-canonical
  serialisation round-trip on every cell, mechanically validating
  that the W23 envelopes survive a wire boundary with no shared
  Python references. On R-70-DELTA-FANOUT n=8:
  ``cross_host_round_trip_bytes_total = 14322`` (per-cell average
  ≈ 1.79 KB). When Mac 2 returns the same proxy interface drops
  in over a real socket with no W23 code changes. Strongest model
  class actually exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE
  Q4) on Mac-1 Ollama.
* **Bounded-context honesty preserved byte-for-byte.** The W23
  layer reads only what the W22 layer below it produced; the W15
  ``tokens_kept`` is byte-for-byte identical between W21, W22, AND
  W23. Mechanically verified:
  * ``Phase70DeltaFanoutTests::test_no_accuracy_regression_on_delta_fanout``
    — W23 ties W22 at 1.000 accuracy on every strategy.
  * ``Phase70SuperTokenTamperedTests::test_super_token_rejected_no_visible_tokens_savings``
    — on rejection, the W23 super-token visible-tokens cost equals
    the W22 baseline (no covert savings claimed).
  * ``Phase70ChainBrokenTests::test_chain_broken_no_visible_tokens_savings``
    — on chain rejection, the W23 delta visible-tokens cost equals
    the W22 baseline.
* **Backward-compat preserved byte-for-byte.**
  * **W23-3-A** (vs W22 / no-trigger paths). With ``enabled = False``
    OR ``schema = None`` OR an inner W22 branch outside
    ``trigger_branches``, the W23 layer reduces to W22
    byte-for-byte on the answer field (mechanically verified by
    ``W23SDKReductionTests``).
  * **W23-3-B** (full programme regression). 703 prior coordpy-anchor
    + capsule + recent-phase tests pass before the W23 milestone
    landed; **742 / 742** coordpy-suite tests pass after (+ 39 new
    W23 tests). The full pre-existing W22 / W21 / W20 / W19 / W18 /
    W17 / W16 / W15 / W14 / W13 / W12 / W11 / W10 / W9 / W8 / W7 /
    W6 / W5 / W4 / W3 test surfaces all pass.
  * **W23-3-C** (W22 anchor regression). The Phase 69
    ``test_w22_strict_visible_tokens_savings_loose`` /
    ``test_w22_strict_visible_tokens_savings_tight`` /
    ``test_w22_correctness_ratified_byte_for_byte`` all pass
    unchanged; the W22-1 strict-efficiency result on
    R-69-CACHE-FANOUT is preserved byte-for-byte
    (``mean_n_visible_tokens_to_decider`` savings +14.51 % loose,
    +16.09 % tight; cache_tokens_saved_total = 88).
* **Audit T-1..T-7 preserved on every cell of every regime.**
  ``Phase70AuditOKTests::test_audit_ok_on_every_w23_cell`` asserts
  ``audit_ok_grid`` is True for every strategy on every bank
  (delta_fanout, super_token, no_delta, super_token_tampered,
  chain_broken).

## What changed from SDK v3.23 → v3.24 (one paragraph)

The W22 family (SDK v3.23) crossed the W21 wire-cost wall at the
*per-cell* layer: schema-passing + delta execution +
shared-read cache + controller-verified latent digest, on one cell.
It explicitly named two open frontiers: (a) the
**W22-C-CACHE-AMPLIFICATION** conjecture — that the SharedReadCache
amplifies probabilistic LLM oracle drift across cells — and (b) the
absence of true cross-cell latent state-sharing or super-token /
dense-control side channels. SDK v3.24 implements the smallest
honest version of those moves *together*: a hash-chained
``SessionDigestEnvelope`` carrying running cross-cell state; a
small ``SessionDeltaEnvelope`` that emits only what changed against
that running state; a ``QuorumKeyedSharedReadCache`` with per-oracle
freshness policies (``byte_identical`` / ``per_cell_nonce`` /
``quorum_locked``) that *mitigates* W22-C-CACHE-AMPLIFICATION on
probabilistic adjudicators while preserving cross-cell wire savings
on deterministic ones; a single-visible-token
``SuperTokenReferenceEnvelope`` (the bounded steganographic /
dense-control payload experiment) verified through a controller-side
``SuperTokenRegistry``; and a within-process
``CrossHostProducerDecoderProxy`` that forces every W23 envelope
through a JSON-canonical wire round-trip — the honest fallback for
the unreachable Mac 2. Every latent payload is hash-chained,
schema-versioned, parent-CID-sealed, and (for super-tokens) registry-
verified; on any verification failure the W23 layer rejects and the
explicit-capsule path stays sound. The W23 surface is purely
additive on top of the W22 surface; the SDK v3.23 runtime contract
is byte-for-byte unchanged.

## Theorem family W23 (minted by this milestone)

We pre-commit eleven W23 statements. Three are **proved-empirical**
(saturated against pre-committed seeds); three are **proved by
inspection + mechanically-checked**; three are **proved-empirical
backward-compat anchors**; one is **conjectural**
(``W23-C-MITIGATION-LIVE-VARIANCE``); one is **proved-conditional +
empirical-research** (``W23-Λ-real``). Codebase status — all
numbered code paths land in
``vision_mvp/coordpy/team_coord.py`` (``SessionDigestEnvelope`` /
``SessionDeltaEnvelope`` / ``verify_session_digest_chain`` /
``verify_session_delta`` / ``SuperTokenReferenceEnvelope`` /
``SuperTokenRegistry`` / ``verify_super_token_reference`` /
``QuorumKeyedSharedReadCache`` /
``QuorumKeyedCachingOracleAdapter`` /
``CrossHostProducerDecoderProxy`` / ``W23SessionResult`` /
``CrossCellDeltaDisambiguator``)
and ``vision_mvp/experiments/phase70_capsule_session_delta.py``
(R-70 driver + cross-regime + seed-stability sweeps +
``FlippingProbabilisticOracle``).

* **W23-1** *(proved-conditional + proved-empirical n=8 saturated × 5
  seeds × 2 cells)*. On R-70-DELTA-FANOUT-LOOSE and
  R-70-DELTA-FANOUT-TIGHT, pairing the W22
  ``LatentDigestDisambiguator`` with the W23
  ``CrossCellDeltaDisambiguator`` over a registered SchemaCapsule
  strictly reduces ``mean_n_w23_visible_tokens_to_decider`` over the
  W22 baseline AND records ``chain_verifies_ok_rate = 1.000`` AND
  preserves ``accuracy_full`` byte-for-byte. Stable across 5/5
  ``bank_seed`` values ``(11, 17, 23, 29, 31)``. Mechanically
  verified by
  ``Phase70DeltaFanoutTests::test_delta_savings_loose``,
  ``Phase70DeltaFanoutTests::test_delta_savings_tight``,
  ``Phase70SeedStabilityTests::test_savings_strictly_positive_across_5_seeds``.
  *Conditions* (any failure collapses the result):
    1. The bench property — at least two cells share a session
       (W23 chain accumulates across ≥ 2 cells; on R-70-NO-DELTA
       the chain resets every cell and the savings collapse to 0).
    2. The inner W22 branch is in ``trigger_branches`` (i.e. fires
       ``W22_BRANCH_LATENT_RESOLVED``).
    3. The schema is registered AND the verifier's registered chain
       head matches the producer's chain head AND (for super-tokens)
       the verifier's super-token registry matches the producer's
       registered prefixes.
* **W23-2** *(proved-conditional + proved-empirical, mitigation)*. On
  R-70-AMPLIFIED-LLM (synthetic ``FlippingProbabilisticOracle``),
  the W22 baseline reproduces ``accuracy_full < 1.0`` (the cache
  amplifies the bad first reply); the W23 quorum-keyed cache
  (``CACHE_FRESHNESS_PER_CELL_NONCE`` on the flipping oracle)
  achieves strictly higher ``accuracy_full`` than the W22 baseline.
  **+0.125 strict mitigation advantage**. **Empirically discharges
  W22-C-CACHE-AMPLIFICATION as a mitigable property** (the
  amplification is real *and* the W23 mitigation is real on the
  same regime). Mechanically verified by
  ``Phase70AmplifiedLLMTests::test_w22_baseline_reproduces_amplification``,
  ``Phase70AmplifiedLLMTests::test_w23_quorum_keyed_strictly_improves_over_w22``,
  ``Phase70AmplifiedLLMTests::test_w23_delta_alone_does_not_fix_amplification``.
* **W23-3** *(proved-empirical n=8 + proved by inspection)*.
  Trust-boundary soundness. On R-70-SUPER-TOKEN-TAMPERED (verifier
  registry split) and R-70-CHAIN-BROKEN (verifier chain-head
  override), every tampered reference / chain link is rejected by
  the controller; the W23 layer fires
  :data:`W23_BRANCH_SUPER_TOKEN_REJECTED` /
  :data:`W23_BRANCH_DELTA_REJECTED`;
  ``correctness_ratified_rate = 1.000`` (the W23 answer field
  equals the W22 answer field byte-for-byte; the rejection does
  NOT corrupt the answer). The verification functions are short
  and the failure modes are enumerated; soundness holds by
  inspection. Mechanically verified by
  ``Phase70SuperTokenTamperedTests::test_super_token_rejected_on_every_cell``,
  ``Phase70ChainBrokenTests::test_chain_broken_rejects_post_genesis_cells``,
  ``SessionDigestVerificationTests`` /
  ``SessionDeltaVerificationTests`` /
  ``SuperTokenReferenceTests`` (one unit test per failure-mode
  label).
* **W23-3-A** *(proved-empirical n=8, backward-compat anchor)*.
  When the inner W22 branch is not in ``trigger_branches`` (W22
  abstained for any reason — NO_TRIGGER / LATENT_REJECTED /
  DISABLED / NO_SCHEMA), the W23 layer fires
  ``W23_BRANCH_NO_TRIGGER`` and reduces to W22 byte-for-byte on
  the answer field. With ``enabled = False`` OR ``schema = None``,
  the W23 layer reduces to W22 byte-for-byte. Mechanically verified
  by ``W23SDKReductionTests::test_disabled_reduces_to_w22``.
* **W23-3-B** *(proved-empirical full programme regression,
  742 / 742 coordpy-anchor + capsule + recent-phase tests pass)*. On
  R-54..R-69 default banks, the W23 layer ties the W22 layer
  byte-for-byte either via ``W23_BRANCH_NO_TRIGGER`` /
  ``W23_BRANCH_GENESIS`` (no cross-cell delta to compress) or via
  ``W23_BRANCH_DELTA_RESOLVED`` /
  ``W23_BRANCH_SUPER_TOKEN_RESOLVED`` (when both fire on the same
  regime). With ``enabled = False`` OR no schema registered, W23
  reduces to W22.
* **W23-3-C** *(proved-empirical, W22 anchor regression)*. On
  R-69-CACHE-FANOUT, the W22-1 strict-efficiency result holds
  byte-for-byte (``test_w22_strict_visible_tokens_savings_loose``
  passes). The W22-2 correctness ratification holds. The W22-3
  trust falsifiers (R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) hold.
* **W23-Λ-no-delta** *(proved-empirical n=8 saturated, named
  falsifier)*. On R-70-NO-DELTA (chain reset every cell),
  ``w23_delta_savings_per_cell = 0.0`` AND
  ``w23_super_token_savings_per_cell = 0.0`` by construction — the
  W23 layer reduces to W22 byte-for-byte. Names the structural
  limit when there is no cross-cell state to delta against.
  Mechanically verified by
  ``Phase70NoDeltaTests::test_no_delta_zero_savings``.
* **W23-Λ-real** *(proved-conditional + empirical-research, n=4)*.
  Live-LLM transfer on Mac-1 Ollama (``mixtral:8x7b``, 47B-MoE) on
  the AMPLIFIED-LLM bench: ``visible_tokens_savings_pct = +2.84 %
  delta``, ``+11.37 % super-token``;
  ``correctness_ratified_rates`` show
  ``capsule_w23_quorum_keyed = 0.5`` (the quorum-keyed mitigation
  changes the outcome on half the cells but does NOT strictly
  improve overall accuracy on this n=4 live probe). The synthetic
  ``+0.125`` mitigation advantage (W23-2) does not cleanly transfer
  here — names ``W23-C-MITIGATION-LIVE-VARIANCE`` as the
  follow-up frontier. ``cross_host_round_trip_bytes_total = 4232``
  (the producer/decoder split is wire-validated). Artifact:
  ``docs/data/phase70_live_mixtral_8x7b_n4.json``.
* **W23-C-MITIGATION-LIVE-VARIANCE** *(conjectural, named for the
  first time in this milestone)*. The synthetic
  ``FlippingProbabilisticOracle`` (consult #1 = decoy-asymmetric;
  consult #2.. = gold-asymmetric) reproduces W22-C-CACHE-
  AMPLIFICATION at maximum strictness, and the W23
  ``QuorumKeyedSharedReadCache`` (PER_CELL_NONCE on the flipping
  oracle) discharges it at +0.125 strict gain. **Conjecture**: the
  same mitigation has *positive expected* improvement on live
  ``mixtral:8x7b`` regimes but is *not strictly per-probe* because
  the live LLM's drift pattern at temperature=0 is not pure
  first-sample-decoy. **Falsifier**: an LLM whose first-sample
  drift fits the FlippingProbabilisticOracle pattern would produce
  strict W23 > W22 mitigation per-probe; an LLM whose drift pattern
  is unbiased symmetric across cells would produce
  ``E[mitigation] = 0``. Empirical observation in this milestone:
  on n=4 live mixtral, the mitigation advantage is +0.000 even
  though the quorum-keyed cache changes the outcome on 0.5 of
  cells — the LLM's drift pattern is approximately symmetric across
  consults at this n.
* **W23-Λ-cross-host** *(proved-conditional + empirical-research,
  partially discharged)*. The W23 surface validates the
  producer/decoder host-split contract through the within-process
  ``CrossHostProducerDecoderProxy``: every delta+session-digest
  envelope is JSON-canonical-encoded and round-tripped on every
  cell. ``cross_host_round_trip_bytes_total > 0`` on every regime
  where the chain advances (≈ 1.79 KB/cell on R-70-DELTA-FANOUT
  n=8). Mac 2 unreachable; no true two-host execution validated
  in SDK v3.24. The W23 envelopes are wire-compatible with
  cross-host deployment when Mac 2 returns — no W23 code changes
  required. Honest scope: this is *not* a true two-host setup; it
  is a structural simulation that lets us measure (a) bytes
  crossing the wire, (b) JSON-canonical encoding contract, (c)
  envelope round-trip preservation.

## Honest scope (W23)

The W23-1 strict gain is *strongly conditional* on:

1. The bench property — the W23 chain accumulates across ≥ 2 cells
   (else the W23 layer reduces to W22 by W23_BRANCH_GENESIS on
   every cell).
2. The inner W22 branch is ``W22_BRANCH_LATENT_RESOLVED`` (else
   W23 reduces to W22 by ``W23_BRANCH_NO_TRIGGER``).
3. The producer's chain head matches the verifier's registered
   chain head (else ``chain_head_mismatch`` rejection).
4. (For super-tokens) The verifier's registry contains the
   producer's registered prefixes (else ``unknown_super_token``
   rejection).

Three named falsifiers make the conditionality sharp:

* **W23-Λ-no-delta** — chain reset every cell → no cross-cell state
  → no savings.
* **R-70-SUPER-TOKEN-TAMPERED** — verifier registry split →
  ``unknown_super_token`` → fall through to W22 verbose digest.
* **R-70-CHAIN-BROKEN** — verifier chain-head override → every
  post-genesis cell rejected → fall through.

The W23 escape from the W22 per-cell envelope cost is *partial* by
design: bounded above by (a) the cross-cell state-sharing rate
(which depends on consecutive-cell bundle similarity), (b) the
super-token registry hit rate, AND (c) the chain verification
success rate (rejected envelopes pay full W22 baseline cost).

What W23 does **NOT** do (do-not-overstate):

* Does NOT touch transformer KV caches, embedding tables,
  attention weights, or any model-internal state. The "cross-cell
  state-sharing" lives at the **capsule layer**; it is an honest
  **proxy** for the LatentMAS *cross-cell latent state-sharing*
  direction, not a runtime KV transplant.
* Does NOT modify embedding-side payloads. The "super-token
  reference" is a single-visible-token CID prefix verified through
  a controller-side registry — it is a bounded, auditable proxy
  for the LatentMAS *super-token side channel* idea, **not** an
  embedding-level steganographic intervention. The bound on the
  channel is sharp: at most one super-token per cell; at most
  ``hex_prefix_len`` characters of payload (default 16); the
  registry is enumerable / auditable; tampering yields a
  ``hash_mismatch`` or ``unknown_super_token`` rejection.
* Does NOT solve "multi-agent context" in any unqualified sense.
  The W23-1 win is *strongly conditional* on the named bench
  property; under R-70-NO-DELTA, R-70-SUPER-TOKEN-TAMPERED, and
  R-70-CHAIN-BROKEN the savings claim collapses or the envelope
  is rejected. See ``HOW_NOT_TO_OVERSTATE.md`` § "W23 forbidden
  moves".
* Does NOT validate true two-host execution. Mac 2 has been
  ARP-incomplete for 17 milestones in a row; the
  CrossHostProducerDecoderProxy is a *structural* proxy that
  validates the wire-encoding contract, not the latency or
  partition-tolerance properties of a real cross-host deployment.

## Prior-conjecture discharge ledger (SDK v3.24)

* **W22-C-CACHE-AMPLIFICATION** *(SDK v3.23, named)*. **Empirically
  discharged as mitigable** by W23-2 on the synthetic R-70-AMPLIFIED-LLM
  regime (+0.125 strict mitigation advantage). The live-LLM
  direction is partially discharged: the W23 quorum-keyed cache
  *changes* outcomes on 0.5 of mixtral cells but does not strictly
  improve overall accuracy on n=4 — names
  W23-C-MITIGATION-LIVE-VARIANCE as the follow-up.
* **W21-C-CALIBRATED-TRUST** *(SDK v3.22, named, wire-cost
  direction partially discharged by SDK v3.23)*. Further
  partially discharged by W23-1 on cross-cell sessions: the
  per-cell delta + super-token replace the per-cell verbose digest
  and reduce visible-token cost an additional **6.67 %** (delta)
  to **25.45 %** (super-token) over the W22 baseline on
  R-70-DELTA-FANOUT.
* **W22-3-B-Λ-cross-host** *(SDK v3.23, conjectural)*. Partially
  discharged by W23-Λ-cross-host: the W23 surface validates the
  wire-format contract via JSON-canonical round-trip on every
  cell. Mac 2 still unreachable — no true two-host execution
  validated.

## What this means for the original goal

The original goal of the Context Zero programme is *solving
context for multi-agent teams* — not solving any one regime, but
demonstrating that a typed, content-addressed, lifecycle-bounded
capsule discipline can carry the load that ad-hoc context-passing
in production multi-agent systems cannot.

The W23 family advances that goal on **three new axes** beyond
the W17–W22 result family:

1. **Cross-cell efficiency.** The W22 wire-cost concern was
   discharged at the per-cell layer. The W23 layer extends that
   discharge to the **session layer**: a multi-cell coordination
   session no longer pays the verbose W22 digest on every cell.
   On R-70-DELTA-FANOUT loose: −6.67 % (delta) / −25.45 %
   (super-token) over W22; tight: −7.53 % / −28.77 %. The savings
   compose multiplicatively with the existing W15
   ``T_decoder``-budget, W21 per-cell N-oracle accounting, W18 /
   W19 / W20 / W21 bundled-context efficiency, AND W22
   shared-read-cache + per-cell digest contributions to a
   meaningful end-to-end reduction in the visible context the
   final decoder pays.

2. **Empirically-discharged probabilistic-cache mitigation.** The
   W22-C-CACHE-AMPLIFICATION conjecture (named in SDK v3.23 as a
   real observed weakness of the W22 cache when paired with
   probabilistic LLM oracles) is discharged on the synthetic
   regime at +0.125 strict gain. This is the **first empirical
   demonstration in the programme that a named conjecture's
   weakness is mitigable** without weakening the conjectured
   strength on other regimes — the quorum-keyed cache preserves
   BYTE_IDENTICAL behaviour on deterministic oracles AND mitigates
   the amplification on probabilistic ones.

3. **First bounded, audited capsule-native dense-control payload.**
   The super-token reference is the **first capsule-native trust
   boundary on a single-visible-token CID-prefix payload** in the
   programme. The verification function rejects unknown / forged /
   tampered references and the explicit-capsule path stays sound.
   The bound on the channel is sharp (one whitespace token per
   cell, ≤ 16 hex chars, registry-enumerable). This is the smallest
   honest version of the LatentMAS *super-token side channel* idea
   that this repo can validate end-to-end — and a structural
   counter-example to the criticism that dense-control side
   channels necessarily smuggle in unaudited coordination.

What W23 does **not** advance:

* The W19-Λ-total wall (no asymmetric witness anywhere in the
  bundle), the W21-Λ-all-compromised wall (every registered oracle
  jointly compromised), and the W22-Λ-no-cache wall (no repeated
  reads to amortise) are real and structural; W23 does not cross
  them. The natural escape remains the same: oracle integrity at
  the registered-set level, or a meta-oracle that audits the
  registered set, or a richer cross-cell bench property that
  amortises shared-state work.
* True transformer-internal KV reuse / latent hidden-state
  transfer between LLM agents is **not implemented** in this
  repo. Every claim about "shared cross-cell latent state" or
  "super-token side channel" in this milestone is at the **capsule
  layer**, with explicit honest proxies (``SessionDigestEnvelope``,
  ``SessionDeltaEnvelope``, ``SuperTokenReferenceEnvelope``); we
  did not modify any model-internal state. If a future programme
  builds true KV-sharing between Apple-Silicon-distributed MLX
  servers, the W23 typed-envelope + verification surface remains a
  useful integration-boundary anchor.
* True two-host execution is **not validated** in this milestone.
  Mac 2 has been ARP-incomplete for 17 milestones in a row; the
  CrossHostProducerDecoderProxy is a structural proxy. The
  cross-host wire bytes are measured (≈ 1.79 KB/cell) but a real
  socket is not.

## Code anchors

| Theorem / claim | Code anchor |
| --------------- | ----------- |
| W23-1 (efficiency) | ``Phase70DeltaFanoutTests::test_delta_savings_loose / _tight``; ``docs/data/phase70_delta_fanout_K12_n8.json``, ``docs/data/phase70_delta_fanout_tight_K12_n8.json`` |
| W23-1 stability   | ``Phase70SeedStabilityTests``; ``docs/data/phase70_seed_sweep_loose_K12_n8.json`` |
| W23-2 (mitigation) | ``Phase70AmplifiedLLMTests``; ``docs/data/phase70_amplified_llm_K12_n8.json`` |
| W23-3 (super-token) | ``Phase70SuperTokenTamperedTests``; ``SuperTokenReferenceTests``; ``docs/data/phase70_super_token_tampered_K12_n8.json`` |
| W23-3 (chain)     | ``Phase70ChainBrokenTests``; ``SessionDigestVerificationTests``; ``docs/data/phase70_chain_broken_K12_n8.json`` |
| W23-3-A (no-trigger / disabled) | ``W23SDKReductionTests`` |
| W23-3-B (regression) | ``test_coordpy_*.py`` + ``test_phase69_*`` + ``test_phase70_*``; 742 / 742 pass |
| W23-3-C (W22 anchor) | ``test_phase69_capsule_latent_hybrid::Phase69CacheFanoutTests`` (passes unchanged) |
| W23-Λ-no-delta    | ``Phase70NoDeltaTests``; ``docs/data/phase70_no_delta_K12_n8.json`` |
| W23-Λ-real        | ``docs/data/phase70_live_mixtral_8x7b_n4.json`` |
| W23-Λ-cross-host  | ``CrossHostProducerDecoderProxyTests``; ``cross_host_round_trip_bytes_total`` field in every R-70 artifact |
| W23-C-MITIGATION-LIVE-VARIANCE | conjectural; observed empirically on mixtral_8x7b live regime |

## Reproducibility

```bash
# R-70-DELTA-FANOUT-LOOSE (W23-1 anchor):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank delta_fanout --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-70-DELTA-FANOUT-TIGHT (W23-1 + W15 composition):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank delta_fanout --decoder-budget 24 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-70-SUPER-TOKEN (W23-3 dense-control anchor):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank super_token --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-70-AMPLIFIED-LLM (W23-2 mitigation anchor):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank amplified_llm --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-70-SUPER-TOKEN-TAMPERED (W23-3 trust falsifier):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank super_token_tampered --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-70-CHAIN-BROKEN (W23-3 trust falsifier):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank chain_broken --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-70-NO-DELTA (W23-Λ-no-delta named falsifier):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank no_delta --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

# Seed-stability sweep:
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank delta_fanout --seed-sweep --K-auditor 12 --n-eval 8 \
    --decoder-budget -1 --out -

# Live LLM probe (W23-Λ-real, mixtral 8x7b on Mac-1):
python3 -m vision_mvp.experiments.phase70_capsule_session_delta \
    --bank amplified_llm --live-llm-adjudicator \
    --adjudicator-model mixtral:8x7b --n-eval 4 --verbose --out -
```

Reproducible from any commit at or after SDK v3.24 with no
configuration beyond the bank and decoder budget knobs.

## Cross-references

* Theorem registry: `docs/THEOREM_REGISTRY.md` (W23 family entry).
* Research status: `docs/RESEARCH_STATUS.md` (axis 20, SDK v3.24).
* Overstatement guard: `docs/HOW_NOT_TO_OVERSTATE.md` § "W23
  forbidden moves".
* Master plan: `docs/context_zero_master_plan.md` § post-W22
  (SDK v3.24).
* Prior milestone (SDK v3.23 W22):
  `docs/RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md`.
* Capsule team-level formalism: `docs/CAPSULE_TEAM_FORMALISM.md`.
