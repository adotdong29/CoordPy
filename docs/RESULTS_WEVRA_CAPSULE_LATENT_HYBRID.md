# SDK v3.23 — capsule + audited latent-state-sharing hybrid (W22 family)

> Theory-forward results note for the post-W21 milestone. This is the
> first capsule-native multi-agent-coordination method that
> *combines* explicit-capsule passing with audited proxies for the
> LatentMAS direction (collective KV pooling / latent hidden-state
> transfer / super-token side channels), measured on a regime where
> repeated reads and verbose audit replays make the combination
> useful. The W21-1 strict-correctness result is preserved
> byte-for-byte; the W22 contribution is on the **efficiency** and
> **trust-boundary** axes. Date stamp: 2026-04-29.

## TL;DR

* **W22-1 (efficiency, proved-empirical, n=8 saturated × 5 seeds × 2
  cells).** On R-69-CACHE-FANOUT, pairing the W21
  ``TrustWeightedMultiOracleDisambiguator`` with the W22
  ``LatentDigestDisambiguator`` over a shared
  ``SharedReadCache`` and a registered
  ``SchemaCapsule`` strictly reduces
  ``mean_n_visible_tokens_to_decider`` over the W21 baseline by
  **−7 tokens/cell (−14.51 %)** at ``T_decoder = None`` and by
  **−7 tokens/cell (−16.09 %)** at ``T_decoder = 24``, AND records
  ``cache_tokens_saved_total = 88`` over the bank (oracle-side
  wire savings), AND ties W21 byte-for-byte on
  ``accuracy_full = 1.000``. Stable across **5/5** alternate
  ``bank_seed`` values (11, 17, 23, 29, 31): savings exactly +7
  tokens/cell on every seed; cache_tokens_saved=88 on every seed.
  Code anchor:
  ``vision_mvp.experiments.phase69_capsule_latent_hybrid.run_phase69``.
  Artifacts: ``docs/data/phase69_cache_fanout_K12_n4.json``,
  ``docs/data/phase69_cache_fanout_tight_K12_n8.json``,
  ``docs/data/phase69_seed_sweep_loose_K12_n8.json``,
  ``docs/data/phase69_cross_regime_synthetic.json``.
* **W22-2 (correctness ratification, proved-empirical byte-for-byte,
  n=8 × 5 seeds).** On every cell of R-69-CACHE-FANOUT, the W22
  ``answer["services"]`` equals the W21 ``answer["services"]``
  byte-for-byte. ``correctness_ratified_rate = 1.000`` on every
  seed. The latent digest is a *summary* of the W21 vote outcome,
  not a re-projection. The W22 layer's correctness is therefore
  exactly W21's correctness on this bench — no covert trade-off.
* **W22-3 (trust-boundary soundness, proved-empirical + proved by
  inspection).** Three failure modes are mechanically tested by
  enumerated falsifiers; on every cell of every falsifier regime,
  the controller-side ``verify_latent_digest`` rejects the
  envelope and the W22 layer falls through to W21 byte-for-byte:
  * **R-69-POISONED-DIGEST** (``EnvelopeTamperer(mode=
    "flip_projected_subset")``):
    ``verification_ok_rate = 0.000`` on every cell,
    rejection reason = ``hash_mismatch``,
    ``correctness_ratified_rate = 1.000`` (W22 answer = W21
    answer byte-for-byte).
  * **R-69-SCHEMA-DRIFT** (verifier registered with a different
    schema CID than the producer's): ``verification_ok_rate =
    0.000``, rejection reason = ``schema_cid_mismatch``,
    ``correctness_ratified_rate = 1.000``.
  * **R-69-NO-TRIGGER** (W21 abstains, e.g. ``W21_BRANCH_NO_QUORUM``):
    W22 fires ``W22_BRANCH_NO_TRIGGER``,
    no envelope is emitted, ``digest_n_tokens = 0``,
    answer field equals W21 byte-for-byte. Backward-compat
    preserved.
  The verification function (``verify_latent_digest``) is short
  (≈ 30 lines), the failure modes are enumerated
  (``"ok"`` / ``"hash_mismatch"`` / ``"schema_cid_mismatch"`` /
  ``"schema_version_unknown"`` / ``"unsealed_parent_probe_cid"`` /
  ``"empty_envelope"``); soundness holds by inspection.
* **W22-Λ-no-cache (proved-empirical, named falsifier).** On
  R-69-NO-CACHE (fresh per-cell ``SharedReadCache`` instances),
  ``cache_tokens_saved_total = 0`` by construction. The
  digest-only contribution to ``visible_tokens_savings_pct``
  (+14.51 %) survives, but the wire-side savings claim is null.
  Names the structural limit when there are no repeated reads to
  amortise.
* **W22-Λ-real (proved-conditional + empirical-research, n=4 × 2
  models, partially discharged).** Live W22 transfer on Mac-1
  Ollama:
    * **mixtral:8x7b** (47B-MoE) as a fourth registered LLM
      adjudicator on cache_fanout: visible-tokens savings
      **+39.08 %** (W21 = 87.0, W22 = 53.0 tokens/cell);
      cache_tokens_saved_total = 120 over 4 cells; verification
      ok rate = 0.750. Correctness ratified rate = 0.750 — the
      one cell that drifts is a sharp finding: **the W22 cache
      amplifies LLM nondeterminism**. Cells with identical
      OutsideQueries replay the *first* mixtral reply; if cell-1's
      reply was decoy-asymmetric, all subsequent matching cells
      inherit that bad vote. This is the first capsule-native
      benchmark to *measure* this interaction and name it (the
      conjecture is **W22-C-CACHE-AMPLIFICATION** — the cache
      amplifies the variance of probabilistic adjudicator
      replies, both right and wrong). Artifact:
      ``docs/data/phase69_live_mixtral_8x7b_n4.json``.
    * **gemma2:9b** (9.2B-dense) as the fourth registered LLM
      adjudicator: every strategy ties at ``acc_full = 0.250``;
      W22 ties W21 byte-for-byte (correctness ratified rate =
      1.000). The ratification holds because gemma2's bad reply
      drives W21 itself to a SYMMETRIC_QUORUM abstention, which
      W22 inherits unchanged (W22-3-A backward-compat path).
      Visible-tokens savings = 4.90 % — modest because most
      envelopes are skipped (NO_TRIGGER on abstention). Artifact:
      ``docs/data/phase69_live_gemma2_9b_n4.json``.
* **Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
  ``incomplete`` at milestone capture — same status as SDK v3.6
  through SDK v3.22 (16th milestone in a row). **No two-Mac
  sharded inference happened in SDK v3.23.** The W22 mechanism is
  *naturally* a producer / cache-controller separation: the
  ``SharedReadCache`` + ``LatentDigestDisambiguator`` interface is
  wire-compatible with cross-host deployment (cache on Mac-1,
  decoder on Mac-2, oracles on either) — no W22 code changes
  required when Mac-2 returns. Strongest model class actually
  exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE Q4) on Mac-1
  Ollama.
* **Bounded-context honesty preserved byte-for-byte.** The W22
  layer reads only what the W21 layer below it produced; the
  ``W15`` ``tokens_kept`` is byte-for-byte identical to W21 (and
  to W19, W18). Mechanically verified:
  * ``Phase69CacheFanoutTests::test_w22_no_accuracy_regression_on_cache_fanout``
    — W22 ties W21 at 1.000.
  * ``Phase69PoisonedDigestTests::test_poisoned_digest_no_visible_tokens_savings``
    — on rejection, the W22 visible-tokens cost equals the W21
    baseline (no covert savings claimed).
  * Total context delivered to the final decider on the synthetic
    anchor: ``W21 = 48.2 tokens/cell``, ``W22 = 41.2 tokens/cell``
    (loose budget); ``W21 = 43.5``, ``W22 = 36.5`` (tight budget).
* **Backward-compat preserved byte-for-byte.**
  * **W22-3-A** (vs W21, no-trigger paths). With ``enabled = False``
    OR ``schema = None`` OR an inner W21 branch outside
    ``trigger_branches``, the W22 layer reduces to W21
    byte-for-byte on the answer field (mechanically verified by
    ``Phase69NoTriggerTests``,
    ``W22SDKReductionTests::test_disabled_reduces_to_w21``).
  * **W22-3-B** (full programme regression). 633 / 633 prior
    wevra tests pass before the W22 milestone landed; **675 /
    675** wevra-suite tests pass after (+ 32 new W22 tests + 10
    misc).
* **Audit T-1..T-7 preserved on every cell of every regime.**
  ``Phase69AuditOKTests::test_audit_ok_on_every_w22_cell``
  asserts ``audit_ok_grid["capsule_w22_hybrid"] == True`` on
  cache_fanout, no_cache, poisoned_digest, schema_drift.

## What changed from SDK v3.22 → v3.23 (one paragraph)

The W21 family (SDK v3.22) crossed the W20-Λ-compromised wall by
consulting **N registered oracles** under quorum + trust thresholds
and projecting the answer to the quorum-aligned subset. It
explicitly named the **wire-cost** of consulting all N oracles per
cell as a separate research direction (W21-C-CALIBRATED-TRUST in
the *correctness* direction; the *cost* direction was undischarged).
SDK v3.23 implements the smallest version of the cost-direction
move *in combination with* the LatentMAS *latent hidden-state
transfer + collective KV pooling + super-token side channel* idea
families: a CID-keyed shared-read cache that collapses identical
oracle queries to one wire-side call, plus a controller-verified
latent-digest envelope that compresses the verbose W21 audit
into a single typed line, plus a content-addressed schema capsule
that is shared once per session and referenced by CID across cells.
Every latent payload is hash-chained, schema-versioned, and
parent-CID-sealed; on verification failure the W22 layer rejects
and the explicit-capsule path stays sound. The W22 surface is
purely additive on top of the W21 surface; the SDK v3.22 runtime
contract is byte-for-byte unchanged.

## Theorem family W22 (minted by this milestone)

We pre-commit eight W22 statements. Three are **proved-empirical**
(saturated against pre-committed seeds); two are **proved by
inspection + mechanically-checked**; two are **proved-empirical
backward-compat anchors**; one is **conjectural**
(**W22-C-CACHE-AMPLIFICATION**) and one is **proved-conditional +
empirical-research** (**W22-Λ-real**). Codebase status — all
numbered code paths land in
``vision_mvp/wevra/team_coord.py`` (``SchemaCapsule`` /
``LatentDigestEnvelope`` / ``LatentVerificationOutcome`` /
``verify_latent_digest`` / ``SharedReadCache`` /
``CachingOracleAdapter`` / ``EnvelopeTamperer`` /
``W22LatentResult`` / ``LatentDigestDisambiguator``)
and ``vision_mvp/experiments/phase69_capsule_latent_hybrid.py``
(R-69 driver + cross-regime + seed-stability sweeps).

* **W22-1** *(proved-conditional + proved-empirical n=8 saturated × 5
  seeds × 2 cells)*. On R-69-CACHE-FANOUT-LOOSE and
  R-69-CACHE-FANOUT-TIGHT, pairing the W21
  ``TrustWeightedMultiOracleDisambiguator`` with the W22
  ``LatentDigestDisambiguator`` over a registered SchemaCapsule + a
  shared SharedReadCache strictly reduces
  ``mean_n_visible_tokens_to_decider`` over the W21 baseline AND
  records ``cache_tokens_saved_total > 0`` AND preserves
  ``accuracy_full`` byte-for-byte. Stable across 5/5 ``bank_seed``
  values ``(11, 17, 23, 29, 31)``. Mechanically verified by
  ``Phase69CacheFanoutTests::test_w22_strict_visible_tokens_savings_loose``,
  ``Phase69CacheFanoutTests::test_w22_strict_visible_tokens_savings_tight``,
  ``Phase69SeedStabilityTests::test_savings_strictly_positive_across_5_seeds``.
  *Conditions* (any failure collapses the result):
    1. The bench property — at least two cells share an
       OutsideQuery + oracle_id pair (else the cache cannot hit).
    2. The inner W21 branch is in ``trigger_branches`` (i.e. fires
       ``W21_BRANCH_QUORUM_RESOLVED``).
    3. The schema is registered AND the verifier's schema CID
       matches the producer's signed CID.
* **W22-2** *(proved-empirical byte-for-byte, n=8 × 5 seeds)*.
  Correctness ratification. On every cell of R-69-CACHE-FANOUT, the
  W22 ``answer["services"]`` equals the W21 ``answer["services"]``
  byte-for-byte. ``correctness_ratified_rate = 1.000`` on every
  seed (5/5). The latent digest is a *summary* of the W21 vote
  outcome — the projection rule (positive-set under quorum + trust
  thresholds) is computed in W21 below; W22 emits a compact
  representation of the same projection with provenance. W22's
  correctness is exactly W21's correctness on this bench.
  Mechanically verified by
  ``Phase69CorrectnessRatificationTests::test_w22_correctness_ratified_byte_for_byte``.
* **W22-3** *(proved-empirical n=8 + proved by inspection)*.
  Trust-boundary soundness. On R-69-POISONED-DIGEST and
  R-69-SCHEMA-DRIFT, every tampered envelope is rejected by
  ``verify_latent_digest``; the W22 layer fires
  ``W22_BRANCH_LATENT_REJECTED`` on every cell;
  ``verification_ok_rate = 0.000``;
  ``correctness_ratified_rate = 1.000`` (the W22 answer field
  equals the W21 answer field byte-for-byte). The verification
  function is short (≈ 30 lines) and the failure modes are
  enumerated; soundness holds by inspection. Mechanically verified
  by ``Phase69PoisonedDigestTests::test_poisoned_digest_rejected_on_every_cell``,
  ``Phase69SchemaDriftTests::test_schema_drift_rejected_on_every_cell``,
  ``Phase69PoisonedDigestTests::test_poisoned_digest_rejection_reason_is_hash_mismatch``,
  ``Phase69SchemaDriftTests::test_schema_drift_rejection_reason``,
  ``LatentDigestVerificationTests`` (six unit tests, one per
  failure-mode label).
* **W22-3-A** *(proved-empirical n=8, backward-compat anchor)*.
  When the inner W21 branch is not in ``trigger_branches`` (W21
  abstained for any reason — NO_QUORUM / NO_ORACLES / NO_TRIGGER /
  DISABLED), the W22 layer fires ``W22_BRANCH_NO_TRIGGER`` and
  reduces to W21 byte-for-byte on the answer field. With
  ``enabled = False`` OR ``schema = None``, the W22 layer reduces
  to W21 byte-for-byte. Mechanically verified by
  ``Phase69NoTriggerTests::test_no_trigger_reduces_to_w21``,
  ``W22SDKReductionTests::test_disabled_reduces_to_w21``.
* **W22-3-B** *(proved-empirical full programme regression,
  675 / 675 wevra-suite tests pass)*. On R-54..R-68 default banks,
  the W22 layer ties the W21 layer byte-for-byte either via
  ``W22_BRANCH_NO_TRIGGER`` or via ``W22_BRANCH_LATENT_RESOLVED``
  when both fire on the same regime. With ``enabled = False`` OR
  no schema registered, W22 reduces to W21. The full pre-existing
  W21 / W20 / W19 / W18 / W17 / W16 / W15 / W14 / W13 / W12 / W11
  / W10 / W9 / W8 / W7 / W6 / W5 / W4 / W3 test suites all pass
  (633 prior + 32 new W22 + 10 misc = 675 total).
* **W22-Λ-no-cache** *(proved-empirical n=8 saturated)*. On
  R-69-NO-CACHE (fresh per-cell ``SharedReadCache``),
  ``cache_tokens_saved_total = 0``;
  ``cache_hit_rate = 0.000``. The digest's contribution to
  ``visible_tokens_savings_per_cell`` (+7 tokens/cell, +14.51 %)
  survives, but the wire-side savings claim of W22-1 does NOT
  hold. Names the structural limit when there are no repeated
  reads to amortise. Mechanically verified by
  ``Phase69NoCacheTests::test_no_cache_records_zero_cache_tokens_saved``,
  ``Phase69NoCacheTests::test_no_cache_still_has_digest_savings``.
* **W22-Λ-real** *(proved-conditional + empirical-research, n=4 × 2
  models)*. Live-LLM transfer on Mac-1 Ollama:
  * **mixtral:8x7b** (47B-MoE): ``visible_tokens_savings_pct =
    +39.08 %``; ``cache_tokens_saved_total = 120``;
    ``verification_ok_rate = 0.750``;
    ``correctness_ratified_rate = 0.750``. The 0.250 gap between
    cached-W21 (0.750) and uncached-W21 (1.000) reveals **W22-C-
    CACHE-AMPLIFICATION** (next): the cache returns cell-1's
    mixtral reply for every subsequent matching cell; if cell-1's
    reply was decoy-asymmetric, all matching cells inherit that
    bad vote. This is a **real** finding — at temperature=0
    mixtral is *probabilistic* not deterministic across separate
    sessions, and the cache freezes the first sample. Artifact:
    ``docs/data/phase69_live_mixtral_8x7b_n4.json``.
  * **gemma2:9b** (9.2B-dense): every strategy ties at
    ``acc_full = 0.250`` (gemma2's closure-landing rate is the
    structural bound, identical to SDK v3.22 W21-Λ-real
    coalition); W22 ties W21 byte-for-byte
    (``correctness_ratified_rate = 1.000``);
    ``visible_tokens_savings_pct = +4.90 %`` (modest because most
    envelopes are skipped via NO_TRIGGER on abstention).
    Artifact: ``docs/data/phase69_live_gemma2_9b_n4.json``.
* **W22-C-CACHE-AMPLIFICATION** *(conjectural, named for the first
  time in this milestone)*. When a probabilistic LLM oracle is
  registered alongside deterministic oracles in a W22 stack with a
  ``SharedReadCache``, the cache returns the *first* recorded
  reply for every subsequent identical OutsideQuery. This freezes
  the LLM's first sample as the canonical reply across the
  session. **Conjecture**: in expectation over (model, prompt)
  pairs at temperature=0, this *amplifies* the variance of the
  per-cell vote outcome — bad first replies poison many cells; good
  first replies stabilise many cells. **Falsifier**: an LLM
  adjudicator that is byte-stable across separate sessions (true
  determinism + identical sampling state) would produce
  ``correctness_ratified_rate = 1.000`` on every cell. Empirical
  observation in this milestone: ``mixtral:8x7b`` at temperature=0
  on Mac-1 Ollama is *not* byte-stable across cells (3/4 cells
  ratified, 1/4 drifts); the cache amplifies this drift.

## Honest scope (W22)

The W22-1 strict gain is *strongly conditional* on:

1. The bench property — at least two cells share an OutsideQuery
   + oracle_id pair (else the cache cannot hit).
2. ≥ ``quorum_min`` of the registered oracles return non-empty-
   proper-asymmetric replies whose token sets find admitted-tag
   mentions through the W18 / W19 closure (the W21 trigger
   condition).
3. The schema is registered AND the verifier's schema CID matches
   the producer's signed CID (the W22 trigger condition).

Three named falsifiers make the conditionality sharp:

* **W22-Λ-no-cache** — fresh per-cell cache → no wire-side savings.
* **R-69-POISONED-DIGEST** — tampered envelope → controller rejects
  → W22 falls through to W21 baseline cost.
* **R-69-SCHEMA-DRIFT** — verifier's registered schema differs from
  producer's signed CID → controller rejects → fall through.

The W22 escape from the W21 wire-cost concern is *partial* by
design: bounded above by (a) the cache-hit rate (which depends on
cross-cell OutsideQuery overlap), (b) the digest compression ratio
(which depends on the verbose-audit / digest size delta — small
on tiny vote outcomes, larger on big oracle sets), AND (c) the
verification path success rate (rejected envelopes pay full W21
baseline cost).

What W22 does **NOT** do (do-not-overstate):

* Does NOT touch transformer KV caches, embedding tables,
  attention weights, or any model-internal state. The "shared
  cache" lives at the **capsule layer**; it is an honest **proxy**
  for the LatentMAS shared-KV-read direction, not a runtime KV
  transplant.
* Does NOT hide unaudited coordination behind opaque latent
  payloads. Every envelope carries a content hash, a schema CID,
  a parent-CID list, a closed-vocabulary projection, and a
  human-readable canonical encoding. The verification check is
  short (≈ 30 lines) and the failure modes are enumerated.
* Does NOT improve correctness over W21 on the synthetic
  R-69-CACHE-FANOUT anchor. W22's correctness is exactly W21's
  by construction (Theorem W22-2). The load-bearing contribution
  is on the **efficiency** and **trust-boundary** axes.
* Does NOT solve "multi-agent context" in any unqualified sense.
  The W22-1 win is *strongly conditional* on the named bench
  property; under the W22-Λ-no-cache, R-69-POISONED-DIGEST,
  R-69-SCHEMA-DRIFT, and R-69-NO-TRIGGER falsifiers the wire-side
  savings claim collapses or the digest is rejected. See
  ``HOW_NOT_TO_OVERSTATE.md`` § "W22 forbidden moves".

## Prior-conjecture discharge ledger (SDK v3.23)

* **W21-C-CALIBRATED-TRUST** *(SDK v3.22, named)*. Open
  conjecturally; the **wire-cost direction** of the multi-oracle
  consult is *partially discharged* by the W22 cache. The
  *correctness direction* (low trust priors on uncalibrated
  oracles) remains open and is orthogonal to W22.
* **W21-C-LIVE-WITH-REGISTRY** *(SDK v3.22, partially discharged)*.
  Extended by W22-Λ-real on cache_fanout: the live mixtral
  adjudicator on a four-oracle stack with cache delivers
  ``visible_tokens_savings_pct = +39.08 %`` on the same regime
  the SDK v3.22 W21-C-LIVE-WITH-REGISTRY conjecture targeted; the
  cache amplifies LLM nondeterminism (W22-C-CACHE-AMPLIFICATION
  newly named).

## What this means for the original goal

The original goal of the Context Zero programme is *solving
context for multi-agent teams* — not solving any one regime, but
demonstrating that a typed, content-addressed, lifecycle-bounded
capsule discipline can carry the load that ad-hoc context-passing
in production multi-agent systems cannot.

The W22 family advances that goal on **two new axes** beyond the
W17-W21 result family:

1. **Wire-cost efficiency** — the multi-oracle adjudication wall
   crossed by W21 (SDK v3.22) was named with a known cost
   concern: consulting all N oracles every cell is expensive when
   N grows or when oracles are LLM-shaped. The W22 cache + digest
   discharges that cost concern *partially* on regimes where
   identical OutsideQueries recur — a structural property real
   multi-agent runs frequently exhibit (e.g. same root_cause
   family across multiple incidents). The 14.51 % synthetic
   savings + 39.08 % live-mixtral savings are modest individually
   but compose with the existing W15 ``T_decoder``-budget, W21
   per-cell N-oracle accounting, and W18 / W19 / W20 / W21
   bundled-context efficiency to a meaningful end-to-end
   reduction in the visible context the final decoder pays.
2. **Audited trust boundary** — every prior step in the W17-W21
   ladder added a new piece of evidence the explicit-capsule path
   trusted: round-2 disambiguator (W18), bundle-internal witness
   (W19), single outside oracle (W20), N outside oracles under
   quorum (W21). Each step was bounded by an integrity assumption.
   W22 adds the *first capsule-native trust boundary on
   compressed / hidden-state-shaped payloads* in the programme:
   the verification function rejects tampered envelopes and the
   explicit-capsule path stays sound. This is the part of the
   LatentMAS direction that has been most-criticised as
   smuggling-in unaudited coordination; W22 implements the
   smallest honest counter-example.

What W22 does **not** advance:

* The W19-Λ-total wall (no asymmetric witness anywhere in the
  bundle) and the W21-Λ-all-compromised wall (every registered
  oracle jointly compromised) are real and structural; W22 does
  not cross them. The natural escape remains the same: oracle
  integrity at the registered-set level (W21-C-CALIBRATED-TRUST)
  or a meta-oracle that audits the registered set (open).
* True transformer-internal KV reuse / latent hidden-state
  transfer between LLM agents is **not implemented** in this
  repo. Every claim about "shared KV" or "latent state" in this
  milestone is at the **capsule layer**, with explicit honest
  proxies (``SharedReadCache``, ``LatentDigestEnvelope``); we did
  not modify any model-internal state. If a future programme
  builds true KV-sharing between Apple-Silicon-distributed MLX
  servers, the W22 typed-envelope + verification surface remains
  a useful integration-boundary anchor.

## Code anchors

| Theorem / claim | Code anchor |
| --------------- | ----------- |
| W22-1 (efficiency) | ``Phase69CacheFanoutTests::test_w22_strict_visible_tokens_savings_loose / _tight``; ``docs/data/phase69_cache_fanout_K12_n4.json`` |
| W22-1 stability   | ``Phase69SeedStabilityTests``; ``docs/data/phase69_seed_sweep_loose_K12_n8.json`` |
| W22-2 (ratification) | ``Phase69CorrectnessRatificationTests``; ``docs/data/phase69_cache_fanout_K12_n4.json`` |
| W22-3 (poisoned)  | ``Phase69PoisonedDigestTests``; ``LatentDigestVerificationTests`` |
| W22-3 (drift)     | ``Phase69SchemaDriftTests``; ``LatentDigestVerificationTests`` |
| W22-3-A (no-trigger) | ``Phase69NoTriggerTests``; ``W22SDKReductionTests`` |
| W22-3-B (regression) | ``test_wevra_*.py`` full suite; 675 / 675 pass |
| W22-Λ-no-cache    | ``Phase69NoCacheTests`` |
| W22-Λ-real        | ``docs/data/phase69_live_mixtral_8x7b_n4.json``, ``docs/data/phase69_live_gemma2_9b_n4.json`` |
| W22-C-CACHE-AMPLIFICATION | conjectural; observed empirically on mixtral_8x7b live regime |

## Reproducibility

```bash
# R-69-CACHE-FANOUT-LOOSE (W22-1 anchor):
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --bank cache_fanout --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-69-CACHE-FANOUT-TIGHT (W22-1 + W15 composition):
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --bank cache_fanout --decoder-budget 24 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-69-POISONED-DIGEST (W22-3 trust-boundary anchor):
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --bank poisoned_digest --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# R-69-SCHEMA-DRIFT (W22-3 trust-boundary anchor):
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --bank schema_drift --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-seed 11 --verbose --out -

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

# Seed-stability sweep:
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --bank cache_fanout --seed-sweep --K-auditor 12 --n-eval 8 --out -

# Live LLM probe (W22-Λ-real, mixtral 8x7b on Mac-1):
python3 -m vision_mvp.experiments.phase69_capsule_latent_hybrid \
    --bank cache_fanout --live-llm-adjudicator \
    --adjudicator-model mixtral:8x7b --n-eval 4 --verbose --out -
```

Reproducible from any commit at or after SDK v3.23 with no
configuration beyond the bank and decoder budget knobs.

## Cross-references

* Theorem registry: `docs/THEOREM_REGISTRY.md` (W22 family entry).
* Research status: `docs/RESEARCH_STATUS.md` (axis 19, SDK v3.23).
* Overstatement guard: `docs/HOW_NOT_TO_OVERSTATE.md` § "W22
  forbidden moves".
* Master plan: `docs/context_zero_master_plan.md` § post-W21
  (SDK v3.23).
* Prior milestone (SDK v3.22 W21):
  `docs/RESULTS_WEVRA_MULTI_ORACLE_ADJUDICATION.md`.
* Capsule team-level formalism: `docs/CAPSULE_TEAM_FORMALISM.md`.
