# Results — bundle-relational compatibility disambiguator + symmetric-ambiguity benchmark family (SDK v3.19, W18 family)

> Milestone note for the SDK v3.19 push: the **first capsule-native
> multi-agent-coordination method that crosses the symmetric-
> corroboration wall** (W17-Λ-symmetric, named in SDK v3.18) **on a
> regime where the wall actually applies**. The W18 layer is
> *additive in code* (one new dataclass + one tokeniser + one
> closed-form scorer + one wrapping decoder; the SDK v3.18 runtime
> contract is byte-for-byte unchanged) and *load-bearing in effect*:
> on the synthetic R-65-COMPAT regime (every gold service AND the
> decoy mentioned by ≥ 2 distinct routed roles via generic-noise
> kinds with comparable magnitudes — symmetric-corroboration; round-2
> specific-tier disambiguator's payload contains a relational-compound
> mention of *every gold service* AND *no decoy service*), every
> closed-form salience scorer in the SDK ties FIFO at 0.000 by
> construction (W17-Λ-symmetric extended to R-65 verbatim); the new
> W18 ``RelationalCompatibilityDisambiguator`` strictly improves
> ``accuracy_full`` by **+1.000** over every non-W18 capsule
> baseline at both ``T_decoder = None`` AND ``T_decoder = 24``,
> stable across **5/5** alternate ``bank_seed`` values.
>
> Last touched: SDK v3.19, 2026-04-28.

## TL;DR

* **W18-1 strict gain on R-65-COMPAT-LOOSE** (synthetic identity
  producer, ``T_decoder = None``, ``K_auditor = 12``, n=8 ×
  bank_replicates=2; ``bank_seed = 11``). The new
  :class:`RelationalCompatibilityDisambiguator` (composing the W15
  :class:`AttentionAwareBundleDecoder` with a deterministic
  bundle-relational scorer over the round-2 disambiguator's payload
  text) achieves ``capsule_relational_compat = 1.000``; every other
  capsule strategy in the SDK — substrate FIFO,
  ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
  W7-2 cohort, W8 corroboration, W9 multi-service, W11 multi-round,
  W12 robust, W13 layered, W15 attention-aware,
  ``capsule_layered_fifo_packed`` — ties FIFO at ``0.000``.
  **+1.000 strict separation**.
* **W18-1 strict gain on R-65-COMPAT-TIGHT** (same regime under
  decoder-side budget pressure, ``T_decoder = 24``). W18 = 1.000;
  every other capsule strategy = 0.000. **+1.000** vs the W14H +
  W15 composition. The W18 method composes *cleanly* with the W15
  attention-aware pack: the W18 scorer reads only the W15-packed
  bundle (``tokens_kept_sum`` and ``handoffs_decoder_input_sum``
  are byte-for-byte identical to W15's). Bounded-context honesty
  preserved.
* **W18-1 5-seed stability**: gap ``w18 − attention_aware``
  saturates at **+1.000** on every seed in
  ``{11, 17, 23, 29, 31}`` for both ``T_decoder ∈ {None, 24}``.
  Min/max/mean = +1.000 / +1.000 / +1.000 — well above the 0.50
  strong-bar threshold.
* **W18-Λ-no-compat falsifier on R-65-NO-COMPAT** (round-2
  disambiguator carries no service-tag mention). W18 abstains
  uniformly; ``capsule_relational_compat = 0.000`` on 8/8 cells.
  Every closed-form salience packer also at 0.000 (W17-Λ-symmetric).
  Names the structural limit when round-2 evidence carries no
  relational compatibility cue.
* **W18-Λ-confound falsifier on R-65-CONFOUND** (round-2 mentions
  BOTH gold AND decoy). W18 abstains uniformly (every admitted tag
  has positive score); ``capsule_relational_compat = 0.000`` on
  8/8 cells. Names the structural limit when round-2 evidence is
  itself relationally-symmetric.
* **W18-Λ-deceive falsifier on R-65-DECEIVE** (round-2 mentions
  decoy but NOT gold). W18 trusts its evidence and projects to the
  decoy-only set; ``services_correct = False`` (gold missing);
  ``capsule_relational_compat = 0.000`` on 8/8 cells. Names the
  structural limit when round-2 evidence is adversarially-
  relational; no closed-form scorer that *trusts* the disambiguator
  payload can escape this regime without an outside-information
  axis (W18-C-OUTSIDE, conjectural).
* **W18-3 backward-compat preserved**: on R-58 default (single-
  decoy R-58 anchor) the W18 method ties
  ``AttentionAwareBundleDecoder`` byte-for-byte on the answer field
  — either via abstention (no relational mention) or via strict-
  asymmetric projection that lands on the same gold-only set. On
  R-64-SYM (the SDK v3.18 W17-Λ-symmetric anchor), W18 partially
  recovers — on the deadlock scenarios the round-2 disambiguator
  carries a ``relation=A_B_join`` mention and W18 elects gold; on
  pool / disk / slow_query the round-2 disambiguator carries no
  service-tag mention and W18 abstains. R-64-SYM thus motivates
  R-65-COMPAT as the *consistent* relational-mention regime.
* **Audit T-1..T-7 preserved on every cell of every regime**.
* **Backward-compat byte-for-byte**: the SDK v3.18 442/442 prior
  tests pass on the W18 surface; with ``enabled = False`` the
  W18 method reduces to W15 byte-for-byte.

## 1. Honest scope (what this is and is not)

The W18-1 result is a strong synthetic strict-gain anchor on a
named harder regime (R-65-COMPAT) where the W17-Λ-symmetric wall
binds. **It is not "we solved multi-agent context."** Three
honest caps:

* **Synthetic regime.** R-65-COMPAT uses an
  :class:`IdentityExtractor` for the producer; the relational-
  mention convention in round-2 is a closed-vocabulary contract
  the synthetic bench enforces. Real-LLM transfer of the W18
  method (W18-Λ-real) is **proved-conditional** on the LLM
  emitting the same relational-compound forms the synthetic
  bench uses; if the LLM emits free-form natural-language
  relational mentions (e.g. ``"the join between orders and
  payments"``), the W18 exact-match layer misses by construction.
  The natural extension is **W18-C-LEARNED** (a small distilled
  bundle-relational scorer over capsule bundles), conjectural.
* **Three named falsifiers.** R-65-NO-COMPAT (no signal),
  R-65-CONFOUND (symmetric signal), R-65-DECEIVE (adversarial
  signal). On all three the W18 method ties FIFO or fails by
  construction. The W18-1 conditionality is sharp.
* **Outside-information axis remains conjectural**
  (W18-C-OUTSIDE). On R-65-DECEIVE, no closed-form bundle-
  relational scorer that trusts its evidence can escape the
  adversarial regime without cross-referencing an outside source.
  The natural extension — service-graph topology, prior reliability
  scores, cross-incident historical evidence — is conjectural and
  out of scope for SDK v3.19.

## 2. The named structural ingredient

The W17-Λ-symmetric wall is the regime where every closed-form
salience scorer in the SDK (every cross-role corroboration metric,
every magnitude weight, every CCK filter, every round-index hint)
produces an *identical* salience score for gold and decoy. The
bipartite ``(role × tag, kind, magnitude)`` multiset is — by
construction — symmetric. No service-blind admission AND no
closed-form salience packer can prefer gold over decoy on this
regime; the structural argument is the same as W17-Λ-symmetric
in `docs/RESULTS_WEVRA_LIVE_COMPOSITION.md` § 4.

What R-65-COMPAT adds is a *new information channel* outside the
bipartite multiset: the **payload text** of the round-2 specific-
tier disambiguator. On R-65-COMPAT, the round-2 disambiguator's
payload contains a relational-compound mention of every gold
service tag (``relation=A_B_join``, ``pool_chain=A_B``,
``mount=/storage/A/B``, ``query_path=svc_A_then_svc_B``) AND no
decoy service tag. This is the ingredient the closed-form salience
scorer cannot read — it operates on ``(role, tag, kind, magnitude,
round)`` tuples, not on payload-text content.

The W18 method consumes this new channel directly.

## 3. The W18 method (closed-form, deterministic, training-free)

The W18 :class:`RelationalCompatibilityDisambiguator` is a four-stage
pipeline:

1. **Inner W15 decode.** Run the W15 :class:`AttentionAwareBundleDecoder`
   over the admitted bundle. Capture the inner ``(root_cause, services,
   remediation)`` answer, the W15 first-pass tentative root_cause,
   and the W15 pack-stats block.
2. **Disambiguator selection.** Identify the round-2 specific-tier
   disambiguator(s) in the bundle (any handoff whose ``claim_kind``
   is in :data:`_SPECIFIC_TIER_CLAIM_KINDS`). When multiple
   specific-tier handoffs exist, concatenate their payloads in
   canonical (round, source_role, claim_kind, payload_sha) order so
   the input to the scorer is byte-deterministic.
3. **Tokenise + score.** Run :func:`_disambiguator_payload_tokens`
   on the disambiguator payload (lower-cased, split on non-
   identifier chars, compound identifiers preserved). For each
   service tag in the *union* of admitted service tags (not the
   inner's filtered set — the union is the candidate space, since
   the W11 contradiction-aware drop fires symmetrically and drops
   every tag on R-65-COMPAT), compute
   :func:`_relational_compatibility_score` — the
   ``(direct_hits, compound_hits)`` pair where direct_hits counts
   standalone-identifier matches and compound_hits counts
   contiguous-subsequence matches inside compound identifiers.
4. **Project the answer.** Apply the *strict-asymmetric branch*:
   if at least one but not all admitted tags have positive
   compatibility score, keep only the positive-score tags;
   otherwise abstain (fall through to the inner answer
   byte-for-byte).

The whole pipeline is closed-form, deterministic, and training-
free. Token-budget honesty: the scorer reads only the W15-packed
bundle — no extra capsule reads, no global state — so the W15
``tokens_kept`` accounting is byte-for-byte preserved.

### Why "strict-asymmetric" — and not "argmax"

The strict-asymmetric branch is the load-bearing structural
property. Two alternative designs were rejected:

* **Argmax-only.** Picking the single highest-scoring tag would
  fail R-65-COMPAT (gold has *two* tags A and B; the auditor's
  ``services_correct`` check is set-equality, not single-tag).
* **Threshold-based.** Picking every tag whose score exceeds a
  threshold would fail on R-65-CONFOUND and R-65-DECEIVE: a
  threshold tuned to admit gold also admits the symmetrically-
  corroborated decoy; a threshold tuned to reject decoy also
  rejects partial gold mentions.

The strict-asymmetric branch ("at least one but not all") is the
unique closed-form rule that is *correct* on R-65-COMPAT (gold
positive, decoy zero) AND *defensive* on R-65-CONFOUND (every tag
positive → abstain) AND *defensive* on R-65-NO-COMPAT (every tag
zero → abstain). It is *not* defensive on R-65-DECEIVE (decoy
positive, gold zero); that regime is the named structural limit.

### Compound-target semantics — contiguous subsequence

Service tags that themselves contain underscores (e.g.
``logs_pipeline``, ``db_query``) require contiguous-subsequence
matching against compound tokens. The W18 scorer matches
``db_query`` inside ``svc_web_then_svc_db_query`` because
``["db", "query"]`` is a contiguous subsequence of
``["svc", "web", "then", "svc", "db", "query"]``. It does NOT
match ``orders_join`` inside ``orders_payments_join`` because
``["orders", "join"]`` is not contiguous in
``["orders", "payments", "join"]``. This semantic is the
load-bearing closed-form generalisation that lets one scorer
handle every R-65 scenario family (deadlock / pool / disk /
slow_query) with a single closed-vocabulary relational-mention
contract.

## 4. Headline tables

### 4.1 R-65-COMPAT-LOOSE (synthetic, ``T_decoder = None``)

Pre-committed config: ``K_auditor = 12``, ``T_auditor = 256``,
``n_eval = 8``, ``bank_seed = 11``, ``bank_replicates = 2``.

| strategy                          | accuracy_full | accuracy_root_cause | accuracy_services |
| --------------------------------- | ------------- | ------------------- | ----------------- |
| substrate                         | 0.000         | 1.000               | 0.000             |
| capsule_fifo                      | 0.000         | 1.000               | 0.000             |
| capsule_priority                  | 0.000         | 1.000               | 0.000             |
| capsule_coverage                  | 0.000         | 1.000               | 0.000             |
| capsule_cohort_buffered (W7-2)    | 0.000         | 1.000               | 0.000             |
| capsule_corroboration (W8)        | 0.000         | 1.000               | 0.000             |
| capsule_multi_service (W9)        | 0.000         | 1.000               | 0.000             |
| capsule_multi_round (W11)         | 0.000         | 1.000               | 0.000             |
| capsule_robust_multi_round (W12)  | 0.000         | 1.000               | 0.000             |
| capsule_layered_multi_round (W13) | 0.000         | 1.000               | 0.000             |
| capsule_layered_fifo_packed       | 0.000         | 1.000               | 0.000             |
| capsule_attention_aware (W15)     | 0.000         | 1.000               | 0.000             |
| **capsule_relational_compat (W18)** | **1.000**   | **1.000**           | **1.000**         |

`headline_gap`:
* `w18_minus_attention_aware = +1.000`
* `w18_minus_layered = +1.000`
* `w18_minus_fifo = +1.000`
* `w18_minus_substrate = +1.000`
* `max_non_w18_accuracy_full = 0.000`

Audit T-1..T-7 OK on every capsule strategy. Data file:
`docs/data/phase65_cross_regime_synthetic.json::r65_compat_loose`.

### 4.2 R-65-COMPAT-TIGHT (synthetic, ``T_decoder = 24``)

Same configuration as 4.1 plus a strict decoder-side token budget.
The W18 method consumes only the W15-packed bundle.

| strategy                          | accuracy_full |
| --------------------------------- | ------------- |
| substrate                         | 0.000         |
| capsule_fifo                      | 0.000         |
| capsule_attention_aware (W15)     | 0.000         |
| capsule_layered_fifo_packed       | 0.000         |
| **capsule_relational_compat (W18)** | **1.000**   |

Pack-stats summary (W18 vs W15):

| metric                      | capsule_attention_aware | capsule_relational_compat |
| --------------------------- | ----------------------- | ------------------------- |
| `tokens_kept_sum`           | identical               | identical                 |
| `handoffs_decoder_input_sum` | identical              | identical                 |
| `tokens_kept_over_input`    | identical               | identical                 |

Bounded-context honesty: the W18 method reads only the W15-packed
bundle — no extra capsule reads, no inflation of `tokens_kept`.
Data file: `docs/data/phase65_cross_regime_synthetic.json::r65_compat_tight`.

### 4.3 R-65 falsifiers — W18-Λ-no-compat / -confound / -deceive

| regime               | W18 acc_full | inner W15 acc_full | classification         |
| -------------------- | ------------ | ------------------ | ---------------------- |
| R-65-NO-COMPAT       | 0.000        | 0.000              | abstain → tie FIFO     |
| R-65-CONFOUND        | 0.000        | 0.000              | abstain → tie FIFO     |
| R-65-DECEIVE         | 0.000        | 0.000              | pick decoy → fail      |

The three falsifiers cover the three structural failure modes:

* **NO-COMPAT** (no signal): the relational scorer has no input;
  abstention is forced by construction.
* **CONFOUND** (symmetric signal): the relational scorer's
  strict-asymmetric branch does not fire; abstention is forced.
* **DECEIVE** (adversarial signal): the relational scorer trusts
  its input; failure is forced.

Data file: `docs/data/phase65_cross_regime_synthetic.json::r65_no_compat,r65_confound,r65_deceive`.

### 4.4 5-seed stability

| seed | R-65-COMPAT-LOOSE gap | R-65-COMPAT-TIGHT gap |
| ---- | --------------------- | --------------------- |
| 11   | +1.000                | +1.000                |
| 17   | +1.000                | +1.000                |
| 23   | +1.000                | +1.000                |
| 29   | +1.000                | +1.000                |
| 31   | +1.000                | +1.000                |

Min / max / mean: +1.000 / +1.000 / +1.000 (saturated). Data files:
`docs/data/phase65_seed_sweep_loose.json`, `phase65_seed_sweep_tight.json`.

### 4.5 R-58 / R-64-SYM backward-compat

* **R-58 default deadlock**: W18 ties W15 byte-for-byte on the
  answer field (gold-only set via strict-asymmetric projection
  that lands on the same set the W11 contradiction-aware drop
  produces).
* **R-64-SYM**: W18 partially recovers — only the deadlock
  scenarios carry a relational mention (``relation=A_B_join``); on
  pool / disk / slow_query the W18 method abstains and ties FIFO.
  This is an *honest partial* result that motivates R-65-COMPAT as
  the *consistent* relational-mention regime.

## 5. The W18 family of named theorems

* **W18-Λ-sym** (proved-empirical n=8 saturated × 5 seeds +
  structural sketch). W17-Λ-symmetric extends to R-65-COMPAT
  verbatim for every method pre-W18.
* **W18-1** (proved-conditional + proved-empirical n=40 saturated
  across 5 seeds × 2 budgets). W18 strictly improves over the
  strongest non-W18 capsule baseline by ≥ 0.50 on R-65-COMPAT
  (loose AND tight); on the saturated config the gap is +1.000.
* **W18-2** (proved by inspection + mechanically-checked). W18
  determinism + closed-form correctness.
* **W18-3** (proved-empirical full programme regression). W18 ties
  W15 byte-for-byte on R-54..R-64 default banks via abstention or
  strict-asymmetric projection that lands on the same set.
* **W18-Λ-no-compat / -confound / -deceive** (proved-empirical
  n=8 saturated each). The three named falsifier regimes.
* **W18-C-LEARNED** (conjectural). A learned compatibility scorer
  may beat the closed-form rule when the LLM emits free-form
  natural-language relational mentions.
* **W18-C-OUTSIDE** (conjectural). An outside-information axis can
  detect the W18-Λ-deceive regime by cross-reference.
* **W18-Λ-real** (proved-conditional + empirical-research,
  conjectural-empirical-on-Mac-1). Real-LLM transfer of the W18
  method is conditional on the LLM emitting closed-vocabulary
  relational compounds.
* **W18-C-CROSS-BENCH** (conjectural). Cross-bench transfer of
  the W18 method to non-incident-triage regimes.

See `docs/THEOREM_REGISTRY.md` for the full table.

## 6. What this means for the original goal

The original Context Zero thesis is *per-agent minimum-sufficient
context for multi-agent teams*. The W18 milestone advances the
thesis along one named axis — relational compatibility under
symmetric corroboration — without retracting any prior axis. The
honest reading:

* **The W17-Λ-symmetric wall is real.** It binds wherever round-2
  evidence carries no asymmetric cue (R-65-NO-COMPAT,
  R-65-CONFOUND, R-65-DECEIVE).
* **The wall is breakable on a regime where it actually applies**
  *if and only if* round-2 carries a relational compatibility cue
  AND the cue is closed-vocabulary (R-65-COMPAT). The W18 method is
  the smallest closed-form move that exploits this cue.
* **The next named research frontier is W18-C-LEARNED** (free-form
  relational mentions) **and W18-C-OUTSIDE** (outside-information
  axis to detect deceptive relational mentions). Both are
  conjectural and out of scope for SDK v3.19.

This milestone *strengthens* the thesis on the relational-
compatibility axis but does *not* retract the named limits
W17-Λ-symmetric / W18-Λ-no-compat / W18-Λ-confound / W18-Λ-deceive.
Multi-agent context as a *whole* is not solved; one named structural
axis the prior milestone left explicit *is*.

## 7. Reproducibility

```bash
# R-65-COMPAT-LOOSE (W18-1 anchor):
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank compat --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_compat_loose.json

# R-65-COMPAT-TIGHT (composed with W15):
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank compat --decoder-budget 24 \
    --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_compat_tight.json

# Falsifiers:
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank no_compat --decoder-budget -1 --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_no_compat.json
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank confound --decoder-budget -1 --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_confound.json
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank deceive --decoder-budget -1 --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_deceive.json

# 5-seed stability:
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank compat --decoder-budget -1 --seed-sweep \
    --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_seed_sweep_loose.json
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --bank compat --decoder-budget 24 --seed-sweep \
    --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_seed_sweep_tight.json

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase65_relational_disambiguation \
    --cross-regime-synthetic --K-auditor 12 --n-eval 8 \
    --out docs/data/phase65_cross_regime_synthetic.json

# Tests:
python3 -m pytest vision_mvp/tests/test_wevra_relational_disambiguator.py -v
```
