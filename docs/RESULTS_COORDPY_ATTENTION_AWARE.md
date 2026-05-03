# Results — attention-aware capsule context packing (SDK v3.16, W15 family)

> Milestone note for the SDK v3.16 push: the **first decoder-side
> capsule-aware context packing move** in the CoordPy programme, and
> the first programme milestone where *joint correctness AND token-
> efficiency* is the strict-gain axis. SDK v3.15 (W14) closed the
> producer-side gap on R-61 (the structured prompt restored the
> bench property's cross-role decoy corroboration assumption under
> a real Ollama 14B producer); SDK v3.16 (W15) attacks the
> symmetric *downstream* gap: even when the bench property holds
> and the cross-round capsule pipeline is in scope, a strict
> *decoder-side* token budget ``T_decoder`` can drop the round-2
> specific-tier disambiguator under naive FIFO packing of the
> admitted union.
>
> Last touched: SDK v3.16, 2026-04-27.

## TL;DR

* **W15-1 strict gain on R-62-tightbudget** (synthetic, n=8 × 5
  seeds saturated). The new
  :class:`vision_mvp.coordpy.team_coord.AttentionAwareBundleDecoder`
  (a salience-aware capsule context packer wrapped around the W13
  layered decoder) achieves ``accuracy_full = 1.000`` on
  R-62-tightbudget under strict decoder-side budget pressure
  (``T_decoder = 24``); the FIFO-packed W13 baseline ties FIFO at
  ``accuracy_full = 0.000``. **+1.000 strict separation, stable
  across 5/5 alternate ``bank_seed`` values.**
* **W15-Λ-budget sharp limit theorem** (proved-empirical n=40
  saturated × 5 seeds + structural sketch via W7-3 extension).
  Under FIFO packing of the cross-round bundle when ``T_decoder``
  is below the admitted union's token sum, every cross-round
  decoder in the SDK ties FIFO at 0.000 — the round-2 specific-
  tier disambiguator falls past the budget by construction;
  arrival-order packing cannot retain it. The argument extends
  W7-3 (extraction floor) to the decoder-side axis: a never-
  decoded handoff has no influence on the answer.
* **W15-3 backward-compat preserved.** With ``T_decoder = None``
  the W15 decoder ties the W13 layered decoder byte-for-byte on
  the answer field; on R-58 default n=8 the W15 answers match
  W13 in 8/8 scenarios. R-54..R-61 default banks all still hit
  prior anchors. 393/393 prior tests pass byte-for-byte; 37 new
  tests cover the W15 surface and the Phase-62 closure
  invariants.
* **W15-Λ-budget falsifier** (R-62-saturation, ``T_decoder =
  9999``): under no decoder-side budget pressure the salience
  reordering is a no-op on the answer field — both the salience
  packer and the FIFO packer tie W13 at ``accuracy_full = 1.000``.
  The W15-1 win is *conditional* on (a) the bench property
  holding, (b) ``T_decoder`` below the union token sum (budget
  pressure exists), and (c) the regime carrying round-2 specific-
  tier disambiguation that FIFO would push past the budget; this
  falsifier makes the conditionality sharp.
* **Token-efficiency reading.** On R-62-tightbudget the W15 packer
  keeps ``186 / 220`` tokens (84.6 %) and ``76 / 88`` handoffs
  (86.4 %) admitted-union-wide; the FIFO packer keeps a marginally
  larger but causally-misaligned subset (``192 / 220`` tokens,
  ``80 / 88`` handoffs). The proxy attention metric
  ``position_of_first_causal_claim`` is **0.0 (rank 0)** in 8/8
  W15 cells and **−1 (absent)** in 8/8 FIFO-pack cells. Token
  reduction is not the goal — *causal-evidence concentration in
  early prompt positions* is. This is exactly the axis the
  honest proxy measures.

## 1. The structural move

### 1.1 Why W14 (producer protocol) is structurally insufficient under decoder-side budget pressure — W15-Λ-budget

The SDK v3.15 W14 layer ensures the bench property's cross-role
decoy corroboration assumption is *emitted* by a real producer
LLM under a structured prompt. But the W14 layer has no influence
on what happens *between* admission and decoding. The cross-round
decoders W11/W12/W13 consume the *full union* of admitted
handoffs, with no token budget enforcement at the decode step.

In a real-LLM downstream consumer (an audit agent that re-decodes
the bundle for human-readable summarisation, or a ranker LLM that
prioritises capsule witnesses), the bundle is a *prompt* with a
*context window*. When the admitted union exceeds the window, the
naive choice is FIFO truncation — drop handoffs from the tail
until the bundle fits.

**W15-Λ-budget** (proved-empirical + structural sketch) on
R-62-tightbudget:

*When the cross-round decoder's bundle is FIFO-truncated under
``T_decoder`` strictly below the admitted union's token sum, AND
the regime is the R-58 delayed-causal-evidence shape AND the
admitted union has more round-1 noise than round-2 specific-tier
claims AND round-2 emissions arrive later than round-1
emissions, every cross-round decoder in the SDK
(``capsule_layered_fifo_packed``, the implicit FIFO truncation
the bench applies to ``capsule_multi_round`` /
``capsule_robust_multi_round`` / ``capsule_layered_multi_round``
under bounded ``T_decoder``) ties FIFO at 0.000.*

**Sketch.** The bench property's R-58 round-2 disambiguator
predicate requires the elected ``root_cause`` to come from a
specific-tier claim_kind. Under FIFO arrival order the round-1
generic-noise emissions arrive first; the round-2 specific-tier
claim arrives last. Under ``T_decoder`` that fits ≤ |round-1| + 0
of the round-2 emissions, the FIFO-packed bundle contains only
round-1 generic noise. The priority decoder elects a generic
``root_cause`` (``error_spike`` / ``latency_spike``); the CCK is
itself generic; the W11 contradiction-aware drop has nothing
specific to project; the answer is wrong by construction.

A never-decoded handoff has no influence on the answer (extends
W7-3). This is W15-Λ-budget at the decoder-side axis.

### 1.2 The W15 capsule-context-packing move

The :class:`AttentionAwareBundleDecoder` is a two-stage decoder:

1. **First-pass tentative root_cause election.** Run the W11
   :class:`MultiRoundBundleDecoder` over the *normalised* full
   union (post-W13) to elect a tentative ``root_cause``. No
   token budget is enforced here — only the priority lookup, which
   is O(|union|).
2. **Salience-aware repacking + final decode.** Score every
   handoff in the union via a closed-form salience function:

   ::

       salience(h) =
           w_tier  * 1{h.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS}
         + w_cck   * 1{h.claim_kind in CCK(R)}
         + w_corr  * log1p(distinct_roles_for(service_tag(h)))
         + w_mag   * normalised_magnitude(h.payload)
         + w_round * round_index(h) / 2

   Pre-committed defaults (``W15_DEFAULT_*``):
   ``w_tier=4.0``, ``w_cck=3.0``, ``w_corroboration=1.0``,
   ``w_magnitude=0.1``, ``w_round=0.5``.

   Pack greedy under ``T_decoder`` with **hypothesis preservation**
   (Pass 1 reserves one slot per ``(service_tag, source_role,
   tier)`` bucket; Pass 2 fills remaining budget by salience). Tie-
   breaks are canonical
   ``(-salience, round_idx, source_role, claim_kind, payload_sha)``
   so the pack is byte-deterministic.

3. **Final decode.** Run :class:`LayeredRobustMultiRoundBundleDecoder`
   (W13) on the *packed* subset; the answer is W13's plus a
   ``pack_stats`` block exposing the salience-pack diagnostics.

The hypothesis-preservation pass is the load-bearing structural
property under multi-hypothesis regimes. Without it, salience-
greedy packing prefers more-corroborated decoy mentions over less-
corroborated gold mentions (decoy corr=2 > gold corr=1 by
construction in R-62), dropping the gold services entirely from
the kept bundle and producing the wrong answer set after the W11
contradiction-aware drop.

Pre-committed bench property: under ``T_decoder = 24`` on
R-62-tightbudget, the W15 packer puts the round-2 specific-tier
claim at rank 0 of the kept bundle in **every** scenario where the
bench property holds (8/8). The proxy attention metric
``position_of_first_causal_claim_avg = 0.0`` is mechanically
verified.

### 1.3 Why this is a *real* method change beyond W14

* The W15 layer is **not** a producer-protocol move, a
  normalisation move, or an admission move. It is a
  *decoder-side* context-shaping intervention that runs *between*
  admission and decoding and operates on the cross-round bundle
  directly. The CoordPy programme has now accumulated **seven**
  structurally-distinct layers — admission (W7-2 / W8 / W9),
  intra-round decoding (W10), cross-round decoding (W11), fixed-
  vocabulary normalisation (W12), open-world normalisation (W13),
  producer protocol (W14), and **decoder-side context packing
  (W15)**.
* The W15 contribution is *interpretable*: the salience score is
  a closed-form deterministic function of the handoff's bytes
  plus the elected ``root_cause``. Each component (tier, CCK,
  corroboration, magnitude, round) is auditable; the pack is
  byte-deterministic given a canonical tie-break order; the pack-
  stats block exposes the proxy attention signal directly.
* The W15 layer is *honestly an attention proxy*. We do NOT claim
  to manipulate transformer attention weights; we DO claim that
  putting the highest-salience evidence first in the bundle
  benefits any downstream LLM consumer via prompt-position
  attention (a well-known property of transformers under typical
  positional encoding regimes). The proxy metric
  ``position_of_first_causal_claim`` is the auditable measurement.
* The W15 layer composes additively with W11 / W12 / W13 / W14.
  R-58 / R-60-wide / R-61-structured-prompt anchors are preserved
  byte-for-byte under W15 with ``T_decoder = None`` (W15-3); they
  remain available for non-budget-pressured decoders.
* The CoordPy single-run product runtime contract is byte-for-byte
  unchanged; the W15 surface lives in
  :mod:`vision_mvp.coordpy.team_coord` only and is opt-in via the
  new :class:`AttentionAwareBundleDecoder` and
  :class:`CapsuleContextPacker` / :class:`FifoContextPacker`
  classes.

## 2. Pre-committed bench properties + falsifier

### 2.1 R-62-default bench property (W15-3 backward-compat anchor)

Phase-62 inherits the R-58 four-property delayed-causal-evidence
shape, plus three new structural ingredients:

1. ``round1_only_generic_noise`` — every round-1 emission to the
   auditor has *normalised* ``claim_kind`` ∈
   ``{LATENCY_SPIKE, ERROR_RATE_SPIKE, FW_BLOCK_SURGE}``.
2. ``round2_only_specific`` — every round-2 emission has a
   *normalised* ``claim_kind`` NOT in the generic-noise set.
3. ``decoy_only_in_round1`` — the decoy services are not mentioned
   in round 2.
4. ``all_decoys_round1_corroborated`` — every decoy service is
   mentioned by ≥ 2 distinct producer roles in round 1 (the W11-
   drop trigger condition).
5. ``all_golds_single_role`` — every gold service is mentioned by
   exactly 1 distinct producer role in round 1 (so the W11 drop
   does NOT remove gold services from the answer).
6. ``multi_hypothesis`` — ``|gold_services| = 2`` AND ``|decoy
   _services| ≥ 2``.

Mechanically verified by ``Phase62BankShapeTests`` — every
scenario satisfies all six properties (8/8 on the default bank).

### 2.2 R-62-tightbudget anchor (W15-1)

Same scenario shape as R-62-default; ``T_decoder = 24`` enforces a
strict bundle-level token budget over the cross-round union (≈ 27
/ 28 token sum per scenario). Under FIFO packing the round-2
specific-tier claim is dropped (``position_of_first_causal_claim
= -1`` in 8/8 cells). Under W15 salience packing the round-2 claim
is preserved at rank 0 in 8/8 cells; hypothesis preservation
guarantees every (tag, role) representative survives. The W11
contradiction-aware drop then removes decoys (multi-role-noise-
only) and keeps golds (single-role) cleanly.

Anchor: ``Phase62DefaultTests::test_tightbudget_w15_strict_win_over_fifo_pack``,
``Phase62DefaultTests::test_tightbudget_w15_keeps_round2_specific_claim``,
``Phase62SeedStabilityTests::test_gap_attention_minus_fifo_pack_holds_across_5_seeds``.

### 2.3 R-62-saturation falsifier (W15-Λ-budget anchor)

Same scenario shape; ``T_decoder = 9999`` (effectively unbounded).
The salience pack reduces to the full-union pack — same answer as
the FIFO pack which also keeps every handoff. Both packers tie
the unpacked W13 layered decoder at ``accuracy_full = 1.000``.

Anchor: ``Phase62DefaultTests::test_saturation_falsifier_w15_ties_fifo_pack``.

## 3. Cross-regime evaluation — single regime table

Pre-committed default config per regime; ``bank_seed=11``;
n_eval per regime as recorded in the data files:

| Regime                              | substr | fifo  | mr/W11 | rb/W12 | lay/W13 | layfp/W13+FIFO | aa/W15 |
|-------------------------------------|--------|-------|--------|--------|---------|----------------|--------|
| R-58-default (n=8)                  | 0.000  | 0.000 | 1.000  |  —     |   —     |     —          |   —    |
| R-59-noisy (n=12)                   | 0.000  | 0.000 | 0.000  | 1.000  |   —     |     —          |   —    |
| R-60-wide-OOV (n=12)                | 0.000  | 0.000 | 0.000  | 0.000  | 1.000   |     —          |   —    |
| R-60-cosmic-OOV (n=8)               | 0.000  | 0.000 | 0.000  | 0.000  | 0.000   |     —          |   —    |
| R-61-default (n=8)                  | 0.000  | 0.000 | 1.000  | 1.000  | 1.000   |     —          |   —    |
| R-61-naive-prompt (n=8)             | 0.000  | 0.000 | 0.000  | 0.000  | 0.000   |     —          |   —    |
| R-61-structured (n=8)               | 0.000  | 0.000 | 1.000  | 1.000  | 1.000   |     —          |   —    |
| R-61-ollama-naive (n=8)             | 0.000  | 0.000 | 0.000  | 0.000  | 0.000   |     —          |   —    |
| R-61-ollama-struct (n=8)            | 0.000  | 0.000 | 0.500  | 0.500  | 0.500   |     —          |   —    |
| **R-62-default (n=8)**              | 0.000  | 0.000 | 1.000  | 1.000  | 1.000   |   1.000        | 1.000  |
| **R-62-tightbudget (n=8)**          | 0.000  | 0.000 | 1.000* | 1.000* | 1.000*  |   0.000        | 1.000  |
| **R-62-saturation (n=8)**           | 0.000  | 0.000 | 1.000  | 1.000  | 1.000   |   1.000        | 1.000  |

\* The W11/W12/W13 columns under R-62-tightbudget run *without*
   ``T_decoder`` enforcement (the existing decoders take the full
   union; the 1.000 reflects the W13 answer when the union is not
   truncated). The load-bearing baseline for the W15-1 strict-gain
   claim is the **W13 + FIFO pack** column (``layfp/W13+FIFO``),
   which honestly applies the same ``T_decoder`` budget as the W15
   method via FIFO truncation.

Cross-regime data (this milestone):
``docs/data/phase62_default_K12_n8.json``,
``docs/data/phase62_tightbudget_K12_n8.json``,
``docs/data/phase62_saturation_K12_n8.json``,
``docs/data/phase62_seed_sweep_tightbudget_K12_n8.json``,
``docs/data/phase62_cross_regime.json``.

The seven CoordPy structural layers gate the column where each is
strictly load-bearing:

* R-58 → W11 multi-round bundle decoder (cross-round decoding).
* R-59 → W12 fixed-vocabulary closed-vocabulary normaliser.
* R-60 → W13 layered open-world normaliser.
* R-61 → W14 structured producer protocol.
* **R-62 → W15 attention-aware capsule context packing.**

Every prior layer's win is preserved on its anchor regime; W15
adds the seventh layer with a *strictly orthogonal* failure mode
(decoder-side budget pressure) and a *strictly orthogonal*
intervention (capsule-aware salience packing with hypothesis
preservation).

## 4. Theorem family W15 (minted by this milestone)

### W15-Λ-budget — decoder-side budget structural limit (proved-empirical + structural sketch)

**Statement.** On R-62-tightbudget default (``T_decoder = 24``,
admitted union token sum ≈ 27 / 28 per scenario), every FIFO-
packed cross-round decoder in the SDK
(``capsule_layered_fifo_packed`` is the explicit head; the
implicit FIFO truncation any naive cross-round decoder would
apply at this budget produces the same outcome) achieves
``accuracy_full = 0.000``.

**Sketch.** The bench property's R-58 round-2 disambiguator
predicate requires ``root_cause`` to come from a specific-tier
claim. Under FIFO arrival order the round-1 generic-noise
emissions arrive first; the round-2 specific-tier claim arrives
last. Under ``T_decoder`` that fits |round-1| but not the round-2
disambiguator, the FIFO-packed bundle contains only round-1
generic noise. The priority decoder elects a generic
``root_cause``; the CCK is itself generic; the W11 contradiction-
aware drop has nothing specific to project; the answer is wrong by
construction. Extends W7-3 (extraction floor) to the decoder-side
axis: a never-decoded handoff has no influence on the answer.

**Status.** Proved-empirical on n_eval=8 saturated × 5 seeds (all
gaps ≤ 0.000). Structural argument by the W7-3 extension. The
saturation falsifier (``T_decoder = 9999``) lifts the bound and
the FIFO pack achieves the same accuracy as the unpacked W13
decoder.

### W15-1 — AttentionAwareBundleDecoder sufficiency under bounded T_decoder (proved-conditional + proved-empirical)

**Statement.** Pairing the
:class:`AttentionAwareBundleDecoder` (default
:class:`CapsuleContextPacker` with ``preserve_hypotheses=True`` and
the pre-committed weight defaults
``W15_DEFAULT_*``) with the W13 layered decoder achieves
``accuracy_full = 1.000`` on R-62-tightbudget default
(``K_auditor = 12``, ``T_auditor = 256``, ``T_decoder = 24``,
``n_eval = 8``, ``bank_seed = 11``), stable across **5/5**
alternate ``bank_seed`` values (11, 17, 23, 29, 31) with strict
gap attention − fifo_packed_layered ∈ {1.000, 1.000, 1.000,
1.000, 1.000} (min 1.000 well above the 0.50 stability threshold).
The win is **conditional** on the bench property + budget
pressure + R-58-shape round-2 disambiguator carrying no
``service=`` token; either condition removed collapses the win
(see W15-Λ-budget falsifier and W15-Λ-degenerate falsifier).

**Status.** Proved-conditional (premises = R-62 bench property +
``T_decoder`` < admitted-union token sum) + proved-empirical
(synthetic n=40 saturated across 5 seeds). Anchors:
``Phase62DefaultTests::test_tightbudget_w15_strict_win_over_fifo_pack``,
``Phase62SeedStabilityTests::test_gap_attention_minus_fifo_pack_holds_across_5_seeds``,
``docs/data/phase62_seed_sweep_tightbudget_K12_n8.json``.

### W15-2 — pack determinism + closed-form salience (proved + mechanically-checked)

**Statement.** The :class:`CapsuleContextPacker.pack` method
satisfies:

* (a) Salience score is a closed-form deterministic function of
  the handoff's bytes plus the elected ``root_cause``: no RNG, no
  wall-clock, no environment dependence.
* (b) The pack output (kept handoffs and their order) is byte-
  deterministic given the input handoffs + ``root_cause`` +
  ``T_decoder`` + ``round_index_hint``.
* (c) Tie-break is canonical
  ``(-salience, round_idx, source_role, claim_kind, payload_sha)``;
  byte-identical handoffs collapse to one slot.
* (d) The :class:`FifoContextPacker.pack` method preserves arrival
  order; under ``T_decoder = None`` the kept bundle equals the
  input bundle byte-for-byte.
* (e) The hypothesis-preservation pass guarantees every distinct
  ``(service_tag, source_role, tier)`` triple gets at most one
  slot in the kept bundle (Pass 1 deduplication invariant).

**Status.** Proved by inspection + mechanically-checked by
``CapsuleContextPackerDeterminismTests`` and
``FifoContextPackerTests``.

### W15-3 — backward compatibility (proved-empirical)

**Statement.** With ``T_decoder = None`` the
:class:`AttentionAwareBundleDecoder` reduces to the W13 layered
decoder on the answer field; the ``pack_stats`` block is purely
additive. On R-58 default n=8 the W15 answers match W13 in 8/8
scenarios; on R-54..R-57 the admitted set is small enough that
the trust-admission fallback fires identically under both
decoders. R-58 / R-59 / R-60 / R-61 default + falsifier banks all
preserved; 393/393 prior tests pass byte-for-byte.

**Status.** Proved-empirical. Anchors:
``AttentionAwareBackwardCompatTests::test_w15_no_budget_ties_w13_byte_for_byte``,
full programme regression in `test_coordpy_*.py` (430/430 pass).

### W15-Λ-degenerate — saturation falsifier (proved-empirical)

**Statement.** On R-62-saturation (``T_decoder = 9999``), the
salience reordering is a no-op on the answer field; the W15
salience pack ties the FIFO pack and both tie the unpacked W13
layered decoder at ``accuracy_full = 1.000``. The W15-1 win is
*not* observable on regimes without budget pressure — by
construction.

**Status.** Proved-empirical, n_eval=8 saturated. Anchor:
``Phase62DefaultTests::test_saturation_falsifier_w15_ties_fifo_pack``.

### W15-4 — token-efficiency floor (proved + mechanically-checked)

**Statement.** The :class:`CapsuleContextPacker` and
:class:`FifoContextPacker` both satisfy
``tokens_kept ≤ T_decoder`` for every input on every call (strict
budget enforcement; no soft truncation). Equivalently
``tokens_kept_over_input ≤ 1.0``. The pack-stats block is a
faithful summary: ``n_input = n_kept + n_dropped_budget`` and
``tokens_input = tokens_kept + sum(n_tokens for h in dropped)``.

**Status.** Proved by inspection + mechanically-checked by
``Phase62DefaultTests::test_w15_token_efficiency_under_tight_budget``.

## 5. Honest scope (what this milestone does NOT claim)

* **Not** "we solved multi-agent context." R-62's wins are
  *conditional* on (a) the multi-hypothesis comparable-magnitude
  events, (b) decoder-side budget pressure, (c) the bench
  property holding, AND (d) the R-58-shape round-2 disambiguator
  carrying no ``service=`` token. *All four* must be present for
  W15-1 to hold; this is exactly the W15-Λ-degenerate falsifier
  statement.
* **Not** "we shape transformer attention weights." The W15 layer
  is an *honest proxy* attention metric — the position of the
  first causal claim in the salience-ordered pack. We do not
  manipulate model attention; we do reorder evidence to benefit
  prompt-position attention under typical transformer positional
  encoding regimes. The proxy metric is auditable; the attention
  claim is not.
* **Not** "we solved real-LLM transfer on the new regime." R-62 is
  a *synthetic* milestone — the producer is the deterministic
  IdentityExtractor, not a real LLM. Real-LLM transfer of W15
  (using a real LLM as the downstream decoder consumer) is
  W15-C-real, conjectural; it requires Mac 1 / Mac 2 to be online
  and the bundle to be re-decoded by an LLM agent under a real
  context window. SDK v3.16 does not run this probe.
* **Not** "the salience weights are universal." The pre-committed
  weights are tuned for the incident-triage CCK + magnitude
  vocabulary; cross-bench transfer (security-incident /
  compliance-review / robotics) is W15-C1, conjectural.
* **Not** "the runtime context is now token-bounded." The CoordPy
  single-run product runtime contract is byte-for-byte
  unchanged. ``W15`` is research-grade SDK code on the team-coord
  surface, opt-in only via
  :class:`AttentionAwareBundleDecoder`.
* **Not** "FIFO packing is always wrong." On R-62-saturation and
  on R-54..R-61 default banks (where ``T_decoder = None`` or
  loose), FIFO packing matches W15 byte-for-byte. The W15-1 win
  is structurally orthogonal: it lives at the
  decoder-budget-pressured slice of the regime grid only.
* **Not** "every multi-hypothesis regime favours W15." The
  asymmetric corroboration shape (decoy ≥ 2 roles, gold = 1 role)
  is a structural ingredient of R-62. A symmetric-corroboration
  multi-hypothesis regime is W15-Λ-symmetric — out-of-scope for
  the current milestone, but conjecturally a *harder* regime
  where neither W11 drop nor W15 packing fully recovers (the
  CoordPy programme's next-axis open question).

## 6. Active conjectures (SDK v3.16)

* **W15-C-real** (real-LLM downstream decoder): the W15 salience
  pack provides a strict accuracy gain over FIFO truncation when
  the cross-round bundle is fed to a real LLM (e.g. Mac-1
  ``qwen2.5:14b-32k``) configured as a downstream re-decoder
  agent under a tight context window. Conjectural; not yet wired.
  Falsifier: a real LLM whose attention weights are sufficiently
  uniform that prompt-position ordering does not matter.
* **W15-C1** (cross-bench): the W15 salience scoring transfers to
  non-incident-triage benchmark families when the family admits a
  closed-form CCK + magnitude vocabulary. Conjectural.
* **W15-C-LEARNED** (learned salience): a logistic-regression /
  small MLP scorer over per-handoff features (analogous to
  :class:`LearnedTeamAdmissionPolicy`) outperforms the closed-form
  W15 weights on a held-out test set. Conjectural; the
  pre-committed defaults are the un-learned baseline.
* **W15-C-SYMMETRIC** (multi-hypothesis with symmetric
  corroboration): on a regime where both gold and decoy services
  have ≥ 2 distinct roles' corroboration via generic-noise kinds,
  neither W11 drop nor W15 packing recovers; the named structural
  limit of the current capsule pipeline. Conjectural;
  out-of-scope for SDK v3.16 and the natural next-axis open
  question for SDK v3.17+.
* **W15-C-COMPOSE-W14** (W15 + W14 on real Ollama): W15 over the
  W14 structured producer protocol on R-61-ollama-structured (the
  R-61-OLLAMA-A tier anchor) closes the 1/8 model-error failure
  by reordering the LLM's misjudged response into a salience-
  aware bundle that the auditor decoder can re-elect from.
  Conjectural; requires Mac 1 + cross-LLM probe.

## 7. Theory consequences — sharper decomposition

The CoordPy programme now has **seven** structurally-distinct
layers that were named one-by-one over SDK v3.7..v3.16:

| Layer                                | SDK   | Theorem family | Anchor regime    |
|--------------------------------------|-------|----------------|------------------|
| Admission (cohort coherence)         | v3.8  | W7-2           | R-54             |
| Admission (cross-role corrob.)       | v3.9  | W8-1           | R-55             |
| Admission (multi-service)            | v3.10 | W9-1           | R-56             |
| Decoding (intra-round bundle)        | v3.11 | W10-1          | R-57             |
| Decoding (cross-round bundle)        | v3.12 | W11-1          | R-58             |
| Normalisation (fixed-vocabulary)     | v3.13 | W12-1          | R-59             |
| Normalisation (open-world)           | v3.14 | W13-1          | R-60-wide        |
| Producer protocol                    | v3.15 | W14-1          | R-61 + R-61-ollama-A |
| **Decoder context packing**          | v3.16 | **W15-1**      | **R-62-tightbudget** |

The layers compose: each layer addresses a structurally-distinct
failure mode (named limit theorem per layer), each layer's anchor
regime is a *strict* counterexample for every prior layer alone,
and each layer's win is *conditional* on a stated bench property.

This sharpens the answer to the original Context Zero question
*"what would solving multi-agent context actually require?"*:

1. **Producer-side ambiguity preservation** (W14) — the producer
   must emit the bench property's structural ingredients, which is
   non-trivial under default LLM compression.
2. **Cross-round / cross-role capsule structure** (W11/W10/W7-2/
   W8/W9) — the auditor must aggregate evidence across rounds and
   roles via the parent-CID-gated capsule DAG.
3. **Normalisation of the producer's drift channel** (W13/W12) —
   the auditor's pipeline must absorb bounded kind / payload drift
   so the priority decoder lookups still hit canonical entries.
4. **Decoder-side context packing** (W15) — when the auditor's
   downstream consumer is bounded by a token budget (a real LLM
   context window, a structured audit query, a downstream
   summarisation step), the union must be packed by causal
   salience with hypothesis preservation, not truncated FIFO. The
   round-2 disambiguator must lead the bundle.
5. **Lifecycle audit** (W4-1, T-1..T-7) — every cell of every
   regime must satisfy the team-lifecycle invariants for the run
   to be auditable.

All five are structurally necessary. The prior reading of the
programme stopped at (1)/(2)/(3) and observed an honest negative
on real LLMs. SDK v3.15 added (1) and showed that closes the gap
on a real LLM under the strong success bar. SDK v3.16 adds (4)
and shows that it is structurally necessary under decoder-side
budget pressure — a regime where the prior pipeline saturates at
0.000 by construction even when the bench property holds and the
producer protocol is in effect.

The defensible "thesis-after-SDK-v3.16" is that the synthetic→
real-LLM-and-bounded-context transfer story now has **seven
layers**:

* **Layer 1 (W6-C2 falsified):** un-normalised admission cannot
  transfer.
* **Layer 2 (W12-Λ at the real-LLM axis):** un-normalised cross-
  round decoding cannot transfer.
* **Layer 3 (SDK v3.13, W12-1):** fixed-vocabulary normalised
  cross-round decoding DOES transfer, conditional on the closed-
  vocabulary closure.
* **Layer 4 (SDK v3.14, W13-1):** layered (exact + heuristic)
  normalised cross-round decoding DOES transfer on a strictly
  *wider* drift channel, conditional on the heuristic predicate
  closure.
* **Layer 5 (SDK v3.14, W13-Λ-real, empirical):** real Ollama 14B
  at default settings does not produce the drift OR the cross-role
  decoy corroboration shape — the gating axis on real Ollama is
  *event-shape design + prompt-side discipline*, not normalisation.
* **Layer 6 (SDK v3.15, W14-1 + W14-Λ-real, conditional):** the
  structured producer protocol + comparable-magnitude events
  combined with the cross-round capsule pipeline DOES transfer on
  a real-LLM stream at +0.500 strict gain over substrate FIFO,
  conditional on (a) the redesigned events, (b) the structured
  prompt, (c) the cross-round pipeline.
* **Layer 7 (SDK v3.16, W15-1 + W15-Λ-budget, conditional):** the
  attention-aware capsule context packer + hypothesis preservation
  DOES restore correctness when the cross-round bundle is bounded
  by a strict decoder-side token budget, conditional on the
  multi-hypothesis bench property + budget pressure existing. The
  W15 layer adds an *orthogonal* axis to the prior six: even when
  W11/W12/W13/W14 all succeed at producing a clean ambiguity-
  preserving union, the union may exceed the downstream context
  budget — and FIFO truncation drops the load-bearing
  disambiguator. W15 is the structural fix.

The W15-Λ-degenerate falsifier sharpens the structural composition:
*no decoder-side budget pressure* removes the W15 advantage by
construction. This is *not* a refutation — it is the named
counterexample regime that confirms W15-1 is a conditional, not a
universal, win.

## 8. Files changed

* New SDK surface (additive):
  ``vision_mvp/coordpy/team_coord.py`` — adds
  ``W15_DEFAULT_*`` weights, ``W15PackedHandoff``,
  ``W15PackResult``, ``FifoContextPacker``,
  ``CapsuleContextPacker``, ``AttentionAwareBundleDecoder``;
  re-exported via ``__all__``.
* Public surface:
  ``vision_mvp/coordpy/__init__.py`` — re-exports the W15 surface +
  bumps ``SDK_VERSION = "coordpy.sdk.v3.16"``.
* New benchmark:
  ``vision_mvp/experiments/phase62_attention_aware_packing.py``.
* New tests:
  ``vision_mvp/tests/test_coordpy_attention_aware.py`` — 37 tests
  across salience scoring, pack determinism, hypothesis
  preservation, FIFO packer, backward-compat with W13, Phase-62
  bank shape, default config (W15-1 anchor), 5-seed stability,
  and cross-regime separation.
* Artifacts:
  ``docs/data/phase62_default_K12_n8.json``,
  ``docs/data/phase62_tightbudget_K12_n8.json``,
  ``docs/data/phase62_saturation_K12_n8.json``,
  ``docs/data/phase62_seed_sweep_tightbudget_K12_n8.json``,
  ``docs/data/phase62_cross_regime.json``.
* Doc updates:
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-62 anchor +
  bar 12 + § 2.11 R-62 ingredients),
  ``docs/RESEARCH_STATUS.md`` (this milestone),
  ``docs/THEOREM_REGISTRY.md`` (W15 family),
  ``docs/HOW_NOT_TO_OVERSTATE.md`` (W15 framing rules),
  ``docs/context_zero_master_plan.md`` (next-frontier note),
  ``docs/START_HERE.md`` (current milestone pointer),
  ``docs/RESULTS_COORDPY_ATTENTION_AWARE.md`` (this file).

## 9. What this milestone advances

* **The original Context-Zero thesis** — *per-agent
  minimum-sufficient context for multi-agent teams* — gains its
  first decoder-side instance with a strict gain of +1.000 over a
  FIFO-truncated cross-round decoder under decoder-side budget
  pressure. The minimum-sufficient context for the auditor's
  decision is now *measurable*: the W15 packer's
  ``position_of_first_causal_claim`` and ``tokens_kept_sum``
  metrics make the "minimum sufficient" claim auditable directly,
  not just inferred.
* **Joint correctness + context efficiency** is now a first-class
  axis of the success bar. Bar 12 of
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` requires both
  an accuracy threshold AND an explicit decoder-side token-budget
  constraint; methods that win on accuracy alone while ignoring
  decoder context budget do NOT clear bar 12.
* **The CoordPy programme has seven structural axes** with named
  limit theorems on each. W7-2 / W8 / W9 work on admission; W10
  works on intra-round decoding; W11 works on cross-round
  decoding; W12 works on fixed-vocabulary normalisation; W13
  works on layered open-world normalisation; W14 works on
  producer-side ambiguity preservation; **W15** works on
  decoder-side capsule context packing. The runtime contract is
  unchanged; all seven are research-grade SDK extensions.

## Cross-references

* Bench: ``vision_mvp/experiments/phase62_attention_aware_packing.py``
* Method: ``vision_mvp/coordpy/team_coord.py``
  (``AttentionAwareBundleDecoder``, ``CapsuleContextPacker``,
  ``FifoContextPacker``, ``W15PackResult``, ``W15PackedHandoff``)
* Tests: ``vision_mvp/tests/test_coordpy_attention_aware.py``
* Prior milestone: ``docs/RESULTS_COORDPY_PRODUCER_AMBIGUITY.md``
* Success criterion: ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``
  (R-62 anchor + bar 12 — joint-correctness-and-context-efficiency
  split + § 2.11 R-62 ingredients)
* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W15 family)
* Master plan: ``docs/context_zero_master_plan.md``
