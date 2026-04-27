# Results — layered open-world normalisation + real-Ollama probe (SDK v3.14, W13 family)

> Milestone note for the SDK v3.14 push: the **first open-world
> normalisation move** in the Wevra programme, and the **first
> real-Ollama benchmark with honest raw-response capture and an
> empirical real-LLM observation**. SDK v3.13 (W12) closed the
> synthetic→real-LLM transfer gap on a *fixed-vocabulary* drift
> channel; the named limit was W12-4 (any kind variant outside
> :data:`CLAIM_KIND_SYNONYMS` survived normalisation unchanged).
> SDK v3.14 (W13) sharpens that limit and partly widens it, and
> separately *measures* (does not claim) what actually happens
> under a real Ollama producer.
>
> Last touched: SDK v3.14, 2026-04-26.

## TL;DR

* **W13-1 strong success on R-60-wide synthetic.** The new
  ``LayeredRobustMultiRoundBundleDecoder`` (closed-vocabulary exact
  table → ordered heuristic abstraction rules → optional abstention,
  ahead of the W11 multi-round bundle decoder) achieves
  ``accuracy_full = 1.000`` on Phase-60 default at
  ``wide_oov_prob=0.50, svc_token_alt_prob=0.30``, while the SDK
  v3.13 W12 ``RobustMultiRoundBundleDecoder`` ties FIFO at 0.000.
  Headline gap **layered − w12 = +1.000** (seed 11), stable across
  **5/5** alternate (bank_seed, llm_seed) values with min gap
  **+0.917** (seed 23) — well above the strong-bar 0.50 stability
  threshold. The first capsule-native multi-agent coordination
  method that crosses the open-world normalisation boundary on a
  regime where fixed-vocabulary normalisation is provably
  insufficient.
* **W13-Λ-fixed sharp limit theorem.** Every variant the wide-OOV
  extractor emits is in :data:`HEURISTIC_RESCUABLE_OOV_KINDS` and
  *not* in :data:`CLAIM_KIND_SYNONYMS`; the W12 fixed-table
  normaliser cannot rescue any. Mechanically verified.
* **W13-4 sharp closure boundary (named falsifier).** On Phase-60
  ``synthetic_cosmic_oov_llm`` (round-2 specific kinds replaced by
  truly arbitrary OOV like XYZZY_QQQQ, COSMIC_RAY_FLIP), the
  layered decoder ties FIFO at 0.000. The closure of any
  predicate-based normaliser is finite; widening it is a research
  move, not a structural fix.
* **W13-3 backward-compat preserved.** On R-58 / R-59 / R-60-clean
  / R-60-synonym (the W12 anchor) and on R-54..R-57, the layered
  decoder produces byte-for-byte the same answer as W11 / W12.
  Cross-regime: every prior best capsule strategy still hits
  ``accuracy_full = 1.000`` on its anchor regime.
* **R-60-ollama honest negative (W13-Λ-real).** On real Ollama
  Mac-1 ``qwen2.5:14b-32k`` at temperature 0, the LLM emits
  *canonical* kinds (zero kind drift across n=4 × 12 producer
  calls) and *filters low-magnitude decoy events as noise* (the
  ``monitor`` role emits ``NONE`` for the deliberately-low-magnitude
  decoy events, breaking the cross-role decoy corroboration
  assumption). The bench property holds in 0/4 scenarios;
  normalisation has nothing to rescue; W13 ties W12 ties multi_round
  at ``accuracy_full = 0.250``. The R-60-ollama probe lands at the
  **R-60-OLLAMA-C tier** (honest null real transfer); the
  synthetic→real-LLM transfer story is gated by event-shape design
  + prompt-side discipline, not by normalisation. R-60-ollama is a
  *measurement* anchor, not a claim.

## 1. The structural move

### 1.1 Why W12 (fixed-vocabulary) fails on R-60-wide — W13-Λ-fixed

The SDK v3.13 ``RobustMultiRoundBundleDecoder`` rewrites drifted
``claim_kind`` strings via the closed-vocabulary
:data:`CLAIM_KIND_SYNONYMS` table (≈ 60 entries covering 11
canonical kinds × 4-5 LLM variants each). On the Phase-60 wide-OOV
extractor:

* The producer LLM emits a kind from
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` with probability
  ``wide_oov_prob`` (default 0.50). Examples:
  ``DEADLOCK_PROBABLY_DETECTED_MAYBE``, ``POOL_LOOKING_BUSY``,
  ``QUERY_SOMEWHAT_SLUGGISH``, ``DISK_GETTING_FULL_PROBABLY``,
  ``POOL_AT_CAPACITY``, ``P95_REGRESSION``, ``SLO_VIOLATED``,
  ``FW_DENIAL_BURST``, ``ERROR_BURST``, ``LATENCY_INCREASE``, …
* Every entry in :data:`HEURISTIC_RESCUABLE_OOV_KINDS` is verified
  *outside* :data:`CLAIM_KIND_SYNONYMS` (mechanically:
  ``W13ClosureTests::test_every_wide_oov_variant_outside_w12_inside_w13``).
* The W12 ``normalize_claim_kind`` lookup misses; the input survives
  unchanged; ``_DECODER_PRIORITY`` cannot match it; the elected
  root_cause stays *generic* (``latency_spike`` / ``error_spike`` /
  ``unknown``).
* The W11 contradiction-aware drop is a no-op on generic root_causes
  (W11-Λ at the temporal axis). The decoy survives; the gold
  services may even be entirely missing.

This is **W13-Λ-fixed** (proved-empirical + structural sketch):
``capsule_robust_multi_round`` (W12) ties FIFO at 0.000 on Phase-60
default across **5/5** seeds. The structural argument is by
inspection: any fixed-vocabulary table has a finite key set;
inputs outside the key set survive normalisation by construction.

### 1.2 The W13 layered move

The ``LayeredClaimNormalizer`` applies *two* layers in order:

1. **Exact synonym table** — :data:`CLAIM_KIND_SYNONYMS` lookup.
   Lossless on calibrated drift; idempotent on canonical input;
   the W12 closure is preserved unchanged. (W13 reduces to W12 on
   any input the W12 layer covers.)
2. **Heuristic abstraction rules** — an ordered set of
   :class:`HeuristicAbstractionRule` predicates over the upper-cased
   kind text:

   ```python
   _HEURISTIC_KIND_RULES = (
     # Storage tier — disk fill family.
     HeuristicAbstractionRule(
       pattern=re.compile(r"\bDISK[_ ]*(FILL|FULL|OVERFLOW|"
                            r"GETTING[_ ]*FULL|...|AT[_ ]*CAPACITY|"
                            r"OUT[_ ]*OF[_ ]*SPACE)"),
       canonical="DISK_FILL_CRITICAL",
       name="disk_fill"),
     # Database — pool exhaustion (conjunctive look-ahead).
     HeuristicAbstractionRule(
       pattern=re.compile(r"(?=.*\b(CONNECTION[_ ]*)?POOL)"
                            r"(?=.*(EXHAUST|...|CAPACITY|LIMIT|...))"),
       canonical="POOL_EXHAUSTION", name="pool_exhaustion"),
     # Database — slow query (conjunctive look-ahead).
     HeuristicAbstractionRule(
       pattern=re.compile(r"(?=.*QUER(Y|IES))"
                            r"(?=.*(SLOW|SLUG|TIMING[_ ]*OUT))"),
       canonical="SLOW_QUERY_OBSERVED", name="slow_query"),
     # ... and 11 more rules covering deadlock, OOM, TLS, DNS,
     # firewall, error, latency tiers.
   )
   ```

Every rule's `canonical` field is in ``_DECODER_PRIORITY``; every
canonical kind self-maps under the heuristic layer (idempotency); the
heuristic layer's union strictly contains the W12 exact table on
non-canonical inputs (mechanically verified by
``HeuristicRuleSoundnessTests``).

3. **Optional abstention** — when both layers miss, return either
   the input unchanged (default; preserves W12 behaviour on OOV) or
   the sentinel :data:`LAYERED_NORMALIZER_ABSTAIN` (opt-in; exposes
   OOV rate as a load-bearing decoder signal — W13-C4).

### 1.3 Why this is a *real* method change beyond W12

* The W13 decoder is **not** a refit of the W11 / W12 algorithm.
  The inner :class:`MultiRoundBundleDecoder` is unchanged. The W13
  contribution is *purely the closure-widening contract* — a small,
  ordered, deterministic set of regex predicates whose union
  strictly contains the W12 fixed table.
* The layered normaliser is a *new structural ingredient* the
  programme did not have before. SDK v3.13 (W12) absorbed real-LLM
  noise via a *closed-vocabulary table* — the named W12-4 limit
  said any variant outside the table escapes. SDK v3.14 (W13)
  absorbs *structurally similar* OOV via predicate abstraction —
  the named W13-4 limit says any input that witnesses *neither*
  the table *nor* any predicate escapes.
* Admission is FIFO; the W13 contribution is purely decoder-side.
  This preserves the cleanest possible test of "decoding alone,
  given a fair admission floor and a *layered* normalisation
  layer, can resolve a regime that no fixed-vocabulary decoder can
  solve."
* The closure-widening is *interpretable*: each rule has a name,
  an explicit pattern, and a canonical output. Per-call counters
  expose the layer breakdown (``n_exact``, ``n_heuristic``,
  ``n_abstained``, ``n_passthrough``, ``rule_hits``) so an
  operator can verify mechanically which rules fired.

## 2. Pre-committed bench properties + falsifier

### 2.1 R-60-wide bench property (W13-1 anchor)

Phase-60 inherits the R-58 four-property delayed-causal-evidence
shape verbatim, plus the closure-membership contract:

1. ``round1_only_generic_noise`` — every round-1 emission to the
   auditor has *normalised* ``claim_kind`` ∈
   ``{LATENCY_SPIKE, ERROR_RATE_SPIKE, FW_BLOCK_SURGE}``.
2. ``round2_only_specific`` — every round-2 emission has
   *normalised* ``claim_kind`` NOT in the generic-noise set.
3. ``decoy_only_in_round1`` — the decoy service is not mentioned
   in round 2 (after payload normalisation).
4. ``round1_decoy_corroborated`` — the decoy is mentioned by ≥ 2
   distinct producer roles in round 1.

**Closure-membership contract (new for R-60):**

* Every variant in :data:`HEURISTIC_RESCUABLE_OOV_KINDS` is *not*
  in :data:`CLAIM_KIND_SYNONYMS` AND resolves to its named
  canonical via :class:`LayeredClaimNormalizer`. Mechanically
  verified by
  ``W13ClosureTests::test_every_wide_oov_variant_outside_w12_inside_w13``.
* Every variant in :data:`COSMIC_OOV_KINDS` is *not* in
  :data:`CLAIM_KIND_SYNONYMS` AND does *not* resolve through any
  heuristic rule. Mechanically verified by
  ``W13ClosureTests::test_every_cosmic_oov_variant_escapes_both_layers``.

### 2.2 R-60-cosmic falsifier (W13-4 anchor)

The Phase-60 ``synthetic_cosmic_oov_llm`` extractor replaces every
round-2 specific-tier disambiguating claim with an entry from
:data:`COSMIC_OOV_KINDS` (XYZZY_QQQQ, COSMIC_RAY_FLIP, FOO_BAR_BAZ,
PINEAPPLE_LANDED, REDACTED_SIGNAL, OPAQUE_TOKEN_X,
INCIDENT_NUMBER_42, ABCDEF_GHIJK). None of these match the W12
table OR any heuristic rule pattern, so neither layer fires. Result:
``capsule_layered_multi_round`` ties FIFO at 0.000 on 8/8 falsifier
scenarios. Anchor:
``Phase60FalsifierTests::test_W13_4_cosmic_oov_collapses_layered_to_zero``.

### 2.3 R-60-ollama observation (W13-Λ-real anchor)

The Phase-60 ``ollama`` extractor talks Mac-1 Ollama with the
no-dedup-instructed Phase-58 prompt and captures every raw producer
string into the report. The empirical observation under
``model='qwen2.5:14b-32k'`` and ``temperature=0`` (n_eval=4):

* **Zero kind drift.** Every claim_kind the LLM emits is canonical;
  the ``LayeredClaimNormalizer``'s exact layer covers every
  emission; the heuristic layer fires zero times.
* **Magnitude filtering.** The ``monitor`` role's prompt presents 4
  events (2 high-magnitude on gold, 2 low-magnitude on decoy); the
  LLM emits 2 claims (the high-magnitude ones) and writes ``NONE``
  for the 2 low-magnitude decoy events, treating them as
  epistemically not-a-spike. This is *reasonable LLM behaviour* but
  breaks the bench's cross-role decoy corroboration assumption
  (``r1_decoy_corroborated`` requires ≥ 2 distinct roles to mention
  decoy — only ``network`` does).
* **Bench property holds 0/4.** Because of magnitude filtering, the
  decoy is mentioned by exactly 1 role in round 1 across all
  scenarios.
* **Downstream:** every per-round capsule strategy ties FIFO at
  0.000; multi_round / robust_multi_round / layered_multi_round
  achieve ``accuracy_full = 0.250`` on the one scenario where the
  bench shape accidentally aligned.

This is **W13-Λ-real** — an *empirical research observation*, not a
theorem. The honest reading: real Ollama at the 14B class on the
calibrated incident-triage prompt does not, by default, emit drift
OR the cross-role decoy corroboration shape. The
synthetic→real-LLM transfer is gated by event-shape design +
prompt-side discipline, not by normalisation.

## 3. Cross-regime evaluation

Single regime table at each bench's pre-committed default,
``bank_seed=11``:

| Regime              | Best prior strategy            | W12 robust | W13 layered | Notes                                         |
|---------------------|--------------------------------|------------|-------------|-----------------------------------------------|
| R-54 (W7-2)         | cohort_buffered = 1.000        | n/a        | n/a         | preserved                                     |
| R-55 (W8)           | corroboration = 1.000          | n/a        | n/a         | preserved                                     |
| R-56 (W9)           | multi_service = 1.000          | n/a        | n/a         | preserved                                     |
| R-57 (W10)          | bundle_decoder = 1.000         | n/a        | n/a         | preserved                                     |
| R-58 (W11)          | multi_round = 1.000            | 1.000      | 1.000       | W13-3 (n_heuristic = 0)                       |
| R-59 noisy (W12)    | robust_multi_round = 1.000     | 1.000      | 1.000       | W13-3 (n_heuristic = 0)                       |
| R-60 clean          | multi_round = 1.000            | 1.000      | 1.000       | W13-3 (n_exact > 0, n_heuristic = 0)          |
| **R-60 wide-OOV**   | **layered = 1.000**            | **0.000**  | **1.000**   | **W13-1 strict win, +1.000 vs W12**           |
| R-60 cosmic-OOV     | every method = 0.000           | 0.000      | 0.000       | W13-4 closure boundary (sharp)                |
| R-60 ollama (n=4)   | (no clear anchor)              | 0.250      | 0.250       | W13-Λ-real (zero drift; W13 = W12 = mr)       |

Cross-regime data: ``docs/data/phase60_cross_regime.json``.
Default-config data: ``docs/data/phase60_default_K8_n12.json``.
Seed-sweep data: ``docs/data/phase60_seed_sweep_K8_n12.json``.
Falsifier data: ``docs/data/phase60_falsifier_K8_n8.json``. Clean
data (W13-3 anchor): ``docs/data/phase60_clean_K8_n8.json``.
Real-Ollama capture: ``docs/data/phase60_real_ollama_qwen2_5_14b_n4.json``.

## 4. Theorem family W13 (minted by this milestone)

### W13-Λ-fixed — fixed-vocabulary closure limit on R-60-wide (proved-empirical + structural sketch)

**Statement.** On R-60-wide default, every variant the producer
emits is in :data:`HEURISTIC_RESCUABLE_OOV_KINDS` and is *not* in
:data:`CLAIM_KIND_SYNONYMS`. Therefore
:class:`RobustMultiRoundBundleDecoder` (W12) ties FIFO at
``accuracy_full = 0.000``.

**Sketch.** Any fixed-vocabulary table has a finite key set; inputs
outside the key set are passed through unchanged; the priority
decoder cannot match a non-canonical kind; the elected root_cause
stays generic; W11 contradiction-aware drop is a no-op. The closure-
membership property is mechanically verified per-cell.

**Status.** Proved-empirical on n_eval=12 saturated across 5/5
(bank_seed, llm_seed) values. Structural argument by inspection of
:func:`normalize_claim_kind` and :data:`CLAIM_KIND_SYNONYMS`.

### W13-1 — LayeredRobustMultiRoundBundleDecoder sufficiency under bounded-OOV-in-heuristic-closure noise (proved-conditional + proved-empirical)

**Statement.** Pairing FIFO admission with the
``LayeredRobustMultiRoundBundleDecoder()`` (default config) achieves
``accuracy_full = 1.000`` on R-60-wide default, stable across **5/5**
alternate (bank_seed, llm_seed) values, with strict gap
``layered − w12 ∈ {0.917, 1.000}`` (min 0.917), **conditional** on
the producer-noise channel being bounded by the heuristic closure
(every variant in :data:`HEURISTIC_RESCUABLE_OOV_KINDS` matches at
least one pattern in :data:`_HEURISTIC_KIND_RULES`).

**Status.** Proved-conditional (premise = closure-membership;
proved by ``W13-2``) + proved-empirical (n_eval=12 saturated × 5
seeds = 60 cells). Anchor:
``Phase60DefaultTests::test_layered_strict_win``,
``Phase60SeedStabilityTests::test_layered_minus_robust_gap_holds_across_5_seeds``.

### W13-2 — heuristic abstraction soundness (proved + mechanically-checked)

**Statement.** The :data:`_HEURISTIC_KIND_RULES` table satisfies:

* (a) Every rule's ``canonical`` field is a key in
  ``_DECODER_PRIORITY``.
* (b) Every canonical kind self-maps through the layered
  normaliser (idempotency on canonical input; the heuristic layer
  never disagrees with the exact-table layer on canonical input).
* (c) When the exact-table layer is removed, every canonical kind
  still self-maps via the heuristic layer (the rules witness their
  own canonicals).

**Status.** Proved by inspection + mechanically-checked by
``HeuristicRuleSoundnessTests``.

### W13-3 — backward compatibility (proved-empirical)

**Statement.** On the R-58 default bank, on Phase-59 noisy default,
on Phase-60 ``synthetic_clean_llm`` and ``synthetic_synonym_llm``,
and on R-54..R-57 default banks, the W13 layered decoder produces
byte-for-byte the same answer as the W11 / W12 decoder; the
heuristic layer fires zero times when the exact layer covers all
drift. Cross-regime: every prior anchor preserved at
``accuracy_full = 1.000``.

**Status.** Proved-empirical via ``Phase60BackwardCompatTests``.

### W13-4 — open-world closure boundary (proved-empirical)

**Statement.** On R-60-cosmic (Phase-60
``synthetic_cosmic_oov_llm``; round-2 specific kinds replaced by
entries from :data:`COSMIC_OOV_KINDS` verified absent from both
:data:`CLAIM_KIND_SYNONYMS` AND the heuristic predicate union),
the layered decoder ties FIFO at 0.000 on 8/8 scenarios.

**Sharper observation.** Any closed-vocabulary set of regex
predicates has a finite firing surface; OOV inputs whose surface
form witnesses none of the patterns escape the closure. A learned
normaliser (W13-C2) widens but does not eliminate the closure: any
predicate-based normaliser, learned or hand-curated, has a finite
VC dimension on the input space and an adversarial OOV channel
exists.

**Status.** Proved-empirical, n_eval=8 saturated. Anchor:
``Phase60FalsifierTests::test_W13_4_cosmic_oov_collapses_layered_to_zero``.

### W13-Λ-real — real-Ollama no-drift, magnitude-filtering observation (empirical-research)

**Statement.** Real Ollama 14B (qwen2.5:14b-32k on Mac 1) at
``temperature=0`` on the calibrated Phase-58 incident-triage prompt,
by default, emits canonical claim kinds (zero kind drift across
n=4 × 12 producer calls) and filters low-magnitude decoy events as
noise (the ``monitor`` role emits ``NONE`` for the deliberately-low-
magnitude decoy events, breaking the cross-role decoy corroboration
assumption). The bench property holds in 0/4 scenarios; W12 / W13
normalisation has nothing to rescue; W13 ties W12 ties multi_round
at ``accuracy_full = 0.250`` on the one scenario where the bench
shape accidentally aligned.

**Status.** Empirical-research (n=4 producer-side observation; 12
real Ollama calls; 0 failures; 49.4 s wall on Mac 1, capture:
``docs/data/phase60_real_ollama_qwen2_5_14b_n4.json``). The
observation is honest data, not a claim — under different (event
× prompt) regimes a real LLM may emit non-trivial drift; W13-C3 is
the conjectural extension.

## 5. Honest scope (what this milestone does NOT claim)

* **Not** "we solved multi-agent context." R-60-wide's win is
  *conditional* on (a) the R-58 delayed-causal-evidence shape, (b)
  the producer-noise channel being bounded by the heuristic closure,
  AND (c) round-N admission not being budget-starved (inherits
  W11-4).
* **Not** "Wevra now generalises to open-world LLM drift." The W13
  closure is wider than the W12 closure but still finite (W13-4).
  Truly arbitrary OOV escapes both layers.
* **Not** "real Ollama 14B emits the bench property." The honest
  R-60-ollama observation is the *opposite*: 14B emits canonical
  kinds and filters low-magnitude decoy events, so the bench
  property does not hold. R-60-ollama lands at the R-60-OLLAMA-C
  tier (honest null real transfer).
* **Not** "synthetic→real-LLM transfer is closed." SDK v3.14
  measures the gap honestly: the real-LLM gating axis is event-
  shape design + prompt-side discipline, not normalisation. A
  redesigned event stream that elicits non-trivial drift from a
  14B-class LLM is the W13-C3 next move.
* **Not** "the W13 method beats W12 on R-58 / R-59." On R-58
  (synthetic canonical) and R-59 (synthetic noisy in-W12-closure),
  W13 ties W12 byte-for-byte (W13-3). The win is *strict* only on
  regimes where the producer-noise channel is non-trivial AND
  bounded by the heuristic closure but not by the W12 fixed table.
* **Not** "the runtime now needs the layered normalisation layer."
  The Wevra single-run product runtime is unchanged. ``W13`` is
  research-grade SDK code, additive on top of W12.

## 6. Active conjectures (SDK v3.14)

* **W13-C1** (cross-bench): the W13 closure-widening contract
  generalises to non-incident-triage benchmark families when the
  family admits a closed-vocabulary tier mapping AND a small
  predicate set covering ≥ 90 % of LLM kind drift on the family's
  prompt. Conjectural.
* **W13-C2** (learned normaliser): a learned (embedding-distance or
  LLM-distilled) normaliser strictly widens the W13 heuristic
  closure on R-60-cosmic. Conjectural; restated as a closure-
  widening move, not a structural fix — the W13-4 boundary applies
  to any predicate-based normaliser.
* **W13-C3** (real-Ollama transfer with redesigned events): under
  redesigned events where decoy and gold services have comparable
  magnitudes AND the prompt instructs the LLM to emit one claim
  per distinct event, real Ollama qwen2.5:14b-32k does emit the
  R-58 bench property AND does drift kinds with non-trivial
  probability; under this regime, W13-1 strict bar is cleared on
  the real-LLM stream. Conjectural; Phase-60 v2 candidate.
* **W13-C4** (abstention-aware decoder): an abstention-aware
  decoder strictly improves over a passthrough decoder on a regime
  where OOV rate is non-trivial but bounded. Conjectural; the
  abstention sentinel is implemented but the abstention-aware
  decoder pathway is not yet wired.

## 7. Files changed

* New SDK class:
  ``vision_mvp/wevra/team_coord.py`` — adds
  ``HeuristicAbstractionRule``, ``LayeredClaimNormalizer``,
  ``LayeredRobustMultiRoundBundleDecoder``,
  :data:`_HEURISTIC_KIND_RULES`, :data:`LAYERED_NORMALIZER_ABSTAIN`;
  re-exported via ``__all__``.
* Public surface:
  ``vision_mvp/wevra/__init__.py`` — re-exports the W13 surface
  + bumps ``SDK_VERSION = "wevra.sdk.v3.14"``.
* New benchmark:
  ``vision_mvp/experiments/phase60_open_world_normalization.py``.
* New tests:
  ``vision_mvp/tests/test_wevra_open_world_normalization.py`` — 22
  tests across heuristic rule soundness, W13 closure properties,
  layered decoder unit semantics, default-config (W13-Λ-fixed +
  W13-1), falsifier (W13-4), backward-compat (W13-3), 5-seed
  stability, and noisy extractor witness.
* Artifacts:
  ``docs/data/phase60_default_K8_n12.json``,
  ``docs/data/phase60_falsifier_K8_n8.json``,
  ``docs/data/phase60_clean_K8_n8.json``,
  ``docs/data/phase60_seed_sweep_K8_n12.json``,
  ``docs/data/phase60_cross_regime.json``,
  ``docs/data/phase60_real_ollama_qwen2_5_14b_n4.json``.
* Doc updates:
  ``docs/RESEARCH_STATUS.md``,
  ``docs/THEOREM_REGISTRY.md``,
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``,
  ``docs/HOW_NOT_TO_OVERSTATE.md``,
  ``docs/context_zero_master_plan.md``,
  ``docs/START_HERE.md``,
  ``docs/RESULTS_WEVRA_OPEN_WORLD_NORMALIZATION.md`` (this file).

## 8. What this milestone advances

* **The original Context-Zero thesis** — *per-agent
  minimum-sufficient context for multi-agent teams* — gains its
  first **open-world-normalisation** instance. The
  minimum-sufficient context for the auditor's decision now
  survives bounded *open-world* producer drift (drift outside the
  fixed-vocabulary closure but inside the heuristic predicate
  closure), provided the runtime adds an explicit *layered*
  normalisation layer with both an exact table and a small set of
  abstraction rules. The W13 result is the first programme result
  to materially advance the thesis on a regime where fixed-
  vocabulary normalisation is provably insufficient.
* **The synthetic→real-LLM transfer story sharpens (and a fifth
  layer is named).** The honest reading is now:
  (i) un-normalised admission cannot transfer (W6-C2 falsified);
  (ii) un-normalised cross-round decoding cannot transfer (W12-Λ);
  (iii) fixed-vocabulary normalisation transfers under bounded
  *synthetic* drift (W12-1, conditional);
  (iv) heuristic-widened normalisation transfers under bounded
  *open-world* drift inside the heuristic closure (W13-1,
  conditional);
  (v) real Ollama 14B at default settings does not produce the
  drift OR the cross-role decoy corroboration shape — the gating
  axis on real Ollama is event-shape design + prompt-side
  discipline, not normalisation (W13-Λ-real, empirical
  observation).
* **The Wevra programme has five structural axes**
  (admission, decoding within a round, decoding across rounds,
  fixed-vocabulary normalisation across the synthetic noise
  channel, *layered open-world normalisation across the real-LLM
  drift channel inside the predicate closure*) with named limit
  theorems on each. W7-2 / W8 / W9 work on admission; W10 works on
  single-round decoding; W11 works on cross-round decoding; W12
  works on fixed-vocabulary normalisation; **W13** works on
  layered open-world normalisation. The runtime contract is
  unchanged; all five are research-grade SDK extensions.

## Cross-references

* Bench: ``vision_mvp/experiments/phase60_open_world_normalization.py``
* Method: ``vision_mvp/wevra/team_coord.py``
  (``LayeredClaimNormalizer``, ``LayeredRobustMultiRoundBundleDecoder``)
* Tests: ``vision_mvp/tests/test_wevra_open_world_normalization.py``
* Prior milestone: ``docs/RESULTS_WEVRA_REAL_LLM_MULTI_ROUND.md``
* Success criterion: ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``
  (R-60 anchor + bar 10 — open-world normalisation split + § 1.4
  R-60-ollama 4-tier grading)
* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W13 family)
* Master plan: ``docs/context_zero_master_plan.md``
