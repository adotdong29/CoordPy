# Start Here

One-pass orientation for this repository. If you read only one doc, read
this one. Everything else in the repo should make sense after this page.

> **Current canonical reading.** The active scientific and product
> position is captured by a small set of files; everything else is
> historical record under [`archive/`](archive/).
>
> | Topic                                | Live doc                                                           |
> | ------------------------------------ | ------------------------------------------------------------------ |
> | One-pass orientation                 | this file (`docs/START_HERE.md`)                                   |
> | What is true *now*                   | [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)                         |
> | Theorem-by-theorem status            | [`THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md)                       |
> | What may be claimed (do-not-overstate) | [`HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md)               |
> | Run-boundary capsule formalism (W3)  | [`CAPSULE_FORMALISM.md`](CAPSULE_FORMALISM.md)                     |
> | Team-boundary capsule formalism (W4) | [`CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md)           |
> | Long-running master plan             | [`context_zero_master_plan.md`](context_zero_master_plan.md)       |
> | Two-Mac MLX runbook                  | [`MLX_DISTRIBUTED_RUNBOOK.md`](MLX_DISTRIBUTED_RUNBOOK.md)         |
> | Latest milestone (SDK v3.25)         | [`RESULTS_WEVRA_W24_SESSION_COMPACTION.md`](RESULTS_WEVRA_W24_SESSION_COMPACTION.md) |
> | Previous milestone (SDK v3.24)       | [`RESULTS_WEVRA_W23_CROSS_CELL_DELTA.md`](RESULTS_WEVRA_W23_CROSS_CELL_DELTA.md) |
> | Previous milestone (SDK v3.23)       | [`RESULTS_WEVRA_CAPSULE_LATENT_HYBRID.md`](RESULTS_WEVRA_CAPSULE_LATENT_HYBRID.md) |
> | Previous milestone (SDK v3.22)       | [`RESULTS_WEVRA_MULTI_ORACLE_ADJUDICATION.md`](RESULTS_WEVRA_MULTI_ORACLE_ADJUDICATION.md) |
> | Previous milestone (SDK v3.21)       | [`RESULTS_WEVRA_OUTSIDE_INFORMATION.md`](RESULTS_WEVRA_OUTSIDE_INFORMATION.md) |
> | Previous milestone (SDK v3.20)       | [`RESULTS_WEVRA_DECEPTIVE_AMBIGUITY.md`](RESULTS_WEVRA_DECEPTIVE_AMBIGUITY.md) |
> | Previous milestone (SDK v3.19)       | [`RESULTS_WEVRA_RELATIONAL_DISAMBIGUATOR.md`](RESULTS_WEVRA_RELATIONAL_DISAMBIGUATOR.md) |
> | Previous milestone (SDK v3.18)       | [`RESULTS_WEVRA_LIVE_COMPOSITION.md`](RESULTS_WEVRA_LIVE_COMPOSITION.md) |
> | Previous milestone (SDK v3.17)       | [`RESULTS_WEVRA_COMPOSED_REAL_LLM.md`](RESULTS_WEVRA_COMPOSED_REAL_LLM.md) |
> | Previous milestone (SDK v3.16)       | [`RESULTS_WEVRA_ATTENTION_AWARE.md`](RESULTS_WEVRA_ATTENTION_AWARE.md) |
> | Previous milestone (SDK v3.15)       | [`RESULTS_WEVRA_PRODUCER_AMBIGUITY.md`](RESULTS_WEVRA_PRODUCER_AMBIGUITY.md) |
> | Previous milestone (SDK v3.14)       | [`RESULTS_WEVRA_OPEN_WORLD_NORMALIZATION.md`](RESULTS_WEVRA_OPEN_WORLD_NORMALIZATION.md) |
> | Previous milestone (SDK v3.13)       | [`RESULTS_WEVRA_REAL_LLM_MULTI_ROUND.md`](RESULTS_WEVRA_REAL_LLM_MULTI_ROUND.md) |
> | Previous milestone (SDK v3.12)       | [`RESULTS_WEVRA_MULTI_ROUND_DECODER.md`](RESULTS_WEVRA_MULTI_ROUND_DECODER.md) |
> | Previous milestone (SDK v3.11)       | [`RESULTS_WEVRA_BUNDLE_DECODER.md`](RESULTS_WEVRA_BUNDLE_DECODER.md) |
> | Pre-committed success bar (SDK v3.13)| [`SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`](SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md) |
> | Previous milestone (SDK v3.10)       | [`RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md`](RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md) |
> | Previous milestone (SDK v3.9)        | [`RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md`](RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md) |
> | Previous milestone (SDK v3.8)        | [`RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md`](RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md) |
> | Previous milestone (SDK v3.7)        | [`RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`](RESULTS_WEVRA_SCALE_VS_STRUCTURE.md) |
> | Repo top-level                       | [`../README.md`](../README.md), [`../ARCHITECTURE.md`](../ARCHITECTURE.md), [`../CHANGELOG.md`](../CHANGELOG.md) |
> | Historical record (read-only)        | [`archive/`](archive/) — pre-Wevra theory, older Wevra milestones, sprint prompts |

---

## What this repo is — in one line

**Wevra is a context-capsule runtime.** Every piece of context that
crosses a role boundary, a layer boundary, or a run boundary is a
typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-stamped **capsule** — never a raw prompt string. As of
**SDK v3.22 (April 2026)**, capsules are load-bearing **inside one
Wevra run** (W3 family, run-boundary → cell → parser axis → LLM
byte boundary), **between agents in a team** (W4 family,
multi-agent coordination *research slice*: TEAM_HANDOFF /
ROLE_VIEW / TEAM_DECISION + T-1..T-7 lifecycle audit + learned
per-role admission policy), **across the model-class gradient**
(W5 family, real cross-LLM parser-boundary measurement
TVD = 1.000 / 0.000), **across the *model regime × admission
strategy* grid on a real-LLM-driven multi-agent coordination
benchmark** (W6 family, SDK v3.7), **across a deterministic
cross-role cohort-coherence benchmark where capsule structure
provides a strict +1.000 ``accuracy_full`` advantage over
substrate FIFO** (W7 family, SDK v3.8), and — most sharply —
**across a strict separation between W7-2 single-tag plurality
cohort and W8 cross-role corroboration on a harder
decoy-plurality benchmark, with cross-bank stability and a named
falsifier regime** (W8 family, SDK v3.9), and — most sharply —
**across a strict separation between W8 single-tag corroboration
and W9 multi-service top-K corroboration on a harder
multi-service-gold benchmark, where every gold answer requires
admitting handoffs from two distinct gold services simultaneously,
again with cross-bank stability and a named falsifier regime**
(W9 family, SDK v3.10), and — most sharply — **across the
admission/decoding split itself: on the new Phase-57
decoder-forcing benchmark (multi-service-gold + corroborated-decoy
via *non-causal* claim_kinds), every service-blind admission policy
in the SDK ties FIFO at 0.000 (the W10-Λ admission-only structural
limit), while pairing W9 admission with the new
``BundleAwareTeamDecoder`` (CCK-projection on admitted services)
achieves 1.000 — the first capsule-native multi-agent coordination
method that crosses the admission/decoding split** (W10 family,
SDK v3.11).
SDK v3.22's headline result is the **first capsule-native multi-
agent-coordination method that crosses the W20-Λ-compromised
wall** (named in SDK v3.21) **on a regime where the wall actually
applies**. The W21 family ships the
:class:`TrustWeightedMultiOracleDisambiguator` — a deterministic,
training-free composition of the W19 inner with **N registered
outside oracles** under bounded context (one query per oracle per
cell, each ≤ ``max_response_tokens``). On R-68-MULTI-MAJORITY
(three registered oracles: ``compromised_registry`` first,
``service_graph``, ``change_history``; W20's single-oracle
interface picks the first-registered compromised oracle and
trusts its decoy reply ⇒ FAILS), the W21 method consults all
three; gold tags receive 2 votes (from the two honest
deterministic oracles), decoy receives 1; quorum forms on gold;
W21 projects the answer to the gold pair. **+1.000 strict gain
over W20** (which fails at 0.000 by trusting the first-registered
compromised oracle on the same regime), stable across **5/5**
alternate ``bank_seed`` values, both at ``T_decoder ∈ {None, 24}``
(loose AND tight). Three named falsifiers (W21-Λ-no-quorum,
W21-Λ-all-compromised, W21-Λ-partial) make the W21-1
conditionality sharp; the partial-recovery axis (W21-C-PARTIAL-
RECOVERY) is empirically discharged at ``quorum_min = 1``. Live
LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-REGISTRY): on a
four-oracle live registry pairing deterministic ``service_graph``
+ ``change_history`` with ``ollama_mixtral:8x7b`` (Mac-1, 47B-MoE)
the W21 method wins at +1.000 over W20 (registry-anchored regime;
the deterministic two form quorum on gold regardless of LLM
output). On the **harder coalition regime** (one deterministic
honest + one LLM + one compromised, ``quorum_min = 2``, LLM vote
required for quorum), the cross-model split is sharp:
``mixtral:8x7b`` lands gold tokens ⇒ W21 = **0.750** (+0.750 over
W20); ``gemma2:9b`` lands decoy tokens ⇒ W21 = 0.000 (+0.000 over
W20). Backward-compat (W21-3-A / W21-3-B) preserved byte-for-byte:
on R-54..R-67 default banks W21 ties W19 byte-for-byte via
``W21_BRANCH_NO_TRIGGER``; on R-67-OUTSIDE-RESOLVES with a single
oracle and ``quorum_min = 1`` W21 ties W20 byte-for-byte. The W21
surface is purely additive on top of the W20 surface (one new
``OracleRegistration`` dataclass + four new oracle adapters
``ChangeHistoryOracle`` / ``OnCallNotesOracle`` /
``SingletonAsymmetricOracle`` / ``DisagreeingHonestOracle`` + one
``W21OracleProbe`` + one ``W21MultiOracleResult`` + one
``TrustWeightedMultiOracleDisambiguator``); the SDK v3.21 runtime
contract is byte-for-byte unchanged. See
`docs/RESULTS_WEVRA_MULTI_ORACLE_ADJUDICATION.md` for the milestone
note.
SDK v3.21's prior headline result is the **first capsule-native
multi-agent-coordination method that crosses the W19-Λ-outside
wall** by acquiring asymmetric outside information (single-oracle
:class:`OutsideWitnessAcquisitionDisambiguator` + deterministic
``ServiceGraphOracle``).
SDK v3.19's headline result is the **first capsule-native multi-
agent-coordination method that crosses the symmetric-corroboration
wall** (W17-Λ-symmetric) **on a regime where the wall actually
applies**. The W18 family ships the
:class:`RelationalCompatibilityDisambiguator` — a closed-form
deterministic bundle-relational scorer that consumes the round-2
specific-tier disambiguator's *payload text* (the channel every
prior decoder ignored) and projects the W11 / W15 answer through
a strict-asymmetric branch that recovers gold-only services on
the new R-65-COMPAT regime. On R-65-COMPAT (every gold service
AND the decoy mentioned by ≥ 2 distinct routed roles via generic-
noise kinds with comparable magnitudes — symmetric-corroboration;
round-2 disambiguator carries a relational-compound mention of
every gold service AND no decoy service), every closed-form
salience scorer in the SDK ties FIFO at 0.000 (W17-Λ-symmetric
extended verbatim); the W18 method achieves
``capsule_relational_compat = 1.000`` at both
``T_decoder ∈ {None, 24}`` (loose AND tight), strictly improving
over every non-W18 capsule baseline by **+1.000**, stable across
**5/5** alternate ``bank_seed`` values. Three named falsifiers
(R-65-NO-COMPAT, R-65-CONFOUND, R-65-DECEIVE) make the W18-1
conditionality sharp: no signal → abstain → tie FIFO; symmetric
signal → abstain → tie FIFO; adversarial signal → trust evidence
→ fail at 0.000. Backward-compat (W18-3) preserved byte-for-byte:
on R-54..R-64 default banks the W18 method ties W15 byte-for-byte
on the answer field via abstention or strict-asymmetric projection
that lands on the same gold subset; with ``enabled = False`` the
W18 method reduces to W15 byte-for-byte. The W18 surface is purely
additive on top of the W15 surface (one new dataclass + one
tokeniser + one closed-form scorer + one wrapping decoder); the
SDK v3.18 runtime contract is byte-for-byte unchanged. See
`docs/RESULTS_WEVRA_RELATIONAL_DISAMBIGUATOR.md` for the milestone
note. SDK v3.18's prior headline result is the **first fresh-live end-to-end
real-LLM strict +1.000 advance** in the programme over the
strongest non-composed baseline AND **the first explicit
symmetric-corroboration limit theorem**. The W17 family ships a
*magnitude-hinted* extension to the W14 producer protocol (one
new prompt-mode + one operational-threshold table) that, on a
*fresh* live ``qwen2.5:14b-32k`` Mac-1 probe at
``T_decoder = 14``, takes the W14 + W15 composition from 7/8 +
0.500 (the W14 / W16-Λ-real-replay envelope) to **8/8 + 1.000**:
+1.000 strict gain over both substrate FIFO AND the FIFO-packed-
W14H-only baseline (W17-1, ``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``).
The W17-Λ-no-hint anchor on the same fresh probe under the legacy
structured prompt reproduces the recorded 7/8 + 0.500 envelope —
the magnitude-hint extension, not a re-run of the same prompt, is
the load-bearing improvement. Cross-model probe on a fresh live
``qwen3.5:35b`` MoE backend transfers: bench property holds in
8/8; +0.750 strict gain (W17-C-XMODEL,
``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``). The
**first explicit symmetric-corroboration limit theorem**
(W17-Λ-symmetric) lands as a *negative* result on the synthetic
``build_phase64_sym_bank``: every capsule strategy in the SDK
ties FIFO at 0.000 under both ``T_decoder ∈ {None, 24}`` by
construction — discharges W15-C-SYMMETRIC / W16-C-SYMMETRIC as a
negative theorem and names the next research frontier
(W17-C-DISAMBIGUATOR, conjectural). Backward-compat preserved
byte-for-byte: 442/442 prior tests pass; with default parameters
the W17 surface reduces to the SDK v3.15 W14 anchor (W17-3).
SDK v3.17's prior headline result is the **first end-to-end
W14 + W15 composition move** in the programme — and the **first
end-to-end real-LLM strict advance over the strongest non-
composed baseline**.
SDK v3.16 (W15) closed the decoder-side gap on R-62-tightbudget at
+1.000 over FIFO-packed-W13 under a deterministic identity
producer. SDK v3.17 (W16) attacks the symmetric *coupling*
question: **does the producer-side W14 layer compose with the
decoder-side W15 layer on a single regime where both are
independently load-bearing?** The new
:mod:`vision_mvp.experiments.phase63_composed_real_llm` driver
combines R-62 multi-hypothesis events with the Phase-61
:class:`MagnitudeFilteringExtractor` (its naive-prompt mode
breaks the bench property upstream; its structured-prompt mode
restores it) and the W15 salience packer under a strict
``T_decoder``. On R-63-COMPOSED-TIGHT (mag-filter structured ×
``T_decoder = 24``) the composition achieves
``accuracy_full = 1.000`` while every non-composed baseline
collapses to 0.000 — **+1.000 strict separation, stable across
5/5 alternate ``bank_seed`` values**. The W16-Λ-compose anchor
on R-63-naive-tight (mag-filter naive × ``T_decoder = 24``) shows
W14-Λ-prompt and W15-Λ-budget compose multiplicatively: every
capsule strategy ties FIFO at 0.000 when *both* upstream emission
and downstream retention fail. The W16-Λ-real-replay anchor
replays the recorded Phase-61 ``qwen2.5:14b-32k`` bytes
(``phase61_real_ollama_structured_qwen2_5_14b_n8.json``, n=8 × 24
producer calls, byte-stable) through the same pipeline at
``T_decoder = 14`` and shows the **first end-to-end real-LLM
strict +0.500 advance over the FIFO-packed-W14-only baseline**;
the recorded naive-prompt bytes confirm joint failure. The
runtime contract is byte-for-byte unchanged; W16 is *additive in
code* (no new SDK class) and *multiplicative in effect*
(jointly-necessary on R-63). Backward-compat preserved
byte-for-byte: 442/442 tests pass.
SDK v3.16's headline result was the **first decoder-side capsule-
context-packing move** in the programme. The
:class:`vision_mvp.wevra.team_coord.AttentionAwareBundleDecoder`
salience-packs the cross-round union (CCK + tier + corroboration
+ magnitude + round) with hypothesis preservation under
``T_decoder`` and defeats the FIFO-packed baseline by +1.000 on
R-62-tightbudget (synthetic). SDK v3.17 carries that decoder-side
intervention into composition with the W14 producer-side layer —
the first end-to-end demonstration that producer-side ambiguity
preservation AND decoder-side capsule context packing are
jointly necessary on the regime where both fire.

SDK v3.15's headline result was the **first producer-protocol move**
in the programme — and the **first real-Ollama benchmark in which a
Wevra cross-round capsule decoder produced a strict +0.50 gain over
substrate FIFO under a real LLM**. SDK v3.14 closed the synthetic
open-world normalisation axis (W13-1) but produced an honest
negative on real Ollama 14B (W13-Λ-real): the bench property was
being erased *upstream* by producer-side magnitude filtering and
compression, so normalisation had nothing to rescue. SDK v3.15
attacks that producer-side bottleneck directly via the
:class:`StructuredProducerProtocol` (a prompt-rendering surface that
splits round-1 *observation* from round-2 *diagnosis* and mandates
one claim per listed event). On the redesigned comparable-magnitude
events under the structured prompt, real Mac-1 ``qwen2.5:14b-32k``
at temperature 0 produces the bench property in **7/8 scenarios**;
the W11/W12/W13 cross-round bundle decoders all achieve
``accuracy_full = 0.500`` (vs 0.000 under naive prompt) — a +0.500
strict separation at exactly the R-61-OLLAMA-A tier threshold of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.5. The
synthetic counterpart (R-61-naive vs R-61-structured) is +1.000
strict, stable across 5/5 seeds. The W13 closure-widening is
*structurally invisible* on this regime because the real LLM emits
canonical kinds (zero drift) — the load-bearing layer is W14, not
W13. The synthetic→real-LLM transfer story now has six layers
named, each with a structurally-distinct failure mode and a sharp
limit theorem; the W14 layer is the **first** layer to materially
advance the programme on a real-LLM stream. Backward-compat
preserved: R-58 / R-59 / R-60 default + falsifier banks, R-54..R-57
anchors, and 393/393 prior tests pass byte-for-byte. Wevra's
single-run product runtime contract is unchanged.

SDK v3.14's headline result is the **first open-world
normalisation move** in the programme: on the new Phase-60
``synthetic_wide_oov_llm`` regime (R-60-wide; producer emits LLM
variants from :data:`HEURISTIC_RESCUABLE_OOV_KINDS` whose every
entry is verified outside :data:`CLAIM_KIND_SYNONYMS`), the new
``LayeredRobustMultiRoundBundleDecoder`` (W13: exact synonym table
→ ordered heuristic abstraction rules → optional abstention,
ahead of the W11 multi-round bundle decoder) achieves
``accuracy_full = 1.000`` while the SDK v3.13 W12
``RobustMultiRoundBundleDecoder`` ties FIFO at 0.000 — a
**+1.000** strict separation, stable across **5/5** alternate
(bank_seed, llm_seed) values with min gap layered − w12 = +0.917.
This sharpens W12-4 into the named limit theorem **W13-Λ-fixed**
(any fixed-vocabulary table has a finite closure) and widens the
practical closure with the heuristic predicate union. The named
falsifier **W13-4** (R-60-cosmic, truly arbitrary OOV outside
both layers) ties FIFO at 0.000, preserving the structural bound
on any predicate-based normaliser. Backward-compat preserved:
R-54 / R-55 / R-56 / R-57 / R-58 / R-59 / R-60-clean all still
hit ``accuracy_full = 1.000``. The opt-in ``ollama`` mode is the
**first real-Ollama probe** through the team-coord pipeline; on
Mac-1 ``qwen2.5:14b-32k`` at temperature 0 it produces a
**R-60-OLLAMA-C tier honest negative** (W13-Λ-real): the LLM
emits canonical kinds AND filters low-magnitude decoy events as
noise, so the bench property holds in 0/4 scenarios and W13/W12
have nothing to rescue. The synthetic→real-LLM transfer story
therefore has five layers and the real-LLM gating axis is
event-shape design + prompt-side discipline (W13-C3), not
normalisation. SDK v3.13's
real-LLM-robust multi-round bundle decoder remains the W12-1
anchor for synthetic noisy transfer; W13 is a strict additive
layer that adds R-60-wide, not a refutation.

SDK v3.11's headline result is the **third consecutive
strong-bar conditional structural win** in the programme: on the
Phase-57 deterministic decoder-forcing benchmark, the new joint
admit-and-decode method (``MultiServiceCorroborationAdmissionPolicy``
with ``top_k=3`` + ``BundleAwareTeamDecoder`` with ``cck_filter=True``,
``role_corroboration_floor=1``,
``fallback_admitted_size_threshold=2``) achieves
``accuracy_full = 1.000`` while substrate FIFO, ``capsule_fifo``,
``capsule_priority``, ``capsule_coverage``,
``capsule_cohort_buffered`` (W7-2),
``capsule_corroboration`` (W8), **AND** ``capsule_multi_service``
(W9) all produce 0.000 — a +1.000 strict separation from every
admission-only baseline including SDK v3.10 W9 itself, stable
across **5/5** alternate ``bank_seed`` values. Backward-compat
preserved (W10-3): bundle decoder ties prior best on R-54
(W7-2), R-55 (W8), R-56 (W9) at 1.000 each via the
trust-admission fallback. The named W10-4 falsifier regime
(decoy CCK-promotion) correctly ties FIFO at 0.000 — the
structural limit of any single-round CCK-style filter.
SDK v3.10's previous-headline result was the **second consecutive
strong-bar conditional structural win** in the programme: on the
Phase-56 deterministic benchmark (multi-service-gold property +
both gold services corroborated by ≥ 2 distinct producer roles +
single-role decoy storm + ``|candidates| > K_auditor``), the new
``MultiServiceCorroborationAdmissionPolicy`` (buffered mode,
``top_k=2, min_corroborated_roles=2``) achieves ``accuracy_full
= 1.000`` while substrate FIFO, ``capsule_fifo``,
``capsule_priority``, ``capsule_coverage``, the SDK v3.8
``capsule_cohort_buffered`` (W7-2), **AND** the SDK v3.9
``capsule_corroboration`` (W8) all produce 0.000 — a +1.000
strict separation **from W8 itself**, stable across 5/5
alternate ``bank_seed`` values. The win is **conditional** on the
named bench property; under the named W9-4 falsifier regime
(decoy-corroborated decoy with ≥ 2 distinct producer roles), the
new policy correctly ties FIFO at 0.000. Backward-compat
preserved (W9-3): on Phase-55 the new policy ties W8 at 1.000 (via
the argmax-by-role-count gate); on Phase-54 it ties W7-2 at 1.000.
No regression on Phase-53 synthetic. The milestone is the
**first SDK release whose strict-gain regime is not solvable by
the previous SDK's strongest method** — a real generalisation,
not just another conditional win on the same axis.
SDK v3.9's headline result was a **strong-bar conditional
structural win** (`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1): on the Phase-55 deterministic benchmark (decoy-plurality
+ cross-role-corroborated-gold property + ``|candidates| >
K_auditor``), the new ``CrossRoleCorroborationAdmissionPolicy``
(buffered mode, pre-fitted dominant tag from the candidate
stream's (role, payload) pairs) achieves
``accuracy_full = 1.000`` while substrate FIFO,
``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
**AND** the SDK v3.8 ``capsule_cohort_buffered`` (W7-2) all
produce 0.000 — a +1.000 strict separation from both baselines,
stable across **5/5** alternate ``bank_seed`` values. The win
is **conditional** on the named bench property; under the
named W8-4 falsifier regime (decoy-corroborated decoy), the
new policy correctly ties FIFO at 0.000. Backward-compat
preserved (W8-3): on Phase-54 the new policy ties W7-2 at
1.000. No regression on Phase-53 synthetic or Phase-53 14B
real-LLM (all strategies tie at 0.800, the W7-1 anchor). The
milestone is the **first SDK release** to clear the strong
success bar — three named regimes, cross-bank stability, named
falsifier, audit T-1..T-7 preserved on every cell.
SDK v3.8's headline result remains a **conditional structural win**:
on the Phase-54 deterministic benchmark (gold-plurality property
+ foreign-service decoys + ``|candidates| > K_auditor``), the
new ``CohortCoherenceAdmissionPolicy`` (buffered mode, pre-fitted
plurality from candidate-stream payloads) achieves
``accuracy_full = 1.000`` while substrate FIFO,
``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
and the streaming-cohort variant all produce 0.000 — a +1.000
gap, stable across 5/5 alternate ``bank_seed`` values. The win
is **conditional** on the bench property; under W7-1 (low
surplus, Phase-53 anchor), substrate FIFO is unbeatable by
construction. The Phase-53 / Phase-54 dichotomy (W7-1 / W7-2)
makes the capsule layer's coordination-performance contribution
*demonstrable in a falsifiable way* under a stated condition,
without invalidating the Phase-53 reading. The streaming variant
is unstable under arrival permutation (W7-1-aux); the buffered
variant is the load-bearing policy. SDK v3.7's reading is
preserved exactly: the SDK v3.5 learned-admission-policy
advantage (synthetic+noise default config) does **not** transfer
out-of-distribution to a real-LLM regime on Phase-53;
``structure_gain`` is non-positive at every model regime tested
(-0.4 / -0.4 / 0.0). The capsule layer's *audit* contribution
(T-1..T-7) is preserved and extends to Phase-54 unchanged. Mac 2 is
still offline (192.168.12.248 ARP "incomplete"); **no two-Mac
sharded inference happened in SDK v3.7 or SDK v3.8** — the
``MLXDistributedBackend`` integration boundary is unchanged
from SDK v3.6 and waits for the runbook
(`docs/MLX_DISTRIBUTED_RUNBOOK.md`) when Mac 2 returns. The
strongest model class actually exercised is single-Mac
qwen3.5:35b (36 B-MoE Q4) on Mac 1 Ollama. SDK v3.6's two-Mac
MLX-distributed integration boundary (chosen path: **MLX
distributed** under `mpirun mlx_lm.server`; **not** Hyperspace,
which is distributed-agent infrastructure rather than
single-model sharding) is byte-for-byte unchanged — experimental
infrastructure, not product; the Wevra single-run product
runtime contract is byte-for-byte unchanged. Up through SDK v3.4, capsules drove execution **one
further structural layer past v3.3** by extending the discipline
into the LLM byte boundary itself. The end-to-end inner-loop chain is
**five typed sealed capsules** —
PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
TEST_VERDICT — with strong parent-CID gating at each step
(Theorems W3-42 / W3-43 / W3-44). A runtime-checkable
``CapsuleLifecycleAudit`` mechanically verifies eleven
invariants L-1..L-11 on every finished run (Theorems W3-40 /
W3-45). Deterministic-mode replay (``RunSpec(deterministic=True)``)
collapses the full capsule DAG byte-for-byte across runs of the
same logical input (Theorem W3-41). Meta-artefacts (the report
itself) are authenticated by a *detached* META_MANIFEST in a
secondary ledger — the rendering circularity (impossible to seal
a report whose bytes encode the seal) is a sharp theorem
(W3-36) with a constructive boundary witness. ``wevra-capsule
verify`` recomputes the chain from on-disk header bytes and
re-hashes every artefact. SDK v3.4 also ships a
**synthetic-LLM mode** (``SweepSpec(mode="synthetic", ...)``)
that exercises the full chain in CI without a network endpoint;
the cross-model parser-boundary research (W3-C6) reports
PARSE_OUTCOME failure-kind TVD up to 1.000 across the
calibrated synthetic distribution library. See *"What capsules
do at runtime now"* below. Context Zero is the research
programme that produced it.

## The load-bearing abstraction — Context Capsule

A **`ContextCapsule`** is an immutable object with:

  * **`cid`**        — SHA-256 content address over
                        `(kind, payload, budget, parents)`.
  * **`kind`**       — closed-vocabulary semantic type (`HANDOFF`,
                        `HANDLE`, `THREAD_RESOLUTION`, `SWEEP_CELL`,
                        `PROVENANCE`, `RUN_REPORT`, `PROFILE`,
                        `ARTIFACT`, `READINESS_CHECK`, `SWEEP_SPEC`,
                        `ADAPTIVE_EDGE`).
  * **`lifecycle`**  — `PROPOSED → ADMITTED → SEALED` (+ optional
                        `RETIRED`).
  * **`budget`**     — `CapsuleBudget(max_tokens, max_bytes,
                        max_rounds, max_witnesses, max_parents)`.
  * **`parents`**    — tuple of parent CIDs (the capsule DAG).

Capsules live in a **`CapsuleLedger`** — append-only, hash-chained,
budget-enforcing, provenance-auditing. Every Wevra run emits a sealed
capsule DAG rooted at a `RUN_REPORT` capsule; the root CID is the
durable identifier for the run.

This abstraction *subsumes and re-centers* everything Wevra already
did. Handles (Phase 19), typed handoffs (Phase 31), thread resolutions
(Phase 35), sweep cells, and the provenance manifest were already
capsules — they just weren't named. SDK v3 names them.

See [`archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md`](archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md) for the
theorem-style statement of the Capsule Contract (invariants C1..C6)
and why it is a better top-level description of the product than
"bounded-context orchestration." See
[`archive/wevra-milestones/RESULTS_WEVRA_CAPSULE_NATIVE.md`](archive/wevra-milestones/RESULTS_WEVRA_CAPSULE_NATIVE.md)
for the SDK v3.1 milestone in which capsules become the runtime's
typed execution contract (W3-32 / W3-33 / W3-34 / W3-35).

---

## What capsules do at runtime now (SDK v3.2)

As of April 2026, when you run Wevra with the default
``RunSpec(capsule_native=True)``, capsules **drive** the run, not
just describe it. Specifically:

  * **Profile is sealed first.** Every other capsule's parent CID
    chain ends at the PROFILE capsule. You cannot seal a
    READINESS_CHECK or SWEEP_SPEC without a sealed PROFILE.
  * **Each runtime stage seals its capsule before the next stage
    can read its result.** Readiness verdict → READINESS_CHECK
    capsule. Sweep spec → SWEEP_SPEC capsule. Each cell's results
    → SWEEP_CELL capsule (sealed *as soon as the cell completes*,
    not after the whole sweep). Provenance manifest → PROVENANCE
    capsule. The RUN_REPORT capsule's parents are every other
    sealed capsule.
  * **Substantive artifacts are content-addressed at write time.**
    ``readiness_verdict.json`` / ``sweep_result.json`` /
    ``provenance.json`` are written via
    ``ctx.seal_and_write_artifact``: SHA-256 is computed in
    memory, baked into a sealed ARTIFACT capsule, then bytes hit
    disk, then re-read and re-hashed to verify. The on-disk
    file's hash matches the sealed CID by construction
    (Theorem W3-33).
  * **Mid-run failure is a typed witness, not a state-bag.** A
    stage that fails (admission rejection, missing parent CID,
    over-budget capsule) leaves a typed entry in the in-flight
    register that never reaches the ledger. The runtime exposes
    ``ctx.in_flight_failures()`` listing every kind / cid /
    failure for forensic inspection.
  * **Capsule view tags itself with construction mode.** The
    ``report["capsules"]`` block now carries
    ``"construction": "in_flight"`` so a downstream consumer can
    tell whether the ledger was built during the run (capsule-
    native) or folded after the fact (legacy post-hoc).

## What changed in SDK v3.10 (multi-service top-K cross-role corroboration — research slice)

SDK v3.10 mints the **W9 theorem family**:

* **W9-1** (proved-empirical, n=50 saturated). On Phase-56
  (multi-service-gold + cross-role-corroborated, single-role
  decoy storm), the buffered
  ``MultiServiceCorroborationAdmissionPolicy``
  (``top_k=2, min_corroborated_roles=2``) achieves
  ``accuracy_full = 1.000`` while substrate FIFO, all fixed
  capsule baselines, the SDK v3.8 W7-2 buffered cohort, AND the
  SDK v3.9 W8 corroboration policy all produce 0.000.
  +1.000 strict separation, stable across 5/5 alternate
  ``bank_seed`` values.
* **W9-2** (proved, structural). The dominant-set fitter
  (``_dominant_tag_set``) has three structural properties:
  (a) any tag with ``|distinct_roles| < min_corroborated_roles``
  is excluded; (b) if the argmax-by-role-count tier has size 1,
  W9 collapses to W8 (backward-compat); (c) if the argmax tier
  has size ≤ ``top_k``, W9 admits the entire tier.
* **W9-3** (proved-empirical, n=10). Backward-compat: on
  Phase-55 default and Phase-54 default, W9 admits the same set
  as W8 / W7-2 respectively, and achieves ``accuracy_full =
  1.000`` on both.
* **W9-4** (proved-empirical, n=10 falsifier saturated). When
  a decoy is also corroborated by ≥ ``min_corroborated_roles``
  distinct producer roles, W9 admits the decoy and ``services_correct``
  fails; W9 ties FIFO at 0.000. The W9-1 win does NOT hold —
  by construction.

The new admission policy
(``MultiServiceCorroborationAdmissionPolicy``) is exported under
the canonical alias
``vision_mvp.wevra.TeamMultiServiceCorroborationAdmissionPolicy``.
The Wevra single-run product runtime contract is **byte-for-byte
unchanged from SDK v3.9**; the new surface is purely additive
(multi-agent coordination research slice). The lifecycle audit
(T-1..T-7) holds on every cell of every regime.

## What changed in SDK v3.2 (intra-cell + detached witness)

Two new structural moves were made:

  * **Intra-cell capsules** (``PATCH_PROPOSAL`` and
    ``TEST_VERDICT``). Each (task, strategy) inside a sweep
    cell now seals two capsules in flight: a PATCH_PROPOSAL
    when the generator returns a patch (parent: SWEEP_SPEC,
    payload: coordinates + content hash + bounded rationale),
    and a TEST_VERDICT when the sandbox returns a result
    (parent: PATCH_PROPOSAL, payload: WorkspaceResult fields).
    The ``patch → verdict`` ordering is enforced at the
    capsule layer (W3-32-extended). On the bundled
    ``local_smoke`` profile this seals 48 PATCH_PROPOSAL +
    48 TEST_VERDICT capsules per run.
  * **Detached META_MANIFEST witness for meta-artefacts.**
    The runtime now writes a fourth file, ``meta_manifest.json``,
    whose payload carries the on-disk SHA-256s of
    ``product_report.json``, ``capsule_view.json``, and
    ``product_summary.txt`` plus the primary ``root_cid``. The
    manifest sits in a *secondary* ledger disjoint from the
    primary — the rendering circularity (Theorem W3-36) makes
    it impossible to seal an ARTIFACT for a report whose bytes
    encode the seal. The manifest is the one-hop trust unit
    beyond the primary view; ``wevra-capsule verify`` now
    re-hashes every meta-artefact and primary artefact at
    audit time (W3-37 / W3-38).

## What changed in SDK v3.3 (deeper intra-cell + audit + determinism)

Three additive moves:

  * **Sub-intra-cell ``PARSE_OUTCOME`` capsule.** Each
    (task, strategy) now seals THREE capsules: a PARSE_OUTCOME
    *before* the PATCH_PROPOSAL (parent: SWEEP_SPEC, payload:
    coordinates + parser ``ok`` boolean + closed-vocabulary
    ``failure_kind`` + ``recovery`` label + bounded ``detail``).
    The PATCH_PROPOSAL's parents now include both SWEEP_SPEC and
    the PARSE_OUTCOME's CID, so the parse → patch → verdict
    chain is a typed witness on the DAG (Theorem W3-39). On
    ``local_smoke`` this seals 48 PARSE_OUTCOME capsules per
    run (all ``failure_kind == "oracle"`` because the bundled
    profile uses the deterministic_oracle path); under real-LLM
    profiles, ``failure_kind`` distributes over the parser's
    closed vocabulary (``ok``, ``unclosed_new``, ``prose_only``,
    ``parse_failed``, ``recovery=closed_at_eos`` etc.) so the
    parser-axis attribution is now lifecycle-governed.
  * **Runtime-checkable lifecycle audit** (``audit_capsule_lifecycle``,
    ``audit_capsule_lifecycle_from_view``). Eight invariants
    L-1..L-8 are checked mechanically on a finished run; the
    audit returns OK/BAD/EMPTY plus typed counterexamples
    (Theorem W3-40). The audit is also runnable from a
    forensic ``capsule_view.json`` alone — auditors do not need
    the runtime ctx that produced it.
  * **Deterministic-mode replay** (``RunSpec(deterministic=True)``).
    Strips per-run timestamps / wall-clock fields / host-local
    paths from PROVENANCE / RUN_REPORT / READINESS_CHECK
    payloads and ARTIFACT capsule paths so two runs of the same
    deterministic profile (mock mode, frozen JSONL,
    ``in_process``/``subprocess`` sandbox) produce byte-identical
    full-DAG CIDs and identical chain head (Theorem W3-41).

## What changed in SDK v3.6 (two-Mac distributed-inference integration boundary + real cross-LLM parser-boundary)

Three additive moves. The Wevra single-run product runtime
contract is *byte-for-byte unchanged*; the new surface is one
duck-typed Protocol + two backend adapters + one new research
experiment. The chosen two-Mac path is **MLX distributed
inference** (Apple-official; supports sharding a single
transformer across N Apple Silicon hosts via `mx.distributed`
and `mlx_lm.server`); Hyperspace is **not** the right tool for
this use case (it is distributed-agent infrastructure, not
single-model sharding).

  * **`LLMBackend` Protocol + two concrete backends.**
    `vision_mvp/wevra/llm_backend.py` ships a runtime-checkable
    Protocol (`model`, `base_url`, `generate`) and two
    implementations: `OllamaBackend` (wraps the existing client
    byte-for-byte) and `MLXDistributedBackend` (talks
    OpenAI-compatible `POST /v1/chat/completions` against an
    `mlx_lm.server` launched under `mpirun`). `run_sweep(spec,
    *, ctx=None, llm_backend=None)` accepts an optional
    backend; when None, behaviour is byte-for-byte identical to
    SDK v3.5 (Theorem W5-2). The wire shape is locked
    (Theorem W5-3).
  * **First real cross-LLM parser-boundary measurement (W5-1).**
    `vision_mvp/experiments/parser_boundary_real_llm.py`
    against the live Mac 1 Ollama endpoint yields cross-model
    PARSE_OUTCOME failure-kind TVD = 1.000 between
    `qwen2.5:14b-32k` (14.8B-dense Q4) and `qwen3.5:35b`
    (36B-MoE Q4 `think=False`) under strict parsing on n=10
    instances — the larger model emits the OLD/NEW close as
    `<<` instead of `<<<` and lands entirely in
    `unclosed_new`, while the smaller model emits `<<<`
    cleanly. Robust-mode `recovery=closed_at_eos` collapses
    cross-model TVD to 0.000. The result **inverts the naive
    prediction** that a stronger model would reduce
    parser-boundary instability.
  * **Experimental infrastructure, not product.** The two-Mac
    MLX-distributed path is opt-in research infrastructure;
    Wevra does **not** ship `mlx`, `mlx-lm`, or `mpirun` as
    dependencies. There is deliberately no
    `pip install wevra[mlx_distributed]` extra. The integration
    is one HTTP-client class. See
    `docs/MLX_DISTRIBUTED_RUNBOOK.md` for operator bring-up.

## What changed in SDK v3.5 (multi-agent capsule coordination — research slice)

Three additive moves. The Wevra single-run product runtime
contract is *byte-for-byte unchanged*; the new surface is a
research slice (`vision_mvp.wevra.team_coord` +
`vision_mvp.wevra.team_policy`).

  * **Three new closed-vocabulary capsule kinds.**
    `TEAM_HANDOFF` (capsule-native multi-agent handoff;
    distinct from `HANDOFF` which adapts a substrate
    `TypedHandoff`), `ROLE_VIEW` (per-role admitted view of one
    coordination round; parents = admitted TEAM_HANDOFF cids;
    `max_parents = K_role`, `max_tokens = T_role`), and
    `TEAM_DECISION` (team-level decision; parents = role views
    consulted). A `TeamCoordinator` orchestrates one round
    end-to-end against a shared `CapsuleLedger`.
  * **Mechanically-checked team-lifecycle audit.**
    `audit_team_lifecycle` verifies invariants T-1..T-7 on every
    coordination round (Theorem W4-1, *proved + mechanically-
    checked*). Theorems W4-2 (*proved-conditional*: coverage-
    implies-correctness) and W4-3 (*proved-negative*: per-role
    budget below the role's causal-share floor cannot be rescued
    by *any* admission policy) anchor the team-level mechanism
    formally.
  * **Learned per-role admission policy + Phase-52 reference
    benchmark.** A logistic-regression scorer over six capsule
    features (per-role weights, SGD-trained on a 60-scenario
    partition) admits **strictly fewer handoffs** than the
    strongest fixed admission baseline (coverage-guided) on
    every train seed of the Phase-52 incident-triage bench
    (12/12 seeds; mean savings ≈ 1.26 handoffs per scenario at
    $K_\text{auditor}=8$, $n_\text{eval}=31$, default noise).
    The learned policy *also* improves pooled team-decision
    accuracy on most train seeds (gap on `accuracy_full` > 0 in
    11/12 seeds, mean $+0.054$; gap on `accuracy_root_cause` > 0
    in 8/12 seeds, mean $+0.032$) — but the accuracy advantage
    reverses at higher noise (`spurious_prob = 0.50`). Conjecture
    W4-C1: budget-efficiency dominance is robust per-seed;
    accuracy advantage is mean-positive on the default noise
    config but not strict per-seed; advantage does not survive
    heavier noise.

## What changed in SDK v3.4 (LLM byte boundary + synthetic mode + parser-boundary research)

Four additive moves:

  * **Sub-sub-intra-cell ``PROMPT`` and ``LLM_RESPONSE``
    capsules.** Every LLM call seals two capsules in flight: a
    PROMPT (parent: SWEEP_SPEC) carrying coordinates + prompt
    SHA-256 + byte length + bounded text snippet (≤ 4 KiB),
    and an LLM_RESPONSE (parent: PROMPT) carrying coordinates
    + response SHA-256 + byte length + snippet + elapsed
    milliseconds. Both kinds are content-addressed (Capsule
    Contract C1) so byte-identical prompts (e.g. a cached LLM
    call shared by naive + routing strategies) collapse to one
    capsule. Theorems W3-42 / W3-43 — proved.
  * **End-to-end typed inner-loop chain.** The PARSE_OUTCOME
    capsule's parent set is now either ``(SWEEP_SPEC,)`` (oracle
    path) or ``(SWEEP_SPEC, LLM_RESPONSE)`` (LLM-backed path).
    The full chain is
    ``PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
    TEST_VERDICT`` — five typed capsules with strong
    parent-CID gating. The lifecycle audit's new rule **L-11**
    mechanically verifies coordinate consistency between
    PARSE_OUTCOME and LLM_RESPONSE (Theorem W3-44). The audit
    soundness extends to L-1..L-11 (Theorem W3-45).
  * **In-process synthetic-LLM mode**
    (``SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)``).
    Uses a deterministic synthetic LLM client
    (``vision_mvp.wevra.synthetic_llm.SyntheticLLMClient``) that
    returns canned strings keyed by model tag. The full
    PROMPT / LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL /
    TEST_VERDICT chain runs end-to-end without an Ollama
    endpoint. CI-runnable on every commit.
  * **Cross-model parser-boundary research (W3-C6, empirical).**
    ``vision_mvp.experiments.parser_boundary_cross_model``
    sweeps seven calibrated synthetic distributions through the
    real parser, computes pairwise PARSE_OUTCOME failure-kind
    Total Variation Distance, and reports a parser-mode
    (strict→robust) shift on each distribution. On the bundled
    bank: max cross-distribution TVD = 1.000; max parser-mode
    shift = 1.000 (the ``synthetic.unclosed`` distribution
    flips entirely from ``unclosed_new`` failure under strict
    to ``ok + recovery=closed_at_eos`` under robust). Honest
    scope: synthetic distribution library, not real cross-LLM.

## What remains post-hoc / audit only

The capsule layer is *substantially* load-bearing in execution
now, but a few axes are still post-hoc / not capsule-tracked:

  * **PROMPT / LLM_RESPONSE bytes are now capsule-tracked
    (SDK v3.4).** The legacy "sub-step bytes outside the slice"
    line is **superseded** by Theorems W3-42 / W3-43.
  * **Sandbox stdout / stderr / test trace** remain plain bytes.
    Capsule-tracking these (an APPLY_OUTCOME capsule between
    PATCH_PROPOSAL and TEST_VERDICT) is the natural SDK v3.5
    candidate.
  * **Parser-internal regex / recovery-heuristic state.** The
    parser's intermediate match objects and recovery iteration
    state are not capsule-tracked; PARSE_OUTCOME captures the
    structured verdict only.
  * **The post-hoc ``build_report_ledger`` adapter** is retained
    as the third-party-facing path for code that has a
    ``product_report`` dict from somewhere outside the runtime
    (disk, an HTTP API, another tool). The two paths produce
    CID-equivalent ledgers on the spine kinds (Theorem W3-34
    preserved under SDK v3.2's intra-cell extension and SDK
    v3.3's sub-intra-cell extension); they differ on ARTIFACT
    (real SHA vs None) and transitively on RUN_REPORT.
  * **Re-execution determinism is opt-in only.** Without
    ``deterministic=True`` two runs produce different
    PROVENANCE / RUN_REPORT CIDs (timestamp / wall-clock variance).
    With the flag, the full DAG is reproducible (W3-41); the
    underlying profile must be deterministic (mock mode,
    frozen JSONL).
  * **META_MANIFEST authentication** is a one-hop trust unit.
    Theorem W3-36 establishes that authenticating the manifest
    *itself* within the primary ledger is impossible without
    structural circularity. Cryptographic signing of the
    manifest is orthogonal and out of scope.

## Quick check: which path is my run on?

```python
from vision_mvp.wevra import RunSpec, run, CONSTRUCTION_IN_FLIGHT
report = run(RunSpec(profile="local_smoke", out_dir="/tmp/x"))
assert report["capsules"]["construction"] == CONSTRUCTION_IN_FLIGHT
# in_flight_stats: every proposed capsule sealed.
assert report["capsules"]["in_flight_stats"]["n_failed"] == 0
```

---

## What this repo is

This repository is the home of two coupled things:

1. **Context Zero** — a research programme on *per-agent minimum-sufficient
   context* in multi-agent LLM systems. Theorems, phase shards, an
   EXTENDED_MATH survey, and ~1500 tests of substrate behaviour.
2. **Wevra** — the first shipped product from that programme. A
   **context-capsule runtime**: one `RunSpec` in, one reproducible,
   provenance-stamped, sealed-capsule-DAG report out.

Neither identity subsumes the other. Context Zero is the body of work;
Wevra is the load-bearing product slice of it that a third party can
install, run, and rely on. The capsule abstraction is Wevra's centre
of gravity; the substrate primitives (CASR router, typed handoffs,
escalation threads, adaptive subscriptions) are Wevra's *instances*
of capsule-shaped objects, not its identity.

If you came here because you want to **use** something, you want Wevra.
If you came here because you want to **extend the theory**, the active
canonical entry points are
[`docs/CAPSULE_FORMALISM.md`](CAPSULE_FORMALISM.md) (run-boundary,
W3 family), [`docs/CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md)
(team-boundary, W4 family), and
[`docs/THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md). The pre-Wevra theory
volumes (`PROOFS.md`, `EXTENDED_MATH_[1-7].md`, `OPEN_QUESTIONS.md`,
`FRAMEWORK.md`, the original 12 theorems and 72-framework survey) are
preserved under [`docs/archive/pre-wevra-theory/`](archive/pre-wevra-theory/);
they are historical record, not the active position. The phase-by-phase
research diary lives in `vision_mvp/RESULTS_PHASE*.md`. The long-running
master plan is [`docs/context_zero_master_plan.md`](context_zero_master_plan.md).

---

## One-sentence summary per layer

| Layer | One sentence | Stability |
|---|---|---|
| **Context Capsule primitives** (`wevra.capsule`) | `ContextCapsule` + `CapsuleLedger` + `CapsuleView`: the load-bearing SDK abstraction. Every cross-boundary artefact is a typed, content-addressed, lifecycle-bounded, budget-bounded, provenance-carrying capsule. | **Stable v1** (contract C1..C6). |
| **Capsule-native runtime** (`wevra.capsule_runtime`) | `CapsuleNativeRunContext` + `seal_and_write_artifact` + intra-cell `seal_patch_proposal` / `seal_test_verdict` + sub-intra-cell `seal_parse_outcome` (SDK v3.3) + detached `seal_meta_manifest`: capsules drive runtime stage transitions at run boundaries AND inside the inner sweep loop AND on the parser axis; substantive artefacts are content-addressed at write time and re-verifiable at audit time. The capsule layer is the runtime's typed execution contract on the spine and now extends two structural layers past the cell boundary. | **Stable v3** (theorems W3-32 / W3-33 / W3-34 / W3-35 / W3-32-extended / W3-36 / W3-37 / W3-38 / W3-39 / W3-40 / W3-41). |
| **Lifecycle audit** (`wevra.lifecycle_audit`) | `CapsuleLifecycleAudit` + `audit_capsule_lifecycle_from_view`: mechanically verifies eight lifecycle invariants L-1..L-8 over a finished run (Theorem W3-40). Returns OK / BAD / EMPTY plus typed counterexamples. Runnable from runtime ctx OR from on-disk `capsule_view.json` alone. | **Stable v1** (SDK v3.3). |
| **Wevra SDK** (`vision_mvp.wevra`) | Profile-driven context-capsule runtime for SWE-bench-Lite-shape banks; `RunSpec` → provenance-stamped report whose root is a sealed `RUN_REPORT` capsule + a detached `meta_manifest.json` witness. ``RunSpec.deterministic=True`` opt-in collapses CIDs across runs (Theorem W3-41). | **Stable v3.3** — public contract. |
| **Wevra console scripts** | `wevra`, `wevra-import`, `wevra-ci`, `wevra-capsule` — installed by `pip install wevra`. | **Stable v3**. |
| **Wevra extension protocols** (`wevra.extensions`) | `SandboxBackend`, `TaskBankLoader`, `ReportSink` — runtime-checkable Protocols, discovered via `importlib.metadata.entry_points`. | **Stable v1**. |
| **Unified runtime** (`wevra.runtime`) | `SweepSpec` + `run_sweep`: one code path for mock and real-LLM runs, with an explicit `acknowledge_heavy` cost gate. Every sweep cell becomes a `SWEEP_CELL` capsule. | **Stable v1**. |
| **Legacy product path** (`vision_mvp.product`) | Pre-Wevra import path. Still works; re-exported by `wevra`. | **Deprecated-compat** — do not import in new code. |
| **Core substrate** (`vision_mvp.core`) | CASR routing, hierarchical router, context ledger, exact_ops, typed role-handoff, dynamic_comm, adaptive_sub. Research primitives Wevra rests on; each is adapter-able into the capsule surface (`capsule_from_handle`, `capsule_from_handoff`, …). | **Settled, but research API** — no SDK guarantees. |
| **Research shards** (`vision_mvp.experiments`, `vision_mvp.tasks`, `vision_mvp/RESULTS_PHASE*.md`, archived `EXTENDED_MATH_*.md`) | 53+ phases of falsifiability experiments, 72-framework theory survey (archived under `docs/archive/pre-wevra-theory/`), proofs. | **Research-grade** — empirical/proved per shard; no product-API guarantee. |
| **Multi-agent capsule coordination** (`wevra.team_coord` + `wevra.team_policy`) | SDK v3.5 (research slice). `TEAM_HANDOFF` / `ROLE_VIEW` / `TEAM_DECISION` capsules + `TeamCoordinator` + `audit_team_lifecycle` (T-1..T-7) + learned per-role admission policy. Theorems W4-1 (proved + mechanically-checked) / W4-2 (proved-conditional) / W4-3 (proved-negative); Conjecture W4-C1 (empirical-positive on default config). | **Research-grade v1** — additive on top of the Wevra product surface; not part of the run-boundary product contract. |
| **Boundary / next-slice** | Docker-first-by-default for public/untrusted JSONLs; first real out-of-tree plugin exemplar; release-on-tag firing. | **Declared, not fired** — see master plan § 10.5. |

For the full living stability matrix, see
[`context_zero_master_plan.md` § 10.1](context_zero_master_plan.md#101-stability-matrix-living).

---

## Minimal mental model

```
    Context Zero (research programme)
    ├── Theory (active):   docs/CAPSULE_FORMALISM.md (W3 family),
    │                      docs/CAPSULE_TEAM_FORMALISM.md (W4 family),
    │                      docs/THEOREM_REGISTRY.md
    ├── Theory (archived): docs/archive/pre-wevra-theory/
    │                      (PROOFS.md, EXTENDED_MATH_[1-7].md, OPEN_QUESTIONS.md, …)
    ├── Substrate: vision_mvp/core/*  (CASR router, exact memory,
    │                                   typed handoff, runtime calibration)
    ├── Research shards: vision_mvp/experiments/*, vision_mvp/tasks/*,
    │                    vision_mvp/RESULTS_PHASE*.md (empirical diary, per phase)
    │
    └── Wevra (shipped product slice)
        ├── SDK:       vision_mvp/wevra/          (stable contract)
        ├── CLI:       wevra / wevra-import / wevra-ci / wevra-capsule
        ├── Runtime:   wevra.runtime.run_sweep    (mock + real, one path)
        ├── Plugins:   wevra.extensions           (3 Protocols + registry)
        ├── Schemas:   phase45.product_report.v2, wevra.provenance.v1, …
        ├── Latest milestone: docs/RESULTS_WEVRA_W23_CROSS_CELL_DELTA.md (SDK v3.24)
        ├── Older milestones: docs/archive/wevra-milestones/ (SDK v3.0 → v3.6)
        └── Legacy:    vision_mvp/product/*       (deprecated-compat)
```

The rule of thumb: **anything imported from `vision_mvp.wevra` is
product; anything else is research substrate or research shard.**

---

## What Wevra is — and what it is not

**Wevra IS:** a drop-in SDK for profile-driven evaluation runs on
SWE-bench-Lite-shape task banks, with a stable report schema, a CI
gate, a provenance manifest on every run, and a plugin surface for
sandboxes / task banks / report sinks.

**Wevra is NOT:**
- The whole Context Zero research programme.
- A universal multi-agent platform.
- A replacement for SWE-bench harnesses on arbitrary-shape tasks.
- An orchestrator for training runs or long-lived agent services.

The distinction matters: Wevra is deliberately narrow so that what it
does, it does with proofs and provenance. Scope creep is resisted on
purpose.

---

## Fastest path from zero to a real report

```bash
git clone <this-repo>
cd context-zero
pip install -e .[docker]            # Docker extra optional, recommended for public JSONLs
wevra --profile local_smoke --out-dir /tmp/wevra-smoke
wevra-ci       --report /tmp/wevra-smoke/product_report.json --min-pass-at-1 1.0
wevra-capsule  view   --report /tmp/wevra-smoke/product_report.json
wevra-capsule  verify --report /tmp/wevra-smoke/product_report.json
```

Four files of interest land in `/tmp/wevra-smoke/`:

- `product_report.json`   — machine-readable report (`phase45.product_report.v2`),
  includes a `capsules` block (`wevra.capsule_view.v1`).
- `capsule_view.json`     — the sealed capsule DAG on disk.
- `provenance.json`       — reproducibility manifest (`wevra.provenance.v1`).
- `product_summary.txt`   — human summary with a capsule-kind histogram,
  `chain_ok` flag, and the RUN_REPORT capsule's root CID.

Send someone the `root_cid` plus the JSONL SHA-256 (already recorded
in `provenance.json`) and they have everything needed to reproduce,
audit, and verify the run — no out-of-band trust required.

For a real-LLM sweep, set `WEVRA_OLLAMA_URL` and add `--acknowledge-heavy`.

---

## Where to go from here

- **I want to use Wevra** → [`README.md § Wevra SDK quick start`](../README.md) and
  [`vision_mvp/wevra/__init__.py`](../vision_mvp/wevra/__init__.py) (module docstring
  lists the full public surface).
- **I want to extend Wevra** → [`vision_mvp/wevra/extensions/`](../vision_mvp/wevra/extensions/)
  and the `examples/out_of_tree_plugin/` folder (minimal standalone package
  demonstrating the `entry_points` path).
- **I want to understand the substrate** → [`ARCHITECTURE.md`](../ARCHITECTURE.md)
  (skip the per-phase callouts on first read) and
  [`context_zero_master_plan.md` § 3](context_zero_master_plan.md).
- **I want the active theory** →
  [`CAPSULE_FORMALISM.md`](CAPSULE_FORMALISM.md) (W3 family),
  [`CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md) (W4 family),
  [`THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md), and
  [`context_zero_master_plan.md` § 1–2](context_zero_master_plan.md).
- **I want the historical theory** → pre-Wevra theory volumes are
  preserved under [`archive/pre-wevra-theory/`](archive/pre-wevra-theory/)
  (`PROOFS.md`, `EXTENDED_MATH_[1-7].md`, `OPEN_QUESTIONS.md`,
  `FRAMEWORK.md`, `MVP.md`, `EVALUATION.md`, `ROADMAP.md`,
  `VISION_MILLIONS.md`, `MATH_AUDIT.md`).
- **I want the research diary** → `vision_mvp/RESULTS_PHASE*.md` in
  order; the latest milestone is
  [`RESULTS_WEVRA_COMPOSED_REAL_LLM.md`](RESULTS_WEVRA_COMPOSED_REAL_LLM.md)
  (SDK v3.17); older Wevra milestone notes are under
  [`archive/wevra-milestones/`](archive/wevra-milestones/).

---

*This document is the canonical orientation. If it and any other file
disagree on identity or scope, this document is right and the other
file is stale.*
