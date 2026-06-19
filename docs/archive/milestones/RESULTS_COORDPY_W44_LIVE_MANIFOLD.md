# RESULTS — W44 Live Manifold-Coupled Coordination (LMCC)

> Programme step: post-W43, post-CoordPy 0.5.20. Mints axis 41 of
> the Context Zero programme. Strictly additive on top of the W43
> PMC layer and the released v3.43 line. The released SDK contract
> is byte-for-byte unchanged; the W44 surface lives at
> `coordpy.live_manifold` and is reachable only through an explicit
> import.

## TL;DR

W44 introduces the **first capsule-native live-coupled manifold
layer** in CoordPy. Where W43 built a closed-form, deterministic,
zero-parameter Product-Manifold Capsule (PMC) over the cells of a
team run, W44 lets three of those W43 channels (spherical / subspace
/ causal) actually *change run behaviour* in a sequential agent
team, and lets a fourth channel (factoradic route) actually *reduce
the visible prompt-token cost* on a real backend.

The R-91 benchmark family across 5 seeds × 7 cell families produces
the following honest, repeatable, code-backed results vs the
released `AgentTeam` baseline and the W43 closed-form audit-only
arm:

| family | metric | baseline_team | w43_closed_form | w44_live_coupled | Δ live vs w43 |
|---|---|---|---|---|---|
| `r91_trivial_live_passthrough` | passthrough_ok | 1.000 | 1.000 | 1.000 | +0.000 (sanity) |
| `r91_live_causal_gate` | downstream_protect_rate | 0.600 | 0.600 | **1.000** | **+0.400** |
| `r91_live_spherical_gate` | downstream_protect_rate | 0.600 | 0.600 | **1.000** | **+0.400** |
| `r91_live_subspace_gate` | downstream_protect_rate | 0.600 | 0.600 | **1.000** | **+0.400** |
| `r91_live_factoradic_compression` | visible_tokens_saved_per_run | 0 | 0 | **314** | **+314** |
| `r91_live_falsifier` | no_false_abstain | 1.000 | 1.000 | 1.000 | +0.000 (no over-claim) |
| `r91_live_dual_channel_collusion` | downstream_protect_rate | 0.000 | 0.000 | 0.000 | +0.000 (limitation reproduces) |

All H1..H10 hypotheses of the pre-committed success criterion
(`docs/SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md`) pass cleanly. The
released CoordPy 0.5.20 stable smoke driver
(`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED" with the
W44 module on disk.

## What is shipped

* **`coordpy/live_manifold.py`** (~1100 LoC, dependency-free): the
  W44 layer. `LiveManifoldRegistry` + `LiveManifoldOrchestrator`
  + `LiveManifoldTeam` + `LiveManifoldHandoffEnvelope` +
  12-mode verifier + builders. Composes with the W43 inner via
  the same `ProductManifoldPolicyRegistry`.

* **`coordpy/r91_benchmark.py`** (~700 LoC, dependency-free):
  the R-91 benchmark family. Seven cell families, three honest
  baselines (`baseline_team`, `w43_closed_form`,
  `w44_live_coupled`), 5-seed aggregator, text-report renderer.
  Comparable to the R-69..R-90 ladder under
  `docs/RESULTS_COORDPY_W*.md`.

* **`tests/test_live_manifold.py`** (32 tests): channels, gates,
  factoradic compressor, verifier (one test per disjoint failure
  mode), schema invariants.

* **`tests/test_r91_benchmark.py`** (19 tests): each H1..H10
  hypothesis is exercised directly; aggregator + per-family +
  per-seed regression coverage.

* **`docs/SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md`**: the
  pre-committed success bar.

## What was NOT done (honest scope)

W44 is a capsule-layer milestone with *behavioural coupling* on top
of the released `AgentTeam` contract. It does NOT close any of:

* **`W43-C-MIXED-CURVATURE-LATENT`** — full transformer-internal
  mixed-curvature attention. The W44 mechanism couples the W43
  channels to *prompt-construction* and *gating* decisions; it
  does not modify the model's attention computation.

* **`W43-C-COLLECTIVE-KV-POOLING`** — host-collective KV-cache
  sharing. W44 operates on a single backend (synthetic or real
  Ollama) within a single process; it does not pool KV state.

* **`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`** — a true continuous
  Gr(k, d) homotopy. The subspace channel still captures a single
  point on the Grassmannian per cell; the W44 layer simply gates
  the run on principal-angle drift.

W44 does NOT claim that real LLMs decode the factoradic header.
The factoradic compressor's *behavioural* effect is measured on a
deterministic `SyntheticLLMClient` testbed; on real LLMs the saving
is a visible-token saving without a guaranteed behavioural-decoding
gain. The bounded local-Ollama probe below is a **realism anchor
only** — it is not load-bearing for the H1..H10 success bar.

## Live mechanism (PART C)

The W44 orchestrator wraps the W43 product-manifold orchestrator
plus the released `AgentTeam` contract. For each turn:

1. The observation builder (default
   `default_live_observation_builder` or a custom callable) emits a
   `CellObservation` from the per-turn state (turn index, role
   arrival order, recent handoffs, causal counts).
2. The orchestrator runs the W43 PMC orchestrator under the hood
   to produce a closed-form decision branch (one of `pmc_ratified`,
   `pmc_*_abstained`, etc.) and a sealed W43 envelope.
3. The W43 branch is mapped to a W44 branch
   (`live_ratified`, `live_*_abstain`, etc.).
4. **If the live branch is an abstain branch AND
   `abstain_substitution_enabled` is True**, the orchestrator
   substitutes a deterministic abstain output for the agent's
   `generate()` call. The downstream agents see the abstain
   marker on their visible-handoff list, *not* the bad upstream
   output.
5. **If `inline_route_mode == factoradic`**, the prompt builder
   replaces the textual rendering of the role-arrival ordering
   with a single `FACTORADIC_ROUTE: <int>` header. The full
   permutation is recoverable bijectively from the integer.
6. The orchestrator seals a `LiveManifoldHandoffEnvelope` carrying
   the W43 envelope CID, the team-handoff capsule CID, the
   prompt-construction witness CID, the live witness CID, and the
   live outer CID. The envelope's outer CID is content-addressed
   from every other field; tampering with any subfield is detected
   by name through one of 12 disjoint failure modes.

The released `AgentTeam.run` is byte-for-byte unchanged; the W44
surface is a new `LiveManifoldTeam` class that sits beside it.

## Architecture triage (PART B)

| W43 component | W44 bucket | Status |
|---|---|---|
| Spherical consensus channel | active live gate | gates downstream prompts via abstain substitution |
| Subspace drift channel | active live gate | same |
| Causal-clock channel | active live gate | same |
| Factoradic route channel | live compressor | replaces textual ordering with single integer in prompt |
| Hyperbolic branch channel | audit-only | recorded in W43 envelope; not yet behaviorally consumed |
| Euclidean attribute channel | audit-only | recorded in W43 envelope; not yet behaviorally consumed |
| Manifest-v13 envelope | audit + provenance | bound under live envelope as `parent_w43_envelope_cid` |
| Verifier (18 modes) | audit + provenance | retained byte-for-byte; W44 adds 12 disjoint modes |

Cumulative trust boundary across W22..W44 = **226 enumerated
capsule-layer failure modes** (W22..W42 cumulative 196, plus 18
new at W43, plus 12 new at W44).

## Capsule strengthening (PART D)

W44 strengthens the released TEAM_HANDOFF capsule with:

1. **Typed live-coupled handoff envelope.** The
   `LiveManifoldHandoffEnvelope` is a frozen dataclass binding the
   underlying TEAM_HANDOFF CID, the W43 envelope CID, the gating
   decision branch + reason, the prompt-construction witness CID,
   the inline route mode, and the visible-token saving — all
   under one content-addressed `live_outer_cid`.

2. **Channel provenance.** Each envelope records the W43 channel
   that drove the gating decision (`pmc_branch`,
   `abstain_reason`) plus the W43 manifold-state CID, so an
   auditor can prove which channel triggered which behaviour.

3. **Causal-gating provenance.** The `causal_admissible` field
   on the W43 inner is preserved; an auditor can prove that an
   abstain branch was triggered by a Lamport partial-order
   violation rather than a forged causal clock (modulo
   `W43-L-FORGED-CAUSAL-CLOCK-CAP`).

4. **Prompt-construction witness.** The
   `prompt_construction_witness_cid` binds together the prompt
   SHA-256, the inline route mode, the factoradic integer, the
   factoradic capacity, and the visible-token textual/actual
   counts. Replay can reconstruct the exact bytes the model saw,
   including the factoradic compression substitution.

5. **Replay artifact for model-coupled runs.** Every
   `LiveManifoldTeam.run` produces both a `team_capsule_view`
   (compatible with `coordpy-capsule verify-view`) and a
   per-turn list of live envelopes. The chain is verifiable from
   bytes alone.

W44 is strictly additive: when configured trivially, the released
TEAM_HANDOFF capsule is sealed unchanged.

## Token / bit cramming with model-facing transfer (PART E)

The W44 layer extends the W43 cram frontier with *visible-token
saving on real backends*:

* W43 already crams **1808 structured bits per visible-token of
  overhead** at n=8 roles (the W43-EMPIRICAL bar).
* W44 preserves all 1808 W43 structured bits per envelope while
  **reducing the visible prompt-token cost by 4..16 tokens per
  active turn** in factoradic compression mode at `n_roles=8`.

On the R-91 `r91_live_factoradic_compression` family (synthetic
backend, n_roles=8, max_visible_handoffs=4), the live arm saves
**314 visible prompt tokens per run** vs the textual baseline
(min == max across 5/5 seeds).

The cramming frontier metric this milestone introduces is
``visible_tokens_saved_per_run`` on the W44 envelope; the W43
``cram_factor_w43`` metric is preserved.

## Discrete routing in real runs (PART F)

The factoradic side-channel is the only *purely discrete* W43
channel. Under W44:

* In `inline_route_mode = factoradic`, the prompt's role-arrival
  ordering is encoded as a single integer header instead of n
  textual lines.
* The route decodes bijectively back to the original permutation
  via the public `decode_factoradic_route` API. The audit chain
  records the integer and its capacity; the canonical permutation
  is recoverable from the bytes alone.
* The synthetic-backend bench measures a clean **+8 tokens saved
  per turn at n_roles=8** lower bound (averaged across 8 turns
  with bounded visible-handoff window).

Honest verdict: the discrete factoradic channel produces a real
visible-token saving even when the model is indifferent to the
header content. When the model *does* respect the route ordering,
the saving is preserved without any behavioural regression.

## Per-channel verdicts under live execution (PART I)

| Channel | Live verdict | R-91 evidence |
|---|---|---|
| Spherical | active gate; behaviourally meaningful | `r91_live_spherical_gate` Δ +0.400 |
| Subspace | active gate; behaviourally meaningful | `r91_live_subspace_gate` Δ +0.400 |
| Causal-clock | active gate; behaviourally meaningful | `r91_live_causal_gate` Δ +0.400 |
| Factoradic | live compressor; saves visible tokens | `r91_live_factoradic_compression` +314 |
| Hyperbolic | **audit-only** | not exercised by R-91 (depth recorded but no behavioural gate) |
| Euclidean | **audit-only** | not exercised by R-91 (attributes recorded but no behavioural gate) |

The hyperbolic and euclidean channels remain *structurally strong
but behaviourally inert* at the live layer. They could become
behaviourally meaningful only via either (a) a learned latent
controller that consumes them (substrate-blocked), or (b) a
domain-specific observation builder that maps them to a
behavioural signal (out of capsule-layer scope at this milestone).

## Live / replay-live evaluation (PART H)

The R-91 benchmark family runs against a deterministic
`SyntheticLLMClient` so the results are reproducible from a fixed
seed set with no external dependency.

A bounded *realism anchor* probes the W44 mechanism against a real
local Ollama backend (`qwen2.5:0.5b` at `localhost:11434`):

* `LiveManifoldTeam` with the trivial registry: 4 agents,
  bounded-context default, 3.5s wall total, real model output,
  capsule chain sealed cleanly.
* `LiveManifoldTeam` with the factoradic compressor and a clean
  policy: 4 agents, 1.5s wall total, **31 visible prompt tokens
  saved across 3 active turns**, all turns ratified, capsule chain
  sealed cleanly.

These probes confirm that the W44 surface survives contact with a
real local LLM and that the factoradic compressor reduces real
prompt tokens. They are *realism anchors only*, not load-bearing
for the H1..H10 bar.

## Theory and limitations (PART J)

### W44-T-LIVE-COUPLING-DETERMINISM (code-backed)

For any deterministic backend (e.g. `SyntheticLLMClient`), two
independent runs of `LiveManifoldTeam.run` over the same task with
the same registry, agents, and observation builder produce
byte-identical `final_output`, byte-identical capsule chain root
CIDs, and byte-identical sequences of `LiveManifoldHandoffEnvelope`.

*Witness*: `tests/test_live_manifold.py::TestTrivialLivePassthrough`
+ `tests/test_r91_benchmark.py::TestPerSeedDeterminism` (per-seed
regressions).

### W44-T-LIVE-GATE-SOUNDNESS (proved-conditional)

When the policy registry's expected-channel state is honest and the
observation faithfully reflects the cell's true state, the live
gate substitutes the abstain output on every turn at which the W43
PMC orchestrator detects a violation; the substituted output
propagates to the downstream agent's visible-handoff list,
strictly reducing the rate at which a bad upstream handoff
propagates to a downstream `generate()` call.

*Witness*: R-91 families `r91_live_causal_gate`,
`r91_live_spherical_gate`, `r91_live_subspace_gate` each register
+0.400 mean improvement over the W43 closed-form arm with min ==
max across 5/5 seeds.

### W44-T-FACTORADIC-COMPRESSION-SOUNDNESS (proved-conditional)

When `inline_route_mode == factoradic`, the prompt-construction
witness CID binds the factoradic integer to the actual prompt
bytes; the integer is bijective with the role-arrival permutation
(W43-T-FACTORADIC-BIJECTION). Therefore the route is recoverable
byte-for-byte from the audit envelope while the visible prompt
emits a single integer header instead of n textual lines.

*Witness*: `tests/test_live_manifold.py::TestFactoradicCompressor`
plus the R-91 `r91_live_factoradic_compression` family.

### W44-L-TRIVIAL-LIVE-PASSTHROUGH (proved by inspection)

When `LiveManifoldRegistry.is_trivial` is True
(`live_enabled=False`, `inline_route_mode='textual'`,
`abstain_substitution_enabled=False`, W43 inner trivial), the
live orchestrator emits no envelope side effects beyond the W43
trivial-passthrough envelope, and `LiveManifoldTeam.run` produces
the same `final_output`, `n_turns`, and capsule chain head as
`AgentTeam.run` for the same backend.

*Witness*: `tests/test_live_manifold.py::TestTrivialLivePassthrough
::test_trivial_run_matches_agent_team` and the R-91
`r91_trivial_live_passthrough` family (5/5 seeds, all three arms).

### W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP (proved-conditional limitation)

When an adversary forges BOTH the spherical channel signature AND
the subspace basis to match the registered policy, the live gate
ratifies on every turn — the W44 mechanism cannot recover at the
capsule layer. This limitation is the live counterpart of
`W43-L-DUAL-CHANNEL-COLLUSION-CAP`. Recovery requires either
(a) a stricter role-handoff signature policy (forcing the attacker
to also forge the W42 invariance signature) or (b) native-latent
evidence outside the capsule layer
(`W43-C-MIXED-CURVATURE-LATENT`).

*Witness*: R-91 `r91_live_dual_channel_collusion` family —
`live.mean == w43.mean == 0.0` across 5/5 seeds.

### W44-L-MODEL-INDIFFERENCE-CAP (proved-conditional limitation)

The factoradic compressor reduces the visible prompt-token cost
unconditionally, but the *behavioural* effect of the substitution
on a real LLM is **not** guaranteed. If the model is indifferent to
the route (i.e. it does not depend on the explicit role-arrival
ordering), the saving is purely visible-token-cost; if the model
*does* depend on the ordering, the factoradic header must be
decoded by the model to preserve correctness. The W44 layer does
not require the model to decode the header; the saving is therefore
a *cost saving*, not a *capability lift*, on real LLMs.

The bounded local-Ollama probe at temperature 0 confirms that the
saving is real (31 visible tokens across 3 turns of `qwen2.5:0.5b`)
without claiming a behavioural gain.

### W44-C-LIVE-LATENT (conjectural)

The hyperbolic and euclidean channels remain audit-only at the
live layer; promoting them to behavioural channels requires either
a learned controller that consumes them (substrate-blocked) or a
domain-specific observation builder that maps them to a
behavioural signal. This is conjectural and out of capsule-layer
scope.

## Product-boundary decisions (PART K, M)

The released CoordPy 0.5.20 stable surface is byte-for-byte
unchanged. The W44 module ships in the source tree but is **not**
re-exported through `coordpy/__init__.py` and is **not** listed in
`coordpy.__experimental__`. The first-run UX (`coordpy-team run
--preset quant_desk ...`) is unaffected; the smoke driver
(`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED".

A sophisticated caller reaches the W44 surface explicitly:

```python
from coordpy.live_manifold import (
    LiveManifoldTeam,
    build_live_manifold_registry,
    W44_ROUTE_MODE_FACTORADIC,
)
from coordpy.product_manifold import (
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.r91_benchmark import run_all_families, render_text_report

# 1. Define a policy that gates the spherical / subspace / causal
#    channels, with the factoradic compressor on.
# 2. Build a LiveManifoldTeam over your agents + a real backend.
# 3. Run; replay later from the sealed envelopes.

results = run_all_families()
print(render_text_report(results))
```

This explicit import reflects the milestone's research-grade
status. A future milestone may promote a stable subset of the W44
surface under `coordpy.__experimental__` once cross-host live
evidence is acquired.

## Validation (PART M)

* **Baseline regression**: `tests/test_smoke_full.py` reports
  "ALL CHECKS PASSED" with the W44 module on disk.
* **W43 regression**: `coordpy/r90_benchmark.py::run_all_families`
  reproduces the W43 R-90 results byte-for-byte; no R-90 family
  is perturbed by the W44 module.
* **W44 unit tests**: `tests/test_live_manifold.py` — 32 tests
  passed, including all 12 verifier-failure-mode tests.
* **R-91 H1..H10**: `tests/test_r91_benchmark.py` — 19 tests passed,
  exercising every pre-committed hypothesis.
* **Aggregate test count**: 109 tests passed across the full
  `tests/` directory (W43 PMC + R-90 + W44 LMCC + R-91).
* **Realism anchor**: live `LiveManifoldTeam` against
  `qwen2.5:0.5b` at `localhost:11434` records 31 visible prompt
  tokens saved across 3 active turns at temperature 0.

## Where this leaves the programme

W44 is the first capsule-layer milestone in CoordPy that lets the
W43 product-manifold channels actually *change run behaviour* while
preserving full content-addressed audit and producing strict
empirical gains over both the released `AgentTeam` baseline and
the W43 closed-form audit-only arm on three distinct R-91 gating
families and one compressor family.

The remaining open frontiers are the W43 conjectures
(`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) plus the new W44 conjecture
(`W44-C-LIVE-LATENT`); these require new architectural substrate
beyond the capsule layer and are explicitly out of scope for the
W44 milestone.
