# Pre-committed success criterion — W44 Live Manifold-Coupled Coordination

> Programme step: post-W43. Mints axis 41 of the Context Zero
> programme. Strictly additive on top of the W43 PMC layer and the
> released v3.43 line. Honest scope: the W44 mechanism couples the
> W43 product-manifold state to *actual agent run behaviour* through
> a manifold-aware prompt builder and a manifold-conditioned gating
> policy. It does NOT close the W43 conjectures
> (`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) — these still require new
> substrate beyond the capsule layer.

## Mechanism

W44 introduces **Live Manifold-Coupled Coordination (LMCC)** — the
first capsule-native layer in CoordPy that lets the W43
product-manifold channels actually *change run behaviour* in a
sequential agent team. The mechanism couples three W43 channels
(spherical / subspace / causal) to per-turn *gating decisions* that
either ratify, abstain, or compress the next agent's prompt, and
binds the gating decision to a strengthened
`coordpy.live_manifold_handoff.v1` envelope that records both the
W43 manifold provenance and the prompt-construction witness.

The four live-coupled mechanisms are:

* **Causal admissibility gate** — when the per-turn Lamport
  vector-clock sequence violates the partial order, the live
  orchestrator substitutes a deterministic abstain output for the
  agent's `generate()` call, sealing a `live_causal_abstain`
  envelope. The downstream agent prompts therefore never see the
  bad upstream handoff.

* **Spherical consensus gate** — when the observed spherical
  signature disagrees with the registered policy entry, the live
  orchestrator substitutes an abstain output and seals a
  `live_spherical_abstain` envelope. Same downstream semantics.

* **Subspace drift gate** — when the observed subspace basis
  drifts beyond the registered tolerance, the live orchestrator
  substitutes an abstain output and seals a `live_subspace_abstain`
  envelope. Same downstream semantics.

* **Factoradic route compressor** — when `inline_route_mode =
  factoradic`, the live prompt builder *replaces* the textual
  rendering of the role-arrival ordering with a single integer
  header (`FACTORADIC_ROUTE: <int> over <role_universe>`),
  reducing the visible prompt-token cost while preserving the
  full route in the audit envelope.

Two W43 channels remain audit-only at the live layer:

* The **hyperbolic branch channel** records branch depth + path
  hash; the live layer does not yet behaviorally consume it.
* The **euclidean attribute channel** records cell attributes; the
  live layer does not yet behaviorally consume it.

W44 is strictly additive on top of W43 and the released v3.43 SDK.
When the live registry is configured trivially
(`live_enabled=False`, `inline_route_mode=textual`,
`abstain_substitution_enabled=False`), the live orchestrator
reduces to the released `AgentTeam` byte-for-byte (the
**W44-L-TRIVIAL-LIVE-PASSTHROUGH** falsifier).

## Pre-committed hypotheses (H1..H10)

Each hypothesis is testable from the bundled R-91 benchmark family
(`coordpy.r91_benchmark`) and its accompanying pytest tests
(`tests/test_r91_benchmark.py`). Pre-committed seed set:
`(0, 1, 2, 3, 4)`. Backend: deterministic
`SyntheticLLMClient` with controlled per-instance responses (the
honest live-coupling testbed; a real Ollama backend is supported
through the same surface but is not load-bearing for the
pre-committed bar).

* **H1 — Trivial live passthrough is byte-for-AgentTeam.**
  Family `r91_trivial_live_passthrough`: with the trivial registry,
  `LiveManifoldTeam.run` produces the same `final_output`,
  `turns`, and `capsule_view` chain head as the released
  `AgentTeam.run` for the same backend.
  **Pass if**: `passthrough_ok = 1.0` across 5/5 seeds.

* **H2 — Causal-violation gate strictly improves downstream
  protection.**
  Family `r91_live_causal_gate`: half the cells inject an
  out-of-order causal clock at index 1 or 2. The W42 baseline and
  the W43 closed-form arm cannot intervene before the next
  agent's `generate()` runs, so the next agent sees the bad
  upstream handoff. The W44 live arm substitutes the abstain
  output before the next prompt is built.
  **Pass if**: `live.mean - w43.mean >= 0.40` AND `live.min == 1.0`
  across 5/5 seeds.

* **H3 — Spherical-divergence gate strictly improves downstream
  protection.**
  Family `r91_live_spherical_gate`: half the cells emit divergent
  claim_kinds. Same comparison.
  **Pass if**: `live.mean - w43.mean >= 0.40` AND `live.min == 1.0`
  across 5/5 seeds.

* **H4 — Subspace-drift gate strictly improves downstream
  protection.**
  Family `r91_live_subspace_gate`: half the cells drift the
  subspace basis to an orthogonal subspace. Same comparison.
  **Pass if**: `live.mean - w43.mean >= 0.40` AND `live.min == 1.0`
  across 5/5 seeds.

* **H5 — Factoradic compressor strictly reduces visible-prompt
  tokens on permutation-heavy regimes.**
  Family `r91_live_factoradic_compression`: with `n_roles >= 6`,
  the W44 live arm sends a compact `FACTORADIC_ROUTE: <int>`
  header instead of the textual list of n role names, while
  preserving the full route in the audit envelope.
  **Pass if**: `live.min - baseline.max >= 4` saved visible
  prompt tokens per turn at `n_roles=8`, AND the recovered
  permutation from the factoradic CID matches the registered
  permutation byte-for-byte across 5/5 seeds.

* **H6 — No false abstention on the linear-flow falsifier.**
  Family `r91_live_falsifier`: a clean linear-flow regime where
  the geometry adds nothing must NOT trigger spurious abstentions
  in the live arm.
  **Pass if**: `live.min == max == 1.0`.

* **H7 — Live dual-channel collusion limitation
  (W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP).**
  Family `r91_live_dual_channel_collusion`: when an adversary
  forges BOTH the spherical signature AND the subspace basis, the
  W44 live arm cannot recover at the capsule layer.
  **Pass if**: `live.mean == w43.mean == 0.0` across 5/5 seeds
  (a *limitation* falsifier — passing means the limitation
  reproduces honestly).

* **H8 — Live envelope verifier enumerates >= 12 disjoint failure
  modes.**
  `verify_live_manifold_handoff` returns the empty-envelope
  failure as the first mode and recomputes every component CID;
  tampering with any subfield is detected through one of the
  disjoint named modes.
  **Pass if**: a successful verification reports `n_checks >= 12`.

* **H9 — Live-coupled cram-frontier preserves W43 audit while
  reducing visible prompt tokens.**
  On the `r91_live_factoradic_compression` family, the structured
  bits per *envelope* (W43's >= 1808 bits at n=8) are preserved
  byte-for-byte, while the *visible prompt tokens per turn* drop
  by at least 4 tokens at `n_roles=8`. The live envelope's
  `n_visible_tokens_saved_factoradic` field reports the saving
  per-turn.
  **Pass if**: `live.min == max >= 1808` structured bits AND
  `live.min - baseline.max >= 4` visible tokens saved across
  5/5 seeds.

* **H10 — Stable SDK contract preserved.**
  CoordPy 0.5.20's stable smoke driver
  (`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED" with
  the W44 module on disk. The W44 surface is held outside
  `coordpy.__experimental__`; it is reachable only through an
  explicit `from coordpy.live_manifold import ...` import. The
  released v0.5.20 wheel's public surface is byte-for-byte
  unchanged.
  **Pass if**: smoke driver reports "ALL CHECKS PASSED" with the
  W44 module installed.

## Falsifiers + per-channel verdicts

Each W43 channel is force-verdicted at the live layer:

| Channel | Live verdict | Behavioural? | Audit? |
|---|---|---|---|
| Spherical consensus | active gate | yes | yes |
| Subspace drift | active gate | yes | yes |
| Causal clock | active gate | yes | yes |
| Factoradic route | compressor | yes (visible tokens) | yes |
| Hyperbolic branch | audit-only | no | yes |
| Euclidean attribute | audit-only | no | yes |

A channel fails its verdict if the corresponding R-91 family does
not register the predicted gain (gates) or saving (compressor).
Hyperbolic and euclidean are explicitly *not* required to produce
behavioural gains; they pass if the audit envelope round-trips
their CIDs.

## Scope (do-not-overstate)

W44 does NOT:

* claim transformer-internal access. The W43 conjectures
  (`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) carry forward unchanged.
* claim that real LLMs decode the factoradic header. The
  factoradic compression's *behavioural* effect is measured on a
  deterministic `SyntheticLLMClient` testbed; on real LLMs the
  saving is a visible-token saving without a guaranteed
  behavioural-decoding gain. Any real-LLM evidence is honest
  realism anchor only.
* claim multi-host live coupling. The mechanism is a
  same-process orchestrator over the released `AgentTeam`
  contract; cross-host live coupling requires substrate beyond
  the capsule layer.

## Strong / partial / failure

* **Strong success**: H1..H10 all pass; the four active live
  channels each register their predicted gain or saving with min
  = max across 5/5 seeds; the released SDK contract is preserved.
* **Partial success**: H1 + H10 pass plus at least two of
  H2..H4 + H5; the rest are honestly downgraded with a per-channel
  retraction.
* **Failure**: H1 or H10 regresses; the live mechanism is rolled
  back behind the trivial registry while the W43 closed-form
  layer remains green.

## Stable boundary preservation (PART K)

The W44 surface ships at `coordpy.live_manifold` and is reachable
only through an explicit import. The first-run UX
(`coordpy-team run --preset quant_desk ...`) is unaffected. The
W43 surface (`coordpy.product_manifold`, `coordpy.r90_benchmark`)
is unchanged. The released `AgentTeam.run` path is byte-for-byte
unchanged — `LiveManifoldTeam` is a new class that sits beside
it. A future milestone may promote a stable subset of the W44
surface under `coordpy.__experimental__` once cross-host live
evidence is acquired.
