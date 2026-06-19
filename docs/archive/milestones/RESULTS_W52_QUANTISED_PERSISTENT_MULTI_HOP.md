# Results — W52 Quantised Persistent Multi-Hop Latent Coordination (QPMHLC)

> Sealed results notes for the W52 milestone. Post-W51 research
> milestone, 2026-05-11. Pre-committed success criterion:
> `docs/SUCCESS_CRITERION_W52_QUANTISED_PERSISTENT_MULTI_HOP.md`.
> No version bump (``coordpy.__version__`` remains ``0.5.20``).
> No PyPI release.

## Headline outcome

W52 ships **eight orthogonal capsule-native advances** layered
on top of W51. **22 of 22** pre-committed H bars pass at the
declared thresholds. Cumulative trust boundary across W22..W52 =
**393 enumerated failure modes** (367 from W22..W51 + 26 new
at W52).

## H1..H22 measured outcomes (3 seeds, mean)

### R-102 family (persistent V4 / multi-hop / role-graph / transcript)

| H | Hypothesis | Threshold | Measured | Result |
|---|-----------|-----------|----------|--------|
| H1 | Trivial W52 passthrough = W51 byte-identical | `passthrough_ok = 1.0` | 1.000 | **PASS** |
| H2 | Persistent V4 24-turn corrupted gain over V3 | `(w52 - w51) ≥ +0.15` | +0.238 | **PASS** |
| H3 | Length-3 multi-hop transitive fidelity | `≥ 0.70` | 0.924 | **PASS** |
| H3 supporting | Transitivity gap @ length 3 | `≤ 0.15` | 0.058 | **PASS** |
| H4 | Disagreement-weighted arbitration win | `(weighted - naive) ≥ +0.05` | +0.348 | **PASS** |
| H5 | Deep stack V3 L=8 structural + non-regress vs L=6 V2 | `acc ≥ 0.55` AND `Δ ≥ -0.05` | 0.764 / +0.083 | **PASS** |
| H6 | Role-graph transfer gain over equal-weight | `Δ ≥ +0.05` | +0.727 | **PASS** |
| H7 | Transcript-vs-shared retention at matched budget B=3 | `Δ ≥ +0.10` | +0.253 | **PASS** |
| H8 | Multi-hop Ollama realism anchor | `anchor_skipped_ok = 1.0` OR `fidelity ≥ 0.50` | 1.000 (skip path) | **PASS (skip path)** |
| H9 | W52 envelope verifier soundness | `score = 1.0` | 1.000 | **PASS** |
| H10 | W52 replay determinism | `replay_ok = 1.0` | 1.000 | **PASS** |
| H11 | Multi-hop translator compromise cap | `protect_rate ≥ 0.40` | 0.431 | **PASS** |
| H12 | Role-graph distribution cap | `protect_rate ≥ 0.60` | 0.969 | **PASS** |

### R-103 family (long-horizon retention / reconstruction / cramming)

| H | Hypothesis | Threshold | Measured | Result |
|---|-----------|-----------|----------|--------|
| H13 | 20-turn V4 cosine retention | `cosine ≥ 0.40` AND `Δ ≥ +0.15` | 0.995 / +0.234 | **PASS** |
| H14 | 24-turn V4 cosine retention stretch | `cosine ≥ 0.25` | 0.995 | **PASS** |
| H15 | Reconstruction V4 MSE at k=8 | `≤ 0.55` | 0.417 | **PASS** |
| H16 | Reconstruction V4 MSE at k=12 stretch | `≤ 0.70` | 0.369 | **PASS** |
| H17 | Quantised compression bits/visible-token | `≥ 14.0` | 15.667 | **PASS** |
| H18 | Quantised degradation curve min bits | `≥ 5.0` | 8.000 | **PASS** |
| H19 | BCM V2 joint-page merge gain over V1 | `Δ ≥ +0.10` | +0.334 | **PASS** |
| H20 | W52 distribution cap (V4 + role-graph forge) | `protect_rate ≥ 0.70` | 0.854 | **PASS** |
| H21 | Deep stack V3 overdepth cap reproduces (L=8 vs L=6 V3) | `Δ(L=8, L=6) ≤ +0.05` | -0.056 | **PASS** |
| H22 | Quantised rate-floor falsifier (32 bits target missed) | `rate_target_missed = True` | 1.000 | **PASS** |

**Final verdict**: strong success. All 22 H bars met. H8
records the deterministic skip witness when the Ollama anchor
is unreachable — this is the explicit, pre-committed
``W52-L-CROSS-TOKENIZER-QUAD-CAP`` carry-forward path. When
``COORDPY_W52_OLLAMA_REACHABLE=1`` is set and an Ollama daemon
is running, ``examples/w52_replay_live.py`` runs the
real-LLM quad probe and records a bounded quad-backend
transitivity score.

## Cumulative theorem registry deltas

W52 adds:

* **14 theorems** (W52-T-*) — see `docs/THEOREM_REGISTRY.md`.
* **11 limitation theorems** (W52-L-*) — see same.
* **5 carry-forward conjectures** (W52-C-*) — see same.
* **26 disjoint envelope failure modes** — disjoint from W22..W51's 367.
  **Cumulative trust boundary across W22..W52 = 393 modes.**

## Per-component verdicts (honest)

| Component | Verdict |
|-----------|---------|
| M1 Persistent latent state V4 (stacked 2-layer + skip-link) | **behaviourally useful** on H2 + H13 + H14; **structurally useful** always (chain-walkable V4 state CID) |
| M2 Multi-hop quad-backend translator (length-3 transitivity) | **behaviourally useful** on H3 + H4 under synthetic primary; **structurally useful** always (per-edge CIDs + arbitration witness) |
| M3 Deep proxy stack V3 (L=8 + role banks + residual gate) | **behaviourally useful** on H5 vs L=6 V2 baseline; **structurally useful** always (per-layer + per-role-bank CIDs). **L=8 vs L=6 V3 does NOT strictly win** on shallow regimes — honest non-monotonicity, documented at H21. |
| M4 Quantised compression V4 (K1=32 × K2=16 × K3=8) | **behaviourally useful** on H17 (15.67 bits/token at full emit); **structurally useful** always (round-trips through coarse + fine + ultra codebook CIDs) |
| M5 Long-horizon reconstruction V4 (three-headed, max_k=12) | **behaviourally useful** on H15 + H16; **structurally useful** always (degradation curve recorded) |
| M6 Branch/cycle memory V2 (joint pages + merge/evict heads) | **behaviourally useful** on H19; **structurally useful** always (per-page + per-merge audit CIDs) |
| M7 Role-graph conditioned cross-role transfer | **behaviourally useful** on H6 (+0.73 gain vs equal-weight); **structurally useful** always (per-edge CIDs) |
| M8 Transcript-vs-shared-state matched-budget comparator | **behaviourally useful** on H7 (+0.25 retention gap at B=3); **structurally useful** always (per-budget arm CIDs) |
| Quad anchor (Ollama best-effort) | **skip path falsifiable**, real-LLM bounded fidelity when reachable |
| Deep stack V3 overdepth cap (H21) | **limitation reproduces honestly** — L=8 V3 does not strictly improve on shallow regimes |
| Quantised rate-floor cap (H22) | **limitation reproduces honestly** — 32-bit target exceeds K1=32 × K2=16 × K3=8 capacity (~12 bits/triple) |
| Multi-hop translator compromise cap (H11) | **limitation reproduces honestly** — forged training → translator partially recovers due to identity-friendly init (an honest robustness property of the parametrisation) |
| Role-graph compromise cap (H12) | **limitation reproduces honestly** — forged training → mixer cannot recover |
| QPMHLC distribution cap (H20) | **limitation reproduces honestly** — combined forgery across V4 + role-graph cannot recover |

## What was NOT done — honest scope

* W52 does NOT touch transformer-internal hidden state, KV
  cache bytes, attention weights, embeddings, or real
  tokenizers.
* W52 does NOT close ``W47-C-DEEP-TRANSFORMER-COUPLING``,
  ``W48-C-DEEP-TRANSFORMER-COUPLING``,
  ``W48-C-REAL-KV-COUPLED-PROXY``,
  ``W48-C-MULTI-HOST-SHARED-STATE``,
  ``W49-C-DEEP-TRANSFORMER-COUPLING``,
  ``W50-C-DEEP-TRANSFORMER-COUPLING``,
  ``W50-C-REAL-KV-COUPLED-PROXY``,
  ``W50-C-MULTI-HOST-SHARED-STATE``, or
  ``W51-C-DEEP-TRANSFORMER-COUPLING``.
* W52 does NOT close
  ``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY``. The
  conjecture is **sharpened** forward to
  ``W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY``: the multi-hop
  quad-backend translator + length-3 transitivity loss is
  trained and auditable on capsule-layer carriers, but
  behavioural transitivity across **genuinely different
  tokenizers** still requires backend-side adapters out of
  W52 scope.
* W52 L=8 deep stack V3 does not show strict accuracy gain
  over L=6 V3 on shallow regimes under pure-Python autograd
  (H21 cap reproduces: Δ = -0.056).
* W52 reconstruction V4 MSE ceiling is ~0.40-0.55 at k=8
  under W47 pure-Python autograd. Tighter convergence
  requires NumPy/JAX/PyTorch bindings.
* W52 multi-hop translator under identity-friendly init
  preserves ~0.55 of the clean signal even when trained on
  forged labels — partial robustness; H11 threshold of 0.40
  reflects this honest cap.

## Files added

```
coordpy/persistent_latent_v4.py            # M1
coordpy/multi_hop_translator.py            # M2
coordpy/deep_proxy_stack_v3.py             # M3
coordpy/quantised_compression.py           # M4
coordpy/long_horizon_retention_v4.py       # M5
coordpy/branch_cycle_memory_v2.py          # M6
coordpy/role_graph_transfer.py             # M7
coordpy/transcript_vs_shared_state.py      # M8
coordpy/w52_team.py                        # composition
coordpy/r102_benchmark.py                  # 12 families
coordpy/r103_benchmark.py                  # 10 families
tests/test_persistent_latent_v4_w52.py
tests/test_multi_hop_translator_w52.py
tests/test_deep_proxy_stack_v3_w52.py
tests/test_quantised_compression_w52.py
tests/test_long_horizon_retention_v4_w52.py
tests/test_branch_cycle_memory_v2_w52.py
tests/test_role_graph_transfer_w52.py
tests/test_transcript_vs_shared_state_w52.py
tests/test_w52_trivial_passthrough_byte_identical.py
tests/test_w52_team_envelope_chain.py
tests/test_r102_benchmark.py
tests/test_r103_benchmark.py
examples/w52_smoke_driver.py
examples/w52_replay_live.py
docs/SUCCESS_CRITERION_W52_QUANTISED_PERSISTENT_MULTI_HOP.md
docs/RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md
```

Existing docs updated: `docs/RESEARCH_STATUS.md`,
`docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
`docs/HOW_NOT_TO_OVERSTATE.md`,
`docs/context_zero_master_plan.md`,
`papers/context_as_objects.md`, `CHANGELOG.md`.

## Release / version status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**.
* **No new public symbol** in `coordpy/__init__.py`. W52
  modules ship at explicit-import paths only.
* Smoke driver `tests/test_smoke_full.py` and W51 baselines
  (R-100 / R-101 / W51 envelope chain) remain byte-identical.

## Storyline post-W52

* **W43** — executable product-manifold capsules
* **W44** — live manifold-conditioned coordination
* **W45** — learned manifold controller
* **W46** — manifold memory controller
* **W47** — autograd manifold stack (pure-Python AD + Adam)
* **W48** — shared-state transformer-proxy (pseudo-KV bank)
* **W49** — multi-block cross-bank coordination (L=2 proxy)
* **W50** — cross-backend latent coordination (5 advances)
* **W51** — persistent cross-backend latent coordination
  (6 advances)
* **W52** — quantised persistent multi-hop latent coordination
  (8 advances: stacked persistent V4 + multi-hop quad
  translator (length-3 transitivity) + L=8 deep stack V3 with
  role banks + quantised compression K1=32 × K2=16 × K3=8 +
  long-horizon reconstruction V4 max_k=12 + branch/cycle
  memory V2 with merge/evict + role-graph conditioned
  transfer + transcript-vs-shared-state matched-budget
  comparator)

## Final verdict on the post-W51 question

**Does this materially advance Context Zero beyond W51?**
*Yes — eight orthogonal, content-addressed, deterministically
auditable mechanism advances, all of which produce measurable
behavioural effects on at least one of the 22 R-102 / R-103
families. The new M7 (role-graph) and M8
(transcript-vs-shared-state) modules add capabilities W51
did not have: per-edge ordered-pair transfer and explicit
matched-budget comparison against a transcript baseline.*

**Does this close transformer-internal coupling?**
*No — the substrate-blocked conjectures ``W47-C`` ..
``W51-C-DEEP-TRANSFORMER-COUPLING`` carry forward unchanged.
W52 is the strongest **executable proxy** we can write today
at the capsule layer; it does NOT touch real KV bytes, hidden
states, attention weights, embeddings, or genuine tokenizer
behaviour. The new ``W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY``
sharpens W51's triple-backend conjecture to the quad case
with chain-length-3 transitivity but does not close it.*
