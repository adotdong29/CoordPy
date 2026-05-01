# Success Criterion — W28 (Ensemble-Verified Cross-Model Multi-Chain Pivot Ratification)

**SDK target**: wevra.sdk.v3.29
**Pre-commit date**: 2026-04-30
**Status**: pre-committed (the bar is fixed BEFORE the headline numbers are
collected — falsifiable in either direction).

This document pre-commits the pass/fail bar for the W28 milestone. The
milestone *advances the multi-agent-context thesis beyond W27* iff every
gate below is met. Each gate is mechanically checkable: it cites the
exact regime, the exact metric, the exact threshold, and the exact
expected branch counts.

---

## 1. The position W28 has to clear

W27 (SDK v3.28) established the strongest existing capsule-native
multi-agent-coordination result the programme has shipped:

* On `R-74-XORACLE-RECOVER` (1 producer + K=3 consumers, 16 cells, 2
  salience signatures, signature_period=4) W27 simultaneously
  - reduced mean visible tokens by **−76.27%** vs W26 (29.50 → 7.00)
    on a regime where W26 architecturally fails, AND
  - lifted `correctness_ratified_rate` from **0.500 → 1.000** by
    routing each cell to the per-signature oracle scope.
* Trust boundary sound across 12 enumerated failure modes
  (`verify_salience_signature` 4 modes + `verify_chain_pivot` 8 modes).
* Backward-compat: 508/508 focused regression (W18..W27) passed;
  `enabled=False` reduces W27 to W26 byte-for-byte.

W27's explicit remaining gaps (named in
`docs/RESULTS_WEVRA_W27_MULTI_CHAIN_PIVOT.md` and reaffirmed by the
master plan's post-W27 next-steps section):

* G1. **Live cross-model robustness of the W27 line** — the W27
  benchmarks were run against the synthetic `ServiceGraphOracle` /
  `ChangeHistoryOracle` ecology; the chain-pivot machinery has not been
  stressed against intermittent oracle drift driven by a real LLM.
* G2. **Cross-host / two-Mac validation absent** — Mac 2 (192.168.12.248)
  has been ARP-incomplete for 22 consecutive milestones. Even where two
  reachable hosts exist with different model families, no W-letter
  result has yet *exercised* both hosts inside a single benchmark cell.
* G3. **Old explicit capsule line and the new dense-control line have
  not yet been synthesised**. W21's trust-weighted multi-oracle
  adjudicator (old line) and W27's salience-keyed multi-chain pool (new
  line) sit side-by-side in the codebase but no one mechanism currently
  composes them inside one cell.
* G4. **Release packaging and stable-vs-experimental boundary still
  loose**. The `__init__.py` re-exports W22..W27 dense-control symbols
  without an explicit experimental marker, so external callers cannot
  tell which surface the SDK contract covers.
* G5. **No named ensemble-verification failure modes**. Every existing
  W{N} verifier (W22..W27) checks integrity *of one envelope*; no
  verifier currently checks the *integrity of an ensemble decision*
  (probe forgery, weight forgery, quorum forgery).

W28 must address G1, G2, G3, G5 with a *real* mechanism change AND make
direct progress on G4 in the same milestone, without regressing on the
prior W27 wins.

---

## 2. Pre-committed bars

W28 is **discharged** iff *every* gate below holds. (If only some hold,
W28 is partially discharged — the results note must say which gates
failed and why.) The bars are split into a hard set (mandatory) and a
soft set (must report honestly, but partial pass is acceptable).

### 2.1 HARD gates

* **H1 — Real mechanism beyond W27**
  W28 must add a new content-addressed envelope kind
  (`EnsemblePivotRatificationEnvelope`) and a new pure verifier
  (`verify_ensemble_pivot_ratification`) with at least 6 enumerated
  failure modes that did not exist in any W22..W27 verifier. The new
  mechanism must compose the W21 trust-weighted oracle quorum with the
  W27 salience-signature pool in a single decision (one or more probes,
  trust-weighted, ratifying or rejecting one W27 pivot/anchor).
  *Mechanically checked* by `test_phase75` enumerating every failure
  mode of the new verifier.

* **H2 — No regression on the strongest W27 regime**
  On `R-75-CHAIN-SHARED` (the W28 analogue of R-74-CHAIN-SHARED) the
  ensemble layer with `enabled=True, K_probes=1, trust_weight=quorum`
  must reduce to W27 byte-for-byte (zero extra tokens, zero
  ratification rejections, identical W27_BRANCH_* counts).
  Threshold: `mean_total_w28_visible_tokens == mean_total_w27_visible_tokens`
  AND `correctness_ratified_rate >= W27 baseline` across 5/5 seeds.

* **H3 — Trust boundary sound**
  `verify_ensemble_pivot_ratification` must reject every enumerated
  tampering mode (≥6 modes). On `R-75-RATIFICATION-TAMPERED` the
  controller must reject ≥ N − 2 tampered ratifications across 16
  cells for 5/5 seeds; correctness must be preserved on 5/5 seeds (no
  silent acceptance of a tampered ratification leading to an incorrect
  answer).

* **H4 — Honest scope of new mechanism stated**
  The W28 module docstring must explicitly state:
  - W28 is a *trust-amplification* layer, not a new information
    channel: it does not add new content; it only ratifies or rejects
    W27's existing pivot/anchor decisions.
  - W28's correctness gain on cross-model drift is bounded above by the
    fraction of cells where at least one probe disagrees with the
    primary signature; W28 cannot exceed the per-cell correctness
    achievable by a perfect trust-weighted quorum.
  - When K_probes = 1, W28 is W27 byte-for-byte (W28-Λ-single-probe).
  - When all probes drift identically (W28-Λ-coordinated-drift), W28
    cannot detect the drift; this is the structural limit.

* **H5 — At least one named falsifier where W28 does not help OR is
  unsafe**
  The bench family must include at least one regime where W28 is
  named-no-help (W28-Λ-single-probe, W28-Λ-coordinated-drift) and at
  least one regime where the ensemble ratification mechanism is
  named-unsafe-without-verification (W28-Λ-spoofed-probe rejected only
  if the verifier is engaged). Both falsifiers must be empirically
  confirmed in the headline run.

* **H6 — Old-line strengthening clause: trust-weighting composes with
  multi-chain**
  The W28 mechanism must thread the W21 `OracleRegistration` /
  `trust_prior` interface into the ensemble's probe table — i.e. an
  ensemble probe is itself an `OracleRegistration` with a trust prior,
  and the ratification quorum is *trust-weighted* not vote-counted.
  This must be visible in the public API: the W28 disambiguator/
  orchestrator must accept a `tuple[OracleRegistration, ...]` for its
  probe table.

* **H7 — Release-readiness clause**
  The `__init__.py` must add an explicit
  `__experimental__` tuple listing all dense-control symbols
  (W22..W28). The CHANGELOG.md must gain a v0.5.2 entry naming W28 and
  the new experimental marker. The pyproject.toml version must bump to
  0.5.2.

* **H8 — Focused regression green**
  Tests `test_phase70..test_phase74 + test_phase75 + test_wevra_capsules
  + test_wevra_team_coord + test_wevra_public_api` must all pass
  (≥ 508 from before W28 + the new W28 tests). No test must be skipped
  or marked xfail without explicit justification in the results note.

### 2.2 SOFT gates (must report honestly)

* **S1 — Cross-model live evidence**
  At least one R-75 sub-bank (`R-75-CROSS-MODEL-DRIFT-LIVE`) must run
  with at least two real, reachable LLM probes from *different model
  families* (e.g. Qwen + Gemma2/Mixtral). If only one host is reachable
  the bench must still run (with one local probe + one synthetic
  control) and must say so. If both hosts are reachable the bench must
  use both. Mac 2 (192.168.12.248) being unreachable does NOT block this
  gate as long as the bench reports the host topology actually used.

* **S2 — Cross-host evidence (best effort)**
  The bench must attempt cross-host probing using both reachable hosts
  (`localhost` + `192.168.12.191`). If both are reachable the bench
  must record `n_cross_host_probe_calls > 0` and report
  `cross_host_round_trip_bytes`. If only one is reachable, the bench
  must record this fact and downgrade to single-host gracefully. This
  is the first W-letter milestone in 22 milestones to actually exercise
  two reachable hosts with different model families inside one bench.

* **S3 — Variance reduction headline**
  On `R-75-CROSS-MODEL-DRIFT` (synthetic drift, no live LLM required),
  W28 with K_probes ≥ 2 should reduce per-cell correctness *variance*
  vs W27 by ≥ 0.10 across 5/5 seeds where the drift regime introduces
  intermittent oracle disagreement. If variance reduction does not
  materialise, the results note must state plainly that S3 was not met
  and explain why (e.g. coordinated drift falsifier triggered).

* **S4 — Token-overhead bound**
  W28's per-cell visible-token overhead vs W27 (the cost of carrying
  the ratification envelope) must be ≤ 2 tokens on the producer side
  in the canonical encoding (one `<ratify_ref:DDDD>` token + one
  trust-weighted vote summary token). If higher, the results note must
  explain why and bound the regression.

### 2.3 Strong vs partial vs failure

* **Strong success** — every H1..H8 gate met AND every S1..S4 gate met.
  W28 is fully discharged; the master plan's "post-W27 next steps"
  section is replaced with a "post-W28 next steps" section.
* **Partial success** — every H1..H8 gate met; one or more S1..S4 gates
  honestly reported as not met. W28 is partially discharged; the
  unmet gates become named open conjectures (e.g.
  W28-C-CROSS-HOST-VARIANCE).
* **Failure** — any H1..H8 gate fails. W28 must be retracted from the
  results note (or downgraded to a conditional pre-print) and the
  master plan's "open" section must be updated accordingly.

---

## 3. Why this bar is honest

It is genuinely possible for W28 to fail any of H1..H8 with the
mechanism we are about to implement:

* H2 can fail if our K=1 reduction is not byte-perfect (a common bug
  pattern is forgetting to skip the ratification envelope when K=1 +
  weight ≥ quorum).
* H3 can fail if the verifier accepts a tampered probe vote.
* H5 can fail if both named falsifiers are not empirically reproduced.
* H6 can fail if the API does not actually thread `OracleRegistration`
  (a common shortcut is to just use raw probe IDs).
* H8 can fail if any prior regression breaks under the new W28 layer.

S1..S2 can fail if the live LLM probes diverge in ways the synthetic
ratification cannot ratify, or if the second host is genuinely useless
for the probe quorum (e.g. the local Qwen probe and remote Qwen probe
agree byte-for-byte).

S3 can fail if the only drift our bench can simulate is fully
coordinated (W28-Λ-coordinated-drift); in that case S3 is a real null
result and W28 is partial.

S4 can fail if our envelope encoding adds more than 2 tokens.

The bar is therefore both stretching and falsifiable.

---

## 4. Named falsifiers

These named falsifiers must be *implemented* in the bench and
*observed* in the headline run:

* **W28-Λ-single-probe** — at K_probes=1 with weight ≥ quorum, W28
  reduces to W27 byte-for-byte. (Bench: `R-75-SINGLE-PROBE`.)
* **W28-Λ-coordinated-drift** — when every probe drifts identically,
  W28 cannot detect the drift; correctness is bounded above by W27's
  correctness on the same regime. (Bench: `R-75-COORDINATED-DRIFT`.)
* **W28-Λ-trust-zero** — when all probe trust priors sum to zero, the
  ratification quorum is unreachable and W28 abstains (no false
  ratification). (Bench: `R-75-TRUST-ZERO`.)
* **W28-Λ-spoofed-probe** — an unregistered probe ID in a ratification
  envelope is rejected with `probe_id_unregistered`. (Bench:
  `R-75-RATIFICATION-TAMPERED`.)
* **W28-Λ-quorum-tampered** — the ratified flag does not match the
  recomputed quorum; rejected with `quorum_recompute_mismatch`.
  (Bench: `R-75-RATIFICATION-TAMPERED`.)
* **W28-Λ-pool-exhausted-passthrough** — when the W27 pool is
  exhausted, W28 must fall through to W27's pool-exhausted path
  without inventing a fresh ratification (no spurious savings from a
  layer that should be inactive). (Bench: `R-75-POOL-EXHAUSTED`.)

---

## 5. Theorem statements W28 commits to

W28 introduces:

* **W28-1 (proved + mechanically-checked)** — Trust-boundary soundness:
  `verify_ensemble_pivot_ratification` rejects every enumerated
  tampering mode. Status: proved by enumeration in
  `test_phase75_*EnsembleVerifier*`.
* **W28-2 (proved + empirical)** — Backward compatibility: at K_probes=1
  with weight ≥ quorum, W28's per-cell visible-token cost equals W27's
  per-cell visible-token cost byte-for-byte.
* **W28-3 (proved-conditional + empirical, headline)** — On a divergent
  regime where some probes disagree with the primary salience
  signature, the trust-weighted ensemble layer raises the ratified
  correctness floor from W27's single-stack correctness to a
  trust-weighted-quorum correctness floor; the empirical magnitude
  on `R-75-CROSS-MODEL-DRIFT` is the headline number.
* **W28-Λ-single-probe (proved-empirical)** — K_probes=1 ⇒ W28 = W27
  byte-for-byte.
* **W28-Λ-coordinated-drift (proved-conditional)** — Coordinated drift
  ⇒ W28 cannot improve over W27.
* **W28-C-CROSS-HOST (conjectural)** — Cross-host probes (different
  model families) reduce per-cell ratification variance vs single-host
  probes by at least ε on `R-75-CROSS-MODEL-DRIFT-LIVE`. Status set in
  the W28 results note, depending on whether two hosts are reachable
  and whether the empirical evidence supports ε > 0.

---

## 6. Honest scope (what W28 does NOT claim)

* W28 does NOT claim "we solved context." The original success criterion
  in `SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` is unchanged.
* W28 does NOT claim transformer-internal latent control. It is an
  *audited proxy* — every "latent" reference is a content-addressed
  envelope on a typed bus, not a hidden activation.
* W28 does NOT solve the W22-C-CACHE-AMPLIFICATION conjecture. That
  remains open.
* W28 does NOT bring up Mac 2. Mac 2 (192.168.12.248) remains
  ARP-incomplete; W28 documents the use of the *other* reachable host
  (192.168.12.191) plus localhost as the two-host topology.
* W28 does NOT promise live evidence beyond what its probes can
  honestly observe; `R-75-CROSS-MODEL-DRIFT-LIVE` is conditional on
  having ≥ 2 reachable models from different families.

---

End of pre-commit. Headline numbers will be appended to
`RESULTS_WEVRA_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md`; gate-by-gate
verdicts will be appended to this file as a "Verdict" section.
