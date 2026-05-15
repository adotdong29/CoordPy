# Results — W62 Trainable Replay-Dominance Hidden-vs-KV Substrate-Coupled Latent OS

> Empirical results for the seventh substrate-attack milestone in
> the Context Zero programme. Pre-committed success criterion:
> `docs/SUCCESS_CRITERION_W62_REPLAY_DOMINANCE.md`. Last updated
> 2026-05-15.

## TL;DR

W62 ships nineteen mechanism advances over W61. R-131..R-133 at
3 seeds (135 cells) returns **45 / 45 H-bars pass at 3 / 3 seeds
(135 / 135 cells)** — strong success per the pre-committed bar.

## R-131 — real-substrate / latent-bridge / replay-dominance / hidden-vs-KV (13 cell families)

* H163  replay V3 dominates transcript fallback on synthetic
        corruption regime — **PASS** at 3/3 seeds.
* H163b replay V3 chooses REUSE strictly more often than V2 on
        CRC-passed-low-drift regime (V3 ≥ V2) — **PASS** at 3/3.
* H163c per-regime ridge head fits converge on all 4 regimes
        (post ≤ pre + 1e-9 per regime) — **PASS** at 3/3.
* H164  cache controller V5 two-objective stacked ridge converges
        on both drop-oracle and retrieval-relevance objectives
        (per-objective post ≤ pre + 1e-9) — **PASS** at 3/3.
* H164b cache controller V5 trained-repair head reduces residual
        (post ≤ pre + 1e-9) — **PASS** at 3/3.
* H164c composite_v5 6-head mixture ridge converges — **PASS** at 3/3.
* H165  hidden-vs-KV regime classifier reaches ≥ 0.8 training
        accuracy on synthetic supervision — **PASS** at 3/3.
* H165b HSB V6 three-target stacked ridge fit converges — **PASS** at 3/3.
* H165c HSB V6 writes propagate into V7 substrate cache-write
        ledger with positive L2 — **PASS** at 3/3.
* H166  prefix V6 drift-curve predictor fits all K=3 steps with
        post ≤ pre + 1e-9 per step — **PASS** at 3/3.
* H166b prefix V6 multi-segment partial reuse saves ≥ 25% flops
        vs full recompute — **PASS** at 3/3 (~46% observed).
* H167  attention V6 two-stage clamp negative-budget falsifier
        returns 0 KL exactly — **PASS** at 3/3.
* H167b attention V6 signed falsifier with per-coarse-bucket
        signs produces non-zero correlation — **PASS** at 3/3.

## R-132 — long-horizon retention / reconstruction / aggressive-compression (12 cell families)

* H168  persistent V14 chain walk depth ≥ 2048 — **PASS** at 3/3.
* H168b persistent V14 decuple skip carries replay-dominance EMA —
        **PASS** at 3/3.
* H168c persistent V14 distractor rank ≥ 8 — **PASS** at 3/3.
* H169  LHR V14 13-way reconstruction head runs without crashing —
        **PASS** at 3/3.
* H169b LHR V14 replay-dominance head produces non-trivial output —
        **PASS** at 3/3.
* H169c LHR V14 four-layer scorer converges — **PASS** at 3/3.
* H170  ECC V14 ≥ 25.0 bits/visible-token — **PASS** at 3/3
        (25.333 observed).
* H170b ECC V14 total codes = 2^23 = 8 388 608 — **PASS** at 3/3.
* H170c ECC V14 4096-bit/token rate-floor falsifier reproduces
        ceiling — **PASS** at 3/3.
* H171  multi-hop V12 chain-length 17 over 20 backends and 380
        directed edges — **PASS** at 3/3.
* H171b multi-hop V12 seven-axis composite used — **PASS** at 3/3.
* H171c multi-hop V12 compromise threshold in [1, 7] — **PASS** at 3/3.

## R-133 — corruption / disagreement / consensus / fallback / abstention (20 cell families)

* H172  CRC V10 KV1024 single-byte detect rate ≥ 0.95 — **PASS**
        at 3/3 (1.0 observed).
* H172b CRC V10 17-bit adversarial burst detect rate ≥ 0.95 —
        **PASS** at 3/3.
* H172c CRC V10 post-repair top-K Jaccard floor ≥ 0.5 — **PASS**
        at 3/3 (1.0 observed).
* H173  consensus V8 12-stage chain enumerated with trained_repair
        — **PASS** at 3/3.
* H173b consensus V8 trained_repair stage fires when corruption
        detected AND repair amount above threshold — **PASS** at 3/3.
* H174  uncertainty V10 9-axis weighted composite returns value
        in [0, 1] — **PASS** at 3/3.
* H174b uncertainty V10 replay_dominance_aware flips True when
        fidelity < 1.0 — **PASS** at 3/3.
* H175  disagreement V8 Wasserstein-1 equivalence identity holds —
        **PASS** at 3/3.
* H175b disagreement V8 Wasserstein-1 falsifier triggers — **PASS**
        at 3/3.
* H176  TVS V11 pick rates sum to 1.0 (within 1e-6) and 12 arms
        enumerated — **PASS** at 3/3.
* H176b TVS V11 replay-dominance arm fires when fidelity is strict
        highest — **PASS** at 3/3.
* H176c TVS V11 reduces to V10 when replay-dominance-fidelity = 0
        — **PASS** at 3/3.
* H177  MLSC V10 replay-dominance witness chain inherits as union —
        **PASS** at 3/3.
* H177b MLSC V10 disagreement Wasserstein-1 distance computed at
        merge — **PASS** at 3/3.
* H178  substrate V7 per-(layer, head, slot) cache-write ledger
        has shape (L, H, T) — **PASS** at 3/3.
* H178b substrate V7 logit-lens probe returns shape (L, V) —
        **PASS** at 3/3.
* H178c substrate V7 attention-receive delta recorded across two
        forwards — **PASS** at 3/3.
* H178d substrate V7 replay-trust ledger updated after replay
        decision — **PASS** at 3/3.
* H179  deep hybrid V7 seven-way fires when all seven axes active
        — **PASS** at 3/3.
* H180  substrate V7 adapter tier `substrate_v7_full` satisfied
        only by the V7 runtime — **PASS** at 3/3.

## Strong success verdict

Per the pre-committed success criterion:
* ≥ 95% H-bars pass at 3 seeds: **100%** (45/45) — **strong
  success**.
* Every per-mechanism advance is exercised end-to-end: **yes**
  (M1..M19 all reached in `coordpy.w62_team`).
* ≥ 65 disjoint failure modes enumerated by the W62 envelope
  verifier: **yes** (68).
* W62 envelope chain preserves the W61 outer CID byte-for-byte:
  **yes** (verified by `test_w62_team_envelope_chain` and
  `test_w62_trivial_passthrough_byte_identical`).

## Honest scope (what W62 does NOT prove)

* W62 does NOT prove third-party transformer-internal access on
  hosted backends. ``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
  carries forward unchanged.
* W62 does NOT use end-to-end backprop or autograd; only twelve
  closed-form linear ridge solves.
  ``W62-L-V7-NO-AUTOGRAD-CAP`` documents.
* W62's hidden-vs-KV classifier reaches ≥ 0.8 accuracy on
  **synthetic** supervision; it does NOT itself prove hidden-
  state injection beats KV injection on real models or real
  workloads.
* W62's trained corruption-repair head outputs an **additive
  correction**; it does NOT un-corrupt the raw cached state.
  ``W62-L-CONSENSUS-V8-REPAIR-STAGE-SYNTHETIC-CAP`` documents.
* W62's V7 substrate is a 9-layer NumPy-on-CPU transformer; NOT
  a frontier model.
  ``W62-L-NUMPY-CPU-V7-SUBSTRATE-CAP`` documents.
* W62's persistent V14 outer wrapper adds one EMA carrier; it
  does NOT train V13's outer GRU end-to-end.
  ``W62-L-V14-OUTER-NOT-TRAINED-CAP`` documents.

## Pointers

* Mechanism modules: ``coordpy.tiny_substrate_v7`` …
  ``coordpy.disagreement_algebra_v8``.
* Team orchestrator: ``coordpy.w62_team``.
* Benchmarks: ``coordpy.r131_benchmark``,
  ``coordpy.r132_benchmark``, ``coordpy.r133_benchmark``.
* Tests: ``tests/test_w62_modules.py``,
  ``tests/test_r131_r132_r133_w62.py``,
  ``tests/test_w62_team_envelope_chain.py``,
  ``tests/test_w62_trivial_passthrough_byte_identical.py``.
* Pre-committed success criterion:
  ``docs/SUCCESS_CRITERION_W62_REPLAY_DOMINANCE.md``.
* Theorem registry (W62-T-* and W62-L-* rows):
  ``docs/THEOREM_REGISTRY.md``.
