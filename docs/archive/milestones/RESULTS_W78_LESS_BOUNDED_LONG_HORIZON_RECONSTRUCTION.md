# W78 — Stronger Less-Bounded Long-Horizon Reconstruction / Bounded-Window-Falsifier Budget-Primary Two-Plane Multi-Agent Substrate Programme

Post-mortem for the W78 milestone landed on 2026-05-17.

## What W78 delivered

W78 is the twenty-third post-release substrate-attack milestone
and the fourteenth multi-agent task-success-bearing substrate
milestone in the Context Zero programme. It introduces the V23
substrate (one new long-horizon-reconstruction axis on top of
V22), the 30-policy MASC V14 across **eighteen** regimes, TCC
V13, hosted control plane V11, the new long-horizon-
reconstruction-aware Plane A↔B handoff coordinator V10, the new
long-horizon-reconstruction-aware provider filter V10, and —
critically — the two new W78-anchor modules:

* ``coordpy.long_horizon_reconstruction_substrate_v1`` — the
  load-bearing W78 win. Reads from a persistent latent V30
  carrier and reconstructs the W77 compound-chain-then-restart-
  then-replacement trajectory CID *without looking at the
  visible transcript at all*.
* ``coordpy.bounded_window_baseline_v1`` — the load-bearing
  anti-goal falsifier. Implements explicit fixed-k transcript
  baselines at k ∈ {4, 8, 16, 32} + rolling-summary, and proves
  that any predictor restricted to the last k visible turns has
  zero information about events outside that window. The W78
  substrate is required to beat them all on the new regime.

This is the W78 philosophical change. Previous milestones
compounded contradiction/replacement/restart/chain mechanisms
but the visible state per agent remained whatever the local
turn observed. W78 moves the programme toward **genuinely less-
bounded** state retention by making the bounded-window
transcript baseline explicit, benchmarking against it, and
demonstrating that the W78 substrate succeeds where any fixed-k
baseline structurally cannot.

## Quantitative wins

* MASC V14: V23 strictly beats V22 on **100 %** of seeds in
  every one of the 18 regimes (verified at 3 seeds × 18 regimes
  = 54 cells; required: ≥ 50 % per regime).
* TSC_V23 strictly beats TSC_V22 on **100 %** of seeds in every
  regime.
* Bounded-window-baseline failure bar: all four fixed-k
  baselines (k ∈ {4, 8, 16, 32}) achieve **zero** reconstructions
  on queries whose source events lie outside their window. The
  W78 long-horizon-reconstruction substrate achieves **100 %**
  reconstruction success on the same queries (deterministic,
  content-addressed lookup from the persistent carrier).
* Cache planner V11 nine-layer rotation saves ≥ 89 % on 20 × 8
  at hit_rate=1.0.
* Handoff V10 cross-plane savings ≥ 88 % over a 100-turn
  schedule (target: ≥ 87 %).
* Boundary V11 enumerates 46 blocked axes at the hosted surface
  (W77's 43 + 3 new V23 axes).
* Substrate LHR dominance flops saves ≥ 0.93 vs full recompute
  over fourteen primitives.
* Substrate LHR pressure throttle saves > 0 visible tokens
  under tight budget.
* Long-horizon-reconstruction substrate vs full replay
  economics: ≥ 0.99 saving on a 240-turn horizon (O(log n)
  Merkle walks vs O(horizon) per-token forward).

## Benchmark families

Four R-1XX benchmark families, **64 H-bars × 3 seed sets**, all
pass:

* ``coordpy.r193_benchmark`` (Plane A V11; 10 H-bars).
* ``coordpy.r194_benchmark`` (Plane B V23; 18 H-bars).
* ``coordpy.r195_benchmark`` (multi-agent task success across 18
  regimes + bounded-window-baseline failure bar + W78
  reconstruction success bar; 20 H-bars).
* ``coordpy.r196_benchmark`` (handoff V10 + bounded-window
  falsifier + long-horizon limitation reproductions; 16 H-bars).

## Tests

Five focused W78 test files; 33 tests; all pass:

* ``tests/test_w78_modules.py`` (11 tests) — focused unit tests
  for each new module's witness CID + cid() stability.
* ``tests/test_w78_bounded_window_baseline.py`` (5 tests) —
  explicit bounded-window-failure-by-construction proof and
  W78-beats-bounded-window-baseline bar.
* ``tests/test_w78_long_horizon_reconstruction.py`` (5 tests) —
  W78 reconstruction module deterministically reconstructs the
  W77 compound-chain trajectory from the persistent latent
  carrier alone.
* ``tests/test_w78_benchmarks.py`` (6 tests) — R-193..R-196
  all_pass under 3 seed sets, plus two-consecutive-runs
  stability for R-193 and R-194.
* ``tests/test_w78_team.py`` (6 tests) — W78Team end-to-end
  smoke + W77 envelope chaining preserved byte-for-byte under
  trivial passthrough + stable-boundary preservation.

W77 regression (25 W77 tests) still passes. Smoke driver
(``tests/test_smoke_full.py``) still passes.

## Theorem registry adds

* ``W78-T-LONG-HORIZON-RECONSTRUCTION-CID-DETERMINISM`` —
  mechanically-checked.
* ``W78-T-MASC-V14-V23-BEATS-V22-ALL-18`` — empirical.
* ``W78-T-MASC-V14-TSC23-BEATS-TSC22-ALL-18`` — empirical.
* ``W78-T-CACHE-V21-18OBJ-CONVERGES`` — empirical.
* ``W78-T-KV-V23-19TARGET-CONVERGES`` — empirical.
* ``W78-T-HANDOFF-V10-WALL-PRESERVED`` — mechanically-checked.
* ``W78-T-HANDOFF-V10-CROSS-PLANE-SAVINGS-GE-87`` — empirical.
* ``W78-T-HOSTED-CACHE-V11-NINE-LAYER-SAVINGS-GE-89`` —
  empirical.
* ``W78-T-BOUNDED-WINDOW-INSUFFICIENT`` — **proved-conditional**
  (proof by construction: a predictor restricted to the last k
  visible turns has zero information about events occurring
  more than k turns ago; therefore cannot strictly beat random
  on ≥ 1-bit reconstruction queries).
* ``W78-T-LHR-SUBSTRATE-BEATS-BW-BASELINE`` —
  mechanically-checked.

## Limitations carried forward

* ``W78-L-NUMPY-CPU-V23-SUBSTRATE-CAP`` — W78 substrate is
  in-repo NumPy.
* ``W78-L-HOSTED-V11-NO-SUBSTRATE-CAP`` — hosted plane does NOT
  pierce the wall; 46 blocked axes.
* ``W78-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP`` — frontier-
  model substrate access still blocked; W70 frontier_blocked_axes
  unchanged.
* ``W78-L-V23-NO-AUTOGRAD-CAP`` — no SGD / autograd / GPU;
  closed-form linear ridges only.
* ``W78-L-BOUNDED-WINDOW-INSUFFICIENT-CAP`` — bounded-window
  baselines fail by construction on reconstruction queries
  outside the window. The proof is structural within the
  synthetic regime; we do NOT claim frontier LLMs with long
  context windows are similarly bounded — they have soft,
  learned attention, not a hard fixed-k window.

## Stable contract preservation

* ``coordpy.__version__ == "0.5.20"`` — byte-for-byte unchanged.
* ``coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`` — byte-for-byte
  unchanged.
* Smoke driver passes.
* W77 regression passes (25 tests).
* W76 regression passes.
* All W78 modules ship at explicit-import paths; not re-exported
  from ``coordpy.__init__``.
* No PyPI release performed.

## Honest scope ("what we did NOT do")

* **No third-party hosted-model substrate coupling.** The wall
  V11 enumerates 46 blocked axes; we do not read hidden states,
  KV-cache bytes, or attention weights from any hosted API.
* **No autograd / SGD / GPU.** All "training" is closed-form
  linear ridge.
* **The bounded-window-insufficiency claim is structural.** We
  prove it within the synthetic regime; we do NOT claim that
  frontier LLMs with long soft-attention context windows are
  similarly bounded.
* **MASC V14 is synthetic.** The 100 % V23-strict-beat-V22 rate
  across 18 regimes is measured *inside* the deterministic
  MASC V14 harness. It is NOT a real-model multi-agent task
  win.
* **The reconstruction substrate is deterministic lookup, not
  learned reconstruction.** The carrier is a content-addressed
  list of events; reconstruction is a Merkle-style lookup, not
  generation. This is the structural claim — bounded-window
  baselines provably fail; deterministic-lookup substrates
  provably succeed.

## What this milestone advances about the original goal

The original goal is **actually solving context for multi-agent
teams**. W78 advances this on three coupled fronts:

1. **Less-bounded state and long-horizon survival.** The V30
   persistent latent (29 layers, max_chain_walk_depth=33554432)
   plus the LHR V30 retention head (max_k=2048; the largest k
   in the programme to date) plus the long-horizon-reconstruction
   substrate V1 (deterministic Merkle-style lookup from the
   persistent carrier) demonstrate state survival across
   blackouts that any fixed-k transcript baseline structurally
   cannot bridge.
2. **Compound failure robustness.** All 17 W77 failure-mode
   regimes carry forward unchanged; W78 adds the
   ``long_delay_reconstruction_after_compound_chain_failure``
   regime as the eighteenth load-bearing regime.
3. **Replay / recompute / handoff economics.** The handoff V10
   saves ≥ 88 % visible tokens vs all-hosted; the LHR substrate
   saves ≥ 99 % FLOPs vs full transcript replay; the cache
   planner V11 saves ≥ 89 % on 20×8 at hit_rate=1.0.

The bounded-window-baseline anti-goal is the W78 philosophical
contribution: it makes load-bearing what the programme has
repeatedly warned against — quietly regressing to fixed-k
visible-window thinking.
