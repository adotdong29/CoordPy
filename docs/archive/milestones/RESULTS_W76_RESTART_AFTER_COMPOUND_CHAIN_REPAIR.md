# W76 — Stronger Restart-After-Compound-Chain-Repair / Compound-Chain-Then-Restart Budget-Primary Two-Plane Multi-Agent Substrate Programme

## TL;DR

W76 mints **research axis 73**: the **twenty-first substrate-
attack milestone**, the **twelfth multi-agent task-success-
bearing substrate milestone** (the first to win across **sixteen**
regimes — W75's fifteen plus
``restart_after_compound_chain_repair_under_budget``), the
**first milestone to operationalise chain-then-restart-aware
Plane A↔B handoff promotion**, and the **first milestone to
expose a content-addressed per-turn compound-chain-then-restart
trajectory CID** that unifies all twelve W75 primitives + the
new post-compound-chain-restart window into a single dominant
signal back into the substrate-routed policy.

The load-bearing W76 win:

* **MASC V12 + TCC V11 + tiny_substrate_v21 + 11 supporting Plane
  B V21 modules + 5 Plane A V9 modules + the new chain-then-
  restart-aware handoff coordinator V8 + the new chain-then-
  restart-aware provider filter V8**.
* V21 strictly beats V20 on ≥ 50 % of seeds in every regime
  (100 % in practice across all sixteen regimes; 15 seeds total).
* TSC_V21 strictly beats TSC_V20 on ≥ 50 % of seeds in every
  regime (100 % in practice).
* Chain-then-restart-aware promotion saves ≥ 88 % cross-plane
  visible tokens at the default workload.

## Architecture split

**Plane A — hosted control plane V9 (HTTP-text-only, no
substrate).** Solved now on hosted APIs:

* Provider routing under (budget, restart, rejoin, replacement,
  compound, compound-chain, chain-then-restart) pressures
  (router V9).
* Logprob fusion with chain-then-restart-aware abstain floor +
  7-pressure tiebreak (logprob V9).
* Prefix-cache planning with seven-layer (fine + coarse + ultra
  + mega + giga + peta + exa) rotation (cache planner V9).
* Cost-per-chain-then-restart-success-under-budget with abstain-
  when-chain-then-restart-pressure-violated (cost planner V9).
* Cross-plane handoff with chain-then-restart-aware promotion
  and restart-after-compound-chain-repair fallback (handoff V8).
* Chain-then-restart-aware provider filter (filter V8).
* Explicit wall enumerating ≥ 40 blocked axes at the hosted
  surface (boundary V9).

**Plane B — real substrate plane V21 (in-repo NumPy).** Solved
now on the controlled in-repo substrate:

* Per-turn content-addressed compound-chain-then-restart-
  trajectory CID (substrate V21).
* Per-layer compound-chain-then-restart-length label in [0..12]
  (substrate V21).
* Per-layer chain-then-restart-pressure gate (substrate V21).
* Seventeen-target stacked ridge + 140-dim chain-then-restart
  fingerprint (KV V21).
* Sixteen-objective stacked ridge + per-role 17-dim chain-then-
  restart-pressure head (cache V19).
* Twenty-four-regime ridge + fourteen-way chain-then-restart-
  aware routing head (replay V17).
* Twenty-one-way bidirectional loop (deep substrate hybrid V21).
* Persistent latent V28 (27 layers, 25th skip carrier,
  max_chain_walk_depth=8388608).
* Long-horizon reconstruction V28 (27 heads, max_k=960,
  eighteen-layer scorer).
* MLSC V24 (chain-then-restart-trajectory chain + post-compound-
  chain-restart chain).
* Consensus fallback V22 (38 stages).
* MASC V12 (26 policies across 16 regimes).
* Team-consensus controller V11 (chain-then-restart-pressure +
  post-compound-chain-restart-after-RTR arbiters).

**Still blocked on third-party hosted-model substrate.**
Frontier-model substrate access remains the unsolved research-
line wall; W76 carries the W70 ``frontier_blocked_axes`` set
forward unchanged (``W76-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-
CAP``).

## Multi-agent task line — the primary scoreboard

R-187 sweeps all sixteen MASC V12 regimes at 5 seeds × 3 seed
sets = 15 seeds per regime per family. Every regime returns
``v21_beats_v20_rate ≥ 0.5`` and
``tsc_v21_beats_tsc_v20_rate ≥ 0.5`` (100 % strict-beat in
practice across all sixteen regimes). The new chain-then-
restart regime
(``restart_after_compound_chain_repair_under_budget``) reaches
100 % strict-beat for V21 vs V20.
``team_success_per_visible_token_v21`` is non-trivial across all
regimes.

## Replay / recompute / handoff economics

* **Recompute saving.** Substrate chain-then-restart-repair-
  dominance flops vs full recompute across twelve primitives
  saves ≥ 0.96 at 128 tokens
  (``substrate_chain_then_restart_repair_dominance_flops_v21``).
* **Cache reuse.** Cache-aware planner V9 saves ≥ 88 % input
  tokens on 20 × 8 plans at hit_rate=1.0 with seven-layer
  rotation (≈ 89.4 % at default constants).
* **Cross-plane handoff.** Handoff V8 cross-plane savings ≥ 85 %
  (≈ 88.2 % in practice at default config).
* **Team success per visible token.**
  ``team_success_per_visible_token_v21`` > 0 across every regime;
  matched-budget visible-token usage is min(budget // 21,
  budget // 24) tokens per turn for the substrate-routed and
  team-substrate-coordination policies respectively.

## Falsifiers and limitation reproductions

* H1156 (R-188): Handoff V8 chain-then-restart falsifier honest
  case scores 0; dishonest case scores 1.
* H1157 (R-188): Boundary V9 falsifier honest=0 on the new V21
  axes, dishonest=1.
* H1158 (R-188): KV V21 chain-then-restart-pressure falsifier
  honest case scores 0.
* H1160-H1163 (R-188): four limitation reproductions —
  in-repo NumPy substrate; wall preserved (≥ 40 blocked axes);
  no-version-bump invariant; frontier substrate still blocked.

## Stable boundary preservation

* ``coordpy.__version__ == "0.5.20"`` byte-for-byte.
* ``SDK_VERSION == "coordpy.sdk.v3.43"`` byte-for-byte.
* Smoke driver passes unchanged.
* Trivial-passthrough preservation: ``W76Params.build_trivial()``
  produces an envelope whose ``w75_outer_cid`` carries the
  caller's W75 outer CID exactly.

## Tests

* ``tests/test_w76_tiny_substrate_v21.py`` — 6 tests.
* ``tests/test_w76_masc_v12.py`` — 5 tests.
* ``tests/test_w76_hosted_handoff_v8.py`` — 10 tests.
* ``tests/test_w76_team.py`` — 6 tests.
* ``tests/test_w76_benchmarks.py`` — 12 tests (R-185..R-188 × 3
  seed sets).
* ``tests/test_w76_modules.py`` — 10 tests.
* W75 regression: 33 tests (unchanged).
* Smoke driver: 1 test (unchanged).

Total: **82 tests pass / 0 fail**.

## Verdict

W76 materially advances Context Zero beyond W75 by making
restart-after-compound-chain-repair load-bearing — the team must
survive a fresh restart of the recovering member *immediately
after* completing the W75 compound-chain repair arc, the most
fragile possible follow-on failure modality. V21 strictly beats
V20 on 100 % of seeds across all sixteen regimes (not just the
new one — all carried-forward W75 regimes too), TSC_V21 strictly
beats TSC_V20 on 100 % of seeds, and the new chain-then-restart-
aware handoff V8 saves ≥ 85 % cross-plane visible tokens. The
wall is preserved (≥ 40 blocked axes at the hosted surface;
W70 frontier_blocked_axes unchanged). No version bump. No PyPI
release.
