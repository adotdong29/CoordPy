# W75 — Stronger Compound-Chain-Repair / Replacement-Then-Delayed-Repair-Then-Rejoin Budget-Primary Two-Plane Multi-Agent Substrate Programme

## TL;DR

W75 mints **research axis 72**: the **twentieth substrate-attack
milestone**, the **eleventh multi-agent task-success-bearing
substrate milestone** (the first to win across **fifteen** regimes
— W74's fourteen plus
``compound_repair_after_replacement_then_rejoin_under_budget``),
the **first milestone to operationalise compound-chain-aware Plane
A↔B handoff promotion** plus the **first milestone to expose a
content-addressed per-turn compound-chain-repair-trajectory CID**
that unifies all eleven W74 primitives + the new replacement-then-
delayed-repair-then-rejoin chain into a single dominant signal
back into the substrate-routed policy.

The load-bearing W75 win:

* **MASC V11 + TCC V10 + tiny_substrate_v20 + 11 supporting Plane
  B V20 modules + 5 Plane A V8 modules + the new compound-chain-
  aware handoff coordinator V7 + the new compound-chain-aware
  provider filter V7**.
* V20 strictly beats V19 on ≥ 50 % of seeds in every regime (≥
  80 % in practice across all regimes; 100 % on the new
  compound-chain regime; ≥ 5 × 4 = 20 seeds total).
* TSC_V20 strictly beats TSC_V19 on ≥ 50 % of seeds in every regime
  (≥ 80 % in practice).
* Compound-chain-pressure-aware promotion saves ≥ 87 % cross-plane
  visible tokens at the default workload.

## Architecture split

**Plane A — hosted control plane V8 (HTTP-text-only, no
substrate).** Solved now on hosted APIs:

* Provider routing under (budget, restart, rejoin, replacement,
  compound, compound-chain) pressures (router V8).
* Logprob fusion with compound-chain-aware abstain floor + per-
  budget+restart+rejoin+replacement+compound+chain tiebreak
  (logprob V8).
* Prefix-cache planning with six-layer (fine + coarse + ultra +
  mega + giga + peta) rotation (cache planner V8).
* Cost-per-compound-chain-success-under-budget with abstain-when-
  compound-chain-pressure-violated (cost planner V8).
* Cross-plane handoff with compound-chain-aware promotion and
  compound-repair-after-replacement-then-rejoin fallback (handoff
  V7).
* Compound-chain-aware provider filter (filter V7).
* Explicit wall enumerating ≥ 37 blocked axes at the hosted
  surface (boundary V8).

**Plane B — real substrate plane V20 (in-repo NumPy).** Solved
now on the controlled in-repo substrate:

* Per-turn content-addressed compound-chain-repair-trajectory CID
  (substrate V20).
* Per-layer compound-chain-length label in [0..11] (substrate
  V20).
* Per-layer compound-chain-pressure gate (substrate V20).
* Sixteen-target stacked ridge + 130-dim compound-chain-repair
  fingerprint (KV V20).
* Fifteen-objective stacked ridge + per-role 16-dim compound-
  chain-pressure head (cache V18).
* Twenty-three-regime ridge + thirteen-way compound-chain-aware
  routing head (replay V16).
* Twenty-way bidirectional loop (deep substrate hybrid V20).
* Persistent latent V27 (26 layers, 24th skip carrier,
  max_chain_walk_depth=4194304).
* Long-horizon reconstruction V27 (26 heads, max_k=896,
  seventeen-layer scorer).
* MLSC V23 (compound-chain-repair-trajectory chain + replacement-
  then-rejoin chain).
* Consensus fallback V21 (36 stages).
* MASC V11 (24 policies across 15 regimes).
* Team-consensus controller V10 (compound-chain-pressure +
  compound-repair-after-RTR arbiters).

**Still blocked on third-party hosted-model substrate.** Frontier-
model substrate access remains the unsolved research-line wall;
W75 carries the W70 ``frontier_blocked_axes`` set forward
unchanged (``W75-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP``).

## Multi-agent task line — the primary scoreboard

R-183 sweeps all fifteen MASC V11 regimes at 5 seeds × 4 seed
sets = 20 seeds per regime per family. Every regime returns
``v20_beats_v19_rate ≥ 0.5`` and
``tsc_v20_beats_tsc_v19_rate ≥ 0.5``. The new chain regime
(``compound_repair_after_replacement_then_rejoin_under_budget``)
reaches 100 % strict-beat for V20 vs V19. ``team_success_per_
visible_token_v20`` is non-trivial across all regimes.

## Replay / recompute / handoff economics

* **Recompute saving.** Substrate compound-chain-repair-dominance
  flops vs full recompute across eleven primitives saves 95.8 %
  at 128 tokens (``substrate_compound_chain_repair_dominance_
  flops_v20``).
* **Cache reuse.** Cache-aware planner V8 saves ≥ 87 % input
  tokens on 18×8 plans at hit_rate=1.0 with six-layer rotation.
* **Cross-plane handoff.** Handoff V7 saves ≥ 87 % visible
  tokens vs forcing every turn through hosted_only (default
  workload).
* **Hosted abstain.** Logprob V8 lowers the abstain floor under
  high compound-chain pressure, shrinking effective top-k by 50 %.

## Live / model-backed findings

W75 does NOT bridge to third-party hosted models at the substrate
layer; the hosted V8 plane operates only at the HTTP text +
logprobs + prefix-cache surface. The substrate V20 plane is a
22-layer NumPy byte-tokenised in-repo runtime. No live frontier-
model run; the multi-agent task-success wins are inside the W75
synthetic MASC V11 harness. ``W75-L-MASC-V11-SYNTHETIC-CAP`` and
``W75-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` document.

## Theory / limitations

See ``docs/THEOREM_REGISTRY.md`` for the canonical W75-T-* and
W75-L-* / W75-C-* claims.

Key new limitations:

* ``W75-L-NUMPY-CPU-V20-SUBSTRATE-CAP`` — V20 substrate is a
  22-layer NumPy runtime, not a frontier model.
* ``W75-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` — W75 does not
  bridge to third-party hosted models at the substrate layer.
* ``W75-L-HOSTED-V8-NO-SUBSTRATE-CAP`` — hosted V8 modules do not
  claim hidden-state / KV / attention access.
* ``W75-L-COMPOUND-CHAIN-PRESSURE-DECLARED-CAP`` — the compound-
  chain-pressure gate is calibrated on caller-declared signals.
* ``W75-L-MASC-V11-SYNTHETIC-CAP`` — the fifteen-regime wins are
  measured inside the in-repo MASC V11 harness.
* ``W75-L-HANDOFF-V7-NOT-CROSSING-WALL-CAP`` — the V7 handoff
  coordinator preserves the wall as a content-addressed invariant.
* ``W75-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP`` — frontier-model
  substrate access remains the unsolved research-line wall.

## Product boundary

* The released SDK contract is **byte-for-byte unchanged**:
  ``coordpy.__version__ == "0.5.20"``,
  ``coordpy.SDK_VERSION == "coordpy.sdk.v3.43"``.
* Smoke driver passes unchanged.
* All W75 modules ship at explicit-import paths
  (``coordpy.tiny_substrate_v20``, etc.); not re-exported through
  ``coordpy.__init__`` or ``coordpy.__experimental__``.
* No PyPI release.

## Validation

* W75 test suite (33 tests across 5 test files): 33/33 pass.
* W74 regression test suite (16 tests): 16/16 pass — no
  regression on the W74 line.
* Smoke driver: all checks pass.
* R-181 (10 H-bars), R-182 (16 H-bars), R-183 (32 H-bars), R-184
  (14 H-bars) × 4 seed sets: 288/288 cells pass.

## Falsifier and limitation reproductions

R-184 includes the following falsifiers and limitation
reproductions:

* H1056 handoff V7 compound-chain falsifier (honest=0 /
  dishonest=1).
* H1057 boundary V8 falsifier (honest=0 / dishonest=1).
* H1058 KV V20 compound-chain-pressure falsifier (honest=0).
* H1060 W75 substrate is in-repo NumPy.
* H1061 W75 hosted control plane V8 does NOT pierce the wall (≥
  37 blocked axes).
* H1062 no-version-bump invariant.
* H1063 frontier substrate access still blocked.
