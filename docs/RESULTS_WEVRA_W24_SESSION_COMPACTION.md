# SDK v3.25 — bounded-window session compaction + intra-cell resample-quorum + real cross-process producer/decoder wire (W24 family)

> Theory-forward results note for the post-W23 milestone. The first
> capsule-native multi-agent-coordination method that *combines*
> bounded-window cross-cell session compaction with intra-cell
> resample-quorum mitigation and a real OS-level cross-process
> producer/decoder wire — measured on regimes where (a) long
> multi-cell sessions amortise cross-cell state to a O(1)-per-cell
> visible footprint, (b) the W23 PER_CELL_NONCE policy is structurally
> insufficient against intra-cell probabilistic drift, and (c) a real
> subprocess pipe validates the wire-format contract end-to-end. The
> W23-1 / W23-2 / W23-3 results are preserved byte-for-byte; W24
> contributes on the **session efficiency**, **intra-cell mitigation**,
> and **cross-process honesty** axes. Date stamp: 2026-04-29.

## TL;DR

* **W24-1 (session-compaction efficiency, proved-conditional +
  proved-empirical, n=16 saturated × 5 seeds).** On
  R-71-LONG-SESSION the
  :class:`MultiCellSessionCompactor` over a registered
  :class:`SchemaCapsule` strictly reduces
  ``mean_n_w24_visible_tokens_to_decider`` over the W23-delta
  baseline. At ``T_decoder = None`` (loose): W23 = 37.88, W24 =
  31.06 (**−6.81 tokens/cell, −18.0 %**). At ``T_decoder = 24``
  (tight): W23 = 33.19, W24 = 26.38 (**−6.81 tokens/cell, −20.5 %**).
  ``compact_verifies_ok_rate = 0.812`` (13/16 cells beyond the
  ``compact_window = 4`` threshold; the first 3 cells are
  ``W24_BRANCH_BELOW_WINDOW`` by construction). Stable across **5/5**
  alternate ``bank_seed`` values (11, 17, 23, 29, 31): savings ≥
  6.69 tokens/cell on every seed; mean savings 6.79 tokens/cell.
  ``correctness_ratified_rate = 1.000`` byte-for-byte vs W22 on
  every cell of every W24 strategy.
  Code anchor:
  ``vision_mvp.experiments.phase71_session_compaction.run_phase71``.
  Artifacts: ``docs/data/phase71_long_session_K12_n16.json``,
  ``docs/data/phase71_long_session_tight_K12_n16.json``,
  ``docs/data/phase71_seed_sweep_loose_K12_n16.json``,
  ``docs/data/phase71_cross_regime_synthetic.json``.

* **W24-2 (intra-cell-drift mitigation, proved-empirical n=8 +
  proved-empirical-research live n=4).** On R-71-INTRA-CELL-FLIP
  (a synthetic :class:`IntraCellFlippingOracle` whose first consult
  per cell returns a decoy-asymmetric reply and consults #2..M
  return gold-asymmetric, isolated as the only registered oracle so
  its vote is decisive in the W21 quorum), the W23 PER_CELL_NONCE
  baseline ties FIFO at ``accuracy_full = 0.000`` (each cell's
  fresh consult is the bad first one); the W24
  :class:`ResampleQuorumCachingOracleAdapter` (M=3, T=2) achieves
  ``accuracy_full = 0.500`` — **+0.500 strict mitigation advantage
  over W23 PER_CELL_NONCE**. On the live ``mixtral:8x7b`` Mac-1
  probe (n=4, four-cell session): W23 PER_CELL_NONCE = 0.500, W24
  resample = **0.750** — **+0.250 strict gain on a fresh live LLM
  stream**. Empirically discharges
  W23-C-MITIGATION-LIVE-VARIANCE on the synthetic intra-cell
  pattern AND demonstrates partial live transfer.
  Artifacts: ``docs/data/phase71_intra_cell_flip_K12_n8.json``,
  ``docs/data/phase71_live_mixtral_8x7b_n4.json``.

* **W24-3 (trust-boundary soundness + real cross-process wire,
  proved-empirical n=16 + proved by inspection).** Two
  trust-boundary failure modes are mechanically tested by
  enumerated falsifiers:
  * **R-71-COMPACT-TAMPERED** (verifier window override: after
    ``compact_window - 1`` cells, the controller's expected window
    CIDs are set to a phantom tuple): every post-genesis compact
    envelope fires ``window_cids_mismatch`` → 12/16 cells rejected;
    ``correctness_ratified_rate = 1.000`` (W24 falls through to
    W23 byte-for-byte on rejection); ``n_w24_compact_resolved_cells
    = 1`` (only the first cell-after-window verifies before tamper
    fires; subsequent cells reject).
  * **CrossProcessProducerDecoderWire end-to-end real wire**: on
    R-71-CROSS-PROCESS the W24 layer carries a real
    :class:`CrossProcessProducerDecoderWire` that spawns a Python
    subprocess and round-trips JSON envelopes via stdin/stdout
    pipes. ``cross_process_round_trip_bytes_total = 12 861`` on
    n=16 (≈ 803 bytes/cell on the wire). ``cross_process_failures =
    0``. ``correctness_ratified_rate = 1.000`` byte-for-byte vs
    W22.
  The verification function (``verify_session_compact``) is short
  (≈ 35 lines), the failure modes are enumerated (``"ok"`` /
  ``"empty_envelope"`` / ``"schema_version_unknown"`` /
  ``"schema_cid_mismatch"`` / ``"window_size_mismatch"`` /
  ``"window_cids_mismatch"`` / ``"window_cid_mismatch"`` /
  ``"hash_mismatch"``); soundness holds by inspection.

* **W24-Λ-no-compact (proved-empirical, named falsifier).** On
  R-71-NO-COMPACT (chain reset every cell), no cell exceeds the
  ``compact_window`` threshold; ``n_w24_compact_resolved_cells =
  0``; W24 fires ``W24_BRANCH_BELOW_WINDOW`` on every cell and
  reduces to W23 byte-for-byte. ``correctness_ratified_rate =
  1.000``. Names the structural limit when the chain length stays
  below the window.

* **Two-Mac infrastructure (W24-Λ-cross-host).** Mac 2
  (192.168.12.248) ARP ``incomplete`` at milestone capture — same
  status as SDK v3.6 through SDK v3.24 (**18th milestone in a row**).
  **No two-Mac sharded inference happened in SDK v3.25.** The W24-3
  :class:`CrossProcessProducerDecoderWire` upgrades the W23
  within-process round-trip to a real OS-level subprocess pipe — a
  strictly stronger cross-process honesty proxy. When Mac 2 returns,
  the same JSON-canonical interface drops in over a real socket
  with no W24 code changes. Strongest model class actually
  exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE Q4) on Mac-1
  Ollama.

* **Bounded-context honesty preserved byte-for-byte.** The W24
  layer reads only what the W23 layer below it produced; the
  W23 / W22 / W21 / ... / W15 ``tokens_kept`` and ``T_decoder``
  contracts hold byte-for-byte. Mechanically verified:
  * ``Phase71LongSessionTests::test_long_session_strict_savings_loose``
    — W24 ties W22 at 1.000 accuracy AND saves > 5 tokens/cell.
  * ``Phase71CompactTamperedTests::test_compact_tampered_rejection``
    — on rejection, W24 visible-tokens cost equals the W23 baseline
    (no covert savings claimed).
  * ``Phase71NoCompactTests::test_no_compact_zero_savings`` — when
    chain resets every cell, W24 reduces to W23 byte-for-byte.

* **Backward-compat preserved byte-for-byte.**
  * **W24-3-A** (vs W23 / no-trigger paths). With ``enabled = False``
    OR ``schema = None`` OR an inner W23 branch outside
    ``trigger_branches``, the W24 layer reduces to W23
    byte-for-byte on the answer field (mechanically verified by
    ``W24SDKReductionTests``).
  * **W24-3-B** (full programme regression). 121/121 wevra-anchor +
    capsule + phase-69/70/71 tests pass; 33/33 new W24 tests pass.
    The full pre-existing W23 / W22 / W21 / W20 / ... / W3 surface
    is unaffected.
  * **W24-3-C** (W23 anchor regression). The Phase 70
    ``Phase70DeltaFanoutTests::test_delta_savings_loose`` /
    ``Phase70AmplifiedLLMTests::test_w23_quorum_keyed_strictly_improves_over_w22``
    / ``Phase70SuperTokenTamperedTests`` /
    ``Phase70ChainBrokenTests`` all pass unchanged; the W23-1 /
    W23-2 / W23-3 results are preserved byte-for-byte.

* **Audit T-1..T-7 preserved on every cell of every regime.**
  ``Phase71AuditOKTests::test_audit_ok_on_every_w24_cell`` asserts
  ``audit_ok_grid`` is True for every strategy on every bank
  (long_session, long_session_super_token, intra_cell_flip,
  cross_process, no_compact, compact_tampered).

## What changed from SDK v3.24 → v3.25 (one paragraph)

The W23 family (SDK v3.24) crossed the W21-wire-cost wall at the
*per-cell* layer (W22) AND at the *cross-cell* layer via a hash-
chained running session digest, per-cell delta, quorum-keyed cache,
and bounded super-token reference. It explicitly named two open
frontiers: (a) the **W23-C-MITIGATION-LIVE-VARIANCE** conjecture —
that intra-cell probabilistic drift on a single consult survives the
PER_CELL_NONCE mitigation (which only changes drift across cells),
and (b) the W23 within-process producer/decoder host-split proxy
remained a structural simulation rather than a real cross-process
wire. SDK v3.25 implements the smallest honest version of those
moves *together*: a fixed-size :class:`SessionCompactEnvelope` that
folds the last ``compact_window - 1`` cell digests into one envelope
(O(window) visible tokens regardless of session length); a
:class:`ResampleQuorumCachingOracleAdapter` that consults the inner
oracle ``sample_count`` times within one cell and returns the
majority verdict; a real :class:`CrossProcessProducerDecoderWire`
that round-trips JSON envelopes through a Python subprocess's
stdin/stdout pipes; and a synthetic :class:`IntraCellFlippingOracle`
whose drift fits the *intra-cell* pattern that W23 PER_CELL_NONCE
cannot mitigate. Every compact envelope is hash-chained,
schema-versioned, parent-CID-sealed, and verifier-rejectable; on any
verification failure or wire failure the W24 layer rejects and the
W23 path stays sound. The W24 surface is purely additive on top of
the W23 surface; the SDK v3.24 runtime contract is byte-for-byte
unchanged.

## Theorem family W24 (minted by this milestone)

We pre-commit eleven W24 statements. Three are **proved-empirical**
(saturated against pre-committed seeds); two are **proved by
inspection + mechanically-checked**; three are **proved-empirical
backward-compat anchors**; one is **proved-conditional + empirical-
research** (``W24-Λ-real``); one is **conjectural**
(``W24-C-LIVE-VARIANCE-COMPLETE``); one is the named falsifier
(``W24-Λ-no-compact``). Codebase status — all numbered code paths
land in
``vision_mvp/wevra/team_coord.py`` (``SessionCompactEnvelope`` /
``verify_session_compact`` / ``W24CompactionResult`` /
``MultiCellSessionCompactor`` /
``ResampleQuorumCachingOracleAdapter`` /
``CrossProcessProducerDecoderWire`` / ``IntraCellFlippingOracle``)
and
``vision_mvp/experiments/phase71_session_compaction.py`` (R-71
driver + cross-regime + seed-stability sweeps + live LLM
adjudicator hook).

* **W24-1** *(proved-conditional + proved-empirical n=16 saturated × 5
  seeds)*. On R-71-LONG-SESSION-LOOSE / -TIGHT, the
  :class:`MultiCellSessionCompactor` over a registered SchemaCapsule
  strictly reduces ``mean_n_w24_visible_tokens_to_decider`` over the
  W23 baseline AND records ``compact_verifies_ok_rate = 0.812``
  (13/16 cells beyond ``compact_window = 4``) AND preserves
  ``accuracy_full`` byte-for-byte. Stable across 5/5
  ``bank_seed`` values ``(11, 17, 23, 29, 31)``: savings ≥ 6.69
  tokens/cell on every seed; mean savings 6.79 tokens/cell.
  Mechanically verified by
  ``Phase71LongSessionTests::test_long_session_strict_savings_loose``,
  ``Phase71LongSessionTests::test_long_session_strict_savings_tight``,
  ``Phase71SeedStabilityTests::test_savings_strictly_positive_across_5_seeds``.
  *Conditions* (any failure collapses the result):
    1. The bench property — at least ``compact_window`` resolved
       cells share a session.
    2. The inner W23 branch is in ``trigger_branches``
       (``W23_BRANCH_DELTA_RESOLVED`` /
       ``W23_BRANCH_SUPER_TOKEN_RESOLVED`` /
       ``W23_BRANCH_GENESIS``).
    3. The schema is registered AND the verifier's expected window
       CIDs match the producer's last (compact_window - 1) chain
       digests.

* **W24-2** *(proved-empirical n=8 mitigation)*. On
  R-71-INTRA-CELL-FLIP (synthetic
  :class:`IntraCellFlippingOracle`, isolated single-oracle bench
  with quorum_min=1), the W23 PER_CELL_NONCE baseline ties FIFO at
  ``accuracy_full = 0.000`` (each cell's first consult is bad);
  the W24 :class:`ResampleQuorumCachingOracleAdapter` (M=3, T=2)
  achieves ``accuracy_full = 0.500``. **+0.500 strict mitigation
  advantage**. **Empirically discharges
  W23-C-MITIGATION-LIVE-VARIANCE as a mitigable property on the
  intra-cell drift axis** — the within-cell variance is real *and*
  the W24 mitigation is real on the same regime. Mechanically
  verified by
  ``Phase71IntraCellFlipTests::test_w24_resample_strict_mitigation_advantage``,
  ``ResampleQuorumCachingOracleAdapterTests``.

* **W24-3** *(proved-empirical n=8 + proved by inspection)*.
  Trust-boundary soundness. On R-71-COMPACT-TAMPERED (verifier
  window override) and ``W24SDKReductionTests``, every tampered
  envelope is rejected by the controller; the W24 layer fires
  :data:`W24_BRANCH_COMPACT_REJECTED` and the W23 answer is preserved
  byte-for-byte. The verification function is short and the
  failure modes are enumerated. Mechanically verified by
  ``Phase71CompactTamperedTests::test_compact_tampered_rejection``,
  ``SessionCompactVerificationTests``.

* **W24-3-A** *(proved-empirical n=8, backward-compat anchor)*.
  When the inner W23 branch is not in ``trigger_branches`` (W23
  abstained for any reason — NO_TRIGGER / DELTA_REJECTED /
  SUPER_TOKEN_REJECTED / DISABLED), the W24 layer fires
  ``W24_BRANCH_NO_TRIGGER`` and reduces to W23 byte-for-byte on
  the answer field. With ``enabled = False`` OR ``schema = None``,
  the W24 layer reduces to W23 byte-for-byte. Mechanically verified
  by ``W24SDKReductionTests::test_disabled_reduces_to_w23_byte_for_byte``.

* **W24-3-B** *(proved-empirical full programme regression,
  121/121 wevra-anchor + capsule + recent-phase tests pass + 33/33
  new W24 tests pass)*. On R-54..R-70 default banks, the W24 layer
  ties the W23 layer byte-for-byte either via
  ``W24_BRANCH_NO_TRIGGER`` / ``W24_BRANCH_BELOW_WINDOW`` (no
  multi-cell window to compact) or via
  ``W24_BRANCH_COMPACT_RESOLVED`` (when the window forms on
  long sessions). With ``enabled = False`` OR no schema registered,
  W24 reduces to W23.

* **W24-3-C** *(proved-empirical, W23 anchor regression)*. On
  R-70-DELTA-FANOUT, the W23-1 strict-efficiency result holds
  byte-for-byte. On R-70-AMPLIFIED-LLM, the W23-2 mitigation
  holds. On R-70-SUPER-TOKEN-TAMPERED / R-70-CHAIN-BROKEN, the W23-3
  trust falsifiers hold.

* **W24-Λ-no-compact** *(proved-empirical n=8 saturated, named
  falsifier)*. On R-71-NO-COMPACT (chain reset every cell),
  ``n_w24_compact_resolved_cells = 0`` AND
  ``mean_n_w24_visible_tokens_to_decider == W23 baseline`` by
  construction — the W24 layer reduces to W23 byte-for-byte.
  Names the structural limit when the chain length stays below the
  window. Mechanically verified by
  ``Phase71NoCompactTests::test_no_compact_zero_savings``.

* **W24-Λ-real** *(proved-conditional + empirical-research, n=4
  partially discharged)*. Live-LLM transfer on Mac-1 Ollama
  (``mixtral:8x7b``, 47B-MoE) on R-71-INTRA-CELL-FLIP:
  ``acc_full(W23 quorum-keyed) = 0.500``; ``acc_full(W24 resample
  M=3) = 0.750``; **+0.250 strict mitigation advantage on a fresh
  live LLM**. The synthetic ``+0.500`` mitigation advantage
  (W24-2) does not fully transfer (the live LLM does not
  perfectly match the deterministic IntraCellFlippingOracle
  pattern; some samples are gold even on consult #1, some are
  decoy on consult #2,3) — names ``W24-C-LIVE-VARIANCE-COMPLETE``
  as the follow-up frontier. ``cross_process_round_trip_bytes_total
  = 0`` on the live probe (no W24-3 cross-process wire used here;
  the cross-process anchor is on R-71-CROSS-PROCESS, n=16).
  Artifact: ``docs/data/phase71_live_mixtral_8x7b_n4.json``.

* **W24-C-LIVE-VARIANCE-COMPLETE** *(conjectural, named for the
  first time in this milestone)*. The synthetic
  :class:`IntraCellFlippingOracle` (consult #1 = decoy-asymmetric;
  consult #2..M = gold-asymmetric) reproduces intra-cell drift at
  maximum strictness, and the W24
  :class:`ResampleQuorumCachingOracleAdapter` (M=3, T=2) discharges
  it at +0.500 strict gain. **Conjecture**: the same mitigation has
  *positive expected* improvement on live ``mixtral:8x7b`` regimes
  (verified at +0.250 on n=4) and on stronger LLMs the live
  improvement is bounded by the LLM's intra-cell drift pattern's
  similarity to the synthetic IntraCellFlippingOracle. **Falsifier**:
  an LLM whose first-sample drift fits the
  IntraCellFlippingOracle pattern would produce strict W24 > W23
  per-probe at the synthetic +0.500 rate; an LLM whose intra-cell
  drift is unbiased (every sample equally likely to be decoy or
  gold) would produce ``E[mitigation] = 0`` at large n. Empirical
  observation in this milestone: on n=4 live mixtral, the
  mitigation advantage is +0.250 — between the synthetic +0.500 and
  the unbiased 0.000. The conjecture is named, not yet falsified or
  saturated.

* **W24-Λ-cross-host** *(proved-conditional + empirical-research,
  partially discharged)*. The W24-3
  :class:`CrossProcessProducerDecoderWire` upgrades the W23
  within-process round-trip to a real OS-level Python subprocess
  pipe. On R-71-CROSS-PROCESS (n=16):
  ``cross_process_round_trip_bytes_total = 12 861`` (≈ 803
  bytes/cell on the wire); ``cross_process_failures = 0`` (zero
  serialisation failures on the synthetic bench);
  ``correctness_ratified_rate = 1.000`` byte-for-byte vs W22. Mac 2
  unreachable; no true two-host execution validated in SDK v3.25.
  The W24 envelopes are wire-compatible with cross-host deployment
  when Mac 2 returns — no W24 code changes required. Honest scope:
  this is *not* a true two-host setup; it is a real OS-level
  cross-process wire that strictly upgrades the W23 within-process
  proxy on (a) bytes crossing a real pipe, (b) JSON-canonical
  encoding contract, (c) envelope round-trip preservation, AND (d)
  process-isolation (no Python state survives the wire).

## Honest scope (W24)

The W24-1 strict gain is *strongly conditional* on:

1. The bench property — the W24 chain accumulates across ≥
   ``compact_window`` resolved cells (else the W24 layer fires
   ``W24_BRANCH_BELOW_WINDOW`` and reduces to W23 by construction).
2. The inner W23 branch is ``W23_BRANCH_DELTA_RESOLVED`` /
   ``W23_BRANCH_SUPER_TOKEN_RESOLVED`` / ``W23_BRANCH_GENESIS``
   (else W24 reduces to W23 by ``W24_BRANCH_NO_TRIGGER``).
3. The verifier's expected window CIDs match the producer's last
   (compact_window - 1) chain digests (else
   ``window_cids_mismatch`` rejection).

Three named falsifiers make the conditionality sharp:

* **W24-Λ-no-compact** — chain reset every cell → no cross-cell
  state to compact → no savings.
* **R-71-COMPACT-TAMPERED** — verifier window override →
  ``window_cids_mismatch`` → fall through to W23.
* **R-71-INTRA-CELL-FLIP-COMPACT-ONLY** (implicit; covered by
  R-71-INTRA-CELL-FLIP grid where the compact strategy doesn't
  resample) — the W24-1 efficiency win does NOT mitigate
  intra-cell drift; the W24-1 and W24-2 axes are *orthogonal*.

The W24 escape from the W23 per-cell delta cost is *partial* by
design: bounded above by (a) the cross-cell state-sharing rate
(which depends on consecutive-cell bundle similarity), (b) the
window verification success rate (rejected envelopes pay full W23
baseline cost), AND (c) the resolved-cell rate (only resolved cells
contribute to the window).

What W24 does **NOT** do (do-not-overstate):

* Does NOT touch transformer KV caches, embedding tables,
  attention weights, or any model-internal state. The "compact
  envelope" lives at the **capsule layer**; it is an honest
  **proxy** for the LatentMAS *bounded-context running summary*
  direction, not a runtime KV transplant.
* Does NOT solve the live-LLM probabilistic drift problem in
  general. The W24-2 mitigation discharges *intra-cell* drift on a
  synthetic IntraCellFlippingOracle pattern at +0.500 and on live
  mixtral at +0.250, BUT a live LLM whose intra-cell drift is
  unbiased symmetric across samples would produce
  ``E[mitigation] = 0``. See ``W24-C-LIVE-VARIANCE-COMPLETE``.
* Does NOT solve "multi-agent context" in any unqualified sense.
  The W24-1 win is *strongly conditional* on the named bench
  property; under R-71-NO-COMPACT, R-71-COMPACT-TAMPERED, the
  savings claim collapses or the envelope is rejected. See
  ``HOW_NOT_TO_OVERSTATE.md`` § "W24 forbidden moves".
* Does NOT validate true two-host execution. Mac 2 has been
  ARP-incomplete for 18 milestones in a row; the
  CrossProcessProducerDecoderWire is a *real cross-process* proxy
  but not a *cross-host* one. The cross-host wire bytes are not
  measured (only cross-process bytes are).

## Prior-conjecture discharge ledger (SDK v3.25)

* **W23-C-MITIGATION-LIVE-VARIANCE** *(SDK v3.24, named)*.
  **Empirically discharged on the intra-cell drift axis** by W24-2
  on the synthetic R-71-INTRA-CELL-FLIP regime (+0.500 strict gain)
  AND partially discharged on live mixtral at n=4 (+0.250 strict
  gain). The full live discharge axis is named
  ``W24-C-LIVE-VARIANCE-COMPLETE`` — the live mitigation rate is
  bounded by the LLM's intra-cell drift pattern's similarity to the
  synthetic oracle.
* **W22-3-B-Λ-cross-host** *(SDK v3.23, conjectural; partially
  discharged by W23-Λ-cross-host)*. **Further partially discharged**
  by W24-Λ-cross-host: the W24 surface validates the wire-format
  contract via a real OS-level subprocess pipe on every cell
  (12 861 bytes round-tripped on n=16). Mac 2 still unreachable —
  no true two-host execution validated.
* **W21-C-CALIBRATED-TRUST** *(SDK v3.22, named)*. Further
  partially discharged by W24-1 on long sessions: the per-cell
  compact envelope replaces the per-cell verbose digest+delta pair
  and reduces visible-token cost an additional **18.0 %** (loose) /
  **20.5 %** (tight) over the W23 baseline on R-71-LONG-SESSION.

## What this means for the original goal

The original goal of the Context Zero programme is *solving
context for multi-agent teams* — not solving any one regime, but
demonstrating that a typed, content-addressed, lifecycle-bounded
capsule discipline can carry the load that ad-hoc context-passing
in production multi-agent systems cannot.

The W24 family advances that goal on **three new axes** beyond
the W17–W23 result family:

1. **Bounded-window session efficiency.** The W23 per-cell delta
   cost was already small but grew with session length on regimes
   where state changes between cells. The W24 layer extends the
   discharge to the **bounded-window summary layer**: a multi-cell
   coordination session no longer pays the W23 verbose digest+delta
   on cells beyond the window. On R-71-LONG-SESSION (loose):
   −18.0 %; tight (T_decoder=24): −20.5 %. The savings compose
   multiplicatively with the existing W15 ``T_decoder``-budget,
   W21 per-cell N-oracle accounting, W22 shared-read-cache + per-
   cell digest contributions, AND W23 cross-cell delta + super-
   token contributions to a **meaningful cumulative end-to-end
   reduction in the visible context the final decoder pays.**

2. **Empirically-discharged intra-cell probabilistic mitigation.**
   The W23-C-MITIGATION-LIVE-VARIANCE conjecture (named in SDK v3.24
   as a real observed weakness of the W23 PER_CELL_NONCE policy
   when paired with intra-cell-drifting probabilistic LLM oracles)
   is discharged on the synthetic regime at +0.500 strict gain AND
   partially discharged on live mixtral at +0.250 strict gain. This
   is the **first programme-internal demonstration that the
   live-LLM mitigation transfer is non-trivially measurable** — the
   live-LLM advantage is bounded by the LLM's drift pattern's
   similarity to the synthetic oracle, but it is measurably
   positive at this n.

3. **First real cross-process producer/decoder wire.** The W24-3
   :class:`CrossProcessProducerDecoderWire` is the **first real
   OS-level cross-process honesty proxy** in the programme. The
   verification function rejects unknown / tampered windows and
   the W23 path stays sound. The wire's bound is sharp: real
   subprocess pipes, real bytes, real serialisation failures
   (zero on the synthetic bench, but mechanically observable). This
   is the strongest cross-process honesty this repo can validate
   end-to-end on Mac-1 alone — and a structural counter-example to
   the criticism that within-process round-trips smuggle in
   cross-host claims they cannot back up.

What W24 does **not** advance:

* The W19-Λ-total wall (no asymmetric witness anywhere in the
  bundle), the W21-Λ-all-compromised wall (every registered oracle
  jointly compromised), the W22-Λ-no-cache wall (no repeated reads
  to amortise), and the W23-Λ-no-delta wall (no cross-cell state)
  are real and structural; W24 does not cross them. The natural
  escape remains the same.
* True transformer-internal KV reuse / latent hidden-state
  transfer between LLM agents is **not implemented** in this
  repo. Every claim about "bounded-window summary" or "compact
  envelope" in this milestone is at the **capsule layer**, with
  explicit honest proxies (``SessionCompactEnvelope``,
  ``verify_session_compact``); we did not modify any model-internal
  state. If a future programme builds true KV-sharing between
  Apple-Silicon-distributed MLX servers, the W24 typed-envelope +
  verification surface remains a useful integration-boundary
  anchor.
* True two-host execution is **not validated** in this milestone.
  Mac 2 has been ARP-incomplete for 18 milestones in a row; the
  CrossProcessProducerDecoderWire is a real cross-PROCESS wire,
  not a cross-HOST wire. The cross-process bytes are measured (≈
  803 bytes/cell) but a real socket between two machines is not.

## Code anchors

| Theorem / claim | Code anchor |
| --------------- | ----------- |
| W24-1 (efficiency) | ``Phase71LongSessionTests::test_long_session_strict_savings_loose / _tight``; ``docs/data/phase71_long_session_K12_n16.json``, ``docs/data/phase71_long_session_tight_K12_n16.json`` |
| W24-1 stability   | ``Phase71SeedStabilityTests``; ``docs/data/phase71_seed_sweep_loose_K12_n16.json`` |
| W24-2 (intra-cell mitigation) | ``Phase71IntraCellFlipTests``; ``ResampleQuorumCachingOracleAdapterTests``; ``docs/data/phase71_intra_cell_flip_K12_n8.json`` |
| W24-3 (compact tampered) | ``Phase71CompactTamperedTests``; ``SessionCompactVerificationTests``; ``docs/data/phase71_compact_tampered_K12_n16.json`` |
| W24-3 (cross-process wire) | ``CrossProcessProducerDecoderWireTests``; ``Phase71CrossProcessTests``; ``docs/data/phase71_cross_process_K12_n16.json`` |
| W24-3-A (no-trigger / disabled) | ``W24SDKReductionTests`` |
| W24-3-B (regression) | ``test_phase69_*`` + ``test_phase70_*`` + ``test_phase71_*`` + ``test_wevra_capsules`` + ``test_theorems``; pass |
| W24-3-C (W23 anchor) | ``test_phase70_capsule_session_delta`` (passes unchanged) |
| W24-Λ-no-compact  | ``Phase71NoCompactTests``; ``docs/data/phase71_no_compact_K12_n8.json`` |
| W24-Λ-real        | ``docs/data/phase71_live_mixtral_8x7b_n4.json`` |
| W24-Λ-cross-host  | ``Phase71CrossProcessTests``; ``cross_process_round_trip_bytes_total`` field in every R-71 artifact |
| W24-C-LIVE-VARIANCE-COMPLETE | conjectural; observed empirically on mixtral_8x7b live regime |

## Reproducibility

```bash
# R-71-LONG-SESSION-LOOSE (W24-1 anchor):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank long_session --decoder-budget -1 \
    --K-auditor 12 --n-eval 16 --bank-replicates 4 --bank-seed 11 \
    --verbose --out -

# R-71-LONG-SESSION-TIGHT (W24-1 + W15 composition):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank long_session --decoder-budget 24 \
    --K-auditor 12 --n-eval 16 --bank-replicates 4 --bank-seed 11 \
    --verbose --out -

# R-71-INTRA-CELL-FLIP (W24-2 anchor):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank intra_cell_flip --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-replicates 4 --bank-seed 11 \
    --verbose --out -

# R-71-CROSS-PROCESS (W24-3 real-wire anchor):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank cross_process --decoder-budget -1 \
    --K-auditor 12 --n-eval 16 --bank-replicates 4 --bank-seed 11 \
    --verbose --out -

# R-71-NO-COMPACT (W24-Λ-no-compact named falsifier):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank no_compact --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --bank-replicates 4 --bank-seed 11 \
    --verbose --out -

# R-71-COMPACT-TAMPERED (W24-3 trust falsifier):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank compact_tampered --decoder-budget -1 \
    --K-auditor 12 --n-eval 16 --bank-replicates 4 --bank-seed 11 \
    --verbose --out -

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --cross-regime-synthetic --K-auditor 12 --n-eval 16 --out -

# Seed-stability sweep:
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank long_session --seed-sweep --K-auditor 12 --n-eval 16 \
    --decoder-budget -1 --out -

# Live LLM probe (W24-Λ-real, mixtral 8x7b on Mac-1):
python3 -m vision_mvp.experiments.phase71_session_compaction \
    --bank intra_cell_flip --live-llm-adjudicator \
    --adjudicator-model mixtral:8x7b --n-eval 4 --verbose --out -
```

Reproducible from any commit at or after SDK v3.25 with no
configuration beyond the bank, decoder budget, compact_window, and
resample_count knobs.

## Cross-references

* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W24 family entry).
* Research status: ``docs/RESEARCH_STATUS.md`` (axis 21, SDK v3.25).
* Overstatement guard: ``docs/HOW_NOT_TO_OVERSTATE.md`` § "W24
  forbidden moves".
* Master plan: ``docs/context_zero_master_plan.md`` § post-W23
  (SDK v3.25).
* Prior milestone (SDK v3.24 W23):
  ``docs/RESULTS_WEVRA_W23_CROSS_CELL_DELTA.md``.
* Capsule team-level formalism: ``docs/CAPSULE_TEAM_FORMALISM.md``.
