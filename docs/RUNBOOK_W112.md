# RUNBOOK — W112 (stronger-model resistant-code gate + NIM-free-earned M3 strengthening + bounded-claim fallback only if deserved)

**Pre-commit contract. Locked BEFORE any NIM call** — including the sub-second
stronger-model reachability sweep (Lane α) and any earned pilot. `COO-9` REMAINS
the lead path. No version bump; no PyPI; `coordpy/__init__.py` untouched;
advanced work explicit-import only. `ultracode` stays OFF (W112 is a bounded
mechanism/scale milestone, not a repo-wide dynamic-workflow problem).

This milestone executes the W111 pre-commit (`docs/RUNBOOK_W111.md` § 8
bounded-claim-fallback branch + `docs/RESULTS_W111_M3_PATCHER_PROBE_70B_V1.md`
§ 7; `COO-35` close): the resistant ceiling is **not reflexion-specific** at 70B
(reflexion 0/2; M3 sub-reflexion), so the live options are exactly (in priority
order): **(α) a genuinely STRONGER code model reopens superiority on resistant
code, (β) a NIM-free-earned M3 strengthening that clears the 33 % rescue floor,
or (γ) acceptance of the tightly-bounded two-retirement
contamination-EXPOSED-HumanEval-family-at-70B claim as the honest code ceiling.**
W112 attacks α and β aggressively and falls back to γ only if BOTH die honestly.

W112 is NOT a new benchmark-family tournament; NOT another 70B reflexion rerun;
NOT a 405B/Llama-3.1 reopening unless reachability genuinely changes; NOT an
APPS main-lane reopening; NOT an MBPP+ V2 reopening; NOT a bounded-context /
compaction / summarization / token-compression drift (those remain
anti-patterns, not the frontier path). It is exactly THREE lanes.

**The scientific question (stated once, sharply):** the resistant-code ceiling
is real and not reflexion-specific at 70B. **Is it a 70B CAPABILITY limit that a
genuinely stronger code model reopens, or a property of same-budget multi-call
mechanisms that persists at scale?** And, orthogonally: **is there any FAIR
in-budget M3 strengthening with a credible path above the 33 % floor, or did
W111 already close the 70B mechanism-search branch?** If a stronger model wins →
that is the new frontier. If both serious avenues die honestly → the bounded
contamination-exposed claim is the honest ceiling, registered SHARPER.

---

## Linear

* `COO-9` (second/next code benchmark battlefield) — High, **lead path**. W112
  carries its "does the resistant ceiling persist at stronger scale?" question.
* `COO-36` (to be created at end-of-milestone) — the W112 milestone sub-issue
  under `COO-6`.
* Sync discipline: GitHub canonical; Linear synced at end-of-milestone via
  `scripts/sync_linear_github_v1.py` + `linear_github_mapping.json`.

---

## What is NOT in scope (anti-drift)

* No reopening MBPP+ V2 (`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`).
* No reopening frozen cross-modal lines (RealWorldQA @ 11B / MathVista / ChartQA).
* No reopening the closed Llama-3.1 rescue branch
  (`W106-L-…-MARGIN-CAP-…-NOT-EARNED-CAP`) — re-opens only via a genuinely
  different battlefield (cross-scale-UP), never a rescue-concentrated re-run.
* No 405B EXPENSIVE run unless the reachability sweep returns HTTP 200 AND the
  pre-committed earn gate (§ 1α) clears.
* No APPS main-lane NIM (2021 contamination-EXPOSED; control evidence only).
* No more 70B reflexion de-noise on resistant code (a +0.00 / −3.33 pp weak-MLB-2
  point cannot be de-noised into a PASS).
* No LLM-as-judge anywhere; functional/`unittest`-based subset only.
* **No mechanism that needs the hidden test source** — oracle leakage, not a fair
  same-budget mechanism. Every arm sees ONLY the visible docstring/spec + the
  executor pass/fail + its 800-char `stderr_tail`, never the `test` source.

---

## Operational state (pre-W112 facts)

| Fact | Value |
|---|---|
| Confirmed retirements | EXACTLY TWO, both `meta/llama-3.3-70b-instruct` @ 70B: **W89** base HumanEval +5.56 pp; **W105** HumanEval+ +7.00 pp. Both contamination-EXPOSED HumanEval-family (2021). |
| Resistant column (reflexion) | **0/2**: W108 LiveCodeBench 2025 B−A1=−3.33 pp (MLB-2 25 %); W110 BigCodeBench 2024 B−A1=+0.00 pp (MLB-1 40 % PASS, MLB-2 25 % FAIL). |
| Resistant column (different mechanism) | M3 executor-grounded patcher (W111): rescue-concentrated UPPER BOUND +15.38 pp; MLB-2 12.5 % < reflexion 25 % < 33 % floor ⇒ no fair pilot earned. |
| Exposed column | 3/3 (W89 +5.56, W105 +7.00, W109 APPS +16.67 non-mech). |
| Contamination-confound | **STRENGTHENED toward a finding, NOT proven** (two single-seed resistant points; orthogonal difficulty not excluded). |
| 405B gate | CLOSED — HTTP 404 ×5 (W104–W108). Re-probed only as part of the W112 honest reachability sweep; assumed dead until a 200 says otherwise. |
| Operational reachability | `NVIDIA_API_KEY` present; BigCodeBench corpus cached + SHA-pinned (`ca4f352e…`); bcb_venv present; W110 fair 30-slice CID `b69bf3a0…` reproducible. |
| Stable boundary | `coordpy.__version__ == 0.5.20`; `SDK_VERSION == coordpy.sdk.v3.43` |

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α — stronger-model resistant-code MAIN lane.** (1) Lock the
  target-selection rule (§ 1α) + the mechanism-selection rule (§ 1α-mech) +
  the benchmark rule (§ 1α-bench) BELOW, BEFORE probing. (2) Run an honest
  NIM reachability/availability sweep over stronger-than-70B code-capable NIM
  targets (405B first, plus any strictly-stronger reachable class). (3) Rank
  eligible targets; pick the strongest that satisfies C-S1∧C-S2∧C-S3∧C-S4.
  (4) If one is eligible AND the earn gate (§ 1α-earn) clears, run the
  smallest honest BigCodeBench pilot on the EXACT W110 fair 30-slice with the
  9 Phase-2 gates + MLB-1 + MLB-2 (§ 5). (5) If the pilot PASSes strongly,
  pre-commit the immediate cross-check / W113 escalation (§ 8). (6) If NO
  eligible stronger target is reachable, CLOSE Lane α sharply.
* **Lane β — NIM-free M3 strengthening lane.** Mine the hidden-test-coupling /
  semantic failure regime HARDER (§ 2). Build only FAIR strengthening ideas
  (no hidden test source, no oracle leakage, same visible-spec/stderr regime).
  Kill weak ideas at $0. A strengthened M3 earns live NIM ONLY IF it has a
  credible path ABOVE the 33 % floor (§ 2 earn rule). NIM-free by default.
* **Lane γ — graphify / claim-discipline lane.** Refresh graphify from HEAD at
  start + end; use `explain`/`path`/`affected`/`query` for file selection +
  claim surfaces; tighten the truth surfaces so the bounded
  contamination-EXPOSED-HumanEval-family-at-70B ceiling is defensible after
  W112 whatever happens (§ 7).

The only contemplated NIM spend is (a) the Lane α sub-second reachability sweep,
and (b) a single earned BigCodeBench cheap pilot on a stronger model IF § 1α-earn
clears. Lane β is NIM-free unless a strengthened M3 honestly earns it (§ 2).

### § 1α — stronger-model TARGET-SELECTION rule (LOCKED — before probing)

An eligible W112 stronger-model target is a NIM-reachable instruct/chat model
that is strictly stronger than `meta/llama-3.3-70b-instruct` on code AND
preserves same-budget comparability with the W110 BigCodeBench bench:

* **C-S1 (reachable):** HTTP 200 on the NIM chat-completions endpoint with a
  plain `{model, messages, max_tokens, temperature}` body (the W105 probe body).
* **C-S2 (strictly stronger):** larger parameter scale OR a higher published
  code pass@1 than Llama-3.3-70B. NOT a smaller/equal model.
* **C-S3 (same-budget comparable):** a PLAIN single-completion code path where
  K=5 self-consistency + iterative patch are byte-exact-budget-comparable to the
  70B bench. **EXCLUDES reasoning models that emit hidden / long chain-of-thought
  by default** (e.g. DeepSeek-R1, any `*-thinking` / `*-reasoning` / o-style
  variant) — their per-call token budget is NOT comparable to a 70B plain model's,
  which would confound any B−A1 read at the heart of the comparison.
* **C-S4 (honest code path):** plain Python code-gen via the SAME `_initial_prompt`
  (no tool/agent scaffold, no retrieval).

**Probe order:** `meta/llama-3.1-405b-instruct` FIRST (the standing extension),
then any other strictly-stronger reachable class enumerated from the live NIM
`/v1/models` catalogue (e.g. larger-than-70B non-reasoning instruct models).
**Ranking:** prefer a same-FAMILY larger Llama instruct (cleanest cross-scale —
only scale changes) > a cross-architecture strictly-larger NON-reasoning instruct
model. Pick the strongest C-S1∧C-S2∧C-S3∧C-S4 target. **If NO eligible target is
reachable → Lane α CLOSED.**

### § 1α-mech — stronger-model MECHANISM-selection rule (LOCKED — before any pilot)

* **Default lead mechanism on the stronger model = REFLEXION (B)** — at 70B it
  is the best-measured resistant mechanism (BigCodeBench fair-30-slice +0.00 pp,
  genuinely-invoked MLB-1 PASS) vs M3's rescue-concentrated upper bound +
  sub-floor MLB-2. The stronger-model question is cleanest with the
  best-measured 70B mechanism held fixed, varying ONLY scale.
* **M3 may enter the stronger-model lane ONLY IF** (a) a NIM-free strengthening
  materially changes its rescue mechanics (Lane β decides this), OR (b) a
  pre-committed tiny canary shows M3 has genuinely different verdict-changing
  power at the stronger model. Do NOT spend stronger-model NIM on the same weak
  M3 variant that already failed to earn a fair pilot at 70B.

### § 1α-bench — PRIMARY / SECONDARY benchmark rule (LOCKED)

* **BigCodeBench is the PRIMARY battlefield** for the W112 scale-up: it gave the
  stronger resistant negative (W110) and its harness is clean, SHA-pinned, and
  bug-fixed. **Reuse the EXACT W110 fair 30-slice (slice CID `b69bf3a0…`)** so
  the stronger-model result is apples-to-apples with W110's A1=B=70 % — only the
  model changes.
* **LiveCodeBench is the SECONDARY resistant cross-check** — used ONLY if the
  primary stronger-model BigCodeBench pilot PASSes strongly enough to earn it
  (a pre-committed cross-check, never speculative).

### § 1α-earn — stronger-model cheap-pilot EARN rule (LOCKED)

The stronger-model BigCodeBench cheap pilot is EARNED iff ALL hold:

1. An eligible target satisfies C-S1∧C-S2∧C-S3∧C-S4 (§ 1α).
2. A ≤ 2-problem live CANARY confirms the target's plain code-gen path produces
   parseable ```python``` solutions at the byte-exact K=5 budget (no scaffold,
   no reasoning-trace overflow of `max_tokens`).
3. The mechanism is the § 1α-mech default (reflexion B) unless M3 earned entry.

If reachable but C-S3 fails (reasoning model) OR the canary shows a broken /
non-comparable code path → NOT earned; record the reason; close Lane α.

---

## § 2 — Lane β: NIM-free M3-strengthening mining + earn/no-earn rule (LOCKED — applied BEFORE any β NIM)

**The harder mining (executed; $0 NIM).**
`scripts/mine_w112_fair_reachability_v1.py` re-executes the EXISTING W110
BigCodeBench pilot transcripts through the real `unittest` executor and, for
every problem in the MLB-2 DENOMINATOR (attempt-0 sample failed ⇒ the patch loop
is invoked), classifies the failure's **fair-reachability** — whether the fix
information is present in the FAIR regime (visible docstring + executor
`stderr_tail`) or only in the hidden `test` source. The fair-reachable fraction
is a **STRUCTURAL CEILING on MLB-2** for ANY fair same-budget patcher:
oracle-entangled failures sit permanently in the denominator.

**Result (`results/w112/fair_reachability/w110_bcb_fair_reachability.json`):**

| Reachability class (of 12 invoked) | count | % |
|---|---|---|
| UNREACHABLE_MOCK_OR_FIXTURE | 4 | 33.3 % |
| BORDERLINE_CONTRACT_UNDER_MOCK | 3 | 25.0 % |
| UNREACHABLE_OTHER (ValueError, no contract) | 2 | 16.7 % |
| **FAIR_REACHABLE_OUTPUT_VALUE** | **1** | **8.3 %** |
| UNREACHABLE_TIMEOUT | 1 | 8.3 % |
| UNREACHABLE_UNDERSPEC_ASSERT | 1 | 8.3 % |

* **STRICT (reliably fair-reachable) ceiling = 8.3 %** (1/12; the one clean
  contract-no-mock problem) — far below the 33 % floor.
* **GENEROUS (best-conceivable) ceiling = 33.3 %** (4/12) — merely TOUCHES the
  floor (+0.3 pp), and only by counting 3 mock-entangled "contracts" as perfect
  rescues (the W111 `/51` lateral-trade failure pattern).
* **58 % (7/12) of the denominator is mock/fixture-coupled** — the expected
  behaviour lives in the hidden test, information-unavailable to any fair mechanism.

**The four candidate strengthenings, all FAIR, all killed at $0 (none expands the
reliably-reachable set):**

| ID | Idea | Kill reason ($0) |
|---|---|---|
| **S-C** richer typed digest | parse assertAlmostEqual / assertTrue-with-locals / assertRaises / multi-line reprs / traceback frame localisation | the newly-actionable invoked failures are mock-coupled ⇒ raises the GENEROUS bound, NOT the reliably-reachable STRICT set |
| **S-A** multi-candidate failure aggregation | condition the patch on ALL K candidates + digests, not just the latest | improves rescue EFFICIENCY on already-reachable problems; cannot expand the reachable set |
| **S-B** patch ranking / rejection | execute each patch, reject regressions vs prior best | reduces REGRESSIONS only; adds NO new rescues ⇒ cannot raise MLB-2 |
| **S-D** visible-spec doctest invariants | parse `>>>` examples into local self-checks | doctests are ALREADY in the visible prompt at generation; a self-check adds ZERO new information |

**Earn / no-earn rule (LOCKED, conservative, falsifiable):** a strengthened M3
earns live NIM **iff the STRICT (reliably fair-reachable) ceiling ≥ the 33 %
floor** — i.e. a fair patcher must be able to clear the floor on cleanly-reachable
problems ALONE (mock-entangled "contracts" are NOT counted as reliable rescues).
STRICT = 8.3 % ≪ 33 % ⇒ **NO fair strengthening can clear the floor ⇒ KILL Lane β
at $0.** Even the best-CONCEIVABLE bound (a perfect patcher rescuing every
borderline-under-mock) only touches the floor with zero headroom, while every
prior mechanism (W111 M3 12.5 %; reflexion 25 %) lands well below it. This is a
STRUCTURAL strengthening of the W111 EMPIRICAL sub-floor finding: the entire fair
in-budget M3-strengthening design space is capped at the floor by BigCodeBench's
hidden-test-coupling structure.

Lane β verdict: **`NO_FAIR_STRENGTHENING_CAN_CLEAR_FLOOR_KILL_AT_0`** — W111 was
already enough to close the 70B fair-mechanism search branch; W112 shows it is
structural, not one-variant-empirical. No strengthened-M3 NIM is warranted.

---

## § 5 — stronger-model BigCodeBench cheap-pilot gates (LOCKED — only if § 1α-earn EARNS)

**Slice (G1):** the EXACT pinned W110 BigCodeBench 30-problem gold-green slice
(slice CID `b69bf3a0…`) — so the stronger model is directly comparable to W110's
A1=B=70 % at 70B.

**Arms (byte-identical budget/regime to W110):** A0 (single-shot T=0), **A1**
(first-pass-among-K=5, T=0.7), **B** (sequential reflexion K=5) — the § 1α-mech
default. Same executor (headless Agg, `unittest` oracle, no LLM judge), budget
byte-exact.

**The 9 Phase-2 gates + MLB sub-gates** (verbatim from W103–W111), evaluated by
`coordpy.contamination_resistant_interpretation_v1.evaluate_phase2_gates_v1`:

| Gate | Pass condition |
|---|---|
| G1 slice pre-committed | slice CID `b69bf3a0…` |
| G2 A1 < 90 % | non-saturated (a stronger model may SATURATE A1 — itself informative) |
| G3 B > A1 | strict |
| G4 (B − A1) ≥ +5 pp | margin bar |
| G5 (B − A0) ≥ +5 pp | vs single-shot |
| G6 per-problem majority (≥ 16/30) | B did not regress vs A1 |
| G7 budget byte-exact | A1/B both K=5 |
| G8 audit chain re-derives | per-call CIDs + Merkle |
| G9 executor clean | no-LLM-judge `unittest` subprocess |
| MLB-1 | reflexion invoked on ≥ 33 % of problems |
| MLB-2 | of invocations, ≥ 33 % rescued |

**Verdict labels:** `PASS_MECHANISM_DRIVEN` iff 9/9 + both MLB;
`PASS_NON_MECHANISM_DRIVEN` iff 9/9 but an MLB fails; `FAIL` otherwise.

**What a stronger-model BigCodeBench fair-slice `PASS_MECHANISM_DRIVEN`
entitles:** exactly "the same-budget reflexion advantage reopens on
contamination-RESISTANT BigCodeBench at a genuinely STRONGER code model — the
resistant ceiling is a 70B CAPABILITY limit, not universal." It does NOT add a
retirement (Phase-3 multi-seed is W113), and it triggers the § 1α-bench
LiveCodeBench secondary cross-check pre-commit + the § 8 W113 escalation.

---

## § 6 — bounded-claim fallback rule (LOCKED — LAST resort, not default)

The bounded contamination-EXPOSED-HumanEval-family-at-70B claim is accepted as
the programme's current honest code ceiling **iff BOTH** hold:

1. **Lane α dead:** the reachability sweep finds NO eligible stronger target
   (no C-S1∧C-S2∧C-S3∧C-S4 model reachable), so no stronger-model pilot is
   earned.
2. **Lane β dead:** the STRICT fair-reachable ceiling < the 33 % floor (§ 2), so
   no fair strengthened-M3 variant earns live NIM. (DONE: 8.3 % ≪ 33 %.)

If triggered, register `W112-L-…` (§ 7) and accept: **"Within reachable scale and
the fair in-budget mechanism space, neither a stronger code model nor a fair M3
strengthening reopens same-budget superiority on contamination-resistant code;
the two confirmed retirements are contamination-EXPOSED-HumanEval-family-specific
at 70B."** This is a SHARPER bounded claim, not a weaker one. It does NOT prove
the contamination confound; it does NOT retire anything; it does NOT close
`COO-9` (W113 may re-probe stronger scale when reachability changes). The
fallback fires ONLY after the stronger-model sweep AND the Lane β mining have
both been executed — never by default.

---

## § 7 — graphify deliverables (LOCKED)

* Refresh at start (`graphify update .`; confirm built from current HEAD — DONE:
  `2985b55`, "No code-graph topology changes detected" ⇒ already current).
* Use concretely (DONE): `explain run_executor_grounded_patcher_bench_v1`;
  `path` M3↔reflexion bench (3-hop shared-import confirmation);
  `affected run_executor_grounded_patcher_bench_v1` (the M3 dependency surface
  that grounds Lane β file selection).
* Re-ingest after adding the W112 mining script + any new module + tests;
  `explain`/`affected` on whatever lands.
* Refresh at end; record exactly what graphify changed in file selection /
  understanding (`docs/RESULTS_W112_*` + the milestone summary).
* Tighten the claim surface (RESEARCH_STATUS / THEOREM_REGISTRY /
  HOW_NOT_TO_OVERSTATE / CONSOLIDATED narrative / CHANGELOG) so the bounded
  boundary is defensible whatever the lanes return.

---

## § 8 — W113 branch logic (LOCKED)

Driven by the Lane α outcome:

* **Stronger-model reachable + pilot EARNS + `PASS_MECHANISM_DRIVEN`** ⇒ **W113 =
  BigCodeBench Phase-3 multi-seed retirement bench at the stronger model**
  (3 seeds × 100 × K=5; mirrors W103→W105) — the earned path to a
  contamination-RESISTANT THIRD retirement at scale — preceded by the § 1α-bench
  LiveCodeBench secondary cross-check (cheap pilot). This is the new frontier.
* **Stronger-model reachable + pilot `FAIL` / `PASS_NON_MECHANISM_DRIVEN`** ⇒
  **W113 = register that the resistant ceiling PERSISTS at the stronger scale**
  (a capability-axis result: the ceiling is not merely a 70B artifact), accept
  the bounded claim more firmly, and consider a THIRD resistant benchmark only if
  warranted. No reflexion de-noise.
* **No eligible stronger target reachable (Lane α CLOSED) AND Lane β dead** ⇒
  bounded-claim fallback fires (§ 6); **W113 = standing-extension watch** —
  re-probe stronger code models when NIM reachability changes; the bounded
  contamination-EXPOSED-HumanEval-family-at-70B claim stands as the honest code
  ceiling. `COO-9` stays lead; the resistant-superiority sub-question is answered
  negatively within current reach.

`COO-9` stays the lead path in all branches unless the evidence forces a
different code-line move.

---

## § 9 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION ==
  "coordpy.sdk.v3.43"`; no PyPI publish; `coordpy/__init__.py` untouched.
* New work is explicit-import only. W112 adds NO new `coordpy.*` module by
  default (Lane β reuses `coordpy.executor_grounded_patcher_v1` +
  `coordpy.bigcodebench_*`; the 9-gate evaluator
  `coordpy.contamination_resistant_interpretation_v1` is REUSED). New scripts:
  `scripts/mine_w112_fair_reachability_v1.py` (NIM-free Lane β mining; landed),
  `scripts/run_w112_stronger_model_reachability_sweep_v1.py` (Lane α reachability),
  and — only if § 1α-earn clears — a stronger-model pilot runner reusing the
  W111 `_build_nim_gen` + the BigCodeBench reflexion bench.

---

## Honest framing

W112 attacks the honest aggressive move — a genuinely STRONGER code model on
contamination-resistant code first, a FAIR M3 strengthening second — rather than
accepting the bounded claim by default. Lane β's harder mining shows the fair
in-budget M3-strengthening design space is structurally capped at the 33 % floor
by BigCodeBench's hidden-test-coupling (STRICT reachable 8.3 %; 58 % of the
invoked failures are mock/fixture-coupled), so no fair strengthening earns NIM —
a STRONGER (structural) close of the 70B mechanism-search branch than W111's
(empirical) one. Lane α then asks whether SCALE reopens it: an honest NIM
reachability sweep over stronger code models, an eligible-target rule locked
before probing, and the smallest honest BigCodeBench pilot ONLY if a strictly
stronger, same-budget-comparable, non-reasoning model is actually reachable. If
one wins → the new resistant-code frontier + a LiveCodeBench cross-check + a W113
Phase-3 bench. If none is reachable and the fair strengthening is dead → the
bounded contamination-EXPOSED-HumanEval-family-at-70B claim is the honest ceiling,
registered SHARPER, not weaker. Either way: NOT a retirement by itself, NOT proof
of the confound, NOT a reopening of any closed line. The two confirmed
retirements remain W89 + W105. `COO-9` stays lead.
