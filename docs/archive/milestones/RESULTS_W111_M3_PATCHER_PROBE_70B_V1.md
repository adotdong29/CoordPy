# RESULTS — W111 M3 (executor-grounded structured-failure patcher) smallest-decisive probe (70B)

**Verdict: M3 did NOT earn the fair 30-slice cheap pilot ⇒ bounded-claim
fallback TRIGGERED. A genuinely-different mechanism does NOT beat same-budget
self-consistency on contamination-resistant code at 70B (cheap-pilot scale).**

W111 attacked the honest aggressive move — a genuinely DIFFERENT mechanism on
contamination-resistant code, not acceptance of defeat by default. After a
NIM-free mechanism-mining pass killed M1 (spec/library planner) and M2 (local
symbol/doc introspection) at $0, the one aligned candidate — **M3, an
executor-grounded structured-failure patcher** (typed expected/actual contract
+ minimal-patch, materially different from prose reflexion) — earned a
smallest-decisive live probe. The probe shows M3 IS interesting (its patch loop
rescued one hard-core problem reflexion could not) but its mechanism is
**sub-reflexion and non-load-bearing** on resistant code (patch-loop rescue rate
12.5 % < reflexion's 25 % < the 33 % floor), so it did not clear the
pre-committed bar to a fair pilot. The two confirmed retirements (W89, W105)
are UNCHANGED. W111 adds NO retirement.

---

## 1. Run identity (audit chain)

| Field | Value |
|---|---|
| Model | `meta/llama-3.3-70b-instruct` (the W89/W105/W108/W109/W110 class — clean single-class contrast) |
| Mechanism under test | **M3** — `coordpy.executor_grounded_patcher_v1` (1 initial sample + 4 executor-grounded structured-patch turns; K=5 byte-exact; same info regime as reflexion B — docstring + executor pass/fail + 800-char `stderr_tail`, NEVER the test source) |
| Baselines | A0 (single-shot T=0) + A1 (first-pass-among-K=5 self-consistency T=0.7), byte-identical to the W110 BigCodeBench bench |
| Corpus | the W110 pinned `bigcode/bigcodebench` v0.1.4; JSONL SHA-256 `ca4f352e…` (re-verified at load) |
| Probe slice | **rescue-CONCENTRATED, hard-core-focused 13-problem slice** (3 both-pass controls `/1,/2,/3` + 8 both-A1+B-fail hard-core + `/26` B-regression + `/51` B-rescue) — an **UPPER BOUND**, NOT a fair slice |
| Slice CID | `b611fae0ed75844232d2e0497b04826e6edf93b6dae1f17f95ce3934cc257d6b` (pinned in `scripts/run_w111_m3_probe.py`; reproduced by `--dry-run`) |
| Seed | 111001 (single-seed cheap probe) |
| K (A1 and M3) | 5, byte-exact; no early-stop |
| NIM calls | 143 (A0 1 + A1 5 + M3 5 per problem × 13) + a 22-call live-path canary |
| Wall | 4 131.1 s (~69 min; heavy HTTP-429 throttling survived, mirroring W110) |
| Executor | `bigcodebench_executor_v1` — fresh `-I` subprocess `unittest` oracle under headless `Agg`; bcb_venv; NO LLM judge |
| Bench Merkle root | `70353e77697f4b523078992e5f9fa412008d2e906fedf751ae2a54c2a0ae8f71` |

---

## 2. Empirical result (rescue-concentrated UPPER BOUND — NOT a fair slice)

| Arm | pass@1 (13-problem hard-core slice) |
|---|---|
| A0 (single-shot T=0) | **30.77 %** (4/13) |
| A1 (first-pass-among-K=5, T=0.7) | **30.77 %** (4/13) |
| M3 (executor-grounded patcher, K=5) | **46.15 %** (6/13) |
| **M3 − A1** | **+15.38 pp** (UPPER BOUND on a rescue-concentrated slice) |

**Per-problem decomposition (the load-bearing detail):**

| problem | A0 | A1 | M3 | M3 first-pass idx | what M3 did |
|---|---|---|---|---|---|
| /1, /2, /3 (controls) | ✅ | ✅ | ✅ | 0 | held (attempt-0) |
| /26 (B-regression) | ✅ | ✅ | ✅ | 0 | held (attempt-0) |
| **/13** (hard-core) | ❌ | ❌ | **✅** | **2** | **PATCH-LOOP rescue** (reflexion B failed this in W110) |
| **/20** (hard-core, OUTPUT_VALUE) | ❌ | ❌ | **✅** | **0** | **attempt-0 SAMPLING win** (NOT the mechanism) |
| /6, /10, /12, /15, /17, /32 (hard-core) | ❌ | ❌ | ❌ | −1 | not rescued |
| /51 (B-rescue) | ❌ | ❌ | ❌ | −1 | NOT rescued (reflexion B rescued this in W110) |

* **M3 = A1 + 2, ZERO regressions vs A1.** The two M3-only wins are `/13` and `/20`.
* **Only ONE of the two is the mechanism.** `/13` is a genuine PATCH-LOOP rescue
  (`first_pass_idx = 2` — a problem the W110 reflexion B arm also failed).
  `/20` is an **attempt-0 sampling win** (`first_pass_idx = 0`): M3's single
  T=0.7 first sample passed where A1's 5 samples missed — luck of the draw, not
  the patch mechanism (the W109 pid-2365 pattern).
* **MLB-1 = 61.5 %** (8/13 invoked the patch loop) but **MLB-2 = 12.5 %** (1/8
  rescued) — **BELOW the W110 reflexion rescue rate (25 %, 3/12) and far below
  the 33 % load-bearing floor.** The M3 patch mechanism is *weaker* than the
  prose reflexion it was designed to beat.
* M3 did **not** replicate reflexion's `/51` rescue. (A1 also failed `/51`, so
  this is not a regression vs A1 — but it means M3 does not dominate reflexion;
  it is a lateral TRADE: M3 gains `/13`, loses `/51`.)

---

## 3. The pre-committed KILL/EARN decision (RUNBOOK_W111 § 4.3) and its honest resolution

The probe slice is rescue-CONCENTRATED by construction → it is an UPPER BOUND,
never a PASS claim. Its job was the KILL/EARN decision:

* **KILL** iff M3 rescued 0 OUTPUT_VALUE hard-core problems (`/15`, `/20`).
* **EARN** the fair 30-slice pilot iff M3 rescued ≥ 1 OUTPUT_VALUE **AND** held
  `/51` **AND** regressed ≤ 1 control.

**Literal result = `AMBIGUOUS`:** M3 rescued 1 OUTPUT_VALUE problem (`/20`) →
not a clean KILL; but it did NOT hold `/51` → did not meet the EARN bar.

**Principled resolution (grounded, not back-fit): M3 did NOT earn the fair
pilot ⇒ bounded-claim fallback.** Three independent reasons, all pre-existing
discipline:

1. **The EARN bar was not met.** The pre-committed gate to the fair pilot
   required holding `/51` (M3 ≥ reflexion's rescue profile). M3 did not.
2. **The margin is NON-mechanism-driven.** The informational 9-gate eval on the
   probe slice is `PASS_NON_MECHANISM_DRIVEN` (9/9 core gates because M3 > A1 on
   THIS slice, but MLB-2 = 12.5 % fails) — a margin WITHOUT a load-bearing
   mechanism, on an UPPER-BOUND slice. The only mechanism rescue is `/13`; `/20`
   is sampling. The W110 reflexion B arm rescued 25 % of its invocations; M3's
   patch loop rescued **12.5 %** — *worse*.
3. **Rescue-concentrated upper bounds erode on fair slices (W104→W105
   precedent).** W104's +10 pp on a rescue-concentrated 30-slice eroded to
   +2.33 pp on the fair 100-slice Phase-3. M3's +15.38 pp here is on a
   *harder-concentrated* 13-slice with a *sub-floor* mechanism rate; on the fair
   30-slice (where A1 ≈ 70 %, only ~9 problems fail A1, mostly mock-coupling
   unreachable by any fair patcher) M3's mechanism would rescue ≈ 1 problem, so
   M3 ≈ A1 ± sampling — it **cannot** produce a `PASS_MECHANISM_DRIVEN** (MLB-2
   would fail the floor). Per the **W106 margin-cap discipline**, the
   verdict-changing power of the ~300-call fair pilot is LOW ⇒ **NOT WARRANTED**.

This is NOT "accepting defeat by default": M3 was given a fair, smallest-decisive
empirical shot, and its mechanism was measured to be sub-reflexion. Running the
fair pilot would violate the pre-commit (EARN not met) AND spend NIM on a run
that cannot change the verdict (W106 logic).

---

## 4. Lane α mechanism-mining (the $0-NIM cheap probe that built the slate)

`scripts/mine_w111_resistant_failure_modes_v1.py` re-executed all 300 W110
BigCodeBench A1+B candidates through the real executor
(`results/w111/mechanism_mining/w110_bcb_failure_census.json`). The
contamination-resistant failure distribution (114 failures):

| class | count | % | which candidate attacks it |
|---|---|---|---|
| **SEMANTIC_LOGIC** | 93 | **81.6 %** | M3 (executor-grounded patcher) |
| TIMEOUT | 9 | 7.9 % | — |
| ENV_HARNESS | 7 | 6.1 % | — |
| OTHER | 3 | 2.6 % | — |
| **API_GROUNDING** | 2 | **1.8 %** | M2 (introspection) — both on `/51`, already rescued |

Hard-core ablation (8 both-A1+B-fail problems): **6/8 MOCK_COUPLING** (the fix
needs the hidden test's mock setup — NOT in `stderr_tail`; a fair mechanism
never sees the test source), **2/8 OUTPUT_VALUE** (`/15`, `/20`; expected value
IS in `stderr_tail`).

**Slate triage (pre-results, RUNBOOK_W111 § 2–3):**

* **M2 (tool-augmented local symbol/doc introspection)** — attacks
  API_GROUNDING = 1.8 % of resistant failures; local stdlib introspection
  cannot reveal hidden-test conventions. **KILLED at $0 NIM.**
* **M1 (library/spec-grounded planner→coder)** — attacks spec-comprehension, but
  the failures are hidden-test-convention (not comprehension), and M1 sacrifices
  a self-consistency sample with no executor grounding. **KILLED at $0 NIM**
  (dominated by M3).
* **M3 (executor-grounded structured-failure patcher)** — the only mechanism
  aligned with the dominant 81.6 % SEMANTIC class AND using the executor signal.
  Admitted to the smallest-decisive live probe (§ 1–3).

The mining finding stands as `W111-T-RESISTANT-FAILURE-IS-SEMANTIC-HIDDEN-TEST-
COUPLING`: contamination-resistant BigCodeBench difficulty is overwhelmingly
hidden-`unittest`-test coupling + spec under-specification, largely
information-unavailable to any *fair* same-budget mechanism at 70B.

---

## 5. Honest interpretation (what this is, and is NOT)

**What it IS — the cleanest possible negative for a genuinely-different
mechanism, with one honest positive.** M3 is materially different from reflexion
(typed digest + explicit expected/actual contract + minimal-patch on the latest
candidate; never the test source) and it **did** rescue `/13` via its patch loop
— a hard-core problem the W89 reflexion mechanism failed in W110. So the
different-mechanism idea is not vacuous; it solved something reflexion couldn't.
But measured fairly, M3's patch loop is **sub-reflexion** (12.5 % vs 25 % rescue
on the hard core) and **non-load-bearing** (below the 33 % floor); its apparent
+15.38 pp is a rescue-concentrated upper bound inflated by one attempt-0
sampling win. It is a **lateral trade** with reflexion, not a beater.

**What it is NOT — a contamination-resistant superiority, a retirement, or proof
of anything.** M3 did not earn the fair pilot; no fair-slice resistant PASS
exists for any mechanism (reflexion 0/2, M3 not earned). W111 adds NO retirement
and retires NO research carry-forward. The two confirmed retirements (W89
+5.56 pp; W105 +7.00 pp) STAND, contamination-EXPOSED-HumanEval-family-specific
at 70B.

**What it sharpens.** The boundary is now defensible against the obvious
"maybe a better mechanism would win" objection: a NIM-free census localised the
resistant failure to hidden-test-coupling semantics; the one mechanism aligned
with that failure (executor-grounded structured patching) was built and probed,
and it underperformed even reflexion. So the resistant-code ceiling is **not
specific to the reflexion mechanism** — it is a property of same-budget
multi-call mechanisms at 70B against hidden-test-coupling difficulty.

---

## 6. Carry-forwards

**Added (theorem / infrastructure anchors):**
* `W111-T-RESISTANT-FAILURE-IS-SEMANTIC-HIDDEN-TEST-COUPLING` — NIM-free
  re-execution census: contamination-resistant BigCodeBench failures are 81.6 %
  SEMANTIC_LOGIC / 1.8 % API_GROUNDING; hard-core 6/8 mock-coupling, 2/8
  output-value (`results/w111/mechanism_mining/w110_bcb_failure_census.json`).
* `W111-T-EXECUTOR-GROUNDED-PATCHER-V1-SHIPS` — `coordpy.executor_grounded_
  patcher_v1` (M3: typed failure digest + explicit expected/actual contract +
  minimal-patch; A0/A1/M3 byte-identical-budget to the W110 line; never the test
  source; 9 PASSing tests incl. a test-source-non-leak fairness guard).

**Added (caps):**
* `W111-L-NO-DIFFERENT-MECHANISM-BEATS-A1-ON-RESISTANT-CODE-AT-70B-CHEAP-CAP` —
  the slate's only aligned candidate (M3) did NOT earn a fair pilot: on the
  smallest-decisive hard-core probe its patch-loop rescue rate was 12.5 % (1/8),
  BELOW reflexion's 25 % and the 33 % floor; its +15.38 pp on the
  rescue-concentrated slice is an upper bound inflated by 1 attempt-0 sampling
  win. M1 + M2 were killed at $0 NIM (attack near-absent failure classes). A
  genuinely-different same-budget mechanism is NOT demonstrated to beat A1 on
  contamination-resistant code at 70B at cheap-pilot scale. NOT a re-runnable
  margin-cap (a sub-floor mechanism cannot be de-noised into a load-bearing PASS).
* `W111-L-M3-PATCHER-SUB-REFLEXION-ON-RESISTANT-HARD-CORE-CAP` — the M3 patch
  loop rescued 1/8 invoked hard-core problems (12.5 %) vs reflexion's 3/12
  (25 %); it rescued `/13` (which reflexion failed) but lost `/51` (which
  reflexion rescued) — a lateral trade, not an improvement. Structured digest +
  minimal-patch framing did not outperform prose reflexion on resistant code.

**Bounded-claim fallback REGISTERED (RUNBOOK_W111 § 6 — LAST resort, now
earned):** at 70B, neither the W89 reflexion mechanism NOR a genuinely-different
executor-grounded patcher beats same-budget self-consistency on
contamination-resistant code at cheap-pilot scale; the two confirmed retirements
are **contamination-EXPOSED-HumanEval-family-specific at 70B**. This does NOT
prove the contamination confound (still two single-seed resistant points); does
NOT retire anything; does NOT close `COO-9`.

**NOT retired:** the two confirmed retirements (W89, W105) — unchanged.

---

## 7. What W112 becomes

Per `docs/RUNBOOK_W111.md` § 8 (bounded-claim-fallback branch): **W112 = the
honest post-fallback move.** The within-budget, fair-regime mechanism space at
70B is now substantially explored on contamination-resistant code (reflexion
0/2; M3 sub-reflexion). The live W112 options, in priority order:

1. **Cross-scale-UP probe** of reflexion and/or M3 on a STRONGER code model if
   one becomes reachable on NIM (405B stays CLOSED at the 5th 404; this is a
   standing extension, not a launch) — the resistant ceiling may be a 70B
   capability limit, testable only at larger scale.
2. **A residual M3 strengthening** (NOT the current sub-reflexion form): the one
   honest positive is that M3's patch loop rescued `/13` where reflexion failed;
   a future patcher with a wider failure window or a different patch policy is a
   *possible* (low-prior) direction — but only after a NIM-free design that
   plausibly clears the 33 % rescue floor, never another sub-floor rerun.
3. **Acceptance of the bounded two-retirement contamination-EXPOSED-HumanEval-
   family claim as the programme's honest code ceiling** — the default if 1–2
   do not produce a NIM-free-earned candidate.

W112 is NOT: a reflexion rerun, an APPS reopening, a 405B/Llama-3.1 reopening, an
MBPP+ V2 reopening, or a bounded-context/compaction drift. `COO-9` stays the
lead path. The closed branches stay closed.

---

## 8. Stable boundary preserved

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`;
no PyPI publish; `coordpy/__init__.py` untouched. New work explicit-import only:
1 new module (`executor_grounded_patcher_v1`) + 2 new scripts
(`mine_w111_resistant_failure_modes_v1.py`, `run_w111_m3_probe.py`) + 9 new
tests. The interpretation evaluator
(`contamination_resistant_interpretation_v1.evaluate_phase2_gates_v1`) was
REUSED (mechanism-agnostic). The one expensive run was the earned 143-call M3
probe; $0 on M1/M2 (NIM-free kills), $0 on a fair pilot (not warranted), $0 on
APPS / reflexion / 405B / Llama-3.1.
