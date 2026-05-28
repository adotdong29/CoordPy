# RUNBOOK — W108 (LiveCodeBench real-data bug-fix + cheap-pilot earning + APPS readiness)

**Pre-commit contract. Locked BEFORE any expensive NIM call** (the
LiveCodeBench functional-subset cheap pilot). The cheap 405B reachability
side-probe (Lane α) is the W104–W107 established sub-second gate probe and is
governed by § 2. `COO-9` REMAINS the lead path. No version bump; no PyPI;
`coordpy/__init__.py` untouched; advanced work explicit-import only.

This milestone executes the W107 pre-commit (`docs/RUNBOOK_W107.md` § 7;
COO-31 close comment): **W108 = LiveCodeBench functional-subset Phase 2 cheap
pilot AFTER operator corpus-fetch confirms schema + live residual, OR the
pre-committed APPS pivot, OR an honest no-go.** The operator corpus-fetch has
landed (real `release_v6`; see § 3).

W108 is NOT a new broad benchmark tournament. It is exactly THREE lanes:
one cheap 405B re-check; LiveCodeBench real-data validation + bug-fix +
pilot-earning attempt (the main lane); APPS backup readiness in parallel.
W108 does NOT drift into bounded-context / compaction / summarization /
token-compression work — those remain anti-patterns, not the frontier path.

---

## Linear

* `COO-9` (second/next code benchmark battlefield) — High, **lead path**.
  W108 executes its DoD on the LiveCodeBench battlefield.
* `COO-32` (to be created) — the W108 milestone sub-issue under `COO-6`.
* Sync discipline: GitHub is canonical truth; Linear synced at end-of-
  milestone via `scripts/sync_linear_github_v1.py` + `linear_github_mapping.json`.

---

## What is NOT in scope (anti-drift)

* No reopening MBPP+ V2 (`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`).
* No reopening the frozen cross-modal lines (RealWorldQA @ 11B / MathVista /
  ChartQA).
* No reopening the closed Llama-3.1 rescue-concentrated branch
  (`W106-L-…-MARGIN-CAP-…-NOT-EARNED-CAP`; closed NO-GO at W106).
* No expensive 405B run (Lane α is a sub-second reachability probe only).
* No LLM-as-judge anywhere; no contamination hand-waving; functional-subset
  only unless a stronger executor story is built first.
* No claim that LiveCodeBench strengthens the retirement claim until a
  Phase-3 multi-seed retirement bench clears (this milestone is Phase-2
  cheap pilot at most).

---

## Operational state (pre-W108 facts)

| Fact | Value |
|---|---|
| Confirmed retirements | EXACTLY TWO, both `meta/llama-3.3-70b-instruct` @ 70B: **W89** base HumanEval +5.56 pp; **W105** HumanEval+ +7.00 pp (6/6 bars; MLB-2 55.62 %) |
| Not entitled | cross-class (Llama-3.1 FAIL_MARGIN +2.33 pp), cross-scale-UP (405B 404 ×4), MBPP-family (W102 cap), cross-modal (frozen @ 11B), "context solved" |
| 405B gate | CLOSED — HTTP 404 at W104/W105/W106/W107 (4 consecutive) |
| LiveCodeBench (W107) | structurally-sound PRIMARY (S1∧S2∧S3); APPS structural-pivot BACKUP; residual published-baseline-grade pending real fetch |
| Stable boundary | `coordpy.__version__ == 0.5.20`; `SDK_VERSION == coordpy.sdk.v3.43` |

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α — 405B reachability gate.** Re-run the cheap probe ONCE
  (`scripts/run_w105_405b_reachability_probe.py`). Record the W108 gate
  decision (§ 2). If still HTTP 404 → close α immediately; β is the main
  lane. Reachable is **signal, not permission** (§ 2).
* **Lane β — LiveCodeBench main lane.** Audit the partial W108 scaffold →
  reproduce the failing gold-path smoke → diagnose + fix the real-data
  bug(s) → lock the fix in tests → confirm corpus fetch / schema / SHA pin
  (§ 3) → re-run the preflight on REAL data → if the gold-path bar (§ 4)
  passes and the real-data preflight passes, launch the cheap pilot (§ 5);
  else pivot to APPS (§ 6) or stop honestly.
* **Lane γ — APPS backup.** Build real APPS loader/executor/preflight
  scaffolding + write the exact pivot conditions (§ 6) so a pivot is
  possible IN THIS milestone, not deferred to W109.

The lanes run in the same milestone. α does NOT derail β unless this RUNBOOK
is explicitly re-locked before any expensive 405B call.

---

## § 2 — 405B reachability gate rule (LOCKED)

Target = `meta/llama-3.1-405b-instruct`. One sub-second probe.

* `status == "reachable"` (HTTP 200) ⇒ **GATE = OPEN** ⇒ α becomes a latent
  live lane. Reachable is a NECESSARY signal, NOT sufficient permission: a
  405B pilot would additionally require the W107 § 3 earning rule
  (`PASS_MECHANISM_DRIVEN` on the pre-locked W105 inner-kernel slice) AND a
  re-lock of this RUNBOOK before any expensive 405B call. Absent that
  re-lock, β remains the W108 main lane.
* `http_status == 404` ⇒ **GATE = CLOSED** ⇒ refresh
  `W104-L-…-405B-UNREACHABLE-ON-NIM-CAP` with the W108 timestamp; this would
  be the 5th consecutive 404. β is the main lane.
* any other status (non-404 error / exception / no key) ⇒ **GATE = CLOSED
  (indeterminate)** ⇒ treated exactly as 404 for branch purposes; raw status
  recorded verbatim.

The probe NEVER widens any W108 matrix by itself.

---

## § 3 — LiveCodeBench real-data validation rule (LOCKED)

The operator fetch has landed. The corpus is PINNED:

* dataset `livecodebench/code_generation_lite`, release **`release_v6`**,
  file `test6.jsonl`, HF tree commit `0fe84c39…`;
* cache `~/.cache/coordpy/livecodebench-test6.jsonl`;
* SHA-256 **`bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`**
  (verified; the loader REFUSES on mismatch / missing cache / schema
  mismatch — the W102 silent-degeneration guard).

**Confirmed real schema** (this discharges
`W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP` for the
field-name / encoding surface):

* `public_test_cases` and `metadata` are BOTH JSON **strings** (not native
  objects);
* each functional test's `input` = one JSON value per newline-separated line
  == one positional argument (signature order, after `self`); `output` = a
  single JSON value;
* `metadata` carries `{"func_name": …}`.

**The W108 bug + fix (the load-bearing correctness work):** the W107 loader
unwrapped the `public_test_cases` string but read `func_name` only when
`metadata` was already a dict — so on real rows `func_name` was silently
`""`, the executor returned `ENTRY_NOT_FOUND` (rc 3) on every arm, and the
gold-path smoke read `A0=A1=B=0.0`. `coordpy.livecodebench_loader_v1.
_resolve_func_name` now parses both metadata encodings (+ a `starter_code`
fallback). `livecodebench_executor_v2` decodes the confirmed
newline-per-argument input. Both are locked by
`tests/test_w108_livecodebench_realdata_v1.py`.

**Real-data preflight** (`scripts/run_w108_livecodebench_preflight.py`;
NIM-free; verdict `results/w108/livecodebench_preflight/preflight_verdict.json`,
verdict CID `61b9961c46943a5db8dbfd66b75844bfa2d092eac5543df2f4a613556e06bf33`):
P1 corpus integrity, P2 executor-V2 self-test (incl. REAL gold zigzag PASS +
real wrong zigzag FAIL), P3 loader real-data self-test (63 functional, ALL
`func_name` resolved, 63 plain-arg, difficulty 17e/26m/20h, dates
2025-01-11…2025-04-05 — ALL post the Llama-3.x 2024-01-01 cutoff), P4
deterministic outcome-blind slice. **Result: OVERALL PASS — pilot EARNED.**

Honest residual: the LIVE A1@K=5 residual is NOT pre-measured (no
re-executed sidecar). It is measured BY the cheap pilot (gate G2). The
published-baseline prior (Llama-3.x-70B LiveCodeBench pass@1 ≈ 30–50 %) is
recorded but is NOT a gate input. `W107-L-LIVECODEBENCH-RESIDUAL-PUBLISHED-
BASELINE-GRADE-CAP` stays until a Phase-3 bench re-executes the residual.

---

## § 4 — Gold-path correctness bar that MUST pass before any pilot (LOCKED)

NO NIM is spent until ALL of the following hold (they DO, as of this lock):

1. `tests/test_w108_livecodebench_realdata_v1.py` — **all green** (19 tests),
   including: metadata-JSON-string → `func_name` resolves; empty `func_name`
   → `ENTRY_NOT_FOUND` (the bug's failure mode, as a regression guard);
   executor_v2 gold Solution-method PASS on the newline encoding; multi-arg
   newline decode; wrong solution FAIL (no false-pass); the FULL bench
   gold-path (A0=A1=B=1.0 with a stub gen returning the gold solution — the
   exact smoke that was failing); slice-selector determinism + difficulty
   stratification.
2. The real-corpus binding tests pass (SHA pin matches; every functional
   problem resolves a non-empty `func_name`; real gold zigzag passes
   end-to-end).
3. The W108 real-data preflight (§ 3) reports `overall_pass=true`.
4. The pilot driver `--dry-run` reproduces the pinned slice CID
   `2afc318cb9a24d9a52b8914082cfbddaa8e941ef85d77be5382981621f43aa82`.

If any of these regress, the pilot is NOT launched.

---

## § 5 — Cheap-pilot gates for LiveCodeBench (LOCKED — Lane β, if earned)

**Slice (G1, pre-committed):** the deterministic outcome-blind
difficulty-stratified 30-problem slice from
`select_livecodebench_functional_slice_v1`; slice CID
`2afc318cb9a24d9a52b8914082cfbddaa8e941ef85d77be5382981621f43aa82`
(8 easy / 12 medium / 10 hard; contest dates 2025-01-11…2025-02-15; all
post-cutoff).

**A0 / A1 / B (byte-identical mechanism to W89/W103/W105):**

* **A0** — stock single-shot at T=0.0 (1 call/problem).
* **A1** — first-pass-among-K=5 self-consistency at T=0.7 (5 calls).
* **B** — sequential-reflexion-K=5 at T=0.7, each turn conditioned on the
  cumulative (candidate, executor-stderr) history (5 calls).
* Budget byte-exact (A1 and B both K=5; no early-stop); same model on all
  arms; executor truth = subprocess exit 0 iff every public functional test
  matches; NO LLM-as-judge.

**Run:** 1 seed (108001) × 30 problems × K=5 ≈ **330 NIM calls** at
`meta/llama-3.3-70b-instruct` (the W89/W105 retirement class). A ≤3-problem
canary (~22–33 calls) runs first to validate the live path proves the fix
(A0/A1/B not all-zero on real model output).

**The 9 Phase-2 gates + MLB sub-gates (verbatim from W103/W104/W105):**

| Gate | Pass condition |
|---|---|
| G1 slice pre-committed | slice CID pinned (above) |
| G2 A1 < 90 % | A1@K=5 non-saturated (real headroom) |
| G3 B > A1 | strict |
| G4 (B − A1) ≥ +5 pp | margin bar |
| G5 (B − A0) ≥ +5 pp | vs single-shot |
| G6 per-problem majority | B did not regress vs A1 on ≥ 16 of 30 |
| G7 budget byte-exact | A1/B both K=5 |
| G8 audit chain re-derives | per-call CIDs + per-seed/bench Merkle |
| G9 executor clean | no-LLM-judge subprocess |
| MLB-1 | reflexion invoked on ≥ 33 % of problems |
| MLB-2 | of invocations, ≥ 33 % rescued by reflexion |

**Verdict labels (exact):**

* `PASS_MECHANISM_DRIVEN` iff 9/9 gates AND both MLB sub-gates pass.
* `PASS_NON_MECHANISM_DRIVEN` iff 9/9 gates but an MLB sub-gate fails.
* `FAIL` otherwise.

**What a W108 Phase-2 PASS entitles (and does NOT):** a Phase-2
`PASS_MECHANISM_DRIVEN` entitles the claim that the W89 mechanism extends to
a THIRD published code-benchmark family (LiveCodeBench, time-anchored
contamination-resistant) at cheap-pilot scale at 70B. It does **NOT** entitle
a retirement — that requires a Phase-3 multi-seed retirement bench (W109+),
exactly as HumanEval+ required W105 after the W103 cheap-pilot PASS.

---

## § 6 — APPS pivot conditions (LOCKED — Lane γ)

APPS is the structural-pivot backup. It becomes the active battlefield IN
THIS milestone iff LiveCodeBench fails its real-data structural-soundness
test, defined as ANY of:

* P1 corpus integrity FAIL (SHA / schema mismatch the loader cannot bind), OR
* P2 executor-V2 self-test FAIL on real data (gold-path does not pass, or a
  wrong solution false-passes), OR
* P3 functional-subset size < 30 after the real fetch, OR
* the gold-path bar (§ 4) cannot be made green.

**W108 status: none of these fired — LiveCodeBench PASSED real-data
soundness, so NO pivot is triggered.** APPS scaffolding is nonetheless built
to real (loader + executor + NIM-free preflight, explicit-import only) so a
future structural failure pivots without a paperwork milestone. APPS stays
BACKUP because its 2021 vintage is contamination-exposed (C7 = C), which
would weaken any claim built on it — the opposite of LiveCodeBench's decisive
time-anchored C7 = A.

---

## § 7 — graphify deliverables (LOCKED)

* Refresh at start (`graphify update .`; confirm built from current HEAD,
  not a stale commit).
* Use concretely on the LiveCodeBench path + the bounded-retirement claim:
  `graphify query` for the modules defining the LCB path; `graphify affected`
  on the touched loader/executor/bench/preflight/driver/doc surfaces;
  `graphify explain` on the relevant modules; `graphify path` to verify the
  W108 bench sits as the right sibling of the W89/W105 benchmark line.
* Refresh at end and state exactly what graphify changed in file selection /
  understanding.

---

## § 8 — W109 branch logic (LOCKED)

* **If W108 LiveCodeBench Phase-2 = `PASS_MECHANISM_DRIVEN`** ⇒ **W109 =
  LiveCodeBench cross-generation cheap confirmation** at a SECOND model class
  (Llama-3.1-70B) on the BYTE-EQUAL W108 slice (mirrors W103→W104), THEN a
  Phase-3 multi-seed retirement bench if cross-gen also PASSes (mirrors
  W104→W105). 405B re-opens only if it becomes reachable AND this RUNBOOK is
  re-locked.
* **If W108 Phase-2 = `PASS_NON_MECHANISM_DRIVEN`** ⇒ W109 records a
  margin-without-mechanism cap (the W100/W104 MLB discipline) and decides
  whether a different slice or model class is worth a cheap re-check; no
  Phase-3 entitlement.
* **If W108 Phase-2 = `FAIL`** ⇒ register the LiveCodeBench Phase-2 cap with
  the exact failing gate(s); decide between (a) an APPS cheap pilot (Lane γ
  scaffolding already real) or (b) an honest no-go on the post-EvalPlus code
  battlefield → next live move decided explicitly (never a re-run of a
  capped/frozen line).
* **If the pilot could not be launched at all** (e.g. NIM unavailable) ⇒
  W108 closes as preflight-earned-but-operator-gated; W109 = launch the
  earned pilot (no new earning work needed).

---

## § 9 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION ==
  "coordpy.sdk.v3.43"`; no PyPI publish; `coordpy/__init__.py` untouched.
* New/extended modules are explicit-import only. W108 edits the W107 loader
  in place (the documented confirm-at-fetch discharge, not a new module) and
  adds the slice selector to the W108 bench module; the executor-V2 + bench
  scaffold were already present from the partial W108 work.

---

## Honest framing

W108, on a Phase-2 PASS, would entitle exactly: "the W89 sequential-reflexion
mechanism extends to a third published code-benchmark family (LiveCodeBench,
time-anchored / contamination-resistant) at cheap-pilot scale at 70B." It
would NOT add a third confirmed retirement (that is W109+ Phase-3 work), NOT
change the cross-class / cross-scale / cross-modal boundaries, and NOT imply
multi-agent context is "solved." The two confirmed retirements remain W89 +
W105. The 405B cross-scale-UP path remains closed unless reachable.
