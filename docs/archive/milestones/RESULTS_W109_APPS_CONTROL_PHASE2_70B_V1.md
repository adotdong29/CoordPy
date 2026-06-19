# RESULTS — W109 APPS contamination-control Phase-2 cheap pilot (70B)

**Verdict: `PASS_NON_MECHANISM_DRIVEN`. 9/9 Phase-2 gates; B − A1 = +16.67 pp; MLB-2 = 57.14 % PASS, MLB-1 = 23.33 % FAIL.**

This is the **contamination-control counterpart** to the W108 LiveCodeBench
pilot. W108's first contamination-RESISTANT test of the W89 mechanism
(LiveCodeBench 2025) FAILed (B − A1 = −3.33 pp). W109 ran the SAME mechanism,
SAME same-budget K=5 contract, on a contamination-EXPOSED 2021 benchmark (APPS
call-based subset). **It recovered a large same-budget win** — a **double
dissociation by vintage** that is **evidence CONSISTENT with a
contamination-confound, but NOT proof, and NOT a retirement** (APPS is
contamination-EXPOSED → control evidence only). The two confirmed retirements
(W89, W105) are UNCHANGED. W109 adds NO retirement.

---

## 1. Run identity (audit chain)

| Field | Value |
|---|---|
| Model | `meta/llama-3.3-70b-instruct` (the W89/W105/W108 retirement class — clean single-class contrast) |
| Corpus | `codeparrot/apps` via `refs/convert/parquet` @ commit `0f10e424e13e1c2a69f851e153097b71b6734a1f`, config `all`, split `test` |
| Parquet shards | `0000.parquet` (424 202 850 B, SHA `f1c36415…`) + `0001.parquet` (319 566 256 B, SHA `3b02746c…`); 5 000 test problems |
| Call-based subset | 38 problems (of 5 000); 28 interview / 10 introductory; materialized `apps-test.jsonl` (40 075 B) |
| Corpus SHA-256 | `f6c44d76be0eea7669f0ccbd90b6b45fb03a4327d06682073b5cd8f905310918` |
| Pilot slice | 30 problems; 22 interview / 8 introductory |
| Slice CID | `783687d6109d2e452aba8a32952b5569ed7c03d8aa1d040f1a22ef18688c6dcc` |
| Preflight verdict CID | `0cf1a8e2b02acc1511c5db1f2fe3ce79771987a2b9b9759c2dfd978bf5498e7b` |
| Seed | 109001 (single-seed cheap pilot) |
| K (A1 and B) | 5, byte-exact; no early-stop |
| max tests/problem | 25 (corpus max is 5 — truncates nothing) |
| NIM calls | 330 (A0 1 + A1 5 + B 5 per problem × 30) |
| Wall | 4 491.2 s (~75 min; heavy HTTP-429 throttling survived) |
| Bench Merkle root | `a571c08bce7387d1024c7834e005e682ba6d047d7612545f05d02c5de13d4d25` |

A0/A1/B are byte-identical in mechanism to W89/W103/W105/W108
(`coordpy.apps_reflexion_bench_v1`, shape-identical to the LiveCodeBench
bench); only the corpus, executor (JSON-arg-list decode + APPS output-wrapper
tolerance) and the prompt (implement the call-based `fn_name`) differ. NO
LLM-as-judge — executor truth is the subprocess exit code over the call-based
tests.

---

## 2. Empirical result

| Arm | pass@1 |
|---|---|
| A0 (single-shot T=0) | **73.33 %** (22/30) |
| A1 (first-pass-among-K=5, T=0.7) | **73.33 %** (22/30) |
| B (sequential-reflexion-K=5, T=0.7) | **90.00 %** (27/30) |
| **B − A1** | **+16.67 pp** |
| B − A0 | +16.67 pp |

Per-problem A1-vs-B transition surface (30 problems):

| both pass | A1-only (B regressed) | B-only (B beat A1) | neither |
|---|---|---|---|
| 22 | **0** | **5** | 3 |

**B regressed on ZERO problems** and beat A1 on 5 — a clean positive, the
OPPOSITE of W108 LiveCodeBench (where B regressed on 5, rescued 4, net −1). Of
the 5 B-wins: **4 are genuine reflexion rescues** (pids 2466, 2469, 2637,
2746 — attempt-0 failed, the executor-stderr-conditioned repair loop fixed
them) + 1 is an attempt-0 sampling win (pid 2365 — B's single T=0.7 attempt-0
passed where A1's 5 samples did not).

---

## 3. The 9 Phase-2 gates + MLB sub-gates

| Gate | Pass | Value |
|---|---|---|
| G1 slice pre-committed | ✅ | slice CID pinned, reproduced by `--dry-run` |
| G2 A1 < 90 % | ✅ | 73.33 % (non-saturated) |
| G3 B > A1 | ✅ | 90.00 % > 73.33 % |
| G4 (B − A1) ≥ +5 pp | ✅ | +16.67 pp |
| G5 (B − A0) ≥ +5 pp | ✅ | +16.67 pp |
| G6 per-problem majority (≥ 16/30) | ✅ | 30/30 (B never regressed) |
| G7 budget byte-exact | ✅ | A1, B both K=5 |
| G8 audit chain re-derives | ✅ | per-call CIDs + per-seed/bench Merkle |
| G9 executor clean | ✅ | no-LLM-judge subprocess |
| MLB-1 invocation ≥ 33 % | ❌ | **23.33 %** (7/30 problems triggered reflexion) |
| MLB-2 rescue ≥ 33 % | ✅ | **57.14 %** (4/7 invocations rescued) |

**9/9 core gates pass; MLB-2 (rescue rate) is healthy at 57.14 % — HumanEval-
family-like (W103 47 % / W105 56 %); but MLB-1 (invocation rate) FAILS at
23.33 %, so the formal verdict is `PASS_NON_MECHANISM_DRIVEN`.**

---

## 4. Honest interpretation (what this is, and is NOT)

**Why MLB-1 fails, and why it matters less than it looks.** Reflexion is only
INVOKED when attempt-0 fails. On APPS, A0 = 73.33 % — the model solves most
problems first-shot — so reflexion was invoked on only 7/30 problems (< the
33 % floor). But WHEN invoked it was strongly load-bearing (MLB-2 = 57.14 %,
4/7 rescued), and B never regressed. So the +16.67 pp margin is real and mostly
mechanism-driven (4 of 5 B-wins are reflexion rescues); the formal
`PASS_NON_MECHANISM_DRIVEN` label reflects the low INVOCATION frequency, not a
variance-driven margin (contrast the W96-C anti-pattern, where the margin
itself was variance-driven and MLB-2 was weak).

**The double dissociation by vintage (the W109 finding).**

| | exposed (≤ 2024) | resistant (≥ 2025) |
|---|---|---|
| **B − A1** | **APPS +16.67 pp** (W109) | LiveCodeBench −3.33 pp (W108) |
| **MLB-2 rescue** | 57.14 % (load-bearing) | 25 % (weak) |
| **B regressions** | 0 | 5 (net −1) |
| **A0 (single-shot)** | **73.33 %** | **43.33 %** |

The SAME mechanism, SAME budget, recovers a large same-budget win on
contamination-EXPOSED APPS and fails on contamination-RESISTANT LiveCodeBench.
This is **evidence CONSISTENT with the contamination-confound hypothesis**
(`W108-L-…-NOT-DEMONSTRATED-ON-CONTAMINATION-RESISTANT-BENCH-CAP`): the
reflexion-superiority replicates on exposed code (APPS, like HumanEval) but not
on resistant code. A striking secondary signal points the same way: **A0 on
exposed APPS (73.33 %) is +30 pp above A0 on resistant LiveCodeBench
(43.33 %)** — a memorization-consistent first-shot gap that ALSO explains the
low MLB-1 (the model rarely needs to repair APPS because it often already
"knows" the 2021 answer).

**What it is NOT — proof of contamination.** One exposed/resistant control
pair, single-seed (n = 30 each), cannot establish the confound. The APPS PASS
is also `PASS_NON_MECHANISM_DRIVEN` (the invocation sub-gate fails). And APPS
and HumanEval may share a difficulty/structure property orthogonal to
contamination. The hypothesis is now **SUPPORTED** (it was merely OPEN after
W108) — but **NOT established**. Settling it requires a contamination-RESISTANT
PASS or a larger control battery.

**What it is NOT — a retirement.** APPS is 2021 vintage = contamination-EXPOSED
(C7 = C). A PASS on exposed data cannot be publication-grade (the model may
have memorised it — indeed the high A0 suggests it has). W109 adds NO third
retirement; that requires a contamination-RESISTANT PASS at Phase-3 multi-seed,
which does not exist (LiveCodeBench FAILed).

**What it does NOT change.** The two confirmed retirements (W89 +5.56 pp; W105
+7.00 pp) STAND. W109 does NOT overwrite the W108 LiveCodeBench FAIL — both
stand as the two halves of the dissociation. What W109 changes is the
**boundary around the retirements: it is now SHARPER** — there is positive
control evidence that the same-budget advantage may be contamination-linked.
This makes the programme's claim MORE carefully bounded and MORE defensible,
not stronger on superiority.

---

## 5. Carry-forwards

**Added (theorem / infrastructure anchors):**
* `W109-T-APPS-REAL-DATA-FETCH-PINNED` — fetched the real `codeparrot/apps`
  corpus (refs/convert/parquet @ `0f10e424`; SHA-verified shards) and
  materialized the SHA-pinned call-based subset (38 problems; JSONL SHA
  `f6c44d76…`) via `scripts/fetch_w109_apps_corpus.py`. Reproducible.
* `W109-T-APPS-CONTROL-PREFLIGHT-EARNED` — real-data preflight P1∧P2∧P3∧P4 PASS
  (verdict CID `0cf1a8e2…`); the cheap pilot was genuinely earned before NIM
  spend. DISCHARGES `W108-L-APPS-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP` and
  `W108-L-APPS-EXECUTOR-V1-OUTPUT-WRAPPER-TOLERANCE-CAP` (real schema +
  heterogeneous wrapper convention confirmed; executor faithful to official
  APPS `output==expected OR output==expected[0]`).
* `W109-T-CONTAMINATION-CONFOUND-SUPPORTED-NOT-PROVEN` — the double dissociation
  by vintage (exposed APPS +16.67 pp / MLB-2 57 % vs resistant LCB −3.33 pp /
  MLB-2 25 %; A0 73 % vs 43 %) is positive evidence CONSISTENT with the
  contamination-confound. The hypothesis is SUPPORTED, NOT established (one
  single-seed control pair; APPS PASS is non-mechanism-driven on invocation).
* `W109-T-LIVECODEBENCH-DENOISE-NOT-WARRANTED` — the Lane β two-gate rule
  (`coordpy.livecodebench_denoise_decision_v1`) returns NOT WARRANTED on the
  W108 result (negative margin + weak MLB-2; +8.33 pp mean shift multi-seed
  cannot supply; decision CID `290afa46…`). $0 further LCB NIM.

**Added (caps):**
* `W109-L-APPS-CONTROL-PHASE2-70B-PASS-NON-MECHANISM-DRIVEN-CAP` — APPS
  call-based subset, `meta/llama-3.3-70b-instruct`, 1 seed × 30 × K=5:
  B − A1 = +16.67 pp; 9/9 gates; MLB-2 = 57.14 % PASS but MLB-1 = 23.33 % FAIL
  ⇒ `PASS_NON_MECHANISM_DRIVEN`. Contamination-EXPOSED (2021) ⇒ CONTROL
  evidence only, NEVER a retirement and NEVER publication-grade.
* `W109-L-APPS-CONTROL-MLB1-INVOCATION-CAP` — reflexion was invoked on only
  23.33 % of APPS problems because A0 = 73.33 % is high (the model solves most
  first-shot — itself a memorization-consistent signal). The +16.67 pp margin
  is real and mostly reflexion-driven, but the formal invocation sub-gate
  fails; do NOT cite W109 as a clean mechanism-driven PASS.

**Discharged (infra caps confirmed on real data — NOT research retirements):**
* `W108-L-APPS-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP` — DISCHARGED (real schema
  confirmed: `input_output` JSON string → `{fn_name, inputs, outputs}`).
* `W108-L-APPS-EXECUTOR-V1-OUTPUT-WRAPPER-TOLERANCE-CAP` — DISCHARGED (the
  heterogeneous wrapper convention is faithfully matched on real data).

**Stands:** `W108-L-APPS-CONTAMINATION-EXPOSED-2021-VINTAGE-CAP`,
`W108-L-APPS-LOADER-V1-CALL-BASED-SUBSET-ONLY-CAP`,
`W108-L-LIVECODEBENCH-REFLEXION-PHASE2-70B-CAP`,
`W108-L-REFLEXION-NOT-DEMONSTRATED-ON-CONTAMINATION-RESISTANT-BENCH-CAP`.

**NOT retired:** the two confirmed retirements (W89, W105) — unchanged.

---

## 6. What W110 becomes

Per `docs/RUNBOOK_W109.md` § 7 (PASS branch, refined by the confound-consistent
result): the APPS control PASSed on margin, so the contamination-confound is
now the live question. The verdict-changing next move is **W110 = a SECOND
contamination-RESISTANT benchmark** (a later LiveCodeBench `release` window or a
2024+-dated functional set) to test whether the LiveCodeBench FAIL is
LCB-specific or GENERAL to contamination-resistant data:
* if a second resistant benchmark also FAILs ⇒ the confound hypothesis becomes
  much stronger (resistant data systematically fails);
* if it PASSes ⇒ the LCB FAIL was LCB-specific and the confound weakens.

A multi-seed APPS de-noise is LOWER value (APPS can't retire; the +16.67 pp
margin is already large and clean; de-noising would only firm up the MLB-1
invocation rate). The closed Llama-3.1 and 405B branches stay closed. `COO-9`
stays the lead path (the second-code-battlefield charter now carries the
contamination-confound investigation).
