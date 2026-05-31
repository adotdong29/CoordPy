# RESULTS — W121: matched EXPOSED official-ICPC control + Maverick same-family contrast

Milestone W121 / `COO-9`. `ultracode` OFF. No version bump; no PyPI; `coordpy/__init__.py`
untouched. Graph refreshed at start (HEAD `8b15e97`) and at end (see milestone summary).

## Headline

W120 ran the W89 mechanism on a ≥30 official-ICPC **RESISTANT** battlefield ⇒ B−A1 =
+0.00 pp (FAIL), but left one live objection: the EXPOSED retirements (W89/W105) and the
RESISTANT ICPC null had never been matched **inside the SAME official ICPC package
family**, so the resistant null could be an ICPC-DIFFICULTY artifact rather than a
CONTAMINATION one. **W121 builds the matched EXPOSED control on the SAME official family
and runs the same-model same-mechanism contrast.**

* **Lane α (matched exposed control): SUCCESS.** A matched EXPOSED battlefield from the
  SAME two official ICPC org surface families W120 used, on the immediately-preceding
  pre-cutoff year-drops ⇒ **42 tier-1 pure pass-fail ≥ 30** (+5 float +1
  custom-with-validator; 2 custom-no-validator excluded), all dated ≤ 2024-08-31
  (EXPOSED for Maverick). Grader self-test **30 all-pass problems / 637 official secret
  cases, each surface PASS**.
* **Lane β (exposed certification + pilot): Maverick EXPOSED-certifiable, pilot EARNED
  and RUN** (see `## Pilot result`).
* **Lane γ (executable dual-battlefield infra): shipped** — 1 module + 3 scripts + 1 test
  file; the exposed-vs-resistant contrast is computed by code, not prose.

## Lane α — matched EXPOSED control battlefield

### The exposed rule (mirror of W120, flipped to the pre-cutoff side)

Same two official `github.com/icpc` surface families as W120 (the `na-ecna-archive`
archive + the `na-rocky-mountain-*-public` repos), on the most-recent pre-cutoff
artifact-complete year-drops, under byte-identical R1/R2/R4/R5/R6/R7/R8 discipline with
the date rule FLIPPED (E3: contest date AT OR BEFORE the Aug-2024 cutoff = EXPOSED).

| surface | source | contest date | tier-1 | matched W120 resistant surface |
|---|---|---|---|---|
| RMRC | `icpc/na-rocky-mountain-2021-public` | 2022-03-14 | 10 | RMRC 2024-2025 |
| ECNA | `icpc/na-ecna-archive` `2022-2023` | 2022-11-12 | 10 | ECNA 2024-2025 |
| RMRC | `icpc/na-rocky-mountain-2022-2023-public` | 2023-02-25 | 10 | RMRC 2025-2026 |
| ECNA | `icpc/na-ecna-archive` `2023-2024` | 2023-11-11 | 12 | ECNA 2025-2026 |

**Typed comparability exclusion:** `icpc/na-rocky-mountain-2023-2024-public` (2024-01-16,
still pre-cutoff) ships a MINIMAL package — secret data only, NO `problem_statement/*.tex`
— so it cannot present a statement to the model (every W120 resistant problem shipped a
statement). The deterministic rule advances to RMRC 2021 to keep all four exposed surfaces
artifact-complete, mirroring W120's 2-ECNA + 2-RMRC structure.

### Combined exposed battlefield (`coordpy_icpc_exposed_control_v1`)

* **50 seen → 48 admitted → 42 tier-1 pure pass-fail** (+5 float +1 custom-with-validator).
* Excluded: 2 custom-no-validator (`teamchange`, `colortubes`), typed + machine-checkable.
* Surfaces: ECNA + RMRC; dates 2022-03-14..2023-11-11 (**all ≤ 2024-08-31 ⇒ EXPOSED**);
  month histogram `{2022-03:13, 2022-11:12, 2023-02:11, 2023-11:12}`.
* `raw_classification_sha256 = 653e3682…`; `manifest_cid = 8acbc7cc…`; core 30-slice CID
  `32d15db5…`.
* Grader self-test (NIM-free, E8): accepted Python references vs official secret cases in
  a fresh subprocess — **30 all-pass problems / 637 cases across all four exposed surfaces
  (RMRC 2021 10/215, ECNA 2022 7/156, RMRC 2022-23 9/192, ECNA 2023 4/74); ≥4 per surface
  ⇒ grader proven executable on EACH exposed surface.** (C++-tuned references that TLE
  under Python at the 25-case/6s cap are NOT counted — honest-floor, as W119 did.)

## Lane β — exposed certification (mirror of the W114 gate, C2 flipped)

`certify_model_exposed_v1` applies C1 (KNOWN cutoff) ∧ **C2e (≥30 problems in months AT
OR BEFORE the cutoff = EXPOSED)** ∧ C3 (reachable/stronger) ∧ C4 (un-run here):

| model | primary cutoff | status | EXPOSED-certifiable |
|---|---|---|---|
| Llama-4-Maverick | **August 2024** (Meta llama4 MODEL_CARD) | KNOWN | **YES** (C1✓ C2e✓[48≥30] C3✓ C4✓) |
| Qwen3-Coder-480B | none stated | UNKNOWN | no (C1) |
| DeepSeek-V4-pro | none stated | UNKNOWN | no (C1) |
| Mistral-Small-4-119B | none stated | UNKNOWN | no (C1) |

`{KNOWN:1, UNKNOWN:3}`; nothing newly disclosed since W120. Maverick is the sole
exposed-certifiable + reachable target (the only model whose exposure side is anchorable),
and the matched-family contrast makes it the right one: W120 established the resistant
cell at the SAME scale.

## Lane γ — matched-family comparison (NIM-free)

`build_matched_family_comparison_v1` asserts (machine-checkable) that the exposed control
and the W120 resistant battlefield **differ only in cutoff side**:

| | EXPOSED (W121) | RESISTANT (W120) |
|---|---|---|
| surface families | ECNA + RMRC | ECNA + RMRC |
| org / package format | github.com/icpc, ICPC-Kattis | github.com/icpc, ICPC-Kattis |
| grader / oracle | `run_icpc_stdin_executor_v1` (secret cases) | same |
| tier discipline | tier-1 core only (gate + pilot) | same |
| difficulty class | ICPC regional | ICPC regional |
| model / evaluator | Maverick / verbatim W108 gates | same |
| tier-1 core | 42 | 45 |
| dates | 2022-03-14..2023-11-11 | 2024-11-11..2025-11-13 |
| core 30-slice CID | `32d15db5…` | `01bf9ef8…` |
| cutoff side | **EXPOSED** (≤ 2024-08-31) | **RESISTANT** (> 2024-08-31) |

`differs_only_in_cutoff_side = True`; the LCB-inherited decision CID re-derives
byte-identically (`258b6ed7…`).

### Executable infrastructure shipped

* `coordpy/coordpy_icpc_exposed_control_v1.py` — exposed constructor + manifest builder +
  exposed admission (E1..E8) + exposed certification gate (C1∧C2e∧C3∧C4) +
  matched-family comparison + the three-branch exposed-vs-resistant interpreter +
  W122 fire condition. Reuses the W120 classifier/oracle/slice-selector + the W114 cutoff
  registry + W117 `run_upstream_construction_v1` (258b6ed7 invariant) with NO duplication.
* `scripts/build_w121_exposed_listing_v1.py` — the LIVE fetch/classify/self-test builder
  (the only network/exec) that produced the pinned snapshot.
* `scripts/run_w121_exposed_control_v1.py` — NIM-free construction/preflight + dual
  comparison + verdict JSON.
* `scripts/run_w121_exposed_pilot.py` — earned-pilot driver (canary + full; dry-run
  loader; reuses the W120 package loaders + verbatim W108 gates).
* `tests/test_w121_exposed_control_v1.py` — 16 tests incl. the three-branch interpreter
  falsifiability + the C2e sub-30 falsifiability + the matched-family invariants.

## Pilot result (W121-β) — EARNED, RAN, FAIL; matched-family contrast

Runbook `docs/RUNBOOK_W121.md` LOCKED before any NIM call (built from HEAD `8b15e97`);
canary (2 problems, ~22 calls) validated the harness end-to-end. Full run:

* **Model** `meta/llama-4-maverick-17b-128e-instruct`; **instrument** the deterministic
  exposed tier-1 core 30-slice (CID `32d15db5…`; surface×year stratified — 7 RMRC-2021 +
  7 ECNA-2022-23 + 7 RMRC-2022-23 + 9 ECNA-2023-24; every problem dated ≤ 2024-08-31 ⇒
  EXPOSED); 1 seed (120001) × 30 × K=5 = **330 NIM calls; WALL 4215 s (~70 min)**.
* **Grader** = official secret cases (token-diff oracle; exit-0-iff-every-secret-case; NO
  LLM judge). **Reflexion feedback** = public samples + judge verdict + stderr only.
* **Gates** = the VERBATIM W108 `_evaluate_phase2_gates` + `_mlb_rates`.

| metric | EXPOSED (W121) | RESISTANT (W120) |
|---|---|---|
| A0 pass@1 | **6.67 %** (2/30) | 20.00 % (6/30) |
| A1 pass@1 | **26.67 %** (8/30) | 23.33 % (7/30) |
| B pass@1 | **30.00 %** (9/30) | 23.33 % (7/30) |
| **B − A1** | **+3.33 pp** | +0.00 pp |
| MLB-1 invocation | 93.33 % (28/30) **PASS** | 83.33 % PASS |
| MLB-2 rescue | 25.00 % (7/28) **FAIL** | 8.00 % FAIL |
| Phase-2 gates | **8/9** | 6/9 |
| **verdict** | **`FAIL`** | `FAIL` |

The mechanism was **genuinely invoked** (reflexion fired on 28/30) and rescued more of its
own attempt-0 failures than on resistant ICPC (25 % vs 8 %), but the net B-vs-A1 effect is
**+1 problem**: 2 rescues (`rsamistake` at reflexion attempt 1, `isbnconversion` at attempt
2) offset by 1 regression (`icouldhavewon`: A1 solved it, B did not). `bench_merkle_root =
618e270a…`.

### Exposed-vs-resistant verdict (the W121 question)

Through the pre-committed three-branch interpreter (`interpret_exposed_vs_resistant_v1`):
**EXPOSED +3.33 pp vs RESISTANT +0.00 pp ⇒ both within the ±3.34 pp null band ⇒**
`EXPOSED_NULL_TOO_CONTAMINATION_CONFOUND_WEAKENS_BOUNDED_CEILING_HARDENS`; **paired seed
NOT earned** (the result is on the null side of the band; chasing a one-problem edge is the
W106 anti-pattern).

* **The difficulty/family loophole is addressed, not closed-by-an-exposed-win.** With the
  family fixed (ICPC) and difficulty comparable (exposed A0 6.67 % ≤ resistant A0 20.00 %,
  so the exposed slice is NOT easier), flipping only exposure produced at most a sub-floor
  +3.33 pp nudge — it did **NOT** reproduce the retirement-grade HumanEval-family margins
  (+5.56 / +7.00, clean PASS). The contamination hypothesis predicted a clean exposed
  margin; it did not appear ⇒ the **strong contamination reading WEAKENS** and
  **difficulty/family-ease is implicated** as the driver of the W89/W105 margins.
* **Contamination not refuted.** A faint contamination-consistent gradient remains (exposed
  +3.33 > resistant +0.00; exposed rescue 25 % > resistant 8 %), sub-floor and single-seed
  — contamination is demoted from "the dominant driver" to "at most a minor contributor".

### What this means

* **No third retirement.** W89 (+5.56) + W105 (+7.00), Llama-3.3-70B contamination-EXPOSED
  HumanEval-family, **remain the only two**. W121 adds none.
* **The mechanism gets a clean retirement-grade margin ONLY on HumanEval-family code** and
  FAILS (sub-floor) on official ICPC code **regardless of exposure** (resistant +0.00 FAIL;
  exposed +3.33 FAIL). The bounded ceiling HARDENS toward
  HumanEval-family-(ease/structure)-specific at 70B.
* **Contamination-confound WEAKENED** (first matched-family within-model exposure control;
  see `CONTAMINATION_CONTROL_FRAMING_W121_V1.md`), not refuted (single seed; faint gradient).

### W122

Pre-committed (`RUNBOOK_W121` § 9, CONFOUND_WEAKENS branch): accept the hardened bounded
ceiling and pursue a genuinely different axis (NOT another ICPC rerun); OR a reachable
stronger-than-Maverick model with a primary-KNOWN cutoff re-opens BOTH matched ICPC
battlefields; OR (optional, not earned now) a single paired seed on BOTH battlefields to
tighten the single-seed caveat. `COO-9` stays lead.

Artifacts: `results/w121/exposed_pilot/w121_exposed_pilot_meta_llama-4-maverick-17b-128e-instruct_*/`
(`exposed_reflexion_bench_report.json` + `provenance.json` + `exposed_reflexion_calls.jsonl`
sidecar with per-call prompt/response SHAs); `results/w121/exposed_control/exposed_control_verdict.json`.
