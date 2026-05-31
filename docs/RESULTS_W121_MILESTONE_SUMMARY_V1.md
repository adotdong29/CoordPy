# W121 — milestone summary (matched EXPOSED ICPC control + same-family contrast)

Milestone W121 / `COO-9` (sibling). `ultracode` OFF. No version bump; no PyPI;
`coordpy/__init__.py` untouched. Graph refreshed at START (HEAD `8b15e97`) and END.

## One line

W121 built the matched **EXPOSED** control on the SAME official ICPC package family W120
used (pre-Aug-2024 editions of the same RMRC + ECNA regionals; 42 tier-1 pure pass-fail ≥
30; grader self-test 30 all-pass / 637 cases each surface), certified Maverick on the
exposed side, locked the runbook, and ran the earned same-model same-mechanism pilot ⇒
**EXPOSED B−A1 = +3.33 pp, FAIL** vs the LOCKED **RESISTANT +0.00 pp, FAIL** (W120). The
matched-family exposure flip did **NOT** reproduce the retirement-grade HumanEval-family
margins (+5.56 / +7.00) ⇒ the strong contamination reading **WEAKENS**, difficulty/
family-ease is implicated, and the bounded ceiling **HARDENS**. **W89 + W105 remain the
only two retirements.**

## Three lanes

* **Lane α (matched exposed control):** SAME two official `github.com/icpc` families
  (`na-ecna-archive` + `na-rocky-mountain-*-public`), pre-cutoff year-drops (RMRC 2021 +
  ECNA 2022-2023 + RMRC 2022-2023 + ECNA 2023-2024) ⇒ 50 seen → 48 admitted → **42 tier-1
  pure pass-fail ≥ 30** (+5 float +1 custom-validator; 2 custom-no-validator excluded), all
  ≤ 2024-08-31 (EXPOSED). SHA `653e3682…`; manifest CID `8acbc7cc…`; 30-slice CID
  `32d15db5…`. Grader self-test **30 all-pass / 637 official secret cases, each surface**.
  RMRC 2023-2024 excluded (typed: minimal package, no shipped statement).
* **Lane β (exposed cert + pilot):** Maverick EXPOSED-certifiable (C1 KNOWN Aug-2024 ∧
  C2e 48 exposed ≥30 ∧ C3 ∧ C4); tier-2 all UNKNOWN. Canary validated; earned pilot RAN
  (330 calls, ~70 min) ⇒ **+3.33 pp FAIL** (8/9; MLB-1 93.33 % PASS / MLB-2 25 % FAIL).
* **Lane γ (executable dual-battlefield pipeline):** `coordpy/coordpy_icpc_exposed_control_v1.py`
  (exposed constructor + manifest + E1..E8 admission + C1∧C2e∧C3∧C4 exposed cert +
  `MatchedFamilyComparisonV1` + three-branch interpreter + W122 fire) + 3 scripts + 16
  tests; reuses the W120 classifier/oracle/slice-selector + W114 cutoff registry + W117
  `run_upstream_construction_v1` (LCB decision CID `258b6ed7…` re-derives byte-identically).

## The 2×N table (after W121)

| setting | model | benchmark | resistance | B − A1 |
|---|---|---|---|---|
| W89 | Llama-3.3-70B | base HumanEval | EXPOSED | **+5.56** (retire) |
| W105 | Llama-3.3-70B | HumanEval+ | EXPOSED | **+7.00** (retire) |
| W108 | Llama-3.3-70B | LiveCodeBench 2025 | RESISTANT | −3.33 (FAIL) |
| W110 | Llama-3.3-70B | BigCodeBench 2024-06 | RESISTANT | +0.00 (FAIL) |
| W113 | Maverick | LiveCodeBench (date-filtered) | RESISTANT | +0.00 (FAIL) |
| W120 | Maverick | official ICPC ≥30 (RMRC+ECNA, post-cutoff) | RESISTANT | +0.00 (FAIL) |
| **W121** | **Maverick** | **official ICPC ≥30 (RMRC+ECNA, PRE-cutoff)** | **EXPOSED** | **+3.33 (FAIL)** |

The clean-PASS margins appear ONLY on HumanEval-family (easy) code. On official ICPC code
the mechanism FAILS regardless of cutoff side — **the W121 EXPOSED cell is the first
exposed-code cell that does NOT show a retirement-grade margin**, which is exactly the
matched-family exposure control that disambiguates exposure from family/difficulty.

## Entitlement after W121

* **IS** (unchanged): TWO confirmed retirements — W89 + W105, Llama-3.3-70B @ 70B
  contamination-EXPOSED-HumanEval-family.
* **Sharper:** the bounded ceiling HARDENS to HumanEval-family-(ease/structure)-specific;
  matched-family exposure within ICPC did NOT reproduce the margin (+3.33 FAIL).
  Contamination-confound **WEAKENED** (first within-family within-model exposure control),
  NOT refuted (faint sub-floor exposure-consistent gradient; single seed).
* **IS NOT:** a third retirement (W121 adds none), a clean exposed win, contamination
  proven OR refuted. Still NOT cross-class / MBPP-family / cross-modal / "context solved".

## Files / tests / discipline

* New: `coordpy/coordpy_icpc_exposed_control_v1.py`; `scripts/build_w121_exposed_listing_v1.py`,
  `scripts/run_w121_exposed_control_v1.py`, `scripts/run_w121_exposed_pilot.py`;
  `tests/test_w121_exposed_control_v1.py` (16 tests); `docs/RUNBOOK_W121.md`,
  `docs/RESULTS_W121_*`, `docs/CONTAMINATION_CONTROL_FRAMING_W121_V1.md`,
  `docs/FRONTIER_RELEVANCE_AUDIT_W121_V1.md`; `results/w121/`.
* Tests: 16 W121 + 42 W119/W120/W114 regression = 58 pass. 31st preflight/earn-discipline
  validation. Explicit-import-only; `coordpy/__init__.py` untouched; no version bump; no PyPI.
* `COO-9` stays the lead path (a within-family FAIL does not force a code-line change).
* **W122** = accept the hardened bounded ceiling / genuinely different axis; OR a
  primary-KNOWN stronger-than-Maverick model on BOTH matched ICPC battlefields; OR (not
  earned now) one paired seed on BOTH battlefields to tighten the single-seed caveat.
