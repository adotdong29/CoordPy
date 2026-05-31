# W120 milestone summary — count-gap closed, Maverick certified + piloted, bounded ceiling HOLDS

`COO-44` (sibling of lead `COO-9`). 2026-05-31. `ultracode` OFF. No version bump; no PyPI;
`coordpy/__init__.py` untouched. 30th consecutive preflight/earn-discipline validation.

## One paragraph

W119 left a **count-only** blocker: the official ICPC grader was present and
self-test-passing, but only 24 < 30 post-cutoff resistant pass-fail tasks came from the
single RMRC surface. W120 did the honest aggressive move on official surfaces only — it
**audited every RMRC exclusion** problem-by-problem (correcting `draftlottery` from
pure-pass-fail to float) and **aggregated a NEW official surface** (`icpc/na-ecna-archive`,
the NA East Division regional archive), reaching **45 tier-1 pure pass-fail resistant
tasks (≥30 with margin), grader self-test 165/165 on each surface**. That made **Maverick
(KNOWN Aug-2024, reachable) certifiable for the first time** on this family. With the
runbook locked and a canary clean, the **earned pilot RAN** (330 NIM calls) and returned
**B − A1 = +0.00 pp, FAIL**. So: the count gap is genuinely closed, the pilot genuinely
ran, and the honest result is that the W89 same-budget multi-agent mechanism **does not
beat a single agent on contamination-resistant official ICPC code** — confirming the
bounded ceiling and **closing the W114–W119 "no certifiable resistant instrument" escape**.

## The three lanes

* **α (count gap): CLOSED.** 51 seen → 49 admitted → **45 tier-1 pure pass-fail** (+3
  float +1 custom = 49 gradeable); 2 typed exclusions (interactive + custom-no-validator).
  RMRC + ECNA, both official `github.com/icpc`, same resistant-date + grader-clean rule.
  Snapshot SHA `b212866f`; manifest CID `bf55bb6c`; 30-slice CID `01bf9ef8`.
* **β (certification): Maverick certifiable.** Primary cutoff KNOWN Aug-2024 (re-verified
  verbatim); reachable; C2 flips 24→45. `{KNOWN:1, UNKNOWN:4}`; nothing new since W119.
* **γ (infra): shipped + tested.** 2 modules + 2 scripts + 13 tests (2 falsifiability);
  decision CID `258b6ed7` invariant preserved via the reused W117 chain.

## Pilot

A0 20.00 / A1 23.33 / B 23.33 %; **B−A1 = +0.00 pp**; MLB-1 83.33% PASS / MLB-2 8.00% FAIL;
6/9 gates; **FAIL** → `BOUNDED_CEILING_HOLDS_ON_RESISTANT_ICPC`. Reflexion invoked on 25/30
but net-zero (2 rescues − 1 regression). ~54 min, single seed 120001.

## Carry-forward ledger

* **Retired:** none added. W89 + W105 STAND as the only two (both exposed-HumanEval, 70B).
* **Added (registered):** `W120-L-RESISTANT-SUPERIORITY-0-CLEAN-ON-OFFICIAL-ICPC-≥30-CAP`
  (resistant superiority 0 clean across four settings; the certifiable-instrument escape
  is closed). `W120-T-OFFICIAL-ICPC-MULTI-SURFACE-≥30-BATTLEFIELD-CONSTRUCTIBLE`
  (45 tier-1 pure pass-fail, grader self-test 165/165, deterministic + SHA/CID-pinned).
* **Strengthened:** contamination-confound (cleanest resistant null) — NOT proven.

## Stronger claim than before?

**Yes, but it is a stronger NEGATIVE.** Before W120 the resistant column was *untestable at
a certified scale* (W114–W119: no certifiable ≥30 grader-clean instrument). After W120 it
is *tested and null*: on a genuinely-new, official, grader-clean, ≥30, post-cutoff
battlefield at the certified Maverick scale, the mechanism ties the single agent
(+0.00 pp). The programme is entitled to say the bounded ceiling is now **empirically
tested on resistant code**, not merely asserted for want of an instrument. It is NOT
entitled to a third retirement, a resistant win, or a proven confound.

## Files

New: `coordpy/coordpy_icpc_battlefield_v1.py`, `coordpy/icpc_reflexion_bench_v1.py`,
`scripts/run_w120_icpc_battlefield_v1.py`, `scripts/run_w120_icpc_pilot.py`,
`tests/test_w120_icpc_battlefield_v1.py`, `docs/RUNBOOK_W120.md`,
`docs/RESULTS_W120_ICPC_BATTLEFIELD_V1.md`, this file,
`docs/CONTAMINATION_CONTROL_FRAMING_W120_V1.md`,
`docs/FRONTIER_RELEVANCE_AUDIT_W120_V1.md`,
`results/w120/icpc_battlefield/{battlefield_snapshot,battlefield_verdict}.json`,
`results/w120/icpc_pilot/.../{icpc_reflexion_bench_report,provenance}.json`.
Updated: `docs/RESEARCH_STATUS.md`, `docs/THEOREM_REGISTRY.md`,
`docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md`, `docs/HOW_NOT_TO_OVERSTATE.md`,
`CHANGELOG.md`, `linear_github_mapping.json`.
