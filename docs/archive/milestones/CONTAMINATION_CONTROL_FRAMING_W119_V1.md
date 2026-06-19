# Contamination-control framing — W119 (official ICPC public-package grader DISSOLUTION + slice-count cap)

**Purpose.** Keep the W119 result honestly bounded: what the official-ICPC-package pivot
DID establish (the executable grader the W118 LCB family lacked is present + self-test-
passing on a genuinely-new resistant battlefield), and what it did NOT (it ran no pilot, it
did not reopen resistant superiority, it did not move the contamination confound).

## The 2×2 is UNCHANGED (no new empirical cell)

W119 spent **$0 NIM** — it ran no model pilot. The contamination 2×2 (exposed vs resistant
× 70B vs Maverick) is exactly as W113 left it:

| | EXPOSED | RESISTANT |
|---|---|---|
| **70B** | W89 +5.56 / W105 +7.00 / W109 APPS +16.67 (retirements / control) | W108 LCB −3.33 / W110 BCB +0.00 (FAIL) |
| **Maverick** | W112 BCB +10.00 (exposure) | W113 LCB +0.00 (FAIL) |

W119 adds NO cell. Resistant superiority remains **0 clean across both scales**. The
contamination-confound status is **UNCHANGED** (STRENGTHENED-not-proven, per W113) — W119
tests **grader supply + slice count**, not the confound.

## What W119 DID establish

* The official ICPC problem-package family (`github.com/icpc`) **ships the executable
  functional grader** — real `data/secret/*.in`+`*.ans` (non-LFS), shipped output
  validators, and accepted reference solutions — the exact artifact W118 proved ABSENT in
  the LiveCodeBench source family.
* The grader is **verifiably EXECUTABLE**: a NIM-free self-test ran official accepted
  Python solutions against the official secret cases under a fresh-subprocess diff oracle =
  **16/16 cases PASS**.
* Both grader-bearing repos (RMRC 2024-2025 + 2025-2026) post-date Maverick's KNOWN
  Aug-2024 cutoff ⇒ **resistant-for-Maverick** on a genuinely-new instrument it never ran.

## What W119 did NOT establish (do not overclaim)

* **NOT a pilot / NOT a resistant win.** No NIM was spent; B−A1 is unmeasured on this
  family. The instrument is grader-admissible but only 24 post-cutoff resistant pass-fail
  problems are available (< 30), which blocks both slice-admissibility AND the C2
  certification gate ⇒ no certifiable model ⇒ no pilot.
* **NOT a contamination proof or move.** W119 tests construction + grader supply + slice
  count; the confound is untouched.
* **NOT an LCB release.** `coordpy_icpc_public_functional_v1` is a CoordPy-OWNED official-
  ICPC-package line, not "LCB v7" and not the W118 Codeforces line.
* **NOT a third retirement.** W119 adds none; W89 + W105 STAND (contamination-EXPOSED-
  HumanEval-family at 70B).
* **A 24-task pilot would not be admissible** — running below the pre-committed 30-slice
  bar would produce an under-powered B−A1 and is refused (the same discipline that gates
  every W93–W118 spend).

## The honest one-paragraph version

W119 dissolved the W118 grader blocker by pivoting to the official ICPC package family,
which ships real, self-test-passing executable graders on a post-Maverick-cutoff resistant
battlefield. But the cleanest single official surface yields only 24 resistant pass-fail
problems — 6 short of the 30 needed to both build the slice and clear the reused C2
certification gate — so no model (not even KNOWN-cutoff Maverick) is pilot-admissible, and
$0 was spent. The blocker moved from "no official grader" to "official grader present, +6
post-cutoff resistant pass-fail tasks short of a clean pilot." Two retirements still stand;
resistant superiority is still 0 clean; the contamination confound is still
STRENGTHENED-not-proven.
