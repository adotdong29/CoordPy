# Frontier-Relevance Audit — W120 official-ICPC multi-surface resistant battlefield

*(Outcome-independent: this audit justifies the INSTRUMENT's frontier relevance; the
pilot RESULT is recorded separately in `RESULTS_W120_*`.)*

## The question

Is a ≥30-task slice of post-Aug-2024 official ICPC problem packages (RMRC + ECNA) a
*frontier-relevant, contamination-resistant* battlefield for testing whether the W89
same-budget multi-agent reflexion mechanism beats a strong single agent — or is it a
soft / stale / leaked target that would inflate a win?

## Why it IS frontier-relevant

1. **Genuinely post-cutoff for the target.** Every admitted contest (ECNA NA-East-Division
   2024-11-11 + 2025-11-10; RMRC 2024-12-03 + 2025-11-13) is strictly after Maverick's
   KNOWN August-2024 pretraining cutoff. The 2025 contests (24 of the 45 core problems)
   post-date it by **14 months**. This is the resistant property W108/W110/W113 had to
   manufacture by date-filtering; here it is intrinsic to the source.

2. **Official, executable, hidden-test graded.** These are official ICPC regional problem
   packages in the ICPC/Kattis format with real `data/secret/*.in`+`*.ans` and the
   default diff oracle — the exact executable-grader artifact W118 proved ABSENT in the
   LiveCodeBench source family. Grading is pass-fail on hidden tests (NO LLM judge, NO
   sample-only false-pass), self-test-verified on each surface (RMRC 16/16 + ECNA
   149/149).

3. **Hard, real competitive problems.** ICPC regional problems are full algorithmic
   problems (graphs, DP, number theory, geometry) with large stress-test secret inputs —
   not LeetCode-easy snippets. A0/A1 are NOT expected to saturate (the G2 anti-ceiling
   concern), so there is genuine headroom for a multi-agent mechanism to be load-bearing
   — or to fail honestly.

4. **stdin/stdout, the original W89 modality.** The W89/W105 retirements were on
   stdin/stdout-style functional code; ICPC is the same modality, so a result here is
   directly comparable to the registered retirements (unlike the cross-modal lines).

## Objections considered (and why they do not inflate a win)

* **"Figures are unavailable to the model."** True for figure-bearing problems; but the
  figure is withheld from ALL arms equally (A0, A1, B), so it cannot manufacture a B−A1
  margin. It can only *depress* absolute pass rates — a conservative bias against a win.

* **"Python may TLE where C++ would pass."** The time limit binds ALL arms equally (every
  arm emits Python); it cannot create a B−A1 gap. Again conservative.

* **"Reflexion could be peeking at hidden tests."** It cannot: reflexion feedback is
  restricted to PUBLIC samples + the judge verdict bit + the Python stderr tail; the
  secret `.in`/`.ans` are used ONLY to score, never shown. This is strictly stronger
  anti-cheat than the LCB bench (which reflected the same public tests it scored on).

* **"Aggregating two surfaces is cherry-picking."** Inclusion is a total deterministic
  machine rule over the official payload; both surfaces are official ICPC org repos under
  the SAME resistant-date + grader-clean rule; the slice is surface×year stratified and
  outcome-blind (CID pinned before any NIM call). No problem was hand-picked.

* **"Float / custom problems were slipped in to reach 30."** No: the ≥30 gate AND the
  pilot use the STRICT tier-1 pure-pass-fail tier only (45 ≥ 30). Float (3) and
  custom-with-validator (1) are documented breadth, not load-bearing for the gate.

## Contamination-confound status

W120 tests the *supply / count / certifiability* of a resistant battlefield and then runs
the pilot; it does **not** by itself prove or disprove the contamination confound (a
single pilot is one data point on one model). The confound framing is recorded, unchanged
in kind, in `CONTAMINATION_CONTROL_FRAMING_W120_V1.md`; what W120 changes is that — for
the first time — the resistant column is tested on a ≥30 grader-clean OFFICIAL battlefield
with a KNOWN-cutoff certified model, removing the "no certifiable resistant instrument"
escape that capped W114–W119.

## Verdict

The W120 battlefield is **frontier-relevant and contamination-resistant by construction**,
with conservative (not inflationary) biases on absolute pass rate. A clean
`PASS_MECHANISM_DRIVEN` here would be the first resistant-code mechanism win; a FAIL
honestly reinforces the bounded ceiling. Either way the result is defensible because the
instrument is honest.
