# Frontier-relevance audit — W118

**Question.** Is W118 frontier-relevant work, or busywork dressed up as progress?

## Verdict: frontier-relevant — a real construction, not a restated blocker

W117 ended with "no post-v6 instrument can be inherited from LCB; waiting on a packaged
`release_v7` or an LCB-published construction provenance." A lazy W118 would re-run that
checker and report "still no v7." W118 did **not** do that. It **built a real
constructor** and ran it LIVE against an official source, producing a concrete,
machine-checkable artifact:

* **894 official post-v6 functional problems** constructed from the official Codeforces
  API (`contest.list` + `problemset.problems`), 2025-04-07 .. 2026-05-30, 130 contests,
  deterministic inclusion, SHA `b6342fd1…` + manifest CID `fb4185a6…`. This is a real
  instrument-identity manifest, committed to `results/w118/`.
* A **family-wide grader sweep** (Codeforces / AtCoder / LeetCode) proving the
  executable grader is absent through any clean official surface — the exact, named,
  machine-checkable blocker.

That is a genuine advance: the post-v6 instrument-identity question moved from
**unanswered/blocked** (W117) to **answered/constructible** (W118), and the residual
blocker is narrowed to ONE artifact.

## What W118 is honestly NOT

* **NOT a retirement** — W118 adds none; W89 + W105 stand.
* **NOT a resistant win** — no pilot ran (grader absent); resistant superiority is still
  0 clean across both scales.
* **NOT contamination proven** — W118 tests construction + grader supply, not the
  confound.
* **NOT an LCB release** — `coordpy_frontier_functional_v1` is a CoordPy-OWNED line; it
  does not pretend to be "LCB v7".
* **NOT a pilot-runnable instrument** — the official source family publishes no
  executable grader; sample-only / operator-synthesised graders are refused.

## Frontier relevance of the constructed instrument

The instrument targets exactly the contamination-resistant frontier the programme needs:
post-2025-04 competitive-programming problems, dated, functional, from an official
source. For Maverick (KNOWN Aug-2024 cutoff) **all 894 problems are resistant** — the
instrument is more favourable, date-wise, than LCB `release_v6` (which tops out at
2025-04-05). The instrument is **identity-certifiable for Maverick on a genuinely-new
slice it never ran** — a verdict-changing pilot the moment a grader exists.

## Honest caveats recorded

* **`W118-L-OFFICIAL-SOURCE-FAMILY-NO-EXECUTABLE-GRADER-CAP`** — the binding blocker.
* **Format caveat** — Codeforces problems are full-program stdin/stdout (not
  LeetCode-style function-completion like LCB-lite); even WITH a grader, executor
  format-compatibility for the W89 mechanism would be a secondary consideration. The
  grader blocks first, so this is recorded but not load-bearing this milestone.
* **Date-snapshot caveat** — the 894-problem manifest is a dated snapshot (2026-05-30);
  re-running yields more problems as Codeforces adds contests. The transform is
  deterministic given the same input bytes (tests verify), and the raw-fetch SHA pins
  provenance.

## Conclusion

W118 is frontier-relevant: it converts the W117 "cannot inherit" blocker into a live,
constructive result (894 official post-v6 functional identities) and isolates the exact
missing official artifact (an executable grader). The honest aggressive move was made;
no close or confounded edge is counted as a win; the bounded ceiling stands; `COO-9`
remains the lead.
