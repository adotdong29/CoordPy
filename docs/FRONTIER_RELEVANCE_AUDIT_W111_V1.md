# Frontier-relevance audit — W111 (supplement to W108/W109/W110 V1)

> Extends `docs/FRONTIER_RELEVANCE_AUDIT_W110_V1.md`; all prior classifications
> remain in force. Classifies the W111 artifacts so future milestones reuse what
> is load-bearing and avoid re-running what is capped.

## Discipline status (#21 consecutive — W93–W111)

W111 executed the pre-committed `docs/RUNBOOK_W111.md` three-lane branch logic;
the bounded-claim fallback was EARNED, not defaulted. W111's distinguishing
additions:

1. **A $0-NIM failure-mode census that drove a slate triage** — re-executing all
   300 W110 candidates localised the resistant failure (81.6 % SEMANTIC / 1.8 %
   API) and killed 2 of 3 candidates (M1, M2) at $0, before any NIM.
2. **A genuinely-different mechanism, built + fairness-guarded** — M3
   (executor-grounded structured patcher) with a unit test that asserts the
   patch prompt NEVER contains the hidden test source (no oracle leakage).
3. **A smallest-decisive probe + a pre-committed refusal to over-spend** — the
   probe measured M3 sub-reflexion; the fair pilot was refused on the
   pre-committed EARN bar + W104/W106 discipline (the ~300-call run cannot change
   the verdict).

## Active frontier arsenal (reusable, load-bearing)

* **`scripts/mine_w111_resistant_failure_modes_v1.py`** — the NIM-free
  failure-mode census (re-execute existing pilot transcripts → taxonomy +
  per-candidate weakness-coverage). Reusable for ANY future "is this mechanism
  worth a NIM run?" triage; it turns committed transcripts into a $0 slate
  filter. THE reusable W111 tool.
* **`coordpy.executor_grounded_patcher_v1`** — the M3 bench (A0/A1/M3
  byte-identical-budget to the W110 line; typed `parse_failure_digest_v1` +
  minimal-patch; reuses the BigCodeBench executor/loader/slice; never the test
  source). Reusable as the scaffold for any future executor-grounded patch
  mechanism (e.g. a wider-window or different-policy variant) — but the CURRENT
  patch policy is capped sub-reflexion (see Dead directions).
* **`parse_failure_digest_v1`** — pure, deterministic `unittest`-stderr →
  `{failing_tests, exception_type, expected_repr, actual_repr}` parser. Reusable
  for any structured-feedback mechanism on a `unittest`-shaped benchmark.

## Useful control / baseline-only (NOT frontier superiority)

* **The 13-problem hard-core rescue-concentrated probe slice (CID `b611fae0…`)**
  — a fast KILL/EARN instrument for a candidate mechanism on resistant code
  (upper bound, never a PASS claim). Reusable for the next mechanism probe.

## Dead directions (capped — do NOT re-run)

* **M2 (tool-augmented local symbol/doc introspection) on resistant code** —
  attacks API_GROUNDING (1.8 % of resistant failures); killed at $0. Do NOT
  present introspection as a resistant-code lever.
* **M1 (library/spec-grounded planner→coder) on resistant code** — attacks
  comprehension (failures are hidden-test-convention), sacrifices a
  self-consistency sample, no executor grounding; killed at $0, dominated by M3.
* **The CURRENT M3 patch policy as a resistant-code beater** — sub-reflexion
  (12.5 % rescue < 25 % < 33 % floor) on the hard core; did NOT earn a fair
  pilot (`W111-L-NO-DIFFERENT-MECHANISM-BEATS-A1-ON-RESISTANT-CODE-AT-70B-CHEAP-CAP`,
  `W111-L-M3-PATCHER-SUB-REFLEXION-ON-RESISTANT-HARD-CORE-CAP`). Do NOT re-run
  the same sub-floor patcher; a future variant must clear the 33 % floor on a
  NIM-free design first.
* **A fair 30-slice M3 pilot** — NOT WARRANTED (rescue-concentrated upper bound +
  sub-floor non-load-bearing mechanism → cannot yield PASS_MECHANISM_DRIVEN; W106
  margin-cap discipline). Closed Llama-3.1 + 405B re-probe stay closed; APPS,
  reflexion reruns stay $0.

## Anti-patterns (reinforced + new at W111)

* **Bounded-context / compaction / prose-summary / token-compression** REMAIN
  explicit anti-patterns, NOT the frontier path. W111 did none of these.
* **NEW W111 anti-pattern**: treating a rescue-concentrated-slice margin as a
  PASS. M3's +15.38 pp is an UPPER BOUND inflated by ONE attempt-0 sampling win;
  the mechanism rescued ONE problem. Always read the per-problem
  `first_pass_idx` (patch-loop rescue vs attempt-0 sampling win) and MLB-2, not
  the headline margin.
* **NEW W111 anti-pattern**: a mechanism that shows a NON-mechanism-driven margin
  (MLB-2 below floor) on an upper-bound slice does NOT earn a fair pilot — the
  W104→W105 erosion + W106 margin-cap logic apply.
* **NEW W111 lesson (positive)**: a $0-NIM re-execution census of committed
  transcripts can kill the majority of a candidate slate before any spend — make
  it the first step of any different-mechanism milestone.

## Do not claim (W111 additions)

* That a genuinely-different mechanism beats same-budget self-consistency on
  contamination-resistant code (M3 did not earn a fair pilot; sub-reflexion).
* That W111 proves no mechanism can ever win (cheap-pilot scale at 70B only; a
  stronger model scale is untested).
* That W111 retires anything, weakens W89/W105, or proves the contamination
  confound (W111 tests a mechanism, not the confound).
* That M3 is worthless (it rescued `/13` reflexion could not — a lateral trade).
* That multi-agent context is "solved".

## Carry-forwards

* **Added (T):** `W111-T-RESISTANT-FAILURE-IS-SEMANTIC-HIDDEN-TEST-COUPLING`,
  `W111-T-EXECUTOR-GROUNDED-PATCHER-V1-SHIPS`.
* **Added (L):** `W111-L-NO-DIFFERENT-MECHANISM-BEATS-A1-ON-RESISTANT-CODE-AT-70B-CHEAP-CAP`,
  `W111-L-M3-PATCHER-SUB-REFLEXION-ON-RESISTANT-HARD-CORE-CAP`.
* **Bounded-claim fallback REGISTERED** (RUNBOOK_W111 § 6) — the bounded
  two-retirement contamination-EXPOSED-HumanEval-family-at-70B claim is the
  honest code ceiling.
* **Retired (research retirements):** NONE. W89 + W105 remain the only two
  confirmed multi-seed same-budget multi-agent superiority retirements.

## Anchors

`docs/RUNBOOK_W111.md`; `docs/RESULTS_W111_M3_PATCHER_PROBE_70B_V1.md`;
`docs/RESULTS_W111_MILESTONE_SUMMARY_V1.md`;
`docs/CONTAMINATION_CONTROL_FRAMING_W111_V1.md`;
`results/w111/mechanism_mining/w110_bcb_failure_census.json`;
`results/w111/m3_probe/.../patcher_bench_report.json`.
