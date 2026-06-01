# Frontier-relevance audit — W125 (controller-native code mechanism)

Classifies the W125 additions and re-affirms the standing arsenal/baseline/dead/anti-pattern
columns. Supplements the W123/W121 audits; all prior classifications remain in force.

## Active-frontier arsenal (NEW in W125)

* **`coordpy.controller_native_code_mechanism_v1`** — the controller-native code mechanism:
  C1 role-specialized planner/controller, C2 router-selected multi-candidate, **C3
  tool-substrate audited repair loop (LEAD)**. The FIRST module to wire the hosted-controller
  stack to the audited tool-call substrate (graphify-confirmed they were 6 hops apart with no
  semantic edge). Genuinely controller-native (structural fake-different test passes) and
  contract-clean (4/4 NIM-free checks). **Active frontier**: this is the executable controller
  mechanism a fresh hosted pilot would run IF the precursor earns it (W126 operator-greenlit
  branch), and it is the pilot-ready promotion of the W124 M6 contract.
* **`AuditedGraderPlaneV1`** — official ICPC grader calls as first-class audited
  `ToolCallSchemaV1`/`ToolResultSchemaV1` events on a `ToolAuditChainV1`, with a never-reads-secret
  guard. Active frontier (reusable for any future tool-substrate code pilot).
* **The $0 resistant headroom replay** (`headroom_probe` + `run_w125_lane_beta_*`) — a reusable
  pattern: re-route already-paid model generations through a controller policy and re-grade with
  the official oracle, at $0, to decide whether a fresh pilot is worth NIM. Active frontier
  (the cheapest honest "is a controller mechanism worth buying?" precursor).

## Promoted from the hosted/controller arsenal (now WIRED to code)

`hosted_router_controller_v12`, `hosted_logprob_router_v12`, `hosted_cache_aware_planner_v12`,
`tool_call_substrate_v1`, `executor_grounded_patcher_v1` — previously W79/W84/W111-era leaves with
no code-task consumer; W125 composes them onto the official ICPC code path. They move from
"unused arsenal" to "active-frontier code-mechanism components".

## Baseline-only (unchanged)

The W120 reflexion bench (`icpc_reflexion_bench_v1`) A0/A1/B arms remain the same-budget
baselines; reflexion B is the FAKE_DIFFERENT null mechanism the structural test discriminates
against. `bounded_window_baseline_v{1,2,3}` remain falsifier targets.

## Dead directions / capped (W125 additions)

* **$0 controller re-routing over the existing resistant pool** — capped by
  `W125-L-RESISTANT-GENERATION-CAP`: the pool reaches 8/30, A1 captures 7, blind headroom 0.
  A controller cannot beat A1 by re-routing what Maverick already produced; only NEW trajectories
  could, which the corpus cannot supply.
* **Public-sample-guided selection on resistant ICPC** — capped by
  `W125-T-RESISTANT-PUBLIC-SAMPLE-SIGNAL-NON-DISCRIMINATING`: 10 generations pass all samples yet
  fail secret, so the only blind in-loop signal is non-discriminating.

## Anti-patterns (REMAIN explicit anti-patterns; W125 reinforces)

Bounded-context / compaction / generic summarization / "cram less, truncate better" remain
anti-patterns, NOT the frontier path. W125 explicitly did NOT drift into them — it built a real
controller-native mechanism with a different control flow, retry/routing policy, and tool-use
plane. Reflexion-with-extra-words (the C0 negative control) is classified FAKE_DIFFERENT and
killed, reinforcing that relabeling reflexion is not a mechanism.

## Do-not-claim (see `docs/HOW_NOT_TO_OVERSTATE.md` W125 section)

The controller stack is real, not fake-different; the resistant field is generation-capped for
$0; a fresh pilot is an untested hope, deliberately not bought; W89+W105 stand as the only two
retirements; multi-agent context is not "solved".
