# RESULTS — W122 Lane β: ICPC different-mechanism (M3) signal audit → KILL NIM-free

Milestone W122 / `COO-9`. `ultracode` OFF. NIM-free ($0). This is the same-family
different-mechanism probe the W122 brief made mandatory: *on the SAME matched ICPC family,
does the strongest non-reflexion mechanism earn any real shot at beating A1 where reflexion
failed?* Answer: **No — and not because the mechanism is weak, but because the ICPC grading
regime structurally denies it its differentiator.** The lane is killed honestly, NIM-free.

## The mechanism under test

M3 = the executor-grounded structured-failure patcher (`coordpy.executor_grounded_patcher_v1`,
W111). Its load-bearing edge over plain reflexion (W111 § 2.3) is the **explicit
expected/actual contract extracted from the FAILING test**: on BigCodeBench the hidden
`unittest` oracle prints `AssertionError: <actual> != <expected>`, so M3 can condition the
patch on the exact value the hidden test required. That is what made it "genuinely
different", not a prompt variant.

## The audit (`audit_icpc_mechanism_signal_v1`, over the real W120+W121 sidecars)

`scripts/run_w122_mechanism_signal_audit_v1.py` re-read the full 660-call reflexion-call
sidecars (W120 resistant + W121 exposed), classified every failing reflexion turn by the
PUBLIC signal the model actually saw, and applied the pre-committed earn rule
(`docs/RUNBOOK_W122.md` § 4):

| signal class | count | what it means for M3 |
|---|---|---|
| total reflexion turns | 240 | (120 resistant + 120 exposed) |
| `PUBLIC_SAMPLE_WRONG` | 152 (63.3 %) | expected value shown — but reflexion ALREADY feeds it |
| `RUNTIME_TRACEBACK` | 27 (11.3 %) | stderr tail shown — reflexion ALREADY shows it |
| `NO_SIGNAL` | 43 (17.9 %) | rejected, no actionable signal for anyone |
| `HIDDEN_ONLY` | 10 (4.2 %) | samples pass, hidden fails — expected is SECRET |
| `TIMEOUT` | 8 (3.3 %) | TLE — no expected/actual contract possible |

* `grader_reveals_hidden_expected = False` (official ICPC = SECRET token-diff oracle;
  anti-cheat returns only "wrong answer on a hidden case", never the value).
* `m3_exclusive_signal_fraction = 0.000` < floor 0.33.

**Verdict: `KILL_M3_LANE_NIM_FREE`. $0 NIM.**

## Why this is the honest aggressive answer (not defeat-by-default)

* M3's differentiator is the hidden expected/actual contract. On ICPC the hidden expected
  is **secret** (it must be — revealing it would be oracle leakage / cheating). So on every
  `HIDDEN_ONLY` failure (the only place M3 could add value over reflexion), M3 gets exactly
  what reflexion got: "rejected, no value". Its exclusive signal is **structurally zero**.
* The expected values that ARE visible (public samples, 63 %) are **already** in reflexion's
  feedback. M3 holds no contract reflexion lacks ⇒ on ICPC it reduces to a reflexion
  prompt-variant.
* And M3 was already **sub-reflexion** (12.5 % rescue < reflexion's 25 % < the 33 % floor)
  on BigCodeBench, where it had its FULL unittest edge (W111). Strictly signal-poorer here,
  it cannot clear the floor.

⇒ Running M3 on ICPC would be spend without verdict-changing power — refused by the
pre-committed earn rule. The same-family ceiling is therefore **mechanism-robust, not merely
reflexion-specific**: it is a property of same-budget multi-call mechanisms against
secret-graded contest difficulty, not of the reflexion prompt shape.

## Falsifiability

The audit flips to `BUILD_M3_PROBE` exactly when (a) the grading regime reveals the hidden
expected AND (b) ≥33 % of failing turns are `HIDDEN_ONLY` (so M3 holds an exclusive
contract on enough failures). Both conditions are tested
(`tests/test_w122_paired_seed_closure_v1.py::test_audit_falsifiability_build_when_regime_reveals_and_enough_hidden`).
On official ICPC neither holds ⇒ the kill is machine-checkable, not a judgment call.

Artifacts: `results/w122/mechanism_audit/mechanism_audit_verdict.json`;
`coordpy/coordpy_icpc_paired_seed_closure_v1.py` (`audit_icpc_mechanism_signal_v1`,
`classify_reflexion_turn_v1`); `scripts/run_w122_mechanism_signal_audit_v1.py`.
