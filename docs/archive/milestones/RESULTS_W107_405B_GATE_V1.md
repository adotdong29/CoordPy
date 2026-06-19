# W107 — 405B reachability gate verdict (V1)

> **2026-05-28.  Lane α of W107.  The 405B reachability gate was
> re-probed FIRST (the only step that spends a NIM call in W107-α,
> and it is a sub-second free side-probe), AFTER `docs/RUNBOOK_W107.md`
> was locked.  Result: HTTP 404 — the FOURTH consecutive 404.  GATE
> = CLOSED.  Lane α is closed for W107; Lane β is the main empirical
> lane.  No 405B cheap pilot was earned or launched.  $0 expensive
> NIM.**

## The gate decision

| Field | Value |
|---|---|
| Target | `meta/llama-3.1-405b-instruct` |
| Endpoint | `https://integrate.api.nvidia.com/v1/chat/completions` |
| Probe status | `http_error` |
| HTTP status | **404 (Not Found)** |
| Wall time | 183 ms |
| Probe timestamp | 2026-05-28T18:03:35Z |
| Probe artifact | `results/w107/405b_reachability_probe/probe_20260528T180335Z/probe.json` |
| **Gate** | **CLOSED** |
| Lane | `beta_main` |
| Gate decision CID | `332d4ef983313f7faf724c7d8b2ad96e8e5964a125dfd30d177bc49ba6b2111e` |
| Gate artifact | `results/w107/405b_reachability_probe/gate_decision.json` |

## Consecutive 404 history

`meta/llama-3.1-405b-instruct` has now returned HTTP 404 on NIM at
**four consecutive milestones**:

| Milestone | Probe date | Result |
|---|---|---|
| W104 | 2026-05-26 | HTTP 404 |
| W105 | 2026-05-27 | HTTP 404 |
| W106 | 2026-05-28 (17:33Z) | HTTP 404 |
| **W107** | **2026-05-28 (18:03Z)** | **HTTP 404** |

## What this means

* The cross-scale-UP path (the genuine single-class → cross-scale
  strengthening of the W105 HumanEval+ retirement) remains blocked.
* `W104-L-HUMANEVAL-PLUS-CROSS-SCALE-UP-PRIMARY-TARGET-405B-UNREACHABLE-ON-NIM-CAP`
  stands and is refreshed with the W107 probe timestamp.
* Per `docs/RUNBOOK_W107.md` § 1–2, the gate decision routes W107
  to **Lane β** as the main empirical lane: the NIM-free preflight
  for the next code battlefield (LiveCodeBench primary / APPS
  backup) under `COO-9`.
* No matrix was widened to include 405B.  The § 3 cheap-pilot rule
  was NOT exercised (it requires GATE = OPEN).

## Discipline note

The gate was decided by re-running the probe, recording the result
sharply, and branching — exactly as the W106 § 7 contract
pre-committed.  W107-α did NOT pretend the cross-scale-UP path was
live for a fourth time; it recorded the 404 and moved to β.  This is
the same "decide honestly and cheaply, don't spend on hope" discipline
that produced the W106 margin-cap NO-GO.

## Anchors

* `docs/RUNBOOK_W107.md` § 1–3 — α/β branch logic + gate rule +
  (un-exercised) cheap-pilot rule.
* `scripts/run_w105_405b_reachability_probe.py` — the reused probe.
* `scripts/run_w107_405b_reachability_gate.py` — the gate decision
  recorder.
* `results/w107/405b_reachability_probe/` — probe + gate artifacts.
