## W106 margin-cap dispatch decision (schema `coordpy.margin_cap_dispatch_v1.v1`)

* failed class: `meta/llama-3.1-70b-instruct` (`FAIL_MARGIN`; 5/6 bars)
* retired class (bounded-claim anchor): `meta/llama-3.3-70b-instruct`
* failed-class signature: margin +2.33 pp; MLB-2 50.54 %; A1 max 87.00 %; per-seed majority 3/3; executor clean True; audit True

### GATE 1 — entitlement

* entitled next step: `multi_seed_cheap_confirmation_at_class`
* entitles a cheap confirmation: **YES** (NIM ceiling ~990 calls)

### GATE 2 — verdict-changing power

| sub-gate | PASS |
|---|---|
| 2a fair battlefield (not rescue-concentrated) | NO |
| 2b no authoritative fair result already exists | NO |
| 2c fixable confound (not clean magnitude miss) | NO |
| **GATE 2 overall** | **NO** |

### Decision — **NO_GO**

GATE 2a FAIL (proposed slice is 'rescue_concentrated', not a fair battlefield — rescue-concentrated is an upper bound, the W102 anti-pattern); GATE 2b FAIL (an authoritative fair broad-slice multi-seed Phase 3 verdict already exists for this class — a cheaper re-run cannot overturn it); GATE 2c FAIL (clean true magnitude miss: executor clean, audit passes, per-seed majority positive, MLB-2 healthy — no confound to fix) ⇒ accept the bounded single-class claim on meta/llama-3.3-70b-instruct; $0 NIM.

Carry-forward registered: `W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-CONFIRMATION-NOT-EARNED-CAP`
