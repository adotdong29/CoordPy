# RESULTS — W86 / P2 #38 Byzantine Fault Tolerance V1

> **Status:** TRULY CLOSED 2026-05-20 (W86 P2 line).
> Canonical evidence: `results/w86/bft/<TS>/bft_v1_suite_report.json`
> with `suite_cid = 3ff0e1797c2b7c7c…`; offline-re-verifiable via
> `scripts/verify_w86_bft_v1_audit_chain.py`.

## What lands

`coordpy.byzantine_fault_tolerance_v1` ships a real PBFT-style
3-phase Byzantine fault-tolerant consensus on top of the
W82+W83 distributed line:

| Component                         | Module                              |
|-----------------------------------|-------------------------------------|
| `BFTReplicaKey` (Ed25519)         | `byzantine_fault_tolerance_v1.py`   |
| `BFTReplicaIdentity` (public key) | "                                   |
| `BFTMembershipV1`                 | "                                   |
| `BFTPhaseMessageV1` (signed)      | "                                   |
| `ByzantineWitnessV1`              | "                                   |
| `ByzantineEquivocationEvidenceV1` | "                                   |
| `BFTConsensusOutcomeV1`           | "                                   |
| `run_pbft_consensus_round_v1`     | "                                   |
| collusion bench                   | `run_collusion_bench_at_byzantine_bound_v1` |
| refuse-to-commit bench            | `run_refuse_to_commit_bench_above_byzantine_bound_v1` |
| equivocation bench                | `run_equivocation_detection_bench_v1` |
| full suite                        | `run_bft_v1_full_suite`             |
| bench driver                      | `scripts/run_w86_bft_v1_bench.py`   |
| audit verifier                    | `scripts/verify_w86_bft_v1_audit_chain.py` |
| safety + liveness proof           | `papers/proofs/w86_proof_byzantine_v1.md` |

## DoD bullets ↔ evidence

| DoD bullet                                       | Status | Where to look |
|--------------------------------------------------|--------|---------------|
| `ByzantineWitnessV1` with cryptographic sigs over value | ✓ | `byzantine_fault_tolerance_v1.ByzantineWitnessV1` + `tests/test_w86_byzantine_fault_tolerance_v1.py::test_byzantine_witness_signs_value_cid_and_verifies` |
| PBFT-style 3-phase protocol                      | ✓ | `run_pbft_consensus_round_v1` (pre_prepare → prepare → commit) |
| Collusion at f = ⌊(n−1)/3⌋ commits within bound  | ✓ | `bft_v1_suite_report.json` → collusion bench, `committed_value = μ = 1.0`, `committed_error = 0.0 ≤ B = 0.0` |
| f > (n−1)/3 refuses to commit                    | ✓ | refuse bench, `committed = False`, `verdict = refused_quorum_not_reached` |
| Equivocation detection → independently verifiable evidence | ✓ | equivocation bench, `evidence_count ≥ 1`, `independently_verifiable = True` |
| Safety + liveness proofs                         | ✓ | `papers/proofs/w86_proof_byzantine_v1.md` |
| `RESULTS_<MILESTONE>_BYZANTINE_V1.md`            | ✓ | this file |

## Anti-cheat coverage

| Anti-cheat clause                                | Honest answer |
|--------------------------------------------------|--------------|
| Do not call the W81 corruption penalty "Byzantine" | The W83 integrity-trust line is preserved unchanged; W86 BFT is a strictly new module. |
| Do not rely on a trusted third party for equivocation | `ByzantineEquivocationEvidenceV1.independently_verify` recomputes from membership public keys only — no trusted oracle. |
| Do not weaken the safety bound past 3f + 1       | `BFTMembershipV1.f_byzantine_bound = (n - 1) // 3` enforced; `quorum_size = 2f + 1`. |
| Do not rely on a synchronous clock               | Theorem 4 (liveness) explicitly assumes partial synchrony; safety holds without any synchrony assumption. |
| Do not "prove" safety informally                 | `papers/proofs/w86_proof_byzantine_v1.md` reuses the classical PBFT safety argument (Castro & Liskov 1999 §5.2) verbatim under our message format. |
| Do not skip the f > (n−1)/3 safety test          | refuse bench exercises exactly this; verifier insists on `committed = False`. |

## Measured numbers (canonical run, seed 86 038)

| Bench                                  | n   | f_target | f_bound | quorum | verdict                       | committed_value | committed_error | evidence |
|----------------------------------------|----:|---------:|--------:|-------:|-------------------------------|----------------:|----------------:|---------:|
| collusion_at_byzantine_bound_v1        |  7  |  2       |  2      |  5     | committed                     | 1.0             | 0.0             | 0        |
| refuse_to_commit_above_byzantine_bound |  4  |  2       |  1      |  3     | refused_quorum_not_reached    | None            | None            | 0        |
| equivocation_detection_v1              |  4  |  1       |  1      |  3     | refused_equivocation          | None            | None            | 1        |

* `committed_error = 0.0` means the colluding `f` byzantines did **not**
  shift the committed value at all — honest 2f+1 quorum strictly
  determines the outcome.
* `refused_quorum_not_reached` at f = f_bound + 1: byzantines
  cannot reach quorum on `μ + δ`, honest cannot reach quorum on `μ`
  (they're outnumbered).  Safety holds; liveness is sacrificed
  *honestly*.
* `evidence_count = 1` at the equivocation bench: one
  `ByzantineEquivocationEvidenceV1` capsule is produced; its
  `conclusively_byzantine = True` under `independently_verify`.

## Reproducibility

Two consecutive runs of the driver at the default seed
produce byte-identical reports:

```
results/w86/bft/w86_bft_20260520T222914Z/bft_v1_suite_report.json
results/w86/bft/w86_bft_20260520T222923Z/bft_v1_suite_report.json
suite_cid (both): 3ff0e1797c2b7c7c3c21cb1e5d866e01092131a3e19ac38aabaaa17a98c331c0
```

Verifier output (final lines):

```
INFO collusion: n=7, f_target=2, committed_value=1.0, committed_error=0.0
PASS collusion: committed μ exactly
INFO refuse: n=4, f_target=2, f_bound=1, verdict=refused_quorum_not_reached
PASS refuse: did not commit (verdict=refused_quorum_not_reached)
INFO equivocation: n=4, evidence_count=1, independently_verifiable=True, committed=False
PASS equivocation: evidence produced and round refused to commit
PASS suite.closed=True

OVERALL: PASS
```

## Honest scope

* `W86-L-BYZANTINE-V1-RESEARCH-ONLY-CAP` — explicit-import only;
  not exposed via `coordpy.__init__`.
* `W86-L-BYZANTINE-V1-IN-PROCESS-CAP` — V1 protocol runs in-process;
  the W86 multi-host substrate (#29 closed) carries it over the
  wire in V2.
* `W86-L-BYZANTINE-V1-ED25519-CAP` — V1 uses Ed25519 via
  `cryptography`; threshold sigs (BLS / Shamir) are V2.
* `W86-L-BYZANTINE-V1-STATIC-MEMBERSHIP-CAP` — V1 quorum against
  static membership; dynamic membership is V3.
* `W86-L-BYZANTINE-V1-PARTIAL-SYNCHRONY-CAP` — liveness (Theorem 4)
  requires partial synchrony; safety (Theorems 1–3) does not.
* `W86-L-BYZANTINE-V1-NO-VIEW-CHANGE-CAP` — V1 has no
  view-change protocol; a Byzantine primary breaks liveness
  (does not break safety). View-change is V2.
* `W86-L-BYZANTINE-V1-N4-MIN-CAP` — `n ≥ 4` enforced. Default
  bench shape: collusion at `n=7`, refuse + equivocation at `n=4`.

These limitations are tracked as W86-L-BYZANTINE-V1-* rows in
`docs/THEOREM_REGISTRY.md` and `docs/HOW_NOT_TO_OVERSTATE.md`.
