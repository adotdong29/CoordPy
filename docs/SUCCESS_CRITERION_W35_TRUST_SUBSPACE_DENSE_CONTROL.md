# Success criterion — Wevra SDK v3.36 / W35
# Trust-subspace dense-control proxy + basis-history projection + manifest-v5

**Pre-committed before final W35 verdict/release decision**:
2026-05-02.
**Target**: SDK v3.36 / W35 family.
**Position relative to W34**: W34 was a strong blocker-clearing
milestone: it replaced single-anchor trust fragility with
multi-anchor consensus + NO_CONSENSUS abstention, closed both named
W33 live-infra follow-ups, and produced first measured live
cross-host gold-correlated disagreement evidence.  The remaining
blockers are now concentrated:

- **W33-C-NATIVE-LATENT remains open**.  The best line is still
  capsule-layer audited proxy, not transformer-internal hidden-state
  projection.
- **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE remains open on the
  stronger magnitude axis**.  W34 showed real disagreement, not a
  systematic magnitude survey.
- **W34-C-MULTI-HOST remains hardware-bounded**.  Mac 2
  (`192.168.12.248`) has not joined the live topology.
- **Old explicit capsule line + new dense-control/geometry line need
  one synthesis mechanism**, not parallel stories.

W35 must therefore be judged by whether it removes a real part of
that blocker set, not by whether it adds another local audit layer.

---

## 1. Mechanism bar

W35 must implement a real method beyond W34:

1. Wrap the W34 live-aware multi-anchor path with a
   **trust-subspace dense-control proxy**.
2. Build one controller-verified basis entry per oracle from:
   W21 probe top_sets, W33 EWMA trust, W34 response-feature/live
   attestation state, top-set stability, and host health.
3. Use the basis only when it can safely convert a W34
   NO_CONSENSUS abstention into a verified reroute.
4. Preserve W34 abstention when the basis is too short, too unstable,
   insufficiently separated, or unverifiable.
5. Seal the new state in a content-addressed manifest-v5 envelope.
6. Keep the native-latent gap explicit: no transformer KV cache,
   hidden-state, embedding-table, or attention-weight access.

The permitted headline is **audited trust-subspace proxy**, not
native latent transfer.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W35 mechanism beyond W34 | New W35 orchestrator + registry + envelope + verifier + projection selector are implemented; W35 can make a routing decision W34 cannot make |
| **H2** | Trust boundary | `verify_trust_subspace_dense_ratification` enumerates 14 disjoint W35 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W35 old-regime preservation | With trust-subspace disabled + manifest-v5 disabled, W35 reduces to W34 byte-for-byte across 5 seeds |
| **H4** | Load-bearing benchmark | On R-82-TRUST-SUBSPACE-SHIFT, W35 improves correctness over W34 by at least +0.25 with no trust-precision regression and <= 1 visible-token overhead/cell |
| **H5** | Old explicit capsule line strengthened | W35 must compose W21/W33/W34 signals rather than bypass them; the result must be explainable as W21 explicit oracle probes + W33 trust state + W34 multi-anchor abstention + W35 basis-history projection |
| **H6** | Dense-control/geometry line strengthened | W35 must transfer more structured controller state per visible token than W34 while keeping that state verifiable |
| **H7** | At least one falsifier | A named regime where W35 does not help or is unsafe must be implemented and measured |
| **H8** | Live/two-host evidence | Re-check usable hosts; if Mac 2 is unreachable, record the exact fallback and run the strongest bounded live probe that is practical |
| **H9** | Release-readiness clause | SDK_VERSION/pyproject/version/changelog/experimental exports updated only if H1-H7 pass and stable runtime remains unchanged |
| **H10** | Focused regression green | W35 unit tests, Phase82 benchmark driver, W33/W34 regression slices, and import/compile checks pass |

**Hard-gate aggregate**:

- **Strong success** = 10/10 gates pass.
- **Partial success** = 8-9/10 gates pass, with explicit open
  blockers carried forward.
- **Failure** = <= 7/10 gates pass, or any trust/audit weakening that
  is not explicitly bounded.

---

## 3. Soft gates

| Gate | Description | Target |
| --- | --- | --- |
| **S1** | Stronger live trust magnitude | Bounded live probe observes at least one cross-host disagreement with a gold-correlated winner, or records honestly-null |
| **S2** | Mac 2 | `192.168.12.248:11434/api/tags` succeeds, or the unreachable state is recorded with concrete timeout evidence |
| **S3** | Token/context efficiency | W35 carries >= 10,000 structured bits per visible W35 token on the load-bearing R-82 bench |
| **S4** | Stable-vs-experimental boundary | W35 remains under `__experimental__`; no stable runtime contract changes |
| **S5** | Theory | Add at least one conditional sufficiency claim and one limitation/falsifier claim |
| **S6** | Paper/master-plan synthesis | Old explicit capsule and dense-control/geometry lines read as one research arc |

Soft gates are not allowed to compensate for failed trust/audit hard
gates.

---

## 4. Named falsifiers

- **W35-L-TRIVIAL-PASSTHROUGH**: when the trust-subspace layer is
  disabled and manifest-v5 is disabled, W35 equals W34 byte-for-byte.
- **W35-L-NO-ANCHOR-DISAGREEMENT**: when W34 already has consensus
  and no anchor disagreement exists, W35 adds no correctness benefit.
- **W35-L-SHORT-OR-UNSTABLE-BASIS**: when the basis history is too
  short or below threshold/margin, W35 must not reroute.
- **W35-L-ALL-BASIS-COMPROMISED**: when every registered basis
  direction moves together to the same wrong answer, W35 cannot
  recover at the capsule layer.
- **W35-L-NATIVE-LATENT-GAP**: if a regime requires hidden-state
  evidence not visible through oracle probes, EWMA, response-feature
  signatures, and host health, W35 must be reported as insufficient.

---

## 5. Claim boundary

W35 may claim:

- an audited trust-subspace dense-control proxy;
- measured correctness gain over W34 on a regime where W34 abstains;
- preserved trust precision and old explicit capsule behavior;
- improved structured-state density per visible token;
- a sharper limitation showing where capsule-layer dense control ends.

W35 may not claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- solved context for multi-agent teams;
- release readiness independent of blocker removal.
