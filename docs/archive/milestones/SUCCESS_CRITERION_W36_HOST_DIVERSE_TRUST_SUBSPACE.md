# Success criterion - CoordPy SDK v3.37 / W36
# Host-diverse trust-subspace guard + manifest-v6

**Pre-committed before final W36 verdict/release decision**:
2026-05-02.
**Target**: SDK v3.37 / W36 family.
**Position relative to W35**: W35 proved that a controller-verified
trust-subspace proxy can convert some W34 abstentions into correct
reroutes.  It did not solve the harder case where a dense projection
is supported by a single host, a spoofed host, or co-located model
state that moves wrong together.

W36 must therefore be judged as a blocker-removal milestone, not as
another local benchmark win.  The target blocker is the live/host
trust boundary of dense control.

---

## 1. Mechanism bar

W36 must implement a real method beyond W35:

1. Wrap W35 with a **host-diverse trust-subspace guard**.
2. Require projected dense-control support to be independently
   attested by at least `min_distinct_hosts` registered healthy hosts.
3. Allow reroute only when host-diverse support is high-score and
   separated by margin.
4. Reject, abstain, or preserve W35 behavior when live attestation is
   missing, host diversity is insufficient, or the host registry is
   unverifiable.
5. Seal the new state in a content-addressed manifest-v6 envelope.
6. Keep the native-latent gap explicit: no transformer hidden-state,
   KV-cache, attention-weight, or embedding-table access.

The permitted headline is **host-diverse audited trust-subspace
guard**, not native latent transfer.

---

## 2. Hard gates

| Gate | Description | Pass iff |
| --- | --- | --- |
| **H1** | Real W36 mechanism beyond W35 | New orchestrator, registry, envelope, verifier, and host-diverse selector are implemented; W36 can make a routing decision W35 cannot make |
| **H2** | Trust boundary | `verify_host_diverse_ratification` enumerates 14 disjoint W36 failure modes; tests exercise every mode plus clean-envelope sanity |
| **H3** | Trivial-W36 old-regime preservation | With host diversity disabled and manifest-v6 disabled, W36 reduces to W35 byte-for-byte across 5 seeds |
| **H4** | Load-bearing host-diverse recovery | On R-83-HOST-DIVERSE-RECOVER, W36 improves correctness over W35 by at least +0.25, restores trust precision to 1.000, and adds <= 1 visible token/cell |
| **H5** | Spoofed-host falsifier | On R-83-HOST-SPOOFED-CONSENSUS, W36 must not claim correctness gain; it may only improve trust by abstaining on unsafe ratifications |
| **H6** | No-live-attestation falsifier | On R-83-NO-LIVE-ATTESTATION, W36 must show that forcing host diversity without live attestations is unsafe for correctness |
| **H7** | Old explicit capsule line preserved | Focused W33, W34, and W35 regression slices stay green; W36 composes W21/W33/W34/W35 rather than bypassing them |
| **H8** | Dense-control/geometry line strengthened | W36 transfers controller-verified structured state at density >= 10,000 bits per visible W36 token on the load-bearing regime |
| **H9** | Live/two-host evidence | Re-check usable hosts; if Mac 2 is unreachable, record exact fallback and run the strongest bounded live probe practical |
| **H10** | Release-readiness clause | Versioning, changelog, success bar, results note, theorem registry, README/START_HERE/master plan/paper markers updated only if H1-H8 pass and the stable runtime remains unchanged |
| **H11** | Focused validation green | W36 unit tests, W35/W34/W33 regression slices, public API import, compile checks, and diff hygiene pass |

**Hard-gate aggregate**:

- **Strong success** = 10-11 gates pass, with no trust/audit weakening.
- **Partial success** = 8-9 gates pass, with exact blockers carried
  forward.
- **Failure** = <= 7 gates pass, any verifier weakening, or any
  unbounded native-latent/live claim.

---

## 3. Soft gates

| Gate | Description | Target |
| --- | --- | --- |
| **S1** | Stronger live disagreement evidence | Bounded live probe observes cross-host disagreement with gold-correlated winner, or records honestly-null |
| **S2** | Mac 2 | `192.168.12.248:11434/api/tags` succeeds, or timeout evidence is recorded |
| **S3** | Stable-vs-experimental boundary | W36 remains under `__experimental__`; stable runtime contract unchanged |
| **S4** | Theory | Add one conditional sufficiency claim, one limitation/falsifier claim, and one native-latent gap claim |
| **S5** | Paper/master-plan synthesis | The old explicit capsule line and dense-control/geometry line read as a single stack with a host-trust boundary |

Soft gates cannot compensate for failed trust/audit hard gates.

---

## 4. Named falsifiers

- **W36-L-TRIVIAL-PASSTHROUGH**: disabled W36 + disabled manifest-v6
  equals W35 byte-for-byte.
- **W36-L-SINGLE-HOST-PROJECTION**: a high-score projection supported
  by only one host is unsafe to ratify as host-diverse.
- **W36-L-HOST-SPOOFED-CONSENSUS**: when every dense basis direction
  agrees on the wrong answer from the same unhealthy/spoofed host,
  W36 cannot recover correctness; the safe behavior is abstention.
- **W36-L-NO-LIVE-ATTESTATION**: if host diversity is required but
  live attestations are absent, W36 may preserve trust precision by
  abstaining, but it can destroy correctness relative to W35.
- **W36-L-NATIVE-LATENT-GAP**: if a regime requires transformer-
  internal evidence not visible through probes, EWMA, response
  signatures, host attestations, and registries, W36 is insufficient.

---

## 5. Claim boundary

W36 may claim:

- an audited host-diverse trust-subspace guard;
- measured W36-over-W35 correctness gain on a regime where W35
  ratifies unsafe single-host dense support;
- measured trust-precision gain on a spoofed-host falsifier through
  abstention, not recovery;
- preserved W35 behavior on the trivial path;
- stronger two-reachable-host live evidence if the bounded live probe
  succeeds.

W36 may not claim:

- native latent transfer;
- transformer-internal trust subspace;
- KV-cache transplant;
- hidden-state projection;
- true three-host evidence while Mac 2 is unavailable;
- solved context for multi-agent teams;
- release readiness independent of blocker removal.
