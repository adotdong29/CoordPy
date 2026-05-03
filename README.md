# CoordPy

**A Python-first SDK and CLI for building auditable AI agent teams
with structured, content-addressed context.** Each piece of context
that crosses a role boundary, a layer boundary, or a run boundary is
a typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-stamped **capsule** — never a raw prompt string. One
`RunSpec` in, one reproducible `RunReport` out, and that report is
the root of a sealed capsule graph you can audit, replay, and trust.

> **Status — SDK v3.43, the final release of the CoordPy SDK v3.4x
> line (May 2026).** This is the first public release of CoordPy.
> See [Final release scope](#final-release-scope-v343) below for
> what is stable, what is experimental, and what is explicitly out
> of scope.

## What CoordPy is

CoordPy is the shipped product surface from the **Context Zero**
research programme. It gives you four things you would otherwise
build by hand on every project that wires multiple LLM agents
together:

* **A capsule contract.** Every cross-boundary artefact (prompt,
  response, parse outcome, role handoff, run report) is a typed
  object with a content-derived ID, declared parents, an explicit
  budget, and a closed lifecycle. The contract is checked at
  runtime; tampering is detectable.
* **A capsule-native runtime.** `coordpy.run(RunSpec(...))` produces a
  `RunReport` whose root is a sealed capsule DAG, written to disk
  alongside a `provenance.json` reproducibility manifest and a
  detached `meta_manifest.json` witness.
* **A team-coordination surface.** Agents exchange `TEAM_HANDOFF` /
  `ROLE_VIEW` / `TEAM_DECISION` capsules with a mechanically-checked
  T-1..T-7 lifecycle audit, so multi-agent runs are reproducible,
  bounded-context, and audit-friendly.
* **A research-grade evaluation harness.** Reproducible profiles
  (`local_smoke`, `bundled_57`, `aspen_mac1_coder`,
  `aspen_mac2_frontier`, `public_jsonl`, …), a `coordpy-ci` gate that
  consumes the report, and a `coordpy-capsule verify` CLI that
  re-hashes the on-disk capsule chain end-to-end.

## Why CoordPy

Most multi-agent stacks treat context as text — prompts, JSON
records, ad-hoc tool traces. That works until something breaks, and
then the failure is a vague "the model was confused." CoordPy treats
context as **objects**: typed, content-addressed, lifecycle-bounded.
The result is a runtime where you can ask "what evidence did the
team actually have?", get a sealed DAG, and re-verify it from the
bytes on disk. Reproducibility, auditability, and a clean integration
boundary for downstream tools come along for free.

## What makes it different

CoordPy is **not** another agent-orchestration framework. It is a
runtime contract — small, stable, opinionated — under which
existing agent code becomes auditable. Capsules are the load-bearing
abstraction; everything else (CLIs, profiles, the team harness, the
research-grade trust ladder) hangs off them. The individual
primitives (content addressing, hash-chained logs, typed claim
kinds, capability-style typed references) are inherited from Git /
Merkle DAGs / IPFS / actor systems / session types. What CoordPy
contributes is the *unification* — one contract, implemented
end-to-end in a runnable SDK.

## Install

```bash
pip install -e .                 # editable install from a clone
# or, once published to PyPI:
# pip install coordpy
# pipx install coordpy              # if you only want the CLIs
```

Only required dependency is NumPy. The optional LLM-agent demo
talks HTTP to a local Ollama instance — no Python binding required.

Optional extras: `coordpy[scientific]`, `coordpy[dl]`, `coordpy[heavy]`,
`coordpy[crypto]`, `coordpy[docker]` (Docker-first sandbox), `coordpy[dev]`.

Installing the package registers four console scripts:

```bash
coordpy --profile local_smoke --out-dir /tmp/coordpy-smoke
coordpy-import   --jsonl /path/to/swe_bench_lite.jsonl --out /tmp/audit.json
coordpy-ci       --report /tmp/coordpy-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule  view   --report /tmp/coordpy-smoke/product_report.json
coordpy-capsule  verify --report /tmp/coordpy-smoke/product_report.json
```

## Quickstart

```python
from vision_mvp.coordpy import RunSpec, run

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/coordpy-smoke"))
assert report["readiness"]["ready"]
assert report["provenance"]["schema"] == "coordpy.provenance.v1"

# Every run ships a sealed capsule graph.
cv = report["capsules"]
assert cv["schema"] == "coordpy.capsule_view.v1"
assert cv["chain_ok"]
print(f"RUN_REPORT CID = {cv['root_cid']}")
print(report["summary_text"])
```

A first real-LLM team is one extra line:

```bash
COORDPY_OLLAMA_URL=http://localhost:11434 \
    coordpy --profile local_smoke --acknowledge-heavy --out-dir /tmp/coordpy-smoke
```

See [`docs/START_HERE.md`](docs/START_HERE.md) for the onboarding
path and [`examples/`](examples/) for short standalone programs.

## Stable vs experimental — at a glance

| Surface | What you get | Stability |
|---|---|---|
| `vision_mvp.coordpy` SDK — `RunSpec`, `run`, `RunReport`, `SweepSpec`, `run_sweep`, `CoordPyConfig`, `profiles`, `ci_gate`, `import_data`, `extensions`, capsule primitives, schema constants | The product / runtime contract | **Stable** (contract-tested, byte-for-byte unchanged across the v3.4x line) |
| Console scripts — `coordpy`, `coordpy-import`, `coordpy-ci`, `coordpy-capsule` | The CLI surface | **Stable v3** |
| Capsule view / provenance / report schemas — `coordpy.capsule_view.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2` | On-disk contracts | **Stable** |
| `vision_mvp.coordpy.__experimental__` — W22..W42 trust-adjudication / multi-agent-coordination ladder, R-69..R-89 benchmark drivers, bounded live cross-host probes | Research surface, included for audit and reproduction | **Experimental** — may move, rename, or be withdrawn as the next programme starts |
| Transformer-internal trust transfer (`W42-C-NATIVE-LATENT`); K+1-host disjoint topology beyond the two-Mac pair (`W42-C-MULTI-HOST`) | Architecture-bound open frontiers | **Out of scope for this release** — see [Out of scope](#out-of-scope-for-this-release) |

The full stability matrix lives further down in
[Stability matrix](#stability-matrix); the canonical, file-by-file
status is in [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
and [`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md).

## The released result, in one paragraph

The v3.43 line closes the **capsule-layer-only research programme**
inside Context Zero. Its strongest internal result is a measured
strict trust-precision recovery on a regime where the prior best
(W41) tied at 0.500: on `R-89-ROLE-INVARIANT-RECOVER`, W42 raises
trust precision from 0.500 to **1.000 across 5/5 seeds**
(`Δ_trust_precision = +0.500`, min = max). This is the first
capsule-native multi-agent-coordination method in the programme
that materially **bounds** `W41-L-COMPOSITE-COLLUSION-CAP` at the
capsule layer via a third orthogonal evidence axis (the
role-handoff invariance axis). W42 is closed-form, deterministic,
zero-parameter, and capsule-layer; it does **not** add a
transformer-internal mechanism, does **not** close
`W42-L-FULL-COMPOSITE-COLLUSION-CAP` (a newly proved-conditional
limitation theorem), and does **not** claim universal solution of
multi-agent context. Live cross-host evidence at temperature 0 on
the two-Mac topology (`localhost` gemma2:9b + `192.168.12.191`
qwen2.5:14b) shows **4/4 paraphrase-invariant gold-correlated
agreement** across K=4 paraphrases of one arithmetic prompt — the
first measured cross-host paraphrase-invariance result in the
programme. Full results note:
[`docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`](docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md);
pre-committed success bar:
[`docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`](docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md);
paper:
[`papers/context_as_objects.md`](papers/context_as_objects.md).

## Final release scope (v3.43)

The SDK v3.43 line is the **final release of the CoordPy SDK v3.4x
research line** -- the **end-of-line for the capsule-layer-only
research programme** in the Context Zero project.  The boundary
between what is stable, what is experimental but included, and
what is explicitly out of scope is now final and frozen for this
release.

### Stable and shipped

* The product/runtime contract: one ``coordpy.RunSpec`` in, one
  reproducible ``RunReport`` out, where the report is the root of
  a sealed capsule graph that can be audited and replayed.  This
  contract is byte-for-byte unchanged from earlier v3.x releases
  and is what users and downstream tools should depend on.
* Public CLIs: ``coordpy``, ``coordpy-import``, ``coordpy-ci``,
  ``coordpy-capsule`` (see ``[project.scripts]`` in
  ``pyproject.toml``).
* Capsule contract types and ``coordpy.run`` /
  ``coordpy.RunReport`` orchestration (W3-7..W3-31 and W3-32..W3-41).
* Public package version: ``vision_mvp.__version__ = 0.5.16`` ==
  ``pyproject.toml`` ``project.version = 0.5.16``;
  ``SDK_VERSION = "coordpy.sdk.v3.43"``.

### Experimental but included

Everything under ``vision_mvp.coordpy.__experimental__`` is
**experimental research surface** included in the release for
audit, reproduction, and downstream research.  This covers the
entire capsule-layer trust-adjudication / multi-agent-coordination
research ladder:

* The W22..W42 capsule-layer research surface (every symbol in
  the cumulative ``__experimental__`` tuple — orchestrators,
  registries, envelopes, verifiers, signature CIDs, manifest
  versions v6 through v12, decision selectors, named decision
  branches, and the 196 cumulative enumerated trust-boundary
  failure modes).
* The R-69..R-89 benchmark family drivers
  (``vision_mvp.experiments.phase69_*`` through
  ``phase89_role_invariant_synthesis``) and the matching unit
  tests under ``vision_mvp/tests/``.
* The bounded live cross-host probes
  (``phase8x_xllm_*`` and ``phase89_xllm_role_invariance_probe``).

These symbols may move, rename, or get withdrawn as the next
research programme starts.  Downstream code that depends on them
should pin against the experimental tuple, not assume API
stability.

### Out of scope for this release

Two open frontiers are explicitly **out of capsule-layer scope**
and are not addressed by the CoordPy SDK v3.4x line.  They are
preserved as named conjectures in
``docs/THEOREM_REGISTRY.md`` so future work has a clean handle on
them, but they are not blockers to the v3.43 final release:

* **``W42-C-NATIVE-LATENT``** — true transformer-internal
  trust-state projection.  Architecture-bound: requires
  hidden-state, KV-cache, attention-weight, or embedding-table
  access.  No mechanism in this repo touches transformer
  internals; the W22..W42 chain is closed-form, deterministic,
  zero-parameter, and capsule-layer.  Closing
  ``W42-C-NATIVE-LATENT`` requires a new architectural substrate
  that is not in this repo.
* **``W42-C-MULTI-HOST``** — K+1-host disjoint topology beyond
  the two-Mac pair (``localhost`` + ``192.168.12.191``).
  Hardware-bound: would let the role-invariance policy registry
  be sourced from a true off-cluster oracle, defeating the
  ``W42-L-FULL-COMPOSITE-COLLUSION-CAP`` attack at the capsule
  layer.  The lab's two-Mac topology is the strongest live
  evidence available in this environment; ``.248`` is gone,
  ``.101`` is Apple TV / AirPlay (``W41-INFRA-1`` carry-forward).

These are next-programme architecture questions, not release
blockers.  Future work addressing them will require new
substrate (transformer-internal access, K+1-host topology, or
both) and is explicitly outside the scope of this repo's v3.43
final release.

### Reproducing the headline result

The forced-verdict result is a strict **+0.500
trust-precision recovery** on R-89-ROLE-INVARIANT-RECOVER across
5/5 seeds (``min = max = +0.500``).  To reproduce:

```
# W42 unit suite (40 tests; ~2 s).
python3 -m pytest vision_mvp/tests/test_phase89_role_invariant_synthesis.py -q

# Focused W22..W42 stack regression (738 tests; ~80 s).
python3 -m pytest vision_mvp/tests/test_phase{69..89}_*.py -q

# R-89 5-seed sweep driver (writes seed-sweep JSON artifacts under
# vision_mvp/experiments/artifacts/phase89/).
python3 -m vision_mvp.experiments.phase89_role_invariant_synthesis
```

The success bar (every hard gate, every named falsifier, the
forced verdict structure) is pre-committed in
``docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md``.  The
results note (every empirical headline + theoretical claim +
hard-gate / soft-gate aggregate + end-of-line declaration) is in
``docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md``.  The
canonical theorem-by-theorem status is in
``docs/THEOREM_REGISTRY.md``; the do-not-overstate rules are in
``docs/HOW_NOT_TO_OVERSTATE.md``.

---

## Detailed milestone history

The sections below preserve the per-milestone notes for the W22..W42
research ladder, in reverse-chronological order, for audit and
research-record purposes. Everything from this point on is
**research surface** — useful if you want to follow exactly how the
released result was built up, but not required reading for using
CoordPy. The active scientific position is in
[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md); the
theorem-by-theorem status is in
[`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md).

**Previous milestone: SDK v3.42 RC2 (May 2026, superseded by v3.43 final).** Integrated
multi-agent context synthesis + manifest-v11 CID + cross-axis
witness CID + producer-axis x trust-axis decision selector (W41
family).  W41 jointly binds the strongest old-line explicit-
capsule trust-adjudication chain (W21..W40) AND the strongest
cross-role / multi-round bundle decoder family (W7..W11) into a
single auditable end-to-end path with one ``manifest-v11``
envelope.  W41-L-COMPOSITE-COLLUSION-CAP names the open wall: when
the adversary controls both axes on the same wrong top_set, W41
cannot recover.  W42 (above) materially BOUNDS this wall via a
third orthogonal evidence axis.

**Previous milestone: SDK v3.41 RC1 (May 2026, superseded by v3.43 final).** Cross-host
response-signature heterogeneity ratification + manifest-v10 CID +
cross-host response-text Jaccard divergence guard (W40 family).
W40 wraps W39's K-of-N mutually-disjoint quorum consensus-reference
adjudication with a **cross-host response-heterogeneity** layer that
operates on an evidence axis ORTHOGONAL to top_set: the per-member
response **text bytes** themselves.  Even if K colluders coordinate
their declared top_set (the W39 full-quorum-collusion attack;
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` fires), naturally-
independent K probes should produce heterogeneous response text
bytes.  When the K member probes' mean pairwise Jaccard divergence
over canonical sorted token bags falls strictly below
``response_text_diversity_min``, W40 abstains via
``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` even when W39 would have
RATIFIED.  This **bounds** (does not close) the
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` limitation theorem:
it raises the capsule-layer adversary bar from "compromise 2 of N
trajectory hosts AND ``quorum_min`` of the K mutually-disjoint
registered consensus references" to "compromise 2 of N trajectory
hosts AND ``quorum_min`` of the K mutually-disjoint registered
consensus references AND inject K diverse response text bytes".
On **R-87-RESPONSE-SIGNATURE-COLLAPSE**, W40 raises trust
precision over W39 from 0.500
(``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` fires) to **1.000**
(**Δ_trust_precision_w40_w39 = +0.500**, min and max equal across
5/5 seeds), abstains via RESPONSE_SIGNATURE_COLLAPSE on 8
cells/seed, and adds one visible token/cell while carrying about
**14.5k structured bits per visible W40 token**.  On
**R-87-COORDINATED-DIVERSE-RESPONSE** the new proved-conditional
limitation theorem **W40-L-COORDINATED-DIVERSE-RESPONSE-CAP**
fires (the smart attacker who diversifies response bytes while
holding the wrong top_set in lock-step ⇒ W40 cannot recover;
closure requires native-latent evidence outside the capsule layer
or a K+1-host disjoint topology with a new uncompromised pool).
On **R-87-TRIVIAL-W40**, **R-87-NO-REGRESSION-DIVERSE-AGREES**,
and **R-87-INSUFFICIENT-RESPONSE-SIGNATURE** falsifiers, W40
preserves W39 behavior with no correctness or trust-precision
regression.  W40 adds 14 verifier failure modes (including the
W40-specific ``w40_response_mutual_disjointness_violation``),
bringing the cumulative W22..W40 trust boundary to **168
enumerated modes**.  This is **not native latent transfer** and
not transformer-internal hidden-state projection; it is an
audited capsule-layer cross-host response-text Jaccard
divergence proxy with two mechanically-enforced disjointness
preconditions inherited from W39 (trajectory disjointness +
mutual disjointness).  **Lab topology -- W40-INFRA-1**:
``192.168.12.101`` re-probed -- ping 0% packet loss (a strict
improvement over W39's 100% packet loss end-state), TCP port
11434 connects, but the Ollama HTTP listener still returns
"Empty reply from server" / "Connection reset by peer" (the
W39-INFRA-1 hung-listener pattern persists; SSH credentials
still unavailable to restart).  Honest verdict: ``.101`` is
TCP-up + HTTP-broken at the Ollama layer; ``.248`` remains
ARP-incomplete (32nd milestone in a row).  **RC1 declared**:
H1..H12 + S3 of the W40 success criterion pass; the SDK v3.41
line is the **first official release-candidate** of the CoordPy
SDK v3.4x line.  Stable-vs-experimental boundary final for RC1:
every W22..W40 symbol is exported under ``__experimental__``;
the stable ``RunSpec → run report`` runtime contract is
byte-for-byte unchanged.  Versioning: ``vision_mvp.__version__``
and ``pyproject.toml`` are now both ``0.5.14`` (alignment
maintained).

**Previous milestone: SDK v3.40 (May 2026).** Multi-host disjoint quorum
consensus-reference ratification + manifest-v9 CID + mutually-disjoint
physical-host topology (W39 family).  W39 wraps W38's disjoint
cross-source consensus-reference adjudication with a **K-of-N
mutually-disjoint quorum** of disjoint probes, each sourced from a
physically-distinct host pool that is both mechanically disjoint from
the W37 trajectory hosts (W38's precondition) AND mutually disjoint
from every other registered quorum probe's host pool (the new W39
precondition; the ``MultiHostDisjointQuorumRegistry`` raises
:class:`MutuallyDisjointTopologyError` when any two pools have
non-empty intersection; the verifier additionally rejects envelopes
claiming an overlapping pool pair).  When at least ``quorum_min`` of
the K member probes diverge from the W37/W38 candidate top_set, W39
abstains via the ``QUORUM_DIVERGENCE_ABSTAINED`` branch.  This
**bounds** (does not close) the W38-L-CONSENSUS-COLLUSION-CAP
limitation theorem: it raises the capsule-layer adversary bar from
"compromise 2 of N trajectory hosts AND the single disjoint registered
consensus reference" to "compromise 2 of N trajectory hosts AND
``quorum_min`` of the K mutually-disjoint registered consensus
references, each on a physically distinct host pool".  On
**R-86-MULTI-HOST-COLLUDED-CONSENSUS**, W39 raises trust precision
over W38 from 0.500 (W38-L-CONSENSUS-COLLUSION-CAP fires) to **1.000**
(**Δ_trust_precision_w39_w38 = +0.500**, min and max equal across 5/5
seeds), abstains via QUORUM_DIVERGENCE on 8 cells/seed, and adds one
visible token/cell while carrying about **24.4k structured bits per
visible W39 token** (~2.7x denser than W38 at the audited-proxy
capsule layer).  On **R-86-FULL-QUORUM-COLLUSION** the new
proved-conditional limitation theorem
**W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP** fires (every disjoint
quorum probe is itself compromised in lock-step ⇒ W39 cannot recover;
closure requires native-latent evidence outside the capsule layer or
a K+1-host disjoint topology with a new uncompromised pool).  On
**R-86-TRIVIAL-W39**, **R-86-NO-REGRESSION-QUORUM-AGREES**, and
**R-86-INSUFFICIENT-QUORUM** falsifiers, W39 preserves W38 behavior
with no correctness or trust-precision regression.  W39 adds 14
verifier failure modes (including the W39-specific
``w39_quorum_mutual_disjointness_violation``), bringing the
cumulative W22..W39 trust boundary to **154 enumerated modes**.
This is **not native latent transfer** and not transformer-internal
hidden-state projection; it is an audited capsule-layer multi-host
disjoint quorum proxy with two mechanically-enforced disjointness
preconditions (trajectory disjointness + mutual disjointness).
**Multi-Mac evidence broadened materially**: the historical
``192.168.12.248`` Mac-2 stale pin (ARP-incomplete for the 31st
milestone in a row) was discharged in favour of ``192.168.12.101``
as the reachable third physical host candidate (preflight-OK on
cold contact with ``qwen3.5:35b`` + ``qwen2.5:14b-32k`` model files
visible) -- partially discharging W38-C-MULTI-HOST at the topology
layer.  The ``.101`` inference path subsequently degraded under
the one-word probe budget (``W39-INFRA-1``).  The W39 live xllm
probe was made robust via a fallback path: when ``.101`` is
unreachable, ``mac_off_cluster_a`` swaps to ``localhost`` running
``llama3.1:8b`` (a model class genuinely different from the
trajectory's ``gemma2:9b``), so the live K=2 quorum becomes
``(localhost llama3.1:8b, .191 qwen2.5-coder:14b-32k)`` -- two
**physically distinct hosts**, each running a different model class
from the trajectory pair AND from the W38 single consensus
reference.  Bounded W39 5-host live xllm probe at temperature 0 +
``num_predict=4`` observed **8/8 responsive on all 5 hosts** (first
5-host live W39 disjoint-quorum probe in the programme), 7/8
trajectory_pair_agrees, 7/8 W38 single consensus agreements, **8/8
quorum_a gold-correlated, 8/8 quorum_b gold-correlated, 8/8 K=2
quorum size simultaneously responsive**.  Notable live finding: on
the ``h2o`` probe, the trajectory pair disagreed (mac1=h2o vs
mac_remote=h due to ``num_predict=4`` truncation), but BOTH quorum
members got h2o correct -- empirical-suggestive evidence for the
new ``W39-C-LIVE-TRUNCATION-RECOVERY`` conjecture.  Versioning:
``vision_mvp.__version__`` and ``pyproject.toml`` are now both
``0.5.13`` (alignment maintained).  Stable runtime contract
unchanged; W39 remains experimental.

**Earlier milestone: SDK v3.39 (May 2026).** Disjoint cross-source
consensus-reference trajectory-divergence adjudication + manifest-v8
CID (W38 family).  W38 wraps W37's anchor-cross-host basis-trajectory
ratification with a controller-pre-registered
``ConsensusReferenceProbe`` whose host topology is *mechanically
disjoint* from W37's trajectory hosts.  When W37 chooses to reroute
on a trajectory-anchored top_set and the disjoint consensus reference
disagrees within ``divergence_margin_min`` (Jaccard), W38 abstains
via the ``CONSENSUS_DIVERGENCE_ABSTAINED`` branch.  On
**R-85-COLLUDED-CROSS-HOST-TRAJECTORY**, W38 raises trust precision
over W37 from 0.500 to **1.000**
(**Δ_trust_precision_w38_w37 = +0.500**, min and max equal across 5/5
seeds), abstains via DIVERGENCE on 8 cells/seed, and adds one visible
token/cell while carrying about **9.07k structured bits per visible
W38 token**.  On **R-85-CONSENSUS-ALSO-COMPROMISED** the
proved-conditional limitation theorem **W38-L-CONSENSUS-COLLUSION-CAP**
fires (further bounded by W39 in SDK v3.40).  Stable runtime contract
unchanged; W38 remains experimental.

**Earlier milestone: SDK v3.38 (May 2026).** Anchor-cross-host basis-
trajectory ratification + manifest-v7 CID (W37 family).  W37 wraps
W36's host-diverse trust-subspace guard with a closed-form,
zero-parameter, per-(host, oracle, top_set) EWMA over *anchored*
historical observations.  W36 abstains whenever the current cell has
fewer than ``min_distinct_hosts`` healthy attested hosts -- even when
the remaining single host has been independently anchored across
earlier cells by other healthy hosts.  W37 makes that historical
cross-host anchoring a typed audited precondition for a safe
single-host reroute.  On **R-84-SINGLE-HOST-TRAJECTORY-RECOVER**, W37
raises correctness over W36 from 0.500 to **1.000**
(**Δ_correctness_w37_w36 = +0.500**, min and max equal across 5/5
seeds) at trust precision **1.000**, recovers all 8 W36 abstentions
per seed, and adds one visible token/cell while carrying about
**29.5k structured bits per visible W37 token**.  On
**R-84-NO-TRAJECTORY-HISTORY**, **R-84-POISONED-TRAJECTORY**, and
**R-84-TRAJECTORY-DISAGREEMENT** falsifiers, W37 preserves W36
behavior byte-for-byte.  W37 adds 14 verifier failure modes, bringing
the cumulative W22..W37 trust boundary to **126 enumerated modes**.
A new proved-conditional limitation theorem
**W37-L-MULTI-HOST-COLLUSION-CAP** is recorded: two registered hosts
emitting a coordinated wrong top_set across enough cells can cross
the anchored thresholds and cannot be separated at the capsule layer.
This is **not native latent transfer** and not transformer-internal
hidden-state projection; it is an audited capsule-layer cross-cell
host-trust proxy.  Fresh preflight found local Ollama and
`192.168.12.191` reachable, with `192.168.12.248` still timing out
(Mac 2 ARP-incomplete for the **30th milestone in a row**); the
bounded W37 cross-host trajectory probe observed **8/8 cross-host
anchored agreements and 8/8 gold-correlated agreements** across local
`gemma2:9b` and remote `qwen2.5:14b` at temperature 0.  Stable
runtime contract unchanged; W37 remains experimental.

**Earlier milestone: SDK v3.37 (May 2026).** Host-diverse trust-
subspace guard + manifest-v6 CID (W36 family).  W36 wraps W35's
trust-subspace dense-control proxy with a host-diverse verifier: a
dense projection is eligible only when its support is independently
attested by at least two registered healthy hosts.  On **R-83-HOST-
DIVERSE-RECOVER**, W36 raises correctness over W35 from 0.625 to
**0.9375** (**+0.3125**) across 5/5 seeds and restores trust
precision from 0.6667 to **1.000**.  W36 adds 14 verifier failure
modes, bringing the cumulative W22..W36 trust boundary to
**112 enumerated modes**.  Stable runtime contract unchanged; W36
remains experimental.

**Earlier milestone: SDK v3.36 (May 2026).** Trust-subspace dense-
control proxy + basis-history projection + W35 manifest-v5 CID (W35
family).  W35 wraps W34's live-aware multi-anchor abstention path
with a controller-verified dense basis over W21 probe top_sets, W33
EWMA trust, W34 live-attestation/response-feature state, top-set
stability, and host health.  On **R-82-TRUST-SUBSPACE-SHIFT**,
W34 abstains on 6 disputed cells; W35 safely reroutes 5/6 through the
stable `change_history` basis direction, raising correctness from
0.625 to **0.9375** (**+0.3125**) across 5/5 seeds while preserving
trust precision at **1.000** and adding one visible token/cell.
Stable runtime contract unchanged; W35 remains experimental.

**Earlier milestone: SDK v3.35 (May 2026).** Live-aware multi-anchor
adjudication + native-latent audited response-feature proxy + W34
manifest-v4 CID + W33 infra-blocker closure (preflight ``/api/tags``
+ chat-template + ``num_predict=4`` + stop tokens for one-word
probes) (W34 family).  The W34
``LiveAwareMultiAnchorOrchestrator`` wraps the W33
``TrustEWMATrackedMultiOracleOrchestrator`` with: (a) a
**multi-anchor consensus reference** that takes the *intersection*
of K registered anchors' top_sets when at least
``anchor_quorum_min`` non-abstaining anchors agree, with
NO_CONSENSUS abstention when the intersection is empty (anchor
disagreement is itself a trust signal — closes the W33
single-anchor *fragility* where a flipped anchor caused W33 to
detrust the honest oracles); (b) a **closed-form 64-bit
response-feature signature** (``compute_response_feature_signature``
returns a SHA-256 hex prefix over (first_token_class, length_bucket,
structural_hash)) — the W34 audited proxy for native-latent;
explicitly NOT a transformer-internal hidden-state projection, NOT
a learned feature embedding, NOT a runtime KV transplant; (c) a
content-addressed **LiveOracleAttestation** per cell (host_id,
model_id, response_feature_signature, latency_ms_bucket,
preflight_ok); (d) a closed-form **host-aware EWMA decay** with
``host_decay_factor ∈ [0.5, 1.0]`` applied multiplicatively to
oracles whose host failed preflight; (e) a **manifest-v4 CID** over
four component CIDs (parent_w33_cid, live_attestation_cid,
multi_anchor_cid, host_topology_cid).  The new
``verify_live_aware_multi_anchor_ratification`` enumerates **14
disjoint failure modes** (cumulative 84 across W22 + W29 + W30 +
W31 + W32 + W33 + W34).  **First capsule-native multi-agent-
coordination method to defeat anchor-itself-flips attacks at the
capsule layer**: on **R-81-DOUBLE-ANCHOR-COMPROMISE** (where the
W33 anchor itself becomes compromised in the final phase), W33
single-anchor commits to wrong with trust precision 0.625; W34
multi-anchor with K=2 + ``anchor_quorum_min=2`` abstains via the
NO_CONSENSUS branch ⇒ trust precision = **1.000** at
**Δ_trust_precision_w34_w33 = +0.375 across 5/5 seeds**, no
correctness regression, max overhead 1 token/cell.  On
**R-81-MANIFEST-V4-TAMPER** the manifest-v4 CID + cross-component
CID checks yield **400/400 = 1.000 tamper rejection rate**.  On
**R-81-RESPONSE-FEATURE-SIGNATURE** the audited proxy is
byte-stable across 10 fixtures × 3 runs = 30/30 byte-equal calls.
On **R-81-TRIVIAL-W34** the trivial passthrough yields **W34 = W33
byte-for-byte across 5/5 seeds**.  Five named W34-Λ falsifiers all
empirically observed.  **W33-INFRA-1 + W33-INFRA-2 jointly closed**:
preflight discipline (an honest empirical correction recorded —
qwen3.5:35b on 192.168.12.191 IS in fact loaded; W33's diagnosis
was wrong, the real W33 infra failure was timeout-budget
exhaustion + chat-template) + ``/api/chat`` + ``num_predict=4`` +
stop tokens + adaptive timeout (small models 30 s, medium 60 s,
large >= 30B 240 s) — both load-bearing in the W34 live xLLM pilot.
**W34-L-MULTI-ANCHOR-CAP limitation theorem**: when all K anchors
are simultaneously compromised at the capsule layer, no
multi-anchor mechanism (including W34) can recover; native-latent
(architecture-dependent; ``W33-C-NATIVE-LATENT``) is required to
break this.  Mac 2 (192.168.12.248) still ARP-incomplete (29th
consecutive milestone).  Stable runtime contract byte-for-byte
unchanged from v3.34.

**Earlier milestone: SDK v3.34 (May 2026).** Trust-EWMA-tracked
multi-oracle adjudication + per-oracle agreement signal + anchor-
oracle reference + content-addressed oracle-trust-state + trust-
trajectory CID + W33 manifest-v3 CID + single-partition long-window
strict-gain regime that exceeds the W32-L-CYCLE-CAP limitation
theorem (W33 family).  The W33
``TrustEWMATrackedMultiOracleOrchestrator`` wraps the **OLD W21
``TrustWeightedMultiOracleDisambiguator``** (the explicit
multi-oracle adjudication line, dormant since W21) with a **closed-
form per-oracle EWMA trust accumulator** (``ewma_new = (1 - α) *
ewma_prev + α * obs`` where ``obs`` is the per-cell agreement
between each oracle's probe and a registered **anchor-oracle
reference** that is trust-by-construction immune to quorum-flip),
a **trust-threshold gate** that excludes detrusted oracles from
the effective tally, a **sealed (cell_idx, oracle_id, observed,
ewma_after) trust trajectory CID**, and a **manifest-v3 CID** over
six component CIDs (w21_oracle_cid, oracle_trust_state_cid,
trust_trajectory_cid, anchor_oracle_set_cid, route_audit_cid_v3,
w32_long_window_cid).  W33 is **NOT** a learned trust model in the
deep-learning sense: zero parameters, zero gradients, zero training
step; the EWMA primitive is the same closed-form W32 primitive
applied to the W21 quorum-agreement signal at the per-oracle level.
The ``TrustEWMARatificationEnvelope`` adds a sealed trajectory of
``(cell_idx, oracle_id, observed_quorum_agreement, ewma_trust_after)``
entries; the new ``verify_trust_ewma_ratification`` enumerates **14
disjoint failure modes** (cumulative 70 across W22 + W29 + W30 +
W31 + W32 + W33).  **First capsule-native multi-agent-coordination
method to simultaneously discharge two named open conjectures from
two different research lines in a single milestone**:
**W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST** are
**jointly discharged at +0.375 trust-precision strict gain** across
5/5 seeds × 16 cells/seed on **R-80-COMPROMISED-SHIFT** (W21
fixed-trust adjudicator commits to the wrong answer when 2 of 3
oracles become compromised mid-session ⇒ trust precision = 0.625;
W33 EWMA-tracks each oracle's anchor-agreement, gates out the
detrusted ones, and recovers ⇒ trust precision = **1.000**, no
correctness regression, max overhead 1 token/cell).
**W32-C-LONG-WINDOW-STRICT-GAIN** is **discharged at +0.10
correctness strict gain** across 5/5 seeds × 80 cells on
**R-79-SINGLE-PARTITION** (a prefix-then-shift regime over a
single-partition signature space that exceeds the W32-L-CYCLE-CAP
``Δ_max ≤ min(c_p/4, c_s)/N ≤ 0.0625`` cycle-capped bound by
construction).  On **R-80-MANIFEST-V3-TAMPER** the manifest-v3 CID
+ cross-component CID checks together yield **400/400 = 1.000
tamper rejection rate** across 5/5 seeds × 16 ratified cell-
positions × 5 named tampers per cell (oracle_trust_state CID byte
corruption, trust_trajectory CID byte corruption, anchor_oracle_set
swap, manifest_v3_cid byte corruption, w33_cid byte corruption).
On **R-80-TRIVIAL-W33** the trivial passthrough yields **W33 = W21
byte-for-byte across 5/5 seeds**.  Four named W33-Λ falsifiers all
empirically observed (W33-Λ-trivial-trust-ewma ⇒ byte-for-W21
passthrough; W33-Λ-no-trust-shift ⇒ all EWMA stay at 1.0;
W33-Λ-frozen-threshold ⇒ gate never fires; W33-Λ-mis-trust-shift
⇒ honest empirical: anchor-oracle design is more robust than the
falsifier predicted).  Live cross-host pilot (mixtral:8x7b on
localhost + qwen3.5:35b on 192.168.12.191) **honestly null on
infrastructure** (qwen3.5:35b not actually loaded on the remote
host; mixtral past-token-budget at temp 0); the
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE conjecture **remains open**
and is recorded with two infrastructure-fix items
(W33-INFRA-1, W33-INFRA-2).  Stable-vs-experimental boundary
further tightened; the new W33 surface lives under
``__experimental__`` (31 unit tests + the verifier);
``SDK_VERSION = "coordpy.sdk.v3.34"``; pyproject.toml ``0.5.7``.
**446/446 phase69-80 regression pass + 31/31 W33 unit tests +
133/133 wider coordpy suite = 610 tests pass**.  Mac 2
(192.168.12.248) **still ARP-incomplete (28th consecutive
milestone)**.  See
[`docs/RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md`](docs/RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md)
and [`CHANGELOG.md`](CHANGELOG.md) for details.

**Previous milestone: SDK v3.33 (May 2026).** Long-window convergent
online geometry-aware dense control + EWMA prior accumulator + Page
CUSUM change-point detector + gold-correlated disagreement-routing +
W32 manifest-v2 CID + first measured live cross-architecture LLM
gold-verifiable agreement at temperature 0 (W32 family).  The W32
``LongWindowConvergentOrchestrator`` wraps the W31
``OnlineCalibratedOrchestrator`` with a **closed-form EWMA prior
accumulator** (``ewma_new = (1 - α) * ewma_prev + α * obs`` with
default α=0.20, ~13× more responsive than W31's effective
1/(n+1) ≈ 0.015 at trajectory_window=64), a **closed-form Page
two-sided CUSUM change-point detector** (cusum_pos / cusum_neg
accumulators bounded by registered cusum_max=16.0; threshold 1.5;
slack 0.10), a **gold-correlated disagreement-routing primitive**
against a registered closed-vocabulary ``GoldCorrelationMap`` (the
map is a structural witness, NOT a runtime ground-truth oracle),
and a **content-addressed convergence-state trajectory + manifest-v2
CID** that closes cross-component swap avenues the W31 manifest
CID cannot detect.  W32 is **NOT** a learned model in the
deep-learning sense: zero parameters, zero gradients, zero training
step.  The ``LongWindowConvergentRatificationEnvelope`` adds a
sealed trajectory of
``(cell_idx, partition_id, ewma_prior_after, cusum_pos, cusum_neg,
change_point_fired)`` entries plus a ``manifest_v2_cid`` over
(w31_online_cid, convergence_state_cid, gold_correlation_cid,
route_audit_cid_v2); the new
``verify_long_window_convergent_ratification`` enumerates **14
disjoint failure modes** (cumulative 56 across W29 + W30 + W31 +
W32).  **First capsule-native multi-agent-coordination method to
discharge W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability
axis**: on **R-79-LONG-WINDOW** (sweep over long_window ∈
{16, 32, 64, 128} on the prefix-then-shift drift_recover regime),
the W32 EWMA + CUSUM mechanism achieves
**byte-for-W31-equal correctness across 5/5 seeds × 4/4 windows =
20/20 cell-window pairs at trust precision 1.000; zero
degradation as window grows**.  On **R-79-MANIFEST-V2-TAMPER** the
manifest-v2 CID + cross-cell convergence_state_cid check together
yield **1525/1525 = 1.000 tamper rejection rate** across 5/5
seeds × 61 ratified cell-positions × 5 named tampers per cell
(cross-cell convergence_state swap with self-consistent manifest-v2
recompute, manifest_v2_cid byte corruption, ewma_prior_after out of
range, cusum_pos out of range, outer w32_cid byte corruption).  On
**R-79-DRIFT-RECOVER** Δ(W32 - W31) = 0.000 (honestly-null per the
**W32-L-CYCLE-CAP limitation theorem**: max strict gain is bounded
above by ``min(c_p / 4, c_s) / N ≤ 0.0625`` on cycle-capped
dispatcher regimes; the mechanism is empirically validated by
``n_change_points = 1`` firing exactly at the shift boundary).
**First measured live cross-architecture LLM gold-verifiable
agreement at temperature 0 in the programme** (29th milestone):
gemma2:9b on localhost + qwen2.5:14b on 192.168.12.191 agree on
**19/20 = 0.950 of gold-verifiable structured-decision prompts at
temp 0** (5/5 arithmetic + 5/5 syntax + 5/5 factoid + 4/5
disambiguation; the unique disagreement is a tokenisation boundary
where neither host matches the registered gold).  **The honest
converse of W31's R-78-XLLM-LIVE result** (6/8 = 0.750 agreement
on operational-decision prompts) — together they characterise the
**prompt-class-dependent cross-architecture disagreement frontier**
at temp 0.  Four named falsifiers empirically observed
(W32-Λ-trivial-long-window ⇒ byte-for-W31 passthrough;
W32-Λ-no-change-point ⇒ stationary regime fires 0 change-points;
W32-Λ-frozen-ewma ⇒ honest empirical correction (did NOT regress);
W32-Λ-mis-correlated-gold ⇒ gate-bounded on synthetic).
**Sharpens W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
prompt-class-dependent agreement frontier (renamed forward as
W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE).**  Proves the new
**W32-L-CYCLE-CAP limitation theorem** that bounds strict-gain
claims on cycle-capped dispatcher regimes — a load-bearing
honest-scope distinction.  Stable-vs-experimental boundary
further tightened; the new W32 surface lives under
``__experimental__`` (45 unit tests + the verifier);
``SDK_VERSION = "coordpy.sdk.v3.33"``; pyproject.toml ``0.5.6``.
**414/414 phase69-79 regression pass + 45/45 W32 unit tests + 77/77
wider coordpy suite = 536 tests pass**.  Mac 2 (192.168.12.248)
**still ARP-incomplete (27th consecutive milestone)**.  See
[`docs/RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md`](docs/RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md)
and [`CHANGELOG.md`](CHANGELOG.md) for details.

**Earlier milestone: SDK v3.32 (May 2026).** Online self-calibrated
geometry-aware dense control + sealed prior trajectory + adaptive
threshold + W31 manifest CID + first measured live cross-architecture
LLM disagreement at temperature 0 (W31 family).  The W31
``OnlineCalibratedOrchestrator`` wraps the W30
``CalibratedGeometryOrchestrator`` with a **closed-form online
running-mean update** for the per-partition calibration prior driven
by a deterministic per-cell agreement signal
(``ratified AND no cross-host disagreement`` ⇒ 1.0; else 0.0), a
**closed-form clipped-median adaptive threshold** bounded to
``[0.20, 0.80]``, and a **content-addressed prior + threshold
trajectory + manifest CID** that closes every cross-component swap
avenue.  W31 is **NOT** a learned model in the deep-learning sense:
zero parameters, zero gradients, zero training step.  The
``OnlineCalibratedRatificationEnvelope`` adds a sealed trajectory of
``(cell_idx, partition_id, observed_agreement, prior_after)``
entries plus a ``manifest_cid`` over (basis_history_cid,
calibration_cid, ancestor_chain_cid, prior_trajectory_cid,
threshold_trajectory_cid, route_audit_cid); the new
``verify_online_calibrated_ratification`` enumerates **14 disjoint
failure modes** (cumulative 42 across W29 + W30 + W31).  **First
capsule-native multi-agent-coordination method to discharge
W30-C-PRIOR-LEARNING on a non-stationary regime**: on
**R-78-NONSTATIONARY-PRIOR**
``correctness_ratified_rate_w30 = 0.750`` (uniform priors cannot
reroute) vs ``correctness_ratified_rate_w31 = 0.875``,
**Δ = +0.125 across 5/5 seeds at trust precision 1.000**, mean
overhead 0.875 tokens/cell, max overhead 1.  On
**R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD** the adaptive
contribution isolates at +0.125 across 5/5 seeds (frozen yields 0).
On **R-78-MANIFEST-TAMPER** the manifest CID + cross-cell trajectory
CID check together yield **65/65 = 1.000 tamper rejection rate**
across five named tampers including the cross-cell prior_trajectory
swap that self-consistently recomputes the manifest CID.  **First
measured live cross-architecture LLM disagreement at temperature 0
in the programme** (28th milestone): gemma2:9b on localhost +
qwen2.5:14b on 192.168.12.191 disagree on **2/8 = 0.250 of
structured-decision prompts at temp 0**, reproducible byte-for-byte
across two runs (Q5: db_query vs logs_pipeline; Q7: api vs storage).
Three named falsifiers all empirically confirmed
(W31-Λ-trivial-online ⇒ byte-for-W30 passthrough; W31-Λ-no-drift ⇒
no help on stationary regime; W31-Λ-frozen-threshold ⇒ no adaptive
contribution when threshold is fixed at 0.5).
**Sharpens W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the
infrastructure-discharge axis.**  Stable-vs-experimental boundary
further tightened; the new W31 surface lives under
``__experimental__`` (41 unit tests + the verifier);
``SDK_VERSION = "coordpy.sdk.v3.32"``; pyproject.toml ``0.5.5``.
**437/437 phase69-78 regression pass byte-for-byte** (was 357/357
in v3.31; +41 W31 unit tests + 39 unchanged from v3.31 + 1
unchanged); 68/68 wider coordpy suite passes.  Mac 2 (192.168.12.248)
**still ARP-incomplete (26th consecutive milestone)**.  See
[`docs/RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md`](docs/RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md)
and [`CHANGELOG.md`](CHANGELOG.md) for details.

**Earlier milestone: SDK v3.31 (May 2026).** Calibrated
geometry-partitioned dense control + multi-stride basis-history +
per-partition calibration prior + cross-host disagreement-routing +
ancestor-chain causal binding (W30 family). The W30
``CalibratedGeometryOrchestrator`` wraps the W29 geometry-partitioned
orchestrator with two **closed-form, audited** override mechanisms
that ride on a single content-addressed envelope at a fixed 1-token
wire cost: a per-partition calibration prior (deterministic running
mean over observed agreement, **NOT** a learned model) that strictly
beats the uniform-trust baseline, and a cross-host
disagreement-routing override that peeks at the inner W28's
last-result ``cross_host_disagreement_count`` and reroutes the
*next* cell to a high-trust partition when the previous cell's
ensemble disagreed. The
``CalibratedGeometryRatificationEnvelope`` additionally carries a
``BasisHistory`` (multi-stride content-addressed accumulator over
``stride`` recent partition basis CIDs — rotation detectable, so
H6 cram amplification is real) and an ``AncestorChain``
(``ancestor_window`` sorted parent CIDs giving deeper causal
binding than W29's single-parent W28 link). The new
``verify_calibrated_geometry_ratification`` enumerates **14
additional failure modes** (basis-history-CID forgery,
calibration-vector-out-of-range, ancestor-chain-CID forgery,
disagreement-route-target forgery, calibrated-CID forgery) —
disjoint from W29's 14, so the trust boundary is tightened on a
fresh axis. **First capsule-native multi-agent-coordination method
to discharge BOTH W29 cram-amplification AND per-partition
calibration on a single milestone**: on **R-77-CHAIN-CRAM**
``cram_w30/w28 = 8.74×`` AND ``cram_w30/w29 = 2.74×`` at
stride=28, window=12, across 5/5 seeds; on
**R-77-CALIBRATION-PRIOR** the calibration override yields
``Δ(W30 − W29) = +0.250 correctness gain`` across 5/5 seeds; on
**R-77-XHOST-DISAGREE** the disagreement-routing override yields
``Δ(W30 − W29) = +0.250 correctness gain`` across 5/5 seeds.
``trust_precision_w30 = 1.000`` on every R-77 sub-bank where W30
ratifies; tamper rejection 100 % (60/60) across five named modes.
Three named falsifiers all empirically confirmed
(W30-Λ-trivial-calibration ⇒ byte-for-W29 passthrough;
W30-Λ-non-calibratable ⇒ no measurable calibration gain when
priors are degenerate uniform; W30-Λ-degenerate-history ⇒
basis-history with a single repeated CID does not amplify cram).
The new vocabulary (calibration prior, disagreement-routing,
basis-history, ancestor-chain) is honestly framed as
**capsule-layer audited proxy**, **NOT** a learned partition
classifier and **NOT** transformer-internal calibration.
**Empirically discharges W29-C-CRAM-AMPLIFICATION (8.74× ≥ 8.0
bar) AND W29-C-PARTITION-CALIBRATION on a single milestone.**
Stable-vs-experimental boundary further tightened; the new W30
surface lives under ``__experimental__`` (36 unit tests + the
verifier); ``SDK_VERSION = "coordpy.sdk.v3.31"``; pyproject.toml
``0.5.4``. **357/357 focused regression pass** (273/273 phase69-77
+ 84/84 wider coordpy suite). See
[`docs/RESULTS_COORDPY_W30_CALIBRATED_GEOMETRY.md`](docs/RESULTS_COORDPY_W30_CALIBRATED_GEOMETRY.md)
and [`CHANGELOG.md`](CHANGELOG.md) for details.

**Previous milestone: SDK v3.30 (April 2026).** Geometry-partitioned
product-manifold dense control + audited subspace-basis payload +
factoradic Lehmer routing index + causal-validity gate + cross-host
variance witness (W29 family). The W29
``GeometryPartitionedOrchestrator`` wraps the W28 ensemble layer
with a **structural geometry-partitioning** step that classifies
every triggered cell into one of three pre-committed labels —
LINEAR (extends most-recent signature), HIERARCHICAL (fresh
anchor), CYCLIC (re-visited signature) — keyed by a deterministic
signature-history heuristic. Per-partition inner W28 stacks get
their own oracle / probe / pool topology. **First capsule-native
multi-agent-coordination method to demonstrate the synthesis (W21
× W27, sealed by W28, geometry-partitioned by W29) strictly
improves correctness on a regime where the prior best (W28) makes
correctness mistakes.** On **R-76-XHOST-DRIFT**
``correctness_ratified_rate_w29 = 0.750`` vs
``correctness_w27 = correctness_w28 = 0.500``, **Δ = +0.250 across
5/5 seeds**, ``trust_precision = 1.000``, ``mean overhead = 0.75
tokens/cell``. **Empirically discharges W28-C-CROSS-HOST-VARIANCE
on the magnitude axis.** See
[`docs/RESULTS_COORDPY_W29_GEOMETRY_PARTITIONED.md`](docs/RESULTS_COORDPY_W29_GEOMETRY_PARTITIONED.md).

**Previous milestone: SDK v3.29 (April 2026).** Ensemble-verified
cross-model multi-chain pivot ratification (W28 family). The W28
``EnsembleVerifiedMultiChainOrchestrator`` wraps the W27 multi-
chain pool with a controller-side **trust-weighted probe quorum**
(each probe is an ``EnsembleProbeRegistration`` with a
``trust_prior`` mirroring W21's ``OracleRegistration``) that
ratifies or rejects every pivot/anchor decision via a content-
addressed ``EnsemblePivotRatificationEnvelope``;
``verify_ensemble_pivot_ratification`` enumerates **11 new
failure modes** (probe forgery, weight forgery, quorum forgery,
hash tampering, schema/signature drift) — none of which existed
in any W22..W27 verifier. **First capsule-native mechanism that
synthesises the explicit-capsule trust line (W21 multi-oracle
adjudication) with the dense-control line (W27 multi-chain
salience-keyed pool) inside one decision.** **First cross-host
live LLM evidence in 23 milestones**: R-75-CROSS-HOST-LIVE on
localhost (gemma2:9b) + 192.168.12.191 (qwen2.5:14b) records 128
cross-host probe calls and 5592 LAN bytes; ensemble ratifies
10/16 cells (real LLM disagreement on 6/16) with trust precision
1.000 and W28 correctness 1.000. Stable-vs-experimental
boundary tightened: dense-control surface (W22..W28) now lives
under an explicit ``__experimental__`` tuple. See
[`docs/RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md`](docs/RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md).

**Previous milestone: SDK v3.28 (April 2026).** Multi-chain
salience-keyed dense-control fanout + per-signature scoping
(W27 family). The W27
``MultiChainPersistedFanoutOrchestrator`` maintains a bounded
pool of independent W26 stacks keyed by the cell's salience
signature (SHA-256 over canonical input handoffs computed by
``compute_input_signature_cid``); the audited
``MultiChainPersistedFanoutDisambiguator`` adds two new content-
addressed envelopes (``SalienceSignatureEnvelope``,
``ChainPivotEnvelope``) plus ``verify_salience_signature`` (4
failure modes) and ``verify_chain_pivot`` (8 failure modes) for
trust-boundary auditing. On R-74-XORACLE-RECOVER (1 producer +
K=3 consumers, 16 cells, 2 signatures, partial oracle on the W26
baseline scoped to GOLD_A) the W27 method **simultaneously**
strictly reduces visible tokens by **−22.5 tokens / cell
(−76.27 %)** over W26 AND raises ``correctness_ratified_rate``
from **0.500 → 1.000**, stable across 5/5 seeds. The first
capsule-native multi-agent-coordination method that
*simultaneously* improves both efficiency and correctness over
the prior best on a regime where the prior best architecturally
limits correctness. Four named falsifiers
(W27-Λ-single-signature / -pool-exhausted / -pivot-tampered /
-signature-drift). **Discharges W26-C-DIVERGENCE-RECOVERY** on
the per-signature scoping axis. See
[`docs/RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md`](docs/RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md).

**Previous milestone: SDK v3.27 (April 2026).** Chain-persisted
dense-control fanout + per-consumer projections (W26 family). The
W26 ``ChainPersistedFanoutDisambiguator`` amortises the producer's
per-cell salience-token cost across cells via a two-tier
content-addressed envelope hierarchy (``ChainAnchorEnvelope`` +
``ChainAdvanceEnvelope``); on R-73-CHAIN-SHARED (1 producer + K=3
consumers, 16 cells) it strictly reduces total visible tokens by
**−12.125 tokens / cell (−68.79 %)** over W25 AND **−53.00 tokens /
cell (−90.60 %)** over W24, stable across 5/5 seeds. Trust boundary:
``verify_chain_anchor`` (6 failure modes), ``verify_chain_advance``
(8), ``verify_projection_subscription`` (2). Four named falsifiers
(W26-Λ-no-chain / -tampered / -projection-mismatch / -divergent).
W25-C-K-SCALING discharged at K∈{3,5,8,10}. See
[`docs/RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md`](docs/RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md).

**As of SDK v3.24 (April 2026), capsules are load-bearing
*inside one CoordPy run*, *between agents in a team*, *across the
model-class gradient*, *across the multi-oracle adjudication
axis*, *across the per-cell latent-hybrid axis (W22)*, and —
most sharply — *across the first capsule-native multi-agent-
coordination method that combines explicit-capsule passing with
audited proxies for the LatentMAS direction at the **cross-cell
session layer** (collective KV pooling / latent hidden-state
transfer / super-token side channels): the W23 family ships
:class:`SessionDigestEnvelope` (hash-chained running cross-cell
state — the LatentMAS *cross-cell latent state-sharing* direction
at the capsule layer), :class:`SessionDeltaEnvelope` (per-cell
delta carrying only what changed), :class:`SuperTokenReferenceEnvelope`
(bounded steganographic / dense-control-payload single-token CID
reference), :class:`SuperTokenRegistry` (controller-side
verifier registry), :class:`QuorumKeyedSharedReadCache` (per-oracle
freshness policy that mitigates W22-C-CACHE-AMPLIFICATION on
probabilistic LLM oracles), :class:`QuorumKeyedCachingOracleAdapter`
(drop-in W23 oracle adapter), :class:`CrossHostProducerDecoderProxy`
(within-process producer/decoder host-split simulation — the honest
fallback for the unreachable Mac 2), :func:`verify_session_digest_chain`
/ :func:`verify_session_delta` / :func:`verify_super_token_reference`
(≈ 30–40-line controller-side verification, enumerated failure
modes), and :class:`CrossCellDeltaDisambiguator` (the wrapping
decoder). On the new R-70-DELTA-FANOUT regime, W23 strictly
reduces visible-tokens-to-decider over the W22 baseline by **−2.75
tokens / cell (−6.67 %)** loose, **−2.75 tokens / cell (−7.53 %)**
tight (delta path); **−10.50 tokens / cell (−25.45 %)** loose,
**−10.50 tokens / cell (−28.77 %)** tight (super-token path); AND
ties W22 byte-for-byte on ``accuracy_full = 1.000`` — stable across
5/5 ``bank_seed`` values. On R-70-AMPLIFIED-LLM (synthetic
``FlippingProbabilisticOracle``), the W23 quorum-keyed cache
**empirically discharges W22-C-CACHE-AMPLIFICATION** at **+0.125
strict mitigation advantage** over the W22 baseline (W22 = 0.875,
W23 quorum-keyed = 1.000). Three named falsifiers (R-70-NO-DELTA,
R-70-SUPER-TOKEN-TAMPERED, R-70-CHAIN-BROKEN) make the W23-1 / W23-3
conditionality sharp: no cross-cell state → no savings; tampered
super-token → ``unknown_super_token`` → fall through to W22; chain
head split → ``chain_head_mismatch`` → fall through. Live LLM
transfer (W23-Λ-real, mixtral:8x7b on Mac-1, n=4): visible-tokens
savings **+2.84 % (delta)**, **+11.37 % (super-token)**;
``chain_verifies_ok_rate = 0.750``;
``super_token_verification_ok_rate = 1.000``;
``cross_host_round_trip_bytes_total = 4232`` bytes — the
producer/decoder split is wire-validated; the synthetic mitigation
advantage does not cleanly transfer at n=4 (newly named conjecture
**W23-C-MITIGATION-LIVE-VARIANCE**). 703 prior coordpy-anchor +
capsule + recent-phase tests pass, 39 new W23 tests pass = **742 /
742**. Mac 2 remains unreachable (**17th milestone in a row**);
the :class:`CrossHostProducerDecoderProxy` forces every W23
envelope through a JSON-canonical wire round-trip on every cell —
the W23 envelopes survive the wire boundary with no shared Python
references; when Mac 2 returns the same proxy interface drops in
over a real socket with no W23 code changes. SDK v3.24's
contribution is the Phase-70 capsule-session-delta benchmark family
([`vision_mvp/experiments/phase70_capsule_session_delta.py`](vision_mvp/experiments/phase70_capsule_session_delta.py))
plus the W23 surface in
[`vision_mvp/coordpy/team_coord.py`](vision_mvp/coordpy/team_coord.py).
Empirically discharges the SDK v3.23 W22-C-CACHE-AMPLIFICATION
conjecture as a *mitigable* property (the synthetic mitigation is
+0.125 strict; the live mitigation is partially discharged). See
[`docs/RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md`](docs/RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md)
for the full SDK v3.24 milestone note.*

**Historical SDK v3.23 reading (preserved for context).** Capsules
were load-bearing
*inside one CoordPy run*, *between agents in a team*, *across the
model-class gradient*, *across the multi-oracle adjudication
axis*, and — most sharply — *across the first capsule-native
multi-agent-coordination method that combines explicit-capsule
passing with audited proxies for the LatentMAS direction
(collective KV pooling / latent hidden-state transfer / super-
token side channels): the W22 family ships
:class:`SchemaCapsule` (closed-vocabulary type schema, content-
addressed and shared once per session), :class:`SharedReadCache`
(CID-keyed write-once-read-many proxy for the LatentMAS shared-KV
direction), :class:`CachingOracleAdapter` (drop-in for any
:class:`OutsideWitnessOracle`),
:class:`LatentDigestEnvelope` (typed, controller-verified compact
summary of one W21 vote outcome — hash-chained, schema-versioned,
parent-CID-sealed), :func:`verify_latent_digest` (≈ 30-line
controller-side verification with enumerated failure modes), and
:class:`LatentDigestDisambiguator` (the wrapping decoder). On the
new R-69-CACHE-FANOUT regime, W22 strictly reduces visible-tokens-
to-decider over the W21 baseline by **−7 tokens / cell (−14.51 %)
loose, −7 tokens / cell (−16.09 %) tight**, AND records
``cache_tokens_saved_total = 88`` over the bank, AND ties W21
byte-for-byte on ``accuracy_full = 1.000`` — stable across 5/5
``bank_seed`` values; three named falsifiers (R-69-NO-CACHE,
R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) and one backward-compat
anchor (R-69-NO-TRIGGER) make the W22-1 conditionality sharp.
Live LLM transfer (W22-Λ-real, mixtral:8x7b on Mac-1):
visible-tokens savings **+39.08 %**; cache_tokens_saved_total =
120; correctness ratified rate = 0.750 — newly named conjecture
**W22-C-CACHE-AMPLIFICATION** (the cache freezes a probabilistic
LLM oracle's first reply across all matching cells). 633 / 633
prior coordpy tests pass, 32 new W22 tests pass, 10 misc = **675 /
675**. Mac 2 remains unreachable (16th milestone in a row); no
two-Mac sharded inference. SDK v3.23's contribution is the
Phase-69 capsule-latent-hybrid benchmark family
([`vision_mvp/experiments/phase69_capsule_latent_hybrid.py`](vision_mvp/experiments/phase69_capsule_latent_hybrid.py))
plus the W22 surface in
[`vision_mvp/coordpy/team_coord.py`](vision_mvp/coordpy/team_coord.py).
Closes the wire-cost half of the SDK v3.22 W21-C-CALIBRATED-TRUST
conjecture (the *correctness* half remains open). See
[`docs/RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md`](docs/RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md)
for the full SDK v3.23 milestone note.*

**Historical SDK v3.22 reading (preserved for context).** Capsules
were load-bearing
*inside one CoordPy run*, *between agents in a team*, *across the
model-class gradient*, and — most sharply — *across the first
capsule-native multi-agent-coordination method that crosses the
W20-Λ-compromised wall (named SDK v3.21) on a regime where the
wall actually applies: W21-1 strict +1.000 gain on
R-68-MULTI-MAJORITY loose AND tight, with a three-oracle
registered set (``compromised_registry`` first, ``service_graph``,
``change_history``) under default ``quorum_min = 2``; W20 (the
prior strongest method) trusts the first-registered compromised
oracle on every cell and FAILS at 0.000; W21 consults all three
oracles, gold pair receives 2 votes (from the two honest
deterministic oracles), decoy receives 1, quorum forms on gold,
W21 projects to the gold pair; stable across 5/5 alternate
``bank_seed`` values; three named falsifiers
(R-68-MULTI-NO-QUORUM = W21-Λ-no-quorum ties FIFO,
R-68-MULTI-ALL-COMPROMISED = W21-Λ-all-compromised picks decoy
and fails, R-68-MULTI-PARTIAL = W21-Λ-partial abstains at default
``quorum_min = 2``) make the W21-1 conditionality sharp; the
conditional W21-C-PARTIAL-RECOVERY (with override
``quorum_min = 1``) is empirically discharged at 1.000 — the
quorum-strictness trade-off is real. Live LLM transfer
(W21-Λ-real / W21-C-LIVE-WITH-REGISTRY) on Mac-1 mixtral 8x7b
(47B-MoE): registry-anchored regime achieves +1.000 over W20
(four-oracle registry incl. mixtral); coalition regime
(LLM-vote-required, three-oracle registry) cross-model split
sharp — mixtral 8x7b lands gold tokens through the W18/W19
closure on 3/4 cells (W21 = 0.750, +0.750 over W20); gemma2:9b
(9.2B-dense) lands decoy tokens through the closure (W21 = 0.000)
— scale + general knowledge matter for the W21-Λ-real escape on
the LLM-vote-required regime. Backward-compat (W21-3-A / W21-3-B)
preserved byte-for-byte: with ``enabled = False`` OR no oracles
registered, W21 reduces to W19; with ``quorum_min = 1`` and a
single registered honest oracle, W21 ties W20 byte-for-byte on
R-67-OUTSIDE-RESOLVES; 633 / 633 coordpy tests pass (= 585 prior +
48 new W21). Mac 2 remains unreachable (ARP ``incomplete``, 15th
milestone in a row); no two-Mac sharded inference. SDK v3.22's
contribution is the Phase-68 multi-oracle adjudication benchmark
family
([`vision_mvp/experiments/phase68_multi_oracle_adjudication.py`](vision_mvp/experiments/phase68_multi_oracle_adjudication.py))
plus a deterministic, training-free, closed-form
[`TrustWeightedMultiOracleDisambiguator`](vision_mvp/coordpy/team_coord.py)
that wraps the W19 inner with **N independently-replied outside
queries** (one per registered oracle, each bounded by
``max_response_tokens``) and a deterministic per-tag voting rule
under quorum + trust-sum thresholds — the channel single-oracle
methods cannot reach. Closes both SDK v3.21 conjectures
(W20-C-MULTI-ORACLE discharged on R-68-MULTI-MAJORITY;
W20-C-LIVE-WITH-REGISTRY partially discharged on Mac-1 mixtral
8x7b). See
[`docs/RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md`](docs/RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md)
for the full SDK v3.22 milestone note.*

**Historical SDK v3.21 reading (preserved for context).** Capsules
were load-bearing inside one CoordPy run, between agents in a team,
across the model-class gradient, and — most sharply — across the
first capsule-native multi-agent-coordination method that crosses
the W19-Λ-outside wall (named SDK v3.20) on a regime where the
wall actually applies: W20-1 strict +1.000 gain on R-67-OUTSIDE-RESOLVES
loose AND tight, with a deterministic ``ServiceGraphOracle``
registered as the outside information source; every closed-form
bundle-only scorer through W19 ties FIFO at 0.000 by W19-Λ-outside
extended verbatim — W19 abstains via
``W19_BRANCH_ABSTAINED_SYMMETRIC`` because the asymmetric-witness
count is uniform across the candidate set; W20 issues exactly one
targeted hypothesis-conditioned outside query (bounded by
``max_response_tokens``) and projects the answer through the
SAME closed-vocabulary closure W18 / W19 use; stable across 5/5
alternate ``bank_seed`` values, bounded-context honesty preserved
byte-for-byte (``tokens_kept_sum`` byte-identical to W19 on the
tight budget cell; ``n_outside_tokens`` accounted as a strict
additional cost), with three named falsifier regimes —
R-67-OUTSIDE-NONE (``W20-Λ-none`` — abstaining oracle ties FIFO),
R-67-OUTSIDE-COMPROMISED (``W20-Λ-compromised`` — adversarial
oracle trusts decoy and FAILS at 0.000), R-67-JOINT-DECEPTION
(``W20-Λ-joint-deception`` — primary + secondary + oracle ALL
favour decoy; ties W19 at 0.000) — that make the W20-1
conditionality sharp and name oracle integrity as the structural
escape limit. A partial live-LLM ``W20-Λ-real`` probe on Mac-1
``mixtral:8x7b`` (47B-MoE) achieves ``acc_full = 0.750`` (+0.750
over W19) on a fresh live LLM stream; ``qwen2.5-coder:7b`` ties
FIFO at 0.000 — cross-model split honestly named. SDK v3.21's
contribution is the Phase-67 outside-information benchmark family
([`vision_mvp/experiments/phase67_outside_information.py`](vision_mvp/experiments/phase67_outside_information.py))
plus a deterministic, training-free, closed-form
[`OutsideWitnessAcquisitionDisambiguator`](vision_mvp/coordpy/team_coord.py)
that wraps the W19 inner with one targeted oracle consult per
cell when the inner W19 abstains via the symmetric branch — the
channel every prior bundle-only scorer cannot reach. Backward-
compat preserved byte-for-byte: 545 / 545 prior coordpy tests pass
+ 40 new W20 tests = 585 / 585. Mac 2 remains unreachable (ARP
``incomplete``); no two-Mac sharded inference. See
[`docs/RESULTS_COORDPY_OUTSIDE_INFORMATION.md`](docs/RESULTS_COORDPY_OUTSIDE_INFORMATION.md)
for the full SDK v3.21 milestone note.

**Historical SDK v3.20 reading (preserved for context).** Capsules
were load-bearing
*inside one CoordPy run*, *between agents in a team*, *across the
model-class gradient*, *across the model regime × admission
strategy grid on a real-LLM benchmark*, *across cross-role
coherence / corroboration / multi-service / decoder-forcing /
multi-round / open-world-normalisation / producer-protocol /
decoder-context-packing / live-end-to-end-composition /
relational-compatibility-disambiguation axes*, and — most sharply
— *across the first capsule-native multi-agent-coordination
method that crosses the deceptive-ambiguity wall on a regime
where bundle-only relational compatibility is structurally
insufficient: W19-1 strict +1.000 gain on R-66-DECEIVE-NAIVE
loose AND tight (W18 = 0.000 because the round-2 disambiguator
is adversarial — mentions decoy, not gold) AND on
R-66-CONFOUND-RESOLVABLE (W18 abstains at 0.000 because both
gold and decoy are mentioned symmetrically — W19 sees the
asymmetric witness count from secondary specific-tier handoffs
and breaks the tie correctly), stable across 5/5 alternate
``bank_seed`` values, bounded-context honesty preserved byte-for-
byte (``tokens_kept_sum`` byte-identical to W18 on the tight
budget cell), with two named falsifier regimes —
R-66-DECEIVE-TOTAL (no witnesses anywhere; W19-Λ-total fires;
W19 abstains; ties FIFO at 0.000) and R-66-OUTSIDE-REQUIRED
(witnesses are symmetric across gold and decoy; W19-Λ-outside
fires; W19 abstains; ties FIFO at 0.000) — that make the W19-1
conditionality sharp and name the structural limit no
bundle-only closed-form scorer can escape.* SDK v3.20's
contribution is the Phase-66 deceptive-ambiguity-under-trust
benchmark family
([`vision_mvp/experiments/phase66_deceptive_ambiguity.py`](vision_mvp/experiments/phase66_deceptive_ambiguity.py))
plus a deterministic, training-free, closed-form
[`BundleContradictionDisambiguator`](vision_mvp/coordpy/team_coord.py)
that asymmetrically counts independent specific-tier witnesses
per tag (excluding the canonical primary, deduplicated by
``(role, kind, payload_sha)``) — the channel every prior
disambiguator ignored. On R-66-CORROBORATED (the existing R-65
positive anchor recreated under the new framework), W19 ties
W18 byte-for-byte at 1.000 (the W19-3 ratification anchor —
W19 reduces to W18 when no witnesses exist). On R-66-DECEIVE-NAIVE
loose AND tight, W18 fails at 0.000 (trusts adversarial
disambiguator); W19 achieves 1.000, **+1.000 strict separation**
vs W18 and every other capsule baseline. On R-66-CONFOUND-RESOLVABLE,
W18 abstains at 0.000 (symmetric round-2 disambiguator); W19
achieves 1.000, **+1.000 strict separation**. Stable across
5/5 alternate ``bank_seed`` values on every deceptive cell. The
W19 surface is purely additive on top of W18. See
[`docs/RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md`](docs/RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md)
for the full SDK v3.20 milestone note.

The previous milestone (**SDK v3.19**) contributed the **Phase-65
relational-compatibility disambiguation benchmark family + W18
family (first capsule-native multi-agent-coordination method to
cross the symmetric-corroboration wall on a regime where the wall
actually applies)** — see
[`docs/RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md`](docs/RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md).

The milestone before that (**SDK v3.18**) contributed the **Phase-64
fresh-live composition + magnitude-hinted producer protocol +
symmetric-corroboration limit theorem** — see
[`docs/RESULTS_COORDPY_LIVE_COMPOSITION.md`](docs/RESULTS_COORDPY_LIVE_COMPOSITION.md).

**Historical SDK v3.17 reading (preserved for context).** Capsules
were load-bearing
*inside one CoordPy run*, *between agents in a team*, *across the
model-class gradient*, *across the model regime × admission
strategy grid on a real-LLM benchmark*, *across cross-role
coherence / corroboration / multi-service / decoder-forcing /
multi-round / open-world-normalisation / producer-protocol /
decoder-context-packing axes*, and — most sharply — *across the
first end-to-end composition where the producer-side W14
ambiguity-preservation layer and the decoder-side W15 capsule
context-packing layer must fire jointly (W16-1 strict +1.000
synthetic gain; W16-Λ-real-replay strict +0.500 gain on recorded
``qwen2.5:14b-32k`` bytes — the first end-to-end real-LLM strict
advance over the strongest non-composed baseline in the
programme).* SDK v3.17's contribution is the
**Phase-63 composed end-to-end W14 + W15 benchmark**
([`vision_mvp/experiments/phase63_composed_real_llm.py`](vision_mvp/experiments/phase63_composed_real_llm.py))
plus the ``OllamaReplayExtractor`` for honest replay over
recorded real-LLM bytes. The headline (n=8 saturated × 5 seeds
on R-63-COMPOSED-TIGHT, ``K_auditor=12, T_auditor=256, T_decoder=24,
bank_seed=11``): pairing the W14 ``StructuredProducerProtocol`` +
W15 ``AttentionAwareBundleDecoder`` simultaneously achieves
``accuracy_full = 1.000`` while every non-composed baseline
collapses to 0.000 — **+1.000 strict separation** vs the
W14-only-budgeted baseline (FIFO-packed-W13 on the structured-prompt
stream) and the W15-only-without-W14 baseline (AttentionAware on
the naive-prompt stream); stable across 5/5 alternate
``bank_seed`` values. On the recorded Phase-61 ``qwen2.5:14b-32k``
bytes (``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``,
n=8 × 24 producer calls, byte-stable) at ``T_decoder = 14`` the
composition delivers a **+0.500 strict gain** over the FIFO-packed-W14-only
baseline — the **first end-to-end real-LLM strict advance** over the
strongest non-composed baseline. The W16-Λ-compose anchor on
R-63-naive-tight (mag-filter naive × ``T_decoder = 24``) shows
W14-Λ-prompt and W15-Λ-budget compose multiplicatively: every
capsule strategy ties FIFO at 0.000 when *both* upstream emission
AND downstream retention fail. The W16-Λ-degenerate falsifier on
R-63-degen-budget (``T_decoder = 2``) confirms the conditionality.
Backward-compat (W16-3) preserved byte-for-byte: 442/442 prior
tests pass; the runtime contract is byte-for-byte unchanged. The
Mac-1 endpoint at 192.168.12.191:11434 was offline at milestone
capture time (``HTTP=000``), so a fresh live LLM probe
(W16-C-LIVE-OLLAMA) is conjectural. See
[`docs/RESULTS_COORDPY_COMPOSED_REAL_LLM.md`](docs/RESULTS_COORDPY_COMPOSED_REAL_LLM.md)
for the full SDK v3.17 milestone note.

The previous milestone (**SDK v3.16**) contributed the **Phase-62
attention-aware capsule context packing benchmark** — see
[`docs/RESULTS_COORDPY_ATTENTION_AWARE.md`](docs/RESULTS_COORDPY_ATTENTION_AWARE.md).
The previous milestone (**SDK v3.15**) contributed the **Phase-61
producer-side ambiguity-preservation benchmark** — see
[`docs/RESULTS_COORDPY_PRODUCER_AMBIGUITY.md`](docs/RESULTS_COORDPY_PRODUCER_AMBIGUITY.md).
The CoordPy single-run product runtime contract is byte-for-byte
unchanged from SDK v3.16.

---

**Historical SDK v3.8 reading (preserved for context).** Capsules
are load-bearing
*inside one CoordPy run*, *between agents in a team*, *across the
model-class gradient*, *across the model regime × admission
strategy grid on a real-LLM benchmark*, and now — most sharply —
*across a deterministic cross-role cohort-coherence benchmark
where capsule structure provides a strict +1.000 ``accuracy_full``
advantage over substrate FIFO under stated bench properties.*
SDK v3.8's contribution is the **Phase-54 cross-role
cohort-coherence benchmark**
([`vision_mvp/experiments/phase54_cross_role_coherence.py`](vision_mvp/experiments/phase54_cross_role_coherence.py))
plus a new admission policy
[`CohortCoherenceAdmissionPolicy`](vision_mvp/coordpy/team_coord.py).
The headline (n=10 saturated, K_auditor=4, bank_seed=11):
``capsule_cohort_buffered`` (pre-fitted plurality from the
candidate stream's payloads) achieves ``accuracy_full = 1.000``
while substrate FIFO, ``capsule_fifo``, ``capsule_priority``,
``capsule_coverage``, and ``capsule_cohort_streaming`` all
produce ``accuracy_full = 0.000`` — a **+1.000** structural win,
stable across 5/5 alternate ``bank_seed`` values. The win is
**conditional** on the bench having a stated *gold-plurality*
property (gold service has strictly more auditor-routed
candidates than any decoy service) plus surplus
(``|candidates| > K_auditor``) plus foreign-service decoys.
Without these conditions (e.g., on Phase-53 where the real LLM
emits a clean below-budget stream), substrate FIFO is unbeatable
by construction (W7-1). The Phase-53 / Phase-54 dichotomy is
itself a sharp pair of theorems: the streaming variant of cohort
coherence is unstable under arrival permutation (W7-1-aux); the
buffered variant is the load-bearing policy. The capsule-team
lifecycle audit (T-1..T-7) holds across every Phase-54 cell. See
[`docs/RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md`](docs/RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md)
for the full SDK v3.8 milestone note. The CoordPy single-run
product runtime contract is byte-for-byte unchanged from SDK
v3.7.

The previous milestone (**SDK v3.7**) contributed the
**Phase-53 stronger-model multi-agent benchmark**
([`vision_mvp/experiments/phase53_scale_vs_structure.py`](vision_mvp/experiments/phase53_scale_vs_structure.py)),
which replaces Phase-52's deterministic producer-role extractor
with a real-LLM extractor (qwen2.5:14b-32k, qwen3.5:35b on Mac 1
Ollama) and decomposes accuracy across (model regime × admission
strategy). The headline result (n=5 saturated, K_auditor=4):
every fixed admission strategy — substrate, capsule_fifo,
capsule_priority, capsule_coverage — achieves
``accuracy_full = 0.800`` in every model regime; only
``capsule_learned`` varies, scoring 0.400 on synthetic and 14B
and recovering to 0.800 on 35B. **``structure_gain`` is
*non-positive* at every regime tested** (-0.4 / -0.4 / 0.0);
**``scale_gain[capsule_learned] = +0.4`` while
``scale_gain[fixed] = 0.0``**; cross-(14B, 35B) candidate-kind
TVD = 0.167. The capsule-team lifecycle audit (T-1..T-7) holds
60/60 across (regime × capsule strategy × scenario) — Theorem
**W6-1**. The honest reading: the SDK v3.5 conjecture **W4-C1**
(learned admission policy beats fixed) is **conditionally
falsified** out-of-distribution on the real-LLM regime — it
holds on its anchor distribution (Phase-52 synthetic+noise
default config) but loses to FIFO on the cleaner real-LLM
candidate stream. Scale closes a *structure deficit* (created by
OOD over-rejection of the synthetic-trained scorer), **not** a
*structure surplus*. **The capsule layer's load-bearing
contribution at this benchmark is the lifecycle audit, not
admission policy gains.** Substrate FIFO is a stronger baseline
than the W4 family suggested when the LLM is the producer (the
LLM does its own implicit filtering). **Mac 2 is still offline**
(192.168.12.248 ARP "incomplete"); **no two-Mac sharded inference
ran in this milestone** — the
[`MLXDistributedBackend`](vision_mvp/coordpy/llm_backend.py)
integration boundary is byte-for-byte unchanged from SDK v3.6
and waits for the runbook
([`docs/MLX_DISTRIBUTED_RUNBOOK.md`](docs/MLX_DISTRIBUTED_RUNBOOK.md))
when Mac 2 returns. The strongest model class actually exercised
is single-Mac qwen3.5:35b (36 B-MoE Q4) on Mac 1 Ollama. See
[`docs/RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md`](docs/RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md)
for the full SDK v3.8 milestone note,
[`docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`](docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md)
for the full SDK v3.7 milestone note,
[`docs/archive/coordpy-milestones/RESULTS_COORDPY_DISTRIBUTED.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_DISTRIBUTED.md)
for SDK v3.6 (cross-LLM parser-boundary + integration boundary),
and
[`docs/archive/coordpy-milestones/RESULTS_COORDPY_TEAM_COORD.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_TEAM_COORD.md)
for SDK v3.5 (multi-agent capsule-coordination research slice).
The CoordPy single-run product runtime contract is byte-for-byte
unchanged.

Up through **SDK v3.5**, the multi-agent capsule coordination
**research slice**
([`vision_mvp/coordpy/team_coord.py`](vision_mvp/coordpy/team_coord.py)
+ [`vision_mvp/coordpy/team_policy.py`](vision_mvp/coordpy/team_policy.py))
added three closed-vocabulary capsule kinds (`TEAM_HANDOFF`,
`ROLE_VIEW`, `TEAM_DECISION`) and a `TeamCoordinator` that drove
one coordination round end-to-end, with a mechanically-checked
`audit_team_lifecycle` over invariants T-1..T-7 (Theorem **W4-1**).
Theorems W4-2 (proved-conditional: coverage-implies-correctness)
and W4-3 (proved-negative: per-role budget below the role's
causal-share floor cannot be rescued by *any* admission policy)
anchor the team-level mechanism formally. The W4-C1 conjecture
(learned admission policy beats the strongest fixed baseline) is
empirical-positive on its synthetic+noise anchor distribution
but is now annotated as **conditional** in
[`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md): SDK v3.7
empirically falsifies it out-of-distribution on the real-LLM
regime. SDK v3.8 supersedes the W4-C1 framing entirely with a
deterministic, training-free cross-role admission rule
(``CohortCoherenceAdmissionPolicy``) that produces a strict
+1.000 ``accuracy_full`` advantage over FIFO under named bench
conditions (W7-2). Honest-reading rules in
[`docs/HOW_NOT_TO_OVERSTATE.md`](docs/HOW_NOT_TO_OVERSTATE.md).

Up through **SDK v3.4 (April 2026)**, capsules drive execution all the
way through the LLM byte boundary. Run-boundary stages
(profile / readiness / sweep_spec / sweep_cell / provenance /
artifact / run_report) seal capsules in flight as before
(W3-32..W3-35). Inside every LLM-backed cell, the end-to-end
inner-loop chain is **five typed sealed capsules**:

  `PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL → TEST_VERDICT`

The PROMPT capsule (parent: SWEEP_SPEC, **W3-42**) records the
prompt's SHA-256 + byte length + bounded text snippet; the
LLM_RESPONSE capsule (parent: PROMPT, **W3-43**) records the
response's SHA-256 + length + snippet + elapsed milliseconds;
the PARSE_OUTCOME capsule on the LLM-backed path parents on
both SWEEP_SPEC and the upstream LLM_RESPONSE (**W3-44**), with
coordinate consistency mechanically verified by audit rule
**L-11** (**W3-45** — soundness of the eleven-rule audit). A new
**in-process synthetic-LLM mode**
(`SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)`) lets
the full chain run end-to-end in CI without an Ollama endpoint.
The cross-model parser-boundary research (**W3-C6**, empirical)
reports PARSE_OUTCOME failure-kind Total Variation Distance up
to **1.000** across the calibrated synthetic distribution
library and parser-mode (strict→robust) shift up to **1.000**
on `synthetic.unclosed`.

The SDK v3.3 layer is unchanged: PARSE_OUTCOME capsule
(W3-39), eight-rule lifecycle audit (W3-40, now extended to
eleven rules in W3-45), and `RunSpec(deterministic=True)` opt-in
that collapses the full capsule DAG byte-for-byte across runs
(W3-41). The meta-artefact boundary
(`product_report.json` / `capsule_view.json` / `product_summary.txt`)
remains a sharp circularity theorem (W3-36) with a constructive
detached `META_MANIFEST` witness in a secondary ledger.
`coordpy-capsule verify` recomputes the chain from on-disk header
bytes (W3-37) and re-hashes every artefact at audit time (W3-38).
See
[`docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md)
for the SDK v3.4 milestone note,
[`docs/archive/coordpy-milestones/RESULTS_COORDPY_DEEP_INTRA_CELL.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_DEEP_INTRA_CELL.md)
for SDK v3.3,
[`docs/archive/coordpy-milestones/RESULTS_COORDPY_INTRA_CELL.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_INTRA_CELL.md)
for SDK v3.2, and
[`docs/archive/coordpy-milestones/RESULTS_COORDPY_CAPSULE_NATIVE.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_CAPSULE_NATIVE.md)
for the SDK v3.1 run-boundary slice. For the unified paper-grade
write-up see
[`papers/coordpy_capsule_native_runtime.md`](papers/coordpy_capsule_native_runtime.md).

CoordPy is the first shipped product from the **Context Zero** research
programme on per-agent minimum-sufficient context.

> ### The one-line mental model
>
> Traditional eval harnesses and agent frameworks pass *strings*
> between roles — prompts, JSON dicts, log lines. CoordPy doesn't.
> Every unit of coordination that crosses a boundary in CoordPy is a
> **`ContextCapsule`**: it has a SHA-256 content-address (`cid`), a
> typed `claim_kind` from a closed vocabulary, an explicit
> `lifecycle` (PROPOSED → ADMITTED → SEALED), an explicit
> `CapsuleBudget` (tokens / bytes / rounds / witnesses / parents),
> a parent-CID DAG, and a hash-chained audit history. "Context" in
> CoordPy is not text — it is an object with identity, type,
> lifecycle, budget, and proof. SDK v3.1 lifted that contract from
> *audit description* to *runtime gate* on the run-boundary spine;
> SDK v3.2 extended the gate past the cell boundary into the inner
> parse→apply→test loop and formalised the meta-artefact circularity
> as a sharp limitation theorem with a constructive detached-witness
> boundary. SDK v3.3 extended the gate one further structural
> layer with a sub-intra-cell PARSE_OUTCOME capsule (parent:
> SWEEP_SPEC, W3-39), a runtime-checkable lifecycle audit (W3-40),
> and deterministic-mode replay collapsing the full DAG
> byte-for-byte across runs (W3-41). **SDK v3.4 extends the gate
> one further structural layer to the LLM byte boundary itself**:
> PROMPT and LLM_RESPONSE capsules (W3-42 / W3-43), the
> PARSE_OUTCOME → LLM_RESPONSE chain coordinate consistency
> theorem (W3-44, mechanically checked by audit rule L-11), the
> extended audit soundness over eleven invariants (W3-45), and
> a synthetic-LLM mode that lets the full chain run in CI
> end-to-end. See
> [`docs/archive/coordpy-milestones/RESULTS_COORDPY_CAPSULE.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_CAPSULE.md)
> for the contract (C1..C6),
> [`docs/archive/coordpy-milestones/RESULTS_COORDPY_CAPSULE_NATIVE.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_CAPSULE_NATIVE.md)
> for the v3.1 run-boundary slice (W3-32..W3-35),
> [`docs/archive/coordpy-milestones/RESULTS_COORDPY_INTRA_CELL.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_INTRA_CELL.md)
> for the v3.2 intra-cell + detached witness milestone
> (W3-32-extended / W3-36 / W3-37 / W3-38),
> [`docs/archive/coordpy-milestones/RESULTS_COORDPY_DEEP_INTRA_CELL.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_DEEP_INTRA_CELL.md)
> for the v3.3 deeper-slice / audit / determinism milestone
> (W3-39 / W3-40 / W3-41), and
> [`docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md`](docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md)
> for the v3.4 PROMPT / LLM_RESPONSE / synthetic mode /
> parser-boundary research milestone
> (W3-42 / W3-43 / W3-44 / W3-45 / W3-C6). Canonical theorem registry:
> [`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md). Canonical
> research-status:
> [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md). How not to
> overstate any claim:
> [`docs/HOW_NOT_TO_OVERSTATE.md`](docs/HOW_NOT_TO_OVERSTATE.md).

[![status](https://img.shields.io/badge/status-beta-blue.svg)](#)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)

> **New to this repo?** Read [`docs/START_HERE.md`](docs/START_HERE.md) first —
> it is the canonical one-pass orientation: what Context Zero is, what CoordPy
> is, what is core substrate, what is product surface, what is boundary, and
> what is research-grade.
>
> **Naming.** `Context Zero` is the research programme — theorems, phase
> shards, and the 72-framework theoretical survey (see `vision_mvp/RESULTS_PHASE*.md`
> for the active phase diary, and the archived pre-CoordPy theory volumes
> under [`docs/archive/pre-coordpy-theory/`](docs/archive/pre-coordpy-theory/)).
> `CoordPy` is the
> shipped SDK/runtime under `vision_mvp/coordpy/`, imported as
> `from vision_mvp import coordpy`. The older `vision_mvp/product/` modules
> remain importable for backwards compatibility but are **not** the public
> contract; see the [Stability matrix](#stability-matrix).
>
> CoordPy is **not** the whole research programme and **not** a universal
> agent platform. It is a context-capsule runtime with a narrow, stable
> product surface: profile-driven SWE-bench-Lite-shape evaluation runs
> with a stable report schema, a CI gate, a provenance manifest, and a
> capsule graph on every run.
>
> **CASR** (Causal-Abstraction Scale-Renormalized Routing) is the original
> substrate proposal from the programme; it lives in `vision_mvp.core.*` as
> research-grade code and informs CoordPy's bounded-context guarantees, but
> it is not itself the product identity.

> Most multi-agent AI frameworks (AutoGen, CrewAI, LangGraph, …) cap out at
> around 10-100 agents because every agent has to read every other agent's
> output each round — context grows like O(N²). **Context Zero** ships a
> coordination layer whose per-agent context grows like **O(log N)**, so the
> same team design scales to 10 000, 100 000, or more agents without
> collapsing under its own context.

> **What this project is — and what it is not.** Context Zero is a *context
> substrate for teams of agents collaborating on a task*. It is NOT a repo
> knowledge-graph tool (Graphify-style or otherwise); those tools represent
> a corpus for a single assistant to traverse, and their object of study is
> the graph. Our object of study is the per-role, per-round information flow
> across a whole team — *who should know what, when, why, and with what loss
> profile*. Graph/index techniques show up *as one layer* in our stack
> (retrieval, call graph, interprocedural analysis); they are not the
> project's identity. See
> [``docs/context_zero_master_plan.md`` § 1.5](docs/context_zero_master_plan.md)
> for the durable version of this distinction, and
> [``vision_mvp/RESULTS_PHASE31.md``](vision_mvp/RESULTS_PHASE31.md)
> Theorem P31-5 for the formal statement (a single-agent corpus compressor
> cannot match a team's bounded-context guarantee by any universal
> compression; the team's guarantee is a property of *role-conditioned
> information flow*).

Empirically validated from **N=10** up to **N=100 000** agents with peak
per-agent context equal to ⌈log₂ N⌉ *exactly* — and **1 000 real local-LLM
agents coordinating on one laptop in 54 seconds** with 100 % accuracy on a
factual question, using **3 750 × fewer tokens** than naive broadcast would
require.

---

## SDK reference — full quick start

The landing-page [Quickstart](#quickstart) covers the minimal path.
This section walks through the full SDK surface — capsule
construction modes, on-disk artifact set, CLIs, who CoordPy is for,
and how to extend it.

CoordPy is a context-capsule runtime. One `RunSpec` in, one
reproducible, provenance-stamped **capsule graph** out.

```python
from vision_mvp.coordpy import RunSpec, run

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/cz-smoke"))
assert report["readiness"]["ready"]
assert report["provenance"]["schema"] == "coordpy.provenance.v1"
# SDK v3 — every run ships a sealed capsule graph.
cv = report["capsules"]
assert cv["schema"] == "coordpy.capsule_view.v1"
assert cv["chain_ok"]
print(f"RUN_REPORT CID = {cv['root_cid']}")
print(report["summary_text"])
```

### Every run is a sealed capsule DAG, built in flight

```python
from vision_mvp.coordpy import (
    RunSpec, run, CONSTRUCTION_IN_FLIGHT, CapsuleKind,
)

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/cz-smoke"))
# Capsule view tagged with its construction mode:
assert report["capsules"]["construction"] == CONSTRUCTION_IN_FLIGHT
# Every proposed capsule sealed; no failures.
assert report["capsules"]["in_flight_stats"]["n_failed"] == 0

# Types you'll see in a smoke run:
#   PROFILE           (root of the DAG)
#   READINESS_CHECK   (parent: PROFILE)
#   SWEEP_SPEC        (parent: PROFILE)
#   SWEEP_CELL × N    (parent: SWEEP_SPEC)
#   PROVENANCE        (parent: PROFILE)
#   ARTIFACT × K      (content-addressed at write time, parent: source capsule)
#   RUN_REPORT        (parents: ALL of the above)
```

For third parties who have a `product_report` dict from outside the
runtime, the post-hoc fold is still available:

```python
from vision_mvp.coordpy import build_report_ledger
ledger, run_cid = build_report_ledger(report)
assert ledger.verify_chain()
```

The two paths produce CID-equivalent ledgers on non-ARTIFACT kinds
(Theorem W3-34); ARTIFACT capsules from a capsule-native run carry
real SHA-256 hashes, while the post-hoc fold's ARTIFACT capsules
carry `None`.

The RUN_REPORT capsule's CID is the durable identifier for the run —
send someone that CID plus `product_report.json` and they can
reproduce every upstream capsule, verify the hash chain end-to-end,
and know the bytes haven't drifted.

### Every run emits seven artifacts

  * `product_report.json` — schema `phase45.product_report.v2`,
    includes the `capsules` block (`coordpy.capsule_view.v1`)
  * `capsule_view.json` — the sealed capsule graph, on disk
  * `product_summary.txt` — human-readable summary with a
    `capsules` line (kind histogram + chain_ok + root CID)
  * `readiness_verdict.json` — per-row readiness verdicts
  * `provenance.json` — git SHA, package version, Python, platform,
    profile, model, endpoint, sandbox, input JSONL + SHA-256, argv,
    timestamp, artifact list (schema `coordpy.provenance.v1`)
  * `sweep_result.json` — the executed-sweep block when the
    profile's sweep ran in-process (mock or real-LLM acknowledged)
  * `meta_manifest.json` — **new in SDK v3.2** — the detached
    META_MANIFEST witness for the meta-artefacts above, in a
    secondary ledger; carries on-disk SHA-256 of
    `product_report.json` / `capsule_view.json` /
    `product_summary.txt` plus the primary `root_cid` and
    `chain_head` (Theorem W3-36 — meta-artefact circularity is
    sharp; the manifest is the strongest authentication achievable
    one trust hop beyond the primary view)

### CLIs

```bash
coordpy --profile local_smoke --out-dir /tmp/cz-smoke
coordpy-ci --report /tmp/cz-smoke/product_report.json --min-pass-at-1 1.0
# Capsule-graph inspection (new in SDK v3, strengthened in v3.2):
coordpy-capsule view   --report /tmp/cz-smoke/product_report.json
coordpy-capsule verify --report /tmp/cz-smoke/product_report.json
coordpy-capsule cid    --report /tmp/cz-smoke/product_report.json
```

`coordpy-capsule verify` (v3.2) runs four independent on-disk
checks and prints each verdict line plus a final `verdict = OK / BAD`:

```
chain_ok_embedded         = True    # writer's self-report
chain_recompute_embedded  = True    # we recompute the chain
chain_recompute_on_disk   = True    # ...from on-disk view bytes
on_disk_view_agrees       = True    # cross-check
artifacts_on_disk         = OK  (3/3 matched, 0 drifts, 0 missing)
meta_manifest_on_disk     = OK  (3/3 matched, 0 drifts, 0 missing)
verdict                   = OK
```

A drift in any check fails the verify (exit code 3) and prints the
specific path / sealed SHA / on-disk SHA tuple — the audit knows
exactly which file lied.

The SDK public surface is contract-tested in
`vision_mvp/tests/test_coordpy_public_api.py` and the Capsule Contract
(invariants C1..C6) is tested in
`vision_mvp/tests/test_coordpy_capsules.py` — any rename or removal is a
breaking change and requires bumping `coordpy.SDK_VERSION` (currently
`coordpy.sdk.v3.4`).

### Who CoordPy is for

  * Research engineers running **profile-driven evaluations** on
    SWE-bench-Lite-shape banks who need a reproducible, provenance-
    stamped, capsule-graph artifact trail instead of ad-hoc scripts.
  * Teams wiring **CI gates** over evaluation quality (`coordpy-ci`
    consumes the report and emits a pass/fail verdict with explicit
    blocker strings; `coordpy-capsule verify` re-hashes the capsule
    chain so a third party can confirm the report bytes haven't
    drifted).
  * Operators who want to swap **profiles** (`local_smoke`,
    `bundled_57`, `aspen_mac1_coder`, `aspen_mac2_frontier`,
    `public_jsonl`, …) without editing core code.
  * Downstream framework authors who want to **lift their own
    substrate into capsules** — the `capsule_from_handle`,
    `capsule_from_handoff`, `capsule_from_sweep_cell` adapters
    let you expose your own typed objects under the same
    Contract, and `CapsuleLedger.admit_and_seal()` makes your
    artefacts composable with CoordPy's.

CoordPy is **not** (yet) for: arbitrary multi-agent orchestration,
agent-platform building, or non-SWE evaluation shapes. Those are
Context Zero research-programme territory, not CoordPy product
territory.

### How to extend CoordPy (current state, honest)

  * **New profiles**: add a declarative entry to
    `vision_mvp/product/profiles.py`. The `profiles` module is part
    of the SDK contract; a profile is a frozen dict with stable keys.
  * **New sandbox backends, new task banks, new reporting sinks**:
    the extension surface for these is *not yet stable*. The planned
    entry-point / registry-based plugin system is Slice 2 (see
    `docs/context_zero_master_plan.md` § CoordPy SDK follow-ups).
    Today, extending these requires editing core modules; this is
    a known limitation marked **boundary** in the stability matrix.

---

## Stability matrix

| Layer | Scope | Stability | Import path |
|---|---|---|---|
| **CoordPy SDK** — `RunSpec`, `run`, `SweepSpec`, `run_sweep`, `HeavyRunNotAcknowledged`, `CoordPyConfig`, `build_manifest`, `profiles`, `report`, `ci_gate`, `import_data`, `extensions`, schema constants. SDK v3.3 adds `RunSpec.deterministic` opt-in. **SDK v3.4 adds `SweepSpec(mode="synthetic", synthetic_model_tag=...)`** — in-process synthetic-LLM mode for CI-runnable end-to-end chain exercise. | The public product contract | **Stable v3.4** (contract-tested) | `vision_mvp.coordpy` |
| **Context Capsule primitives** — `ContextCapsule`, `CapsuleKind`, `CapsuleLifecycle`, `CapsuleBudget`, `CapsuleLedger`, `CapsuleView`, `render_view`, `build_report_ledger`, `capsule_from_*` adapters | The **load-bearing SDK abstraction**: every cross-boundary artefact is a capsule | **Stable v1** (contract-tested: invariants C1..C6) | `vision_mvp.coordpy.capsule` (re-exported from `vision_mvp.coordpy`) |
| **Capsule-native runtime** — `CapsuleNativeRunContext`, `seal_and_write_artifact`, `ContentAddressMismatch`, `CONSTRUCTION_IN_FLIGHT`/`CONSTRUCTION_POST_HOC`, `RunSpec.capsule_native` + intra-cell `seal_patch_proposal` / `seal_test_verdict` + sub-intra-cell `seal_parse_outcome` (SDK v3.3) + **sub-sub-intra-cell `seal_prompt` / `seal_llm_response` (SDK v3.4)** + detached `seal_meta_manifest` + on-disk `verify_chain_from_view_dict` / `verify_artifacts_on_disk` / `verify_meta_manifest_on_disk` | Capsules drive runtime stage transitions at the run boundary AND inside the inner sweep loop AND on the parser axis AND at the LLM byte boundary; substantive artifacts are content-addressed at write time and re-verifiable at audit time; meta-artefacts are authenticated by a detached META_MANIFEST | **Stable v3.4** (contract-tested: theorems W3-32 / W3-33 / W3-34 / W3-35 / W3-32-extended / W3-36 / W3-37 / W3-38 / W3-39 / W3-40 / W3-41 / W3-42 / W3-43 / W3-44 / W3-45) | `vision_mvp.coordpy.capsule_runtime` (re-exported from `vision_mvp.coordpy`) |
| **Lifecycle audit** — `CapsuleLifecycleAudit`, `LifecycleAuditReport`, `audit_capsule_lifecycle`, `audit_capsule_lifecycle_from_view` (SDK v3.3, **extended in SDK v3.4 to L-9..L-11**) | Mechanically verifies **eleven** lifecycle invariants L-1..L-11 over a finished run (eight from v3.3 + L-9 / L-10 / L-11 covering the PROMPT / LLM_RESPONSE / coordinate-consistency chain). Returns OK / BAD / EMPTY plus typed counterexamples. | **Stable v1.1** (contract-tested: theorems W3-40 / W3-45) | `vision_mvp.coordpy.lifecycle_audit` (re-exported) |
| **CoordPy console scripts** — `coordpy`, `coordpy-import`, `coordpy-ci`, `coordpy-capsule` | CLI surface | **Stable v3** (Slice 3: `coordpy-capsule view / verify / cid`) | `[project.scripts]` |
| **Provenance manifest** — `coordpy.provenance.v1` | Reproducibility artifact | **Stable v1** | `vision_mvp.coordpy.provenance` |
| **Capsule view artifact** — `coordpy.capsule_view.v1` | Sealed capsule graph on disk | **Stable v1** | `capsule_view.json` next to every report |
| **Extension Protocols** — `SandboxBackend`, `TaskBankLoader`, `ReportSink` | Plugin surface | **Stable v1** (runtime-checkable Protocols, `entry_points` discovery) | `vision_mvp.coordpy.extensions` |
| **Unified runtime** — `SweepSpec`, `run_sweep`, `coordpy.sweep.v2` | One execution path for mock + real-executed + real-staged | **Stable v1** | `vision_mvp.coordpy.runtime` |
| **Report / CI-gate schemas** — `phase45.product_report.v2` (v1 accepted), `phase46.ci_verdict.v1`, `phase46.import_audit.v1` | On-disk contract | **Stable** | — |
| **Core substrate** — CASR router, hierarchical router, ledger, exact_ops, role_handoff | Research substrate used *by* CoordPy | **Settled** (proofs + tests) but **research API** | `vision_mvp.core.*` |
| **Legacy product path** — `vision_mvp.product.*` | Pre-Slice-1 import path | **Deprecated-compat** (still works; re-exported by `coordpy`) | `vision_mvp.product` |
| **Docker sandbox** | Untrusted-input isolation | **Available** (backend registered as `coordpy.extensions.get_sandbox("docker")`); **not yet the default** | `vision_mvp.coordpy.extensions` |
| **Docker-first-by-default** for public JSONLs | Slice 3 target | **Boundary / next-slice** (default-flip) | n/a yet |
| **First real out-of-tree plugin** | Slice 3 community target | **Boundary / next-slice** | n/a yet |
| **Research shards** — Phases 1–44 RESULTS_*.md, EXTENDED_MATH_*.md, per-phase experiment scripts, 72-framework survey | The Context Zero research programme | **Research-grade** (empirical or proved per shard; no product-API guarantee) | `vision_mvp.experiments.*`, `vision_mvp.tasks.*`, docs |

See `docs/context_zero_master_plan.md` for the living version of this
matrix and the concrete next-slice follow-ups.

---

## Research substrate quick start (CASR)

The CoordPy SDK is built on top of `CASRRouter` (Causal-Abstraction
Scale-Renormalized Routing) — the research substrate that grounds
CoordPy's bounded-context guarantees. CASR is **not** part of the
CoordPy SDK contract; it is exposed for research use.

```python
from vision_mvp import CASRRouter
import numpy as np

# 1 000 agents, each carrying a 64-dim state vector.
router = CASRRouter(n_agents=1000, state_dim=64, task_rank=10)

# Feed observations each round — get back consensus estimates.
for _ in range(100):
    obs = np.random.randn(1000, 64)     # (N, d) noisy observations
    estimates = router.step(obs)         # (N, d) consensus estimates

print(router.stats)
# {'peak_context_per_agent': 10, 'total_tokens': 1_010_000,
#  'total_messages': 100_100, 'mean_context_per_agent': 10.0,
#  'manifold_dim': 10, 'workspace_size': 10, 'rounds_executed': 100}
```

Peak context per agent = 10 = ⌈log₂ 1000⌉. Workspace size also = 10. Both
independent of the state dimension d=64.

---

## Research-substrate CLI

The legacy `casr` console script (and the equivalent
`python -m vision_mvp ...`) drives the CASR research demos and is
retained for continuity with the Phase-1..44 scripts. The product
CLI is `coordpy` — see [Install](#install) above.

```bash
python -m vision_mvp demo --n 500 --d 32 --rounds 20
python -m vision_mvp scale --n-values 10 50 200 1000 5000
python -m vision_mvp phase 3
python -m vision_mvp test
```

---

## What's in the repo

```
context-zero/
├── README.md                   # you are here
├── LICENSE                     # MIT
├── CHANGELOG.md                # SDK release history
├── ARCHITECTURE.md             # substrate + product architecture
├── pyproject.toml              # installable as coordpy
├── docs/                       # canonical research + product docs (see § Where to read next)
│   ├── START_HERE.md                    # one-pass orientation
│   ├── RESEARCH_STATUS.md               # what is true now, single source of truth
│   ├── THEOREM_REGISTRY.md              # theorem-by-theorem status
│   ├── HOW_NOT_TO_OVERSTATE.md          # do-not-overstate rules
│   ├── CAPSULE_FORMALISM.md             # run-boundary capsule formalism (W3 family)
│   ├── CAPSULE_TEAM_FORMALISM.md        # team-boundary capsule formalism (W4 family)
│   ├── context_zero_master_plan.md      # long-running master plan
│   ├── MLX_DISTRIBUTED_RUNBOOK.md       # two-Mac MLX distributed-inference runbook
│   ├── RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md  # latest milestone (SDK v3.43, final)
│   ├── SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md  # pre-committed success bar (SDK v3.43)
│   └── archive/                         # historical milestones + pre-CoordPy theory (see archive/README.md)
├── papers/                     # paper-grade write-ups (context_as_objects.md is the main paper)
├── examples/                   # short standalone programs (basic consensus, drift tracking, scaling, local LLM, real code review)
└── vision_mvp/                 # working implementation + research diary
    ├── coordpy/                  # public CoordPy SDK (stable contract — see § Stability matrix)
    │   └── __experimental__/   # W22..W42 research surface (under the experimental tuple)
    ├── core/                   # CASR substrate primitives (research-grade)
    ├── tasks/                  # task banks + adapters
    ├── experiments/            # phase 1–89 experiment harnesses (incl. R-69..R-89 benchmark drivers)
    ├── tests/                  # ~1500 substrate + capsule + W22..W42 tests
    ├── RESULTS_PHASE*.md       # phase-by-phase research diary (Phase 1 → 89+)
    └── README.md               # implementation-level README
```

The pre-CoordPy theory volumes (`PROOFS.md`, `EXTENDED_MATH_[1-7].md`,
`OPEN_QUESTIONS.md`, `FRAMEWORK.md`, `EVALUATION.md`, `MVP.md`,
`ROADMAP.md`, `VISION_MILLIONS.md`, `MATH_AUDIT.md`,
`HIERARCHICAL_DECOMPOSITION.md`, `WAVES.md`) and older CoordPy
milestone notes (`RESULTS_COORDPY_*.md` SDK v3.0 → v3.6, the
`RESULTS_CAPSULE_RESEARCH_MILESTONE*.md` series) are intact under
[`docs/archive/`](docs/archive/) — the active scientific position is
in `docs/`, the archive is historical record only.

---

## The empirical story in one table

Measured on the drifting-consensus task (d=64, intrinsic_rank=⌈log₂ N⌉,
noise σ=1.0, drift σ=0.05, 3 seeds averaged).

| N | log₂ N | Peak ctx/agent | Writes/round | Workspace | Steady err |
|---:|---:|---:|---:|---:|---:|
| 50 | 5.6 | **6** | 6 | 6 | 0.22 |
| 200 | 7.6 | **8** | 8 | 8 | 0.25 |
| 1 000 | 10.0 | **10** | 10 | 10 | 0.08 |
| 5 000 | 12.3 | **13** | 13 | 13 | 0.11 |
| 10 000 | 13.3 | **14** | 14 | 14 | 0.09 |
| 50 000 | 15.6 | **16** | 16 | 16 | 0.12 |
| **100 000** | **16.6** | **17** | **17** | **17** | **0.15** |

Three independently-measured metrics — peak context, writes per round,
workspace capacity — **all equal ⌈log₂ N⌉** at every scale tested.

Compare with naive broadcast, which at N=100 000 would use **6 500 000 tokens
of peak per-agent context**. That's a **≈382 000× reduction**.

And for **real LLM agents** (qwen2.5:0.5b via local Ollama), wall time stays
essentially flat across two orders of magnitude of team size:

| N | Wall time | Accuracy | LLM tokens | Naive/vision ratio |
|---:|---:|---:|---:|---:|
| 100 | 46 s | 100 % | 5 669 | 43 × |
| 1 000 | 54 s | 100 % | 8 242 | **3 750 ×** |
| 2 000 | 24 s | 100 % | 5 934 | 15 745 × |
| **5 000** | **46 s** | **100 %** | **7 513** | **76 840 ×** |

At N = 5 000, naive broadcast would cost ~577 million tokens ≈ **333 days** of
continuous decoding on this laptop. The vision stack does the same job in
under a minute.

Reproduce with `python -m vision_mvp.experiments.phase6_llm_1000 --n 5000`.

### And on a real reasoning task

Phase 7: **100 AI agents review actual code and find the critical bug.**
Same CASR machinery, qwen2.5-coder:7b model, 25 specialist reviewer
personas. The team reviews a Python helper with an SQL-injection
vulnerability and produces a structured report:

> **CRITICAL ISSUE:** SQL injection vulnerability due to direct string
> concatenation of user input into the SQL query.

10/10 sampled agents flag SQL injection, all 100 do via NN heuristic,
the synthesis step outputs the report above. See
`vision_mvp/RESULTS_PHASE7.md` for the full trace and
`python -m vision_mvp.experiments.phase7_code_review --n 100 --task sql`
to reproduce.

A second run on a **different bug type** (race condition in a counter)
also succeeded: 50/50 agents correctly identified the thread-safety
issue, synthesis produced the correct structured report. Two tasks, two
distinct bug categories, both 100 % accuracy — the team preserves
*reasoning quality* across task types, not just the original demo.

### Phase 8 — truly distributed task: no single agent can solve it

The phases above are useful scaling demos but don't require
*collaboration* — each agent could independently name the SQL-injection
bug. Phase 8 is the first task where **no single agent has enough
information** to answer. Each of 16 agents sees ONE chunk (~47 words)
of a 757-word fictional incident review. The task: identify the top 3
systemic risks that emerge only from **cross-chunk patterns**.

| Mode | Risks found | What it proves |
|---|---:|---|
| Isolated (1 chunk) | **0 / 3** | Correctly refuses — task really is distributed |
| Oracle (full 757 words, 1 agent) | 2 / 3 | Upper bound with unbounded context |
| **Map-reduce team (16 × 1-chunk agents + synth)** | **2 / 3** | **Matches oracle; each member saw 1/16th of the doc** |

The map-reduce team, where **no single agent saw more than 47 words**,
produced the same quality report as a single agent with the full
document. Both identified vendor concentration (NordAxis across multiple
incidents) and documentation/runbook gaps. This is the qualitative shift
— **collaborative output exceeds any individual member's capability**.

See `vision_mvp/RESULTS_PHASE8.md` for the full trace. Reproduce with:

```bash
python -m vision_mvp.experiments.phase8_mapreduce --n 16
python -m vision_mvp.experiments.phase8_distributed --n 16  # full 3-way compare
```

### Phase 9 — multi-role pipeline + longer-than-context document

Phase 8 showed pooling across chunks. Phase 9 pushes two directions:

**9a — Multi-role pipeline (quant strategy).** 19 agents in 4 distinct
roles: researchers distill notes, market-analysts read individual asset
time-series, strategists combine signals, a PM produces the final
portfolio. Each role has its own prompt and reads different inputs.
Result: team produced a real book with written rationales (+1.05 %
gross on its 5 committed bets, 3/5 hit rate). Too conservative —
FLATted 7 of 12 assets because the PM prompt didn't enforce coverage.
Real end-to-end work across genuinely different roles, honestly modest
score.

**9b — Longer-than-context document.** Fictional 11 000-word / 14 500-
token incident review (~3.5× Ollama's default 4 k context). Single-
agent oracle **TIMED OUT after 300 s**: it simply cannot complete the
task at default settings. Map-reduce team (40 agents, each sees 1/40
of the document) continues running — the distributed approach works
because each chunk fits any single agent.

See `vision_mvp/RESULTS_PHASE9.md` for the full phase-9 writeup and
[`docs/archive/pre-coordpy-theory/MATH_AUDIT.md`](docs/archive/pre-coordpy-theory/MATH_AUDIT.md)
for an honest accounting of which of the 72 frameworks
in the extended math docs are actually in the running code (6 USED,
13 STRUCTURAL, 3 BUILT-not-tested, 50 THEORY-only).

### Phase 10 — The Agent Network: 500 agents collaborating on one task

Phase 9 used a classical DAG pipeline — what CrewAI / LangGraph already do.
Phase 10 is the thing they can't: **hundreds of agents send targeted
messages to each other**, each picking up a piece of one interconnected
task, with per-agent context **bounded regardless of team size**.

The architecture wires in three mechanisms from the math that were
previously theory-only:

1. **Sparse MoE routing** (from Routing Transformer / Mixtral) — each
   agent has a learned key; messages route to top-k recipients by
   clustered cosine-similarity lookup, cost O(√N · d).
2. **Hyperbolic (Lorentz-model) address space** — tree-structured task
   decompositions embed without sibling-subtree crosstalk.
3. **Sheaf H¹ diagnostic** — per-edge discord localizes exactly which
   agents disagree, not whether the team as a whole does.

Plus a shared task board (claim/complete/deps) and per-agent inbox with
capacity limits.

### The scaling result (mock LLM, same 40-subtask interconnected build)

| | **N = 30** | **N = 200** | **N = 500** |
|---|---:|---:|---:|
| Subtasks completed | **40 / 40** | **40 / 40** | **40 / 40** |
| Rounds | 8 | 4 | **4** |
| **Max inbox per agent** | **73** | **42** | **48** |
| Integration score | 1.00 | 1.00 | **1.00** |
| Total inter-agent messages | 232 | 706 | 1 728 |
| Wall (mock) | 0.1 s | 0.2 s | 0.9 s |

**Max per-agent inbox stays bounded at 40–75 across a 17× range of team
size.** That's the thing AutoGen / CrewAI cannot do — their context grows
linearly with team size and breaks past ~100 agents.

Reproduce:
```bash
python -m vision_mvp.experiments.phase10_network --mock --n 500
```

Full phase-10 writeup: `vision_mvp/RESULTS_PHASE10.md`. Design:
[`docs/archive/legacy-progress-notes/AGENT_NETWORK_DESIGN.md`](docs/archive/legacy-progress-notes/AGENT_NETWORK_DESIGN.md).

### Phase 43 — public-style-scale audit, frontier semantic headroom, and the post-parser-recovery semantic taxonomy

Phase 42 closed the parser-compliance layer and shipped the
three-axis attribution surface. The residue on the
``qwen2.5-coder:14b`` cluster run (4/57 failures on every
strategy) was the first purely semantic residue — format-
compliant, byte-matching, structurally-valid patches that
still fail the hidden test. Phase 43 *characterises* that
residue and audits that the programme is ready for public
SWE-bench-Lite drop-in.

* **Part A — Semantic failure taxonomy**
  (``vision_mvp/tasks/swe_semantic_taxonomy.py``). A pure
  deterministic ``classify_semantic_outcome`` that takes
  ``(buggy_source, gold_patch, proposed_patch, error_kind,
  test_passed)`` and returns exactly one label from a closed
  nine-element set (``SEM_OK`` / ``SEM_PARSE_FAIL`` /
  ``SEM_WRONG_EDIT_SITE`` / ``SEM_RIGHT_SITE_WRONG_LOGIC`` /
  ``SEM_INCOMPLETE_MULTI_HUNK`` / ``SEM_TEST_OVERFIT`` /
  ``SEM_STRUCTURAL_SEMANTIC_INERT`` / ``SEM_SYNTAX_INVALID``
  / ``SEM_NO_MATCH_RESIDUAL``). ``SemanticCounter`` aggregates
  per-strategy + pooled histograms with a ``failure_mix``
  helper that normalises by non-``SEM_OK`` total so
  cross-model comparisons are composition-level.
* **Part B — Public-style loader self-test**
  (``phase43_frontier_headroom.verify_public_style_loader``).
  Round-trips every instance in a target JSONL through
  ``load_jsonl_bank → SWEBenchAdapter.from_swe_bench_dict
  → parse_unified_diff → apply_patch → run_patched_test``
  under the strict matcher. On the bundled 57-instance bank:
  **57 / 57 oracle saturation** (Theorem P41-2 reproduced at
  scale). The externalisation gap to public SWE-bench-Lite
  is now a ``--jsonl <path>`` drop-in.
* **Part C — Frontier semantic-headroom run.** The ASPEN
  cluster runs ``qwen3.5:35b`` (36B MoE) on mac1 at
  ``n_distractors = 6`` over the full 57-instance bank and
  on mac2 at ``n_distractors ∈ {0, 24}`` over a 20-instance
  subset for the bounded-context stress. The 35B needs
  ``--think off`` to free its 600-token output budget from
  internal thinking; the ``LLMClient.think`` Phase-43
  extension threads this through Ollama's ``/api/generate``
  body.
* **Part D — Analysis driver**
  (``phase43_frontier_headroom``). Ingests Phase-42-shape
  artifacts, re-derives per-cell semantic labels, and emits
  the cross-model summary JSON. Analysis-only, no LLM
  dependency.

| metric (Phase-43 canonical cell: parser=robust/nd=6/apply=strict) | pass@1 (N, R, S) | S−N gap | dominant residue label | residue mix |
|---|:-:|:-:|---|---|
| ``qwen3.5:35b`` (36B MoE, cluster mac1)      | **0.965 / 0.965 / 0.965 (55/57)** | **0.0 pp** | SEM_WRONG_EDIT_SITE* | 100 % wrong_site* (§ D.7 caveat: test_exception + test_assert merged) |
| ``qwen2.5-coder:14b`` (cluster mac1)        | 0.930 / 0.930 / 0.930 | **0.0 pp** | SEM_WRONG_EDIT_SITE | 50 % wrong_site / 25 % multi_hunk / 25 % no_match |
| ``qwen2.5-coder:7b`` (localhost)            | 0.842 / 0.842 / 0.842 | **0.0 pp** | SEM_WRONG_EDIT_SITE | 56 % wrong_site / 33 % no_match / 11 % syntax |
| ``gemma2:9b`` (28 subset)                    | 0.857 / 0.857 / 0.857 | **0.0 pp** | SEM_WRONG_EDIT_SITE | 50 % wrong_site / 25 % no_match / 25 % syntax |
| ``qwen2.5:14b-32k`` (general, cluster mac2)  | 0.544 / 0.544 / 0.526 | 1.8 pp      | SEM_SYNTAX_INVALID  | 52 % syntax / 46 % no_match / 3 % multi_hunk  |

**Three new theorems** (P43-1 bounded-context preservation
on the external-validity bank — substrate 205.9 tokens flat
across the full cross product; P43-2 post-parser-recovery
semantic residue is structurally classifiable — total,
exhaustive, deterministic; P43-3 semantic-ceiling separation
on coder-finetuned models at N ≥ 50 — gap-zero + strategy-
invariant composition + training-mix-indexed dominance) and
**four new conjectures** (C43-1 frontier coder closes
wrong-edit-site without re-opening the substrate gap; C43-2
residue composition is training-mix-indexed, not parameter-
count-indexed; C43-3 substrate bounded-context invariant is
model-independent; C43-4 semantic residue does not decompose
under existing substrate primitives).

The ``qwen3.5:35b`` cluster run surfaced a Phase-43
regression shape: the 35B closes the NEW block with ``<<``
instead of ``<<<`` at end-of-generation. Without the
Phase-43 fix, the ``RECOVERY_CLOSED_AT_EOS`` heuristic kept
the trailing ``<<`` in the NEW payload, producing syntax
errors on every instance (171/171). The one-pattern fix in
``_strip_trailing_prose`` (``\n\s*<{2,4}\s*\Z``) strips
partial or full trailing delimiters; byte-safe under
Theorem P42-2.

**Headline**: on every coder-class model tested at N ≥ 50 the
substrate-vs-naive pass@1 gap is **0 pp**, and the frontier
``qwen3.5:35b`` MoE **beats the 14B-coder** at 0.965 vs 0.930
— direct evidence for Conjecture C43-1 (frontier model closes
the semantic residue without re-opening the substrate gap).
The substrate's durable claim is *bounded active context per
role* — substrate prompt is flat at **205.9 tokens** across
the full parser × matcher × distractor cross product on the
full 57-instance bank, while naive grows 197 → 527 tokens.
The remaining residue is *model-shaped* (wrong-edit-site on
coder models, syntax-invalid on general-purpose) and neither
the parser nor the matcher nor the substrate can close it
without re-generating the patch.

Reproduce:
```bash
# Phase-43 cluster run — qwen3.5:35b on mac1, full 57-instance bank
python -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen3.5:35b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --think off --max-tokens 600 --llm-timeout 600 \
    --out vision_mvp/results_phase43_parser_35b_moe_mac1.json

# Phase-43 analysis — cross-model semantic summary
python -m vision_mvp.experiments.phase43_frontier_headroom \
    --artifacts \
        vision_mvp/results_phase42_parser_14b_coder.json \
        vision_mvp/results_phase42_parser_7b_coder.json \
        vision_mvp/results_phase42_parser_9b_gemma.json \
        vision_mvp/results_phase42_parser_14b_general.json \
        vision_mvp/results_phase43_parser_35b_moe_mac1.json \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out vision_mvp/results_phase43_frontier_summary.json
```

### Phase 44 — raw-text semantic residue capture, refined taxonomy, and public-SWE-bench-Lite drop-in readiness

Phase 43 characterised the post-parser-recovery residue with a
nine-label closed vocabulary, but every Phase-43 result was
derived from a *sentinel* proposed-patch (§ D.7) because the
Phase-42 artifact schema does not preserve raw LLM output.
Phase 44 removes that limitation, runs the strongest practical
coder-class cell on the ASPEN cluster with raw capture on, and
promotes the public-SWE-bench-Lite drop-in claim from
documented to validated code.

* **Part A — Raw-text capture** (``vision_mvp/tasks/swe_raw_capture.py``).
  ``RawCaptureRecord`` / ``RawCaptureStore`` with schema version
  ``phase44.v1``. Each record carries raw LLM bytes + SHA-256,
  the ``ParseOutcome`` dict, proposed + applied ``(old, new)``
  pairs, and the patched-source SHA-256.
  ``make_capturing_generator`` wraps a bridge generator or a
  fresh ``llm_call`` and plumbs the raw text into the store
  while preserving the Phase-42 LLM-output cache discipline.
  Opt-in — every pre-Phase-44 path runs unchanged.

* **Part B — Refined semantic taxonomy (v2 classifier).** Five
  new sub-labels partition the Phase-43 coarse buckets when raw
  bytes are available: ``SEM_RIGHT_FILE_WRONG_SPAN``
  (anchored in right file, wrong span), ``SEM_RIGHT_SPAN_WRONG_LOGIC``,
  ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``,
  ``SEM_NARROW_FIX_TEST_OVERFIT``,
  ``SEM_STRUCTURAL_VALID_INERT``.
  ``classify_semantic_outcome_v2`` subsumes v1 on sentinel
  inputs (Theorem P44-2). ``REFINEMENT_MAP`` is reflexive so
  v2 can stay at the coarse level when bytes don't allow
  narrowing.

* **Part C — Phase-44 driver**
  (``vision_mvp/experiments/phase44_semantic_residue.py``).
  Sweep mode runs the Phase-42-shape experiment with raw
  capture on; analyse-only mode consumes (parent, capture)
  pairs and emits a ``phase44.summary.v1`` JSON with per-cell
  coarse + refined counters + a ``coarse_to_refined_partition``
  audit.

* **Part D — Public-SWE-bench-Lite readiness validator**
  (``vision_mvp/experiments/phase44_public_readiness.py``).
  Five checks (``schema`` → ``adapter`` → ``parser`` →
  ``matcher`` → ``test_runner``) on any local JSONL. CI-gate
  verdict: ``{"ready": true, "n": 57, "n_passed_all": 57,
  "blockers": []}`` on the bundled bank in **5.2 s wall**
  through SubprocessSandbox. The externalisation gap to public
  SWE-bench-Lite is now purely a data-availability gap.

| metric (Phase-44 canonical cell: parser=robust/nd=6/apply=strict) | pass@1 (N, R, S) | S−N gap | refined dominant residue (v2) |
|---|:-:|:-:|---|
| ``qwen3.5:35b`` (36B MoE, cluster mac2)   | 0.965 / 0.965 / 0.965 (55/57) | **0.0 pp** | right_site / test_overfit (refined from v1 wrong_site) |
| ``qwen2.5-coder:14b`` (cluster mac1)     | 0.930 / 0.930 / 0.930 | **0.0 pp** | right_file_wrong_span + partial_multi_hunk (v1 wrong_site/multi_hunk partitioned) |

**Three new theorems** (P44-1 raw capture is a lossless
projection of pipeline state; P44-2 refined classifier monotone
on sentinel inputs; P44-3 public-readiness saturates on bundled
bank) and **four new conjectures** (C44-1 coder-class
wrong_edit_site refines to right_file_wrong_span; C44-2
frontier residue refines to overfit or wrong-logic; C44-3
substrate gap is refinement-invariant on coder-class; C44-4
readiness closed under row-level filtering).

**Headline**: the substrate-vs-naive gap is **0 pp** on every
coder-class model under the v2 refined classifier (Conjecture
C44-3), and the Phase-43 § D.7 sentinel-path limitation is
closed — the 14B-coder's 4/57 residue and the 35B's 2/57
residue are now attributable to refined sub-labels by
inspection of stored bytes, no LLM replay required.
The substrate's durable claim (bounded active context per role,
Theorem P43-1) is preserved: the v2 classifier does not move a
measurement between SEM_OK and non-SEM_OK.

Reproduce:
```bash
# Phase-44 public-SWE-bench-Lite readiness validator (bundled bank)
python -m vision_mvp.experiments.phase44_public_readiness \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out vision_mvp/results_phase44_readiness_bundled.json

# Phase-44 sweep with raw capture — qwen2.5-coder:14b on mac1
python -m vision_mvp.experiments.phase44_semantic_residue \
    --mode real --model qwen2.5-coder:14b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --parser-modes strict robust --apply-modes strict \
    --n-distractors 6 --think default --max-tokens 400 \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out-parent vision_mvp/results_phase44_parser_14b_coder.json \
    --out-capture vision_mvp/results_phase44_capture_14b_coder.json

# Phase-44 sweep with raw capture — qwen3.5:35b on mac2
python -m vision_mvp.experiments.phase44_semantic_residue \
    --mode real --model qwen3.5:35b \
    --ollama-url http://192.168.12.248:11434 \
    --sandbox subprocess \
    --parser-modes strict robust --apply-modes strict \
    --n-distractors 6 --think off --max-tokens 600 \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out-parent vision_mvp/results_phase44_parser_35b_moe.json \
    --out-capture vision_mvp/results_phase44_capture_35b_moe.json

# Phase-44 analysis — refined cross-model summary
python -m vision_mvp.experiments.phase44_semantic_residue \
    --analyse-only \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --artifacts \
        vision_mvp/results_phase44_parser_14b_coder.json \
        vision_mvp/results_phase44_parser_35b_moe.json \
    --captures \
        vision_mvp/results_phase44_capture_14b_coder.json \
        vision_mvp/results_phase44_capture_35b_moe.json \
    --out vision_mvp/results_phase44_refined_summary.json
```

Full phase-43 writeup: ``vision_mvp/RESULTS_PHASE43.md``.

### Phase 45 — one-command product runner, release-candidate validation, finished-product checklist

Phase 44 closed the science side (raw capture + refined
taxonomy + five-check readiness validator). Phase 45 closes the
*operability* side: a declared profile set, a one-command
runner, a report renderer, and a durable Finished-Product
Checklist in the master plan (§9 of
``docs/context_zero_master_plan.md``).

* **One command** —
  ``python3 -m vision_mvp.product --profile <name> --out-dir <d>``
  composes the Phase-44 readiness validator, the Phase-42
  parser sweep, and the Phase-44 raw-capture / refined
  taxonomy behind a stable schema
  (``phase45.product_report.v1`` + ``phase45.profile.v1``).
* **Six stable profiles** — ``local_smoke``, ``bundled_57``,
  ``bundled_57_mock_sweep``, ``aspen_mac1_coder``,
  ``aspen_mac2_frontier``, ``public_jsonl``.
* **Release-candidate artifacts** — under
  ``vision_mvp/artifacts/phase45_rc_bundled/`` (readiness
  57/57, 5.16 s), ``vision_mvp/artifacts/phase45_rc_mock_sweep/``
  (oracle pass@1 = 1.000 every strategy every cell), and
  ``vision_mvp/artifacts/phase45_mac1_recorded/`` (real-LLM
  launch command recorded for the ASPEN cluster operator).
* **Three new theorems** — P45-1 (runner composition is
  faithful), P45-2 (readiness is a hard gate for the sweep),
  P45-3 (finished-product state is the logical product of the
  per-layer theorems). **Three new conjectures** (C45-1
  profile-set completeness; C45-2 operator overhead <5 %;
  C45-3 remaining blockers are model/data shaped, not
  architecture shaped).
* **What still materially blocks finished-product status** —
  named in §9.8 of the master plan: (1) a public SWE-bench-
  Lite JSONL on local disk (🧱 external data availability),
  and (2) a ≥70B local coder-finetuned model on the cluster
  (◐ engineering). Neither is an architecture blocker.

Reproduce the Phase-45 release-candidate validation pass:
```bash
# Local smoke (mock, 8 instances, <1 s wall)
python3 -m vision_mvp.product --profile local_smoke \
    --out-dir /tmp/cz-smoke

# Bundled 57 readiness release-candidate (subprocess, ~5 s)
python3 -m vision_mvp.product --profile bundled_57 \
    --out-dir vision_mvp/artifacts/phase45_rc_bundled

# Bundled 57 + mock oracle sweep (~30 s)
python3 -m vision_mvp.product --profile bundled_57_mock_sweep \
    --out-dir vision_mvp/artifacts/phase45_rc_mock_sweep

# Real-LLM launch command record (ASPEN mac1, no LLM budget)
python3 -m vision_mvp.product --profile aspen_mac1_coder \
    --out-dir vision_mvp/artifacts/phase45_mac1_recorded

# Public SWE-bench-Lite drop-in (requires JSONL)
python3 -m vision_mvp.product --profile public_jsonl \
    --jsonl /path/to/swe_bench_lite.jsonl \
    --out-dir vision_mvp/artifacts/phase45_public_lite
```

Full Phase-45 writeup: ``vision_mvp/RESULTS_PHASE45.md``.

### Phase 46 — External-exercise readiness: public-data import, CI gate, frontier-model slot

Phase 45 closed "finished product within programme control".
Phase 46 closes the *boundary* between the finished product and
the outside world — the remaining blockers named in master plan
§9.8 (public JSONL, ≥70B model, CI pipeline) now meet the code at
a single command.

* **Public-data import CLI** —
  ``python3 -m vision_mvp.product.import_data --jsonl X --out Y``.
  Schema audit (native / hermetic / ambiguous / unusable),
  duplicate-``instance_id`` detection, decode-error / non-object
  / empty-bank enumeration, delegated Theorem-P44-3 readiness.
  Exit codes: ``0`` clean / ``1`` blocker / ``2`` file-not-found.
* **CI gate** —
  ``python3 -m vision_mvp.product.ci_gate --report <product_report.json> ...``.
  Five checks (schema / profile / readiness threshold / sweep
  outcome / artifact presence), multi-report aggregation,
  ``--min-ready-fraction`` + ``--min-pass-at-1`` thresholds,
  ``--require-profile`` whitelist. Machine-readable
  ``phase46.ci_verdict.v1``.
* **Frontier-model slot** — profile
  ``aspen_mac1_coder_70b`` + ``profiles.model_availability()``.
  Adding a 70B coder model is a one-string config change; the
  runner attaches ``model_metadata`` to the recorded launch
  payload so downstream tooling knows whether the model is
  resident or pending.
* **Three new theorems** — P46-1 (import-audit saturates on the
  bundled bank), P46-2 (CI-gate composition is faithful), P46-3
  (capability declaration ≠ residency). **Three new conjectures**
  (C46-1 remaining blockers are boundary-shaped; C46-2 70B
  ceiling lift + substrate invariance; C46-3 CI consumption
  closed under profile extension).

Reproduce the Phase-46 boundary-surface exercises:
```bash
# Audit the bundled bank (57/57 rows, readiness READY)
python3 -m vision_mvp.product.import_data \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out   vision_mvp/artifacts/phase46_public_audit/bundled_bank_audit.json

# Audit a missing path (exit 2)
python3 -m vision_mvp.product.import_data \
    --jsonl /nonexistent/public_swe_bench_lite.jsonl

# Gate the three Phase-45 RC artifacts (aggregate ok)
python3 -m vision_mvp.product.ci_gate \
    --report \
        vision_mvp/artifacts/phase45_rc_bundled/product_report.json \
        vision_mvp/artifacts/phase45_rc_mock_sweep/product_report.json \
        vision_mvp/artifacts/phase45_mac1_recorded/product_report.json \
    --out vision_mvp/artifacts/phase46_ci_gate/aggregate_verdict.json

# Exercise the frontier-model slot (no LLM invoked)
python3 -m vision_mvp.product --profile aspen_mac1_coder_70b \
    --out-dir vision_mvp/artifacts/phase46_frontier_slot
```

Full Phase-46 writeup: ``vision_mvp/RESULTS_PHASE46.md``.

### Phase 42 — parser-compliance attribution layer, 57-instance SWE-bench-Lite bank, cluster rerun

Phase 41 surfaced a new attribution boundary above the matcher
axis: the LLM-output parser. On the Phase-41 bank ``gemma2:9b``
emitted the semantically correct fix on every instance but
failed to close the bridge's ``<<<`` output delimiter, so every
patch landed as ``patch_no_match`` before the matcher axis
became measurable. Phase 42 promotes that boundary to a
first-class attribution surface and closes the ≥ 50-instance
external-validity threshold.

* **Part A — Parser-compliance layer**
  (``vision_mvp/tasks/swe_patch_parser.py``). A
  ``parse_patch_block(text, mode, unified_diff_parser)`` entry
  point with three modes (``PARSER_STRICT`` = Phase-41
  baseline; ``PARSER_ROBUST`` = Phase-42 default with five
  heuristics — tolerant block closing at end-of-generation,
  trailing-prose stripping, unified-diff fallback, two-fence
  pairing, ``OLD:``/``NEW:`` label-prefix; ``PARSER_UNIFIED``
  = diff-only) and a closed vocabulary: ten failure kinds
  (``PARSE_OK`` / ``PARSE_EMPTY_OUTPUT`` / ``PARSE_NO_BLOCK``
  / ``PARSE_UNCLOSED_NEW`` / ``PARSE_UNCLOSED_OLD`` /
  ``PARSE_MALFORMED_DIFF`` / ``PARSE_EMPTY_PATCH`` /
  ``PARSE_MULTI_BLOCK`` / ``PARSE_PROSE_ONLY`` /
  ``PARSE_FENCED_ONLY``), six recovery labels
  (``RECOVERY_NONE`` / ``RECOVERY_CLOSED_AT_EOS`` /
  ``RECOVERY_FENCED_CODE`` / ``RECOVERY_LABEL_PREFIX`` /
  ``RECOVERY_UNIFIED_DIFF`` / ``RECOVERY_LOOSE_DELIM``).
  ``ParserComplianceCounter`` aggregates per-cell and
  exposes ``compliance_rate`` / ``raw_compliance_rate`` /
  ``recovery_lift``; the bridge's ``llm_patch_generator``
  gains opt-in ``parser_mode`` / ``parser_counter`` /
  ``prompt_style`` kwargs. ``None`` preserves Phase-41
  behaviour.
* **Part B — 57-instance SWE-bench-Lite-style bank.** The
  Phase-41 28-instance ``swe_lite_style_bank.jsonl`` grown
  with 29 new instances covering broader edit classes
  (string manipulation, numeric guards, sequence
  construction, dict helpers, recursion/iteration,
  exception handling, set algebra, class state
  transitions, binary search, graph walk, multi-hunk
  class edits). Every instance round-trips through the
  oracle before being written — the bank-builder refuses
  any instance whose diff doesn't parse, whose OLD blocks
  aren't unique, or whose oracle-patched source doesn't
  pass the hidden test.
* **Part C — Parser sweep + cluster driver**
  (``experiments/phase42_parser_sweep``). Sweeps
  ``(parser_mode × apply_mode × n_distractors)`` with an
  LLM-output cache so the second parser cell costs only
  sandbox wall. ``LLMClient(base_url=…)`` + ``--ollama-url``
  forward to the ASPEN cluster: coding/generation runs on
  macbook-1 (``http://192.168.12.191:11434``), secondary
  runs on macbook-2 (``http://192.168.12.248:11434``) or
  localhost — fan out in parallel.

| metric (Phase-42 mock, 57 instances) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle)                         | 1.000 | 1.000 | 1.000 |
| substrate prompt tokens (n_d ∈ {0..24}) | 197 → 527 | 93 → 423 | **205.9 (constant)** |
| events to patch_gen                     | 4 → 28 | 0 → 24 | **0** |
| wall (1 368 sandboxed measurements)     | — | — | **122 s** |

**Three new theorems** (P42-1 parser-compliance attribution
decomposition ``Δ pass@1 = |R_recovered_parser| −
|R_regressed_parser|``; P42-2 parser recovery cannot produce
a false pass — byte-provenance argument on recovery
heuristics; P42-3 robust parser dominates on format-
noncompliant generators whose dominant failure mode is one
of ``{unclosed_new, prose_only, fenced_only_2,
label_prefix, fence_wrapped_payload}``) and **three new
conjectures** (C42-1 substrate-vs-naive gap ≤ 1 pp at N ≥
50; C42-2 parser-compliance dominates matcher-
permissiveness at 7B–30B; C42-3 three-axis decomposition
completeness ``pass@1 = P_parse · P_match · P_semantic ·
P_sandbox``).

**Headline empirical result**: ``qwen2.5-coder:14b`` on
the ASPEN cluster macbook-1, 57 instances, strict matcher
— pass@1 jumps from **0.018 / 0.018 / 0.018**
(naive / routing / substrate, strict parser, 56 × 3
`patch_no_match` from fence-wrapped OLD payloads) to
**0.930 / 0.930 / 0.930** (robust parser, with
`RECOVERY_FENCE_WRAPPED`): **+91.2 percentage-point pass@1
lift, 52 instances recovered on every strategy, 0
regressed, substrate-vs-naive gap = 0 pp**. On cluster
macbook-2 ``qwen2.5:14b-32k`` (general) the same parser
axis lifts +1.8 pp (12 % of outputs fence-wrapped — model-
specific η). On localhost ``qwen2.5-coder:7b`` — the Phase-41
headline model at N = 57 — the parser axis is empirically
null (no fence-wrapping); pass@1 = **0.842** on every
strategy. And on localhost ``gemma2:9b`` (the Phase-41
§ D.4 failure-mode replication), pass@1 flips from
**0/28 → 24/28 = 0.857** on every strategy under the
robust parser — **+85.7 pp lift** from the exact same
LLM output, changing only the parser mode. Substrate-vs-
naive gap is **0 pp** on all four real-LLM cells
(14B-coder cluster, 14B-general cluster, 7B-coder
localhost, 9B-gemma localhost) — the strongest empirical
support for Conjecture C42-1 in the programme to date.
Full per-cell tables in
``vision_mvp/RESULTS_PHASE42.md`` § D.3, § D.4, § D.4b,
§ D.5.

Reproduce:
```bash
# Phase-42 mock — 1 368 measurements in ~122 s, no LLM, no docker
python -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode mock --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase42_swe_lite_mock.json

# Phase-42 real LLM — cluster mac1 qwen2.5-coder:14b
python -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen2.5-coder:14b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase42_parser_14b_coder.json

# Phase-42 secondary — localhost qwen2.5-coder:7b
python -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen2.5-coder:7b \
    --ollama-url http://localhost:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase42_parser_7b_coder.json

# Phase-42 test slice (parser + bank regression; 31 tests)
python -m pytest vision_mvp/tests/test_phase42_parser.py -q
```

### Phase 41 — larger SWE-bench-Lite-style empirical sweep, patch-matcher permissiveness attribution, stronger-model datapoint

Phase 40 proved the real SWE-bench-style loop exists. Phase
41 moves the next credibility step: **scale plus
attribution**. Three tightly coupled artifacts ship, keeping
the agent-team substrate central:

* **Part A — 28-instance real-shape bank**
  (``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``).
  ~4.7× the Phase-40 mini bank, authored to cover a
  disciplined spectrum of edit shapes (single-hunk,
  multi-hunk, multi-function, operator-typo, off-by-one,
  wrong-branch, seed-wrong, aggregate-missing, mutation-
  vs-copy, parity-partition, slice-direction, index-
  return, polarity-flipped, empty-guard, type-conversion,
  unicode edge, ambiguous comparator). The bank-builder
  (``_build_swe_lite_bank.py``) refuses to register any
  instance whose diff doesn't parse, whose OLD blocks
  aren't unique, or whose oracle-patched source doesn't
  pass the hidden test.
* **Part B — Permissive patch-matcher axis.**
  ``apply_patch(..., mode=…)`` accepts one of
  ``strict`` (Phase-40 default, byte-exact),
  ``lstrip`` (leading-whitespace drift tolerance),
  ``ws_collapse`` (internal-whitespace drift),
  ``line_anchored`` (trailing-whitespace drift). All
  three permissive modes preserve the **unique-match
  discipline** — a normalised OLD that matches more than
  one source region is rejected. ``apply_mode`` is
  threaded through ``run_swe_loop``, ``Sandbox.run``, and
  ``run_swe_loop_sandboxed``; ``SWEReport.config`` records
  it for audit.
* **Part C — Attribution-aware driver**
  (``experiments/phase41_swe_lite_sweep``). Caches every
  LLM call per ``(instance, strategy, n_distractors)``
  so permissive cells reuse strict cells' proposals; emits
  a per-strategy ``{recovered, regressed, unchanged_pass,
  unchanged_fail}`` set delta between each permissive
  mode and strict.

| metric (Phase-41 mock, 28 instances) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle)                       | 1.000 | 1.000 | 1.000 |
| substrate prompt chars (n_d ∈ {0..24}) | 807 → 2 126 | 373 → 1 692 | **746 (constant)** |
| events to patch_gen                   | 4 → 28 | 0 → 24 | **0** |
| wall (672 sandboxed measurements)     | — | — | **53.0 s** |

**Three new theorems** (P41-1 bounded-context
preservation at 4.7× scale, P41-2 oracle-ceiling is
matcher-mode-invariant, P41-3 matcher-permissiveness
attribution decomposition ``Δ pass@1 = |R_recovered| −
|R_regressed|``) and **five new conjectures** (C41-1
communication-bounded at ≥ 50 instances, C41-2
matcher-permissiveness saturation, C41-3 stronger-model
strict-floor saturation, C41-4 regime decomposition
``pass@1 = P_comm · P_gen``, C41-5 parser-compliance
attribution boundary). Real-LLM sweeps on
``qwen2.5-coder:7b`` (28 instances, **pass@1 = 0.929 /
0.929 / 0.893** naive / routing / substrate under strict;
byte-identical under permissive matchers;
``R_recovered = R_regressed = ∅``) and ``gemma2:9b`` (28
instances, **pass@1 = 0 / 0 / 0** — the general-purpose
9B emits the semantically correct fix but fails the
bridge's ``<<<`` output-delimiter contract, surfacing
a new attribution boundary named by Conjecture C41-5);
see ``vision_mvp/RESULTS_PHASE41.md`` § D.3 and § D.4.

Reproduce:
```bash
# Phase-41 mock — 672 measurements in ~ 53 s, no LLM, no docker
python -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode mock --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase41_swe_lite_mock.json

# Phase-41 real LLM — qwen2.5-coder:7b on all 28 instances
python -m vision_mvp.experiments.phase41_swe_lite_sweep \
    --mode real --model qwen2.5-coder:7b --sandbox subprocess \
    --apply-modes strict lstrip \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase41_swe_lite_7b.json

# Phase-41 test slice (18 tests; ~ 28 s)
python -m pytest vision_mvp/tests/test_phase41_swe_lite.py -q
```

### Phase 40 — real SWE-bench-style loader, sandboxed execution boundary, first end-to-end real-shape evaluation

Phase 39 left two gaps to end-to-end SWE-bench: the
*mechanical* (unidiff parser + sandboxed runner +
JSONL loader, Theorem P39-4) and the *empirical*
(does substrate dominance hold at SWE-bench Lite scale,
Conjectures C39-3 / C39-4). Phase 40 closes the
mechanical gap end-to-end:

* **Part A — Real-shape loader / adapter
  (``tasks/swe_bench_bridge`` extension).**
  ``parse_unified_diff`` (a tolerant ``git diff``
  parser), ``SWEBenchAdapter.from_swe_bench_dict``
  (the real-shape adapter — derives ``buggy_function``
  from the diff hunk, promotes a ``test_patch`` to a
  runnable ``test_source``), and ``load_jsonl_bank``
  (hermetic JSONL loader with per-instance file
  namespacing). A bundled six-instance JSONL artifact
  (``vision_mvp/tasks/data/swe_real_shape_mini.jsonl``)
  exercises the full path offline in sub-second.
* **Part B — Sandboxed execution boundary
  (``tasks/swe_sandbox``).** Three backends behind
  one ``Sandbox`` protocol: ``InProcessSandbox``
  (Phase-39 wrapped), ``SubprocessSandbox`` (new —
  wall-clock timeout, tempdir cwd, sanitised env, JSON
  outcome protocol), ``DockerSandbox`` (new — optional;
  ``--network=none --read-only`` rootfs, ``tmpfs``
  ``/work``). ``select_sandbox("auto")`` picks
  Docker → subprocess → in-process by availability;
  ``run_swe_loop_sandboxed`` wires it into the bridge.
* **Part C — End-to-end driver
  (``experiments/phase40_real_swe_bridge``).** Loader
  + substrate + sandbox + (optional) real LLM patch
  generator. Mock run on the bundled JSONL across
  n_distractors ∈ {0, 6, 12, 24} = 72 measurements
  in **5.6 s**, pass@1 = 1.000 / 1.000 / 1.000 (oracle
  ceiling). Real-LLM ``qwen2.5:0.5b``: every cell
  hits ``patch_no_match`` (transcription-bounded
  regime). Real-LLM ``qwen2.5-coder:7b``: pass@1 =
  0.833 / 0.833 / 0.667 (naive / routing /
  substrate) on 6 instances — substrate ranks one
  instance below naive on byte-strict matcher
  variance, sitting cleanly in P39-2's
  transcription-bounded regime.

| metric (Phase-40 mock JSONL bank) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle generator)         | 1.000 | 1.000 | 1.000 |
| substrate prompt chars (n_d ∈ {0..24}) | 826 → 2 145 | 373 → 1 692 | **813 (constant)** |
| events to patch_gen               | 4 → 28 | 0 → 24 | **0** |
| substrate handoffs                | 2     | 2     | **5**   |

| metric (Phase-40 real LLM, n_d=6) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (qwen2.5:0.5b)             | 0.000 | 0.000 | 0.000 |
| pass@1 (qwen2.5-coder:7b)         | 0.833 | 0.833 | 0.667 |
| dominant failure                  | patch_no_match (1/6 → 2/6 substrate) ||

**Three new theorems** (P40-1 unidiff round-trip,
P40-2 real-shape bounded-context, P40-3 sandbox-
boundary preservation) and **three new conjectures**
(C40-1 sandbox cost amortisable, C40-2 loader
sufficiency for SWE-bench Lite, C40-3 sandbox-axis
equivalence). See ``vision_mvp/RESULTS_PHASE40.md``.

Reproduce:
```bash
# Phase-40 mock — sub-second, no LLM, no docker required
python -m vision_mvp.experiments.phase40_real_swe_bridge \
    --mode mock --sandbox subprocess \
    --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \
    --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase40_real_swe_bridge_mock.json

# Phase-40 real LLM — qwen2.5:0.5b (~ 100 s)
python -m vision_mvp.experiments.phase40_real_swe_bridge \
    --mode real --model qwen2.5:0.5b --sandbox subprocess \
    --jsonl vision_mvp/tasks/data/swe_real_shape_mini.jsonl \
    --n-distractors 6 \
    --out vision_mvp/results_phase40_real_swe_bridge_0p5b.json

# Phase-40 test slice (26 tests; ~ 11 s)
python -m pytest vision_mvp/tests/test_phase40_real_swe_bridge.py -q
```

### Phase 39 — real-LLM prompt-variant sweep, frontier-model substrate breadth, SWE-bench-style bridge

Phase 39 attacks three coupled gaps Phase 38 left open:
(A) the real-LLM prompt-variant measurement; (B)
cross-family frontier-model breadth on the substrate
slice; (C) a runnable SWE-bench-style bridge that wires
the existing typed-handoff substrate through a multi-role
patch / test team.

* **Part A — Real-LLM prompt-variant measurement
  (``--mode real``).** The Phase-38 mock predicted that
  ``rubric`` and ``contrastive`` variants would cut
  ``sem_wrong`` from 0.69 → 0.23. On real
  ``qwen2.5:0.5b`` the prediction is *wrong* — four of
  five variants reproduce the Phase-37 default
  distribution to within ±0 calls; the fifth
  (``forced_order``) merely converts semantic errors
  into malformed parses. **Theorem P39-1**: on the 0.5B
  / 7B size class, the Phase-37
  ``sem_root_as_symptom`` bias is *model-shaped, not
  prompt-shaped*. This empirically refutes the
  optimistic read of Conjecture C38-3 on these models.
* **Part B — Frontier-model bounded substrate sweep
  (``experiments/phase39_frontier_substrate``).** A
  cross-family bench on Phase-31 incident triage at
  k = 6, seed = 31 across mock + 2–3 local LLMs
  (``llama3.1:8b``, ``gemma2:9b``,
  ``qwen2.5-coder:7b``). Substrate-side correctness
  preservation reproduces across families.
* **Part C — SWE-bench-style bridge
  (``tasks/swe_bench_bridge`` +
  ``experiments/phase39_swe_bridge``).** A
  ``SWEBenchStyleTask`` schema mirroring SWE-bench's
  public instance shape; a four-instance ``MiniSWEBank``
  with real Python files, real bugs, real gold patches,
  real in-process tests; a four-role team
  (``issue_reader`` / ``code_searcher`` /
  ``patch_generator`` / ``test_runner``) wired through
  the unchanged Phase-31 ``HandoffRouter``.
  ``SWEBenchAdapter.from_dict`` documents the schema
  mapping for a future real-SWE-bench loader.
  **Theorem P39-3**: the substrate's bounded-context
  invariant extends to the SWE-style team — patch_
  generator prompt size is constant in n_distractors
  while naive grows linearly. **Theorem P39-4**: the
  schema gap to public SWE-bench is adapter-shaped, not
  architectural.

| metric (mini-SWE bank) | naive | routing | substrate |
|---|---:|---:|---:|
| pass@1 (oracle generator)  | 1.000 | 1.000 | 1.000 |
| pass@1 (qwen2.5:0.5b)      | 0.000 | 0.000 | 0.000 |
| pass@1 (qwen2.5-coder:7b)  | 0.250 | 0.250 | 0.250 |
| mean prompt chars (n_d=24) | 1936  | 1360  | **842** |
| events to patch_gen        | 14    | 10    | **0**   |
| substrate handoffs         | 2     | 2     | **5**   |

| prompt variant | 0.5b correct | 7b correct | 7b dyn ctst |
|---|---:|---:|---:|
| default        | 0.100 | 0.100 | 0.000 |
| contrastive    | 0.100 | **0.500** | 0.000 |
| few_shot       | 0.100 | 0.200 | **0.250** |
| rubric         | 0.100 | **0.400** | 0.000 |
| forced_order   | 0.100 | 0.200 | 0.000 |

(0.5B: every variant pinned at correct = 0.10 — model-shaped bias.
7B: contrastive lifts correct 5×; few_shot is the only variant
that lifts downstream contested accuracy — partial prompt-shape.)

| frontier substrate slice (incident triage, k=6) | naive | substrate | substrate_wrap |
|---|---:|---:|---:|
| qwen2.5-coder:7b acc_full | 0.000 | 0.400 | **0.800** |
| llama3.1:8b acc_full      | 0.000 | 0.200 | **0.600** |
| gemma2:9b acc_full        | 0.000 | 0.000 | **1.000** |
| gemma2:9b acc_root_cause  | 0.000 | 0.400 | **1.000** |
| substrate prompt tokens   | 573  | **196** | 229 |

(Cross-family reproduction of the Phase-31 substrate dominance:
substrate_wrap dominates naive by +60 to +100 pp on 7B / 8B / 9B;
**gemma2:9b saturates at 1.000/1.000** — first real LLM in the
programme to fully match the substrate ceiling on a non-code team.
Substrate prompt is constant at 196 chars regardless of model.)

See ``vision_mvp/RESULTS_PHASE39.md`` for theorems
(P39-1..P39-4) and conjectures (C39-1..C39-4).

Reproduce:
```bash
# Real LLM prompt-variant sweep (~ 100s on 0.5b)
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5:0.5b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase39_prompt_calibration_0p5b.json

# SWE-bench-style mini bridge (sub-second mock)
python3 -m vision_mvp.experiments.phase39_swe_bridge \
    --mode mock --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase39_swe_bridge_mock.json

# Frontier-model substrate sweep on incident triage
python3 -m vision_mvp.experiments.phase39_frontier_substrate \
    --models llama3.1:8b gemma2:9b qwen2.5-coder:7b \
    --domains incident --distractor-counts 6 --seeds 31 \
    --out vision_mvp/results_phase39_frontier_substrate.json
```

### Phase 38 — two-layer ensemble composition, minimum dynamic primitive ablation, prompt-shaped reply calibration

Phase 38 closes three Phase-37 frontier items at once: (A)
two-layer ensemble composition across the extractor and
reply axes; (B) a per-feature falsifier table for the
five-feature candidate minimum dynamic primitive; (C) a
pipeline for measuring whether the Phase-37
``sem_root_as_symptom`` bias is prompt-shaped.

* **Part A — Two-layer ensemble composition
  (``core/two_layer_ensemble`` +
  ``core/extractor_adversary``).**
  ``PathUnionCausalityExtractor`` + ``UnionClaimExtractor``
  stack layer-1 (extractor-axis) and layer-2 (reply-axis)
  ensembles. **Theorem P38-1**: on a conjunction attack
  (layer-1 adversary drops the gold claim AND layer-2
  biased primary emits IR on every candidate), the two-
  layer stack ``UnionClaimExtractor ∘
  EnsembleReplier(MODE_DUAL_AGREE)`` is the unique
  configuration that recovers (0.833 vs 0.333 for every
  single-layer alternative; 1.000 contested). **Theorem
  P38-2**: on the Phase-37 ``adv_drop_root`` cell where
  Theorem P37-4 proved every reply-axis ensemble powerless,
  ``PathUnionCausalityExtractor(PATH_MODE_UNION_ROOT)``
  above-noise combiner recovers to 1.000 accuracy.
* **Part B — Minimum primitive ablation
  (``core/primitive_ablation``).** Feature-flagged thread
  runner with five toggles; ablation table across Phase-35
  contested + Phase-37 nested banks. **Theorem P38-3**:
  ``typed_vocab``, ``terminating_resolution``, and
  ``round_aware_state`` are individually load-bearing;
  ``bounded_witness`` is null-control on accuracy but
  load-bearing for Theorem P35-2's context bound;
  ``frozen_membership`` is null-control on tested families
  (Conjecture C38-2 asserts a family exists where it is
  load-bearing).
* **Part C — Prompt-variant calibration
  (``core/prompt_variants``).** Five surgical variants of
  the Phase-36 default prompt (default, contrastive,
  few_shot, rubric, forced_order). Driver supports
  ``--mode mock`` (sub-second deterministic simulation)
  and ``--mode real`` (Ollama sweep). Mock headline:
  ``rubric`` and ``contrastive`` cut the semantic-wrong
  rate from 0.688 → 0.225. **Theorem P38-4**: every
  variant preserves the Phase-36 typed-reply contract;
  bias shifts are measurable without enlarging the
  substrate surface. Real-LLM measurement is one CLI
  parameter away (Conjecture C38-3).

| cell / config      | baseline | ext_only | reply_only | **two_layer** | two_layer_path_union |
|---|---:|---:|---:|---:|---:|
| clean              | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ext_drop_gold      | 0.333 | **0.833** | 0.333 | 0.833 | 0.833 |
| rep_biased_primary | 0.333 | 0.333 | **1.000** | **1.000** | 0.333 |
| **conjunction**    | 0.333 | 0.333 | 0.333 | **0.833** | 0.333 |
| adv_drop_root      | 0.333 | 0.333 | 0.333 | 0.333 | **1.000** |

| feature removed          | contested | nested |
|---|---:|---:|
| none (full)              | 1.000 | 1.000 |
| typed_vocab              | 0.500 | 0.333 |
| terminating_resolution   | 0.333 | 0.000 |
| round_aware_state        | 1.000 | 0.000 |
| bounded_witness          | 1.000 | 1.000 |
| frozen_membership        | 1.000 | 1.000 |

| prompt variant | correct_rate (mock) | sem_wrong (mock) |
|---|---:|---:|
| default        | 0.312 | 0.688 |
| **contrastive** | **0.775** | **0.225** |
| few_shot       | 0.463 | 0.537 |
| **rubric**     | **0.775** | **0.225** |
| forced_order   | 0.500 | 0.500 |

See ``vision_mvp/RESULTS_PHASE38.md`` for theorems
(P38-1..P38-4) and conjectures (C38-1..C38-3).

Reproduce:
```bash
# Two-layer ensemble sweep (sub-second)
python3 -m vision_mvp.experiments.phase38_two_layer_ensemble \
    --seeds 35 36 37 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_two_layer_ensemble.json

# Minimum primitive ablation (sub-second)
python3 -m vision_mvp.experiments.phase38_primitive_ablation \
    --seeds 35 36 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_primitive_ablation.json

# Prompt calibration, mock mode (sub-second)
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode mock --seeds 35 36 --distractor-counts 4 6 \
    --out vision_mvp/results_phase38_prompt_calibration_mock.json

# Prompt calibration, real Ollama (one variant at ~20s on 0.5b)
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5:0.5b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase38_prompt_calibration_real.json
```

### Phase 37 — real-LLM reply calibration, reply-axis ensembles, nested-contest equivalence

Phase 37 sharpens three axes Phase 36 left open: (A) real-LLM
calibration of the Phase-36 synthetic reply-noise channel; (B)
reply-axis ensemble defenses; (C) thread-vs-adaptive equivalence
on a task family with multi-round state.

* **Part A — Real-LLM reply calibration
  (``core/reply_calibration``).** ``CalibratingReplier`` wraps
  an ``LLMThreadReplier`` with a per-call oracle comparator and
  records every call into a 9-bucket taxonomy. On the Phase-35
  contested bank: both ``qwen2.5:0.5b`` and ``qwen2.5-coder:7b``
  emit 100 % well-formed JSON, but 90 % semantically wrong
  (50 % ``sem_root_as_symptom`` + 40 %
  ``sem_uncertain_as_symptom``). **Theorem P37-1:** real-LLM
  reply noise is dominated by semantic mislabel, not syntactic
  failure — the Phase-36 synthetic malformed_prob knob is a
  near-useless surrogate on this task.
* **Part B — Reply-axis ensemble (``core/reply_ensemble``).**
  Three modes: ``dual_agree`` (AND-gated parallel),
  ``primary_fallback`` (chatty primary + deterministic
  fallback), ``verified`` (primary + deterministic verifier).
  **Theorem P37-2:** under a biased primary (always emits IR),
  dual_agree and verified recover from 33 % to 100 %.
  **Theorem P37-3:** under synthetic malformed_prob=0.5,
  primary_fallback recovers from 83 % to 100 %. **Theorem P37-4**
  (structural limit): under extractor-output-level noise
  (adversarial drop_root, synthetic mislabel), no ensemble
  mode helps — the ensemble is above the noise wrapper.
* **Part C — Nested-contest thread vs adaptive sub
  (``tasks/nested_contested_incident``).** Three scenarios
  where round-1 replies are insufficient. Four strategies ×
  18 measurements: static=0.000, adaptive_sub_1r=0.000,
  **adaptive_sub_2r=1.000** (18 briefings), **dynamic_nested_2r=
  1.000** (0 briefings). **Theorem P37-5:** accuracy
  equivalence EXTENDS to nested contests while exposing a
  structural-complexity separation — the thread reads round-1
  replies natively, adaptive-sub needs explicit inter-round
  briefing edges.

| noise cell | mode | dyn acc | dyn contested |
|---|---|---:|---:|
| clean                 | single            | 1.000 | 1.000 |
| synth_malformed_0.5   | single            | 0.833 | 0.750 |
| synth_malformed_0.5   | **primary_fallback** | **1.000** | **1.000** |
| biased_primary_ir     | single            | 0.333 | 0.000 |
| biased_primary_ir     | **dual_agree**    | **1.000** | **1.000** |
| biased_primary_ir     | **verified**      | **1.000** | **1.000** |
| adv_drop_root         | any mode          | 0.333 | 0.000 |
| synth_mislabel_0.5    | any mode          | 0.333 | 0.000 |

| strategy (nested bank) | acc | briefings |
|---|---:|---:|
| static_handoff        | 0.000  | 0  |
| adaptive_sub_1r       | 0.000  | 0  |
| **adaptive_sub_2r**   | **1.000** | **18** |
| **dynamic_nested_2r** | **1.000** | **0** |

See ``vision_mvp/RESULTS_PHASE37.md`` for theorems (P37-1..
P37-5) and conjectures (C37-1..C37-4).

Reproduce:
```bash
# Real-LLM calibration (Ollama required; 20s for 0.5b)
python3 -m vision_mvp.experiments.phase37_real_reply_calibration \
    --models qwen2.5:0.5b qwen2.5-coder:7b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase37_real_reply_calibration.json

# Reply-axis ensemble sweep (sub-second mock)
python3 -m vision_mvp.experiments.phase37_reply_ensemble \
    --seeds 35 36 --distractor-counts 6 \
    --out vision_mvp/results_phase37_reply_ensemble.json

# Nested-contest comparison (sub-second mock)
python3 -m vision_mvp.experiments.phase37_nested_contest \
    --seeds 37 38 39 --distractor-counts 4 6 \
    --out vision_mvp/results_phase37_nested_contest.json
```

### Phase 36 — dynamic coordination under reply noise, LLM-driven typed replies, and adaptive subscriptions

Phase 36 stresses the Phase-35 dynamic-coordination primitive
along three coupled axes:

* **Part A — Reply-axis noise (``core/reply_noise``).**
  ``ReplyNoiseConfig`` perturbs producer-local causality replies
  with Bernoulli drop / mislabel; ``AdversarialReplyConfig``
  targets the gold ``INDEPENDENT_ROOT`` reply under a per-scenario
  budget. **Theorem P36-1:** under i.i.d. noise the dynamic
  accuracy satisfies ``Pr[D_dyn = gold] = (1-p)·(1-q)``; static
  is capped at ``≤ 1/2``. Dominance persists for ``p + q < 1/2``.
  Empirically, dynamic is at 91.7 % at drop_prob=0.25 and
  degrades to the static baseline at drop_prob≥0.75.
  **Theorem P36-2:** a single targeted adversarial
  ``drop_root`` (budget ``b = 1``) collapses both dynamic and
  adaptive-sub to the static baseline (33.3 %).
* **Part B — LLM-driven typed replies
  (``core/llm_thread_replier``).** ``LLMThreadReplier`` drives
  a narrow LLM call (one JSON line — reply_kind ∈ Phase-35 enum,
  bounded witness), filters out-of-vocab / malformed at parse,
  falls back to UNCERTAIN. **Theorem P36-3:** under well-formed
  in-vocab replies, the LLM replier is behaviourally identical
  to the deterministic oracle (100 % contested accuracy).
  Graceful decay at ``malformed_prob = 0.5`` to 66.7 %.
* **Part C — Bounded adaptive subscriptions
  (``core/adaptive_sub``).** ``AdaptiveSubRouter`` extends the
  Phase-31 router with a bounded, TTL-expiring edge-install /
  tick primitive (hard cap ``max_active_edges``). A new strategy
  ``STRATEGY_ADAPTIVE_SUB`` installs one temporary edge per
  producer per contested scenario. **Theorem P36-4:** across
  96 paired measurements (drop × mislabel × k × seed) the
  dynamic-thread vs adaptive-sub accuracy gap is **0.000 pp** at
  every cell; token overhead is +12 %. Conjecture C35-5
  empirically confirmed on this task family.

| noise | dynamic | adaptive_sub | static |
|---|---:|---:|---:|
| drop=0.0, mis=0.0  | 1.000 | 1.000 | 0.333 |
| drop=0.25, mis=0.0 | 0.917 | 0.917 | 0.333 |
| drop=0.5, mis=0.0  | 0.667 | 0.667 | 0.333 |
| drop=1.0, mis=0.0  | 0.333 | 0.333 | 0.333 |
| adv/drop_root, b=1 | 0.333 | 0.333 | 0.333 |

See ``vision_mvp/RESULTS_PHASE36.md`` for theorems + the full
noise × primitive × seed grid.

Reproduce:
```bash
# Mock sweeps (all sub-second)
python3 -m vision_mvp.experiments.phase36_noisy_dynamic \
    --mock --seeds 35 36 \
    --drop-probs 0.0 0.1 0.25 0.5 0.75 1.0 --mislabel-probs 0.0 0.25 \
    --out vision_mvp/results_phase36_noisy_dynamic.json

python3 -m vision_mvp.experiments.phase36_llm_replies \
    --mock --seeds 35 36 --malformed-probs 0.0 0.1 0.25 0.5 \
    --out vision_mvp/results_phase36_llm_replies_mock.json

python3 -m vision_mvp.experiments.phase36_adaptive_sub \
    --mock --seeds 35 36 --distractor-counts 6 20 \
    --drop-probs 0.0 0.25 0.5 1.0 \
    --out vision_mvp/results_phase36_adaptive_sub.json
```

### Phase 35 — dynamic, bounded communication primitives and a contested-incident benchmark

Phases 31–34 established that **typed** handoffs + a **static**
role-subscription table suffice whenever the auditor's decoder can
pick the right answer from a fixed-priority rule over the
delivered bundle. Phase 35 identifies the smallest task family
where that precondition fails — *contested* incidents where two
plausible root-cause claims arrive with inverted static priority
— and ships a minimal primitive that recovers correctness while
preserving the Phase-31 bounded-context guarantee.

* **Escalation threads (``core/dynamic_comm``).** A typed,
  frozen-membership, explicitly-terminated coordination object
  with bounded per-member reply budget and bounded witness-token
  cap. The thread's only public output is a single
  ``CLAIM_THREAD_RESOLUTION`` handoff routed through the
  unchanged Phase-31 ``HandoffRouter``; thread-internal events are
  hash-chained in the existing log but never enter non-member
  inboxes (Theorem P35-4, no-leak invariant).
* **Contested-incident benchmark (``tasks/contested_incident``).**
  6-scenario bank: 4 contested root-cause pairs (deadlock vs
  shadow cron, TLS vs disk shadow, cron vs OOM shadow, DNS vs
  TLS shadow) + 2 controls. Under the mock auditor across
  k ∈ {6, 20, 60, 120} × 2 seeds × 4 strategies = 192
  measurements:

| k | strategy | full_acc | contested_acc | mean_tok |
|---:|---|---:|---:|---:|
| 6   | naive          | 0.333 | 0.000 |   591 |
| 6   | static_handoff | 0.333 | 0.000 |   215 |
| 6   | **dynamic**    | **1.000** | **1.000** |   **246** |
| 120 | naive          | 0.167 | 0.250 | 2 950 |
| 120 | static_handoff | 0.333 | 0.000 |   215 |
| 120 | **dynamic**    | **1.000** | **1.000** |   **246** |

  Dynamic coordination is **flat at 246 tokens / 100 % accuracy
  on every k**; static is **flat at 215 tokens but capped at
  33 % full / 50 % root-cause** (Theorem P35-1 separation).
  Messaging budget: exactly one 3-member thread per contested
  scenario, ≤ 2 replies of ≤ 12 witness tokens. Real-LLM spot
  check under ``qwen2.5:0.5b`` at k=6 seed=35 reproduces the
  separation: dynamic root-cause 1.00 vs static 0.50 (+50 pp).

* **Theorems.** P35-1 (expressivity separation between static
  handoffs and dynamic coordination — a pigeonhole argument on
  priority orderings), P35-2 (bounded-context preservation with
  additive ``T·R_max·W`` per role per round — independent of
  |X|), P35-3 (correctness under sound producer-local causality
  extraction), P35-4 (no-leak invariant). Conjectures C35-5
  (bounded threads ≡ bounded adaptive subscriptions in decoder
  correctness), C35-6 (dynamic coordination is necessary, not
  only sufficient — predicts an information-theoretic lower-
  bound dual of P35-1).

See ``vision_mvp/RESULTS_PHASE35.md`` for theorems + benchmark
tables.

Reproduce:
```bash
# Mock sweep (~0.2 s)
python3 -m vision_mvp.experiments.phase35_contested_incident \
    --mock --distractor-counts 6 20 60 120 --seeds 35 36 \
    --out vision_mvp/results_phase35_mock.json

# Real-LLM spot check (~70 s for 24 calls on qwen2.5:0.5b)
python3 -m vision_mvp.experiments.phase35_contested_incident \
    --model qwen2.5:0.5b --distractor-counts 6 --seeds 35 \
    --out vision_mvp/results_phase35_llm_0p5b.json
```

### Phase 34 — structured noise, adversarial noise, and honest ensemble extractors

Phase 34 closes the three medium-term frontier items Phase 33
surfaced: per-role-adaptive calibration (§ 4.11 bullet h),
adversarial extractor noise (§ 4.11 bullet i), ensemble extractor
composition (§ 4.11 bullet j).

* **Per-role calibration (Part A).**
  ``core/extractor_calibration.per_role_audit_summary`` + an audit-
  fit ``PerRoleNoiseConfig`` + ``per_role_noisy_extractor``. The
  Phase-34 mock benchmark across three domains shows max per-role
  drop-rate spread ≥ 0.33 (incident), 0.50 (compliance), 0.67
  (security) — the Phase-33 C33-3 pattern reproduces on every
  domain. Theorem P34-1: ``A_real ≤ Π_k (1 − δ_k) ≤
  (1 − δ̄)^{R*}`` by AM-GM; the pooled replay over-estimates
  accuracy whenever per-role noise is heterogeneous.
* **Adversarial noise (Part B).**
  ``core/extractor_noise.adversarial_extractor`` with three target
  modes (load-bearing drop with priority ordering, role silencing,
  severity escalation). At matched nominal budget δ·R*, the
  targeted-drop adversary collapses substrate accuracy to **0 %** at
  budget = 1 on all three domains while matched i.i.d. preserves
  20 %–80 % — pooled gap **+0.47 pp** (Theorem P34-2). Severity
  escalation confirms the Theorem-P33-3 precision-to-severity
  failure mode on the max-ordinal security decoder.
* **Ensemble extractor (Part C).**
  ``core/ensemble_extractor.UnionExtractor`` on a compliance *mixed*
  bank (5 canonical + 5 narrative; regex cannot parse narrative
  phrasings and narrative-LLM cannot match canonical). Regex alone
  **50 %**, LLM alone **0 %**, ensemble **100 %** at pooled
  δ_u = 0.00 ≤ δ_r · δ_l = 0.188 — Conjecture C33-4 promoted to
  empirical bound (Theorem P34-3).

See ``vision_mvp/RESULTS_PHASE34.md`` for theorems + benchmark tables.

Reproduce:
```bash
python -m vision_mvp.experiments.phase34_per_role_calibration \
    --mode mock --domains incident compliance security \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase34_per_role_calibration_mock.json

python -m vision_mvp.experiments.phase34_adversarial_noise \
    --domains incident compliance security --seeds 34 35 \
    --drop-budgets 1 2 3 \
    --out vision_mvp/results_phase34_adversarial_noise.json

python -m vision_mvp.experiments.phase34_ensemble_extractor \
    --seeds 34 35 --distractor-counts 6 \
    --out vision_mvp/results_phase34_ensemble_extractor.json
```

### Phase 33 — LLM-driven extractors, real-noise calibration, and a third non-code domain

The Phase-31/32 substrate evaluated with *regex-perfect* extractors
— clean but unrealistic. Phase 33 closes that gap: an LLM-driven
extractor (``core/llm_extractor``) is a drop-in replacement for any
Phase-31/32 regex extractor, and a calibration layer
(``core/extractor_calibration``) measures its empirical noise
profile (δ̂ drop, ε̂ spurious, μ̂ mislabel, π̂ payload-corrupt)
against gold causal chains and maps it to the closest Phase-32
synthetic sweep grid point.

Headline on ``qwen2.5:0.5b`` against the compliance-review bank
(k = 6, seed = 33, 40 LLM calls, 91 s wall):

| metric | real 0.5b LLM extractor | Phase-32 closest (drop=0.5 sp=0.1 mis=0.25) | gap |
|---|---:|---:|---:|
| drop rate (δ̂)      | **0.70** | 0.50 | +0.20 |
| spurious per event (ε̂) | **0.12** | 0.10 | +0.02 |
| mislabel rate (μ̂) | **0.40** | 0.25 | +0.15 |
| substrate accuracy  | 0.00 | 0.00 | 0.00 |
| handoff recall      | **0.50** | 0.60 | −0.10 |
| handoff precision   | **0.27** | 0.21 | +0.06 |
| **verdict**         | | | **approximates** (γ = 0.10) |

The Phase-32 synthetic i.i.d. Bernoulli sweep **approximates** the
real LLM extractor's pooled noise profile on compliance — max-abs
gap of 0.10 on the recall axis. Per-role noise is highly
heterogeneous (legal 50 % drop, finance 100 % drop) — the pooled
match hides structure that Conjecture C33-3 names explicitly.

A **third non-code domain** — security-audit escalation
(``tasks/security_escalation``) — with a *max-ordinal severity +
claim-set classification* decoder (structurally distinct from both
prior domains) confirms the substrate signature at K = 3:

| k | strategy | tokens | acc (mock) |
|---:|---|---:|---:|
| 6   | naive          |   767 | 100 % |
| 6   | routing        |   151 |   0 % |
| 6   | **substrate**  | **242** | **100 %** |
| 120 | naive          | 4 216 |  20 % |
| 120 | **substrate**  | **242** | **100 %** |

Three domains, three decoder shapes (Phase 31 priority-order /
Phase 32 monotone-verdict + strict-flags / Phase 33 max-ordinal
severity + claim-set), one substrate module unchanged. Theorems
P33-1 (LLM-extractor subsumption), P33-2 (cross-domain
correctness at K = 3), P33-3 (two-regime bound on max-ordinal
decoder); Conjectures C33-3 (per-role heterogeneity), C33-4
(ensemble composition). See
``vision_mvp/RESULTS_PHASE33.md`` for theorems + benchmark tables.

Reproduce:
```bash
# LLM-extractor benchmark across all three domains (mock — instant)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode mock \
    --distractor-counts 6 20 60 120 --seeds 33 34 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_mock.json

# Real 0.5b LLM calibration on compliance (~90 s)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode real \
    --model qwen2.5:0.5b --domains compliance --seeds 33 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_0p5b.json

# Third-domain substrate benchmark (mock — instant)
python3 -m vision_mvp.experiments.phase33_security_escalation --mock \
    --distractor-counts 6 20 60 120 --seeds 33 34 \
    --out vision_mvp/results_phase33_security_mock.json
```

Writeup: `vision_mvp/RESULTS_PHASE33.md`.

### Phase 32 — cross-domain substrate, noisy-extractor robustness, frontier-model spot check

A second non-code domain confirms the Phase-31 substrate isn't
specific to operational telemetry. Five role-typed agents (legal,
security, privacy, finance, compliance officer) review a vendor
onboarding request — compound compliance issues (missing DPA,
uncapped liability, cross-border transfer without SCCs, weak
encryption, budget breach) require cross-role handoffs to resolve
correctly:

| k | strategy | mean tokens | accuracy (mock) | accuracy (7B) |
|---:|---|---:|---:|---:|
| 6   | naive          |   658 | 100 % |   0 % |
| 6   | routing        |   132 |   0 % | — |
| 6   | **substrate**  | **171** | **100 %** | **100 %** |
| 6   | **substrate_wrap** | 204 |   100 % | **100 %** |
| 120 | naive          | 4 047 |  40 % | — |
| 120 | **substrate**  | **171** | **100 %** | — |

Same substrate module (``core/role_handoff``) unchanged — the
*only* change is a new scenario catalogue, extractor set, and
decoder. Substrate flat at **171 tokens** across four orders of
magnitude of document-stream size; matches Phase-31's 196-token
flatline on a structurally distinct domain. On ``qwen2.5-coder:7b``
the substrate path **saturates the mock ceiling at 100 %** on
compliance review (vs naive's 0 %); on incident triage the 7B
reaches 80 % via ``substrate_wrap`` vs naive's 0 %. **Theorem
P32-1** formalises cross-domain correctness preservation;
**Theorem P32-2** formalises graceful degradation under extractor
noise (two regimes, closing Conjecture C31-7 in the monotone
case); **Theorem P32-3** shows the token bound survives bounded
noise as long as inbox capacity absorbs the spurious blow-up. The
Phase-32 noise sweep (``core/extractor_noise``, 96 noise points ×
2 domains, 0.5 s wall) confirms all three empirically, with a new
first-class failure attribution (``spurious_claim``) on the flag
side of the histogram.

Reproduce:
```bash
# Part A — cross-domain benchmark
python3 -m vision_mvp.experiments.phase32_compliance_review --mock \
    --distractor-counts 6 20 60 120 --seeds 32 33 \
    --out vision_mvp/results_phase32_compliance_mock.json

# Part B — noisy-extractor sweep (both domains)
python3 -m vision_mvp.experiments.phase32_noise_sweep --domain both \
    --drop-probs 0.0 0.1 0.25 0.5 --spurious-probs 0.0 0.05 0.1 \
    --mislabel-probs 0.0 0.25 --seeds 31 32 \
    --out vision_mvp/results_phase32_noise_sweep.json

# Part C — qwen2.5-coder:7b spot check on both non-code benchmarks
python3 -m vision_mvp.experiments.phase32_stronger_model \
    --model qwen2.5-coder:7b --seeds 32 --distractor-counts 6 \
    --out vision_mvp/results_phase32_llm_7b_spot.json
```

Writeup: `vision_mvp/RESULTS_PHASE32.md`.

### Phase 31 — typed handoffs and the first non-code multi-role benchmark

The same substrate, without code in sight. Five role-typed agents
(monitor, DBA, sysadmin, network, auditor) investigate a cascading
outage on a simulated fleet — each role owns a different slice of
telemetry. Under a deterministic five-scenario catalogue with a
distractor-density sweep (k ∈ {6, 20, 60, 120} per role), the
substrate path delivers:

| k | strategy | mean tokens | accuracy (mock) | accuracy (0.5b) |
|---:|---|---:|---:|---:|
| 6   | naive          |   574 | 100 % | 0 % |
| 6   | routing        |   147 |   0 % | 0 % |
| 6   | **substrate**  | **196** | **100 %** | **40 %** |
| 120 | naive          | 2 925 |  20 % | — |
| 120 | **substrate**  | **196** | **100 %** | — |

Substrate prompt size is **flat at 196 tokens** across four orders
of magnitude of event-stream size. Role-keyed routing alone cannot
rescue the auditor role — its concerns are content-level, so
header-filtering delivers nothing. The new substrate primitive is
``vision_mvp/core/role_handoff.py``: typed, content-addressed,
hash-chained handoffs between roles, with a role-subscription
table at claim-kind granularity. Theorems P31-1..P31-5 formalise
the separation; P31-5 is the formal statement behind the
master-plan distinction from graph/index tools.

Reproduce:
```bash
python3 -m vision_mvp.experiments.phase31_incident_triage --mock \
    --distractor-counts 6 20 60 120 --seeds 31 32 \
    --out vision_mvp/results_phase31_mock.json
```

Writeup: `vision_mvp/RESULTS_PHASE31.md`.

### Phase 30 — substrate vs naive full-context, with a real LLM on the answer path

First LLM-in-loop external-validity result. `qwen2.5:0.5b` via
local Ollama answers 20 SWE-style queries on the Python stdlib
`json` module under three delivery strategies:

| strategy | mean prompt tokens | accuracy | wall s/call |
|---|---:|---:|---:|
| naive full-context | 2 615 | 20 % | 19.97 |
| role routing       | 2 554 | 10 % |  0.86 |
| **substrate_wrap** | **163** | **80 %** |  **1.22** |

That's **16.0×** fewer tokens **and** **+60 percentage points** of
accuracy. Same harness, same model, same corpus — the only variable
is delivery strategy. See `vision_mvp/RESULTS_PHASE30.md` for the
theorem set, external causal-relevance numbers on `click` and
`json-stdlib`, and the reproduction commands.

Reproduce:
```bash
python -m vision_mvp.experiments.phase30_llm_swe_benchmark \
    --model qwen2.5:0.5b --corpora json-stdlib \
    --out vision_mvp/results_phase30_json_stdlib.json
```

The harness `vision_mvp/tasks/swe_loop_harness.py` takes any
`Callable[[str], str]`, so swapping in a frontier-model API or a
SWE-bench task generator is a drop-in replacement.

---

## Why this works

The repo separates the context substrate into five distinct layers,
each with its own loss profile:

| Layer | What it bounds | Loss profile | Where in code |
|---|---|---|---|
| **Routing** | who-talks-to-whom | lossy — projections / sparse selection are intentional | `core/api.CASRRouter`, `core/agent_keys`, `core/sparse_router`, `core/hierarchical_router` |
| **Trigger** | when to refine | lossy — drops drafts that look "agreed" | `core/trigger`, `core/general_trigger`, `core/event_trigger`, `core/behavior_trigger` |
| **Exact external memory** | what content is preserved | **lossless** — content-addressed Merkle DAG | `core/merkle_dag`, `core/context_ledger` |
| **Retrieval** | what content reaches the prompt | lossy in *ranking*; never lossy in *content* | `core/retrieval_store` (dense), `core/lexical_index` (BM25), hybrid RRF in `core/context_ledger.search(mode=)` |
| **Computation / planning** | how aggregations and joins answer without LLM-in-the-loop | **lossless and deterministic** — typed operators over handles | `core/exact_ops`, `core/query_planner`, `core/code_planner` |
| **Render** | whether the planner's exact answer is wrapped by the LLM or returned verbatim | **lossless on the direct-exact path** (zero LLM, zero prompt) | `experiments/phase22_codebase.run_direct_exact` |

The full project's research-first framing, arc structure, and
long-running open problems live in
[docs/context_zero_master_plan.md](docs/context_zero_master_plan.md).
That document is the durable reference; results notes below are
the per-phase empirical record.

Phases 1–18 built the routing and trigger layers (CASR + hybrid-structural
trigger). Phase 19 introduced the exact-memory + retrieval layers as a
lossless context substrate. Phase 20 strengthened retrieval with hybrid
BM25+dense and structural multi-hop expansion. Phase 21 added the
computation layer — a small natural-language → operator planner that
answers aggregation queries deterministically without an LLM in the
inner loop. Phase 22 generalised the substrate to real Python codebases
(AST-derived typed metadata via `core/code_index`) and added the
direct-exact render path that bypasses the LLM entirely when the planner
has the answer. Phase 23 extends Phase 22 to multi-codebase external
validity: direct-exact holds at 65 / 65 (100 %, σ = 0) across six real
Python corpora (vision-mvp modules / tests / experiments, the `click`
third-party CLI framework, and the stdlib `json` module), with a
reusable `CorpusRegistry` and a coverage-accounting ingestion pass
(`IngestionStats`). Phase 24 extends the exact slice from syntactic
structure to conservative *intraprocedural* static-semantic properties —
per-function predicates for `may_raise`, `is_recursive`,
`may_write_global`, `calls_subprocess`/`filesystem`/`network`, computed
from the AST via `core/code_semantics`. Direct-exact holds at **44 / 44
(100 %, σ = 0)** across the same six corpora on a 44-question semantic
battery with zero LLM calls. **Phase 25 extends the exact slice further
to conservative *interprocedural* semantic properties** — transitive
closures of the Phase-24 predicates over a local call graph plus exact
SCC-based recursion-cycle detection, computed by a new
`core/code_interproc` module that runs as a corpus-wide post-pass.
Direct-exact scores **50 / 50 (100 %, σ = 0)** on the Phase-25
interprocedural battery across the same six corpora, zero LLM calls,
zero prompt chars; retrieval-multihop averages 38.0 % (σ = 23.1) with
every failure attributed to `retrieval_miss`. The widening is dramatic
per corpus — on `click`, intra `may_raise = 46` becomes trans
`may_raise = 96` (+50 functions recovered); on `vision-core`, mutual
recursion is detected over a 19-function SCC. **Phase 26 introduces a
separate truth axis — *runtime calibration of the conservative
analyzer* — via instrumented probes over an executable snippet corpus.
On 21 snippets × 6 runtime-decidable predicates (126 applicable
measurements), the analyzer agrees with runtime-observed truth on
123 / 126 (97.6 %). The three divergences are one false-positive
(`may_raise` on `if False: raise` — the analyzer is control-flow-
insensitive by design) and two false-negatives (`calls_subprocess` via
`eval`, `calls_filesystem` via `getattr` — reflection holes explicitly
documented as Phase-24 boundaries). Every divergence lands on a pre-
documented boundary condition; the direct-exact planner round-trip
still matches the analyzer at 126 / 126 (100 %), confirming the
substrate's `render_error = 0` guarantee is independent of analyzer
runtime calibration.** **Phase 27 extends the runtime-calibration
axis from the curated 21-snippet corpus to *real corpus functions*.
On `vision-core` (~791 functions), the Phase-24/25 analyzer emits
flags for every function; only ~35.7 % are *runtime-calibratable*
under the Phase-27 default invocation-recipe strategy (ready_no_args
+ ready_typed + ready_curated). The gap — $|F_R| / |F_A| \approx
0.36$ — is the formal research finding: runtime truth at corpus
scale is **witness-availability-bounded, not planner-exactness-
bounded** (Theorem P27-1). On the entered subset, analyzer and
runtime agree on the overwhelming majority of predicates;
divergences concentrate exactly on the Phase-24 pre-documented
boundary classes (Conjecture P27-4). Planner-vs-analyzer round-trip
remains at 100 % on every predicate across every corpus — the
Phase-22 substrate guarantee is independent of Phase-27 coverage
(Theorem P27-2).** **Phase 28 extends runtime calibration from
one primary corpus (`vision-core`) to four local corpora
(`vision-core` / `vision-tasks` / `vision-tests` /
`vision-experiments`) in a single benchmark, and makes the
explicit-vs-implicit raise distinction first-class in both the
analyzer and the runtime observer. On the pooled entered slice
(306 observations across 2 140 functions), `may_raise_explicit`
is sound (FN = 0) with 98.7 % agreement; the new
`may_raise_implicit` predicate is essentially sound (FN = 1 /
116 on runtime-positives, ≈ 0.9 %) and over-approximating
by design. The Phase-24 `may_raise` contract and every Phase-22..27
substrate guarantee are preserved byte-for-byte. Coverage is
reported as a first-class cross-corpus variable — `ready_fraction`
ranges from **2.9 %** (`vision-tests`, 97 % methods) to **80.2 %**
(`vision-experiments`, 80 % typed top-level) — while
analyzer/runtime agreement on the other five predicates stays at
100 % on every corpus's entered slice (Theorem P28-2, P28-3,
P28-4).**

**Phase 29 runs the first task-scale falsifiability check of the
core routing/substrate thesis on a realistic SWE-style multi-role
task distribution drawn from the same four corpora, and closes the
Phase-27/28 method-coverage gap with a conservative
instance-auto-constructor. On 80 queries / 5 718 events across four
corpora, the pooled aggregator-role *causal*-relevance fraction
under naive broadcast is **4.54 %** (σ ≈ 0.002 across 5 seeds) —
strictly below the ROADMAP-specified 50 % confirmation gate. The
substrate matches 95 % of tasks and collapses aggregator context
from 13 849 → 13.75 tokens (**1 007×**) at **100 %** answer
correctness. Role-level routing alone reduces non-aggregator
context 1.3×–1 154× but leaves the aggregator untouched, confirming
that routing-by-type cannot resolve content-level aggregation
(Theorem P29-2). In parallel, the Phase-29 method-instance recipe
promotes methods on safely-zero-arg-constructable classes
(inherited-`object.__init__` / all-defaulted init / dataclass-all-
defaults) to a new `ready_method` status, lifting runtime
`ready_fraction` on `vision-tests` from **2.9 %** (Phase 28) to
**98.8 %** (Phase 29) and pooled entered slice 4.83× (306 → 1 477)
with `may_raise_explicit` FN preserved at 0 pooled and
construct-failed rate < 1 % (Theorem P29-5). Pooled falsifiability
decision: **CONFIRMED** (Theorem P29-1; full eight-theorem set in
`RESULTS_PHASE29.md`).**

**Phase 30 closes the theoretical–empirical bridge and runs the
programme's first LLM-in-loop external-validity benchmark. Four
theorems (P30-1..P30-4) formalise minimum-sufficient context
`T_i*`, connect it to the Phase-29 causal-relevance fraction,
and close one special case of OQ-1 (fixed-point convergence) in
the matched-substrate regime with a unique one-step fixed point.
Two conjectures (P30-5, P30-6) give OQ-1 its first concrete
mathematical shape under a stochastic answer path. The benchmark
runs a real local Ollama model on real external Python corpora:
on the Python stdlib `json` module under `qwen2.5:0.5b`, the
substrate path delivers a **16.0×** prompt-token reduction and
a **+60 percentage-point** answer-accuracy lift (80 % vs 20 %)
over naive full-context delivery. Substrate-matched slice
accuracy is **78.9 %** on 0.5b — bounded below by the model's
transcription fidelity, not by any substrate guarantee. Routing
alone *does not* rescue this model (10 % — confirming Phase-29
Theorem P29-2 on a live LLM). External corpora (`click`,
`json-stdlib`) land in the same causal-relevance band as the
internal four-corpus set (0.047–0.122 vs 0.032–0.056),
supporting Conjecture P30-5. The harness
(`vision_mvp/tasks/swe_loop_harness.py`,
`vision_mvp/experiments/phase30_llm_swe_benchmark.py`) accepts
any `Callable[[str], str]`; a future SWE-bench driver is a
drop-in replacement. See `vision_mvp/RESULTS_PHASE30.md`.**

See
`vision_mvp/RESULTS_PHASE19.md`, `RESULTS_PHASE20.md`,
`RESULTS_PHASE21.md`, `RESULTS_PHASE22.md`, `RESULTS_PHASE23.md`,
`RESULTS_PHASE24.md`, `RESULTS_PHASE25.md`, `RESULTS_PHASE26.md`,
`RESULTS_PHASE27.md`, `RESULTS_PHASE28.md`, `RESULTS_PHASE29.md`,
and `RESULTS_PHASE30.md`.

Five ideas stacked at the routing/trigger layers (see
[`docs/archive/pre-coordpy-theory/VISION_MILLIONS.md`](docs/archive/pre-coordpy-theory/VISION_MILLIONS.md)
for the full vision of 10):

1. **Shared Latent Manifold** — every agent projects to, and reads from, a
   small shared subspace instead of talking to every other agent directly.
2. **Streaming PCA** — the subspace is learned from observations, no oracle.
3. **Global Workspace** — only the ⌈log₂ N⌉ most-surprised agents get to
   write each round.
4. **Neural-net predictor per agent** — agents predict their own next state;
   surprise = prediction error (the "world model" of each agent).
5. **Decaying shared register** — the manifold forgets old evidence at an
   exponential rate, so drift is tracked without a sliding window.

The math behind why O(log N) is the right bound, not O(N), is derived from
72 independent mathematical frameworks in
[`docs/archive/pre-coordpy-theory/EXTENDED_MATH.md`](docs/archive/pre-coordpy-theory/EXTENDED_MATH.md)
through `EXTENDED_MATH_7.md`:
Information Bottleneck, Kolmogorov cascade, gauge theory, spin glasses,
expander graphs, holographic entropy, TQFT, … — they all converge on the
same scaling law.

---

## Honest caveats

- **Low intrinsic rank assumption.** These results hold when the task's
  relevant structure has effective rank ≤ O(log N). Fully-general tasks
  with dim-d complexity need Ω(d) bandwidth by Theorem 11 in
  [`docs/archive/pre-coordpy-theory/PROOFS.md`](docs/archive/pre-coordpy-theory/PROOFS.md).
- **LLM experiments were run at N=10.** The numpy experiments go to
  N=100 000 but the bridge (N=100 real LLMs) was out of scope for the
  initial pass — that's the obvious next thing to check.
- **Protocol is synchronous.** Async variants are straightforward in
  principle (CRDT semantics are already commutative) but not yet built.
- **Not yet peer-reviewed.** If you're a referee or reviewer, please dig
  in and break things.

---

## Project status

This is one continuous research push (Apr 2026) producing:
- a 72-framework theoretical survey (`docs/archive/pre-coordpy-theory/EXTENDED_MATH_[1-7].md`)
- 12 formal theorems (`docs/archive/pre-coordpy-theory/PROOFS.md`; superseded by the W3..W6 families in `docs/THEOREM_REGISTRY.md`)
- 5 experiment phases from pure NumPy to local LLMs
- 94 passing unit + integration tests
- a clean public API (CASRRouter)

If it holds up under wider scrutiny, O(log N) coordination is the kind of
foundational result that would sit next to Shannon's channel capacity — a
statement about the minimum communication required for a class of
distributed problems. The way to find out is to throw it at more problems,
and to invite many pairs of eyes to check the math. That is what this
repository exists for.

---

## License

MIT.
