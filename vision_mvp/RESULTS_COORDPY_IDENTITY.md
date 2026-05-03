# RESULTS — CoordPy Identity & Clarity Milestone

*Post-Slice-2 consolidation pass. Date: 2026-04-22. No new subsystem
landed in this pass; what shipped is a coherent product identity,
repo legibility for new readers, and the first out-of-tree plugin
exemplar. Results are in the form of definitions, theorem-style
claims, and conjectures — with an honest ledger of what is
empirical, what is proved, what is conjectural, and what remains
external.*

## 0. Scope

This note belongs to the CoordPy product track, not a CASR research
arc. It is about the *projection* of a settled architecture onto a
legible product identity. It does not introduce new substrate
primitives or new empirical claims about multi-agent coordination.

The milestone's five fronts, kept close together because they are
tightly coupled:

1. **De-legacy** the active product narrative (CASR framing moved
   from headline to research-context footnote; the CoordPy SDK and
   its contract are now first-impression).
2. **Repo legibility** — one canonical orientation document
   (`docs/START_HERE.md`) that answers "what is this repo?" in one
   pass, without duplicating the master plan or the README.
3. **Public identity / API naming** — docstrings and help text on
   the CASR-era public surface no longer conflate research
   terminology with user-facing API contract.
4. **Final 10/10 defaults** — first out-of-tree plugin exemplar
   (`examples/out_of_tree_plugin/`) closing master-plan § 10.5
   ledger item 2 *as machinery-plus-artifact*, not just machinery.
5. **Definitive narrative** — one short paragraph each for the
   README, the architecture doc, and the master plan, so the
   project description is the same across surfaces.

## 1. Definitions used in this note

* **Product identity.** The name, one-sentence scope, and stable
  public surface a third party sees when they install and use the
  artifact.
* **Research-programme identity.** The body of claims, shards, and
  proofs that justify the product's guarantees but are not
  themselves the product's contract.
* **Identity drift.** The gap between (a) what a new reader infers
  the primary artifact is from the first-impression surfaces
  (README headline, top-level `__init__` docstring, CLI help,
  package `description`) and (b) what the code actually ships as
  its stable contract.
* **Self-explanatory repo (to an external agent).** A repo is
  *self-explanatory* if an agent arriving cold can, in one
  read-pass of ≤ 1 k words, correctly classify every top-level
  surface into {core substrate, product surface, boundary, research
  shard, legacy} with zero ambiguity.

## 2. Theorem-style claims

### W-IDN-1 (proved, constructive). Identity projection.

*Let `S` be the settled set of SDK-surface modules (§ 10.1 stability
matrix, rows marked Stable). Let `N` be the set of first-impression
surfaces — README headline, `vision_mvp/__init__.py` docstring,
`vision_mvp.coordpy.__init__.py` docstring, `ARCHITECTURE.md` headline,
`pyproject.toml` `description`, and the three console-script
`--help` texts. After this milestone, every element of `N` names
**CoordPy** as the shipped product, and every element of `N` that
mentions CASR contextualises it as research substrate rather than as
current product identity.*

**Proof sketch.** The set `N` is enumerable and finite; each element
was audited and either already clean or edited to meet the
condition (see file list in the milestone summary). The condition
is locally checkable on each element, so the global property holds
constructively. □

### W-IDN-2 (proved, constructive). Orientation sufficiency.

*`docs/START_HERE.md` provides, in ≤ 1 k words, a classification of
every top-level directory and every `vision_mvp.*` public subpackage
into one of {CoordPy SDK, CoordPy CLI, CoordPy extensions, Unified runtime,
Legacy product path, Core substrate, Research shards, Boundary /
next-slice}.*

**Proof sketch.** The START_HERE matrix is exhaustive against the
current `vision_mvp/` layout and the current `docs/` layout. Any
new directory added in the future lands in exactly one of the
declared categories by construction of the categories (the partition
is designed to cover the stability matrix plus the research shard
category plus the boundary category). □

### W-IDN-3 (proved, constructive). Extension-surface reality.

*The CoordPy extension surface admits a non-trivial out-of-tree
consumer: a package that (a) is installable with `pip install -e`
from outside the main repo, (b) registers a new `ReportSink` only
via `importlib.metadata.entry_points` under group
`coordpy.report_sinks`, and (c) requires no edit to any file under
`vision_mvp/`.*

**Proof sketch.** `examples/out_of_tree_plugin/coordpy-markdown-sink/`
is such a package. The registration path is
`coordpy.extensions.registry → importlib.metadata.entry_points →
coordpy_markdown_sink.sink:register → coordpy.extensions.register_report_sink`.
No edit to `vision_mvp/` is required; verified by construction of
the exemplar. □

## 3. Conjectures

### W-IDN-C1 (testable). Cold-agent classification accuracy.

*An external coding agent (any strong generalist model), given only
`docs/START_HERE.md` plus one `ls vision_mvp/` listing, correctly
classifies ≥ 95 % of `vision_mvp.*` subpackages into the seven
stability-matrix categories, on first try, without hallucinating
categories not in the matrix.*

This is falsifiable: pick an agent, run the test, score. If it
fails, the remedy is to tighten START_HERE's matrix — not to invent
a new category — because the partition is the contract.

### W-IDN-C2 (empirical, weak). Stable-identity robustness.

*Once product identity is decoupled from substrate lineage (i.e.
CASR is research, CoordPy is product), future additions to the
research shard pile do not destabilise the product identity. That
is, adding Phase 47, 48, … to `RESULTS_PHASE*.md` and to
`vision_mvp.core.*` does not require re-editing the README
headline, the `vision_mvp/__init__.py` docstring, or the CoordPy SDK
docstring to keep identity drift at zero.*

Evidence for this conjecture is the structural split made in this
pass: the first-impression surfaces talk about *CoordPy*, which has a
bounded definition; they talk about *CASR* only as historical
substrate, which is monotone. As long as the bounded-context
guarantee referenced in `api.py` remains Theorem 3, the conjecture
holds; if a future research arc weakens Theorem 3, the claim needs
re-stating. No empirical test here beyond "does the README headline
need a rewrite after each phase?"; answer, so far, no.

### W-IDN-C3 (conjectural). Distinctiveness via bounded-context runtime, not primitive novelty.

*CoordPy's distinctiveness now sits in the coherent combination
(bounded-context substrate × profile-driven runtime × provenance
manifest × plugin protocols × CI gate × unified mock/real path),
not in any one primitive being novel. In particular, CoordPy does
not need to claim that content-addressed memory, hierarchical
routing, or typed handoffs were invented here; the claim that holds
is that they are composed here into a single SDK surface whose
public contract fits on one page.*

This is a judgement, not a theorem. It is falsifiable in the
following sense: a reviewer who can replicate CoordPy's
bounded-context + provenance + CI-gate guarantee from any existing
SDK on PyPI, without assembling pieces themselves, falsifies the
distinctiveness claim. No such SDK is known to the authors as of
this milestone.

## 4. What is empirical, proved, conjectural, external

| Status | Content |
|---|---|
| **Proved (constructive)** | W-IDN-1 (identity projection), W-IDN-2 (orientation sufficiency), W-IDN-3 (extension-surface reality). Each is anchored to a finite, locally checkable artifact set. |
| **Empirical, carried over** | Slice 2's W2-2 (1349/1349 tests pass under unified runtime). This milestone did not rerun experiments; the test suite remained green after the documentation-layer and plugin-exemplar edits. |
| **Conjectural** | W-IDN-C1 (cold-agent classification), W-IDN-C2 (stable-identity robustness), W-IDN-C3 (distinctiveness via composition). All three are stated so that they are falsifiable and not trivially true. |
| **External** | The § 9.8 blockers (public SWE-bench-Lite JSONL on disk, resident ≥ 70 B coder model) — unchanged by this milestone, still 🧱 external. Docker-first-*by-default* for untrusted input — still Slice 3 D.1, still not flipped. GitHub Actions release-on-real-tag — still Slice 3 ops, workflow file exists but has not fired on a real tag. |

## 5. What, if anything, still blocks a true 10/10 CoordPy

Honest ledger, post-this-milestone:

1. **Docker-first-by-default for untrusted public JSONLs.** Backend
   exists; the `trust_input=False` default-flip is still Slice 3
   D.1. Decision *not* to flip in this pass was deliberate: a
   default-flip is a behaviour change and needs its own test
   pass; the identity milestone should not bundle behaviour changes.
2. **Release-on-real-tag firing.** `.github/workflows/coordpy-ci.yml`
   is checked in and covers SDK contract tests + sdist + wheel +
   tag-triggered release. It has not yet been exercised end-to-end
   on a real version tag. Requires a real `git tag` push.
3. **External blockers from § 9.8** — public SWE-bench-Lite JSONL
   on local disk, resident ≥ 70 B coder-finetuned model. Orthogonal
   to any SDK-layer slice.

Item 2 in master-plan § 10.5 ledger (*first real out-of-tree
plugin*) is closed by `examples/out_of_tree_plugin/` — both as
machinery (Slice 2) and as artifact (this milestone).

*End of note. Next CoordPy work, if any, is Slice 3: default-flip,
tagged release, and — orthogonally — the § 9.8 external blockers.*
