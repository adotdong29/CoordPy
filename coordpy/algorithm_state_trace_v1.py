"""W136 / COO-9 — machine-structured ALGORITHM-STATE TRACE instrument on the resistant-by-construction
battlefield.

W133 proved the exact-oracle EW1 COUNTEREXAMPLE witness adds +0.00 pp over blind reflexion on the
WRONG_ALGORITHM / SEARCH_ENUM modes (the wrong-algorithm capability ceiling).  W135 then asked whether
oracle-derived SOLUTION STRUCTURE in PROSE — a greedy-failure certificate, an optimal-substructure
LADDER, a search-frontier exact-count — could break that ceiling, and found a sharp null: on the
held-out non-complexity dev field the full structure controller S4 TIES the counterexample
(S4 - C1 = +0.00) AND blind reflexion (S4 - B0 = +0.00).  The W135 ladder is a FLAT list of optimal
values rendered in prose ("the optimum reuses optima of smaller sub-instances").  So the live blocker is
no longer feedback FORM in prose; W136 asks the one level deeper: **can CoordPy expose machine-structured
algorithm STATE — the optimal AND the naive trajectories through the sub-instance state space, the exact
step where they diverge, and the increment/branching deltas — and does representing solver state DIRECTLY
(not describing it in prose) unlock a mechanism class prose witnesses could not, WITHOUT leaking the
answer?**

The trace is a TYPED, CONTENT-ADDRESSED capsule (AT5) carrying, for the SAME token-minimal disjoint
counterexample ``X`` W133/W135 anchor on, a dual-trajectory STATE-TRANSITION TABLE over the OBVIOUS
disjoint sub-instances of ``X``:

* **AT1 — decision-path trace**: for greedy-vs-opt families, the per-sub-instance optimal value V(i) AND
  the naive value G(i), with the marked FIRST DIVERGENCE step i* = min{i : V(i) != G(i)} — the exact
  sub-instance where the naive's decision path first departs from the optimum (a machine-readable
  divergence trajectory, not a prose "the local choice is dominated").
* **AT2 — subproblem-state trace**: the optimal-value state table with the per-step increment trajectory
  ``optimal_delta(i) = V(i) - V(i-1)`` (and the naive's increments) — the optimal-substructure transition
  structure W135's flat optimal-only ladder collapsed.
* **AT3 — search-frontier trace**: for counting families, the exact count C(i) and the naive's count
  Cn(i) per disjoint sub-instance with the branching delta — the search frontier exposing the wrong
  recurrence as a SEQUENCE of (correct, wrong) counts, not a single scalar contrast.
* **AT5 — typed trace capsule**: the ``AlgorithmStateTraceV1`` object itself — bounded, content-addressed,
  reproducible from the sanctioned oracle-execution API.
* **AT4 — invariant-state trace**: DELIBERATELY EXCLUDED.  A clean loop-invariant / augmented-DP-state
  trace cannot be derived from oracle OUTPUTS; it requires INSTRUMENTING the reference solver to emit its
  internal state, which would render the recurrence / augmented state itself — the human insight that, per
  Pu (OOPSLA 2011), IS the algorithm.  Disclosing it is answer-adjacent leakage (it hands over the
  algorithm, not just the structure), so AT4 is documented-and-excluded, exactly as W135 excluded the
  SW4-invariant arm.

The KEY difference vs the W135 prose structure witness (the "genuinely-new vs S4" guard,
``trace_is_genuinely_new_vs_structure_v1``): W135's ladder carried ONLY V(i) (optimal values) + a prose
property hint.  The trace carries the DUAL trajectory (V(i) AND G(i) per sub-instance), the ``diverges``
flag per row, the marked first-divergence step, and the increment/branching ``delta`` trajectory — the
state-transition structure prose collapsed.  A trace that reduces to S4's optimal-only ladder (no naive
trajectory, no divergence, no deltas) is NOT genuinely-new (it is S4 reformatted = "a prose witness in
JSON clothing") and the guard fails it.

Anti-cheat / no-leakage (LOCKED — see ``docs/RUNBOOK_W136.md`` §4):

* the trace is the ONLY sanctioned feedback object; ``ref_source`` / ``brute_source`` / ``naive_source``
  are NEVER placed in any model-facing path — they are EXECUTED (subprocess) to derive OUTPUTS, never
  rendered as source (a structural test asserts ``to_capsule_block`` carries no ``def ``/``import ``/
  ``class `` solver source);
* the counterexample ``X`` AND every sub-instance in the table are asserted byte-disjoint from the graded
  ``secret_cases`` (``leakage_clean``); the recurrence / algorithm / augmented DP state is NEVER stated
  (AT4 excluded for exactly this reason);
* the table is bounded (``TRACE_MAX_ROWS`` rows on SMALL sub-instances) — minimal by construction, per the
  primary-source finding that LONGER execution traces HURT repair (arXiv:2505.04441); a fat trace is the
  documented falsification risk, not a feature;
* grading is the audited ``grade_on_secret_v1`` on a DISJOINT hidden bank (pass iff ALL secret cases
  pass), so memorising the shown table CANNOT pass — the trace tests GENERALISATION, not memorisation
  (the W133/W135 discipline).

Research-grounded FORM (primary sources, γ): unlike Scratchpad (arXiv:2112.00114) / Self-Debugging
(arXiv:2304.05128) / LDB (arXiv:2402.16906), which feed the model its OWN program's runtime trace to
localise its own bug, this feeds an ORACLE-derived trace of the CORRECT algorithm's STATE (the dual
trajectory) — genuinely new in FORM relative to the closest primary work; NAR (arXiv:2205.15659) /
TransNAR (arXiv:2406.09308) model correct-algorithm execution state but only by TRAINING a GNN/LLM on it
(not at inference on a frozen model); the bounded-trace discipline answers arXiv:2505.04441's
longer-traces-hurt falsification risk.

The trace arm (``run_trace_arm_v1``) is a strict **same-budget** swap of the W120 reflexion arm: identical
K / model / temperature / attempt-0 prompt; the ONLY change is the between-attempt feedback object.  It is
scored by the SAME audited grader + the SAME W108 evaluator that scored W89 / W105 / W120 / W132 / W133 /
W134 / W135, so "T - A1" is computed byte-identically to "B - A1".

Reuses (explicit-import only, NO duplication): the exact-oracle counterexample search + probe set from
``exact_oracle_witness_v1``; the sub-instance generator + scalar-gap + summariser from
``solution_structure_witness_v1`` (so the trace is anchored on the SAME X and sub-instances as the W135
ladder — the cleanest possible "machine-structured state vs prose" contrast); the minted-problem record +
subprocess from ``resistant_by_construction_battlefield_v1``; the pilot record, candidate extractor,
secret grader, and prompt scaffolds from ``icpc_reflexion_bench_v1``; the typed ``FailureDigestV1`` +
``ControllerAction`` arsenal (``executor_grounded_patcher_v1`` / ``controller_native_code_mechanism_v1``)
for the forward-only, weightless T2 controller route.  Pure / deterministic except the (already-audited)
program-execution subprocess; NO model inference lives here (that is the W136 driver script).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional

from .resistant_by_construction_battlefield_v1 import (
    MODE_SEARCH_ENUM,
    MODE_WRONG_ALGORITHM,
    MintedProblemV1,
    MintedTemplateV1,
    _exec_capture_v1,
    _tok_count,
)
from .exact_oracle_witness_v1 import (
    WitnessV1,
    WitnessProbeSetV1,
    _none_witness,
    find_counterexample_witness_v1,
)
from .solution_structure_witness_v1 import (
    _int_or_none,
    _scalar_gap_v1,
    _subinstances_v1,
)
from .icpc_reflexion_bench_v1 import (
    IcpcArmOutcomeV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt,
    _samples_block,
    extract_candidate_code_v1,
    grade_on_secret_v1,
    sample_feedback_v1,
)
from .executor_grounded_patcher_v1 import FailureDigestV1, parse_failure_digest_v1
from .controller_native_code_mechanism_v1 import ControllerAction

W136_ALGORITHM_STATE_TRACE_V1_SCHEMA_VERSION: str = (
    "coordpy.algorithm_state_trace_v1.v1")

# ---- trace kinds (oracle-side route label; NEVER shown to the model) ---------------
TRACE_DECISION_PATH: str = "DECISION_PATH"        # AT1
TRACE_SUBPROBLEM_STATE: str = "SUBPROBLEM_STATE"  # AT2
TRACE_SEARCH_FRONTIER: str = "SEARCH_FRONTIER"    # AT3
TRACE_NONE: str = "NONE"

# ---- trace arms (the same-budget feedback policies) --------------------------------
ARM_T1_TRACE_REWRITE: str = "T1"      # full machine-structured trace capsule (LEAD)
ARM_T2_TRACE_CONTROLLER: str = "T2"   # forward-only trace-routed controller (staged)
ARM_T_IO: str = "T_IO"                # execution-grounded I/O-format repair + trace (W136 root-cause win)
TRACE_ARMS: tuple[str, ...] = (ARM_T1_TRACE_REWRITE, ARM_T2_TRACE_CONTROLLER, ARM_T_IO)

# The generic, no-leakage I/O-format directive (fires ONLY when the model's own code fails a VALID public
# sample — execution-grounded).  The W136 root cause: the W132 battlefield emits whitespace-FLATTENED
# input (grid rows / pairs / triples on one line; the ref reads ``sys.stdin.read().split()``), but the
# model assumes one-row-per-line, so it crashes / misparses even the public samples — and the algorithm
# feedback (counterexample / structure / 2-D state table) never helped because it addressed the wrong bug.
# Proven $0: the SAME correct DP passes 7/7 with split() parsing and fails 7/7 with input()-per-line.
IO_REPAIR_DIRECTIVE: str = (
    "CRITICAL — FIX YOUR INPUT READING FIRST. Your program crashed, printed nothing, or printed an error "
    "on a VALID sample input, which means it is reading the input in the wrong SHAPE. In THIS problem "
    "every token is WHITESPACE-separated: numbers, grid rows, pairs and triples may be separated by "
    "spaces and/or newlines interchangeably — do NOT assume one row/pair/triple per line. Read the whole "
    "input at once and consume tokens in order:\n    import sys\n    data = sys.stdin.buffer.read().split()\n"
    "    # then take data[0], data[1], ... (grid rows are the next R whitespace tokens)\n"
    "After fixing the input reading, apply your algorithm. Exact diagnostic of the failing attempt:\n")

# ---- trace sizing (LOCKED; bounded by construction — longer traces HURT repair) ----
TRACE_PROBE_TIMEOUT_S: float = 2.0        # the candidate finishes a small probe within this
ORACLE_SUBINSTANCE_TIMEOUT_S: float = 4.0  # ref/naive on a sub-instance
TRACE_MAX_ROWS: int = 6                    # at most this many state rows (bounded prompt; arXiv:2505.04441)
MAX_VALUE_CHARS: int = 80                  # truncate a rendered oracle value


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== the state-transition row + trace records

@dataclasses.dataclass(frozen=True)
class StateTransitionRowV1:
    """One sub-instance of the counterexample X carrying BOTH trajectories: the oracle-optimal value AND
    the naive (wrong-algorithm) value, the divergence flag, and the per-step increment deltas.  Every
    field is an oracle OUTPUT or a derived summary — never the oracle PROGRAM."""

    idx: int
    size_tokens: int
    summary: str
    optimal_value: str        # ref(sub) — the oracle output
    naive_value: str          # naive(sub) — the wrong algorithm's output ("" if the naive failed)
    diverges: bool            # optimal_value != naive_value (the decision-path divergence)
    optimal_delta: str        # str(V(i) - V(i-1)) when both scalar; "" otherwise (the transition)
    naive_delta: str          # str(G(i) - G(i-1)) when both scalar; "" otherwise
    leakage_clean: bool

    def to_dict(self) -> dict[str, Any]:
        return {"idx": int(self.idx), "size_tokens": int(self.size_tokens), "summary": self.summary,
                "optimal_value_len": len(self.optimal_value), "naive_value_len": len(self.naive_value),
                "diverges": bool(self.diverges), "optimal_delta": self.optimal_delta,
                "naive_delta": self.naive_delta, "leakage_clean": bool(self.leakage_clean)}


@dataclasses.dataclass(frozen=True)
class AlgorithmStateTraceV1:
    """A machine-structured algorithm-state trace (AT5), anchored on a token-minimal disjoint
    counterexample.  ``to_capsule_block(arm)`` is the ONLY model-facing text; it carries oracle OUTPUTS +
    the dual-trajectory state table + the divergence step + increment deltas, never the oracle PROGRAM and
    never the recurrence (AT4 excluded)."""

    kind: str                         # TRACE_*
    at_family: str                    # AT1 | AT2 | AT3 | NONE
    counterexample: WitnessV1         # the underlying EW1 counterexample (the anchor)
    optimal_value: str                # V* = ref(X)
    naive_value: str                  # G  = naive(X) ("" if the naive failed on X)
    objective_gap: str                # str(|V* - G|) when both integer scalars; "" otherwise
    naive_overcounts: Optional[bool]  # SE: True if naive>exact, False if naive<exact, None if NA
    rows: tuple[StateTransitionRowV1, ...]
    first_divergence_idx: int         # the first row index where optimal != naive; -1 if none / no dual
    leakage_clean: bool
    grid: Optional["StateGridV1"] = None   # the full 2-D subproblem-state table (AT2) when 2-parameter

    def found(self) -> bool:
        return self.kind != TRACE_NONE and self.counterexample.found()

    def has_grid(self) -> bool:
        return bool(self.grid is not None and self.grid.n_cells() >= 4)

    def has_dual_trajectory(self) -> bool:
        """True iff >=1 row carries BOTH the optimal AND the naive value (the trajectory pair S4's
        optimal-only ladder structurally lacked), or the 2-D grid carries both."""
        return bool(any(r.naive_value != "" for r in self.rows)
                    or (self.has_grid() and any(v != "" for v in self.grid.naive.values())))

    def has_transition_structure(self) -> bool:
        """True iff the trace carries a STATE-TRANSITION signal beyond a flat value list: a marked
        divergence step, a per-step increment (delta) trajectory, OR a 2-D subproblem-state grid."""
        return bool(self.first_divergence_idx >= 0
                    or any(r.optimal_delta or r.naive_delta for r in self.rows)
                    or self.has_grid())

    def n_divergent_rows(self) -> int:
        return sum(1 for r in self.rows if r.diverges)

    def _shift_left_block(self) -> str:
        return (
            "Before writing code, list 2-3 worst-case / adversarial inputs your previous approach gets "
            "wrong (boundary sizes, ties, all-equal, the state the trace below exposes). Then write a "
            "solution that handles them.\n")

    def _observed(self) -> str:
        ce = self.counterexample
        return (ce.observed_output.rstrip() if ce.observed_kind == "WRONG_ANSWER"
                else "(your program timed out)" if ce.observed_kind == "TIMEOUT"
                else "(your program crashed)")

    def _table_text(self) -> str:
        """The machine-readable state-transition table (the core of the trace; NOT prose).  Each row is
        an independently oracle-verified (sub_instance -> optimal, your_approach, diverges, delta)
        record over a SMALL disjoint sub-instance of the counterexample."""
        if not self.rows:
            return ""
        lines = ["state_transition_table (each row = a small sub-instance of the counterexample, "
                 "independently verified by the reference oracle):",
                 "  idx | sub_instance | optimal | your_approach | diverges | optimal_delta"]
        for r in self.rows:
            nv = r.naive_value if r.naive_value != "" else "(failed)"
            dv = "YES" if r.diverges else "no"
            od = r.optimal_delta if r.optimal_delta != "" else "-"
            lines.append(f"  {r.idx} | {r.summary} | {r.optimal_value} | {nv} | {dv} | {od}")
        return "\n".join(lines) + "\n"

    def _trajectory_text(self) -> str:
        opt = [r.optimal_value for r in self.rows]
        you = [r.naive_value if r.naive_value != "" else "?" for r in self.rows]
        if not opt:
            return ""
        out = f"optimal_value_trajectory: [{', '.join(opt)}]\n"
        if self.has_dual_trajectory():
            out += f"your_approach_trajectory: [{', '.join(you)}]\n"
        if self.first_divergence_idx >= 0:
            out += (f"first_divergence: at sub-instance idx={self.first_divergence_idx} your approach "
                    f"first deviates from the optimum.\n")
        return out

    def to_capsule_block(self, arm: str) -> str:
        """The sanctioned MACHINE-STRUCTURED feedback object, rendered per arm.  Always anchored on the
        EW1 counterexample; the emphasis differs by kind (AT3 counts / AT2 deltas / AT1 divergence).  No
        solver source / no recurrence formula / no hidden-case answer."""
        if not self.found():
            return ("No counterexample was found and the program looked correct on small inputs; "
                    "re-examine the problem statement for an overlooked structural case.")
        x = self.counterexample.probe_input.rstrip()
        vstar = self.optimal_value.rstrip()
        head = (
            "Machine-readable algorithm-state trace (derived by executing a reference oracle on SMALL "
            "inputs disjoint from the hidden tests; do NOT special-case or hard-code these values — the "
            "hidden tests use different inputs):\n"
            f"counterexample_input:\n{x}\n"
            f"correct_output (optimal): {vstar}\n"
            f"your_output: {self._observed()}\n\n")
        # the full 2-D subproblem-state table (AT2) is the richest state representation; when present it
        # REPLACES the 1-D projection (a fatter trace would only add noise — arXiv:2505.04441).
        if self.has_grid():
            table = self.grid.to_table_text()
            traj = ""
        else:
            table = self._table_text()
            traj = self._trajectory_text()
        recurrence_scaffold = ""
        if self.has_grid() or self.kind == TRACE_SUBPROBLEM_STATE:
            recurrence_scaffold = (
                "\nDerive-the-recurrence step: each cell above is the optimum of a SUBPROBLEM and is "
                "determined by SMALLER subproblems (optimal substructure), NOT by a greedy/local choice. "
                "Before writing code, write the recurrence that reproduces this table in one line (e.g. "
                "cell[i][j] = a max/sum over smaller cells), then implement THAT recurrence as a general "
                "solution (do not hard-code the table).\n")

        if self.has_grid():
            tail = ("\nEach cell above is the optimum of a SUBPROBLEM; a greedy/local rule does not reach "
                    "it (see the divergence cells, where your approach departs from the optimum in state "
                    "space). The optimum of each cell is built from SMALLER cells — recover that "
                    "recurrence and implement it as a general solution.\n")
        elif self.kind == TRACE_SEARCH_FRONTIER:
            if self.naive_overcounts is True:
                dirn = (f"OVER-counts by {self.objective_gap}" if self.objective_gap else "OVER-counts")
            elif self.naive_overcounts is False:
                dirn = (f"UNDER-counts by {self.objective_gap}" if self.objective_gap else "UNDER-counts")
            else:
                dirn = "computes the wrong count"
            tail = (f"\nThe table is the exact search-frontier: optimal_value is the EXACT count on each "
                    f"sub-instance; your_approach is your program's count. Your recurrence {dirn} — read "
                    f"the (correct, your) count sequence above, find which transition your recurrence "
                    f"gets wrong (ordered vs unordered, reuse allowed vs not, or the wrong state), and "
                    f"re-implement the recurrence that reproduces the optimal sequence.\n\n")
        elif self.kind == TRACE_DECISION_PATH:
            tail = ("\nThe two trajectories agree until the first_divergence step, then your approach's "
                    "value departs from the optimum: a locally-greedy / obvious decision is DOMINATED "
                    "from that sub-instance on. Use the divergence point and the optimal trajectory to "
                    "find the non-local combination your approach misses, and re-implement a global "
                    "method (not a greedy/local extension).\n\n")
        else:  # TRACE_SUBPROBLEM_STATE (AT2)
            tail = ("\nThe optimal_delta column is the per-step increment of the optimum: each optimal "
                    "value reuses the optima of smaller sub-instances (optimal substructure) — it is NOT "
                    "a greedy local extension. Recover the recurrence that reproduces the optimal column "
                    "and its increments, and implement it (do not hard-code the table).\n\n")
        return (head + table + traj + tail + recurrence_scaffold + self._shift_left_block() +
                "Provide ONLY the corrected complete Python 3 program in a ```python ... ``` fence:")

    def cid(self) -> str:
        return _sha256_hex({"kind": "w136_algorithm_state_trace_v1", "at_family": self.at_family,
                            "trace_kind": self.kind,
                            "probe_input_sha256": _sha256_hex(self.counterexample.probe_input),
                            "optimal_value": self.optimal_value, "naive_value": self.naive_value,
                            "objective_gap": self.objective_gap,
                            "first_divergence_idx": int(self.first_divergence_idx),
                            "rows": [f"{r.idx}:{r.summary}={r.optimal_value}/{r.naive_value}"
                                     f":{r.optimal_delta}" for r in self.rows],
                            "grid": (sorted(f"{k}={v}" for k, v in self.grid.opt.items())
                                     if self.grid is not None else None)})

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "at_family": self.at_family, "found": self.found(),
                "counterexample": self.counterexample.to_dict(),
                "optimal_value_len": len(self.optimal_value), "naive_value_len": len(self.naive_value),
                "objective_gap": self.objective_gap, "naive_overcounts": self.naive_overcounts,
                "n_rows": len(self.rows), "rows": [r.to_dict() for r in self.rows],
                "first_divergence_idx": int(self.first_divergence_idx),
                "n_divergent_rows": self.n_divergent_rows(),
                "has_dual_trajectory": self.has_dual_trajectory(),
                "has_transition_structure": self.has_transition_structure(),
                "has_grid": self.has_grid(),
                "grid": (self.grid.to_dict() if self.grid is not None else None),
                "leakage_clean": bool(self.leakage_clean), "trace_cid": self.cid()}


def _none_trace(ce: Optional[WitnessV1] = None) -> AlgorithmStateTraceV1:
    return AlgorithmStateTraceV1(
        kind=TRACE_NONE, at_family="NONE", counterexample=ce or _none_witness(),
        optimal_value="", naive_value="", objective_gap="", naive_overcounts=None,
        rows=(), first_divergence_idx=-1, leakage_clean=True)


# ===================================================== typed (family-aware) sub-instances

def _is_int_token(s: str) -> bool:
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False


def _summarize_typed_subinstance_v1(s: str) -> str:
    """Compact single-line label of a (possibly multi-line / structured) sub-instance: ``header | body``
    so a knapsack ``N W`` + pairs / grid ``R C`` + rows sub-instance is shown UNAMBIGUOUSLY (the W135
    summariser dropped the header and showed only the last line, which is misleading for structured
    inputs)."""
    parts = s.split("\n")
    head = parts[0].strip()
    body_toks = " ".join(parts[1:]).split()
    preview = " ".join(body_toks[:8]) + (" ..." if len(body_toks) > 8 else "")
    if head and body_toks:
        return f"{head} | {preview}"
    return (head or preview)[:60]


def _ladder_sizes(count: int) -> list[int]:
    """A bounded, deterministic ladder of sub-counts 1..count (<= TRACE_MAX_ROWS rungs)."""
    if count <= TRACE_MAX_ROWS:
        return list(range(1, count + 1))
    cand = sorted({1, 2, 3, count // 2, count - 1, count})
    return [m for m in cand if 1 <= m <= count][:TRACE_MAX_ROWS]


def _grid_ladder_pairs(R: int, C: int) -> list[tuple[int, int]]:
    """A monotone diagonal ladder of (r', c') sub-grid sizes growing to (R, C) (<= TRACE_MAX_ROWS)."""
    steps = max(R, C, 1)
    seen: set = set()
    pairs: list[tuple[int, int]] = []
    for t in range(1, steps + 1):
        rp = min(R, max(1, round(R * t / steps)))
        cp = min(C, max(1, round(C * t / steps)))
        if (rp, cp) not in seen:
            seen.add((rp, cp))
            pairs.append((rp, cp))
    if (R, C) not in seen:
        pairs.append((R, C))
    if len(pairs) > TRACE_MAX_ROWS:
        idx = sorted({0, len(pairs) - 1}
                     | {int(i * (len(pairs) - 1) / (TRACE_MAX_ROWS - 1)) for i in range(TRACE_MAX_ROWS)})
        pairs = [pairs[i] for i in sorted(idx)][:TRACE_MAX_ROWS]
    return pairs


def _typed_subinstances_v1(x: str) -> list[str]:
    """Family-AWARE sub-instances of the counterexample X for the structured input shapes the generic
    1D slicer cannot handle: interleaved tuples (knapsack ``N W`` + (w,v) pairs / weighted-interval
    ``N`` + (s,e,w) triples; stride detected from header-count vs body length) and 2D grids
    (``R C`` + R row strings ⇒ top-left r'×c' sub-grids).

    Parses X as (header ints on line 0, flattened body tokens after) and produces VALID smaller
    sub-instances of the SAME problem; a malformed parse simply yields inputs the reference oracle
    rejects downstream (filtered), so this is safe.  Falls back to the generic 1D/integer ladder
    (``_subinstances_v1``) for unrecognised shapes.  Leakage-safe: it emits only INPUTS — the answers
    come from the owned oracle on disjoint small inputs and grading is on the disjoint hidden bank."""
    lines = x.split("\n")
    if not lines:
        return []
    header = lines[0].split()
    body = " ".join(lines[1:]).split()

    # shape 1: header all-int + numeric body ⇒ interleaved tuples / 1D array (stride = body/count)
    if (header and all(_is_int_token(h) for h in header)
            and body and all(_is_int_token(b) for b in body)):
        best: Optional[tuple[int, int, int]] = None   # (p, stride, count) with the finest decomposition
        for p, h in enumerate(header):
            c = int(h)
            if 1 <= c <= len(body) and len(body) % c == 0:
                stride = len(body) // c
                if 1 <= stride <= 6 and (best is None or c > best[2]):
                    best = (p, stride, c)
        if best is not None:
            p, stride, count = best
            out: list[str] = []
            for m in _ladder_sizes(count):
                nh = list(header)
                nh[p] = str(m)
                out.append(" ".join(nh) + "\n" + " ".join(body[:m * stride]))
            return out

    # shape 2: 2D grid 'R C' + R string rows of length C ⇒ top-left r'×c' sub-grids
    if len(header) >= 2 and _is_int_token(header[0]) and _is_int_token(header[1]):
        R, C = int(header[0]), int(header[1])
        rows = body
        if (R >= 1 and C >= 1 and len(rows) >= R and all(len(t) == C for t in rows[:R])
                and not all(_is_int_token(t) for t in rows[:R])):
            grid = rows[:R]
            out_g: list[str] = []
            for rp, cp in _grid_ladder_pairs(R, C):
                sub_rows = [row[:cp] for row in grid[:rp]]
                out_g.append(f"{rp} {cp}\n" + " ".join(sub_rows))
            return out_g

    # fallback: the generic 1D / single-integer ladder (reused from W135)
    return _subinstances_v1(x)


# ===================================================== 2-D subproblem-state grid (AT2 full table)

GRID_MAX_DIM: int = 4   # cap each grid axis (bounded prompt; arXiv:2505.04441 longer-traces-hurt)


def _axis_ladder(n: int, k: int = GRID_MAX_DIM) -> list[int]:
    """A small monotone ladder of <=k distinct values in 1..n (the sub-instance sizes along one axis)."""
    if n <= k:
        return list(range(1, n + 1))
    cand = sorted({1, 2, max(1, n // 2), n - 1, n})
    return [v for v in cand if 1 <= v <= n][:k]


def _state_grid_subinstances_v1(x: str):
    """For a 2-PARAMETER DP family, return (p1_name, p2_name, p1_vals, p2_vals, cells) where
    ``cells[(i, j)]`` is a sub-instance string with parameter-1 = i and parameter-2 = j — the exact
    SUBPROBLEM-STATE TABLE (AT2) the operator's spec calls for, assembled from BLACK-BOX oracle calls on
    disjoint small sub-instances (no solver instrumentation, no recurrence stated).  Returns None for
    non-2-parameter shapes.

    * knapsack-shape ``N K`` + body stride-s over N  ⇒ axes (items i=1..N, capacity j over a small
      ladder up to K) — the textbook (items × capacity) DP state, varied along BOTH axes;
    * grid-shape ``R C`` + R row strings of length C  ⇒ axes (rows r'=1..R, cols c'=1..C) — sub-grids
      to the top-left, varied along BOTH axes.

    Leakage-safe: emits only INPUTS; cell VALUES come from the owned oracle on disjoint small inputs and
    grading is on the disjoint hidden bank, so the model must recover + GENERALISE the recurrence."""
    lines = x.split("\n")
    if not lines:
        return None
    header = lines[0].split()
    body = " ".join(lines[1:]).split()
    # knapsack-shape: header [N, K], numeric body, stride = body/N
    if (len(header) == 2 and all(_is_int_token(h) for h in header)
            and body and all(_is_int_token(b) for b in body)):
        N, K = int(header[0]), int(header[1])
        if N >= 2 and K >= 2 and len(body) % N == 0:
            stride = len(body) // N
            if 1 <= stride <= 4:
                p1 = _axis_ladder(N)                                          # items
                p2 = sorted({max(1, K // 4), max(1, K // 2), max(1, 3 * K // 4), K})[:GRID_MAX_DIM]
                cells = {(i, j): f"{i} {j}\n" + " ".join(body[:i * stride]) for i in p1 for j in p2}
                return ("items", "capacity", p1, p2, cells)
    # grid-shape: header [R, C], R row strings of length C
    if len(header) >= 2 and _is_int_token(header[0]) and _is_int_token(header[1]):
        R, C = int(header[0]), int(header[1])
        rows = body
        if (R >= 2 and C >= 2 and len(rows) >= R and all(len(t) == C for t in rows[:R])
                and not all(_is_int_token(t) for t in rows[:R])):
            grid = rows[:R]
            p1 = _axis_ladder(R)
            p2 = _axis_ladder(C)
            cells = {(rp, cp): f"{rp} {cp}\n" + " ".join(row[:cp] for row in grid[:rp])
                     for rp in p1 for cp in p2}
            return ("rows", "cols", p1, p2, cells)
    return None


@dataclasses.dataclass(frozen=True)
class StateGridV1:
    """The exact 2-D subproblem-state table (AT2), assembled from black-box oracle calls.  ``opt[(i,j)]``
    is the optimal sub-value; ``naive[(i,j)]`` is the wrong algorithm's sub-value (the dual 2-D
    trajectory — where the naive's recurrence diverges in state space)."""

    p1_name: str
    p2_name: str
    p1_vals: tuple[int, ...]
    p2_vals: tuple[int, ...]
    opt: dict[str, str]      # "i,j" -> optimal value
    naive: dict[str, str]    # "i,j" -> naive value ("" if the naive failed)
    leakage_clean: bool

    def n_cells(self) -> int:
        return len(self.opt)

    def n_divergent_cells(self) -> int:
        return sum(1 for k, v in self.opt.items() if self.naive.get(k, "") not in ("", v))

    def to_table_text(self) -> str:
        """Render the dual 2-D state table (optimal grid; naive grid only where it diverges)."""
        cols = list(self.p2_vals)
        head = f"      {self.p2_name}:" + " ".join(f"{c:>8}" for c in cols)
        lines = [f"subproblem-state table (optimal value for each ({self.p1_name}, {self.p2_name}); "
                 "each cell independently verified by the reference oracle):", head]
        for i in self.p1_vals:
            cells = []
            for j in cols:
                v = self.opt.get(f"{i},{j}", "?")
                cells.append(f"{v:>8}")
            lines.append(f"  {self.p1_name}={i:<3}" + " ".join(cells))
        # the naive's divergence cells (where the wrong recurrence departs)
        div = [(k, self.opt[k], self.naive.get(k, "")) for k in self.opt
               if self.naive.get(k, "") not in ("", self.opt[k])]
        if div:
            lines.append("your_approach diverges from the optimum at these "
                         f"({self.p1_name},{self.p2_name}) cells (optimal vs yours):")
            for k, o, n in div[:GRID_MAX_DIM * GRID_MAX_DIM]:
                lines.append(f"  ({k}): optimal {o} vs yours {n}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"p1_name": self.p1_name, "p2_name": self.p2_name, "p1_vals": list(self.p1_vals),
                "p2_vals": list(self.p2_vals), "n_cells": self.n_cells(),
                "n_divergent_cells": self.n_divergent_cells(), "leakage_clean": bool(self.leakage_clean)}


# ===================================================== the trace builder

def build_algorithm_state_trace_v1(code: str, problem: MintedProblemV1, probe: WitnessProbeSetV1,
                                   template: MintedTemplateV1, *,
                                   timeout_s: float = TRACE_PROBE_TIMEOUT_S,
                                   oracle_timeout_s: float = ORACLE_SUBINSTANCE_TIMEOUT_S,
                                   ) -> AlgorithmStateTraceV1:
    """Build the machine-structured algorithm-state trace for one candidate.

    Anchors on a token-minimal fresh disjoint counterexample (reusing EW1 — the SAME anchor as the W135
    ladder, so the trace-vs-prose contrast is clean); if the candidate is value-correct on every small
    probe (no counterexample) the trace is NONE (the complexity negative control — no counterexample ⇒ no
    state to trace).  Then runs BOTH ``ref_source`` (optimal) AND ``naive_source`` (the wrong algorithm)
    on the OBVIOUS disjoint sub-instances of X, building the dual-trajectory state-transition table with
    per-step increment deltas and the marked first-divergence step."""
    ce = find_counterexample_witness_v1(code, problem, probe, template, timeout_s=timeout_s)
    if not ce.found():
        return _none_trace(ce)

    secret_inputs = {inp for inp, _ in problem.secret_cases}
    x = ce.probe_input
    vstar = ce.expected_output            # = ref(X), already computed by EW1
    nr = _exec_capture_v1(template.naive_source, x, timeout_s=oracle_timeout_s)
    vnaive = nr.stdout.strip()[:MAX_VALUE_CHARS] if (not nr.timed_out and nr.returncode == 0) else ""
    gap, overcounts = _scalar_gap_v1(vstar, vnaive)

    rows: list[StateTransitionRowV1] = []
    seen: set = set()
    prev_opt: Optional[int] = None
    prev_naive: Optional[int] = None
    for s in _typed_subinstances_v1(x):
        if s in seen:
            continue
        seen.add(s)
        if s in secret_inputs:          # leakage: never reveal an oracle value of a graded case
            continue
        ro = _exec_capture_v1(template.ref_source, s, timeout_s=oracle_timeout_s)
        if ro.timed_out or ro.returncode != 0:
            continue
        opt = ro.stdout.strip()[:MAX_VALUE_CHARS]
        rn = _exec_capture_v1(template.naive_source, s, timeout_s=oracle_timeout_s)
        nai = rn.stdout.strip()[:MAX_VALUE_CHARS] if (not rn.timed_out and rn.returncode == 0) else ""
        oi, ni = _int_or_none(opt), _int_or_none(nai)
        odelta = str(oi - prev_opt) if (oi is not None and prev_opt is not None) else ""
        ndelta = str(ni - prev_naive) if (ni is not None and prev_naive is not None) else ""
        rows.append(StateTransitionRowV1(
            idx=len(rows), size_tokens=_tok_count(s), summary=_summarize_typed_subinstance_v1(s),
            optimal_value=opt, naive_value=nai, diverges=bool(nai != "" and nai != opt),
            optimal_delta=odelta, naive_delta=ndelta, leakage_clean=True))
        if oi is not None:
            prev_opt = oi
        if ni is not None:
            prev_naive = ni
        if len(rows) >= TRACE_MAX_ROWS:
            break

    # assemble the full 2-D subproblem-state grid (AT2) for 2-parameter families (black-box oracle only)
    grid_obj: Optional[StateGridV1] = None
    g = _state_grid_subinstances_v1(x)
    if g is not None:
        p1n, p2n, p1v, p2v, cells = g
        opt_cells: dict[str, str] = {}
        naive_cells: dict[str, str] = {}
        gleak = True
        for (i, j), sub in cells.items():
            if sub in secret_inputs:        # never reveal an oracle value of a graded case
                gleak = False
                continue
            ro = _exec_capture_v1(template.ref_source, sub, timeout_s=oracle_timeout_s)
            if ro.timed_out or ro.returncode != 0:
                continue
            opt_cells[f"{i},{j}"] = ro.stdout.strip()[:MAX_VALUE_CHARS]
            rn = _exec_capture_v1(template.naive_source, sub, timeout_s=oracle_timeout_s)
            naive_cells[f"{i},{j}"] = (rn.stdout.strip()[:MAX_VALUE_CHARS]
                                       if (not rn.timed_out and rn.returncode == 0) else "")
        if len(opt_cells) >= 4:
            grid_obj = StateGridV1(p1_name=p1n, p2_name=p2n, p1_vals=tuple(p1v), p2_vals=tuple(p2v),
                                   opt=opt_cells, naive=naive_cells, leakage_clean=gleak)

    first_div = next((r.idx for r in rows if r.diverges), -1)
    leak_clean = bool(ce.leakage_clean and all(r.leakage_clean for r in rows)
                      and (grid_obj is None or grid_obj.leakage_clean))
    has_contrast = bool(vnaive != "" and vnaive != vstar)
    has_dual = any(r.naive_value != "" for r in rows)

    # kind classification (oracle-side route for the rendering EMPHASIS; the mode is a property of the
    # owned battlefield, NEVER shown to the model — it only makes the prose phrasing CORRECT, e.g. a
    # counting problem gets count-talk and a greedy-vs-DP optimisation gets divergence-talk; the
    # count-vs-scalar-gap ambiguity is otherwise indistinguishable from outputs alone).  The trace
    # CONTENT (the dual-trajectory table) is purely behaviour-derived.
    mode = getattr(problem, "mode", "")
    if mode == MODE_SEARCH_ENUM and has_contrast:
        kind, fam = TRACE_SEARCH_FRONTIER, "AT3"          # the search-frontier count contrast
    elif has_dual and first_div >= 0:
        kind, fam = TRACE_DECISION_PATH, "AT1"            # the marked decision-path divergence
    elif len(rows) >= 2:
        kind, fam = TRACE_SUBPROBLEM_STATE, "AT2"         # the optimal-substructure state table
    elif has_contrast:
        kind, fam = TRACE_DECISION_PATH, "AT1"
    else:
        kind, fam = TRACE_SUBPROBLEM_STATE, "AT2"  # degrade; the genuinely-new guard will judge it
    return AlgorithmStateTraceV1(
        kind=kind, at_family=fam, counterexample=ce, optimal_value=vstar, naive_value=vnaive,
        objective_gap=gap, naive_overcounts=overcounts, rows=tuple(rows),
        first_divergence_idx=int(first_div), leakage_clean=leak_clean, grid=grid_obj)


def trace_is_genuinely_new_vs_structure_v1(trace: AlgorithmStateTraceV1,
                                           problem: MintedProblemV1) -> dict[str, Any]:
    """Machine-checkable 'not a prose witness in JSON clothing' test.  The trace is genuinely NEW vs the
    W135 prose structure witness (S4) iff it carries the DUAL-TRAJECTORY + TRANSITION structure S4's
    flat optimal-only ladder structurally lacks: (a) >=1 row with BOTH the optimal AND the naive value
    and a ``diverges`` flag (the trajectory pair), AND (b) a marked first-divergence step OR a per-step
    increment (delta) trajectory (the transition structure) — AND it is leakage-clean (X + every
    sub-instance disjoint from the graded bank).  A trace that reduces to S4's optimal-only ladder (no
    naive trajectory, no divergence, no deltas) is NOT genuinely-new."""
    sample_inputs = {inp for inp, _ in problem.samples}
    found = trace.found()
    ce = trace.counterexample
    is_new_input = bool(found and ce.probe_input not in sample_inputs)
    has_dual = trace.has_dual_trajectory()
    has_transition = trace.has_transition_structure()
    carries_state = bool(has_dual and has_transition)
    genuinely_new = bool(found and is_new_input and carries_state and trace.leakage_clean)
    return {"genuinely_new": genuinely_new, "found": found,
            "input_not_a_public_sample": is_new_input,
            "has_dual_trajectory": has_dual, "has_transition_structure": has_transition,
            "n_divergent_rows": trace.n_divergent_rows(), "first_divergence_idx": trace.first_divergence_idx,
            "n_rows": len(trace.rows), "leakage_clean": bool(trace.leakage_clean),
            "kind": trace.kind, "at_family": trace.at_family}


# ===================================================== forward-only T2 controller route (weightless)

def route_trace_action_v1(trace: AlgorithmStateTraceV1, *, stderr_tail: str, timed_out: bool
                          ) -> tuple[ControllerAction, str]:
    """Forward-only, weightless trace-conditioned controller (the T2 route).  Bridges the W125
    ``ControllerAction`` arsenal + the W111 typed ``FailureDigestV1`` to the trace path: it builds a
    digest from the last attempt's executor stderr (the SAME signal reflexion B saw — never the hidden
    test source) and routes between trace renderings on (trace features, digest).  No learned weights;
    deterministic.

    * a crash / exception with NO usable state trace        -> DRAFT  (fall back to the counterexample)
    * a dual-trajectory trace with a clear early divergence  -> PATCH  (targeted state-divergence rewrite)
    * a dual-trajectory trace whose divergence is structural -> REPLAN (re-derive the recurrence)
    * a value-correct candidate (no counterexample)          -> ABSTAIN (the trace has nothing to teach)
    """
    digest = parse_failure_digest_v1(stderr_tail=str(stderr_tail or ""), timed_out=bool(timed_out))
    if not trace.found():
        return ControllerAction.ABSTAIN, "no_counterexample_value_correct"
    if digest.exception_type and digest.exception_type not in ("Timeout",) and not trace.has_dual_trajectory():
        return ControllerAction.DRAFT, "crash_no_state_trace_fallback_counterexample"
    if trace.has_dual_trajectory() and 0 <= trace.first_divergence_idx <= 1:
        return ControllerAction.PATCH, "early_divergence_targeted_rewrite"
    if trace.has_dual_trajectory() and trace.has_transition_structure():
        return ControllerAction.REPLAN, "structural_divergence_recurrence_rewrite"
    return ControllerAction.DRAFT, "weak_trace_fallback_counterexample"


def _trace_block_for_action_v1(trace: AlgorithmStateTraceV1, action: ControllerAction) -> str:
    """The feedback rendering the T2 route selects.  PATCH/REPLAN -> the full machine-structured capsule
    (the divergence/recurrence emphasis is already kind-routed inside the capsule); DRAFT -> the bare
    counterexample (no state table); ABSTAIN -> the bare counterexample (correctly minimal)."""
    if action in (ControllerAction.PATCH, ControllerAction.REPLAN):
        return trace.to_capsule_block(ARM_T1_TRACE_REWRITE)
    # DRAFT / ABSTAIN: fall back to the bare counterexample (never WORSE than C1)
    return trace.counterexample.to_prompt_block()


# ===================================================== trace-reflexion prompt + arm

def _trace_reflexion_prompt(problem, history, block: str, attempt_idx: int) -> str:
    """Between-attempt prompt: the SAME scaffold as the W120 reflexion prompt (judge bit + stderr +
    public-sample results) PLUS the machine-structured algorithm-state trace block — a strict superset
    of the blind feedback AND of EW1's counterexample, so any gain is attributable to the trace."""
    chunks: list[str] = []
    for i, (cand, passed, stderr_tail, sample_fb) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (cand[:1500] + "\n# ...(truncated)\n")
        verdict = ("ACCEPTED by the judge (all hidden tests passed)" if passed
                   else "REJECTED by the judge (failed at least one hidden test)")
        se = f"\nExecutor stderr (tail):\n{stderr_tail.strip()}" if stderr_tail.strip() else ""
        sf = f"\nPublic sample results:\n{sample_fb}" if sample_fb.strip() else ""
        chunks.append(f"--- Attempt {i+1} ({verdict}) ---\n"
                      f"```python\n{cand_trim}\n```{se}{sf}")
    return (
        "You are an expert ICPC competitor on a reflective debugging loop. You are on "
        f"attempt {attempt_idx + 1} out of 5. Below are your previous attempts with the judge verdict "
        "and the PUBLIC sample-case results, followed by a MACHINE-READABLE algorithm-state trace of the "
        "last attempt's failure derived from a reference oracle. Use the trace to produce a NEW corrected "
        "COMPLETE Python 3 stdin/stdout program. Do not repeat a previous attempt verbatim and do not "
        "hard-code the shown values.\n\n"
        f"Problem:\n{problem.statement}\n\n"
        f"{_samples_block(problem)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        f"=== Machine-readable algorithm-state trace of attempt {len(history)} ===\n{block}")


@dataclasses.dataclass(frozen=True)
class TraceArmTraceV1:
    """Per-problem audit of one trace arm: which trace fired + which controller action routed each
    invoked attempt."""
    problem_id: str
    arm_id: str
    trace_kinds: tuple[str, ...]
    at_families: tuple[str, ...]
    controller_actions: tuple[str, ...]
    any_trace_found: bool
    any_genuinely_new: bool
    all_leakage_clean: bool
    max_rows: int
    max_divergent_rows: int

    def rescue_is_structural(self) -> bool:
        """True iff any invoked attempt fired a genuinely-new state trace (dual trajectory + transition),
        i.e. the arm's effect is NOT trivial parse/format repair."""
        return bool(self.any_genuinely_new)

    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "arm_id": self.arm_id,
                "trace_kinds": list(self.trace_kinds), "at_families": list(self.at_families),
                "controller_actions": list(self.controller_actions),
                "any_trace_found": bool(self.any_trace_found),
                "any_genuinely_new": bool(self.any_genuinely_new),
                "all_leakage_clean": bool(self.all_leakage_clean),
                "max_rows": int(self.max_rows), "max_divergent_rows": int(self.max_divergent_rows)}


def run_trace_arm_v1(*, seed: int, template: MintedTemplateV1, problem: MintedProblemV1,
                     probe: WitnessProbeSetV1, gen, K: int, temperature: float, max_tokens: int,
                     timeout_s: float, arm: str, minted_date: str,
                     witness_timeout_s: float = TRACE_PROBE_TIMEOUT_S,
                     oracle_timeout_s: float = ORACLE_SUBINSTANCE_TIMEOUT_S,
                     ) -> tuple[IcpcArmOutcomeV1, TraceArmTraceV1]:
    """Same-budget algorithm-state-trace reflexion arm.  Byte-identical structure to the W120 ``_run_b``
    (attempt-0 = the standard initial prompt; K attempts; one model call per attempt; no early stop),
    except the between-attempt feedback object is the machine-structured trace.

    * T1 — always render the FULL machine-structured trace capsule (the LEAD; if T1 does not beat S4 no
      trace arm would, since it renders the richest applicable trace).
    * T2 — a forward-only, weightless controller routes between the full capsule and the bare
      counterexample fallback per ``route_trace_action_v1`` (so T2 is never WORSE than C1).

    The model is graded ONLY on ``problem.secret_cases`` (the audited grader); the trace is computed from
    a FRESH disjoint probe set — never the graded cases."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass_idx = -1
    tkinds: list[str] = []
    tfams: list[str] = []
    actions: list[str] = []
    any_new = False
    leak_clean = True
    max_rows = 0
    max_div = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            last_code, _lp, last_stderr, _lsfb = history[-1]
            trace = build_algorithm_state_trace_v1(last_code, problem, probe, template,
                                                   timeout_s=witness_timeout_s,
                                                   oracle_timeout_s=oracle_timeout_s)
            tkinds.append(trace.kind)
            tfams.append(trace.at_family)
            gn = trace_is_genuinely_new_vs_structure_v1(trace, problem)
            any_new = any_new or bool(gn["genuinely_new"])
            max_rows = max(max_rows, len(trace.rows))
            max_div = max(max_div, trace.n_divergent_rows())
            if trace.found() and not trace.leakage_clean:
                leak_clean = False
            if arm == ARM_T2_TRACE_CONTROLLER:
                action, _why = route_trace_action_v1(
                    trace, stderr_tail=last_stderr, timed_out=False)
                actions.append(action.value if hasattr(action, "value") else str(action))
                block = _trace_block_for_action_v1(trace, action)
            else:  # T1 — full capsule
                actions.append("FULL_TRACE")
                block = trace.to_capsule_block(ARM_T1_TRACE_REWRITE)
            prompt = _trace_reflexion_prompt(pilot, tuple(history), block, attempt_idx=k)
        text, _ = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=timeout_s)
        per_call.append(bool(passed))
        sfb = sample_feedback_v1(pilot, code, timeout_s=timeout_s)
        history.append((code, bool(passed), stderr_tail, sfb))
        if passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id=str(arm),
        final_passed=bool(first_pass_idx >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass_idx))
    trace_audit = TraceArmTraceV1(
        problem_id=problem.problem_id, arm_id=str(arm),
        trace_kinds=tuple(tkinds), at_families=tuple(tfams), controller_actions=tuple(actions),
        any_trace_found=any(kk != TRACE_NONE for kk in tkinds),
        any_genuinely_new=bool(any_new), all_leakage_clean=bool(leak_clean),
        max_rows=int(max_rows), max_divergent_rows=int(max_div))
    return outcome, trace_audit


def code_fails_public_io_v1(code: str, pilot, *, timeout_s: float = 4.0, n_samples: int = 3) -> bool:
    """Execution-grounded I/O check (the W136 root-cause detector): does the model's OWN code crash,
    print nothing, or print an 'invalid'/'error' marker on a VALID PUBLIC sample input?  The traps'
    public samples are the SAME whitespace-flattened shape as the hidden cases, so a per-line parser
    fails the public samples too — a clean, no-leakage signal that the bug is I/O parsing, not the
    algorithm.  Pure / deterministic except the (audited) execution subprocess."""
    for inp, _exp in pilot.samples[:n_samples]:
        r = _exec_capture_v1(code, inp, timeout_s=timeout_s)
        out = (r.stdout or "").strip().lower()
        if r.timed_out or r.returncode != 0 or out == "" or "invalid" in out or "error" in out:
            return True
    return False


def run_io_grounded_trace_arm_v1(*, seed: int, template: MintedTemplateV1, problem: MintedProblemV1,
                                 probe: WitnessProbeSetV1, gen, K: int, temperature: float, max_tokens: int,
                                 timeout_s: float, minted_date: str,
                                 witness_timeout_s: float = TRACE_PROBE_TIMEOUT_S,
                                 oracle_timeout_s: float = ORACLE_SUBINSTANCE_TIMEOUT_S,
                                 ) -> tuple[IcpcArmOutcomeV1, TraceArmTraceV1]:
    """Same-budget EXECUTION-GROUNDED I/O-repair arm (T_IO; the W136 root-cause win).  Byte-identical
    structure to the W120 ``_run_b`` except the between-attempt feedback is the machine-structured trace
    PREFIXED with the generic ``IO_REPAIR_DIRECTIVE`` IFF the model's own last code fails a valid public
    sample (``code_fails_public_io_v1``).  No leakage: the directive is generic whitespace-parsing
    guidance triggered by the model's OWN crash; the model is graded ONLY on the disjoint hidden bank."""
    pilot = problem.to_pilot_problem(minted_date=str(minted_date))
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass_idx = -1
    tkinds: list[str] = []
    tfams: list[str] = []
    actions: list[str] = []
    any_new = False
    leak_clean = True
    max_rows = 0
    max_div = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            last_code = history[-1][0]
            trace = build_algorithm_state_trace_v1(last_code, problem, probe, template,
                                                   timeout_s=witness_timeout_s, oracle_timeout_s=oracle_timeout_s)
            tkinds.append(trace.kind)
            tfams.append(trace.at_family)
            gn = trace_is_genuinely_new_vs_structure_v1(trace, problem)
            any_new = any_new or bool(gn["genuinely_new"])
            max_rows = max(max_rows, len(trace.rows))
            max_div = max(max_div, trace.n_divergent_rows())
            if trace.found() and not trace.leakage_clean:
                leak_clean = False
            io_bad = code_fails_public_io_v1(last_code, pilot, timeout_s=witness_timeout_s + 2.0)
            actions.append("IO_REPAIR" if io_bad else "TRACE_ONLY")
            cap = trace.to_capsule_block(ARM_T1_TRACE_REWRITE)
            block = (IO_REPAIR_DIRECTIVE + cap) if io_bad else cap
            prompt = _trace_reflexion_prompt(pilot, tuple(history), block, attempt_idx=k)
        text, _ = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=timeout_s)
        per_call.append(bool(passed))
        sfb = sample_feedback_v1(pilot, code, timeout_s=timeout_s)
        history.append((code, bool(passed), stderr_tail, sfb))
        if passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    outcome = IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=problem.problem_id, arm_id=ARM_T_IO, final_passed=bool(first_pass_idx >= 0),
        n_model_calls=int(K), per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass_idx))
    audit = TraceArmTraceV1(
        problem_id=problem.problem_id, arm_id=ARM_T_IO, trace_kinds=tuple(tkinds), at_families=tuple(tfams),
        controller_actions=tuple(actions), any_trace_found=any(kk != TRACE_NONE for kk in tkinds),
        any_genuinely_new=bool(any_new), all_leakage_clean=bool(leak_clean), max_rows=int(max_rows),
        max_divergent_rows=int(max_div))
    return outcome, audit


__all__ = [
    "W136_ALGORITHM_STATE_TRACE_V1_SCHEMA_VERSION",
    "TRACE_DECISION_PATH", "TRACE_SUBPROBLEM_STATE", "TRACE_SEARCH_FRONTIER", "TRACE_NONE",
    "ARM_T1_TRACE_REWRITE", "ARM_T2_TRACE_CONTROLLER", "ARM_T_IO", "IO_REPAIR_DIRECTIVE",
    "code_fails_public_io_v1", "run_io_grounded_trace_arm_v1", "TRACE_ARMS",
    "TRACE_MAX_ROWS", "TRACE_PROBE_TIMEOUT_S", "ORACLE_SUBINSTANCE_TIMEOUT_S", "GRID_MAX_DIM",
    "StateTransitionRowV1", "StateGridV1", "AlgorithmStateTraceV1",
    "build_algorithm_state_trace_v1", "trace_is_genuinely_new_vs_structure_v1",
    "route_trace_action_v1", "TraceArmTraceV1", "run_trace_arm_v1",
]
