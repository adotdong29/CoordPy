"""W127 Lane α — Resistant Capability Atlas V1 (COO-9 sibling).

W126 proved the 22 uniformly-unsolved resistant ICPC problems are **capability
failures**: 894 leakage-clean synthesized candidates reach an ORACLE ceiling of 0/22 ⇒
deterministic recombination/repair/consensus over the already-paid generations cannot
manufacture a correct algorithm at $0.  "Capability failure" is where W126 stopped.
W127 Lane α refuses to stop there and asks: **WHICH capabilities are missing, HOW MANY
problems per capability, and which look plausibly recoverable with a family-specific
scaffold?**

This module builds a machine-checkable capability atlas for those 22 problems
(RUNBOOK_W127 § 2).  It separates TWO signal layers explicitly:

* **HARD, re-executable signals** (from the W126 grade cache + the already-graded
  generations): failure visibility, the typed per-generation failure-category
  distribution, generation diversity, best public-sample pass fraction.  These are
  reproducible facts, not heuristics.
* **SOFT, transparent, evidence-recorded heuristic layer**: a deterministic
  lexicon + code-signal classifier over PUBLIC inputs only (statement + public samples +
  the model's OWN generations) that assigns a `dominant_algorithm_family` from a LOCKED
  10-family taxonomy, with the full score vector + the exact signal hits recorded.  This
  is machine-checkable (deterministic + auditable), NOT ground truth.

An INDEPENDENT analyst cross-check (`reference_family_signal`) infers the family from the
resistant target's OWN accepted-solution structure ONLY to report
`atlas_label_agreement` — a validation metric of the public-signal label.  It is NEVER
passed to any scaffold-generation path (the scaffold module does not import it); this is
the no-leakage boundary (RUNBOOK_W127 § 3).

Held OUTSIDE the stable SDK contract: explicit-import-only, ``coordpy/__init__.py``
untouched, ``coordpy.__version__ == "0.5.20"``, no PyPI publish.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Optional, Sequence

W127_RESISTANT_CAPABILITY_ATLAS_V1_SCHEMA_VERSION: str = (
    "coordpy.resistant_capability_atlas_v1.v1")

# ============================================================ LOCKED family taxonomy

# The SAME ordered taxonomy is used by Lane β's retriever + the "spans >= 2 families"
# earn metric (RUNBOOK_W127 § 2/§ 4/§ 5).  Order is the deterministic tie-break.
LOCKED_FAMILY_TAXONOMY: tuple[str, ...] = (
    "graph_flow",
    "dp_optimization",
    "geometry",
    "number_theory_math",
    "string_processing",
    "greedy_scheduling",
    "simulation_grid",
    "search_enumeration",
    "data_structure",
    "adhoc_math",
)

# Families with a well-defined, transferable reusable skeleton (RUNBOOK_W127 § 2).
SCAFFOLDABLE_FAMILIES: frozenset = frozenset({
    "graph_flow", "dp_optimization", "geometry", "number_theory_math",
    "string_processing", "data_structure", "greedy_scheduling",
})

# Per-family signal lexicons.  ``stmt`` hits (statement, public) weigh 3; ``code`` hits
# (the model's own generations, public) weigh 1.  Transparent + auditable, not learned.
FAMILY_LEXICON: dict[str, dict[str, tuple[str, ...]]] = {
    "graph_flow": {
        "stmt": ("graph", "vertex", "vertices", "edge", "edges", " node", " nodes",
                 "tree", "forest", "connected", "component", "directed", "undirected",
                 "neighbor", "adjacent", "shortest path", "spanning", "bipartite",
                 "matching", "max flow", "maximum flow", "min cut", "reachable",
                 "cycle", "dag", "topological"),
        "code": ("adjacency", "deque(", "collections.deque", "def bfs", "def dfs",
                 "dijkstra", "heappush", "heappop", "visited", "union(", "find("),
    },
    "dp_optimization": {
        "stmt": ("minimum number of", "maximum number of", "minimum cost", "maximum value",
                 "subsequence", "subarray", "partition", "knapsack", "longest",
                 "fewest", "minimize", "maximize", "number of ways to", "optimal",
                 "at most k", "best way"),
        "code": ("dp[", "memo", "lru_cache", "@cache", "[[0]", "[0] * (", "[0]*(",
                 "table[", "for i in range(n", "prev["),
    },
    "geometry": {
        "stmt": ("point", "points", "polygon", "circle", "rectangle", "triangle",
                 " area", "perimeter", "coordinate", "convex", "segment", "collinear",
                 "intersection", "radius", "vector", "angle", "euclidean", "distance"),
        "code": ("math.hypot", "math.sqrt", "math.atan2", "cross", " * dy", " * dx",
                 ".real", ".imag", "abs("),
    },
    "number_theory_math": {
        "stmt": ("modulo", "modular", "prime", "primes", "divisor", "divisible", "gcd",
                 " lcm", "factorial", "binomial", "combinations of", "probability",
                 "expected value", "remainder", "coprime", "congruent", "factor",
                 "10^9", "modulus", "number of integers"),
        "code": ("math.gcd", "pow(", "factorial", "math.comb", "1000000007", "10**9 + 7",
                 "10 ** 9 + 7", "% mod", "% (10"),
    },
    "string_processing": {
        "stmt": ("string", "strings", "substring", "palindrome", "character",
                 "characters", "letter", "letters", " word", " words", "prefix",
                 "suffix", "alphabet", "lowercase", "uppercase", "concatenat", "pattern"),
        "code": (".find(", ".count(", ".replace(", ".split(", "ord(", "chr(", "[::-1]",
                 ".startswith", ".endswith", "ascii"),
    },
    "greedy_scheduling": {
        "stmt": ("schedule", "scheduling", "interval", "intervals", "deadline",
                 "earliest", "latest", "as many as possible", "as few as possible",
                 "greedily", "non-overlapping", "assign each"),
        "code": (".sort(", "sorted(", "heapq", "key=lambda", "reverse=True"),
    },
    "simulation_grid": {
        "stmt": ("grid", "maze", " cell", " cells", " row ", " rows", "column",
                 "columns", "board", "robot", "moves", " step", " steps", "simulate",
                 "direction", "north", "south", "rotate", "game of"),
        "code": ("grid[", "board[", "directions", "dx", "dy", "for r in range",
                 "for c in range", "[[", "rows", "cols"),
    },
    "search_enumeration": {
        "stmt": ("all possible", "enumerate", "every possible", "try all", "brute",
                 "permutation of", "subsets", "combinations of", "backtrack",
                 "exhaustive", "all subsets"),
        "code": ("itertools", "product(", "permutations(", "combinations(",
                 "for mask in range(1 <<", "1 << n", "2 ** n", "def backtrack"),
    },
    "data_structure": {
        "stmt": ("query", "queries", "update", "updates", "range sum", "prefix sum",
                 "segment tree", "fenwick", "binary indexed", "priority queue",
                 " stack", " queue", "balanced", "online"),
        "code": ("bisect", "SortedList", "prefix", "fenwick", "seg_tree", "BIT[",
                 "import heapq"),
    },
    "adhoc_math": {
        "stmt": ("formula", "closed form", "arithmetic", "sum of the", "sequence",
                 "series", "ratio", "percentage", "average", "count the number of",
                 "how many", "total number"),
        "code": ("print(sum", "//", "math.", "round(", "int("),
    },
}

_STMT_WEIGHT = 3
_CODE_WEIGHT = 1


# ============================================================ text helpers

def _sha256_hex(payload: Any) -> str:
    if not isinstance(payload, (bytes, bytearray)):
        payload = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strip_latex(text: str) -> str:
    """Lightweight LaTeX → plain lowercase text for keyword matching."""
    t = str(text or "")
    t = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", " ", t)   # drop \commands{arg}
    t = re.sub(r"[{}$\\&%#~^_]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.lower()


_CONSTRAINT_RE = re.compile(
    r"(?:\\le|\\leq|<=|≤|at most|up to|no more than|less than or equal to)\s*"
    r"\\?[a-z]?\s*([0-9][0-9\s,]*(?:\^[0-9]+|e[0-9]+|\s*\\cdot\s*10\^?[0-9]+)?)",
    re.IGNORECASE)


def _extract_max_constraint(statement: str) -> int:
    """Best-effort largest stated upper bound (a complexity-pressure signal)."""
    raw = str(statement or "")
    best = 0
    # explicit 10^k / 10^{k}
    for m in re.finditer(r"10\s*\^?\s*\{?\s*([0-9]{1,2})\s*\}?", raw):
        try:
            best = max(best, 10 ** int(m.group(1)))
        except (ValueError, OverflowError):
            pass
    for m in _CONSTRAINT_RE.finditer(raw):
        tok = m.group(1)
        digits = re.sub(r"[^\d]", "", tok.split("^")[0])
        if digits:
            try:
                best = max(best, int(digits[:12]))
            except ValueError:
                pass
    return int(best)


# ============================================================ family classifiers

@dataclasses.dataclass(frozen=True)
class FamilyClassificationV1:
    family: str
    scores: dict
    evidence: dict           # family -> list of hit tokens
    margin: float            # top score - runner-up (label confidence)


def classify_family_v1(*, statement: str, sample_text: str,
                       generation_codes: Sequence[str]) -> FamilyClassificationV1:
    """Deterministic PUBLIC-signal family classifier (RUNBOOK_W127 § 2 item 7).

    Inputs are PUBLIC only: the statement, public samples, and the model's OWN
    generations.  Never touches secret cases or any accepted solution.  Returns the
    argmax family + the full score vector + the exact hit tokens (auditable)."""
    stmt = _strip_latex(statement) + " " + _strip_latex(sample_text)
    code = "\n".join(str(c) for c in generation_codes).lower()
    scores: dict[str, int] = {}
    evidence: dict[str, list] = {}
    for fam in LOCKED_FAMILY_TAXONOMY:
        lex = FAMILY_LEXICON[fam]
        hits: list[str] = []
        s = 0
        for kw in lex["stmt"]:
            n = stmt.count(kw)
            if n:
                s += _STMT_WEIGHT * min(n, 3)
                hits.append(f"stmt:{kw.strip()}({n})")
        for kw in lex["code"]:
            n = code.count(kw.lower())
            if n:
                s += _CODE_WEIGHT * min(n, 3)
                hits.append(f"code:{kw.strip()}({n})")
        scores[fam] = int(s)
        evidence[fam] = hits
    ordered = sorted(LOCKED_FAMILY_TAXONOMY, key=lambda f: (-scores[f],
                     LOCKED_FAMILY_TAXONOMY.index(f)))
    top = ordered[0]
    runner = scores[ordered[1]] if len(ordered) > 1 else 0
    # if nothing fires, fall back to adhoc_math (the residual family)
    family = top if scores[top] > 0 else "adhoc_math"
    margin = float(scores[top] - runner)
    return FamilyClassificationV1(family=family, scores=scores,
                                  evidence={k: v for k, v in evidence.items() if v},
                                  margin=margin)


# Code-structure signals for the reference cross-check (analyst-only; NOT model-facing).
_REF_FAMILY_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graph_flow", ("flow", "dinitz", "ford_fulkerson", "dijkstra", "bfs", "dfs",
                    "adjacency", "networkx", "union_find", "scc", "bipartite", "mst",
                    "kruskal", "prim")),
    ("dp_optimization", ("dp[", "memo", "lru_cache", "knapsack", "dp =", "dptable")),
    ("geometry", ("hypot", "atan2", "cross(", "convex_hull", "shoelace", "ccw(",
                  ".real", ".imag")),
    ("number_theory_math", ("gcd", "math.comb", "factorial", "sieve", "modpow",
                            "miller", "crt", "1000000007", "10**9+7")),
    ("string_processing", ("kmp", "z_function", "suffix", "trie", "rolling_hash",
                           "[::-1]", "palindrome")),
    ("data_structure", ("fenwick", "segment_tree", "segtree", "bisect", "sortedlist",
                        "bit[", "treap")),
    ("search_enumeration", ("itertools", "permutations", "combinations", "product(",
                            "backtrack", "1 <<")),
    ("greedy_scheduling", ("greedy", ".sort(", "heapq", "heappush")),
    ("simulation_grid", ("grid", "directions", "dx", "dy", "board")),
)


def classify_reference_family_v1(reference_texts: Sequence[str]) -> tuple[str, list]:
    """Analyst-only cross-check: infer the family from a target's OWN reference
    accepted-solution structure (RUNBOOK_W127 § 2 item 15).  NEVER model-facing — used
    solely to compute ``atlas_label_agreement``.  Works on .py/.cpp/.kt/.java text."""
    blob = "\n".join(str(t) for t in reference_texts).lower()
    if not blob.strip():
        return "unknown", []
    best_fam, best_hits = "unknown", []
    best_n = 0
    for fam, sigs in _REF_FAMILY_SIGNALS:
        hits = [s for s in sigs if s in blob]
        if len(hits) > best_n:
            best_n, best_fam, best_hits = len(hits), fam, hits
    return best_fam, best_hits


# ============================================================ failure decomposition

_FAIL_CATEGORIES = ("ok", "wrong_answer", "timeout", "runtime_error", "parse_error")


def _gen_failure_category(gen: dict) -> str:
    """Map one cached generation to the LOCKED 5-category failure taxonomy (the typed
    PUBLIC-sample digest already stored by the W126 recon; no re-execution)."""
    if not gen.get("parses", True):
        return "parse_error"
    exc = str(gen.get("digest_exc") or "").strip()
    if exc == "Timeout" or exc.lower().startswith("timeout"):
        return "timeout"
    if exc:
        return "runtime_error"
    if gen.get("secret_pass"):
        return "ok"
    return "wrong_answer"


# Naive / brute-force structural signals (a complexity-mismatch indicator).
_NAIVE_RE = re.compile(
    r"for .+:\s*\n\s+for .+:\s*\n\s+for .+:|itertools\.permutations|"
    r"itertools\.product|while True:|range\(1\s*<<", re.MULTILINE)


# ============================================================ atlas entry

@dataclasses.dataclass(frozen=True)
class CapabilityAtlasEntryV1:
    # hard, re-executable
    problem_id: str
    short_name: str
    surface: str
    contest_date: str
    n_samples: int
    n_secret: int
    failure_visibility: str            # visible | hidden
    n_generations: int
    n_distinct_codes: int
    n_distinct_digests: int
    digest_distribution: dict
    best_sample_pass_frac: float
    # soft, transparent heuristic layer
    dominant_algorithm_family: str
    family_scores: dict
    family_evidence: dict
    family_margin: float
    likely_missing_technique: str
    complexity_mismatch_evidence: dict
    parsing_impl_mismatch_evidence: dict
    teacher_family_coverage: int
    scaffoldable_flag: bool
    scaffoldable_reasons: dict
    # analyst cross-check (NEVER model-facing)
    reference_family_signal: str
    reference_family_hits: list
    atlas_label_agrees: bool
    family_confidence: str             # ref_confirmed | ref_conflict | unconfirmed

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_MISSING_TECHNIQUE = {
    "graph_flow": "graph modeling + a flow/matching/shortest-path/connectivity algorithm",
    "dp_optimization": "a correct DP state/transition (or proof a greedy is optimal)",
    "geometry": "computational-geometry primitives (orientation/area/intersection)",
    "number_theory_math": "modular arithmetic / combinatorics / number-theory identity",
    "string_processing": "string algorithm (pattern match / palindrome / suffix structure)",
    "greedy_scheduling": "the correct greedy exchange argument + ordering",
    "simulation_grid": "an exact stateful simulation / grid BFS with correct transitions",
    "search_enumeration": "a feasible search space + pruning (the naive enumeration is too big)",
    "data_structure": "an efficient query/update structure (segment/Fenwick/heap/bisect)",
    "adhoc_math": "the closed-form / case-analysis insight",
}


def _surface_of(problem_id: str, source_repo: str) -> str:
    pid = (problem_id + " " + str(source_repo)).lower()
    if "ecna" in pid:
        return "ECNA-NA-East-2024-2025"
    if "2025-2026" in pid:
        return "RMRC-2025-2026"
    if "2024-2025" in pid:
        return "RMRC-2024-2025"
    return str(source_repo)


def build_atlas_entry_v1(*, problem, generation_codes: Sequence[str],
                         cache_problem: dict, teacher_family_count: dict,
                         reference_texts: Sequence[str] = ()) -> CapabilityAtlasEntryV1:
    """Build one capability-atlas entry (RUNBOOK_W127 § 2)."""
    gens = list(cache_problem.get("gens", []))
    sample_text = "\n".join(f"{i}\n{o}" for i, o in getattr(problem, "samples", ()))
    # hard layer ------------------------------------------------------------------
    visibility = "hidden" if cache_problem.get("any_all_sample_pass") else "visible"
    dist = {c: 0 for c in _FAIL_CATEGORIES}
    for g in gens:
        dist[_gen_failure_category(g)] += 1
    best_frac = 0.0
    for g in gens:
        tot = g.get("sample_total") or 0
        if tot:
            best_frac = max(best_frac, float(g.get("sample_pass", 0)) / float(tot))
    # soft layer ------------------------------------------------------------------
    fc = classify_family_v1(statement=getattr(problem, "statement", ""),
                            sample_text=sample_text, generation_codes=generation_codes)
    fam = fc.family
    max_constraint = _extract_max_constraint(getattr(problem, "statement", ""))
    naive_signal = any(bool(_NAIVE_RE.search(str(c))) for c in generation_codes)
    complexity_ev = {
        "max_constraint_seen": int(max_constraint),
        "any_timeout": bool(dist["timeout"] > 0),
        "naive_signal": bool(naive_signal),
        "large_constraint": bool(max_constraint >= 100_000),
    }
    parsing_ev = {
        "any_parse_error": bool(dist["parse_error"] > 0),
        "sample_shape_mismatch": bool(best_frac == 0.0 and dist["parse_error"] == 0
                                      and dist["runtime_error"] == 0),
    }
    cov = int(teacher_family_count.get(fam, 0))
    reasons = {
        "teacher_coverage_ge_2": bool(cov >= 2),
        "family_has_skeleton": bool(fam in SCAFFOLDABLE_FAMILIES),
    }
    scaffoldable = bool(reasons["teacher_coverage_ge_2"] and reasons["family_has_skeleton"])
    # analyst cross-check (segregated) -------------------------------------------
    ref_fam, ref_hits = classify_reference_family_v1(reference_texts)
    agrees = bool(ref_fam != "unknown" and ref_fam == fam)
    if ref_fam == "unknown":
        confidence = "unconfirmed"          # no reference solution to validate against
    elif agrees:
        confidence = "ref_confirmed"        # the actual-algorithm signal confirms the label
    else:
        confidence = "ref_conflict"         # theme-biased: surface label != actual algorithm
    return CapabilityAtlasEntryV1(
        problem_id=str(getattr(problem, "problem_id", "")),
        short_name=str(getattr(problem, "short_name", "")),
        surface=_surface_of(getattr(problem, "problem_id", ""),
                            getattr(problem, "source_repo", "")),
        contest_date=str(getattr(problem, "contest_date", "")),
        n_samples=len(getattr(problem, "samples", ())),
        n_secret=len(getattr(problem, "secret_cases", ())),
        failure_visibility=visibility,
        n_generations=len(gens),
        n_distinct_codes=int(cache_problem.get("distinct_codes", 0)),
        n_distinct_digests=int(cache_problem.get("distinct_digests", 0)),
        digest_distribution=dist,
        best_sample_pass_frac=round(best_frac, 4),
        dominant_algorithm_family=fam,
        family_scores=fc.scores,
        family_evidence=fc.evidence,
        family_margin=fc.margin,
        likely_missing_technique=_MISSING_TECHNIQUE.get(fam, "unknown"),
        complexity_mismatch_evidence=complexity_ev,
        parsing_impl_mismatch_evidence=parsing_ev,
        teacher_family_coverage=cov,
        scaffoldable_flag=scaffoldable,
        scaffoldable_reasons=reasons,
        reference_family_signal=ref_fam,
        reference_family_hits=ref_hits,
        atlas_label_agrees=agrees,
        family_confidence=confidence)


# ============================================================ full atlas

@dataclasses.dataclass(frozen=True)
class CapabilityAtlasV1:
    schema: str
    n_problems: int
    entries: list
    cluster_counts: dict
    top_clusters: list
    dominant_cluster: str
    concentration_top2_frac: float
    scaffoldable_count: int
    scaffoldable_by_family: dict
    atlas_label_agreement: float
    n_ref_confirmed: int
    n_ref_conflict: int
    n_unconfirmed: int
    reference_cluster_counts: dict      # the more-accurate actual-algorithm diagnosis
    failure_mode_summary: dict          # hard layer roll-up across the 22
    atlas_cid: str

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["entries"] = [e.to_dict() if hasattr(e, "to_dict") else e for e in self.entries]
        return d


def build_capability_atlas_v1(entries: Sequence[CapabilityAtlasEntryV1]
                              ) -> CapabilityAtlasV1:
    """Cluster the entries by family + compute concentration (RUNBOOK_W127 § 2)."""
    counts: dict[str, int] = {f: 0 for f in LOCKED_FAMILY_TAXONOMY}
    ref_counts: dict[str, int] = {}
    scaff_by_fam: dict[str, int] = {f: 0 for f in LOCKED_FAMILY_TAXONOMY}
    n_agree = n_ref = 0
    n_conf = n_conflict = n_unconf = 0
    fail_modes: dict[str, int] = {c: 0 for c in _FAIL_CATEGORIES}
    n_visible = n_hidden = 0
    for e in entries:
        counts[e.dominant_algorithm_family] += 1
        if e.scaffoldable_flag:
            scaff_by_fam[e.dominant_algorithm_family] += 1
        # reference-space (best-available actual-algorithm) diagnosis
        rfam = e.reference_family_signal if e.reference_family_signal != "unknown" \
            else e.dominant_algorithm_family
        ref_counts[rfam] = ref_counts.get(rfam, 0) + 1
        if e.reference_family_signal != "unknown":
            n_ref += 1
            if e.atlas_label_agrees:
                n_agree += 1
        n_conf += int(e.family_confidence == "ref_confirmed")
        n_conflict += int(e.family_confidence == "ref_conflict")
        n_unconf += int(e.family_confidence == "unconfirmed")
        for c in _FAIL_CATEGORIES:
            fail_modes[c] += int(e.digest_distribution.get(c, 0))
        n_visible += int(e.failure_visibility == "visible")
        n_hidden += int(e.failure_visibility == "hidden")
    nonzero = {f: c for f, c in counts.items() if c > 0}
    ref_nonzero = dict(sorted(ref_counts.items(),
                              key=lambda kv: (-kv[1], kv[0])))
    ordered = sorted(nonzero, key=lambda f: (-nonzero[f], LOCKED_FAMILY_TAXONOMY.index(f)))
    n = max(1, len(entries))
    top2 = sum(nonzero[f] for f in ordered[:2])
    atlas = CapabilityAtlasV1(
        schema=W127_RESISTANT_CAPABILITY_ATLAS_V1_SCHEMA_VERSION,
        n_problems=len(entries),
        entries=list(entries),
        cluster_counts=nonzero,
        top_clusters=[{"family": f, "count": nonzero[f],
                       "scaffoldable": scaff_by_fam[f]} for f in ordered],
        dominant_cluster=ordered[0] if ordered else "",
        concentration_top2_frac=round(top2 / n, 4),
        scaffoldable_count=sum(1 for e in entries if e.scaffoldable_flag),
        scaffoldable_by_family={f: c for f, c in scaff_by_fam.items() if c > 0},
        atlas_label_agreement=round(n_agree / n_ref, 4) if n_ref else 0.0,
        n_ref_confirmed=n_conf, n_ref_conflict=n_conflict, n_unconfirmed=n_unconf,
        reference_cluster_counts=ref_nonzero,
        failure_mode_summary={"by_category": fail_modes, "n_visible": n_visible,
                              "n_hidden": n_hidden,
                              "wrong_answer_frac": round(
                                  fail_modes["wrong_answer"]
                                  / max(1, sum(fail_modes.values())), 4)},
        atlas_cid=_sha256_hex([e.problem_id + ":" + e.dominant_algorithm_family
                               for e in entries]))
    return atlas
