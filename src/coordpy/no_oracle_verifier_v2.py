"""W142b / COO-9 — robust no-oracle correct+efficient verifier (fixes the W142 FN + FP).

The W142 pilot capped because the v1 verifier (S1 = agree with ONE public-trusted self-brute) failed two
ways, both because a single self-brute validated only on the (non-discriminating) public samples is the
sole point of trust:
  * FALSE-NEGATIVE (count_pairs_sum): the model's self-brute was BUGGY (passes public, fails secret — a
    counting-convention bug); the 39/80 genuinely correct+efficient candidates DISAGREED with it ⇒ S1
    vetoed every correct candidate ⇒ 0 winners.
  * FALSE-POSITIVE (subarrays_sum_and_range): the sum-only naive agrees with the brute on the small bank
    (the range cap never binds there) ⇒ committed a wrong winner.

v2 fixes BOTH with one principled, still-no-oracle rule. Build a CONSTRAINT-COVERING bank (public-sample
arrays mutated to HIGH SPREAD + deterministic extremes — valid under the PUBLIC value range, so a cap that
binds only on spread is exercised in isolation). Cluster {public-passing self-brutes} ∪ {efficient,
public-passing candidates} by their output-vector on that bank. The **reference cluster = the LARGEST
cluster that contains ≥1 public-passing brute**; the winner is the efficient candidate in it (consensus).
  * subarrays: the correct two-constraint brute sits with the correct two-deque candidates; the sum-only
    naive cluster has NO brute ⇒ rejected (FP gone).
  * count_pairs: the 39 correct candidates + ≥1 correct brute form the largest brute-bearing cluster; the
    tiny buggy-brute cluster (no correct efficient member) loses (FN gone) — PROVIDED ≥1 of the K_b brutes
    is correct (so the controller generates K_b>1 brutes).
ABSTAIN (⇒ caller KEEPs, non-negative) when no brute-bearing cluster exists or it has no efficient member.

Strictly no-oracle: uses only the public statement+samples, self-generated inputs (high-spread mutations
of PUBLIC samples + spec-valid extremes), self-written brutes, and candidate runtime. NEVER the hidden
bank / ref / naive / brute answer-key. The extractor, leak gate, and NEG control are UNCHANGED (v2 changes
only winner SELECTION). Reuses v1's `_run_sig`, `_fast_efficient_v1`, `brute_is_trusted_v1`, `_StubProblem`,
`WinnerSelectionV1`, `CandidateVerdictV1` verbatim. Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import random as _random
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

from .no_oracle_verifier_v1 import (
    _run_sig, _fast_efficient_v1, brute_is_trusted_v1, _StubProblem,
    WinnerSelectionV1, CandidateVerdictV1, _CRASH, _TLE)
from .parser_neutral_io_v1 import (
    IoShapeV1, parse_all_tokens_v1, render_normal_form_v1, NormalFormError)

NO_ORACLE_VERIFIER_V2_SCHEMA_VERSION: str = "coordpy.no_oracle_verifier_v2.v1"

_AUG_SIZES: tuple[int, ...] = (12, 24, 40)
_MAX_CLUSTER_INPUT_CHARS: int = 1200   # cap clustering inputs to small (O(N^2) brutes stay fast/stable)


def _high_spread_arrays(n: int, value_hi: int, rng: _random.Random) -> list[list[int]]:
    """Spec-valid (1..value_hi) small arrays whose max-min SPREAD is large, so a range/extent cap binds
    in isolation while the SUM can stay moderate.  No oracle — built from the PUBLIC value range."""
    hi = max(2, min(int(value_hi), 10 ** 9))
    return [
        list(range(n, 0, -1)),                                   # strictly decreasing -> large spread
        [1] * (n // 2) + [hi] * (n - n // 2),                    # two-block extreme spread
        [1 if i % 2 == 0 else hi for i in range(n)],             # alternating min/max
        [rng.randint(1, hi) for _ in range(n)],                  # uniform over the full range
        [1] * n,                                                 # all-equal -> range cap never binds
    ]


_THRESH_LO: int = 1
_THRESH_HI: int = 10 ** 9


def _threshold_settings(thresholds: list[str]) -> list[dict[str, int]]:
    """Per-threshold isolation settings: each threshold set to LO while the rest are HI (so THAT cap
    binds while the others are slack), and vice-versa, plus all-LO and all-HI.  This exercises EACH
    stated constraint in isolation — the key to catching a multi-constraint naive that drops one cap."""
    out: list[dict[str, int]] = [{t: _THRESH_HI for t in thresholds}, {t: _THRESH_LO for t in thresholds}]
    for t in thresholds:
        out.append({**{u: _THRESH_HI for u in thresholds}, t: _THRESH_LO})   # isolate t binding (tight)
        out.append({**{u: _THRESH_LO for u in thresholds}, t: _THRESH_HI})   # isolate t slack
    # dedupe
    seen, uniq = set(), []
    for s in out:
        k = tuple(sorted(s.items()))
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq


def constraint_covering_bank_v1(io_shape: Optional[IoShapeV1], samples: Sequence[tuple[str, str]],
                                *, seed: int = 142) -> list[str]:
    """Mutate each PUBLIC sample to (HIGH-SPREAD array) × (each THRESHOLD scalar bound in isolation):
    set one threshold tight (LO) while the others are slack (HI) and vice-versa, so a cap that binds
    only on value-spread (a range/extent cap) is exercised WITHOUT the sum cap also binding.  This is
    what catches a multi-constraint naive that drops one cap (the W142 subarrays FP).  No oracle:
    public-sample mutation + spec-valid threshold settings (within the stated bounds) + the public
    value range.  Re-rendered in the family's canonical normal form."""
    if io_shape is None:
        return []
    rng = _random.Random(seed)
    value_hi = 20
    for inp, _ in samples:
        for tok in inp.split():
            try:
                value_hi = max(value_hi, abs(int(tok)))
            except ValueError:
                pass
    value_hi = max(value_hi, 1000)
    out: list[str] = []
    for inp, _ in samples:
        try:
            d = parse_all_tokens_v1(inp, io_shape)
        except Exception:  # noqa: BLE001
            continue
        arr_fields = [k for k, v in d.items()
                      if isinstance(v, list) and v and all(isinstance(x, int) for x in v)]
        if not arr_fields:
            continue
        af = arr_fields[0]
        n = len(d[af])
        if n < 2:
            continue
        # thresholds = integer scalars whose value is NOT the array length (heuristic: not the size N)
        thresholds = [k for k, v in d.items()
                      if isinstance(v, int) and v != n and k not in arr_fields]
        settings = _threshold_settings(thresholds) if thresholds else [{}]
        for variant in _high_spread_arrays(n, value_hi, rng):
            arr = variant[:n] if len(variant) >= n else variant + [variant[-1]] * (n - len(variant))
            for st in settings:
                d2 = dict(d)
                d2[af] = arr
                d2.update(st)
                try:
                    out.append(render_normal_form_v1(io_shape, d2))
                except Exception:  # noqa: BLE001
                    continue
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        k = s.strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq[:60]


def _output_vector(code: str, bank: Sequence[str], *, timeout_s: float) -> Optional[tuple[str, ...]]:
    """The candidate/brute's output on EVERY bank input; None if it crashes/TLEs on any.  (Kept for the
    $0 tests; the selector uses _vec_over which TOLERATES per-input crashes via markers.)"""
    row: list[str] = []
    for si in bank:
        s = _run_sig(code, si, timeout_s=timeout_s)
        if s.startswith((_CRASH, _TLE)) or not s:
            return None
        row.append(s)
    return tuple(row)


def _vec_over(code: str, bank: Sequence[str], *, timeout_s: float) -> tuple[str, ...]:
    """Output-vector over a fixed bank, with crash/TLE recorded as a marker token (NOT None) — so a
    witness that mishandles one extreme input still clusters by its behaviour on the rest (the v1
    tolerance restored: v1 SKIPPED brute-crash inputs)."""
    return tuple(_run_sig(code, si, timeout_s=timeout_s) for si in bank)


def _usable_bank(brutes: Sequence[str], bank: Sequence[str], *, timeout_s: float,
                 min_clean: int) -> list[str]:
    """Inputs where >= min_clean of the trusted brutes run CLEAN (no crash/TLE).  A large adversarial
    input that TLEs the O(N^2) brute is dropped from the CORRECTNESS-clustering bank (correctness is a
    small-input property; efficiency is the separate S2 check) — fixing the W142b 'one TLE excludes the
    whole brute' bug."""
    out = []
    for si in bank:
        n_clean = sum(1 for b in brutes
                      if not _run_sig(b, si, timeout_s=timeout_s).startswith((_CRASH, _TLE)))
        if n_clean >= min_clean:
            out.append(si)
    return out


def _passes_public_v2(code: str, samples: Sequence[tuple[str, str]], *, timeout_s: float) -> bool:
    for inp, exp in samples:
        if _run_sig(code, inp, timeout_s=timeout_s) != exp.strip():
            return False
    return True


def select_winner_v2(candidates: Sequence[str], *, statement: str,
                     samples: Sequence[tuple[str, str]], small_inputs: Sequence[str],
                     brute_codes: Sequence[str], io_shape: Optional[IoShapeV1] = None,
                     consensus_probe: str = "", timeout_s: float = 4.0) -> WinnerSelectionV1:
    """Robust no-oracle winner: reference = the largest output-cluster (over public-passing brutes +
    efficient public-passing candidates, on a constraint-covering bank) that CONTAINS a public-passing
    brute; winner = an efficient candidate in it.  ABSTAIN (caller KEEPs) otherwise.  Field-compatible
    with ``select_winner_v1`` (returns ``WinnerSelectionV1``)."""
    # CORRECTNESS-clustering bank = the CONTROLLED DETERMINISTIC small bank: the public samples + the
    # constraint-covering aug (which isolates each stated cap). We do NOT cluster on the passed
    # ``small_inputs`` (model-adversarial / random): correctness is a small-input property, and those
    # inputs add (a) large cases where the O(N^2) brutes TLE inconsistently and (b) noise that fragments
    # the correct cluster (the W142b discover bug — public+aug alone clusters cleanly). Efficiency is the
    # separate S2 check; small_inputs ≤ cap may still augment coverage but the aug bank is the guarantee.
    raw = [inp for inp, _ in samples] + constraint_covering_bank_v1(io_shape, samples)
    seen: set[str] = set()
    bank = [b for b in raw if b.strip() and not (b.strip() in seen or seen.add(b.strip()))]
    if not bank:
        return WinnerSelectionV1(None, None, True, "empty_bank", (), 0.0)

    tbrutes = [b for b in brute_codes if b.strip()
               and brute_is_trusted_v1(b, _StubProblem(list(samples)), timeout_s=timeout_s)]
    pub_idx = [i for i, c in enumerate(candidates)
               if c.strip() and _passes_public_v2(c, samples, timeout_s=timeout_s)]
    eff = {i: _fast_efficient_v1(candidates[i], statement, list(samples))[0] for i in pub_idx}
    if not tbrutes:
        return WinnerSelectionV1(None, None, True, "no_trusted_brute", (), 0.0)

    # usable correctness bank: inputs where a MAJORITY of trusted brutes run clean (tolerate the rare
    # large adversarial input that TLEs the O(N^2) brute — correctness is a small-input property).
    min_clean = max(1, (len(tbrutes) + 1) // 2)
    usable = _usable_bank(tbrutes, bank, timeout_s=timeout_s, min_clean=min_clean)
    if not usable:
        return WinnerSelectionV1(None, None, True, "no_usable_bank", (), 0.0)

    # cluster brutes (tag "B") + public-passing candidates (tag "C") by their output-vector on the
    # usable bank (crash/TLE recorded as a marker so a fragile witness clusters by the rest).
    clusters: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for j, b in enumerate(tbrutes):
        clusters[_vec_over(b, usable, timeout_s=timeout_s)].append(("B", j))
    for i in pub_idx:
        clusters[_vec_over(candidates[i], usable, timeout_s=timeout_s)].append(("C", i))

    brute_clusters = [(v, keys) for v, keys in clusters.items() if any(k[0] == "B" for k in keys)]
    verdicts = _build_verdicts(candidates, pub_idx, eff, clusters, consensus_probe, timeout_s)
    if not brute_clusters:
        return WinnerSelectionV1(None, None, True, "no_trusted_brute_cluster", verdicts, 0.0)
    # reference cluster = the largest cluster anchored by >=1 brute
    ref_v, ref_keys = max(brute_clusters, key=lambda x: len(x[1]))
    winner_idxs = [k[1] for k in ref_keys if k[0] == "C" and eff.get(k[1])]
    n_brute_in_ref = sum(1 for k in ref_keys if k[0] == "B")
    if not winner_idxs:
        return WinnerSelectionV1(None, None, True, "no_efficient_winner_in_ref_cluster", verdicts, 0.0)
    pick = winner_idxs[0]
    frac = (n_brute_in_ref + len(winner_idxs)) / max(1, len(clusters))
    return WinnerSelectionV1(pick, candidates[pick], False, "committed_v2", verdicts, round(frac, 3))


def _build_verdicts(candidates, pub_idx, eff, clusters, consensus_probe, timeout_s
                    ) -> tuple[CandidateVerdictV1, ...]:
    # which cluster each candidate landed in (for the verdict's is_winner flag)
    cand_cluster: dict[int, tuple] = {}
    for v, keys in clusters.items():
        for tag, idx in keys:
            if tag == "C":
                cand_cluster[idx] = v
    brute_clusters = {v for v, keys in clusters.items() if any(k[0] == "B" for k in keys)}
    ref_v = None
    if brute_clusters:
        ref_v = max(((v, keys) for v, keys in clusters.items() if v in brute_clusters),
                    key=lambda x: len(x[1]))[0]
    out = []
    for i, c in enumerate(candidates):
        is_pub = i in pub_idx
        is_eff = bool(eff.get(i))
        in_ref = (cand_cluster.get(i) == ref_v) and ref_v is not None
        out.append(CandidateVerdictV1(
            idx=i, parses_runs=is_pub, passes_public=is_pub,
            agrees_with_brute=in_ref, n_brute_cases=0, efficient=is_eff,
            witness_kind=("ref_cluster" if in_ref else "off_cluster"),
            output_sig="", is_winner=bool(is_pub and is_eff and in_ref)))
    return tuple(out)


__all__ = [
    "NO_ORACLE_VERIFIER_V2_SCHEMA_VERSION", "constraint_covering_bank_v1",
    "select_winner_v2",
]
