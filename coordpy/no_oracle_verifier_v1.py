"""W141 / COO-9 — no-oracle correct+efficient verifier (Lane β core).

Picks the technique-bearing WINNER from a team's K candidate programs using ONLY:
  * the public problem statement + public sample I/O,
  * SELF-GENERATED small inputs (drawn from the public constraints),
  * a SELF-WRITTEN brute (the obvious algorithm, a team generation) — validated on the public
    samples first, then used as a slow-but-correct reference on the small self-gen inputs,
  * the candidate's OWN measured runtime growth (W134 ``build_deployable_witness_v1``, oracle-free).

It NEVER reads the hidden bank, the ``ref_source``, the ``naive_source``, or the ``brute_source``
answer-key — those exist only for the bench's $0 HONESTY audit (validate-against-oracle then deploy
oracle-free, the W134 discipline).  A candidate WINS iff it (1) passes the public samples, (2) AGREES
with the self-brute on every small self-gen input (correctness), and (3) is NOT flagged too-slow by
the deployable complexity witness (efficiency).  Among winners, commit by output-signature consensus;
ABSTAIN when none qualify or the top class is not a strict majority (the W128/W129 abstain discipline).

This is the module that breaks the W128 selection cap: it adds the two signals public-sample
selection lacked — a self-brute correctness reference + the complexity/timeout efficiency signal
(the W129-proven cap-breaker).  Pure/deterministic except the audited execution subprocess; NO model
inference here.  Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import random as _random

from .resistant_by_construction_battlefield_v1 import MintedProblemV1, _exec_capture_v1
from .deployable_complexity_witness_v1 import (
    build_deployable_witness_v1, DW_COMPLEXITY,
    derive_budget_fact_v1, parse_input_shape_v1, synth_input_v1)

_FAST_EFF_CAP: int = 30000          # large-probe size cap for the single-point efficiency check
_FAST_EFF_TIMEOUT_S: float = 2.0    # TLE on a worst-case large input => inadmissible growth

NO_ORACLE_VERIFIER_V1_SCHEMA_VERSION: str = "coordpy.no_oracle_verifier_v1.v1"

_NORMAL = "OK"
_CRASH = "CRASH"
_TLE = "TLE"


def _run_sig(code: str, stdin_text: str, *, timeout_s: float) -> str:
    """A behaviour signature for differential comparison: stdout on success, else a crash/TLE tag."""
    r = _exec_capture_v1(code, stdin_text, timeout_s=float(timeout_s))
    if r.timed_out:
        return _TLE
    if r.returncode != 0:
        return f"{_CRASH}:{r.returncode}"
    return r.stdout.strip()


def _fast_efficient_v1(code: str, statement: str, samples: Sequence[tuple[str, str]], *,
                       timeout_s: float = _FAST_EFF_TIMEOUT_S, cap: int = _FAST_EFF_CAP
                       ) -> tuple[bool, str]:
    """Oracle-free efficiency (S2): run the candidate on ONE worst-case-structure LARGE
    self-SYNTHESIZED input (descending, then all-equal) with a wall-clock timeout; a TLE ⇒
    inadmissible growth (the slow-brute cap-breaker, W129/W134).  The dual structure also rejects
    distribution-shortcut "fast" candidates (Agent A's recommendation).  Self-synthesizes the input
    from the public statement constraint (no oracle); falls back to the W134 growth witness when the
    input shape is unparseable."""
    try:
        shape = parse_input_shape_v1(statement, list(samples))
        budget = derive_budget_fact_v1(statement)
    except Exception:  # noqa: BLE001
        shape, budget = None, None
    if shape is None or not getattr(shape, "parseable", False) or not (budget and budget.n_max):
        wit = build_deployable_witness_v1(code, statement=statement, samples=list(samples),
                                          timeout_s=timeout_s)
        return (not (wit.kind == DW_COMPLEXITY and wit.fired and wit.confidence_ok)), wit.kind
    size = min(int(budget.n_max), int(cap))
    for kind in ("descending", "constant"):
        inp = synth_input_v1(shape, size=size, kind=kind, rng=_random.Random(141))
        r = _exec_capture_v1(code, inp, timeout_s=float(timeout_s))
        if r.timed_out:
            return False, "TLE_LARGE"
    return True, "FAST_OK"


def brute_is_trusted_v1(brute_code: str, problem: MintedProblemV1, *, timeout_s: float = 4.0) -> bool:
    """A self-written brute is trusted ONLY if it reproduces every PUBLIC sample (no oracle needed —
    public samples are given).  This bootstraps the brute into a correctness reference."""
    if not brute_code.strip():
        return False
    cases = list(problem.samples)
    if not cases:
        return False
    for inp, exp in cases:
        if _run_sig(brute_code, inp, timeout_s=timeout_s) != exp.strip():
            return False
    return True


@dataclasses.dataclass(frozen=True)
class CandidateVerdictV1:
    idx: int
    parses_runs: bool
    passes_public: bool
    agrees_with_brute: bool      # on the small self-gen inputs (correctness, no oracle)
    n_brute_cases: int
    efficient: bool              # deployable witness did NOT flag too-slow (oracle-free)
    witness_kind: str
    output_sig: str              # signature on a fixed probe input (for consensus)
    is_winner: bool

    def to_dict(self) -> dict[str, Any]:
        return {"idx": self.idx, "passes_public": self.passes_public,
                "agrees_with_brute": self.agrees_with_brute, "n_brute_cases": self.n_brute_cases,
                "efficient": self.efficient, "witness_kind": self.witness_kind,
                "is_winner": self.is_winner}


@dataclasses.dataclass(frozen=True)
class WinnerSelectionV1:
    winner_idx: Optional[int]
    winner_code: Optional[str]
    abstained: bool
    reason: str
    verdicts: tuple[CandidateVerdictV1, ...]
    consensus_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {"winner_idx": self.winner_idx, "abstained": self.abstained, "reason": self.reason,
                "consensus_fraction": round(self.consensus_fraction, 3),
                "n_candidates": len(self.verdicts),
                "n_winners": sum(1 for v in self.verdicts if v.is_winner),
                "verdicts": [v.to_dict() for v in self.verdicts]}


def verify_candidate_v1(code: str, idx: int, *, statement: str,
                        samples: Sequence[tuple[str, str]], small_inputs: Sequence[str],
                        brute_code: str, consensus_probe: str,
                        timeout_s: float = 4.0) -> CandidateVerdictV1:
    """No-oracle per-candidate verdict.  ``brute_code`` must already be trusted (passes public)."""
    # public correctness (given, no oracle)
    passes_public = True
    for inp, exp in samples:
        if _run_sig(code, inp, timeout_s=timeout_s) != exp.strip():
            passes_public = False
            break
    parses_runs = passes_public or any(
        not _run_sig(code, inp, timeout_s=timeout_s).startswith((_CRASH, _TLE)) for inp, _ in samples)

    # differential correctness vs the self-brute on small self-gen inputs.  The brute is the
    # slow-but-correct reference; SKIP any input where the brute itself crashes/TLEs (an invalid
    # reference there) — only count cases where the brute produced a clean answer.  Per the W141
    # de-risk (Agent A, 115 real samples): this S1 signal is load-bearing (it caught 8/8 real
    # fast-but-WRONG candidates that the efficiency signal alone accepted), PROVIDED the bank is
    # constraint-adversarial, not public-style.
    agree = True
    n_cases = 0
    for si in small_inputs:
        bs = _run_sig(brute_code, si, timeout_s=timeout_s)
        if bs.startswith((_CRASH, _TLE)) or not bs:
            continue  # invalid reference on this case — skip
        cs = _run_sig(code, si, timeout_s=timeout_s)
        n_cases += 1
        if bs != cs:
            agree = False
            break
    agrees = bool(agree and n_cases > 0)

    # efficiency: the candidate's OWN behavior on a worst-case large self-synthesized input (S2)
    efficient, wit_kind = _fast_efficient_v1(code, statement, list(samples))

    sig = _run_sig(code, consensus_probe, timeout_s=timeout_s) if consensus_probe else ""
    is_winner = bool(passes_public and agrees and efficient)
    return CandidateVerdictV1(idx=idx, parses_runs=parses_runs, passes_public=passes_public,
                              agrees_with_brute=agrees, n_brute_cases=n_cases, efficient=efficient,
                              witness_kind=wit_kind, output_sig=sig, is_winner=is_winner)


def select_winner_v1(candidates: Sequence[str], *, statement: str,
                     samples: Sequence[tuple[str, str]], small_inputs: Sequence[str],
                     brute_code: str, consensus_probe: str = "",
                     timeout_s: float = 4.0) -> WinnerSelectionV1:
    """Pick the correct+efficient winner with NO oracle.  Commit by output-signature consensus among
    winners; ABSTAIN if no winner or the consensus class is not a strict majority of winners."""
    if not brute_is_trusted_v1(brute_code, _StubProblem(samples), timeout_s=timeout_s):
        return WinnerSelectionV1(None, None, True, "untrusted_self_brute", (), 0.0)
    verdicts = tuple(
        verify_candidate_v1(c, i, statement=statement, samples=samples, small_inputs=small_inputs,
                            brute_code=brute_code, consensus_probe=consensus_probe,
                            timeout_s=timeout_s)
        for i, c in enumerate(candidates))
    winners = [v for v in verdicts if v.is_winner]
    if not winners:
        return WinnerSelectionV1(None, None, True, "no_correct_efficient_candidate", verdicts, 0.0)
    # consensus by output signature on the probe input
    from collections import Counter
    sig_counts = Counter(v.output_sig for v in winners if v.output_sig)
    if sig_counts:
        top_sig, top_n = sig_counts.most_common(1)[0]
        frac = top_n / len(winners)
        pick = next(v for v in winners if v.output_sig == top_sig)
    else:
        frac = 1.0 / len(winners)
        pick = winners[0]
    if len(winners) > 1 and sig_counts and frac <= 0.5:
        return WinnerSelectionV1(None, None, True, "winner_consensus_tie", verdicts, frac)
    return WinnerSelectionV1(pick.idx, candidates[pick.idx], False, "committed", verdicts, frac)


@dataclasses.dataclass(frozen=True)
class _StubProblem:
    """Minimal shim so brute_is_trusted_v1 can validate a self-brute against public samples without a
    full MintedProblemV1 (the verifier never needs the hidden bank)."""
    _samples: Sequence[tuple[str, str]]

    @property
    def samples(self) -> Sequence[tuple[str, str]]:
        return self._samples
