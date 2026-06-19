"""W131 Lane β — stronger-GENERATOR hard-cluster dev bench with the W129 selector held FIXED.

The variable under test is **base generator MODEL capability**, not a new selector and not more
prompt engineering.  This module wires a swappable code model (local Ollama OR hosted NIM, via the
W131 census ``build_openai_compat_gen_v1`` seam) through a same-budget K=5 generator slate, with the
W129 NIM-free SOLEAD selector (``public_signal_selection_oracle_v1.select_so_v1``, gen=None) held
FIXED downstream — so any committed-win lift is attributable to GENERATION, not selection.

The B-slate (mapped onto the audited W128/W130 arms so the ONLY new variable is the model):

  * **B0** — the W128/W129/W130 Maverick baseline (REUSED from results/w130; old pool ceiling 3/11,
    plain baseline 2/11; GG2 cracked ``doubleup``).  Not re-run here ($0).
  * **B1_PLAIN** — plain same-budget generation: K i.i.d. full-solution implements (NO analyze
    role), then the fixed selector.  The new arm in this module.
  * **B2_RDIV** — role-diverse multi-sketch generation (W128 shape) via the audited ``run_gg1_v1``
    (complexity-gated role handoff) + fixed selector.
  * **B3_GG2** — counterexample-to-rewrite (the W130 winning lever) via ``run_gg2_v1`` + fixed
    selector.
  * **B4_GGLEAD** — the GG1→GG2 composite handoff/planner via ``run_gglead_v1`` + fixed selector.

Same-budget accounting (LOCKED): every arm spends EXACTLY <= K model calls per target
(``assert_same_budget_v1``); the selector, grading, and leakage checks are NIM-free.  The earn gate
is the W130 ``apply_gg_dev_bench_earn_gate_v1`` (>= 2 NEW pool solves spanning >= 2
families/atlas-modes, realness-REAL + leakage-clean).  A stronger model EARNS only by creating
genuinely new committed headroom — never by re-cashing old pool wins (the stored-regression trio
guards that) and never via same-problem leakage.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import coordpy.family_scaffold_generation_v1 as _G
import coordpy.role_diverse_algorithm_search_v1 as _R
import coordpy.stronger_generator_slate_v1 as _S
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1
from coordpy.code_model_supply_census_v1 import build_openai_compat_gen_v1

GENERATOR_MODEL_BENCH_V1_SCHEMA = "coordpy.generator_model_bench_v1.v1"

# B-slate labels and their mapping to the audited underlying arms.
B_ARMS = ("B1_PLAIN", "B2_RDIV", "B3_GG2", "B4_GGLEAD")
B_ARM_UNDERLYING = {
    "B1_PLAIN": "plain",
    "B2_RDIV": "GG1",       # role-diverse multi-sketch + complexity gate (W128 shape)
    "B3_GG2": "GG2",        # counterexample-to-rewrite (W130 winner)
    "B4_GGLEAD": "GGLEAD",  # GG1->GG2 composite handoff/planner
}
DEFAULT_K = 5

# Backends.
NIM_BASE_URL = "https://integrate.api.nvidia.com"
OLLAMA_BASE_URL = "http://localhost:11434"


def _parses_v1(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        compile(code, "<cand>", "exec")
        return True
    except Exception:  # noqa: BLE001
        return False


def build_dev_gen_v1(model: str, *, backend: str,
                     sidecar_writer: Optional[Callable[[dict], None]] = None,
                     read_timeout_s: float = 120.0, max_retries: int = 8,
                     ollama_base_url: str = OLLAMA_BASE_URL,
                     nim_base_url: str = NIM_BASE_URL) -> Callable[[str, int, float], tuple]:
    """Build a ``gen(prompt, max_tokens, temperature) -> (text, wall_ms)`` for the chosen backend.

    ``backend`` ∈ {"local","ollama"} (→ Ollama, $0) | {"hosted","nim"} (→ NVIDIA NIM, $).
    """
    b = backend.lower()
    if b in ("local", "ollama"):
        return build_openai_compat_gen_v1(
            model, base_url=ollama_base_url, api_key="ollama",
            read_timeout_s=read_timeout_s, max_retries=max_retries,
            sidecar_writer=sidecar_writer, backend_tag="ollama")
    if b in ("hosted", "nim"):
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            raise RuntimeError("hosted backend requires NVIDIA_API_KEY")
        return build_openai_compat_gen_v1(
            model, base_url=nim_base_url, api_key=key,
            read_timeout_s=read_timeout_s, max_retries=max_retries,
            sidecar_writer=sidecar_writer, backend_tag="nim")
    raise ValueError(f"unknown backend: {backend}")


def run_plain_arm_v1(gen: Callable[[str, int, float], Any], problem, *, K: int = DEFAULT_K,
                     impl_temp: float = 0.2, max_tokens: int = 1536, timeout_s: float = 5.0,
                     secret_timeout_s: float = 8.0,
                     accepted_codes: Sequence[str] = ()) -> Any:
    """B1 — plain same-budget generation: K i.i.d. full-solution implements + the FIXED W129
    selector.  No ANALYZE role (so the selector sees public samples + auto-derived cases only).

    Reuses the audited ``stronger_generator_slate_v1._finalize_arm`` (which runs
    ``select_so_v1(..., variant="SOLEAD", gen=None)``, grades secret, and leakage-checks) so the
    selection/grading/leakage are byte-identical to every other arm — the only difference is the
    plain generation shape.
    """
    plain_prompt = _G.build_plain_prompt_v1(problem)
    cands = []
    calls = 0
    for i in range(max(1, int(K))):
        text, _w = gen(plain_prompt, int(max_tokens), float(impl_temp))
        calls += 1
        code = extract_candidate_code_v1(response_text=text)
        cands.append(_S.GenCandidateV1(f"P{i}", code, _parses_v1(code), "plain"))
    arts = _R.RoleArtifactsV1(spec="", invariants=(), complexity="", sketches=(),
                              counterexamples=(), raw="")
    diagnostics = {"arm_kind": "plain", "n_impl": calls}
    realness = {"real": True, "kind": "plain_no_analyze", "passes": True}
    return _S._finalize_arm(problem, "B1_PLAIN", arts, cands, calls, diagnostics, realness,
                            accepted_codes=tuple(accepted_codes),
                            secret_timeout_s=float(secret_timeout_s),
                            public_timeout_s=float(timeout_s))


def run_b_arm_v1(gen: Callable[[str, int, float], Any], problem, b_arm: str, *,
                 family: str = "", library: Any = None, K: int = DEFAULT_K,
                 n_sketches: int = 4, analyze_temp: float = 0.5, impl_temp: float = 0.2,
                 max_tokens: int = 1536, timeout_s: float = 5.0, secret_timeout_s: float = 8.0,
                 accepted_codes: Sequence[str] = ()) -> Any:
    """Dispatch one B-slate arm on ``problem`` using the injected ``gen``.  Returns a
    ``GgArmOutcomeV1`` (the audited outcome dataclass; every arm ends in the fixed W129 selector).
    """
    underlying = B_ARM_UNDERLYING.get(b_arm, b_arm)
    common = dict(K=K, n_sketches=n_sketches, analyze_temp=analyze_temp, impl_temp=impl_temp,
                  max_tokens=max_tokens, accepted_codes=tuple(accepted_codes))
    if underlying == "plain":
        return run_plain_arm_v1(gen, problem, K=K, impl_temp=impl_temp, max_tokens=max_tokens,
                                timeout_s=timeout_s, secret_timeout_s=secret_timeout_s,
                                accepted_codes=accepted_codes)
    if underlying == "GG1":
        return _S.run_gg1_v1(gen, problem, timeout_s=timeout_s, **common)
    if underlying == "GG2":
        return _S.run_gg2_v1(gen, problem, timeout_s=timeout_s, **common)
    if underlying == "GG4":
        return _S.run_gg4_v1(gen, problem, timeout_s=timeout_s, **common)
    if underlying == "GGLEAD":
        return _S.run_gglead_v1(gen, problem, timeout_s=timeout_s, **common)
    if underlying == "GG3":
        return _S.run_gg3_v1(gen, problem, family=family, library=library, **common)
    raise ValueError(f"unknown B-arm: {b_arm}")


def assert_same_budget_v1(outcome: Any, K: int = DEFAULT_K) -> bool:
    """Same-budget guard: an arm must spend <= K model calls per target.  Returns True if OK."""
    n = int(getattr(outcome, "n_calls", 0))
    if n > int(K):
        raise AssertionError(f"same-budget violation: arm spent {n} > K={K} calls")
    return True


# Re-export the W130 earn gate so the W131 driver applies the IDENTICAL +2-spanning bar.
apply_dev_bench_earn_gate_v1 = _S.apply_gg_dev_bench_earn_gate_v1


__all__ = [
    "GENERATOR_MODEL_BENCH_V1_SCHEMA", "B_ARMS", "B_ARM_UNDERLYING", "DEFAULT_K",
    "NIM_BASE_URL", "OLLAMA_BASE_URL", "build_dev_gen_v1", "run_plain_arm_v1",
    "run_b_arm_v1", "assert_same_budget_v1", "apply_dev_bench_earn_gate_v1",
]
