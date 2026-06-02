"""W131 — code-model supply census + stronger-generator model bench (offline, $0).

Falsifiability-first: covers the census schema/classification, the fence-format normalization fix
(the qwen-coder ```\\npython\\n quirk that crashes the audited extractor), the code smoke gate
(incl. the constant-output cheat guard), the B-slate arm wiring + same-budget guard, the earn-gate
reuse, and a real end-to-end plain-arm run on a synthetic problem (correct gen ⇒ commit+pass; wrong
gen ⇒ no commit).  No network: all model calls are fakes.
"""
import dataclasses

import coordpy.code_model_supply_census_v1 as C
import coordpy.generator_model_bench_v1 as B
import coordpy.stronger_generator_slate_v1 as S
from coordpy.role_diverse_algorithm_search_v1 import IcpcPilotProblemV1


# --------------------------------------------------------------------------- census schema

def test_census_enums_locked():
    assert C.ACCESS_PATHS == ("LOCAL_HF", "LOCAL_OLLAMA", "HOSTED_NIM")
    assert set(C.CUTOFF_DISCLOSURES) == {"PRIMARY_KNOWN", "UNKNOWN"}
    assert "FRONTIER_ELIGIBLE" in C.USAGE_CLASSES and "DEV_ONLY" in C.USAGE_CLASSES
    assert C.CODE_MODEL_SUPPLY_CENSUS_V1_SCHEMA == "coordpy.code_model_supply_census_v1.v1"


def test_classify_code_prior():
    assert C.classify_code_prior_v1("qwen2.5-coder:32b") == "CODE_TUNED"
    assert C.classify_code_prior_v1("qwen/qwen3-coder-480b-a35b-instruct") == "CODE_TUNED"
    assert C.classify_code_prior_v1("mistralai/codestral-22b-instruct-v0.1") == "CODE_TUNED"
    assert C.classify_code_prior_v1("meta/codellama-70b") == "CODE_TUNED"
    assert C.classify_code_prior_v1("deepseek-r1:7b") == "REASONING"
    assert C.classify_code_prior_v1("nvidia/nv-embedcode-7b-v1") == "EMBED_OR_OTHER"
    assert C.classify_code_prior_v1("nvidia/nemotron-4-340b-reward") == "EMBED_OR_OTHER"
    assert C.classify_code_prior_v1("meta/llama-4-maverick-17b-128e-instruct") == "GENERAL_LM"


# --------------------------------------------------------------------------- fence normalization

def test_normalize_fence_fixes_lone_language_tag():
    # the qwen-coder quirk: ```\npython\n<code> -> ```python\n<code>
    bad = "```\npython\nn = int(input())\nprint(n * n)\n```"
    fixed = C.normalize_fence_v1(bad)
    assert "```python\n" in fixed
    code = C._extract_code_block_v1(fixed)
    assert code.splitlines()[0].strip() != "python"
    assert C._smoke_run_v1(code, "7\n", timeout_s=5.0) == "49"


def test_normalize_fence_idempotent_on_clean():
    clean = "```python\nprint(1)\n```"
    assert C.normalize_fence_v1(clean) == clean
    # a bare ```python with no lone-tag line is unchanged
    assert C.normalize_fence_v1("no fence here") == "no fence here"


# --------------------------------------------------------------------------- smoke gate

def _fake_gen(text):
    def g(prompt, max_tokens=300, temperature=0.2):
        return text, 1
    return g


def test_smoke_gate_pass_on_correct_code():
    ok, detail = C.code_smoke_gate_v1(_fake_gen("```python\nn=int(input())\nprint(n*n)\n```"))
    assert ok, detail


def test_smoke_gate_fail_on_wrong_code():
    ok, _ = C.code_smoke_gate_v1(_fake_gen("```python\nn=int(input())\nprint(n+n)\n```"))
    assert not ok


def test_smoke_gate_fail_on_no_code():
    ok, detail = C.code_smoke_gate_v1(_fake_gen("I cannot help with that."))
    # bare text is treated as code and will fail to produce 49
    assert not ok


def test_smoke_gate_rejects_constant_output_cheat():
    # printing a constant 49 passes case 1 but must FAIL case 2 (144) — the 2-case guard.
    ok, _ = C.code_smoke_gate_v1(_fake_gen("```python\nprint(49)\n```"))
    assert not ok


# --------------------------------------------------------------------------- surface probes

def test_probe_local_hf_returns_tuple():
    loadable, reason = C.probe_local_hf_v1()
    assert isinstance(loadable, bool) and isinstance(reason, str)


def test_supply_record_to_dict_and_census_cid_deterministic():
    r = C.CodeModelSupplyRecordV1(
        model_id="m", access_path="HOSTED_NIM", code_prior="CODE_TUNED", param_hint="480b-a35b",
        context_hint=262144, load_success=True, blocked_reason="", smoke_ran=False,
        smoke_pass=False, smoke_detail="defer", cutoff_disclosure="UNKNOWN", cutoff_boundary="",
        stronger_than_maverick=True, usage_class="DEV_ONLY", realistic_for_dev_bench=True)
    d = r.to_dict()
    assert d["model_id"] == "m" and d["usage_class"] == "DEV_ONLY"
    c1 = C.CodeModelSupplyCensusV1(schema="x", local_hf_loadable=False, local_hf_reason="",
                                   ollama_reachable=True, hosted_nim_reachable=True,
                                   hosted_blocked_reason="", records=(r,))
    c2 = C.CodeModelSupplyCensusV1(schema="x", local_hf_loadable=False, local_hf_reason="",
                                   ollama_reachable=True, hosted_nim_reachable=True,
                                   hosted_blocked_reason="", records=(r,))
    assert c1.cid() == c2.cid()


def test_param_billions():
    assert C._param_billions("32.8B") == 32.8
    assert C._param_billions("480b-a35b") == 480.0
    assert C._param_billions("") == 0.0


# --------------------------------------------------------------------------- bench wiring

def test_b_arm_map_and_underlying():
    assert B.B_ARMS == ("B1_PLAIN", "B2_RDIV", "B3_GG2", "B4_GGLEAD")
    assert B.B_ARM_UNDERLYING["B1_PLAIN"] == "plain"
    assert B.B_ARM_UNDERLYING["B3_GG2"] == "GG2"
    assert B.B_ARM_UNDERLYING["B4_GGLEAD"] == "GGLEAD"


def test_earn_gate_is_w130_gate():
    # W131 applies the IDENTICAL +2-spanning earn bar as W130.
    assert B.apply_dev_bench_earn_gate_v1 is S.apply_gg_dev_bench_earn_gate_v1


class _FakeOutcome:
    def __init__(self, n):
        self.n_calls = n


def test_same_budget_guard():
    assert B.assert_same_budget_v1(_FakeOutcome(5), 5) is True
    assert B.assert_same_budget_v1(_FakeOutcome(3), 5) is True
    import pytest
    with pytest.raises(AssertionError):
        B.assert_same_budget_v1(_FakeOutcome(6), 5)


def test_build_dev_gen_backends():
    g_local = B.build_dev_gen_v1("qwen2.5-coder:7b", backend="local")
    assert callable(g_local)
    # hosted requires NVIDIA_API_KEY; only assert it builds when the key is present
    import os
    if os.environ.get("NVIDIA_API_KEY"):
        assert callable(B.build_dev_gen_v1("meta/llama-4-maverick-17b-128e-instruct",
                                           backend="hosted"))


# --------------------------------------------------------------------------- end-to-end plain arm

def _square_problem():
    return IcpcPilotProblemV1(
        problem_id="t-sq", short_name="square", source_repo="test", contest_date="2024-01-01",
        statement="Read an integer N from stdin and print N*N.", kind="KIND_PASSFAIL",
        float_tol=0.0,
        samples=(("7\n", "49\n"), ("3\n", "9\n")),
        secret_cases=(("10\n", "100\n"), ("0\n", "0\n"), ("100\n", "10000\n")))


def test_plain_arm_commits_correct_solution():
    prob = _square_problem()
    gen = _fake_gen("```python\nn=int(input())\nprint(n*n)\n```")
    out = B.run_plain_arm_v1(gen, prob, K=5)
    assert out.n_calls == 5            # exactly K model calls, selector is NIM-free
    assert out.pool_pass is True       # the pool contains a secret-passing program
    assert out.committed_pass is True  # the fixed selector commits it
    assert out.leakage_clean is True


def test_plain_arm_does_not_commit_wrong_solution():
    prob = _square_problem()
    gen = _fake_gen("```python\nn=int(input())\nprint(n+n)\n```")  # N+N, fails public sample
    out = B.run_plain_arm_v1(gen, prob, K=5)
    assert out.n_calls == 5
    assert out.committed_pass is False  # no public survivor ⇒ abstain / no commit
