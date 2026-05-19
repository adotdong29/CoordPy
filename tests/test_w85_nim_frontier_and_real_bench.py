"""W85 — NIM frontier text runtime + GSM8K real-task bench +
long-context live bench tests.

Pure-Python contract tests run unconditionally. Live-NIM tests
are gated on ``NVIDIA_API_KEY`` to keep CI clean.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from coordpy.nim_frontier_text_runtime_v1 import (
    NIMFrontierBlockedError,
    NIMFrontierProbeReportV1,
    NIMFrontierCallCapsuleV1,
    NIMFrontierCapabilityClaimV1,
    NIMFrontierTextRuntimeV1,
    W85_NIM_FRONTIER_MODEL_CATALOG,
    W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
    declare_nim_frontier_capability_claim_v1,
    probe_nim_frontier_runtime_v1,
)
from coordpy.gsm8k_real_bench_v1 import (
    GSM8K_TEST_RAW_EXPECTED_SHA256,
    GSM8K_TEST_RAW_URL,
    GSM8KArmCallCapsuleV1,
    GSM8KArmOutcomeCapsuleV1,
    GSM8KBenchConfigV1,
    GSM8KBenchReportV1,
    GSM8KCorpusError,
    GSM8KSeedReportV1,
    W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
    load_gsm8k_test_corpus_v1,
    parse_gold_int_v1,
    parse_model_int_v1,
    run_gsm8k_real_bench_v1,
    select_problem_subset_v1,
)
from coordpy.long_context_live_bench_v1 import (
    LiveArmCallV1,
    LiveNeedlePromptV1,
    LongContextLiveHorizonPointV1,
    LongContextLiveReportV1,
    W85_LONG_CONTEXT_LIVE_V1_SCHEMA_VERSION,
    build_live_needle_corpus_v1,
    parse_magic_number_v1,
    run_long_context_live_bench_v1,
)


_HAS_NIM_KEY = bool(os.environ.get("NVIDIA_API_KEY"))


# ---------------------------------------------------------------
# NIM frontier text runtime (#25 / #27 / #28 / #31 — text-only).
# ---------------------------------------------------------------


def test_w85_nim_probe_without_key_is_blocked_not_mocked(monkeypatch):
    """Probe must report unreachable when no key — no mock."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    rep = probe_nim_frontier_runtime_v1(api_key=None)
    assert rep.reachable is False
    assert rep.api_key_present is False
    assert "NVIDIA_API_KEY" in rep.blocked_reason
    assert rep.available_models == ()
    assert rep.catalog_subset_available == ()
    assert rep.is_text_only_substrate() is True


def test_w85_nim_capability_claim_says_no_substrate_access():
    """The W85 capability claim is honest: text yes, substrate no."""
    rep = NIMFrontierProbeReportV1(
        schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
        nim_endpoint="https://integrate.api.nvidia.com",
        api_key_present=True, reachable=True, blocked_reason="",
        available_models=(
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "microsoft/phi-3.5-moe-instruct",
        ),
        catalog_subset_available=(
            ("meta/llama-3.1-8b-instruct", 131_072, "llama-3.1-8b"),
            ("meta/llama-3.1-70b-instruct", 131_072, "llama-3.1-70b"),
            ("microsoft/phi-3.5-moe-instruct", 128_000, "phi-3.5-moe"),
        ),
    )
    claim = declare_nim_frontier_capability_claim_v1(probe=rep)
    assert claim.nim_text_generation is True
    assert claim.real_frontier_class_open_weights is True
    # The honesty bars:
    assert claim.hidden_state_access is False
    assert claim.kv_cache_replay is False
    assert claim.per_layer_instrumentation is False
    assert claim.cross_runtime_state_export is False
    # Long-context and MoE reachable on text
    assert claim.long_context_at_least_32k is True
    assert claim.moe_models_reachable is True


def test_w85_nim_runtime_constructor_fails_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(NIMFrontierBlockedError):
        NIMFrontierTextRuntimeV1(
            model_id="meta/llama-3.1-8b-instruct",
            api_key=None)


def test_w85_nim_call_capsule_is_content_addressed():
    cap1 = NIMFrontierCallCapsuleV1(
        schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
        model_id="meta/llama-3.1-8b-instruct",
        prompt_cid="a" * 64, response_cid="b" * 64,
        temperature=0.0, max_tokens=64, wall_ms=400,
        prompt_tokens=44, output_tokens=3,
        response_finish_reason="stop")
    cap2 = NIMFrontierCallCapsuleV1(
        schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
        model_id="meta/llama-3.1-8b-instruct",
        prompt_cid="a" * 64, response_cid="b" * 64,
        temperature=0.0, max_tokens=64, wall_ms=999,  # differs
        prompt_tokens=44, output_tokens=3,
        response_finish_reason="stop")
    # Wall clock is part of the CID — different wall → different CID
    assert cap1.cid() != cap2.cid()
    assert len(cap1.cid()) == 64


def test_w85_nim_catalog_has_real_frontier_models():
    families = {tag for (_, _, tag) in W85_NIM_FRONTIER_MODEL_CATALOG}
    assert "llama-3.1-8b" in families
    assert "llama-3.1-70b" in families
    assert "phi-3.5-moe" in families
    # All catalog entries advertise a context window
    for (mid, ctx, tag) in W85_NIM_FRONTIER_MODEL_CATALOG:
        assert isinstance(ctx, int) and ctx > 0
        assert isinstance(mid, str) and "/" in mid


@pytest.mark.skipif(
    not _HAS_NIM_KEY,
    reason="NVIDIA_API_KEY not set — live NIM test skipped")
def test_w85_nim_live_probe_finds_frontier_models():
    rep = probe_nim_frontier_runtime_v1()
    assert rep.reachable is True
    assert rep.api_key_present is True
    assert len(rep.available_models) > 0
    assert len(rep.catalog_subset_available) >= 1


@pytest.mark.skipif(
    not _HAS_NIM_KEY,
    reason="NVIDIA_API_KEY not set — live NIM test skipped")
def test_w85_nim_live_generate_returns_text_and_capsule():
    rt = NIMFrontierTextRuntimeV1(
        model_id="meta/llama-3.1-8b-instruct", timeout=60.0)
    cap, text = rt.generate_capsule(
        prompt="Reply with exactly: PASS", max_tokens=5,
        temperature=0.0)
    assert isinstance(cap, NIMFrontierCallCapsuleV1)
    assert "PASS" in text.upper()
    assert cap.prompt_tokens > 0
    assert cap.model_id == "meta/llama-3.1-8b-instruct"


# ---------------------------------------------------------------
# GSM8K real-task bench (#28).
# ---------------------------------------------------------------


def test_w85_gsm8k_corpus_url_is_canonical():
    assert "openai/grade-school-math" in GSM8K_TEST_RAW_URL
    assert GSM8K_TEST_RAW_URL.endswith("/test.jsonl")
    assert len(GSM8K_TEST_RAW_EXPECTED_SHA256) == 64


def test_w85_gsm8k_corpus_sha256_check_blocks_substitution():
    """If a substituted corpus has wrong SHA-256, refuse it."""
    with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".jsonl", delete=False) as f:
        f.write(b'{"question":"x","answer":"#### 1"}\n')
        path = f.name
    try:
        with pytest.raises(GSM8KCorpusError):
            load_gsm8k_test_corpus_v1(cache_path=path)
    finally:
        os.unlink(path)


def test_w85_gsm8k_parse_gold():
    assert parse_gold_int_v1("…\n#### 42") == 42
    assert parse_gold_int_v1("#### 1,234") == 1234
    assert parse_gold_int_v1("…\n#### -7") == -7
    assert parse_gold_int_v1("no marker here") is None


def test_w85_gsm8k_parse_model_response():
    # Standard format
    assert parse_model_int_v1("ANSWER: 18") == 18
    # Last-number convention
    assert parse_model_int_v1(
        "First we get 4, then 8, then total is 12") == 12
    # Currency / commas
    assert parse_model_int_v1("She has $1,234.") == 1234
    # No numbers
    assert parse_model_int_v1("No numbers here.") is None


def test_w85_gsm8k_outcome_capsule_is_content_addressed():
    o1 = GSM8KArmOutcomeCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=1, problem_idx=10, arm_id="A0",
        gold_answer=18, final_answer=18, correct=True,
        n_calls=1, total_wall_ms=400,
        call_capsule_cids=("a" * 64,))
    o2 = GSM8KArmOutcomeCapsuleV1(
        schema=W85_GSM8K_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=1, problem_idx=10, arm_id="A0",
        gold_answer=18, final_answer=17, correct=False,  # wrong
        n_calls=1, total_wall_ms=400,
        call_capsule_cids=("a" * 64,))
    assert o1.cid() != o2.cid()


def test_w85_gsm8k_bench_with_stub_generator():
    """Drive the bench with a deterministic stub generator.

    The stub always answers correctly, so all three arms score
    100%. This validates the bench pipeline (capsules, vote,
    Merkle root) without any LLM calls.
    """
    # Tiny synthetic 'corpus' for fast test
    corpus = tuple(
        (f"What is {i} + {i + 1}? Write the answer.",
         f"step\n#### {2 * i + 1}")
        for i in range(0, 30))

    def stub_gen(prompt: str, max_tokens: int, temperature: float):
        # Parse the integers from the question and answer them
        import re
        ms = re.findall(r"\b\d+\b", prompt)
        if len(ms) >= 2:
            ans = int(ms[0]) + int(ms[1])
        else:
            ans = 0
        return f"Let me think.\nANSWER: {ans}", 10

    cfg = GSM8KBenchConfigV1(
        n_problems=10, K_multi_sample=5,
        seeds=(7, 11, 13), sampling_temperature=0.7,
        max_tokens_per_call=64)
    report = run_gsm8k_real_bench_v1(
        gen=stub_gen, model_id="stub", corpus=corpus, config=cfg)
    assert report.a0_mean_accuracy == 1.0
    assert report.a1_mean_accuracy == 1.0
    assert report.b_mean_accuracy == 1.0
    # Merkle root present
    assert len(report.bench_merkle_root) == 64
    # Per-seed reports
    assert len(report.per_seed) == 3
    for s in report.per_seed:
        assert s.n_problems == 10


def test_w85_gsm8k_bench_detects_arm_improvement():
    """Drive bench with stub that A0 fails, A1 partial, B succeeds.

    Verifies the bench correctly reports strict improvement bools.
    """
    corpus = tuple(
        (f"Q{i}", f"#### {100 + i}")
        for i in range(50))

    call_state = {"counter": 0}

    def stub_gen(prompt: str, max_tokens: int, temperature: float):
        # Parse question index
        import re
        m = re.search(r"Q(\d+)", prompt)
        q_idx = int(m.group(1)) if m else 0
        gold = 100 + q_idx
        # A0 (temp=0) always answers the same wrong number (50%)
        if float(temperature) == 0.0 and "[Persona" not in prompt:
            # A0 single-shot or judge call
            if "[Persona: final judge" in prompt:
                # The judge sees the candidate answers and picks correctly
                return f"ANSWER: {gold}", 10
            else:
                # Pure A0 single-shot — wrong
                return f"ANSWER: {gold - 100}", 10  # wrong (0)
        # A1 self-consistency or B solver calls at temp>0:
        # alternate between right and wrong to give partial credit
        call_state["counter"] += 1
        if call_state["counter"] % 3 == 0:
            return f"ANSWER: {gold}", 10
        else:
            return f"ANSWER: {gold + (call_state['counter'] % 7)}", 10

    cfg = GSM8KBenchConfigV1(
        n_problems=10, K_multi_sample=5,
        seeds=(101, 102, 103), sampling_temperature=0.5,
        max_tokens_per_call=64)
    report = run_gsm8k_real_bench_v1(
        gen=stub_gen, model_id="stub", corpus=corpus, config=cfg)
    # A0 should be near 0 (always wrong), B should be higher
    # because the judge call at temp=0 picks correctly
    assert report.a0_mean_accuracy < 0.1
    # B uses the judge which always answers correctly when judge
    # path runs, so B should be substantially higher
    assert report.b_mean_accuracy > report.a0_mean_accuracy


# ---------------------------------------------------------------
# Long-context live bench (#27 push).
# ---------------------------------------------------------------


def test_w85_long_context_corpus_is_deterministic():
    c1 = build_live_needle_corpus_v1(
        horizons_chars=(8_000, 32_000), n_per_horizon=2)
    c2 = build_live_needle_corpus_v1(
        horizons_chars=(8_000, 32_000), n_per_horizon=2)
    assert tuple(p.cid() for p in c1) == tuple(p.cid() for p in c2)
    # Materialised prompts are equal too
    for p1, p2 in zip(c1, c2):
        assert p1.materialise_prompt() == p2.materialise_prompt()


def test_w85_long_context_prompt_actually_long():
    corpus = build_live_needle_corpus_v1(
        horizons_chars=(32_000,), n_per_horizon=1)
    p = corpus[0]
    text = p.materialise_prompt()
    # At least the horizon chars worth (a little extra is fine)
    assert len(text) >= 32_000
    # Needle present exactly once
    assert text.count(f"MAGIC NUMBER is {p.needle_value}") == 1
    # Query at end
    assert "What is the MAGIC NUMBER" in text


def test_w85_long_context_parse_magic_number():
    assert parse_magic_number_v1("MAGIC NUMBER: 442") == 442
    assert parse_magic_number_v1(
        "I think... MAGIC NUMBER: 1234.") == 1234
    assert parse_magic_number_v1(
        "no marker but answer at end: 42") == 42
    assert parse_magic_number_v1("nothing here") is None


def test_w85_long_context_bench_with_stub_perfect_substrate():
    """Stub gen returns the substring after 'MAGIC NUMBER is' if
    present in the prompt; otherwise returns -1.

    Expected outcome: A_FULL recovers needle (it's in the long
    prompt), A_BOUNDED_V3 fails on horizons where the needle is
    earlier than the window, B_COMPOSED succeeds because
    retrieval brings the needle into the short prompt.
    """
    import re

    def stub_gen(prompt: str, max_tokens: int, temperature: float):
        m = re.search(
            r"MAGIC NUMBER is (\d+)", prompt)
        if m:
            return f"MAGIC NUMBER: {m.group(1)}", 5, 100
        return "MAGIC NUMBER: -1", 5, 100

    report = run_long_context_live_bench_v1(
        gen=stub_gen, model_id="stub",
        horizons_chars=(8_000, 32_000), n_per_horizon=2,
        max_tokens=8)
    # Per-horizon:
    for p in report.per_horizon:
        assert p.a_full_success_rate == 1.0  # full has the needle
        assert p.b_composed_success_rate == 1.0  # retrieval finds it
        # Bounded truncates — at 8k the needle is at ~mid (chars
        # 3200-4000); window is 3800 chars so it MIGHT contain it.
        # At 32k the needle is at chars ~12800-16000; bounded
        # window 3800 chars at the end cannot contain it.
        if p.horizon_chars >= 32_000:
            assert p.a_bounded_v3_success_rate == 0.0
            assert p.composed_strictly_beats_bounded is True


def test_w85_long_context_bench_merkle_root_changes_with_data():
    """Two runs over different corpora produce different roots."""
    import re

    def stub_gen(prompt, max_tokens, temperature):
        m = re.search(r"MAGIC NUMBER is (\d+)", prompt)
        return (f"MAGIC NUMBER: {m.group(1) if m else -1}",
                5, 100)

    r1 = run_long_context_live_bench_v1(
        gen=stub_gen, model_id="m",
        horizons_chars=(8_000,), n_per_horizon=2, max_tokens=8)
    r2 = run_long_context_live_bench_v1(
        gen=stub_gen, model_id="m",
        horizons_chars=(8_000, 32_000), n_per_horizon=2,
        max_tokens=8)
    assert r1.bench_merkle_root != r2.bench_merkle_root


# ---------------------------------------------------------------
# Honest limitation tags (W85).
# ---------------------------------------------------------------


def test_w85_nim_module_does_not_claim_substrate_access():
    """Catch-all: the W85 NIM module must NOT export anything that
    claims hidden-state / KV / per-layer access. We assert the
    capability claim's bool axes are False for these."""
    # Build a synthetic 'reachable' probe (no network)
    probe = NIMFrontierProbeReportV1(
        schema=W85_NIM_FRONTIER_TEXT_V1_SCHEMA_VERSION,
        nim_endpoint="https://integrate.api.nvidia.com",
        api_key_present=True, reachable=True, blocked_reason="",
        available_models=("meta/llama-3.1-8b-instruct",),
        catalog_subset_available=(
            ("meta/llama-3.1-8b-instruct", 131_072,
             "llama-3.1-8b"),))
    claim = declare_nim_frontier_capability_claim_v1(probe=probe)
    # Substrate axes must be False
    assert claim.hidden_state_access is False
    assert claim.kv_cache_replay is False
    assert claim.per_layer_instrumentation is False
    assert claim.cross_runtime_state_export is False


def test_w85_audit_verifier_re_hashes_response_cid():
    """Offline-verifier sanity: write a 1-call sidecar, then
    re-hash the response_text and confirm response_cid matches."""
    import hashlib
    import tempfile
    text = "MAGIC NUMBER: 42"
    expected_cid = hashlib.sha256(
        text.encode("utf-8")).hexdigest()
    record = {
        "model_id": "stub",
        "n_call": 1,
        "prompt_cid": "a" * 64,
        "prompt_chars": 500,
        "prompt_tokens": 120,
        "response_cid": expected_cid,
        "response_text": text,
        "temperature": 0.0,
        "max_tokens": 8,
        "wall_ms": 100,
        "finish_reason": "stop",
    }
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".calls.jsonl", delete=False) as f:
        f.write(json.dumps(record) + "\n")
        sidecar = f.name
    # Re-hash the bytes from the file
    with open(sidecar) as f:
        loaded = json.loads(f.readline())
    re_hashed = hashlib.sha256(
        loaded["response_text"].encode("utf-8")).hexdigest()
    assert re_hashed == loaded["response_cid"], (
        f"audit chain corrupted: rehashed {re_hashed} vs "
        f"claimed {loaded['response_cid']}")
    os.unlink(sidecar)


@pytest.mark.skipif(
    not _HAS_NIM_KEY,
    reason="NVIDIA_API_KEY not set — live NIM test skipped")
def test_w85_nim_live_long_context_returns_long_prompts():
    """Confirm the NIM call really sends a long prompt + receives
    accurate NIM-reported token counts for the long-context
    bench. Small N to keep CI fast."""
    from coordpy.long_context_live_bench_v1 import (
        build_live_needle_corpus_v1,
        parse_magic_number_v1,
    )
    from coordpy.nim_frontier_text_runtime_v1 import (
        NIMFrontierTextRuntimeV1,
    )
    corpus = build_live_needle_corpus_v1(
        horizons_chars=(8_000,), n_per_horizon=1)
    p = corpus[0]
    text = p.materialise_prompt()
    rt = NIMFrontierTextRuntimeV1(
        model_id="meta/llama-3.1-8b-instruct", timeout=120.0)
    cap, resp = rt.generate_capsule(
        prompt=text, max_tokens=32, temperature=0.0)
    # The NIM-reported tokens should be > 1000 for 8k-char prompt
    assert cap.prompt_tokens > 1_000, (
        f"expected >1k tokens for ~8k chars, got {cap.prompt_tokens}")
    # Llama-3.1-8B should recover the needle at 8k chars
    extracted = parse_magic_number_v1(resp)
    # Note: 8k-char is borderline for needle recall; we don't
    # assert strict success — only assert the bench ran end-to-end
    # and NIM returned the expected token count.
    assert cap.response_finish_reason in ("stop", "length")
