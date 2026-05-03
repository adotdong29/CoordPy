"""Pilot probe (W34 milestone): live cross-architecture LLM
trust-calibration agreement at temperature 0 with the W34 preflight
discipline.

This corrects the W33 xLLM pilot (which was infra-bounded for two
named reasons recorded as W33-INFRA-1 + W33-INFRA-2):

  * **W33-INFRA-1** — pre-flight ``/api/tags`` model verification.
    The W33 milestone diagnosed qwen3.5:35b on 192.168.12.191 as
    "model not loaded" but a fresh ``/api/tags`` curl on 2026-05-01
    confirms the model IS loaded.  The W33 infra failure was timeout-
    budget combined with chat-template mismatch, NOT model absence.

  * **W33-INFRA-2** — strict token-budget / chat-template for
    one-word probes.  Mixtral:8x7b at default settings emits its
    full chain-of-thought regardless of "EXACTLY one word"
    instructions; the fix is to use ``/api/chat`` with
    ``messages=[{role: system, ...}, {role: user, ...}]`` AND
    ``num_predict=4`` AND ``options.stop=["\n", " ", ".", ","]``.

The W34 probe also attaches a content-addressed
:class:`LiveOracleAttestation` per probe (host_id, model_id,
response_feature_signature, latency_ms_bucket, preflight_ok) — the
W34 audited proxy for native-latent.

Honestly-null acceptable: if the available LLMs at temp 0 agree on
every prompt OR neither is systematically correct on disagreed cells,
the live trust-calibration regime does not exist on these prompts AND
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE remains open.  The W34 probe's
infra closure (W33-INFRA-1 + W33-INFRA-2) is independent of the
agreement-magnitude question.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

from vision_mvp.coordpy.team_coord import (
    LiveOracleAttestation,
    compute_response_feature_signature,
    HostRegistration,
)


# Hosts to probe.  Adaptive timeout per host: small models 30 s,
# medium 60 s, large (>= 30B) 240 s.
HOSTS = (
    ("http://localhost:11434", "localhost", "gemma2:9b", 30_000),
    ("http://localhost:11434", "localhost", "llama3.1:8b", 30_000),
    ("http://localhost:11434", "localhost", "mixtral:8x7b", 60_000),
    ("http://192.168.12.191:11434", "192.168.12.191",
     "qwen2.5:14b", 60_000),
    ("http://192.168.12.191:11434", "192.168.12.191",
     "qwen3.5:35b", 240_000),
)


# Chat-completion-style prompts with a strict system message.  The
# user message asks the question; the system message enforces the
# one-token answer.  Each prompt is gold-verifiable.
SYSTEM_PROMPT = (
    "You are a one-token answerer.  Reply with EXACTLY one word "
    "or symbol.  Do NOT explain, do NOT show reasoning, do NOT "
    "write punctuation, do NOT include preamble.  Output the answer "
    "and nothing else."
)


PROMPTS = (
    # Specialised factoid
    ("K1", "The database isolation level that prevents all "
           "anomalies including phantom reads is named what? "
           "(One word, all lowercase.)", "serializable"),
    ("K2", "The OSI model layer at which TCP operates is named "
           "what? (One word: the layer name.)", "transport"),
    ("K4", "The cryptographic hash function in the SHA-2 family "
           "that produces a 256-bit output is named what? "
           "(One word, no version number.)", "sha256"),
    ("K5", "The Linux system call that creates a new process is "
           "named what? (One word, lowercase.)", "fork"),
    # Symbol questions
    ("S1", "The Python operator that performs floor division is "
           "what? (Just the symbol.)", "//"),
    ("S2", "In regular expressions, the metacharacter that "
           "matches any single character is what? (Just the "
           "symbol.)", "."),
    ("S3", "In C, the operator that returns the size of a type "
           "is named what? (One word.)", "sizeof"),
    ("S4", "The Bash redirection operator that appends to a file "
           "is what? (Just the symbol.)", ">>"),
    ("S5", "The C-family operator that means logical AND is what? "
           "(Just the symbol.)", "&&"),
    # Algorithm / theory
    ("T1", "The complexity class of problems solvable in "
           "polynomial time on a non-deterministic Turing machine "
           "is named what? (One letter.)", "np"),
    ("T2", "The algorithm that finds the shortest path in a "
           "weighted graph with non-negative weights is named "
           "what? (One word, the inventor's surname, lowercase.)",
           "dijkstra"),
    ("T3", "The term for a graph that has no cycles is what? "
           "(One word.)", "acyclic"),
    ("T5", "The term for a function f(x) where f(a*x) = a*f(x) "
           "for all scalars a is what? (One word.)", "homogeneous"),
)


def preflight_check_tags(host_url: str, model_id: str,
                          timeout_s: float = 5.0) -> dict:
    """Closed-form preflight check: query ``/api/tags`` and confirm
    the model_id is in the loaded tags list.  This closes
    W33-INFRA-1.

    Returns a dict with:
      * ``host_url``, ``model_id``
      * ``preflight_ok`` — True iff /api/tags returned 200 AND
        model_id is in the tags list.
      * ``available_tags`` — list of model names the host advertises.
      * ``error`` — error string when preflight_ok is False.
    """
    out = {
        "host_url": host_url,
        "model_id": model_id,
        "preflight_ok": False,
        "available_tags": [],
        "error": "",
    }
    try:
        req = urllib.request.Request(
            f"{host_url}/api/tags", method="GET")
        with urllib.request.urlopen(req,
                                     timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            d = json.loads(body)
            tags = [m.get("name", "") for m in d.get("models", [])]
            out["available_tags"] = tags
            if model_id in tags:
                out["preflight_ok"] = True
            else:
                out["error"] = (
                    f"model {model_id!r} NOT in tags list "
                    f"({len(tags)} models available)")
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def chat_one_token(
        host_url: str, model_id: str, system: str, user: str,
        timeout_s: float,
) -> tuple[str, float, str]:
    """Closed-form one-token probe via ``/api/chat`` with strict
    num_predict=4 and stop tokens.  This closes W33-INFRA-2.

    Returns (raw_response, latency_s, error_str).
    """
    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 0,
            "num_predict": 4,
            "stop": ["\n", " ", ".", ",", "!", "?"],
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req,
                                     timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        dt = time.monotonic() - t0
        d = json.loads(body)
        msg = d.get("message", {})
        content = str(msg.get("content", "")).strip()
        return content, dt, ""
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        dt = time.monotonic() - t0
        return "", dt, f"{type(e).__name__}: {e}"
    except Exception as e:
        dt = time.monotonic() - t0
        return "", dt, f"{type(e).__name__}: {e}"


def first_word(text: str) -> str:
    """Closed-form first-token extraction: lowercase the first
    whitespace-or-punctuation-delimited token.  Handles the symbol
    answers (//, >>, &&, .) by detecting non-alphanumeric prefixes.
    """
    if not text:
        return ""
    s = text.strip().lower()
    # Strip backticks / quotes around answers (some LLMs emit
    # `dijkstra` or 'np').
    s = s.strip("`\"'")
    if not s:
        return ""
    # If the first character is a symbol (//, >>, &&, .) emit the
    # contiguous run of symbols.
    if not s[0].isalnum():
        i = 0
        while i < len(s) and not s[i].isalnum() and s[i] not in (
                " \t\n,;:!?"):
            i += 1
        return s[:i]
    # Otherwise emit the contiguous run of alphanumerics.
    i = 0
    while i < len(s) and (s[i].isalnum() or s[i] == "_"):
        i += 1
    return s[:i]


def latency_bucket_ms(latency_s: float) -> str:
    n = latency_s * 1000.0
    if n <= 1_000:
        return "0..1k"
    if n <= 10_000:
        return "1k..10k"
    if n <= 60_000:
        return "10k..60k"
    if n <= 240_000:
        return "60k..240k"
    return ">240k"


def main() -> int:
    print(f"=== W34 xLLM live pilot (W33-INFRA-1 + W33-INFRA-2 "
          f"closure) ===\n")
    # Step 1: preflight per host+model.
    preflight_results = {}
    print("--- Preflight (/api/tags) ---")
    for host_url, host_id, model_id, _ in HOSTS:
        result = preflight_check_tags(host_url, model_id)
        key = f"{host_id}::{model_id}"
        preflight_results[key] = result
        if result["preflight_ok"]:
            print(f"  [ OK] {key}")
        else:
            print(f"  [SKIP] {key}: {result['error']}")
    print()

    # Step 2: probe each (host, model) where preflight_ok.  For each
    # prompt, send the chat probe with adaptive timeout.
    probe_results = []
    n_total_probes = 0
    n_total_responsive = 0
    n_total_correct = 0
    per_host_correct: dict[str, int] = {}
    per_host_total: dict[str, int] = {}
    pairs = [(host_url, host_id, model_id, timeout_ms)
              for (host_url, host_id, model_id, timeout_ms) in HOSTS
              if preflight_results[
                  f"{host_id}::{model_id}"]["preflight_ok"]]
    if not pairs:
        print("All hosts skipped on preflight.  Honestly null.")
        out = {
            "preflight_results": {
                k: {kk: vv for kk, vv in v.items()
                    if kk != "available_tags"}
                for k, v in preflight_results.items()
            },
            "probes": [],
            "n_total_probes": 0,
            "verdict": "all_hosts_skipped_preflight",
        }
        out_path = os.environ.get(
            "PHASE81_XLLM_OUT",
            "vision_mvp/experiments/artifacts/phase81/"
            "xllm_live_pilot.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {out_path}")
        return 0

    print("--- Probes (chat one-token w/ num_predict=4 + stop) ---")
    for tag, prompt, gold in PROMPTS:
        print(f"\n=== {tag} (gold='{gold}') ===")
        for host_url, host_id, model_id, timeout_ms in pairs:
            timeout_s = timeout_ms / 1000.0
            content, dt, err = chat_one_token(
                host_url, model_id,
                SYSTEM_PROMPT, prompt, timeout_s)
            fw = first_word(content)
            sig = compute_response_feature_signature(
                response_text=content)
            correct = (fw == gold.lower())
            n_total_probes += 1
            host_key = f"{host_id}::{model_id}"
            per_host_total[host_key] = per_host_total.get(
                host_key, 0) + 1
            if not err and content:
                n_total_responsive += 1
            if correct:
                n_total_correct += 1
                per_host_correct[host_key] = per_host_correct.get(
                    host_key, 0) + 1
            att = LiveOracleAttestation(
                oracle_id=f"ollama_{model_id}",
                host_id=host_id,
                model_id=model_id,
                response_feature_signature=sig,
                latency_ms_bucket=latency_bucket_ms(dt),
                preflight_ok=True,
            )
            probe_results.append({
                "tag": tag,
                "gold": gold,
                "host_id": host_id,
                "model_id": model_id,
                "raw_response": content[:120],
                "first_word": fw,
                "correct": correct,
                "latency_s": round(dt, 2),
                "latency_bucket": latency_bucket_ms(dt),
                "response_feature_signature": sig,
                "attestation_cid": att.attestation_cid,
                "preflight_ok": True,
                "error": err,
            })
            status = "OK" if not err else "ERR"
            mark = "C" if correct else " "
            print(f"  [{status}][{mark}] {host_id:18s} "
                  f"{model_id:25s} fw={fw!r:25s} "
                  f"dt={dt:6.2f}s sig={sig}")

    # Step 3: per-host accuracy + cross-host agreement on
    # gold-verifiable prompts.
    summary = {
        "preflight_results": {
            k: {"preflight_ok": v["preflight_ok"],
                "model_id": v["model_id"],
                "error": v["error"],
                "n_available_tags": len(v["available_tags"])}
            for k, v in preflight_results.items()
        },
        "n_total_probes": int(n_total_probes),
        "n_total_responsive": int(n_total_responsive),
        "n_total_correct": int(n_total_correct),
        "responsive_rate": round(
            n_total_responsive / n_total_probes, 4)
            if n_total_probes else 0.0,
        "overall_accuracy_when_responsive": round(
            n_total_correct / n_total_responsive, 4)
            if n_total_responsive else 0.0,
        "per_host_accuracy": {
            k: {
                "n_probes": int(per_host_total[k]),
                "n_correct": int(per_host_correct.get(k, 0)),
                "accuracy": round(
                    per_host_correct.get(k, 0)
                    / max(1, per_host_total[k]), 4),
            }
            for k in per_host_total
        },
        "n_prompts": len(PROMPTS),
        "n_hosts_in_topology": len(HOSTS),
        "n_hosts_passed_preflight": len(pairs),
        "probes": probe_results,
    }

    # Step 4: cross-host agreement / disagreement (when ≥ 2 hosts
    # responded on the same prompt).
    n_agree = 0
    n_disagree = 0
    n_one_correct_on_disagree = 0
    n_neither_correct_on_disagree = 0
    n_both_correct_on_agree = 0
    cross_host_per_prompt = {}
    by_tag: dict[str, list[dict]] = {}
    for p in probe_results:
        by_tag.setdefault(p["tag"], []).append(p)
    for tag, probes in by_tag.items():
        responsive = [pp for pp in probes
                       if not pp["error"]
                       and pp["raw_response"]]
        if len(responsive) < 2:
            cross_host_per_prompt[tag] = {
                "skipped_lt_2_responsive": True}
            continue
        first_words = set(pp["first_word"] for pp in responsive)
        if len(first_words) == 1:
            n_agree += 1
            if any(pp["correct"] for pp in responsive):
                n_both_correct_on_agree += 1
        else:
            n_disagree += 1
            if any(pp["correct"] for pp in responsive):
                n_one_correct_on_disagree += 1
            else:
                n_neither_correct_on_disagree += 1
        cross_host_per_prompt[tag] = {
            "n_responsive": len(responsive),
            "first_words": sorted(first_words),
            "any_correct": any(pp["correct"]
                                for pp in responsive),
            "all_agree": len(first_words) == 1,
        }
    summary["cross_host_per_prompt"] = cross_host_per_prompt
    summary["n_cross_host_agreement"] = int(n_agree)
    summary["n_cross_host_disagreement"] = int(n_disagree)
    summary["n_one_correct_on_disagreement"] = int(
        n_one_correct_on_disagree)
    summary["n_neither_correct_on_disagreement"] = int(
        n_neither_correct_on_disagree)
    summary["n_both_correct_on_agreement"] = int(
        n_both_correct_on_agree)

    # Verdict
    summary["verdict"] = (
        "infra_closure_only_honestly_null_on_magnitude"
        if (n_disagree == 0 or n_one_correct_on_disagree == 0)
        else "infra_closure_AND_live_disagreement_with_gold_correlated_winner"
    )

    out_path = os.environ.get(
        "PHASE81_XLLM_OUT",
        "vision_mvp/experiments/artifacts/phase81/"
        "xllm_live_pilot.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===")
    short = {k: v for k, v in summary.items()
              if k not in ("probes", "cross_host_per_prompt")}
    print(json.dumps(short, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
