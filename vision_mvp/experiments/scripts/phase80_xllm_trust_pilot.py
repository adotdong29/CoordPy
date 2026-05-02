"""Pilot probe (W33 milestone): live cross-architecture LLM
trust-calibration agreement at temperature 0.

Tests whether two LLM hosts of *different scales and architectures*
disagree at temperature 0 on a small bench of trust-calibration
prompts (multi-step reasoning, specialised knowledge, ambiguous
syntax) AND, when they disagree, whether one host's answer is
systematically more often the gold answer than the other.

This is the live evidence we need to construct R-80-XLLM-LIVE-TRUST
for the W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE conjecture.

Honestly-null acceptable: if the two models agree on every prompt at
temp 0 OR neither is systematically correct on the disagreed cells,
the live trust-calibration regime does not exist on these prompts.

Design notes
------------
* Pair the largest available model on each host: ``mixtral:8x7b`` on
  localhost (47B-MoE) vs ``qwen3.5:35b`` on 192.168.12.191 (35B
  MoE) — much wider scale + architecture split than the W31/W32
  pair (gemma2:9b vs qwen2.5:14b).
* Prompts are designed to expose trust-calibration: multi-step
  reasoning where the larger model is expected to win, edge-case
  factoids where domain priors differ, ambiguous syntax where
  tokenisation may differ.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request


HOSTS = [
    ("http://localhost:11434", "localhost", "mixtral:8x7b"),
    ("http://192.168.12.191:11434", "192.168.12.191", "qwen3.5:35b"),
]


# Each prompt is gold-verifiable: "gold" is the unambiguous correct
# answer.  Categories chosen to expose trust-calibration: multi-step
# reasoning, specialised factoid, ambiguous syntax.
PROMPTS = (
    # Multi-step reasoning — gold is mechanical but requires several
    # steps.  Larger reasoning models expected to win on disagreement.
    ("R1", "Answer with EXACTLY one number, no words. If a train "
           "leaves at 9:00 AM going 60 mph and another leaves at "
           "10:30 AM going 80 mph from the same station in the same "
           "direction, at what hour (24-hour clock, integer) does "
           "the second train catch the first?", "16"),
    ("R2", "Answer with EXACTLY one integer, no words. Compute "
           "the number of integers between 1 and 1000 inclusive "
           "that are divisible by 3 but not by 5.", "267"),
    ("R3", "Answer with EXACTLY one integer, no words. The greatest "
           "common divisor of 252 and 198 is what?", "18"),
    ("R4", "Answer with EXACTLY one integer, no words. How many "
           "diagonals does a regular hexagon have?", "9"),
    ("R5", "Answer with EXACTLY one integer, no words. If "
           "f(x) = x^2 - 4x + 7, find the minimum value of f.",
           "3"),
    # Specialised factoid — domain priors may differ.
    ("K1", "Answer with EXACTLY one word, no punctuation. The "
           "name of the database isolation level that prevents "
           "all anomalies including phantom reads is what? "
           "(One word, all lowercase.)", "serializable"),
    ("K2", "Answer with EXACTLY one word, no punctuation. The "
           "OSI model layer at which TCP operates is which? "
           "(One word: the layer name.)", "transport"),
    ("K3", "Answer with EXACTLY one word, no punctuation. In "
           "category theory, the dual of a functor is called what? "
           "(One word.)", "cofunctor"),
    ("K4", "Answer with EXACTLY one word, no punctuation. The "
           "name of the cryptographic hash function that produces "
           "a 256-bit output and is part of the SHA-2 family is "
           "what? (One word, no version number.)", "sha256"),
    ("K5", "Answer with EXACTLY one word, no punctuation. The "
           "name of the Linux system call that creates a new "
           "process is what? (One word, lowercase.)", "fork"),
    # Ambiguous syntax — tokenisation boundaries may differ.
    ("S1", "Answer with EXACTLY one word, no punctuation. The "
           "Python operator that performs floor division is what? "
           "(One word, just the symbol or its name.)", "//"),
    ("S2", "Answer with EXACTLY one word, no punctuation. In "
           "regular expressions, the metacharacter that matches "
           "any single character is what? (One word.)", "."),
    ("S3", "Answer with EXACTLY one word, no punctuation. In "
           "C, the operator that returns the size of a type is "
           "what? (One word.)", "sizeof"),
    ("S4", "Answer with EXACTLY one word, no punctuation. The "
           "Bash redirection operator that appends to a file "
           "is what? (One word, just the symbol.)", ">>"),
    ("S5", "Answer with EXACTLY one word, no punctuation. The "
           "C-family operator that means 'logical AND' is what? "
           "(One word, just the symbol.)", "&&"),
    # Trust-calibration: prompts where one model is *systematically*
    # expected to be more correct (e.g. a reasoning model on a
    # reasoning-heavy prompt).
    ("T1", "Answer with EXACTLY one word, no punctuation. The "
           "computational complexity class that contains problems "
           "solvable in polynomial time on a non-deterministic "
           "Turing machine is named what? (One letter.)", "np"),
    ("T2", "Answer with EXACTLY one word, no punctuation. The "
           "name of the algorithm that finds the shortest path "
           "in a weighted graph with non-negative weights is what? "
           "(One word, the surname of its inventor, lowercase.)",
           "dijkstra"),
    ("T3", "Answer with EXACTLY one word, no punctuation. The "
           "term for a graph that has no cycles is what? "
           "(One word.)", "acyclic"),
    ("T4", "Answer with EXACTLY one word, no punctuation. The "
           "name of the parameter-update rule that adjusts learning "
           "rate per parameter using running averages of squared "
           "gradients is what? (One word.)", "rmsprop"),
    ("T5", "Answer with EXACTLY one word, no punctuation. The "
           "term for the property of a function f(x) where "
           "f(a*x) = a*f(x) for all scalars a is what? "
           "(One word.)", "homogeneous"),
)


def call_ollama(host_url: str, model: str, prompt: str,
                timeout_s: float = 120.0) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 0,
                     "num_predict": 60},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return f"<<error: {type(e).__name__}: {e}>>"
    dt = time.monotonic() - t0
    try:
        d = json.loads(body)
        return d.get("response", "").strip()
    except Exception:
        return body.strip()


def first_word(text: str) -> str:
    if not text:
        return ""
    # Strip leading punctuation and whitespace, take first whitespace
    # delimited token, lower-case it.
    cleaned = "".join(
        c for c in text.lstrip() if c.isalnum() or c in ("_-./*+&|<>"))
    if not cleaned:
        return text.strip().lower().split()[0] if text.strip() else ""
    return cleaned.lower().split()[0] if cleaned else ""


def main() -> int:
    print(f"Hosts: {HOSTS}\n")
    results = []
    n_agree = 0
    n_disagree = 0
    n_a_correct = 0
    n_b_correct = 0
    n_a_correct_on_disagree = 0
    n_b_correct_on_disagree = 0
    n_neither_correct_on_disagree = 0
    for tag, prompt, gold in PROMPTS:
        print(f"\n=== {tag} (gold='{gold}') ===")
        per_host = []
        for url, host_id, model in HOSTS:
            t0 = time.monotonic()
            raw = call_ollama(url, model, prompt)
            dt = time.monotonic() - t0
            fw = first_word(raw)
            per_host.append({
                "host_id": host_id, "model": model, "url": url,
                "raw": raw, "first_token": fw,
                "dt": round(dt, 2),
            })
            print(f"  {host_id} ({model}): raw={raw!r} first={fw!r} dt={dt:.2f}s")
        a, b = per_host[0], per_host[1]
        agree = (a["first_token"] == b["first_token"])
        a_correct = (a["first_token"] == gold)
        b_correct = (b["first_token"] == gold)
        if agree:
            n_agree += 1
        else:
            n_disagree += 1
            if a_correct and not b_correct:
                n_a_correct_on_disagree += 1
            elif b_correct and not a_correct:
                n_b_correct_on_disagree += 1
            else:
                n_neither_correct_on_disagree += 1
        if a_correct:
            n_a_correct += 1
        if b_correct:
            n_b_correct += 1
        results.append({
            "tag": tag, "prompt": prompt, "gold": gold,
            "host_a": a, "host_b": b,
            "agree": agree,
            "host_a_correct": a_correct,
            "host_b_correct": b_correct,
        })
    n_total = len(PROMPTS)
    summary = {
        "n_total": int(n_total),
        "n_agreement": int(n_agree),
        "n_disagreement": int(n_disagree),
        "agreement_rate": round(n_agree / n_total if n_total else 0.0, 4),
        "disagreement_rate": round(
            n_disagree / n_total if n_total else 0.0, 4),
        "n_a_correct": int(n_a_correct),
        "n_b_correct": int(n_b_correct),
        "host_a_overall_accuracy": round(
            n_a_correct / n_total if n_total else 0.0, 4),
        "host_b_overall_accuracy": round(
            n_b_correct / n_total if n_total else 0.0, 4),
        "n_a_correct_on_disagreement": int(n_a_correct_on_disagree),
        "n_b_correct_on_disagreement": int(n_b_correct_on_disagree),
        "n_neither_correct_on_disagreement": int(
            n_neither_correct_on_disagree),
        "hosts": [{"host_id": h, "model": m, "url": u}
                   for u, h, m in HOSTS],
        "results": results,
    }
    print("\n\n=== SUMMARY ===")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "results"},
        indent=2))
    out_path = os.environ.get(
        "PHASE80_XLLM_OUT",
        "vision_mvp/experiments/artifacts/phase80/xllm_live_trust_pilot.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
