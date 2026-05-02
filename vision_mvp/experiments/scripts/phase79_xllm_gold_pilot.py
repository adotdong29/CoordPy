"""Pilot probe: live cross-architecture LLM agreement + gold correlation.

Tests whether gemma2:9b (localhost) and qwen2.5:14b (192.168.12.191)
disagree at temperature 0 on a small bench of *gold-verifiable*
structured-decision prompts AND, when they disagree, whether one host's
answer is systematically more often the gold answer than the other.

This is the live evidence we need to construct
**R-79-XLLM-LIVE-GOLD** for the W31-C-CROSS-HOST-VARIANCE-LIVE-
MAGNITUDE-LIVE gold-correlation axis.

The probe is honestly-null acceptable: if the two models agree on
every gold-verifiable prompt at temp 0, the gold-correlation regime
does not exist on these prompts AND the W32 mechanism cannot be
exercised here.  In that case the W32 layer is still meaningful at
the synthetic + long-window axes; the live gold-correlation axis
remains conjectural at SDK v3.34.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request


HOSTS = [
    ("http://localhost:11434", "localhost", "gemma2:9b"),
    ("http://192.168.12.191:11434", "192.168.12.191", "qwen2.5:14b"),
]


# Each prompt is gold-verifiable: "gold" is the unambiguous correct
# answer.  Categories: simple arithmetic, single-character syntax,
# closed-vocabulary factoid where consensus is documented.
PROMPTS = (
    # Arithmetic — gold is mechanical.
    ("A1", "Answer with EXACTLY one number, no words. What is "
           "17 + 26 - 8?", "35"),
    ("A2", "Answer with EXACTLY one number, no words. What is "
           "12 * 7?", "84"),
    ("A3", "Answer with EXACTLY one number, no words. What is "
           "144 / 12?", "12"),
    ("A4", "Answer with EXACTLY one number, no words. What is "
           "2 ** 10?", "1024"),
    ("A5", "Answer with EXACTLY one number, no words. What is "
           "(3 + 4) * 5?", "35"),
    # Syntax / closed-vocab factoid.
    ("S1", "Answer with EXACTLY one word, no punctuation. The "
           "ISO 3166-1 alpha-2 code for Japan is what?", "jp"),
    ("S2", "Answer with EXACTLY one word, no punctuation. In the "
           "linux kernel, the system call to read a file descriptor "
           "is named what?", "read"),
    ("S3", "Answer with EXACTLY one word, no punctuation. The Python "
           "built-in keyword used to begin a function definition is "
           "what?", "def"),
    ("S4", "Answer with EXACTLY one word, no punctuation. The HTTP "
           "status code for 'Not Found' as a number is what?", "404"),
    ("S5", "Answer with EXACTLY one word, no punctuation. The Unix "
           "command that lists files in a directory is what?", "ls"),
    # Closed-vocabulary factoids with single-token gold.
    ("F1", "Answer with EXACTLY one word, no punctuation. The "
           "currency of Japan is what?", "yen"),
    ("F2", "Answer with EXACTLY one word, no punctuation. The "
           "chemical symbol for gold is what?", "au"),
    ("F3", "Answer with EXACTLY one word, no punctuation. The "
           "capital of France is what?", "paris"),
    ("F4", "Answer with EXACTLY one word, no punctuation. The "
           "first letter of the Greek alphabet is what?", "alpha"),
    ("F5", "Answer with EXACTLY one word, no punctuation. The "
           "smallest prime number is what?", "2"),
    # Disambiguation prompts where one model family may bias.
    ("D1", "Answer with EXACTLY one word, no punctuation. In "
           "Python, the recommended type-hint annotation for a "
           "deprecated function uses which decorator from typing? "
           "(One word.)", "deprecated"),
    ("D2", "Answer with EXACTLY one word, no punctuation. In a "
           "binary tree's level-order traversal, which data "
           "structure is used? (One word; queue or stack.)", "queue"),
    ("D3", "Answer with EXACTLY one word, no punctuation. The "
           "default Git remote name when you clone a repository "
           "is what?", "origin"),
    ("D4", "Answer with EXACTLY one word, no punctuation. In "
           "Markov chain analysis, the stationary distribution is "
           "the right or left eigenvector? (One word.)", "left"),
    ("D5", "Answer with EXACTLY one word, no punctuation. In a "
           "TCP three-way handshake, the second packet sent is "
           "called what? (One word; e.g. ack.)", "syn-ack"),
)


def call_ollama(host_url: str, model: str, prompt: str,
                timeout_s: float = 90.0) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 0,
                    "num_predict": 16},
    }).encode()
    req = urllib.request.Request(
        f"{host_url}/api/generate",
        data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read().decode())
        return str(data.get("response", "")).strip()
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        return f"<error:{e}>"


def first_token(text: str) -> str:
    text = text.strip().lower()
    for ch in ('"', "'", "`", ".", ",", ":", ";", "!", "?", "*", "\n"):
        text = text.replace(ch, " ")
    text = text.strip()
    if not text:
        return ""
    return text.split()[0]


def main() -> int:
    print(f"Live cross-architecture LLM gold-correlation pilot @ "
          f"{time.ctime()}")
    print(f"Hosts: {HOSTS}\n")
    print(f"Total prompts: {len(PROMPTS)}\n")
    n_total = 0
    n_agreement = 0
    n_disagreement = 0
    n_a_correct = 0
    n_b_correct = 0
    n_a_correct_on_disagreement = 0
    n_b_correct_on_disagreement = 0
    n_neither_correct_on_disagreement = 0
    rows = []
    for tag, prompt, gold in PROMPTS:
        words = []
        out = []
        for url, host_id, model in HOSTS:
            t0 = time.time()
            resp = call_ollama(url, model, prompt)
            dt = time.time() - t0
            w = first_token(resp)
            out.append({"host_id": host_id, "model": model,
                        "first_token": w, "raw": resp[:80],
                        "dt": round(dt, 2)})
            words.append(w)
        gold_tok = first_token(gold)
        a_correct = (words[0] == gold_tok and
                     not words[0].startswith("<error"))
        b_correct = (words[1] == gold_tok and
                     not words[1].startswith("<error"))
        agree = (words[0] == words[1] and words[0] not in ("",) and
                 not any(w.startswith("<error") for w in words))
        n_total += 1
        if agree:
            n_agreement += 1
        else:
            n_disagreement += 1
            if a_correct:
                n_a_correct_on_disagreement += 1
            elif b_correct:
                n_b_correct_on_disagreement += 1
            else:
                n_neither_correct_on_disagreement += 1
        if a_correct:
            n_a_correct += 1
        if b_correct:
            n_b_correct += 1
        rows.append({
            "tag": tag,
            "prompt": prompt,
            "gold": gold_tok,
            "host_a": out[0],
            "host_b": out[1],
            "agree": bool(agree),
            "host_a_correct": bool(a_correct),
            "host_b_correct": bool(b_correct),
        })
        marker = "AGREE" if agree else (
            f"DISAGREE A={words[0]!r} B={words[1]!r}")
        gold_marker = (
            f"  [gold={gold_tok!r} A_correct={a_correct} "
            f"B_correct={b_correct}]")
        print(f"  {tag}: {marker}{gold_marker}")

    print()
    print(f"agreement: {n_agreement}/{n_total} = "
          f"{n_agreement / n_total:.3f}")
    print(f"disagreement: {n_disagreement}/{n_total} = "
          f"{n_disagreement / n_total:.3f}")
    print(f"host_a correct (overall): {n_a_correct}/{n_total} = "
          f"{n_a_correct / n_total:.3f}")
    print(f"host_b correct (overall): {n_b_correct}/{n_total} = "
          f"{n_b_correct / n_total:.3f}")
    if n_disagreement > 0:
        print(f"on disagreement: A correct {n_a_correct_on_disagreement} / "
              f"B correct {n_b_correct_on_disagreement} / "
              f"neither {n_neither_correct_on_disagreement}")

    out_path = os.environ.get(
        "OUT_JSON",
        "vision_mvp/experiments/artifacts/phase79/"
        "xllm_live_gold_pilot.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "hosts": [{"url": u, "host_id": h, "model": m}
                      for u, h, m in HOSTS],
            "n_total": n_total,
            "n_agreement": n_agreement,
            "agreement_rate": n_agreement / n_total,
            "n_disagreement": n_disagreement,
            "disagreement_rate": n_disagreement / n_total,
            "n_a_correct": n_a_correct,
            "n_b_correct": n_b_correct,
            "host_a_overall_accuracy": n_a_correct / n_total,
            "host_b_overall_accuracy": n_b_correct / n_total,
            "n_a_correct_on_disagreement": n_a_correct_on_disagreement,
            "n_b_correct_on_disagreement": n_b_correct_on_disagreement,
            "n_neither_correct_on_disagreement":
                n_neither_correct_on_disagreement,
            "results": rows,
            "timestamp": time.ctime(),
        }, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
