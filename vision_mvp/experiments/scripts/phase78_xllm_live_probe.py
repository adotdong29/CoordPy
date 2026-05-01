"""Minimal live cross-architecture LLM agreement probe.

Tests whether gemma2:9b on localhost and qwen2.5:14b on
192.168.12.191 systematically agree or disagree at temperature 0
on a small bench of structured-decision prompts.  This is the live
evidence for W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE / the
sharpened W31 conjecture.

The probe is honestly-null acceptable: if the two models agree on
every prompt at temp 0, the cross-host variance witness is correctly
empty and W31 = W30 on this regime.
"""
from __future__ import annotations

import json
import urllib.request
import time
import os
import socket


HOSTS = [
    ("http://localhost:11434", "localhost", "gemma2:9b"),
    ("http://192.168.12.191:11434", "192.168.12.191", "qwen2.5:14b"),
]


PROMPTS = [
    # Each prompt asks for a structured one-line decision.  The prompts
    # are crafted so that different model families could honestly disagree.
    ("Q1", "Answer with EXACTLY one word: orders or payments. "
     "Which of these two services is more likely to handle credit-card "
     "tokenization at the storage layer?"),
    ("Q2", "Answer with EXACTLY one word: api or db. Which of these two "
     "services is the most direct cause of a 502 Bad Gateway error chain?"),
    ("Q3", "Answer with EXACTLY one word: storage or web. Which of these "
     "two services is more likely to be the root cause of a sudden disk-fill?"),
    ("Q4", "Answer with EXACTLY one word: orders or payments. Which "
     "service is more likely to be the SOURCE of an idempotency-key "
     "duplication?"),
    ("Q5", "Answer with EXACTLY one word: db_query or logs_pipeline. "
     "Which service is more likely to be the IMMEDIATE downstream consumer "
     "of a slow REST API response?"),
    ("Q6", "Answer with EXACTLY one word: orders or payments. In a "
     "deadlock between an order checkout and a payment authorization "
     "transaction, which service is more typically the holder of the "
     "FIRST lock?"),
    ("Q7", "Answer with EXACTLY one word: api or storage. In a typical "
     "incident where users see 'image upload failed', which service is "
     "more likely to be the IMMEDIATE error source?"),
    ("Q8", "Answer with EXACTLY one word: payments or storage. In a "
     "PCI-compliant architecture, which service is more likely to be "
     "subject to the strictest data-retention SLA?"),
]


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


def first_word(text: str) -> str:
    text = text.strip().lower()
    # Strip common formatting characters.
    for ch in ('"', "'", "`", ".", ",", ":", ";", "!", "?", "*"):
        text = text.replace(ch, " ")
    text = text.strip()
    if not text:
        return ""
    return text.split()[0]


def main() -> int:
    print(f"Live cross-architecture LLM agreement probe @ {time.ctime()}")
    print(f"Hosts: {HOSTS}\n")
    n_total = 0
    n_agreement = 0
    rows = []
    for tag, prompt in PROMPTS:
        out = []
        words = []
        for url, host_id, model in HOSTS:
            t0 = time.time()
            resp = call_ollama(url, model, prompt)
            dt = time.time() - t0
            w = first_word(resp)
            out.append((host_id, model, w, resp[:80], dt))
            words.append(w)
        agree = (words[0] == words[1] and
                  words[0] not in ("", ) and
                  not any(w.startswith("<error") for w in words))
        n_total += 1
        if agree:
            n_agreement += 1
        rows.append({
            "tag": tag,
            "agree": bool(agree),
            "host_a": out[0][1], "word_a": out[0][2],
            "host_b": out[1][1], "word_b": out[1][2],
            "raw_a": out[0][3], "raw_b": out[1][3],
            "dt_a": round(out[0][4], 2), "dt_b": round(out[1][4], 2),
        })
        print(f"  {tag}: {out[0][2]} ({out[0][1]}, {out[0][4]:.1f}s) "
              f"vs {out[1][2]} ({out[1][1]}, {out[1][4]:.1f}s) "
              f"-> {'AGREE' if agree else 'DISAGREE'}")

    print()
    print(f"agreement: {n_agreement}/{n_total} = "
            f"{n_agreement / n_total:.3f}")
    print(f"disagreement: {n_total - n_agreement}/{n_total} = "
            f"{(n_total - n_agreement) / n_total:.3f}")

    out_path = os.environ.get("OUT_JSON",
                               "vision_mvp/experiments/artifacts/phase78/"
                               "xllm_live_agreement_probe.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "hosts": [{"url": u, "host_id": h, "model": m}
                       for u, h, m in HOSTS],
            "n_total": n_total,
            "n_agreement": n_agreement,
            "agreement_rate": n_agreement / n_total,
            "n_disagreement": n_total - n_agreement,
            "disagreement_rate": (n_total - n_agreement) / n_total,
            "results": rows,
            "timestamp": time.ctime(),
        }, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
