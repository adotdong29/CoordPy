"""W42 / Phase 89 -- live cross-host paraphrase-invariance probe.

Bounded live probe at temperature 0 on the two-Mac topology:

* localhost (Mac 1) — gemma2:9b
* 192.168.12.191 (Mac 2 / HSC136047-MAC.lan) — qwen2.5:14b

For each of K=4 paraphrases of one gold-verifiable closed-
vocabulary arithmetic prompt, query both hosts and record the
returned token.  The W42 mechanism is closed-form and capsule-
layer; this live probe is a *realism anchor* for the
role-handoff invariance / paraphrase-invariance thesis (when
the same logical question is rephrased, the model's answer
should be invariant) — not load-bearing for the W42 success
criterion.  H10 only requires that the probe is *recorded
honestly*; it does not require any particular pass rate.

Output is written to
``vision_mvp/experiments/artifacts/phase89/xllm_paraphrase_probe_<DATE>.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import urllib.error
import urllib.request


_DEFAULT_LOCALHOST = "http://localhost:11434"
_DEFAULT_REMOTE = "http://192.168.12.191:11434"
_DEFAULT_LOCAL_MODEL = "gemma2:9b"
_DEFAULT_REMOTE_MODEL = "qwen2.5:14b"

# K=4 paraphrases of the same closed-vocabulary arithmetic
# question.  Gold answer: "Four".
_PARAPHRASES: tuple[str, ...] = (
    "What is 2+2? Answer with one word.",
    "Compute two plus two. Answer with one word.",
    "Add two and two. Answer with one word.",
    "2+2 equals what? Answer with one word.",
)
_GOLD = "four"


def _ollama_generate(
        *, base_url: str, model: str, prompt: str,
        timeout: float = 60.0,
        num_predict: int = 4) -> dict:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, socket.timeout, ConnectionError,
            TimeoutError, OSError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        return {"ok": True, "raw": json.loads(data.decode("utf-8"))}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"JSONDecodeError: {exc}"}


def _normalise(token: str) -> str:
    return "".join(
        c for c in str(token).strip().lower()
        if c.isalnum())


def run_xllm_paraphrase_probe(
        *,
        local_url: str = _DEFAULT_LOCALHOST,
        remote_url: str = _DEFAULT_REMOTE,
        local_model: str = _DEFAULT_LOCAL_MODEL,
        remote_model: str = _DEFAULT_REMOTE_MODEL,
        timeout: float = 60.0,
) -> dict:
    """Run the K=4 paraphrase probe on the two-Mac topology."""
    records = []
    for paraphrase_idx, prompt in enumerate(_PARAPHRASES):
        local = _ollama_generate(
            base_url=local_url, model=local_model,
            prompt=prompt, timeout=timeout)
        remote = _ollama_generate(
            base_url=remote_url, model=remote_model,
            prompt=prompt, timeout=timeout)
        local_text = (
            local["raw"].get("response", "")
            if local.get("ok") else "")
        remote_text = (
            remote["raw"].get("response", "")
            if remote.get("ok") else "")
        local_norm = _normalise(local_text)
        remote_norm = _normalise(remote_text)
        records.append({
            "paraphrase_idx": int(paraphrase_idx),
            "prompt": prompt,
            "local_host": local_url,
            "local_model": local_model,
            "local_ok": bool(local.get("ok")),
            "local_error": local.get("error", ""),
            "local_response_text": local_text,
            "local_response_norm": local_norm,
            "local_correct": bool(local_norm == _GOLD),
            "remote_host": remote_url,
            "remote_model": remote_model,
            "remote_ok": bool(remote.get("ok")),
            "remote_error": remote.get("error", ""),
            "remote_response_text": remote_text,
            "remote_response_norm": remote_norm,
            "remote_correct": bool(remote_norm == _GOLD),
            "cross_host_agree_norm": bool(
                local_norm == remote_norm
                and local.get("ok") and remote.get("ok")),
        })

    n = len(records) or 1
    n_local_ok = sum(1 for r in records if r["local_ok"])
    n_remote_ok = sum(1 for r in records if r["remote_ok"])
    n_local_correct = sum(1 for r in records if r["local_correct"])
    n_remote_correct = sum(1 for r in records if r["remote_correct"])
    n_cross_host_agree = sum(
        1 for r in records if r["cross_host_agree_norm"])
    # Within-host paraphrase invariance: count distinct normalised
    # answers per host.
    local_norms = {
        r["local_response_norm"] for r in records
        if r["local_ok"]}
    remote_norms = {
        r["remote_response_norm"] for r in records
        if r["remote_ok"]}
    summary = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_url": local_url,
        "remote_url": remote_url,
        "local_model": local_model,
        "remote_model": remote_model,
        "n_paraphrases": int(n),
        "n_local_ok": int(n_local_ok),
        "n_remote_ok": int(n_remote_ok),
        "n_local_correct": int(n_local_correct),
        "n_remote_correct": int(n_remote_correct),
        "n_cross_host_agree_norm": int(n_cross_host_agree),
        "local_correct_rate": float(n_local_correct / n),
        "remote_correct_rate": float(n_remote_correct / n),
        "cross_host_agree_rate": float(n_cross_host_agree / n),
        "n_distinct_local_normalised": int(len(local_norms)),
        "n_distinct_remote_normalised": int(len(remote_norms)),
        "local_paraphrase_invariant": bool(
            len(local_norms) == 1 and n_local_ok == n),
        "remote_paraphrase_invariant": bool(
            len(remote_norms) == 1 and n_remote_ok == n),
        "gold_norm": _GOLD,
    }
    return {"summary": summary, "records": records}


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="W42 cross-host paraphrase-invariance probe.")
    parser.add_argument(
        "--local-url", default=_DEFAULT_LOCALHOST)
    parser.add_argument(
        "--remote-url", default=_DEFAULT_REMOTE)
    parser.add_argument(
        "--local-model", default=_DEFAULT_LOCAL_MODEL)
    parser.add_argument(
        "--remote-model", default=_DEFAULT_REMOTE_MODEL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--out-dir", default=os.path.join(
            "vision_mvp", "experiments", "artifacts", "phase89"))
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    result = run_xllm_paraphrase_probe(
        local_url=args.local_url,
        remote_url=args.remote_url,
        local_model=args.local_model,
        remote_model=args.remote_model,
        timeout=args.timeout)
    today = dt.date.today().isoformat().replace("-", "_")
    out_path = os.path.join(
        out_dir, f"xllm_paraphrase_probe_{today}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
