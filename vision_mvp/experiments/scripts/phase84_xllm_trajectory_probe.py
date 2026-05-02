"""W37 (SDK v3.38) bounded live cross-host trajectory probe.

Phase 84 needs evidence that, on real LLM hosts, the W37 trajectory
state can accumulate cross-host anchored observations across cells
even when only two hosts are reachable.  Mac 2 (192.168.12.248) has
been ARP-incomplete for the 30th milestone in a row; this probe
honestly records the two-reachable-host fallback and runs the
strongest bounded W37 trajectory probe practical against:

  * localhost: 11434 (mac1)        -- gemma2:9b
  * 192.168.12.191:11434 (mac_remote) -- qwen2.5:14b

The probe asks a small panel of gold-verifiable prompts and groups
the per-host first-token answers by ``(host_id, prompt_id, answer)``.
A trajectory key is "cross-host anchored" iff the same answer was
produced by both hosts on the same prompt.  We then report:

  * how many gold-verifiable prompts were responsive on both hosts;
  * how many of those produced an anchored agreement;
  * the distinct anchored-host count per (prompt_id, answer);
  * the gold-correlated agreement rate.

This is bounded live evidence, not a closure of the W37-C-MULTI-HOST
conjecture (which requires three reachable hosts).
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request


PROBES = (
    ("p1", "Answer in exactly one word: capital city of France?",
     "paris"),
    ("p2", "Answer in exactly one word: chemical symbol for water?",
     "h2o"),
    ("p3", "Answer in exactly one word: how many continents?",
     "seven"),
    ("p4", "Answer in exactly one word: opposite of hot?", "cold"),
    ("p5", "Answer in exactly one word: planet closest to the sun?",
     "mercury"),
    ("p6", "Answer in exactly one word: largest ocean by area?",
     "pacific"),
    ("p7", "Answer in exactly one word: number of sides in a "
     "triangle?", "three"),
    ("p8", "Answer in exactly one word: color of the sky on a "
     "clear day?", "blue"),
)


HOSTS = (
    ("mac1", "http://localhost:11434", "gemma2:9b", 30.0),
    ("mac_remote", "http://192.168.12.191:11434", "qwen2.5:14b",
     60.0),
)


def _preflight_tags(base_url: str, model_id: str,
                     timeout_s: float = 5.0) -> bool:
    try:
        req = urllib.request.Request(
            f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req,
                                    timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        d = json.loads(body)
        tags = [m.get("name", "") for m in d.get("models", [])]
        return model_id in tags
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


def _normalise(answer: str) -> str:
    a = answer.strip().lower()
    # Strip a trailing period; keep first whitespace token only.
    a = a.split()[0] if a else ""
    a = a.strip(" .,:;!?\"'`*-_")
    return a


def _ask(base_url: str, model_id: str, prompt: str,
         timeout_s: float) -> tuple[str, float]:
    body_obj = {
        "model": model_id,
        "messages": [
            {"role": "system",
             "content": "You are a one-token answerer.  Respond with "
             "exactly one lowercase word followed by a period."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 4,
            "stop": ["\n", " ", ".", ",", "!", "?"],
        },
    }
    body = json.dumps(body_obj).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat", method="POST", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req,
                                    timeout=timeout_s) as resp:
            response_body = resp.read().decode("utf-8")
        latency_s = round(time.monotonic() - t0, 3)
        d = json.loads(response_body)
        answer = d.get("message", {}).get("content", "")
        return _normalise(answer), latency_s
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        latency_s = round(time.monotonic() - t0, 3)
        return f"<error:{type(e).__name__}>", latency_s


def main() -> int:
    print("=== W37 bounded live cross-host trajectory probe ===\n")
    # Preflight.
    available: list[tuple[str, str, str, float]] = []
    for host_id, base_url, model_id, timeout_s in HOSTS:
        ok = _preflight_tags(base_url, model_id)
        if ok:
            available.append(
                (host_id, base_url, model_id, timeout_s))
            print(f"  [OK]   {host_id:<12} {model_id:<25}")
        else:
            print(f"  [SKIP] {host_id:<12} {model_id:<25}")
    if len(available) < 2:
        print("\nFewer than two hosts available; honest fallback.\n")
        result = {
            "verdict": "fewer_than_two_hosts_reachable",
            "n_available_hosts": len(available),
            "available": [h[0] for h in available],
            "mac2_status": "ARP-incomplete (30th milestone)",
        }
    else:
        print(f"\nProbing {len(PROBES)} gold-verifiable prompts on "
              f"{len(available)} reachable hosts...\n")
        per_probe: list[dict] = []
        for probe_id, prompt, gold in PROBES:
            row = {
                "probe_id": probe_id,
                "prompt": prompt,
                "gold": gold,
                "responses": {},
            }
            for host_id, base_url, model_id, timeout_s in available:
                ans, latency = _ask(
                    base_url, model_id, prompt, timeout_s)
                row["responses"][host_id] = {
                    "answer": ans,
                    "latency_s": latency,
                    "model_id": model_id,
                }
            per_probe.append(row)
            agree = (
                row["responses"][available[0][0]]["answer"]
                == row["responses"][available[1][0]]["answer"]
                and not row["responses"][available[0][0]]["answer"]
                .startswith("<error"))
            print(f"  {probe_id}: "
                  + " | ".join(
                      f"{available[i][0]}={row['responses'][available[i][0]]['answer']!r}"
                      for i in range(len(available)))
                  + (f"  (agree, gold={gold!r})" if agree
                     else f"  (disagree or error, gold={gold!r})"))
        # Aggregate trajectory metrics.
        n_responsive_both = 0
        n_anchored = 0
        n_anchored_gold = 0
        anchored_keys: dict[tuple[str, str], list[str]] = {}
        for row in per_probe:
            ans_a = row["responses"][available[0][0]]["answer"]
            ans_b = row["responses"][available[1][0]]["answer"]
            if (ans_a.startswith("<error")
                    or ans_b.startswith("<error")):
                continue
            n_responsive_both += 1
            if ans_a == ans_b:
                n_anchored += 1
                key = (row["probe_id"], ans_a)
                anchored_keys.setdefault(key, []).extend(
                    [available[0][0], available[1][0]])
                if ans_a == row["gold"]:
                    n_anchored_gold += 1
        result = {
            "verdict": "bounded_two_host_trajectory_probe",
            "n_probes": len(PROBES),
            "n_available_hosts": len(available),
            "available": [h[0] for h in available],
            "mac2_status": "ARP-incomplete (30th milestone)",
            "n_responsive_both": n_responsive_both,
            "n_anchored": n_anchored,
            "n_anchored_gold_correlated": n_anchored_gold,
            "anchored_rate": (
                round(n_anchored / max(1, n_responsive_both), 4)
                if n_responsive_both else 0.0),
            "gold_correlated_anchored_rate": (
                round(n_anchored_gold / max(1, n_anchored), 4)
                if n_anchored else 0.0),
            "anchored_keys_count": len(anchored_keys),
            "per_probe": per_probe,
        }
        print()
        print("=== SUMMARY ===")
        print(json.dumps({k: v for k, v in result.items()
                          if k != "per_probe"}, indent=2))
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "phase84",
        "xllm_trajectory_probe_2026_05_02.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
