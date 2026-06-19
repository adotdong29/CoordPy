"""W127 Lane β — EXPOSED-family scaffold-generation development bench (NIM; dev spend
ALLOWED for mechanism validation only — RUNBOOK_W127 § 5).

Validates the family-specific scaffold-generation line (G1/G2/G3) on a disjoint same-family
EXPOSED dev bench BEFORE any resistant spend.  Deterministic short-name-hash split into
TEACHER problems (their accepted .py -> G1 library) and held-out DEV-TARGET problems (graded;
accepted solution NEVER shown to the generator).  Two arms, SAME K=5 budget:

* baseline ``A_dev`` — plain hosted generation (== W120/W121 A1), pass@5 on official secret.
* scaffold ``G_dev`` — G2-retrieved family scaffold -> G3 fresh generation, pass@5 on secret.

Applies the R1 earn gate (net scaffold gain >= +2 spanning >= 2 families, leakage-clean,
nontrivial).  Emits results/w127/dev_bench/exposed_dev_bench_verdict.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
from coordpy.family_adapted_repair_synthesis_v1 import SynthesisLeakageGuardV1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    grade_icpc_candidate_case_v1, grade_on_secret_v1)
from scripts.run_w108_livecodebench_pilot import NIM_CHAT_URL  # noqa: E402

W127_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
OUT_DIR = os.path.join(ROOT, "results", "w127", "dev_bench")


def _build_local_nim_gen(*, model, sidecar_writer=None, read_timeout_s=120.0,
                         max_retries=10):
    """NIM chat-completion gen with a TUNABLE read timeout (the shared
    ``_build_nim_gen`` hardcodes 240s/12 — too long when the endpoint stalls; a tighter
    timeout abandons a stalled call fast so the retry can land on a healthy path)."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY not set; W127 dev bench requires NIM.")

    def _gen(prompt, max_tokens, temperature):
        body = {"model": str(model),
                "messages": [{"role": "user", "content": str(prompt)}],
                "max_tokens": int(max_tokens), "temperature": float(temperature),
                "stream": False}
        data = json.dumps(body).encode("utf-8")
        last = None
        for attempt in range(int(max_retries)):
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    NIM_CHAT_URL, data=data, headers={
                        "Content-Type": "application/json", "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}"}, method="POST")
                with urllib.request.urlopen(req, timeout=read_timeout_s) as r:
                    payload = json.loads(r.read().decode("utf-8", errors="replace"))
                wall_ms = int((time.time() - t0) * 1000)
                ch = payload.get("choices") or []
                text = str((ch[0].get("message") or {}).get("content") or "") if ch else ""
                if sidecar_writer is not None:
                    sidecar_writer({
                        "model_id": str(model), "backend": "nim",
                        "prompt_len": len(prompt),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response_len": len(text),
                        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "temperature": float(temperature), "max_tokens": int(max_tokens),
                        "wall_ms": wall_ms, "attempt": attempt,
                        "prompt": str(prompt), "response_text": str(text)})
                return str(text), wall_ms
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2.0 ** attempt + random.random() * 3.0, 60.0))
        raise RuntimeError(f"NIM failed after {max_retries} attempts: {last}")
    return _gen


def _bucket(short: str) -> int:
    return int(hashlib.sha256(short.encode()).hexdigest(), 16) % 3


def _passes_all_samples(problem, code, *, timeout_s):
    """Cheap PUBLIC-sample prescreen: a candidate that fails any public sample is
    definitively wrong (the sample is a known-correct I/O pair) so it cannot pass secret —
    safe to skip the expensive secret grading for pass@K."""
    for inp, exp in problem.samples:
        r = grade_icpc_candidate_case_v1(
            candidate_code=code, stdin_text=inp, expected_stdout=exp,
            kind=problem.kind, float_tol=problem.float_tol, timeout_s=timeout_s)
        if not r.passed:
            return False
    return True


def _pass_at_k(problem, codes, *, timeout_s, sample_timeout_s=5.0):
    """Return (any_pass, first_pass_k, n_parseable).  Prescreens on public samples, then
    grades only sample-passing candidates on the official secret cases (short-circuit)."""
    first = -1
    n_parse = 0
    for k, code in enumerate(codes):
        if not code.strip():
            continue
        n_parse += 1
        if not _passes_all_samples(problem, code, timeout_s=sample_timeout_s):
            continue
        passed, _tail, _n = grade_on_secret_v1(problem, code, timeout_s=timeout_s)
        if passed and first < 0:
            first = k
            return True, first, n_parse
    return False, first, n_parse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--model", default=W127_TARGET_MODEL)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--R", type=int, default=2, help="scaffolds retrieved per target")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=5, help="concurrent NIM dispatch")
    ap.add_argument("--read-timeout-s", type=float, default=120.0)
    ap.add_argument("--dev-bucket", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    probs = G.load_exposed_problems_v1(args.exposed_root)
    teacher = [p for p in probs if _bucket(p.short_name) != args.dev_bucket]
    dev = [p for p in probs if _bucket(p.short_name) == args.dev_bucket]
    dev = sorted(dev, key=lambda p: p.short_name)
    if args.limit:
        dev = dev[:args.limit]
    lib = G.build_scaffold_library_v1(teacher)
    teacher_shorts = {p.short_name for p in teacher}
    dev_shorts = {p.short_name for p in dev}
    assert teacher_shorts.isdisjoint(dev_shorts), "teacher/dev NOT disjoint"
    teacher_cid = hashlib.sha256(
        json.dumps(sorted(teacher_shorts)).encode()).hexdigest()
    dev_cid = hashlib.sha256(json.dumps(sorted(dev_shorts)).encode()).hexdigest()
    print(f"  exposed: {len(probs)} problems; teacher={len(teacher)} dev={len(dev)} "
          f"(disjoint); library={lib.summary()}")
    print(f"  teacher_cid={teacher_cid[:16]}…  dev_cid={dev_cid[:16]}…")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lbl = (f"_{args.label}" if args.label else "")
    out_run = os.path.join(OUT_DIR, f"w127_dev_bench_{run_id}{lbl}")
    os.makedirs(out_run, exist_ok=True)
    sidecar = open(os.path.join(out_run, "dev_bench_calls.jsonl"), "w")
    _slock = threading.Lock()

    def sidecar_writer(rec):
        with _slock:
            sidecar.write(json.dumps(rec, separators=(",", ":")) + "\n")
            sidecar.flush()

    from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1
    gen = _build_local_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer,
                               read_timeout_s=args.read_timeout_s)

    if not args.no_canary:
        print("  canary: 1 plain gen on the first dev target …", flush=True)
        ctext, _w = gen(G.build_plain_prompt_v1(dev[0].as_pilot_problem()),
                        args.max_tokens, 0.0)
        ccode = extract_candidate_code_v1(response_text=ctext)
        print(f"    canary returned {len(ctext)}b, extracted {len(ccode)}b code "
              f"(parses={bool(ccode.strip())})")
        if not ccode.strip():
            raise SystemExit("canary produced no code; aborting before bench spend.")

    # --- precompute per-target prompts + retrieval (deterministic, NO NIM) ---
    plan = []
    for ep in dev:
        prob = ep.as_pilot_problem()
        cls = G.target_family_ranking_v1(ep.statement, ep.samples)
        prio = G.prioritized_families_v1(cls)
        rr = G.retrieve_scaffolds_v1(
            target_short=ep.short_name, target_statement=ep.statement,
            target_family=cls.family, library=lib, R=args.R, candidate_families=prio)
        plan.append({"ep": ep, "prob": prob, "cls": cls, "rr": rr,
                     "base_prompt": G.build_plain_prompt_v1(prob),
                     "scaf_prompt": G.build_scaffolded_prompt_v1(prob, rr.scaffolds)})

    # --- concurrent generation (network-bound; identical prompts/K/temperature) ---
    jobs = []
    for ti, pl in enumerate(plan):
        for _s in range(args.K):
            jobs.append((ti, "base", pl["base_prompt"]))
            jobs.append((ti, "scaf", pl["scaf_prompt"]))

    def _run(job):
        ti, arm, prompt = job
        try:
            text, _w = gen(prompt, args.max_tokens, args.temperature)
            return ti, arm, extract_candidate_code_v1(response_text=text)
        except Exception as e:  # noqa: BLE001 — one dead call must not tank the bench
            print(f"    [gen-fail t{ti} {arm}] {type(e).__name__}: {e}", flush=True)
            return ti, arm, ""   # a failed generation = empty candidate (honest non-pass)

    t0 = time.time()
    gen_out: dict = {}
    n_calls = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ti, arm, code in ex.map(_run, jobs):
            gen_out.setdefault((ti, arm), []).append(code)
            n_calls += 1
            if n_calls % 20 == 0:
                print(f"    … {n_calls}/{len(jobs)} gens ({time.time()-t0:.0f}s)",
                      flush=True)
    print(f"  generation done: {n_calls} calls in {time.time()-t0:.0f}s; grading …",
          flush=True)

    # --- sequential grading (cheap: public-sample prescreen) + results ---
    results = []
    for ti, pl in enumerate(plan):
        ep, prob, rr, cls = pl["ep"], pl["prob"], pl["rr"], pl["cls"]
        base_codes = gen_out.get((ti, "base"), [])
        scaf_codes = gen_out.get((ti, "scaf"), [])
        base_pass, base_k, base_nparse = _pass_at_k(prob, base_codes,
                                                    timeout_s=args.timeout_s)
        scaf_pass, scaf_k, _ = _pass_at_k(prob, scaf_codes, timeout_s=args.timeout_s)
        prov = "\n".join([sc.skeleton for sc in rr.scaffolds] + [ep.statement]
                         + [i + o for i, o in ep.samples])
        guard = SynthesisLeakageGuardV1(  # SECRET-only tripwire (accepted = block check)
            prob, target_accepted_texts=(), provenance_texts=[prov])
        clean, _reason = G.assert_scaffold_pipeline_clean_v1(
            target_short=ep.short_name, scaffolds=rr.scaffolds,
            candidate_texts=scaf_codes, guard=guard,
            target_accepted_texts=list(ep.accepted_codes), provenance=prov)
        trivial = bool(base_nparse == 0)
        r = G.DevBenchTargetResultV1(
            short_name=ep.short_name, family=cls.family,
            families_pulled=rr.families_pulled, n_scaffolds=len(rr.scaffolds),
            baseline_pass=base_pass, scaffold_pass=scaf_pass,
            baseline_first_pass_k=base_k, scaffold_first_pass_k=scaf_k,
            failure_family_was_trivial=trivial, leakage_clean=clean)
        results.append(r)
        flag = ("  *** SCAFFOLD UNIQUE SOLVE ***" if (scaf_pass and not base_pass) else
                ("  (regression)" if (base_pass and not scaf_pass) else ""))
        print(f"   [{ti+1:2d}/{len(dev)}] {ep.short_name:28s} fam={cls.family:16s} "
              f"base={int(base_pass)} scaf={int(scaf_pass)} "
              f"nscaf={len(rr.scaffolds)} clean={int(clean)}{flag}", flush=True)
    wall = time.time() - t0
    sidecar.close()

    verdict_gate = G.apply_dev_bench_earn_gate_v1(results)
    verdict = {
        "schema": "coordpy.w127_exposed_dev_bench.v1", "lane": "beta_exposed_dev_bench",
        "verified_on": _dt.date.today().isoformat(),
        "model_id": str(args.model), "K": args.K, "R": args.R,
        "temperature": args.temperature, "timeout_s": args.timeout_s,
        "nim_calls": n_calls, "wall_s": round(wall, 1),
        "exposed_root": args.exposed_root, "dev_bucket": args.dev_bucket,
        "teacher_corpus_cid": teacher_cid, "dev_target_cid": dev_cid,
        "library": lib.summary(),
        "per_target": [r.to_dict() for r in results],
        "earn_gate": verdict_gate.to_dict(),
        "dev_min_net_gain": G.DEV_MIN_NET_GAIN,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_run, "exposed_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "exposed_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "latest_run.txt"), "w") as f:
        f.write(os.path.basename(out_run))

    print()
    print(f"  baseline {verdict_gate.baseline_total_pass}/{verdict_gate.n_targets} -> "
          f"scaffold {verdict_gate.scaffold_total_pass}/{verdict_gate.n_targets}  "
          f"net={verdict_gate.net_scaffold_gain:+d}")
    print(f"  unique_solves={verdict_gate.scaffold_unique_solves} "
          f"regressions={verdict_gate.scaffold_regressions} "
          f"gain_families={list(verdict_gate.gain_families)}")
    print(f"  EARN GATE: {verdict_gate.verdict_label} (earned={verdict_gate.earned})")
    print(f"    {verdict_gate.rationale}")
    print(f"  nim_calls={n_calls}  wall={wall:.0f}s  -> {out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
