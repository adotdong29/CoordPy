#!/usr/bin/env python3
"""W120-α — Maverick × official-ICPC resistant battlefield cheap-pilot driver.

CONDITIONAL ON the W120 battlefield construction
(``results/w120/icpc_battlefield/battlefield_verdict.json``) having
``pilot_earned=true`` AND ``docs/RUNBOOK_W120.md`` being locked.

Runs the W89 sequential-reflexion B-pipeline + A0 + A1 baselines (stdin/stdout variant,
``icpc_reflexion_bench_v1``) against the deterministic core-tier 30-slice of the official
ICPC multi-surface resistant battlefield (RMRC + ECNA; every problem dated strictly after
Maverick's KNOWN August-2024 cutoff) at ``meta/llama-4-maverick-17b-128e-instruct``,
1 seed × N × K=5 (default 30 ⇒ 330 NIM calls).

Grader = the OFFICIAL secret cases (the W119/W120 self-test-passing oracle); the
reflexion feedback uses ONLY public samples + judge verdict + stderr (never secret data).
The 9 Phase-2 gates + MLB-1/MLB-2 are scored by the VERBATIM W108 evaluator
(``_evaluate_phase2_gates`` / ``_mlb_rates``), byte-identical to W103/W105/W108/W113.

Requires ``NVIDIA_API_KEY``.

Usage::

    python scripts/run_w120_icpc_pilot.py --dry-run                 # 0 NIM: load+validate
    python scripts/run_w120_icpc_pilot.py --n-problems 2 --label canary
    python scripts/run_w120_icpc_pilot.py --n-problems 30
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import re
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    KIND_PASSFAIL,
    classify_battlefield_listing_v1,
    core_slice_cid_v1,
    run_battlefield_construction_v1,
    select_battlefield_core_slice_v1,
)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1,
    IcpcPilotProblemV1,
    run_icpc_reflexion_bench_v1,
)
# Reuse the proven W108 NIM generator + gate evaluator verbatim (no duplication).
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen,
    _evaluate_phase2_gates,
    _mlb_rates,
    _sha256_hex,
)

W120_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
W120_EXPECTED_CORE_SLICE_CID_30 = (
    "01bf9ef869a56e20")  # prefix; full asserted in main()
PKG_CACHE = Path("/tmp/w120_icpc/pkgcache")
ECNA_YEAR_FOLDER = {"2024": "2024-2025", "2025": "2025-2026"}


# ----------------------------------------------------------- statement cleaning
_FIG_RE = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.DOTALL)
_INCLUDE_RE = re.compile(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}")
_CMD1_RE = re.compile(r"\\(?:texttt|emph|textit|textbf|text|mathit|mathrm)\{([^{}]*)\}")
_TTBRACE_RE = re.compile(r"\{\\(?:tt|it|bf|em)\s+([^{}]*)\}")
_PROBNAME_RE = re.compile(r"\\problemname\{([^}]*)\}")
_SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")


def clean_icpc_statement_v1(tex: str) -> str:
    """Deterministic, lossless-where-it-matters LaTeX -> prompt text.

    Removes figure environments + \\includegraphics (images are unavailable to ALL arms
    equally), unwraps common text-formatting macros, renders \\problemname / \\section*
    as headers, and normalizes a few math operators; otherwise keeps the statement
    verbatim (the model reads residual LaTeX competently). No human curation."""
    s = tex
    s = _FIG_RE.sub("\n[figure omitted]\n", s)
    s = _INCLUDE_RE.sub("[figure omitted]", s)
    s = _PROBNAME_RE.sub(r"# \1", s)
    s = _SECTION_RE.sub(r"\n## \1", s)
    for _ in range(3):  # unwrap nested formatting macros
        s = _CMD1_RE.sub(r"\1", s)
        s = _TTBRACE_RE.sub(r"\1", s)
    s = (s.replace("\\ldots", "...").replace("\\dots", "...")
         .replace("\\leq", "<=").replace("\\geq", ">=").replace("\\le", "<=")
         .replace("\\ge", ">=").replace("\\times", "*").replace("\\cdot", "*")
         .replace("\\%", "%").replace("\\#", "#").replace("\\&", "&")
         .replace("\\{", "{").replace("\\}", "}").replace("\\_", "_")
         .replace("\\,", " ").replace("\\;", " ").replace("\\:", " ")
         .replace("\\!", "").replace("~", " ").replace("\\\\", "\n"))
    s = "\n".join(ln for ln in s.splitlines() if not ln.strip().startswith("%"))
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# ----------------------------------------------------------- package fetch + parse
def _gh_bytes(args: list[str], timeout: float = 240.0) -> bytes:
    return subprocess.run(["gh", *args], capture_output=True, timeout=timeout).stdout


def _ensure_rmrc_tarball(repo: str) -> Path:
    dest = PKG_CACHE / repo.replace("/", "_")
    if dest.exists() and any(dest.iterdir()):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    raw = _gh_bytes(["api", f"repos/{repo}/tarball"], timeout=300)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(dest)
    return dest


def _ensure_ecna_zip(short: str, year_folder: str) -> Path:
    dest = PKG_CACHE / "ecna" / year_folder / short
    if dest.exists() and (dest / "problem.yaml").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    url = _gh_bytes(["api",
                     f"repos/icpc/na-ecna-archive/contents/{year_folder}/{short}.zip",
                     "--jq", ".download_url"]).decode().strip()
    raw = subprocess.run(["curl", "-sL", url], capture_output=True, timeout=240).stdout
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        # strip the leading "<short>/" dir
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            rel = n.split("/", 1)[1] if "/" in n else n
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_bytes(zf.read(n))
    return dest


def _find_pkg_dir(root: Path, short: str) -> Path:
    for p in root.rglob("problem.yaml"):
        if p.parent.name == short and (p.parent / "data" / "secret").exists():
            return p.parent
    raise FileNotFoundError(f"package dir for {short} under {root}")


def _read_cases(pkg: Path, sub: str) -> list[tuple[str, str]]:
    d = pkg / "data" / sub
    if not d.exists():
        return []
    cases = []
    for inp in sorted(d.glob("*.in"), key=lambda p: (p.stat().st_size, p.name)):
        ans = inp.with_suffix(".ans")
        if ans.exists():
            cases.append((inp.read_text(errors="replace"),
                          ans.read_text(errors="replace")))
    return cases


def _read_statement(pkg: Path) -> str:
    cand = list((pkg / "problem_statement").glob("*.tex")) if (
        pkg / "problem_statement").exists() else []
    if not cand:
        cand = list(pkg.rglob("problem*.tex"))
    if not cand:
        return f"(statement .tex not found for {pkg.name})"
    # prefer an english statement file
    cand.sort(key=lambda p: (("en" not in p.name.lower()), len(p.name)))
    return clean_icpc_statement_v1(cand[0].read_text(errors="replace"))


def load_pilot_problems(slice_records) -> list[IcpcPilotProblemV1]:
    PKG_CACHE.mkdir(parents=True, exist_ok=True)
    rmrc_roots: dict[str, Path] = {}
    out: list[IcpcPilotProblemV1] = []
    for rec in slice_records:
        repo = rec.source_repo
        if rec.surface == "ECNA":
            yf = ECNA_YEAR_FOLDER[rec.contest_date[:4]]
            pkg = _ensure_ecna_zip(rec.short_name, yf)
        else:
            if repo not in rmrc_roots:
                rmrc_roots[repo] = _ensure_rmrc_tarball(repo)
            pkg = _find_pkg_dir(rmrc_roots[repo], rec.short_name)
        out.append(IcpcPilotProblemV1(
            problem_id=rec.problem_id, short_name=rec.short_name,
            source_repo=repo, contest_date=rec.contest_date,
            statement=_read_statement(pkg), kind=KIND_PASSFAIL, float_tol=1e-6,
            samples=tuple(_read_cases(pkg, "sample")),
            secret_cases=tuple(_read_cases(pkg, "secret"))))
    return out


def _interpret_w120(verdict_label: str, b_minus_a1_pp: float,
                    mlb2: float) -> dict:
    """Pre-committed W120 outcome mapping (RUNBOOK_W120 § 8)."""
    if verdict_label == "PASS_MECHANISM_DRIVEN":
        outcome = "RESISTANT_SUPERIORITY_DEMONSTRATED_SINGLE_SEED"
        w121 = ("CLEAN resistant superiority on an official grader-clean ICPC "
                "battlefield (single seed). W121 = multi-seed same-budget confirmation "
                "to reach W89/W105 retirement-grade on resistant code.")
    elif verdict_label == "PASS_NON_MECHANISM_DRIVEN":
        outcome = "MARGIN_WITHOUT_MECHANISM_LOAD_BEARING"
        w121 = ("Margin present but reflexion not load-bearing (MLB sub-gate fail) => "
                "NOT a clean mechanism win; register bounded; W121 strengthens the "
                "mechanism or accepts the bounded resistant ceiling.")
    else:
        outcome = "BOUNDED_CEILING_HOLDS_ON_RESISTANT_ICPC"
        w121 = ("FAIL on the resistant ICPC battlefield => the bounded contamination-"
                "EXPOSED-HumanEval-family-at-70B ceiling STANDS; resistant superiority "
                "still 0 clean. W121 = accept bounded claim / genuinely different axis.")
    return {"outcome": outcome, "verdict_label": verdict_label,
            "b_minus_a1_pp": float(b_minus_a1_pp), "mlb2_rescue_rate": float(mlb2),
            "w121_branch": w121}


def main() -> int:
    import os
    ap = argparse.ArgumentParser(description="W120 Maverick × resistant-ICPC pilot")
    ap.add_argument("--model", default=W120_TARGET_MODEL)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=120_001)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument(
        "--out-dir", default=str(ROOT / "results" / "w120" / "icpc_pilot"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="load packages + build slice, NO NIM")
    args = ap.parse_args()

    # PILOT-EARNED GATE — refuse to spend unless the battlefield certifies a model.
    res = run_battlefield_construction_v1(verified_on=_dt.date.today().isoformat())
    if not res.pilot_earned:
        raise SystemExit("battlefield not pilot-admissible (core <30 or no certifiable "
                         "model); refusing to spend NIM (W120 § 7).")
    pilot_models = [m.model_id for m in res.per_model if m.pilot_admissible]
    if str(args.model) not in pilot_models:
        raise SystemExit(f"{args.model} is not pilot-admissible {pilot_models}; refusing.")
    print(f"  battlefield: core={res.manifest.n_core_passfail} pilot_earned=True "
          f"certifiable={pilot_models}")

    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(W120_EXPECTED_CORE_SLICE_CID_30):
        raise SystemExit(f"core 30-slice CID {slice_cid[:16]} != expected "
                         f"{W120_EXPECTED_CORE_SLICE_CID_30}; refusing (slice drift).")
    run_slice = list(slice30)[:int(args.n_problems)]
    print(f"  30-slice cid={slice_cid[:16]}…; running {len(run_slice)} problems")

    print("  fetching official ICPC packages (cached) ...")
    problems = load_pilot_problems(run_slice)
    for p in problems:
        print(f"    {p.problem_id:52s} samples={len(p.samples)} "
              f"secret={len(p.secret_cases)} stmt={len(p.statement)}b")
    bad = [p.problem_id for p in problems if not p.secret_cases or not p.statement
           or p.statement.startswith("(statement")]
    if bad:
        raise SystemExit(f"package load failed for {bad}; refusing to spend NIM.")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = Path(args.out_dir) / f"w120_icpc_pilot_{safe_model}_{run_id}{lbl}"
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schema": "coordpy.w120_icpc_pilot.v1", "milestone": "W120-alpha",
        "model_id": str(args.model), "seed": int(args.seed),
        "n_problems": len(run_slice), "K_multi_sample": 5,
        "battlefield_instrument": "coordpy_icpc_battlefield_v1",
        "battlefield_manifest_cid": res.manifest.manifest_cid(),
        "battlefield_surfaces": list(res.manifest.surfaces),
        "core_slice_cid": slice_cid,
        "slice_problem_ids": [p.problem_id for p in problems],
        "slice_contest_date_min": min(p.contest_date for p in problems),
        "slice_contest_date_max": max(p.contest_date for p in problems),
        "cutoff_boundary": "2024-08-31", "cutoff_confidence": "KNOWN",
        "contamination_window": ("RESISTANT for Maverick: every problem dated strictly "
                                 "after the Aug-2024 cutoff (NA East Div 2024/2025 + "
                                 "RMRC 2024-25/2025-26)"),
        "grader": "official secret cases (data/secret); token-diff oracle; NO LLM judge",
        "reflexion_feedback": "public samples + judge verdict + stderr ONLY (no secret leak)",
        "max_tokens_per_call": int(args.max_tokens),
        "executor_timeout_s": float(args.timeout_s),
        "clean_reopening_bar": "verdict_label == PASS_MECHANISM_DRIVEN",
        "label": str(args.label),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    if args.dry_run:
        print(f"  --dry-run: validated slice + {len(problems)} packages; "
              f"stopping before NIM. ({out_dir})")
        # statement preview for the first problem (manual faithfulness check)
        print("  --- statement preview (problem 1) ---")
        print("\n".join(problems[0].statement.splitlines()[:14]))
        return 0

    sidecar_f = open(out_dir / "icpc_reflexion_calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = IcpcBenchConfigV1(
        K_multi_sample=5, seeds=(int(args.seed),), sampling_temperature=0.7,
        max_tokens_per_call=int(args.max_tokens),
        executor_timeout_s=float(args.timeout_s))
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=problems, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(problems)} qid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    interp = _interpret_w120(gates["verdict_label"], gates["b_minus_a1_pp"],
                             float(mlb["mlb2_rescue_rate"]))

    rep = report.to_dict()
    rep.update({"wall_s": round(wall_s, 2), "provenance": provenance, "mlb": mlb,
                "phase2_evaluation": gates, "w120_interpretation": interp})
    (out_dir / "icpc_reflexion_bench_report.json").write_text(
        json.dumps(rep, indent=2, default=str))
    (Path(args.out_dir) / "latest_run.txt").write_text(out_dir.name + "\n")

    print()
    print(f"  WALL {wall_s:.1f}s; A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% B={report.b_mean_pass_at_1*100:.2f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  MLB-1 {mlb['mlb1_invocation_rate']*100:.2f}% "
          f"({mlb['n_b_invoked_reflexion']}/{mlb['n_problems_total']}) "
          f"{'PASS' if mlb['mlb1_passes'] else 'FAIL'}; "
          f"MLB-2 {mlb['mlb2_rescue_rate']*100:.2f}% "
          f"({mlb['n_b_rescued_via_reflexion']}/{mlb['n_b_invoked_reflexion']}) "
          f"{'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(f"  Phase-2 {gates['n_phase2_passed_of_9']}/9; verdict {gates['verdict_label']}")
    print(f"  W120 outcome: {interp['outcome']}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
