"""W124 / COO-9 — Transformer-native code-intervention mechanism line (M4/M5/M6).

The W120-W123 matched-ICPC battlefield is saturated (resistant +0.00 / exposed
+3.33 / paired-seed B4 unresolvable at n=30 / large-n supply-capped). W124 stops
treating the battlefield as the only lever and mines the repo's unused
transformer/substrate arsenal: AST-aware code reads (``code_substrate_v1``),
real hidden-state intercepts (``transformers_runtime_v1`` /
``hidden_state_intercept_bench_v1``), a learned projector
(``cross_runtime_hidden_state_projector_v1``), the tool-call substrate
(``tool_call_substrate_v1``) and the executor-grounded failure digest
(``executor_grounded_patcher_v1``).

Honesty boundary (this host): the only locally-loadable real transformer is
``distilbert/distilgpt2`` (82M general LM; NO code-fine-tuned model is present and
transformers 4.28.1 is too old for modern code models). distilgpt2 cannot
GENERATE competent ICPC solutions, so this module does NOT claim a
mechanism-beats-A1-in-solve-rate result locally. It runs the **necessary
precursor** for the whole transformer-native line:

  **M4** — does a REAL transformer's hidden state, read at AST function
  boundaries, separate ACCEPTED from FAILED ICPC code on the matched family
  **beyond surface features**? If not, no hidden-state intervention can be built
  on this host (blocked at the precursor) and the hosted lane is NOT earned.

The hosted Maverick API (NIM) is text-only and exposes no hidden state, so
M4/M5 (hidden-state read/write) are local-only and NOT hosted-translatable as
hidden-state interventions; only the text-level M6 controller is. See
``docs/RUNBOOK_W124.md`` §7.

Explicit-import-only (NOT re-exported from ``coordpy/__init__.py``). Stdlib +
numpy at the boundary; torch/transformers/sklearn load lazily only on the real
paths, so this module imports anywhere.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Optional, Sequence

W124_TRANSFORMER_NATIVE_CODE_INTERVENTION_V1_SCHEMA = (
    "coordpy.transformer_native_code_intervention_v1.v1")

# Pre-committed M4 gate thresholds (RUNBOOK_W124 §4). Locked before results.
W124_M4_AUC_FLOOR = 0.60
W124_M4_AUC_MARGIN_OVER_SURFACE = 0.05


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------
# Labeled dataset row
# ---------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LabeledCodeRowV1:
    """One recovered model generation + its official-grader label."""

    problem_id: str
    field: str           # "resistant" | "exposed"
    arm: str             # "A0" | "A1" | "B"
    source: str          # recovered candidate code
    passed: bool         # official grade_on_secret_v1 verdict
    label_source: str    # "regraded" | "report_a0" | "report_b_firstpass" | "report_failed_arm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id, "field": self.field, "arm": self.arm,
            "source_sha": _sha(self.source), "source_len": len(self.source),
            "passed": bool(self.passed), "label_source": self.label_source,
        }


# ---------------------------------------------------------------
# Surface-feature baseline (NO model) — the M4 confound control
# ---------------------------------------------------------------
def surface_features_v1(source: str) -> "list[float]":
    """Cheap structural features computable WITHOUT any transformer.

    The M4 gate requires the hidden-state probe to beat THIS baseline, so a
    positive cannot be a length/shape confound.
    """
    import ast as _ast

    s = source or ""
    lines = s.splitlines() or [""]
    n_chars = float(len(s))
    n_lines = float(len(lines))
    max_indent = float(max((len(ln) - len(ln.lstrip(" ")) for ln in lines), default=0))
    approx_tokens = float(len(s.split()))
    n_def = float(s.count("def "))
    n_for = float(s.count("for ") + s.count("while "))
    n_if = float(s.count("if "))
    try:
        tree = _ast.parse(s)
        n_funcs = float(sum(isinstance(n, _ast.FunctionDef) for n in _ast.walk(tree)))
        parses = 1.0
    except SyntaxError:
        n_funcs = 0.0
        parses = 0.0
    return [n_chars, n_lines, max_indent, approx_tokens, n_def, n_for, n_if,
            n_funcs, parses,
            math.log1p(n_chars), n_chars / (n_lines + 1.0)]


# ---------------------------------------------------------------
# M4 encoder — REAL distilgpt2 hidden state pooled at AST boundaries
# ---------------------------------------------------------------
class AstBoundaryHiddenEncoderV1:
    """Reads REAL transformer hidden state (default distilgpt2) at AST function
    boundaries, reusing ``code_substrate_v1.extract_function_boundaries_v1`` for
    the AST plane.

    The hidden-state READ is a direct HuggingFace load (``AutoModelForCausalLM``,
    ``output_hidden_states=True``) with faithful token->line alignment via the
    fast tokenizer's offset mapping. We deliberately do NOT route through
    ``transformers_runtime_v1`` because that wrapper passes
    ``attn_implementation=`` to ``from_pretrained``, which the only locally
    installed transformers (4.28.1) does not accept — an honest env-incompat
    documented in the W124 verdict. The LOAD-BEARING property (read real hidden
    state from a real transformer at AST boundaries) is preserved.

    ``model_name="stub"`` uses the deterministic stub embedding (no torch) — for
    contract tests ONLY, never for an empirical claim.
    """

    def __init__(self, *, model_name: str = "distilbert/distilgpt2",
                 require_real_model: bool = True, max_length: int = 1024,
                 cache_path: Optional[str] = None) -> None:
        self.model_name = str(model_name)
        self.is_stub = (model_name == "stub")
        self.max_length = int(max_length)
        self.cache_path = cache_path
        self._tok = None
        self._model = None
        self._cache: dict[str, Any] = {}
        self._since_flush = 0
        if cache_path:
            self._load_cache()
        if not self.is_stub and require_real_model:
            self._load_direct_or_raise()

    def _load_cache(self) -> None:
        import os
        import numpy as _np
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                z = _np.load(self.cache_path, allow_pickle=True)
                for k in z.files:
                    self._cache[k] = z[k]
            except Exception:  # noqa: BLE001
                self._cache = {}

    def flush(self) -> None:
        import numpy as _np
        if self.cache_path and self._cache:
            _np.savez(self.cache_path, **self._cache)
            self._since_flush = 0

    @staticmethod
    def _safe_boundaries(source: str):
        from .code_substrate_v1 import extract_function_boundaries_v1
        try:
            return extract_function_boundaries_v1(source)
        except Exception:  # noqa: BLE001 - recovered text may not be valid Python
            return ()

    def _load_direct_or_raise(self) -> None:
        import os
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            from .multi_modal_payload_v1 import BlockedOnHardwareError
            raise BlockedOnHardwareError(
                modality="code", missing=("transformers",),
                hint=f"no torch/transformers for the real M4 encoder: {exc}")
        try:  # CPU forward defaults to 1 thread in this env; use the box.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:  # noqa: BLE001
            pass
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, output_hidden_states=True)
        self._model.eval()

    def encoder_id(self) -> str:
        kind = "stub_sha256_v1" if self.is_stub else "hf_causal_lm_direct"
        return f"{kind}:{self.model_name}"

    def _stub_vec(self, source: str):
        import numpy as _np
        from .code_substrate_v1 import encode_source_with_stub_v1
        full = _np.asarray(encode_source_with_stub_v1(source, embedding_dim=16),
                           dtype=_np.float32)
        if full.ndim != 2 or full.shape[0] == 0:
            full = _np.zeros((1, 16), dtype=_np.float32)
        rows = []
        for b in self._safe_boundaries(source):
            sl = full[max(0, b.start_line - 1):b.end_line, :]
            if sl.shape[0] > 0:
                rows.append(sl.mean(axis=0))
        return (_np.stack(rows).mean(axis=0) if rows else full.mean(axis=0)).astype(_np.float32)

    def encode(self, source: str) -> "Any":
        import numpy as _np
        from .code_substrate_v1 import extract_function_boundaries_v1

        key = _sha(source)
        if key in self._cache:
            return self._cache[key]
        if self.is_stub:
            vec = self._stub_vec(source)
            self._cache[key] = vec
            return vec
        import torch
        try:
            src = source if source.strip() else "\n"
            enc = self._tok(src, return_offsets_mapping=True, truncation=True,
                            max_length=self.max_length, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            with torch.no_grad():
                out = self._model(**enc)
            last = out.hidden_states[-1][0].detach().to(torch.float32).numpy()  # [seq, dim]
            # token -> line via char offset
            nl_prefix = _np.cumsum([1 if ch == "\n" else 0 for ch in src])
            tok_line = _np.asarray([
                int(nl_prefix[min(max(0, int(a)), len(src) - 1)]) if len(src) else 0
                for (a, _b) in offsets])
            rows = []
            for bd in self._safe_boundaries(source):
                mask = (tok_line >= (bd.start_line - 1)) & (tok_line <= (bd.end_line - 1))
                if mask.any():
                    rows.append(last[mask].mean(axis=0))
            vec = (_np.stack(rows).mean(axis=0) if rows else last.mean(axis=0)).astype(_np.float32)
        except Exception:  # noqa: BLE001 - one bad row must not kill the sweep
            vec = _np.zeros(768, dtype=_np.float32)
        self._cache[key] = vec
        self._since_flush += 1
        if self.cache_path and self._since_flush >= 500:
            self.flush()
        return vec


# ---------------------------------------------------------------
# M4 separability probe (problem-disjoint grouped CV, pooled OOF AUC)
# ---------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class M4SeparabilityReportV1:
    schema: str
    n_rows: int
    n_pos: int
    n_groups: int
    encoder_id: str
    auc_hidden: float
    auc_surface: float
    auc_hidden_resistant: float
    auc_hidden_exposed: float
    auc_surface_resistant: float
    auc_surface_exposed: float
    n_splits: int
    label_source_counts: dict

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def cid(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            default=str).encode()).hexdigest()


def _pooled_oof_auc(X, y, groups, *, n_splits: int, seed: int) -> float:
    """Pooled out-of-fold ROC-AUC under problem-disjoint GroupKFold.

    Pooled (not per-fold-mean) so it is well defined even when individual
    grouped folds are single-class — the right choice for the imbalanced,
    grouped ICPC label set.
    """
    import numpy as _np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X = _np.asarray(X, dtype=_np.float64)
    y = _np.asarray(y, dtype=_np.int64)
    groups = _np.asarray(groups)
    oof = _np.full(len(y), _np.nan, dtype=_np.float64)
    gkf = GroupKFold(n_splits=int(n_splits))
    for tr, te in gkf.split(X, y, groups):
        if len(_np.unique(y[tr])) < 2:
            continue  # train fold needs both classes to fit logistic
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 random_state=int(seed))
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    mask = ~_np.isnan(oof)
    if mask.sum() < 2 or len(_np.unique(y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], oof[mask]))


def run_m4_separability_probe_v1(
        rows: Sequence[LabeledCodeRowV1], *,
        encoder: AstBoundaryHiddenEncoderV1,
        n_splits: int = 5, seed: int = 124_000,
        progress: Optional[Any] = None) -> M4SeparabilityReportV1:
    import numpy as _np

    rows = [r for r in rows]
    Xh, Xs, y, groups, fields = [], [], [], [], []
    for i, r in enumerate(rows):
        Xh.append(encoder.encode(r.source))
        Xs.append(surface_features_v1(r.source))
        y.append(1 if r.passed else 0)
        groups.append(r.problem_id)
        fields.append(r.field)
        if progress is not None and (i + 1) % 50 == 0:
            progress(i + 1, len(rows))
    Xh = _np.asarray(Xh, dtype=_np.float64)
    Xs = _np.asarray(Xs, dtype=_np.float64)
    y = _np.asarray(y, dtype=_np.int64)
    groups = _np.asarray(groups)
    fields = _np.asarray(fields)
    n_groups = int(len(set(groups.tolist())))
    eff_splits = max(2, min(int(n_splits), n_groups))

    def _slice_auc(X, mask):
        if mask.sum() < 4 or len(set(y[mask].tolist())) < 2:
            return float("nan")
        gg = groups[mask]
        sp = max(2, min(eff_splits, len(set(gg.tolist()))))
        return _pooled_oof_auc(X[mask], y[mask], gg, n_splits=sp, seed=seed)

    res_mask = fields == "resistant"
    exp_mask = fields == "exposed"
    from collections import Counter
    lsc = dict(Counter(r.label_source for r in rows))
    return M4SeparabilityReportV1(
        schema=W124_TRANSFORMER_NATIVE_CODE_INTERVENTION_V1_SCHEMA,
        n_rows=len(rows), n_pos=int(y.sum()), n_groups=n_groups,
        encoder_id=encoder.encoder_id(),
        auc_hidden=_pooled_oof_auc(Xh, y, groups, n_splits=eff_splits, seed=seed),
        auc_surface=_pooled_oof_auc(Xs, y, groups, n_splits=eff_splits, seed=seed),
        auc_hidden_resistant=_slice_auc(Xh, res_mask),
        auc_hidden_exposed=_slice_auc(Xh, exp_mask),
        auc_surface_resistant=_slice_auc(Xs, res_mask),
        auc_surface_exposed=_slice_auc(Xs, exp_mask),
        n_splits=eff_splits, label_source_counts=lsc)


def m4_gate_v1(report: M4SeparabilityReportV1) -> dict[str, Any]:
    """Pre-committed §4 gate. Returns the verdict + which sub-conditions held."""
    ah, as_ = report.auc_hidden, report.auc_surface
    margin = (ah - as_) if (not math.isnan(ah) and not math.isnan(as_)) else float("nan")
    floor_ok = (not math.isnan(ah)) and ah >= W124_M4_AUC_FLOOR
    margin_ok = (not math.isnan(margin)) and margin >= W124_M4_AUC_MARGIN_OVER_SURFACE
    # sign-consistency across slices: hidden beats surface on BOTH fields
    def _beats(h, s):
        return (not math.isnan(h)) and (not math.isnan(s)) and (h - s) >= 0.0
    sign_consistent = _beats(report.auc_hidden_resistant, report.auc_surface_resistant) and \
        _beats(report.auc_hidden_exposed, report.auc_surface_exposed)
    if floor_ok and margin_ok and sign_consistent:
        verdict = "M4_REAL_SIGNAL"
    elif (not math.isnan(margin)) and margin > 0.0:
        verdict = "M4_CLOSE_BLIP_NOT_A_GAIN"
    else:
        verdict = "M4_SIGNAL_POOR"
    return {
        "verdict": verdict,
        "auc_hidden": ah, "auc_surface": as_, "margin_over_surface": margin,
        "auc_floor": W124_M4_AUC_FLOOR, "margin_floor": W124_M4_AUC_MARGIN_OVER_SURFACE,
        "floor_ok": bool(floor_ok), "margin_ok": bool(margin_ok),
        "sign_consistent": bool(sign_consistent),
        "real_signal": verdict == "M4_REAL_SIGNAL",
    }


# ---------------------------------------------------------------
# M5 — learned repair-steering projector (GATED on M4 real signal)
# ---------------------------------------------------------------
def run_m5_projector_if_earned_v1(
        rows: Sequence[LabeledCodeRowV1], *,
        encoder: AstBoundaryHiddenEncoderV1,
        m4_gate: dict[str, Any]) -> dict[str, Any]:
    """Fit a learned projector on the accepted-vs-failed hidden-state contrast,
    ONLY if M4 earned it. Steering (hidden-state WRITE) is local-only and NOT
    hosted-translatable (RUNBOOK_W124 §7)."""
    if not m4_gate.get("real_signal"):
        return {"status": "NOT_RUN_M4_SIGNAL_POOR",
                "reason": "M5 gated on M4 real signal; M4 did not earn it.",
                "hosted_translatable": False}
    try:
        import numpy as _np
        from .cross_runtime_hidden_state_projector_v1 import (
            fit_learned_hidden_state_projector_v1)
        acc = _np.stack([encoder.encode(r.source) for r in rows if r.passed])
        fail = _np.stack([encoder.encode(r.source) for r in rows if not r.passed])
        delta = acc.mean(axis=0) - fail.mean(axis=0)
        return {"status": "RAN", "contrast_norm": float(_np.linalg.norm(delta)),
                "n_acc": int(len(acc)), "n_fail": int(len(fail)),
                "projector_available": bool(fit_learned_hidden_state_projector_v1),
                "hosted_translatable": False}
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                "hosted_translatable": False}


# ---------------------------------------------------------------
# M6 — deterministic tool-call-substrate code controller (text-translatable)
# ---------------------------------------------------------------
# Materially different from prose reflexion: a TYPED router over the executor
# failure digest that can ABSTAIN (save budget) or PATCH (targeted) vs REPLAN
# (fresh sample), NEVER reading the hidden test source. Deterministic, no hidden
# state -> the ONLY slate member translatable to the hosted text API.
M6_ROUTE_PATCH = "PATCH"
M6_ROUTE_REPLAN = "REPLAN"
M6_ROUTE_ABSTAIN = "ABSTAIN"


@dataclasses.dataclass(frozen=True)
class M6ControllerDecisionV1:
    route: str
    reason: str
    attempt_idx: int
    digest_kind: str


class M6ToolSubstrateCodeControllerV1:
    """Deterministic controller routing patch/replan/abstain on a parsed
    executor failure digest (``executor_grounded_patcher_v1`` +
    ``tool_call_substrate_v1`` plane). Contract-only here (no competent local
    generator); earns a hosted probe only if M4 shows a real local gain AND the
    text-level rule has verdict-changing power (RUNBOOK_W124 §7)."""

    def __init__(self, *, k_budget: int = 5) -> None:
        self.k_budget = int(k_budget)

    def route(self, *, stderr_tail: str, timed_out: bool,
              attempt_idx: int) -> M6ControllerDecisionV1:
        from .executor_grounded_patcher_v1 import parse_failure_digest_v1
        digest = parse_failure_digest_v1(stderr_tail=stderr_tail, timed_out=timed_out)
        kind = getattr(digest, "error_type", None) or getattr(digest, "kind", "unknown")
        kind = str(kind)
        remaining = self.k_budget - int(attempt_idx) - 1
        # Deterministic typed policy (NEVER consults the hidden test source):
        if timed_out:
            route, reason = M6_ROUTE_REPLAN, "timeout => algorithmic; reprompt fresh"
        elif kind.lower() in {"syntaxerror", "indentationerror"}:
            route, reason = M6_ROUTE_PATCH, "structural parse error => targeted minimal patch"
        elif kind.lower() in {"none", "unknown", ""} and remaining <= 0:
            route, reason = M6_ROUTE_ABSTAIN, "no actionable signal + no budget => abstain"
        elif remaining <= 0:
            route, reason = M6_ROUTE_ABSTAIN, "budget exhausted => return best-so-far"
        else:
            route, reason = M6_ROUTE_PATCH, "typed runtime error => targeted patch"
        return M6ControllerDecisionV1(
            route=route, reason=reason, attempt_idx=int(attempt_idx), digest_kind=kind)

    def is_hosted_translatable(self) -> bool:
        """True: the route() rule is deterministic over the executor's TEXT
        stderr — reproducible over the NIM text API; no hidden state."""
        return True


# ---------------------------------------------------------------
# Lane-α verdict assembly
# ---------------------------------------------------------------
def assemble_lane_alpha_verdict_v1(
        *, m4: M4SeparabilityReportV1, m4_gate: dict, m5: dict,
        m6_contract: dict, dataset_manifest: dict) -> dict[str, Any]:
    real = bool(m4_gate.get("real_signal"))
    hosted_earned = bool(
        real and m5.get("hosted_translatable") or False)  # M5/M4 are not hosted-translatable
    # Hosted earn requires a text-translatable gain; M4/M5 are hidden-state (local-only).
    hosted_earned = bool(real and m6_contract.get("hosted_translatable") and
                         m6_contract.get("local_gain_demonstrated", False))
    return {
        "schema": W124_TRANSFORMER_NATIVE_CODE_INTERVENTION_V1_SCHEMA,
        "lane": "alpha_local_transformer_native_code_intervention",
        "dataset_manifest": dataset_manifest,
        "m4_separability": m4.to_dict(),
        "m4_cid": m4.cid(),
        "m4_gate": m4_gate,
        "m5_projector": m5,
        "m6_controller_contract": m6_contract,
        "candidates_survived": ([] if not real else ["M4"]),
        "hosted_maverick_probe_earned": hosted_earned,
        "interpretation": (
            "M4 REAL SIGNAL on distilgpt2 hidden state — transformer-native line "
            "advances; see RUNBOOK_W124 §7/§9." if real else
            "M4 SIGNAL-POOR: distilgpt2 (82M general LM) hidden state does not "
            "separate accepted-vs-failed ICPC code beyond surface features. The "
            "transformer-native hidden-state line is BLOCKED AT THE PRECURSOR on "
            "this host (no code-competent local encoder; transformers 4.28.1 too "
            "old for modern code models) — NOT a refutation of the mechanism "
            "idea. No hosted Maverick probe earned ($0 NIM). M5 not run; M6 "
            "stays contract-only."),
    }


__all__ = [
    "W124_TRANSFORMER_NATIVE_CODE_INTERVENTION_V1_SCHEMA",
    "W124_M4_AUC_FLOOR", "W124_M4_AUC_MARGIN_OVER_SURFACE",
    "LabeledCodeRowV1", "surface_features_v1", "AstBoundaryHiddenEncoderV1",
    "M4SeparabilityReportV1", "run_m4_separability_probe_v1", "m4_gate_v1",
    "run_m5_projector_if_earned_v1",
    "M6ToolSubstrateCodeControllerV1", "M6ControllerDecisionV1",
    "M6_ROUTE_PATCH", "M6_ROUTE_REPLAN", "M6_ROUTE_ABSTAIN",
    "assemble_lane_alpha_verdict_v1",
]
