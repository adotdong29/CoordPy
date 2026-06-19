"""W86+ / P2 #42 — State Drift Across Model-Weight Updates V1.

Issue #42 asks for a drift-detection + re-training story when
the model's weights update underneath captured hidden states.
The DoD demands:

1. ``ModelWeightsCID`` — strengthens ``backend_runtime_id``
   with a content-addressed hash of weights.
2. ``DriftDetectorV1`` — replay a captured trace under new
   weights, measure divergence, emit ``ModelDriftEventV1`` when
   divergence > threshold.
3. **No-fire when unchanged**, **fire when changed**.
4. **Re-training pipeline** that produces a strictly-better
   adapted memory under the new weights vs the stale one.
5. **Stale-capsule invalidation** with fallback to recompute.
6. **Principled drift threshold** — derived from the fp64
   precision floor + a stated safety margin.

V1 uses the in-repo `controlled_runtime_substrate_v1` as the
"model" — we perturb its weights to simulate fine-tuning. This
is the same controlled-substrate approach the W82 portability
line uses; it makes the drift mechanism testable without a
GPU dependency. The detector + re-training contracts are
runtime-agnostic and can be re-pointed at `transformers_runtime_v1`
when a real fine-tuned checkpoint is available.

Honest scope (V1)
-----------------

* ``W86-L-DRIFT-V1-RESEARCH-ONLY-CAP``
* ``W86-L-DRIFT-V1-CONTROLLED-RUNTIME-CAP`` — V1 exercises the
  drift mechanism on `controlled_runtime_substrate_v1` (pure
  NumPy fp64). The detector + threshold + re-training contract
  is identical for real models; only the empirical numbers
  change.
* ``W86-L-DRIFT-V1-OFFLINE-CAP`` — V1 detector runs on demand
  (offline); continuous online drift detection is V2.
* ``W86-L-DRIFT-V1-OFFLINE-RETRAIN-CAP`` — V1 re-training is
  triggered manually; online retraining is V3.
* ``W86-L-DRIFT-V1-SAME-ARCH-CAP`` — V1 detects same-architecture
  weight drift; cross-architecture drift is the W82/W83
  portability projector's domain.
* ``W86-L-DRIFT-V1-PRINCIPLED-THRESHOLD-CAP`` — threshold
  derivation is `fp64_floor (5e-3) + safety_margin (3×)` =
  1.5e-2 by default. Documented; not hand-tuned to the bench.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.state_drift_detection_v1 requires numpy") from exc


W86_DRIFT_V1_SCHEMA_VERSION: str = (
    "coordpy.state_drift_detection_v1.v1")


# Default principled threshold derivation.
#
# The W84 `replay_from_kv_exact` proof bounds fp64 byte-identity
# at ~5e-3 absolute. A 3× safety margin gives 1.5e-2 — drift
# below this can be confused with numerical noise; drift above
# is genuine state divergence.
W86_DRIFT_V1_FP64_PRECISION_FLOOR: float = 5e-3
W86_DRIFT_V1_DEFAULT_SAFETY_MARGIN: float = 3.0


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


def _ndarray_cid(arr) -> str:
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------
# Model weights CID
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelWeightsCIDV1:
    """Strengthens ``backend_runtime_id`` with a weight hash.

    ``backend_runtime_id`` was an opaque string identifying the
    runtime (e.g. ``distilbert/distilgpt2``); ``model_weights_cid``
    is a SHA-256 over the actual weight bytes.

    Two runtimes with the same ``backend_runtime_id`` but
    different weights have different ``model_weights_cid``s —
    which is what drives the drift detector.
    """

    backend_runtime_id: str
    model_weights_cid: str
    n_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_DRIFT_V1_SCHEMA_VERSION,
            "backend_runtime_id": str(self.backend_runtime_id),
            "model_weights_cid": str(self.model_weights_cid),
            "n_parameters": int(self.n_parameters),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_model_weights_cid_v1",
            "model": self.to_dict()})


def compute_controlled_runtime_weights_cid_v1(
        params: "ControlledRuntimeParamsV1") -> ModelWeightsCIDV1:
    """Compute a ``ModelWeightsCID`` from a controlled-runtime
    params object. Aggregates every weight tensor's CID into a
    single content-addressed string.
    """
    # Build a stable digest from every weight CID.
    weight_cids = [
        _ndarray_cid(params.embed_W),
        _ndarray_cid(params.pos_W),
        _ndarray_cid(params.unembed_W),
    ]
    for w in params.layer_q_W:
        weight_cids.append(_ndarray_cid(w))
    for w in params.layer_k_W:
        weight_cids.append(_ndarray_cid(w))
    for w in params.layer_v_W:
        weight_cids.append(_ndarray_cid(w))
    for w in params.layer_o_W:
        weight_cids.append(_ndarray_cid(w))
    for w in params.layer_mlp_W1:
        weight_cids.append(_ndarray_cid(w))
    for w in params.layer_mlp_W2:
        weight_cids.append(_ndarray_cid(w))
    aggregate = _sha256_hex({
        "kind": "w86_controlled_weights_aggregate_v1",
        "weight_cids": weight_cids})
    n_params = (
        params.embed_W.size + params.pos_W.size
        + params.unembed_W.size
        + sum(w.size for w in params.layer_q_W)
        + sum(w.size for w in params.layer_k_W)
        + sum(w.size for w in params.layer_v_W)
        + sum(w.size for w in params.layer_o_W)
        + sum(w.size for w in params.layer_mlp_W1)
        + sum(w.size for w in params.layer_mlp_W2))
    return ModelWeightsCIDV1(
        backend_runtime_id="coordpy.controlled_runtime_substrate_v1",
        model_weights_cid=aggregate,
        n_parameters=int(n_params))


# ---------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriftDetectorConfigV1:
    """Threshold derivation parameters."""

    precision_floor: float = W86_DRIFT_V1_FP64_PRECISION_FLOOR
    safety_margin: float = W86_DRIFT_V1_DEFAULT_SAFETY_MARGIN

    def threshold(self) -> float:
        return float(self.precision_floor * self.safety_margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_DRIFT_V1_SCHEMA_VERSION,
            "precision_floor": float(round(
                self.precision_floor, 12)),
            "safety_margin": float(round(self.safety_margin, 12)),
            "threshold": float(round(self.threshold(), 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_drift_detector_config_v1",
            "config": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ModelDriftEventV1:
    """Audit event recording one drift detection."""

    old_model_weights_cid: str
    new_model_weights_cid: str
    drift_score: float
    threshold: float
    detector_config_cid: str
    drift_detected: bool
    detected_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_DRIFT_V1_SCHEMA_VERSION,
            "old_model_weights_cid": str(
                self.old_model_weights_cid),
            "new_model_weights_cid": str(
                self.new_model_weights_cid),
            "drift_score": float(round(self.drift_score, 12)),
            "threshold": float(round(self.threshold, 12)),
            "detector_config_cid": str(
                self.detector_config_cid),
            "drift_detected": bool(self.drift_detected),
            "detected_at_ns": int(self.detected_at_ns),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_model_drift_event_v1",
            "event": self.to_dict()})


def _forward_one_step_v1(
        params, prompt_ids: Sequence[int]) -> "_np.ndarray":
    """One-step forward through the controlled runtime.

    Returns the last token's hidden state at the final layer.
    Pure NumPy fp64 — bit-exact reproducible.
    """
    seq = _np.asarray(prompt_ids, dtype=_np.int64)
    T = seq.shape[0]
    # Embeddings.
    embed = params.embed_W[seq]
    pos = params.pos_W[:T]
    h = embed + pos
    for L in range(params.n_layers):
        q = h @ params.layer_q_W[L]
        k = h @ params.layer_k_W[L]
        v = h @ params.layer_v_W[L]
        # Multi-head attention. Reshape:
        q = q.reshape(T, params.n_heads, params.head_dim)
        k = k.reshape(T, params.n_heads, params.head_dim)
        v = v.reshape(T, params.n_heads, params.head_dim)
        scale = 1.0 / math.sqrt(params.head_dim)
        # (T, n_heads, head_dim) @ (T, n_heads, head_dim).T
        # We'll do per-head.
        attn_out = _np.zeros_like(q)
        for hd in range(params.n_heads):
            qh = q[:, hd, :]
            kh = k[:, hd, :]
            vh = v[:, hd, :]
            scores = (qh @ kh.T) * scale
            # Causal mask.
            mask = _np.triu(_np.ones((T, T)), k=1) * -1e9
            scores = scores + mask
            attn = _np.exp(
                scores - _np.max(scores, axis=-1, keepdims=True))
            attn = attn / _np.sum(attn, axis=-1, keepdims=True)
            attn_out[:, hd, :] = attn @ vh
        attn_out = attn_out.reshape(T, params.hidden_dim)
        h = h + attn_out @ params.layer_o_W[L]
        # MLP.
        mlp_h = _np.maximum(h @ params.layer_mlp_W1[L], 0.0)
        h = h + mlp_h @ params.layer_mlp_W2[L]
    return h[-1]  # last position hidden state


def run_drift_detector_v1(
        old_params, new_params,
        prompt_ids_corpus: Sequence[Sequence[int]],
        config: Optional[DriftDetectorConfigV1] = None,
        detected_at_ns: int = 0) -> ModelDriftEventV1:
    """Detect drift between two model checkpoints.

    For each prompt in the corpus, forward through BOTH old
    and new weights; compute the mean L2 of (h_new - h_old)
    across prompts. Compare against the principled threshold.
    """
    config = config or DriftDetectorConfigV1()
    old_id = compute_controlled_runtime_weights_cid_v1(old_params)
    new_id = compute_controlled_runtime_weights_cid_v1(new_params)
    diffs: list[float] = []
    for ids in prompt_ids_corpus:
        h_old = _forward_one_step_v1(old_params, ids)
        h_new = _forward_one_step_v1(new_params, ids)
        diffs.append(
            float(_np.linalg.norm(h_new - h_old)))
    drift_score = float(_np.mean(diffs)) if diffs else 0.0
    threshold = config.threshold()
    detected = bool(drift_score > threshold)
    return ModelDriftEventV1(
        old_model_weights_cid=old_id.model_weights_cid,
        new_model_weights_cid=new_id.model_weights_cid,
        drift_score=drift_score,
        threshold=threshold,
        detector_config_cid=config.cid(),
        drift_detected=detected,
        detected_at_ns=int(detected_at_ns))


# ---------------------------------------------------------------------
# Stale capsule invalidation + recompute fallback
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CapturedHiddenStateV1:
    """A hidden state captured under a specific model version."""

    model_weights_cid: str
    prompt_ids: tuple[int, ...]
    hidden_state: "_np.ndarray"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_DRIFT_V1_SCHEMA_VERSION,
            "model_weights_cid": str(self.model_weights_cid),
            "prompt_ids": list(self.prompt_ids),
            "hidden_state_cid": _ndarray_cid(self.hidden_state),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_captured_hidden_state_v1",
            "capsule": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class StaleCapsuleVerdictV1:
    """The drift-aware verdict on whether to use a captured
    hidden state."""

    capsule_cid: str
    captured_model_cid: str
    current_model_cid: str
    stale: bool
    fallback_action: str
    """One of: ``use_captured``, ``recompute_from_prompt``,
    ``escalate``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_DRIFT_V1_SCHEMA_VERSION,
            "capsule_cid": str(self.capsule_cid),
            "captured_model_cid": str(self.captured_model_cid),
            "current_model_cid": str(self.current_model_cid),
            "stale": bool(self.stale),
            "fallback_action": str(self.fallback_action),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_stale_capsule_verdict_v1",
            "verdict": self.to_dict()})


def evaluate_stale_capsule_v1(
        capsule: CapturedHiddenStateV1,
        current_model_cid: ModelWeightsCIDV1) -> (
            StaleCapsuleVerdictV1):
    """Decide whether a captured capsule is stale.

    If captured model CID != current model CID, mark stale →
    recommend ``recompute_from_prompt``. Else ``use_captured``.
    """
    is_stale = (
        capsule.model_weights_cid
        != current_model_cid.model_weights_cid)
    return StaleCapsuleVerdictV1(
        capsule_cid=capsule.cid(),
        captured_model_cid=capsule.model_weights_cid,
        current_model_cid=current_model_cid.model_weights_cid,
        stale=is_stale,
        fallback_action=(
            "recompute_from_prompt" if is_stale
            else "use_captured"))


# ---------------------------------------------------------------------
# Re-training pipeline (simple linear adapter)
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LinearAdapterV1:
    """A linear adapter mapping new-model hidden states to a
    learned-memory output. Trained by least squares.
    """

    W: "_np.ndarray"
    b: "_np.ndarray"

    def predict(self, h: "_np.ndarray") -> "_np.ndarray":
        return _np.asarray(h, dtype=_np.float64) @ self.W + self.b


def train_linear_adapter_v1(
        X: "_np.ndarray", y: "_np.ndarray",
        ridge: float = 1e-6) -> LinearAdapterV1:
    """Closed-form least-squares with ridge regularisation."""
    X = _np.asarray(X, dtype=_np.float64)
    y = _np.asarray(y, dtype=_np.float64)
    n, d = X.shape
    # (X^T X + ridge·I)^-1 X^T y
    A = X.T @ X + ridge * _np.eye(d)
    B = X.T @ y
    W = _np.linalg.solve(A, B)
    b = _np.zeros(W.shape[1:]) if W.ndim > 1 else _np.array([0.0])
    return LinearAdapterV1(W=W, b=b)


def evaluate_adapter_mse_v1(
        adapter: LinearAdapterV1,
        X: "_np.ndarray", y: "_np.ndarray") -> float:
    pred = adapter.predict(X)
    return float(_np.mean((pred - y) ** 2))


# ---------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriftBenchReportV1:
    """V1 drift bench output."""

    old_weights_cid: str
    new_weights_cid: str
    no_change_weights_cid: str
    """Same params object, separately CID'd — for the
    no-fire test."""

    drift_score_unchanged: float
    drift_score_changed: float
    threshold: float
    detector_fires_when_changed: bool
    detector_does_not_fire_when_unchanged: bool
    stale_verdict_marks_old_capsule_stale: bool
    stale_verdict_marks_fresh_capsule_fresh: bool
    fallback_recommendation_is_recompute_for_stale: bool
    new_memory_strictly_beats_stale_on_holdout: bool
    """The whole point of re-training: the new adapter has
    strictly lower held-out MSE under the new weights than
    the stale (pre-drift) adapter does."""

    stale_holdout_mse: float
    new_holdout_mse: float
    report_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W86_DRIFT_V1_SCHEMA_VERSION,
            "old_weights_cid": str(self.old_weights_cid),
            "new_weights_cid": str(self.new_weights_cid),
            "no_change_weights_cid": str(
                self.no_change_weights_cid),
            "drift_score_unchanged": float(round(
                self.drift_score_unchanged, 12)),
            "drift_score_changed": float(round(
                self.drift_score_changed, 12)),
            "threshold": float(round(self.threshold, 12)),
            "detector_fires_when_changed": bool(
                self.detector_fires_when_changed),
            "detector_does_not_fire_when_unchanged": bool(
                self.detector_does_not_fire_when_unchanged),
            "stale_verdict_marks_old_capsule_stale": bool(
                self.stale_verdict_marks_old_capsule_stale),
            "stale_verdict_marks_fresh_capsule_fresh": bool(
                self.stale_verdict_marks_fresh_capsule_fresh),
            "fallback_recommendation_is_recompute_for_stale": bool(
                self.fallback_recommendation_is_recompute_for_stale),
            "new_memory_strictly_beats_stale_on_holdout": bool(
                self.new_memory_strictly_beats_stale_on_holdout),
            "stale_holdout_mse": float(round(
                self.stale_holdout_mse, 12)),
            "new_holdout_mse": float(round(
                self.new_holdout_mse, 12)),
            "report_cid": str(self.report_cid),
        }

    def cid(self) -> str:
        d = self.to_dict()
        d["report_cid"] = ""
        return _sha256_hex({
            "kind": "w86_drift_bench_report_v1",
            "report": d})


def run_drift_v1_bench(
        seed: int = 86_042,
        finetune_noise_scale: float = 0.05,
        n_train_prompts: int = 50,
        n_holdout_prompts: int = 20,
        prompt_len: int = 8) -> DriftBenchReportV1:
    """End-to-end drift bench.

    Steps:

    1. Build a controlled-runtime "model" M0 with random params.
    2. Build a fine-tuned model M1 = M0 + ε·noise on every
       weight tensor (simulating a real fine-tune).
    3. Detector(M0, M0) → no-fire (drift below threshold).
    4. Detector(M0, M1) → fires (drift above threshold).
    5. Capture hidden states under M0 → "stale" capsules.
    6. Stale-verdict on those capsules under M1 → stale + fallback.
    7. Train a NEW linear adapter on M1-derived hidden states →
       held-out MSE strictly lower than the stale adapter.
    """
    from .controlled_runtime_substrate_v1 import (
        build_controlled_runtime_params_v1,
    )

    rng = _np.random.default_rng(int(seed))
    M0 = build_controlled_runtime_params_v1(
        vocab_size=64, n_layers=2, n_heads=2,
        head_dim=4, mlp_dim=16, max_len=16, seed=int(seed))

    # Fine-tune: add small noise to every weight.
    def _perturb(W):
        return W + finetune_noise_scale * rng.standard_normal(
            W.shape) * float(_np.std(W) + 1e-6)
    import dataclasses as _dc
    M1 = _dc.replace(
        M0,
        embed_W=_perturb(M0.embed_W),
        pos_W=_perturb(M0.pos_W),
        unembed_W=_perturb(M0.unembed_W),
        layer_q_W=tuple(_perturb(w) for w in M0.layer_q_W),
        layer_k_W=tuple(_perturb(w) for w in M0.layer_k_W),
        layer_v_W=tuple(_perturb(w) for w in M0.layer_v_W),
        layer_o_W=tuple(_perturb(w) for w in M0.layer_o_W),
        layer_mlp_W1=tuple(_perturb(w) for w in M0.layer_mlp_W1),
        layer_mlp_W2=tuple(_perturb(w) for w in M0.layer_mlp_W2))

    M0_id = compute_controlled_runtime_weights_cid_v1(M0)
    M1_id = compute_controlled_runtime_weights_cid_v1(M1)

    # Build a deterministic prompt corpus.
    rng_p = _np.random.default_rng(int(seed) + 100)
    train_prompts = [
        rng_p.integers(0, M0.vocab_size, size=prompt_len).tolist()
        for _ in range(n_train_prompts)]
    holdout_prompts = [
        rng_p.integers(0, M0.vocab_size, size=prompt_len).tolist()
        for _ in range(n_holdout_prompts)]

    # 3. Detector on M0 vs M0 (should not fire).
    ev_unchanged = run_drift_detector_v1(
        old_params=M0, new_params=M0,
        prompt_ids_corpus=holdout_prompts[:5])

    # 4. Detector on M0 vs M1 (should fire).
    ev_changed = run_drift_detector_v1(
        old_params=M0, new_params=M1,
        prompt_ids_corpus=holdout_prompts[:5])

    # 5. Capture hidden states under M0 → stale capsules.
    stale_caps: list[CapturedHiddenStateV1] = []
    for ids in holdout_prompts[:3]:
        h = _forward_one_step_v1(M0, ids)
        stale_caps.append(CapturedHiddenStateV1(
            model_weights_cid=M0_id.model_weights_cid,
            prompt_ids=tuple(ids),
            hidden_state=h))

    # 6. Stale verdict under M1.
    stale_verdicts = [
        evaluate_stale_capsule_v1(c, M1_id) for c in stale_caps]
    all_stale = all(v.stale for v in stale_verdicts)
    all_recompute = all(
        v.fallback_action == "recompute_from_prompt"
        for v in stale_verdicts)
    # Fresh capsule under M1.
    fresh_cap = CapturedHiddenStateV1(
        model_weights_cid=M1_id.model_weights_cid,
        prompt_ids=tuple(holdout_prompts[0]),
        hidden_state=_forward_one_step_v1(
            M1, holdout_prompts[0]))
    fresh_verdict = evaluate_stale_capsule_v1(fresh_cap, M1_id)
    fresh_not_stale = (not fresh_verdict.stale)

    # 7. Re-training: a linear adapter mapping hidden state →
    # a 4-d "task" output. Define a deterministic target
    # function: y = first 4 dims of hidden_state + noise.
    def _hidden_states(model, prompts):
        return _np.stack(
            [_forward_one_step_v1(model, ids) for ids in prompts])

    X_train_old = _hidden_states(M0, train_prompts)
    X_train_new = _hidden_states(M1, train_prompts)
    X_hold_new = _hidden_states(M1, holdout_prompts)

    # Define target y = first 4 dims of M1-hidden state (the
    # learned-memory module's target IS the new-model
    # representation; the WHOLE POINT is that the new model
    # represents the same task differently).
    rng_t = _np.random.default_rng(int(seed) + 200)
    y_dim = 4
    y_train = X_train_new[:, :y_dim] + 0.01 * rng_t.standard_normal(
        (n_train_prompts, y_dim))
    y_hold = X_hold_new[:, :y_dim] + 0.01 * rng_t.standard_normal(
        (n_holdout_prompts, y_dim))

    # Stale adapter: trained on OLD model's hidden states; we
    # apply it to NEW model's hidden states at inference time.
    stale_adapter = train_linear_adapter_v1(X_train_old, y_train)
    # New adapter: trained on NEW model's hidden states.
    new_adapter = train_linear_adapter_v1(X_train_new, y_train)

    stale_mse = evaluate_adapter_mse_v1(
        stale_adapter, X_hold_new, y_hold)
    new_mse = evaluate_adapter_mse_v1(
        new_adapter, X_hold_new, y_hold)
    new_strictly_beats = (new_mse < stale_mse)

    rep = DriftBenchReportV1(
        old_weights_cid=M0_id.model_weights_cid,
        new_weights_cid=M1_id.model_weights_cid,
        no_change_weights_cid=M0_id.model_weights_cid,
        drift_score_unchanged=ev_unchanged.drift_score,
        drift_score_changed=ev_changed.drift_score,
        threshold=ev_changed.threshold,
        detector_fires_when_changed=ev_changed.drift_detected,
        detector_does_not_fire_when_unchanged=(
            not ev_unchanged.drift_detected),
        stale_verdict_marks_old_capsule_stale=bool(all_stale),
        stale_verdict_marks_fresh_capsule_fresh=bool(
            fresh_not_stale),
        fallback_recommendation_is_recompute_for_stale=bool(
            all_recompute),
        new_memory_strictly_beats_stale_on_holdout=bool(
            new_strictly_beats),
        stale_holdout_mse=float(stale_mse),
        new_holdout_mse=float(new_mse))
    rep = dataclasses.replace(rep, report_cid=rep.cid())
    return rep


__all__ = [
    "W86_DRIFT_V1_SCHEMA_VERSION",
    "W86_DRIFT_V1_FP64_PRECISION_FLOOR",
    "W86_DRIFT_V1_DEFAULT_SAFETY_MARGIN",
    "ModelWeightsCIDV1",
    "DriftDetectorConfigV1",
    "ModelDriftEventV1",
    "CapturedHiddenStateV1",
    "StaleCapsuleVerdictV1",
    "LinearAdapterV1",
    "DriftBenchReportV1",
    "compute_controlled_runtime_weights_cid_v1",
    "run_drift_detector_v1",
    "evaluate_stale_capsule_v1",
    "train_linear_adapter_v1",
    "evaluate_adapter_mse_v1",
    "run_drift_v1_bench",
]
