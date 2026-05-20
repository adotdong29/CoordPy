"""W86 / P0 #26 — Live training of the W83 composed learned memory.

Closes the empirical gap that W84 left open on #26: a real,
content-addressed training run of
:mod:`coordpy.composed_learned_memory_v1` on *live* hidden states
extracted from a real pretrained transformer (default:
Llama-3.1-8B-Instruct via :mod:`coordpy.transformers_runtime_v1`
in bf16 on CUDA).

Anti-cheat (from issue #26):

* The training set is materialised *fresh* every run from
  (prompt corpus, model weights, layer index, precision tier);
  there is no pickled "live" data. If torch / transformers are
  unavailable, the module raises
  :class:`coordpy.live_hidden_state_dataset_v1.
  LiveTrainingBlockedOnHardwareError` rather than mocking.
* Held-out disjointness is enforced by the dataset capsule
  (prompt-CID sets must be disjoint, see
  :class:`coordpy.live_hidden_state_dataset_v1.
  LiveHiddenStateDatasetCapsuleV1.__post_init__`).
* Both the live-trained module and the synthetic-trained
  baseline are evaluated on the SAME held-out live hidden-state
  set. The synthetic baseline is trained on
  :func:`coordpy.composed_learned_memory_v1.
  build_composed_long_horizon_dataset_v1`-shaped data at the same
  dimensions — same architecture, same optimiser config, same
  seed, same number of steps. The only difference is the
  training distribution.
* The "live training beats synthetic training" claim is decided
  by the empirical strict MSE inequality on the held-out live
  set. It is *not* asserted; it is *reported* honestly in the
  output capsule.

Honest scope (W86):

* ``W86-L-LIVE-CM-TRAIN-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only.
* ``W86-L-LIVE-CM-TRAIN-V1-PROJECTION-CAP`` — Llama-3.1-8B's
  hidden_dim is 4096, far larger than the W83 composed module's
  default dimensions (≤ 16). To make the live task and the
  W83-shaped synthetic task comparable we project the live
  hidden states down via a fixed, deterministic, content-
  addressed random projection ``P : R^{4096} → R^{D}`` (default
  D=8). The projection is part of the dataset CID.
* ``W86-L-LIVE-CM-TRAIN-V1-NEXT-STEP-TASK-CAP`` — V1 trains the
  composed module on the task of predicting the *next token's*
  projected hidden state at the same layer from the current
  position. This is a meaningful sequence-modelling task on real
  transformer internals; richer tasks (cross-layer prediction,
  perplexity-coupled training) are V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.live_composed_memory_training_v1 requires numpy"
    ) from exc

from .composed_learned_memory_v1 import (
    ComposedLearnedMemoryModuleV1,
    build_composed_learned_memory_module_v1,
    build_composed_long_horizon_dataset_v1,
    train_composed_learned_memory_module,
)
from .live_hidden_state_dataset_v1 import (
    LiveHiddenStateDatasetCapsuleV1,
    LiveTrainingBlockedOnHardwareError,
    TrainingTraceWitnessV1,
    build_live_hidden_state_dataset_v1,
)


W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION: str = (
    "coordpy.live_composed_memory_training_v1.v1")

# Default projection dim D for projecting live hidden states
# down to the W83 composed module's working dimensionality.
W86_LIVE_CM_DEFAULT_PROJECTION_DIM: int = 8

# Default composed-module dimensions used for both the live-
# trained and synthetic-trained baselines (must match between
# the two — the load-bearing #26 bar is "same arch + same train
# config; only the train distribution differs").
W86_LIVE_CM_DEFAULT_HIDDEN_DIM: int = 16
W86_LIVE_CM_DEFAULT_MEMORY_DIM: int = 12
W86_LIVE_CM_DEFAULT_K_SLOTS: int = 6
W86_LIVE_CM_DEFAULT_TRAIN_ITERS: int = 70
W86_LIVE_CM_DEFAULT_LEARNING_RATE: float = 0.05
W86_LIVE_CM_DEFAULT_MOMENTUM: float = 0.9
W86_LIVE_CM_DEFAULT_WEIGHT_DECAY: float = 1e-4
W86_LIVE_CM_DEFAULT_SEED: int = 86_026_001
W86_LIVE_CM_DEFAULT_LAYER: int = 12
W86_LIVE_CM_DEFAULT_SEQ_LEN: int = 24


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray") -> str:
    return hashlib.sha256(
        _np.ascontiguousarray(
            arr, dtype=_np.float64).tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class W86HiddenStateProjectionV1:
    """A fixed, deterministic, content-addressed random projection.

    Maps high-dimensional live hidden states (e.g. Llama-3.1-8B's
    R^{4096}) down to the W83 composed module's working
    dimensionality. Seeded so the same model + projection_seed
    always yields the same projection matrix. Hashed into the
    dataset CID so a third party can re-derive the matrix.
    """

    schema: str
    in_dim: int
    out_dim: int
    seed: int
    matrix_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "in_dim": int(self.in_dim),
            "out_dim": int(self.out_dim),
            "seed": int(self.seed),
            "matrix_cid": str(self.matrix_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_hidden_state_projection_v1",
            "projection": self.to_dict()})


def build_hidden_state_projection_v1(
        *,
        in_dim: int, out_dim: int, seed: int,
) -> tuple[W86HiddenStateProjectionV1, "_np.ndarray"]:
    """Build a fixed orthogonal-ish projection matrix.

    Uses ``numpy.random.default_rng(seed)`` so the matrix is
    deterministic and the projection CID is reproducible from
    (in_dim, out_dim, seed) alone.
    """
    rng = _np.random.default_rng(int(seed))
    P = rng.standard_normal(
        (int(in_dim), int(out_dim))).astype(_np.float64)
    # Normalise columns to unit norm.
    norms = _np.linalg.norm(P, axis=0) + 1e-12
    P = P / norms[None, :]
    matrix_cid = _ndarray_cid(P)
    proj = W86HiddenStateProjectionV1(
        schema=W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION,
        in_dim=int(in_dim),
        out_dim=int(out_dim),
        seed=int(seed),
        matrix_cid=str(matrix_cid),
    )
    return proj, P


@dataclasses.dataclass(frozen=True)
class LiveHiddenStateTensorsV1:
    """The materialised hidden-state tensors per prompt.

    Held in memory only; never written to disk except via the
    content-addressed report.
    """

    schema: str
    dataset_cid: str
    projection_cid: str
    layer_index: int
    train_X: tuple["_np.ndarray", ...]
    train_Y: tuple["_np.ndarray", ...]
    eval_X: tuple["_np.ndarray", ...]
    eval_Y: tuple["_np.ndarray", ...]
    n_prompts_train: int
    n_prompts_eval: int
    seq_len: int


def materialise_live_hidden_state_tensors_v1(
        *,
        capsule: LiveHiddenStateDatasetCapsuleV1,
        projection_matrix: "_np.ndarray",
        prompts_train: Sequence[str],
        prompts_eval: Sequence[str],
        model_name: str,
        device: str,
        precision_tier: str,
        seq_len: int,
        runtime: Any = None,
) -> LiveHiddenStateTensorsV1:
    """Build (X, Y) sequences from live model hidden states.

    For each prompt:
    1. Tokenize to ``seq_len`` tokens (truncated; padded if too
       short — padded prompts are dropped honestly).
    2. Run model forward, capture layer ``capsule.layer_index``
       hidden state (T, hidden_dim).
    3. Project to (T, D) via the fixed projection matrix.
    4. Build X = projected[:-1], Y = projected[1:] — the next-
       step forecasting task.

    Anti-cheat: never mock the forward call; if torch is
    unavailable, raises
    :class:`LiveTrainingBlockedOnHardwareError`.
    """
    if runtime is None:
        try:
            from .transformers_runtime_v1 import (
                TransformersRuntimeV1,
            )
            runtime = TransformersRuntimeV1(
                model_name=str(model_name),
                device=str(device),
                precision_tier=str(precision_tier),
            )
        except Exception as exc:  # noqa: BLE001
            raise LiveTrainingBlockedOnHardwareError(
                "transformers / torch not available — cannot "
                "materialise live hidden states; "
                f"reason={type(exc).__name__}: {exc}") from exc

    P = _np.asarray(projection_matrix, dtype=_np.float64)
    L = int(capsule.layer_index)

    def _seqs_for(prompts: Sequence[str]) -> tuple[
            list["_np.ndarray"], list["_np.ndarray"]]:
        Xs: list["_np.ndarray"] = []
        Ys: list["_np.ndarray"] = []
        for prompt in prompts:
            ids = runtime.tokenize(
                str(prompt), max_len=int(seq_len))
            if len(ids) < 4:
                continue  # honestly drop too-short prompts
            trace = runtime.forward(input_token_ids=ids)
            if trace.hidden is None or len(
                    trace.hidden.per_layer) <= L:
                continue
            h = _np.asarray(
                trace.hidden.per_layer[L], dtype=_np.float64)
            # h shape: (T, hidden_dim).
            proj = h @ P  # (T, D)
            X = proj[:-1]
            Y = proj[1:]
            Xs.append(X.astype(_np.float64))
            Ys.append(Y.astype(_np.float64))
        return Xs, Ys

    train_X, train_Y = _seqs_for(list(prompts_train))
    eval_X, eval_Y = _seqs_for(list(prompts_eval))

    # Effective seq_len varies across prompts (tokenizer);
    # report the median.
    all_lens = [int(x.shape[0]) for x in train_X + eval_X]
    eff_len = int(_np.median(all_lens)) if all_lens else 0
    proj_cid = _ndarray_cid(P)
    return LiveHiddenStateTensorsV1(
        schema=W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION,
        dataset_cid=str(capsule.cid()),
        projection_cid=str(proj_cid),
        layer_index=int(capsule.layer_index),
        train_X=tuple(train_X),
        train_Y=tuple(train_Y),
        eval_X=tuple(eval_X),
        eval_Y=tuple(eval_Y),
        n_prompts_train=int(len(train_X)),
        n_prompts_eval=int(len(eval_X)),
        seq_len=int(eff_len),
    )


def _mse_module_on_live(
        *,
        module: ComposedLearnedMemoryModuleV1,
        Xs: Sequence["_np.ndarray"],
        Ys: Sequence["_np.ndarray"],
) -> float:
    """Mean MSE of ``module`` on a list of (X, Y) live sequences."""
    if len(Xs) == 0:
        return float("nan")
    total = 0.0
    n_seq = 0
    for X, Y in zip(Xs, Ys):
        _H, _S_seq, _R, Y_hat = module.forward_sequence(
            _np.asarray(X, dtype=_np.float64))
        d = _np.asarray(Y, dtype=_np.float64) - Y_hat
        total += float(_np.mean(d * d))
        n_seq += 1
    return float(total / max(1, n_seq))


@dataclasses.dataclass(frozen=True)
class LiveComposedMemoryTrainReportV1:
    schema: str
    transformers_available: bool
    model_name: str
    precision_tier: str
    device: str
    layer_index: int
    projection: dict[str, Any]
    dataset_cid: str
    n_prompts_train: int
    n_prompts_eval: int
    seq_len_median: int
    composed_module_arch: dict[str, int]
    train_iters: int
    seed: int
    # Live training run.
    live_module_cid_pre: str
    live_module_cid_post: str
    live_pre_loss: float
    live_post_loss: float
    live_mse_on_holdout_live: float
    # Synthetic baseline.
    synthetic_module_cid_pre: str
    synthetic_module_cid_post: str
    synthetic_pre_loss: float
    synthetic_post_loss: float
    synthetic_mse_on_holdout_live: float
    # Verdict bool — the load-bearing #26 strict-beat claim.
    live_strictly_beats_synthetic_on_holdout: bool
    # Wall-clock.
    wall_seconds_live_train: float
    wall_seconds_synthetic_train: float
    wall_seconds_materialise: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "transformers_available": bool(
                self.transformers_available),
            "model_name": str(self.model_name),
            "precision_tier": str(self.precision_tier),
            "device": str(self.device),
            "layer_index": int(self.layer_index),
            "projection": dict(self.projection),
            "dataset_cid": str(self.dataset_cid),
            "n_prompts_train": int(self.n_prompts_train),
            "n_prompts_eval": int(self.n_prompts_eval),
            "seq_len_median": int(self.seq_len_median),
            "composed_module_arch": dict(
                self.composed_module_arch),
            "train_iters": int(self.train_iters),
            "seed": int(self.seed),
            "live_module_cid_pre": str(
                self.live_module_cid_pre),
            "live_module_cid_post": str(
                self.live_module_cid_post),
            "live_pre_loss": float(round(
                self.live_pre_loss, 12)),
            "live_post_loss": float(round(
                self.live_post_loss, 12)),
            "live_mse_on_holdout_live": float(round(
                self.live_mse_on_holdout_live, 12)),
            "synthetic_module_cid_pre": str(
                self.synthetic_module_cid_pre),
            "synthetic_module_cid_post": str(
                self.synthetic_module_cid_post),
            "synthetic_pre_loss": float(round(
                self.synthetic_pre_loss, 12)),
            "synthetic_post_loss": float(round(
                self.synthetic_post_loss, 12)),
            "synthetic_mse_on_holdout_live": float(round(
                self.synthetic_mse_on_holdout_live, 12)),
            "live_strictly_beats_synthetic_on_holdout": bool(
                self.live_strictly_beats_synthetic_on_holdout),
            "wall_seconds_live_train": float(round(
                self.wall_seconds_live_train, 6)),
            "wall_seconds_synthetic_train": float(round(
                self.wall_seconds_synthetic_train, 6)),
            "wall_seconds_materialise": float(round(
                self.wall_seconds_materialise, 6)),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_live_composed_memory_train_report_v1",
            "report": self.to_dict()})


def train_composed_learned_memory_on_live_hidden_states_v1(
        *,
        prompts_train: Sequence[str],
        prompts_eval: Sequence[str],
        model_name: str,
        device: str = "cuda:0",
        precision_tier: str = "tier_bf16",
        layer_index: int = W86_LIVE_CM_DEFAULT_LAYER,
        projection_dim: int = (
            W86_LIVE_CM_DEFAULT_PROJECTION_DIM),
        projection_seed: int = (
            W86_LIVE_CM_DEFAULT_SEED + 1),
        hidden_dim: int = W86_LIVE_CM_DEFAULT_HIDDEN_DIM,
        memory_dim: int = W86_LIVE_CM_DEFAULT_MEMORY_DIM,
        K_slots: int = W86_LIVE_CM_DEFAULT_K_SLOTS,
        n_train_iters: int = W86_LIVE_CM_DEFAULT_TRAIN_ITERS,
        learning_rate: float = (
            W86_LIVE_CM_DEFAULT_LEARNING_RATE),
        momentum: float = W86_LIVE_CM_DEFAULT_MOMENTUM,
        weight_decay: float = (
            W86_LIVE_CM_DEFAULT_WEIGHT_DECAY),
        seed: int = W86_LIVE_CM_DEFAULT_SEED,
        seq_len: int = W86_LIVE_CM_DEFAULT_SEQ_LEN,
        runtime: Any = None,
) -> tuple[
        LiveComposedMemoryTrainReportV1,
        TrainingTraceWitnessV1,
        TrainingTraceWitnessV1]:
    """End-to-end live-vs-synthetic head-to-head training.

    Returns
    -------
    (report, live_witness, synthetic_witness)

    Raises
    ------
    LiveTrainingBlockedOnHardwareError
        If torch + transformers cannot run a real forward (live
        training cannot be done honestly).
    """
    capsule = build_live_hidden_state_dataset_v1(
        prompts_train=list(prompts_train),
        prompts_eval=list(prompts_eval),
        model_name=str(model_name),
        layer_index=int(layer_index),
        precision_tier=str(precision_tier),
    )
    # Materialise live hidden states.
    # We need to know the model's hidden_dim to build the
    # projection. We do this by instantiating the runtime first
    # (cheap once weights are cached).
    if runtime is None:
        try:
            from .transformers_runtime_v1 import (
                TransformersRuntimeV1,
            )
            runtime = TransformersRuntimeV1(
                model_name=str(model_name),
                device=str(device),
                precision_tier=str(precision_tier),
            )
        except Exception as exc:  # noqa: BLE001
            raise LiveTrainingBlockedOnHardwareError(
                "transformers / torch not available — cannot "
                "materialise live hidden states; "
                f"reason={type(exc).__name__}: {exc}") from exc
    in_dim = int(runtime.hidden_dim)
    projection_record, P = build_hidden_state_projection_v1(
        in_dim=int(in_dim),
        out_dim=int(projection_dim),
        seed=int(projection_seed),
    )

    t_mat_0 = time.time()
    tensors = materialise_live_hidden_state_tensors_v1(
        capsule=capsule,
        projection_matrix=P,
        prompts_train=list(prompts_train),
        prompts_eval=list(prompts_eval),
        model_name=str(model_name),
        device=str(device),
        precision_tier=str(precision_tier),
        seq_len=int(seq_len),
        runtime=runtime,
    )
    t_mat = float(time.time() - t_mat_0)

    if tensors.n_prompts_train == 0 or tensors.n_prompts_eval == 0:
        raise LiveTrainingBlockedOnHardwareError(
            "materialised dataset is empty; nothing to train on")

    # Live training.
    live_module = build_composed_learned_memory_module_v1(
        input_dim=int(projection_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(projection_dim),
        K_slots=int(K_slots),
        seed=int(seed),
    )
    cid_pre_live = str(live_module.cid())
    t0 = time.time()
    live_fitted, live_report = (
        train_composed_learned_memory_module(
            module=live_module,
            train_sequences=list(tensors.train_X),
            train_targets=list(tensors.train_Y),
            n_iters=int(n_train_iters),
            learning_rate=float(learning_rate),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        ))
    t_live = float(time.time() - t0)
    cid_post_live = str(live_fitted.cid())
    live_mse = _mse_module_on_live(
        module=live_fitted,
        Xs=list(tensors.eval_X), Ys=list(tensors.eval_Y))

    # Synthetic baseline — same arch, same train config, same
    # seed; only the training distribution differs. We use the
    # W83 composed long-horizon credit-assignment task with the
    # projection-dim shape. We use a sequence length close to
    # the live seq_len so the synthetic baseline runs through
    # the same recurrence depth.
    eff_seq_len = max(14, int(tensors.seq_len) - 1)
    n_synth = max(8, int(tensors.n_prompts_train))
    X_syn, Y_syn = build_composed_long_horizon_dataset_v1(
        n_sequences=int(n_synth),
        seq_len=int(eff_seq_len),
        input_dim=int(projection_dim),
        output_dim=int(projection_dim),
        seed=int(seed),
    )
    syn_module = build_composed_learned_memory_module_v1(
        input_dim=int(projection_dim),
        hidden_dim=int(hidden_dim),
        memory_dim=int(memory_dim),
        output_dim=int(projection_dim),
        K_slots=int(K_slots),
        seed=int(seed),
    )
    cid_pre_syn = str(syn_module.cid())
    t0 = time.time()
    syn_fitted, syn_report = (
        train_composed_learned_memory_module(
            module=syn_module,
            train_sequences=[
                X_syn[i] for i in range(int(X_syn.shape[0]))],
            train_targets=[
                Y_syn[i] for i in range(int(Y_syn.shape[0]))],
            n_iters=int(n_train_iters),
            learning_rate=float(learning_rate),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        ))
    t_syn = float(time.time() - t0)
    cid_post_syn = str(syn_fitted.cid())
    syn_mse = _mse_module_on_live(
        module=syn_fitted,
        Xs=list(tensors.eval_X), Ys=list(tensors.eval_Y))

    strict_beat = bool(
        (live_mse == live_mse)  # not NaN
        and (syn_mse == syn_mse)
        and float(live_mse) < float(syn_mse))

    arch = {
        "input_dim": int(projection_dim),
        "hidden_dim": int(hidden_dim),
        "memory_dim": int(memory_dim),
        "output_dim": int(projection_dim),
        "K_slots": int(K_slots),
    }

    report = LiveComposedMemoryTrainReportV1(
        schema=W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION,
        transformers_available=True,
        model_name=str(model_name),
        precision_tier=str(precision_tier),
        device=str(device),
        layer_index=int(layer_index),
        projection=projection_record.to_dict(),
        dataset_cid=str(capsule.cid()),
        n_prompts_train=int(tensors.n_prompts_train),
        n_prompts_eval=int(tensors.n_prompts_eval),
        seq_len_median=int(tensors.seq_len),
        composed_module_arch=dict(arch),
        train_iters=int(n_train_iters),
        seed=int(seed),
        live_module_cid_pre=str(cid_pre_live),
        live_module_cid_post=str(cid_post_live),
        live_pre_loss=float(live_report.pre_loss),
        live_post_loss=float(live_report.post_loss),
        live_mse_on_holdout_live=float(live_mse),
        synthetic_module_cid_pre=str(cid_pre_syn),
        synthetic_module_cid_post=str(cid_post_syn),
        synthetic_pre_loss=float(syn_report.pre_loss),
        synthetic_post_loss=float(syn_report.post_loss),
        synthetic_mse_on_holdout_live=float(syn_mse),
        live_strictly_beats_synthetic_on_holdout=bool(
            strict_beat),
        wall_seconds_live_train=float(t_live),
        wall_seconds_synthetic_train=float(t_syn),
        wall_seconds_materialise=float(t_mat),
        detail=(
            f"live={tensors.n_prompts_train} train / "
            f"{tensors.n_prompts_eval} eval prompts; "
            f"layer={layer_index}; D_proj={projection_dim}; "
            f"strict_beat={strict_beat}"
        ),
    )

    live_witness = TrainingTraceWitnessV1(
        schema=W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION,
        seed=int(seed),
        optimiser_config_cid=_sha256_hex({
            "kind": "w86_optimiser_config_v1",
            "n_iters": int(n_train_iters),
            "learning_rate": float(learning_rate),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
        }),
        loss_curve_cid=str(live_report.loss_curve_cid),
        fitted_module_cid=str(cid_post_live),
        dataset_cid=str(capsule.cid()),
    )
    synthetic_witness = TrainingTraceWitnessV1(
        schema=W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION,
        seed=int(seed),
        optimiser_config_cid=_sha256_hex({
            "kind": "w86_optimiser_config_v1",
            "n_iters": int(n_train_iters),
            "learning_rate": float(learning_rate),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
        }),
        loss_curve_cid=str(syn_report.loss_curve_cid),
        fitted_module_cid=str(cid_post_syn),
        dataset_cid=_sha256_hex({
            "kind": "w83_synthetic_baseline_dataset_marker",
            "n_synth": int(n_synth),
            "seq_len": int(eff_seq_len),
            "projection_dim": int(projection_dim),
            "seed": int(seed),
        }),
    )
    return report, live_witness, synthetic_witness


__all__ = [
    "W86_LIVE_CM_TRAIN_V1_SCHEMA_VERSION",
    "W86_LIVE_CM_DEFAULT_PROJECTION_DIM",
    "W86_LIVE_CM_DEFAULT_HIDDEN_DIM",
    "W86_LIVE_CM_DEFAULT_MEMORY_DIM",
    "W86_LIVE_CM_DEFAULT_K_SLOTS",
    "W86_LIVE_CM_DEFAULT_TRAIN_ITERS",
    "W86_LIVE_CM_DEFAULT_LEARNING_RATE",
    "W86_LIVE_CM_DEFAULT_MOMENTUM",
    "W86_LIVE_CM_DEFAULT_WEIGHT_DECAY",
    "W86_LIVE_CM_DEFAULT_SEED",
    "W86_LIVE_CM_DEFAULT_LAYER",
    "W86_LIVE_CM_DEFAULT_SEQ_LEN",
    "W86HiddenStateProjectionV1",
    "LiveHiddenStateTensorsV1",
    "LiveComposedMemoryTrainReportV1",
    "build_hidden_state_projection_v1",
    "materialise_live_hidden_state_tensors_v1",
    "train_composed_learned_memory_on_live_hidden_states_v1",
]
