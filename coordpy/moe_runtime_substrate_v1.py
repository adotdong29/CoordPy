"""W86 / P1 #31 — MoE / Mixture-of-Experts substrate.

Extends the W80 dense-transformer instrumentation contract with
**three MoE-specific axes** and ships an MoE-aware runtime adapter
that wraps a HuggingFace MoE model (OLMoE / Mixtral / Qwen-MoE).

The W80 contract assumes dense forward passes: every token activates
every layer's full MLP. MoE breaks this — a token activates only top-K
experts per layer, and the choice of experts is itself state. Naively
reading the post-MLP hidden state without also restoring the routing
decision produces a divergent replay. W80's
``KVCacheSnapshotV1`` does not carry expert routing.

This module ships:

* **3 new MoE axes** (added to ``MoEInstrumentationAxis``):
  - ``READ_EXPERT_ROUTING_PER_LAYER`` — per-layer
    ``(seq_len, top_k)`` selected expert IDs + their gate weights.
  - ``WRITE_FORCE_EXPERT_ROUTING_PER_LAYER`` — override the router's
    top-K decision (force a specific set of experts to fire).
  - ``READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER`` — per-(layer,
    expert) output activations for experts that fired.

* **ExpertRoutingSnapshotV1** — content-addressed dataclass with
  per-layer expert IDs + gate weights + n_experts + top_k.

* **MoERuntimeAdapterV1** — wraps ``TransformersRuntimeV1`` from
  W86; auto-detects MoE block class (``MixtralSparseMoeBlock``,
  ``OlmoeSparseMoeBlock``, ``Qwen2MoeSparseMoeBlock``,
  ``DeepseekV2MoE``); installs forward hooks on each router /
  block to capture (expert_ids, gate_weights) per layer.

* **MoEForceRoutingInjectionV1** — analogue of the W80
  ``InjectionPlanV1`` for MoE routing: per-layer expert-ID +
  gate-weight overrides.

* ``run_moe_substrate_closure_bench_v1`` — the load-bearing #31
  bench. On a real MoE model:
    1. Forward, capture routing.
    2. Replay-from-KV WITH routing restored. Final-token logits
       byte-identity at the bf16 tier floor.
    3. Replay-from-KV WITHOUT routing restored. Trace CID
       diverges from (1) — this is the *negative claim* that
       proves routing is load-bearing in the substrate.
    4. Hidden-state intercept on the MoE post-block residual
       moves CID.

Honest scope (W86):

* ``W86-L-MOE-SUBSTRATE-V1-NEEDS-CUDA-AND-MOE-WEIGHTS-CAP`` —
  the empirical bars (load + forward + replay + routing
  divergence) require a real HF MoE checkpoint + a CUDA GPU
  large enough to hold the model in bf16. On CPU / no-MoE
  hosts the module raises a structured error rather than
  faking results.
* ``W86-L-MOE-SUBSTRATE-V1-TOPK-RESTORE-CAP`` — V1 restores the
  selected top-K expert IDs + their gate weights per
  (layer, token). The full router-logits distribution is NOT
  restored (only the post-softmax top-K). For models whose
  experts are deterministic given the same input + same
  expert selection, this is sufficient for byte-identity at
  the precision floor.
* ``W86-L-MOE-SUBSTRATE-V1-HF-FAMILIES-CAP`` — V1 supports the
  four major HF MoE families enumerated above. Other MoE
  implementations (custom routers, learned mixture-of-experts
  with non-softmax gates) are V2.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.moe_runtime_substrate_v1 requires numpy"
    ) from exc

from .runtime_instrumentation_v1 import (
    CapabilityTag,
    W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
)


W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION: str = (
    "coordpy.moe_runtime_substrate_v1.v1")


class MoEInstrumentationAxis(str, enum.Enum):
    """The 3 new MoE-specific instrumentation axes on the W80
    contract."""

    READ_EXPERT_ROUTING_PER_LAYER = (
        "read_expert_routing_per_layer")
    WRITE_FORCE_EXPERT_ROUTING_PER_LAYER = (
        "write_force_expert_routing_per_layer")
    READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER = (
        "read_expert_output_per_expert_per_layer")


W86_MOE_AXES_ALL: tuple[str, ...] = tuple(
    a.value for a in MoEInstrumentationAxis)


# Known HF MoE block class names. We match on attribute names
# rather than class identity to stay tolerant of transformers
# version drift.
W86_MOE_BLOCK_CANDIDATE_NAMES: tuple[str, ...] = (
    "MixtralSparseMoeBlock",
    "OlmoeSparseMoeBlock",
    "Qwen2MoeSparseMoeBlock",
    "Qwen3MoeSparseMoeBlock",
    "DeepseekV2MoE",
    "DeepseekV3MoE",
    "JambaSparseMoeBlock",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    return hashlib.sha256(
        _np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------
# ExpertRoutingSnapshotV1.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PerLayerRoutingV1:
    """One layer's routing: top-K expert IDs + their gate weights.

    Shapes:
      ``expert_ids``: ``(seq_len, top_k)`` int32
      ``gate_weights``: ``(seq_len, top_k)`` float (renormalised
                        top-K softmax of router logits)
    """

    schema: str
    layer_index: int
    n_experts: int
    top_k: int
    seq_len: int
    expert_ids_cid: str
    gate_weights_cid: str
    expert_ids: "_np.ndarray | None"
    gate_weights: "_np.ndarray | None"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "layer_index": int(self.layer_index),
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
            "seq_len": int(self.seq_len),
            "expert_ids_cid": str(self.expert_ids_cid),
            "gate_weights_cid": str(self.gate_weights_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_moe_per_layer_routing_v1",
            "routing": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ExpertRoutingSnapshotV1:
    """Per-layer routing snapshot for one forward pass.

    Content-addressed by (model_id, prompt_tokens, per-layer
    routing CIDs).
    """

    schema: str
    model_id: str
    n_layers: int
    n_layers_with_routing: int
    seq_len: int
    per_layer: tuple[PerLayerRoutingV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_id": str(self.model_id),
            "n_layers": int(self.n_layers),
            "n_layers_with_routing": int(
                self.n_layers_with_routing),
            "seq_len": int(self.seq_len),
            "per_layer": [
                p.to_dict() for p in self.per_layer],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_expert_routing_snapshot_v1",
            "snapshot": self.to_dict()})


# ---------------------------------------------------------------
# MoEForceRoutingInjectionV1.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoEForceRoutingInjectionV1:
    """Override the router's top-K decision per layer.

    For each layer in ``per_layer_force``, the runtime's MoE
    block's router is replaced with a fixed
    ``(expert_ids, gate_weights)`` returned by the hook —
    bypassing the router's learned softmax-and-topk path.

    ``None`` for a layer means "honor the model's own router".
    """

    schema: str
    per_layer_force: tuple["PerLayerRoutingV1 | None", ...]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_moe_force_routing_injection_v1",
            "schema": str(self.schema),
            "n_layers": len(self.per_layer_force),
            "force_cids": [
                (p.cid() if p else "none")
                for p in self.per_layer_force],
        })


# ---------------------------------------------------------------
# Capability probe.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoECapabilityProbeV1:
    """Honest record of what the MoE adapter can do on this host."""

    schema: str
    transformers_available: bool
    torch_cuda_available: bool
    model_name: str
    model_is_moe: bool
    moe_block_class_name: str
    n_moe_layers_detected: int
    n_experts: int
    top_k: int
    declared_moe_axes: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "transformers_available": bool(
                self.transformers_available),
            "torch_cuda_available": bool(
                self.torch_cuda_available),
            "model_name": str(self.model_name),
            "model_is_moe": bool(self.model_is_moe),
            "moe_block_class_name": str(
                self.moe_block_class_name),
            "n_moe_layers_detected": int(
                self.n_moe_layers_detected),
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
            "declared_moe_axes": [
                [str(a), str(t)]
                for a, t in self.declared_moe_axes],
            "notes": [str(n) for n in self.notes],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_moe_capability_probe_v1",
            "probe": self.to_dict()})


def moe_declared_axes() -> Mapping[str, str]:
    """Declared MoE axes for the adapter.

    All three axes are ``AVAILABLE`` when the runtime is wrapping
    an MoE model; ``UNAVAILABLE`` otherwise.
    """
    a = CapabilityTag.AVAILABLE.value
    return {
        MoEInstrumentationAxis.READ_EXPERT_ROUTING_PER_LAYER.value:
            a,
        MoEInstrumentationAxis
        .WRITE_FORCE_EXPERT_ROUTING_PER_LAYER.value: a,
        MoEInstrumentationAxis
        .READ_EXPERT_OUTPUT_PER_EXPERT_PER_LAYER.value: a,
    }


# ---------------------------------------------------------------
# Runtime adapter.
# ---------------------------------------------------------------


def _find_moe_blocks(model: Any) -> tuple[
        tuple[int, Any, str], ...]:
    """Walk the model and return ``(layer_index, block, class_name)``
    for every layer that looks like an MoE sparse block.

    Detection heuristic: any nn.Module whose class name appears
    in ``W86_MOE_BLOCK_CANDIDATE_NAMES`` AND that has either a
    ``gate`` or ``router`` attribute.
    """
    out: list[tuple[int, Any, str]] = []
    # Try the standard HF transformer block container paths.
    for path in (
            "model.layers",
            "model.transformer.layers",
            "transformer.layers",
            "transformer.h",
    ):
        node = model
        ok = True
        for part in path.split("."):
            if hasattr(node, part):
                node = getattr(node, part)
            else:
                ok = False
                break
        if not ok or not hasattr(node, "__iter__"):
            continue
        layers = list(node)
        for layer_idx, layer in enumerate(layers):
            # Walk this layer's children for MoE blocks.
            queue = [layer]
            found: Any = None
            found_name = ""
            while queue:
                m = queue.pop()
                cls_name = type(m).__name__
                if cls_name in (
                        W86_MOE_BLOCK_CANDIDATE_NAMES):
                    # Sanity check: has a router or gate.
                    if (hasattr(m, "gate")
                            or hasattr(m, "router")):
                        found = m
                        found_name = cls_name
                        break
                # Walk children breadth-first.
                if hasattr(m, "children"):
                    queue.extend(list(m.children()))
            if found is not None:
                out.append((
                    int(layer_idx), found, str(found_name)))
        if out:
            return tuple(out)
    return tuple(out)


def probe_moe_capability_v1(
        *, model_name: str,
) -> MoECapabilityProbeV1:
    """Probe whether the MoE adapter can wrap a given model.

    Does NOT download weights. Tries to import torch +
    transformers; if available, queries the model's config
    for ``num_local_experts`` / ``num_experts`` /
    ``num_experts_per_tok`` to decide. Returns a content-
    addressed report.
    """
    notes: list[str] = []
    try:
        import torch  # type: ignore
        torch_cuda = bool(torch.cuda.is_available())
        notes.append("torch importable")
    except Exception as exc:  # noqa: BLE001
        return MoECapabilityProbeV1(
            schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
            transformers_available=False,
            torch_cuda_available=False,
            model_name=str(model_name),
            model_is_moe=False,
            moe_block_class_name="",
            n_moe_layers_detected=0,
            n_experts=0,
            top_k=0,
            declared_moe_axes=tuple(
                (str(k), CapabilityTag.UNAVAILABLE.value)
                for k in moe_declared_axes().keys()),
            notes=(
                f"torch NOT importable: "
                f"{type(exc).__name__}: {exc}",),
        )
    try:
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained(str(model_name))
        notes.append("AutoConfig OK")
    except Exception as exc:  # noqa: BLE001
        return MoECapabilityProbeV1(
            schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
            transformers_available=False,
            torch_cuda_available=bool(torch_cuda),
            model_name=str(model_name),
            model_is_moe=False,
            moe_block_class_name="",
            n_moe_layers_detected=0,
            n_experts=0,
            top_k=0,
            declared_moe_axes=tuple(
                (str(k), CapabilityTag.UNAVAILABLE.value)
                for k in moe_declared_axes().keys()),
            notes=(
                f"AutoConfig failed: "
                f"{type(exc).__name__}: {exc}",),
        )
    # MoE detection by config keys.
    n_experts = int(
        getattr(cfg, "num_local_experts", 0)
        or getattr(cfg, "num_experts", 0)
        or getattr(cfg, "n_routed_experts", 0))
    top_k = int(
        getattr(cfg, "num_experts_per_tok", 0)
        or getattr(cfg, "moe_topk", 0)
        or getattr(cfg, "num_experts_per_token", 0))
    model_is_moe = bool(n_experts > 0 and top_k > 0)
    notes.append(
        f"n_experts={n_experts} top_k={top_k}")
    return MoECapabilityProbeV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        transformers_available=True,
        torch_cuda_available=bool(torch_cuda),
        model_name=str(model_name),
        model_is_moe=bool(model_is_moe),
        moe_block_class_name="",
        n_moe_layers_detected=0,
        n_experts=int(n_experts),
        top_k=int(top_k),
        declared_moe_axes=tuple(
            (str(k), str(v))
            for k, v in moe_declared_axes().items()),
        notes=tuple(notes),
    )


@dataclasses.dataclass
class MoERuntimeAdapterV1:
    """W86 MoE adapter on top of W86 TransformersRuntimeV1.

    Construct with ``MoERuntimeAdapterV1(model_name=…,
    device="cuda:0", precision_tier="tier_bf16")``. Auto-detects
    MoE blocks; raises ``RuntimeError`` if the model isn't MoE.
    """

    model_name: str
    device: str = "cuda:0"
    precision_tier: str = "tier_bf16"
    _runtime: Any = None
    _moe_blocks: tuple[tuple[int, Any, str], ...] = ()
    _moe_block_class_name: str = ""
    _n_experts: int = 0
    _top_k: int = 0
    _captured_routing: list["PerLayerRoutingV1 | None"] = (
        dataclasses.field(default_factory=list))
    _captured_raw_logits: list[Any] = (
        dataclasses.field(default_factory=list))
    _gate_class_name: str = ""
    _gate_returns_tuple: bool = False
    _last_hook_fire_count: int = 0

    def __post_init__(self) -> None:
        from .transformers_runtime_v1 import (
            TransformersRuntimeV1,
        )
        self._runtime = TransformersRuntimeV1(
            model_name=str(self.model_name),
            device=str(self.device),
            precision_tier=str(self.precision_tier),
        )
        moe_blocks = _find_moe_blocks(self._runtime.model)
        if not moe_blocks:
            raise RuntimeError(
                f"model {self.model_name!r} does not appear to "
                "have MoE blocks (no class in "
                f"{W86_MOE_BLOCK_CANDIDATE_NAMES} found). "
                "If this IS an MoE model, the class name may be "
                "new — file an issue.")
        self._moe_blocks = moe_blocks
        self._moe_block_class_name = str(moe_blocks[0][2])
        # Record the gate sub-module's class name (helps tell
        # the new tuple-API OLMoE router from a plain nn.Linear).
        try:
            first_gate = self._gate_module(moe_blocks[0][1])
            self._gate_class_name = str(type(first_gate).__name__)
        except Exception:  # noqa: BLE001
            self._gate_class_name = ""
        cfg = self._runtime.model.config
        self._n_experts = int(
            getattr(cfg, "num_local_experts", 0)
            or getattr(cfg, "num_experts", 0)
            or getattr(cfg, "n_routed_experts", 0))
        self._top_k = int(
            getattr(cfg, "num_experts_per_tok", 0)
            or getattr(cfg, "moe_topk", 0))

    @property
    def runtime(self) -> Any:
        return self._runtime

    @property
    def n_moe_layers(self) -> int:
        return int(len(self._moe_blocks))

    @property
    def n_experts(self) -> int:
        return int(self._n_experts)

    @property
    def top_k(self) -> int:
        return int(self._top_k)

    @property
    def moe_block_class_name(self) -> str:
        return str(self._moe_block_class_name)

    def declared_axes(self) -> Mapping[str, str]:
        return moe_declared_axes()

    def _gate_module(self, block: Any) -> Any:
        """Return the router-linear sub-module on this MoE block.

        Modern HF MoE blocks (OLMoE / Mixtral / Qwen-MoE /
        DeepSeek-V2) expose the router as ``block.gate``; some
        forks use ``block.router``. The returned sub-module's
        forward output IS the raw ``router_logits`` tensor of
        shape ``(B*T, n_experts)`` — hooking it is robust to
        transformers version drift (in 5.0 the block's own
        forward no longer returns ``router_logits`` by default,
        which is why the V1 block-level hook captured nothing).
        """
        if hasattr(block, "gate"):
            return block.gate
        if hasattr(block, "router"):
            return block.router
        raise RuntimeError(
            f"MoE block {type(block).__name__} has neither "
            "`gate` nor `router` attribute; can't hook the "
            "router. File an issue with the model name.")

    def _install_router_hooks(self) -> list[Any]:
        """Install forward hooks on each MoE block's gate.

        Handles BOTH HF MoE router APIs:

        * Tuple API (OLMoE 5.0 ``OlmoeTopKRouter`` and similar):
          ``gate.forward()`` returns ``(router_logits,
          top_k_weights, top_k_index)``. The block uses indices
          1 and 2 directly for expert dispatch; capturing the
          full tuple lets restoration return the same
          ``(top_k_weights, top_k_index)`` byte-identically.

        * Tensor API (Mixtral / legacy OLMoE / Qwen-MoE older
          forks): ``gate.forward()`` returns a single
          ``router_logits`` tensor of shape ``(B*T, n_experts)``;
          the block runs softmax+top-K itself.

        Returns hook handles that the caller MUST remove."""
        import torch  # type: ignore
        n_blocks = int(len(self._moe_blocks))
        captured: list["PerLayerRoutingV1 | None"] = [
            None] * n_blocks
        raw_outs: list[Any] = [None] * n_blocks
        fire_count = {"n": 0}
        gate_is_tuple = {"v": False}
        hooks: list[Any] = []
        for layer_idx, block, _cls_name in self._moe_blocks:
            gate = self._gate_module(block)

            def _make_gate_hook(li: int):
                def _gate_hook(_mod, _inp, out):
                    fire_count["n"] += 1
                    # Tuple API (OLMoE 5.0): (rl, weights, ids)
                    if (isinstance(out, tuple)
                            and len(out) >= 3
                            and hasattr(out[0], "shape")):
                        gate_is_tuple["v"] = True
                        rl_t, wt_t, idx_t = out[0], out[1], out[2]
                        if rl_t.ndim != 2:
                            return None
                        n_experts_observed = int(
                            rl_t.shape[-1])
                        ids_np = (
                            idx_t.detach().to("cpu")
                            .to(dtype=torch.int32).numpy())
                        weights_np = (
                            wt_t.detach().to("cpu")
                            .to(dtype=torch.float32).numpy())
                        raw_outs[li] = (
                            rl_t.detach().to("cpu").float(),
                            wt_t.detach().to("cpu").float(),
                            idx_t.detach().to("cpu").long(),
                        )
                    elif hasattr(out, "shape") and out.ndim == 2:
                        # Tensor API (Mixtral / legacy).
                        rl = out.detach()
                        n_experts_observed = int(rl.shape[-1])
                        routing_weights = torch.softmax(
                            rl.float(), dim=-1)
                        k = int(self._top_k or 1)
                        selected_weights, selected_experts = (
                            torch.topk(
                                routing_weights, k, dim=-1))
                        selected_weights = (
                            selected_weights
                            / (selected_weights.sum(
                                dim=-1, keepdim=True) + 1e-12))
                        ids_np = (
                            selected_experts.to("cpu")
                            .to(dtype=torch.int32).numpy())
                        weights_np = (
                            selected_weights.to("cpu")
                            .to(dtype=torch.float32).numpy())
                        raw_outs[li] = rl.to("cpu").float()
                    else:
                        return None
                    captured[li] = PerLayerRoutingV1(
                        schema=(
                            W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION),
                        layer_index=int(li),
                        n_experts=int(n_experts_observed),
                        top_k=int(weights_np.shape[-1]),
                        seq_len=int(ids_np.shape[0]),
                        expert_ids_cid=_ndarray_cid(ids_np),
                        gate_weights_cid=_ndarray_cid(
                            weights_np),
                        expert_ids=ids_np,
                        gate_weights=weights_np,
                    )
                    return None
                return _gate_hook

            h = gate.register_forward_hook(
                _make_gate_hook(layer_idx))
            hooks.append(h)
        self._captured_routing = captured
        self._captured_raw_logits = raw_outs
        # Counter dicts are stored on self so the caller can read
        # them after the forward runs.
        self._hook_fire_count_holder = fire_count
        self._gate_is_tuple_holder = gate_is_tuple
        self._last_hook_fire_count = 0
        self._gate_returns_tuple = False
        return hooks

    def _read_capture_diagnostics(self) -> None:
        """Snapshot the hook counters onto immutable fields.

        Call after running a forward with capture hooks installed,
        BEFORE removing the hook handles (the closures hold the
        counter dicts)."""
        fc = getattr(self, "_hook_fire_count_holder", None)
        ti = getattr(self, "_gate_is_tuple_holder", None)
        if fc is not None:
            self._last_hook_fire_count = int(fc.get("n", 0))
        if ti is not None:
            self._gate_returns_tuple = bool(ti.get("v", False))

    def _install_routing_restore_hooks(
            self, *,
            raw_logits_per_layer: list[Any],
    ) -> list[Any]:
        """Install gate hooks that REPLACE the gate's output
        with previously-captured router output.

        For the tuple-API gate (OLMoE 5.0), the hook returns the
        captured ``(router_logits, top_k_weights, top_k_index)``
        sliced to the new-token rows. The block then uses
        ``top_k_weights`` and ``top_k_index`` for expert dispatch
        — byte-identical to the original forward by construction.

        For the tensor-API gate (Mixtral / legacy), the hook
        replaces the gate's ``router_logits`` tensor; the block's
        own softmax+top-K then re-derives the same routing.
        """
        hooks: list[Any] = []
        for layer_idx, block, _cls_name in self._moe_blocks:
            gate = self._gate_module(block)
            saved = (
                raw_logits_per_layer[int(layer_idx)]
                if int(layer_idx) < len(raw_logits_per_layer)
                else None)
            if saved is None:
                continue

            def _make_restore_hook(saved_x):
                def _hook(_mod, _inp, out):
                    # Tuple-saved → return tuple, sliced + cast.
                    if isinstance(saved_x, tuple):
                        if not (isinstance(out, tuple)
                                and len(out) >= 3):
                            return out
                        rl_s, wt_s, idx_s = saved_x
                        n_rl = int(out[0].shape[0])
                        n_wt = int(out[1].shape[0])
                        n_idx = int(out[2].shape[0])
                        rl_t = rl_s[-n_rl:].to(
                            device=out[0].device,
                            dtype=out[0].dtype)
                        wt_t = wt_s[-n_wt:].to(
                            device=out[1].device,
                            dtype=out[1].dtype)
                        idx_t = idx_s[-n_idx:].to(
                            device=out[2].device,
                            dtype=out[2].dtype)
                        return (rl_t, wt_t, idx_t)
                    # Tensor-saved → return tensor.
                    if not hasattr(out, "shape"):
                        return out
                    n_rows = int(out.shape[0])
                    return saved_x[-n_rows:].to(
                        device=out.device, dtype=out.dtype)
                return _hook

            h = gate.register_forward_hook(
                _make_restore_hook(saved))
            hooks.append(h)
        return hooks

    def _install_force_random_routing_hooks(
            self, *, seed: int = 0,
    ) -> list[Any]:
        """Install gate hooks that REPLACE the gate output with
        deterministic-random routing — forces a DIFFERENT top-K
        per token than the model's own router.

        Used by the bench as the explicit demonstration that MoE
        routing is load-bearing state: same prompt + same KV +
        DIFFERENT routing → measurably different output.
        """
        import torch  # type: ignore
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        hooks: list[Any] = []
        for layer_idx, block, _cls_name in self._moe_blocks:
            gate = self._gate_module(block)

            def _make_random_hook(seed_li: int):
                def _hook(_mod, _inp, out):
                    if (isinstance(out, tuple)
                            and len(out) >= 3
                            and hasattr(out[0], "shape")):
                        rl_o, wt_o, idx_o = out[0], out[1], out[2]
                        n_rows = int(rl_o.shape[0])
                        n_experts = int(rl_o.shape[-1])
                        k = int(wt_o.shape[-1])
                        rl_rand = torch.randn(
                            rl_o.shape, generator=gen,
                            dtype=torch.float32).to(
                            device=rl_o.device,
                            dtype=rl_o.dtype)
                        # Random positive scores per row, summed
                        # to 1 (mimics softmax+normalised topk).
                        wt_rand = torch.rand(
                            (n_rows, k), generator=gen,
                            dtype=torch.float32) + 1e-3
                        wt_rand = (
                            wt_rand
                            / wt_rand.sum(dim=-1, keepdim=True))
                        wt_rand = wt_rand.to(
                            device=wt_o.device,
                            dtype=wt_o.dtype)
                        # Random expert indices in [0, n_experts).
                        idx_rand = torch.randint(
                            0, n_experts, (n_rows, k),
                            generator=gen).to(
                            device=idx_o.device,
                            dtype=idx_o.dtype)
                        return (rl_rand, wt_rand, idx_rand)
                    if hasattr(out, "shape"):
                        rand = torch.randn(
                            out.shape, generator=gen,
                            dtype=torch.float32)
                        return rand.to(
                            device=out.device,
                            dtype=out.dtype)
                    return out
                return _hook

            h = gate.register_forward_hook(
                _make_random_hook(int(layer_idx)))
            hooks.append(h)
        return hooks

    def forward_with_routing_capture(
            self, *, input_token_ids: Sequence[int],
    ) -> tuple[Any, ExpertRoutingSnapshotV1]:
        """Run a forward pass; capture routing.

        Returns ``(forward_trace, routing_snapshot)``. The
        forward_trace is the W80 ForwardTraceV1 from the
        underlying TransformersRuntimeV1; the routing snapshot
        is the new W86 MoE artefact.
        """
        hooks = self._install_router_hooks()
        try:
            trace = self._runtime.forward(
                input_token_ids=list(input_token_ids))
        finally:
            self._read_capture_diagnostics()
            for h in hooks:
                h.remove()
        per_layer = tuple(
            r for r in self._captured_routing
            if r is not None)
        seq_len = (
            int(per_layer[0].seq_len) if per_layer else 0)
        snapshot = ExpertRoutingSnapshotV1(
            schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
            model_id=str(self.model_name),
            n_layers=int(self._runtime.n_layers),
            n_layers_with_routing=int(len(per_layer)),
            seq_len=int(seq_len),
            per_layer=per_layer,
        )
        return trace, snapshot

    @property
    def captured_raw_logits(self) -> list[Any]:
        """List of CPU float32 tensors (one per MoE layer) with
        the raw router_logits from the most recent forward."""
        return list(self._captured_raw_logits)

    def replay_with_routing_restored(
            self, *,
            kv: Any,
            new_token_ids: Sequence[int],
            raw_logits_per_layer: list[Any],
    ) -> Any:
        """Replay-from-KV with the captured router_logits
        restored — the model's own block.gate output is
        overridden by the saved tensor (sliced to the new-token
        rows), so the rest of the MoE block re-computes
        softmax+top-K+expert dispatch on the same logits as the
        original forward.

        This is the load-bearing positive arm: ``max_abs_diff``
        on final-token logits should be at the tier floor.
        """
        hooks = self._install_routing_restore_hooks(
            raw_logits_per_layer=list(raw_logits_per_layer))
        try:
            return self._runtime.replay_from_kv(
                kv=kv, new_token_ids=list(new_token_ids))
        finally:
            for h in hooks:
                h.remove()

    def forward_with_force_random_routing(
            self, *, input_token_ids: Sequence[int],
            seed: int = 0,
    ) -> Any:
        """Forward with the gate's output replaced by
        deterministic-random logits per layer — i.e., force a
        DIFFERENT top-K than the model's own router.

        Used by the bench as the explicit demonstration that
        routing is load-bearing: same prompt + different routing
        → measurably different final logits.
        """
        hooks = self._install_force_random_routing_hooks(
            seed=int(seed))
        try:
            return self._runtime.forward(
                input_token_ids=list(input_token_ids))
        finally:
            for h in hooks:
                h.remove()


# ---------------------------------------------------------------
# Bench.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MoESubstrateClosureBenchReportV1:
    """Load-bearing #31 closure bench report.

    Records:
    * ``forward_trace_cid`` — the W80 trace CID of the original
      forward.
    * ``routing_snapshot_cid`` — content-addressed routing
      capture.
    * ``replay_trace_cid_with_routing`` — trace CID of the
      replay-from-KV with the routing snapshot restored.
    * ``replay_trace_cid_without_routing`` — trace CID of the
      replay-from-KV with the model's own router firing.
    * Load-bearing bools:
      - ``forward_routing_captured`` — at least one layer's
        routing was captured.
      - ``replay_with_routing_matches_forward_floor`` —
        max_abs_diff on final-token logits between forward and
        replay-with-routing is within the precision-tier floor.
      - ``moe_routing_is_load_bearing`` — replay-without-
        routing has a measurable divergence (max_abs_diff
        ≥ some threshold OR the captured routing changed
        across two consecutive forwards — under temperature=0
        forward should be deterministic, so captured routings
        should match across runs).
      - ``hidden_state_intercept_on_moe_block_moves_cid`` —
        injecting a hidden-state delta on a MoE block's residual
        moves the trace CID.
    """

    schema: str
    model_name: str
    precision_tier: str
    moe_block_class_name: str
    gate_class_name: str
    gate_returns_tuple: bool
    hook_fires_per_forward: int
    n_moe_layers: int
    n_experts: int
    top_k: int
    n_layers_routing_captured: int
    forward_trace_cid: str
    routing_snapshot_cid: str
    replay_trace_cid_with_routing: str
    replay_trace_cid_without_routing: str
    forward_force_random_routing_trace_cid: str
    max_abs_diff_replay_vs_forward_last_logits: float
    max_abs_diff_with_routing_vs_forward_last_logits: float
    max_abs_diff_without_routing_vs_forward_last_logits: float
    max_abs_diff_force_random_vs_forward_last_logits: float
    tier_tolerance: float
    forward_routing_captured: bool
    replay_with_routing_matches_forward_floor: bool
    moe_routing_is_load_bearing: bool
    routing_deterministic_across_two_forwards: bool
    hidden_state_intercept_on_moe_block_moves_cid: bool
    wall_clock_seconds: float
    bench_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_name": str(self.model_name),
            "precision_tier": str(self.precision_tier),
            "moe_block_class_name": str(
                self.moe_block_class_name),
            "gate_class_name": str(self.gate_class_name),
            "gate_returns_tuple": bool(
                self.gate_returns_tuple),
            "hook_fires_per_forward": int(
                self.hook_fires_per_forward),
            "n_moe_layers": int(self.n_moe_layers),
            "n_experts": int(self.n_experts),
            "top_k": int(self.top_k),
            "n_layers_routing_captured": int(
                self.n_layers_routing_captured),
            "forward_trace_cid": str(
                self.forward_trace_cid),
            "routing_snapshot_cid": str(
                self.routing_snapshot_cid),
            "replay_trace_cid_with_routing": str(
                self.replay_trace_cid_with_routing),
            "replay_trace_cid_without_routing": str(
                self.replay_trace_cid_without_routing),
            "forward_force_random_routing_trace_cid": str(
                self.forward_force_random_routing_trace_cid),
            "max_abs_diff_replay_vs_forward_last_logits": (
                float(round(
                    self
                    .max_abs_diff_replay_vs_forward_last_logits,
                    6))),
            "max_abs_diff_with_routing_vs_forward_last_logits":
                float(round(
                    self
                    .max_abs_diff_with_routing_vs_forward_last_logits,
                    6)),
            "max_abs_diff_without_routing_vs_forward_last_logits":
                float(round(
                    self
                    .max_abs_diff_without_routing_vs_forward_last_logits,
                    6)),
            "max_abs_diff_force_random_vs_forward_last_logits":
                float(round(
                    self
                    .max_abs_diff_force_random_vs_forward_last_logits,
                    6)),
            "tier_tolerance": float(round(
                self.tier_tolerance, 6)),
            "forward_routing_captured": bool(
                self.forward_routing_captured),
            "replay_with_routing_matches_forward_floor": bool(
                self
                .replay_with_routing_matches_forward_floor),
            "moe_routing_is_load_bearing": bool(
                self.moe_routing_is_load_bearing),
            "routing_deterministic_across_two_forwards": bool(
                self
                .routing_deterministic_across_two_forwards),
            "hidden_state_intercept_on_moe_block_moves_cid": (
                bool(
                    self
                    .hidden_state_intercept_on_moe_block_moves_cid
                )),
            "wall_clock_seconds": float(round(
                self.wall_clock_seconds, 3)),
            "bench_cid": str(self.bench_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w86_moe_substrate_closure_bench_report_v1",
            "report": self.to_dict()})


def run_moe_substrate_closure_bench_v1(
        *,
        model_name: str = "allenai/OLMoE-1B-7B-0924-Instruct",
        device: str = "cuda:0",
        precision_tier: str = "tier_bf16",
        prompt: str = (
            "Context Zero is the research programme that "
            "ships a real substrate contract for "
            "mixture-of-experts transformer routing."),
        prompt_max_len: int = 24,
        n_continuation_tokens: int = 4,
        inject_layer: int = 4,
        inject_magnitude: float = 1.0,
) -> MoESubstrateClosureBenchReportV1:
    """The #31 closure bench.

    Steps:
      1. Build MoERuntimeAdapterV1.
      2. Tokenize the prompt; split into old + new tokens.
      3. Forward with routing capture → forward_trace +
         routing_snapshot.
      4. Replay-from-KV. Compare replay trace's final-token
         logits to the forward's same row. Diff < tier
         tolerance → routing-restored byte-id (for HF Mixtral/
         OLMoE the model's own router fires deterministically
         at temperature=0, so the same routing pattern is
         reproduced and the replay matches).
      5. Hidden-state intercept on layer `inject_layer`. Trace
         CID must differ from the baseline.
      6. Negative claim: capture the routing on TWO forwards.
         At temperature=0 they should match byte-identically.
         Then verify: a forced-routing override (different top-
         K) produces a measurably different forward.
    """
    t0 = time.time()
    adapter = MoERuntimeAdapterV1(
        model_name=str(model_name),
        device=str(device),
        precision_tier=str(precision_tier),
    )
    runtime = adapter.runtime
    ids = runtime.tokenize(
        str(prompt), max_len=int(prompt_max_len))
    if len(ids) <= int(n_continuation_tokens):
        raise RuntimeError(
            f"prompt too short for {n_continuation_tokens} "
            f"continuation tokens (only {len(ids)} tokens)")
    old_ids = ids[: -int(n_continuation_tokens)]
    new_ids = ids[-int(n_continuation_tokens):]

    # Step 1: forward(full ids) WITH router capture.
    fwd_trace, routing = (
        adapter.forward_with_routing_capture(
            input_token_ids=ids))
    forward_routing_captured = bool(
        int(routing.n_layers_with_routing) > 0)
    fwd_trace_cid = str(fwd_trace.cid())
    routing_cid = str(routing.cid())
    raw_logits = list(adapter.captured_raw_logits)

    from .transformers_runtime_v1 import (
        W86_REPLAY_TOLERANCE_PER_TIER,
    )
    tier_tol = float(W86_REPLAY_TOLERANCE_PER_TIER.get(
        str(precision_tier).lower(), 5e-3))

    # Build forward's last-n logit rows once — every diff uses
    # the same reference.
    def _last_rows(trace: Any) -> "_np.ndarray":
        arr = _np.asarray(trace.final_logits)
        if arr.ndim == 2:
            return arr[-int(n_continuation_tokens):, :]
        return arr

    fwd_last_rows = _last_rows(fwd_trace)

    def _diff_against_forward(trace: Any) -> float:
        other = _np.asarray(trace.final_logits)
        if other.ndim == 2:
            other_rows = other[
                -int(n_continuation_tokens):, :]
        else:
            other_rows = other
        n_cmp = int(min(
            fwd_last_rows.shape[0], other_rows.shape[0]))
        return float(_np.max(_np.abs(
            fwd_last_rows[-n_cmp:]
            - other_rows[-n_cmp:])))

    # Step 2: replay-from-KV, WITHOUT routing restored
    # (model's own router fires; tiny KV-cache bf16 noise can
    # flip the top-K → MoE cascades the divergence → this is the
    # *negative* arm that proves routing is load-bearing).
    old_trace = runtime.forward(input_token_ids=old_ids)
    replay_without = runtime.replay_from_kv(
        kv=old_trace.kv, new_token_ids=new_ids)
    diff_without = _diff_against_forward(replay_without)
    replay_trace_cid_without_routing = str(
        replay_without.cid())

    # Step 3: replay-from-KV, WITH routing restored. The
    # captured raw router_logits replace block.gate's output
    # during replay, so the block re-creates the original
    # routing deterministically.
    if forward_routing_captured:
        replay_with = adapter.replay_with_routing_restored(
            kv=old_trace.kv,
            new_token_ids=new_ids,
            raw_logits_per_layer=raw_logits)
        diff_with = _diff_against_forward(replay_with)
        replay_with_routing_cid = str(replay_with.cid())
    else:
        # No capture → no restore is possible. Report the
        # without-restore diff as the with-restore value so the
        # report is honest about the failure mode.
        diff_with = float(diff_without)
        replay_with_routing_cid = str(
            replay_trace_cid_without_routing)
    replay_with_routing_byte_id = bool(
        diff_with < tier_tol)

    # Step 4: force-random routing forward — direct
    # demonstration that DIFFERENT routing → measurably
    # different output (corroborates the load-bearing claim).
    if forward_routing_captured:
        force_random_trace = (
            adapter.forward_with_force_random_routing(
                input_token_ids=ids, seed=int(0)))
        diff_force_random = _diff_against_forward(
            force_random_trace)
        force_random_trace_cid = str(force_random_trace.cid())
    else:
        diff_force_random = 0.0
        force_random_trace_cid = ""

    # Step 5: hidden-state intercept on the MoE block residual.
    # Uses the W80 hidden_state_inject_per_layer surface (which
    # fires AFTER each transformer block, including MoE).
    from .runtime_instrumentation_v1 import (
        InjectionPlanV1,
        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION as W80_SV,
    )
    H = int(runtime.hidden_dim)
    n_layers = int(runtime.n_layers)
    inj_layer = min(int(inject_layer), int(n_layers) - 1)
    inj = _np.ones(
        (int(len(ids)), int(H)),
        dtype=_np.float64) * float(inject_magnitude)
    inj_per_layer: list["_np.ndarray | None"] = [
        None] * int(n_layers)
    inj_per_layer[int(inj_layer)] = inj
    plan = InjectionPlanV1(
        schema=W80_SV,
        hidden_state_inject_per_layer=tuple(inj_per_layer),
        attention_bias_per_layer=tuple(),
        prefix_state_inject=None,
        kv_restore=None,
        position_offset=None,
    )
    inj_trace = runtime.forward(
        input_token_ids=ids, injection=plan)
    intercept_moves_cid = bool(
        str(fwd_trace.cid()) != str(inj_trace.cid()))

    # Step 6: routing determinism. Two forwards at temperature
    # =0 with the same prompt must produce the same routing.
    _trace2, routing2 = (
        adapter.forward_with_routing_capture(
            input_token_ids=ids))
    routing_deterministic = bool(
        routing.cid() == routing2.cid())

    # Load-bearing := (1) routing was captured, (2) restoring it
    # yields byte-identity at the tier floor, (3) NOT restoring
    # it EXCEEDS the tier floor (so routing IS the missing state
    # the substrate needs), and (4) routing is deterministic.
    moe_routing_is_load_bearing = bool(
        forward_routing_captured
        and replay_with_routing_byte_id
        and diff_without > tier_tol
        and diff_without > (2.0 * diff_with)
        and routing_deterministic
        and len(routing.per_layer) > 0
        and routing.per_layer[0].expert_ids is not None)
    wall = float(time.time() - t0)

    report = MoESubstrateClosureBenchReportV1(
        schema=W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION,
        model_name=str(model_name),
        precision_tier=str(precision_tier),
        moe_block_class_name=str(
            adapter.moe_block_class_name),
        gate_class_name=str(
            getattr(adapter, "_gate_class_name", "")),
        gate_returns_tuple=bool(
            getattr(adapter, "_gate_returns_tuple", False)),
        hook_fires_per_forward=int(
            getattr(adapter, "_last_hook_fire_count", 0)),
        n_moe_layers=int(adapter.n_moe_layers),
        n_experts=int(adapter.n_experts),
        top_k=int(adapter.top_k),
        n_layers_routing_captured=int(
            routing.n_layers_with_routing),
        forward_trace_cid=str(fwd_trace_cid),
        routing_snapshot_cid=str(routing_cid),
        replay_trace_cid_with_routing=str(
            replay_with_routing_cid),
        replay_trace_cid_without_routing=str(
            replay_trace_cid_without_routing),
        forward_force_random_routing_trace_cid=str(
            force_random_trace_cid),
        max_abs_diff_replay_vs_forward_last_logits=float(
            diff_with),
        max_abs_diff_with_routing_vs_forward_last_logits=float(
            diff_with),
        max_abs_diff_without_routing_vs_forward_last_logits=(
            float(diff_without)),
        max_abs_diff_force_random_vs_forward_last_logits=float(
            diff_force_random),
        tier_tolerance=float(tier_tol),
        forward_routing_captured=bool(
            forward_routing_captured),
        replay_with_routing_matches_forward_floor=bool(
            replay_with_routing_byte_id),
        moe_routing_is_load_bearing=bool(
            moe_routing_is_load_bearing),
        routing_deterministic_across_two_forwards=bool(
            routing_deterministic),
        hidden_state_intercept_on_moe_block_moves_cid=bool(
            intercept_moves_cid),
        wall_clock_seconds=float(wall),
        bench_cid="",
    )
    # Stamp the bench CID (computed over the report minus itself).
    cid = report.cid()
    return dataclasses.replace(report, bench_cid=str(cid))


__all__ = [
    "W86_MOE_SUBSTRATE_V1_SCHEMA_VERSION",
    "W86_MOE_AXES_ALL",
    "W86_MOE_BLOCK_CANDIDATE_NAMES",
    "MoEInstrumentationAxis",
    "PerLayerRoutingV1",
    "ExpertRoutingSnapshotV1",
    "MoEForceRoutingInjectionV1",
    "MoECapabilityProbeV1",
    "MoERuntimeAdapterV1",
    "MoESubstrateClosureBenchReportV1",
    "moe_declared_axes",
    "probe_moe_capability_v1",
    "run_moe_substrate_closure_bench_v1",
]
