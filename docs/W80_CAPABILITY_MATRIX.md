# W80 Capability Matrix — hosted vs local substrate access

The W80 capability matrix V1 is a **living** hosted-vs-local
substrate capability matrix. Where the W79 boundary V12 module
documents the *hosted side* of the wall as a static prose list,
the W80 matrix re-probes both the hosted side and the local
side on every build call, surfaces every backend as a column,
every substrate axis as a row, and is content-addressed so the
CID changes whenever the matrix does.

Build the matrix:

```python
from coordpy.capability_matrix_v1 import (
    build_capability_matrix_v1,
    emit_capability_matrix_v1_witness,
)

matrix = build_capability_matrix_v1(include_transformers=True)
print(matrix.render_markdown())
print(matrix.to_json())

witness = emit_capability_matrix_v1_witness(matrix)
print(witness.to_dict())
```

## Surfaces

The W80 capability matrix V1 tracks four surfaces:

| Surface | Transport | Deployment | Notes |
|---------|-----------|------------|-------|
| `hosted_third_party`              | `https`      | `third_party_cloud` | text + optional logprobs + optional prefix-cache; substrate axes blocked |
| `local_openai_facade`             | `in_process` | `local_facade`      | OpenAI-shaped surface over the in-repo controlled runtime; carries substrate side channel |
| `controlled_runtime_numpy`        | `in_process` | `in_repo`           | NumPy controlled runtime V1; every W80 axis is `available` |
| `controlled_runtime_transformers` | `in_process` | `local_pretrained`  | HF transformers controlled runtime V1; some axes `backend_specific` (see parity matrix) |

## Axes

The W80 instrumentation contract V1 defines 12 substrate axes,
partitioned into reads / writes / witnessing. The capability
matrix also records 5 transport/deployment axes for context.

Read axes:

- `read_hidden_state`
- `read_kv_cache`
- `read_attention_probs`
- `read_per_layer_logits`
- `read_final_logits`

Write axes:

- `write_hidden_state_inject`
- `write_kv_restore`
- `write_attention_bias`
- `inject_prefix_state`

Witness axes:

- `replay_from_kv`
- `deterministic_replay`
- `content_addressed_trace`

Transport / deployment axes:

- `transport_text_completion`
- `transport_logprobs_optional`
- `transport_prefix_cache_optional`
- `side_channel_available`
- `deployment_mode`

## Capability tags

Every cell in the matrix carries one of four canonical tags:

- `available` — full contract coverage, canonical shape.
- `backend_specific` — implemented but with backend-known
  caveats (e.g. HF attention probs only with
  `attn_implementation="eager"`).
- `best_effort` — the runtime tries but may degrade silently.
- `unavailable` — the backend honestly does not support this
  axis.

Hosted plane axes are typically `unavailable` for substrate
reads/writes; controlled runtime backends are typically
`available` or `backend_specific`.

## Asymmetry report

The capability matrix exposes an `asymmetry_report()` that
isolates axes which are **hosted-blocked AND local-universal** —
the load-bearing W80 surface where the controlled runtimes
meaningfully extend what the hosted plane can do.

```python
report = matrix.asymmetry_report()
# report["n_asymmetry_axes"] >= 5
# report["asymmetry_axes"] is a subset of both hosted_blocked_axes
# and local_universal_axes
```

## Cross-link to the W79 boundary V12

The W80 matrix cross-links each new instrumentation axis to the
W79 boundary V12's hosted-blocked axis name, so any caller can
ask either question:

- *what does this backend expose?* (W80 axes + capability tags)
- *what does the hosted plane block?* (W79 boundary V12 axes)

The cross-link is a stable mapping carried as `axis_cross_link`
on the matrix dataclass.

## Refresh semantics

The matrix is **living**:

- `build_capability_matrix_v1()` re-probes the hosted boundary
  V12, the local OpenAI façade descriptor, the in-repo NumPy
  runtime, and the HF transformers runtime (if installed).
- `refreshed_at_ns` is a unix-ns timestamp; two consecutive
  builds produce strictly increasing timestamps.
- The matrix CID changes on every refresh; the
  `hosted_boundary_v12_cid` field stays constant unless the
  boundary V12 changes.

## Backend availability

Building the matrix without the `transformers` / `torch`
optional deps produces a matrix with three available surfaces
and one explicit "transformers unavailable" entry. CI on lean
environments still produces a deterministic matrix.
