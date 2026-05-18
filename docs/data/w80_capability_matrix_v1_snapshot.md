# W80 capability matrix V1 - sample snapshot

This file is an auto-generated sample of the W80 capability
matrix V1. The matrix is *living* - the `refreshed_at_ns`
field and the matrix CID change on every fresh build.
Regenerate with:

```python
from coordpy.capability_matrix_v1 import (
    build_capability_matrix_v1,
)
matrix = build_capability_matrix_v1(include_transformers=True)
print(matrix.render_markdown())
print(matrix.asymmetry_report())
```

## Matrix

| Axis | hosted_third_party | local_openai_facade | controlled_runtime_numpy | controlled_runtime_transformers |
|------|------|------|------|------|
| read_hidden_state | unavailable | available | available | available |
| read_kv_cache | unavailable | available | available | available |
| read_attention_probs | unavailable | available | available | backend_specific |
| read_per_layer_logits | unavailable | available | available | backend_specific |
| read_final_logits | best_effort | available | available | available |
| write_hidden_state_inject | unavailable | available | available | backend_specific |
| write_kv_restore | unavailable | available | available | available |
| write_attention_bias | unavailable | available | available | backend_specific |
| inject_prefix_state | unavailable | available | available | backend_specific |
| replay_from_kv | unavailable | available | available | available |
| deterministic_replay | best_effort | available | available | available |
| content_addressed_trace | best_effort | available | available | available |
| transport_text_completion | available | available | available | available |
| transport_logprobs_optional | best_effort | available | available | available |
| transport_prefix_cache_optional | best_effort | backend_specific | backend_specific | backend_specific |
| side_channel_available | unavailable | available | available | available |
| deployment_mode | available | available | available | available |

## Asymmetry report

- n_hosted_blocked_axes: 9
- n_local_universal_axes: 12
- n_asymmetry_axes (hosted-blocked AND local-universal): 9
- asymmetry_axes: ['read_hidden_state', 'read_kv_cache', 'read_attention_probs', 'read_per_layer_logits', 'write_hidden_state_inject', 'write_kv_restore', 'write_attention_bias', 'inject_prefix_state', 'replay_from_kv']

## Probed surfaces

### hosted_third_party

- label: hosted third-party LLM HTTP API
- transport_mode: https
- deployment_mode: third_party_cloud
- notes:
  - blocked on every controlled-runtime substrate axis
  - carries text, optional logprobs, optional prefix-cache accounting only
  - hosted-blocked-axis count at probe: 56

### local_openai_facade

- label: local OpenAI-compatible façade over controlled runtime
- transport_mode: in_process
- deployment_mode: local_facade
- notes:
  - OpenAI-shaped HTTP surface; in-process façade reroutes calls to the in-repo controlled runtime
  - exposes a substrate side-channel that hosted APIs cannot carry

### controlled_runtime_numpy

- label: in-repo NumPy controlled runtime (controlled_runtime_substrate_v1)
- transport_mode: in_process
- deployment_mode: in_repo
- notes:
  - small NumPy transformer, frozen Xavier-init weights, byte-identical replay
  - exposes every W80 substrate axis as AVAILABLE

### controlled_runtime_transformers

- label: HF transformers controlled runtime (transformers_runtime_v1)
- transport_mode: in_process
- deployment_mode: local_pretrained
- notes:
  - pretrained transformers backend, lazy-loaded
  - attention-bias steer is BACKEND_SPECIFIC via attention_mask augmentation
  - byte-identical replay-from-KV under fp32 CPU

