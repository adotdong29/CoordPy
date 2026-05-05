# coordpy

Python SDK and CLI for coordinating teams of LLM agents with
content-addressed, lifecycle-bounded context objects ("capsules"),
plus a reproducible run/report contract.

PyPI: [`coordpy-ai`](https://pypi.org/project/coordpy-ai/) ·
import: `coordpy`

## What it does

Multi-agent stacks usually pass context around as raw prompts and
JSON. That works until something breaks and you can't reconstruct
what each agent actually saw. coordpy treats context as typed
objects with content-derived IDs, declared parents, byte budgets,
and a fixed lifecycle. A run produces a `RunReport` whose root is
a sealed capsule DAG written to disk alongside a provenance
manifest, and you can re-verify the whole thing from the bytes
later.

The four pieces:

- **Capsules.** Every cross-boundary artefact (prompt, response,
  parse outcome, role handoff, run report) is a typed object with
  a SHA-256 content ID and an enforced lifecycle.
- **Runtime.** `coordpy.run(RunSpec(...))` produces a `RunReport`
  whose root is a sealed capsule DAG, plus a `provenance.json`
  manifest and a detached `meta_manifest.json` witness.
- **Team coordination.** Agents exchange `TEAM_HANDOFF`,
  `ROLE_VIEW`, and `TEAM_DECISION` capsules under a
  mechanically-checked T-1..T-7 lifecycle audit.
- **Evaluation harness.** Reproducible profiles (`local_smoke`,
  `bundled_57`, `public_jsonl`, ...), a `coordpy-ci` gate that
  consumes the report, and a `coordpy-capsule verify` CLI that
  re-hashes the on-disk capsule chain end-to-end.

## Install

```bash
pip install coordpy-ai
```

The PyPI distribution is `coordpy-ai`; the import name is
`coordpy`. The only required dependency is NumPy. Optional extras:
`coordpy-ai[scientific]`, `[dl]`, `[heavy]`, `[crypto]`,
`[docker]`, `[dev]`.

The package installs four console scripts:

```bash
coordpy --profile local_smoke --out-dir /tmp/cp-smoke
coordpy-import   --jsonl path/to/swe_bench_lite.jsonl --out audit.json
coordpy-ci       --report /tmp/cp-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule  view   --report /tmp/cp-smoke/product_report.json
coordpy-capsule  verify --report /tmp/cp-smoke/product_report.json
```

For development:

```bash
git clone https://github.com/adotdong29/context-zero.git
cd context-zero
pip install -e ".[dev]"
```

## Quickstart

```python
import coordpy

report = coordpy.run(coordpy.RunSpec(
    profile="local_smoke",
    out_dir="/tmp/cp-smoke",
))

assert report["readiness"]["ready"]
assert report["provenance"]["schema"] == "coordpy.provenance.v1"
assert report["capsules"]["chain_ok"]

print(report["capsules"]["root_cid"])
print(report["summary_text"])
```

## Agent teams

```python
from coordpy import AgentTeam, agent

team = AgentTeam.from_env(
    [
        agent("planner",    "Break the task into 2-3 concrete steps."),
        agent("researcher", "Gather the facts that matter."),
        agent("writer",     "Write the final answer for the user."),
    ],
    model="gpt-4o-mini",
    backend_name="openai",
    team_instructions=(
        "Reuse visible handoffs instead of restating the task."
    ),
)
result = team.run("Explain what coordpy does.")
print(result.final_output)
```

Backend configuration via environment:

```bash
# Local Ollama
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:0.5b
export COORDPY_OLLAMA_URL=http://localhost:11434

# OpenAI-compatible provider
export COORDPY_BACKEND=openai
export COORDPY_MODEL=gpt-4o-mini
export COORDPY_API_KEY=...
# Optional, for non-default providers:
# export COORDPY_API_BASE_URL=https://your-provider.example/v1
```

See [`examples/agent_team.py`](examples/agent_team.py) for the
minimal version, and
[`examples/build_with_coordpy.py`](examples/build_with_coordpy.py)
for an eight-step demo that drives every public layer of the SDK
against the synthetic backend (no network or API key required).

## Public surface

| Surface | Stability |
|---|---|
| `coordpy` SDK: `RunSpec`, `run`, `RunReport`, `SweepSpec`, `run_sweep`, `CoordPyConfig`, `Agent`, `AgentTeam`, `agent`, `create_team`, `profiles`, `report`, `ci_gate`, `import_data`, `extensions`, capsule primitives, schema constants, `OpenAICompatibleBackend`, `OllamaBackend`, `backend_from_env` | Stable |
| Console scripts: `coordpy`, `coordpy-import`, `coordpy-ci`, `coordpy-capsule` | Stable |
| On-disk schemas: `coordpy.capsule_view.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2` | Stable |
| `coordpy.__experimental__`: research-grade trust adjudication and multi-agent coordination ladder | Experimental, may move or disappear |

The experimental surface ships in the same wheel for reproducibility
and audit. Pin against the experimental tuple if you depend on it.

## Limitations

- The runtime is the capsule layer. It does not provide
  transformer-internal trust transfer or hidden-state access.
- Cross-host evidence in the test corpus is bounded by the lab
  topology that produced it.
- Not peer-reviewed. The code, tests, results notes, and theorem
  registry are public so they can be challenged.

## License

MIT. See [`LICENSE`](LICENSE).
