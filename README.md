# coordpy

Python SDK and CLI for coordinating teams of LLM agents with
content-addressed, lifecycle-bounded context objects ("capsules"),
plus a reproducible run/report contract.

PyPI: [`coordpy-ai`](https://pypi.org/project/coordpy-ai/) · import: `coordpy`

## Overview

Multi-agent stacks usually pass context around as raw prompts and
JSON. That works until something breaks and you can't reconstruct
what each agent actually saw. `coordpy` treats context as typed
objects with content-derived IDs, declared parents, byte budgets,
and a fixed lifecycle. A run produces a `RunReport` whose root is
a sealed capsule DAG written to disk alongside a provenance
manifest, and you can re-verify the whole thing from the bytes
later.

## Install

Requires Python 3.10 or newer.

```bash
pip install coordpy-ai
```

Verify:

```bash
coordpy --version           # coordpy 0.5.16 (coordpy.sdk.v3.43)
python -c "import coordpy; print(coordpy.__version__)"
```

The first parenthetical (`coordpy.sdk.v3.43`) is the
research-line tag exposed at `coordpy.SDK_VERSION`. It tracks
the underlying research programme and is independent of the
PyPI version.

The only required dependency is NumPy. Optional extras:

| Extra | Pulls in | When you want it |
|---|---|---|
| `[scientific]` | `scipy`, `networkx` | numerical / graph helpers |
| `[crypto]` | `cryptography` | optional signed-capsule paths |
| `[dl]` | `torch`, `peft` | the deep-learning research path |
| `[heavy]` | `hnswlib`, `transformers`, `RestrictedPython` | full research stack (heavy) |
| `[docker]` | `docker` | Docker-backed sandbox |
| `[dev]` | `ruff`, `black`, `mypy`, `pytest`, `build`, `twine` | contributing |

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
```

`coordpy.run` writes seven files into `out_dir`. The two you
will reach for most are `product_report.json` (the same shape
as the returned dict) and `capsule_view.json` (the sealed
capsule chain that `coordpy-capsule verify` re-hashes); the
others (`provenance.json`, `meta_manifest.json`,
`readiness_verdict.json`, `product_summary.txt`,
`sweep_result.json`) are always written and are useful for
audit. The `root_cid` is the SHA-256 of the run's `RUN_REPORT`
capsule; it is stable for a given input but differs between
runs because provenance includes a wall-clock timestamp.

## Console scripts

| Command | Purpose |
|---|---|
| `coordpy --profile <name> --out-dir <dir>` | Run a profile end to end and write the seven artefacts. |
| `coordpy-ci --report <product_report.json>` | Apply the CI pass/fail gate to a finished report. |
| `coordpy-capsule view --report ...` | Summarise the capsule graph. |
| `coordpy-capsule verify --report ...` | Re-hash the capsule chain end to end. |
| `coordpy-import --jsonl <file>` | Audit a SWE-bench-Lite-style JSONL for compatibility. |

A typical chain:

```bash
coordpy --profile local_smoke --out-dir /tmp/cp-smoke
coordpy-ci --report /tmp/cp-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule view   --report /tmp/cp-smoke/product_report.json
coordpy-capsule verify --report /tmp/cp-smoke/product_report.json
```

To exercise `coordpy-import` against the bundled mini fixture
(no external file required):

```bash
FIXTURE=$(python -c 'import coordpy, os; print(os.path.join(os.path.dirname(coordpy.__file__), "_internal/tasks/data/swe_real_shape_mini.jsonl"))')
coordpy-import --jsonl "$FIXTURE" --out /tmp/audit.json
```

## Agent teams

`AgentTeam.from_env` reads its backend from `COORDPY_*`
environment variables and **requires a configured backend** to
run — either a reachable Ollama server or an OpenAI-compatible
API key. To run a team without a network, see the
`SyntheticLLMClient` example below.

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

Local Ollama:

```bash
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:0.5b
export COORDPY_OLLAMA_URL=http://localhost:11434
```

OpenAI-compatible provider:

```bash
export COORDPY_BACKEND=openai
export COORDPY_MODEL=gpt-4o-mini
export COORDPY_API_KEY=...
# Optional, for non-default providers:
# export COORDPY_API_BASE_URL=https://your-provider.example/v1
```

To run a team without a network or an API key, pass a
`SyntheticLLMClient` directly:

```python
from coordpy import create_team, agent
from coordpy.synthetic_llm import SyntheticLLMClient

team = create_team(
    [agent("planner", "..."), agent("writer", "...")],
    backend=SyntheticLLMClient(default_response="ok"),
)
print(team.run("hi").final_output)
```

[`examples/build_with_coordpy.py`](examples/build_with_coordpy.py)
is an eight-step demo that drives every public layer this way.

## Public surface

| Surface | Stability |
|---|---|
| `coordpy` SDK: `RunSpec`, `run`, `RunReport`, `SweepSpec`, `run_sweep`, `CoordPyConfig`, `Agent`, `AgentTeam`, `agent`, `create_team`, `profiles`, `report`, `ci_gate`, `import_data`, `extensions`, capsule primitives, schema constants, `OpenAICompatibleBackend`, `OllamaBackend`, `backend_from_env` | Stable |
| Console scripts: `coordpy`, `coordpy-import`, `coordpy-ci`, `coordpy-capsule` | Stable |
| On-disk schemas: `coordpy.capsule_view.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2` | Stable |
| `coordpy.__experimental__` (a tuple of names exported under that attribute): research-grade trust-adjudication primitives and the multi-agent coordination ladder behind the research papers | Experimental, may move or disappear between releases |

The experimental surface ships in the same wheel for
reproducibility and audit. Pin against
`coordpy.__experimental__` if you depend on it.

## Limitations

- `coordpy` works at the capsule layer. It does not provide
  transformer-internal trust transfer or hidden-state access.
- The bundled cross-host evidence comes from the small two-node
  lab where it was generated. Behaviour at larger scales has not
  been measured.
- Not peer-reviewed. The code, tests, results notes, and theorem
  registry are public so they can be challenged.

## Where to go next

- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Releasing to PyPI: [`RELEASING.md`](RELEASING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

## License

MIT. See [`LICENSE`](LICENSE).
