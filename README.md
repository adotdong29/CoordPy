# coordpy

Python SDK and CLI for coordinating teams of LLM agents through
**content-addressed, lifecycle-bounded context capsules** instead of
ever-growing token-crammed prompts. Every agent handoff seals into a
hash-chained capsule you can re-verify from disk, and every run produces
a manifest you can replay against another model.

PyPI: [`coordpy-ai`](https://pypi.org/project/coordpy-ai/) · import: `coordpy`

## Why CoordPy

Multi-agent stacks usually pass context around as raw prompts and
JSON. That scales until the prompt grows past the model's useful
window and the run silently devolves into token cramming — and then
when something breaks you can't reconstruct what each agent saw.

CoordPy attacks both edges of that frontier:

* **Bounded context, not token cramming.** Each agent sees the team
  instructions plus the latest N visible handoffs (default `N=4`),
  not the full transcript. Every team run reports the
  bounded-context savings in real tokens, not vibes.
* **Auditable handoffs.** Every agent output seals into a
  `TEAM_HANDOFF` capsule with a content-derived ID, declared
  parents, a per-handoff byte/token budget, and witness fields
  (`prompt_sha256`, `model_tag`) that prove which prompt produced
  which output.
* **Replayable runs.** A `team_result.json` manifest records each
  turn's prompt, generation params, and capsule CID. `coordpy-team
  replay` re-runs the same prompts on a different backend/model at
  the original sampling settings, so the audit story holds across
  models.

## Install

Requires Python 3.10 or newer.

```bash
pip install coordpy-ai
```

Verify:

```bash
coordpy --version           # coordpy 0.5.20 (coordpy.sdk.v3.43)
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

## Five-minute first run

The fastest way to see CoordPy do something useful is the bundled
`coordpy-team` CLI driving a curated preset against a local Ollama
endpoint or any OpenAI-compatible provider:

```bash
# Local Ollama (no API key needed)
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:14b
export COORDPY_OLLAMA_URL=http://localhost:11434

coordpy-team run \
    --preset quant_desk \
    --task examples/scenario_bullish.txt \
    --out-dir /tmp/desk-run
```

That writes four files into `/tmp/desk-run`:

| file | purpose |
|---|---|
| `final_output.txt` | the final agent's plain-text answer |
| `team_capsule_view.json` | the sealed `coordpy.capsule_view.v1` chain |
| `team_result.json` | the `coordpy.team_result.v1` manifest used for replay |
| `team_report.md` | a polished Markdown summary (telemetry + savings + audit) |

Re-verify the chain from bytes alone:

```bash
coordpy-capsule verify-view --view /tmp/desk-run/team_capsule_view.json
```

Replay the same prompts on a different model and compare:

```bash
coordpy-team compare \
    --preset quant_desk \
    --task examples/scenario_bullish.txt \
    --backend ollama --model qwen2.5:14b \
    --replay-backend ollama --replay-model gemma2:9b \
    --out-dir /tmp/desk-compare
```

The compare report shows whether the per-turn prompt SHAs match
and whether the synthesizer's parsed `ACTION` agrees across models.

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
| `coordpy-team run / replay / sweep / compare` | Drive an `AgentTeam` preset from the CLI; dump or replay a sealed bundle. The recommended front door for new users. |
| `coordpy-capsule view / verify / verify-view / audit` | Summarise / re-hash a sealed capsule chain (works on both team and `RunSpec` runs). |
| `coordpy --profile <name> --out-dir <dir>` | Run a research profile end to end and write the seven artefacts. |
| `coordpy-ci --report <product_report.json>` | Apply the CI pass/fail gate to a finished report. |
| `coordpy-import --jsonl <file>` | Audit a SWE-bench-Lite-style JSONL for compatibility. |

The research-profile chain is still useful for the structured
`RunSpec → RunReport` path:

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

## Agent teams in Python

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
    max_visible_handoffs=3,
    task_summary="Answer briefly using only the prior handoffs.",
)
result = team.run("Explain what coordpy does.")
print(result.final_output)

# Bounded-context savings vs naive token cramming.
print(result.cramming_estimate())

# Dump a four-file replayable bundle.
result.dump("/tmp/team-run")
```

For curated multi-role teams (quant desk, code review, research
writer) skip the role-prompt typing and use the bundled
`coordpy.presets`:

```python
from coordpy import presets

team = presets.quant_desk_team()
result = team.run(open("examples/scenario_bullish.txt").read())
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

The bundled [`examples/`](examples/) ladder (`01_quickstart.py`,
`02_quant_desk.py`, `03_replay_and_audit.py`) drives the same
public surface end-to-end against a real backend.

## Public surface

| Surface | Stability |
|---|---|
| `coordpy` SDK: `RunSpec`, `run`, `RunReport`, `SweepSpec`, `run_sweep`, `CoordPyConfig`, `Agent`, `AgentTurn`, `ActionDecision`, `AgentTeam`, `TeamResult`, `agent`, `create_team`, `replay_team_result`, `presets`, `TEAM_RESULT_SCHEMA`, `profiles`, `report`, `ci_gate`, `import_data`, `extensions`, capsule primitives, schema constants, `OpenAICompatibleBackend`, `OllamaBackend`, `backend_from_env` | Stable |
| Console scripts: `coordpy-team`, `coordpy-capsule`, `coordpy`, `coordpy-import`, `coordpy-ci` | Stable |
| On-disk schemas: `coordpy.capsule_view.v1`, `coordpy.team_result.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2` | Stable |
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
