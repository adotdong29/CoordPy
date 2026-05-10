# Examples

Three short standalone programs that exercise the stable CoordPy
SDK surface end-to-end. They are ordered so each step adds one new
concept on top of the previous one.

| # | file | what it shows | runtime |
|--:|---|---|---|
| 01 | [`01_quickstart.py`](01_quickstart.py) | three-agent team via `AgentTeam.from_env(...)`; bounded-context savings; sealed capsule chain | < 30 s on local Ollama with `qwen2.5:0.5b` |
| 02 | [`02_quant_desk.py`](02_quant_desk.py) | `presets.quant_desk_team(...)` four-role desk; per-turn telemetry; `parse_action()`; full sealed bundle on disk | ~ 60 s on local Ollama with `qwen2.5:14b` |
| 03 | [`03_replay_and_audit.py`](03_replay_and_audit.py) | dump a manifest, replay it on a fresh backend, re-hash the new capsule chain; per-turn generation params restored faithfully | < 60 s on the same local backend |

All three pick the backend up from the standard CoordPy env vars:

```bash
# Local Ollama
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:0.5b
export COORDPY_OLLAMA_URL=http://localhost:11434

# OpenAI-compatible provider
export COORDPY_BACKEND=openai
export COORDPY_MODEL=gpt-4o-mini
export COORDPY_API_KEY=...
# Optional for non-default compatible providers:
# export COORDPY_API_BASE_URL=https://your-provider.example/v1
```

Then:

```bash
python3 examples/01_quickstart.py
python3 examples/02_quant_desk.py --out-dir /tmp/desk-run
python3 examples/03_replay_and_audit.py
```

For a CLI-only walkthrough that doesn't need any Python, see the
[Five-minute first run](../README.md#five-minute-first-run) section
of the README.

## Out-of-tree extension example

[`out_of_tree_plugin/`](out_of_tree_plugin/) ships a separate
distribution (`coordpy-markdown-sink`) that registers a custom
`ReportSink` via the `coordpy.report_sink` entry-point group. Use it
as a template when you want to publish your own report sink as a
standalone package.

## Sample input

[`scenario_bullish.txt`](scenario_bullish.txt) is the bundled US-
equity quant scenario referenced from the CLI examples in the README
and START_HERE. The same three scenarios (`bullish`, `risk_off`,
`ambiguous`) are also embedded in [`02_quant_desk.py`](02_quant_desk.py)
so the example runs without any external file.
