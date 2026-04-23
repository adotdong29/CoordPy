# `wevra-markdown-sink` — minimal out-of-tree Wevra plugin

This directory is a **standalone, pip-installable Python package** that
adds a new `ReportSink` to Wevra without editing any file in the main
repo. It exists as a worked exemplar of the Wevra extension surface: if
this package installs alongside `wevra` and works via `entry_points`,
the plugin contract is real.

The sink itself is deliberately tiny: it renders the Wevra product
report as a short Markdown digest and writes it to a configurable path.
The *point* is not the sink — it's the packaging.

## Layout

```
out_of_tree_plugin/
├── README.md              — this file
├── pyproject.toml         — declares the `wevra.report_sinks` entry_point
└── wevra_markdown_sink/
    ├── __init__.py
    └── sink.py            — the ReportSink implementation
```

## Install against a real Wevra checkout

```bash
# from a venv that already has wevra installed:
pip install -e ./examples/out_of_tree_plugin

# verify registration:
python -c "from vision_mvp.wevra.extensions import list_report_sinks; \
           print(list_report_sinks())"
# -> ['jsonfile', 'markdown', 'stdout']

# use it from the CLI:
wevra --profile local_smoke --out-dir /tmp/wevra-smoke \
      --report-sink markdown
```

The `markdown` sink will write `report.md` into the run's `out_dir`.

## How the wiring works

`pyproject.toml` in this package declares:

```toml
[project.entry-points."wevra.report_sinks"]
markdown = "wevra_markdown_sink.sink:register"
```

When Wevra first needs a report sink, `wevra.extensions.registry`
discovers this entry point via `importlib.metadata.entry_points`, calls
`register()`, and the sink becomes resolvable by name. No in-tree edit
to the Wevra repo is required.

## What this closes

Master plan § 10.5 ledger item: *"First real out-of-tree plugin — e.g.
a Slack `ReportSink` shipped from a separate pip-installable package —
not yet demonstrated."* This package is that exemplar, in its cheapest
possible form. The extension machinery itself landed in Slice 2; this
directory is the first *consumer* artifact proving the contract is real.
