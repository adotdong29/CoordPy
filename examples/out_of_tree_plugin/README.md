# `coordpy-markdown-sink` — minimal out-of-tree CoordPy plugin

This directory is a **standalone, pip-installable Python package** that
adds a new `ReportSink` to CoordPy without editing any file in the main
repo. It exists as a worked exemplar of the CoordPy extension surface: if
this package installs alongside `coordpy` and works via `entry_points`,
the plugin contract is real.

The sink itself is deliberately tiny: it renders the CoordPy product
report as a short Markdown digest and writes it to a configurable path.
The *point* is not the sink — it's the packaging.

## Layout

```
out_of_tree_plugin/
├── README.md              — this file
├── pyproject.toml         — declares the `coordpy.report_sinks` entry_point
└── coordpy_markdown_sink/
    ├── __init__.py
    └── sink.py            — the ReportSink implementation
```

## Install against a real CoordPy checkout

```bash
# from a venv that already has coordpy installed:
pip install -e ./examples/out_of_tree_plugin

# verify registration:
python -c "from vision_mvp.coordpy.extensions import list_report_sinks; \
           print(list_report_sinks())"
# -> ['jsonfile', 'markdown', 'stdout']

# use it from the CLI:
coordpy --profile local_smoke --out-dir /tmp/coordpy-smoke \
      --report-sink markdown
```

The `markdown` sink will write `report.md` into the run's `out_dir`.

## How the wiring works

`pyproject.toml` in this package declares:

```toml
[project.entry-points."coordpy.report_sinks"]
markdown = "coordpy_markdown_sink.sink:register"
```

When CoordPy first needs a report sink, `coordpy.extensions.registry`
discovers this entry point via `importlib.metadata.entry_points`, calls
`register()`, and the sink becomes resolvable by name. No in-tree edit
to the CoordPy repo is required.

## What this closes

Master plan § 10.5 ledger item: *"First real out-of-tree plugin — e.g.
a Slack `ReportSink` shipped from a separate pip-installable package —
not yet demonstrated."* This package is that exemplar, in its cheapest
possible form. The extension machinery itself landed in Slice 2; this
directory is the first *consumer* artifact proving the contract is real.
