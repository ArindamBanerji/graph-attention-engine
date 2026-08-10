# hello-gae — 5-Minute Quickstart

## Install

```bash
pip install -e ../../
```

Or install the published package:

```bash
pip install graph-attention-engine
```

## Run

```bash
python hello_gae.py
```

## What it does

- Creates a toy three-category domain: `priority`, `normal`, and `low`.
- Writes ten decisions and outcomes to an in-memory SQLite graph.
- Computes conservation status (`GREEN`, `AMBER`, or `RED`).
- Prints the graph summary and calibration values.

No server. No Neo4j. No AGE. Just Python and NumPy.

