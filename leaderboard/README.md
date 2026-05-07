# MolPuzzle-Gym Leaderboards

This directory contains lightweight static leaderboards built from cached diagnostic outputs. They are meant to be GitHub-readable summaries, not a replacement for the full paper analysis.

## Current Table

- [Audit-18 Partial-Autonomous Diagnostic Leaderboard](audit18_pa.md)

The main leaderboard intentionally excludes Transfer-24 and hidden-transfer evaluations. Those probes are useful for robustness analysis, but the public-facing table focuses on the Audit-18 partial-autonomous diagnostic profile.

## Regeneration

```bash
python scripts/build_leaderboard.py
```

The command reads `analysis/broad_pa_audit18_diagnostic_snapshot.csv/json` when available. If the snapshot is missing, it rebuilds the CSV/JSON from cached broad PA Audit-18 diagnostic logs under `interactive/results/v3_multimodel` without making any LLM calls.
