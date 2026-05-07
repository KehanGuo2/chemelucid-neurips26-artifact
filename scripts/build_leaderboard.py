"""Build the static MolPuzzle-Gym leaderboard markdown.

This script is cache-only. It reads the Audit-18 partial-autonomous diagnostic
snapshot if present, or rebuilds that snapshot from cached broad PA diagnostic
JSON logs. It does not call model providers and intentionally excludes
Transfer-24 / hidden-transfer rows from the public-facing leaderboard.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO / "analysis"
SNAPSHOT_CSV = ANALYSIS_DIR / "broad_pa_audit18_diagnostic_snapshot.csv"
SNAPSHOT_JSON = ANALYSIS_DIR / "broad_pa_audit18_diagnostic_snapshot.json"
LEADERBOARD_DIR = REPO / "leaderboard"
AUDIT18_MD = LEADERBOARD_DIR / "audit18_pa.md"
INDEX_MD = LEADERBOARD_DIR / "README.md"


def ensure_snapshot() -> None:
    if SNAPSHOT_CSV.exists() and SNAPSHOT_JSON.exists():
        return

    sys.path.insert(0, str(ANALYSIS_DIR))
    sys.path.insert(0, str(REPO))
    import generate_broad_pa_audit18_diagnostic_snapshot as snapshot

    rows = snapshot.build_rows()
    snapshot.write_csv(rows)
    snapshot.write_json(rows)


def read_rows() -> list[dict[str, str]]:
    ensure_snapshot()
    with SNAPSHOT_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in {SNAPSHOT_CSV}")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def fmt_num(row: dict[str, str], key: str, places: int = 3) -> str:
    value = row.get(key, "")
    if value == "":
        return "NA"
    return f"{float(value):.{places}f}"


def fmt_one(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return "NA"
    return f"{float(value):.1f}"


def fmt_em(row: dict[str, str]) -> str:
    exact = as_int(row, "exact_n")
    n = as_int(row, "n")
    pct = as_float(row, "exact_match_pct")
    return f"{exact}/{n} ({pct:.1f}%)"


def note_for(row: dict[str, str]) -> str:
    exact_pct = as_float(row, "exact_match_pct")
    l1 = as_float(row, "mean_l1")
    calls = as_float(row, "mean_pre_submit_tool_calls")
    low_tool_n = as_int(row, "low_tool_n")

    notes: list[str] = []
    if exact_pct >= 75:
        notes.append("high EM; compare process profile")
    if l1 >= 0.75 and exact_pct < 50:
        notes.append("broad evidence, weaker outcome")
    if calls <= 5:
        notes.append("compact trace")
    if low_tool_n:
        notes.append(f"{low_tool_n}/18 low-tool run(s)")
    if not notes:
        notes.append("diagnostic profile")
    return "; ".join(notes)


def sorted_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            -as_float(row, "exact_match_pct"),
            -as_float(row, "mean_tanimoto"),
            as_float(row, "mean_pre_submit_tool_calls"),
            row["model"].casefold(),
        ),
    )


def leaderboard_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Rank | Model | EM ↑ | Tanimoto ↑ | Calls ↓ | Cost ↓ | L1 ↑ | L2 ↑ | L3 ↑ | Notes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(sorted_rows(rows), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["model"],
                    fmt_em(row),
                    fmt_num(row, "mean_tanimoto"),
                    fmt_one(row, "mean_pre_submit_tool_calls"),
                    fmt_one(row, "mean_spent_cost"),
                    fmt_num(row, "mean_l1"),
                    fmt_num(row, "mean_l2"),
                    fmt_num(row, "mean_l3"),
                    note_for(row),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def snapshot_meta() -> dict:
    if not SNAPSHOT_JSON.exists():
        return {}
    with SNAPSHOT_JSON.open() as f:
        return json.load(f)


def write_audit18(rows: list[dict[str, str]]) -> None:
    meta = snapshot_meta()
    n = as_int(rows[0], "n")
    body = f"""# Audit-18 Partial-Autonomous Diagnostic Leaderboard

MolPuzzle-Gym is a behavior-diagnostic assessment, not a one-number leaderboard. EM is sorted for convenience, but the intended readout is the diagnostic profile across final-answer quality, tool use, and process metrics.

L1 evidence-acquisition breadth, L2 expert-roadmap coverage, L3 observation-conditioned dependency.

**Scope.** Audit-18 only; partial-information access; autonomous prompting; cached diagnostic outputs only. Transfer-24 and hidden-transfer results are intentionally excluded from this main leaderboard.

**Anonymous-release hygiene.** The table reports public model names and aggregate metrics only. It does not include submitter names, institutions, private molecule identities, prompts, traces, or hidden answers.

{leaderboard_table(rows)}

## How To Read This Table

- `EM` is exact-match final-answer accuracy over `{n}` Audit-18 molecules.
- `Tanimoto` is the mean fingerprint similarity of submitted structures to targets.
- `Calls` counts pre-submit non-submit tool calls; lower is not automatically better because sparse evidence can indicate shortcut behavior.
- `Cost` is the cached total tool cost when present in the diagnostic JSON.
- `L1`, `L2`, and `L3` are process diagnostics, not optimization targets in isolation.

## Data Source

- Snapshot CSV: `{SNAPSHOT_CSV.relative_to(REPO)}`
- Snapshot JSON: `{SNAPSHOT_JSON.relative_to(REPO)}`
- Source scope: `{meta.get("scope", "Audit-18 partial-autonomous diagnostic snapshot")}`

Regenerate with:

```bash
python scripts/build_leaderboard.py
```
"""
    AUDIT18_MD.write_text(body)


def write_index() -> None:
    body = """# MolPuzzle-Gym Leaderboards

This directory contains lightweight static leaderboards built from cached diagnostic outputs. They are meant to be GitHub-readable summaries, not a replacement for the full paper analysis.

## Current Table

- [Audit-18 Partial-Autonomous Diagnostic Leaderboard](audit18_pa.md)

The main leaderboard intentionally excludes Transfer-24 and hidden-transfer evaluations. Those probes are useful for robustness analysis, but the public-facing table focuses on the Audit-18 partial-autonomous diagnostic profile.

## Regeneration

```bash
python scripts/build_leaderboard.py
```

The command reads `analysis/broad_pa_audit18_diagnostic_snapshot.csv/json` when available. If the snapshot is missing, it rebuilds the CSV/JSON from cached broad PA Audit-18 diagnostic logs under `interactive/results/v3_multimodel` without making any LLM calls.
"""
    INDEX_MD.write_text(body)


def main() -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    write_audit18(rows)
    write_index()
    print(f"wrote {AUDIT18_MD}")
    print(f"wrote {INDEX_MD}")


if __name__ == "__main__":
    main()
