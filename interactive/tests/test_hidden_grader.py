"""Tests for the private-grader CLI (`python -m interactive.grade`).

The CLI must:
1. Reproduce the inline grader's L2/coverage values within tolerance for a
   canonical episode log (no drift between inline vs CLI grader).
2. Error gracefully (non-zero exit) when the molecule is absent from the
   requested manifest.
3. Never leak any private answer field (e.g., gt_smiles) into metrics.json.
4. Always emit metrics.json with the documented schema keys.

The anonymous review package omits the private HRE assets, so tests that need
exact private grading skip unless those controlled-release assets are present.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator

import pytest

from interactive.grade import grade_episode, main

REPO_ROOT = Path(__file__).resolve().parents[2]

# A canonical episode log: gpt-5.2 on 2_4_dimethyl_aniline, partial_autonomous.
# Diagnostic file already records coverage=0.5769, cf_coverage=0.5769.
CANONICAL_EPISODE = (
    REPO_ROOT
    / "interactive"
    / "results"
    / "v3_multimodel"
    / "gpt-5.2_partial_autonomous"
    / "2_4_dimethyl_aniline_gpt-5.2_stateful_20260419_234038.json"
)

PRIVATE_HRE_AVAILABLE = (
    (REPO_ROOT / "data" / "CNMR_HRE_v5.json").exists()
    and any((REPO_ROOT / "data_split" / "private_grader_data").glob("*/grader_*.json"))
)

EXPECTED_KEYS = {
    "molecule_id",
    "manifest",
    "spectrum_type",
    "model",
    "info_mode",
    "scaffold",
    "exact_match",
    "tanimoto",
    "n_calls",
    "total_cost",
    "low_tool_submit",
    "L1",
    "L1_unique_regions",
    "L1_categories_used",
    "L2",
    "L2_cf",
    "L2_failure_type",
    "L3_dependency_rate",
    "L3_pivot_reactivity",
}

# Fields that, if present in metrics.json, would constitute a privacy leak.
PRIVATE_FIELDS = {"gt_smiles", "gt", "answer", "gt_canonical", "ground_truth"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_strings(obj) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k  # also scan keys: a leak via key name should also fail
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def _all_known_smiles() -> list[str]:
    """Collect every gt_smiles from private grader files when present.

    Used by the privacy test to confirm the metrics output never echoes any
    ground-truth answer string.
    """
    grader_root = REPO_ROOT / "data_split" / "private_grader_data"
    smiles: list[str] = []
    if not grader_root.exists():
        return smiles
    for f in grader_root.glob("*/grader_*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        s = d.get("ground_truth", {}).get("smiles")
        if s:
            smiles.append(s)
    return smiles


# ---------------------------------------------------------------------------
# Test 1: end-to-end consistency with inline grader
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CANONICAL_EPISODE.exists() or not PRIVATE_HRE_AVAILABLE,
    reason=(
        "Canonical episode or private HRE assets missing; private grader assets "
        "are intentionally omitted from the anonymous review package."
    ),
)
def test_cli_matches_inline_grader(tmp_path):
    out = tmp_path / "metrics.json"
    rc = main([
        "--episode-log", str(CANONICAL_EPISODE),
        "--manifest", "core",
        "--out", str(out),
    ])
    assert rc == 0, f"CLI exited {rc}"
    assert out.exists(), "metrics.json was not written"

    metrics = json.loads(out.read_text())
    inline = json.loads(CANONICAL_EPISODE.read_text())

    # Coverage / cf_coverage must agree with the inline-graded log
    # (allow a small tolerance for rounding).
    assert metrics["L2"] == pytest.approx(inline["coverage"], abs=1e-3)
    assert metrics["L2_cf"] == pytest.approx(inline["cf_coverage"], abs=1e-3)
    # L1 score recomputed from trajectory should match the cached score.
    assert metrics["L1"] == pytest.approx(inline["layer1"]["score"], abs=1e-3)
    # L3 metrics also come straight from causal_analysis on the same trajectory.
    assert metrics["L3_dependency_rate"] == pytest.approx(
        inline["layer3_dependency_rate"], abs=1e-3
    )
    assert metrics["L3_pivot_reactivity"] == pytest.approx(
        inline["layer3_conflict_reactivity"], abs=1e-3
    )
    # Cost / call counts derived from trajectory.
    assert metrics["n_calls"] == len(inline["trajectory"])
    assert metrics["total_cost"] == inline["total_cost"]


# ---------------------------------------------------------------------------
# Test 2: graceful error on unknown molecule
# ---------------------------------------------------------------------------

def test_unknown_molecule_errors(tmp_path):
    # Build a minimal valid episode log with a molecule_id in no manifest.
    fake_log = tmp_path / "fake_episode.json"
    fake_log.write_text(json.dumps({
        "molecule_id": "definitely_not_a_real_molecule_xyz",
        "spectrum_type": "CNMR",
        "trajectory": [],
        "submitted_smiles": "C",
    }))
    out = tmp_path / "metrics.json"
    rc = main([
        "--episode-log", str(fake_log),
        "--manifest", "auto",
        "--out", str(out),
    ])
    assert rc == 1, "Expected exit code 1 for unknown molecule"
    assert not out.exists(), "metrics.json must not be written on lookup failure"


# ---------------------------------------------------------------------------
# Test 3: privacy — no GT SMILES (or any private field name) in metrics.json
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CANONICAL_EPISODE.exists() or not PRIVATE_HRE_AVAILABLE,
    reason=(
        "Canonical episode or private HRE assets missing; private grader assets "
        "are intentionally omitted from the anonymous review package."
    ),
)
def test_metrics_has_no_private_leakage(tmp_path):
    out = tmp_path / "metrics.json"
    rc = main([
        "--episode-log", str(CANONICAL_EPISODE),
        "--manifest", "core",
        "--out", str(out),
    ])
    assert rc == 0
    raw = out.read_text()

    # No private field name should appear, not even as a key.
    metrics = json.loads(raw)
    assert isinstance(metrics, dict)
    for k in PRIVATE_FIELDS:
        assert k not in metrics, f"Private field '{k}' leaked into metrics.json"

    # No known gt_smiles string should appear anywhere in the file's text.
    for s in _all_known_smiles():
        # An empty or single-char SMILES would create false positives; skip
        # entries shorter than 4 chars (e.g. 'C', 'CC') — those aren't unique
        # ground-truth markers.
        if len(s) < 4:
            continue
        assert s not in raw, f"Ground-truth SMILES '{s}' leaked into metrics.json"


# ---------------------------------------------------------------------------
# Test 4: schema — all expected keys present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CANONICAL_EPISODE.exists() or not PRIVATE_HRE_AVAILABLE,
    reason=(
        "Canonical episode or private HRE assets missing; private grader assets "
        "are intentionally omitted from the anonymous review package."
    ),
)
def test_metrics_schema(tmp_path):
    out = tmp_path / "metrics.json"
    rc = main([
        "--episode-log", str(CANONICAL_EPISODE),
        "--manifest", "auto",
        "--out", str(out),
    ])
    assert rc == 0
    metrics = json.loads(out.read_text())

    missing = EXPECTED_KEYS - set(metrics.keys())
    assert not missing, f"metrics.json missing keys: {sorted(missing)}"

    # Type smoke checks
    assert isinstance(metrics["exact_match"], bool)
    assert isinstance(metrics["low_tool_submit"], bool)
    assert isinstance(metrics["n_calls"], int)
    assert isinstance(metrics["total_cost"], int)
    for k in ("L1", "L2", "L2_cf", "L3_dependency_rate", "L3_pivot_reactivity", "tanimoto"):
        assert isinstance(metrics[k], (int, float)), f"{k} should be numeric"
    assert metrics["L2_failure_type"] in {
        "none", "knowledge", "reasoning", "tool_use", "strategy",
    }
    assert metrics["manifest"] == "core"
