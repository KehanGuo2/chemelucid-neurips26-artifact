"""Reviewer-harness tests for `interactive.task_bundle`.

These tests are Docker-independent: they validate the ingest/validate CLI,
the public/private split, the leakage scanner, and the cached-trajectory
grader's invariants. None of them require an LLM API.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
EXAMPLE_CNMR = REPO / "examples" / "acetaminophen_cnmr_peaks.csv"
EXAMPLE_DAG = REPO / "examples" / "custom_dag.yaml"
EXAMPLE_RUBRIC = REPO / "examples" / "custom_rubric.yaml"

# A cached non-error trajectory used by the grader smoke-test.
CACHED_DIAGNOSTIC = (
    REPO
    / "interactive"
    / "results"
    / "dummy_baselines"
    / "random_tool_roulette"
    / "partial_autonomous"
    / "Benzil.json"
)


def _run(cmd: list[str], cwd: Path = REPO, env_extra: dict[str, str] | None = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Ingest + validate (end-to-end)
# ---------------------------------------------------------------------------


@pytest.fixture
def ingested_bundle(tmp_path: Path) -> Path:
    """Ingest a bundle and return the path; pre-empts a per-test re-ingest."""
    out = tmp_path / "bundle"
    rc, _stdout, stderr = _run([
        sys.executable, "-m", "interactive.task_bundle", "ingest",
        "--smiles", "CC(=O)Nc1ccc(O)cc1",
        "--formula", "C8H9NO2",
        "--cnmr", str(EXAMPLE_CNMR),
        "--source", "test",
        "--license", "test-only",
        "--out", str(out),
    ])
    assert rc == 0, f"ingest failed: stderr={stderr}"
    return out


def test_ingest_creates_full_bundle_layout(ingested_bundle: Path) -> None:
    """All four required subdirectories and files are present."""
    assert (ingested_bundle / "public" / "task.json").is_file()
    assert (ingested_bundle / "public" / "tool_schema.json").is_file()
    assert (ingested_bundle / "public" / "prompt.md").is_file()
    assert (ingested_bundle / "private" / "answer.json").is_file()
    assert (ingested_bundle / "private" / "provenance.json").is_file()
    assert (ingested_bundle / "tests" / "test_no_leakage.py").is_file()
    assert (ingested_bundle / "tests" / "test_task_validity.py").is_file()
    assert (ingested_bundle / "tests" / "test_grader_smoke.py").is_file()


def test_validate_passes_on_clean_ingest(ingested_bundle: Path) -> None:
    rc, stdout, _stderr = _run([
        sys.executable, "-m", "interactive.task_bundle", "validate",
        str(ingested_bundle),
    ])
    assert rc == 0, f"validate exited non-zero: {stdout}"
    assert "issues           : 0" in stdout
    assert "OK: bundle passes all hard checks." in stdout


def test_public_files_contain_no_smiles(ingested_bundle: Path) -> None:
    """The public layer must not expose the canonical SMILES."""
    public_dir = ingested_bundle / "public"
    target = "CC(=O)Nc1ccc(O)cc1"
    for f in public_dir.rglob("*"):
        if not f.is_file():
            continue
        text = f.read_text()
        assert target not in text, f"hidden SMILES found in {f}"
        assert "InChI=" not in text, f"InChI found in {f}"


def test_public_files_contain_no_hard_leak_keys(ingested_bundle: Path) -> None:
    """`smiles:`, `inchi:`, `cas:`, `gt_smiles:`, ... must not appear in
    public/* keys. tool_schema.json and prompt.md may legitimately mention
    `validate_smiles` (a tool name), so we look for the keyed forms."""
    from interactive.task_bundle import _scan_text_for_leaks

    public_dir = ingested_bundle / "public"
    for f in public_dir.rglob("*"):
        if not f.is_file():
            continue
        leaks = _scan_text_for_leaks(f.read_text())
        assert not leaks, f"leak keys found in {f}: {leaks}"


def test_validate_catches_deliberate_smiles_leak(tmp_path: Path) -> None:
    """A bundle whose public/task.json includes the hidden SMILES must fail
    validation with a leakage issue."""
    bundle = tmp_path / "leak_bundle"
    (bundle / "public").mkdir(parents=True)
    (bundle / "private").mkdir(parents=True)
    (bundle / "public" / "task.json").write_text(json.dumps({
        "molecule_id": "leak", "spectrum_type": "CNMR",
        "molecular_formula": "C8H9NO2", "spectrum_peaks": [169.8, 24.1],
        "task_prompt": "find me",
        # Deliberate leak via both keyed form and canonical SMILES string.
        "smiles": "CC(=O)Nc1ccc(O)cc1",
    }))
    (bundle / "private" / "answer.json").write_text(json.dumps({
        "molecule_id": "leak",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "molecular_formula": "C8H9NO2",
    }))
    rc, stdout, _stderr = _run([
        sys.executable, "-m", "interactive.task_bundle", "validate", str(bundle),
    ])
    assert rc == 1, "leak bundle must fail validation"
    assert "hard_leak_key:smiles" in stdout
    assert "canonicalizes to the hidden answer" in stdout


# ---------------------------------------------------------------------------
# Custom DAG / rubric schema
# ---------------------------------------------------------------------------


def test_validate_dag_accepts_example() -> None:
    rc, stdout, _ = _run([
        sys.executable, "-m", "interactive.task_bundle", "validate-dag",
        str(EXAMPLE_DAG),
    ])
    assert rc == 0
    assert "issues           : 0" in stdout
    # The not-comparable note is part of the contract.
    assert "NOT part of the official" in stdout


def test_validate_rubric_accepts_example() -> None:
    rc, stdout, _ = _run([
        sys.executable, "-m", "interactive.task_bundle", "validate-rubric",
        str(EXAMPLE_RUBRIC),
    ])
    assert rc == 0
    assert "issues           : 0" in stdout
    assert "NOT comparable to official" in stdout


def test_validate_dag_rejects_missing_edges(tmp_path: Path) -> None:
    bad = tmp_path / "bad_dag.yaml"
    bad.write_text("schema_version: 1\nnodes:\n  - {id: F1, weight: 1}\n")
    rc, stdout, _ = _run([
        sys.executable, "-m", "interactive.task_bundle", "validate-dag",
        str(bad),
    ])
    assert rc == 1
    assert "missing or non-list `edges`" in stdout


def test_validate_rubric_rejects_missing_criteria(tmp_path: Path) -> None:
    bad = tmp_path / "bad_rubric.yaml"
    bad.write_text("schema_version: 1\nname: bad\n")
    rc, stdout, _ = _run([
        sys.executable, "-m", "interactive.task_bundle", "validate-rubric",
        str(bad),
    ])
    assert rc == 1
    assert "missing or non-list `criteria`" in stdout


# ---------------------------------------------------------------------------
# Cached-trajectory grader: must produce metrics WITHOUT any private fields.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not CACHED_DIAGNOSTIC.exists(),
    reason="cached dummy trajectory not present; grader smoke test skipped",
)
def test_grade_cached_trajectory_emits_no_private_fields(tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    rc, stdout, stderr = _run([
        sys.executable, "-m", "interactive.grade",
        "--episode-log", str(CACHED_DIAGNOSTIC),
        "--manifest", "core",
        "--out", str(out),
    ])
    assert rc == 0, f"grader failed: stdout={stdout} stderr={stderr}"
    assert out.exists(), "grader must write metrics.json"
    metrics = json.loads(out.read_text())

    # The grader emits aggregate metrics only; assert no private GT key
    # leaks into the public metrics file.
    forbidden_keys = {
        "smiles", "canonical_smiles", "inchi", "inchi_key",
        "answer", "ground_truth", "gt_smiles",
        "private_id", "legacy_internal_id",
    }

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k.lower() not in forbidden_keys, (
                    f"forbidden private key {k!r} present in metrics.json"
                )
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(metrics)
