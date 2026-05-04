"""Tests for the 48-molecule withheld-probe outcome-only grader pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from interactive.probe_grader import score_outcome_only

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO_ROOT / "data" / "withheld_probe"
MANIFEST_PATH = PROBE_DIR / "probe_48_manifest.csv"
SPECTRA_DIR = PROBE_DIR / "spectra"


# ---------------------------------------------------------------------------
# Outcome-only grader correctness
# ---------------------------------------------------------------------------

def test_score_outcome_only_exact_match():
    """Submitting the same SMILES (different valid form) returns inchi_exact=True."""
    # Aspirin: two valid SMILES forms that should InChI-collide.
    submitted = "CC(=O)Oc1ccccc1C(=O)O"
    hidden = "O=C(C)Oc1ccccc1C(=O)O"  # same molecule, different SMILES start atom
    out = score_outcome_only(submitted, hidden)
    assert out["inchi_exact"] is True
    assert out["valid_smiles"] is True
    assert out["tanimoto"] == pytest.approx(1.0, abs=1e-6)


def test_score_outcome_only_partial_match():
    """A close-but-different molecule gives inchi_exact=False with Tanimoto in (0,1)."""
    # Acetaminophen vs phenacetin (one is methyl-amide, the other ethyl-ether-amide)
    submitted = "CC(=O)Nc1ccc(O)cc1"  # acetaminophen
    hidden = "CCOc1ccc(NC(C)=O)cc1"  # phenacetin
    out = score_outcome_only(submitted, hidden)
    assert out["inchi_exact"] is False
    assert out["valid_smiles"] is True
    assert 0.0 < out["tanimoto"] < 1.0


def test_score_outcome_only_invalid_submission():
    """An unparseable SMILES marks valid_smiles=False and zero scores."""
    out = score_outcome_only("not_a_smiles_string@@@", "CCO")
    assert out["valid_smiles"] is False
    assert out["inchi_exact"] is False
    assert out["tanimoto"] == 0.0


def test_score_outcome_only_empty_submission():
    """None or empty string short-circuits to all-zero outcome."""
    out = score_outcome_only(None, "CCO")
    assert out == {"inchi_exact": False, "tanimoto": 0.0, "valid_smiles": False}
    out = score_outcome_only("   ", "CCO")
    assert out == {"inchi_exact": False, "tanimoto": 0.0, "valid_smiles": False}


def test_score_outcome_only_rejects_empty_hidden():
    """Empty hidden GT is a programming error, not a valid call."""
    with pytest.raises(ValueError):
        score_outcome_only("CCO", "")


def test_score_outcome_only_no_gt_in_output():
    """The grader output dict must never contain a 'gt' / 'hidden' / 'answer' key."""
    out = score_outcome_only("CCO", "CCO")
    forbidden = {"gt", "gt_smiles", "hidden", "hidden_smiles", "answer", "ground_truth"}
    assert not (set(out.keys()) & forbidden)


# ---------------------------------------------------------------------------
# Sampling reproducibility / artifact integrity
# ---------------------------------------------------------------------------

def test_probe_manifest_exists_and_has_48_rows():
    """The probe artifact is built and contains exactly 48 entries with seed=42."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"probe manifest missing at {MANIFEST_PATH}; run scripts/build_withheld_probe.py")

    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 48, f"expected 48 probe entries, got {len(rows)}"

    # All rows must record the same sampling_seed (deterministic provenance).
    seeds = {r["sampling_seed"] for r in rows}
    assert seeds == {"42"}, f"expected single seed=42, got {seeds}"

    # probe_ids are zero-padded probe_001..probe_048, no duplicates.
    pids = [r["probe_id"] for r in rows]
    assert pids == [f"probe_{i:03d}" for i in range(1, 49)], "probe_id ordering is non-canonical"
    assert len(set(pids)) == 48

    # Every row must have a non-empty hidden SMILES.
    for r in rows:
        assert r["smiles_hidden"], f"{r['probe_id']} has empty smiles_hidden"


def test_probe_spectra_have_no_hidden_answer():
    """Agent-facing spectrum JSONs must NEVER contain SMILES, InChI, or 'answer' fields."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"probe manifest missing at {MANIFEST_PATH}")

    with MANIFEST_PATH.open() as f:
        manifest = {r["probe_id"]: r for r in csv.DictReader(f)}

    forbidden_keys = {"smiles", "canonical_smiles", "inchi", "inchi_key", "answer", "ground_truth"}

    # Sanity-check at least one CNMR file exists per probe.
    leaks = []
    for probe_id, row in manifest.items():
        spec_path = SPECTRA_DIR / f"{probe_id}_CNMR.json"
        if not spec_path.exists():
            leaks.append(f"{probe_id}: missing CNMR spectrum file")
            continue
        with spec_path.open() as fh:
            doc = json.load(fh)

        # Walk every key/value, flag any forbidden key or string == hidden_smiles
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k.lower() in forbidden_keys:
                        leaks.append(f"{probe_id}{path}/{k}: forbidden key in agent-facing JSON")
                    walk(v, f"{path}/{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif isinstance(node, str):
                if node == row["smiles_hidden"]:
                    leaks.append(f"{probe_id}{path}: literal hidden SMILES present")
        walk(doc)

    assert not leaks, f"hidden-answer leakage detected:\n  " + "\n  ".join(leaks[:10])


def test_probe_no_overlap_with_core_18():
    """No probe SMILES may overlap the canonical 18-core (canonicalized comparison)."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"probe manifest missing at {MANIFEST_PATH}")
    try:
        from rdkit import Chem
    except ImportError:
        pytest.skip("RDKit unavailable")

    from interactive.config import ALL_MOLECULES, CNMR_DATA_DIR

    # Build canonical 18-core SMILES set.
    core_canon = set()
    for mol_id in ALL_MOLECULES:
        p = CNMR_DATA_DIR / f"{mol_id}_CNMR_HRE.json"
        if not p.exists():
            continue
        with p.open() as fh:
            doc = json.load(fh)
        for n in doc.get("Nodes", []):
            if n.get("id") == "BIG QUESTION":
                smi = n.get("answer", "")
                if smi:
                    m = Chem.MolFromSmiles(smi)
                    if m is not None:
                        core_canon.add(Chem.MolToSmiles(m, canonical=True))
                break

    # Verify each probe SMILES is outside that set.
    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))
    overlaps = []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles_hidden"])
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, canonical=True)
        if canon in core_canon:
            overlaps.append(f"{r['probe_id']} ({canon})")
    assert not overlaps, f"probes overlap core 18: {overlaps}"


def test_probe_sampling_reproducibility():
    """The probe set is fully determined by the recorded seed.

    We cannot re-invoke the upstream sampler in a unit test (it depends on
    a 50-mol pool of nmrshiftdb JSON files that may not be present), but we can
    verify that re-running build_withheld_probe.py produces a byte-identical
    manifest. This catches non-determinism in the mirroring step.
    """
    if not MANIFEST_PATH.exists():
        pytest.skip(f"probe manifest missing at {MANIFEST_PATH}")

    import subprocess
    before = MANIFEST_PATH.read_bytes()
    result = subprocess.run(
        ["python3", "scripts/build_withheld_probe.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"build_withheld_probe.py failed:\n{result.stderr}"
    after = MANIFEST_PATH.read_bytes()
    assert before == after, "rebuild changed the manifest — sampling is non-deterministic"
