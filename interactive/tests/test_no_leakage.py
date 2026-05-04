"""Regression test: public_task_data/ must contain no SMILES leakage.

For each JSON under data_split/public_task_data/, walk every string token and
attempt RDKit parsing. If any token parses to a valid molecule with at least
3 heavy atoms, we treat that as a SMILES leak and fail.

Also enforces the inverse: every public task file must have a matching
private grader file with a non-empty gt_smiles.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

from interactive.config import DATA_DIR

REPO_ROOT = DATA_DIR.parent
PUBLIC_ROOT = REPO_ROOT / "data_split" / "public_task_data"
PRIVATE_ROOT = REPO_ROOT / "data_split" / "private_grader_data"

# Field names whose string contents are known prose, not SMILES candidates.
# Even so, we still scan their tokens; this list is informational only.
# Token rules: 5 <= len <= 200; alphanumeric plus the SMILES punctuation set;
# at least one letter and one ring/bond/branch character to weed out plain
# words like "Chloroform" or "Molecule".
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\\/\.%:]+$")
_SMILES_HINT = re.compile(r"[\[\]\(\)=#/\\1-9]")


def _iter_strings(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def _candidate_tokens(s: str) -> Iterable[str]:
    """Split a string into substrings that *might* be SMILES.

    We split on whitespace and a few common separators so that a token like
    `Cc1ccc(N)c(C)c1` extracted from "is Cc1ccc(N)c(C)c1." is testable.
    """
    for tok in re.split(r"[\s,;{}\"]+", s):
        tok = tok.strip(".,:;!?\"'`")
        if 5 <= len(tok) <= 200 and _SMILES_CHARS.match(tok) and _SMILES_HINT.search(tok):
            yield tok


def _scan_for_smiles_leaks(data, source: Path) -> List[Tuple[str, str]]:
    """Return list of (token, parent_string) pairs that parse as SMILES."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pytest.skip("RDKit unavailable; cannot run leakage test")

    leaks: List[Tuple[str, str]] = []
    for s in _iter_strings(data):
        for tok in _candidate_tokens(s):
            mol = Chem.MolFromSmiles(tok)
            if mol is None:
                continue
            if mol.GetNumHeavyAtoms() >= 3:
                leaks.append((tok, s))
    return leaks


def _public_task_files() -> List[Path]:
    if not PUBLIC_ROOT.exists():
        return []
    return sorted(PUBLIC_ROOT.rglob("task_*.json"))


@pytest.mark.parametrize("path", _public_task_files(), ids=lambda p: p.relative_to(PUBLIC_ROOT).as_posix())
def test_public_file_has_no_smiles_leak(path: Path):
    """No string in any public task file should parse to a >=3-HAC molecule."""
    with open(path) as f:
        data = json.load(f)
    leaks = _scan_for_smiles_leaks(data, path)
    assert not leaks, (
        f"SMILES leak in {path.relative_to(REPO_ROOT)}: "
        + "; ".join(f"{tok!r} (in {snippet[:80]!r})" for tok, snippet in leaks[:3])
    )


def test_split_layout_has_files():
    """Sanity: ensure the migration produced something to test."""
    files = _public_task_files()
    assert len(files) >= 18, f"Expected >=18 public task files, got {len(files)}"


@pytest.mark.parametrize("path", _public_task_files(), ids=lambda p: p.relative_to(PUBLIC_ROOT).as_posix())
def test_each_public_file_has_matching_private_grader(path: Path):
    """For every public file there must be a private grader with gt_smiles."""
    rel = path.relative_to(PUBLIC_ROOT)
    # rel is e.g. mol_id/task_CNMR.json
    mol_id = rel.parts[0]
    mode = rel.name.replace("task_", "").replace(".json", "")
    priv = PRIVATE_ROOT / mol_id / f"grader_{mode}.json"
    assert priv.exists(), f"Missing private grader for {mol_id}/{mode}"
    with open(priv) as f:
        gd = json.load(f)
    smi = gd.get("ground_truth", {}).get("smiles", "")
    assert smi, f"Empty gt_smiles in {priv.relative_to(REPO_ROOT)}"
