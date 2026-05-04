"""Split-data loader: separates agent-visible task data from grader-only ground truth.

Addresses the leak where {mol_id}_CNMR_HRE.json contained the SMILES answer in the
BIG QUESTION node and F1-F8 reasoning checkpoints — fields that ToolServer also
read on disk, meaning a Terminal-Bench-style agent with shell access could `cat`
the file.

This module loads from two parallel directories:
- public_task_data/{mol_id}/task_{TYPE}.json  -> safe, agent-visible
- private_grader_data/{mol_id}/grader_{TYPE}.json -> SMILES + DAG checkpoints

All 18 core molecules × {CNMR, HNMR, COMBO} are migrated; data_loader falls
back to the legacy single-file layout only for molecules outside that set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PublicTask:
    """Agent-visible data for one molecule + spectrum type."""
    molecule_id: str
    spectrum_type: str
    molecular_formula: str
    solvent: str = ""
    mhz: str = ""
    cnmr_peaks: List[float] = field(default_factory=list)
    hnmr_peaks_raw: str = ""  # Rich format string for HNMRPeak parser
    task_prompt: str = ""


@dataclass
class PrivateGrader:
    """Grader-only data for one molecule + spectrum type."""
    molecule_id: str
    spectrum_type: str
    gt_smiles: str
    molecular_formula: str
    dag_nodes: List[Dict[str, Any]] = field(default_factory=list)


def _split_root(data_dir: Path) -> Path:
    """The split layout lives alongside data/ at <repo>/data_split/."""
    return data_dir.parent / "data_split"


def has_split_data(molecule_id: str, spectrum_type: str, data_dir: Path) -> bool:
    """Return True iff the molecule has been migrated to split layout."""
    pub = _split_root(data_dir) / "public_task_data" / molecule_id / f"task_{spectrum_type}.json"
    priv = _split_root(data_dir) / "private_grader_data" / molecule_id / f"grader_{spectrum_type}.json"
    return pub.exists() and priv.exists()


def load_public_task(
    molecule_id: str, spectrum_type: str, data_dir: Path
) -> Optional[PublicTask]:
    """Load agent-visible task data. Returns None if file missing.

    Supports CNMR (cnmr_peaks), HNMR (hnmr_peaks_raw), and COMBO (both).
    For COMBO mode, the public file uses cnmr_solvent/cnmr_mhz; we surface
    those via solvent/mhz so callers don't need a third code path.
    """
    pub = _split_root(data_dir) / "public_task_data" / molecule_id / f"task_{spectrum_type}.json"
    if not pub.exists():
        return None
    with open(pub) as f:
        d = json.load(f)

    if spectrum_type == "COMBO":
        cnmr_peaks = list(d.get("cnmr_spectrum_peaks", []))
        hnmr_raw = d.get("hnmr_spectrum_peaks_raw", "")
        solvent = d.get("cnmr_solvent", "") or d.get("hnmr_solvent", "")
        mhz = d.get("cnmr_mhz", "") or d.get("hnmr_mhz", "")
    else:
        cnmr_peaks = d.get("spectrum_peaks", []) if spectrum_type == "CNMR" else []
        hnmr_raw = d.get("spectrum_peaks_raw", "") if spectrum_type == "HNMR" else ""
        solvent = d.get("solvent", "")
        mhz = d.get("mhz", "")

    return PublicTask(
        molecule_id=d["molecule_id"],
        spectrum_type=d["spectrum_type"],
        molecular_formula=d.get("molecular_formula", ""),
        solvent=solvent,
        mhz=mhz,
        cnmr_peaks=cnmr_peaks,
        hnmr_peaks_raw=hnmr_raw,
        task_prompt=d.get("task_prompt", ""),
    )


def load_private_grader(
    molecule_id: str, spectrum_type: str, data_dir: Path
) -> Optional[PrivateGrader]:
    """Load grader-only ground truth. Returns None if file missing."""
    priv = _split_root(data_dir) / "private_grader_data" / molecule_id / f"grader_{spectrum_type}.json"
    if not priv.exists():
        return None
    with open(priv) as f:
        d = json.load(f)
    return PrivateGrader(
        molecule_id=d["molecule_id"],
        spectrum_type=d["spectrum_type"],
        gt_smiles=d["ground_truth"]["smiles"],
        molecular_formula=d["ground_truth"].get("molecular_formula", ""),
        dag_nodes=d.get("dag_nodes", []),
    )


# ---------------------------------------------------------------------------
# Withheld-probe layer: agent-facing spectra under data/withheld_probe/spectra/
# Hidden ground truth lives in data/withheld_probe/probe_48_manifest.csv and
# is only ever loaded by the grader, never by the agent-facing ToolServer path.
# ---------------------------------------------------------------------------

def _probe_root(data_dir: Path) -> Path:
    """The withheld-probe layout lives under data/withheld_probe/."""
    return data_dir / "withheld_probe"


def has_probe_spectrum(probe_id: str, spectrum_type: str, data_dir: Path) -> bool:
    """Return True iff the probe has an agent-facing spectrum file for this type."""
    spec = _probe_root(data_dir) / "spectra" / f"{probe_id}_{spectrum_type}.json"
    return spec.exists()


def load_probe_spectrum(
    probe_id: str, spectrum_type: str, data_dir: Path
) -> Optional[PublicTask]:
    """Load agent-facing spectrum for a withheld-probe molecule.

    Mirrors load_public_task, but reads from data/withheld_probe/spectra/.
    Returns None if the file is missing.
    """
    spec = _probe_root(data_dir) / "spectra" / f"{probe_id}_{spectrum_type}.json"
    if not spec.exists():
        return None
    with open(spec) as f:
        d = json.load(f)

    cnmr_peaks = d.get("spectrum_peaks", []) if spectrum_type == "CNMR" else []
    hnmr_raw = d.get("spectrum_peaks_raw", "") if spectrum_type == "HNMR" else ""

    return PublicTask(
        molecule_id=d.get("molecule_id", probe_id),
        spectrum_type=d["spectrum_type"],
        molecular_formula=d.get("molecular_formula", ""),
        solvent=d.get("solvent", ""),
        mhz=d.get("mhz", ""),
        cnmr_peaks=cnmr_peaks,
        hnmr_peaks_raw=hnmr_raw,
        task_prompt=d.get("task_prompt", ""),
    )


def load_probe_hidden_smiles(probe_id: str, data_dir: Path) -> Optional[str]:
    """Load the hidden ground-truth SMILES for a probe from probe_48_manifest.csv.

    Caller is the grader, NOT the ToolServer. Returns None if probe not in manifest.
    """
    import csv as _csv
    manifest = _probe_root(data_dir) / "probe_48_manifest.csv"
    if not manifest.exists():
        return None
    with open(manifest) as f:
        for row in _csv.DictReader(f):
            if row.get("probe_id") == probe_id:
                return row.get("smiles_hidden") or None
    return None
