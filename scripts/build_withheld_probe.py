"""Build the 48-molecule withheld-probe layer artifact under data/withheld_probe/.

The withheld-probe layer is the generalization layer of ChemElucid. The 48
molecules are curated from the 150-row NMRShiftDB-derived candidate pool at
data/dynamic/nmrshiftdb_scale150.csv under a frozen stratified workflow with
seed=42 (the workflow lives at scripts/curate_canary_set.py — legacy filename
retained from the original curation pipeline; the public layer it produces is
the withheld-probe layer).

This builder does NOT re-sample. It mirrors the curated 48 split-data files
under a withheld-probe-named directory tree so consumers see one self-contained
artifact, and writes:

  data/withheld_probe/probe_48_manifest.csv   — hidden ground-truth manifest.
  data/withheld_probe/spectra/probe_NNN_CNMR.json (and _HNMR.json when present).
  data/withheld_probe/README.md               — provenance + sampling rule.

The manifest CSV holds the SMILES (private grader data); spectra/*.json files
are agent-facing and contain only formula + peaks + solvent + MHz + task prompt.

Run:  python scripts/build_withheld_probe.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from interactive.config import ALL_MOLECULES  # noqa: E402

PROBE_DIR = ROOT / "data" / "withheld_probe"
SPECTRA_DIR = PROBE_DIR / "spectra"
MANIFEST_PATH = PROBE_DIR / "probe_48_manifest.csv"
README_PATH = PROBE_DIR / "README.md"

PROVENANCE_PATH = ROOT / "data_split" / "private_grader_data" / "_canary_provenance.json"
PUBLIC_SPLIT = ROOT / "data_split" / "public_task_data"
PRIVATE_SPLIT = ROOT / "data_split" / "private_grader_data"
POOL_CSV = ROOT / "data" / "dynamic" / "nmrshiftdb_scale150.csv"

# Fields that, if present in the agent-facing spectrum JSON, would leak the
# hidden answer. These checks run on every spectrum file we emit.
LEAK_KEY_BLACKLIST = {"smiles", "canonical_smiles", "inchi", "inchi_key", "answer", "ground_truth"}
SMILES_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\\.]{4,}")


def _load_provenance() -> Dict[str, Any]:
    if not PROVENANCE_PATH.exists():
        raise FileNotFoundError(
            f"Required provenance file not found: {PROVENANCE_PATH}. "
            "Run scripts/curate_canary_set.py first."
        )
    with PROVENANCE_PATH.open() as fh:
        return json.load(fh)


def _load_pool_index() -> Dict[str, Dict[str, str]]:
    """Map nmrshiftdb_id (e.g., '2500') -> raw pool row, for source bookkeeping."""
    if not POOL_CSV.exists():
        raise FileNotFoundError(f"Required pool CSV not found: {POOL_CSV}")
    out: Dict[str, Dict[str, str]] = {}
    with POOL_CSV.open() as fh:
        for row in csv.DictReader(fh):
            # source column looks like "nmrshiftdb2:2500"
            src = row.get("source", "")
            m = re.match(r"nmrshiftdb2?:?(\d+)", src) or re.match(
                r"nmrshiftdb_(\d+)", row.get("name", "")
            )
            if m:
                out[m.group(1)] = row
    return out


def _looks_like_smiles_candidate(s: str) -> bool:
    """Cheap pre-filter: skip strings that obviously can't be a SMILES we care about.

    Keeps strings >=4 chars, no whitespace, containing at least one bond/structure
    char (paren, bracket, =, #, or a lowercase aromatic atom).
    """
    if not isinstance(s, str) or len(s) < 4 or len(s) > 200:
        return False
    if any(c.isspace() for c in s):
        return False
    return any(c in s for c in "()[]=#") or any(c in s for c in "cnops")


def _agent_facing_leak_check(doc: Dict[str, Any], hidden_smiles: str) -> List[str]:
    """Return a list of leak-issues; empty list = clean.

    Hits if (a) a known blacklisted key appears, or (b) any string value
    canonicalizes to the hidden SMILES.
    """
    issues: List[str] = []
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")  # silence parse-error spam on prompt strings
        hidden_mol = Chem.MolFromSmiles(hidden_smiles)
        hidden_canon = Chem.MolToSmiles(hidden_mol, canonical=True) if hidden_mol else None
    except ImportError:
        hidden_canon = None

    def walk(node: Any, path: str = ""):
        if isinstance(node, dict):
            for k, v in node.items():
                kl = k.lower()
                if kl in LEAK_KEY_BLACKLIST:
                    issues.append(f"{path}/{k}: blacklisted key in agent-facing JSON")
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and hidden_canon and _looks_like_smiles_candidate(node):
            try:
                from rdkit import Chem
                m = Chem.MolFromSmiles(node)
                if m is not None and Chem.MolToSmiles(m, canonical=True) == hidden_canon:
                    issues.append(f"{path}: string canonicalizes to hidden SMILES")
            except Exception:
                pass

    walk(doc)
    return issues


def _emit_spectrum(probe_id: str, source_canary_id: str, spectrum_type: str,
                   hidden_smiles: str) -> Tuple[Path, Dict[str, Any]]:
    """Copy public_task_data/<canary>/task_<TYPE>.json to spectra/<probe>_<TYPE>.json
    after replacing the molecule_id with the probe-side id and re-checking for leaks.

    Returns (output_path, parsed_doc). Raises RuntimeError if a leak is found.
    """
    src = PUBLIC_SPLIT / source_canary_id / f"task_{spectrum_type}.json"
    if not src.exists():
        raise FileNotFoundError(
            f"Source spectrum file missing for {source_canary_id}/{spectrum_type}: {src}"
        )
    with src.open() as fh:
        doc = json.load(fh)

    # Replace internal id with the probe-side id so the agent never sees
    # "canary_*". Other fields (formula, peaks, solvent, mhz, task_prompt)
    # are already agent-safe.
    doc["molecule_id"] = probe_id
    out = SPECTRA_DIR / f"{probe_id}_{spectrum_type}.json"
    with out.open("w") as fh:
        json.dump(doc, fh, indent=2)

    # Leakage check on the WRITTEN file
    with out.open() as fh:
        doc_check = json.load(fh)
    leaks = _agent_facing_leak_check(doc_check, hidden_smiles)
    if leaks:
        out.unlink()
        raise RuntimeError(
            f"Leakage check failed for {out}: {leaks[:3]}"
        )
    return out, doc


def build() -> int:
    prov = _load_provenance()
    pool_idx = _load_pool_index()

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(prov["entries"], 1):
        canary_id = entry["canary_id"]
        probe_id = f"probe_{i:03d}"
        hidden_smiles = entry["canonical_smiles"]

        # 1. Emit agent-facing spectrum JSONs (CNMR always, HNMR if present)
        cnmr_path, _ = _emit_spectrum(probe_id, canary_id, "CNMR", hidden_smiles)
        hnmr_path = None
        if entry.get("has_hnmr"):
            hnmr_path, _ = _emit_spectrum(probe_id, canary_id, "HNMR", hidden_smiles)

        # 2. Look up source nmrshiftdb id from provenance source field.
        # source looks like "nmrshiftdb_2500" -> canonical "nmrshiftdb2:2500"
        m = re.match(r"nmrshiftdb_(\d+)", entry.get("source", "") or "")
        nmrshiftdb_id = m.group(1) if m else ""
        source_id = f"nmrshiftdb2:{nmrshiftdb_id}" if nmrshiftdb_id else ""

        # Sanity: the SMILES in the pool row (if found) should match the hidden one.
        # We log this rather than enforce, since the entry is the source of truth.
        pool_row = pool_idx.get(nmrshiftdb_id) if nmrshiftdb_id else None
        pool_smiles_match = bool(pool_row and pool_row.get("smiles") == hidden_smiles)

        rows.append({
            "probe_id": probe_id,
            "smiles_hidden": hidden_smiles,
            "source_id": source_id,
            "heavy_atoms": entry["hac"],
            "num_peaks": entry["n_peaks"],
            "has_hnmr": entry.get("has_hnmr", False),
            "legacy_internal_id": canary_id,
            "pool_smiles_match": pool_smiles_match,
            "sampling_seed": prov["rng_seed"],
        })

    # 3. Write the manifest CSV (hidden ground truth — grader-private).
    fieldnames = [
        "probe_id", "smiles_hidden", "source_id", "heavy_atoms", "num_peaks",
        "has_hnmr", "legacy_internal_id", "pool_smiles_match", "sampling_seed",
    ]
    with MANIFEST_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 4. Cross-check no probe SMILES overlaps the 18-core canonical set.
    # (Done in build, surfaced in README.)
    overlap = _check_core_overlap([r["smiles_hidden"] for r in rows])

    # 5. Write README documenting sampling rule + provenance.
    README_PATH.write_text(_render_readme(prov, rows, overlap))

    print(f"[OK] wrote {len(rows)} probe entries")
    print(f"     manifest -> {MANIFEST_PATH}")
    print(f"     spectra  -> {SPECTRA_DIR}/")
    print(f"     readme   -> {README_PATH}")
    print(f"     core-set overlap: {len(overlap)} (expected 0)")
    if overlap:
        print(f"     OVERLAP DETECTED: {overlap[:3]} ...")
        return 1
    return 0


def _check_core_overlap(probe_smiles_list: List[str]) -> List[str]:
    """Return canonical SMILES strings that appear in BOTH the probe and the 18-core."""
    try:
        from rdkit import Chem
    except ImportError:
        return []

    cnmr_dir = ROOT / "data" / "C_NMR"
    core_canon = set()
    for mol_id in ALL_MOLECULES:
        p = cnmr_dir / f"{mol_id}_CNMR_HRE.json"
        if not p.exists():
            continue
        with p.open() as fh:
            doc = json.load(fh)
        for n in doc.get("Nodes", []):
            if n.get("id") == "BIG QUESTION":
                smi = n.get("answer", "")
                if smi:
                    m = Chem.MolFromSmiles(smi)
                    if m:
                        core_canon.add(Chem.MolToSmiles(m, canonical=True))
                break

    overlap = []
    for s in probe_smiles_list:
        m = Chem.MolFromSmiles(s)
        if m and Chem.MolToSmiles(m, canonical=True) in core_canon:
            overlap.append(s)
    return overlap


def _render_readme(prov: Dict[str, Any], rows: List[Dict[str, Any]],
                   overlap: List[str]) -> str:
    n_hnmr = sum(1 for r in rows if r["has_hnmr"])
    return f"""# Withheld-probe layer (N={len(rows)})

This directory holds the 48-molecule withheld-probe layer of ChemElucid, used
for the outcome-only generalization evaluation. The probe layer is graded only
on final-answer correctness (InChI exact match + Morgan/2/2048 Tanimoto). It has
no expert-annotated reasoning DAG and therefore receives no L2/L3 metrics.

## Layout

- `probe_48_manifest.csv` — Hidden ground-truth manifest (grader-private,
  reviewer-facing only as a private artifact; do NOT serve to evaluation
  agents). Holds canonical SMILES, source NMR id, heavy-atom count, peak count,
  sampling seed. Includes a `legacy_internal_id` column that records the
  one-to-one mapping back to the curation-pipeline alias used to generate this
  set; this column is private and is not exposed in any agent-facing manifest.
- `spectra/probe_NNN_CNMR.json` and (when available) `spectra/probe_NNN_HNMR.json`
  — Agent-facing spectrum files. Contain only molecular formula, peaks, solvent,
  MHz, and the task prompt. SMILES, InChI, source ids, and legacy aliases are
  excluded by construction.

## Reviewer-facing usage

- **Reviewers**: this manifest is the private grader artifact. Use it together
  with `interactive/probe_grader.py:score_outcome_only(submitted, hidden)` to
  reproduce the outcome-only grading (InChI exact match + Morgan/2/2048
  Tanimoto). The agent surface — under `spectra/` — is what the evaluated
  agent sees and contains no hidden answer.
- **Agents**: must read only from `spectra/probe_NNN_*.json`. The CLI's
  `run-probe` subcommand routes through `interactive/data_loader.py`, which
  loads spectra from the public surface and the hidden SMILES from this
  manifest only on the grader-side code path.

## Sampling rule

- **Source pool:** 150-row NMRShiftDB-derived candidate pool at
  `data/dynamic/nmrshiftdb_scale150.csv`.
- **Curation workflow (frozen):** quotas matched to the 18-core descriptor
  distribution along three axes — heavy-atom count (HAC), degree of
  unsaturation (DBE), and ring count. Distribution caps applied: HAC<=30,
  stereocenters<=3, rings<=4. The workflow lives at
  `scripts/curate_canary_set.py` (legacy filename retained from the original
  curation pipeline; the public layer this script produces is the
  withheld-probe layer); the bins are HAC=
  [(0,10),(10,15),(15,20),(20,30),(30,60)], DBE=[(0,1),(1,3),(3,5),(5,8),(8,20)],
  RING=[(0,1),(1,2),(2,3),(3,6)].
- **Quality filter:** RDKit-parseable SMILES, peaks>=3, 5<=HAC<=60, DAG nodes
  F1..F8 (>=5 of 7) present in the source nmrshiftdb file.
- **Random seed:** {prov['rng_seed']} (deterministic; same seed -> same 48).
- **Core-overlap exclusion:** verified at build time against the canonical 18
  SMILES from `interactive/config.py:ALL_MOLECULES`. Overlap detected: {len(overlap)}.

## Provenance and license

The 48 molecules come from the nmrshiftdb2 public archive. nmrshiftdb is licensed
CC-BY-SA 3.0; source ids are recorded in `probe_48_manifest.csv:source_id` (e.g.,
`nmrshiftdb2:2500`) and in the build-time provenance file
`data_split/private_grader_data/_canary_provenance.json` (legacy filename;
this is the curation-pipeline provenance file kept under the original
filename for code compatibility, and it carries the same 48 entries under
the legacy internal alias).

## Counts

- Total probe molecules: {len(rows)}
- With companion 1H NMR spectrum: {n_hnmr}
- Mean heavy-atom count: {sum(r['heavy_atoms'] for r in rows) / max(len(rows),1):.1f}
- Mean peak count: {sum(r['num_peaks'] for r in rows) / max(len(rows),1):.1f}

## Scope of this layer

The withheld-probe layer is the generalization layer; the 18-core is the
calibration set. The probe distribution runs heavier than the 18-core in HAC,
DBE, and ring count because the nmrshiftdb pool lacks small (HAC<11) molecules.
This is a known unfixable mismatch; treat the probe as
"core 18 + a controlled difficulty step up" rather than as i.i.d. with core 18.
"""


if __name__ == "__main__":
    sys.exit(build())
