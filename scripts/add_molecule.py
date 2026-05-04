"""Ingest a single (SMILES, 13C peak list) pair into the benchmark.

Produces a ChemElucid-compatible HRE JSON in data/C_NMR/.
Auto-fills F1.x.1 motif nodes via RDKit SMARTS matching (same logic as
scripts/fix_cnmr_ground_truth.py) and peak-range nodes by matching peaks
to conventional 13C chemical-shift regions.

Usage:
    python scripts/add_molecule.py \\
        --name my_compound \\
        --smiles "<your_canonical_smiles>" \\
        --peaks "<comma_separated_13C_shifts_in_ppm>" \\
        --solvent "Chloroform-d" --freq 101 \\
        [--dir data/C_NMR] [--source "<repo:id>"] [--apply]
    # NOTE: Do not paste a SMILES that is already in the core 18 or
    # canary 48 sets. The leakage audit will flag the resulting commit.

Bulk mode (from a CSV):
    python scripts/add_molecule.py --csv molecules.csv
    # CSV header: name,smiles,peaks,solvent,freq,source
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError:
    sys.exit("RDKit required: pip install rdkit-pypi")

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "CNMR_HRE_v5.json"

# ---------- SMARTS (mirrors scripts/fix_cnmr_ground_truth.py) ----------
MOTIF_SMARTS = {
    "alkene":   "[C;!a]=[C;!a]",      # F1.1.1
    "carbonyl": "[#6X3]=[OX1]",       # F1.2.1
    "imine":    "[C]=[NX2;!a]",       # F1.3.1
    "alkyne":   "[#6]#[#6]",          # F1.5.1
    "nitrile":  "[#6]#[#7]",          # F1.6.1
}
NODE_MOTIF_MAP = {
    "F1.1.1": "alkene",
    "F1.2.1": "carbonyl",
    "F1.3.1": "imine",
    "F1.5.1": "alkyne",
    "F1.6.1": "nitrile",
}

# ---------- Conventional 13C spectral regions (ppm) ----------
# For F1.x.2 "list peaks that might represent [motif]" nodes.
# Ranges are intentionally wide; F1.x.1 is the authoritative motif-presence node.
PEAK_REGIONS = {
    "F1.1.2": (100.0, 150.0),   # alkene / aromatic sp2 C (overlap expected, filtered by motif)
    "F1.2.2": (155.0, 220.0),   # carbonyl C
    "F1.3.2": (155.0, 175.0),   # imine C (C=N)
    "F1.4.2": (100.0, 150.0),   # aromatic sp2 C
    "F1.5.2": (65.0, 95.0),     # alkyne (sp) C
    "F1.6.2": (110.0, 125.0),   # nitrile (sp C)
    "F1.7.1": None,             # F1.7.1/F1.8.1 computed after aromatic check (see fix script)
    "F1.8.1": None,
}

AROMATIC_RANGE = (100.0, 150.0)   # for F1.4.1


# ---------- helpers ----------
def compute_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    return rdMolDescriptors.CalcMolFormula(mol)


def compute_dbe(smiles: str) -> int:
    """Degree of unsaturation (Bond-equivalent count). Integer formula DBE."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    # Count atoms
    atoms = {"C": 0, "H": 0, "N": 0, "halogen": 0}
    # Use explicit+implicit H
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        if sym == "C": atoms["C"] += 1
        elif sym == "N": atoms["N"] += 1
        elif sym in ("F", "Cl", "Br", "I"): atoms["halogen"] += 1
    # Add Hs
    hs = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    # DBE = C - H/2 - X/2 + N/2 + 1
    dbe = atoms["C"] - hs/2 - atoms["halogen"]/2 + atoms["N"]/2 + 1
    return int(round(dbe))


def has_motif(smiles: str, motif: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    pat = Chem.MolFromSmarts(MOTIF_SMARTS[motif])
    return mol.HasSubstructMatch(pat)


def has_aromatic(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return any(a.GetIsAromatic() for a in mol.GetAtoms())


def peaks_in_range(peaks: List[float], lo: float, hi: float) -> List[float]:
    return [p for p in peaks if lo <= p <= hi]


# ---------- main build function ----------
def build_hre_json(
    name: str,
    smiles: str,
    peaks: List[float],
    solvent: str,
    freq_mhz: float,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce the HRE JSON dict for this molecule (writes-ready)."""
    if not Chem.MolFromSmiles(smiles):
        raise ValueError(f"Invalid SMILES: {smiles}")
    formula = compute_formula(smiles)
    dbe = compute_dbe(smiles)

    # Load template
    tmpl = json.load(open(TEMPLATE))
    out_nodes: List[Dict[str, Any]] = []

    # Populate BIG QUESTION with our data
    big_q = tmpl["nodes"][0].copy()
    big_q["label"] = (
        f"Here is the text carbon NMR spectra of Molecule X: {{Spectra:  δ "
        + ", ".join(f"{p:.2f}" for p in peaks)
        + "}}.   This NMR was taken in {{Solvent: " + solvent + "}} solvent, "
        + "on a machine with a frequency of {{MHz Value: " + f"{int(freq_mhz)}MHz" + "}}. "
        + "Molecule X has a Molecular formula of {{Molecular Formula:  " + formula + "}}. "
        + "Given Molecule X's carbon NMR and molecular formula to provide the corresponding SMILES string of the molecule."
    )
    big_q["answer"] = smiles
    out_nodes.append(big_q)

    # Motif presence (SMARTS-verified)
    motif_results: Dict[str, bool] = {}
    for nid, motif in NODE_MOTIF_MAP.items():
        motif_results[motif] = has_motif(smiles, motif)

    aromatic = has_aromatic(smiles)

    # Sp3 inferred: (C atoms not in aromatic ring) and any sp3 carbons
    mol_obj = Chem.MolFromSmiles(smiles)
    aliph_sp3_exists = any(
        a.GetSymbol() == "C"
        and not a.GetIsAromatic()
        and a.GetHybridization() == Chem.HybridizationType.SP3
        for a in mol_obj.GetAtoms()
    )
    # Heteroatom-bound sp3 C (C-X where X in O, N, S, halogen)
    c_het_sp3 = False
    for a in mol_obj.GetAtoms():
        if a.GetSymbol() == "C" and a.GetHybridization() == Chem.HybridizationType.SP3 and not a.GetIsAromatic():
            if any(nb.GetSymbol() in ("O", "N", "S", "F", "Cl", "Br", "I") for nb in a.GetNeighbors()):
                c_het_sp3 = True
                break

    # Process each template node
    for node in tmpl["nodes"][1:]:
        nid = node["id"]
        new_n = {"id": nid, "label": node.get("question") or node.get("label"), "weight": node.get("weight", 0)}
        # Scored nodes (weight=1) need answers
        w = node.get("weight", 0)
        if w == 1:
            ans = _answer_for_node(nid, smiles, peaks, dbe, motif_results, aromatic, aliph_sp3_exists, c_het_sp3)
            if ans is not None:
                new_n["answer"] = ans
        out_nodes.append(new_n)

    out = {
        "Nodes": out_nodes,
        "edges": tmpl["edges"],
        "_meta": {
            "smiles": smiles,
            "formula": formula,
            "dbe": dbe,
            "solvent": solvent,
            "freq_mhz": freq_mhz,
            "num_peaks": len(peaks),
            "source": source or "manual",
            "generator": "scripts/add_molecule.py",
        },
    }
    return out


def _answer_for_node(
    nid: str,
    smiles: str,
    peaks: List[float],
    dbe: int,
    motif_results: Dict[str, bool],
    aromatic: bool,
    aliph_sp3_exists: bool,
    c_het_sp3: bool,
) -> Optional[str]:
    """Compute ground-truth answer for a scored node."""
    # F1: unsaturation number
    if nid == "F1":
        return str(dbe)

    # F1.x.1: motif presence (Yes/No)
    if nid in NODE_MOTIF_MAP:
        return "Yes" if motif_results[NODE_MOTIF_MAP[nid]] else "No"

    # F1.4.1: aromatic present?
    if nid == "F1.4.1":
        return "Yes" if aromatic else "No"

    # F1.7.1: "IN ADDITION to aromatic, is there carbonyl/alkene/imine?"
    # Following fix_cnmr_ground_truth.py logic.
    if nid == "F1.7.1":
        if not aromatic:
            return "N/A"  # question only makes sense if aromatic
        add = motif_results.get("carbonyl") or motif_results.get("alkene") or motif_results.get("imine")
        return "Yes" if add else "No"

    # F1.8.1: is there aliphatic sp3 C?
    if nid == "F1.8.1":
        return "Yes" if aliph_sp3_exists else "No"

    # F1.x.2 peak-range nodes: list peaks in the region for that motif
    if nid in PEAK_REGIONS and PEAK_REGIONS[nid]:
        lo, hi = PEAK_REGIONS[nid]
        ps = peaks_in_range(peaks, lo, hi)
        if not ps:
            return "None"
        return ", ".join(f"{p:.2f}" for p in ps)

    # F7/F8 and other open-form nodes — left blank (model-generated at inference)
    return None


def write_molecule_file(out: Dict[str, Any], name: str, out_dir: Path, apply: bool) -> Path:
    dest = out_dir / f"{name}_CNMR_HRE.json"
    if not apply:
        print(f"[dry-run] would write {dest}")
        print(f"  formula={out['_meta']['formula']}, DBE={out['_meta']['dbe']}, "
              f"{out['_meta']['num_peaks']} peaks, source={out['_meta']['source']}")
        # quick sanity motif summary
        yn = {n['id']: n.get('answer') for n in out['Nodes']
              if n['id'] in ("F1.1.1","F1.2.1","F1.3.1","F1.4.1","F1.5.1","F1.6.1","F1.7.1","F1.8.1")}
        print(f"  motifs: {yn}")
        return dest
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ wrote {dest}")
    return dest


def parse_peaks(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", help="Molecule ID (used as filename)")
    p.add_argument("--smiles", help="Canonical SMILES")
    p.add_argument("--peaks", help="Comma-separated 13C peaks in ppm")
    p.add_argument("--solvent", default="Chloroform-d")
    p.add_argument("--freq", type=float, default=101.0, help="Spectrometer MHz")
    p.add_argument("--source", default=None, help="Provenance tag (e.g. NAPROC-13:ID12345)")
    p.add_argument("--dir", default="data/C_NMR", help="Output directory")
    p.add_argument("--csv", help="Bulk mode: CSV with columns name,smiles,peaks,solvent,freq,source")
    p.add_argument("--apply", action="store_true", help="Actually write the JSON (default: dry-run)")
    args = p.parse_args()

    out_dir = ROOT / args.dir

    if args.csv:
        rows = list(csv.DictReader(open(args.csv)))
        print(f"Loaded {len(rows)} rows from {args.csv}")
        ok = 0
        for i, row in enumerate(rows, 1):
            try:
                out = build_hre_json(
                    name=row["name"],
                    smiles=row["smiles"],
                    peaks=parse_peaks(row["peaks"]),
                    solvent=row.get("solvent") or "Chloroform-d",
                    freq_mhz=float(row.get("freq") or 101.0),
                    source=row.get("source"),
                )
                write_molecule_file(out, row["name"], out_dir, args.apply)
                ok += 1
            except Exception as e:
                print(f"[{i}/{len(rows)}] FAIL {row.get('name','?')}: {e}")
        print(f"\n{ok}/{len(rows)} succeeded")
        return

    if not (args.name and args.smiles and args.peaks):
        p.error("Need --name, --smiles, --peaks (or use --csv)")
    out = build_hre_json(
        name=args.name, smiles=args.smiles, peaks=parse_peaks(args.peaks),
        solvent=args.solvent, freq_mhz=args.freq, source=args.source,
    )
    write_molecule_file(out, args.name, out_dir, args.apply)


if __name__ == "__main__":
    main()
