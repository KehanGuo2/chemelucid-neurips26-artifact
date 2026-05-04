"""Parse nmrshiftdb2 SDF dump → filtered CSV → feed into add_molecule.py.

Input:  data/dynamic/raw/nmrshiftdb2withsignals.sd
Output: data/dynamic/nmrshiftdb_ingested.csv (name,smiles,peaks,solvent,freq,source)

Filters applied to keep benchmark instances tractable and matched to our core set:
  - Molecule has >= 1 13C spectrum with >= 5 and <= 25 unique peaks
  - SMILES parses via RDKit
  - Molecular weight in [150, 600]
  - Number of heavy atoms in [10, 50]
  - Solvent listed (preferably CDCl3/DMSO/CD3OD; we keep track not enforce)
  - Unique InChI key (dedup)
  - No metals / heavy atoms outside {H,C,N,O,S,F,Cl,Br,I,P}
"""
import csv
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except ImportError:
    sys.exit("RDKit required")

ROOT = Path(__file__).resolve().parents[1]
SDF  = ROOT / "data/dynamic/raw/nmrshiftdb2withsignals.sd"
OUT  = ROOT / "data/dynamic/nmrshiftdb_ingested.csv"

ALLOWED_ELEMENTS = {"H","C","N","O","S","F","Cl","Br","I","P"}
MIN_PEAKS = 5
MAX_PEAKS = 25
MIN_MW, MAX_MW = 150, 600
MIN_HA, MAX_HA = 10, 50

def parse_13c_spectrum(raw: str) -> List[float]:
    """Parse '17.6;0.0Q;10|18.3;0.0T;0|...' → [17.6, 18.3, ...] unique, sorted desc."""
    peaks: List[float] = []
    for item in raw.split("|"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(";")
        if len(parts) < 1:
            continue
        try:
            shift = float(parts[0])
            # Keep only reasonable 13C range
            if -10 < shift < 260:
                peaks.append(shift)
        except ValueError:
            pass
    # dedupe within 0.05 ppm and sort descending (chemist convention)
    peaks = sorted(set(round(p, 2) for p in peaks), reverse=True)
    return peaks

def ingest(apply: bool = False, limit: Optional[int] = None) -> int:
    supplier = Chem.SDMolSupplier(str(SDF), sanitize=True, removeHs=True)
    n_read = 0
    n_kept = 0
    seen_inchi = set()
    rows = []
    for mol in supplier:
        n_read += 1
        if limit and n_read > limit * 10:
            break
        if mol is None:
            continue
        # --- pull 13C spectrum ---
        spec_key = next((k for k in mol.GetPropsAsDict() if k.startswith("Spectrum 13C")), None)
        if spec_key is None:
            continue
        peaks = parse_13c_spectrum(mol.GetProp(spec_key))
        if not (MIN_PEAKS <= len(peaks) <= MAX_PEAKS):
            continue
        # --- SMILES + validation ---
        try:
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            continue
        if not smi or "." in smi:
            continue  # skip salts / mixtures
        m2 = Chem.MolFromSmiles(smi)
        if m2 is None:
            continue
        # --- element filter ---
        elems = {a.GetSymbol() for a in m2.GetAtoms()}
        if not elems.issubset(ALLOWED_ELEMENTS):
            continue
        # --- size filter ---
        mw = Descriptors.ExactMolWt(m2)
        if not (MIN_MW <= mw <= MAX_MW):
            continue
        ha = m2.GetNumHeavyAtoms()
        if not (MIN_HA <= ha <= MAX_HA):
            continue
        # --- #aromatic + #sp3 balance (ensure some complexity) ---
        n_arom = sum(1 for a in m2.GetAtoms() if a.GetIsAromatic())
        n_c = sum(1 for a in m2.GetAtoms() if a.GetSymbol() == "C")
        if n_c < 5:
            continue
        # --- dedupe by InChI key ---
        try:
            ikey = Chem.InchiToInchiKey(Chem.MolToInchi(m2))
        except Exception:
            continue
        if not ikey or ikey in seen_inchi:
            continue
        seen_inchi.add(ikey)
        # --- solvent & freq ---
        solvent = mol.GetProp("Solvent") if mol.HasProp("Solvent") else ""
        # parse "0:Chloroform-D1 (CDCl3)" → normalize
        sol_str = solvent.split(":", 1)[-1].strip() if ":" in solvent else solvent
        if "Chloroform" in sol_str or "CDCl3" in sol_str:
            sol_norm = "Chloroform-d"
        elif "DMSO" in sol_str:
            sol_norm = "DMSO-d6"
        elif "CD3OD" in sol_str or "Methanol" in sol_str:
            sol_norm = "Methanol-d4"
        else:
            sol_norm = sol_str or "Unknown"
        # freq
        freq_str = mol.GetProp("Field Strength [MHz]") if mol.HasProp("Field Strength [MHz]") else ""
        # "0:50" means spectrum-index-0 at 50 MHz; take the number after the colon
        freq_after_colon = freq_str.split(":", 1)[-1].strip() if ":" in freq_str else freq_str
        mfreq = re.search(r"(\d+(?:\.\d+)?)", freq_after_colon)
        freq = float(mfreq.group(1)) if mfreq else 100.0
        # name: use nmrshiftdb2 ID
        nid = mol.GetProp("nmrshiftdb2 ID") if mol.HasProp("nmrshiftdb2 ID") else str(n_read)
        name = f"nmrshiftdb_{nid}"
        rows.append({
            "name": name,
            "smiles": smi,
            "peaks": ",".join(f"{p:.2f}" for p in peaks),
            "solvent": sol_norm,
            "freq": freq,
            "source": f"nmrshiftdb2:{nid}",
        })
        n_kept += 1
        if limit and n_kept >= limit:
            break
    print(f"Read {n_read} mol from SDF; kept {n_kept} after filtering")
    if apply:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name","smiles","peaks","solvent","freq","source"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {OUT}")
    else:
        print("(dry run; pass --apply to write CSV)")
        for r in rows[:3]:
            print(f"  {r['name']}: {r['smiles'][:60]}  peaks={len(r['peaks'].split(','))}  solvent={r['solvent']}")
    return n_kept

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Stop after N accepted molecules")
    args = p.parse_args()
    ingest(apply=args.apply, limit=args.limit)
