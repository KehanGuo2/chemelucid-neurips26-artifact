#!/usr/bin/env python3
"""Fix false-positive motif labels in CNMR ground truth JSONs.

The original autofill script used PPM range overlap + DoU thresholds to
decide F1.x.1 (motif present? Yes/No) nodes. Because ranges overlap
heavily (alkyne 110-140 is entirely within aromatic 100-170), this
produced systematic false positives — e.g., marking alkyne=Yes for
molecules with no triple bonds.

This script uses RDKit substructure matching against the ground-truth
SMILES to verify each motif actually exists in the molecule, then
corrects the F1.x.1 answers and the dependent F1.7.1 / F1.8.1 nodes.

F1.x.2 (peak listing) nodes are NOT changed — they list peaks in the
PPM range regardless of whether the motif exists, which is correct
for the DAG's pedagogical "what might these peaks represent?" framing.

Usage:
    python scripts/fix_cnmr_ground_truth.py          # dry run
    python scripts/fix_cnmr_ground_truth.py --apply   # write changes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem

# SMARTS patterns for motif detection (non-aromatic where applicable)
MOTIF_SMARTS = {
    "alkene":   "[C;!a]=[C;!a]",   # non-aromatic C=C
    "imine":    "[C]=[NX2;!a]",    # C=N where N is not aromatic
    "alkyne":   "[#6]#[#6]",       # C≡C
    "nitrile":  "[#6]#[#7]",       # C≡N
    "carbonyl": "[#6X3]=[OX1]",    # C=O incl. aromatic-labeled carbons (ketone, aldehyde, acid, ester, amide, anhydride, aromatic lactam)
}

# Map F1.x.1 node IDs to motif names
NODE_MOTIF_MAP = {
    "F1.1.1": "alkene",
    "F1.2.1": "carbonyl",
    "F1.3.1": "imine",
    "F1.5.1": "alkyne",
    "F1.6.1": "nitrile",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "C_NMR"


def has_motif(smiles: str, motif: str) -> bool:
    """Check if molecule contains the given motif via SMARTS matching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    pattern = Chem.MolFromSmarts(MOTIF_SMARTS[motif])
    return mol.HasSubstructMatch(pattern)


def has_aromatic(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.HasSubstructMatch(Chem.MolFromSmarts("a"))


def fix_molecule(filepath: Path, apply: bool) -> list[str]:
    """Fix one molecule's ground truth. Returns list of changes made."""
    with open(filepath) as f:
        data = json.load(f)

    nodes = {n["id"]: n for n in data["Nodes"]}
    smiles = nodes["BIG QUESTION"]["answer"]

    changes = []
    mol_id = filepath.stem.replace("_CNMR_HRE", "")

    # Fix F1.x.1 motif presence nodes
    motif_results = {}
    for node_id, motif in NODE_MOTIF_MAP.items():
        if node_id not in nodes or "answer" not in nodes[node_id]:
            continue

        gt_answer = nodes[node_id]["answer"].strip()
        gt_yes = gt_answer.lower() in ("yes", "y")
        actual = has_motif(smiles, motif)
        motif_results[motif] = actual

        if gt_yes and not actual:
            new_answer = "No"
            changes.append(f"  {node_id} ({motif}): Yes -> No")
            nodes[node_id]["answer"] = new_answer
        elif not gt_yes and actual:
            new_answer = "Yes"
            changes.append(f"  {node_id} ({motif}): No -> Yes")
            nodes[node_id]["answer"] = new_answer

    # Fix F1.7.1: "carbonyl, alkene, and/or imine IN ADDITION TO aromatic?"
    if "F1.7.1" in nodes and "answer" in nodes["F1.7.1"]:
        is_aromatic = has_aromatic(smiles)
        has_carbonyl_real = motif_results.get("carbonyl", False)
        has_alkene_real = motif_results.get("alkene", False)
        has_imine_real = motif_results.get("imine", False)

        should_yes = is_aromatic and (has_carbonyl_real or has_alkene_real or has_imine_real)
        gt_yes = nodes["F1.7.1"]["answer"].strip().lower() in ("yes", "y")

        if gt_yes != should_yes:
            new_answer = "Yes" if should_yes else "No"
            changes.append(f"  F1.7.1 (carbonyl/alkene/imine + aromatic): {nodes['F1.7.1']['answer']} -> {new_answer}")
            nodes["F1.7.1"]["answer"] = new_answer

    # Fix F1.8.1: "alkyne or cyanide IN ADDITION TO aromatic?"
    if "F1.8.1" in nodes and "answer" in nodes["F1.8.1"]:
        is_aromatic = has_aromatic(smiles)
        has_alkyne_real = motif_results.get("alkyne", False)
        has_nitrile_real = motif_results.get("nitrile", False)

        should_yes = is_aromatic and (has_alkyne_real or has_nitrile_real)
        gt_yes = nodes["F1.8.1"]["answer"].strip().lower() in ("yes", "y")

        if gt_yes != should_yes:
            new_answer = "Yes" if should_yes else "No"
            changes.append(f"  F1.8.1 (alkyne/nitrile + aromatic): {nodes['F1.8.1']['answer']} -> {new_answer}")
            nodes["F1.8.1"]["answer"] = new_answer

    if changes and apply:
        # Rebuild Nodes list preserving order
        data["Nodes"] = [nodes[n["id"]] for n in data["Nodes"]]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    return changes


def main():
    parser = argparse.ArgumentParser(description="Fix CNMR ground truth false positives")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    args = parser.parse_args()

    total_changes = 0
    for filepath in sorted(DATA_DIR.glob("*_CNMR_HRE.json")):
        mol_id = filepath.stem.replace("_CNMR_HRE", "")
        changes = fix_molecule(filepath, apply=args.apply)
        if changes:
            print(f"{mol_id}:")
            for c in changes:
                print(c)
            total_changes += len(changes)

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"\n[{mode}] Total changes: {total_changes}")


if __name__ == "__main__":
    main()
