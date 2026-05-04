"""Outcome-only grader for the 48-molecule withheld-probe layer.

The withheld-probe layer is graded on final-answer correctness only — there is
no expert-annotated reasoning DAG, so L2/L3 metrics do not apply. The grader
returns three scalars:

    inchi_exact   : bool   — InChI of submitted matches InChI of hidden GT.
    tanimoto      : float  — Morgan/2/2048 Tanimoto similarity in [0, 1].
    valid_smiles  : bool   — Submitted string parses with RDKit.

Reuses canonicalization and Tanimoto helpers from chem_defense.utils.smiles_utils
so canonical-form rules stay consistent with the core-set grader.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from chem_defense.utils.smiles_utils import (
    canonicalize_smiles,
    compute_tanimoto_similarity,
    smiles_are_equivalent,
)


def score_outcome_only(
    submitted_smiles: Optional[str],
    hidden_smiles: str,
) -> Dict[str, Any]:
    """Score one withheld-probe submission against the hidden ground-truth SMILES.

    Args:
        submitted_smiles: Agent-submitted SMILES (may be None or invalid).
        hidden_smiles:   Ground-truth SMILES; never exposed in the return value.

    Returns:
        {"inchi_exact": bool, "tanimoto": float, "valid_smiles": bool}.
        No GT field is included so this dict is safe to write to disk.
    """
    if not hidden_smiles:
        raise ValueError(
            "score_outcome_only requires a non-empty hidden_smiles; "
            "the withheld-probe manifest must always carry the hidden answer."
        )

    if submitted_smiles is None or not str(submitted_smiles).strip():
        return {"inchi_exact": False, "tanimoto": 0.0, "valid_smiles": False}

    submitted = str(submitted_smiles).strip()

    # Validity = RDKit can canonicalize.
    canon_sub = canonicalize_smiles(submitted)
    valid = canon_sub is not None and canon_sub != ""

    if not valid:
        return {"inchi_exact": False, "tanimoto": 0.0, "valid_smiles": False}

    inchi_exact = bool(smiles_are_equivalent(submitted, hidden_smiles))
    tanimoto = float(compute_tanimoto_similarity(submitted, hidden_smiles))

    return {
        "inchi_exact": inchi_exact,
        "tanimoto": round(tanimoto, 4),
        "valid_smiles": True,
    }
