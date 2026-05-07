"""Private-grader CLI for controlled-release ChemElucid assets.

Reads a public episode log (trajectory + submitted SMILES) and grades it against
the optional private ground-truth grader files under
data_split/private_grader_data/. The anonymous review package intentionally
omits those files; in that mode the CLI exits with a clear release-boundary
message instead of exposing or fabricating hidden-grader metrics.

The CLI never prints or writes any private field (notably gt_smiles); it only
emits aggregate metrics when private assets are available.

Typical usage:

    python -m interactive.grade \
        --episode-log interactive/results/v3_multimodel/gpt-5.2_partial_autonomous/x.json \
        --manifest core \
        --out metrics.json

Exit codes:
    0 — success (metrics.json written)
    1 — molecule_id not found in the requested manifest
    2 — referenced private grader asset missing on disk
    3 — episode log malformed (missing required fields)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from interactive.config import (
    CNMR_DAG_TEMPLATE,
    HNMR_DAG_TEMPLATE,
    PROJECT_ROOT,
)
from interactive.dag_grader import DAGGrader
from interactive.metrics.causal_analysis import compute_causal_metrics
from interactive.metrics.layer1 import compute_layer1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest lookup
# ---------------------------------------------------------------------------

CORE_MANIFEST = PROJECT_ROOT / "manifests" / "core18_full_matrix.jsonl"


REVIEW_BOUNDARY_MESSAGE = (
    "Private HRE assets are intentionally omitted from the anonymous review "
    "package. This package supports inspection of the public benchmark "
    "interface, episode logging, scoring definitions, and sanitized toy HRE "
    "examples. Exact private-grader evaluation requires controlled-release "
    "assets after public paper release."
)


def _iter_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _molecule_in_manifest(molecule_id: str, manifest_path: Path) -> bool:
    """True iff manifest_path lists a row with this molecule_id."""
    if not manifest_path.exists():
        return False
    for row in _iter_jsonl(manifest_path):
        if row.get("molecule_id") == molecule_id:
            return True
    return False


def resolve_manifest(molecule_id: str, manifest_choice: str) -> Tuple[str, Path]:
    """Return (resolved_choice, manifest_path) for the given molecule.

    manifest_choice is one of: "core", "auto".
    Raises LookupError if the molecule is not in any allowed manifest.
    """
    if manifest_choice == "core":
        if _molecule_in_manifest(molecule_id, CORE_MANIFEST):
            return "core", CORE_MANIFEST
        raise LookupError(
            f"molecule_id '{molecule_id}' not found in core manifest "
            f"({CORE_MANIFEST.name})"
        )
    if manifest_choice == "auto":
        if _molecule_in_manifest(molecule_id, CORE_MANIFEST):
            return "core", CORE_MANIFEST
        raise LookupError(
            f"molecule_id '{molecule_id}' not found in the review core manifest"
        )
    raise ValueError(
        f"Unknown manifest choice '{manifest_choice}'. "
        f"Expected one of: core, auto"
    )


# ---------------------------------------------------------------------------
# Episode log + grader file plumbing
# ---------------------------------------------------------------------------

def _load_episode(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Episode log not found: {path}")
    with path.open() as fh:
        return json.load(fh)


def _grader_path(molecule_id: str, spectrum_type: str) -> Path:
    return (
        PROJECT_ROOT
        / "data_split"
        / "private_grader_data"
        / molecule_id
        / f"grader_{spectrum_type}.json"
    )


def _dag_template_for(spectrum_type: str) -> Path:
    if spectrum_type == "CNMR":
        return Path(CNMR_DAG_TEMPLATE)
    if spectrum_type == "HNMR":
        return Path(HNMR_DAG_TEMPLATE)
    raise ValueError(
        f"Expected CNMR or HNMR for spectrum_type, got '{spectrum_type}'"
    )


# ---------------------------------------------------------------------------
# Accuracy without leaking GT
# ---------------------------------------------------------------------------

def _check_accuracy(submitted: Optional[str], gt: str) -> Tuple[bool, float]:
    """Compute (exact_match, tanimoto) without exposing gt to caller-side state.

    Returns (False, 0.0) on any parsing error so the CLI never fails on bad
    SMILES — those failures show up as low metrics, not exceptions.
    """
    if not submitted or not gt:
        return False, 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit.Chem import inchi as rdkit_inchi
    except ImportError:
        # Without RDKit, fall back to literal comparison.
        match = submitted.strip() == gt.strip()
        return match, (1.0 if match else 0.0)

    mol_sub = Chem.MolFromSmiles(submitted)
    mol_gt = Chem.MolFromSmiles(gt)
    if mol_sub is None or mol_gt is None:
        return False, 0.0

    inchi_sub = rdkit_inchi.MolToInchi(mol_sub)
    inchi_gt = rdkit_inchi.MolToInchi(mol_gt)
    exact = bool(inchi_sub) and bool(inchi_gt) and inchi_sub == inchi_gt

    fp_sub = AllChem.GetMorganFingerprintAsBitVect(mol_sub, 2, nBits=2048)
    fp_gt = AllChem.GetMorganFingerprintAsBitVect(mol_gt, 2, nBits=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fp_sub, fp_gt)
    return exact, round(tanimoto, 4)


def _dominant_failure_type(failure_analysis: Dict[str, List[str]]) -> str:
    """Return the failure category with the most uncovered nodes ('none' if all covered)."""
    if not failure_analysis:
        return "none"
    ranked = sorted(
        failure_analysis.items(),
        key=lambda kv: (-len(kv[1] or []), kv[0]),
    )
    if not ranked or not ranked[0][1]:
        return "none"
    return ranked[0][0]


# ---------------------------------------------------------------------------
# Top-level grading entry
# ---------------------------------------------------------------------------

def grade_episode(episode_log_path: Path, manifest_choice: str) -> Dict[str, Any]:
    """Grade one episode log; returns the metrics dict (no private fields).

    Raises:
        LookupError: molecule not found in chosen manifest.
        FileNotFoundError: episode log or grader file missing.
        KeyError: episode log missing required fields.
    """
    episode = _load_episode(episode_log_path)

    # Validate minimum required fields up front.
    required = ["molecule_id", "trajectory"]
    missing = [k for k in required if k not in episode]
    if missing:
        raise KeyError(
            f"Episode log missing required fields: {missing}. "
            f"Path: {episode_log_path}"
        )

    molecule_id = episode["molecule_id"]
    trajectory = episode.get("trajectory") or []
    spectrum_type = episode.get("spectrum_type", "CNMR")
    submitted = episode.get("submitted_smiles")

    # Resolve manifest (errors out cleanly if unknown molecule).
    resolved_manifest, manifest_path = resolve_manifest(molecule_id, manifest_choice)

    grader_file = _grader_path(molecule_id, spectrum_type)
    if not grader_file.exists():
        raise FileNotFoundError(
            f"Grader file not found for {molecule_id}/{spectrum_type}: {grader_file}. "
            f"{REVIEW_BOUNDARY_MESSAGE}"
        )

    dag_template = _dag_template_for(spectrum_type)
    if not dag_template.exists():
        raise FileNotFoundError(
            f"DAG template not found: {dag_template}. {REVIEW_BOUNDARY_MESSAGE}"
        )

    # Load grader to extract gt_smiles for accuracy *only*; never returned.
    with grader_file.open() as fh:
        grader_data = json.load(fh)
    gt_smiles = grader_data.get("ground_truth", {}).get("smiles", "")

    # DAGGrader accepts the split private grader file directly.
    grader = DAGGrader(dag_path=str(dag_template), gt_path=str(grader_file))
    grading = grader.grade(trajectory)

    # Accuracy
    exact_match, tanimoto = _check_accuracy(submitted, gt_smiles)

    # L1 + L3
    layer1 = compute_layer1(trajectory)
    layer3 = compute_causal_metrics(trajectory)

    # Cost summary
    n_calls = len(trajectory)
    total_cost = int(episode.get("total_cost", sum(int(c.get("cost", 0)) for c in trajectory)))

    metrics = {
        "molecule_id": molecule_id,
        "manifest": resolved_manifest,
        "spectrum_type": spectrum_type,
        "model": episode.get("model"),
        "info_mode": episode.get("info_mode"),
        "scaffold": episode.get("scaffold"),
        "exact_match": bool(exact_match),
        "tanimoto": float(tanimoto),
        "n_calls": n_calls,
        "total_cost": total_cost,
        "low_tool_submit": n_calls <= 2,
        "L1": float(layer1.get("score", 0.0)),
        "L1_unique_regions": int(layer1.get("unique_regions", 0)),
        "L1_categories_used": int(layer1.get("categories_used", 0)),
        "L2": float(grading.get("coverage", 0.0)),
        "L2_cf": float(grading.get("cf_coverage", 0.0)),
        "L2_failure_type": _dominant_failure_type(grading.get("failure_analysis", {})),
        # L3 conflict_reactivity is the "pivot reactivity" metric (renamed in
        # contribution #1 spec); expose under the spec name.
        "L3_dependency_rate": float(layer3.get("dependency_rate", 0.0)),
        "L3_pivot_reactivity": float(layer3.get("conflict_reactivity", 0.0)),
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m interactive.grade",
        description=(
            "Hidden-grader: read a public episode log and emit metrics.json "
            "without exposing any private ground-truth field."
        ),
    )
    p.add_argument(
        "--episode-log", required=True, type=Path,
        help="Path to episode JSON (e.g., interactive/results/...x.json)",
    )
    p.add_argument(
        "--manifest", default="auto", choices=["core", "auto"],
        help="Which manifest to look the molecule_id up in (default: auto).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Where to write metrics.json.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        metrics = grade_episode(args.episode_log, args.manifest)
    except LookupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyError as e:
        print(f"error: malformed episode log: {e}", file=sys.stderr)
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
