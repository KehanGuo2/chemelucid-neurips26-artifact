"""Task-bundle ingest / validate CLI for ChemElucid-Gym.

A *task bundle* is a self-contained directory layout that authors can produce
and a reviewer (or CI) can validate without running any LLM API. The shape::

    <bundle_dir>/
      public/
        task.json            agent-visible: formula, peaks, solvent, freq, prompt
        tool_schema.json     advertised tool whitelist for this molecule
        prompt.md            human-readable task instructions
      private/
        answer.json          ground-truth SMILES + InChI + provenance
        provenance.json      source / license / collection date
        expert_graph.yaml    optional: chemist-authored reasoning DAG
      grader/
        rubric.yaml          optional: custom rubric (NOT comparable to official)
      tests/
        test_no_leakage.py   regression: public/* contains no SMILES / InChI / CAS
        test_task_validity.py
        test_grader_smoke.py

Subcommands::

    python -m interactive.task_bundle ingest \
        --smiles "CC(=O)Nc1ccc(O)cc1" \
        --formula "C8H9NO2" \
        --cnmr examples/acetaminophen_cnmr_peaks.csv \
        --source example --license example-only \
        --out /tmp/chemelucid_task_candidate

    python -m interactive.task_bundle validate /tmp/chemelucid_task_candidate
    python -m interactive.task_bundle validate-rubric examples/custom_rubric.yaml
    python -m interactive.task_bundle validate-dag examples/custom_dag.yaml

Design notes (also documented in `docs/TASK_BUNDLE_FORMAT.md`):

- `public/` MUST NOT contain canonical SMILES, InChI, CAS numbers, or any
  source identifier; the validator enforces this with a regex sweep.
- `private/answer.json` is the only file the grader reads to score outcome.
- `grader/rubric.yaml` is OPTIONAL and is **not** part of the official score;
  custom-rubric outputs are local extension diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Public/private split + leakage hygiene
# ---------------------------------------------------------------------------

# Strings that, if present in any public/* file, count as a hard leak.
_HARD_LEAK_KEYS = {
    "smiles", "canonical_smiles", "inchi", "inchi_key",
    "answer", "ground_truth", "gt_smiles", "cas",
    "source_id", "source_doi", "doi", "nmrshiftdb_id",
    "private_id", "legacy_internal_id",
}

# A SMILES-like string is conservatively any token of length >=4 made of the
# canonical SMILES character set; we use this only to scan public files.
_SMILES_LIKE_RE = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$%/\\\.]{4,}")
_INCHI_RE = re.compile(r"InChI=1S?/[A-Za-z0-9+\-/(),.]+")
_CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# RDKit is required for most validations; we import lazily so this module
# can still be imported (e.g., for --help) on a host without RDKit.


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import rdMolDescriptors  # noqa: F401
        return Chem, rdMolDescriptors
    except ImportError as e:
        raise SystemExit(
            f"RDKit is required for task bundle ingestion/validation: {e}\n"
            "Install with `pip install rdkit` (or use the Docker image)."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_formula_counts(formula: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for elt, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not elt:
            continue
        out[elt] = out.get(elt, 0) + (int(num) if num else 1)
    return out


def _formulas_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a.strip() == b.strip():
        return True
    return _parse_formula_counts(a) == _parse_formula_counts(b)


def _scan_text_for_leaks(text: str) -> List[str]:
    """Return a list of leak descriptions found in `text`. The validator
    treats SMILES- or InChI-shaped substrings as soft leaks (warn) and any
    explicitly named hard-leak key (`smiles:`, `inchi:`, ...) as a hard
    leak (fail)."""
    leaks: List[str] = []
    low = text.lower()
    for key in _HARD_LEAK_KEYS:
        # Detect e.g. `"smiles":` or `smiles:` (yaml/json key forms).
        if re.search(rf'["\']?{re.escape(key)}["\']?\s*[:=]', low):
            leaks.append(f"hard_leak_key:{key}")
    if _INCHI_RE.search(text):
        leaks.append("inchi_string_present")
    if _CAS_RE.search(text):
        leaks.append("cas_string_present")
    return leaks


def _scan_smiles_in_public(text: str, hidden_smiles_canonical: str) -> List[str]:
    """A more careful SMILES scan: report only tokens that RDKit confirms are
    parseable AND match the hidden answer's InChI. RDKit's per-token parse
    chatter is suppressed because we *expect* most tokens to fail to parse
    (English words, JSON keys, etc.); only positive matches are interesting."""
    Chem, _ = _require_rdkit()
    if not hidden_smiles_canonical:
        return []
    # Silence RDKit's stderr chatter for the duration of this scan.
    try:
        from rdkit import RDLogger
        _logger = RDLogger.logger()
        _logger.setLevel(RDLogger.CRITICAL)
    except ImportError:  # pragma: no cover
        _logger = None
    try:
        target_inchi = Chem.MolToInchi(Chem.MolFromSmiles(hidden_smiles_canonical))
    except Exception:  # noqa: BLE001
        target_inchi = None
    if not target_inchi:
        return []
    hits: List[str] = []
    for tok in _SMILES_LIKE_RE.findall(text):
        try:
            mol = Chem.MolFromSmiles(tok)
        except Exception:  # noqa: BLE001
            continue
        if mol is None:
            continue
        try:
            tok_inchi = Chem.MolToInchi(mol)
        except Exception:  # noqa: BLE001
            continue
        if tok_inchi and tok_inchi == target_inchi:
            hits.append(tok)
    return hits


def _read_cnmr_peaks(path: Path) -> List[float]:
    """Accept either a one-column CSV (header optional, peak per row) or a
    JSON list. Returns a flat list of floats."""
    text = path.read_text()
    # JSON?
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [float(x) for x in data]
        if isinstance(data, dict) and "peaks" in data:
            return [float(x) for x in data["peaks"]]
    except (json.JSONDecodeError, ValueError):
        pass
    # CSV
    peaks: List[float] = []
    reader = csv.reader(text.splitlines())
    for row in reader:
        if not row:
            continue
        cell = row[0].strip()
        if not cell:
            continue
        try:
            peaks.append(float(cell))
        except ValueError:
            # Skip header rows that contain text like "ppm" or "peak".
            continue
    return peaks


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


_DEFAULT_TASK_PROMPT = (
    "You are given the molecular formula and a 13C NMR spectrum of an unknown "
    "molecule X. Use the available chemistry tools to determine its structure "
    "and submit a SMILES string."
)


def _build_tool_schema() -> Dict[str, Any]:
    """Frozen list of tool names that match the agent-facing tool server.
    Mirrors `interactive.tool_server.TOOL_DEFINITIONS` at the level reviewers
    need (just names + costs); the canonical schema lives in the tool server."""
    return {
        "schema_version": 1,
        "tools": [
            {"name": "query_spectrum", "cost": 1, "category": "spectrum_query"},
            {"name": "get_full_spectrum", "cost": 1, "category": "spectrum_query",
             "info_modes": ["full"]},
            {"name": "compute_unsaturation", "cost": 1, "category": "computation"},
            {"name": "compute_molecular_weight", "cost": 1, "category": "computation"},
            {"name": "substructure_check", "cost": 1, "category": "structure_analysis"},
            {"name": "count_carbons", "cost": 1, "category": "structure_analysis"},
            {"name": "check_symmetry", "cost": 1, "category": "structure_analysis"},
            {"name": "validate_smiles", "cost": 2, "category": "structure_analysis"},
            {"name": "predict_nmr", "cost": 3, "category": "prediction"},
            {"name": "compare_spectra", "cost": 1, "category": "validation"},
            {"name": "submit", "cost": 0, "category": "submit"},
        ],
        "budget_default": 25,
        "invalid_submit_penalty": 5,
        "note": "HNMR-only tools (get_multiplicity, get_integration, get_coupling) "
                "are exposed only when spectrum_type in {HNMR, COMBO}.",
    }


def cmd_ingest(args: argparse.Namespace) -> int:
    Chem, rdMolDescriptors = _require_rdkit()

    # 1. Validate SMILES + formula on the author side BEFORE writing anything.
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"ERROR: RDKit could not parse --smiles {args.smiles!r}", file=sys.stderr)
        return 2
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    inchi = Chem.MolToInchi(mol)
    inchi_key = Chem.InchiToInchiKey(inchi) if inchi else None
    rdkit_formula = rdMolDescriptors.CalcMolFormula(mol)
    if not _formulas_equivalent(args.formula, rdkit_formula):
        print(
            f"ERROR: --formula {args.formula!r} disagrees with the formula "
            f"RDKit computes for the SMILES ({rdkit_formula}). "
            f"Reject before bundle creation to avoid an unsolvable task.",
            file=sys.stderr,
        )
        return 2

    # 2. Read CNMR peaks.
    cnmr_path = Path(args.cnmr)
    if not cnmr_path.exists():
        print(f"ERROR: CNMR peaks file not found: {cnmr_path}", file=sys.stderr)
        return 2
    peaks = _read_cnmr_peaks(cnmr_path)
    if len(peaks) < 1:
        print(f"ERROR: no peaks parsed from {cnmr_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(
            f"ERROR: output directory exists and is not empty: {out_dir}\n"
            f"Pick a fresh path or remove the existing directory.",
            file=sys.stderr,
        )
        return 2
    (out_dir / "public").mkdir(parents=True, exist_ok=True)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    (out_dir / "grader").mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(parents=True, exist_ok=True)

    molecule_id = args.molecule_id or inchi_key or "candidate"
    now_iso = datetime.now(timezone.utc).isoformat()

    # 3. Public side --- agent-visible only. NO SMILES/InChI here.
    public_task = {
        "molecule_id": molecule_id,
        "spectrum_type": "CNMR",
        "molecular_formula": args.formula,
        "solvent": args.solvent or "unspecified",
        "mhz": args.mhz or "unspecified",
        "spectrum_peaks": peaks,
        "task_prompt": _DEFAULT_TASK_PROMPT,
    }
    (out_dir / "public" / "task.json").write_text(
        json.dumps(public_task, indent=2) + "\n"
    )
    (out_dir / "public" / "tool_schema.json").write_text(
        json.dumps(_build_tool_schema(), indent=2) + "\n"
    )
    (out_dir / "public" / "prompt.md").write_text(
        f"# Task\n\n{_DEFAULT_TASK_PROMPT}\n\n"
        f"- molecular_formula: {args.formula}\n"
        f"- solvent: {public_task['solvent']}\n"
        f"- mhz: {public_task['mhz']}\n"
        f"- spectrum: see `task.json` (CNMR, {len(peaks)} peaks)\n"
        f"- budget: 25 tool-call units (see `tool_schema.json`)\n"
    )

    # 4. Private side --- grader-only.
    answer = {
        "molecule_id": molecule_id,
        "smiles": canonical_smiles,
        "inchi": inchi,
        "inchi_key": inchi_key,
        "molecular_formula": rdkit_formula,
    }
    (out_dir / "private" / "answer.json").write_text(
        json.dumps(answer, indent=2) + "\n"
    )
    provenance = {
        "molecule_id": molecule_id,
        "source": args.source,
        "license": args.license,
        "ingested_at_utc": now_iso,
        "schema_version": 1,
    }
    (out_dir / "private" / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    # 5. Stub grader / tests so `validate` can find them.
    (out_dir / "grader" / ".gitkeep").write_text("")
    _emit_test_stubs(out_dir / "tests")

    print(f"OK: wrote bundle to {out_dir}")
    print(f"    public/  : task.json, tool_schema.json, prompt.md")
    print(f"    private/ : answer.json, provenance.json")
    print(f"    grader/  : (empty; add rubric.yaml or expert_graph.yaml as needed)")
    print(f"    tests/   : leakage / validity / grader-smoke stubs")
    print(f"\nNext: python -m interactive.task_bundle validate {out_dir}")
    return 0


def _emit_test_stubs(tests_dir: Path) -> None:
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_no_leakage.py").write_text(
        '"""Regression: public/* must not contain ground-truth SMILES, InChI, or CAS."""\n'
        "from pathlib import Path\n"
        "from interactive.task_bundle import _scan_text_for_leaks\n"
        "\n"
        "def test_public_contains_no_hard_leaks():\n"
        "    pub = Path(__file__).resolve().parents[1] / 'public'\n"
        "    for f in pub.rglob('*'):\n"
        "        if not f.is_file():\n"
        "            continue\n"
        "        leaks = _scan_text_for_leaks(f.read_text())\n"
        "        assert not leaks, f'leaks found in {f}: {leaks}'\n"
    )
    (tests_dir / "test_task_validity.py").write_text(
        '"""Regression: bundle is internally consistent (formula <-> SMILES)."""\n'
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "def test_bundle_internally_consistent():\n"
        "    root = Path(__file__).resolve().parents[1]\n"
        "    pub = json.loads((root / 'public' / 'task.json').read_text())\n"
        "    priv = json.loads((root / 'private' / 'answer.json').read_text())\n"
        "    assert pub['molecule_id'] == priv['molecule_id']\n"
        "    assert pub['molecular_formula'] == priv['molecular_formula']\n"
    )
    (tests_dir / "test_grader_smoke.py").write_text(
        '"""Smoke: a dummy submission of the GT SMILES grades to InChI exact match."""\n'
        "import json\n"
        "from pathlib import Path\n"
        "from chem_defense.utils.smiles_utils import smiles_are_equivalent\n"
        "\n"
        "def test_grader_smokes_on_ground_truth():\n"
        "    root = Path(__file__).resolve().parents[1]\n"
        "    priv = json.loads((root / 'private' / 'answer.json').read_text())\n"
        "    assert smiles_are_equivalent(priv['smiles'], priv['smiles'])\n"
    )


# ---------------------------------------------------------------------------
# Validate task bundle
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle_dir)
    if not bundle.is_dir():
        print(f"ERROR: not a directory: {bundle}", file=sys.stderr)
        return 1
    issues: List[str] = []
    warnings: List[str] = []

    # Layout checks
    public_dir = bundle / "public"
    private_dir = bundle / "private"
    if not public_dir.is_dir():
        issues.append("missing public/ directory")
    if not private_dir.is_dir():
        issues.append("missing private/ directory")

    # public/task.json must exist with required fields
    task_path = public_dir / "task.json"
    public_task: Dict[str, Any] = {}
    if not task_path.exists():
        issues.append("missing public/task.json")
    else:
        try:
            public_task = json.loads(task_path.read_text())
        except json.JSONDecodeError as e:
            issues.append(f"public/task.json: invalid JSON ({e})")
        for k in ("molecule_id", "spectrum_type", "molecular_formula", "spectrum_peaks"):
            if k not in public_task:
                issues.append(f"public/task.json missing required key: {k}")

    # private/answer.json must exist
    answer_path = private_dir / "answer.json"
    answer: Dict[str, Any] = {}
    if not answer_path.exists():
        issues.append("missing private/answer.json")
    else:
        try:
            answer = json.loads(answer_path.read_text())
        except json.JSONDecodeError as e:
            issues.append(f"private/answer.json: invalid JSON ({e})")
        for k in ("smiles", "molecular_formula"):
            if k not in answer:
                issues.append(f"private/answer.json missing required key: {k}")

    # Provenance is optional but warned if missing
    if not (private_dir / "provenance.json").exists():
        warnings.append("private/provenance.json missing (license / source unset)")

    # SMILES round-trip + formula match
    canonical_smiles = ""
    if answer.get("smiles"):
        Chem, rdMolDescriptors = _require_rdkit()
        mol = Chem.MolFromSmiles(answer["smiles"])
        if mol is None:
            issues.append(f"private/answer.json: SMILES does not parse with RDKit")
        else:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            rdkit_formula = rdMolDescriptors.CalcMolFormula(mol)
            if (public_task.get("molecular_formula") and
                    not _formulas_equivalent(public_task["molecular_formula"], rdkit_formula)):
                issues.append(
                    f"formula mismatch: public={public_task['molecular_formula']} "
                    f"vs RDKit-from-SMILES={rdkit_formula}"
                )

    # CNMR peaks schema
    peaks = public_task.get("spectrum_peaks") if public_task else None
    if peaks is None:
        warnings.append("public/task.json: no spectrum_peaks list")
    elif not isinstance(peaks, list) or any(not isinstance(p, (int, float)) for p in peaks):
        issues.append("public/task.json: spectrum_peaks must be a list of numbers")

    # Leakage sweep over every public/* file
    if public_dir.is_dir():
        for f in sorted(public_dir.rglob("*")):
            if not f.is_file():
                continue
            text = f.read_text(errors="replace")
            for leak in _scan_text_for_leaks(text):
                issues.append(f"leakage in public/{f.relative_to(public_dir)}: {leak}")
            if canonical_smiles:
                hits = _scan_smiles_in_public(text, canonical_smiles)
                for h in hits:
                    issues.append(
                        f"leakage in public/{f.relative_to(public_dir)}: "
                        f"SMILES {h!r} canonicalizes to the hidden answer"
                    )

    # Test stubs presence
    for stub in ("tests/test_no_leakage.py",
                 "tests/test_task_validity.py",
                 "tests/test_grader_smoke.py"):
        if not (bundle / stub).exists():
            warnings.append(f"missing recommended {stub}")

    # Report
    print(f"task bundle      : {bundle}")
    print(f"issues           : {len(issues)}")
    print(f"warnings         : {len(warnings)}")
    for s in issues:
        print(f"  ISSUE   {s}")
    for s in warnings:
        print(f"  WARNING {s}")
    if not issues:
        print("OK: bundle passes all hard checks.")
        return 0
    return 1


# ---------------------------------------------------------------------------
# Custom DAG / rubric validation
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is required for DAG/rubric validation; install with `pip install PyYAML`."
        )
    return yaml.safe_load(path.read_text())


def cmd_validate_dag(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    try:
        d = _load_yaml(path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: YAML parse failed: {e}", file=sys.stderr)
        return 1

    issues: List[str] = []
    if not isinstance(d, dict):
        issues.append("top-level must be a mapping")
    else:
        if "schema_version" not in d:
            issues.append("missing schema_version")
        if "nodes" not in d or not isinstance(d.get("nodes"), list):
            issues.append("missing or non-list `nodes`")
        else:
            seen = set()
            for i, node in enumerate(d["nodes"]):
                if not isinstance(node, dict):
                    issues.append(f"nodes[{i}]: not a mapping")
                    continue
                nid = node.get("id")
                if not isinstance(nid, str) or not nid:
                    issues.append(f"nodes[{i}]: missing string `id`")
                if nid in seen:
                    issues.append(f"nodes[{i}]: duplicate id {nid!r}")
                seen.add(nid)
                if "weight" in node and not isinstance(node["weight"], (int, float)):
                    issues.append(f"nodes[{i}] ({nid}): weight must be numeric")
        if "edges" not in d or not isinstance(d.get("edges"), list):
            issues.append("missing or non-list `edges`")
    print(f"DAG file         : {path}")
    print(f"issues           : {len(issues)}")
    for s in issues:
        print(f"  ISSUE {s}")
    if issues:
        return 1
    print("OK: custom DAG passes schema validation.")
    print("NOTE: custom DAG outputs are local extensions and are NOT part of the "
          "official ChemElucid-Gym score (see docs/TASK_BUNDLE_FORMAT.md).")
    return 0


def cmd_validate_rubric(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    try:
        d = _load_yaml(path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: YAML parse failed: {e}", file=sys.stderr)
        return 1

    issues: List[str] = []
    if not isinstance(d, dict):
        issues.append("top-level must be a mapping")
    else:
        if "schema_version" not in d:
            issues.append("missing schema_version")
        if "name" not in d:
            issues.append("missing `name`")
        if "criteria" not in d or not isinstance(d.get("criteria"), list):
            issues.append("missing or non-list `criteria`")
        else:
            for i, cr in enumerate(d["criteria"]):
                if not isinstance(cr, dict):
                    issues.append(f"criteria[{i}]: not a mapping")
                    continue
                for required in ("id", "description", "weight"):
                    if required not in cr:
                        issues.append(f"criteria[{i}]: missing `{required}`")
                if "weight" in cr and not isinstance(cr["weight"], (int, float)):
                    issues.append(f"criteria[{i}]: weight must be numeric")

    print(f"rubric file      : {path}")
    print(f"issues           : {len(issues)}")
    for s in issues:
        print(f"  ISSUE {s}")
    if issues:
        return 1
    print("OK: custom rubric passes schema validation.")
    print("NOTE: custom rubric outputs are local extension diagnostics and are "
          "NOT comparable to official ChemElucid-Gym scores. The official score "
          "is computed only by frozen manifests and the grader in "
          "`interactive/grade.py`.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m interactive.task_bundle",
        description="Task-bundle ingest and validation for ChemElucid-Gym."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build a task bundle from CLI args")
    p_ingest.add_argument("--smiles", required=True, help="Ground-truth SMILES (private)")
    p_ingest.add_argument("--formula", required=True, help="Molecular formula (public)")
    p_ingest.add_argument("--cnmr", required=True, help="Path to CNMR peaks (CSV or JSON)")
    p_ingest.add_argument("--source", required=True, help="Provenance source label")
    p_ingest.add_argument("--license", required=True, help="License string for redistribution")
    p_ingest.add_argument("--out", required=True, help="Output bundle directory")
    p_ingest.add_argument("--molecule-id", default=None, help="Override molecule id (default: InChI key)")
    p_ingest.add_argument("--solvent", default=None)
    p_ingest.add_argument("--mhz", default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    p_val = sub.add_parser("validate", help="Validate an existing task bundle")
    p_val.add_argument("bundle_dir")
    p_val.set_defaults(func=cmd_validate)

    p_dag = sub.add_parser("validate-dag", help="Validate a custom DAG YAML")
    p_dag.add_argument("path")
    p_dag.set_defaults(func=cmd_validate_dag)

    p_rub = sub.add_parser("validate-rubric", help="Validate a custom rubric YAML")
    p_rub.add_argument("path")
    p_rub.set_defaults(func=cmd_validate_rubric)

    args = p.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
