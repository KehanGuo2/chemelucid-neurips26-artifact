"""Comprehensive leakage audit for MolElucidate-gym.

Walks every agent-facing surface in the repo and looks for ground-truth
leakage: SMILES strings that match a canonical answer, InChI strings, CAS
numbers, and IUPAC-like names.

Outputs:
  analysis/leakage_audit_report.md  (full categorized findings)
  process exit code 0 if no HARD_LEAK, 1 otherwise (when --strict).

Severity classification:
  HARD_LEAK   — leak in an agent-facing surface (public task data, public
                manifests, scaffolds, tool responses, system prompt). FAILS CI.
  SOFT_LEAK   — leak in tests / dev tooling / scripts. Warning only.
  PUBLIC_DOC  — leak in README/paper/SI drafts. Warning, flagged for review.

A "leak" is defined as any of:
  - A SMILES-shaped token whose canonical InChI matches a canonical answer.
  - A literal canonical SMILES (string-equality after canonicalization).
  - An InChI=1S/... string that matches a canonical answer.
  - A CAS number (regex \\d{2,7}-\\d{2}-\\d) — always flagged for review.

Run:
  python scripts/audit_leakage.py                  # Write report only
  python scripts/audit_leakage.py --strict         # Fail (exit 1) on HARD_LEAK
  python scripts/audit_leakage.py --hard-only      # Print only HARD_LEAK
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = REPO_ROOT / "data_split" / "public_task_data"
PRIVATE_ROOT = REPO_ROOT / "data_split" / "private_grader_data"
PRIVATE_PROVENANCE = PRIVATE_ROOT / "_private_provenance.json"

# Severity levels
HARD = "HARD_LEAK"
SOFT = "SOFT_LEAK"
DOC = "PUBLIC_DOC"

# Token-shape regex used to extract SMILES candidates from arbitrary strings.
# Mirrors the existing test_no_leakage.py heuristic so behavior stays aligned.
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\\/\.%:]+$")
_SMILES_HINT = re.compile(r"[\[\]\(\)=#/\\1-9]")
_INCHI_RE = re.compile(r"InChI=1S?/[\w\d\(\)\-/+,;.\*]+")
_CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


@dataclass
class Finding:
    severity: str
    surface: str
    file: str
    line: Optional[int]
    leak_type: str  # 'smiles' / 'inchi' / 'cas' / 'mol_id'
    snippet: str
    matched_answer_for: Optional[str] = None  # mol_id whose answer it leaks


@dataclass
class AuditResult:
    findings: List[Finding] = field(default_factory=list)
    hard: int = 0
    soft: int = 0
    doc: int = 0
    canonical_inchi_to_mol: Dict[str, str] = field(default_factory=dict)
    canonical_smiles_set: Set[str] = field(default_factory=set)
    mol_ids: Set[str] = field(default_factory=set)


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:
        return None


def _canon_inchi(Chem, smiles: str) -> Optional[str]:
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 3:
            return None
        return Chem.MolToInchi(mol)
    except Exception:
        return None


def load_canonical_answers(Chem) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """Return (inchi -> mol_id), set of canonical SMILES, set of mol_ids."""
    inchi_to_mol: Dict[str, str] = {}
    canon_smiles: Set[str] = set()
    mol_ids: Set[str] = set()

    if not PRIVATE_ROOT.exists():
        return inchi_to_mol, canon_smiles, mol_ids

    for grader in PRIVATE_ROOT.rglob("grader_*.json"):
        try:
            with open(grader) as f:
                gd = json.load(f)
        except Exception:
            continue
        mol_id = gd.get("molecule_id") or grader.parent.name
        mol_ids.add(mol_id)
        smi = gd.get("ground_truth", {}).get("smiles", "")
        if not smi:
            continue
        inchi = _canon_inchi(Chem, smi)
        if inchi:
            inchi_to_mol.setdefault(inchi, mol_id)
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon_smiles.add(Chem.MolToSmiles(mol))
        except Exception:
            pass

    # Controlled-release private provenance may list additional canonical
    # answers. It is absent from the anonymous review package.
    if PRIVATE_PROVENANCE.exists():
        try:
            with open(PRIVATE_PROVENANCE) as f:
                prov = json.load(f)
            for entry in prov.get("entries", []):
                private_id = entry.get("private_id") or entry.get("molecule_id")
                if private_id:
                    mol_ids.add(private_id)
                smi = entry.get("canonical_smiles")
                if not smi:
                    continue
                inchi = _canon_inchi(Chem, smi)
                if inchi:
                    inchi_to_mol.setdefault(inchi, private_id or "private")
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        canon_smiles.add(Chem.MolToSmiles(mol))
                except Exception:
                    pass
        except Exception:
            pass

    return inchi_to_mol, canon_smiles, mol_ids


def _candidate_tokens(s: str) -> Iterable[str]:
    for tok in re.split(r"[\s,;{}\"]+", s):
        tok = tok.strip(".,:;!?\"'`")
        if 5 <= len(tok) <= 200 and _SMILES_CHARS.match(tok) and _SMILES_HINT.search(tok):
            yield tok


def _iter_strings(obj) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def _scan_text_for_leaks(
    text: str,
    Chem,
    audit: AuditResult,
    severity: str,
    surface: str,
    file_str: str,
    line: Optional[int] = None,
) -> None:
    """Scan a string for SMILES / InChI / CAS leaks and record findings."""
    # InChI
    for m in _INCHI_RE.finditer(text):
        inchi = m.group(0)
        matched = audit.canonical_inchi_to_mol.get(inchi)
        audit.findings.append(Finding(
            severity=severity,
            surface=surface,
            file=file_str,
            line=line,
            leak_type="inchi",
            snippet=inchi[:120],
            matched_answer_for=matched,
        ))
    # CAS — always flagged (lower confidence; humans triage)
    for m in _CAS_RE.finditer(text):
        cas = m.group(0)
        # Skip obvious false positives: 4-digit years are not CAS; CAS has
        # a strict 2-7 / 2 / 1 layout. Regex already enforces this.
        audit.findings.append(Finding(
            severity=severity,
            surface=surface,
            file=file_str,
            line=line,
            leak_type="cas",
            snippet=cas,
            matched_answer_for=None,
        ))
    # SMILES tokens
    if Chem is None:
        return
    for tok in _candidate_tokens(text):
        try:
            mol = Chem.MolFromSmiles(tok)
        except Exception:
            mol = None
        if mol is None:
            continue
        if mol.GetNumHeavyAtoms() < 3:
            continue
        try:
            inchi = Chem.MolToInchi(mol)
        except Exception:
            inchi = None
        matched = audit.canonical_inchi_to_mol.get(inchi) if inchi else None
        audit.findings.append(Finding(
            severity=severity,
            surface=surface,
            file=file_str,
            line=line,
            leak_type="smiles",
            snippet=tok[:120],
            matched_answer_for=matched,
        ))


def _scan_json_file(path: Path, Chem, audit: AuditResult, severity: str, surface: str) -> None:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return
    rel = str(path.relative_to(REPO_ROOT))
    for s in _iter_strings(data):
        _scan_text_for_leaks(s, Chem, audit, severity, surface, rel)


def _scan_text_file(path: Path, Chem, audit: AuditResult, severity: str, surface: str) -> None:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                _scan_text_for_leaks(line, Chem, audit, severity, surface, str(path.relative_to(REPO_ROOT)), line=i)
    except Exception:
        pass


def _scan_jsonl_file(path: Path, Chem, audit: AuditResult, severity: str, surface: str) -> None:
    rel = str(path.relative_to(REPO_ROOT))
    try:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # Try JSON first, then plain text
                try:
                    obj = json.loads(line)
                    for s in _iter_strings(obj):
                        _scan_text_for_leaks(s, Chem, audit, severity, surface, rel, line=i)
                except json.JSONDecodeError:
                    _scan_text_for_leaks(line, Chem, audit, severity, surface, rel, line=i)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Surface walkers
# ---------------------------------------------------------------------------

def audit_public_task_data(audit: AuditResult, Chem) -> None:
    if not PUBLIC_ROOT.exists():
        return
    for path in PUBLIC_ROOT.rglob("task_*.json"):
        _scan_json_file(path, Chem, audit, severity=HARD, surface="public_task_data")


def audit_manifests(audit: AuditResult, Chem) -> None:
    mdir = REPO_ROOT / "manifests"
    if not mdir.exists():
        return
    # Public manifests must contain no leakage in the anonymous review package.
    for path in mdir.glob("*.jsonl"):
        sev = HARD
        _scan_jsonl_file(path, Chem, audit, severity=sev, surface="manifests")
    for path in mdir.glob("*.json"):
        # JSON config (not list of episodes)
        _scan_json_file(path, Chem, audit, severity=HARD, surface="manifests")
    readme = mdir / "README.md"
    if readme.exists():
        _scan_text_file(readme, Chem, audit, severity=DOC, surface="manifests/README")


def audit_scaffolds_and_prompts(audit: AuditResult, Chem) -> None:
    sc = REPO_ROOT / "interactive" / "scaffolds.py"
    if sc.exists():
        _scan_text_file(sc, Chem, audit, severity=HARD, surface="scaffolds.py")
    al = REPO_ROOT / "interactive" / "agent_loop.py"
    if al.exists():
        _scan_text_file(al, Chem, audit, severity=HARD, surface="agent_loop.py")
    bb = REPO_ROOT / "interactive" / "blackboard_ext.py"
    if bb.exists():
        _scan_text_file(bb, Chem, audit, severity=HARD, surface="blackboard_ext.py")


def audit_tool_server(audit: AuditResult, Chem) -> None:
    """Static check: tool_server.py source must not contain canonical SMILES
    string-literals that match an answer. (Tool *responses* echo agent inputs;
    that is fine. We're flagging whether the source itself embeds answers.)
    """
    ts = REPO_ROOT / "interactive" / "tool_server.py"
    if ts.exists():
        _scan_text_file(ts, Chem, audit, severity=HARD, surface="tool_server.py")


def audit_tests(audit: AuditResult, Chem) -> None:
    tdir = REPO_ROOT / "interactive" / "tests"
    if not tdir.exists():
        return
    for path in tdir.rglob("*.py"):
        # Skip the new audit-clean test itself and the no-leakage test (they
        # are about *detecting* leaks, not embedding them).
        if path.name in {"test_audit_leakage_clean.py", "test_no_leakage.py"}:
            continue
        _scan_text_file(path, Chem, audit, severity=SOFT, surface="tests")


def audit_paper_and_readme(audit: AuditResult, Chem) -> None:
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        _scan_text_file(readme, Chem, audit, severity=DOC, surface="README.md")
    pdir = REPO_ROOT / "paper"
    if pdir.exists():
        for path in list(pdir.glob("*.tex")) + list(pdir.glob("*.md")):
            _scan_text_file(path, Chem, audit, severity=DOC, surface="paper")
    # Private SI drafts should not appear in the anonymous review package; if a
    # local controlled-release tree has them, flag for review.
    si = REPO_ROOT / "data_split" / "private_si_drafts"
    if si.exists():
        for path in si.rglob("*.json"):
            _scan_json_file(path, Chem, audit, severity=DOC, surface="private_si_drafts")


def audit_scripts_and_analysis(audit: AuditResult, Chem) -> None:
    for sub in ("scripts", "analysis"):
        sdir = REPO_ROOT / sub
        if not sdir.exists():
            continue
        for path in sdir.rglob("*.py"):
            if path.name == "audit_leakage.py":
                continue
            _scan_text_file(path, Chem, audit, severity=SOFT, surface=sub)
        for path in sdir.rglob("*.md"):
            _scan_text_file(path, Chem, audit, severity=SOFT, surface=sub)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summarize(audit: AuditResult) -> None:
    audit.hard = sum(1 for f in audit.findings if f.severity == HARD)
    audit.soft = sum(1 for f in audit.findings if f.severity == SOFT)
    audit.doc = sum(1 for f in audit.findings if f.severity == DOC)


def _is_real_leak(f: Finding, audit: AuditResult) -> bool:
    """True iff finding actually matches a canonical answer.

    SMILES tokens that parse but don't match any answer (e.g. random
    SMARTS-shaped strings, sample structures unrelated to the benchmark)
    are NOT real leaks. CAS and InChI are flagged regardless.
    """
    if f.leak_type == "smiles":
        return f.matched_answer_for is not None
    if f.leak_type == "inchi":
        return f.matched_answer_for is not None
    return True  # CAS — always reviewable


def filter_real_leaks(audit: AuditResult) -> List[Finding]:
    return [f for f in audit.findings if _is_real_leak(f, audit)]


def write_report(audit: AuditResult, out_path: Path) -> None:
    real = filter_real_leaks(audit)
    hard = [f for f in real if f.severity == HARD]
    soft = [f for f in real if f.severity == SOFT]
    doc = [f for f in real if f.severity == DOC]

    lines: List[str] = []
    lines.append("# Leakage Audit Report")
    lines.append("")
    lines.append(f"- Canonical answers loaded: {len(audit.canonical_inchi_to_mol)} (by InChI)")
    lines.append(f"- Mol IDs tracked: {len(audit.mol_ids)}")
    lines.append(f"- Total candidate findings (incl. unmatched SMILES tokens): {len(audit.findings)}")
    lines.append(f"- Real leaks (match a canonical answer / InChI / CAS): {len(real)}")
    lines.append("")
    lines.append(f"## Counts by severity")
    lines.append(f"- HARD_LEAK: {len(hard)}")
    lines.append(f"- SOFT_LEAK: {len(soft)}")
    lines.append(f"- PUBLIC_DOC: {len(doc)}")
    lines.append("")

    def _section(title: str, items: List[Finding]) -> None:
        lines.append(f"## {title} ({len(items)})")
        if not items:
            lines.append("(none)")
            lines.append("")
            return
        # Group by file
        by_file: Dict[str, List[Finding]] = {}
        for f in items:
            by_file.setdefault(f.file, []).append(f)
        for file, fs in sorted(by_file.items()):
            lines.append(f"### {file}")
            for f in fs[:20]:
                line = f.line if f.line is not None else "?"
                snippet = f.snippet
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                tag = f"matches `{f.matched_answer_for}`" if f.matched_answer_for else "no-mol-match"
                lines.append(f"- L{line} [{f.leak_type}] [{tag}] `{snippet}`")
            if len(fs) > 20:
                lines.append(f"- ... and {len(fs) - 20} more")
            lines.append("")

    _section("HARD_LEAK", hard)
    _section("SOFT_LEAK", soft)
    _section("PUBLIC_DOC", doc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit() -> AuditResult:
    Chem = _load_rdkit()
    audit = AuditResult()
    if Chem is None:
        # Without RDKit we cannot canonicalize. Still scan for InChI / CAS.
        pass
    if Chem is not None:
        audit.canonical_inchi_to_mol, audit.canonical_smiles_set, audit.mol_ids = (
            load_canonical_answers(Chem)
        )
    audit_public_task_data(audit, Chem)
    audit_manifests(audit, Chem)
    audit_scaffolds_and_prompts(audit, Chem)
    audit_tool_server(audit, Chem)
    audit_tests(audit, Chem)
    audit_paper_and_readme(audit, Chem)
    audit_scripts_and_analysis(audit, Chem)
    _summarize(audit)
    return audit


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MolElucidate-gym leakage audit")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any HARD_LEAK is found.")
    p.add_argument("--hard-only", action="store_true",
                   help="Print only HARD_LEAK findings to stdout.")
    p.add_argument("--out", default=str(REPO_ROOT / "analysis" / "leakage_audit_report.md"),
                   help="Path to write Markdown report.")
    p.add_argument("--quiet", action="store_true", help="No stdout summary.")
    args = p.parse_args(argv)

    audit = run_audit()
    write_report(audit, Path(args.out))

    real = filter_real_leaks(audit)
    hard = [f for f in real if f.severity == HARD]
    soft = [f for f in real if f.severity == SOFT]
    doc = [f for f in real if f.severity == DOC]

    if not args.quiet:
        sys.stdout.write(
            f"Leakage audit: HARD={len(hard)} SOFT={len(soft)} DOC={len(doc)} "
            f"(report: {args.out})\n"
        )
        if args.hard_only:
            for f in hard[:50]:
                sys.stdout.write(
                    f"  [HARD] {f.file}:{f.line} {f.leak_type} -> "
                    f"{f.matched_answer_for}: {f.snippet[:60]}\n"
                )

    if args.strict and len(hard) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
