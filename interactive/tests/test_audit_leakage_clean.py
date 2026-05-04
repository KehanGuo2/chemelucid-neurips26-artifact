"""CI gate: scripts/audit_leakage.py must report zero HARD_LEAK findings.

SOFT_LEAK and PUBLIC_DOC findings are warnings, not failures — they live in
the report so a human can triage. This test only fails when something on an
agent-facing surface (public task data, public manifests, scaffolds, system
prompts, tool_server source) embeds a canonical answer.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_leakage.py"


def _load_audit_module():
    if not AUDIT_PATH.exists():
        pytest.skip(f"audit script not found at {AUDIT_PATH}")
    spec = importlib.util.spec_from_file_location("audit_leakage", AUDIT_PATH)
    if spec is None or spec.loader is None:
        pytest.skip("could not load audit script")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_leakage"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_no_hard_leaks_in_agent_facing_surfaces():
    """No agent-facing surface may embed a canonical answer.

    Runs the full audit, then asserts the count of HARD_LEAK findings (after
    filtering to those actually matching a canonical answer / InChI / CAS) is
    zero. SOFT_LEAK / PUBLIC_DOC are intentionally NOT asserted on.
    """
    mod = _load_audit_module()
    audit = mod.run_audit()
    real = mod.filter_real_leaks(audit)
    hard = [f for f in real if f.severity == mod.HARD]
    if hard:
        # Build a compact diagnostic of the first 10
        lines = []
        for f in hard[:10]:
            lines.append(
                f"  {f.file}:{f.line} type={f.leak_type} "
                f"matches={f.matched_answer_for} snippet={f.snippet[:60]!r}"
            )
        rest = max(0, len(hard) - 10)
        msg = (
            f"{len(hard)} HARD_LEAK finding(s) in agent-facing surfaces:\n"
            + "\n".join(lines)
            + (f"\n  ... and {rest} more" if rest else "")
        )
        pytest.fail(msg)


def test_audit_report_is_writable():
    """Sanity: the audit reports without crashing and produces a report file."""
    mod = _load_audit_module()
    audit = mod.run_audit()
    out = REPO_ROOT / "analysis" / "leakage_audit_report.md"
    mod.write_report(audit, out)
    assert out.exists() and out.stat().st_size > 0
