"""Smoke test for the 48-molecule withheld-probe outcome-only pipeline.

Runs ONE probe (probe_001) end-to-end with a canned trajectory, NOT a real LLM.
Verifies:
  (a) ToolServer can read the agent-facing spectrum without seeing the hidden answer
  (b) score_outcome_only produces {inchi_exact, tanimoto, valid_smiles}
  (c) the diagnostic JSON saves correctly with no gt_smiles leak

Usage: python -m interactive.experiments.smoke_test_probe
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from interactive.config import DATA_DIR, ExperimentConfig
from interactive.data_loader import load_probe_hidden_smiles
from interactive.environment import ChemElucidEnv
from interactive.probe_grader import score_outcome_only
from interactive.tool_server import ToolServer

PROBE_ID = "probe_001"


def _canned_trajectory(ts: ToolServer) -> List[Dict[str, Any]]:
    """Drive a minimal canned trajectory through the live ToolServer.

    Three tool calls (formula DoU + one spectral query + a deliberately-wrong
    SMILES submission), enough to exercise the cost + log paths without
    needing the agent-loop API surface.
    """
    steps: List[Dict[str, Any]] = []

    r = ts.execute("compute_unsaturation", {"formula": ts.formula})
    steps.append({
        "tool": "compute_unsaturation",
        "args": {"formula": ts.formula}, "result": r, "cost": 1,
    })

    r = ts.execute("query_spectrum", {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160})
    steps.append({
        "tool": "query_spectrum",
        "args": {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160},
        "result": r, "cost": 1,
    })

    # Submit a deliberately wrong-but-parseable SMILES; the smoke test cares
    # about pipeline plumbing, not getting probe_001 right.
    wrong_smiles = "CCO"
    r = ts.execute("submit", {"smiles": wrong_smiles})
    steps.append({
        "tool": "submit",
        "args": {"smiles": wrong_smiles}, "result": r, "cost": 0,
    })
    return steps


def main() -> int:
    print("=" * 60)
    print(f"Withheld-probe smoke test: {PROBE_ID}")
    print("=" * 60)

    # -------- (a) ToolServer reads spectrum without exposing GT --------
    ts = ToolServer(
        molecule_id=PROBE_ID, data_dir=str(DATA_DIR),
        spectrum_type="CNMR", info_mode="partial",
    )
    print(f"  formula:        {ts.formula}")
    print(f"  n peaks loaded: {len(ts.cnmr_peaks)}")
    assert ts.formula, "ToolServer failed to load formula for probe spectrum"
    assert ts.cnmr_peaks, "ToolServer failed to load peaks for probe spectrum"

    hidden = load_probe_hidden_smiles(PROBE_ID, DATA_DIR)
    assert hidden, f"manifest is missing hidden SMILES for {PROBE_ID}"
    print(f"  hidden SMILES (grader-side only, length): {len(hidden)} chars")
    # ToolServer holds gt_smiles for the accuracy path but never exposes it
    # to the agent. We assert here that the manifest hidden SMILES matches.
    assert ts.gt_smiles == hidden, "ToolServer gt_smiles disagrees with probe manifest"

    # The agent-facing initial observation must NOT contain hidden SMILES.
    obs = ts.get_initial_observation(info_mode="partial")
    assert hidden not in obs, "agent-facing observation leaks the hidden SMILES"
    print("  (a) initial observation contains NO hidden SMILES — ok")

    # -------- (b) score_outcome_only returns the documented schema --------
    trajectory = _canned_trajectory(ts)
    submitted = "CCO"
    outcome = score_outcome_only(submitted, hidden)
    assert set(outcome.keys()) == {"inchi_exact", "tanimoto", "valid_smiles"}, outcome
    assert isinstance(outcome["tanimoto"], float)
    assert outcome["valid_smiles"] is True
    print(f"  (b) outcome-only score: {outcome}")

    # -------- (c) diagnostic JSON saves correctly without gt_smiles leak --------
    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=25,
        output_dir="interactive_results/withheld_probe_smoke",
        spectrum_type="CNMR",
    )
    env = ChemElucidEnv(config)

    # Build a minimal diagnostic by reusing env._check_smiles + env._strip_gt
    # rather than running the full agent loop (which needs an API key).
    accuracy = env._check_smiles(submitted, ts.gt_smiles)
    diagnostic = {
        "molecule_id": PROBE_ID,
        "model": "smoke-canned-trajectory",
        "provider": "none",
        "stateful": False,
        "spectrum_type": "CNMR",
        "info_mode": "partial",
        "probe_mode": True,
        "accuracy": env._strip_gt(accuracy),
        "submitted_smiles": submitted,
        "gt_smiles": None,  # NEVER set in probe mode
        "trajectory": trajectory,
        "total_cost": sum(int(s.get("cost", 0)) for s in trajectory),
        "budget": 25,
        "num_turns": len(trajectory),
        "terminated_by": "submit",
        "outcome_only": outcome,
    }

    saved = env.save_diagnostic(diagnostic, config.output_dir)
    print(f"  saved diagnostic -> {saved}")

    with open(saved) as fh:
        on_disk = json.load(fh)
    assert on_disk.get("gt_smiles") is None, "diagnostic on disk leaks gt_smiles"
    assert "outcome_only" in on_disk, "outcome_only missing from saved diagnostic"
    forbidden_in_accuracy = {"gt", "gt_canonical"}
    assert not (set(on_disk["accuracy"].keys()) & forbidden_in_accuracy), (
        f"accuracy block leaks GT keys: {on_disk['accuracy'].keys()}"
    )
    print("  (c) saved diagnostic is GT-clean — ok")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
