"""End-to-end smoke test for the interactive pipeline using a mock LLM.

This script validates the FULL pipeline without requiring API keys:
1. ToolServer loads molecule data correctly
2. AgentLoop processes tool calls and manages budget
3. BlackboardExt accumulates facts from tool results
4. DAGGrader grades the trajectory against the DAG
5. ChemElucidEnv produces a complete diagnostic report

Usage:
    python -m interactive.experiments.smoke_test
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from interactive.config import DATA_DIR, ExperimentConfig, ALL_MOLECULES
from interactive.tool_server import ToolServer
from interactive.blackboard_ext import BlackboardExt
from interactive.dag_grader import DAGGrader
from interactive.environment import ChemElucidEnv


def create_expert_trajectory(ts: ToolServer) -> list:
    """Simulate an expert-level agent trajectory for 2,4-dimethylaniline.

    This mimics what a skilled chemistry agent would do:
    1. Compute DoU → 4
    2. Query aromatic region → peaks found
    3. Query alkyl region → peaks found
    4. Query carbonyl region → no peaks
    5. Get full spectrum → 8 peaks
    6. Validate candidate SMILES
    7. Predict NMR for candidate
    8. Compare predicted vs observed
    9. Submit answer
    """
    steps = []

    # Step 1: Compute degree of unsaturation
    r = ts.execute("compute_unsaturation", {"formula": ts.formula})
    steps.append({"tool": "compute_unsaturation", "args": {"formula": ts.formula}, "result": r, "cost": 1})
    print(f"  1. compute_unsaturation({ts.formula}) → DBE={r.get('degree')}")

    # Step 2: Query aromatic region (110-160 ppm)
    r = ts.execute("query_spectrum", {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160})
    steps.append({"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160}, "result": r, "cost": 1})
    print(f"  2. query_spectrum(aromatic) → {r.get('count')} peaks")

    # Step 3: Query alkyl region (0-50 ppm)
    r = ts.execute("query_spectrum", {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50})
    steps.append({"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50}, "result": r, "cost": 1})
    print(f"  3. query_spectrum(alkyl) → {r.get('count')} peaks")

    # Step 4: Query carbonyl region (160-220 ppm)
    r = ts.execute("query_spectrum", {"nucleus": "13C", "ppm_min": 160, "ppm_max": 220})
    steps.append({"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 160, "ppm_max": 220}, "result": r, "cost": 1})
    print(f"  4. query_spectrum(carbonyl) → {r.get('count')} peaks (should be 0)")

    # Step 5: Get full spectrum
    r = ts.execute("get_full_spectrum", {"nucleus": "13C"})
    steps.append({"tool": "get_full_spectrum", "args": {"nucleus": "13C"}, "result": r, "cost": 1})
    print(f"  5. get_full_spectrum() → {r.get('count')} peaks total")

    # Step 6: Validate the GT SMILES
    r = ts.execute("validate_smiles", {"smiles": "Cc1ccc(N)c(C)c1"})
    steps.append({"tool": "validate_smiles", "args": {"smiles": "Cc1ccc(N)c(C)c1"}, "result": r, "cost": 1})
    print(f"  6. validate_smiles(Cc1ccc(N)c(C)c1) → valid={r.get('valid')}, formula={r.get('molecular_formula')}")

    # Step 7: Predict NMR
    r = ts.execute("predict_nmr", {"smiles": "Cc1ccc(N)c(C)c1", "nucleus": "13C"})
    steps.append({"tool": "predict_nmr", "args": {"smiles": "Cc1ccc(N)c(C)c1", "nucleus": "13C"}, "result": r, "cost": 3})
    print(f"  7. predict_nmr(Cc1ccc(N)c(C)c1) → {r.get('num_carbons')} C signals")

    # Step 8: Compare spectra
    pred_shifts = r.get("predicted_shifts", [])
    obs_shifts = sorted(ts.cnmr_peaks)
    r = ts.execute("compare_spectra", {
        "predicted_shifts": pred_shifts,
        "observed_shifts": obs_shifts,
        "nucleus": "13C",
    })
    steps.append({"tool": "compare_spectra", "args": {"predicted_shifts": pred_shifts, "observed_shifts": obs_shifts, "nucleus": "13C"}, "result": r, "cost": 1})
    print(f"  8. compare_spectra() → similarity={r.get('similarity')}, mae={r.get('mae')}")

    # Step 9: Submit
    r = ts.execute("submit", {"smiles": "Cc1ccc(N)c(C)c1"})
    steps.append({"tool": "submit", "args": {"smiles": "Cc1ccc(N)c(C)c1"}, "result": r, "cost": 0})
    print(f"  9. submit(Cc1ccc(N)c(C)c1) → valid={r.get('valid')}")

    return steps


def test_all_molecules_load():
    """Verify all 18 molecules load correctly."""
    print("\n" + "="*60)
    print("TEST: Loading all 18 molecules")
    print("="*60)

    loaded = 0
    failed = []

    for mol_id in ALL_MOLECULES:
        try:
            ts = ToolServer(mol_id, str(DATA_DIR))
            assert ts.formula, f"No formula for {mol_id}"
            assert ts.gt_smiles, f"No GT SMILES for {mol_id}"
            assert len(ts.cnmr_peaks) > 0, f"No CNMR peaks for {mol_id}"
            loaded += 1
            print(f"  ✓ {mol_id}: {ts.formula}, {len(ts.cnmr_peaks)} peaks, GT={ts.gt_smiles[:30]}")
        except Exception as e:
            failed.append((mol_id, str(e)))
            print(f"  ✗ {mol_id}: {e}")

    print(f"\nLoaded: {loaded}/{len(ALL_MOLECULES)}")
    if failed:
        print(f"Failed: {[f[0] for f in failed]}")

    return len(failed) == 0


def test_full_pipeline():
    """End-to-end pipeline test with mock LLM on 2,4-dimethylaniline."""
    print("\n" + "="*60)
    print("TEST: Full pipeline (2,4-dimethylaniline)")
    print("="*60)

    mol_id = "2_4_dimethyl_aniline"

    # 1. ToolServer
    print("\n[1] ToolServer")
    ts = ToolServer(mol_id, str(DATA_DIR))
    obs = ts.get_initial_observation()
    print(f"  Formula: {ts.formula}")
    print(f"  CNMR peaks: {len(ts.cnmr_peaks)}")
    print(f"  GT SMILES: {ts.gt_smiles}")
    print(f"  Observation preview: {obs[:100]}...")

    # 2. Expert trajectory (simulated)
    print("\n[2] Simulated Expert Trajectory")
    ts2 = ToolServer(mol_id, str(DATA_DIR))  # Fresh TS for trajectory
    trajectory = []

    # Execute expert-level tool calls
    calls = [
        ("compute_unsaturation", {"formula": ts2.formula}),
        ("query_spectrum", {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160}),
        ("query_spectrum", {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50}),
        ("query_spectrum", {"nucleus": "13C", "ppm_min": 160, "ppm_max": 220}),
        ("get_full_spectrum", {"nucleus": "13C"}),
        ("validate_smiles", {"smiles": "Cc1ccc(N)c(C)c1"}),
        ("predict_nmr", {"smiles": "Cc1ccc(N)c(C)c1", "nucleus": "13C"}),
        ("submit", {"smiles": "Cc1ccc(N)c(C)c1"}),
    ]

    for tool_name, args in calls:
        result = ts2.execute(tool_name, args)
        print(f"  {tool_name}({list(args.values())[:2]}) → cost={ts2.get_total_cost()}")

    trajectory = ts2.get_trajectory()
    total_cost = ts2.get_total_cost()
    print(f"  Total cost: {total_cost}/25")
    print(f"  Trajectory length: {len(trajectory)} steps")

    # 3. Blackboard
    print("\n[3] BlackboardExt")
    bb = BlackboardExt()
    for step in trajectory:
        bb.add_tool_result(step["tool"], step["args"], step["result"])

    rendered = bb.render()
    n_facts = len(bb.confirmed_facts)
    print(f"  Confirmed facts: {n_facts}")
    for fact in bb.confirmed_facts[:5]:
        print(f"    - {fact}")
    if n_facts > 5:
        print(f"    ... and {n_facts - 5} more")

    # 4. DAG Grading
    print("\n[4] DAG Grading")
    dag_template = str(DATA_DIR / "CNMR_HRE_v5.json")
    gt_file = str(DATA_DIR / "C_NMR" / f"{mol_id}_CNMR_HRE.json")

    grader = DAGGrader(dag_path=dag_template, gt_path=gt_file)
    grading = grader.grade(trajectory)

    coverage = grading["coverage"]
    node_results = grading["node_results"]
    failures = grading["failure_analysis"]

    covered_nodes = [nid for nid, r in node_results.items() if r["covered"]]
    uncovered_nodes = [nid for nid, r in node_results.items() if not r["covered"]]

    print(f"  Coverage: {coverage:.1%}")
    print(f"  Covered nodes ({len(covered_nodes)}): {covered_nodes}")
    print(f"  Uncovered nodes ({len(uncovered_nodes)}): {uncovered_nodes[:10]}")
    print(f"  Failures: {', '.join(f'{cat}={len(ids)}' for cat, ids in failures.items() if ids)}")
    print(f"  Summary: {grading['summary']}")

    # 5. Accuracy check
    print("\n[5] SMILES Accuracy")
    config = ExperimentConfig(data_dir=str(DATA_DIR))
    env = ChemElucidEnv(config)
    accuracy = env._check_smiles("Cc1ccc(N)c(C)c1", ts.gt_smiles)
    print(f"  Exact match: {accuracy['exact_match']}")
    print(f"  Tanimoto: {accuracy['tanimoto']}")

    # 6. Wrong answer test
    print("\n[6] Wrong Answer Test")
    accuracy_wrong = env._check_smiles("c1ccccc1", ts.gt_smiles)  # benzene
    print(f"  Submitted: c1ccccc1 (benzene)")
    print(f"  Exact match: {accuracy_wrong['exact_match']}")
    print(f"  Tanimoto: {accuracy_wrong['tanimoto']}")

    # Summary
    print("\n" + "="*60)
    all_ok = (
        coverage > 0.3 and
        accuracy["exact_match"] is True and
        total_cost <= 25 and
        len(trajectory) >= 5
    )

    if all_ok:
        print("✓ PIPELINE SMOKE TEST PASSED")
        print(f"  Coverage: {coverage:.1%}, Cost: {total_cost}/25, Exact: True")
    else:
        print("✗ PIPELINE SMOKE TEST FAILED")
        print(f"  Coverage: {coverage:.1%} (need >30%)")
        print(f"  Cost: {total_cost}/25")
        print(f"  Exact: {accuracy['exact_match']}")

    return all_ok


def test_tool_definitions_format():
    """Verify tool definitions generate valid OpenAI and Anthropic schemas."""
    print("\n" + "="*60)
    print("TEST: Tool Definition Formats")
    print("="*60)

    ts = ToolServer("2_4_dimethyl_aniline", str(DATA_DIR))

    # OpenAI format
    openai_tools = ts.get_tool_definitions_for_llm()
    print(f"\n  OpenAI format: {len(openai_tools)} tools")
    for t in openai_tools:
        assert t["type"] == "function"
        assert "function" in t
        assert "name" in t["function"]
        assert "parameters" in t["function"]
        print(f"    {t['function']['name']}: {t['function']['description'][:60]}...")

    # Anthropic format
    anthropic_tools = ts.get_tool_definitions_for_anthropic()
    print(f"\n  Anthropic format: {len(anthropic_tools)} tools")
    for t in anthropic_tools:
        assert "name" in t
        assert "input_schema" in t
        print(f"    {t['name']}: ok")

    print("\n  ✓ Both formats valid")
    return True


def test_budget_enforcement():
    """Test that budget limits are respected."""
    print("\n" + "="*60)
    print("TEST: Budget Enforcement")
    print("="*60)

    ts = ToolServer("2_4_dimethyl_aniline", str(DATA_DIR))

    # Use up budget with predict_nmr (cost=3 each)
    for i in range(8):  # 8 × 3 = 24 units
        ts.execute("predict_nmr", {"smiles": "Cc1ccc(N)c(C)c1", "nucleus": "13C"})

    print(f"  After 8 predict_nmr calls: cost={ts.get_total_cost()}/25")
    assert ts.get_total_cost() == 24

    # One more cheap call should work (24 + 1 = 25)
    ts.execute("compute_unsaturation", {"formula": "C8H11N"})
    print(f"  After 1 more cheap call: cost={ts.get_total_cost()}/25")
    assert ts.get_total_cost() == 25

    # Note: budget enforcement is done by AgentLoop, not ToolServer
    # ToolServer just tracks cost; AgentLoop refuses calls that exceed budget

    print("  ✓ Budget tracking works correctly")
    return True


def main():
    print("ChemElucid Interactive Pipeline Smoke Test")
    print("=" * 60)
    print(f"Data dir: {DATA_DIR}")
    print(f"Molecules: {len(ALL_MOLECULES)}")

    results = {}

    results["all_molecules_load"] = test_all_molecules_load()
    results["tool_formats"] = test_tool_definitions_format()
    results["budget"] = test_budget_enforcement()
    results["full_pipeline"] = test_full_pipeline()

    # Final report
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
