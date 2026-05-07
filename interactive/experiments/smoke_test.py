"""Review-mode smoke test for the ChemElucid artifact.

This script validates the no-API path that is available in the anonymous review
package:

1. Public task bundles load without private HRE files.
2. Tool calls are logged with the public benchmark interface.
3. L1 and L3 scoring definitions run on the logged trajectory.
4. The sanitized toy HRE example exercises L2 trajectory alignment.

Full private HRE templates, hidden labels, withheld probes, and expert graphs
are intentionally omitted from this package.
"""

from __future__ import annotations

import json

from interactive.config import ALL_MOLECULES, DATA_DIR, PROJECT_ROOT
from interactive.dag_grader import DAGGrader
from interactive.metrics.causal_analysis import compute_causal_metrics
from interactive.metrics.layer1 import compute_layer1
from interactive.tool_server import ToolServer


def test_public_tasks_load() -> bool:
    print("\n" + "=" * 60)
    print("TEST: review-mode public task loading")
    print("=" * 60)

    loaded = 0
    failed: list[tuple[str, str]] = []
    for mol_id in ALL_MOLECULES:
        try:
            ts = ToolServer(mol_id, str(DATA_DIR), spectrum_type="CNMR")
            assert ts.formula, f"No formula for {mol_id}"
            assert len(ts.cnmr_peaks) > 0, f"No CNMR peaks for {mol_id}"
            assert not ts.gt_smiles, "Private GT should be omitted in review mode"
            loaded += 1
            print(f"  OK {mol_id}: {ts.formula}, {len(ts.cnmr_peaks)} public CNMR peaks")
        except Exception as exc:
            failed.append((mol_id, str(exc)))
            print(f"  FAIL {mol_id}: {exc}")

    print(f"\nLoaded public tasks: {loaded}/{len(ALL_MOLECULES)}")
    return not failed


def test_public_tool_logging() -> bool:
    print("\n" + "=" * 60)
    print("TEST: public tool interface + L1/L3 diagnostics")
    print("=" * 60)

    ts = ToolServer("2_4_dimethyl_aniline", str(DATA_DIR), spectrum_type="CNMR")
    calls = [
        ("compute_unsaturation", {"formula": ts.formula}),
        ("query_spectrum", {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160}),
        ("query_spectrum", {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50}),
        ("query_spectrum", {"nucleus": "13C", "ppm_min": 160, "ppm_max": 220}),
        ("submit", {"smiles": "Cc1ccc(N)c(C)c1"}),
    ]

    for tool_name, args in calls:
        result = ts.execute(tool_name, args)
        print(f"  {tool_name}: cost={ts.get_total_cost()} result_keys={sorted(result)[:5]}")

    trajectory = ts.get_trajectory()
    l1 = compute_layer1(trajectory)
    l3 = compute_causal_metrics(trajectory)
    print(f"  L1 score: {l1['score']:.3f}")
    print(f"  L3 dependency rate: {l3['dependency_rate']:.3f}")

    return len(trajectory) == len(calls) and l1["score"] >= 0 and l3["dependency_rate"] >= 0


def test_toy_hre_scoring() -> bool:
    print("\n" + "=" * 60)
    print("TEST: sanitized toy HRE L2 alignment")
    print("=" * 60)

    toy_dir = PROJECT_ROOT / "examples" / "hre_toy"
    trajectory = json.loads((toy_dir / "toy_trajectory.json").read_text())
    grader = DAGGrader(
        dag_path=str(toy_dir / "toy_cnmr_template.json"),
        gt_path=str(toy_dir / "toy_grader_CNMR.json"),
    )
    result = grader.grade(trajectory)
    print(f"  Toy L2 coverage: {result['coverage']:.3f}")
    print(f"  Toy summary: {result['summary']}")
    return result["coverage"] > 0


def main() -> int:
    checks = [
        test_public_tasks_load(),
        test_public_tool_logging(),
        test_toy_hre_scoring(),
    ]

    print("\n" + "=" * 60)
    if all(checks):
        print("REVIEW-MODE SMOKE TEST PASSED")
        print("Private HRE assets are omitted by design; see examples/hre_toy/.")
        return 0

    print("REVIEW-MODE SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
