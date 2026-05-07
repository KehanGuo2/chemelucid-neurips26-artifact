"""Run the sanitized toy HRE scoring example.

This script is intentionally small and deterministic: it loads the toy
trajectory, aligns it to toy HRE nodes with DAGGrader, and computes the public
L1/L2/L3 diagnostic profile. It does not use private benchmark assets or LLM
calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interactive.dag_grader import DAGGrader
from interactive.metrics.causal_analysis import compute_causal_metrics
from interactive.metrics.layer1 import compute_layer1


HERE = Path(__file__).resolve().parent


def main() -> None:
    trajectory = json.loads((HERE / "toy_trajectory.json").read_text())

    grader = DAGGrader(
        dag_path=str(HERE / "toy_cnmr_template.json"),
        gt_path=str(HERE / "toy_grader_CNMR.json"),
    )
    l2 = grader.grade(trajectory)
    l1 = compute_layer1(trajectory)
    l3 = compute_causal_metrics(trajectory)

    report = {
        "molecule_id": "toy_cnmr_review_example",
        "private_benchmark_assets_used": False,
        "L1": l1,
        "L2": {
            "coverage": l2["coverage"],
            "summary": l2["summary"],
            "node_results": l2["node_results"],
        },
        "L3": {
            "dependency_rate": l3["dependency_rate"],
            "conflict_reactivity": l3["conflict_reactivity"],
            "edges": l3["edges"],
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
