"""Offline regrade for episodes whose trajectory is missing the submit step.

Background
----------
A bug in agent_loop.py caused early canonical episodes to terminate the agent
loop without recording a `submit` tool call in the trajectory. The DAGGrader
relies on a `submit` step to mark BIG QUESTION / F8 / F14 (final-proposal)
nodes as covered, so those episodes were under-counted. This script:

  1. Walks every episode in manifests/core18_full_matrix.jsonl whose
     status == "present" (canonical scope used by the paper).
  2. If the trajectory is non-empty, lacks a `submit` step, and the episode
     has a non-null `submitted_smiles`, append a synthetic `submit` step
     matching ToolServer.execute()'s schema (no step_idx; that key is not
     used in the existing trajectories).
  3. Re-instantiate DAGGrader with the same split-aware GT path resolution
     as interactive.environment, recompute coverage / cf_coverage / failure
     analysis, and write the regraded JSON plus a `_delta.json` summary
     into interactive/results/v3_multimodel/_regrade_submit/{condition}/.

Idempotent: re-running overwrites outputs.
No API calls.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from interactive.config import ExperimentConfig  # noqa: E402
from interactive.dag_grader import DAGGrader  # noqa: E402
from interactive.data_loader import _split_root, has_split_data  # noqa: E402

MANIFEST = REPO_ROOT / "manifests" / "core18_full_matrix.jsonl"
OUT_ROOT = REPO_ROOT / "interactive" / "results" / "v3_multimodel" / "_regrade_submit"


def _resolve_gt_path(molecule_id: str, spectrum_type: str, data_dir: Path) -> Optional[Path]:
    """Mirror interactive.environment._grade_single GT-path resolution."""
    if spectrum_type == "CNMR":
        legacy = data_dir / "C_NMR" / f"{molecule_id}_CNMR_HRE.json"
    elif spectrum_type == "HNMR":
        legacy = data_dir / "H_NMR" / f"{molecule_id}_HNMR_HRE.json"
    else:
        return None
    if has_split_data(molecule_id, spectrum_type, data_dir):
        return (
            _split_root(data_dir)
            / "private_grader_data"
            / molecule_id
            / f"grader_{spectrum_type}.json"
        )
    return legacy if legacy.exists() else None


def _make_synthetic_submit(smiles: str) -> Dict[str, Any]:
    """Construct a step matching ToolServer.execute()'s appended record.

    Schema observed in tool_server.py (call_log.append, lines 583-609):
      {tool, args, result, cost, category}
    Result mirrors _exec_submit's success branch.
    """
    return {
        "tool": "submit",
        "args": {"smiles": smiles},
        "result": {"submitted": smiles, "canonical": smiles, "valid": True},
        "cost": 0,
        "category": "submit",
    }


def _grade(
    molecule_id: str,
    spectrum_type: str,
    trajectory: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> Optional[Dict[str, Any]]:
    data_dir = Path(cfg.data_dir)
    if spectrum_type == "CNMR":
        dag_template = cfg.cnmr_dag_template
    elif spectrum_type == "HNMR":
        dag_template = cfg.hnmr_dag_template
    else:
        return None
    gt_path = _resolve_gt_path(molecule_id, spectrum_type, data_dir)
    if gt_path is None or not gt_path.exists() or not Path(dag_template).exists():
        return None
    grader = DAGGrader(dag_path=dag_template, gt_path=str(gt_path))
    return grader.grade(trajectory)


def _node_covered(graded: Dict[str, Any], node_id: str) -> Optional[bool]:
    nr = graded.get("node_results", {}).get(node_id)
    if nr is None:
        return None
    return bool(nr.get("covered"))


def _failure_category_for(graded: Dict[str, Any], node_id: str) -> Optional[str]:
    fa = graded.get("failure_analysis", {}) or {}
    for cat, ids in fa.items():
        if node_id in ids:
            return cat
    return None


def regrade_one(rec: Dict[str, Any], cfg: ExperimentConfig) -> Dict[str, Any]:
    """Regrade a single manifest record. Returns delta summary dict."""
    rel = rec["result_path"]
    src = REPO_ROOT / rel
    fname = src.name
    condition = rec["condition"]
    out_dir = OUT_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    delta_path = out_dir / fname.replace(".json", "_delta.json")

    delta: Dict[str, Any] = {
        "molecule_id": rec["molecule_id"],
        "model_key": rec["model_key"],
        "condition": condition,
        "source": rel,
        "status": "ok",
    }

    try:
        original = json.loads(src.read_text())
    except Exception as e:
        delta["status"] = "error"
        delta["error"] = f"failed to read source: {e}"
        return delta

    trajectory: List[Dict[str, Any]] = list(original.get("trajectory") or [])
    submitted = original.get("submitted_smiles")
    spectrum_type = original.get("spectrum_type", "CNMR")

    has_submit = any(s.get("tool") == "submit" for s in trajectory)
    no_submit_episode = (
        not trajectory
        or submitted is None
        or (isinstance(submitted, str) and not submitted.strip())
    )

    injected = False
    if not has_submit and not no_submit_episode:
        trajectory = trajectory + [_make_synthetic_submit(submitted)]
        injected = True

    graded = _grade(rec["molecule_id"], spectrum_type, trajectory, cfg)
    if graded is None:
        delta["status"] = "error"
        delta["error"] = f"DAG/GT not resolvable for {rec['molecule_id']}/{spectrum_type}"
        return delta

    # Old values from the original file (already on disk).
    old_coverage = original.get("coverage")
    old_cf = original.get("cf_coverage")

    # Build regraded output: copy the original, overlay new grader fields, keep
    # all other layer/accuracy fields untouched (they're not affected).
    regraded = copy.deepcopy(original)
    regraded["coverage"] = graded["coverage"]
    regraded["cf_coverage"] = graded["cf_coverage"]
    regraded["node_results"] = graded["node_results"]
    regraded["failure_analysis"] = graded["failure_analysis"]
    regraded["grading_summary"] = graded["summary"]
    regraded["trajectory"] = trajectory
    regraded["_regrade_submit_injected"] = injected
    regraded["_regrade_submit_no_submit_episode"] = no_submit_episode and not injected

    out_path.write_text(json.dumps(regraded, indent=2))

    # Per-node final-proposal toggle detection: BIG QUESTION / F8 / F14.
    # Compare old node_results (from original) to new (graded).
    final_nodes = ("BIG QUESTION", "F8", "F14")
    final_toggles = {}
    for nid in final_nodes:
        old = original.get("node_results", {}).get(nid)
        new = graded["node_results"].get(nid)
        if old is None and new is None:
            continue
        old_c = bool(old["covered"]) if old else None
        new_c = bool(new["covered"]) if new else None
        if old_c != new_c:
            final_toggles[nid] = {"old": old_c, "new": new_c}

    # Detect failure-category reclassification per node.
    old_fa = original.get("failure_analysis", {}) or {}
    new_fa = graded["failure_analysis"] or {}
    old_node_to_cat: Dict[str, str] = {}
    for cat, ids in old_fa.items():
        for nid in ids or []:
            old_node_to_cat[nid] = cat
    new_node_to_cat: Dict[str, str] = {}
    for cat, ids in new_fa.items():
        for nid in ids or []:
            new_node_to_cat[nid] = cat
    reclassifications = {}
    for nid in set(old_node_to_cat) | set(new_node_to_cat):
        o = old_node_to_cat.get(nid)
        n = new_node_to_cat.get(nid)
        if o != n:
            reclassifications[nid] = {"old": o, "new": n}

    delta.update({
        "old_coverage": old_coverage,
        "new_coverage": graded["coverage"],
        "delta_coverage": (
            None if old_coverage is None else round(graded["coverage"] - old_coverage, 6)
        ),
        "old_cf_coverage": old_cf,
        "new_cf_coverage": graded["cf_coverage"],
        "delta_cf_coverage": (
            None if old_cf is None else round(graded["cf_coverage"] - old_cf, 6)
        ),
        "old_l2": old_coverage,  # alias matching prompt vocabulary
        "new_l2": graded["coverage"],
        "delta_l2": (
            None if old_coverage is None else round(graded["coverage"] - old_coverage, 6)
        ),
        "submit_injected": injected,
        "no_submit_episode": no_submit_episode and not injected,
        "had_submit_already": has_submit,
        "submitted_smiles": submitted,
        "tanimoto_old": (original.get("accuracy") or {}).get("tanimoto"),
        "tanimoto_new": (regraded.get("accuracy") or {}).get("tanimoto"),
        "final_node_toggles": final_toggles,
        "failure_reclassifications": reclassifications,
        "spectrum_type": spectrum_type,
    })
    delta_path.write_text(json.dumps(delta, indent=2))
    return delta


def main() -> None:
    cfg = ExperimentConfig.default()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise SystemExit(f"manifest not found: {MANIFEST}")

    records = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    present = [r for r in records if r.get("status") == "present"]
    missing = [r for r in records if r.get("status") != "present"]

    print(f"manifest: {len(records)} entries, {len(present)} present, {len(missing)} missing")

    deltas: List[Dict[str, Any]] = []
    n_ok = n_injected = n_no_submit = n_already = n_err = 0
    for rec in present:
        d = regrade_one(rec, cfg)
        deltas.append(d)
        if d["status"] != "ok":
            n_err += 1
            continue
        n_ok += 1
        if d.get("submit_injected"):
            n_injected += 1
        if d.get("no_submit_episode"):
            n_no_submit += 1
        if d.get("had_submit_already"):
            n_already += 1

    # Aggregated summary file at top of _regrade_submit/.
    agg_path = OUT_ROOT / "_regrade_submit_summary.json"
    agg_path.write_text(json.dumps({
        "n_manifest": len(records),
        "n_present": len(present),
        "n_missing": len(missing),
        "n_regraded_ok": n_ok,
        "n_errored": n_err,
        "n_submit_injected": n_injected,
        "n_no_submit_episode": n_no_submit,
        "n_already_had_submit": n_already,
        "deltas": deltas,
    }, indent=2))
    print(
        f"regraded ok={n_ok}, injected={n_injected}, no_submit={n_no_submit}, "
        f"already_had_submit={n_already}, errored={n_err}"
    )
    print(f"wrote summary -> {agg_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
