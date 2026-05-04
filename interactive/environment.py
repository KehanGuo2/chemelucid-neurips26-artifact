"""Component 5: Orchestrator that ties all interactive components together.

ChemElucidEnv is the main entry point for running structure elucidation
episodes. It wires together ToolServer, AgentLoop, BlackboardExt, and DAGGrader.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from interactive.agent_loop import AgentLoop
from interactive.blackboard_ext import BlackboardExt
from interactive.config import ExperimentConfig
from interactive.dag_grader import DAGGrader
from interactive.metrics.causal_analysis import compute_causal_metrics
from interactive.metrics.layer1 import compute_layer1
from interactive.tool_server import ToolServer

logger = logging.getLogger(__name__)


class ChemElucidEnv:
    """Gym-style environment for interactive molecular structure elucidation."""

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig.default()

    def run_episode(
        self,
        molecule_id: str,
        provider: str,
        model: str,
        stateful: bool = True,
        temperature: float = 0.0,
        budget: Optional[int] = None,
        spectrum_type: Optional[str] = None,
        info_mode: str = "full",
        reasoning_scaffold: Optional[str] = None,
        probe_mode: bool = False,
    ) -> Dict[str, Any]:
        """Run a single episode for one molecule.

        Args:
            molecule_id: e.g., "2_4_dimethyl_aniline"
            provider: "openai", "anthropic", etc.
            model: Model name/ID
            stateful: Whether to use BlackboardExt (True) or stateless (False)
            temperature: LLM temperature
            budget: Tool-call budget (default from config)
            spectrum_type: "CNMR", "HNMR", or "COMBO"
            info_mode: "full" (spectrum in prompt) or "partial" (formula only)

        Returns:
            Diagnostic report dict.
        """
        budget = budget or self.config.budget
        spectrum_type = spectrum_type or self.config.spectrum_type

        start_time = time.time()

        # 1. Initialize tool server
        tool_server = ToolServer(
            molecule_id=molecule_id,
            data_dir=self.config.data_dir,
            spectrum_type=spectrum_type,
            info_mode=info_mode,
        )

        # 2. Initialize blackboard (or None for stateless)
        blackboard = BlackboardExt() if stateful else None

        # 3. Build initial observation
        observation = tool_server.get_initial_observation(info_mode=info_mode)

        # 4. Run agent loop
        agent = AgentLoop(
            provider=provider,
            model=model,
            tool_server=tool_server,
            blackboard=blackboard,
            budget=budget,
            max_turns=self.config.max_turns,
            temperature=temperature,
            reasoning_scaffold=reasoning_scaffold,
        )
        result = agent.run(observation)

        # 5. Grade with DAG (skipped in withheld-probe mode — no DAG exists).
        if probe_mode:
            grading = None
        else:
            grading = self._grade_trajectory(
                molecule_id=molecule_id,
                trajectory=result["trajectory"],
                spectrum_type=spectrum_type,
            )

        # 6. Check SMILES accuracy
        accuracy = self._check_smiles(result["submitted_smiles"], tool_server.gt_smiles)

        # 7. Compute efficiency score
        efficiency = self._compute_efficiency(result["total_cost"], budget)

        # 8. Layer 1 (information acquisition) and Layer 3 (causal topology)
        layer1 = compute_layer1(result["trajectory"])
        layer3 = compute_causal_metrics(result["trajectory"])

        elapsed = time.time() - start_time

        # 9. Compile diagnostic report. In probe_mode the DAG-derived fields are
        # all None and gt_smiles is NOT echoed (would leak hidden ground truth).
        diagnostic = {
            "molecule_id": molecule_id,
            "model": model,
            "provider": provider,
            "stateful": stateful,
            "spectrum_type": spectrum_type,
            "info_mode": info_mode,
            "probe_mode": probe_mode,
            "accuracy": accuracy if not probe_mode else self._strip_gt(accuracy),
            "efficiency": efficiency,
            "coverage": grading["coverage"] if grading else None,
            "cf_coverage": grading.get("cf_coverage") if grading else None,
            "cnmr_coverage": grading.get("cnmr_coverage") if grading else None,
            "hnmr_coverage": grading.get("hnmr_coverage") if grading else None,
            "node_results": grading["node_results"] if grading else None,
            "failure_analysis": grading["failure_analysis"] if grading else None,
            "grading_summary": grading["summary"] if grading else None,
            "submitted_smiles": result["submitted_smiles"],
            "gt_smiles": tool_server.gt_smiles if not probe_mode else None,
            "total_cost": result["total_cost"],
            "budget": budget,
            "num_turns": result["num_turns"],
            "terminated_by": result["terminated_by"],
            "trajectory": result["trajectory"],
            "layer1": layer1,
            "layer3_dependency_rate": layer3["dependency_rate"],
            "layer3_conflict_reactivity": layer3["conflict_reactivity"],
            "layer3_edges": layer3["edges"],
            "layer3_conflicts": layer3["conflicts"],
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }

        if probe_mode:
            from interactive.probe_grader import score_outcome_only
            outcome = score_outcome_only(
                result["submitted_smiles"], tool_server.gt_smiles,
            )
            diagnostic["outcome_only"] = outcome

        return diagnostic

    @staticmethod
    def _strip_gt(accuracy: Dict[str, Any]) -> Dict[str, Any]:
        """Drop any field that exposes the hidden ground-truth SMILES from a
        diagnostic dict. Used when emitting withheld-probe results to disk."""
        if not isinstance(accuracy, dict):
            return accuracy
        scrubbed = {}
        for k, v in accuracy.items():
            if k in {"gt", "gt_canonical"}:
                continue
            scrubbed[k] = v
        return scrubbed

    def _grade_trajectory(
        self,
        molecule_id: str,
        trajectory: List[Dict[str, Any]],
        spectrum_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Grade trajectory against DAG.

        For COMBO mode, grades against both CNMR and HNMR DAGs independently
        and returns a merged result with per-modality breakdowns.
        """
        if spectrum_type == "COMBO":
            return self._grade_combo(molecule_id, trajectory)

        # Single-modality grading
        return self._grade_single(molecule_id, trajectory, spectrum_type)

    def _grade_single(
        self,
        molecule_id: str,
        trajectory: List[Dict[str, Any]],
        spectrum_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Grade trajectory against a single DAG (CNMR or HNMR).

        Prefers the split private grader file
        (data_split/private_grader_data/{mol}/grader_{TYPE}.json) when present;
        falls back to the legacy single-file layout otherwise.
        """
        from interactive.data_loader import has_split_data, _split_root

        if spectrum_type == "CNMR":
            dag_template = self.config.cnmr_dag_template
            legacy_gt = Path(self.config.data_dir) / "C_NMR" / f"{molecule_id}_CNMR_HRE.json"
        elif spectrum_type == "HNMR":
            dag_template = self.config.hnmr_dag_template
            legacy_gt = Path(self.config.data_dir) / "H_NMR" / f"{molecule_id}_HNMR_HRE.json"
        else:
            logger.warning(f"Unknown spectrum_type: {spectrum_type}")
            return None

        # Prefer split private grader file when available.
        data_dir = Path(self.config.data_dir)
        if has_split_data(molecule_id, spectrum_type, data_dir):
            gt_file = (
                _split_root(data_dir)
                / "private_grader_data"
                / molecule_id
                / f"grader_{spectrum_type}.json"
            )
        else:
            gt_file = legacy_gt

        if not Path(dag_template).exists() or not gt_file.exists():
            logger.warning(f"DAG or GT file not found for {molecule_id}/{spectrum_type}")
            return None

        grader = DAGGrader(dag_path=dag_template, gt_path=str(gt_file))
        return grader.grade(trajectory)

    def _grade_combo(
        self,
        molecule_id: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Grade COMBO trajectory against both CNMR and HNMR DAGs.

        Returns a merged result with per-modality coverage and combined metrics.
        """
        cnmr_result = self._grade_single(molecule_id, trajectory, "CNMR")
        hnmr_result = self._grade_single(molecule_id, trajectory, "HNMR")

        if cnmr_result is None and hnmr_result is None:
            return None

        # Merge node_results from both modalities (prefixed to avoid collision)
        merged_nodes = {}
        if cnmr_result and cnmr_result.get("node_results"):
            for nid, val in cnmr_result["node_results"].items():
                merged_nodes[f"CNMR:{nid}"] = val
        if hnmr_result and hnmr_result.get("node_results"):
            for nid, val in hnmr_result["node_results"].items():
                merged_nodes[f"HNMR:{nid}"] = val

        # Compute combined coverage (micro-average across both DAGs)
        cnmr_cov = cnmr_result["coverage"] if cnmr_result else 0.0
        hnmr_cov = hnmr_result["coverage"] if hnmr_result else 0.0
        cnmr_n = len(cnmr_result.get("node_results", {})) if cnmr_result else 0
        hnmr_n = len(hnmr_result.get("node_results", {})) if hnmr_result else 0
        total_n = cnmr_n + hnmr_n
        combined_cov = (
            (cnmr_cov * cnmr_n + hnmr_cov * hnmr_n) / total_n
            if total_n > 0 else 0.0
        )

        # Merge failure analyses (format: {category: [node_ids]})
        failures: Dict[str, List[str]] = {}
        if cnmr_result and cnmr_result.get("failure_analysis"):
            fa = cnmr_result["failure_analysis"]
            for cat, node_ids in fa.items():
                prefixed = [f"CNMR:{nid}" for nid in node_ids]
                failures.setdefault(cat, []).extend(prefixed)
        if hnmr_result and hnmr_result.get("failure_analysis"):
            fa = hnmr_result["failure_analysis"]
            for cat, node_ids in fa.items():
                prefixed = [f"HNMR:{nid}" for nid in node_ids]
                failures.setdefault(cat, []).extend(prefixed)

        summary_parts = []
        if cnmr_result and cnmr_result.get("summary"):
            summary_parts.append(f"[CNMR] {cnmr_result['summary']}")
        if hnmr_result and hnmr_result.get("summary"):
            summary_parts.append(f"[HNMR] {hnmr_result['summary']}")

        return {
            "coverage": round(combined_cov, 4),
            "cnmr_coverage": cnmr_cov if cnmr_result else None,
            "hnmr_coverage": hnmr_cov if hnmr_result else None,
            "node_results": merged_nodes,
            "failure_analysis": failures,
            "summary": " | ".join(summary_parts),
            "cnmr_grading": cnmr_result,
            "hnmr_grading": hnmr_result,
        }

    def _check_smiles(
        self,
        submitted: Optional[str],
        gt: str,
    ) -> Dict[str, Any]:
        """Check SMILES accuracy using InChI comparison + Tanimoto similarity."""
        if not submitted:
            return {
                "exact_match": False,
                "tanimoto": 0.0,
                "submitted": None,
                "gt": gt,
                "error": "No SMILES submitted",
            }

        try:
            from rdkit import Chem
            from rdkit.Chem import inchi as rdkit_inchi
            from rdkit.Chem import DataStructs, AllChem

            mol_sub = Chem.MolFromSmiles(submitted)
            mol_gt = Chem.MolFromSmiles(gt)

            if mol_sub is None:
                return {
                    "exact_match": False,
                    "tanimoto": 0.0,
                    "submitted": submitted,
                    "gt": gt,
                    "error": "Invalid submitted SMILES",
                }

            if mol_gt is None:
                return {
                    "exact_match": False,
                    "tanimoto": 0.0,
                    "submitted": submitted,
                    "gt": gt,
                    "error": "Invalid GT SMILES",
                }

            # InChI exact match
            inchi_sub = rdkit_inchi.MolToInchi(mol_sub)
            inchi_gt = rdkit_inchi.MolToInchi(mol_gt)
            exact_match = inchi_sub is not None and inchi_gt is not None and inchi_sub == inchi_gt

            # Tanimoto similarity
            fp_sub = AllChem.GetMorganFingerprintAsBitVect(mol_sub, 2, nBits=2048)
            fp_gt = AllChem.GetMorganFingerprintAsBitVect(mol_gt, 2, nBits=2048)
            tanimoto = DataStructs.TanimotoSimilarity(fp_sub, fp_gt)

            return {
                "exact_match": exact_match,
                "tanimoto": round(tanimoto, 4),
                "submitted": submitted,
                "submitted_canonical": Chem.MolToSmiles(mol_sub, canonical=True),
                "gt": gt,
                "gt_canonical": Chem.MolToSmiles(mol_gt, canonical=True),
            }

        except ImportError:
            # Fallback: string comparison
            return {
                "exact_match": submitted.strip() == gt.strip(),
                "tanimoto": 1.0 if submitted.strip() == gt.strip() else 0.0,
                "submitted": submitted,
                "gt": gt,
                "note": "RDKit not available; using string comparison",
            }

    def _compute_efficiency(self, cost: int, budget: int) -> float:
        """Efficiency score: lower cost = higher efficiency."""
        if budget <= 0:
            return 0.0
        return round(max(0.0, 1.0 - cost / budget), 4)

    def save_diagnostic(
        self,
        diagnostic: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> str:
        """Save a diagnostic report to disk."""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        mol_id = diagnostic["molecule_id"]
        model = diagnostic["model"].replace("/", "_")
        stateful = "stateful" if diagnostic["stateful"] else "stateless"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{mol_id}_{model}_{stateful}_{ts}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(diagnostic, f, indent=2, default=str)

        logger.info(f"Saved diagnostic to {filepath}")
        return filepath
