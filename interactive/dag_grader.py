"""Component 3: Post-hoc DAG evaluation of agent trajectories.

Given an agent's tool-call trajectory, grades it against the existing DAG
checkpoints using:
1. Deterministic tool-based matching rules (preferred)
2. LLM judge fallback for reasoning nodes (optional)

Also classifies uncovered nodes into failure categories:
- knowledge: agent lacked domain knowledge
- reasoning: agent had info but drew wrong conclusion
- tool_use: agent used tools incorrectly or inefficiently
- strategy: agent's exploration strategy was suboptimal

PPM ranges are aligned with scripts/cnmr_hre_autofill_gt.py (source of truth).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chem_defense.utils.formula_utils import parse_formula_counts as _parse_formula

# ---------------------------------------------------------------------------
# Canonical PPM ranges — must match scripts/cnmr_hre_autofill_gt.py exactly
# ---------------------------------------------------------------------------
CNMR_PPM_RANGES: Dict[str, Tuple[float, float]] = {
    "alkene":        (100, 150),   # F1.1.x
    "carbonyl":      (150, 220),   # F1.2.x
    "imine":         (150, 170),   # F1.3.x
    "aromatic":      (100, 170),   # F1.4.x
    "alkyne":        (110, 140),   # F1.5.x
    "nitrile":       (105, 135),   # F1.6.x
    "alkyl":         (8,   50),    # F3.x
    "C-heteroatom":  (30,  90),    # F5.x
    "C-halogen":     (20,  80),    # F6.x
}

# Carbonyl sub-ranges for F4 structured matching (from cnmr_hre_autofill_gt.py)
CARBONYL_SUBRANGES: Dict[str, Tuple[float, float]] = {
    "Aldehyde":         (170, 220),
    "Ketone":           (170, 220),
    "Carboxylic Acid":  (165, 185),
    "Ester":            (165, 185),
    "Amide":            (155, 185),
}

# ---------------------------------------------------------------------------
# HNMR canonical PPM ranges — must match scripts/hnmr_hre_autofill_gt.py
# ---------------------------------------------------------------------------
HNMR_PPM_RANGES: Dict[str, Tuple[float, float]] = {
    "alkene":          (4.5, 7.0),     # F1.1.x
    "aromatic":        (6.5, 8.0),     # F1.2.x
    "alkyne":          (1.5, 3.0),     # F1.3.x  (terminal alkyne C-H)
    "alkyl":           (0.25, 2.5),    # F4.x
    "aromatic_alkyl":  (2.0, 3.0),     # F4.1.1 (secondary range)
    "aldehyde":        (8.5, 10.0),    # F6.x
    "cooh":            (10.0, 12.0),   # F7.x
    "amide":           (5.0, 9.0),     # F8.x
    "hetero_c":        (3.0, 4.5),     # F9.x  (C adjacent to heteroatom)
    "singlet_hetero":  (2.0, 4.5),     # F11.1.2/F11.1.3
}


class DAGGrader:
    """Post-hoc evaluation of agent trajectory against DAG checkpoints.

    The DAG defines a series of reasoning steps (nodes) with ground-truth
    answers. We check whether the agent's tool calls effectively covered
    the knowledge that each node tests.
    """

    def __init__(self, dag_path: str, gt_path: str):
        """
        Args:
            dag_path: Path to CNMR_HRE_v5.json or HNMR_HRE_v5.json (template)
            gt_path:  Path to per-molecule ground truth JSON. Accepts either
                      the legacy single-file layout (Nodes/nodes key, with
                      BIG QUESTION label carrying formula/spectrum) or the
                      split private grader file (dag_nodes + ground_truth).
        """
        self.dag_path = Path(dag_path)
        self.gt_path = Path(gt_path)

        with open(self.dag_path) as f:
            self.dag = json.load(f)

        with open(self.gt_path) as f:
            self.gt_data = json.load(f)

        # Build GT lookup: node_id -> {label, answer, weight}.
        # Split private files use `dag_nodes`; legacy files use `Nodes`/`nodes`.
        self.gt_nodes: Dict[str, Dict[str, Any]] = {}
        gt_nodes_list = (
            self.gt_data.get("dag_nodes")
            or self.gt_data.get("Nodes")
            or self.gt_data.get("nodes", [])
        )
        for node in gt_nodes_list:
            nid = node.get("id", "")
            self.gt_nodes[nid] = {
                "label": node.get("label", ""),
                "answer": node.get("answer", ""),
                "weight": node.get("weight", 0),
            }

        # Extract formula and spectrum from BIG QUESTION
        bq = self.gt_nodes.get("BIG QUESTION", {})
        self._bq_label = bq.get("label", "")
        self._gt_smiles = bq.get("answer", "") or self.gt_data.get(
            "ground_truth", {}
        ).get("smiles", "")

        # Parse formula
        form_match = re.search(r"\{Molecular Formula:\s*([^}]+)\}", self._bq_label)
        self._formula = form_match.group(1).strip() if form_match else ""

        # Detect DAG type: HNMR vs CNMR
        dag_name = self.dag_path.stem.upper()
        gt_name = self.gt_path.stem.upper()
        self._is_hnmr = (
            "HNMR" in dag_name or "HNMR" in gt_name
            or "proton NMR" in self._bq_label.lower()
            or "proton nmr" in self._bq_label.lower()
        )

        # Parse spectrum peaks (CNMR: simple floats; HNMR: rich peaks)
        self._spectrum_peaks: List[float] = []
        self._hnmr_peaks: list = []  # HNMRPeak objects if HNMR

        if self._is_hnmr:
            self._parse_hnmr_peaks()
        else:
            spec_match = re.search(r"\{Spectra:\s*(?:δ\s*)?([\d.,\s]+)\}", self._bq_label)
            if spec_match:
                for v in spec_match.group(1).split(","):
                    v = v.strip()
                    if v:
                        try:
                            self._spectrum_peaks.append(float(v))
                        except ValueError:
                            pass

        # Parse formula counts for formula-based checks
        self._formula_counts = _parse_formula(self._formula)

        self.node_results: Dict[str, Dict[str, Any]] = {}

    def _parse_hnmr_peaks(self):
        """Parse rich HNMR peaks from BIG QUESTION label."""
        try:
            from chem_defense.utils.hnmr_parser import parse_hnmr_peaks
            spec_match = re.search(r"\{Spectra:\s*(?:δ\s*)?(.+?)\}", self._bq_label, re.DOTALL)
            if spec_match:
                raw = spec_match.group(1).strip()
                self._hnmr_peaks = parse_hnmr_peaks(raw)
                # Also populate _spectrum_peaks with ppm floats for compatibility
                self._spectrum_peaks = [p.ppm for p in self._hnmr_peaks]
        except (ImportError, Exception):
            # Fallback: extract simple ppm values
            spec_match = re.search(r"\{Spectra:\s*(?:δ\s*)?(.+?)\}", self._bq_label, re.DOTALL)
            if spec_match:
                for m in re.finditer(r"([\d.]+)\s*(?:\(|,|$)", spec_match.group(1)):
                    try:
                        self._spectrum_peaks.append(float(m.group(1)))
                    except ValueError:
                        pass

    def grade(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Grade a trajectory against DAG checkpoints.

        Args:
            trajectory: List of tool-call records from ToolServer.

        Returns:
            {
                "coverage": float (weighted),
                "node_results": {node_id: {"covered": bool, "method": str, ...}},
                "failure_analysis": {category: [node_ids]},
                "summary": str,
            }
        """
        self.node_results = {}

        # Only grade scored nodes (weight > 0)
        scored_nodes = [
            (nid, info) for nid, info in self.gt_nodes.items()
            if info.get("weight", 0) > 0
        ]

        for nid, info in scored_nodes:
            self.node_results[nid] = self._check_node(nid, info, trajectory)

        # Compute weighted coverage
        total_weight = sum(info["weight"] for _, info in scored_nodes)
        covered_weight = sum(
            self.gt_nodes[nid]["weight"]
            for nid in self.node_results
            if self.node_results[nid]["covered"]
        )
        coverage = covered_weight / total_weight if total_weight > 0 else 0.0

        # Counterfactual regrading: recompute coverage excluding nodes
        # that were only covered via get_full_spectrum (not targeted queries).
        cf_covered_weight = 0
        for nid in self.node_results:
            nr = self.node_results[nid]
            if nr["covered"]:
                details = nr.get("details", "")
                via_full = "get_full_spectrum" in details
                if not via_full:
                    cf_covered_weight += self.gt_nodes[nid]["weight"]
        cf_coverage = cf_covered_weight / total_weight if total_weight > 0 else 0.0

        # Classify failures
        failure_analysis = self._classify_failures(trajectory)

        # Summary
        covered_count = sum(1 for r in self.node_results.values() if r["covered"])
        total_count = len(scored_nodes)

        summary = (
            f"DAG Coverage: {covered_count}/{total_count} nodes "
            f"({coverage:.1%} weighted, cf={cf_coverage:.1%}). "
            f"Failures: {', '.join(f'{cat}={len(ids)}' for cat, ids in failure_analysis.items() if ids)}"
        )

        return {
            "coverage": round(coverage, 4),
            "cf_coverage": round(cf_coverage, 4),
            "node_results": self.node_results,
            "failure_analysis": failure_analysis,
            "summary": summary,
        }

    def _check_node(
        self,
        node_id: str,
        node_info: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check if a single node's checkpoint was covered by the trajectory."""
        gt_answer = node_info.get("answer", "")

        # Method 1: Deterministic tool-based matching (HNMR or CNMR)
        if self._is_hnmr:
            tool_result = self._try_tool_match_hnmr(node_id, gt_answer, trajectory)
        else:
            tool_result = self._try_tool_match(node_id, gt_answer, trajectory)
        if tool_result is not None:
            return tool_result

        # Method 2: Heuristic text matching (check if trajectory mentions relevant info)
        heuristic = self._try_heuristic_match(node_id, node_info, trajectory)
        if heuristic is not None:
            return heuristic

        # Default: not covered
        return {
            "covered": False,
            "method": "no_match",
            "details": f"No tool call or heuristic matched node {node_id}",
        }

    def _try_tool_match(
        self,
        node_id: str,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Deterministic tool-based matching rules for specific node types."""

        # F1: Degree of unsaturation (shared with HNMR)
        if node_id == "F1":
            return self._match_compute_unsaturation(gt_answer, trajectory)

        # F1.1.1: Alkene present? (Yes/No)
        if node_id == "F1.1.1":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["alkene"],
                motif_name="alkene",
            )

        # F1.2.1: Carbonyl present?
        if node_id == "F1.2.1":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["carbonyl"],
                motif_name="carbonyl",
            )

        # F1.3.1: Imine present?
        if node_id == "F1.3.1":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["imine"],
                motif_name="imine",
            )

        # F1.4.1: Aromatic ring present?
        if node_id == "F1.4.1":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["aromatic"],
                motif_name="aromatic",
            )

        # F1.5.1: Alkyne present?
        if node_id == "F1.5.1":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["alkyne"],
                motif_name="alkyne",
            )

        # F1.6.1: Nitrile present?
        if node_id == "F1.6.1":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["nitrile"],
                motif_name="nitrile",
            )

        # F2: Symmetry (Yes/No)
        if node_id == "F2":
            for step in trajectory:
                if step["tool"] == "check_symmetry":
                    has_sym = step["result"].get("has_symmetry")
                    gt_yes = gt_answer.strip().lower() in ("yes", "y")
                    if has_sym == gt_yes:
                        return {"covered": True, "method": "tool_match",
                                "details": f"check_symmetry matched GT (sym={has_sym})"}
                    else:
                        return {"covered": False, "method": "tool_match",
                                "details": f"check_symmetry returned sym={has_sym} but GT={gt_answer}"}
            return None  # Fall through to heuristic

        # F3: Alkyl carbon present?
        if node_id == "F3":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["alkyl"],
                motif_name="alkyl",
            )

        # F4: Carbonyl functional group assignment — structured matching
        if node_id == "F4":
            return self._match_carbonyl_assignment(gt_answer, trajectory)

        # F5: C adjacent to heteroatom?
        if node_id == "F5":
            return self._match_yes_no_with_spectrum_query(
                gt_answer, trajectory,
                ppm_range=CNMR_PPM_RANGES["C-heteroatom"],
                motif_name="C-heteroatom",
            )

        # F6: C adjacent to halogen?
        if node_id == "F6":
            # First: check if agent queried the halogen ppm range
            pmin, pmax = CNMR_PPM_RANGES["C-halogen"]
            gt_yes = gt_answer.strip().lower() in ("yes", "y")
            for step in trajectory:
                if step["tool"] == "query_spectrum":
                    args = step["args"]
                    if args.get("nucleus") == "13C":
                        q_min = args.get("ppm_min", 0)
                        q_max = args.get("ppm_max", 0)
                        if q_min <= pmax and q_max >= pmin:
                            peaks = step["result"].get("peaks", [])
                            has_peaks = len(peaks) > 0
                            if (gt_yes and has_peaks) or (not gt_yes and not has_peaks):
                                return {"covered": True, "method": "tool_match",
                                        "details": f"query_spectrum in C-halogen range [{q_min},{q_max}], GT={gt_answer}"}
                            else:
                                return {"covered": False, "method": "tool_match",
                                        "details": f"query_spectrum in C-halogen range, found {'peaks' if has_peaks else 'no peaks'}, but GT={gt_answer}"}
            # Fallback: check formula for halogens
            for step in trajectory:
                if step["tool"] in ("compute_unsaturation", "compute_molecular_weight"):
                    formula = step["args"].get("formula", "")
                    counts = _parse_formula(formula)
                    has_halogens = any(counts.get(e, 0) > 0 for e in ("F", "Cl", "Br", "I"))
                    if not has_halogens and not gt_yes:
                        return {"covered": True, "method": "tool_match",
                                "details": "No halogens in formula; GT=No"}
            return None

        # F1.x.2 nodes: peak labeling nodes (check if agent queried appropriate regions)
        # PPM ranges aligned with cnmr_hre_autofill_gt.py
        peak_label_nodes = {
            "F1.1.2": (*CNMR_PPM_RANGES["alkene"], "alkene"),
            "F1.2.2": (*CNMR_PPM_RANGES["carbonyl"], "carbonyl"),
            "F1.3.2": (*CNMR_PPM_RANGES["imine"], "imine"),
            "F1.4.2": (*CNMR_PPM_RANGES["aromatic"], "aromatic"),
            "F1.5.2": (*CNMR_PPM_RANGES["alkyne"], "alkyne"),
            "F1.6.2": (*CNMR_PPM_RANGES["nitrile"], "nitrile"),
            "F3.1.1": (*CNMR_PPM_RANGES["alkyl"], "alkyl"),
            "F5.1.1": (*CNMR_PPM_RANGES["C-heteroatom"], "C-heteroatom"),
            "F6.1.1": (*CNMR_PPM_RANGES["C-halogen"], "C-halogen"),
        }

        if node_id in peak_label_nodes:
            pmin, pmax, motif = peak_label_nodes[node_id]
            for step in trajectory:
                if step["tool"] == "query_spectrum":
                    args = step["args"]
                    if args.get("nucleus") == "13C":
                        q_min = args.get("ppm_min", 0)
                        q_max = args.get("ppm_max", 0)
                        # Check overlap
                        if q_min <= pmax and q_max >= pmin:
                            return {"covered": True, "method": "tool_match",
                                    "details": f"Agent queried {motif} region [{q_min},{q_max}]"}
            return None

        return None  # No rule for this node

    def _match_yes_no_with_spectrum_query(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
        ppm_range: Tuple[float, float],
        motif_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if agent queried the relevant ppm range for a yes/no motif question."""
        pmin, pmax = ppm_range
        gt_yes = gt_answer.strip().lower() in ("yes", "y")

        for step in trajectory:
            if step["tool"] == "query_spectrum":
                args = step["args"]
                if args.get("nucleus") == "13C":
                    q_min = args.get("ppm_min", 0)
                    q_max = args.get("ppm_max", 0)
                    # Check if queried range overlaps target range
                    if q_min <= pmax and q_max >= pmin:
                        # Agent queried the right region
                        peaks = step["result"].get("peaks", [])
                        has_peaks = len(peaks) > 0
                        # If GT says yes and peaks found, or GT says no and no peaks: covered
                        if (gt_yes and has_peaks) or (not gt_yes and not has_peaks):
                            return {"covered": True, "method": "tool_match",
                                    "details": f"query_spectrum in {motif_name} range, "
                                               f"found {'peaks' if has_peaks else 'no peaks'}, GT={gt_answer}"}
                        else:
                            return {"covered": False, "method": "tool_match",
                                    "details": f"query_spectrum in {motif_name} range, "
                                               f"found {'peaks' if has_peaks else 'no peaks'}, but GT={gt_answer}"}

            # Also check substructure_check for motif detection
            if step["tool"] == "substructure_check":
                pattern = str(step["args"].get("pattern", "")).lower()
                # Match pattern against motif
                motif_patterns = {
                    "aromatic": ["c1ccccc1", "aromatic", "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1"],
                    "carbonyl": ["[CX3]=O", "C=O", "carbonyl"],
                    "alkene": ["C=C", "[CX2]=[CX2]"],
                    "imine": ["C=N", "[CX2]=[NX2]"],
                    "alkyl": ["[CH3]", "[CX4]"],
                }
                if motif_name in motif_patterns:
                    for mp in motif_patterns[motif_name]:
                        if mp.lower() in pattern:
                            has_match = step["result"].get("has_match", False)
                            if has_match == gt_yes:
                                return {"covered": True, "method": "tool_match",
                                        "details": f"substructure_check for {motif_name} matched GT"}
                            else:
                                return {"covered": False, "method": "tool_match",
                                        "details": f"substructure_check: match={has_match}, GT={gt_answer}"}

        return None  # No matching tool call found

    def _match_carbonyl_assignment(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Structured matching for F4: carbonyl functional group assignment.

        Checks whether the agent queried sub-ranges that overlap with any of
        the carbonyl functional group ranges defined in CARBONYL_SUBRANGES.
        """
        # Check if agent queried any carbonyl sub-range
        for step in trajectory:
            if step["tool"] == "query_spectrum":
                args = step["args"]
                if args.get("nucleus") == "13C":
                    q_min = args.get("ppm_min", 0)
                    q_max = args.get("ppm_max", 0)
                    # Check overlap with any carbonyl sub-range
                    for group_name, (sr_min, sr_max) in CARBONYL_SUBRANGES.items():
                        if q_min <= sr_max and q_max >= sr_min:
                            return {"covered": True, "method": "tool_match",
                                    "details": f"Agent queried range [{q_min},{q_max}] "
                                               f"overlapping {group_name} ({sr_min}-{sr_max})"}
            # Also accept substructure_check for carbonyl
            if step["tool"] == "substructure_check":
                pattern = str(step["args"].get("pattern", "")).lower()
                carbonyl_patterns = ["[cx3]=o", "c=o", "carbonyl", "[ch]=o", "aldehyde", "ketone"]
                if any(cp in pattern for cp in carbonyl_patterns):
                    return {"covered": True, "method": "tool_match",
                            "details": f"substructure_check for carbonyl pattern: {pattern}"}
        return None

    # ------------------------------------------------------------------
    # HNMR-specific tool matching rules
    # ------------------------------------------------------------------

    def _try_tool_match_hnmr(
        self,
        node_id: str,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Deterministic tool-based matching rules for HNMR DAG nodes."""

        # F1: Degree of unsaturation — same as CNMR
        if node_id == "F1":
            return self._match_compute_unsaturation(gt_answer, trajectory)

        # F1.1.1: Alkene peaks in 4.5–7.0 ppm?
        if node_id == "F1.1.1":
            return self._match_hnmr_yes_no_query(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["alkene"],
                motif_name="alkene (1H)",
            )

        # F1.2.1: Aromatic peaks in 6.5–8.0 ppm?
        if node_id == "F1.2.1":
            return self._match_hnmr_yes_no_query(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["aromatic"],
                motif_name="aromatic (1H)",
            )

        # F1.3.1: Terminal alkyne peaks in 1.5–3.0 ppm?
        if node_id == "F1.3.1":
            return self._match_hnmr_yes_no_query(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["alkyne"],
                motif_name="terminal alkyne (1H)",
            )

        # F1.4.1: Compound node — alkene+aromatic (DBE>=5 AND F1.1.1=Yes AND F1.2.1=Yes)
        if node_id == "F1.4.1":
            return self._match_hnmr_compound_node(gt_answer, trajectory,
                                                   ["alkene", "aromatic"])

        # F1.5.1: Compound node — alkyne+aromatic (DBE>=6 AND F1.3.1=Yes AND F1.2.1=Yes)
        if node_id == "F1.5.1":
            return self._match_hnmr_compound_node(gt_answer, trajectory,
                                                   ["alkyne", "aromatic"])

        # F1.1.2, F1.2.2, F1.3.2: Peak extraction nodes
        hnmr_peak_nodes = {
            "F1.1.2": ("alkene", "alkene peaks (1H 4.5–7.0)"),
            "F1.1.3": ("alkene", "J-coupling analysis"),
            "F1.2.2": ("aromatic", "aromatic peaks (1H 6.5–8.0)"),
            "F1.2.3": ("aromatic", "aromatic peaks with integration"),
            "F1.3.2": ("alkyne", "alkyne peaks (1H 1.5–3.0)"),
        }
        if node_id in hnmr_peak_nodes:
            motif_key, desc = hnmr_peak_nodes[node_id]
            pmin, pmax = HNMR_PPM_RANGES[motif_key]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax, desc)

        # F2: Magnetic equivalence — check_symmetry tool
        if node_id == "F2":
            for step in trajectory:
                if step["tool"] == "check_symmetry":
                    has_sym = step["result"].get("has_symmetry")
                    gt_yes = gt_answer.strip().lower() in ("yes", "y")
                    if has_sym == gt_yes:
                        return {"covered": True, "method": "tool_match",
                                "details": f"check_symmetry matched GT (sym={has_sym})"}
                    else:
                        return {"covered": False, "method": "tool_match",
                                "details": f"check_symmetry returned sym={has_sym} but GT={gt_answer}"}
            return None

        # F2.1.1: Number of equivalent systems — check_symmetry
        if node_id == "F2.1.1":
            for step in trajectory:
                if step["tool"] == "check_symmetry":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent called check_symmetry (provides equivalent environments)"}
            return None

        # F2.1.2: Equivalent peaks labeled — needs check_symmetry + spectrum query
        if node_id == "F2.1.2":
            has_sym = any(s["tool"] == "check_symmetry" for s in trajectory)
            has_spec = any(s["tool"] in ("get_full_spectrum", "query_spectrum")
                         and s["args"].get("nucleus") == "1H" for s in trajectory)
            if has_sym and has_spec:
                return {"covered": True, "method": "tool_match",
                        "details": "Agent called check_symmetry + queried 1H spectrum"}
            return None

        # F3: Integration sum matches H count?
        if node_id == "F3":
            return self._match_hnmr_integration_check(gt_answer, trajectory)

        # F3.1.1: Difference between H count and integration sum
        if node_id == "F3.1.1":
            for step in trajectory:
                if step["tool"] == "get_integration":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent called get_integration (provides total_integration)"}
            return None

        # F3.1.2: Heteroatom list
        if node_id == "F3.1.2":
            for step in trajectory:
                if step["tool"] == "compute_unsaturation":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent analyzed formula (heteroatoms identifiable)"}
            return None

        # F4: Alkyl protons in 0.25–2.5 ppm?
        if node_id == "F4":
            return self._match_hnmr_yes_no_query(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["alkyl"],
                motif_name="alkyl (1H)",
            )

        # F4.1.1: Alkyl peaks with labels
        if node_id == "F4.1.1":
            pmin, pmax = HNMR_PPM_RANGES["alkyl"]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax, "alkyl peaks (1H 0.25–2.5)")

        # F5: Heteroatom-attached protons (singlet 1H in 1–7 ppm + O/N/S in formula)?
        if node_id == "F5":
            return self._match_hnmr_singlet_heteroatom(gt_answer, trajectory)

        # F5.1.1: Heteroatom proton peaks
        if node_id == "F5.1.1":
            # Accept any singlet query or get_multiplicity that identifies singlets
            for step in trajectory:
                if step["tool"] == "get_multiplicity":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent called get_multiplicity (can identify singlet heteroatom peaks)"}
                if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent queried 1H spectrum (heteroatom peak identification)"}
            return None

        # F5.1.2: Heteroatom assignment (which heteroatom?)
        if node_id == "F5.1.2":
            # Need both multiplicity info AND formula analysis
            has_mult = any(s["tool"] == "get_multiplicity" for s in trajectory)
            has_formula = any(s["tool"] in ("compute_unsaturation", "compute_molecular_weight")
                           for s in trajectory)
            if has_mult and has_formula:
                return {"covered": True, "method": "tool_match",
                        "details": "Agent has multiplicity + formula info for heteroatom assignment"}
            return None

        # F6: Aldehyde present? (singlet in 8.5–10 ppm + O in formula + DBE>=1)
        if node_id == "F6":
            return self._match_hnmr_yes_no_query_with_formula(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["aldehyde"],
                motif_name="aldehyde (1H)",
                required_elements=["O"],
            )

        # F6.1.1: Aldehyde peak extraction
        if node_id == "F6.1.1":
            pmin, pmax = HNMR_PPM_RANGES["aldehyde"]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax, "aldehyde peaks (1H 8.5–10)")

        # F7: Carboxylic acid present? (singlet in 10–12 ppm + O in formula + DBE>=1)
        if node_id == "F7":
            return self._match_hnmr_yes_no_query_with_formula(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["cooh"],
                motif_name="carboxylic acid (1H)",
                required_elements=["O"],
            )

        # F7.1.1: Carboxylic acid peak extraction
        if node_id == "F7.1.1":
            pmin, pmax = HNMR_PPM_RANGES["cooh"]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax, "COOH peaks (1H 10–12)")

        # F8: Amide present? (peaks in 5–9 ppm + O + N in formula + DBE>=1)
        if node_id == "F8":
            return self._match_hnmr_yes_no_query_with_formula(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["amide"],
                motif_name="amide (1H)",
                required_elements=["O", "N"],
            )

        # F8.1.1: Amide peak extraction
        if node_id == "F8.1.1":
            pmin, pmax = HNMR_PPM_RANGES["amide"]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax, "amide peaks (1H 5–9)")

        # F9: C adjacent to heteroatom (peaks in 3–4.5 ppm + O/N/S in formula)
        if node_id == "F9":
            return self._match_hnmr_yes_no_query_with_formula(
                gt_answer, trajectory,
                ppm_range=HNMR_PPM_RANGES["hetero_c"],
                motif_name="C-heteroatom adjacent (1H)",
                required_elements=["O", "N", "S"],
                require_any=True,
            )

        # F9.1.1: C-heteroatom adjacent proton peaks
        if node_id == "F9.1.1":
            pmin, pmax = HNMR_PPM_RANGES["hetero_c"]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax,
                                                "C-heteroatom peaks (1H 3–4.5)")

        # F10: Adjacent proton analysis (coupling patterns)
        if node_id == "F10":
            for step in trajectory:
                if step["tool"] == "get_coupling":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent called get_coupling for adjacent proton analysis"}
            return None

        # F11: Any singlets present?
        if node_id == "F11":
            return self._match_hnmr_singlet_check(gt_answer, trajectory)

        # F11.1.1: All singlet peaks
        if node_id == "F11.1.1":
            for step in trajectory:
                if step["tool"] == "get_multiplicity":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent called get_multiplicity (can extract singlets)"}
            return None

        # F11.1.2: Singlets in heteroatom range (2–4.5 ppm)?
        if node_id == "F11.1.2":
            for step in trajectory:
                if step["tool"] == "get_multiplicity":
                    args = step["args"]
                    q_min = args.get("ppm_min", 0)
                    q_max = args.get("ppm_max", 15)
                    pmin, pmax = HNMR_PPM_RANGES["singlet_hetero"]
                    if q_min <= pmax and q_max >= pmin:
                        return {"covered": True, "method": "tool_match",
                                "details": f"Agent called get_multiplicity in range [{q_min},{q_max}]"}
            # Also accept query_spectrum in that range
            for step in trajectory:
                if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                    q_min = step["args"].get("ppm_min", 0)
                    q_max = step["args"].get("ppm_max", 15)
                    pmin, pmax = HNMR_PPM_RANGES["singlet_hetero"]
                    if q_min <= pmax and q_max >= pmin:
                        return {"covered": True, "method": "tool_match",
                                "details": f"Agent queried 1H in singlet-heteroatom range [{q_min},{q_max}]"}
            return None

        # F11.1.3: Singlets in 2–4.5 ppm (peak extraction)
        if node_id == "F11.1.3":
            pmin, pmax = HNMR_PPM_RANGES["singlet_hetero"]
            return self._match_hnmr_peak_query(trajectory, pmin, pmax,
                                                "singlet peaks in heteroatom range (1H 2–4.5)")

        # F12: Multiple methyls (integration > 3 AND divisible by 3)?
        if node_id == "F12":
            return self._match_hnmr_integration_tool(gt_answer, trajectory,
                                                      "multiple methyls")

        # F12.1.1: Multiple methyl peaks with integration
        if node_id == "F12.1.1":
            for step in trajectory:
                if step["tool"] == "get_integration":
                    return {"covered": True, "method": "tool_match",
                            "details": "Agent called get_integration (can identify methyl groups)"}
            return None

        # F12.1.2: What group(s) do these represent
        if node_id == "F12.1.2":
            # Needs integration + multiplicity together
            has_integ = any(s["tool"] == "get_integration" for s in trajectory)
            has_mult = any(s["tool"] == "get_multiplicity" for s in trajectory)
            if has_integ and has_mult:
                return {"covered": True, "method": "tool_match",
                        "details": "Agent has integration + multiplicity for group assignment"}
            return None

        return None  # No HNMR rule for this node

    # ------------------------------------------------------------------
    # HNMR helper methods
    # ------------------------------------------------------------------

    def _match_compute_unsaturation(
        self, gt_answer: str, trajectory: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Match F1 (DoU) — shared between CNMR and HNMR."""
        gt_dbe = None
        try:
            gt_dbe = int(gt_answer.strip())
        except (ValueError, AttributeError):
            pass

        for step in trajectory:
            if step["tool"] == "compute_unsaturation":
                computed = step["result"].get("degree")
                if gt_dbe is not None and computed == gt_dbe:
                    return {"covered": True, "method": "tool_match",
                            "details": f"compute_unsaturation returned {computed} (GT={gt_dbe})"}
                elif gt_dbe is not None:
                    return {"covered": False, "method": "tool_match",
                            "details": f"compute_unsaturation returned {computed} but GT={gt_dbe}"}

        return {"covered": False, "method": "tool_match",
                "details": "Agent never called compute_unsaturation"}

    def _match_hnmr_yes_no_query(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
        ppm_range: Tuple[float, float],
        motif_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if agent queried the relevant 1H ppm range for a yes/no question."""
        pmin, pmax = ppm_range
        gt_yes = gt_answer.strip().lower() in ("yes", "y")

        for step in trajectory:
            if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                q_min = step["args"].get("ppm_min", 0)
                q_max = step["args"].get("ppm_max", 15)
                if q_min <= pmax and q_max >= pmin:
                    peaks = step["result"].get("peaks", [])
                    has_peaks = len(peaks) > 0
                    if (gt_yes and has_peaks) or (not gt_yes and not has_peaks):
                        return {"covered": True, "method": "tool_match",
                                "details": f"query_spectrum(1H) in {motif_name} range, "
                                           f"found {'peaks' if has_peaks else 'no peaks'}, GT={gt_answer}"}
                    else:
                        return {"covered": False, "method": "tool_match",
                                "details": f"query_spectrum(1H) in {motif_name} range, "
                                           f"found {'peaks' if has_peaks else 'no peaks'}, but GT={gt_answer}"}

            # Also accept get_full_spectrum if they got all peaks
            if step["tool"] == "get_full_spectrum" and step["args"].get("nucleus") == "1H":
                all_peaks = step["result"].get("peaks", [])
                # Check if any peak falls in range
                in_range = [p for p in all_peaks
                           if pmin <= (p.get("ppm", 0) if isinstance(p, dict) else 0) <= pmax]
                has_peaks = len(in_range) > 0
                if (gt_yes and has_peaks) or (not gt_yes and not has_peaks):
                    return {"covered": True, "method": "tool_match",
                            "details": f"get_full_spectrum(1H) covers {motif_name} range, GT={gt_answer}"}

        return None

    def _match_hnmr_yes_no_query_with_formula(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
        ppm_range: Tuple[float, float],
        motif_name: str,
        required_elements: List[str],
        require_any: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Check 1H query + formula element requirements for a yes/no node.

        For F6/F7/F8/F9: the GT condition requires specific elements in the formula.
        """
        pmin, pmax = ppm_range
        gt_yes = gt_answer.strip().lower() in ("yes", "y")

        # Check if agent queried the range
        for step in trajectory:
            if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                q_min = step["args"].get("ppm_min", 0)
                q_max = step["args"].get("ppm_max", 15)
                if q_min <= pmax and q_max >= pmin:
                    peaks = step["result"].get("peaks", [])
                    has_peaks = len(peaks) > 0
                    if (gt_yes and has_peaks) or (not gt_yes and not has_peaks):
                        return {"covered": True, "method": "tool_match",
                                "details": f"query_spectrum(1H) in {motif_name} range, GT={gt_answer}"}
                    else:
                        return {"covered": False, "method": "tool_match",
                                "details": f"query_spectrum(1H) in {motif_name} range, "
                                           f"found {'peaks' if has_peaks else 'no peaks'}, but GT={gt_answer}"}

        # Fallback: If GT=No and formula lacks required elements, agent can infer from formula
        if not gt_yes:
            if require_any:
                has_elements = any(self._formula_counts.get(e, 0) > 0 for e in required_elements)
            else:
                has_elements = all(self._formula_counts.get(e, 0) > 0 for e in required_elements)

            if not has_elements:
                for step in trajectory:
                    if step["tool"] in ("compute_unsaturation", "compute_molecular_weight"):
                        return {"covered": True, "method": "tool_match",
                                "details": f"GT=No; formula lacks {required_elements}; "
                                           f"agent analyzed formula"}
        return None

    def _match_hnmr_peak_query(
        self,
        trajectory: List[Dict[str, Any]],
        ppm_min: float,
        ppm_max: float,
        desc: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if agent queried a 1H ppm range (for peak extraction nodes)."""
        for step in trajectory:
            if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                q_min = step["args"].get("ppm_min", 0)
                q_max = step["args"].get("ppm_max", 15)
                if q_min <= ppm_max and q_max >= ppm_min:
                    return {"covered": True, "method": "tool_match",
                            "details": f"Agent queried {desc} [{q_min},{q_max}]"}
            # Also accept get_full_spectrum
            if step["tool"] == "get_full_spectrum" and step["args"].get("nucleus") == "1H":
                return {"covered": True, "method": "tool_match",
                        "details": f"Agent got full 1H spectrum (covers {desc})"}
        return None

    def _match_hnmr_compound_node(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
        required_motifs: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Check compound HNMR nodes (F1.4.1, F1.5.1) that require multiple region queries."""
        queried_motifs = set()
        for step in trajectory:
            if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                q_min = step["args"].get("ppm_min", 0)
                q_max = step["args"].get("ppm_max", 15)
                for motif in required_motifs:
                    pmin, pmax = HNMR_PPM_RANGES[motif]
                    if q_min <= pmax and q_max >= pmin:
                        queried_motifs.add(motif)
            if step["tool"] == "get_full_spectrum" and step["args"].get("nucleus") == "1H":
                queried_motifs.update(required_motifs)

        if len(queried_motifs) >= len(required_motifs):
            return {"covered": True, "method": "tool_match",
                    "details": f"Agent explored all required regions: {queried_motifs}"}
        elif len(queried_motifs) > 0:
            return {"covered": True, "method": "heuristic",
                    "details": f"Agent explored {queried_motifs} of {required_motifs}"}
        return None

    def _match_hnmr_integration_check(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Match F3: Integration sum == H count check."""
        gt_yes = gt_answer.strip().lower() in ("yes", "y")

        for step in trajectory:
            if step["tool"] == "get_integration":
                total_integ = step["result"].get("total_integration", 0)
                # Agent checked integration
                h_count = self._formula_counts.get("H", 0)
                matches = (total_integ == h_count)
                if matches == gt_yes:
                    return {"covered": True, "method": "tool_match",
                            "details": f"get_integration total={total_integ}, H_count={h_count}, GT={gt_answer}"}
                else:
                    return {"covered": False, "method": "tool_match",
                            "details": f"get_integration total={total_integ}, H_count={h_count}, but GT={gt_answer}"}
        return None

    def _match_hnmr_singlet_heteroatom(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Match F5: Heteroatom-attached protons (singlet + formula check)."""
        gt_yes = gt_answer.strip().lower() in ("yes", "y")

        # Check if agent used get_multiplicity (can identify singlets)
        for step in trajectory:
            if step["tool"] == "get_multiplicity":
                peaks = step["result"].get("peaks", [])
                singlets = [p for p in peaks if p.get("multiplicity", "") == "s"]
                has_ons = any(self._formula_counts.get(e, 0) > 0 for e in ("O", "N", "S"))
                has_singlet_hetero = bool(singlets) and has_ons
                if has_singlet_hetero == gt_yes:
                    return {"covered": True, "method": "tool_match",
                            "details": f"get_multiplicity found {len(singlets)} singlets, "
                                       f"formula has O/N/S={has_ons}, GT={gt_answer}"}
                else:
                    return {"covered": False, "method": "tool_match",
                            "details": f"get_multiplicity: singlets={len(singlets)}, "
                                       f"O/N/S={has_ons}, but GT={gt_answer}"}

        # Fallback: query_spectrum in broad range
        for step in trajectory:
            if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "1H":
                q_min = step["args"].get("ppm_min", 0)
                q_max = step["args"].get("ppm_max", 15)
                if q_min <= 7.0 and q_max >= 1.0:
                    return {"covered": True, "method": "heuristic",
                            "details": "Agent queried broad 1H range (includes heteroatom region)"}
        return None

    def _match_hnmr_singlet_check(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Match F11: Any singlets present?"""
        gt_yes = gt_answer.strip().lower() in ("yes", "y")

        for step in trajectory:
            if step["tool"] == "get_multiplicity":
                peaks = step["result"].get("peaks", [])
                singlets = [p for p in peaks if p.get("multiplicity", "") == "s"]
                has_singlets = len(singlets) > 0
                if has_singlets == gt_yes:
                    return {"covered": True, "method": "tool_match",
                            "details": f"get_multiplicity: {len(singlets)} singlets found, GT={gt_answer}"}
                else:
                    return {"covered": False, "method": "tool_match",
                            "details": f"get_multiplicity: {len(singlets)} singlets, but GT={gt_answer}"}
        return None

    def _match_hnmr_integration_tool(
        self,
        gt_answer: str,
        trajectory: List[Dict[str, Any]],
        motif_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Match nodes that require get_integration tool (F12)."""
        for step in trajectory:
            if step["tool"] == "get_integration":
                return {"covered": True, "method": "tool_match",
                        "details": f"Agent called get_integration ({motif_name})"}
        return None

    def _try_heuristic_match(
        self,
        node_id: str,
        node_info: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Heuristic matching for nodes that don't have direct tool mappings.

        E.g., reasoning nodes (F1.7.1, F1.8.1) are hard to match deterministically.
        We check if the trajectory demonstrates relevant knowledge.
        """
        # For combined-motif nodes (F1.7.1, F1.8.1): check if agent explored both regions
        if node_id in ("F1.7.1", "F1.8.1"):
            # These ask about multiple motifs; check general exploration
            queried_regions = set()
            for step in trajectory:
                if step["tool"] == "query_spectrum" and step["args"].get("nucleus") == "13C":
                    q_min = step["args"].get("ppm_min", 0)
                    q_max = step["args"].get("ppm_max", 0)
                    if q_max >= 160:
                        queried_regions.add("high")
                    if q_min <= 50:
                        queried_regions.add("low")
                    if 100 <= q_max and q_min <= 160:
                        queried_regions.add("mid")
            if len(queried_regions) >= 2:
                return {"covered": True, "method": "heuristic",
                        "details": f"Agent explored multiple spectral regions: {queried_regions}"}

        # F8 / F14 / BIG QUESTION: check if agent submitted a SMILES
        if node_id in ("F8", "F14", "BIG QUESTION"):
            for step in trajectory:
                if step["tool"] == "submit":
                    return {"covered": True, "method": "heuristic",
                            "details": "Agent submitted a SMILES answer"}
                if step["tool"] == "validate_smiles":
                    return {"covered": True, "method": "heuristic",
                            "details": "Agent validated a candidate structure"}

        # F13 (HNMR summary node): check if agent explored multiple regions
        if node_id == "F13":
            tools_used = set(s["tool"] for s in trajectory)
            if len(tools_used) >= 3:
                return {"covered": True, "method": "heuristic",
                        "details": f"Agent used {len(tools_used)} different tools (comprehensive analysis)"}

        return None

    def _classify_failures(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Classify uncovered nodes into failure categories."""
        failures: Dict[str, List[str]] = {
            "knowledge": [],
            "reasoning": [],
            "tool_use": [],
            "strategy": [],
        }

        tools_used = set(step["tool"] for step in trajectory)

        for nid, result in self.node_results.items():
            if result["covered"]:
                continue

            method = result.get("method", "")

            # Knowledge failure: agent never tried to investigate this area
            if method == "no_match":
                failures["strategy"].append(nid)

            # Tool use failure: agent used the tool but got wrong result
            elif method == "tool_match" and "returned" in result.get("details", ""):
                failures["tool_use"].append(nid)

            # Reasoning failure: agent queried the right region but drew wrong conclusion
            elif method == "tool_match" and "found" in result.get("details", ""):
                failures["reasoning"].append(nid)

            else:
                # Default: strategy failure (didn't explore this area)
                failures["strategy"].append(nid)

        return failures
