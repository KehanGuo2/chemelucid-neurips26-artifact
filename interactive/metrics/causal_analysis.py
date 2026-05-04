"""Layer 3: Reasoning topology metrics — dependency and conflict analysis.

Analyzes an agent's tool-call trajectory to measure:
- **Dependency rate**: How often consecutive tool calls are causally linked
  (i.e., output of one informs input of the next).
- **Conflict reactivity**: When a tool returns a negative signal, does the
  agent adapt (explore more) or blindly retry?

These metrics distinguish hypothesis-driven reasoning from guess-and-check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Tolerance constants for numeric value matching
# ---------------------------------------------------------------------------
_PPM_TOL = 3.0       # chemical shift values
_INT_TOL = 0         # integer values (DoU, carbon count) — exact
_FLOAT_TOL = 0.5     # other floats (MW, similarity)

# Max look-back window for non-consecutive dependency search
_MAX_LOOKBACK = 5

# Tools considered "exploratory" for conflict-reactivity scoring
_EXPLORATORY_TOOLS: Set[str] = {
    "query_spectrum", "get_multiplicity", "get_integration",
    "get_coupling", "substructure_check", "check_symmetry",
}

# Tools whose negative results constitute a conflict
_CONFLICT_CONDITIONS: Dict[str, str] = {
    "validate_smiles":  "valid_false",
    "compare_spectra":  "low_similarity",
    "submit_rejected":  "always",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_causal_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Layer 3 causal metrics from a tool-call trajectory.

    Args:
        trajectory: List of tool-call records, each with keys
                    "tool", "args", "result", "cost", "category".

    Returns:
        Dict with keys: dependency_rate, conflict_reactivity,
        num_edges, num_nodes, num_conflicts, num_reactive,
        edges (list), conflicts (list).
    """
    n = len(trajectory)
    if n == 0:
        return _empty_result()

    edges = _find_edges(trajectory)
    conflicts = _find_conflicts(trajectory)

    num_nodes = n
    num_edges = len(edges)
    num_conflicts = len(conflicts)
    num_reactive = sum(1 for c in conflicts if c["reactive"])

    # Normalized dependency rate (R2 fix, 2026-04-20):
    # Fraction of tool calls (excluding the first) that have at least one
    # incoming edge from a prior step. Interpretable as "proportion of steps
    # that build on prior evidence", cleanly bounded to [0, 1].
    if num_nodes > 1:
        nodes_with_incoming = set()
        for e in edges:
            dst = e.get("to_idx") if isinstance(e, dict) else None
            if dst is None:
                continue
            nodes_with_incoming.add(dst)
        dependency_rate = len(nodes_with_incoming) / (num_nodes - 1)
    else:
        dependency_rate = 0.0
    # Keep the legacy edges/(n-1) ratio for debugging under a new name.
    dependency_rate_v1_edgecount = (
        num_edges / (num_nodes - 1) if num_nodes > 1 else 0.0
    )
    conflict_reactivity = (
        num_reactive / num_conflicts if num_conflicts > 0 else 1.0
    )

    return {
        "dependency_rate": round(dependency_rate, 4),
        "dependency_rate_v1_edgecount": round(dependency_rate_v1_edgecount, 4),
        "conflict_reactivity": round(conflict_reactivity, 4),
        "num_edges": num_edges,
        "num_nodes": num_nodes,
        "num_conflicts": num_conflicts,
        "num_reactive": num_reactive,
        "edges": edges,
        "conflicts": conflicts,
    }


# ---------------------------------------------------------------------------
# Edge (dependency) detection
# ---------------------------------------------------------------------------

def _find_edges(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find causal dependency edges between tool calls."""
    edges: List[Dict[str, Any]] = []
    n = len(trajectory)
    # Track which target indices already have an edge from a specific source
    # to avoid duplicate edges for the same pair
    seen_pairs: Set[Tuple[int, int]] = set()

    for j in range(1, n):
        step_b = trajectory[j]
        # Look back up to _MAX_LOOKBACK steps
        start_i = max(0, j - _MAX_LOOKBACK)
        for i in range(start_i, j):
            if (i, j) in seen_pairs:
                continue
            step_a = trajectory[i]
            dep = _check_dependency(step_a, step_b, i, j)
            if dep is not None:
                edges.append(dep)
                seen_pairs.add((i, j))

    return edges


def _check_dependency(
    step_a: Dict[str, Any],
    step_b: Dict[str, Any],
    idx_a: int,
    idx_b: int,
) -> Optional[Dict[str, Any]]:
    """Check if step_b depends on step_a. Returns edge dict or None."""

    # 1. Numeric value transfer
    edge = _check_numeric_transfer(step_a, step_b, idx_a, idx_b)
    if edge:
        return edge

    # 2. String transfer (SMILES, formulas)
    edge = _check_string_transfer(step_a, step_b, idx_a, idx_b)
    if edge:
        return edge

    # 3. Static rule dependencies (only for consecutive pairs)
    if idx_b == idx_a + 1:
        edge = _check_static_rules(step_a, step_b, idx_a, idx_b)
        if edge:
            return edge

    return None


def _check_numeric_transfer(
    step_a: Dict[str, Any],
    step_b: Dict[str, Any],
    idx_a: int,
    idx_b: int,
) -> Optional[Dict[str, Any]]:
    """Check if any numeric value in B's args appears in A's result."""
    a_nums = _extract_numbers(step_a.get("result", {}))
    b_nums = _extract_numbers(step_b.get("args", {}))

    for b_val, b_key in b_nums:
        for a_val, a_key in a_nums:
            # Determine tolerance based on value type
            if isinstance(a_val, int) and isinstance(b_val, int):
                if a_val == b_val and a_val != 0:
                    return _edge(idx_a, idx_b, "numeric",
                                 f"{a_key}={a_val} → {b_key}")
            elif isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                a_f, b_f = float(a_val), float(b_val)
                # Determine if this looks like a ppm value
                is_ppm = b_key in ("ppm_min", "ppm_max") or "ppm" in b_key
                tol = _PPM_TOL if is_ppm else _FLOAT_TOL
                if abs(a_f - b_f) <= tol and a_f != 0:
                    return _edge(idx_a, idx_b, "numeric",
                                 f"{a_key}={a_val} → {b_key}={b_val}")

    return None


def _check_string_transfer(
    step_a: Dict[str, Any],
    step_b: Dict[str, Any],
    idx_a: int,
    idx_b: int,
) -> Optional[Dict[str, Any]]:
    """Check if any string in B's args appears in A's result."""
    a_strings = _extract_strings(step_a.get("result", {}))
    b_strings = _extract_strings(step_b.get("args", {}))

    for b_val, b_key in b_strings:
        if len(b_val) < 4:  # Skip trivially short strings (e.g., "13C", "1H")
            continue
        for a_val, a_key in a_strings:
            if a_val == b_val:
                return _edge(idx_a, idx_b, "string",
                             f"{a_key}='{a_val}' → {b_key}")

    return None


def _check_static_rules(
    step_a: Dict[str, Any],
    step_b: Dict[str, Any],
    idx_a: int,
    idx_b: int,
) -> Optional[Dict[str, Any]]:
    """Check hardcoded rule-based dependencies for consecutive pairs."""
    tool_a = step_a.get("tool", "")
    tool_b = step_b.get("tool", "")
    result_a = step_a.get("result", {})
    args_b = step_b.get("args", {})

    # compute_unsaturation → query_spectrum (aromatic check)
    if tool_a == "compute_unsaturation" and tool_b == "query_spectrum":
        dbe = result_a.get("degree", 0)
        ppm_min = args_b.get("ppm_min", 0)
        if dbe >= 4 and ppm_min >= 90:
            return _edge(idx_a, idx_b, "rule",
                         f"DoU={dbe} → aromatic query")
        if dbe >= 1 and ppm_min >= 140:
            return _edge(idx_a, idx_b, "rule",
                         f"DoU={dbe} → carbonyl query")

    # query_spectrum → get_multiplicity / get_integration (overlapping range)
    if tool_a == "query_spectrum" and tool_b in ("get_multiplicity", "get_integration", "get_coupling"):
        a_range = result_a.get("range", [])
        if isinstance(a_range, list) and len(a_range) == 2:
            b_min = args_b.get("ppm_min", 0)
            b_max = args_b.get("ppm_max", 15)
            if a_range[0] <= b_max and a_range[1] >= b_min:
                return _edge(idx_a, idx_b, "rule",
                             f"spectrum query → {tool_b} in overlapping range")

    # validate_smiles → predict_nmr (same SMILES)
    if tool_a == "validate_smiles" and tool_b == "predict_nmr":
        canonical = result_a.get("canonical_smiles", "")
        pred_smiles = args_b.get("smiles", "")
        if canonical and pred_smiles and canonical == pred_smiles:
            return _edge(idx_a, idx_b, "rule",
                         f"validate → predict_nmr for '{canonical}'")

    # predict_nmr → compare_spectra (shifts transfer)
    if tool_a == "predict_nmr" and tool_b == "compare_spectra":
        pred_shifts = result_a.get("predicted_shifts", [])
        comp_shifts = args_b.get("predicted_shifts", [])
        if pred_shifts and comp_shifts and pred_shifts == comp_shifts:
            return _edge(idx_a, idx_b, "rule",
                         "predict_nmr shifts → compare_spectra")

    return None


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def _find_conflicts(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify conflict signals and whether the agent reacted to them."""
    conflicts: List[Dict[str, Any]] = []
    n = len(trajectory)

    for i, step in enumerate(trajectory):
        tool = step.get("tool", "")
        result = step.get("result", {})

        is_conflict = False
        signal = ""

        if tool == "validate_smiles" and result.get("valid") is False:
            is_conflict = True
            signal = "invalid"

        elif tool == "compare_spectra":
            sim = result.get("similarity")
            if sim is not None and sim < 0.5:
                is_conflict = True
                signal = f"similarity={sim}"

        elif tool == "submit_rejected":
            is_conflict = True
            signal = "rejected"

        elif tool == "validate_smiles":
            # Check formula mismatch — if agent is checking a candidate
            # and the formula doesn't match what's expected
            mol_formula = result.get("molecular_formula", "")
            if mol_formula and "error" not in result:
                # We can't know the expected formula here, so skip this check
                # unless there's explicit evidence of mismatch in the trajectory
                pass

        if not is_conflict:
            continue

        # Check for reactive follow-up in the next 3 steps
        reactive = False
        followup_idx = None
        followup_tool = None

        for j in range(i + 1, min(i + 4, n)):
            if trajectory[j]["tool"] in _EXPLORATORY_TOOLS:
                reactive = True
                followup_idx = j
                followup_tool = trajectory[j]["tool"]
                break

        conflicts.append({
            "idx": i,
            "tool": tool,
            "signal": signal,
            "reactive": reactive,
            "followup_idx": followup_idx,
            "followup_tool": followup_tool,
        })

    return conflicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge(from_idx: int, to_idx: int, etype: str, detail: str) -> Dict[str, Any]:
    return {
        "from_idx": from_idx,
        "to_idx": to_idx,
        "type": etype,
        "detail": detail,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "dependency_rate": 0.0,
        "conflict_reactivity": 1.0,
        "num_edges": 0,
        "num_nodes": 0,
        "num_conflicts": 0,
        "num_reactive": 0,
        "edges": [],
        "conflicts": [],
    }


def _extract_numbers(
    obj: Any, prefix: str = ""
) -> List[Tuple[Any, str]]:
    """Recursively extract (value, key_path) for all numeric values."""
    results: List[Tuple[Any, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            results.extend(_extract_numbers(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                results.append((v, f"{prefix}[{i}]"))
            elif isinstance(v, (dict, list)):
                results.extend(_extract_numbers(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        results.append((obj, prefix))
    return results


def _extract_strings(
    obj: Any, prefix: str = ""
) -> List[Tuple[str, str]]:
    """Recursively extract (value, key_path) for all string values."""
    results: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            results.extend(_extract_strings(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                results.append((v, f"{prefix}[{i}]"))
            elif isinstance(v, (dict, list)):
                results.extend(_extract_strings(v, f"{prefix}[{i}]"))
    elif isinstance(obj, str):
        results.append((obj, prefix))
    return results
