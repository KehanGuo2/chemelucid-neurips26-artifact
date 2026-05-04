"""Always-chain negative-control dummy agent (L3 metric stress-test).

Reviewer concern (NeurIPS 2026 E&D): the canonical L3 dependency-rate accepts
"domain-adjacent" edges via static rules in
``interactive.metrics.causal_analysis._check_static_rules``. An adversarial
agent that mechanically pipes the previous tool's output into the next tool's
input could game L3 toward 1.0 without any reasoning.

This script implements such an agent. On each turn it:

    1. Picks a tool from a small fixed roster.
    2. For at least one argument of that tool, copies a value taken
       LITERALLY from the previous tool's result — a string for
       string-typed args (e.g. ``smiles``, ``pattern``), a number for
       numeric-typed args (e.g. ``ppm_min``, ``ppm_max``).
    3. Runs the call, records the result, repeats until budget is spent.
    4. Submits a fixed dummy SMILES (cost 0).

The agent never reasons about chemistry — it only mechanically forwards
values from result[i] to args[i+1]. This is the worst case for an L3
metric that rewards "any kind of chaining".

Run on 18 core molecules x {partial, full} = 36 episodes. Pure Python,
no API. We compute BOTH the canonical L3 (with static-rule edges) and
the strict-edge L3 (numeric + string transfer only) per episode.

Output:
    interactive/results/dummy_baselines/always_chain/{condition}/{mol}.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.l3_strict import compute_causal_metrics_strict  # noqa: E402
from interactive.config import ALL_MOLECULES, DATA_DIR  # noqa: E402
from interactive.metrics.causal_analysis import compute_causal_metrics  # noqa: E402
from interactive.metrics.layer1 import compute_layer1  # noqa: E402
from interactive.tool_server import TOOL_DEFINITIONS, ToolServer  # noqa: E402

DUMMY_SMILES = "CCO"


# ---------------------------------------------------------------------------
# Tool selection roster. Excludes:
#   - submit (handled separately)
#   - get_full_spectrum (hidden in partial mode; using it would skew the
#     comparison across conditions)
#   - compare_spectra (requires two coupled list args; harder to gimmick
#     deterministically and not interesting for the chaining attack)
# ---------------------------------------------------------------------------
_CHAIN_TOOL_ROSTER: List[str] = [
    "query_spectrum",
    "compute_unsaturation",
    "compute_molecular_weight",
    "substructure_check",
    "count_carbons",
    "check_symmetry",
    "validate_smiles",
    "predict_nmr",
    "get_multiplicity",
    "get_integration",
    "get_coupling",
]

# ppm-window arg keys (treated as numeric chain targets).
_PPM_KEYS = {"ppm_min", "ppm_max"}

# Default spectrum bounds when seeding the very first call.
_PPM_DEFAULT_RANGE: Dict[str, Tuple[float, float]] = {
    "13C": (20.0, 60.0),
    "1H": (1.0, 4.0),
}


# ---------------------------------------------------------------------------
# Helpers: pull a literal value from a previous tool result for chaining.
# ---------------------------------------------------------------------------

def _walk_strings(obj: Any) -> List[str]:
    """All string values reachable in obj. Order: dict-iteration order, depth-first."""
    out: List[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk_strings(v))
    return out


def _walk_numbers(obj: Any) -> List[float]:
    out: List[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk_numbers(v))
    return out


def _scrape_smiles_like(prev_result: Any) -> Optional[str]:
    """Return any string from prev_result that is plausibly SMILES-shaped (>=4 chars)."""
    for s in _walk_strings(prev_result):
        if len(s) >= 4 and not s.startswith("13C") and not s.startswith("1H"):
            return s
    return None


def _scrape_string_with_min_len(prev_result: Any, min_len: int = 4) -> Optional[str]:
    for s in _walk_strings(prev_result):
        if len(s) >= min_len:
            return s
    return None


def _scrape_number(
    prev_result: Any,
    rng: random.Random,
    fallback: float,
    *,
    in_range: Tuple[float, float] = (-1e9, 1e9),
) -> float:
    """Pick a number from prev_result inside ``in_range``; fallback otherwise.

    Picking deterministically (first-match) would always return the same
    result — using rng.choice over the eligible numbers gives a bit of
    variation while still copying-from-previous (which is what the L3 metric
    is supposed to detect).
    """
    nums = [n for n in _walk_numbers(prev_result) if in_range[0] <= n <= in_range[1]]
    if nums:
        return rng.choice(nums)
    return fallback


def _formula_from_result(prev_result: Any) -> Optional[str]:
    """Extract a molecular-formula-shaped string (e.g. 'C8H11N')."""
    if isinstance(prev_result, dict):
        for k in ("formula", "molecular_formula"):
            v = prev_result.get(k)
            if isinstance(v, str) and v:
                return v
    return None


# ---------------------------------------------------------------------------
# Always-chain agent
# ---------------------------------------------------------------------------

class AlwaysChainAgent:
    """Each turn picks a tool and forces at least one arg to come from the
    previous tool's RESULT (literal copy). On turn 0 (no previous result),
    seeds with a sane default. Submits dummy SMILES at end of budget.
    """

    name = "always_chain"

    def __init__(
        self,
        server: ToolServer,
        budget: int,
        rng: random.Random,
    ) -> None:
        self.server = server
        self.budget = budget
        self.rng = rng
        self._prev_result: Optional[Dict[str, Any]] = None

    # -- arg builders, each forced to chain at least one value -----------------

    def _args_query_spectrum(self) -> Dict[str, Any]:
        nucleus = self.rng.choice(["13C", "1H"])
        lo_def, hi_def = _PPM_DEFAULT_RANGE[nucleus]
        if self._prev_result is None:
            return {"nucleus": nucleus, "ppm_min": lo_def, "ppm_max": hi_def}
        # Pull a number from prev_result; clamp to the nucleus's range.
        max_ppm = 220.0 if nucleus == "13C" else 12.0
        ppm_min = _scrape_number(
            self._prev_result, self.rng, fallback=lo_def, in_range=(0.0, max_ppm - 1.0)
        )
        ppm_min = max(0.0, min(ppm_min, max_ppm - 1.0))
        ppm_max = min(ppm_min + 10.0, max_ppm)
        return {"nucleus": nucleus, "ppm_min": round(ppm_min, 2), "ppm_max": round(ppm_max, 2)}

    def _args_formula_tool(self) -> Dict[str, Any]:
        # compute_unsaturation / compute_molecular_weight take {"formula": ...}
        if self._prev_result is None:
            return {"formula": self.server.formula or "C8H11N"}
        formula = _formula_from_result(self._prev_result) or (self.server.formula or "C8H11N")
        return {"formula": formula}

    def _args_smiles_only(self) -> Dict[str, Any]:
        # count_carbons / check_symmetry / validate_smiles take {"smiles": ...}
        if self._prev_result is None:
            return {"smiles": "CCO"}
        s = _scrape_smiles_like(self._prev_result) or "CCO"
        return {"smiles": s}

    def _args_substructure_check(self) -> Dict[str, Any]:
        if self._prev_result is None:
            return {"smiles": "CCO", "pattern": "[OH]"}
        smiles = _scrape_smiles_like(self._prev_result) or "CCO"
        # Pattern is a string from prev result if any (forces string-transfer
        # edge to be detectable).
        strings = _walk_strings(self._prev_result)
        pattern = next((s for s in strings if 1 <= len(s) <= 10), "[OH]")
        return {"smiles": smiles, "pattern": pattern}

    def _args_predict_nmr(self) -> Dict[str, Any]:
        if self._prev_result is None:
            return {"smiles": "CCO", "nucleus": "13C"}
        smiles = _scrape_smiles_like(self._prev_result) or "CCO"
        return {"smiles": smiles, "nucleus": self.rng.choice(["13C", "1H"])}

    def _args_hnmr_window(self) -> Dict[str, Any]:
        # get_multiplicity / get_integration / get_coupling — all take
        # ppm_min/ppm_max in 1H range
        if self._prev_result is None:
            return {"ppm_min": 1.0, "ppm_max": 4.0}
        ppm_min = _scrape_number(
            self._prev_result, self.rng, fallback=1.0, in_range=(0.0, 11.0)
        )
        ppm_min = max(0.0, min(ppm_min, 11.0))
        return {"ppm_min": round(ppm_min, 2), "ppm_max": round(min(ppm_min + 1.5, 12.0), 2)}

    def _build_args(self, tool: str) -> Dict[str, Any]:
        if tool == "query_spectrum":
            return self._args_query_spectrum()
        if tool in ("compute_unsaturation", "compute_molecular_weight"):
            return self._args_formula_tool()
        if tool in ("count_carbons", "check_symmetry", "validate_smiles"):
            return self._args_smiles_only()
        if tool == "substructure_check":
            return self._args_substructure_check()
        if tool == "predict_nmr":
            return self._args_predict_nmr()
        if tool in ("get_multiplicity", "get_integration", "get_coupling"):
            return self._args_hnmr_window()
        # Fallback (shouldn't happen with the roster above)
        return {}

    def _pick_tool(self, remaining: int) -> str:
        """Pick a tool whose cost fits remaining budget. Prefers cheap tools
        when budget is tight to keep the trajectory long enough for L3 to
        be defined (n_nodes > 1)."""
        affordable = [t for t in _CHAIN_TOOL_ROSTER if TOOL_DEFINITIONS[t]["cost"] <= remaining]
        if not affordable:
            return ""
        # When budget is tight, drop the cost-3 predict_nmr.
        if remaining < 3 and "predict_nmr" in affordable:
            affordable = [t for t in affordable if t != "predict_nmr"]
        return self.rng.choice(affordable)

    def run(self) -> Dict[str, Any]:
        while True:
            remaining = self.budget - self.server.get_total_cost()
            if remaining <= 0:
                break
            tool = self._pick_tool(remaining)
            if not tool:
                break
            args = self._build_args(tool)
            try:
                result = self.server.execute(tool, args)
            except Exception:
                # Skip malformed call; don't crash the dummy run.
                continue
            # Remember the latest result for chaining on the next turn.
            self._prev_result = result if isinstance(result, dict) else {"value": result}

        # Submit (cost 0) — dummy answer.
        self.server.execute("submit", {"smiles": DUMMY_SMILES})
        return {
            "trajectory": self.server.get_trajectory(),
            "total_cost": self.server.get_total_cost(),
        }


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def run_episode(
    molecule_id: str,
    info_mode: str,
    seed: int,
    data_dir: str,
    budget: int = 25,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    server = ToolServer(
        molecule_id=molecule_id,
        data_dir=data_dir,
        spectrum_type="CNMR",
        info_mode=info_mode,
    )
    agent = AlwaysChainAgent(server=server, budget=budget, rng=rng)
    started = datetime.now()
    out = agent.run()
    elapsed = (datetime.now() - started).total_seconds()

    trajectory = out["trajectory"]
    layer1 = compute_layer1(trajectory)
    cur_l3 = compute_causal_metrics(trajectory)
    strict_l3 = compute_causal_metrics_strict(trajectory)

    return {
        "molecule_id": molecule_id,
        "agent": AlwaysChainAgent.name,
        "model": f"dummy/{AlwaysChainAgent.name}",
        "provider": "dummy",
        "spectrum_type": "CNMR",
        "info_mode": info_mode,
        "condition": f"{info_mode}_autonomous",
        "scaffold": False,
        "seed": seed,
        "budget": budget,
        "total_cost": out["total_cost"],
        "num_turns": len(trajectory),
        "submitted_smiles": DUMMY_SMILES,
        "gt_smiles": server.gt_smiles,
        "terminated_by": "submit",
        "trajectory": trajectory,
        "layer1": layer1,
        "layer3_dependency_rate": cur_l3["dependency_rate"],
        "layer3_dependency_rate_v1_edgecount": cur_l3["dependency_rate_v1_edgecount"],
        "layer3_conflict_reactivity": cur_l3["conflict_reactivity"],
        "layer3_num_edges": cur_l3["num_edges"],
        "layer3_edge_types": [e["type"] for e in cur_l3["edges"]],
        "layer3_strict_dependency_rate": strict_l3["dependency_rate"],
        "layer3_strict_num_edges": strict_l3["num_edges"],
        "layer3_strict_edge_types": [e["type"] for e in strict_l3["edges"]],
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": started.isoformat(),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(out_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for cond_dir in sorted(out_root.iterdir()):
        if not cond_dir.is_dir():
            continue
        for path in sorted(cond_dir.glob("*.json")):
            with open(path) as f:
                rec = json.load(f)
            rows.append({
                "condition": rec["condition"],
                "info_mode": rec["info_mode"],
                "molecule_id": rec["molecule_id"],
                "L3_current": rec["layer3_dependency_rate"],
                "L3_strict": rec["layer3_strict_dependency_rate"],
                "n_calls": rec["num_turns"],
                "n_edges_current": rec["layer3_num_edges"],
                "n_edges_strict": rec["layer3_strict_num_edges"],
            })
    if not rows:
        return {"n": 0}
    n = len(rows)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    by_mode: Dict[str, Dict[str, float]] = {}
    for mode in ("partial", "full"):
        sub = [r for r in rows if r["info_mode"] == mode]
        if not sub:
            continue
        by_mode[mode] = {
            "n": len(sub),
            "mean_L3_current": _mean([r["L3_current"] for r in sub]),
            "mean_L3_strict": _mean([r["L3_strict"] for r in sub]),
        }

    return {
        "n": n,
        "mean_L3_current": _mean([r["L3_current"] for r in rows]),
        "mean_L3_strict": _mean([r["L3_strict"] for r in rows]),
        "by_info_mode": by_mode,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "interactive" / "results" / "dummy_baselines" / "always_chain",
    )
    parser.add_argument(
        "--molecules",
        nargs="*",
        default=ALL_MOLECULES,
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["partial", "full"],
        choices=["partial", "full"],
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
    )
    parser.add_argument("--rerun", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    n_run, n_skipped = 0, 0
    for cond in args.conditions:
        cond_dir = args.output_root / f"{cond}_autonomous"
        cond_dir.mkdir(parents=True, exist_ok=True)
        for i, mol in enumerate(args.molecules):
            out_path = cond_dir / f"{mol}.json"
            if args.skip_existing and out_path.exists():
                n_skipped += 1
                continue
            seed = args.seed + 100 * args.conditions.index(cond) + i
            rec = run_episode(
                molecule_id=mol,
                info_mode=cond,
                seed=seed,
                data_dir=args.data_dir,
                budget=args.budget,
            )
            out_path.write_text(json.dumps(rec, indent=2, default=str))
            n_run += 1

    print(f"Ran {n_run} always-chain episodes ({n_skipped} skipped).")
    summary = aggregate(args.output_root)
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "rows"},
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
