"""Dummy-agent baselines for L1 sanity-check.

Two non-LLM agents that emit valid tool-call trajectories without any reasoning.
Used to demonstrate that L1 (information-acquisition score) is an activity proxy:
a random tool-caller can score competitively on L1 despite producing no useful
structure inference.

Agents:
  - RandomPPMScanner: each turn picks a random ppm window and calls
    query_spectrum until budget is exhausted, then submits a fixed dummy SMILES.
  - RandomToolRoulette: each turn picks a random tool category and calls a
    representative tool with valid-but-unmotivated args.

Output:
  interactive/results/dummy_baselines/{agent}/{condition}/{mol_id}.json
  with the same trajectory + L1 fields as real-LLM episodes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add repo root to path so we can import interactive.* without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interactive.config import ALL_MOLECULES, DATA_DIR  # noqa: E402
from interactive.metrics.layer1 import compute_layer1  # noqa: E402
from interactive.tool_server import ToolServer  # noqa: E402

DUMMY_SMILES = "CCO"  # always parseable, always wrong for the 18 core molecules


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class _DummyAgentBase:
    """Common loop: call act() each turn, stop when budget would overflow, submit."""

    name: str = "dummy"

    def __init__(self, server: ToolServer, budget: int, rng: random.Random):
        self.server = server
        self.budget = budget
        self.rng = rng

    def act(self) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def run(self) -> Dict[str, Any]:
        # Greedy: keep acting until next call would overflow budget.
        # Reserve 0 for submit; submit costs 0 so we can always submit at the end.
        while True:
            remaining = self.budget - self.server.get_total_cost()
            if remaining <= 0:
                break
            tool, args = self.act()
            # Look up cost up-front so we don't overspend.
            from interactive.tool_server import TOOL_DEFINITIONS
            cost = TOOL_DEFINITIONS.get(tool, {}).get("cost", 1)
            if cost > remaining:
                # Skip this draw rather than truncating mid-call. If we can't fit
                # any spectrum_query (cost=1), we are at zero and should stop.
                if remaining < 1:
                    break
                # Try to swap to a cost=1 fallback so we don't waste a turn.
                tool, args = self._fallback_cheap()
                cost = TOOL_DEFINITIONS.get(tool, {}).get("cost", 1)
                if cost > remaining:
                    break
            self.server.execute(tool, args)

        # Submit dummy SMILES (cost 0).
        self.server.execute("submit", {"smiles": DUMMY_SMILES})
        return {
            "trajectory": self.server.get_trajectory(),
            "total_cost": self.server.get_total_cost(),
        }

    def _fallback_cheap(self) -> Tuple[str, Dict[str, Any]]:
        """Cheapest sensible action: a random ppm scan."""
        nucleus = self.rng.choice(["13C", "1H"])
        return "query_spectrum", _random_ppm_args(self.rng, nucleus)


def _random_ppm_args(rng: random.Random, nucleus: str = "13C") -> Dict[str, Any]:
    """Random ppm window: width 10-30 ppm inside the nucleus's typical range."""
    if nucleus == "13C":
        lo_min, hi_max = 0.0, 220.0
    else:
        lo_min, hi_max = 0.0, 12.0
    width = rng.uniform(10.0, 30.0) if nucleus == "13C" else rng.uniform(0.5, 3.0)
    width = min(width, hi_max - lo_min)
    ppm_min = rng.uniform(lo_min, hi_max - width)
    ppm_max = ppm_min + width
    return {
        "nucleus": nucleus,
        "ppm_min": round(ppm_min, 2),
        "ppm_max": round(ppm_max, 2),
    }


class RandomPPMScanner(_DummyAgentBase):
    """Calls query_spectrum with a random ppm window every turn."""

    name = "random_ppm_scanner"

    def act(self) -> Tuple[str, Dict[str, Any]]:
        # Always 13C for CNMR-mode molecules; allow 1H sporadically since the
        # tool just returns empty if no HNMR is loaded — that still counts as a
        # query_spectrum call (and L1 records the region).
        nucleus = "13C" if self.rng.random() < 0.85 else "1H"
        return "query_spectrum", _random_ppm_args(self.rng, nucleus)


class RandomToolRoulette(_DummyAgentBase):
    """Picks a random tool category and calls a representative tool.

    Uses valid-looking but unmotivated args. Never calls submit (the base loop
    handles termination)."""

    name = "random_tool_roulette"

    # Representative tools per category (excluding submit and validation, which
    # need specific kinds of inputs the dummy can't reasonably synthesize).
    _CATEGORIES: Dict[str, List[str]] = {
        "spectrum_query": ["query_spectrum", "get_multiplicity", "get_integration", "get_coupling"],
        "computation": ["compute_unsaturation", "compute_molecular_weight"],
        "structure_analysis": ["substructure_check", "count_carbons", "check_symmetry"],
        "prediction": ["predict_nmr"],
    }

    # Filler SMILES + SMARTS for unmotivated-but-valid args.
    _FILLER_SMILES = ["c1ccccc1", "CCO", "CC(=O)O", "CCN", "c1ccncc1"]
    _FILLER_SMARTS = ["c1ccccc1", "[OH]", "[NH2]", "C=O", "C#N"]

    def act(self) -> Tuple[str, Dict[str, Any]]:
        category = self.rng.choice(list(self._CATEGORIES.keys()))
        tool = self.rng.choice(self._CATEGORIES[category])
        formula = self.server.formula or "C8H11N"  # fall back to dummy formula

        if tool == "query_spectrum":
            return tool, _random_ppm_args(self.rng, self.rng.choice(["13C", "1H"]))
        if tool in ("get_multiplicity", "get_integration", "get_coupling"):
            args = _random_ppm_args(self.rng, "1H")
            # These tools take ppm_min/ppm_max only (no nucleus key).
            return tool, {"ppm_min": args["ppm_min"], "ppm_max": args["ppm_max"]}
        if tool in ("compute_unsaturation", "compute_molecular_weight"):
            return tool, {"formula": formula}
        if tool in ("count_carbons", "check_symmetry"):
            return tool, {"smiles": self.rng.choice(self._FILLER_SMILES)}
        if tool == "substructure_check":
            return tool, {
                "smiles": self.rng.choice(self._FILLER_SMILES),
                "pattern": self.rng.choice(self._FILLER_SMARTS),
            }
        if tool == "predict_nmr":
            return tool, {
                "smiles": self.rng.choice(self._FILLER_SMILES),
                "nucleus": self.rng.choice(["13C", "1H"]),
            }
        # Fallback (shouldn't reach).
        return self._fallback_cheap()


_AGENTS = {
    RandomPPMScanner.name: RandomPPMScanner,
    RandomToolRoulette.name: RandomToolRoulette,
}


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def run_episode(
    agent_cls,
    molecule_id: str,
    info_mode: str,
    seed: int,
    data_dir: str,
    budget: int = 25,
) -> Dict[str, Any]:
    """Run one dummy episode and return a result dict matching the LLM schema."""
    rng = random.Random(seed)
    server = ToolServer(
        molecule_id=molecule_id,
        data_dir=data_dir,
        spectrum_type="CNMR",
        info_mode=info_mode,
    )
    agent = agent_cls(server=server, budget=budget, rng=rng)
    started = datetime.now()
    out = agent.run()
    elapsed = (datetime.now() - started).total_seconds()

    trajectory = out["trajectory"]
    layer1 = compute_layer1(trajectory)

    return {
        "molecule_id": molecule_id,
        "agent": agent_cls.name,
        "model": f"dummy/{agent_cls.name}",
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
        "elapsed_seconds": round(elapsed, 3),
        "timestamp": started.isoformat(),
    }


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def aggregate(out_root: Path) -> Dict[str, Any]:
    """Aggregate L1 across all dummy episodes."""
    rows: List[Dict[str, Any]] = []
    for agent_dir in sorted(out_root.iterdir()):
        if not agent_dir.is_dir():
            continue
        for cond_dir in sorted(agent_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            for path in sorted(cond_dir.glob("*.json")):
                with open(path) as f:
                    rec = json.load(f)
                rows.append({
                    "agent": rec["agent"],
                    "condition": rec["condition"],
                    "molecule_id": rec["molecule_id"],
                    "L1": rec["layer1"]["score"],
                    "regions": rec["layer1"]["unique_regions"],
                    "categories": rec["layer1"]["categories_used"],
                    "n_calls": rec["num_turns"],
                })

    by_agent: Dict[str, List[float]] = {}
    by_cell: Dict[str, List[float]] = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r["L1"])
        cell = f"{r['agent']}__{r['condition']}"
        by_cell.setdefault(cell, []).append(r["L1"])

    summary = {
        "n_episodes": len(rows),
        "by_agent": {a: {"n": len(v), "mean_L1": _mean(v)} for a, v in by_agent.items()},
        "by_cell": {c: {"n": len(v), "mean_L1": _mean(v)} for c, v in by_cell.items()},
        "rows": rows,
    }
    return summary


def _mean(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def write_report(summary: Dict[str, Any], llm_mean_l1: float, report_path: Path) -> None:
    """Write analysis/dummy_baseline_l1.md."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Dummy-baseline L1 sanity check")
    lines.append("")
    lines.append(
        "Two LLM-free dummy agents (random PPM scanner, random tool roulette) "
        "were run on 18 core molecules x 2 info-mode conditions (partial, full) "
        "with seed=42, budget=25. Trajectories follow the same JSON schema as "
        "real-LLM episodes; L1 was computed via "
        "`interactive.metrics.layer1.compute_layer1`."
    )
    lines.append("")
    lines.append("## Aggregate L1 by agent")
    lines.append("")
    lines.append("| Agent | n | Mean L1 |")
    lines.append("| --- | --- | --- |")
    for agent, rec in sorted(summary["by_agent"].items()):
        lines.append(f"| {agent} | {rec['n']} | {rec['mean_L1']:.4f} |")
    lines.append("")
    lines.append("## Aggregate L1 by agent x condition")
    lines.append("")
    lines.append("| Cell | n | Mean L1 |")
    lines.append("| --- | --- | --- |")
    for cell, rec in sorted(summary["by_cell"].items()):
        lines.append(f"| {cell} | {rec['n']} | {rec['mean_L1']:.4f} |")
    lines.append("")
    lines.append("## Comparison to real-LLM episodes")
    lines.append("")
    lines.append(
        f"Canonical core_complete LLM mean L1 (n=216 episodes across "
        f"3 models x 18 molecules x 4 conditions): **{llm_mean_l1:.4f}**"
    )
    lines.append("")
    dummy_overall = _mean([r["L1"] for r in summary["rows"]])
    lines.append(f"Dummy-agents pooled mean L1 (n={summary['n_episodes']}): **{dummy_overall:.4f}**")
    lines.append("")

    # Per-agent verdict
    lines.append("## Verdict")
    lines.append("")
    threshold = 0.5
    over = [a for a, rec in summary["by_agent"].items() if rec["mean_L1"] > threshold]
    if over:
        lines.append(
            f"Agents with mean L1 > {threshold} (the threshold above which the paper "
            f"claim 'L1 is activity proxy not reasoning' is supported): "
            + ", ".join(f"`{a}`" for a in over) + "."
        )
    else:
        lines.append(
            f"No dummy agent exceeded mean L1 = {threshold}. The strongest claim "
            f"the data supports is that dummy L1 is comparable to but below the "
            f"LLM mean, weakening but not refuting the activity-proxy interpretation."
        )
    lines.append("")
    if dummy_overall >= llm_mean_l1:
        lines.append(
            f"Dummy pooled mean ({dummy_overall:.3f}) meets or exceeds the LLM mean "
            f"({llm_mean_l1:.3f}); a no-reasoning agent matches or beats real LLMs on "
            f"L1, confirming that L1 tracks observable activity rather than reasoning quality."
        )
    elif dummy_overall >= llm_mean_l1 * 0.85:
        lines.append(
            f"Dummy pooled mean ({dummy_overall:.3f}) is within 15% of LLM mean "
            f"({llm_mean_l1:.3f}); a no-reasoning agent reaches near-LLM L1, "
            f"confirming that L1 tracks observable activity rather than reasoning quality."
        )
    elif dummy_overall >= llm_mean_l1 * 0.5:
        lines.append(
            f"Dummy pooled mean ({dummy_overall:.3f}) is roughly half the LLM mean "
            f"({llm_mean_l1:.3f}); the activity-proxy interpretation is partially "
            f"supported but L1 also retains some signal."
        )
    else:
        lines.append(
            f"Dummy pooled mean ({dummy_overall:.3f}) is well below the LLM mean "
            f"({llm_mean_l1:.3f}); L1 distinguishes reasoning agents from random "
            f"baselines, contradicting the simple activity-proxy interpretation."
        )

    report_path.write_text("\n".join(lines) + "\n")


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
        default=_REPO_ROOT / "interactive" / "results" / "dummy_baselines",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPO_ROOT / "analysis" / "dummy_baseline_l1.md",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=_REPO_ROOT / "analysis" / "canonical_metrics.json",
    )
    parser.add_argument(
        "--molecules",
        nargs="*",
        default=ALL_MOLECULES,
        help="Subset of molecules (default: all 18 core).",
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
        help="Skip episodes whose result JSON already exists.",
    )
    parser.add_argument(
        "--rerun",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing episode JSONs.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    n_run = 0
    n_skipped = 0
    for agent_name, agent_cls in _AGENTS.items():
        for cond in args.conditions:
            cond_dir = args.output_root / agent_name / f"{cond}_autonomous"
            cond_dir.mkdir(parents=True, exist_ok=True)
            for i, mol in enumerate(args.molecules):
                out_path = cond_dir / f"{mol}.json"
                if args.skip_existing and out_path.exists():
                    n_skipped += 1
                    continue
                # Distinct, deterministic seed per (agent, cond, molecule).
                seed = args.seed + 1000 * list(_AGENTS).index(agent_name) + \
                    100 * args.conditions.index(cond) + i
                rec = run_episode(
                    agent_cls,
                    molecule_id=mol,
                    info_mode=cond,
                    seed=seed,
                    data_dir=args.data_dir,
                    budget=args.budget,
                )
                out_path.write_text(json.dumps(rec, indent=2, default=str))
                n_run += 1

    print(f"Ran {n_run} dummy episodes ({n_skipped} skipped).")

    # Aggregate + report
    summary = aggregate(args.output_root)
    canonical_mean_l1 = 0.0
    if args.canonical.exists():
        with open(args.canonical) as f:
            canon = json.load(f)
        canonical_mean_l1 = (
            canon.get("scopes", {}).get("core_complete", {}).get("overall", {}).get("mean_L1", 0.0)
        )
    write_report(summary, canonical_mean_l1, args.report)
    print(f"Report: {args.report}")
    print(f"Dummy mean L1 (pooled): {_mean([r['L1'] for r in summary['rows']]):.4f}")
    print(f"LLM mean L1 (canonical, core_complete): {canonical_mean_l1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
