"""V2 experiments: partial vs full mode comparison.

Runs 4 representative molecules in both partial and full modes, computes
all three metric layers, and generates a comparison report.

Usage:
    python -m interactive.experiments.run_v2_experiments [--model MODEL] [--analyze-only]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from interactive.config import DATA_DIR, ExperimentConfig, MODELS
from interactive.environment import ChemElucidEnv
from interactive.metrics.causal_analysis import compute_causal_metrics
from interactive.metrics.layer1 import compute_layer1
from interactive.metrics.trajectory_graph import save_trajectory_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 4 representative molecules spanning difficulty
TARGET_MOLECULES = [
    "2_4_dimethyl_aniline",   # easy, DoU=4
    "ibuprofen",              # medium, DoU=5
    "saccharine",             # hard, DoU=6, heteroatoms
    "theophylline",           # hard, DoU=6, purine
]

RESULTS_BASE = Path("interactive/results")


def run_experiment_set(
    model_key: str,
    info_mode: str,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run all target molecules with the given model and info_mode."""
    if model_key in MODELS:
        mc = MODELS[model_key]
        provider, model = mc.provider, mc.model
    else:
        provider, model = "openai", model_key

    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=25,
        output_dir=str(output_dir),
        spectrum_type="CNMR",
    )
    env = ChemElucidEnv(config)
    results = []

    for i, mol_id in enumerate(TARGET_MOLECULES, 1):
        logger.info(f"[{i}/{len(TARGET_MOLECULES)}] {mol_id} ({info_mode} mode)...")
        try:
            diagnostic = env.run_episode(
                molecule_id=mol_id,
                provider=provider,
                model=model,
                stateful=True,
                budget=25,
                spectrum_type="CNMR",
                info_mode=info_mode,
            )
            env.save_diagnostic(diagnostic, str(output_dir))

            acc = diagnostic["accuracy"]
            logger.info(
                f"  exact={acc['exact_match']}, tanimoto={acc['tanimoto']:.3f}, "
                f"cost={diagnostic['total_cost']}, L1={diagnostic['layer1']['score']:.3f}, "
                f"L3_dep={diagnostic['layer3_dependency_rate']:.3f}"
            )
            results.append(diagnostic)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"molecule_id": mol_id, "error": str(e)})

    return results


def generate_comparison_report(
    partial_results: List[Dict[str, Any]],
    full_results: List[Dict[str, Any]],
    output_path: Path,
    graph_dir: Path,
):
    """Generate a markdown comparison report."""
    lines = ["# V2 Experiment Comparison: Partial vs Full Mode\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Molecule | Mode | Exact | Tanimoto | L1 | L2 (cov) | L3 dep | L3 react | Cost |")
    lines.append("|----------|------|-------|----------|----|----------|--------|----------|------|")

    for results, mode in [(full_results, "full"), (partial_results, "partial")]:
        for r in results:
            if "error" in r:
                lines.append(f"| {r['molecule_id']} | {mode} | ERROR | — | — | — | — | — | — |")
                continue
            acc = r.get("accuracy", {})
            l1 = r.get("layer1", {})
            lines.append(
                f"| {r['molecule_id']} | {mode} | "
                f"{'Y' if acc.get('exact_match') else 'N'} | "
                f"{acc.get('tanimoto', 0):.3f} | "
                f"{l1.get('score', 0):.3f} | "
                f"{r.get('coverage', 0):.3f} | "
                f"{r.get('layer3_dependency_rate', 0):.3f} | "
                f"{r.get('layer3_conflict_reactivity', 1.0):.3f} | "
                f"{r.get('total_cost', '?')} |"
            )

    lines.append("")

    # Per-molecule analysis
    lines.append("## Per-Molecule Analysis\n")
    for mol_id in TARGET_MOLECULES:
        full_r = next((r for r in full_results if r.get("molecule_id") == mol_id and "error" not in r), None)
        partial_r = next((r for r in partial_results if r.get("molecule_id") == mol_id and "error" not in r), None)

        lines.append(f"### {mol_id}\n")

        if full_r:
            acc = full_r["accuracy"]
            l1 = full_r.get("layer1", {})
            lines.append(
                f"**Full mode**: exact={acc.get('exact_match')}, tanimoto={acc.get('tanimoto', 0):.3f}, "
                f"coverage={full_r.get('coverage', 0):.3f}, L1={l1.get('score', 0):.3f}, "
                f"L3_dep={full_r.get('layer3_dependency_rate', 0):.3f}, "
                f"cost={full_r.get('total_cost')}, turns={full_r.get('num_turns')}"
            )
        if partial_r:
            acc = partial_r["accuracy"]
            l1 = partial_r.get("layer1", {})
            lines.append(
                f"**Partial mode**: exact={acc.get('exact_match')}, tanimoto={acc.get('tanimoto', 0):.3f}, "
                f"coverage={partial_r.get('coverage', 0):.3f}, L1={l1.get('score', 0):.3f}, "
                f"L3_dep={partial_r.get('layer3_dependency_rate', 0):.3f}, "
                f"cost={partial_r.get('total_cost')}, turns={partial_r.get('num_turns')}"
            )

        # Generate trajectory graphs
        for r, mode in [(full_r, "full"), (partial_r, "partial")]:
            if r and r.get("trajectory"):
                traj = r["trajectory"]
                analysis = compute_causal_metrics(traj)
                ep_name = f"{mol_id}_{mode}"
                title = f"{mol_id} ({mode}) dep={analysis['dependency_rate']:.2f}"
                save_trajectory_graph(traj, analysis, str(graph_dir), ep_name, title=title)

        lines.append("")

    # Key comparison
    lines.append("## Key Comparison: Partial vs Full\n")

    full_ok = [r for r in full_results if "error" not in r]
    partial_ok = [r for r in partial_results if "error" not in r]

    if full_ok and partial_ok:
        avg_l1_full = sum(r.get("layer1", {}).get("score", 0) for r in full_ok) / len(full_ok)
        avg_l1_partial = sum(r.get("layer1", {}).get("score", 0) for r in partial_ok) / len(partial_ok)
        avg_l3_full = sum(r.get("layer3_dependency_rate", 0) for r in full_ok) / len(full_ok)
        avg_l3_partial = sum(r.get("layer3_dependency_rate", 0) for r in partial_ok) / len(partial_ok)

        lines.append(f"- **Avg L1 (full)**: {avg_l1_full:.3f} vs **Avg L1 (partial)**: {avg_l1_partial:.3f}")
        lines.append(f"- **Avg L3 dependency (full)**: {avg_l3_full:.3f} vs **Avg L3 dependency (partial)**: {avg_l3_partial:.3f}")
        lines.append("")

        if avg_l1_partial > avg_l1_full:
            lines.append("Partial mode increases L1 (agents explore more spectral regions when data isn't provided upfront).")
        else:
            lines.append("Partial mode did not increase L1 in this sample.")

        if avg_l3_partial > avg_l3_full:
            lines.append("Partial mode increases L3 dependency rate (more hypothesis-driven reasoning).")
        else:
            lines.append("Partial mode did not increase L3 dependency rate in this sample.")
    else:
        lines.append("Insufficient data for comparison — run experiments first.")

    lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info(f"Comparison report saved to {output_path}")


def analyze_existing(output_path: Path, graph_dir: Path):
    """Generate comparison report from existing episode files (no API keys needed)."""
    results_dir = RESULTS_BASE

    all_results = []
    for ep_file in sorted(results_dir.glob("live_episode_*.json")):
        with open(ep_file) as f:
            data = json.load(f)
        # Enrich with layer metrics if missing
        traj = data.get("trajectory", [])
        if "layer1" not in data:
            data["layer1"] = compute_layer1(traj)
        if "layer3_dependency_rate" not in data:
            l3 = compute_causal_metrics(traj)
            data["layer3_dependency_rate"] = l3["dependency_rate"]
            data["layer3_conflict_reactivity"] = l3["conflict_reactivity"]
        all_results.append(data)

    # All existing episodes are full mode
    generate_comparison_report(
        partial_results=[],
        full_results=all_results,
        output_path=output_path,
        graph_dir=graph_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="V2 experiments: partial vs full mode")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results (no API calls)")
    args = parser.parse_args()

    graph_dir = RESULTS_BASE / "v2_graphs"
    report_path = RESULTS_BASE / "v2_comparison.md"

    if args.analyze_only:
        analyze_existing(report_path, graph_dir)
        return

    # Run experiments
    partial_dir = RESULTS_BASE / "v2_partial"
    full_dir = RESULTS_BASE / "v2_full"

    logger.info("=== Experiment A: Partial mode ===")
    partial_results = run_experiment_set(args.model, "partial", partial_dir)

    logger.info("=== Experiment B: Full mode ===")
    full_results = run_experiment_set(args.model, "full", full_dir)

    logger.info("=== Experiment C: Comparison report ===")
    generate_comparison_report(partial_results, full_results, report_path, graph_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
