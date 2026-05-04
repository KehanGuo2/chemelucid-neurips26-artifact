"""Run V2 experiments on two models: gpt-5.2 and claude-sonnet-4.5.

4 molecules x 2 models x 2 modes (partial + full) = 16 episodes.

Usage:
    python -m interactive.experiments.run_dual_model
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
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

TARGET_MOLECULES = [
    "2_4_dimethyl_aniline",
    "ibuprofen",
    "saccharine",
    "theophylline",
]

MODEL_KEYS = ["gpt-5.2", "claude-sonnet-4"]
INFO_MODES = ["partial", "full"]

RESULTS_BASE = Path("interactive/results/v2_dual")


def run_one(
    env: ChemElucidEnv,
    mol_id: str,
    model_key: str,
    info_mode: str,
    output_dir: Path,
) -> Dict[str, Any]:
    mc = MODELS[model_key]
    logger.info(f"  {mol_id} | {model_key} | {info_mode}")
    try:
        diag = env.run_episode(
            molecule_id=mol_id,
            provider=mc.provider,
            model=mc.model,
            stateful=True,
            budget=25,
            spectrum_type="CNMR",
            info_mode=info_mode,
        )
        env.save_diagnostic(diag, str(output_dir))
        acc = diag["accuracy"]
        l1 = diag.get("layer1", {})
        logger.info(
            f"    exact={acc.get('exact_match')}, tani={acc.get('tanimoto', 0):.3f}, "
            f"cost={diag['total_cost']}, L1={l1.get('score', 0):.3f}, "
            f"L3dep={diag.get('layer3_dependency_rate', 0):.3f}, "
            f"ended={diag['terminated_by']}"
        )
        return diag
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        return {"molecule_id": mol_id, "model": mc.model, "info_mode": info_mode, "error": str(e)}


def generate_report(all_results: List[Dict[str, Any]], report_path: Path, graph_dir: Path):
    lines = ["# V2 Dual-Model Experiment: gpt-5.2 vs claude-sonnet-4.5\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Summary table
    lines.append("## Summary Table\n")
    lines.append("| Molecule | Model | Mode | Exact | Tanimoto | L1 | L2 (cov) | L3 dep | L3 react | Cost | Turns |")
    lines.append("|----------|-------|------|-------|----------|----|----------|--------|----------|------|-------|")

    for r in sorted(all_results, key=lambda x: (x.get("molecule_id", ""), x.get("model", ""), x.get("info_mode", ""))):
        if "error" in r:
            lines.append(f"| {r.get('molecule_id', '?')} | {r.get('model', '?')} | {r.get('info_mode', '?')} | ERROR | — | — | — | — | — | — | — |")
            continue
        acc = r.get("accuracy", {})
        l1 = r.get("layer1", {})
        lines.append(
            f"| {r['molecule_id']} | {r['model']} | {r['info_mode']} | "
            f"{'Y' if acc.get('exact_match') else 'N'} | "
            f"{acc.get('tanimoto', 0):.3f} | "
            f"{l1.get('score', 0):.3f} | "
            f"{r.get('coverage', 0) or 0:.3f} | "
            f"{r.get('layer3_dependency_rate', 0):.3f} | "
            f"{r.get('layer3_conflict_reactivity', 1.0):.3f} | "
            f"{r.get('total_cost', '?')} | "
            f"{r.get('num_turns', '?')} |"
        )

    lines.append("")

    # Per-model averages
    lines.append("## Per-Model Averages\n")
    for model_key in MODEL_KEYS:
        mc = MODELS[model_key]
        model_name = mc.model
        for mode in INFO_MODES:
            subset = [r for r in all_results
                      if r.get("model") == model_name and r.get("info_mode") == mode and "error" not in r]
            if not subset:
                continue
            n = len(subset)
            avg_exact = sum(1 for r in subset if r["accuracy"].get("exact_match")) / n
            avg_tani = sum(r["accuracy"].get("tanimoto", 0) for r in subset) / n
            avg_l1 = sum(r.get("layer1", {}).get("score", 0) for r in subset) / n
            avg_cov = sum((r.get("coverage") or 0) for r in subset) / n
            avg_dep = sum(r.get("layer3_dependency_rate", 0) for r in subset) / n
            avg_react = sum(r.get("layer3_conflict_reactivity", 1.0) for r in subset) / n
            avg_cost = sum(r.get("total_cost", 0) for r in subset) / n

            lines.append(
                f"**{model_key} ({mode})**: n={n}, "
                f"exact={avg_exact:.0%}, tani={avg_tani:.3f}, "
                f"L1={avg_l1:.3f}, L2={avg_cov:.3f}, "
                f"L3_dep={avg_dep:.3f}, L3_react={avg_react:.3f}, "
                f"cost={avg_cost:.1f}"
            )

    lines.append("")

    # Key questions
    lines.append("## Key Questions\n")
    lines.append("### Does partial mode increase L1 and L3 vs full mode?\n")

    for model_key in MODEL_KEYS:
        mc = MODELS[model_key]
        model_name = mc.model
        full_set = [r for r in all_results if r.get("model") == model_name and r.get("info_mode") == "full" and "error" not in r]
        partial_set = [r for r in all_results if r.get("model") == model_name and r.get("info_mode") == "partial" and "error" not in r]
        if full_set and partial_set:
            l1_full = sum(r.get("layer1", {}).get("score", 0) for r in full_set) / len(full_set)
            l1_part = sum(r.get("layer1", {}).get("score", 0) for r in partial_set) / len(partial_set)
            l3_full = sum(r.get("layer3_dependency_rate", 0) for r in full_set) / len(full_set)
            l3_part = sum(r.get("layer3_dependency_rate", 0) for r in partial_set) / len(partial_set)
            lines.append(f"**{model_key}**: L1 full={l1_full:.3f} → partial={l1_part:.3f} ({'↑' if l1_part > l1_full else '↓'}), "
                         f"L3 full={l3_full:.3f} → partial={l3_part:.3f} ({'↑' if l3_part > l3_full else '↓'})")

    lines.append("")

    # Per-molecule narratives
    lines.append("## Per-Molecule Trajectory Analysis\n")
    for mol_id in TARGET_MOLECULES:
        lines.append(f"### {mol_id}\n")
        mol_results = [r for r in all_results if r.get("molecule_id") == mol_id and "error" not in r]
        for r in mol_results:
            acc = r["accuracy"]
            l1 = r.get("layer1", {})
            model = r["model"]
            mode = r["info_mode"]
            conflicts = r.get("layer3_conflicts", [])
            edges = r.get("layer3_edges", [])

            lines.append(
                f"- **{model} ({mode})**: "
                f"{'correct' if acc.get('exact_match') else 'wrong'} "
                f"(tani={acc.get('tanimoto', 0):.3f}), "
                f"{r.get('total_cost', 0)} cost in {r.get('num_turns', 0)} turns, "
                f"L1={l1.get('score', 0):.3f} ({l1.get('unique_regions', 0)} regions), "
                f"L3_dep={r.get('layer3_dependency_rate', 0):.3f} ({len(edges)} edges), "
                f"{len(conflicts)} conflicts"
            )

            # Graph
            if r.get("trajectory"):
                traj = r["trajectory"]
                analysis = compute_causal_metrics(traj)
                ep_name = f"{mol_id}_{model}_{mode}"
                title = f"{mol_id} ({model}, {mode})"
                save_trajectory_graph(traj, analysis, str(graph_dir), ep_name, title=title)

        lines.append("")

    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {report_path}")


def main():
    total_start = time.time()

    # Verify API keys
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    config = ExperimentConfig(data_dir=str(DATA_DIR), budget=25, spectrum_type="CNMR")
    env = ChemElucidEnv(config)

    all_results: List[Dict[str, Any]] = []

    total_runs = len(TARGET_MOLECULES) * len(MODEL_KEYS) * len(INFO_MODES)
    run_num = 0

    for model_key in MODEL_KEYS:
        for info_mode in INFO_MODES:
            output_dir = RESULTS_BASE / f"{model_key}_{info_mode}"
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n=== {model_key} / {info_mode} mode ===")
            for mol_id in TARGET_MOLECULES:
                run_num += 1
                logger.info(f"[{run_num}/{total_runs}]")
                result = run_one(env, mol_id, model_key, info_mode, output_dir)
                all_results.append(result)

    # Generate report
    graph_dir = RESULTS_BASE / "graphs"
    report_path = RESULTS_BASE / "comparison.md"
    generate_report(all_results, report_path, graph_dir)

    elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results in: {RESULTS_BASE}")


if __name__ == "__main__":
    main()
