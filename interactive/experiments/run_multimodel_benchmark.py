"""Multi-model benchmark: N molecules × M models × K conditions.

Runs all combinations of models, molecules, and experimental conditions
(autonomous vs skill-augmented, partial vs full info mode).

Usage:
    # Default: all 18 molecules × all specified models × partial + autonomous
    python -m interactive.experiments.run_multimodel_benchmark

    # With skill-augmented condition
    python -m interactive.experiments.run_multimodel_benchmark --with-scaffold

    # Specific models only
    python -m interactive.experiments.run_multimodel_benchmark --models gpt-5.2 claude-sonnet-4 gemini-2.5-pro

    # Include full mode for ablation
    python -m interactive.experiments.run_multimodel_benchmark --include-full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from interactive.config import ALL_MOLECULES, DATA_DIR, ExperimentConfig, MODELS
from interactive.environment import ChemElucidEnv
from interactive.scaffolds import NMR_ELUCIDATION_PROTOCOL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["gpt-5.2", "claude-sonnet-4", "gemini-2.5-pro", "deepseek-r1", "o4-mini"]
RESULTS_DIR = Path("interactive/results/v3_multimodel")


def result_exists(results_dir: Path, mol_id: str, model_key: str, condition: str) -> bool:
    """Check if a result file already exists for this combination."""
    condition_dir = results_dir / f"{model_key}_{condition}"
    if not condition_dir.exists():
        return False
    for f in condition_dir.glob("*.json"):
        try:
            data = json.load(open(f))
            if data.get("molecule_id") == mol_id and not data.get("error"):
                return True
        except (json.JSONDecodeError, KeyError):
            continue
    return False


_CONDITION_PARAMS = {
    "partial_autonomous": ("partial", None),
    "partial_scaffold":   ("partial", NMR_ELUCIDATION_PROTOCOL),
    "full_autonomous":    ("full",    None),
    "full_scaffold":      ("full",    NMR_ELUCIDATION_PROTOCOL),
}


def _execute_one(env, mol_id, model_key, mc, condition, budget, spectrum_type, idx, total):
    info_mode, scaffold = _CONDITION_PARAMS[condition]
    out_dir = RESULTS_DIR / f"{model_key}_{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{idx}/{total}] START {mol_id} | {model_key} | {condition}")
    try:
        diag = env.run_episode(
            molecule_id=mol_id,
            provider=mc.provider,
            model=mc.model,
            stateful=True,
            budget=budget,
            spectrum_type=spectrum_type,
            info_mode=info_mode,
            reasoning_scaffold=scaffold,
        )
        diag["condition"] = condition
        diag["scaffold"] = condition.endswith("_scaffold")
        env.save_diagnostic(diag, str(out_dir))

        acc = diag["accuracy"]
        l1 = diag.get("layer1", {})
        logger.info(
            f"[{idx}/{total}] DONE  {mol_id} | {model_key} | {condition} | "
            f"exact={acc.get('exact_match')}, tani={acc.get('tanimoto', 0):.3f}, "
            f"L2={diag.get('coverage', 0):.3f}, cf={diag.get('cf_coverage', 0):.3f}, "
            f"L1={l1.get('score', 0):.3f}, L3dep={diag.get('layer3_dependency_rate', 0):.3f}, "
            f"cost={diag['total_cost']}"
        )
        return diag
    except Exception as e:
        logger.error(f"[{idx}/{total}] FAILED {mol_id} | {model_key} | {condition}: {e}")
        return {"molecule_id": mol_id, "model": mc.model, "condition": condition, "error": str(e)}


def run_benchmark(
    model_keys: List[str],
    molecules: List[str],
    conditions: List[str],
    budget: int = 25,
    spectrum_type: str = "CNMR",
    concurrency: int = 1,
):
    """Run the full benchmark matrix."""
    config = ExperimentConfig(data_dir=str(DATA_DIR), budget=budget, spectrum_type=spectrum_type)
    env = ChemElucidEnv(config)  # thread-safe: run_episode creates fresh per-call state

    # Build run list
    runs = []
    for model_key in model_keys:
        if model_key not in MODELS:
            logger.warning(f"Unknown model key: {model_key}, skipping")
            continue
        mc = MODELS[model_key]
        for condition in conditions:
            for mol_id in molecules:
                if result_exists(RESULTS_DIR, mol_id, model_key, condition):
                    logger.info(f"  Skip (exists): {mol_id} / {model_key} / {condition}")
                    continue
                runs.append((mol_id, model_key, mc, condition))

    total = len(runs)
    logger.info(f"Running {total} episodes with concurrency={concurrency}")
    logger.info(f"Skipped {len(model_keys) * len(molecules) * len(conditions) - total} existing results")

    results: List[Dict[str, Any]] = []
    if concurrency <= 1:
        for i, (mol_id, model_key, mc, condition) in enumerate(runs, 1):
            results.append(_execute_one(env, mol_id, model_key, mc, condition, budget, spectrum_type, i, total))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_execute_one, env, mol_id, model_key, mc, condition,
                            budget, spectrum_type, i, total): (mol_id, model_key, condition)
                for i, (mol_id, model_key, mc, condition) in enumerate(runs, 1)
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    return results


def generate_report(results_dir: Path, model_keys: List[str], conditions: List[str]):
    """Generate summary report from all result files."""
    all_results = []
    for model_key in model_keys:
        for condition in conditions:
            cond_dir = results_dir / f"{model_key}_{condition}"
            if not cond_dir.exists():
                continue
            for f in sorted(cond_dir.glob("*.json")):
                try:
                    data = json.load(open(f))
                    if "error" not in data:
                        data.setdefault("condition", condition)
                        data.setdefault("model_key", model_key)
                        all_results.append(data)
                except (json.JSONDecodeError, KeyError):
                    continue

    if not all_results:
        logger.warning("No results found for report generation")
        return

    lines = [f"# Multi-Model Benchmark Report\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Episodes: {len(all_results)}")
    lines.append(f"Models: {', '.join(model_keys)}")
    lines.append(f"Conditions: {', '.join(conditions)}")
    lines.append("")

    # Summary table
    lines.append("## Summary by Model × Condition\n")
    lines.append("| Model | Condition | N | Exact% | Avg Tani | Avg L1 | Avg L2 | Avg CF L2 | Avg L3dep | Avg L3react | Avg Cost |")
    lines.append("|-------|-----------|---|--------|----------|--------|--------|-----------|-----------|-------------|----------|")

    for model_key in model_keys:
        for condition in conditions:
            subset = [r for r in all_results
                      if r.get("model_key", "") == model_key
                      and r.get("condition", "") == condition]
            if not subset:
                continue
            n = len(subset)
            ex = sum(1 for r in subset if r.get("accuracy", {}).get("exact_match"))
            at = sum(r.get("accuracy", {}).get("tanimoto", 0) for r in subset) / n
            al1 = sum(r.get("layer1", {}).get("score", 0) for r in subset) / n
            al2 = sum((r.get("coverage") or 0) for r in subset) / n
            acf = sum((r.get("cf_coverage") or 0) for r in subset) / n
            al3 = sum(r.get("layer3_dependency_rate", 0) for r in subset) / n
            ar = sum(r.get("layer3_conflict_reactivity", 1.0) for r in subset) / n
            ac = sum(r.get("total_cost", 0) for r in subset) / n
            lines.append(
                f"| {model_key} | {condition} | {n} | {ex/n:.0%} | {at:.3f} | "
                f"{al1:.3f} | {al2:.3f} | {acf:.3f} | {al3:.3f} | {ar:.3f} | {ac:.1f} |"
            )
    lines.append("")

    report_path = results_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model benchmark runner")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model keys to evaluate")
    parser.add_argument("--molecules", nargs="+", default=None,
                        help="Molecule IDs (default: all 18)")
    parser.add_argument("--with-scaffold", action="store_true",
                        help="Include skill-augmented condition")
    parser.add_argument("--include-full", action="store_true",
                        help="Include full info mode conditions")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--spectrum-type", default="CNMR", choices=["CNMR", "HNMR", "COMBO"])
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of episodes to run in parallel (threads)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing results")
    args = parser.parse_args()

    molecules = args.molecules or ALL_MOLECULES

    # Build conditions list
    conditions = ["partial_autonomous"]
    if args.with_scaffold:
        conditions.append("partial_scaffold")
    if args.include_full:
        conditions.append("full_autonomous")
        if args.with_scaffold:
            conditions.append("full_scaffold")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        t0 = time.time()
        run_benchmark(args.models, molecules, conditions, args.budget, args.spectrum_type, args.concurrency)
        elapsed = time.time() - t0
        logger.info(f"Total runtime: {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    generate_report(RESULTS_DIR, args.models, conditions)


if __name__ == "__main__":
    main()
