"""Phase 2: Full benchmark run — 18 molecules × N models × 2 modes.

Runs the complete ChemElucid benchmark for publication. Each combination
produces a diagnostic JSON; results are aggregated into a summary table.

Usage:
    # Full benchmark with all 5 models
    export OPENAI_API_KEY="..."
    export ANTHROPIC_API_KEY="..."
    python -m interactive.experiments.run_benchmark

    # Single model
    python -m interactive.experiments.run_benchmark --models gpt-4.1

    # Multiple models
    python -m interactive.experiments.run_benchmark --models gpt-4.1 gpt-4.1-mini claude-sonnet

    # Resume from a previous run (skip completed molecules)
    python -m interactive.experiments.run_benchmark --resume --output interactive_results/benchmark_20260401/

Estimated costs:
    gpt-4.1:      ~$5  (36 runs × ~10 tool calls × ~2K tokens)
    gpt-4.1-mini: ~$1
    gpt-4.1-nano: ~$0.20
    claude-sonnet: ~$5
    claude-haiku:  ~$0.50
    Total:        ~$12

Estimated time:
    ~2-3 hours for all 5 models (mainly waiting on API latency)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from interactive.config import (
    ALL_MOLECULES,
    DATA_DIR,
    ExperimentConfig,
    MODELS,
    PROJECT_ROOT,
)
from interactive.environment import ChemElucidEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


ALL_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "claude-sonnet", "claude-haiku"]


def get_completed_runs(output_dir: str) -> set:
    """Scan output dir for completed runs to support resumption."""
    completed = set()
    out = Path(output_dir)
    if out.exists():
        for f in out.glob("*.json"):
            if f.name.startswith("_"):
                continue
            try:
                with open(f) as fh:
                    d = json.load(fh)
                mol = d.get("molecule_id", "")
                model = d.get("model", "")
                stateful = d.get("stateful", True)
                mode = "stateful" if stateful else "stateless"
                if mol and model and "error" not in d:
                    completed.add((mol, model, mode))
            except Exception:
                pass
    return completed


def run_benchmark(
    model_keys: list,
    output_dir: str,
    budget: int = 25,
    resume: bool = False,
):
    """Run the full benchmark."""
    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=budget,
        output_dir=output_dir,
    )
    env = ChemElucidEnv(config)

    # Get completed runs for resumption
    completed = get_completed_runs(output_dir) if resume else set()
    if completed:
        logger.info(f"Resuming: {len(completed)} runs already completed")

    # Validate API keys
    for mk in model_keys:
        mc = MODELS[mk]
        if mc.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.error(f"OPENAI_API_KEY not set (needed for {mk})")
            return []
        if mc.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            logger.error(f"ANTHROPIC_API_KEY not set (needed for {mk})")
            return []

    modes = [True, False]  # stateful, stateless
    total_runs = len(ALL_MOLECULES) * len(model_keys) * len(modes)
    skipped = 0
    results = []
    run_idx = 0

    for model_key in model_keys:
        mc = MODELS[model_key]

        for stateful in modes:
            mode_str = "stateful" if stateful else "stateless"

            for mol_id in ALL_MOLECULES:
                run_idx += 1

                # Check if already done
                if (mol_id, mc.model, mode_str) in completed:
                    logger.info(f"[{run_idx}/{total_runs}] SKIP (completed): {mol_id} | {model_key} | {mode_str}")
                    skipped += 1
                    continue

                logger.info(f"[{run_idx}/{total_runs}] {mol_id} | {model_key} | {mode_str}")

                start = time.time()
                try:
                    diagnostic = env.run_episode(
                        molecule_id=mol_id,
                        provider=mc.provider,
                        model=mc.model,
                        stateful=stateful,
                        temperature=0.0,
                    )

                    elapsed = time.time() - start
                    acc = diagnostic["accuracy"]

                    logger.info(
                        f"  → exact={acc['exact_match']}, tani={acc['tanimoto']:.3f}, "
                        f"cov={diagnostic.get('coverage', 'N/A')}, "
                        f"cost={diagnostic['total_cost']}, turns={diagnostic['num_turns']}, "
                        f"end={diagnostic['terminated_by']}, time={elapsed:.1f}s"
                    )

                    env.save_diagnostic(diagnostic, output_dir)
                    results.append(diagnostic)

                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"  FAILED ({elapsed:.1f}s): {e}")
                    results.append({
                        "molecule_id": mol_id,
                        "model": mc.model,
                        "stateful": stateful,
                        "error": str(e),
                    })

                # Rate limit protection
                time.sleep(0.5)

    # Print summary
    print_benchmark_summary(results, model_keys)

    # Save aggregated results
    save_benchmark_summary(results, model_keys, output_dir, skipped)

    return results


def print_benchmark_summary(results: list, model_keys: list):
    """Print a per-model summary table."""
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")

    for mk in model_keys:
        mc = MODELS[mk]
        model_results = [r for r in results if r.get("model") == mc.model and "error" not in r]

        if not model_results:
            print(f"\n{mk}: No successful results")
            continue

        for stateful in [True, False]:
            mode_str = "stateful" if stateful else "stateless"
            mode_results = [r for r in model_results if r.get("stateful") == stateful]

            if not mode_results:
                continue

            exact = sum(1 for r in mode_results if r["accuracy"]["exact_match"])
            avg_tani = sum(r["accuracy"]["tanimoto"] for r in mode_results) / len(mode_results)
            coverages = [r["coverage"] for r in mode_results if r.get("coverage") is not None]
            avg_cov = sum(coverages) / len(coverages) if coverages else 0
            avg_cost = sum(r["total_cost"] for r in mode_results) / len(mode_results)
            avg_turns = sum(r["num_turns"] for r in mode_results) / len(mode_results)

            print(f"\n  {mk} ({mode_str}):")
            print(f"    Exact match: {exact}/{len(mode_results)} ({100*exact/len(mode_results):.0f}%)")
            print(f"    Avg Tanimoto: {avg_tani:.3f}")
            print(f"    Avg Coverage: {avg_cov:.3f}")
            print(f"    Avg Cost: {avg_cost:.1f}")
            print(f"    Avg Turns: {avg_turns:.1f}")


def save_benchmark_summary(results: list, model_keys: list, output_dir: str, skipped: int):
    """Save aggregated benchmark summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": model_keys,
        "molecules": ALL_MOLECULES,
        "total_runs": len(results),
        "skipped": skipped,
        "errors": sum(1 for r in results if "error" in r),
        "per_model": {},
    }

    for mk in model_keys:
        mc = MODELS[mk]
        for stateful in [True, False]:
            mode_str = "stateful" if stateful else "stateless"
            key = f"{mk}_{mode_str}"

            mode_results = [
                r for r in results
                if r.get("model") == mc.model and r.get("stateful") == stateful and "error" not in r
            ]

            if not mode_results:
                continue

            exact = sum(1 for r in mode_results if r["accuracy"]["exact_match"])
            tanimoto_values = [r["accuracy"]["tanimoto"] for r in mode_results]
            coverages = [r["coverage"] for r in mode_results if r.get("coverage") is not None]

            summary["per_model"][key] = {
                "n": len(mode_results),
                "exact_match_count": exact,
                "exact_match_rate": exact / len(mode_results),
                "tanimoto_mean": sum(tanimoto_values) / len(tanimoto_values),
                "tanimoto_values": tanimoto_values,
                "coverage_mean": sum(coverages) / len(coverages) if coverages else None,
                "avg_cost": sum(r["total_cost"] for r in mode_results) / len(mode_results),
                "avg_turns": sum(r["num_turns"] for r in mode_results) / len(mode_results),
            }

    path = os.path.join(output_dir, "_benchmark_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nBenchmark summary saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Full benchmark")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Models to benchmark (default: all 5)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run (skip completed)")

    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(PROJECT_ROOT / f"interactive_results/benchmark_{ts}")

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output: {args.output}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Molecules: {len(ALL_MOLECULES)}")
    logger.info(f"Total runs: {len(ALL_MOLECULES) * len(args.models) * 2}")

    run_benchmark(
        model_keys=args.models,
        output_dir=args.output,
        budget=args.budget,
        resume=args.resume,
    )


if __name__ == "__main__":
    sys.exit(main())
