"""Phase 1: First Live Episodes — validate end-to-end with real API calls.

Runs GPT-4.1 and GPT-4.1-mini on 2,4-dimethylaniline in both stateful
and stateless modes. This is the minimum viable experiment to confirm
the interactive pipeline works with real LLMs.

Usage:
    # Set API key first
    export OPENAI_API_KEY="your-key"

    # Run Phase 1 smoke test (single molecule, single model)
    python -m interactive.experiments.run_phase1

    # Run Phase 1 with specific model
    python -m interactive.experiments.run_phase1 --model gpt-4.1-mini

    # Run on all 18 molecules
    python -m interactive.experiments.run_phase1 --all

Environment:
    OPENAI_API_KEY: Required for GPT models
    ANTHROPIC_API_KEY: Required for Claude models
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


def run_single(
    env: ChemElucidEnv,
    molecule_id: str,
    model_key: str,
    stateful: bool,
    output_dir: str,
) -> dict:
    """Run a single episode and save results."""
    mc = MODELS[model_key]

    mode = "stateful" if stateful else "stateless"
    logger.info(f"Running: {molecule_id} | {model_key} | {mode}")

    start = time.time()
    try:
        diagnostic = env.run_episode(
            molecule_id=molecule_id,
            provider=mc.provider,
            model=mc.model,
            stateful=stateful,
            temperature=0.0,
        )

        elapsed = time.time() - start
        acc = diagnostic["accuracy"]

        logger.info(
            f"  → exact={acc['exact_match']}, tanimoto={acc['tanimoto']:.3f}, "
            f"coverage={diagnostic.get('coverage', 'N/A')}, "
            f"cost={diagnostic['total_cost']}/{diagnostic['budget']}, "
            f"turns={diagnostic['num_turns']}, "
            f"ended={diagnostic['terminated_by']}, "
            f"time={elapsed:.1f}s"
        )

        # Save
        filepath = env.save_diagnostic(diagnostic, output_dir)
        logger.info(f"  Saved: {filepath}")

        return diagnostic

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return {"molecule_id": molecule_id, "model": model_key, "error": str(e)}


def print_summary(results: list):
    """Print a summary table of results."""
    print(f"\n{'='*100}")
    print(f"{'Molecule':<35} {'Model':<16} {'Mode':<10} {'Exact':<7} {'Tani':<7} {'Cov':<7} {'Cost':<7} {'Turns':<7} {'End':<10} {'Time':<8}")
    print(f"{'-'*100}")

    for r in results:
        if "error" in r:
            print(f"{r.get('molecule_id', '?'):<35} {r.get('model', '?'):<16} {'?':<10} ERROR: {r['error'][:40]}")
            continue

        acc = r.get("accuracy", {})
        mode = "stateful" if r.get("stateful") else "stateless"
        print(
            f"{r['molecule_id']:<35} "
            f"{r['model']:<16} "
            f"{mode:<10} "
            f"{'Y' if acc.get('exact_match') else 'N':<7} "
            f"{acc.get('tanimoto', 0):<7.3f} "
            f"{r.get('coverage', 'N/A')!s:<7} "
            f"{r['total_cost']:<7} "
            f"{r['num_turns']:<7} "
            f"{r['terminated_by']:<10} "
            f"{r.get('elapsed_seconds', '?')!s:<8}"
        )

    print(f"{'='*100}")

    # Aggregates
    successful = [r for r in results if "error" not in r]
    if successful:
        exact_count = sum(1 for r in successful if r["accuracy"]["exact_match"])
        avg_tani = sum(r["accuracy"]["tanimoto"] for r in successful) / len(successful)
        avg_cost = sum(r["total_cost"] for r in successful) / len(successful)
        coverages = [r["coverage"] for r in successful if r.get("coverage") is not None]
        avg_cov = sum(coverages) / len(coverages) if coverages else 0
        print(f"\n  Exact matches: {exact_count}/{len(successful)} ({100*exact_count/len(successful):.0f}%)")
        print(f"  Avg Tanimoto:  {avg_tani:.3f}")
        print(f"  Avg Coverage:  {avg_cov:.3f}")
        print(f"  Avg Cost:      {avg_cost:.1f}")


def check_api_keys(model_key: str) -> bool:
    """Check if required API keys are set."""
    mc = MODELS[model_key]
    if mc.provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            print(f"ERROR: OPENAI_API_KEY not set. Required for {model_key}.")
            print("  export OPENAI_API_KEY='your-key'")
            return False
    elif mc.provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            print(f"ERROR: ANTHROPIC_API_KEY not set. Required for {model_key}.")
            print("  export ANTHROPIC_API_KEY='your-key'")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 1: First Live Episodes")
    parser.add_argument("--model", default="gpt-4.1", choices=list(MODELS.keys()),
                        help="Model to test (default: gpt-4.1)")
    parser.add_argument("--molecule", "-m", default="2_4_dimethyl_aniline",
                        help="Molecule ID (default: 2_4_dimethyl_aniline)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 18 molecules instead of just one")
    parser.add_argument("--stateful-only", action="store_true",
                        help="Only run stateful mode")
    parser.add_argument("--stateless-only", action="store_true",
                        help="Only run stateless mode")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: interactive_results/phase1_<model>_<timestamp>)")
    parser.add_argument("--budget", type=int, default=25)

    args = parser.parse_args()

    # Check API key
    if not check_api_keys(args.model):
        return 1

    # Output directory
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(PROJECT_ROOT / f"interactive_results/phase1_{args.model}_{ts}")

    os.makedirs(args.output, exist_ok=True)

    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=args.budget,
        output_dir=args.output,
    )
    env = ChemElucidEnv(config)

    # Determine molecules
    molecules = ALL_MOLECULES if args.all else [args.molecule]

    # Determine modes
    modes = []
    if not args.stateless_only:
        modes.append(True)   # stateful
    if not args.stateful_only:
        modes.append(False)  # stateless

    results = []

    total_runs = len(molecules) * len(modes)
    run_idx = 0

    for mol_id in molecules:
        for stateful in modes:
            run_idx += 1
            mode_str = "stateful" if stateful else "stateless"
            logger.info(f"\n[{run_idx}/{total_runs}] {mol_id} ({mode_str})")

            r = run_single(env, mol_id, args.model, stateful, args.output)
            results.append(r)

    # Print summary
    print_summary(results)

    # Save aggregated results
    agg_path = os.path.join(args.output, "_phase1_summary.json")
    with open(agg_path, "w") as f:
        json.dump({
            "model": args.model,
            "molecules": molecules,
            "modes": ["stateful" if m else "stateless" for m in modes],
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "molecule_id": r.get("molecule_id"),
                    "exact_match": r.get("accuracy", {}).get("exact_match"),
                    "tanimoto": r.get("accuracy", {}).get("tanimoto"),
                    "coverage": r.get("coverage"),
                    "total_cost": r.get("total_cost"),
                    "num_turns": r.get("num_turns"),
                    "terminated_by": r.get("terminated_by"),
                    "stateful": r.get("stateful"),
                    "error": r.get("error"),
                }
                for r in results
            ],
        }, f, indent=2, default=str)

    print(f"\nSummary saved to: {agg_path}")
    print(f"Full results in: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
