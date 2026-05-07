#!/usr/bin/env python
"""Run a single live episode and write diagnostic + summary outputs.

Usage:
    export OPENAI_API_KEY="sk-..."
    python -m interactive.run_live_episode

Or with custom args:
    python -m interactive.run_live_episode \
        --model gpt-5.2 \
        --molecule 2_4_dimethyl_aniline \
        --spectrum COMBO \
        --budget 25 \
        --output-dir interactive/results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from interactive.environment import ChemElucidEnv
from interactive.config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def write_summary_md(diagnostic: dict, path: str) -> None:
    """Write a human-readable summary markdown file from a diagnostic dict."""
    acc = diagnostic.get("accuracy", {})
    lines = [
        f"# Live Episode Summary",
        f"",
        f"**Timestamp:** {diagnostic.get('timestamp', 'N/A')}",
        f"",
        f"## Configuration",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Molecule | `{diagnostic.get('molecule_id', 'N/A')}` |",
        f"| Model | `{diagnostic.get('model', 'N/A')}` |",
        f"| Provider | `{diagnostic.get('provider', 'N/A')}` |",
        f"| Spectrum | `{diagnostic.get('spectrum_type', 'N/A')}` |",
        f"| Stateful (Blackboard) | `{diagnostic.get('stateful', 'N/A')}` |",
        f"| Budget | {diagnostic.get('budget', 'N/A')} |",
        f"",
        f"## Results",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Submitted SMILES | `{diagnostic.get('submitted_smiles', 'N/A')}` |",
        f"| Private Grader | `{ 'available' if diagnostic.get('private_grader_available') else 'omitted in anonymous review package' }` |",
        f"| Exact Match | {'✅' if acc.get('exact_match') else '❌'} |",
        f"| Tanimoto Similarity | {acc.get('tanimoto', 'N/A')} |",
        f"| Efficiency | {diagnostic.get('efficiency', 'N/A')} |",
        f"| Tool Calls Used | {diagnostic.get('total_cost', 'N/A')} / {diagnostic.get('budget', 'N/A')} |",
        f"| Turns | {diagnostic.get('num_turns', 'N/A')} |",
        f"| Terminated By | `{diagnostic.get('terminated_by', 'N/A')}` |",
        f"| Elapsed (s) | {diagnostic.get('elapsed_seconds', 'N/A')} |",
        f"",
    ]

    # DAG Grading
    lines.append("## DAG Grading")
    if diagnostic.get("coverage") is not None:
        lines.append(f"**Combined Coverage:** {diagnostic['coverage']:.1%}")
        lines.append("")

        if diagnostic.get("cnmr_coverage") is not None:
            lines.append(f"**CNMR Coverage:** {diagnostic['cnmr_coverage']:.1%}")
        if diagnostic.get("hnmr_coverage") is not None:
            lines.append(f"**HNMR Coverage:** {diagnostic['hnmr_coverage']:.1%}")
        lines.append("")

        if diagnostic.get("grading_summary"):
            lines.append(f"**Summary:** {diagnostic['grading_summary']}")
            lines.append("")

        # Failure analysis (format: {category: [node_ids]} or list)
        failures = diagnostic.get("failure_analysis", {})
        if failures:
            lines.append("### Failure Analysis")
            if isinstance(failures, dict):
                lines.append(f"| Category | Uncovered Nodes |")
                lines.append(f"|----------|----------------|")
                for cat, node_ids in failures.items():
                    if node_ids:
                        lines.append(f"| {cat} | {', '.join(str(n) for n in node_ids)} |")
            elif isinstance(failures, list):
                lines.append(f"| Node | Details |")
                lines.append(f"|------|---------|")
                for f in failures:
                    lines.append(f"| {f} |  |")
            lines.append("")
    else:
        lines.append("No DAG grading available.")
        lines.append("")

    # Trajectory summary
    traj = diagnostic.get("trajectory", [])
    if traj:
        lines.append("## Trajectory")
        lines.append(f"| # | Tool | Cost | Key Args |")
        lines.append(f"|---|------|------|----------|")
        for i, step in enumerate(traj, 1):
            tool = step.get("tool", "")
            cost = step.get("cost", "")
            args = step.get("args", {})
            # Summarize args (first 80 chars)
            args_str = json.dumps(args, default=str)
            if len(args_str) > 80:
                args_str = args_str[:77] + "..."
            lines.append(f"| {i} | `{tool}` | {cost} | `{args_str}` |")
        lines.append("")

    # Error info
    if diagnostic.get("error"):
        lines.append("## Error")
        lines.append(f"```\n{diagnostic['error']}\n```")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Summary written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run a live ChemElucid episode")
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--molecule", default="2_4_dimethyl_aniline", help="Molecule ID")
    parser.add_argument("--spectrum", default="COMBO", help="CNMR, HNMR, or COMBO")
    parser.add_argument("--budget", type=int, default=25, help="Tool-call budget")
    parser.add_argument("--stateful", action="store_true", default=True, help="Enable Blackboard")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--output-prefix", default="live_episode_001", help="Output file prefix")
    args = parser.parse_args()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if args.provider == "openai" and not api_key:
        logger.error("OPENAI_API_KEY is not set. Please export it first.")
        sys.exit(1)

    output_dir = args.output_dir or str(Path(__file__).parent / "results")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{args.output_prefix}.json")
    md_path = os.path.join(output_dir, f"{args.output_prefix}_summary.md")

    config = ExperimentConfig(
        budget=args.budget,
        spectrum_type=args.spectrum,
    )

    env = ChemElucidEnv(config=config)

    logger.info(f"Starting episode: {args.molecule} | {args.model} | {args.spectrum} | budget={args.budget}")

    try:
        diagnostic = env.run_episode(
            molecule_id=args.molecule,
            provider=args.provider,
            model=args.model,
            stateful=args.stateful,
            temperature=args.temperature,
            budget=args.budget,
            spectrum_type=args.spectrum,
        )
    except Exception as e:
        # Capture error in diagnostic rather than silently catching
        logger.error(f"Episode failed with error: {e}")
        traceback.print_exc()
        diagnostic = {
            "molecule_id": args.molecule,
            "model": args.model,
            "provider": args.provider,
            "stateful": args.stateful,
            "spectrum_type": args.spectrum,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(diagnostic, f, indent=2, default=str)
    logger.info(f"Diagnostic JSON written to {json_path}")

    # Write summary MD
    write_summary_md(diagnostic, md_path)

    # Print key results
    print("\n" + "=" * 60)
    print("EPISODE COMPLETE")
    print("=" * 60)
    if "error" in diagnostic:
        print(f"  ERROR: {diagnostic['error']}")
    else:
        acc = diagnostic.get("accuracy", {})
        print(f"  Molecule:     {diagnostic['molecule_id']}")
        print(f"  Model:        {diagnostic['model']}")
        print(f"  Submitted:    {diagnostic.get('submitted_smiles', 'N/A')}")
        print(
            "  Private Grader: "
            + ("available" if diagnostic.get("private_grader_available") else "omitted in anonymous review package")
        )
        print(f"  Exact Match:  {acc.get('exact_match', 'N/A')}")
        print(f"  Tanimoto:     {acc.get('tanimoto', 'N/A')}")
        print(f"  Coverage:     {diagnostic.get('coverage', 'N/A')}")
        if diagnostic.get("cnmr_coverage") is not None:
            print(f"  CNMR Cov:     {diagnostic['cnmr_coverage']}")
        if diagnostic.get("hnmr_coverage") is not None:
            print(f"  HNMR Cov:     {diagnostic['hnmr_coverage']}")
        print(f"  Cost:         {diagnostic.get('total_cost', 'N/A')}/{diagnostic.get('budget', 'N/A')}")
        print(f"  Turns:        {diagnostic.get('num_turns', 'N/A')}")
        print(f"  Terminated:   {diagnostic.get('terminated_by', 'N/A')}")
        print(f"  Elapsed:      {diagnostic.get('elapsed_seconds', 'N/A')}s")
    print(f"\n  JSON: {json_path}")
    print(f"  Summary: {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
