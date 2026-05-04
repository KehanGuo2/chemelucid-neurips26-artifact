"""CLI entry point for interactive structure elucidation experiments.

Usage:
    # Run single molecule
    python -m interactive.cli run --molecule 2_4_dimethyl_aniline --model gpt-4.1 --stateful

    # Run all molecules
    python -m interactive.cli run-all --model gpt-4.1 --stateful --output interactive_results/

    # List available molecules
    python -m interactive.cli list-molecules

    # Analyze results
    python -m interactive.cli analyze --input interactive_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from interactive.config import (
    ALL_MOLECULES,
    DATA_DIR,
    ExperimentConfig,
    MODELS,
    get_all_molecule_ids,
)
from interactive.environment import ChemElucidEnv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_run(args):
    """Run a single molecule episode."""
    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=args.budget,
        output_dir=args.output,
        spectrum_type=args.spectrum_type,
    )
    env = ChemElucidEnv(config)

    # Resolve model config
    if args.model in MODELS:
        mc = MODELS[args.model]
        provider = mc.provider
        model = mc.model
    else:
        provider = args.provider or "openai"
        model = args.model

    logger.info(
        f"Running episode: molecule={args.molecule}, model={model}, "
        f"provider={provider}, stateful={args.stateful}, budget={args.budget}"
    )

    scaffold = None
    if getattr(args, "scaffold", False):
        from interactive.scaffolds import NMR_ELUCIDATION_PROTOCOL
        scaffold = NMR_ELUCIDATION_PROTOCOL

    diagnostic = env.run_episode(
        molecule_id=args.molecule,
        provider=provider,
        model=model,
        stateful=args.stateful,
        temperature=args.temperature,
        budget=args.budget,
        spectrum_type=args.spectrum_type,
        info_mode=args.info_mode,
        reasoning_scaffold=scaffold,
    )

    # Print summary
    acc = diagnostic["accuracy"]
    print(f"\n{'='*60}")
    print(f"Molecule:  {diagnostic['molecule_id']}")
    print(f"Model:     {diagnostic['model']}")
    print(f"Stateful:  {diagnostic['stateful']}")
    print(f"Submitted: {diagnostic['submitted_smiles']}")
    print(f"GT:        {diagnostic['gt_smiles']}")
    print(f"Exact:     {acc['exact_match']}")
    print(f"Tanimoto:  {acc['tanimoto']}")
    print(f"Coverage:  {diagnostic.get('coverage', 'N/A')}")
    print(f"Cost:      {diagnostic['total_cost']}/{diagnostic['budget']}")
    print(f"Turns:     {diagnostic['num_turns']}")
    print(f"Ended by:  {diagnostic['terminated_by']}")
    print(f"Time:      {diagnostic['elapsed_seconds']}s")
    print(f"{'='*60}")

    if diagnostic.get("grading_summary"):
        print(f"Grading:   {diagnostic['grading_summary']}")

    # Save
    filepath = env.save_diagnostic(diagnostic, args.output)
    print(f"\nSaved to: {filepath}")


def cmd_run_all(args):
    """Run all molecules with a given model."""
    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=args.budget,
        output_dir=args.output,
        spectrum_type=args.spectrum_type,
    )
    env = ChemElucidEnv(config)

    # Resolve model config
    if args.model in MODELS:
        mc = MODELS[args.model]
        provider = mc.provider
        model = mc.model
    else:
        provider = args.provider or "openai"
        model = args.model

    scaffold = None
    if getattr(args, "scaffold", False):
        from interactive.scaffolds import NMR_ELUCIDATION_PROTOCOL
        scaffold = NMR_ELUCIDATION_PROTOCOL

    molecules = ALL_MOLECULES
    results = []

    for i, mol_id in enumerate(molecules, 1):
        logger.info(f"[{i}/{len(molecules)}] Running {mol_id}...")

        try:
            diagnostic = env.run_episode(
                molecule_id=mol_id,
                provider=provider,
                model=model,
                stateful=args.stateful,
                temperature=args.temperature,
                budget=args.budget,
                spectrum_type=args.spectrum_type,
                info_mode=args.info_mode,
                reasoning_scaffold=scaffold,
            )

            # Save individual result
            env.save_diagnostic(diagnostic, args.output)

            acc = diagnostic["accuracy"]
            logger.info(
                f"  exact={acc['exact_match']}, tanimoto={acc['tanimoto']:.3f}, "
                f"cost={diagnostic['total_cost']}, ended={diagnostic['terminated_by']}"
            )
            results.append(diagnostic)

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"molecule_id": mol_id, "error": str(e)})

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Molecule':<40} {'Exact':<8} {'Tanimoto':<10} {'Cost':<8} {'Ended':<10}")
    print(f"{'-'*80}")
    for r in results:
        if "error" in r:
            print(f"{r['molecule_id']:<40} ERROR: {r['error'][:30]}")
        else:
            acc = r["accuracy"]
            print(
                f"{r['molecule_id']:<40} "
                f"{'Y' if acc['exact_match'] else 'N':<8} "
                f"{acc['tanimoto']:<10.3f} "
                f"{r['total_cost']:<8} "
                f"{r['terminated_by']:<10}"
            )
    print(f"{'='*80}")

    # Aggregate stats
    successful = [r for r in results if "error" not in r]
    if successful:
        exact_count = sum(1 for r in successful if r["accuracy"]["exact_match"])
        avg_tanimoto = sum(r["accuracy"]["tanimoto"] for r in successful) / len(successful)
        avg_cost = sum(r["total_cost"] for r in successful) / len(successful)
        print(f"\nExact matches: {exact_count}/{len(successful)}")
        print(f"Avg Tanimoto:  {avg_tanimoto:.3f}")
        print(f"Avg Cost:      {avg_cost:.1f}")


def cmd_run_probe(args):
    """Run a single withheld-probe episode in outcome-only mode (no DAG grading)."""
    config = ExperimentConfig(
        data_dir=str(DATA_DIR),
        budget=args.budget,
        output_dir=args.output,
        spectrum_type=args.spectrum_type,
    )
    env = ChemElucidEnv(config)

    if args.model in MODELS:
        mc = MODELS[args.model]
        provider = mc.provider
        model = mc.model
    else:
        provider = args.provider or "openai"
        model = args.model

    scaffold = None
    if getattr(args, "scaffold", False):
        from interactive.scaffolds import NMR_ELUCIDATION_PROTOCOL
        scaffold = NMR_ELUCIDATION_PROTOCOL

    logger.info(
        f"Running withheld-probe episode: probe={args.probe_id}, model={model}, "
        f"info_mode={args.info_mode}, budget={args.budget}"
    )

    diagnostic = env.run_episode(
        molecule_id=args.probe_id,
        provider=provider,
        model=model,
        stateful=args.stateful,
        temperature=args.temperature,
        budget=args.budget,
        spectrum_type=args.spectrum_type,
        info_mode=args.info_mode,
        reasoning_scaffold=scaffold,
        probe_mode=True,
    )

    outcome = diagnostic.get("outcome_only", {})
    print(f"\n{'='*60}")
    print(f"Probe:       {diagnostic['molecule_id']}")
    print(f"Model:       {diagnostic['model']}")
    print(f"Submitted:   {diagnostic['submitted_smiles']}")
    print(f"InChI exact: {outcome.get('inchi_exact')}")
    print(f"Tanimoto:    {outcome.get('tanimoto')}")
    print(f"Valid:       {outcome.get('valid_smiles')}")
    print(f"Cost:        {diagnostic['total_cost']}/{diagnostic['budget']}")
    print(f"Turns:       {diagnostic['num_turns']}")
    print(f"Ended by:    {diagnostic['terminated_by']}")
    print(f"{'='*60}")

    filepath = env.save_diagnostic(diagnostic, args.output)
    print(f"\nSaved to: {filepath}")


def cmd_list_molecules(args):
    """List molecules. Default scope is the canonical "core 18" set."""
    scope = getattr(args, "scope", "core")
    if scope == "all":
        molecules = get_all_molecule_ids() or ALL_MOLECULES
    else:
        molecules = list(ALL_MOLECULES)
    print(f"Available molecules ({len(molecules)}, scope={scope}):")
    for i, mol in enumerate(molecules, 1):
        print(f"  {i:2d}. {mol}")


def cmd_analyze(args):
    """Analyze results from a results directory."""
    results_dir = Path(args.input)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not results:
        print("No results found.")
        return

    def _fmt(v, spec=".3f"):
        if v is None:
            return "  -  "
        try:
            return format(float(v), spec)
        except (TypeError, ValueError):
            return str(v)

    print(f"\nLoaded {len(results)} results from {results_dir}")
    width = 128
    header = (
        f"{'Molecule':<32} {'Model':<18} "
        f"{'Exact':<6} {'Tani':<6} "
        f"{'L1':<6} {'L2':<6} {'cfL2':<6} "
        f"{'L3dep':<7} {'L3rea':<7} "
        f"{'Cost':<5} {'End':<10}"
    )
    print(f"\n{'=' * width}")
    print(header)
    print(f"{'-' * width}")

    for r in results:
        acc = r.get("accuracy", {})
        l1 = r.get("layer1") or {}
        print(
            f"{r.get('molecule_id', '?'):<32.32} "
            f"{r.get('model', '?'):<18.18} "
            f"{'Y' if acc.get('exact_match') else 'N':<6} "
            f"{_fmt(acc.get('tanimoto'), '.3f'):<6} "
            f"{_fmt(l1.get('score'), '.3f'):<6} "
            f"{_fmt(r.get('coverage'), '.3f'):<6} "
            f"{_fmt(r.get('cf_coverage'), '.3f'):<6} "
            f"{_fmt(r.get('layer3_dependency_rate'), '.3f'):<7} "
            f"{_fmt(r.get('layer3_conflict_reactivity'), '.3f'):<7} "
            f"{r.get('total_cost', '?')!s:<5} "
            f"{r.get('terminated_by', '?'):<10}"
        )
    print(f"{'=' * width}")

    # Aggregate means for the 5 metrics we now display
    def _mean(key_getter):
        vals = [key_getter(r) for r in results]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    mean_tani = _mean(lambda r: r.get("accuracy", {}).get("tanimoto"))
    mean_l1 = _mean(lambda r: (r.get("layer1") or {}).get("score"))
    mean_l2 = _mean(lambda r: r.get("coverage"))
    mean_cfl2 = _mean(lambda r: r.get("cf_coverage"))
    mean_dep = _mean(lambda r: r.get("layer3_dependency_rate"))
    mean_rea = _mean(lambda r: r.get("layer3_conflict_reactivity"))
    n_exact = sum(1 for r in results if r.get("accuracy", {}).get("exact_match"))
    print(
        f"Means (n={len(results)}): "
        f"Exact={n_exact}/{len(results)}  "
        f"Tani={_fmt(mean_tani)}  "
        f"L1={_fmt(mean_l1)}  L2={_fmt(mean_l2)}  cfL2={_fmt(mean_cfl2)}  "
        f"L3dep={_fmt(mean_dep)}  L3rea={_fmt(mean_rea)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="ChemElucid: Interactive structure elucidation",
        prog="python -m interactive.cli",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run single molecule episode")
    p_run.add_argument("--molecule", "-m", required=True, help="Molecule ID")
    p_run.add_argument("--model", default="gpt-4.1", help="Model name or config key")
    p_run.add_argument("--provider", default=None, help="Provider (auto-detected from model)")
    p_run.add_argument("--stateful", action="store_true", default=True, help="Use Blackboard (default)")
    p_run.add_argument("--stateless", action="store_true", help="Disable Blackboard")
    p_run.add_argument("--budget", type=int, default=25, help="Tool-call budget")
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--spectrum-type", default="CNMR", choices=["CNMR", "HNMR", "COMBO"])
    p_run.add_argument("--info-mode", default="full", choices=["full", "partial"],
                       help="full=spectrum in prompt; partial=formula only, must query")
    p_run.add_argument("--scaffold", action="store_true",
                       help="Inject expert NMR elucidation protocol into system prompt")
    p_run.add_argument("--output", "-o", default="interactive_results")
    p_run.set_defaults(func=cmd_run)

    # --- run-all ---
    p_all = subparsers.add_parser("run-all", help="Run all molecules")
    p_all.add_argument("--model", default="gpt-4.1")
    p_all.add_argument("--provider", default=None)
    p_all.add_argument("--stateful", action="store_true", default=True)
    p_all.add_argument("--stateless", action="store_true")
    p_all.add_argument("--budget", type=int, default=25)
    p_all.add_argument("--temperature", type=float, default=0.0)
    p_all.add_argument("--spectrum-type", default="CNMR", choices=["CNMR", "HNMR", "COMBO"])
    p_all.add_argument("--info-mode", default="full", choices=["full", "partial"],
                       help="full=spectrum in prompt; partial=formula only, must query")
    p_all.add_argument("--scaffold", action="store_true",
                       help="Inject expert NMR elucidation protocol into system prompt")
    p_all.add_argument("--output", "-o", default="interactive_results")
    p_all.set_defaults(func=cmd_run_all)

    # --- run-probe ---
    p_probe = subparsers.add_parser(
        "run-probe",
        help="Run a single withheld-probe episode in outcome-only mode (no DAG grading)",
    )
    p_probe.add_argument("--probe-id", required=True, help="Probe ID, e.g. probe_001")
    p_probe.add_argument("--model", default="gpt-4.1")
    p_probe.add_argument("--provider", default=None)
    p_probe.add_argument("--stateful", action="store_true", default=True)
    p_probe.add_argument("--stateless", action="store_true")
    p_probe.add_argument("--budget", type=int, default=25)
    p_probe.add_argument("--temperature", type=float, default=0.0)
    p_probe.add_argument(
        "--spectrum-type", default="CNMR", choices=["CNMR", "HNMR", "COMBO"],
    )
    p_probe.add_argument(
        "--info-mode", default="partial", choices=["full", "partial"],
        help="Default partial; the probe layer is intended for shortcut-aware eval",
    )
    p_probe.add_argument("--scaffold", action="store_true")
    p_probe.add_argument("--output", "-o", default="interactive_results/withheld_probe")
    p_probe.set_defaults(func=cmd_run_probe)

    # --- list-molecules ---
    p_list = subparsers.add_parser("list-molecules", help="List available molecules")
    p_list.add_argument(
        "--scope",
        default="core",
        choices=["core", "all"],
        help="core=canonical 18 molecules from ALL_MOLECULES (default); all=every JSON in data/C_NMR/",
    )
    p_list.set_defaults(func=cmd_list_molecules)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze results")
    p_analyze.add_argument("--input", "-i", required=True, help="Results directory")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()

    # Handle stateless flag
    if hasattr(args, "stateless") and args.stateless:
        args.stateful = False

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
