"""Exp 6: Repeated trials for noise-floor estimation (R4).

For each (model, molecule) pair, run N_RUNS episodes with temp=0.3.
Saves to interactive/results/v4_repeated/<model>_rep<i>/*.json
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from interactive.config import ALL_MOLECULES, DATA_DIR, ExperimentConfig, MODELS
from interactive.environment import ChemElucidEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("interactive/results/v4_repeated")


def exists(out_dir: Path, mol_id: str) -> bool:
    """Check for an existing non-zero-cost result."""
    if not out_dir.exists():
        return False
    import json
    for f in out_dir.glob("*.json"):
        try:
            d = json.load(open(f))
            if d.get("molecule_id") == mol_id and not d.get("error"):
                return True
        except Exception:
            continue
    return False


def _run_one(env, mol_id, model_key, mc, run_idx, idx, total, temperature):
    out_dir = RESULTS_DIR / f"{model_key}_rep{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if exists(out_dir, mol_id):
        logger.info(f"[{idx}/{total}] SKIP  {mol_id} | {model_key} | rep{run_idx} (exists)")
        return None

    logger.info(f"[{idx}/{total}] START {mol_id} | {model_key} | rep{run_idx} | temp={temperature}")
    try:
        diag = env.run_episode(
            molecule_id=mol_id, provider=mc.provider, model=mc.model,
            stateful=True, temperature=temperature, info_mode="partial",
        )
        diag["run_idx"] = run_idx
        diag["condition"] = "repeated_partial_auto"
        env.save_diagnostic(diag, str(out_dir))
        acc = diag["accuracy"]
        l1 = diag.get("layer1", {})
        logger.info(
            f"[{idx}/{total}] DONE  {mol_id} | {model_key} | rep{run_idx} | "
            f"exact={acc.get('exact_match')}, tani={acc.get('tanimoto', 0):.3f}, "
            f"L2={diag.get('coverage', 0):.3f}, L1={l1.get('score', 0):.3f}, cost={diag['total_cost']}"
        )
        return diag
    except Exception as e:
        logger.error(f"[{idx}/{total}] FAILED {mol_id} | {model_key} | rep{run_idx}: {e}")
        return {"molecule_id": mol_id, "model": mc.model, "run_idx": run_idx, "error": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["claude-sonnet-4", "o4-mini"])
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Default: 10 diverse molecules")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    # Default 10-molecule subset: span easy to hard
    default_mols = [
        "2_4_dimethyl_aniline", "acetaminophen", "benzoin", "Benzil",
        "saccharine", "theophylline", "pamoic_acid", "ibuprofen",
        "vanillin_acetate", "cinnamaldehyde_trans",
    ]
    mols = args.molecules or default_mols

    config = ExperimentConfig(data_dir=str(DATA_DIR))
    env = ChemElucidEnv(config)

    runs = []
    for model_key in args.models:
        if model_key not in MODELS:
            logger.warning(f"Unknown model: {model_key}")
            continue
        mc = MODELS[model_key]
        for run_idx in range(args.runs):
            for mol_id in mols:
                runs.append((mol_id, model_key, mc, run_idx))

    total = len(runs)
    logger.info(f"Running {total} episodes ({len(args.models)} models × {len(mols)} mols × {args.runs} runs) at temp={args.temperature}, concurrency={args.concurrency}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.concurrency <= 1:
        for i, (mol_id, mk, mc, ri) in enumerate(runs, 1):
            _run_one(env, mol_id, mk, mc, ri, i, total, args.temperature)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(_run_one, env, mol_id, mk, mc, ri, i, total, args.temperature): (mol_id, mk, ri)
                    for i, (mol_id, mk, mc, ri) in enumerate(runs, 1)}
            for fut in as_completed(futs):
                fut.result()
    logger.info(f"Total runtime: {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
