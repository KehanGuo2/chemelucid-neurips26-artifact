#!/usr/bin/env python3
"""Probe batch runner: canonical models x withheld-probe molecules x info modes.

The default scope expands the withheld-probe transfer experiment from the
two-Anthropic, N=8 setup to the canonical four-model core scope. Existing
results under interactive/results/canary_probe/ are reused via the
probe_NNN <-> canary_NNN mapping in data/withheld_probe/probe_48_manifest.csv
(legacy_internal_id), so only the genuinely new cells are queued for API runs.

Default new cells (with claude-sonnet-4 and claude-opus already on disk):

    gpt-5.2 + o4-mini  x  probe_001..008  x  {partial_autonomous, full_autonomous}
    = 32 episodes

Usage:
    python scripts/run_probe_batch.py --dry-run
    python scripts/run_probe_batch.py --models gpt-5.2 o4-mini --concurrency 2
    python scripts/run_probe_batch.py --models all --concurrency 2

Output goes to interactive/results/withheld_probe/<probe_id>_<model_key>_<info_mode>_autonomous/.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from interactive.config import DATA_DIR, ExperimentConfig, MODELS  # noqa: E402
from interactive.environment import ChemElucidEnv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_MODELS = ["gpt-5.2", "claude-sonnet-4", "claude-opus", "o4-mini"]
DEFAULT_NEW_MODELS = ["gpt-5.2", "o4-mini"]
PROBE_IDS = [f"probe_{i:03d}" for i in range(1, 9)]
INFO_MODES = ["partial", "full"]

PROBE_OUT_DIR = REPO / "interactive" / "results" / "withheld_probe"
LEGACY_DIR = REPO / "interactive" / "results" / "canary_probe"
MANIFEST = REPO / "data" / "withheld_probe" / "probe_48_manifest.csv"


def load_legacy_alias_map() -> dict[str, str]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing manifest: {MANIFEST}")
    aliases: dict[str, str] = {}
    with MANIFEST.open() as fh:
        for row in csv.DictReader(fh):
            pid = row.get("probe_id")
            legacy = row.get("legacy_internal_id")
            if pid and legacy:
                aliases[pid] = legacy
    return aliases


def existing_canary_result(
    probe_id: str, model_key: str, info_mode: str, alias_map: dict[str, str]
) -> Path | None:
    legacy_id = alias_map.get(probe_id)
    if not legacy_id:
        return None
    cond = f"{info_mode}_autonomous"
    sub = LEGACY_DIR / f"{legacy_id}_{model_key}_{cond}"
    if not sub.exists():
        return None
    diags = list(sub.glob("*.json"))
    return diags[0] if diags else None


def out_subdir(probe_id: str, model_key: str, info_mode: str) -> Path:
    return PROBE_OUT_DIR / f"{probe_id}_{model_key}_{info_mode}_autonomous"


def already_have_new_result(probe_id: str, model_key: str, info_mode: str) -> bool:
    sub = out_subdir(probe_id, model_key, info_mode)
    return sub.exists() and any(sub.glob("*.json"))


def run_one(env: ChemElucidEnv, probe_id: str, model_key: str, info_mode: str) -> dict:
    mc = MODELS[model_key]
    out_dir = out_subdir(probe_id, model_key, info_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond = f"{info_mode}_autonomous"
    logger.info(f"START {probe_id} | {model_key} | {cond}")
    t0 = time.time()
    try:
        diag = env.run_episode(
            molecule_id=probe_id,
            provider=mc.provider,
            model=mc.model,
            stateful=True,
            budget=25,
            spectrum_type="CNMR",
            info_mode=info_mode,
            reasoning_scaffold=None,
            probe_mode=True,
        )
        diag["condition"] = cond
        diag["scaffold"] = False
        env.save_diagnostic(diag, str(out_dir))
        outcome = diag.get("outcome_only", {})
        logger.info(
            f"DONE  {probe_id} | {model_key} | {cond} | "
            f"inchi={outcome.get('inchi_exact')} "
            f"tani={outcome.get('tanimoto', 0):.3f} "
            f"cost={diag.get('total_cost')} "
            f"elapsed={time.time()-t0:.1f}s"
        )
        return diag
    except Exception as e:  # noqa: BLE001
        logger.error(f"FAIL  {probe_id} | {model_key} | {cond}: {e}")
        return {
            "molecule_id": probe_id,
            "model_key": model_key,
            "info_mode": info_mode,
            "error": str(e),
        }


def enumerate_cells(
    models: list[str],
) -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str, str, Path]],
    list[tuple[str, str, str]],
]:
    alias_map = load_legacy_alias_map()
    to_run: list[tuple[str, str, str]] = []
    reusable: list[tuple[str, str, str, Path]] = []
    already: list[tuple[str, str, str]] = []
    for model_key in models:
        for probe in PROBE_IDS:
            for info in INFO_MODES:
                if already_have_new_result(probe, model_key, info):
                    already.append((probe, model_key, info))
                    continue
                legacy = existing_canary_result(probe, model_key, info, alias_map)
                if legacy is not None:
                    reusable.append((probe, model_key, info, legacy))
                    continue
                to_run.append((probe, model_key, info))
    return to_run, reusable, already


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_NEW_MODELS,
        help="Model keys (must be in MODELS). Use 'all' for the canonical four.",
    )
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument(
        "--dry-run", action="store_true", help="Enumerate cells without running."
    )
    args = ap.parse_args()

    models = CANONICAL_MODELS if args.models == ["all"] else args.models
    for k in models:
        if k not in MODELS:
            raise SystemExit(f"Unknown model key: {k}")

    to_run, reusable, already = enumerate_cells(models)

    print(f"Models requested            : {models}")
    print(f"Cells already in withheld_probe/: {len(already)}")
    print(f"Cells reusable from canary_probe/: {len(reusable)}")
    print(f"Cells to run                : {len(to_run)}")
    if reusable:
        sample = [(p, m, i) for p, m, i, _ in reusable[:3]]
        print(f"  (reusable examples: {sample})")
    if to_run:
        print(f"  (to-run examples:   {to_run[:3]})")

    if args.dry_run:
        print("\nDry run; no API calls.")
        return 0

    if not to_run:
        print("\nNothing to run.")
        return 0

    config = ExperimentConfig(data_dir=str(DATA_DIR), budget=25, spectrum_type="CNMR")
    env = ChemElucidEnv(config)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, env, p, m, i): (p, m, i) for p, m, i in to_run}
        for fut in as_completed(futs):
            results.append(fut.result())

    n_ok = sum(1 for r in results if "error" not in r)
    n_fail = sum(1 for r in results if "error" in r)
    logger.info(f"Completed {n_ok} ok, {n_fail} fail.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
