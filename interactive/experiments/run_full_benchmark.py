"""Full benchmark: 18 molecules × 2 models, partial mode only.

Reuses existing partial-mode results for the 4 pilot molecules,
runs the remaining 14 × 2 = 28 new episodes.

Usage:
    python -m interactive.experiments.run_full_benchmark
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from interactive.config import ALL_MOLECULES, DATA_DIR, ExperimentConfig, MODELS
from interactive.environment import ChemElucidEnv
from interactive.metrics.causal_analysis import compute_causal_metrics
from interactive.metrics.layer1 import compute_layer1
from interactive.metrics.trajectory_graph import save_trajectory_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_KEYS = ["gpt-5.2", "claude-sonnet-4"]
RESULTS_DIR = Path("interactive/results/v2_full_benchmark")

# Existing partial results from the pilot run
PILOT_DIR = Path("interactive/results/v2_dual")
PILOT_MOLECULES = {"2_4_dimethyl_aniline", "ibuprofen", "saccharine", "theophylline"}


def load_pilot_results() -> List[Dict[str, Any]]:
    """Load existing partial-mode results for the 4 pilot molecules."""
    results = []
    for model_key in MODEL_KEYS:
        pilot_sub = PILOT_DIR / f"{model_key}_partial"
        if not pilot_sub.exists():
            continue
        for f in sorted(pilot_sub.glob("*.json")):
            data = json.load(open(f))
            if data.get("terminated_by") == "error":
                continue
            mol_id = data.get("molecule_id", "")
            if mol_id in PILOT_MOLECULES and data.get("info_mode") == "partial":
                results.append(data)
                logger.info(f"  Reusing pilot: {mol_id} / {data.get('model', '?')}")
    return results


def run_remaining(env: ChemElucidEnv, existing_mols: Dict[str, set]) -> List[Dict[str, Any]]:
    """Run episodes for molecules not yet covered."""
    results = []
    remaining = []
    for model_key in MODEL_KEYS:
        mc = MODELS[model_key]
        done = existing_mols.get(mc.model, set())
        for mol_id in ALL_MOLECULES:
            if mol_id not in done:
                remaining.append((mol_id, model_key, mc))

    total = len(remaining)
    logger.info(f"Running {total} new episodes...")

    for i, (mol_id, model_key, mc) in enumerate(remaining, 1):
        out_dir = RESULTS_DIR / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{i}/{total}] {mol_id} | {model_key} | partial")
        try:
            diag = env.run_episode(
                molecule_id=mol_id,
                provider=mc.provider,
                model=mc.model,
                stateful=True,
                budget=25,
                spectrum_type="CNMR",
                info_mode="partial",
            )
            env.save_diagnostic(diag, str(out_dir))
            acc = diag["accuracy"]
            l1 = diag.get("layer1", {})
            logger.info(
                f"  exact={acc.get('exact_match')}, tani={acc.get('tanimoto',0):.3f}, "
                f"cost={diag['total_cost']}, L1={l1.get('score',0):.3f}, "
                f"L3dep={diag.get('layer3_dependency_rate',0):.3f}, "
                f"ended={diag['terminated_by']}"
            )
            results.append(diag)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({"molecule_id": mol_id, "model": mc.model, "info_mode": "partial", "error": str(e)})

    return results


def get_dou(mol_id: str) -> Optional[int]:
    """Get degree of unsaturation from the molecule's GT data."""
    from chem_defense.utils.formula_utils import parse_formula_counts, calculate_unsaturation
    gt_path = Path(DATA_DIR) / "C_NMR" / f"{mol_id}_CNMR_HRE.json"
    if not gt_path.exists():
        return None
    import re
    data = json.load(open(gt_path))
    nodes = data.get("Nodes", data.get("nodes", []))
    for node in nodes:
        if node.get("id") == "BIG QUESTION":
            label = node.get("label", "")
            m = re.search(r"\{Molecular Formula:\s*([^}]+)\}", label)
            if m:
                formula = m.group(1).strip()
                return calculate_unsaturation(formula)
    return None


def generate_report(all_results: List[Dict[str, Any]]):
    """Generate the full benchmark report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    graph_dir = RESULTS_DIR / "graphs"
    graph_dir.mkdir(exist_ok=True)

    # Filter valid results
    valid = [r for r in all_results if "error" not in r]
    logger.info(f"Generating report from {len(valid)} valid results ({len(all_results)} total)")

    # Get DoU for each molecule
    dou_map = {}
    for mol_id in ALL_MOLECULES:
        dou_map[mol_id] = get_dou(mol_id)

    lines = ["# Full Benchmark: 18 Molecules × 2 Models (Partial Mode)\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Episodes: {len(valid)} valid / {len(all_results)} total")
    lines.append(f"Mode: partial (agent must query spectrum via tools)")
    lines.append(f"Budget: 25 per episode")
    lines.append("")

    # =========================================================
    # 1. Full summary table
    # =========================================================
    lines.append("## 1. Full Summary Table\n")
    lines.append("| Molecule | DoU | Model | Exact | Tanimoto | L1 | L2 (cov) | L3 dep | L3 react | Cost | Turns |")
    lines.append("|----------|-----|-------|-------|----------|----|----------|--------|----------|------|-------|")

    for r in sorted(valid, key=lambda x: (x.get("molecule_id", ""), x.get("model", ""))):
        acc = r.get("accuracy", {})
        l1 = r.get("layer1", {})
        mol_id = r["molecule_id"]
        model_short = "gpt-5.2" if "gpt" in r.get("model", "") else "claude-4"
        dou = dou_map.get(mol_id, "?")
        lines.append(
            f"| {mol_id} | {dou} | {model_short} | "
            f"{'Y' if acc.get('exact_match') else 'N'} | "
            f"{acc.get('tanimoto', 0):.3f} | "
            f"{l1.get('score', 0):.3f} | "
            f"{(r.get('coverage') or 0):.3f} | "
            f"{r.get('layer3_dependency_rate', 0):.3f} | "
            f"{r.get('layer3_conflict_reactivity', 1.0):.3f} | "
            f"{r.get('total_cost', '?')} | "
            f"{r.get('num_turns', '?')} |"
        )
    lines.append("")

    # =========================================================
    # 2. Overall pass rates
    # =========================================================
    lines.append("## 2. Overall Pass Rates\n")

    n = len(valid)
    exact_n = sum(1 for r in valid if r["accuracy"].get("exact_match"))
    tani50 = sum(1 for r in valid if r["accuracy"].get("tanimoto", 0) >= 0.5)
    tani30 = sum(1 for r in valid if r["accuracy"].get("tanimoto", 0) >= 0.3)
    avg_tani = sum(r["accuracy"].get("tanimoto", 0) for r in valid) / n if n else 0

    lines.append(f"| Metric | Count | Rate |")
    lines.append(f"|--------|-------|------|")
    lines.append(f"| Exact match | {exact_n}/{n} | **{exact_n/n:.1%}** |")
    lines.append(f"| Tanimoto ≥ 0.5 | {tani50}/{n} | **{tani50/n:.1%}** |")
    lines.append(f"| Tanimoto ≥ 0.3 | {tani30}/{n} | **{tani30/n:.1%}** |")
    lines.append(f"| Avg Tanimoto | — | **{avg_tani:.3f}** |")
    lines.append("")
    lines.append(f"TB-Science target: 10–20% solve rate for frontier models.")
    lines.append("")

    # =========================================================
    # 3. Per-model comparison
    # =========================================================
    lines.append("## 3. Per-Model Comparison\n")
    lines.append("| Metric | GPT-5.2 | Claude Sonnet 4 |")
    lines.append("|--------|---------|-----------------|")

    for model_label, model_substr in [("GPT-5.2", "gpt"), ("Claude Sonnet 4", "claude")]:
        subset = [r for r in valid if model_substr in r.get("model", "")]
        ns = len(subset)
        if ns == 0:
            continue
        ex = sum(1 for r in subset if r["accuracy"].get("exact_match"))
        t5 = sum(1 for r in subset if r["accuracy"].get("tanimoto", 0) >= 0.5)
        t3 = sum(1 for r in subset if r["accuracy"].get("tanimoto", 0) >= 0.3)
        at = sum(r["accuracy"].get("tanimoto", 0) for r in subset) / ns
        al1 = sum(r.get("layer1", {}).get("score", 0) for r in subset) / ns
        al2 = sum((r.get("coverage") or 0) for r in subset) / ns
        al3 = sum(r.get("layer3_dependency_rate", 0) for r in subset) / ns
        ar = sum(r.get("layer3_conflict_reactivity", 1.0) for r in subset) / ns
        ac = sum(r.get("total_cost", 0) for r in subset) / ns

        if model_substr == "gpt":
            gpt_vals = (ex, ns, t5, t3, at, al1, al2, al3, ar, ac)
        else:
            cla_vals = (ex, ns, t5, t3, at, al1, al2, al3, ar, ac)

    g = gpt_vals
    c = cla_vals
    lines.append(f"| Episodes | {g[1]} | {c[1]} |")
    lines.append(f"| Exact match | {g[0]}/{g[1]} ({g[0]/g[1]:.0%}) | {c[0]}/{c[1]} ({c[0]/c[1]:.0%}) |")
    lines.append(f"| Tanimoto ≥ 0.5 | {g[2]}/{g[1]} ({g[2]/g[1]:.0%}) | {c[2]}/{c[1]} ({c[2]/c[1]:.0%}) |")
    lines.append(f"| Tanimoto ≥ 0.3 | {g[3]}/{g[1]} ({g[3]/g[1]:.0%}) | {c[3]}/{c[1]} ({c[3]/c[1]:.0%}) |")
    lines.append(f"| Avg Tanimoto | {g[4]:.3f} | {c[4]:.3f} |")
    lines.append(f"| Avg L1 | {g[5]:.3f} | {c[5]:.3f} |")
    lines.append(f"| Avg L2 (coverage) | {g[6]:.3f} | {c[6]:.3f} |")
    lines.append(f"| Avg L3 dependency | {g[7]:.3f} | {c[7]:.3f} |")
    lines.append(f"| Avg L3 reactivity | {g[8]:.3f} | {c[8]:.3f} |")
    lines.append(f"| Avg cost | {g[9]:.1f} | {c[9]:.1f} |")
    lines.append("")

    # =========================================================
    # 4. Molecules ranked by difficulty
    # =========================================================
    lines.append("## 4. Molecules Ranked by Difficulty (Easiest → Hardest)\n")
    lines.append("Ranked by average Tanimoto across both models.\n")

    mol_stats: Dict[str, Dict] = {}
    for mol_id in ALL_MOLECULES:
        mol_results = [r for r in valid if r["molecule_id"] == mol_id]
        if not mol_results:
            mol_stats[mol_id] = {"avg_tani": 0, "n": 0, "exact": 0, "dou": dou_map.get(mol_id, "?")}
            continue
        nm = len(mol_results)
        avg_t = sum(r["accuracy"].get("tanimoto", 0) for r in mol_results) / nm
        ex = sum(1 for r in mol_results if r["accuracy"].get("exact_match"))
        mol_stats[mol_id] = {"avg_tani": avg_t, "n": nm, "exact": ex, "dou": dou_map.get(mol_id, "?")}

    ranked = sorted(mol_stats.items(), key=lambda x: -x[1]["avg_tani"])

    lines.append("| Rank | Molecule | DoU | Avg Tanimoto | Exact | N |")
    lines.append("|------|----------|-----|--------------|-------|---|")
    for i, (mol_id, st) in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {mol_id} | {st['dou']} | {st['avg_tani']:.3f} | "
            f"{st['exact']}/{st['n']} | {st['n']} |"
        )
    lines.append("")

    # =========================================================
    # 5. Trajectory graphs: highest and lowest L3
    # =========================================================
    lines.append("## 5. Notable Trajectory Graphs\n")

    # Sort by L3 dependency rate
    by_l3 = sorted(valid, key=lambda r: r.get("layer3_dependency_rate", 0))
    lowest2 = by_l3[:2]
    highest2 = by_l3[-2:]

    for label, subset in [("Highest L3 dependency", highest2), ("Lowest L3 dependency", lowest2)]:
        lines.append(f"### {label}\n")
        for r in subset:
            mol_id = r["molecule_id"]
            model_short = "gpt52" if "gpt" in r.get("model", "") else "claude4"
            l3d = r.get("layer3_dependency_rate", 0)
            acc = r["accuracy"]
            lines.append(
                f"- **{mol_id}** ({model_short}): L3dep={l3d:.3f}, "
                f"tani={acc.get('tanimoto', 0):.3f}, cost={r.get('total_cost', '?')}"
            )
            # Generate graph
            traj = r.get("trajectory", [])
            if traj:
                analysis = compute_causal_metrics(traj)
                ep_name = f"{mol_id}_{model_short}_partial_notable"
                title = f"{mol_id} ({model_short}) L3dep={l3d:.2f}"
                save_trajectory_graph(traj, analysis, str(graph_dir), ep_name, title=title)
                lines.append(f"  - Graph: `graphs/{ep_name}_graph.dot`")
        lines.append("")

    report_path = RESULTS_DIR / "benchmark_report.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved to {report_path}")


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    # 1. Load pilot results
    logger.info("Loading pilot results...")
    pilot = load_pilot_results()
    logger.info(f"  Loaded {len(pilot)} pilot results")

    # Track what we already have: model -> set of mol_ids
    existing: Dict[str, set] = {}
    for r in pilot:
        model = r.get("model", "")
        existing.setdefault(model, set()).add(r["molecule_id"])

    # 2. Run remaining
    config = ExperimentConfig(data_dir=str(DATA_DIR), budget=25, spectrum_type="CNMR")
    env = ChemElucidEnv(config)
    new_results = run_remaining(env, existing)

    # 3. Combine
    all_results = pilot + new_results
    logger.info(f"Total results: {len(all_results)}")

    # 4. Generate report
    generate_report(all_results)

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
