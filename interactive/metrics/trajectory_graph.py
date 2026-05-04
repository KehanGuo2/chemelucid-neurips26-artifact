"""Trajectory visualization — DOT graph generation.

Converts a tool-call trajectory and its causal analysis into a Graphviz DOT
graph for visual inspection of agent reasoning flow.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Category → color mapping for node styling
_CATEGORY_COLORS: Dict[str, str] = {
    "spectrum_query": "#4A90D9",   # blue
    "computation":    "#26A69A",   # teal
    "structure_analysis": "#FFA726",  # amber
    "prediction":     "#EF5350",   # coral
    "validation":     "#AB47BC",   # purple
    "submit":         "#66BB6A",   # green
    "unknown":        "#BDBDBD",   # grey
}


def trajectory_to_dot(
    trajectory: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    title: str = "Agent Trajectory",
) -> str:
    """Generate a DOT graph string from a trajectory and its causal analysis.

    Args:
        trajectory: Tool-call records from an episode.
        analysis: Output of compute_causal_metrics().
        title: Graph title.

    Returns:
        DOT-format string.
    """
    conflict_indices = {c["idx"] for c in analysis.get("conflicts", [])}

    lines = [
        f'digraph trajectory {{',
        f'  label="{_escape(title)}";',
        '  labelloc="t";',
        '  fontsize=14;',
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontsize=10];',
        '  edge [fontsize=8];',
        '',
    ]

    # Nodes
    for i, step in enumerate(trajectory):
        tool = step.get("tool", "?")
        category = step.get("category", "unknown")
        color = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["unknown"])
        label = f"{i}: {tool}"

        # Add key info to label
        args_summary = _summarize_args(step.get("args", {}))
        if args_summary:
            label += f"\\n{args_summary}"

        border_color = "#D32F2F" if i in conflict_indices else "#424242"
        penwidth = "2.5" if i in conflict_indices else "1.0"

        lines.append(
            f'  n{i} [label="{_escape(label)}", '
            f'fillcolor="{color}", fontcolor="white", '
            f'color="{border_color}", penwidth={penwidth}];'
        )

    lines.append('')

    # Edges
    for edge in analysis.get("edges", []):
        src = edge["from_idx"]
        dst = edge["to_idx"]
        etype = edge.get("type", "")
        detail = edge.get("detail", "")
        edge_label = detail[:40] if detail else etype
        lines.append(
            f'  n{src} -> n{dst} [label="{_escape(edge_label)}", '
            f'color="#2E7D32", fontcolor="#2E7D32"];'
        )

    lines.append('}')
    return '\n'.join(lines)


def render_dot(
    dot_str: str,
    output_path: str,
    fmt: str = "png",
) -> Optional[str]:
    """Render DOT string to an image file using graphviz.

    Returns output_path on success, None if graphviz is not installed.
    """
    if shutil.which("dot") is None:
        return None

    dot_path = output_path.rsplit(".", 1)[0] + ".dot"
    Path(dot_path).write_text(dot_str)

    try:
        subprocess.run(
            ["dot", f"-T{fmt}", dot_path, "-o", output_path],
            check=True, capture_output=True, timeout=30,
        )
        return output_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def save_trajectory_graph(
    trajectory: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    output_dir: str,
    episode_name: str,
    title: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Generate and save DOT (and optionally PNG) for a trajectory.

    Returns dict with "dot" and "png" file paths (png may be None).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    title = title or episode_name
    dot_str = trajectory_to_dot(trajectory, analysis, title=title)

    dot_path = str(Path(output_dir) / f"{episode_name}_graph.dot")
    Path(dot_path).write_text(dot_str)

    png_path = str(Path(output_dir) / f"{episode_name}_graph.png")
    png_result = render_dot(dot_str, png_path)

    return {"dot": dot_path, "png": png_result}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    """Escape special characters for DOT labels."""
    return s.replace('"', '\\"').replace('\n', '\\n')


def _summarize_args(args: Dict[str, Any]) -> str:
    """Compact one-line summary of tool arguments."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            v_str = v[:20] + "..." if len(v) > 20 else v
            parts.append(f"{k}={v_str}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list) and len(v) <= 3:
            parts.append(f"{k}=[{len(v)} items]")
    s = ", ".join(parts)
    return s[:50] + "..." if len(s) > 50 else s
