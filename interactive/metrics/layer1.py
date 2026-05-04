"""Layer 1: Information Acquisition Score.

Measures how thoroughly the agent explored the available spectral and
analytical space — unique spectral regions queried and tool categories used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def compute_layer1(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Layer 1 information-acquisition metrics.

    Args:
        trajectory: Tool-call records from an episode.

    Returns:
        Dict with score, unique_regions, regions, categories_used, categories.
    """
    queried_regions: Set[str] = set()
    categories: Set[str] = set()

    for call in trajectory:
        cat = call.get("category", "unknown")
        categories.add(cat)

        tool = call.get("tool", "")
        if tool in ("query_spectrum", "get_multiplicity",
                     "get_integration", "get_coupling"):
            args = call.get("args", {})
            nucleus = args.get("nucleus", "1H")
            ppm_min = int(args.get("ppm_min", 0))
            ppm_max = int(args.get("ppm_max", 15))
            region = f"{nucleus}_{ppm_min}_{ppm_max}"
            queried_regions.add(region)

    # Normalize: max reasonable regions ~8, max categories = 6
    region_score = min(len(queried_regions) / 8, 1.0)
    category_score = min(len(categories) / 5, 1.0)

    return {
        "score": round((region_score + category_score) / 2, 4),
        "unique_regions": len(queried_regions),
        "regions": sorted(queried_regions),
        "categories_used": len(categories),
        "categories": sorted(categories),
    }
