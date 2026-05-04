"""Shared molecular-formula utilities used by CNMR and HNMR autofill scripts."""

from __future__ import annotations

import re
from typing import Dict, List


def parse_formula_counts(formula: str) -> Dict[str, int]:
    """Parse molecular formula into element counts.

    >>> parse_formula_counts("C8H9NO2")
    {'C': 8, 'H': 9, 'N': 1, 'O': 2}
    """
    if not formula:
        return {}
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    counts: Dict[str, int] = {}
    for elem, num in tokens:
        if not elem:
            continue
        count = int(num) if num else 1
        counts[elem] = counts.get(elem, 0) + count
    return counts


def calculate_unsaturation(formula: str) -> int:
    """Calculate the degree of unsaturation (DBE) from a molecular formula.

    DBE = (2C + 2 + N - H - X) / 2
    where X counts halogens (F, Cl, Br, I).
    """
    counts = parse_formula_counts(formula)
    if not counts:
        raise ValueError(f"Could not parse molecular formula: {formula}")

    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    x = sum(counts.get(e, 0) for e in ("F", "Cl", "Br", "I"))

    dbe = (2 * c + 2 + n - h - x) / 2.0
    return int(round(dbe))


def get_h_count(formula: str) -> int:
    """Get hydrogen count from molecular formula."""
    counts = parse_formula_counts(formula)
    return counts.get("H", 0)


def has_element(counts: Dict[str, int], element: str) -> bool:
    """Check if formula contains a specific element."""
    return counts.get(element, 0) > 0


def has_any_element(counts: Dict[str, int], elements: List[str]) -> bool:
    """Check if formula contains any of the specified elements."""
    return any(has_element(counts, e) for e in elements)


def get_heteroatoms(counts: Dict[str, int]) -> List[str]:
    """Get list of heteroatoms (O, N, S) present in formula."""
    heteroatoms = []
    for h in ["O", "N", "S"]:
        if counts.get(h, 0) > 0:
            heteroatoms.append(h)
    return heteroatoms


def get_halogens(counts: Dict[str, int]) -> List[str]:
    """Return list of halogens present in the formula (full names)."""
    halogens = []
    names = {"F": "Fluorine", "Cl": "Chlorine", "Br": "Bromine", "I": "Iodine"}
    for h in ["F", "Cl", "Br", "I"]:
        if counts.get(h, 0) > 0:
            halogens.append(names[h])
    return halogens
