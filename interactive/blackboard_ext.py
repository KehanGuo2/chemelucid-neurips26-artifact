"""Component 4: Extended Blackboard with tool call history.

Extends the concept from the existing CanonicalFactsState to track tool calls
and automatically extract structured facts from tool results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class BlackboardExt:
    """Extended Blackboard that tracks tool call history and extracts facts.

    This is a standalone class (does not inherit from CanonicalFactsState) to
    avoid tight coupling with the pydantic model. Instead, it maintains a
    compatible interface with confirmed_facts, partially_confirmed_facts, and
    refuted_claims lists.
    """

    def __init__(self):
        self.confirmed_facts: List[str] = []
        self.partially_confirmed_facts: List[str] = []
        self.refuted_claims: List[str] = []
        self.tool_history: List[Dict[str, Any]] = []

        # Structured knowledge slots for quick programmatic access
        self._degree_of_unsaturation: Optional[int] = None
        self._molecular_formula: Optional[str] = None
        self._molecular_weight: Optional[float] = None
        self._has_aromatic: Optional[bool] = None
        self._candidate_smiles: List[str] = []

    def add_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """Add a tool call result to history and extract facts."""
        self.tool_history.append({
            "tool": tool_name,
            "args": args,
            "result": result,
        })

        # Skip extraction if tool returned an error
        if "error" in result:
            return

        self._extract_facts(tool_name, args, result)

    def _extract_facts(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """Extract structured facts from tool results into Blackboard."""

        if tool_name == "compute_unsaturation":
            dbe = result.get("degree")
            formula = result.get("formula", "")
            if dbe is not None:
                self._degree_of_unsaturation = dbe
                self._molecular_formula = formula
                self._add_confirmed(f"Degree of unsaturation (DBE): {dbe} (from {formula})")

        elif tool_name == "compute_molecular_weight":
            mw = result.get("molecular_weight")
            if mw is not None:
                self._molecular_weight = mw
                self._add_confirmed(f"Molecular weight: {mw} g/mol")

        elif tool_name == "query_spectrum":
            n_peaks = result.get("count", 0)
            nucleus = result.get("nucleus", "?")
            rng = result.get("range", [])
            peaks = result.get("peaks", [])
            if isinstance(rng, list) and len(rng) == 2:
                if n_peaks > 0:
                    if nucleus == "13C":
                        peak_strs = ", ".join(f"{p:.1f}" for p in peaks[:8])
                        self._add_confirmed(
                            f"{nucleus} NMR: {n_peaks} peak(s) in {rng[0]}-{rng[1]} ppm "
                            f"[{peak_strs}{'...' if n_peaks > 8 else ''}]"
                        )
                    else:
                        self._add_confirmed(
                            f"{nucleus} NMR: {n_peaks} peak group(s) in {rng[0]}-{rng[1]} ppm"
                        )
                else:
                    self._add_confirmed(
                        f"{nucleus} NMR: no peaks in {rng[0]}-{rng[1]} ppm"
                    )

        elif tool_name == "get_full_spectrum":
            nucleus = result.get("nucleus", "?")
            n_peaks = result.get("count", 0)
            self._add_confirmed(f"{nucleus} NMR: total {n_peaks} peak(s) observed")

        elif tool_name == "substructure_check":
            has_match = result.get("has_match", False)
            pattern = args.get("pattern", "?")
            smiles = args.get("smiles", "?")
            verb = "contains" if has_match else "does NOT contain"
            self._add_confirmed(f"Structure '{smiles}' {verb} pattern '{pattern}'")

        elif tool_name == "check_symmetry":
            unique_envs = result.get("unique_carbon_environments")
            total_c = result.get("total_carbons")
            has_sym = result.get("has_symmetry", False)
            if unique_envs is not None and total_c is not None:
                self._add_confirmed(
                    f"Symmetry: {total_c} carbons → {unique_envs} unique environments "
                    f"({'symmetric' if has_sym else 'asymmetric'})"
                )

        elif tool_name == "count_carbons":
            total = result.get("total_carbons")
            aromatic = result.get("aromatic_carbons", 0)
            if total is not None:
                self._add_confirmed(
                    f"Carbon count: {total} total ({aromatic} aromatic, {total - aromatic} aliphatic)"
                )

        elif tool_name == "validate_smiles":
            if result.get("valid"):
                canonical = result.get("canonical_smiles", "")
                formula = result.get("molecular_formula", "")
                self._add_confirmed(f"Valid SMILES: {canonical} (formula: {formula})")
            else:
                self._add_confirmed(f"Invalid SMILES: {args.get('smiles', '?')}")

        elif tool_name == "compare_spectra":
            similarity = result.get("similarity")
            mae = result.get("mae")
            matched = result.get("matched_peaks", 0)
            total_obs = result.get("total_observed", 0)
            if similarity is not None:
                self._add_confirmed(
                    f"Spectrum comparison: similarity={similarity:.3f}, "
                    f"matched {matched}/{total_obs} observed peaks"
                    + (f", MAE={mae:.1f} ppm" if mae is not None else "")
                )

        elif tool_name == "predict_nmr":
            nucleus = result.get("nucleus", "?")
            n_carbons = result.get("num_carbons", result.get("num_groups", 0))
            smiles = args.get("smiles", "?")
            self._add_confirmed(
                f"Predicted {nucleus} NMR for {smiles}: {n_carbons} signals"
            )

    def _add_confirmed(self, fact: str):
        """Add a fact only if not already present."""
        if fact not in self.confirmed_facts:
            self.confirmed_facts.append(fact)

    def render(self) -> str:
        """Render Blackboard state as text for injection into LLM prompt."""
        lines = []
        lines.append("=== BLACKBOARD (Accumulated Knowledge) ===")

        if self.confirmed_facts:
            lines.append("\nCONFIRMED FACTS:")
            for i, fact in enumerate(self.confirmed_facts, 1):
                lines.append(f"  {i}. {fact}")
        else:
            lines.append("\n(No confirmed facts yet)")

        if self.partially_confirmed_facts:
            lines.append("\nPARTIALLY CONFIRMED:")
            for fact in self.partially_confirmed_facts:
                lines.append(f"  - {fact}")

        if self.refuted_claims:
            lines.append("\nREFUTED:")
            for claim in self.refuted_claims:
                lines.append(f"  - {claim}")

        # Recent tool history (last 10)
        if self.tool_history:
            lines.append("\n=== RECENT TOOL CALLS ===")
            recent = self.tool_history[-10:]
            for i, call in enumerate(recent, max(1, len(self.tool_history) - 9)):
                args_str = _summarize_args(call["args"])
                result_str = _summarize_result(call["result"])
                lines.append(f"  {i}. {call['tool']}({args_str}) → {result_str}")

        return "\n".join(lines)


def _summarize_args(args: Dict[str, Any]) -> str:
    """Produce a compact representation of tool arguments."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v!r}")
    s = ", ".join(parts)
    return s[:120] + "..." if len(s) > 120 else s


def _summarize_result(result: Dict[str, Any]) -> str:
    """Produce a compact representation of a tool result."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    # Pick the most informative key
    for key in ("degree", "molecular_weight", "has_match", "similarity",
                "count", "valid", "submitted"):
        if key in result:
            return f"{key}={result[key]}"

    # Fallback: dump top-level keys
    keys = list(result.keys())[:4]
    return "{" + ", ".join(f"{k}=..." for k in keys) + "}"
