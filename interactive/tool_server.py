"""Component 1: Chemistry Tool Server for interactive structure elucidation.

Provides a structured set of chemistry tools that an LLM agent can call to
query spectra, compute properties, analyze structures, predict NMR, enumerate
isomers, and validate candidates.

Each tool takes structured input and returns structured output. All calls are
logged for post-hoc DAG grading.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chem_defense.utils.formula_utils import (
    parse_formula_counts,
    calculate_unsaturation,
    get_heteroatoms,
    get_halogens,
    has_element,
    has_any_element,
)
from chem_defense.utils.smiles_utils import (
    canonicalize_smiles,
    compute_tanimoto_similarity,
)

# ---------------------------------------------------------------------------
# Tool Definitions (used to generate OpenAI/Anthropic function schemas)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Category 1: Spectrum Query
    "query_spectrum": {
        "description": "Retrieve NMR peaks within a specified chemical shift range.",
        "parameters": {
            "type": "object",
            "properties": {
                "nucleus": {
                    "type": "string",
                    "enum": ["13C", "1H"],
                    "description": "NMR nucleus type",
                },
                "ppm_min": {
                    "type": "number",
                    "description": "Minimum chemical shift in ppm",
                },
                "ppm_max": {
                    "type": "number",
                    "description": "Maximum chemical shift in ppm",
                },
            },
            "required": ["nucleus", "ppm_min", "ppm_max"],
        },
        "cost": 1,
        "category": "spectrum_query",
    },
    "get_full_spectrum": {
        "description": "Retrieve the complete NMR spectrum (all peaks) for a given nucleus.",
        "parameters": {
            "type": "object",
            "properties": {
                "nucleus": {
                    "type": "string",
                    "enum": ["13C", "1H"],
                    "description": "NMR nucleus type",
                },
            },
            "required": ["nucleus"],
        },
        "cost": 1,
        "category": "spectrum_query",
    },
    # Category 2: Computation
    "compute_unsaturation": {
        "description": "Calculate degree of unsaturation (DBE) from molecular formula. DBE = (2C + 2 + N - H - X) / 2.",
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Molecular formula e.g. C8H11N",
                },
            },
            "required": ["formula"],
        },
        "cost": 1,
        "category": "computation",
    },
    "compute_molecular_weight": {
        "description": "Calculate molecular weight from molecular formula.",
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {
                    "type": "string",
                    "description": "Molecular formula e.g. C8H11N",
                },
            },
            "required": ["formula"],
        },
        "cost": 1,
        "category": "computation",
    },
    # Category 3: Structure Analysis
    "substructure_check": {
        "description": "Check if a SMILES structure contains a specific substructure (SMARTS pattern).",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string to check",
                },
                "pattern": {
                    "type": "string",
                    "description": "SMARTS pattern to search for (e.g. 'c1ccccc1' for aromatic ring)",
                },
            },
            "required": ["smiles", "pattern"],
        },
        "cost": 1,
        "category": "structure_analysis",
    },
    "count_carbons": {
        "description": "Count the number of carbon atoms in a SMILES structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string",
                },
            },
            "required": ["smiles"],
        },
        "cost": 1,
        "category": "structure_analysis",
    },
    "check_symmetry": {
        "description": "Analyze symmetry of a molecular structure. Returns the number of unique carbon environments.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string",
                },
            },
            "required": ["smiles"],
        },
        "cost": 1,
        "category": "structure_analysis",
    },
    "validate_smiles": {
        "description": "Validate a SMILES string and return canonical form, molecular formula, and basic properties.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string to validate",
                },
            },
            "required": ["smiles"],
        },
        "cost": 2,
        "category": "structure_analysis",
    },
    # Category 4: Prediction
    "predict_nmr": {
        "description": "Predict approximate NMR chemical shifts for a candidate SMILES structure using functional group heuristics.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Candidate SMILES string",
                },
                "nucleus": {
                    "type": "string",
                    "enum": ["13C", "1H"],
                    "description": "NMR nucleus type",
                },
            },
            "required": ["smiles", "nucleus"],
        },
        "cost": 3,
        "category": "prediction",
    },
    # Category 5: Validation
    "compare_spectra": {
        "description": "Compare predicted vs observed NMR spectra quantitatively (MAE, matched peaks).",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_shifts": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Predicted chemical shifts",
                },
                "observed_shifts": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Observed chemical shifts",
                },
                "nucleus": {
                    "type": "string",
                    "enum": ["13C", "1H"],
                    "description": "NMR nucleus type",
                },
            },
            "required": ["predicted_shifts", "observed_shifts", "nucleus"],
        },
        "cost": 1,
        "category": "validation",
    },
    # Category 7: HNMR-specific tools
    "get_multiplicity": {
        "description": "Get the multiplicity (s, d, t, q, m, dd, dt, ...) for 1H NMR peaks. Returns multiplicity for all peaks or within a ppm range.",
        "parameters": {
            "type": "object",
            "properties": {
                "ppm_min": {
                    "type": "number",
                    "description": "Minimum ppm (optional, default 0)",
                },
                "ppm_max": {
                    "type": "number",
                    "description": "Maximum ppm (optional, default 15)",
                },
            },
            "required": [],
        },
        "cost": 1,
        "category": "spectrum_query",
        "requires_hnmr": True,
    },
    "get_integration": {
        "description": "Get the integration (number of protons) for 1H NMR peaks. Returns integration for all peaks or within a ppm range.",
        "parameters": {
            "type": "object",
            "properties": {
                "ppm_min": {
                    "type": "number",
                    "description": "Minimum ppm (optional, default 0)",
                },
                "ppm_max": {
                    "type": "number",
                    "description": "Maximum ppm (optional, default 15)",
                },
            },
            "required": [],
        },
        "cost": 1,
        "category": "spectrum_query",
        "requires_hnmr": True,
    },
    "get_coupling": {
        "description": "Get J-coupling values (in Hz) for 1H NMR peaks. Returns coupling constants for all peaks or within a ppm range.",
        "parameters": {
            "type": "object",
            "properties": {
                "ppm_min": {
                    "type": "number",
                    "description": "Minimum ppm (optional, default 0)",
                },
                "ppm_max": {
                    "type": "number",
                    "description": "Maximum ppm (optional, default 15)",
                },
            },
            "required": [],
        },
        "cost": 1,
        "category": "spectrum_query",
        "requires_hnmr": True,
    },
    # Category 6: Submit (special — terminates the episode)
    "submit": {
        "description": "Submit your final answer as a SMILES string. This ends the episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Your proposed SMILES string for the unknown molecule",
                },
            },
            "required": ["smiles"],
        },
        "cost": 0,
        "category": "submit",
    },
}


# ---------------------------------------------------------------------------
# Helper: Use shared formula parsing from chem_defense.utils.formula_utils
# ---------------------------------------------------------------------------

_parse_formula = parse_formula_counts  # alias for backward compat

# Atomic weights for MW calculation
_ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "S": 32.06, "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.90,
    "P": 30.974, "Si": 28.086,
}


# ---------------------------------------------------------------------------
# Spectrum parsing from molecule JSON
# ---------------------------------------------------------------------------

def _parse_spectrum_from_big_question(label: str) -> Tuple[List[float], str, str, str]:
    """Extract spectrum peaks, solvent, MHz, and formula from BIG QUESTION label.

    Returns (peaks, solvent, mhz, formula).
    """
    peaks: List[float] = []
    solvent = ""
    mhz = ""
    formula = ""

    # Extract spectra from {Spectra: ...} or δ values
    spec_match = re.search(r"\{Spectra:\s*(?:δ\s*)?([\d.,\s]+)\}", label)
    if spec_match:
        raw = spec_match.group(1)
        for v in raw.split(","):
            v = v.strip()
            if v:
                try:
                    peaks.append(float(v))
                except ValueError:
                    pass

    # Extract solvent
    sol_match = re.search(r"\{Solvent:\s*([^}]+)\}", label)
    if sol_match:
        solvent = sol_match.group(1).strip()

    # Extract MHz
    mhz_match = re.search(r"\{MHz Value:\s*([^}]+)\}", label)
    if mhz_match:
        mhz = mhz_match.group(1).strip()

    # Extract formula
    form_match = re.search(r"\{Molecular Formula:\s*([^}]+)\}", label)
    if form_match:
        formula = form_match.group(1).strip()

    return peaks, solvent, mhz, formula


def _parse_hnmr_spectrum_rich(label: str) -> list:
    """Parse 1H NMR spectrum from BIG QUESTION using the rich HNMRPeak parser.

    Returns a list of HNMRPeak objects with multiplicity, integration, and J-values.
    Falls back to simplified parsing if the rich parser fails.
    """
    try:
        from chem_defense.utils.hnmr_parser import parse_hnmr_peaks, HNMRPeak

        # Extract spectra string from BIG QUESTION label
        spec_match = re.search(r"\{Spectra:\s*(?:δ\s*)?(.+?)\}", label, re.DOTALL)
        if not spec_match:
            return []

        raw = spec_match.group(1).strip()
        peaks = parse_hnmr_peaks(raw)
        return peaks
    except (ImportError, Exception):
        # Fall back to simplified parsing
        return _parse_hnmr_spectrum_simple(label)


def _parse_hnmr_spectrum_simple(label: str) -> list:
    """Simplified fallback HNMR parser returning dicts (legacy compat)."""
    peaks: list = []

    spec_match = re.search(r"\{Spectra:\s*(?:δ\s*)?(.+?)\}", label, re.DOTALL)
    if not spec_match:
        return peaks

    raw = spec_match.group(1).strip()

    for m in re.finditer(r"([\d.]+)\s*(?:\(([^)]*)\))?", raw):
        ppm_str = m.group(1)
        info = m.group(2) or ""
        try:
            ppm = float(ppm_str)
            peak_info: Dict[str, Any] = {"ppm": ppm}
            if info:
                peak_info["info"] = info.strip()
            peaks.append(peak_info)
        except ValueError:
            pass

    return peaks


def _hnmr_peak_to_dict(peak) -> Dict[str, Any]:
    """Convert an HNMRPeak object to a serializable dict for API response."""
    if isinstance(peak, dict):
        return peak  # Already a dict (legacy path)
    # HNMRPeak dataclass
    d: Dict[str, Any] = {
        "ppm": round(peak.ppm, 2),
        "multiplicity": peak.multiplicity,
        "integration": peak.integration,
    }
    if peak.j_values:
        d["j_values"] = [round(j, 1) for j in peak.j_values]
    if peak.ppm_range:
        d["ppm_range"] = [round(peak.ppm_range[0], 2), round(peak.ppm_range[1], 2)]
    if peak.raw_text:
        d["raw_text"] = peak.raw_text
    return d


def _hnmr_peak_ppm(peak) -> float:
    """Get ppm value from either an HNMRPeak object or legacy dict."""
    if isinstance(peak, dict):
        return peak.get("ppm", 0.0)
    return peak.ppm


def _hnmr_peak_in_range(peak, ppm_min: float, ppm_max: float) -> bool:
    """Check if an HNMR peak is within a ppm range."""
    ppm = _hnmr_peak_ppm(peak)
    return ppm_min <= ppm <= ppm_max


# ---------------------------------------------------------------------------
# ToolServer
# ---------------------------------------------------------------------------

class ToolServer:
    """Provides chemistry tools for interactive structure elucidation.

    Loads spectrum data and ground truth for a specific molecule, then
    executes tool calls and logs all interactions for post-hoc grading.
    """

    def __init__(
        self,
        molecule_id: str,
        data_dir: str,
        spectrum_type: str = "CNMR",
        info_mode: str = "full",
    ):
        """
        Args:
            molecule_id: Molecule identifier (e.g., "2_4_dimethyl_aniline")
            data_dir: Path to data directory containing C_NMR/, H_NMR/, etc.
            spectrum_type: "CNMR", "HNMR", or "COMBO"
            info_mode: "full" or "partial". In partial mode, get_full_spectrum
                       is hidden from the agent (must use query_spectrum).
        """
        self.molecule_id = molecule_id
        self.data_dir = Path(data_dir)
        self.spectrum_type = spectrum_type
        self.info_mode = info_mode
        self.call_log: List[Dict[str, Any]] = []

        # Load molecule data
        self._cnmr_data: Optional[Dict] = None
        self._hnmr_data: Optional[Dict] = None
        self._cnmr_peaks: List[float] = []
        self._hnmr_peaks: list = []  # HNMRPeak objects or legacy dicts
        self._formula: str = ""
        self._solvent: str = ""
        self._mhz: str = ""
        self._gt_smiles: str = ""

        self._load_data()

    def _load_data(self):
        """Load spectrum data from molecule JSON files.

        Prefers split-data layout (data_split/public_task_data/) when present —
        this keeps SMILES answer + DAG checkpoints out of any file the agent
        could read via shell access. Falls back to legacy single-file layout.
        """
        from interactive.data_loader import (
            has_split_data, load_public_task, load_private_grader,
            has_probe_spectrum, load_probe_spectrum, load_probe_hidden_smiles,
        )

        # ---- CNMR (also feeds COMBO) ----
        loaded_cnmr_via_split = False

        # Withheld-probe layer takes precedence: agent-facing spectra live under
        # data/withheld_probe/spectra/ and have no DAG. Hidden SMILES is loaded
        # from the private manifest but is never returned by any tool.
        if self.spectrum_type in ("CNMR", "COMBO") and has_probe_spectrum(
            self.molecule_id, "CNMR", self.data_dir
        ):
            pub = load_probe_spectrum(self.molecule_id, "CNMR", self.data_dir)
            if pub:
                self._cnmr_peaks = list(pub.cnmr_peaks)
                self._formula = pub.molecular_formula
                self._solvent = pub.solvent
                self._mhz = pub.mhz
                loaded_cnmr_via_split = True
                hidden = load_probe_hidden_smiles(self.molecule_id, self.data_dir)
                if hidden:
                    self._gt_smiles = hidden

        if not loaded_cnmr_via_split and self.spectrum_type in ("CNMR", "COMBO") and has_split_data(
            self.molecule_id, "CNMR", self.data_dir
        ):
            pub = load_public_task(self.molecule_id, "CNMR", self.data_dir)
            priv = load_private_grader(self.molecule_id, "CNMR", self.data_dir)
            if pub:
                self._cnmr_peaks = list(pub.cnmr_peaks)
                self._formula = pub.molecular_formula
                self._solvent = pub.solvent
                self._mhz = pub.mhz
                loaded_cnmr_via_split = True
            if priv:
                # gt_smiles is needed by environment.py for the accuracy check.
                # Stored on the server but never returned by any tool.
                self._gt_smiles = priv.gt_smiles

        cnmr_path = self.data_dir / "C_NMR" / f"{self.molecule_id}_CNMR_HRE.json"
        if not loaded_cnmr_via_split and cnmr_path.exists():
            with open(cnmr_path) as f:
                self._cnmr_data = json.load(f)
            nodes = self._cnmr_data.get("Nodes", self._cnmr_data.get("nodes", []))
            for node in nodes:
                if node.get("id") == "BIG QUESTION":
                    self._cnmr_peaks, self._solvent, self._mhz, self._formula = \
                        _parse_spectrum_from_big_question(node.get("label", ""))
                    self._gt_smiles = node.get("answer", "")
                    break

        # ---- HNMR (also feeds COMBO) ----
        loaded_hnmr_via_split = False

        if self.spectrum_type in ("HNMR", "COMBO") and has_probe_spectrum(
            self.molecule_id, "HNMR", self.data_dir
        ):
            pub = load_probe_spectrum(self.molecule_id, "HNMR", self.data_dir)
            if pub and pub.hnmr_peaks_raw:
                self._hnmr_peaks = _parse_hnmr_spectrum_rich(
                    "{Spectra: " + pub.hnmr_peaks_raw + "}"
                )
                if not self._formula:
                    self._formula = pub.molecular_formula
                if not self._solvent:
                    self._solvent = pub.solvent
                if not self._mhz:
                    self._mhz = pub.mhz
                loaded_hnmr_via_split = True
                if not self._gt_smiles:
                    hidden = load_probe_hidden_smiles(self.molecule_id, self.data_dir)
                    if hidden:
                        self._gt_smiles = hidden

        if not loaded_hnmr_via_split and self.spectrum_type in ("HNMR", "COMBO") and has_split_data(
            self.molecule_id, "HNMR", self.data_dir
        ):
            pub = load_public_task(self.molecule_id, "HNMR", self.data_dir)
            priv = load_private_grader(self.molecule_id, "HNMR", self.data_dir)
            if pub and pub.hnmr_peaks_raw:
                # Synthesize a minimal label so the rich parser path is reused.
                self._hnmr_peaks = _parse_hnmr_spectrum_rich(
                    "{Spectra: " + pub.hnmr_peaks_raw + "}"
                )
                if not self._formula:
                    self._formula = pub.molecular_formula
                if not self._solvent:
                    self._solvent = pub.solvent
                if not self._mhz:
                    self._mhz = pub.mhz
                loaded_hnmr_via_split = True
            if priv and not self._gt_smiles:
                self._gt_smiles = priv.gt_smiles

        hnmr_path = self.data_dir / "H_NMR" / f"{self.molecule_id}_HNMR_HRE.json"
        if not loaded_hnmr_via_split and hnmr_path.exists():
            with open(hnmr_path) as f:
                self._hnmr_data = json.load(f)
            nodes = self._hnmr_data.get("Nodes", self._hnmr_data.get("nodes", []))
            for node in nodes:
                if node.get("id") == "BIG QUESTION":
                    # Use rich parser that returns HNMRPeak objects
                    self._hnmr_peaks = _parse_hnmr_spectrum_rich(node.get("label", ""))
                    if not self._formula:
                        _, self._solvent, self._mhz, self._formula = \
                            _parse_spectrum_from_big_question(node.get("label", ""))
                    if not self._gt_smiles:
                        self._gt_smiles = node.get("answer", "")
                    break

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def gt_smiles(self) -> str:
        return self._gt_smiles

    @property
    def cnmr_peaks(self) -> List[float]:
        return self._cnmr_peaks

    @property
    def hnmr_peaks(self) -> list:
        """Return HNMR peaks (HNMRPeak objects or legacy dicts)."""
        return self._hnmr_peaks

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return result. Log the call."""
        if tool_name not in TOOL_DEFINITIONS:
            result = {"error": f"Unknown tool: {tool_name}"}
            self.call_log.append({
                "tool": tool_name,
                "args": args,
                "result": result,
                "cost": 0,
                "category": "unknown",
            })
            return result

        defn = TOOL_DEFINITIONS[tool_name]

        try:
            handler = getattr(self, f"_exec_{tool_name}", None)
            if handler is None:
                result = {"error": f"Tool '{tool_name}' not implemented"}
            else:
                result = handler(args)
        except Exception as e:
            result = {"error": f"Tool execution failed: {str(e)}"}

        self.call_log.append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "cost": defn["cost"],
            "category": defn["category"],
        })
        return result

    def get_total_cost(self) -> int:
        return sum(c["cost"] for c in self.call_log)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        return self.call_log.copy()

    def _active_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Return tool definitions filtered by current spectrum_type and info_mode.

        HNMR-specific tools (get_multiplicity, get_integration, get_coupling)
        are excluded in CNMR-only mode. In partial info_mode, get_full_spectrum
        is hidden so the agent must use query_spectrum with specific ppm ranges.
        """
        has_hnmr = self.spectrum_type in ("HNMR", "COMBO")
        is_partial = self.info_mode == "partial"
        return {
            name: defn
            for name, defn in TOOL_DEFINITIONS.items()
            if (has_hnmr or not defn.get("requires_hnmr", False))
            and not (is_partial and name == "get_full_spectrum")
        }

    def get_available_tools(self) -> List[str]:
        """Return names of tools currently exposed to the agent (post mode-filter).

        Used by the agent prompt to advertise only tools that can actually be
        called — e.g. get_full_spectrum is hidden in partial info_mode.
        """
        return list(self._active_tool_definitions().keys())

    def get_tool_definitions_for_llm(self) -> List[Dict[str, Any]]:
        """Return tool definitions in OpenAI function calling format."""
        tools = []
        for name, defn in self._active_tool_definitions().items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": defn["description"],
                    "parameters": defn["parameters"],
                },
            })
        return tools

    def get_tool_definitions_for_anthropic(self) -> List[Dict[str, Any]]:
        """Return tool definitions in Anthropic tool_use format."""
        tools = []
        for name, defn in self._active_tool_definitions().items():
            tools.append({
                "name": name,
                "description": defn["description"],
                "input_schema": defn["parameters"],
            })
        return tools

    def get_initial_observation(self, info_mode: str = "full") -> str:
        """Build the initial observation text the agent sees.

        Args:
            info_mode: "full" — formula + spectrum in prompt (default).
                       "partial" — formula only; agent must query spectrum via tools.
        """
        lines = []
        lines.append(f"Molecular Formula: {self._formula}")

        if info_mode == "partial":
            lines.append(
                "\nSpectrum data is NOT provided upfront. "
                "Use query_spectrum or get_full_spectrum to retrieve spectral data."
            )
            lines.append(f"\nYour task: Determine the molecular structure of this unknown compound")
            lines.append(f"and submit your answer as a SMILES string using the submit tool.")
            return "\n".join(lines)

        if self._cnmr_peaks:
            peaks_str = ", ".join(f"{p:.2f}" for p in sorted(self._cnmr_peaks, reverse=True))
            lines.append(f"\n13C NMR Spectrum (δ ppm): {peaks_str}")
            lines.append(f"Number of 13C peaks: {len(self._cnmr_peaks)}")

        if self._hnmr_peaks:
            hnmr_strs = []
            for p in self._hnmr_peaks:
                ppm = _hnmr_peak_ppm(p)
                d = _hnmr_peak_to_dict(p)
                s = f"{ppm:.2f}"
                # Build info string from rich data
                info_parts = []
                if d.get("multiplicity"):
                    info_parts.append(d["multiplicity"])
                if d.get("integration"):
                    info_parts.append(f"{d['integration']}H")
                if d.get("j_values"):
                    j_str = ", ".join(f"{j:.1f}" for j in d["j_values"])
                    info_parts.append(f"J={j_str} Hz")
                if info_parts:
                    s += f" ({', '.join(info_parts)})"
                elif d.get("raw_text"):
                    s += f" ({d['raw_text']})"
                hnmr_strs.append(s)
            lines.append(f"\n1H NMR Spectrum (δ ppm): {', '.join(hnmr_strs)}")
            lines.append(f"Number of 1H peak groups: {len(self._hnmr_peaks)}")

        if self._solvent:
            lines.append(f"\nSolvent: {self._solvent}")
        if self._mhz:
            lines.append(f"Spectrometer Frequency: {self._mhz}")

        lines.append(f"\nYour task: Determine the molecular structure of this unknown compound")
        lines.append(f"and submit your answer as a SMILES string using the submit tool.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _exec_query_spectrum(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve NMR peaks within a specified ppm range."""
        nucleus = args.get("nucleus", "13C")
        ppm_min = float(args.get("ppm_min", 0))
        ppm_max = float(args.get("ppm_max", 250))

        if nucleus == "13C":
            filtered = [p for p in self._cnmr_peaks if ppm_min <= p <= ppm_max]
            return {
                "nucleus": "13C",
                "range": [ppm_min, ppm_max],
                "peaks": sorted(filtered, reverse=True),
                "count": len(filtered),
            }
        elif nucleus == "1H":
            filtered = [p for p in self._hnmr_peaks
                        if _hnmr_peak_in_range(p, ppm_min, ppm_max)]
            return {
                "nucleus": "1H",
                "range": [ppm_min, ppm_max],
                "peaks": [_hnmr_peak_to_dict(p) for p in filtered],
                "count": len(filtered),
            }
        else:
            return {"error": f"Unknown nucleus: {nucleus}"}

    def _exec_get_full_spectrum(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve the complete NMR spectrum."""
        nucleus = args.get("nucleus", "13C")

        if nucleus == "13C":
            return {
                "nucleus": "13C",
                "peaks": sorted(self._cnmr_peaks, reverse=True),
                "count": len(self._cnmr_peaks),
            }
        elif nucleus == "1H":
            return {
                "nucleus": "1H",
                "peaks": [_hnmr_peak_to_dict(p) for p in self._hnmr_peaks],
                "count": len(self._hnmr_peaks),
            }
        else:
            return {"error": f"Unknown nucleus: {nucleus}"}

    def _exec_compute_unsaturation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate degree of unsaturation from molecular formula."""
        formula = args.get("formula", "")
        counts = parse_formula_counts(formula)

        if not counts:
            return {"error": f"Could not parse formula: {formula}"}

        dbe = calculate_unsaturation(formula)
        c = counts.get("C", 0)
        h = counts.get("H", 0)
        n = counts.get("N", 0)
        x = sum(counts.get(e, 0) for e in ("F", "Cl", "Br", "I"))

        return {
            "formula": formula,
            "degree": dbe,
            "explanation": f"DBE = (2*{c} + 2 + {n} - {h} - {x}) / 2 = {dbe}",
        }

    def _exec_compute_molecular_weight(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate molecular weight from formula."""
        formula = args.get("formula", "")
        counts = _parse_formula(formula)

        if not counts:
            return {"error": f"Could not parse formula: {formula}"}

        mw = 0.0
        breakdown = []
        for elem, count in counts.items():
            w = _ATOMIC_WEIGHTS.get(elem, 0.0)
            if w == 0.0:
                return {"error": f"Unknown element: {elem}"}
            mw += w * count
            breakdown.append(f"{elem}{count}={w*count:.3f}")

        return {
            "formula": formula,
            "molecular_weight": round(mw, 3),
            "breakdown": breakdown,
        }

    def _exec_substructure_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Check if SMILES contains a SMARTS substructure pattern."""
        try:
            from rdkit import Chem
        except ImportError:
            return {"error": "RDKit not available"}

        smiles = args.get("smiles", "")
        pattern = args.get("pattern", "")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        # Try as SMARTS first, then SMILES
        pat_mol = Chem.MolFromSmarts(pattern)
        if pat_mol is None:
            pat_mol = Chem.MolFromSmiles(pattern)
        if pat_mol is None:
            return {"error": f"Invalid pattern: {pattern}"}

        has_match = mol.HasSubstructMatch(pat_mol)
        matches = mol.GetSubstructMatches(pat_mol)

        return {
            "smiles": smiles,
            "pattern": pattern,
            "has_match": has_match,
            "num_matches": len(matches),
        }

    def _exec_count_carbons(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Count carbon atoms in a SMILES structure."""
        try:
            from rdkit import Chem
        except ImportError:
            return {"error": "RDKit not available"}

        smiles = args.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        total_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        aromatic_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetIsAromatic())

        return {
            "smiles": smiles,
            "total_carbons": total_c,
            "aromatic_carbons": aromatic_c,
            "aliphatic_carbons": total_c - aromatic_c,
        }

    def _exec_check_symmetry(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze molecular symmetry — count unique carbon environments."""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolops
        except ImportError:
            return {"error": "RDKit not available"}

        smiles = args.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        # Use canonical ranking to identify equivalent atoms
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

        # Count unique carbon environments
        carbon_ranks = set()
        total_carbons = 0
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 6:
                carbon_ranks.add(ranks[i])
                total_carbons += 1

        unique_c_envs = len(carbon_ranks)
        has_symmetry = unique_c_envs < total_carbons

        return {
            "smiles": smiles,
            "total_carbons": total_carbons,
            "unique_carbon_environments": unique_c_envs,
            "has_symmetry": has_symmetry,
            "symmetry_note": (
                f"Molecule has a plane of symmetry: {total_carbons} carbons map to "
                f"{unique_c_envs} unique environments"
                if has_symmetry
                else f"No carbon symmetry: all {total_carbons} carbons are in unique environments"
            ),
        }

    def _exec_validate_smiles(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SMILES and return canonical form + basic properties."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
        except ImportError:
            return {"error": "RDKit not available"}

        smiles = args.get("smiles", "")

        # Use shared canonicalize_smiles (handles edge cases consistently)
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return {"valid": False, "smiles": smiles, "error": "Invalid SMILES string"}

        mol = Chem.MolFromSmiles(canonical)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = Descriptors.MolWt(mol)
        num_atoms = mol.GetNumAtoms()
        num_heavy = mol.GetNumHeavyAtoms()

        return {
            "valid": True,
            "smiles": smiles,
            "canonical_smiles": canonical,
            "molecular_formula": formula,
            "molecular_weight": round(mw, 3),
            "num_atoms": num_atoms,
            "num_heavy_atoms": num_heavy,
        }

    def _exec_predict_nmr(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Predict approximate NMR shifts using functional group heuristics.

        This is a simplified prediction using known chemical shift ranges.
        For production use, a proper NMR prediction engine (nmrshiftdb2, etc.)
        would be needed.
        """
        try:
            from rdkit import Chem
        except ImportError:
            return {"error": "RDKit not available"}

        smiles = args.get("smiles", "")
        nucleus = args.get("nucleus", "13C")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        if nucleus == "13C":
            return self._predict_13c(mol, smiles)
        elif nucleus == "1H":
            return self._predict_1h(mol, smiles)
        else:
            return {"error": f"Unknown nucleus: {nucleus}"}

    def _predict_13c(self, mol, smiles: str) -> Dict[str, Any]:
        """Heuristic 13C NMR prediction based on atom environment."""
        from rdkit import Chem

        predicted: List[Dict[str, Any]] = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Only carbons
                continue

            idx = atom.GetIdx()
            is_aromatic = atom.GetIsAromatic()

            # Get neighbors
            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            has_O = 8 in neighbors
            has_N = 7 in neighbors
            has_halogen = any(n in (9, 17, 35, 53) for n in neighbors)

            # Check for double bonds
            has_double_bond = any(
                bond.GetBondTypeAsDouble() == 2.0
                for bond in atom.GetBonds()
            )

            # Heuristic shift assignment
            if is_aromatic:
                if has_O:
                    shift = 155.0  # phenol C-O
                elif has_N:
                    shift = 148.0  # aniline C-N
                else:
                    # Generic aromatic: vary by position
                    shift = 128.0 + (idx % 5) * 3.0
                env_type = "aromatic"
            elif has_double_bond and has_O and not is_aromatic:
                # Carbonyl
                shift = 195.0
                env_type = "carbonyl"
            elif has_double_bond and not is_aromatic:
                shift = 130.0
                env_type = "alkene"
            elif has_O:
                shift = 65.0
                env_type = "C-O alkyl"
            elif has_N:
                shift = 45.0
                env_type = "C-N alkyl"
            elif has_halogen:
                shift = 40.0
                env_type = "C-X alkyl"
            else:
                # Saturated carbon
                n_c_neighbors = sum(1 for n in neighbors if n == 6)
                shift = 15.0 + n_c_neighbors * 8.0
                env_type = "alkyl"

            predicted.append({
                "atom_idx": idx,
                "predicted_shift": round(shift, 1),
                "environment": env_type,
            })

        shifts = sorted([p["predicted_shift"] for p in predicted], reverse=True)

        return {
            "smiles": smiles,
            "nucleus": "13C",
            "predicted_shifts": shifts,
            "num_carbons": len(predicted),
            "details": predicted,
            "note": "Heuristic prediction — approximate only. Use compare_spectra for quantitative validation.",
        }

    def _predict_1h(self, mol, smiles: str) -> Dict[str, Any]:
        """Simplified 1H NMR prediction."""
        from rdkit import Chem

        # RDKit drops implicit hydrogens by default — make them explicit so
        # the per-atom loop below actually finds the H atoms to predict.
        mol = Chem.AddHs(mol)

        # Very simplified: group by carbon type
        predicted_groups: List[Dict[str, Any]] = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                continue

            # Get the atom this H is attached to
            neighbors = atom.GetNeighbors()
            if not neighbors:
                continue
            parent = neighbors[0]

            if parent.GetIsAromatic():
                shift = 7.2
                env = "aromatic H"
            elif parent.GetAtomicNum() == 8:
                shift = 4.5  # O-H
                env = "O-H"
            elif parent.GetAtomicNum() == 7:
                shift = 3.5  # N-H
                env = "N-H"
            else:
                # Carbon-attached H
                p_neighbors = [n.GetAtomicNum() for n in parent.GetNeighbors()]
                if 8 in p_neighbors:
                    shift = 3.8
                    env = "H-C-O"
                elif 7 in p_neighbors:
                    shift = 2.8
                    env = "H-C-N"
                elif parent.GetIsAromatic():
                    shift = 7.2
                    env = "aromatic H"
                else:
                    shift = 1.5
                    env = "alkyl H"

            predicted_groups.append({"shift": round(shift, 1), "environment": env})

        shifts = sorted(set(p["shift"] for p in predicted_groups), reverse=True)

        return {
            "smiles": smiles,
            "nucleus": "1H",
            "predicted_shifts": shifts,
            "num_groups": len(predicted_groups),
            "note": "Heuristic prediction — approximate only.",
        }

    def _exec_compare_spectra(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compare predicted vs observed spectra."""
        predicted = sorted(args.get("predicted_shifts", []))
        observed = sorted(args.get("observed_shifts", []))
        nucleus = args.get("nucleus", "13C")

        if not predicted or not observed:
            return {
                "error": "Both predicted_shifts and observed_shifts must be non-empty",
                "predicted_count": len(predicted),
                "observed_count": len(observed),
            }

        # Tolerance depends on nucleus
        tol = 5.0 if nucleus == "13C" else 0.3

        # Greedy matching: for each observed peak, find closest predicted
        matched = []
        unmatched_obs = []
        used_pred = set()

        for obs in observed:
            best_idx = -1
            best_dist = float("inf")
            for j, pred in enumerate(predicted):
                if j in used_pred:
                    continue
                d = abs(obs - pred)
                if d < best_dist:
                    best_dist = d
                    best_idx = j

            if best_idx >= 0 and best_dist <= tol:
                matched.append({
                    "observed": obs,
                    "predicted": predicted[best_idx],
                    "error": round(best_dist, 2),
                })
                used_pred.add(best_idx)
            else:
                unmatched_obs.append(obs)

        unmatched_pred = [predicted[j] for j in range(len(predicted)) if j not in used_pred]

        # Compute MAE over matched peaks
        if matched:
            mae = sum(m["error"] for m in matched) / len(matched)
        else:
            mae = float("inf")

        # Similarity score
        total = max(len(predicted), len(observed))
        similarity = len(matched) / total if total > 0 else 0.0

        return {
            "nucleus": nucleus,
            "similarity": round(similarity, 3),
            "mae": round(mae, 2) if mae != float("inf") else None,
            "matched_peaks": len(matched),
            "total_observed": len(observed),
            "total_predicted": len(predicted),
            "matches": matched[:10],  # Limit output
            "unmatched_observed": unmatched_obs[:10],
            "unmatched_predicted": unmatched_pred[:10],
        }

    # ------------------------------------------------------------------
    # HNMR-specific tool implementations
    # ------------------------------------------------------------------

    def _exec_get_multiplicity(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get multiplicity data for 1H NMR peaks in a ppm range."""
        if not self._hnmr_peaks:
            return {"error": "No 1H NMR data available for this molecule"}

        ppm_min = float(args.get("ppm_min", 0))
        ppm_max = float(args.get("ppm_max", 15))

        results = []
        for p in self._hnmr_peaks:
            if _hnmr_peak_in_range(p, ppm_min, ppm_max):
                d = _hnmr_peak_to_dict(p)
                results.append({
                    "ppm": d["ppm"],
                    "multiplicity": d.get("multiplicity", "unknown"),
                })

        return {
            "range": [ppm_min, ppm_max],
            "peaks": results,
            "count": len(results),
        }

    def _exec_get_integration(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get integration (proton count) for 1H NMR peaks in a ppm range."""
        if not self._hnmr_peaks:
            return {"error": "No 1H NMR data available for this molecule"}

        ppm_min = float(args.get("ppm_min", 0))
        ppm_max = float(args.get("ppm_max", 15))

        results = []
        total_integration = 0
        for p in self._hnmr_peaks:
            if _hnmr_peak_in_range(p, ppm_min, ppm_max):
                d = _hnmr_peak_to_dict(p)
                integ = d.get("integration", None)
                results.append({
                    "ppm": d["ppm"],
                    "integration": integ,
                })
                if integ is not None:
                    total_integration += integ

        return {
            "range": [ppm_min, ppm_max],
            "peaks": results,
            "count": len(results),
            "total_integration": total_integration,
        }

    def _exec_get_coupling(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get J-coupling constants for 1H NMR peaks in a ppm range."""
        if not self._hnmr_peaks:
            return {"error": "No 1H NMR data available for this molecule"}

        ppm_min = float(args.get("ppm_min", 0))
        ppm_max = float(args.get("ppm_max", 15))

        results = []
        for p in self._hnmr_peaks:
            if _hnmr_peak_in_range(p, ppm_min, ppm_max):
                d = _hnmr_peak_to_dict(p)
                results.append({
                    "ppm": d["ppm"],
                    "j_values": d.get("j_values", []),
                    "multiplicity": d.get("multiplicity", "unknown"),
                })

        return {
            "range": [ppm_min, ppm_max],
            "peaks": results,
            "count": len(results),
        }

    def _exec_submit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle submission — just returns the SMILES. Episode termination
        is handled by AgentLoop."""
        smiles = args.get("smiles", "")

        # Validate using shared canonicalize_smiles
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return {"submitted": smiles, "valid": False, "error": "Invalid SMILES"}
        return {"submitted": smiles, "canonical": canonical, "valid": True}
