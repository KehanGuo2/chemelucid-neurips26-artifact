"""HNMR peak parsing utilities.

Extracted from scripts/hnmr_hre_autofill_gt.py to provide rich HNMR peak
parsing (multiplicity, integration, J-coupling) for the interactive gym.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HNMRPeak:
    """Represents a single HNMR peak with all its properties."""

    ppm: float  # Single value or midpoint of range
    ppm_range: Optional[Tuple[float, float]] = None  # (low, high) if range
    integration: int = 1
    multiplicity: str = "m"
    j_values: List[float] = field(default_factory=list)
    raw_text: str = ""

    def format_ppm(self) -> str:
        """Format ppm value for display."""
        if self.ppm_range:
            return f"{self.ppm_range[0]:.2f}-{self.ppm_range[1]:.2f}"
        return f"{self.ppm:.2f}"

    def ppm_in_range(self, lo: float, hi: float) -> bool:
        """Check if peak ppm falls within [lo, hi]."""
        if self.ppm_range:
            return self.ppm_range[0] <= hi and self.ppm_range[1] >= lo
        return lo <= self.ppm <= hi

    def is_singlet(self) -> bool:
        """Check if peak is a singlet."""
        return self.multiplicity.lower() == "s"


def parse_hnmr_peaks(spectra: str) -> List[HNMRPeak]:
    """Parse HNMR spectra string into list of HNMRPeak objects.

    Handles formats like:
        - "7.87 – 7.74 (m, 2H)"
        - "7.44 (s, 1H)"
        - "5.06 (tdq, J = 7.3, 7.3, 2.9, 1.5, 1.5, 1.5 Hz, 1H)"
    """
    peaks: List[HNMRPeak] = []

    # Clean up the spectra string
    spectra = spectra.replace("\u03b4", "").strip()  # remove δ

    # Pattern for peaks: ppm1 [– ppm2] (parenthetical info)
    peak_pattern = r"(\d+\.?\d*)\s*(?:[–\-]\s*(\d+\.?\d*))?\s*\(([^)]+)\)"

    for match in re.finditer(peak_pattern, spectra):
        ppm1 = float(match.group(1))
        ppm2 = float(match.group(2)) if match.group(2) else ppm1

        paren_content = match.group(3)

        # Extract integration (e.g., "2H", "1H")
        # Use negative lookahead to avoid matching "4Hz" as "4H"
        int_match = re.search(r"(\d+)H(?![zZ])", paren_content)
        integration = int(int_match.group(1)) if int_match else 1

        # Extract multiplicity (first token)
        mult_match = re.match(r"([a-zA-Z]+)", paren_content.strip())
        multiplicity = mult_match.group(1) if mult_match else "m"

        # Extract J values
        j_values: List[float] = []
        j_match = re.search(r"J\s*=\s*([\d.,\s]+)\s*Hz", paren_content)
        if j_match:
            j_str = j_match.group(1)
            j_values = [
                float(x.strip())
                for x in re.split(r"[,\s]+", j_str)
                if x.strip()
            ]

        peak = HNMRPeak(
            ppm=(ppm1 + ppm2) / 2 if ppm1 != ppm2 else ppm1,
            ppm_range=(min(ppm1, ppm2), max(ppm1, ppm2)) if ppm1 != ppm2 else None,
            integration=integration,
            multiplicity=multiplicity,
            j_values=j_values,
            raw_text=match.group(0).strip(),
        )
        peaks.append(peak)

    return peaks
