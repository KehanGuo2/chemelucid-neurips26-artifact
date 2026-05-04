"""Tests for ToolServer — using 2,4-dimethylaniline (C8H11N) as default molecule.

Ground truth:
- SMILES: Cc1ccc(N)c(C)c1
- Formula: C8H11N
- DoU: 4
- 13C peaks: δ 142.20, 131.28, 127.88, 127.49, 122.57, 115.23, 20.60, 17.48
- 8 carbon peaks (8 unique environments → no symmetry)
- Aromatic carbons: 6 (110-160 ppm)
- Alkyl carbons: 2 (0-50 ppm)
- No carbonyl, no halogen
"""

import os
import pytest
from pathlib import Path

from interactive.tool_server import ToolServer, _parse_formula, TOOL_DEFINITIONS
from interactive.config import DATA_DIR as _DATA_DIR

DATA_DIR = str(_DATA_DIR)
MOL_ID = "2_4_dimethyl_aniline"


@pytest.fixture
def server():
    """Create a ToolServer for 2,4-dimethylaniline."""
    return ToolServer(molecule_id=MOL_ID, data_dir=DATA_DIR, spectrum_type="CNMR")


class TestDataLoading:
    def test_loads_formula(self, server):
        assert "C8H11N" in server.formula.replace(" ", "")

    def test_loads_cnmr_peaks(self, server):
        assert len(server.cnmr_peaks) == 8
        # Check known peaks exist
        peaks = sorted(server.cnmr_peaks)
        assert any(abs(p - 142.20) < 0.1 for p in peaks), "Missing 142.20 ppm peak"
        assert any(abs(p - 17.48) < 0.1 for p in peaks), "Missing 17.48 ppm peak"

    def test_loads_gt_smiles(self, server):
        assert server.gt_smiles  # Non-empty
        # Should be some variant of dimethylaniline
        assert "c" in server.gt_smiles.lower() or "C" in server.gt_smiles

    def test_initial_observation(self, server):
        obs = server.get_initial_observation()
        assert "C8H11N" in obs.replace(" ", "")
        assert "142.20" in obs
        assert "SMILES" in obs


class TestFormulaParser:
    def test_simple(self):
        assert _parse_formula("C8H11N") == {"C": 8, "H": 11, "N": 1}

    def test_with_oxygen(self):
        assert _parse_formula("C9H11NO2") == {"C": 9, "H": 11, "N": 1, "O": 2}

    def test_with_halogen(self):
        counts = _parse_formula("C7H5ClO3")
        assert counts["Cl"] == 1

    def test_empty(self):
        assert _parse_formula("") == {}


class TestComputeUnsaturation:
    def test_dimethylaniline(self, server):
        result = server.execute("compute_unsaturation", {"formula": "C8H11N"})
        assert result["degree"] == 4

    def test_benzene(self, server):
        result = server.execute("compute_unsaturation", {"formula": "C6H6"})
        assert result["degree"] == 4

    def test_ethanol(self, server):
        result = server.execute("compute_unsaturation", {"formula": "C2H6O"})
        assert result["degree"] == 0

    def test_acetaminophen(self, server):
        result = server.execute("compute_unsaturation", {"formula": "C8H9NO2"})
        assert result["degree"] == 5

    def test_with_halogen(self, server):
        result = server.execute("compute_unsaturation", {"formula": "CHCl3"})
        assert result["degree"] == 0  # Chloroform: (2*1 + 2 + 0 - 1 - 3)/2 = 0

    def test_cost_logged(self, server):
        server.execute("compute_unsaturation", {"formula": "C8H11N"})
        assert server.get_total_cost() == 1
        assert len(server.get_trajectory()) == 1


class TestQuerySpectrum:
    def test_aromatic_region(self, server):
        result = server.execute("query_spectrum", {
            "nucleus": "13C", "ppm_min": 110, "ppm_max": 160
        })
        assert result["count"] == 6  # 6 aromatic carbons
        assert all(110 <= p <= 160 for p in result["peaks"])

    def test_alkyl_region(self, server):
        result = server.execute("query_spectrum", {
            "nucleus": "13C", "ppm_min": 0, "ppm_max": 50
        })
        assert result["count"] == 2  # 2 methyl carbons
        assert all(0 <= p <= 50 for p in result["peaks"])

    def test_carbonyl_region_empty(self, server):
        result = server.execute("query_spectrum", {
            "nucleus": "13C", "ppm_min": 160, "ppm_max": 220
        })
        assert result["count"] == 0

    def test_full_range(self, server):
        result = server.execute("query_spectrum", {
            "nucleus": "13C", "ppm_min": 0, "ppm_max": 250
        })
        assert result["count"] == 8


class TestGetFullSpectrum:
    def test_cnmr(self, server):
        result = server.execute("get_full_spectrum", {"nucleus": "13C"})
        assert result["count"] == 8
        assert len(result["peaks"]) == 8


class TestSubstructureCheck:
    def test_aromatic_ring(self, server):
        result = server.execute("substructure_check", {
            "smiles": "Cc1ccc(N)c(C)c1",
            "pattern": "c1ccccc1",
        })
        assert result["has_match"] is True

    def test_no_carbonyl(self, server):
        result = server.execute("substructure_check", {
            "smiles": "Cc1ccc(N)c(C)c1",
            "pattern": "[CX3]=O",
        })
        assert result["has_match"] is False

    def test_amine(self, server):
        result = server.execute("substructure_check", {
            "smiles": "Cc1ccc(N)c(C)c1",
            "pattern": "[NX3;H2]",
        })
        assert result["has_match"] is True

    def test_invalid_smiles(self, server):
        result = server.execute("substructure_check", {
            "smiles": "INVALID_SMILES",
            "pattern": "c1ccccc1",
        })
        assert "error" in result


class TestCountCarbons:
    def test_dimethylaniline(self, server):
        result = server.execute("count_carbons", {"smiles": "Cc1ccc(N)c(C)c1"})
        assert result["total_carbons"] == 8
        assert result["aromatic_carbons"] == 6
        assert result["aliphatic_carbons"] == 2


class TestCheckSymmetry:
    def test_dimethylaniline(self, server):
        result = server.execute("check_symmetry", {"smiles": "Cc1ccc(N)c(C)c1"})
        assert result["total_carbons"] == 8
        # 2,4-dimethylaniline has 8 unique C environments (asymmetric)
        assert result["unique_carbon_environments"] == 8
        assert result["has_symmetry"] is False

    def test_para_xylene(self, server):
        # p-xylene has symmetry: 4 unique C environments
        result = server.execute("check_symmetry", {"smiles": "Cc1ccc(C)cc1"})
        assert result["total_carbons"] == 8
        assert result["unique_carbon_environments"] < 8
        assert result["has_symmetry"] is True


class TestValidateSmiles:
    def test_valid(self, server):
        result = server.execute("validate_smiles", {"smiles": "Cc1ccc(N)c(C)c1"})
        assert result["valid"] is True
        assert "C8H11N" in result["molecular_formula"]

    def test_invalid(self, server):
        result = server.execute("validate_smiles", {"smiles": "INVALID"})
        assert result["valid"] is False


class TestPredictNMR:
    def test_predict_13c(self, server):
        result = server.execute("predict_nmr", {
            "smiles": "Cc1ccc(N)c(C)c1",
            "nucleus": "13C",
        })
        assert result["num_carbons"] == 8
        assert len(result["predicted_shifts"]) > 0

    def test_predict_invalid(self, server):
        result = server.execute("predict_nmr", {
            "smiles": "INVALID",
            "nucleus": "13C",
        })
        assert "error" in result

    def test_predict_1h_toluene_returns_shifts(self, server):
        # Regression for Bug 2: implicit H atoms must be added before
        # iterating, otherwise predict_nmr(nucleus="1H") returns []
        result = server.execute("predict_nmr", {
            "smiles": "Cc1ccccc1",  # toluene
            "nucleus": "1H",
        })
        assert "error" not in result, result
        assert result["nucleus"] == "1H"
        # Toluene has aromatic Hs (~7.2) and methyl Hs (~1.5) -> >= 2
        # unique shifts; the underlying loop should find at least 3 atoms.
        assert len(result["predicted_shifts"]) >= 2
        assert result["num_groups"] >= 3


class TestCompareSpectra:
    def test_perfect_match(self, server):
        peaks = [142.20, 131.28, 127.88, 127.49, 122.57, 115.23, 20.60, 17.48]
        result = server.execute("compare_spectra", {
            "predicted_shifts": peaks,
            "observed_shifts": peaks,
            "nucleus": "13C",
        })
        assert result["similarity"] == 1.0
        assert result["matched_peaks"] == 8

    def test_partial_match(self, server):
        observed = [142.20, 131.28, 127.88, 127.49, 122.57, 115.23, 20.60, 17.48]
        predicted = [140.0, 130.0, 128.0, 127.0, 122.0, 115.0, 21.0, 18.0]
        result = server.execute("compare_spectra", {
            "predicted_shifts": predicted,
            "observed_shifts": observed,
            "nucleus": "13C",
        })
        assert result["similarity"] > 0.5  # Should be mostly matched within 5 ppm

    def test_empty(self, server):
        result = server.execute("compare_spectra", {
            "predicted_shifts": [],
            "observed_shifts": [],
            "nucleus": "13C",
        })
        assert "error" in result


class TestSubmit:
    def test_submit_valid(self, server):
        result = server.execute("submit", {"smiles": "Cc1ccc(N)c(C)c1"})
        assert result["valid"] is True
        assert result["submitted"] == "Cc1ccc(N)c(C)c1"

    def test_submit_invalid(self, server):
        result = server.execute("submit", {"smiles": "INVALID"})
        assert result["valid"] is False


class TestCostTracking:
    def test_cumulative_cost(self, server):
        server.execute("compute_unsaturation", {"formula": "C8H11N"})  # cost 1
        server.execute("query_spectrum", {"nucleus": "13C", "ppm_min": 0, "ppm_max": 250})  # cost 1
        server.execute("predict_nmr", {"smiles": "Cc1ccc(N)c(C)c1", "nucleus": "13C"})  # cost 3
        assert server.get_total_cost() == 5

    def test_trajectory(self, server):
        server.execute("compute_unsaturation", {"formula": "C8H11N"})
        server.execute("submit", {"smiles": "Cc1ccc(N)c(C)c1"})
        traj = server.get_trajectory()
        assert len(traj) == 2
        assert traj[0]["tool"] == "compute_unsaturation"
        assert traj[1]["tool"] == "submit"
        assert traj[1]["cost"] == 0  # submit is free


class TestToolDefinitions:
    def test_openai_format(self, server):
        tools = server.get_tool_definitions_for_llm()
        assert len(tools) > 0
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_anthropic_format(self, server):
        tools = server.get_tool_definitions_for_anthropic()
        assert len(tools) > 0
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


class TestAvailableTools:
    """Regression for Bug 3: prompt-advertised tools must match exposed schema."""

    def test_full_mode_includes_get_full_spectrum(self):
        srv = ToolServer(molecule_id=MOL_ID, data_dir=DATA_DIR,
                         spectrum_type="CNMR", info_mode="full")
        names = srv.get_available_tools()
        assert "get_full_spectrum" in names
        assert "query_spectrum" in names

    def test_partial_mode_hides_get_full_spectrum(self):
        srv = ToolServer(molecule_id=MOL_ID, data_dir=DATA_DIR,
                         spectrum_type="CNMR", info_mode="partial")
        names = srv.get_available_tools()
        assert "get_full_spectrum" not in names
        assert "query_spectrum" in names

    def test_available_tools_match_llm_schema(self):
        srv = ToolServer(molecule_id=MOL_ID, data_dir=DATA_DIR,
                         spectrum_type="CNMR", info_mode="partial")
        names = set(srv.get_available_tools())
        schema_names = {t["function"]["name"] for t in srv.get_tool_definitions_for_llm()}
        assert names == schema_names

    def test_unknown_tool(self, server):
        result = server.execute("nonexistent_tool", {})
        assert "error" in result


class TestComputeMolecularWeight:
    def test_dimethylaniline(self, server):
        result = server.execute("compute_molecular_weight", {"formula": "C8H11N"})
        # C8H11N = 8*12.011 + 11*1.008 + 14.007 = 96.088 + 11.088 + 14.007 = 121.183
        assert abs(result["molecular_weight"] - 121.183) < 0.01

    def test_water(self, server):
        result = server.execute("compute_molecular_weight", {"formula": "H2O"})
        assert abs(result["molecular_weight"] - 18.015) < 0.01


# ======================================================================
# HNMR-specific tools (require COMBO server with HNMR data)
# ======================================================================

@pytest.fixture
def combo_server():
    """Create a ToolServer for 2,4-dimethylaniline with COMBO (CNMR+HNMR)."""
    return ToolServer(molecule_id=MOL_ID, data_dir=DATA_DIR, spectrum_type="COMBO")


class TestHNMRRichParsing:
    """Verify the rich HNMRPeak parser extracts multiplicity, integration, J-values."""

    def test_loads_hnmr_peaks(self, combo_server):
        assert len(combo_server.hnmr_peaks) == 6

    def test_peak_has_multiplicity(self, combo_server):
        """All peaks should have multiplicity attribute."""
        for p in combo_server.hnmr_peaks:
            assert hasattr(p, "multiplicity") or isinstance(p, dict)

    def test_initial_observation_hnmr(self, combo_server):
        obs = combo_server.get_initial_observation()
        assert "1H NMR" in obs
        # Should contain multiplicity info like (s, 1H) or (d, 1H, J=...)
        assert "s," in obs or "(s" in obs


class TestQuerySpectrum1H:
    def test_aromatic_region(self, combo_server):
        result = combo_server.execute("query_spectrum", {
            "nucleus": "1H", "ppm_min": 6.0, "ppm_max": 8.0,
        })
        assert result["count"] == 3  # 3 aromatic H peaks
        # Each peak should have rich data
        for peak in result["peaks"]:
            assert "ppm" in peak
            assert "multiplicity" in peak

    def test_alkyl_region(self, combo_server):
        result = combo_server.execute("query_spectrum", {
            "nucleus": "1H", "ppm_min": 0.0, "ppm_max": 4.0,
        })
        assert result["count"] == 3  # 2 methyl + NH2

    def test_full_range(self, combo_server):
        result = combo_server.execute("query_spectrum", {
            "nucleus": "1H", "ppm_min": 0.0, "ppm_max": 15.0,
        })
        assert result["count"] == 6


class TestGetFullSpectrum1H:
    def test_returns_all_peaks(self, combo_server):
        result = combo_server.execute("get_full_spectrum", {"nucleus": "1H"})
        assert result["count"] == 6
        assert len(result["peaks"]) == 6

    def test_rich_peak_data(self, combo_server):
        result = combo_server.execute("get_full_spectrum", {"nucleus": "1H"})
        for peak in result["peaks"]:
            assert "ppm" in peak
            assert "multiplicity" in peak
            assert "integration" in peak


class TestGetMultiplicity:
    def test_all_peaks(self, combo_server):
        result = combo_server.execute("get_multiplicity", {})
        assert result["count"] == 6
        multiplicities = [p["multiplicity"] for p in result["peaks"]]
        assert "s" in multiplicities  # singlets exist
        assert "d" in multiplicities  # doublets exist

    def test_aromatic_only(self, combo_server):
        result = combo_server.execute("get_multiplicity", {
            "ppm_min": 6.0, "ppm_max": 8.0,
        })
        assert result["count"] == 3


class TestGetIntegration:
    def test_all_peaks(self, combo_server):
        result = combo_server.execute("get_integration", {})
        assert result["count"] == 6
        # Total integration should be 11 (1+1+1+2+3+3)
        assert result["total_integration"] == 11

    def test_aromatic_integration(self, combo_server):
        result = combo_server.execute("get_integration", {
            "ppm_min": 6.0, "ppm_max": 8.0,
        })
        assert result["total_integration"] == 3  # 1+1+1

    def test_methyl_integration(self, combo_server):
        result = combo_server.execute("get_integration", {
            "ppm_min": 1.5, "ppm_max": 2.5,
        })
        assert result["total_integration"] == 6  # 3+3


class TestGetCoupling:
    def test_all_peaks(self, combo_server):
        result = combo_server.execute("get_coupling", {})
        assert result["count"] == 6

    def test_doublets_have_j(self, combo_server):
        result = combo_server.execute("get_coupling", {
            "ppm_min": 6.0, "ppm_max": 7.0,
        })
        # The two doublets (6.82 and 6.54) should have J-values
        peaks_with_j = [p for p in result["peaks"] if p["j_values"]]
        assert len(peaks_with_j) >= 2
        # J-values should be ~8 Hz (aromatic coupling)
        for p in peaks_with_j:
            assert any(6.0 <= j <= 10.0 for j in p["j_values"])

    def test_singlets_no_j(self, combo_server):
        result = combo_server.execute("get_coupling", {
            "ppm_min": 1.5, "ppm_max": 2.5,
        })
        # Methyl singlets should have empty J-values
        for p in result["peaks"]:
            assert p["j_values"] == []


class TestHNMRNoData:
    """Test graceful handling when no HNMR data is available."""

    @pytest.fixture
    def no_hnmr_server(self, tmp_path):
        """Create a server with no HNMR data (only CNMR)."""
        # Create minimal CNMR data dir
        cnmr_dir = tmp_path / "C_NMR"
        cnmr_dir.mkdir()
        import json, shutil
        src = Path(DATA_DIR) / "C_NMR" / f"{MOL_ID}_CNMR_HRE.json"
        shutil.copy(src, cnmr_dir / f"{MOL_ID}_CNMR_HRE.json")
        # No H_NMR directory → no HNMR data
        return ToolServer(molecule_id=MOL_ID, data_dir=str(tmp_path), spectrum_type="CNMR")

    def test_no_hnmr_multiplicity(self, no_hnmr_server):
        """Server without HNMR data should return error for HNMR tools."""
        result = no_hnmr_server.execute("get_multiplicity", {})
        assert "error" in result

    def test_no_hnmr_query(self, no_hnmr_server):
        result = no_hnmr_server.execute("query_spectrum", {
            "nucleus": "1H", "ppm_min": 0, "ppm_max": 15,
        })
        assert result["count"] == 0
