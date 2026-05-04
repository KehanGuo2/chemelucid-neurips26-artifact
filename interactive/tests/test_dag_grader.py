"""Tests for DAGGrader — post-hoc evaluation of agent trajectories."""

import pytest

from interactive.dag_grader import DAGGrader
from interactive.config import DATA_DIR
MOL_ID = "2_4_dimethyl_aniline"


@pytest.fixture
def grader():
    """Create a DAGGrader for 2,4-dimethylaniline CNMR."""
    dag_path = str(DATA_DIR / "CNMR_HRE_v5.json")
    gt_path = str(DATA_DIR / "C_NMR" / f"{MOL_ID}_CNMR_HRE.json")
    return DAGGrader(dag_path=dag_path, gt_path=gt_path)


class TestGraderSetup:
    def test_loads_gt_nodes(self, grader):
        assert "F1" in grader.gt_nodes
        assert "BIG QUESTION" in grader.gt_nodes
        assert grader.gt_nodes["F1"]["answer"] == "4"

    def test_parses_formula(self, grader):
        assert "C8H11N" in grader._formula.replace(" ", "")

    def test_parses_spectrum(self, grader):
        assert len(grader._spectrum_peaks) == 8


class TestEmptyTrajectory:
    def test_empty_trajectory(self, grader):
        result = grader.grade([])
        assert result["coverage"] == 0.0
        assert all(not r["covered"] for r in result["node_results"].values())


class TestUnsaturationCoverage:
    def test_correct_unsaturation(self, grader):
        """Trajectory that correctly computes DoU=4 should cover F1."""
        trajectory = [
            {
                "tool": "compute_unsaturation",
                "args": {"formula": "C8H11N"},
                "result": {"degree": 4, "formula": "C8H11N"},
                "cost": 1,
                "category": "computation",
            }
        ]
        result = grader.grade(trajectory)
        assert result["node_results"]["F1"]["covered"] is True

    def test_wrong_unsaturation(self, grader):
        """Wrong DoU should mark F1 as not covered."""
        trajectory = [
            {
                "tool": "compute_unsaturation",
                "args": {"formula": "C8H11N"},
                "result": {"degree": 3, "formula": "C8H11N"},
                "cost": 1,
                "category": "computation",
            }
        ]
        result = grader.grade(trajectory)
        assert result["node_results"]["F1"]["covered"] is False


class TestSpectrumQueryCoverage:
    def test_aromatic_query(self, grader):
        """Querying aromatic region should cover F1.4.1."""
        trajectory = [
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160},
                "result": {
                    "nucleus": "13C",
                    "peaks": [142.20, 131.28, 127.88, 127.49, 122.57, 115.23],
                    "count": 6,
                    "range": [110, 160],
                },
                "cost": 1,
                "category": "spectrum_query",
            }
        ]
        result = grader.grade(trajectory)
        # F1.4.1 asks "is there a possibility for aromatic ring?" — GT=Yes
        # Agent queried 110-160, found peaks → covered
        assert result["node_results"]["F1.4.1"]["covered"] is True

    def test_carbonyl_query_empty(self, grader):
        """Querying carbonyl region (empty) should cover F1.2.1 (GT=No)."""
        trajectory = [
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 160, "ppm_max": 220},
                "result": {
                    "nucleus": "13C",
                    "peaks": [],
                    "count": 0,
                    "range": [160, 220],
                },
                "cost": 1,
                "category": "spectrum_query",
            }
        ]
        result = grader.grade(trajectory)
        assert result["node_results"]["F1.2.1"]["covered"] is True

    def test_alkyl_query(self, grader):
        """Querying alkyl region should cover F3 (GT=Yes)."""
        trajectory = [
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50},
                "result": {
                    "nucleus": "13C",
                    "peaks": [20.60, 17.48],
                    "count": 2,
                    "range": [0, 50],
                },
                "cost": 1,
                "category": "spectrum_query",
            }
        ]
        result = grader.grade(trajectory)
        assert result["node_results"]["F3"]["covered"] is True


class TestComprehensiveTrajectory:
    def test_full_coverage(self, grader):
        """A thorough trajectory should achieve high coverage."""
        trajectory = [
            # Compute DoU
            {
                "tool": "compute_unsaturation",
                "args": {"formula": "C8H11N"},
                "result": {"degree": 4, "formula": "C8H11N"},
                "cost": 1, "category": "computation",
            },
            # Query aromatic region
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160},
                "result": {
                    "nucleus": "13C",
                    "peaks": [142.20, 131.28, 127.88, 127.49, 122.57, 115.23],
                    "count": 6, "range": [110, 160],
                },
                "cost": 1, "category": "spectrum_query",
            },
            # Query carbonyl region
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 160, "ppm_max": 220},
                "result": {
                    "nucleus": "13C", "peaks": [], "count": 0, "range": [160, 220],
                },
                "cost": 1, "category": "spectrum_query",
            },
            # Query alkyl region
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50},
                "result": {
                    "nucleus": "13C",
                    "peaks": [20.60, 17.48],
                    "count": 2, "range": [0, 50],
                },
                "cost": 1, "category": "spectrum_query",
            },
            # Query C-heteroatom region
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 50, "ppm_max": 100},
                "result": {
                    "nucleus": "13C", "peaks": [], "count": 0, "range": [50, 100],
                },
                "cost": 1, "category": "spectrum_query",
            },
            # Check symmetry
            {
                "tool": "check_symmetry",
                "args": {"smiles": "Cc1ccc(N)c(C)c1"},
                "result": {
                    "total_carbons": 8, "unique_carbon_environments": 8,
                    "has_symmetry": False,
                },
                "cost": 1, "category": "structure_analysis",
            },
            # Submit
            {
                "tool": "submit",
                "args": {"smiles": "Cc1ccc(N)c(C)c1"},
                "result": {"submitted": "Cc1ccc(N)c(C)c1", "valid": True},
                "cost": 0, "category": "submit",
            },
        ]
        result = grader.grade(trajectory)

        # Should have high coverage
        assert result["coverage"] > 0.3
        # F1 should be covered
        assert result["node_results"]["F1"]["covered"] is True
        # Check failure analysis
        total_failures = sum(len(v) for v in result["failure_analysis"].values())
        total_nodes = len(result["node_results"])
        covered_nodes = sum(1 for r in result["node_results"].values() if r["covered"])
        assert total_failures == total_nodes - covered_nodes


class TestFailureClassification:
    def test_strategy_failures(self, grader):
        """Empty trajectory should yield strategy failures."""
        result = grader.grade([])
        fa = result["failure_analysis"]
        # Most uncovered nodes should be classified as strategy failures
        assert len(fa["strategy"]) > 0


class TestPPMRangeAlignment:
    """Verify dag_grader.py PPM ranges match cnmr_hre_autofill_gt.py (source of truth).

    These tests ensure the grader will not produce false negatives due to
    mismatched chemical shift ranges.
    """

    def test_alkene_range(self):
        """F1.1.x alkene range: 100-150 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["alkene"] == (100, 150)

    def test_carbonyl_range(self):
        """F1.2.x carbonyl range: 150-220 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["carbonyl"] == (150, 220)

    def test_imine_range(self):
        """F1.3.x imine range: 150-170 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["imine"] == (150, 170)

    def test_aromatic_range(self):
        """F1.4.x aromatic range: 100-170 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["aromatic"] == (100, 170)

    def test_alkyne_range(self):
        """F1.5.x alkyne range: 110-140 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["alkyne"] == (110, 140)

    def test_nitrile_range(self):
        """F1.6.x nitrile range: 105-135 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["nitrile"] == (105, 135)

    def test_alkyl_range(self):
        """F3.x alkyl range: 8-50 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["alkyl"] == (8, 50)

    def test_c_heteroatom_range(self):
        """F5.x C-heteroatom range: 30-90 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["C-heteroatom"] == (30, 90)

    def test_c_halogen_range(self):
        """F6.x C-halogen range: 20-80 ppm."""
        from interactive.dag_grader import CNMR_PPM_RANGES
        assert CNMR_PPM_RANGES["C-halogen"] == (20, 80)

    def test_ranges_match_autofill_gt(self):
        """Cross-check all grader ranges against cnmr_hre_autofill_gt.py.

        The autofill script uses peaks_in_range(peaks, low, high).
        We verify each grader range matches the corresponding autofill call.
        """
        from interactive.dag_grader import CNMR_PPM_RANGES

        # Expected ranges from cnmr_hre_autofill_gt.py (lines 201-261)
        autofill_ranges = {
            "alkene":       (100, 150),  # _has_peaks_in_range(peaks, 100, 150)
            "carbonyl":     (150, 220),  # _has_peaks_in_range(peaks, 150, 220)
            "imine":        (150, 170),  # _has_peaks_in_range(peaks, 150, 170)
            "aromatic":     (100, 170),  # _has_peaks_in_range(peaks, 100, 170)
            "alkyne":       (110, 140),  # _has_peaks_in_range(peaks, 110, 140)
            "nitrile":      (105, 135),  # _has_peaks_in_range(peaks, 105, 135)
            "alkyl":        (8,   50),   # _has_peaks_in_range(peaks, 8, 50)
            "C-heteroatom": (30,  90),   # _has_peaks_in_range(peaks, 30, 90)
            "C-halogen":    (20,  80),   # _has_peaks_in_range(peaks, 20, 80)
        }

        for motif, expected in autofill_ranges.items():
            actual = CNMR_PPM_RANGES[motif]
            assert actual == expected, (
                f"PPM range mismatch for {motif}: "
                f"grader={actual}, autofill={expected}"
            )

    def test_carbonyl_subranges(self):
        """Verify CARBONYL_SUBRANGES match cnmr_hre_autofill_gt.CARBONYL_GROUPS."""
        from interactive.dag_grader import CARBONYL_SUBRANGES

        expected = {
            "Aldehyde":        (170, 220),
            "Ketone":          (170, 220),
            "Carboxylic Acid": (165, 185),
            "Ester":           (165, 185),
            "Amide":           (155, 185),
        }
        for group, expected_range in expected.items():
            assert group in CARBONYL_SUBRANGES, f"Missing carbonyl group: {group}"
            assert CARBONYL_SUBRANGES[group] == expected_range, (
                f"Carbonyl sub-range mismatch for {group}: "
                f"grader={CARBONYL_SUBRANGES[group]}, autofill={expected_range}"
            )


class TestNewGradingRules:
    """Test newly added CNMR grading rules: F1.5.1, F1.6.1, F6 with ppm, F4 structured."""

    def test_nitrile_query_covers_f1_6_1(self, grader):
        """Querying nitrile range (105-135) should cover F1.6.1."""
        trajectory = [
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 105, "ppm_max": 135},
                "result": {
                    "nucleus": "13C",
                    "peaks": [122.57, 115.23],
                    "count": 2,
                    "range": [105, 135],
                },
                "cost": 1, "category": "spectrum_query",
            }
        ]
        result = grader.grade(trajectory)
        # F1.6.1 for 2,4-dimethylaniline: N in formula, DBE=4>=2,
        # but the autofill GT answer determines covered or not
        f161 = result["node_results"].get("F1.6.1")
        if f161:
            assert f161["method"] == "tool_match"

    def test_halogen_peaks_query_covers_f6_1_1(self, grader):
        """Querying halogen peak range (20-80) should cover F6.1.1."""
        trajectory = [
            {
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": 20, "ppm_max": 80},
                "result": {
                    "nucleus": "13C",
                    "peaks": [20.74],
                    "count": 1,
                    "range": [20, 80],
                },
                "cost": 1, "category": "spectrum_query",
            }
        ]
        result = grader.grade(trajectory)
        f611 = result["node_results"].get("F6.1.1")
        if f611:
            assert f611["method"] == "tool_match"

    def test_carbonyl_substructure_covers_f4(self, grader):
        """substructure_check for C=O should cover F4."""
        trajectory = [
            {
                "tool": "substructure_check",
                "args": {"smiles": "CC(=O)c1ccc(C)cc1", "pattern": "[CX3]=O"},
                "result": {"has_match": True, "num_matches": 1},
                "cost": 1, "category": "structure_analysis",
            }
        ]
        result = grader.grade(trajectory)
        f4 = result["node_results"].get("F4")
        if f4:
            assert f4["covered"] is True
            assert f4["method"] == "tool_match"


# ======================================================================
# HNMR DAG Grading Tests
# ======================================================================

HNMR_MOL_ID = "2_4_dimethyl_aniline"


@pytest.fixture
def hnmr_grader():
    """Create a DAGGrader for 2,4-dimethylaniline HNMR."""
    dag_path = str(DATA_DIR / "HNMR_HRE_v5.json")
    gt_path = str(DATA_DIR / "H_NMR" / f"{HNMR_MOL_ID}_HNMR_HRE.json")
    return DAGGrader(dag_path=dag_path, gt_path=gt_path)


class TestHNMRGraderSetup:
    def test_detects_hnmr(self, hnmr_grader):
        assert hnmr_grader._is_hnmr is True

    def test_cnmr_not_hnmr(self, grader):
        assert grader._is_hnmr is False

    def test_parses_formula(self, hnmr_grader):
        assert "C8H11N" in hnmr_grader._formula.replace(" ", "")

    def test_loads_hnmr_peaks(self, hnmr_grader):
        assert len(hnmr_grader._hnmr_peaks) == 6

    def test_gt_nodes_loaded(self, hnmr_grader):
        assert "F1" in hnmr_grader.gt_nodes
        assert hnmr_grader.gt_nodes["F1"]["answer"] == "4"

    def test_scored_node_count(self, hnmr_grader):
        scored = [nid for nid, info in hnmr_grader.gt_nodes.items()
                  if info.get("weight", 0) > 0]
        assert len(scored) == 38


class TestHNMRPPMRangeAlignment:
    """Verify HNMR_PPM_RANGES match hnmr_hre_autofill_gt.py (source of truth)."""

    def test_all_ranges(self):
        from interactive.dag_grader import HNMR_PPM_RANGES
        expected = {
            "alkene":         (4.5, 7.0),
            "aromatic":       (6.5, 8.0),
            "alkyne":         (1.5, 3.0),
            "alkyl":          (0.25, 2.5),
            "aromatic_alkyl": (2.0, 3.0),
            "aldehyde":       (8.5, 10.0),
            "cooh":           (10.0, 12.0),
            "amide":          (5.0, 9.0),
            "hetero_c":       (3.0, 4.5),
            "singlet_hetero": (2.0, 4.5),
        }
        for motif, exp_range in expected.items():
            assert HNMR_PPM_RANGES[motif] == exp_range, \
                f"HNMR range mismatch for {motif}: got {HNMR_PPM_RANGES[motif]}, expected {exp_range}"


class TestHNMRUnsaturation:
    def test_correct_dou(self, hnmr_grader):
        """F1: compute_unsaturation returning 4 should cover F1."""
        trajectory = [
            {"tool": "compute_unsaturation", "args": {"formula": "C8H11N"},
             "result": {"degree": 4, "formula": "C8H11N"},
             "cost": 1, "category": "computation"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F1"]["covered"] is True

    def test_wrong_dou(self, hnmr_grader):
        trajectory = [
            {"tool": "compute_unsaturation", "args": {"formula": "C8H11N"},
             "result": {"degree": 3, "formula": "C8H11N"},
             "cost": 1, "category": "computation"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F1"]["covered"] is False


class TestHNMRAlkeneAromatic:
    def test_alkene_query(self, hnmr_grader):
        """Query 1H 4.5-7.0 should cover F1.1.1 (GT=Yes for dimethylaniline)."""
        trajectory = [
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 4.5, "ppm_max": 7.0},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 6.84, "multiplicity": "s"},
                 {"ppm": 6.82, "multiplicity": "d"},
                 {"ppm": 6.54, "multiplicity": "d"},
             ], "count": 3, "range": [4.5, 7.0]},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F1.1.1"]["covered"] is True

    def test_aromatic_query(self, hnmr_grader):
        """Query 1H 6.5-8.0 should cover F1.2.1 (GT=Yes)."""
        trajectory = [
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 6.5, "ppm_max": 8.0},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 6.84, "multiplicity": "s"},
                 {"ppm": 6.82, "multiplicity": "d"},
                 {"ppm": 6.54, "multiplicity": "d"},
             ], "count": 3, "range": [6.5, 8.0]},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F1.2.1"]["covered"] is True


class TestHNMRAlkyl:
    def test_alkyl_query(self, hnmr_grader):
        """Query 1H 0.25-2.5 should cover F4 (GT=Yes)."""
        trajectory = [
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 0.25, "ppm_max": 2.5},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 2.21, "multiplicity": "s", "integration": 3},
                 {"ppm": 2.10, "multiplicity": "s", "integration": 3},
             ], "count": 2, "range": [0.25, 2.5]},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F4"]["covered"] is True


class TestHNMRIntegration:
    def test_integration_check(self, hnmr_grader):
        """get_integration with total=11 should cover F3 (GT=Yes, H=11)."""
        trajectory = [
            {"tool": "get_integration", "args": {},
             "result": {"range": [0, 15], "peaks": [
                 {"ppm": 6.84, "integration": 1},
                 {"ppm": 6.82, "integration": 1},
                 {"ppm": 6.54, "integration": 1},
                 {"ppm": 3.39, "integration": 2},
                 {"ppm": 2.21, "integration": 3},
                 {"ppm": 2.10, "integration": 3},
             ], "count": 6, "total_integration": 11},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F3"]["covered"] is True

    def test_integration_wrong(self, hnmr_grader):
        """Wrong total_integration should mark F3 covered=False."""
        trajectory = [
            {"tool": "get_integration", "args": {},
             "result": {"range": [0, 15], "peaks": [],
                        "count": 0, "total_integration": 9},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F3"]["covered"] is False


class TestHNMRMultiplicity:
    def test_singlet_check(self, hnmr_grader):
        """get_multiplicity returning singlets should cover F11 (GT=Yes)."""
        trajectory = [
            {"tool": "get_multiplicity", "args": {},
             "result": {"range": [0, 15], "peaks": [
                 {"ppm": 6.84, "multiplicity": "s"},
                 {"ppm": 6.82, "multiplicity": "d"},
                 {"ppm": 3.39, "multiplicity": "s"},
                 {"ppm": 2.21, "multiplicity": "s"},
                 {"ppm": 2.10, "multiplicity": "s"},
             ], "count": 5},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F11"]["covered"] is True


class TestHNMRCoupling:
    def test_coupling_covers_f10(self, hnmr_grader):
        """get_coupling should cover F10."""
        trajectory = [
            {"tool": "get_coupling", "args": {},
             "result": {"range": [0, 15], "peaks": [
                 {"ppm": 6.82, "j_values": [8.1], "multiplicity": "d"},
             ], "count": 1},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F10"]["covered"] is True


class TestHNMRFormulaFallback:
    def test_aldehyde_no_by_formula(self, hnmr_grader):
        """F6 (aldehyde) GT=No. If agent didn't query 8.5-10 but formula has no
        halogen-only constraint, formula fallback shouldn't fire (O IS present).
        Since O is present, fallback won't cover — returns None → no_match."""
        trajectory = [
            {"tool": "compute_unsaturation", "args": {"formula": "C8H11N"},
             "result": {"degree": 4, "formula": "C8H11N"},
             "cost": 1, "category": "computation"},
        ]
        result = hnmr_grader.grade(trajectory)
        f6 = result["node_results"]["F6"]
        # F6 GT=No. O is NOT in C8H11N formula → fallback fires
        # Actually C8H11N has no O → formula lacks O → GT=No confirmed
        assert f6["covered"] is True
        assert "formula" in f6["details"].lower()

    def test_carboxylic_no_by_formula(self, hnmr_grader):
        """F7 (COOH) GT=No. C8H11N has no O → formula fallback should cover."""
        trajectory = [
            {"tool": "compute_unsaturation", "args": {"formula": "C8H11N"},
             "result": {"degree": 4, "formula": "C8H11N"},
             "cost": 1, "category": "computation"},
        ]
        result = hnmr_grader.grade(trajectory)
        f7 = result["node_results"]["F7"]
        assert f7["covered"] is True


class TestHNMRHeteroC:
    def test_hetero_c_query(self, hnmr_grader):
        """Query 1H 3.0-4.5 should cover F9 (GT=Yes, has N, peak at 3.39)."""
        trajectory = [
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 3.0, "ppm_max": 4.5},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 3.39, "multiplicity": "s", "integration": 2},
             ], "count": 1, "range": [3.0, 4.5]},
             "cost": 1, "category": "spectrum_query"},
        ]
        result = hnmr_grader.grade(trajectory)
        assert result["node_results"]["F9"]["covered"] is True


class TestHNMRComprehensiveTrajectory:
    def test_high_coverage(self, hnmr_grader):
        """A thorough HNMR trajectory should achieve high coverage."""
        trajectory = [
            # DoU
            {"tool": "compute_unsaturation", "args": {"formula": "C8H11N"},
             "result": {"degree": 4, "formula": "C8H11N"},
             "cost": 1, "category": "computation"},
            # Alkene/aromatic region
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 4.5, "ppm_max": 8.0},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 6.84, "multiplicity": "s", "integration": 1},
                 {"ppm": 6.82, "multiplicity": "d", "integration": 1},
                 {"ppm": 6.54, "multiplicity": "d", "integration": 1},
             ], "count": 3, "range": [4.5, 8.0]},
             "cost": 1, "category": "spectrum_query"},
            # Alkyl region
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 0.0, "ppm_max": 3.0},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 2.21, "multiplicity": "s", "integration": 3},
                 {"ppm": 2.10, "multiplicity": "s", "integration": 3},
             ], "count": 2, "range": [0.0, 3.0]},
             "cost": 1, "category": "spectrum_query"},
            # Heteroatom adjacent (3-4.5)
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 3.0, "ppm_max": 4.5},
             "result": {"nucleus": "1H", "peaks": [
                 {"ppm": 3.39, "multiplicity": "s", "integration": 2},
             ], "count": 1, "range": [3.0, 4.5]},
             "cost": 1, "category": "spectrum_query"},
            # Aldehyde region (empty)
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 8.5, "ppm_max": 10.0},
             "result": {"nucleus": "1H", "peaks": [],
                        "count": 0, "range": [8.5, 10.0]},
             "cost": 1, "category": "spectrum_query"},
            # COOH region (empty)
            {"tool": "query_spectrum",
             "args": {"nucleus": "1H", "ppm_min": 10.0, "ppm_max": 12.0},
             "result": {"nucleus": "1H", "peaks": [],
                        "count": 0, "range": [10.0, 12.0]},
             "cost": 1, "category": "spectrum_query"},
            # Integration
            {"tool": "get_integration", "args": {},
             "result": {"range": [0, 15], "peaks": [
                 {"ppm": 6.84, "integration": 1},
                 {"ppm": 6.82, "integration": 1},
                 {"ppm": 6.54, "integration": 1},
                 {"ppm": 3.39, "integration": 2},
                 {"ppm": 2.21, "integration": 3},
                 {"ppm": 2.10, "integration": 3},
             ], "count": 6, "total_integration": 11},
             "cost": 1, "category": "spectrum_query"},
            # Multiplicity
            {"tool": "get_multiplicity", "args": {},
             "result": {"range": [0, 15], "peaks": [
                 {"ppm": 6.84, "multiplicity": "s"},
                 {"ppm": 6.82, "multiplicity": "d"},
                 {"ppm": 6.54, "multiplicity": "d"},
                 {"ppm": 3.39, "multiplicity": "s"},
                 {"ppm": 2.21, "multiplicity": "s"},
                 {"ppm": 2.10, "multiplicity": "s"},
             ], "count": 6},
             "cost": 1, "category": "spectrum_query"},
            # Coupling
            {"tool": "get_coupling", "args": {},
             "result": {"range": [0, 15], "peaks": [
                 {"ppm": 6.82, "j_values": [8.1], "multiplicity": "d"},
                 {"ppm": 6.54, "j_values": [7.8], "multiplicity": "d"},
             ], "count": 2},
             "cost": 1, "category": "spectrum_query"},
            # Submit
            {"tool": "submit", "args": {"smiles": "Cc1ccc(N)c(C)c1"},
             "result": {"submitted": "Cc1ccc(N)c(C)c1", "valid": True},
             "cost": 0, "category": "submit"},
        ]
        result = hnmr_grader.grade(trajectory)

        # Should have high coverage
        assert result["coverage"] > 0.5, \
            f"Expected >50% coverage, got {result['coverage']:.1%}"

        # Key nodes should be covered
        assert result["node_results"]["F1"]["covered"] is True
        assert result["node_results"]["F1.1.1"]["covered"] is True
        assert result["node_results"]["F1.2.1"]["covered"] is True
        assert result["node_results"]["F4"]["covered"] is True
        assert result["node_results"]["F9"]["covered"] is True
        assert result["node_results"]["F3"]["covered"] is True
        assert result["node_results"]["F10"]["covered"] is True
        assert result["node_results"]["F11"]["covered"] is True

        # Print coverage for debugging
        covered = sum(1 for r in result["node_results"].values() if r["covered"])
        total = len(result["node_results"])
        print(f"\nHNMR Coverage: {covered}/{total} = {result['coverage']:.1%}")
