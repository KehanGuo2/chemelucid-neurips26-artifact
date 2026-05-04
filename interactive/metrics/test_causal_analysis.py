"""Tests for Layer 3 causal analysis metrics."""

import pytest
from interactive.metrics.causal_analysis import compute_causal_metrics


# ---------------------------------------------------------------------------
# Fixtures: fake trajectories
# ---------------------------------------------------------------------------

def _hypothesis_driven_trajectory():
    """A well-structured trajectory: compute DoU → query aromatic → validate → predict → compare."""
    return [
        {
            "tool": "compute_unsaturation",
            "args": {"formula": "C7H5NO3S"},
            "result": {"degree": 6, "formula": "C7H5NO3S"},
            "cost": 1, "category": "computation",
        },
        {
            "tool": "query_spectrum",
            "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
            "result": {"nucleus": "13C", "range": [100, 170],
                       "peaks": [124.4, 127.7, 136.2, 136.8], "count": 4},
            "cost": 1, "category": "spectrum_query",
        },
        {
            "tool": "query_spectrum",
            "args": {"nucleus": "13C", "ppm_min": 150, "ppm_max": 220},
            "result": {"nucleus": "13C", "range": [150, 220],
                       "peaks": [159.1], "count": 1},
            "cost": 1, "category": "spectrum_query",
        },
        {
            "tool": "validate_smiles",
            "args": {"smiles": "O=C1NS(=O)(=O)c2ccccc21"},
            "result": {"valid": True, "canonical_smiles": "O=C1NS(=O)(=O)c2ccccc21",
                       "molecular_formula": "C7H5NO3S"},
            "cost": 2, "category": "structure_analysis",
        },
        {
            "tool": "predict_nmr",
            "args": {"smiles": "O=C1NS(=O)(=O)c2ccccc21", "nucleus": "13C"},
            "result": {"predicted_shifts": [159.0, 137.0, 135.0, 128.0, 125.0],
                       "nucleus": "13C"},
            "cost": 3, "category": "prediction",
        },
        {
            "tool": "compare_spectra",
            "args": {"predicted_shifts": [159.0, 137.0, 135.0, 128.0, 125.0],
                     "observed_shifts": [159.1, 136.8, 136.2, 127.7, 124.4],
                     "nucleus": "13C"},
            "result": {"similarity": 0.9, "mae": 1.2, "matched_peaks": 5,
                       "total_observed": 5},
            "cost": 1, "category": "validation",
        },
        {
            "tool": "submit",
            "args": {"smiles": "O=C1NS(=O)(=O)c2ccccc21"},
            "result": {"submitted": "O=C1NS(=O)(=O)c2ccccc21", "valid": True},
            "cost": 0, "category": "submit",
        },
    ]


def _guess_and_check_trajectory():
    """A random-walk trajectory: repeated validate_smiles with unrelated SMILES."""
    return [
        {
            "tool": "validate_smiles",
            "args": {"smiles": "c1ccccc1"},
            "result": {"valid": True, "canonical_smiles": "c1ccccc1",
                       "molecular_formula": "C6H6"},
            "cost": 2, "category": "structure_analysis",
        },
        {
            "tool": "validate_smiles",
            "args": {"smiles": "CC(=O)O"},
            "result": {"valid": True, "canonical_smiles": "CC(=O)O",
                       "molecular_formula": "C2H4O2"},
            "cost": 2, "category": "structure_analysis",
        },
        {
            "tool": "validate_smiles",
            "args": {"smiles": "CCO"},
            "result": {"valid": True, "canonical_smiles": "CCO",
                       "molecular_formula": "C2H6O"},
            "cost": 2, "category": "structure_analysis",
        },
        {
            "tool": "submit",
            "args": {"smiles": "CCO"},
            "result": {"submitted": "CCO", "valid": True},
            "cost": 0, "category": "submit",
        },
    ]


def _trajectory_with_conflicts():
    """Trajectory with invalid SMILES conflict and reactive follow-up."""
    return [
        {
            "tool": "compute_unsaturation",
            "args": {"formula": "C8H11N"},
            "result": {"degree": 4, "formula": "C8H11N"},
            "cost": 1, "category": "computation",
        },
        {
            "tool": "validate_smiles",
            "args": {"smiles": "INVALID_SMILES"},
            "result": {"valid": False, "error": "Invalid SMILES"},
            "cost": 2, "category": "structure_analysis",
        },
        {
            "tool": "query_spectrum",
            "args": {"nucleus": "13C", "ppm_min": 110, "ppm_max": 160},
            "result": {"nucleus": "13C", "range": [110, 160],
                       "peaks": [142.2, 131.3, 127.9], "count": 3},
            "cost": 1, "category": "spectrum_query",
        },
        {
            "tool": "validate_smiles",
            "args": {"smiles": "Cc1ccc(N)c(C)c1"},
            "result": {"valid": True, "canonical_smiles": "Cc1ccc(N)c(C)c1",
                       "molecular_formula": "C8H11N"},
            "cost": 2, "category": "structure_analysis",
        },
        {
            "tool": "submit",
            "args": {"smiles": "Cc1ccc(N)c(C)c1"},
            "result": {"submitted": "Cc1ccc(N)c(C)c1", "valid": True},
            "cost": 0, "category": "submit",
        },
    ]


def _trajectory_with_low_similarity():
    """Trajectory with a compare_spectra conflict (low similarity) and non-reactive follow-up."""
    return [
        {
            "tool": "predict_nmr",
            "args": {"smiles": "c1ccccc1", "nucleus": "13C"},
            "result": {"predicted_shifts": [128.0], "nucleus": "13C"},
            "cost": 3, "category": "prediction",
        },
        {
            "tool": "compare_spectra",
            "args": {"predicted_shifts": [128.0],
                     "observed_shifts": [142.2, 131.3, 127.9, 20.6, 17.5],
                     "nucleus": "13C"},
            "result": {"similarity": 0.2, "mae": 15.0, "matched_peaks": 1,
                       "total_observed": 5},
            "cost": 1, "category": "validation",
        },
        {
            "tool": "submit",
            "args": {"smiles": "c1ccccc1"},
            "result": {"submitted": "c1ccccc1", "valid": True},
            "cost": 0, "category": "submit",
        },
    ]


def _submit_rejected_trajectory():
    """Trajectory with submit_rejected entry from invalid-submit penalty."""
    return [
        {
            "tool": "compute_unsaturation",
            "args": {"formula": "C8H11N"},
            "result": {"degree": 4, "formula": "C8H11N"},
            "cost": 1, "category": "computation",
        },
        {
            "tool": "submit_rejected",
            "args": {"smiles": "NOT_SMILES"},
            "result": {"error": "Invalid SMILES. -5 budget penalty."},
            "cost": 5, "category": "submit",
        },
        {
            "tool": "query_spectrum",
            "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
            "result": {"nucleus": "13C", "range": [100, 170],
                       "peaks": [142.2], "count": 1},
            "cost": 1, "category": "spectrum_query",
        },
        {
            "tool": "submit",
            "args": {"smiles": "Cc1ccc(N)c(C)c1"},
            "result": {"submitted": "Cc1ccc(N)c(C)c1", "valid": True},
            "cost": 0, "category": "submit",
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHypothesisDriven:
    def test_has_dependencies(self):
        result = compute_causal_metrics(_hypothesis_driven_trajectory())
        assert result["num_edges"] > 0
        assert result["dependency_rate"] > 0.0

    def test_dependency_rate_is_high(self):
        result = compute_causal_metrics(_hypothesis_driven_trajectory())
        # A well-structured trajectory should have dependency_rate > 0.3
        assert result["dependency_rate"] >= 0.3

    def test_edges_have_correct_structure(self):
        result = compute_causal_metrics(_hypothesis_driven_trajectory())
        for edge in result["edges"]:
            assert "from_idx" in edge
            assert "to_idx" in edge
            assert "type" in edge
            assert "detail" in edge
            assert edge["from_idx"] < edge["to_idx"]

    def test_no_conflicts(self):
        result = compute_causal_metrics(_hypothesis_driven_trajectory())
        assert result["num_conflicts"] == 0
        assert result["conflict_reactivity"] == 1.0


class TestGuessAndCheck:
    def test_zero_or_low_dependency(self):
        result = compute_causal_metrics(_guess_and_check_trajectory())
        # Unrelated validate_smiles calls should have few dependencies.
        # One edge is expected: submit("CCO") depends on validate("CCO") via string match.
        assert result["dependency_rate"] <= 0.34

    def test_num_nodes(self):
        result = compute_causal_metrics(_guess_and_check_trajectory())
        assert result["num_nodes"] == 4


class TestConflictDetection:
    def test_detects_invalid_smiles_conflict(self):
        result = compute_causal_metrics(_trajectory_with_conflicts())
        assert result["num_conflicts"] == 1
        assert result["conflicts"][0]["tool"] == "validate_smiles"
        assert result["conflicts"][0]["signal"] == "invalid"

    def test_conflict_is_reactive(self):
        result = compute_causal_metrics(_trajectory_with_conflicts())
        assert result["conflicts"][0]["reactive"] is True
        assert result["conflicts"][0]["followup_tool"] == "query_spectrum"

    def test_reactivity_score(self):
        result = compute_causal_metrics(_trajectory_with_conflicts())
        assert result["conflict_reactivity"] == 1.0  # 1/1 reactive

    def test_low_similarity_conflict(self):
        result = compute_causal_metrics(_trajectory_with_low_similarity())
        assert result["num_conflicts"] == 1
        assert result["conflicts"][0]["tool"] == "compare_spectra"
        # Follow-up is submit (not exploratory) → non-reactive
        assert result["conflicts"][0]["reactive"] is False

    def test_non_reactive_score(self):
        result = compute_causal_metrics(_trajectory_with_low_similarity())
        assert result["conflict_reactivity"] == 0.0  # 0/1 reactive


class TestSubmitRejected:
    def test_detects_submit_rejected_conflict(self):
        result = compute_causal_metrics(_submit_rejected_trajectory())
        assert result["num_conflicts"] == 1
        assert result["conflicts"][0]["tool"] == "submit_rejected"

    def test_submit_rejected_reactive(self):
        result = compute_causal_metrics(_submit_rejected_trajectory())
        # After rejection, agent queries spectrum → reactive
        assert result["conflicts"][0]["reactive"] is True


class TestEdgeCases:
    def test_empty_trajectory(self):
        result = compute_causal_metrics([])
        assert result["dependency_rate"] == 0.0
        assert result["conflict_reactivity"] == 1.0
        assert result["num_nodes"] == 0
        assert result["num_edges"] == 0

    def test_single_tool_call(self):
        traj = [{
            "tool": "compute_unsaturation",
            "args": {"formula": "C8H11N"},
            "result": {"degree": 4},
            "cost": 1, "category": "computation",
        }]
        result = compute_causal_metrics(traj)
        assert result["dependency_rate"] == 0.0
        assert result["num_nodes"] == 1
        assert result["num_edges"] == 0

    def test_two_unrelated_calls(self):
        traj = [
            {
                "tool": "compute_unsaturation",
                "args": {"formula": "C8H11N"},
                "result": {"degree": 4},
                "cost": 1, "category": "computation",
            },
            {
                "tool": "compute_molecular_weight",
                "args": {"formula": "C6H6"},
                "result": {"molecular_weight": 78.11},
                "cost": 1, "category": "computation",
            },
        ]
        result = compute_causal_metrics(traj)
        # Different formulas, no data flow → no edges
        assert result["num_edges"] == 0
        assert result["dependency_rate"] == 0.0
