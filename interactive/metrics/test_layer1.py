"""Tests for Layer 1 (information acquisition) metrics."""

from interactive.metrics.layer1 import compute_layer1


class TestEmptyTrajectory:
    def test_empty_returns_zero(self):
        result = compute_layer1([])
        assert result["score"] == 0.0
        assert result["unique_regions"] == 0
        assert result["categories_used"] == 0

    def test_empty_regions_list(self):
        result = compute_layer1([])
        assert result["regions"] == []
        assert result["categories"] == []


class TestSingleQuery:
    def test_one_13c_query(self):
        traj = [
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
             "result": {"peaks": [128.5]}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert result["unique_regions"] == 1
        assert result["categories_used"] == 1
        assert "13C_100_170" in result["regions"]
        assert result["score"] > 0

    def test_one_1h_query(self):
        traj = [
            {"tool": "query_spectrum", "args": {"nucleus": "1H", "ppm_min": 6.5, "ppm_max": 8.0},
             "result": {"peaks": [7.2]}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert result["unique_regions"] == 1
        assert "1H_6_8" in result["regions"]


class TestMultipleRegions:
    def test_distinct_regions_counted(self):
        traj = [
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 170, "ppm_max": 220},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert result["unique_regions"] == 3

    def test_duplicate_regions_deduplicated(self):
        traj = [
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert result["unique_regions"] == 1


class TestCategories:
    def test_multiple_categories(self):
        traj = [
            {"tool": "compute_unsaturation", "args": {"formula": "C10H10O4"},
             "result": {"unsaturation": 6}, "cost": 1, "category": "computation"},
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 100, "ppm_max": 170},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
            {"tool": "validate_smiles", "args": {"smiles": "c1ccccc1"},
             "result": {"valid": True}, "cost": 2, "category": "structure_analysis"},
        ]
        result = compute_layer1(traj)
        assert result["categories_used"] == 3
        assert "computation" in result["categories"]
        assert "spectrum_query" in result["categories"]
        assert "structure_analysis" in result["categories"]

    def test_same_category_not_double_counted(self):
        traj = [
            {"tool": "validate_smiles", "args": {"smiles": "C"},
             "result": {"valid": True}, "cost": 2, "category": "structure_analysis"},
            {"tool": "substructure_check", "args": {"smiles": "C", "pattern": "C"},
             "result": {"has_match": True}, "cost": 1, "category": "structure_analysis"},
        ]
        result = compute_layer1(traj)
        assert result["categories_used"] == 1


class TestScoreRange:
    def test_score_between_0_and_1(self):
        traj = [
            {"tool": "query_spectrum", "args": {"nucleus": "13C", "ppm_min": 0, "ppm_max": 50},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert 0 <= result["score"] <= 1.0

    def test_max_coverage_caps_at_1(self):
        # 10 regions + 6 categories = should still cap at 1.0
        traj = []
        for i in range(10):
            traj.append({
                "tool": "query_spectrum",
                "args": {"nucleus": "13C", "ppm_min": i * 20, "ppm_max": i * 20 + 15},
                "result": {"peaks": []}, "cost": 1, "category": "spectrum_query",
            })
        for cat in ["computation", "structure_analysis", "prediction", "validation", "submit"]:
            traj.append({"tool": "x", "args": {}, "result": {}, "cost": 1, "category": cat})
        result = compute_layer1(traj)
        assert result["score"] <= 1.0


class TestHNMRTools:
    def test_multiplicity_adds_region(self):
        traj = [
            {"tool": "get_multiplicity", "args": {"ppm_min": 6.5, "ppm_max": 8.0},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert result["unique_regions"] >= 1

    def test_integration_adds_region(self):
        traj = [
            {"tool": "get_integration", "args": {"ppm_min": 0.5, "ppm_max": 2.5},
             "result": {"peaks": []}, "cost": 1, "category": "spectrum_query"},
        ]
        result = compute_layer1(traj)
        assert result["unique_regions"] >= 1
